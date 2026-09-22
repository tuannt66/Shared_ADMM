"""Mobile-base local controller for the revised modular CLF-ADMM design.

The base owns one local responsibility: trajectory tracking.  Its nominal
command ``u_b^0`` is supplied by the path-tracking law.  The distributed
coordination layer may add a correction ``delta_u_b``, but the base-local
objective contains no manipulator decision variable and no Cartesian
"support-arm" tracking term.

The local ADMM variable is the scalar CLF contribution

    s_b = a_b^T delta_u_b.
"""

import math
import threading
import numpy as np

try:
    from .box_qp import solve_box_qp
except ImportError:  # scripts add src/shared directly to sys.path
    from box_qp import solve_box_qp


class BaseSubsystem:
    """Husky mobile-base controller (ADMM agent 1)."""

    # Wheel layout: [front-left, front-right, rear-left, rear-right]
    _WHEEL_FWD  = [1, 1, 1, 1]
    _WHEEL_TURN = [1, -1, 1, -1]

    def __init__(self, sim, husky_id, kuka_id,
                 yaw_init, offset_husky_to_ee_xy,
                 wheels=(2, 3, 4, 5), wheel_radius=0.165,
                 max_vel=1.0,
                 w_tracking=5.0, alpha=0.03,
                 reg_v=2.0, reg_omega=1.0, path_weight=1.0,
                 wheel_force=150.0,
                 allow_reverse=True,
                 sim_lock=None):
        """
        Parameters
        ----------
        sim : module
            ``pybullet`` module handle.
        husky_id : int
            PyBullet body ID for the Husky platform.
        kuka_id : int
            PyBullet body ID for the Kuka arm (mounted on Husky).
        yaw_init : float
            Desired heading angle (rad) — maintained throughout tracking.
        offset_husky_to_ee_xy : ndarray (2,)
            XY offset from Husky base-link to EE at the rest configuration.
        wheels : tuple[int]
            Husky wheel joint indices.
        wheel_radius : float
            Wheel radius (m) — used for visual wheel-spin only.
        max_vel : float
            Saturation for both linear and angular velocity (m/s, rad/s).
        w_tracking : float
            Deprecated compatibility parameter.  The revised correction cost
            is normalized by physical actuator authority and does not use this
            scalar.
        alpha : float
            Deprecated legacy tuning argument retained for command-line/API
            compatibility.  The revised base nominal controller is the
            external path-tracking command ``u_b^0``; this parameter does not
            participate in the local correction QP and is unrelated to the
            adaptive role index.
        reg_v : float
            Deprecated compatibility parameter from the pre-normalized local
            metric.  It is retained for old command lines/log readers only.
        reg_omega : float
            Deprecated compatibility parameter from the pre-normalized local
            metric.  It is retained for old command lines/log readers only.
        path_weight : float
            Weight of the explicit mobile-base path tracking term.
        allow_reverse : bool
            If True, the local base QP uses symmetric linear-velocity bounds
            ``-max_vel <= v <= max_vel`` so the base can recover when the
            moving path reference lies behind it.  If False, the legacy
            forward-only bound ``0 <= v <= max_vel`` is used.
        """
        self.p = sim
        self.husky_id  = husky_id
        self.kuka_id   = kuka_id
        self.yaw_init  = yaw_init
        self.offset    = np.asarray(offset_husky_to_ee_xy, dtype=float)
        self.wheels    = list(wheels)
        self.wheel_r   = wheel_radius
        self.max_vel   = max_vel
        self.legacy_tracking_fraction = float(alpha)
        if not np.isfinite(self.legacy_tracking_fraction) or self.legacy_tracking_fraction <= 0.0:
            raise ValueError("legacy base alpha must be a positive finite scalar")
        self.alpha = self.legacy_tracking_fraction  # deprecated compatibility alias
        self.w_track        = w_tracking
        self._w_track_target = w_tracking
        self._w_track_tau    = 0.92        # EMA smoothing (per solve() call)
        self.reg_v       = float(reg_v)
        self.reg_omega   = float(reg_omega)
        self.path_weight = float(path_weight)
        if self.path_weight < 0.0 or not np.isfinite(self.path_weight):
            raise ValueError("path_weight must be a nonnegative finite scalar")
        self.wheel_force = wheel_force
        self.allow_reverse = bool(allow_reverse)
        self._lock       = sim_lock or threading.Lock()
        self._dyn_dist   = None  # set via set_dynamic_disturbance()
        self._last_J_ee  = np.zeros((6, 2))
        self._last_ee_pos = np.zeros(3)
        self._last_qp_info = {}
        self._last_qp_model = {}
        self._last_local_diag = {}

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_pose(self):
        """Return ``(position_xyz, yaw)`` from the base-local sensor API.

        This is the only PyBullet pose query used by the base MCU.  The arm
        receives this pose through the peer-to-peer state channel instead of
        querying the Husky body directly.
        """
        with self._lock:
            pos, orn = self.p.getBasePositionAndOrientation(self.husky_id)
            yaw = self.p.getEulerFromQuaternion(orn)[2]
        return np.asarray(pos, dtype=float), float(yaw)

    def get_state(self):
        """Return ``(xy, yaw)`` of the Husky base link.

        Returns
        -------
        xy : ndarray (2,)
            World-frame XY position.
        yaw : float
            Heading angle (rad).
        """
        pos, yaw = self.get_pose()
        return pos[:2].copy(), yaw

    # ------------------------------------------------------------------
    # ADMM subproblem
    # ------------------------------------------------------------------

    def solve_clf_admm(self, a_clf, z_clf, lambda_clf, rho,
                       pos_desired=None, vel_desired=None, dt=1e-3,
                       ee_pos=None, base_pos=None, R_base=None,
                       task_share=None,
                       path_v_ref=0.0, path_omega_ref=0.0,
                       role_weight=1.0,
                       command_lower=None, command_upper=None):
        """Solve the revised modular base CLF-ADMM subproblem.

        The base local responsibility is trajectory tracking.  Therefore the
        nominal command is the path-tracking command

            u_b^0 = [v_path, omega_path]^T,

        projected only onto the base's own actuator bounds.  No end-effector
        tracking term and no arm/task-share variable appears in the local
        objective.

        ADMM optimises only the correction delta_u_b = u_b-u_b^0:

            min  w_b/2 ||delta_u_b||^2_{W_b}
               + rho/2 (a_b^T delta_u_b-z_b+lambda_b)^2.

        ``role_weight`` is the adaptive preference weight from Layer 2.
        """
        _, h_yaw = self.get_state()

        # Optional kinematic telemetry is retained strictly for coordinator
        # geometry/logging.  It does not enter the local base objective.
        if ee_pos is not None and base_pos is not None and R_base is not None:
            ee_curr = np.asarray(ee_pos, dtype=float).reshape(3)
            base_pos_arr = np.asarray(base_pos, dtype=float).reshape(3)
            Rb = np.asarray(R_base, dtype=float).reshape(3, 3)
            cy, sy = math.cos(h_yaw), math.sin(h_yaw)
            # Full spatial EE twist induced by the planar mobile base.
            # u_b = [v, omega].  Forward motion contributes only linear
            # velocity; yaw contributes both z x r linear velocity and
            # angular velocity [0,0,omega].
            e_fwd = np.array([cy, sy, 0.0], dtype=float)
            z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
            r_ee = ee_curr - base_pos_arr
            J_lin = np.column_stack([
                e_fwd,
                np.cross(z_axis, r_ee),
            ])
            J_ang = np.column_stack([
                np.zeros(3),
                z_axis,
            ])
            J = np.vstack([J_lin, J_ang])
            self._last_J_ee = J.copy()
            self._last_ee_pos = ee_curr.copy()

        u_ref = np.array([float(path_v_ref), float(path_omega_ref)], dtype=float)
        v_lower = -self.max_vel if self.allow_reverse else 0.0
        lower = np.array([v_lower, -self.max_vel], dtype=float)
        upper = np.array([self.max_vel, self.max_vel], dtype=float)

        # Optional per-step operational bounds.  These are local physical
        # constraints, not role weights.  The pick/place executor uses them
        # during terminal release dwell to make the base quasi-static while
        # the suspended object settles.  Because the bounds are exposed in
        # ``qp_model``, Layer 2 sees the same reduced feasible set when it
        # computes gamma_b.
        if command_lower is not None:
            lower = np.maximum(lower,
                               np.asarray(command_lower, dtype=float).reshape(2))
        if command_upper is not None:
            upper = np.minimum(upper,
                               np.asarray(command_upper, dtype=float).reshape(2))
        if np.any(lower > upper):
            raise ValueError(
                f"inconsistent base command bounds: lower={lower}, upper={upper}")

        # Dimensionless correction metric.  A correction equal to one
        # actuator-speed rating has unit normalized magnitude, regardless of
        # whether the component is linear velocity (m/s) or angular velocity
        # (rad/s).  This prevents arbitrary legacy gains from overwhelming
        # the adaptive role weights.  Remaining instantaneous margin is *not*
        # used here; it is already represented by the hard bounds and gamma_b.
        corr_scale = np.maximum(
            np.maximum(np.abs(lower), np.abs(upper)), 1e-6)
        Wb = np.diag(1.0 / (corr_scale ** 2))
        w_role = max(float(role_weight), 1e-9)
        H_corr = w_role * Wb

        # The nominal base controller is the path tracker itself.  Its output
        # is independent of the role preference; only local actuator clipping
        # is applied before coordination.
        u_nom = np.clip(u_ref, lower, upper)
        nom_info = {"converged": True, "method": "direct_clip"}

        # Solve the correction problem in full-command coordinates:
        #   1/2 (u-u_nom)^T H_corr (u-u_nom)
        # + rho/2 (a^T(u-u_nom)-z+lambda)^2.
        a = np.asarray(a_clf, dtype=float).reshape(2)
        H = H_corr + rho * np.outer(a, a)
        shifted_target = (float(z_clf) - float(lambda_clf)
                          + float(a @ u_nom))
        rhs = H_corr @ u_nom + rho * a * shifted_target

        u, qp_info = solve_box_qp(
            H + 1e-9*np.eye(2), rhs, lower, upper,
            x0=u_nom, tol=1e-10, max_sweeps=100)

        delta_u = u - u_nom
        s_corr = float(a @ delta_u)
        admm_res = float(s_corr - float(z_clf) + float(lambda_clf))

        self._last_qp_model = {
            "H": (H_corr + 1e-9*np.eye(2)).copy(),
            "rhs": (H_corr @ u_nom).copy(),
            "lower": lower.copy(),
            "upper": upper.copy(),
            "x_nom": u_nom.copy(),
            "role_weight": float(w_role),
        }
        self._last_qp_info = {
            **qp_info,
            "lower": lower.copy(),
            "upper": upper.copy(),
            "nominal_qp_converged": bool(nom_info.get("converged", True)),
        }
        self._last_local_diag = {
            "u_path": u_ref.copy(),
            "u_nom": u_nom.copy(),
            "u_cmd": u.copy(),
            "u_correction": delta_u.copy(),
            "role_weight": float(w_role),
            "correction_cost": float(0.5*w_role*(delta_u @ Wb @ delta_u)),
            "correction_scale": corr_scale.copy(),
            "correction_metric_diag": np.diag(Wb).copy(),
            "admm_cost": float(0.5*rho*admm_res*admm_res),
            "path_cmd_gap": float(np.linalg.norm(u - u_ref)),
            "nom_cmd_gap": float(np.linalg.norm(delta_u)),
            "path_nom_gap": float(np.linalg.norm(u_nom - u_ref)),
            # Legacy fields deliberately zero/NaN: no EE-support cost remains.
            "ee_cost": 0.0,
            "path_cost": float(0.5*((u_nom-u_ref) @ Wb @ (u_nom-u_ref))),
            "reg_cost": 0.0,
            "v_ref_cart": np.zeros(6),
            "v_task_cart": np.full(6, np.nan),
            "v_path_ee_cart": (self._last_J_ee @ u_nom).copy(),
            "v_residual_cart": np.full(6, np.nan),
            "v_shareable_cart": np.full(6, np.nan),
            "v_unshareable_cart": np.full(6, np.nan),
            "base_task_projector": np.full((6, 6), np.nan),
            "v_lower_active": bool(u[0] <= lower[0] + 1e-6),
            "v_upper_active": bool(u[0] >= upper[0] - 1e-6),
            "omega_lower_active": bool(u[1] <= lower[1] + 1e-6),
            "omega_upper_active": bool(u[1] >= upper[1] - 1e-6),
            "path_v_outside_bounds": bool(u_ref[0] < lower[0]-1e-9 or u_ref[0] > upper[0]+1e-9),
            "path_omega_outside_bounds": bool(u_ref[1] < lower[1]-1e-9 or u_ref[1] > upper[1]+1e-9),
        }
        return float(u[0]), float(u[1]), float(s_corr), u_nom

    # ------------------------------------------------------------------
    def set_tracking_weight(self, w_target):
        """Update the local tracking metric once per outer control step.

        Keeping this update outside ``solve_clf_admm`` freezes the local
        objective during all inner ADMM iterations, which is required by the
        standard ADMM formulation.
        """
        self._w_track_target = float(w_target)
        self.w_track = (self._w_track_tau * self.w_track
                        + (1.0 - self._w_track_tau) * self._w_track_target)

    # ------------------------------------------------------------------
    # Actuation
    # ------------------------------------------------------------------

    def set_dynamic_disturbance(self, dyn_dist):
        """Attach a dynamic disturbance layer for base velocity filtering.

        When set, ``apply()`` routes velocity commands through the
        disturbance layer's first-order lag + acceleration limiter
        before calling ``resetBaseVelocity``.

        Parameters
        ----------
        dyn_dist : DynamicDisturbance or None
            The disturbance layer.  Pass None to revert to instant
            velocity mode (original behaviour).
        """
        self._dyn_dist = dyn_dist

    def apply(self, v_cmd, omega_cmd, yaw, dt=None):
        """Drive Husky via wheel velocity control (physics-based).

        Husky is driven through wheel VELOCITY_CONTROL so the physics engine
        (inertia, wheel-ground friction) determines actual motion — the robot
        accelerates and decelerates naturally.

        Kuka's base velocity is synchronised to the commanded velocity via
        resetBaseVelocity so the constraint between them stays stable.
        Without this sync the Kuka inertia fights the wheel motors and
        prevents Husky from moving.

        Parameters
        ----------
        v_cmd : float
            Desired forward velocity (m/s).
        omega_cmd : float
            Desired yaw rate (rad/s).
        yaw : float
            Current heading (rad).
        dt : float or None
            Timestep — used by augmentation lag filter if attached.
        """
        # Change 3: Route through BaseDynamicsAugmentation for actuator lag.
        # The augmentation's apply_base_dynamics() returns lagged velocities
        # that reflect real-world wheel inertia — what the wheels actually receive.
        if (self._dyn_dist is not None
                and dt is not None
                and hasattr(self._dyn_dist, "apply_base_dynamics")):
            v_cmd, omega_cmd = self._dyn_dist.apply_base_dynamics(
                v_cmd, omega_cmd, dt)

        wspin = v_cmd / self.wheel_r
        cy, sy = math.cos(yaw), math.sin(yaw)
        lin = [v_cmd * cy, v_cmd * sy, 0.0]
        ang = [0.0, 0.0, omega_cmd]
        with self._lock:
            for i in range(4):
                wv = wspin * self._WHEEL_FWD[i] + omega_cmd * self._WHEEL_TURN[i]
                self.p.setJointMotorControl2(
                    self.husky_id, self.wheels[i], self.p.VELOCITY_CONTROL,
                    targetVelocity=wv, force=self.wheel_force)
            # Synchronise Kuka velocity to commanded value so the rigid
            # constraint between them stays stable.  KNOWN MODELLING
            # SIMPLIFICATION (verified 2026-07): this makes the Kuka a
            # kinematic tractor and the wheels cosmetic.  Removing it
            # (wheel-driven base) was tested and does NOT work in this
            # contact setup: forward traction stalls whenever the
            # constraint budget is large or the arm loads it, and
            # skid-steer yaw authority is ~0 in every tested
            # (maxForce, friction, wheel_force) combination — see
            # docs/planned_trajectory_verification.md §6.
            self.p.resetBaseVelocity(self.kuka_id, lin, ang)
