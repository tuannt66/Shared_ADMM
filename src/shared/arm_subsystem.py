"""Manipulator local controller for the revised modular CLF-ADMM design.

The arm first computes an independent nominal joint-velocity command from a
local-frame end-effector reference using damped least-squares inverse
kinematics plus arm-local null-space objectives.  The coordination layer then
adds ``delta_dq_a`` through a minimum-correction local QP.

No alpha-scaled Cartesian task share and no command-smoothing quadratic term
appears in the local objective.  The only ADMM coupling quantity is

    s_a = a_a^T delta_dq_a.
"""

import math
import threading
import numpy as np

try:
    from .task_space_6d import so3_log, task_scale_matrix
except ImportError:
    from task_space_6d import so3_log, task_scale_matrix

try:
    from .box_qp import solve_box_qp
except ImportError:  # scripts add src/shared directly to sys.path
    from box_qp import solve_box_qp


class ArmSubsystem:
    """Kuka 7-DOF arm controller (ADMM agent 2)."""

    def __init__(self, sim, kuka_id, ee_index, num_joints,
                 q_lower, q_upper,
                 damping=0.05, max_joint_vel=2.0,
                 k_jl=2.0, k_manip=0.5, w_tracking=5.0,
                 alpha=0.15, ema_smooth=0.0,
                 acc_limit=None, joint_force=200.0,
                 manip_update_rate=10, joint_indices=None, sim_lock=None,
                 q_pref=None, orientation_tracking_fraction=0.02,
                 orientation_length=0.25, max_omega_ref=0.50,
                 max_linear_ref=0.35,
                 adaptive_damping_sigma=0.06,
                 adaptive_damping_max=0.14):
        """
        Parameters
        ----------
        sim : module
            ``pybullet`` module handle.
        kuka_id : int
            PyBullet body ID for the Kuka arm.
        ee_index : int
            End-effector link index.
        num_joints : int
            Number of revolute joints (7 for iiwa).
        q_lower, q_upper : array-like (num_joints,)
            Joint position limits (rad).
        damping : float
            DLS damping factor λ  (J Jᵀ + λ²I)⁻¹.
        max_joint_vel : float
            Joint velocity saturation (rad/s).
        k_jl : float
            Null-space gain for joint-limit avoidance gradient.
        k_manip : float
            Null-space gain for manipulability maximisation gradient.
        w_tracking : float
            Deprecated compatibility parameter from the pre-normalized
            correction metric.  The revised coordination cost is normalized
            by joint-velocity authority and does not use this scalar.
        alpha : float
            Legacy constructor name for the arm-local tracking fraction.
            It is unrelated to the adaptive role index ``alpha`` used by the
            coordinator.  Internally it is stored as ``tracking_fraction``.
            A value of 0.15 closes roughly 15% of the local Cartesian error
            per control step before DLS inversion.
        ema_smooth : float
            Deprecated compatibility parameter.  The revised local QP does
            not contain a command-smoothing objective; this value is ignored.
        acc_limit : float or None
            Maximum joint acceleration (rad/s²).  When set, velocity
            commands are rate-limited so the prediction x_a accounts
            for realistic acceleration constraints.  None = no limit.
        manip_update_rate : int
            Recompute manipulability gradient every N steps.
        q_pref : array-like (num_joints,) or None
            Null-space target for the joint-limit-avoidance term.
            Defaults to the geometric mid-range ``(q_lower+q_upper)/2``,
            which for arms with symmetric joint limits (e.g. Kuka iiwa)
            is ``q = 0`` — a kinematic singularity (manipulability = 0),
            not a safe "avoid limits" target.  Pass the carry/rest pose
            here to stop the null space from spring-pulling the arm
            straight whenever the primary task has spare capacity
            (e.g. while dwelling during grasp/release approach).
        orientation_tracking_fraction : float
            Fraction of local full-orientation error closed per control step
            when a pose reference is active.  In this V10.8 branch, the Stage-3
            planner is still free to choose wrist yaw on its tool-axis manifold,
            but the resulting planned full orientation is tracked during execution.
        orientation_length : float
            Characteristic length (m/rad) used to scale all three angular rows
            in the 6-D pose DLS task so translational and rotational errors have
            compatible numerical units.
        max_omega_ref : float
            Norm limit for the commanded local angular velocity (rad/s).
        """
        self.p = sim
        self.kuka_id  = kuka_id
        self.ee_index = ee_index
        self.nj       = num_joints
        self._joint_ids = list(joint_indices) if joint_indices is not None \
                          else list(range(num_joints))

        # Total non-fixed DOF for calculateJacobian (may differ from nj when
        # the URDF has extra fixed joints, e.g. xArm6's world_joint).
        _FIXED = 4
        self._n_dof = sum(
            1 for i in range(self.p.getNumJoints(kuka_id))
            if self.p.getJointInfo(kuka_id, i)[2] != _FIXED
        )

        self.q_lower = np.asarray(q_lower, dtype=float)
        self.q_upper = np.asarray(q_upper, dtype=float)
        self.q_mid   = (self.q_lower + self.q_upper) / 2.0
        self.q_range = self.q_upper - self.q_lower
        self.q_pref  = (self.q_mid if q_pref is None
                        else np.asarray(q_pref, dtype=float))

        self.damping = damping
        self.max_dq  = max_joint_vel
        self.k_jl    = k_jl
        self.k_manip = k_manip
        self.w_track         = w_tracking
        self._w_track_target = w_tracking
        self._w_track_tau    = 0.92        # EMA smoothing (per solve() call)
        self.tracking_fraction = float(alpha)
        # Deprecated compatibility alias. Do not use this as the role index.
        self.alpha = self.tracking_fraction
        self.ema_smooth = ema_smooth
        self.acc_limit   = acc_limit
        self.joint_force = joint_force
        self.manip_rate  = manip_update_rate
        self.orientation_tracking_fraction = float(orientation_tracking_fraction)
        self.orientation_length = max(float(orientation_length), 1e-6)
        self.max_omega_ref = max(float(max_omega_ref), 1e-6)
        self.max_linear_ref = max(float(max_linear_ref), 1e-6)
        self.adaptive_damping_sigma = max(
            float(adaptive_damping_sigma), 1e-6)
        self.adaptive_damping_max = max(
            float(adaptive_damping_max), float(self.damping))
        self._S6 = task_scale_matrix(self.orientation_length)
        self._lock       = sim_lock or threading.Lock()

        # Persistent state
        self._grad_manip = np.zeros(num_joints)
        self._dq_prev    = np.zeros(num_joints)   # previously applied command
        self._last_v_arm = np.zeros(6)            # last EE spatial-twist command
        self.w_current   = 0.0   # last-computed Yoshikawa manipulability

        # State cache for external logging (updated every solve() call)
        self._last_q  = np.zeros(num_joints)
        self._last_qd = np.zeros(num_joints)
        self._last_J  = np.zeros((6, num_joints))
        self._last_J_ang = np.zeros((3, num_joints))
        self._last_ee_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self._last_orientation_error = np.zeros(3)
        self._last_tool_axis_error = np.zeros(3)
        self._last_omega_arm_local = np.zeros(3)

        # EE/mount telemetry cache, published to the base over CAN
        # (see ArmMCU in coordinator.py) — None until the first
        # get_state() call (either via solve() or a lazy get_ee_telemetry()).
        self._last_ee_pos   = None
        self._last_base_pos = None
        self._last_R        = None
        self._last_ee_rot   = None
        self._last_qp_info  = {}
        self._last_qp_model = {}

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_state(self):
        """Return ``(q, qd, ee_pos, base_pos, R_base, J_world)``.

        Returns
        -------
        q : list[float]
            Current joint positions (rad).
        qd : list[float]
            Current joint velocities (rad/s).
        ee_pos : ndarray (3,)
            End-effector world position.
        base_pos : ndarray (3,)
            Kuka mount world position.
        R_base : ndarray (3, 3)
            Rotation matrix from Kuka base to world frame.
        J : ndarray (6, nj)
            Full spatial Jacobian [linear; angular] in world frame.
        """
        with self._lock:
            js  = [self.p.getJointState(self.kuka_id, j)
                   for j in self._joint_ids]
            q   = [s[0] for s in js]
            qd  = [s[1] for s in js]

            ls  = self.p.getLinkState(self.kuka_id, self.ee_index,
                                     computeForwardKinematics=True)
            ee  = np.array(ls[4])
            # Real PyBullet link state provides worldLinkFrameOrientation at
            # index 5.  A few lightweight repo test doubles pre-date the pose
            # upgrade and expose position only; keep those tests compatible.
            ee_quat = (np.array(ls[5], dtype=float) if len(ls) > 5 else
                       np.array([0.0, 0.0, 0.0, 1.0]))

            base_pos, base_orn = self.p.getBasePositionAndOrientation(self.kuka_id)
            base_pos = np.array(base_pos)
            R = np.array(
                self.p.getMatrixFromQuaternion(base_orn)).reshape(3, 3)

            q_pad  = list(q)  + [0.0] * (self._n_dof - self.nj)
            qd_pad = list(qd) + [0.0] * (self._n_dof - self.nj)
            zero_vec = [0.0] * self._n_dof
            jac_lin, jac_ang = self.p.calculateJacobian(
                self.kuka_id, self.ee_index, [0, 0, 0],
                q_pad, qd_pad, zero_vec)
            J_lin_world = R @ np.array(jac_lin)[:, :self.nj]
            J_ang = R @ np.array(jac_ang)[:, :self.nj]
            J = np.vstack([J_lin_world, J_ang])

        self._last_J_ang = J_ang.copy()
        self._last_ee_quat = ee_quat.copy()
        self._last_ee_rot = self._quat_to_matrix(ee_quat)

        return q, qd, ee, base_pos, R, J

    def get_diagnostics(self):
        """Return a dict of diagnostic info for display / logging.

        Keys: ``manipulability``, ``jl_margins_deg``, ``min_margin``,
        ``min_margin_joint``, ``q``.
        """
        with self._lock:
            q = [self.p.getJointState(self.kuka_id, j)[0]
                 for j in self._joint_ids]
        margins = [math.degrees(min(q[j] - self.q_lower[j],
                                    self.q_upper[j] - q[j]))
                   for j in range(self.nj)]
        return {
            "manipulability":   self.w_current,
            "jl_margins_deg":   margins,
            "min_margin":       min(margins),
            "min_margin_joint": margins.index(min(margins)) + 1,
            "q":                q,
            "orientation_error_deg": math.degrees(
                float(np.linalg.norm(self._last_orientation_error))),
        }

    def get_ee_telemetry(self):
        """Return ``(ee_pos, base_pos, R_base)`` for CAN-bus publication.

        Used by ``ArmMCU`` to broadcast the arm's own kinematic state to
        the base over ``arm_to_base`` (see ``coordinator.py``), replacing
        the base's former direct pybullet query of the arm's link state.

        Returns the cache populated by the last ``solve()`` call; if
        called before any ``solve()`` has run, lazily calls
        ``get_state()`` once to populate it.
        """
        if self._last_ee_pos is None:
            _, _, ee_pos, base_pos, R, _ = self.get_state()
            self._last_ee_pos   = ee_pos.copy()
            self._last_base_pos = base_pos.copy()
            self._last_R        = R.copy()
        return self._last_ee_pos, self._last_base_pos, self._last_R

    def get_ee_pose_telemetry(self):
        """Return (ee_pos, R_ee_world, base_pos, R_base)."""
        if self._last_ee_pos is None or self._last_ee_rot is None:
            _, _, ee_pos, base_pos, R, _ = self.get_state()
            self._last_ee_pos = ee_pos.copy()
            self._last_base_pos = base_pos.copy()
            self._last_R = R.copy()
        return (self._last_ee_pos.copy(), self._last_ee_rot.copy(),
                self._last_base_pos.copy(), self._last_R.copy())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _manipulability(J):
        """Yoshikawa manipulability  w = √det(J Jᵀ)."""
        return math.sqrt(max(np.linalg.det(J @ J.T), 1e-20))

    @staticmethod
    def _quat_conj(q):
        q = np.asarray(q, dtype=float).reshape(4)
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)

    @staticmethod
    def _quat_mul(q1, q2):
        """Hamilton product for PyBullet xyzw quaternions."""
        x1, y1, z1, w1 = np.asarray(q1, dtype=float).reshape(4)
        x2, y2, z2, w2 = np.asarray(q2, dtype=float).reshape(4)
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ], dtype=float)

    @classmethod
    def _quat_error_rotvec(cls, q_des, q_cur):
        """Shortest rotation vector taking ``q_cur`` to ``q_des``.

        Both quaternions are expressed in the same reference frame.  The
        returned rotation vector is therefore expressed in that frame and can
        be paired directly with a spatial/geometric angular Jacobian.
        """
        qd = np.asarray(q_des, dtype=float).reshape(4)
        qc = np.asarray(q_cur, dtype=float).reshape(4)
        qd = qd / max(float(np.linalg.norm(qd)), 1e-12)
        qc = qc / max(float(np.linalg.norm(qc)), 1e-12)
        qe = cls._quat_mul(qd, cls._quat_conj(qc))
        qe = qe / max(float(np.linalg.norm(qe)), 1e-12)
        if qe[3] < 0.0:  # shortest quaternion representative
            qe = -qe
        v = qe[:3]
        nv = float(np.linalg.norm(v))
        if nv < 1e-10:
            return np.zeros(3)
        angle = 2.0 * math.atan2(nv, float(np.clip(qe[3], -1.0, 1.0)))
        return (angle / nv) * v

    def _quat_to_matrix(self, q):
        return np.asarray(
            self.p.getMatrixFromQuaternion(
                list(np.asarray(q, dtype=float).reshape(4))),
            dtype=float).reshape(3, 3)

    @staticmethod
    def _axis_error_rotvec(a_des, a_cur):
        """Shortest rotation vector that aligns ``a_cur`` with ``a_des``.

        Rotation *about* the tool axis is intentionally absent.  The vector
        therefore represents a two-DOF direction-alignment task rather than a
        full three-DOF orientation task.
        """
        ad = np.asarray(a_des, dtype=float).reshape(3)
        ac = np.asarray(a_cur, dtype=float).reshape(3)
        ad /= max(float(np.linalg.norm(ad)), 1e-12)
        ac /= max(float(np.linalg.norm(ac)), 1e-12)
        c = np.cross(ac, ad)
        nc = float(np.linalg.norm(c))
        dot = float(np.clip(np.dot(ac, ad), -1.0, 1.0))
        angle = math.acos(dot)
        if angle < 1e-10:
            return np.zeros(3)
        if nc < 1e-9:
            # Anti-parallel degeneracy: choose any stable tangent axis.
            seed = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(seed, ac))) > 0.9:
                seed = np.array([0.0, 1.0, 0.0])
            axis = np.cross(ac, seed)
            axis /= max(float(np.linalg.norm(axis)), 1e-12)
        else:
            axis = c / nc
        return angle * axis

    @staticmethod
    def _tangent_basis(axis):
        """Return a 2x3 orthonormal basis perpendicular to ``axis``."""
        a = np.asarray(axis, dtype=float).reshape(3)
        a /= max(float(np.linalg.norm(a)), 1e-12)
        seed = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(seed, a))) > 0.85:
            seed = np.array([0.0, 1.0, 0.0])
        b1 = seed - a * float(np.dot(a, seed))
        b1 /= max(float(np.linalg.norm(b1)), 1e-12)
        b2 = np.cross(a, b1)
        b2 /= max(float(np.linalg.norm(b2)), 1e-12)
        return np.vstack((b1, b2))

    def _joint_limit_gradient(self, q):
        """Gradient pushing joints toward q_pref (∂H/∂q)."""
        return -(np.asarray(q) - self.q_pref) / (self.nj * self.q_range ** 2)

    def _manipulability_gradient(self, q, R, delta=1e-4):
        """Central-difference gradient of manipulability w.r.t. q."""
        grad = np.zeros(self.nj)
        pad  = [0.0] * (self._n_dof - self.nj)
        zero_vec = [0.0] * self._n_dof
        with self._lock:
            for i in range(self.nj):
                q_p = list(q) + pad; q_p[i] += delta
                q_m = list(q) + pad; q_m[i] -= delta
                jp_lin, jp_ang = self.p.calculateJacobian(
                    self.kuka_id, self.ee_index, [0, 0, 0],
                    q_p, zero_vec, zero_vec)
                jm_lin, jm_ang = self.p.calculateJacobian(
                    self.kuka_id, self.ee_index, [0, 0, 0],
                    q_m, zero_vec, zero_vec)
                Jp = np.vstack([
                    R @ np.array(jp_lin)[:, :self.nj],
                    R @ np.array(jp_ang)[:, :self.nj],
                ])
                Jm = np.vstack([
                    R @ np.array(jm_lin)[:, :self.nj],
                    R @ np.array(jm_ang)[:, :self.nj],
                ])
                grad[i] = (self._manipulability(self._S6 @ Jp)
                           - self._manipulability(self._S6 @ Jm)) / (2.0 * delta)
        return grad

    # ------------------------------------------------------------------
    # ADMM subproblem
    # ------------------------------------------------------------------

    def solve_clf_admm(self, a_clf, z_clf, lambda_clf, rho,
                       pos_desired=None, vel_desired=None, dt=1e-3,
                       step_count=0, task_share=None, v_compensation=None,
                       base_path_ee_velocity=None,
                       shareable_residual_velocity=None,
                       local_pos_desired=None, local_vel_desired=None,
                       local_quat_desired=None, local_rot_desired=None,
                       local_omega_desired=None, role_weight=1.0):
        """Solve the revised modular arm CLF-ADMM subproblem.

        The manipulator first computes an independent *local-frame* nominal
        velocity command.  The local task is expressed in the arm mounting
        frame, so the controller does not use the mobile-base command or an
        alpha-scaled Cartesian share.

        ADMM then optimises only the correction

            delta_dq_a = dq_a-dq_a^0

        with cost

            w_a/2 ||delta_dq_a||^2_{W_a}
            + rho/2 (a_a^T delta_dq_a-z_a+lambda_a)^2.

        ``task_share`` and the historical residual-allocation arguments are
        accepted only for API compatibility and are intentionally ignored.
        """
        q, qd, ee_curr, base_pos, R, J_world = self.get_state()
        self._last_q = np.asarray(q)
        self._last_qd = np.asarray(qd)
        self._last_J = J_world.copy()
        self._last_ee_pos = ee_curr.copy()
        self._last_base_pos = base_pos.copy()
        self._last_R = R.copy()

        # Arm-local kinematics: r_e^B and J_a^B.  This is the nominal local
        # controller interface discussed in Section 2.
        r_curr = R.T @ (ee_curr - base_pos)
        J_local = np.vstack([
            R.T @ J_world[:3, :],
            R.T @ J_world[3:, :],
        ])
        J_lin_local = J_local[:3, :]
        J_ang_local = J_local[3:, :]

        if local_pos_desired is None:
            # Compatibility fallback for callers that only provide a world
            # target.  The revised main loop supplies a planner-generated local
            # reference, so this branch is not used in the intended setup.
            if pos_desired is None:
                r_des = r_curr.copy()
            else:
                r_des = R.T @ (np.asarray(pos_desired, dtype=float) - base_pos)
        else:
            r_des = np.asarray(local_pos_desired, dtype=float).reshape(3)

        if local_vel_desired is None:
            rdot_des = np.zeros(3)
        else:
            rdot_des = np.asarray(local_vel_desired, dtype=float).reshape(3)

        kp = max(float(self.tracking_fraction), 1e-3) / max(float(dt), 1e-6)
        e_local = r_curr - r_des
        v_local_ref = rdot_des - kp * e_local
        v_local_norm = float(np.linalg.norm(v_local_ref))
        if v_local_norm > self.max_linear_ref:
            v_local_ref *= self.max_linear_ref / v_local_norm
        self._last_local_pos = r_curr.copy()
        self._last_local_pos_desired = r_des.copy()

        # V10.8 full orientation execution.  Stage-3 still PLANS on the
        # direction-only top-down manifold, so wrist yaw is chosen by the
        # redundant planner/IK.  Once a path is accepted, its full orientation
        # is a reachable reference and is tracked here as a 3-DOF SO(3) task.
        orientation_active = (local_rot_desired is not None or
                              local_quat_desired is not None)
        if orientation_active:
            with self._lock:
                _, base_quat = self.p.getBasePositionAndOrientation(self.kuka_id)
            q_base_inv = self._quat_conj(base_quat)
            q_cur_local = self._quat_mul(q_base_inv, self._last_ee_quat)
            R_cur_local = self._quat_to_matrix(q_cur_local)
            if local_rot_desired is not None:
                R_des_local = np.asarray(local_rot_desired, dtype=float).reshape(3, 3)
            else:
                R_des_local = self._quat_to_matrix(
                    np.asarray(local_quat_desired, dtype=float).reshape(4))

            e_rot = so3_log(R_des_local @ R_cur_local.T)
            if local_omega_desired is None:
                omega_ff = np.zeros(3)
            else:
                omega_ff = np.asarray(local_omega_desired, dtype=float).reshape(3)
            kp_rot = (max(self.orientation_tracking_fraction, 1e-4)
                      / max(float(dt), 1e-6))
            omega_local_ref = omega_ff + kp_rot * e_rot
            omega_norm = float(np.linalg.norm(omega_local_ref))
            if omega_norm > self.max_omega_ref:
                omega_local_ref *= self.max_omega_ref / omega_norm

            twist_local_ref = np.r_[v_local_ref, omega_local_ref]
            J_task = self._S6 @ J_local
            task_ref = self._S6 @ twist_local_ref
            sigma_min_task = float(
                np.linalg.svd(J_task, compute_uv=False)[-1])
            # Increase DLS damping only near a poorly conditioned full-6D
            # posture.  This suppresses the qdot spikes seen in the first
            # V10.8 physics run without changing the task definition.
            blend = float(np.clip(
                (self.adaptive_damping_sigma - sigma_min_task)
                / self.adaptive_damping_sigma, 0.0, 1.0))
            damping_eff = (
                self.damping
                + blend * (self.adaptive_damping_max - self.damping))
            JJT = J_task @ J_task.T + (damping_eff ** 2) * np.eye(6)
            J_pinv = J_task.T @ np.linalg.solve(JJT, np.eye(6))
            dq_primary = J_pinv @ task_ref
            N = np.eye(self.nj) - J_pinv @ J_task

            a_cur = R_cur_local[:, 2]
            a_des = R_des_local[:, 2]
            tool_axis_error = self._axis_error_rotvec(a_des, a_cur)
            self._last_orientation_error = e_rot.copy()
            self._last_tool_axis_error = tool_axis_error.copy()
            self._last_omega_arm_local = omega_local_ref.copy()
            self._last_local_error = np.r_[e_local, -e_rot]
        else:
            JJT = J_lin_local @ J_lin_local.T + (self.damping ** 2) * np.eye(3)
            J_pinv = J_lin_local.T @ np.linalg.solve(JJT, np.eye(3))
            dq_primary = J_pinv @ v_local_ref
            N = np.eye(self.nj) - J_pinv @ J_lin_local
            omega_local_ref = np.zeros(3)
            self._last_orientation_error = np.zeros(3)
            self._last_tool_axis_error = np.zeros(3)
            self._last_omega_arm_local = np.zeros(3)
            self._last_local_error = e_local.copy()

        self._last_v_arm_local = np.r_[v_local_ref, omega_local_ref]
        self._last_v_arm = np.r_[R @ v_local_ref, R @ omega_local_ref]
        self._last_v_task = self._last_v_arm.copy()
        self._last_v_path_ee = np.zeros(6)
        self._last_v_residual = np.zeros(6)
        self._last_v_shareable = np.zeros(6)
        self._last_v_mandatory = np.zeros(6)

        grad_jl = self._joint_limit_gradient(q)
        self.w_current = self._manipulability(self._S6 @ J_local)
        if step_count % self.manip_rate == 0:
            # Existing helper works in world frame; manipulability is invariant
            # to the base rotation, so the gradient remains valid.
            self._grad_manip = self._manipulability_gradient(q, R)
        dq_raw = dq_primary + N @ (
            self.k_jl * grad_jl + self.k_manip * self._grad_manip)

        # Complete local feasible box: velocity, one-step position viability,
        # directional joint-limit guard, and optional physical acceleration
        # limits.  No command-smoothing cost is used.
        lower = -self.max_dq * np.ones(self.nj, dtype=float)
        upper =  self.max_dq * np.ones(self.nj, dtype=float)
        dt_safe = max(float(dt), 1e-6)
        q_arr = np.asarray(q, dtype=float)
        lower = np.maximum(lower, (self.q_lower - q_arr) / dt_safe)
        upper = np.minimum(upper, (self.q_upper - q_arr) / dt_safe)

        margin = math.radians(5.0)
        near_lower = q_arr <= self.q_lower + margin
        near_upper = q_arr >= self.q_upper - margin
        lower[near_lower] = np.maximum(lower[near_lower], 0.0)
        upper[near_upper] = np.minimum(upper[near_upper], 0.0)

        safety_lower = lower.copy()
        safety_upper = upper.copy()
        accel_fallback = np.zeros(self.nj, dtype=bool)
        if self.acc_limit is not None:
            acc = float(self.acc_limit)
            acc_lower = self._dq_prev - acc * dt_safe
            acc_upper = self._dq_prev + acc * dt_safe
            lower = np.maximum(lower, acc_lower)
            upper = np.minimum(upper, acc_upper)
            infeasible = lower > upper
            if np.any(infeasible):
                accel_fallback[infeasible] = True
                lower[infeasible] = safety_lower[infeasible]
                upper[infeasible] = safety_upper[infeasible]

        infeasible = lower > upper
        if np.any(infeasible):
            for j in np.where(infeasible)[0]:
                safe = float(np.clip(0.0, safety_lower[j], safety_upper[j]))
                lower[j] = safe
                upper[j] = safe

        # Local nominal command: DLS/null-space request projected onto the
        # arm's own feasible box.  Crucially, qdot_a^0 is independent of the
        # adaptive role preference.
        dq_nom = np.clip(dq_raw, lower, upper)
        nom_info = {"converged": True, "method": "direct_clip"}

        # Dimensionless correction metric.  Each joint correction is scaled
        # by its physical velocity authority.  Thus a 50% correction on one
        # joint has the same basic quadratic meaning as a 50% base-actuator
        # correction before the scalar role weights are applied.  The current
        # position/acceleration margin is intentionally excluded here because
        # it is already represented by the hard feasible box and gamma_a.
        max_dq_arr = np.asarray(self.max_dq, dtype=float)
        if max_dq_arr.ndim == 0:
            dq_scale = np.full(self.nj, max(abs(float(max_dq_arr)), 1e-6))
        else:
            dq_scale = np.maximum(np.abs(max_dq_arr).reshape(self.nj), 1e-6)
        Wa = np.diag(1.0 / (dq_scale ** 2))
        w_role = max(float(role_weight), 1e-9)
        H_corr = w_role * Wa

        a = np.asarray(a_clf, dtype=float).reshape(self.nj)
        H = H_corr + rho * np.outer(a, a)
        shifted_target = (float(z_clf) - float(lambda_clf)
                          + float(a @ dq_nom))
        rhs = H_corr @ dq_nom + rho * a * shifted_target

        dq, qp_info = solve_box_qp(
            H + 1e-9*np.eye(self.nj), rhs, lower, upper,
            x0=dq_nom, tol=1e-9, max_sweeps=200)

        delta_dq = dq - dq_nom
        s_corr = float(a @ delta_dq)
        self._last_qp_model = {
            "H": (H_corr + 1e-9*np.eye(self.nj)).copy(),
            "rhs": (H_corr @ dq_nom).copy(),
            "lower": lower.copy(),
            "upper": upper.copy(),
            "x_nom": dq_nom.copy(),
            "role_weight": float(w_role),
            "correction_scale": dq_scale.copy(),
            "correction_metric_diag": np.diag(Wa).copy(),
        }
        self._last_qp_info = {
            **qp_info,
            "lower": lower.copy(),
            "upper": upper.copy(),
            "accel_fallback": accel_fallback.copy(),
            "smooth_weight": 0.0,
            "nominal_qp_converged": bool(nom_info.get("converged", True)),
            "role_weight": float(w_role),
            "correction_scale": dq_scale.copy(),
            "correction_metric_diag": np.diag(Wa).copy(),
            "correction_cost": float(0.5*w_role*(delta_dq @ Wa @ delta_dq)),
        }
        return dq, s_corr, dq_nom

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

    def set_dynamic_compensation(self, dyn_comp):
        """Attach a low-level dynamic compensation layer.

        When set, ``apply()`` delegates to the dynamic layer (torque
        control) instead of using PyBullet's built-in position controller.

        Parameters
        ----------
        dyn_comp : DynamicCompensation or None
            The computed-torque layer.  Pass None to revert to
            position-control mode.
        """
        self._dyn_comp = dyn_comp

    def apply(self, dq_cmd, dt):
        """Execute joint velocity commands via physics-based velocity control.

        Uses PyBullet VELOCITY_CONTROL so joint inertia and friction from
        the physics engine affect the actual motion.  The physics engine
        determines how quickly each joint reaches the target velocity.

        Parameters
        ----------
        dq_cmd : ndarray (nj,)
            Joint velocity command (rad/s).
        dt : float
            Timestep — unused here, kept for API compatibility.
        """
        self._dq_prev = np.asarray(dq_cmd, dtype=float).copy()
        with self._lock:
            for idx, ji in enumerate(self._joint_ids):
                self.p.setJointMotorControl2(
                    bodyIndex=self.kuka_id, jointIndex=ji,
                    controlMode=self.p.VELOCITY_CONTROL,
                    targetVelocity=float(dq_cmd[idx]),
                    force=self.joint_force)
