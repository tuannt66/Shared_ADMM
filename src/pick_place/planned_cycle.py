"""Planned-trajectory pick-place executor (progress-based execution).

Replaces the event-driven ``Phase`` state machine of
``pick_place_cycle.py`` with the hybrid architecture:

  * **Motion planning layer** — long transfers use RRT-Connect in the
    Kuka's 7-D joint space, while the terminal hover→grasp/drop motion is
    generated as a collision-checked Cartesian IK line with a top-down tool
    axis.  Post-action motion first reverses this vertical line back to hover;
    selected long retract/transfer edges then use a tool-axis-constrained,
    IK-projected RRT-Connect so top-down tilt is preserved while wrist yaw
    remains free along the joint-space path.  Planning runs in a *shadow*
    PyBullet client (the real robot is never touched during planning).

  * **Progress-based execution** — each segment owns a progress
    variable ``s`` integrated in time with an adjustable rate ds/dt
    (slowed by arm tracking error and base lag, like the old
    ``_adaptive_speed``).  MOTION segments hand over at ``s >= 1``.
    ACTION segments (grasp / release) additionally require the *actual*
    EE position (read from the simulator) to be within
    ``grasp_tolerance`` of the target; otherwise the trajectory
    **dwells** at s = 1 (arm keeps servoing to the final target through
    the unchanged DLS/ADMM pipeline).  A dwell longer than
    ``dwell_timeout`` aborts the attempt and retries next lap — the old
    "missed pre-reach window" path, now an exception instead of the
    main flow.

  * The base follows one **analytic circular path** (no waypoint/polyline
    approximation).  Geometry is obstacle-free (R_TABLE < R_CIRCLE), and
    the reference is evaluated directly as p_d(theta_ref) from sin/cos.

The ADMM consensus pipeline is untouched: this class only changes
*where* ``arm_target`` / feedforward speeds come from each tick.
"""

import math
import os
import sys
import time as _time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'planning'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'shared'))

from rrt_connect_joint import RRTConnectJointPlanner, shortcut_path
from time_param import ProgressTrajectory, SampledPath3
from reach_safety_cbf import ReachSafetyCBF
from task_space_6d import so3_log, so3_exp, rate_limit_rotation


def _smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _tracking_progress_factor(error_m, soft_m=0.030, hard_m=0.090):
    """Continuous slowdown factor for tracking-sensitive path segments.

    1.0 below ``soft_m`` and 0.0 at/above ``hard_m``.
    The caller adds hysteresis around the hard stop so the progress state does
    not chatter when the measured tracking error sits near the boundary.
    """
    e = max(float(error_m), 0.0)
    soft = max(float(soft_m), 0.0)
    hard = max(float(hard_m), soft + 1e-6)
    if e <= soft:
        return 1.0
    if e >= hard:
        return 0.0
    x = (e - soft) / (hard - soft)
    return 1.0 - _smoothstep(x)


class AdaptiveBaseSpeedGenerator:
    """Path/state-driven base speed scheduler.

    The scheduler is intentionally independent of the arm/base role index
    ``alpha`` and of task phase names.  It uses only path geometry and the
    distance to a planner-declared stop event:

        v_curve = sqrt(a_lat_max / (|kappa| + eps))
        v_stop  = sqrt(2 a_dec_max d_rem)
        v_des   = min(v_max, v_curve, v_stop)

    ``v_stop`` is active only when the current segment must settle at its
    endpoint (currently an ACTION segment).  For ordinary transit/motion
    pieces it is +inf, so no artificial stop is inserted at every segment
    boundary.
    """

    def __init__(self, v_max=0.35, v_min=0.08,
                 a_lat_max=0.05, a_dec_max=0.35, eps=1e-9):
        self.v_max = max(float(v_max), 1e-6)
        self.v_min = max(0.0, min(float(v_min), self.v_max))
        self.a_lat_max = max(float(a_lat_max), 1e-6)
        self.a_dec_max = max(float(a_dec_max), 1e-6)
        self.eps = max(float(eps), 1e-12)
        self.last = {
            "v_des": self.v_max,
            "v_curve": self.v_max,
            "v_stop": float("inf"),
            "curvature": 0.0,
            "remaining": float("inf"),
            "stop_required": False,
        }

    def compute(self, curvature, remaining_distance, stop_required=False):
        kappa = abs(float(curvature))
        d_rem = max(float(remaining_distance), 0.0)

        v_curve = math.sqrt(self.a_lat_max / (kappa + self.eps))
        v_stop = (math.sqrt(2.0 * self.a_dec_max * d_rem)
                  if stop_required else float("inf"))
        v_des = min(self.v_max, v_curve, v_stop)

        # Away from a required stop, retain a small locomotion floor.
        # At a stop event v_des is allowed to approach zero naturally.
        if not stop_required:
            v_des = max(v_des, self.v_min)
        v_des = max(0.0, min(v_des, self.v_max))

        self.last = {
            "v_des": float(v_des),
            "v_curve": float(v_curve),
            "v_stop": float(v_stop),
            "curvature": float(kappa),
            "remaining": float(d_rem),
            "stop_required": bool(stop_required),
        }
        return float(v_des)


def _rot2(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


class AnalyticCirclePath:
    """Continuous analytic base path with no waypoint/polyline geometry.

    The reference is parameterized by an unwrapped angular coordinate theta:

        p_d(theta)   = [cx + R cos(theta), cy + R sin(theta)]
        psi_d(theta) = theta + pi/2
        kappa        = 1/R

    Task phases may change the progress rate, but never the path geometry.
    """

    def __init__(self, cx, cy, radius):
        self.cx = float(cx)
        self.cy = float(cy)
        self.radius = float(radius)
        if abs(self.radius) < 1e-9:
            raise ValueError("circle radius must be non-zero")

    def position(self, theta):
        th = float(theta)
        return np.array([self.cx + self.radius * math.cos(th),
                         self.cy + self.radius * math.sin(th)], dtype=float)

    def tangent_heading(self, theta):
        return _wrap(float(theta) + math.pi / 2.0)

    def curvature(self, theta=None):
        del theta
        return 1.0 / abs(self.radius)

    def feedforward(self, theta_dot):
        thd = float(theta_dot)
        return abs(self.radius) * thd, thd


class _Segment:
    """One progress-parameterized piece of the cycle trajectory."""

    def __init__(self, name, kind, th0, th1, duration,
                 traj=None, ee_local=None, ee_pose_local=None,
                 anchor_start=None, anchor_end=None,
                 action=None, target_world=None, final_of_visit=False,
                 min_duration=None, orientation_local_quat=None,
                 cartesian_world=None):
        self.name       = name        # reuses the old phase names
        self.kind       = kind        # 'carry' | 'motion' | 'action'
        self.th0        = th0         # unwrapped base angle at s=0
        self.th1        = th1         # unwrapped base angle at s=1
        self.duration   = max(float(duration), 1e-3)
        # ``duration`` is retained for compatibility/reporting.  The new
        # adaptive speed scheduler uses ``min_duration`` only as an arm-motion
        # safety cap on ds/dt; it no longer uses phase-specific base speeds.
        self.min_duration = max(float(
            self.duration if min_duration is None else min_duration), 1e-3)
        self.traj       = traj        # ProgressTrajectory (joint space)
        self.ee_local   = ee_local    # SampledPath3 (arm-base frame)
        self.ee_pose_local = ee_pose_local  # legacy compatibility; unused in V10.9
        self.anchor_start = anchor_start   # None | ('world', p3) | ('body', off3)
        self.anchor_end   = anchor_end
        self.action     = action      # None | ('grasp', ti) | ('release', ti)
        self.target_world = target_world   # exact world point for the action
        # Optional local-frame EE orientation target.  Keeping the target in
        # the arm mounting frame makes top-down grasping invariant to base yaw.
        self.orientation_local_quat = (None if orientation_local_quat is None
                                       else np.asarray(
                                           orientation_local_quat,
                                           dtype=float).reshape(4))
        # Optional exact world Cartesian line segment.  ACTION descend paths
        # use this instead of the FK image of a joint-space path so the final
        # approach is geometrically vertical even while the base is moving.
        self.cartesian_world = (None if cartesian_world is None else
                                (np.asarray(cartesian_world[0], dtype=float).reshape(3),
                                 np.asarray(cartesian_world[1], dtype=float).reshape(3)))
        self.final_of_visit = final_of_visit
        self.place_xy   = None        # desired CUBE landing point (release)
        self.aim_count  = 0           # closed-loop release aim iterations
        self.last_aim_time = None      # require a fresh settle after each aim shift
        self.aim_exhausted = False     # hard gate holds if max aim attempts reached
        self.aim_exhausted_reported = False

    def __repr__(self):
        return (f"<Segment {self.name} kind={self.kind} "
                f"th=[{self.th0:.2f},{self.th1:.2f}] dur={self.duration:.2f}s>")


class PlannedCycleExecutor:
    """Drop-in replacement for ``PickPlaceCycleSM``.

    Exposes the same surface the coordination loop uses:
    ``update()``, ``phase_name``, ``finished``, ``_t0``, ``carried``,
    ``place_count``, ``table_items``, ``target_table`` — plus
    ``base_guidance()`` (replaces the waypoint follower) and
    ``post_tick()`` / ``report()`` for grasp/placement verification.
    """

    # ---- schedule geometry (rad on the drive circle) ----
    REACH_ARC  = 0.44     # base arc covered while the arm reaches out
    ACT_SPAN   = 0.12     # arc covered during the descend window
    LIFT_ARC   = 0.15     # base arc covered while the arm lifts back
    LIFT_VERT_FRACTION = 0.40  # split lift into vertical + RRT diagnostics
    TRANSFER_CLEARANCE = 0.07  # short vertical clearance after place before next pick (m)
    # Comfortable horizontal base-to-target distance for the action.
    # The descend window ENDS where the base passes at this distance
    # (behind the target when it lies close to the circle, abeam when
    # it is far inside), so the dwell/grasp happens exactly there.
    # Too small a standoff folds the elbow (j3) against its limit and
    # the 5-deg hard clamp freezes the arm ~45 mm short (measured).
    D_COMFORT  = 0.62

    # ---- nominal base speeds per segment type (m/s) ----
    V_TRANSIT = 0.35
    V_REACH   = 0.22
    V_ACT     = 0.12
    V_LIFT    = 0.25

    # ---- legacy action-settle parameters ----
    # V10.4 no longer uses pre-action dwell or EE/object velocity as action
    # gates.  These constants are retained only for backwards-compatible
    # diagnostics/CLI parsing and are not used in the decision logic.
    DEFAULT_GRASP_SETTLE   = 0.0
    DEFAULT_RELEASE_SETTLE = 0.0
    EE_STATIC              = 0.06  # diagnostic threshold only

    # ---- planner parameters ----
    MARGIN_TRANSFER = 0.02    # carry <-> hover clearance (m)
    MARGIN_ACT      = 0.01    # hover <-> grasp clearance (m)
    RRT_STEP        = 0.25    # rad
    RRT_EDGE_RES    = 0.06    # rad
    RRT_MAX_ITERS   = 3000

    def __init__(self, sim, sim_lock,
                 husky_id, kuka_id, ee_index, joint_indices,
                 q_lower, q_upper, shadow,
                 cube_ids, circle, table_surface, object_half,
                 place_target_fn, carry_offset_body,
                 cruise_speed=0.35, min_speed=0.02,
                 grasp_tolerance=0.05, release_tolerance=0.025,
                 dwell_timeout=2.5, grasp_settle=0.10, release_settle=0.20,
                 release_clearance=0.015, release_tilt_max_deg=15.0,
                 release_ang_static=0.35,
                 grasp_z_off=0.05, hover_dz=0.15,
                 rng_seed=0, dyn_dist=None,
                 reach_safe_radius=None, reach_safe_alpha=2.0,
                 adaptive_speed=True, speed_a_lat_max=0.05,
                 speed_a_dec_max=0.35, allow_reverse=True,
                 action_ref_filter=True, action_ref_wn=35.0,
                 top_down_grasp=True,
                 top_down_local_euler=(math.pi, 0.0, 0.0),
                 preferred_grasp_q=None,
                 full6d_orientation_rate=0.50,
                 full6d_progress_length=0.20,
                 full6d_handover_pos_tol=0.045,
                 full6d_handover_rot_tol_deg=12.0,
                 full6d_action_rot_tol_deg=8.0,
                 rrt_cart_ref_speed=0.18,
                 rrt_joint_path_speed=0.55,
                 reach_cart_ref_speed=0.22,
                 reach_joint_path_speed=0.68,
                 tracking_soft_error=0.035,
                 tracking_hard_error=0.100,
                 tracking_resume_error=0.055,
                 base_decoupled=True,
                 base_work_speed=0.36,
                 base_reach_speed=0.24,
                 base_lift_speed=0.22,
                 base_action_speed=0.10,
                 base_ref_accel=0.55,
                 base_ref_decel=0.80,
                 base_ref_max_lead=0.18):
        self.p         = sim
        self._lock     = sim_lock
        self.husky_id  = husky_id
        self.kuka_id   = kuka_id
        self.ee_index  = ee_index
        self.joint_ids = list(joint_indices)
        self.q_lower   = np.asarray(q_lower, float)
        self.q_upper   = np.asarray(q_upper, float)
        self.shadow    = shadow
        self.CX, self.CY, self.R = circle
        self.base_path = AnalyticCirclePath(self.CX, self.CY, self.R)
        self.table_surface = table_surface
        self.object_half   = object_half
        self.place_target_fn = place_target_fn
        self.carry_offset_body = np.asarray(carry_offset_body, float)
        self.cruise = cruise_speed
        self.min_speed = min_speed
        self.adaptive_speed = bool(adaptive_speed)
        self.allow_reverse = bool(allow_reverse)
        # V10.2 implementation-layer reference shaping.  The theoretical
        # local QP remains unchanged; only sudden terminal ACTION target
        # changes (notably release re-aim steps) are passed through a
        # critically damped second-order command filter before entering
        # the local controller and global CLF reference.
        self.action_ref_filter = bool(action_ref_filter)
        self.action_ref_wn = max(float(action_ref_wn), 1e-6)
        self._action_ref_pos = None
        self._action_ref_vel = np.zeros(3, dtype=float)
        self.speed_generator = AdaptiveBaseSpeedGenerator(
            v_max=self.cruise, v_min=self.min_speed,
            a_lat_max=speed_a_lat_max, a_dec_max=speed_a_dec_max)
        self._speed_sched = self.cruise
        self.grasp_tolerance   = grasp_tolerance
        self.release_tolerance = release_tolerance
        self.dwell_timeout     = dwell_timeout
        self.grasp_settle       = max(0.0, float(grasp_settle))
        self.release_settle     = max(0.0, float(release_settle))
        # Keep the constrained cube slightly above the table while waiting
        # for its pendulum orientation to settle.  Releasing from exact
        # contact height allowed a tilted cube to become statically wedged
        # against the table/constraint and then slide after detach.
        self.release_clearance = max(0.0, float(release_clearance))
        self.release_tilt_max_deg = max(0.0, float(release_tilt_max_deg))
        self.release_ang_static = max(0.0, float(release_ang_static))
        self.grasp_z_off = grasp_z_off
        self.hover_dz    = hover_dz
        self.top_down_grasp = bool(top_down_grasp)
        self.top_down_local_quat = np.asarray(
            self.p.getQuaternionFromEuler(list(top_down_local_euler)),
            dtype=float)
        # V10.9 smooth full-6D reference governor.  Stage-3 still
        # plans the tool-axis manifold, but execution uses a deliberately
        # smooth full orientation instead of inheriting arbitrary wrist yaw
        # from every RRT sample.
        self.full6d_orientation_rate = max(
            float(full6d_orientation_rate), 1e-3)
        self.full6d_progress_length = max(
            float(full6d_progress_length), 1e-4)
        self.full6d_handover_pos_tol = max(
            float(full6d_handover_pos_tol), 1e-4)
        self.full6d_handover_rot_tol = math.radians(
            max(float(full6d_handover_rot_tol_deg), 0.1))
        self.full6d_action_rot_tol = math.radians(
            max(float(full6d_action_rot_tol_deg), 0.1))

        # V10.11: tracking-aware time parameterization.  These limits shape
        # only planner progress / reference speed.  The 6D CLF-ADMM controller
        # itself is unchanged.
        self.rrt_cart_ref_speed = max(float(rrt_cart_ref_speed), 1e-3)
        self.rrt_joint_path_speed = max(float(rrt_joint_path_speed), 1e-3)
        self.reach_cart_ref_speed = max(float(reach_cart_ref_speed), 1e-3)
        self.reach_joint_path_speed = max(float(reach_joint_path_speed), 1e-3)
        self.tracking_soft_error = max(float(tracking_soft_error), 0.0)
        self.tracking_hard_error = max(
            float(tracking_hard_error), self.tracking_soft_error + 1e-3)
        self.tracking_resume_error = min(
            max(float(tracking_resume_error), 0.0),
            self.tracking_hard_error - 1e-3)

        # V10.11: the base path reference has its OWN progress state.  Arm
        # tracking slowdown is no longer allowed to reduce the nominal Husky
        # path speed.  The base still deliberately slows/stops at physical
        # grasp/drop windows.
        self.base_decoupled = bool(base_decoupled)
        self.base_work_speed = max(float(base_work_speed), 0.0)
        self.base_reach_speed = max(float(base_reach_speed), 0.0)
        self.base_lift_speed = max(float(base_lift_speed), 0.0)
        self.base_action_speed = max(float(base_action_speed), 0.0)
        self.base_ref_accel = max(float(base_ref_accel), 1e-3)
        self.base_ref_decel = max(float(base_ref_decel), 1e-3)
        self.base_ref_max_lead = max(float(base_ref_max_lead), 0.02)

        self._R_topdown_local = np.array(
            self.p.getMatrixFromQuaternion(
                self.top_down_local_quat.tolist()), dtype=float).reshape(3, 3)
        self.preferred_grasp_q = (None if preferred_grasp_q is None else
                                  np.asarray(preferred_grasp_q,
                                             dtype=float).reshape(len(self.joint_ids)))
        self._rng_seed   = rng_seed
        self._dyn_dist   = dyn_dist
        # CBF-QP reference governor: keeps the commanded EE reference
        # within a safe working radius of the arm base (see
        # reach_safety_cbf.py) instead of the raw, world-anchored
        # blend potentially leading the base into a near-singular
        # over-reach.  None = disabled (reference passed through as-is).
        self.reach_safety = (ReachSafetyCBF(reach_safe_radius, reach_safe_alpha)
                             if reach_safe_radius is not None else None)

        # ---- task state (same semantics as the old SM) ----
        self.table_items = {i: [cube_ids[i]] for i in range(len(cube_ids))}
        self.carried     = None
        self.grasp_cid   = None
        # Constraint-defined cube-centre offset from the EE frame origin.
        # With the orientation-decoupled point suspension used in V10.5,
        # the nominal carried-object geometry is [0, 0, -grasp_z_off] in
        # gravity/world coordinates.  This is task geometry, not an object
        # feedback gate.
        self._grasp_offset_world = None
        self.place_count = 0
        self.target_table = 0
        self.finished    = False
        self._t0         = 0.0
        self._place_target = None

        # Visit plan: (table, do_place, do_pick)
        self.visits = [(0, False, True), (1, True, True),
                       (2, True, True), (0, True, False)]
        self._visit_idx = 0
        self._retry_offsets = 0.0   # extra laps from aborted attempts

        # ---- verification bookkeeping (Definition-of-Done metrics) ----
        self.events = []            # grasp/release event dicts
        self.grasp_monitors = []    # cube-follows-EE checks after grasp
        self.placements = []        # release target vs final cube pos
        self.timeout_events = []    # dwell timeouts (DoD #5 wants zero)
        self.plan_fallbacks = []    # planner degradations (report loudly)
        self.plan_times = []        # seconds per visit planning
        self.base_track_err = 0.0   # |base_xy - circle(theta_ref)| this tick
        self.dwell_frac_last = 0.0

        # ---- kinematic calibration for nominal-pose prediction ----
        with self._lock:
            h_pos, h_orn = sim.getBasePositionAndOrientation(husky_id)
            k_pos, k_orn = sim.getBasePositionAndOrientation(kuka_id)
        h_yaw = sim.getEulerFromQuaternion(h_orn)[2]
        k_yaw = sim.getEulerFromQuaternion(k_orn)[2]
        self._kuka_rel_xy = _rot2(h_yaw).T @ (np.array(k_pos[:2])
                                              - np.array(h_pos[:2]))
        self._kuka_z       = k_pos[2]
        self._kuka_yaw_rel = _wrap(k_yaw - h_yaw)

        # ---- progress state ----
        th_now = math.atan2(h_pos[1] - self.CY, h_pos[0] - self.CX)
        self._theta_base_wrapped  = th_now
        self._theta_base = th_now      # unwrapped, monotone-ish
        self._theta_ref  = th_now
        self._theta_dot  = 0.0
        self._base_ref_v = 0.0
        self._base_ref_diag = {
            "base_progress_decoupled": bool(self.base_decoupled),
            "base_reference_theta": float(th_now),
            "base_reference_speed": 0.0,
            "base_reference_target_speed": 0.0,
            "base_reference_horizon_theta": float(th_now),
            "base_reference_horizon_distance_m": 0.0,
            "base_reference_stop_required": False,
            "base_reference_lead_m": 0.0,
            "base_reference_phase_cap": 0.0,
        }
        self.s           = 0.0
        self._rate_ema   = None
        self._progress_pause = False
        self._rate_diag = {
            "progress_rate_raw": 0.0,
            "progress_rate_filtered": 0.0,
            "progress_rate_cart_cap": float("inf"),
            "progress_rate_joint_cap": float("inf"),
            "progress_tracking_factor": 1.0,
            "progress_paused": False,
            "progress_dpds_m": 0.0,
            "segment_min_duration_s": 0.0,
        }
        self._dwell_since = None
        self._t_prev     = None
        self._planned    = False
        self._q_carry    = None
        self._queue      = []
        self._last_target = None
        self._vel_ff     = np.zeros(3)   # EE velocity feedforward (EMA)
        # Planner-side arm-local reference used by the modular arm controller.
        # This is a reference-generation state, not a command smoother.
        self._local_ref_prev = None
        self._local_ref_phase = None
        self._local_rot_prev = None

        # V10.8 full-orientation reference state.  Planner constraints remain
        # tool-axis-only, but execution tracks the full orientation selected
        # by the accepted joint path.
        with self._lock:
            _ls0 = self.p.getLinkState(
                self.kuka_id, self.ee_index, computeForwardKinematics=True)
            _kpos0, _korn0 = self.p.getBasePositionAndOrientation(self.kuka_id)
        _R_ee0 = np.array(
            self.p.getMatrixFromQuaternion(_ls0[5])).reshape(3, 3)
        _R_b0 = np.array(
            self.p.getMatrixFromQuaternion(_korn0)).reshape(3, 3)
        self._carry_R_local = _R_b0.T @ _R_ee0
        self._rot_target_world = _R_ee0.copy()
        self._rot_prev_world = _R_ee0.copy()
        self._omega_ff = np.zeros(3)
        self._omega_local_ff = np.zeros(3)
        self._R_base_prev_world = _R_b0.copy()
        self._orientation_phase = None
        self._orientation_phase_start_local = self._carry_R_local.copy()
        self._orientation_phase_goal_local = self._carry_R_local.copy()
        self._orientation_cmd_local = self._carry_R_local.copy()
        self._actual_ee_rot_world = _R_ee0.copy()
        self._last_orientation_track_error = 0.0
        self._last_combined_track_error = 0.0

        self._ee_prev    = None
        self._ee_speed   = 0.0           # actual EE speed (EMA, m/s)
        self._action_diag = self._empty_action_diag()
        self._seg = self._make_transit()

    # ==================================================================
    # Public API used by the coordination loop
    # ==================================================================

    @property
    def phase_name(self):
        return self._seg.name

    @property
    def base_reference_xy(self):
        """Current world-frame XY reference for the mobile base path.

        This is the geometric point ``circle(theta_ref)`` used by
        ``base_guidance()`` and by ``base_track_err``.  It is intentionally
        distinct from the Cartesian EE/base-local task reference passed to
        the CLF-ADMM local base controller.
        """
        return self._circle_pt(self._theta_ref).copy()

    def _shape_terminal_action_reference(self, raw_target, dt, active):
        """Critically damp sudden terminal ACTION reference changes.

        The filter is deliberately an implementation/reference-generation
        element, not part of the local optimization objective.  It is active
        only once an ACTION segment has reached its terminal dwell (s=1),
        where release re-aim can otherwise step the EE target by up to 10 mm
        in one 240-Hz tick.  Outside that condition the state is synchronized
        to the raw planner target so ordinary planned trajectories are not
        delayed.

        For each Cartesian axis the continuous model is

            x_ddot + 2*w_n*x_dot + w_n^2*(x-r) = 0,

        with zeta=1.  The update below is the exact discrete solution for a
        piecewise-constant target over one control interval, avoiding Euler
        stability/tuning artefacts.
        """
        r = np.asarray(raw_target, dtype=float).reshape(3)
        h = max(float(dt), 0.0)

        if (not self.action_ref_filter) or (not active) or h <= 1e-9:
            self._action_ref_pos = r.copy()
            self._action_ref_vel = np.zeros(3, dtype=float)
            return r.copy(), np.zeros(3, dtype=float)

        if self._action_ref_pos is None:
            self._action_ref_pos = r.copy()
            self._action_ref_vel = np.zeros(3, dtype=float)
            return r.copy(), np.zeros(3, dtype=float)

        x = self._action_ref_pos
        v = self._action_ref_vel
        w = self.action_ref_wn
        a = w * h
        decay = math.exp(-a)
        e = x - r
        e_next = decay * ((1.0 + a) * e + h * v)
        v_next = decay * ((1.0 - a) * v - (w * w) * h * e)
        x_next = r + e_next

        self._action_ref_pos = x_next
        self._action_ref_vel = v_next
        return x_next.copy(), v_next.copy()

    def arm_local_reference(self, arm_target_world, dt, phase_name=None):
        """Return the arm task expressed in the *nominal arm-base frame*.

        The local arm controller must not depend on the mobile-base command.
        The planner therefore converts its world-space EE target into the
        current arm-mount frame.  The planner may use system geometry to
        express a task locally, but the arm controller itself receives only
        this local reference.  During TRANSIT/DONE the body-anchored carry
        target therefore becomes constant in the arm frame, so the arm does
        not actively cancel base motion.

        Returns
        -------
        r_des : ndarray (3,)
            Desired EE position relative to the nominal arm mounting frame.
        rdot_des : ndarray (3,)
            Feedforward local-reference velocity.  It is zero in base-dominant
            carry/transit operation and finite-differenced in manipulation
            segments.
        """
        target = np.asarray(arm_target_world, dtype=float).reshape(3)
        pos_local, R_local = self._kuka_pose()
        r_des = R_local.T @ (target - np.asarray(pos_local, dtype=float))

        phase = self.phase_name if phase_name is None else str(phase_name)
        dt_safe = max(float(dt), 1e-9)
        base_dominant_hold = phase in ("TRANSIT", "DONE")
        if (base_dominant_hold or self._local_ref_prev is None or
                self._local_ref_phase != phase):
            rdot_des = np.zeros(3, dtype=float)
        else:
            rdot_des = (r_des - self._local_ref_prev) / dt_safe
            n = float(np.linalg.norm(rdot_des))
            if n > 0.8:
                rdot_des *= 0.8 / n

        self._local_ref_prev = r_des.copy()
        self._local_ref_phase = phase
        return r_des, rdot_des

    def arm_local_pose_reference(self, arm_target_world, dt, phase_name=None):
        """Return one consistent local SE(3) reference.

        The world/global and local arm controllers use the SAME smoothed
        orientation command produced by :meth:`update`.  No RRT-sample wrist
        yaw is reintroduced here.
        """
        r_des, rdot_des = self.arm_local_reference(
            arm_target_world, dt, phase_name=phase_name)

        _, R_base = self._kuka_pose()
        R_des_local = R_base.T @ self._rot_target_world
        # The world target follows mobile-base yaw by construction.  The arm
        # local task must track only the orientation motion *relative to the
        # mount*, not cancel the base angular velocity.
        omega_des_local = self._omega_local_ff.copy()
        return r_des, rdot_des, R_des_local, omega_des_local

    @staticmethod
    def _empty_action_diag():
        return {
            "release_aim_count": 0,
            "release_cube_residual": float("nan"),
            "release_cube_residual_xy": np.array([np.nan, np.nan]),
            "release_cube_speed": float("nan"),
            "release_cube_ang_speed": float("nan"),
            "release_cube_tilt_ang_speed": float("nan"),
            "release_cube_tilt_deg": float("nan"),
            "release_cube_aligned": False,
            "release_cube_upright": False,
            "release_gate_ready": False,
            "release_aim_exhausted": False,
            "action_dwell_elapsed": 0.0,
            "action_settle_elapsed": 0.0,
            "action_ref_filter_active": False,
            "action_terminal_dwell": False,
            "action_ref_filter_error": 0.0,
            "action_ref_filter_speed": 0.0,
            "full6d_action_orientation_ready": False,
            "full6d_action_orientation_error_deg": float("nan"),
        }

    @property
    def orientation_reference_world(self):
        return self._rot_target_world.copy()

    @property
    def angular_velocity_reference_world(self):
        return self._omega_ff.copy()

    @property
    def full6d_tracking_diagnostics(self):
        return {
            "combined_error_m": float(self._last_combined_track_error),
            "orientation_error_rad": float(
                self._last_orientation_track_error),
            "orientation_error_deg": float(math.degrees(
                self._last_orientation_track_error)),
            "orientation_rate_limit": float(
                self.full6d_orientation_rate),
        }

    @property
    def action_diagnostics(self):
        """Latest grasp/release gate diagnostics for the step logger."""
        out = {}
        for k, v in self._action_diag.items():
            out[k] = v.copy() if isinstance(v, np.ndarray) else v
        return out

    def update(self, t, ee_pos, base_xy, wp_follower=None, ee_rot=None):
        """Advance progress and return
        ``(arm_target, arm_vel, base_speed_nominal, segment_name)``.

        Call once per control tick BEFORE ``base_guidance()``.
        """
        dt = 0.0 if self._t_prev is None else max(0.0, t - self._t_prev)
        self._t_prev = t
        self._update_theta_base(base_xy)
        if ee_rot is not None:
            self._actual_ee_rot_world = np.asarray(
                ee_rot, dtype=float).reshape(3, 3).copy()

        # Actual EE speed (EMA of finite difference) for the settle gate
        if dt > 1e-6 and self._ee_prev is not None:
            v_ee = float(np.linalg.norm(ee_pos - self._ee_prev)) / dt
            self._ee_speed = 0.8 * self._ee_speed + 0.2 * min(v_ee, 2.0)
        self._ee_prev = ee_pos.copy()

        if self.finished:
            tgt = self._body_to_world(self.carry_offset_body)
            R_target, omega_target = self._smooth_orientation_reference(
                self._seg, 1.0, dt)
            self._rot_target_world = R_target.copy()
            self._omega_ff = omega_target.copy()
            self._rot_prev_world = R_target.copy()
            self._last_target = tgt
            self._theta_dot = 0.0
            self.base_track_err = float(np.linalg.norm(
                base_xy - self._circle_pt(self._theta_ref)))
            return tgt, np.zeros(3), 0.0, "DONE"

        seg = self._seg
        self._action_diag = self._empty_action_diag()

        # ---- Lazy per-visit motion planning during TRANSIT ----------
        # Runs synchronously (option (a): the trajectory reference
        # freezes while we "think"; collision checks happen only in the
        # shadow client so the real robot is never disturbed).
        if (seg.kind == 'carry' and not self._planned
                and self._visit_idx < len(self.visits)
                and (self.s >= 0.5 or seg.duration < 2.5)):
            self._planned = self._plan_upcoming_visit(t)
            seg = self._seg

        # ---- Progress integration / handover ------------------------
        # FULL-6D progress feedback.  The old Stage-3 scheduler looked only
        # at XYZ error, so the path could keep advancing while orientation
        # lag grew past 90 deg.  Convert SO(3) error to an equivalent length
        # and slow progress whenever either part of the pose falls behind.
        arm_err_pos = 0.0
        arm_err_rot = 0.0
        if self._last_target is not None:
            arm_err_pos = float(np.linalg.norm(ee_pos - self._last_target))
        if (self._actual_ee_rot_world is not None and
                self._rot_target_world is not None):
            arm_err_rot = float(np.linalg.norm(so3_log(
                self._actual_ee_rot_world @ self._rot_target_world.T)))
        arm_err = math.sqrt(
            arm_err_pos**2
            + (self.full6d_progress_length * arm_err_rot)**2)
        self._last_orientation_track_error = arm_err_rot
        self._last_combined_track_error = arm_err
        rate = self._compute_rate(seg, arm_err)

        dwelling = False
        if self.s >= 1.0 - 1e-9:
            if seg.kind == 'action':
                # V10.4 minimal action gate.  Grasp/release decisions use
                # only the terminal trajectory condition and the measured
                # end-effector position error.  Cube pose/velocity/tilt and
                # EE speed are logged for evaluation only; they do not gate
                # the physical action.  There is also no pre-action settle
                # timer and no closed-loop release re-aim.
                err_act = float(np.linalg.norm(ee_pos - seg.target_world))
                is_release = (seg.action[0] == 'release')
                tol = (self.release_tolerance if is_release
                       else self.grasp_tolerance)
                if self._dwell_since is None:
                    self._dwell_since = t
                dwell_elapsed = max(0.0, t - self._dwell_since)

                # Simulation-only diagnostics.  These values are retained in
                # the log/DoD report, but are deliberately NOT part of the
                # grasp/release decision so the action logic does not depend
                # on object-state signals unavailable on the real platform.
                if is_release and self.carried is not None:
                    release_status = self._release_cube_status(seg)
                    self._action_diag.update({
                        "release_aim_count": int(seg.aim_count),
                        "release_cube_residual": float(release_status["residual_norm"]),
                        "release_cube_residual_xy": release_status["residual"].copy(),
                        "release_cube_speed": float(release_status["speed"]),
                        "release_cube_ang_speed": float(release_status["ang_speed"]),
                        "release_cube_tilt_ang_speed": float(release_status["tilt_ang_speed"]),
                        "release_cube_tilt_deg": float(release_status["tilt_deg"]),
                        "release_cube_aligned": bool(release_status["residual_norm"] <= self.AIM_TOL),
                        "release_cube_upright": bool(release_status["tilt_deg"] <= self.release_tilt_max_deg),
                        "release_aim_exhausted": False,
                        "action_dwell_elapsed": float(dwell_elapsed),
                        "action_settle_elapsed": 0.0,
                    })
                else:
                    self._action_diag.update({
                        "action_dwell_elapsed": float(dwell_elapsed),
                        "action_settle_elapsed": 0.0,
                    })

                err_rot_act = float(self._last_orientation_track_error)
                orientation_ready = (
                    err_rot_act <= self.full6d_action_rot_tol)
                self._action_diag["full6d_action_orientation_ready"] = bool(
                    orientation_ready)
                self._action_diag["full6d_action_orientation_error_deg"] = (
                    math.degrees(err_rot_act))

                if err_act < tol and orientation_ready:
                    self._action_diag["release_gate_ready"] = bool(is_release)
                    self._perform_action(seg, t, err_act)
                    self._advance(t)
                    dwelling = False
                else:
                    # Keep servoing the same terminal EE target until the
                    # position tolerance is reached.  Timeout remains only as
                    # fault handling for an unreachable terminal pose.
                    dwelling = True

                if (dwelling and self._dwell_since is not None
                        and t - self._dwell_since > self.dwell_timeout):
                    self._abort_attempt(t, err_act)
                    dwelling = False
            else:
                # Do not hand a large 6D lag to the next segment.  Hold the
                # terminal reference until both position and orientation are
                # inside a moderate handover tube.
                endpoint_pos_ok = (
                    arm_err_pos <= self.full6d_handover_pos_tol)
                endpoint_rot_ok = (
                    arm_err_rot <= self.full6d_handover_rot_tol)
                if endpoint_pos_ok and endpoint_rot_ok:
                    self._advance(t)
                    dwelling = False
                else:
                    dwelling = True
            seg = self._seg
        else:
            self.s = min(1.0, self.s + rate * dt)

        # ---- Independent base reference on the same analytic circle ----
        # Arm progress s and base circle progress theta_ref are now separate
        # states.  The base is allowed to keep moving during arm retract/RRT
        # work, but it still approaches grasp/drop windows with a deliberate
        # stop.
        self._advance_base_reference(seg, dt, dwelling, base_xy)
        self.base_track_err = float(np.linalg.norm(
            base_xy - self._circle_pt(self._theta_ref)))

        raw_target = self._arm_target(seg, self.s)

        # ---- CBF-QP reach-safety filter -------------------------------
        # The world-anchored end of a MOTION segment (e.g. the hover
        # point) blends in as a function of progress `s`, not of how
        # close the base has actually driven -- so early in a REACH
        # segment the raw target can sit farther from the CURRENT base
        # than the arm can comfortably reach, forcing a near-singular,
        # fully-extended posture (measured: manipulability -> ~0 at
        # ~1.0-1.1 m from the base, healthy ~0.10-0.15 below ~0.9 m).
        # Filtered HERE (not inside ArmSubsystem) so self._last_target /
        # the rate-adaptation error below sees the same reference the
        # controller is actually tracking.
        if self.reach_safety is not None:
            base_pos, _ = self._kuka_pose()
            arm_target = self.reach_safety.filter(
                raw_target, self._last_target, base_pos, dt)
        else:
            arm_target = raw_target

        # ---- V10.2 terminal ACTION command filter --------------------
        # A bounded release re-aim changes seg.target_world discretely.
        # At 240 Hz, a 10 mm step would appear as ~2.4 m/s under finite
        # differencing and can excite the arm/cube.  Shape only terminal
        # ACTION dwell references; planned reach/lift/transit trajectories
        # remain untouched.  The filtered position is used consistently by
        # both the global CLF reference and the arm-local reference.
        action_filter_active = (seg.kind == 'action'
                                and self.s >= 1.0 - 1e-9)
        arm_target_shaped, action_v_ff = self._shape_terminal_action_reference(
            arm_target, dt, action_filter_active)

        self._action_diag.update({
            "action_terminal_dwell": bool(action_filter_active),
            "action_ref_filter_active": bool(action_filter_active
                                              and self.action_ref_filter),
            "action_ref_filter_error": float(np.linalg.norm(
                np.asarray(arm_target) - arm_target_shaped)),
            "action_ref_filter_speed": float(np.linalg.norm(action_v_ff)),
        })

        # ---- EE velocity feedforward ---------------------------------
        # During the filtered terminal ACTION dwell, use the filter's
        # analytically generated velocity directly.  Else retain the
        # existing numeric+EMA feedforward for moving planned references.
        if action_filter_active and self.action_ref_filter:
            self._vel_ff = action_v_ff.copy()
        elif dt > 1e-6 and self._last_target is not None:
            v_raw = (arm_target_shaped - self._last_target) / dt
            n = float(np.linalg.norm(v_raw))
            if n > 0.8:
                v_raw *= 0.8 / n
            self._vel_ff = 0.8 * self._vel_ff + 0.2 * v_raw

        arm_target = arm_target_shaped

        R_target, omega_target = self._smooth_orientation_reference(
            seg, self.s, dt)
        self._rot_target_world = R_target.copy()
        self._omega_ff = omega_target.copy()
        self._rot_prev_world = R_target.copy()

        self._last_target = arm_target
        v_nominal = max(self._theta_dot * self.R, 0.0)
        return arm_target, self._vel_ff.copy(), v_nominal, seg.name

    def base_guidance(self, base_xy, base_yaw, law="kanayama",
                      kx=1.0, ky=2.0, ktheta=2.0,
                      k_arc=0.7, lookahead=0.20, kp_heading=2.0):
        """Return ``(v_path_ref, omega_path_ref, heading_target)``.

        Two explicit base path-following laws are supported:

        ``kanayama`` (default)
            Closed-loop unicycle trajectory tracking.  The desired point is
            ``circle(theta_ref)`` and the desired heading is the circle
            tangent.  Position error is expressed in the current base/body
            frame before forming the velocity command::

                e_x =  cos(yaw) dx + sin(yaw) dy
                e_y = -sin(yaw) dx + cos(yaw) dy
                e_th = wrap(theta_d - yaw)
                v = v_d cos(e_th) + kx e_x
                w = w_d + ky e_y + ktheta sin(e_th)

            This gives the base a genuine trajectory-tracking feedback law,
            rather than merely increasing the weight of the EE contribution.

        ``geometric``
            The previous arc-lag + look-ahead heading law.  It is retained
            only as an ablation/baseline for experiments.

        In both cases the returned command is ``u_path`` for the base local
        QP.  There is no separate ``u_ff``-centred regularisation term.
        """
        if self.finished:
            return 0.0, 0.0, base_yaw

        base_xy = np.asarray(base_xy, dtype=float).reshape(2)
        law = str(law).lower()

        if law == "geometric":
            arc_err = (self._theta_ref - self._theta_base) * self.R
            v_des = self._theta_dot * self.R + k_arc * arc_err
            v_lo = -(self.cruise + 0.10) if self.allow_reverse else 0.0
            v_des = float(np.clip(v_des, v_lo, self.cruise + 0.10))

            th_look = self._theta_base_wrapped + lookahead
            pt = self._circle_pt(th_look)
            heading_target = math.atan2(pt[1] - base_xy[1],
                                        pt[0] - base_xy[0])
            heading_err = _wrap(heading_target - base_yaw)
            omega_cmd = v_des * self.base_path.curvature(th_look) + kp_heading * heading_err
            omega_cmd = float(np.clip(omega_cmd, -1.5, 1.5))
            return v_des, omega_cmd, heading_target

        if law != "kanayama":
            raise ValueError(f"Unknown base path law: {law!r}")

        # Reference pose and feedforward twist from ONE analytic circle.
        # Task phases can change only theta-dot, never the geometric path.
        p_ref = self.base_path.position(self._theta_ref)
        theta_d = self.base_path.tangent_heading(self._theta_ref)
        v_d, omega_d = self.base_path.feedforward(self._theta_dot)
        v_d = max(float(v_d), 0.0)

        # Tracking error represented in the current base/body frame.
        dx, dy = p_ref - base_xy
        c, s = math.cos(base_yaw), math.sin(base_yaw)
        e_x = c * dx + s * dy
        e_y = -s * dx + c * dy
        e_theta = _wrap(theta_d - base_yaw)

        # Nonlinear unicycle tracking command.  The cosine/sine terms prevent
        # the controller from demanding full forward speed while badly
        # misaligned and keep the heading correction bounded.
        v_cmd = v_d * math.cos(e_theta) + float(kx) * e_x
        omega_cmd = (omega_d + float(ky) * e_y
                     + float(ktheta) * math.sin(e_theta))

        v_lo = -(self.cruise + 0.15) if self.allow_reverse else 0.0
        v_cmd = float(np.clip(v_cmd, v_lo, self.cruise + 0.15))
        omega_cmd = float(np.clip(omega_cmd, -1.5, 1.5))
        return v_cmd, omega_cmd, _wrap(theta_d)

    def post_tick(self, t, ee_pos):
        """Per-tick verification bookkeeping (grasp-hold monitoring).

        Samples |cube - EE| for the 2 s following each grasp, after a
        0.3 s attach transient (the fixed constraint snaps the cube
        from the approach offset to its 2 cm carry offset)."""
        for mon in self.grasp_monitors:
            if mon['done']:
                continue
            if t - mon['t0'] <= 2.3:
                if self.carried != mon['cube']:
                    mon['dropped'] = True
                    mon['done'] = True
                    continue
                if t - mon['t0'] < 0.3:
                    continue
                with self._lock:
                    c_pos, _ = self.p.getBasePositionAndOrientation(
                        mon['cube'])
                mon['samples'].append(
                    float(np.linalg.norm(np.array(c_pos) - ee_pos)))
            else:
                mon['done'] = True
        # Landing trace: cube position 1 s after each release (to tell
        # a bad landing apart from a later disturbance).
        for pl in self.placements:
            if 'settled_err' not in pl and t - pl['t'] >= 1.0:
                with self._lock:
                    c_pos, _ = self.p.getBasePositionAndOrientation(
                        pl['cube'])
                pl['settled_err'] = float(np.linalg.norm(
                    np.array(c_pos[:2]) - pl['target'][:2]))

    def report(self):
        """Final metrics for the Definition-of-Done summary."""
        grasps = []
        for mon in self.grasp_monitors:
            samples = np.array(mon['samples']) if mon['samples'] else \
                np.array([np.inf])
            grasps.append({
                'cube': mon['cube'], 'table': mon['table'],
                't': mon['t0'],
                'n': len(mon['samples']),
                'max_dev': float(samples.max()),
                'mean_dev': float(samples.mean()),
                'dropped': mon.get('dropped', False),
            })
        placements = []
        for pl in self.placements:
            with self._lock:
                c_pos, _ = self.p.getBasePositionAndOrientation(pl['cube'])
            placements.append({
                'cube': pl['cube'], 'table': pl['table'],
                'target': pl['target'],
                'final': np.array(c_pos),
                'err': float(np.linalg.norm(np.array(c_pos)
                                            - pl['target'])),
                'err_xy': float(np.linalg.norm(np.array(c_pos[:2])
                                               - pl['target'][:2])),
                'err_1s': pl.get('settled_err'),
                'release_cube_residual': pl.get('release_cube_residual'),
                'release_cube_speed': pl.get('release_cube_speed'),
                'release_cube_ang_speed': pl.get('release_cube_ang_speed'),
                'release_cube_tilt_deg': pl.get('release_cube_tilt_deg'),
                'release_aim_count': pl.get('release_aim_count'),
            })
        return {
            'grasps': grasps,
            'placements': placements,
            'timeouts': self.timeout_events,
            'plan_fallbacks': self.plan_fallbacks,
            'plan_times': self.plan_times,
        }

    # ==================================================================
    # Internals — progress & geometry
    # ==================================================================

    def _circle_pt(self, theta):
        """Analytic circular reference point (never a chord/waypoint)."""
        return self.base_path.position(theta)

    def _update_theta_base(self, base_xy):
        th = math.atan2(base_xy[1] - self.CY, base_xy[0] - self.CX)
        self._theta_base += _wrap(th - self._theta_base_wrapped)
        self._theta_base_wrapped = th

    def _ahead(self, theta, cursor):
        """Unwrap ``theta`` to the first value >= cursor - 0.3."""
        out = cursor + _wrap(theta - cursor)
        while out < cursor - 0.3:
            out += 2 * math.pi
        return out

    def _next_visit_transit_end_theta(self, start_theta):
        """Predict the next visit's reach-window entrance without mutating state."""
        next_idx = self._visit_idx + 1
        if next_idx >= len(self.visits):
            return float(start_theta + 0.35 / max(self.R, 1e-6))

        table, do_place, do_pick = self.visits[next_idx]
        if do_place:
            anchor_xy = np.asarray(self.place_target_fn(table), dtype=float)[:2]
        else:
            item = self._item_pos(table)
            if item is None:
                return float(start_theta + 0.35 / max(self.R, 1e-6))
            anchor_xy = np.asarray(item, dtype=float)[:2]

        th_anchor = self._ahead(
            math.atan2(anchor_xy[1] - self.CY, anchor_xy[0] - self.CX),
            float(start_theta) + 0.05)
        return float(max(
            th_anchor - self._act_delta(anchor_xy)
            - self.ACT_SPAN - self.REACH_ARC,
            float(start_theta) + 0.02))

    def _base_progress_horizon(self, seg):
        """Furthest safe circle angle the independent base may pursue now.

        The base may work ahead of the current ARM segment, but it never drives
        past the next planned reach/action window.  This preserves grasp/drop
        reachability while allowing genuine simultaneous base+arm motion.
        """
        phase = str(seg.name)

        # Grasp/drop and the incoming reach must not be passed.
        if phase in ("PCK_RCH", "PLC_RCH", "PCK_GRP", "PLC_DRP"):
            return float(seg.th1), True

        # If a queued reach/action already exists, that is the next physical
        # manipulation window.  The base can move through retract/lift work
        # toward it while the arm is still busy.
        for future in self._queue:
            if str(future.name) in ("PCK_RCH", "PLC_RCH",
                                    "PCK_GRP", "PLC_DRP"):
                return float(future.th1), True

        # TRANSIT has an explicit end at the incoming reach window.
        if phase == "TRANSIT":
            return float(seg.th1), True

        # Final retract of a visit: begin the next transit while the arm
        # retracts, but stop at the next visit's reach-window entrance.
        if bool(seg.final_of_visit):
            return self._next_visit_transit_end_theta(
                max(float(seg.th1), float(self._theta_ref))), True

        # Within a visit, allow progress at least to the current segment end.
        return float(max(seg.th1, self._theta_ref)), False

    def _base_phase_speed_cap(self, seg, dwelling):
        phase = str(seg.name)
        if self.finished:
            return 0.0
        if phase in ("PCK_GRP", "PLC_DRP"):
            # Intentionally settle the physical grasp/release.
            return 0.0 if dwelling or self.s >= 0.82 else self.base_action_speed
        if phase in ("PCK_RCH", "PLC_RCH"):
            return self.base_reach_speed
        if phase in ("LIFT_VERT", "PLC_LFT_VERT"):
            return self.base_lift_speed
        if phase in ("LIFT_RRT", "PLC_LFT_RRT"):
            return self.base_work_speed
        if phase == "TRANSIT":
            return max(self.base_work_speed, float(self.cruise))
        return self.base_lift_speed

    def _advance_base_reference(self, seg, dt, dwelling, base_xy):
        """Advance the mobile-base reference independently of arm progress.

        This is a reference scheduler only.  The base local QP / CLF / ADMM
        still receives the same geometric circle tracking problem and the same
        hard command bounds.  No command is clipped after optimization.
        """
        h = max(float(dt), 0.0)
        if h <= 1e-12:
            self._theta_dot = self._base_ref_v / max(self.R, 1e-9)
            return

        if not self.base_decoupled:
            s_clip = min(float(self.s), 1.0)
            self._theta_ref = seg.th0 + (seg.th1 - seg.th0) * s_clip
            self._theta_dot = (
                0.0 if dwelling else
                (seg.th1 - seg.th0)
                * float(self._rate_diag.get("progress_rate_filtered", 0.0)))
            self._base_ref_v = max(self._theta_dot * self.R, 0.0)
            return

        horizon, stop_required = self._base_progress_horizon(seg)
        remaining = max((horizon - self._theta_ref) * self.R, 0.0)
        phase_cap = self._base_phase_speed_cap(seg, dwelling)

        # Curvature and action-window stopping limits.
        kappa = abs(float(self.base_path.curvature(self._theta_ref)))
        v_curve = math.sqrt(
            self.speed_generator.a_lat_max
            / (kappa + self.speed_generator.eps))
        v_target = min(phase_cap, v_curve)

        if stop_required:
            v_stop = math.sqrt(
                max(2.0 * self.speed_generator.a_dec_max * remaining, 0.0))
            v_target = min(v_target, v_stop)

        # Do not let the reference run far ahead of the measured base.  This
        # avoids path-tracker spikes while keeping the base free to move.
        lead_m = max((self._theta_ref - self._theta_base) * self.R, 0.0)
        if lead_m >= self.base_ref_max_lead:
            v_target = 0.0
        elif lead_m > 0.70 * self.base_ref_max_lead:
            taper = (
                self.base_ref_max_lead - lead_m
            ) / max(0.30 * self.base_ref_max_lead, 1e-6)
            v_target *= float(np.clip(taper, 0.0, 1.0))

        # Reference acceleration governor.
        dv = v_target - self._base_ref_v
        if dv >= 0.0:
            dv_lim = self.base_ref_accel * h
        else:
            dv_lim = self.base_ref_decel * h
        self._base_ref_v += float(np.clip(dv, -dv_lim, dv_lim))
        self._base_ref_v = max(self._base_ref_v, 0.0)

        dtheta = self._base_ref_v * h / max(self.R, 1e-9)
        theta_next = self._theta_ref + dtheta
        if stop_required:
            theta_next = min(theta_next, horizon)
            if theta_next >= horizon - 1e-9:
                self._base_ref_v = 0.0

        self._theta_ref = float(theta_next)
        self._theta_dot = float(self._base_ref_v / max(self.R, 1e-9))
        self._base_ref_diag = {
            "base_progress_decoupled": True,
            "base_reference_theta": float(self._theta_ref),
            "base_reference_speed": float(self._base_ref_v),
            "base_reference_target_speed": float(v_target),
            "base_reference_horizon_theta": float(horizon),
            "base_reference_horizon_distance_m": float(remaining),
            "base_reference_stop_required": bool(stop_required),
            "base_reference_lead_m": float(lead_m),
            "base_reference_phase_cap": float(phase_cap),
        }

    def _act_delta(self, target_xy):
        """How far (rad) BEFORE the target's circle angle the descend
        window is centred.

        Chosen so the base passes the action point at horizontal
        distance ~D_COMFORT: targets close to the circle are acted on
        while still behind them (arm reaches forward, near its carry
        posture); targets deep inside the circle are acted on abeam
        (the radial gap already uses up the reach budget).
        """
        r_t = float(np.linalg.norm(
            np.asarray(target_xy[:2]) - np.array([self.CX, self.CY])))
        radial_gap = max(self.R - r_t, 0.0)
        tang = math.sqrt(max(self.D_COMFORT ** 2 - radial_gap ** 2, 0.0))
        return tang / self.R

    def _kuka_pose(self):
        with self._lock:
            pos, orn = self.p.getBasePositionAndOrientation(self.kuka_id)
        R = np.array(self.p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        return np.array(pos), R

    def _body_to_world(self, offset_body):
        pos, R = self._kuka_pose()
        return pos + R @ np.asarray(offset_body)

    def _read_arm_q(self):
        with self._lock:
            return np.array([self.p.getJointState(self.kuka_id, j)[0]
                             for j in self.joint_ids])

    def _arm_target(self, seg, s):
        """Evaluate the planned trajectory at progress s (world frame).

        The joint path's FK image is stored in the arm-base frame at
        the *nominal* base pose; endpoints are warped so that WORLD
        anchors are hit exactly no matter where the base actually is,
        while BODY anchors ride along with the base (carry pose).
        """
        if seg.cartesian_world is not None:
            p0, p1 = seg.cartesian_world
            u = _smoothstep(s)
            return (1.0 - u) * p0 + u * p1
        if seg.traj is None:
            return self._body_to_world(self.carry_offset_body)
        base_pos, R = self._kuka_pose()
        p_now = base_pos + R @ seg.ee_local.p(s)
        w = _smoothstep(s)
        tgt = p_now
        if seg.anchor_start is not None and w < 1.0:
            tgt = tgt + (1.0 - w) * self._anchor_delta(
                seg.anchor_start, seg.ee_local.p(0.0), base_pos, R)
        if seg.anchor_end is not None and w > 0.0:
            tgt = tgt + w * self._anchor_delta(
                seg.anchor_end, seg.ee_local.p(1.0), base_pos, R)
        return tgt

    @staticmethod
    def _frame_with_axis_closest(R_reference, z_desired):
        """Full frame with desired tool axis and minimum wrist-spin change.

        The Stage-3 planner intentionally leaves spin about the tool axis
        unconstrained.  For 6D execution we must choose that third rotational
        DOF.  Projecting the measured x-axis onto the plane normal to the
        desired tool axis selects the closest yaw instead of imposing a fixed
        arbitrary wrist angle.
        """
        Rref = np.asarray(R_reference, dtype=float).reshape(3, 3)
        z = np.asarray(z_desired, dtype=float).reshape(3)
        z /= max(float(np.linalg.norm(z)), 1e-12)

        x = Rref[:, 0] - z * float(np.dot(z, Rref[:, 0]))
        nx = float(np.linalg.norm(x))
        if nx < 1e-8:
            x = Rref[:, 1] - z * float(np.dot(z, Rref[:, 1]))
            nx = float(np.linalg.norm(x))
        if nx < 1e-8:
            seed = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(seed, z))) > 0.9:
                seed = np.array([0.0, 1.0, 0.0])
            x = seed - z * float(np.dot(seed, z))
            nx = float(np.linalg.norm(x))
        x /= max(nx, 1e-12)
        y = np.cross(z, x)
        y /= max(float(np.linalg.norm(y)), 1e-12)
        x = np.cross(y, z)
        return np.column_stack([x, y, z])

    def _phase_orientation_goal_local(self, seg):
        del seg
        return self._orientation_phase_goal_local.copy()

    def _smooth_orientation_reference(self, seg, s, dt):
        _, Rb = self._kuka_pose()
        phase = str(seg.name)

        # Re-seed from the measured pose at every phase change so the
        # reference itself never jumps.
        if self._orientation_phase != phase:
            if self._actual_ee_rot_world is not None:
                self._orientation_phase_start_local = (
                    Rb.T @ self._actual_ee_rot_world)
                self._orientation_cmd_local = (
                    self._orientation_phase_start_local.copy())

            if phase in ("PCK_RCH", "PCK_GRP", "LIFT_VERT",
                         "PLC_RCH", "PLC_DRP", "PLC_LFT_VERT"):
                # Use only the prescribed top-down TOOL AXIS from Stage 3;
                # choose the full-frame spin closest to the measured pose.
                self._orientation_phase_goal_local = (
                    self._frame_with_axis_closest(
                        self._orientation_phase_start_local,
                        self._R_topdown_local[:, 2]))
            else:
                self._orientation_phase_goal_local = (
                    self._carry_R_local.copy())

            self._orientation_phase = phase

        R_goal = self._phase_orientation_goal_local(seg)

        # ACTION and vertical segments hold the exact top-down frame.
        # Reach/retract segments interpolate smoothly over progress.
        if phase in ("PCK_GRP", "LIFT_VERT",
                     "PLC_DRP", "PLC_LFT_VERT",
                     "TRANSIT", "DONE"):
            u = 1.0
        else:
            s0 = float(np.clip((float(s) - 0.05) / 0.90, 0.0, 1.0))
            u = _smoothstep(s0)

        R_raw = (
            so3_exp(u * so3_log(
                R_goal @ self._orientation_phase_start_local.T))
            @ self._orientation_phase_start_local
        )

        # Reference governor: limit desired orientation angular speed before
        # it reaches either the local arm controller or the global 6D CLF.
        h = max(float(dt), 0.0)
        R_cmd, omega_local = rate_limit_rotation(
            self._orientation_cmd_local,
            R_raw,
            self.full6d_orientation_rate,
            h,
        )

        self._orientation_cmd_local = R_cmd.copy()

        # World angular feedforward must include motion of the arm mounting
        # frame itself.  Without this term, the global 6D CLF interprets
        # ordinary base yaw as an orientation tracking disturbance.
        if h > 1e-9:
            omega_base_world = so3_log(
                Rb @ self._R_base_prev_world.T) / h
        else:
            omega_base_world = np.zeros(3)
        self._R_base_prev_world = Rb.copy()
        self._omega_local_ff = omega_local.copy()
        omega_world = omega_base_world + Rb @ omega_local
        return Rb @ R_cmd, omega_world

    def _arm_orientation_target(self, seg, s):
        """Compatibility accessor for the already-governed reference."""
        del seg, s
        return self._rot_target_world.copy()

    def _anchor_delta(self, anchor, p_edge_local, base_pos, R):
        kind, val = anchor
        if kind == 'world':
            return np.asarray(val) - (base_pos + R @ p_edge_local)
        return R @ (np.asarray(val) - p_edge_local)   # 'body'

    def _compute_rate(self, seg, arm_err):
        """Return ds/dt with predictive RRT/reference-speed limits.

        V10.11 keeps the controller unchanged and fixes the remaining smooth
        positional lag in RRT/reach phases at the time-parameterization layer.

        The progress rate is limited by FOUR independent mechanisms:

        1. mobile-base path speed / curvature scheduler;
        2. minimum segment duration;
        3. joint-path arc-length speed, because ProgressTrajectory is
           joint-arc-length parameterized;
        4. predicted Cartesian reference speed from d p_ref / d s.

        Tracking-sensitive RRT/reach segments also use a 6D-error pause with
        hysteresis.  When tracking falls behind, progress freezes and the
        controller is allowed to catch up to the SAME reference.  No task
        slack, command clipping, or post-optimization smoothing is introduced.
        """
        arc_len = max((seg.th1 - seg.th0) * self.R, 1e-6)

        if self.adaptive_speed:
            # ARM progress uses the arm segment's own progress.  In V10.11 it
            # still read theta_ref, so slowing one subsystem could indirectly
            # slow the other.  V10.11 fully separates those two clocks.
            theta_arm = (
                float(seg.th0)
                + (float(seg.th1) - float(seg.th0))
                * float(np.clip(self.s, 0.0, 1.0))
            )
            curvature = self.base_path.curvature(theta_arm)
            remaining = max(
                (1.0 - float(np.clip(self.s, 0.0, 1.0)))
                * (seg.th1 - seg.th0) * self.R,
                0.0)
            stop_required = (seg.kind == 'action')
            v_sched = self.speed_generator.compute(
                curvature, remaining, stop_required=stop_required)
            rate_speed = v_sched / arc_len
            rate_arm_cap = 1.0 / max(seg.min_duration, 1e-3)
            base = min(rate_speed, rate_arm_cap)
            self._speed_sched = float(v_sched)
        else:
            base = 1.0 / seg.duration
            self._speed_sched = float(base * arc_len)

        phase = str(seg.name)
        tracking_sensitive = (
            seg.kind == 'motion'
            and (phase.endswith("RRT") or phase.endswith("RCH"))
        )

        rate_joint_cap = float("inf")
        rate_cart_cap = float("inf")
        dpds = 0.0

        if tracking_sensitive and seg.traj is not None:
            if phase.endswith("RRT"):
                q_path_speed = self.rrt_joint_path_speed
                cart_speed = self.rrt_cart_ref_speed
            else:
                q_path_speed = self.reach_joint_path_speed
                cart_speed = self.reach_cart_ref_speed

            # q(s) is joint-arc-length parameterized:
            # ||dq/dt||_path = traj.length * ds/dt.
            if float(seg.traj.length) > 1e-9:
                rate_joint_cap = (
                    q_path_speed / float(seg.traj.length)
                )

            # Predict the world reference's local derivative w.r.t progress.
            # This includes the endpoint anchor warp, which was the source of
            # the early 100--300 mm error spike in the V10.9 RRT phases.
            s0 = float(np.clip(self.s, 0.0, 1.0))
            ds_probe = min(0.01, max(1.0 - s0, 0.0))
            if ds_probe > 1e-7:
                p0 = self._arm_target(seg, s0)
                p1 = self._arm_target(seg, s0 + ds_probe)
                dpds = float(np.linalg.norm(p1 - p0)) / ds_probe
                if dpds > 1e-8:
                    rate_cart_cap = cart_speed / dpds

            base = min(base, rate_joint_cap, rate_cart_cap)

        # Tracking-error slowdown / freeze.  The hard pause is hysteretic:
        # enter at hard_error and do not resume until resume_error.
        if tracking_sensitive:
            if self._progress_pause:
                if arm_err <= self.tracking_resume_error:
                    self._progress_pause = False
            elif arm_err >= self.tracking_hard_error:
                self._progress_pause = True

            if self._progress_pause:
                f_err = 0.0
            else:
                f_err = _tracking_progress_factor(
                    arm_err,
                    self.tracking_soft_error,
                    self.tracking_hard_error,
                )
        elif seg.kind == 'carry':
            f_err = 1.0
            self._progress_pause = False
        else:
            e_norm = np.clip((arm_err - 0.03) / 0.10, 0.0, 1.0)
            f_err = 1.0 - 0.85 * _smoothstep(e_norm)
            self._progress_pause = False

        if self.base_decoupled:
            # Base path tracking no longer throttles the arm trajectory.
            f_lag = 1.0
        else:
            lag = (self._theta_ref - self._theta_base) * self.R
            if lag > 0.10:
                f_lag = float(np.clip(
                    1.0 - (lag - 0.10) / 0.20, 0.25, 1.0))
            else:
                f_lag = 1.0

        raw = max(base * f_err * f_lag, 0.0)

        # Asymmetric rate filter:
        #   - slow down rapidly when tracking worsens;
        #   - recover progressively so the reference does not burst forward
        #     immediately after the arm catches up.
        if self._progress_pause:
            self._rate_ema = 0.0
        elif self._rate_ema is None:
            self._rate_ema = raw
        elif raw < self._rate_ema:
            self._rate_ema = 0.25 * self._rate_ema + 0.75 * raw
        else:
            self._rate_ema = 0.94 * self._rate_ema + 0.06 * raw

        self._rate_diag = {
            "progress_rate_raw": float(raw),
            "progress_rate_filtered": float(self._rate_ema),
            "progress_rate_cart_cap": float(rate_cart_cap),
            "progress_rate_joint_cap": float(rate_joint_cap),
            "progress_tracking_factor": float(f_err),
            "progress_paused": bool(self._progress_pause),
            "progress_dpds_m": float(dpds),
            "segment_min_duration_s": float(seg.min_duration),
        }
        return self._rate_ema

    @property
    def adaptive_speed_info(self):
        """Latest scheduler and tracking-aware progress diagnostics."""
        out = dict(self.speed_generator.last)
        out["v_scheduled"] = float(self._speed_sched)
        out.update(self._rate_diag)
        out.update(self._base_ref_diag)
        return out

    def _advance(self, t):
        """Hand over to the next segment (s reset, rate EMA reseeded)."""
        prev = self._seg
        if self._queue:
            self._seg = self._queue.pop(0)
        else:
            if prev.final_of_visit:
                # A visit just finished
                self._visit_idx += 1
            if self._visit_idx >= len(self.visits):
                self._seg = _Segment("DONE", 'carry',
                                     prev.th1, prev.th1, 1.0)
                self.finished = True
                self._t0 = t
                print("\n" + "=" * 40)
                print("  CYCLE COMPLETE — all items rotated!")
                print("=" * 40 + "\n")
                return
            self._planned = False
            self._seg = self._make_transit(
                start_th=max(float(prev.th1), float(self._theta_ref)))
        self.s = 0.0
        self._dwell_since = None
        self._rate_ema = None
        self._progress_pause = False
        self._t0 = t
        print(f"  >>> {prev.name} -> {self._seg.name}  "
              f"table={self.target_table}  "
              f"th=[{self._seg.th0:.2f},{self._seg.th1:.2f}] "
              f"dur={self._seg.duration:.1f}s")

    # Closed-loop release aiming.  The cube may be released ONLY when
    # its own XY residual is within AIM_TOL.  AIM_MAX limits how many
    # target corrections we issue; it is not a bypass for the hard gate.
    AIM_TOL = 0.012       # m — hard cube-over-target release tolerance
    AIM_INNER_TOL = 0.010 # m — re-aim toward a small margin inside the hard gate
    AIM_STEP_MAX = 0.010  # m — max XY target shift per re-aim (limits pendulum excitation)
    AIM_MAX = 6           # correction attempts before hold-until-timeout
    CUBE_STATIC = 0.05    # m/s — cube must be quasi-static before release/aim

    def _release_cube_status(self, seg):
        """Measure cube residual/speed/tilt for the release hard gate."""
        if self.carried is None or seg.place_xy is None:
            return {
                "residual": np.array([np.nan, np.nan]),
                "residual_norm": float("inf"),
                "speed": float("inf"),
                "ang_speed": float("inf"),
                "tilt_ang_speed": float("inf"),
                "tilt_deg": float("nan"),
            }
        with self._lock:
            c_pos, c_orn = self.p.getBasePositionAndOrientation(self.carried)
            c_vel, c_avel = self.p.getBaseVelocity(self.carried)
        resid = np.asarray(c_pos[:2], dtype=float) - seg.place_xy
        up = np.array(self.p.getMatrixFromQuaternion(c_orn)).reshape(3, 3)[:, 2]
        tilt = math.degrees(math.acos(np.clip(up[2], -1.0, 1.0)))
        # Only the angular-velocity component perpendicular to world Z
        # changes the cube tilt.  Yaw spin about world Z does not threaten
        # upright placement and must not block release.
        c_avel_arr = np.asarray(c_avel, dtype=float)
        tilt_ang_speed = float(np.linalg.norm(c_avel_arr[:2]))
        return {
            "residual": resid,
            "residual_norm": float(np.linalg.norm(resid)),
            "speed": float(np.linalg.norm(c_vel)),
            "ang_speed": float(np.linalg.norm(c_avel_arr)),
            "tilt_ang_speed": tilt_ang_speed,
            "tilt_deg": float(tilt),
        }

    def _aim_release_for_cube(self, seg, t, status=None):
        """Shift the EE release target to cancel measured cube XY error.

        Returns True only when a new correction was issued.  If AIM_MAX
        has been reached while the cube is still outside AIM_TOL, the
        segment enters ``aim_exhausted`` and the caller keeps HOLDING; the
        cube is never released merely because the attempt limit was hit.
        """
        if self.carried is None or seg.place_xy is None:
            return False
        if status is None:
            status = self._release_cube_status(seg)
        resid = np.asarray(status["residual"], dtype=float)
        if float(status["residual_norm"]) <= self.AIM_TOL:
            seg.aim_exhausted = False
            return False
        if seg.aim_count >= self.AIM_MAX:
            seg.aim_exhausted = True
            return False

        # V5.3: deadband + bounded release re-aim.  The previous law
        # shifted the EE target by the *entire* measured cube residual,
        # which repeatedly over-excited the suspended cube (especially when
        # the residual was only slightly outside the 12 mm hard gate).
        #
        # Instead, aim only far enough to bring the cube toward a small
        # inner target radius, and cap every correction.  The hard release
        # condition remains AIM_TOL; AIM_INNER_TOL merely supplies 2 mm of
        # hysteresis so measurement noise does not immediately trigger a
        # second correction at the boundary.
        r = float(status["residual_norm"])
        if r <= 1e-12:
            return False
        desired_step = max(r - self.AIM_INNER_TOL, 0.0)
        step_mag = min(desired_step, self.AIM_STEP_MAX)
        if step_mag <= 1e-9:
            return False
        shift_xy = -(step_mag / r) * resid

        target = seg.target_world.copy()
        target[:2] += shift_xy
        seg.target_world = target
        seg.anchor_end = ('world', target.copy())
        seg.aim_count += 1
        seg.last_aim_time = float(t)
        seg.aim_exhausted = (seg.aim_count >= self.AIM_MAX)
        print(f"      release aim #{seg.aim_count}: bounded shift "
              f"[{shift_xy[0]*1000:.1f}, {shift_xy[1]*1000:.1f}] mm "
              f"(|shift|={step_mag*1000:.1f} mm, "
              f"cube residual={r*1000:.1f} mm, "
              f"inner target={self.AIM_INNER_TOL*1000:.0f} mm)")
        return True

    def _make_transit(self, start_th=None):
        """TRANSIT segment driving to the next visit's reach window."""
        table, do_place, do_pick = self.visits[self._visit_idx]
        self.target_table = table
        if do_place:
            anchor_xy = self.place_target_fn(table)[:2]
        else:
            anchor_xy = self._item_pos(table)[:2]
        th0 = self._theta_ref if start_th is None else start_th
        th_anchor = self._ahead(
            math.atan2(anchor_xy[1] - self.CY, anchor_xy[0] - self.CX),
            th0 + 0.05) + self._retry_offsets
        self._retry_offsets = 0.0
        th1 = max(th_anchor - self._act_delta(anchor_xy)
                  - self.ACT_SPAN - self.REACH_ARC, th0 + 0.02)
        dur = max((th1 - th0) * self.R / self.V_TRANSIT, 0.5)
        return _Segment("TRANSIT", 'carry', th0, th1, dur,
                        min_duration=0.5)

    def _item_pos(self, table_idx):
        items = self.table_items.get(table_idx, [])
        if items:
            with self._lock:
                pos, _ = self.p.getBasePositionAndOrientation(items[0])
            return np.array(pos)
        return None

    # ==================================================================
    # Internals — per-visit motion planning
    # ==================================================================

    def _predict_kuka_pose(self, theta):
        """Predicted arm-base world pose when the base is at circle
        angle theta (uses the husky→kuka transform calibrated at
        startup)."""
        h_xy = self._circle_pt(theta)
        h_yaw = theta + math.pi / 2
        k_xy = h_xy + _rot2(h_yaw) @ self._kuka_rel_xy
        pos = [k_xy[0], k_xy[1], self._kuka_z]
        orn = self.p.getQuaternionFromEuler(
            [0, 0, h_yaw + self._kuka_yaw_rel])
        return pos, orn

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

    def _topdown_world_quat(self, base_orn):
        """Top-down EE orientation in world coordinates for a base pose."""
        if not self.top_down_grasp:
            return None
        q = self._quat_mul(base_orn, self.top_down_local_quat)
        return q / max(float(np.linalg.norm(q)), 1e-12)

    def _topdown_world_axis(self, base_orn):
        """Desired world direction of the EE local +Z tool axis."""
        q = self._topdown_world_quat(base_orn)
        if q is None:
            return None
        R = np.asarray(self.p.getMatrixFromQuaternion(list(q)),
                       dtype=float).reshape(3, 3)
        a = R[:, 2]
        return a / max(float(np.linalg.norm(a)), 1e-12)

    def _preferred_ik_rest(self, q_near):
        """Bias redundant IK toward the configured comfortable arm posture."""
        q_near = np.asarray(q_near, dtype=float).reshape(len(self.joint_ids))
        if self.preferred_grasp_q is None:
            return q_near
        # Preserve continuity while giving the redundant solver a repeatable
        # elbow/wrist preference.  This is a soft preference, not a task-space
        # constraint; top-down orientation remains the hard kinematic request.
        rest = 0.70 * q_near + 0.30 * self.preferred_grasp_q
        return np.clip(rest, self.q_lower, self.q_upper)

    def _topdown_carry_q(self, base_pose, q_near, margin):
        """Return a top-down *axis-only* IK solution at the carry point.

        The carry point is a BODY-frame position.  Solving a pose IK there
        gives the constrained RRT a goal that satisfies the same top-down
        tool-axis manifold as the hover pose, instead of asking the controller
        to preserve a full orientation while following an unconstrained
        joint-space retract to the historical carry configuration.
        """
        base_pos, base_orn = base_pose
        R = np.array(self.p.getMatrixFromQuaternion(base_orn),
                     dtype=float).reshape(3, 3)
        carry_world = (np.asarray(base_pos, dtype=float) +
                       R @ np.asarray(self.carry_offset_body, dtype=float))
        axis_top = self._topdown_world_axis(base_orn)
        self.shadow.set_base_pose(*base_pose)
        q = self.shadow.ik_axis(
            carry_world, rest_q=self._preferred_ik_rest(q_near),
            tries_per_yaw=5, pos_tol=0.010, collision_margin=margin,
            target_axis_world=axis_top, axis_tol=math.radians(8.0))
        return q, axis_top

    def _cartesian_ik_path(self, q_start, p_start, p_end, margin,
                           axis_world, label):
        """Build a collision-checked straight Cartesian IK path.

        This is used for the final descend and the initial post-action lift.
        Unlike a joint-space RRT edge, every waypoint lies on the prescribed
        Cartesian line and keeps only the top-down tool *axis*.  Wrist yaw is
        selected continuously from the previous waypoint and remains free.
        """
        p0 = np.asarray(p_start, dtype=float).reshape(3)
        p1 = np.asarray(p_end, dtype=float).reshape(3)
        dist = float(np.linalg.norm(p1 - p0))
        n_steps = int(np.clip(math.ceil(dist / 0.015), 6, 24))
        path = [np.asarray(q_start, dtype=float).copy()]
        q_prev = path[0]
        for i in range(1, n_steps + 1):
            s = i / n_steps
            target = (1.0 - s) * p0 + s * p1
            q_next = self.shadow.ik_axis(
                target, rest_q=q_prev, pos_tol=0.008,
                collision_margin=margin,
                target_axis_world=axis_world,
                tries_per_yaw=4, axis_tol=math.radians(8.0),
                yaw_offsets=(0.0,))
            if q_next is None:
                raise _PlanningError(
                    f"{label}: Cartesian top-down-axis IK failed at {100*s:.0f}%")
            if self.shadow.in_collision(q_next, margin=margin):
                raise _PlanningError(
                    f"{label}: Cartesian top-down path violates "
                    f"{margin*1000:.0f} mm clearance at {100*s:.0f}%")
            path.append(np.asarray(q_next, dtype=float))
            q_prev = path[-1]
        return path

    def _plan_upcoming_visit(self, t):
        """RRT-Connect-plan all arm segments of the next visit.

        Synchronous (the robot 'thinks' for a few hundred ms at most —
        the progress reference simply does not advance meanwhile).
        """
        table, do_place, do_pick = self.visits[self._visit_idx]
        t_wall0 = _time.perf_counter()
        q_carry = self._read_arm_q()
        self._q_carry = q_carry.copy()
        th_cursor = max(float(self._seg.th1), float(self._theta_ref))
        segs = []
        try:
            # V10.5: when a visit contains both place and pick, plan the two
            # actions as one continuous arm transfer.  After release the arm
            # no longer returns to the old carry pose before reaching the next
            # object.  Instead it makes a short collision-checked clearance
            # lift and transfers directly to the next pre-grasp hover.
            if do_place and do_pick:
                if self._place_target is None:
                    self._place_target = self.place_target_fn(table)
                    print(f"      place target T{table+1}: "
                          f"{[round(x,3) for x in self._place_target]}")
                item = self._item_pos(table)
                if item is None:
                    raise _PlanningError(f"no item available on T{table+1} for next pick")
                pair, th_cursor = self._plan_place_pick_pair(
                    self._place_target, item, q_carry, th_cursor, table)
                segs += pair
            else:
                if do_place:
                    if self._place_target is None:
                        self._place_target = self.place_target_fn(table)
                        print(f"      place target T{table+1}: "
                              f"{[round(x,3) for x in self._place_target]}")
                    pt = self._place_target
                    act = self._place_ee_target(pt)
                    hover = np.array([act[0], act[1],
                                      self.table_surface + self.object_half
                                      + self.hover_dz])
                    trio, th_cursor = self._plan_trio(
                        "PLC", hover, act, q_carry, th_cursor,
                        action=('release', table))
                    segs += trio
                if do_pick:
                    item = self._item_pos(table)
                    if item is None:
                        raise _PlanningError(f"no item available on T{table+1}")
                    hover = np.array([item[0], item[1],
                                      self.table_surface + self.object_half
                                      + self.hover_dz])
                    act = item + np.array([0.0, 0.0, self.grasp_z_off])
                    trio, th_cursor = self._plan_trio(
                        "PCK", hover, act, q_carry, th_cursor,
                        action=('grasp', table))
                    segs += trio
        except _PlanningError as e:
            msg = (f"PLANNING FAILED for visit T{table+1} "
                   f"({'place' if do_place else ''}"
                   f"{'+pick' if do_pick else ''}): {e}")
            print(f"  !!! {msg}")
            self.plan_fallbacks.append({'t': t, 'visit': self._visit_idx,
                                        'reason': str(e)})
            # Retry next lap
            self._retry_offsets = 2 * math.pi
            self._queue.clear()
            self._seg = self._make_transit(
                start_th=max(float(self._seg.th1), float(self._theta_ref)))
            self.s = 0.0
            return False
        segs[-1].final_of_visit = True
        self._queue.extend(segs)
        dt_plan = _time.perf_counter() - t_wall0
        self.plan_times.append(dt_plan)
        print(f"  [plan] visit T{table+1} "
              f"({'place' if do_place else ''}"
              f"{'+' if do_place and do_pick else ''}"
              f"{'pick' if do_pick else ''}): "
              f"{len(segs)} segments in {dt_plan*1000:.0f} ms")
        if dt_plan > 2.0:
            print(f"  [plan] WARNING: planning took {dt_plan:.2f}s "
                  f"(> 2 s) — consider pre-planning earlier or a "
                  f"persistent parallel planning client.")
        return True

    def _place_ee_target(self, cube_target_world):
        """Return the EE action point that places the carried cube at target.

        The carried-object geometry is defined by the grasp constraint, not
        by cube velocity/tilt feedback.  The release controller therefore
        computes the EE placement point from the known grasp offset.  If no
        grasp is active, the nominal vertical offset is used.
        """
        cube_target = np.asarray(cube_target_world, dtype=float).reshape(3)
        if self._grasp_offset_world is None:
            offset = np.array([0.0, 0.0, -self.grasp_z_off], dtype=float)
        else:
            offset = np.asarray(self._grasp_offset_world, dtype=float).reshape(3)
            # Guard against a corrupted simulator sample.  A valid grasp
            # offset is of the order of GRASP_Z_OFF, not tens of centimetres.
            if (not np.all(np.isfinite(offset)) or
                    float(np.linalg.norm(offset)) > 0.25):
                offset = np.array([0.0, 0.0, -self.grasp_z_off], dtype=float)
        desired_cube = cube_target + np.array([0.0, 0.0, self.release_clearance])
        return desired_cube - offset

    @staticmethod
    def _join_paths(*paths):
        """Concatenate joint-space waypoint lists without duplicate seams."""
        out = []
        for path in paths:
            if path is None:
                continue
            for q in path:
                q = np.asarray(q, dtype=float)
                if not out or float(np.linalg.norm(q - out[-1])) > 1e-10:
                    out.append(q)
        return out

    def _plan_place_pick_pair(self, place_pt, item_pt, q_carry,
                              th_cursor, table_idx):
        """Plan PLACE -> short clearance -> next PICK without carry return.

        Sequence:
            PLC_RCH -> PLC_DRP -> PCK_RCH -> PCK_GRP
                    -> LIFT_VERT -> LIFT_RRT

        ``PCK_RCH`` is a direct transfer segment.  It first clears the newly
        placed object by a small vertical amount, then moves to the next
        pre-grasp hover.  Only after the new object is grasped do we return to
        the normal carry posture.  This removes the old
        PLC_DRP -> PLC_LFT -> PCK_RCH detour.
        """
        place_pt = np.asarray(place_pt, dtype=float).reshape(3)
        item_pt = np.asarray(item_pt, dtype=float).reshape(3)
        place_act = self._place_ee_target(place_pt)
        place_hover = np.array([
            place_act[0], place_act[1],
            self.table_surface + self.object_half + self.hover_dz
        ], dtype=float)
        pick_act = item_pt + np.array([0.0, 0.0, self.grasp_z_off])
        pick_hover = np.array([
            item_pt[0], item_pt[1],
            self.table_surface + self.object_half + self.hover_dz
        ], dtype=float)

        # --- PLACE timing / IK ---
        th_a_p = self._ahead(
            math.atan2(place_act[1] - self.CY, place_act[0] - self.CX),
            th_cursor)
        th_c_p = th_a_p - self._act_delta(place_act)
        th_r_p = max(th_c_p - self.ACT_SPAN, th_cursor + 0.02)
        th_d_p = max(th_c_p, th_r_p + 0.02)

        place_pose = self._predict_kuka_pose(th_d_p)
        self.shadow.set_base_pose(*place_pose)
        axis_top_place = self._topdown_world_axis(place_pose[1])
        q_place_hover = self.shadow.ik_axis(
            place_hover, rest_q=self._preferred_ik_rest(q_carry),
            collision_margin=self.MARGIN_TRANSFER,
            target_axis_world=axis_top_place)
        if q_place_hover is None:
            raise _PlanningError(f"PLC: IK to hover {place_hover} failed")
        q_place_act = self.shadow.ik_axis(
            place_act, rest_q=q_place_hover,
            collision_margin=self.MARGIN_ACT + 0.002,
            target_axis_world=axis_top_place)
        if q_place_act is None:
            raise _PlanningError(f"PLC: IK to action point {place_act} failed")

        p_place_reach = self._rrt(q_carry, q_place_hover,
                                  self.MARGIN_TRANSFER,
                                  "PLC: carry->hover")
        p_place_act = self._cartesian_ik_path(
            q_place_hover, place_hover, place_act,
            self.MARGIN_ACT, axis_top_place, "PLC: hover->act")

        # --- PICK timing / IK ---
        # Start the pick reach immediately after the place action.  _ahead()
        # and the max() below avoid an unnecessary full lap when both objects
        # are on the same table.
        th_a_k = self._ahead(
            math.atan2(pick_act[1] - self.CY, pick_act[0] - self.CX),
            th_d_p)
        th_c_k = th_a_k - self._act_delta(pick_act)
        th_r_k = max(th_c_k - self.ACT_SPAN, th_d_p + 0.02)
        th_d_k = max(th_c_k, th_r_k + 0.02)
        th_l_k = th_d_k + self.LIFT_ARC
        th_v_k = th_d_k + self.LIFT_VERT_FRACTION * (th_l_k - th_d_k)

        pick_pose = self._predict_kuka_pose(th_d_k)
        self.shadow.set_base_pose(*pick_pose)
        axis_top_pick = self._topdown_world_axis(pick_pose[1])
        q_pick_hover = self.shadow.ik_axis(
            pick_hover, rest_q=self._preferred_ik_rest(q_place_act),
            collision_margin=self.MARGIN_TRANSFER,
            target_axis_world=axis_top_pick)
        if q_pick_hover is None:
            raise _PlanningError(f"PCK: IK to hover {pick_hover} failed")
        q_pick_act = self.shadow.ik_axis(
            pick_act, rest_q=q_pick_hover,
            collision_margin=self.MARGIN_ACT + 0.002,
            target_axis_world=axis_top_pick)
        if q_pick_act is None:
            raise _PlanningError(f"PCK: IK to action point {pick_act} failed")

        # Short safety lift after release.  This is deliberately not the old
        # carry/home posture: it only clears the just-placed object/table.
        clear_w = place_act.copy()
        clear_w[2] += self.TRANSFER_CLEARANCE
        self.shadow.set_base_pose(*place_pose)
        q_clear = self.shadow.ik_axis(
            clear_w, rest_q=q_place_act,
            collision_margin=self.MARGIN_ACT,
            target_axis_world=axis_top_place)

        if q_clear is not None:
            p_up = self._cartesian_ik_path(
                q_place_act, place_act, clear_w, self.MARGIN_ACT,
                axis_top_place, "place->short-clearance")
            # Check the cross-table part near the nominal pick geometry.
            self.shadow.set_base_pose(*self._predict_kuka_pose(th_r_k))
            p_cross = self._rrt(
                q_clear, q_pick_hover, self.MARGIN_TRANSFER,
                "clearance->next-pregrasp",
                axis_world=axis_top_pick)
            p_transfer = self._join_paths(p_up, p_cross)
            transfer_mode = "short-clearance-constrained"
            transfer_topdown = True
        else:
            # Rare fallback: use the old carry posture only when the local
            # clearance IK itself is unavailable.
            print("  [plan] NOTE: short-clearance IK unavailable; "
                  "using carry posture as transfer fallback")
            self.shadow.set_base_pose(*self._predict_kuka_pose(th_r_k))
            p_back = self._rrt(q_place_act, q_carry, self.MARGIN_ACT,
                               "place->carry fallback")
            p_fwd = self._rrt(q_carry, q_pick_hover,
                              self.MARGIN_TRANSFER,
                              "carry->next-pregrasp fallback")
            p_transfer = self._join_paths(p_back, p_fwd)
            transfer_mode = "carry-fallback"
            transfer_topdown = False

        p_pick_act = self._cartesian_ik_path(
            q_pick_hover, pick_hover, pick_act, self.MARGIN_ACT,
            axis_top_pick, "PCK: hover->act")
        # Always lift vertically back to the pre-grasp height.  The following
        # retract is now a *tool-axis-constrained* RRT: every checked edge must
        # remain close to top-down tilt while wrist yaw stays free.
        p_pick_vertical_lift = list(reversed(p_pick_act))
        q_carry_top, axis_top_carry = self._topdown_carry_q(
            pick_pose, q_pick_hover, self.MARGIN_TRANSFER)
        if q_carry_top is None:
            print("  [plan] NOTE: top-down carry IK unavailable; "
                  "using unconstrained hover->carry fallback")
            p_pick_retract = self._rrt(
                q_pick_hover, q_carry, self.MARGIN_TRANSFER,
                "PCK: hover->carry fallback")
            pick_lift_topdown = False
        else:
            p_pick_retract = self._rrt(
                q_pick_hover, q_carry_top, self.MARGIN_TRANSFER,
                "PCK: hover->carry [top-down-axis constrained]",
                axis_world=axis_top_carry)
            pick_lift_topdown = True
        def mk(name, path, th0, th1, v_nom, min_dur, kind,
               a_start, a_end, action=None, target=None,
               cartesian_world=None, topdown=False):
            traj = ProgressTrajectory(path)
            ee_local = SampledPath3.from_trajectory(
                traj, self.shadow.fk_local, n=60)
            ee_pose_local = None

            auto_min = float(min_dur)
            cart_len = float(np.linalg.norm(
                np.diff(ee_local.points, axis=0), axis=1).sum())
            if kind == 'motion' and name.endswith("RRT"):
                auto_min = max(
                    auto_min,
                    float(traj.length) / self.rrt_joint_path_speed,
                    cart_len / self.rrt_cart_ref_speed,
                )
            elif kind == 'motion' and name.endswith("RCH"):
                auto_min = max(
                    auto_min,
                    float(traj.length) / self.reach_joint_path_speed,
                    cart_len / self.reach_cart_ref_speed,
                )

            dur = max((th1 - th0) * self.R / v_nom, auto_min)
            seg_obj = _Segment(
                name, kind, th0, th1, dur,
                traj=traj, ee_local=ee_local,
                ee_pose_local=ee_pose_local,
                anchor_start=a_start, anchor_end=a_end,
                action=action, target_world=target,
                min_duration=auto_min,
                orientation_local_quat=(
                    self.top_down_local_quat
                    if (self.top_down_grasp and topdown) else None),
                cartesian_world=cartesian_world)
            seg_obj.planned_joint_length = float(traj.length)
            seg_obj.planned_cart_length = cart_len
            return seg_obj

        place_reach = mk("PLC_RCH", p_place_reach,
                         th_cursor, th_r_p, self.V_REACH, 1.6,
                         'motion', ('body', self.carry_offset_body),
                         ('world', place_hover), topdown=False)
        place_drop = mk("PLC_DRP", p_place_act,
                        th_r_p, th_d_p, self.V_ACT, 2.0,
                        'action', ('world', place_hover),
                        ('world', place_act),
                        action=('release', table_idx), target=place_act,
                        cartesian_world=(place_hover, place_act),
                        topdown=True)
        place_drop.place_xy = np.asarray(place_pt[:2], dtype=float).copy()

        transfer = mk("PCK_RCH", p_transfer,
                      th_d_p, th_r_k, self.V_REACH, 1.2,
                      'motion', ('world', place_act),
                      ('world', pick_hover), topdown=transfer_topdown)
        # Diagnostic only; plotting/log code can inspect this if desired.
        transfer.transfer_mode = transfer_mode

        pick_grasp = mk("PCK_GRP", p_pick_act,
                        th_r_k, th_d_k, self.V_ACT, 2.0,
                        'action', ('world', pick_hover),
                        ('world', pick_act),
                        action=('grasp', table_idx), target=pick_act,
                        cartesian_world=(pick_hover, pick_act),
                        topdown=True)
        pick_lift_vert = mk(
            "LIFT_VERT", p_pick_vertical_lift,
            th_d_k, th_v_k, self.V_LIFT, 0.55,
            'motion', ('world', pick_act), ('world', pick_hover),
            cartesian_world=(pick_act, pick_hover), topdown=True)
        pick_lift_rrt = mk(
            "LIFT_RRT", p_pick_retract,
            th_v_k, th_l_k, self.V_LIFT, 0.75,
            'motion', ('world', pick_hover),
            ('body', self.carry_offset_body),
            topdown=pick_lift_topdown)

        print(f"  [plan] T{table_idx+1} place->pick transfer: "
              f"{transfer_mode}, clearance={self.TRANSFER_CLEARANCE*1000:.0f} mm")
        return [place_reach, place_drop, transfer, pick_grasp,
                pick_lift_vert, pick_lift_rrt], th_l_k

    def _plan_trio(self, prefix, hover_w, act_w, q_carry, th_cursor,
                   action):
        """Plan carry→hover→act→carry as three progress segments."""
        th_a = self._ahead(
            math.atan2(act_w[1] - self.CY, act_w[0] - self.CX),
            th_cursor)
        # End the descend window where the base passes the target at a
        # comfortable reach distance (see _act_delta) — the dwell and
        # the grasp/release happen exactly there.
        th_c  = th_a - self._act_delta(act_w)
        th_r1 = max(th_c - self.ACT_SPAN, th_cursor + 0.02)
        th_d1 = max(th_c, th_r1 + 0.02)
        th_l1 = th_d1 + self.LIFT_ARC
        th_v1 = th_d1 + self.LIFT_VERT_FRACTION * (th_l1 - th_d1)

        # Nominal base pose at the action moment (where positional
        # accuracy and table clearance matter most)
        pose = self._predict_kuka_pose(th_d1)
        self.shadow.set_base_pose(*pose)
        axis_top = self._topdown_world_axis(pose[1])

        q_hover = self.shadow.ik_axis(
            hover_w, rest_q=self._preferred_ik_rest(q_carry),
            collision_margin=self.MARGIN_TRANSFER,
            target_axis_world=axis_top)
        if q_hover is None:
            raise _PlanningError(f"{prefix}: IK to hover {hover_w} failed")
        q_act = self.shadow.ik_axis(
            act_w, rest_q=q_hover,
            collision_margin=self.MARGIN_ACT + 0.002,
            target_axis_world=axis_top)
        if q_act is None:
            raise _PlanningError(f"{prefix}: IK to action point {act_w} "
                                 f"failed")

        p1 = self._rrt(q_carry, q_hover, self.MARGIN_TRANSFER,
                       f"{prefix}: carry->hover")
        p2 = self._cartesian_ik_path(
            q_hover, hover_w, act_w, self.MARGIN_ACT, axis_top,
            f"{prefix}: hover->act")
        # Lift back to hover before retracting.  Prefer a top-down carry IK and
        # use constrained RRT for the retract so the gripper does not tilt
        # away from top-down; wrist yaw is intentionally unconstrained.
        p_vertical_lift = list(reversed(p2))
        q_carry_top, axis_top_carry = self._topdown_carry_q(
            pose, q_hover, self.MARGIN_TRANSFER)
        if q_carry_top is None:
            print(f"  [plan] NOTE: {prefix} top-down carry IK unavailable; "
                  "using unconstrained retract fallback")
            p_retract = self._rrt(q_hover, q_carry, self.MARGIN_TRANSFER,
                                  f"{prefix}: hover->carry fallback")
            lift_topdown = False
        else:
            p_retract = self._rrt(
                q_hover, q_carry_top, self.MARGIN_TRANSFER,
                f"{prefix}: hover->carry [top-down-axis constrained]",
                axis_world=axis_top_carry)
            lift_topdown = True
        carry_off = self.carry_offset_body
        names = {
            "PCK": ("PCK_RCH", "PCK_GRP", "LIFT_VERT", "LIFT_RRT"),
            "PLC": ("PLC_RCH", "PLC_DRP",
                    "PLC_LFT_VERT", "PLC_LFT_RRT"),
        }[prefix]

        def mk(name, path, th0, th1, v_nom, min_dur, kind,
               a_start, a_end, action=None, target=None,
               cartesian_world=None, topdown=False):
            traj = ProgressTrajectory(path)
            ee_local = SampledPath3.from_trajectory(
                traj, self.shadow.fk_local, n=60)
            ee_pose_local = None

            auto_min = float(min_dur)
            cart_len = float(np.linalg.norm(
                np.diff(ee_local.points, axis=0), axis=1).sum())
            if kind == 'motion' and name.endswith("RRT"):
                auto_min = max(
                    auto_min,
                    float(traj.length) / self.rrt_joint_path_speed,
                    cart_len / self.rrt_cart_ref_speed,
                )
            elif kind == 'motion' and name.endswith("RCH"):
                auto_min = max(
                    auto_min,
                    float(traj.length) / self.reach_joint_path_speed,
                    cart_len / self.reach_cart_ref_speed,
                )

            dur = max((th1 - th0) * self.R / v_nom, auto_min)
            seg_obj = _Segment(
                name, kind, th0, th1, dur,
                traj=traj, ee_local=ee_local,
                ee_pose_local=ee_pose_local,
                anchor_start=a_start, anchor_end=a_end,
                action=action, target_world=target,
                min_duration=auto_min,
                orientation_local_quat=(
                    self.top_down_local_quat
                    if (self.top_down_grasp and topdown) else None),
                cartesian_world=cartesian_world)
            seg_obj.planned_joint_length = float(traj.length)
            seg_obj.planned_cart_length = cart_len
            return seg_obj

        segs = [
            mk(names[0], p1, th_cursor, th_r1, self.V_REACH, 1.6,
               'motion', ('body', carry_off), ('world', hover_w),
               topdown=False),
            mk(names[1], p2, th_r1, th_d1, self.V_ACT, 2.0,
               'action', ('world', hover_w), ('world', act_w),
               action=action, target=np.asarray(act_w, float),
               cartesian_world=(hover_w, act_w), topdown=True),
            mk(names[2], p_vertical_lift,
               th_d1, th_v1, self.V_LIFT, 0.55,
               'motion', ('world', act_w), ('world', hover_w),
               cartesian_world=(act_w, hover_w), topdown=True),
            mk(names[3], p_retract,
               th_v1, th_l1, self.V_LIFT, 0.75,
               'motion', ('world', hover_w), ('body', carry_off),
               topdown=lift_topdown),
        ]
        if action[0] == 'release':
            segs[1].place_xy = np.asarray(act_w[:2], dtype=float).copy()
        return segs, th_l1

    def _rrt(self, qa, qb, margin, label, axis_world=None,
             axis_tol=math.radians(12.0)):
        """RRT-Connect with optional top-down *tool-axis* projection.

        Ordinary calls preserve the historical collision-only joint-space
        planner.  When ``axis_world`` is supplied, each steered sample is
        projected by pose IK so the EE local +Z axis points in that direction
        while preserving the sample's own wrist yaw as much as possible.
        Every edge is then checked for BOTH collision and tool-axis tilt.
        """
        qa = np.asarray(qa, dtype=float)
        qb = np.asarray(qb, dtype=float)
        a_des = (None if axis_world is None else
                 np.asarray(axis_world, dtype=float).reshape(3))
        if a_des is not None:
            a_des = a_des / max(float(np.linalg.norm(a_des)), 1e-12)
        constrained = a_des is not None

        for m in (margin, max(margin - 0.008, 0.002)):
            collision_only = self.shadow.make_collision_fn(m)

            if constrained:
                def invalid(q, _collision=collision_only, _axis=a_des):
                    if _collision(q):
                        return True
                    _, orn = self.shadow.fk_world_pose(q)
                    return self.shadow.tool_axis_error(orn, _axis) > axis_tol

                def project(q_trial, _axis=a_des, _m=m):
                    # Preserve both the trial Cartesian position and its
                    # current yaw about the tool axis.  Only tilt is projected
                    # back to the top-down manifold.
                    pos, orn_trial = self.shadow.fk_world_pose(q_trial)
                    q_target = self.shadow._quat_with_tool_axis(
                        orn_trial, _axis)
                    return self.shadow.ik(
                        pos, rest_q=q_trial, tries=5, pos_tol=0.012,
                        collision_margin=_m,
                        target_orientation_world=q_target,
                        orn_tol=min(math.radians(8.0), 0.8 * axis_tol))

                collision_fn = invalid
                project_fn = project
                step_size = min(self.RRT_STEP, 0.16)
                edge_res = min(self.RRT_EDGE_RES, 0.025)
                max_iters = max(self.RRT_MAX_ITERS, 4500)
            else:
                collision_fn = collision_only
                project_fn = None
                step_size = self.RRT_STEP
                edge_res = self.RRT_EDGE_RES
                max_iters = self.RRT_MAX_ITERS

            planner = RRTConnectJointPlanner(
                self.q_lower, self.q_upper, collision_fn,
                step_size=step_size, edge_resolution=edge_res,
                max_iters=max_iters,
                seed=self._rng_seed + len(self.plan_times),
                project_fn=project_fn)
            path = planner.plan(qa, qb)
            if path is not None:
                if m != margin:
                    print(f"  [plan] NOTE: {label} planned with relaxed "
                          f"margin {m*1000:.0f} mm (from "
                          f"{margin*1000:.0f} mm)")
                if constrained:
                    max_axis = 0.0
                    for q in path:
                        _, orn = self.shadow.fk_world_pose(q)
                        max_axis = max(
                            max_axis,
                            self.shadow.tool_axis_error(orn, a_des))
                    print(f"  [plan] constrained RRT {label}: "
                          f"{len(path)} nodes, max tool-axis tilt "
                          f"{math.degrees(max_axis):.1f} deg")
                return shortcut_path(path, planner.edge_free, planner.rng)

        if constrained:
            # Do not silently replace a requested orientation-constrained path
            # by an unchecked straight segment.  Retrying the visit is safer
            # and makes failures visible in the experiment log.
            raise _PlanningError(
                f"tool-axis constrained RRT failed for {label} "
                f"(reason={planner.last_failure})")

        print(f"  [plan] WARNING: RRT-Connect failed for {label} "
              f"(reason={planner.last_failure}); falling back to a "
              f"straight joint-space segment (NOT collision-checked).")
        self.plan_fallbacks.append({'t': self._t_prev, 'label': label,
                                    'reason': f"rrt:{planner.last_failure}"})
        return [qa.copy(), qb.copy()]

    # ==================================================================
    # Internals — physical actions & failure handling
    # ==================================================================

    def _perform_action(self, seg, t, err_act):
        verb, table = seg.action
        if verb == 'grasp':
            self._do_grasp(table, t, err_act)
        else:
            self._do_release(table, t, err_act)

    def _do_grasp(self, table_idx, t, err_act):
        if self.carried is not None:
            return
        items = self.table_items.get(table_idx, [])
        if not items:
            print(f"  !!! No item on T{table_idx+1}!")
            return
        cid_cube = items[0]

        # V10.5 orientation-decoupled suspension geometry.
        #
        # The previous point constraint placed its parent pivot below the EE
        # frame by a vector expressed in the wrist/link frame.  Because the
        # controller is position-only, wrist orientation can change during
        # transport; that rotated parent offset and created a systematic XY
        # displacement of the cube at placement (~3-4 cm in V10.4 logs).
        #
        # Here the parent pivot is the EE FRAME ORIGIN itself.  The cube-side
        # pivot is GRASP_Z_OFF above its centre.  Therefore wrist orientation
        # no longer creates a lateral attachment bias, while the cube remains
        # free to stay upright under gravity.  At grasp, the actual cube-to-EE
        # position offset is also recorded and used when generating the later
        # placement EE target.
        with self._lock:
            ls = self.p.getLinkState(self.kuka_id, self.ee_index,
                                     computeForwardKinematics=True)
            c_pos, _ = self.p.getBasePositionAndOrientation(cid_cube)

            # createConstraint() expects parentFramePosition in the parent
            # link's inertial/COM frame.  Convert the kinematic EE frame origin
            # (ls[4]) into that frame (ls[0], ls[1]).
            inv_pos, inv_orn = self.p.invertTransform(ls[0], ls[1])
            rel_pos, _ = self.p.multiplyTransforms(
                inv_pos, inv_orn, ls[4], [0, 0, 0, 1])

            self.grasp_cid = self.p.createConstraint(
                self.kuka_id, self.ee_index, cid_cube, -1,
                self.p.JOINT_POINT2POINT, [0, 0, 0],
                parentFramePosition=list(rel_pos),
                childFramePosition=[0, 0, self.grasp_z_off])
            self.p.changeConstraint(self.grasp_cid, maxForce=500)
            self.p.changeDynamics(cid_cube, -1,
                                  linearDamping=0.8, angularDamping=0.9)

        measured_offset = (np.asarray(c_pos, dtype=float)
                           - np.asarray(ls[4], dtype=float))
        # The point constraint enforces the EE-frame origin to coincide with
        # the cube-side pivot at +GRASP_Z_OFF.  Hence the intended carried
        # object transform is known from grasp geometry and does not depend
        # on the pre-attach tracking residual.  Keep the measured offset only
        # as a diagnostic; use the constraint-defined offset for placement.
        self._grasp_offset_world = np.array(
            [0.0, 0.0, -self.grasp_z_off], dtype=float)
        self.carried = cid_cube
        items.remove(cid_cube)
        if self._dyn_dist is not None:
            self._dyn_dist.set_carrying(True)
        self.events.append({'type': 'grasp', 't': t, 'cube': cid_cube,
                            'table': table_idx, 'ee_err': err_act,
                            'grasp_offset_world': self._grasp_offset_world.copy(),
                            'measured_grasp_offset_world': measured_offset.copy()})
        self.grasp_monitors.append({'t0': t, 'cube': cid_cube,
                                    'table': table_idx,
                                    'samples': [], 'done': False})
        print(f"  *** GRASPED cube {cid_cube} from T{table_idx+1} "
              f"(ee_err={err_act*1000:.0f} mm, "
              f"grasp_geom_mm={[round(x*1000,1) for x in self._grasp_offset_world]}, "
              f"measured_pre_attach_mm={[round(x*1000,1) for x in measured_offset]}) ***")

    def _do_release(self, table_idx, t, err_act):
        if self.carried is None or self.grasp_cid is None:
            return
        with self._lock:
            c_pos, c_orn = self.p.getBasePositionAndOrientation(self.carried)
            c_vel, _ = self.p.getBaseVelocity(self.carried)
        _rest_z = self.table_surface + self.object_half
        _up = np.array(self.p.getMatrixFromQuaternion(c_orn)).reshape(3, 3)[:, 2]
        print(f"      release state: cube z-above-rest="
              f"{(c_pos[2]-_rest_z)*1000:.0f} mm  "
              f"|v|={np.linalg.norm(c_vel):.3f} m/s  "
              f"tilt={math.degrees(math.acos(np.clip(_up[2], -1, 1))):.1f} deg")
        with self._lock:
            # Restore the cube's free-body damping before letting go
            self.p.changeDynamics(self.carried, -1,
                                  linearDamping=0.04, angularDamping=0.04)
            self.p.removeConstraint(self.grasp_cid)
        self.table_items.setdefault(table_idx, []).append(self.carried)
        if self._dyn_dist is not None:
            self._dyn_dist.set_carrying(False)
        seg = self._seg
        release_status = self._release_cube_status(seg)
        self.events.append({'type': 'release', 't': t,
                            'cube': self.carried, 'table': table_idx,
                            'ee_err': err_act,
                            'cube_residual': release_status['residual_norm'],
                            'cube_speed': release_status['speed'],
                            'cube_ang_speed': release_status['ang_speed'],
                            'cube_tilt_ang_speed': release_status['tilt_ang_speed'],
                            'cube_tilt_deg': release_status['tilt_deg'],
                            'aim_count': seg.aim_count})
        self.placements.append({'cube': self.carried, 'table': table_idx,
                                'target': self._place_target.copy(),
                                't': t,
                                'release_cube_residual': release_status['residual_norm'],
                                'release_cube_speed': release_status['speed'],
                                'release_cube_ang_speed': release_status['ang_speed'],
                                'release_cube_tilt_ang_speed': release_status['tilt_ang_speed'],
                                'release_cube_tilt_deg': release_status['tilt_deg'],
                                'release_aim_count': seg.aim_count})
        print(f"  *** RELEASED cube {self.carried} on T{table_idx+1} "
              f"(ee_err={err_act*1000:.0f} mm) ***")
        self.carried = None
        self.grasp_cid = None
        self._grasp_offset_world = None
        self._place_target = None
        self.place_count += 1

    def _abort_attempt(self, t, err_act):
        """Terminal-pose timeout: cancel this grasp/release attempt, retreat
        to carry, and retry the same visit one lap later.  In V10.4 this can
        only be triggered by failure to reach the EE position tolerance."""
        seg = self._seg
        verb, table = seg.action
        tol = (self.release_tolerance if verb == 'release'
               else self.grasp_tolerance)
        detail = (f"ee_err={err_act*1000:.0f} mm "
                  f"(tol={tol*1000:.0f} mm)")
        msg = (f"DWELL TIMEOUT in {seg.name} (action={verb} T{table+1}): "
               f"{detail} after {self.dwell_timeout:.1f}s — aborting "
               f"attempt, retrying next lap")
        print(f"  !!! {msg}")
        self.timeout_events.append({'t': t, 'segment': seg.name,
                                    'action': verb, 'table': table,
                                    'ee_err': err_act})
        self._queue.clear()
        # Retreat: straight joint-space path back to carry (planned at
        # the current predicted pose; margin relaxed since we start
        # near the table).
        q_now = self._read_arm_q()
        pose = self._predict_kuka_pose(self._theta_base)
        self.shadow.set_base_pose(*pose)
        path = self._rrt(q_now, self._q_carry, 0.005, "retreat->carry")
        traj = ProgressTrajectory(path)
        ee_local = SampledPath3.from_trajectory(
            traj, self.shadow.fk_local, n=40)
        ee_pose_local = None
        th0 = self._theta_ref
        retreat = _Segment("LIFT", 'motion', th0, th0 + 0.15, 1.5,
                           traj=traj, ee_local=ee_local,
                           ee_pose_local=ee_pose_local,
                           anchor_start=('world',
                                         self._last_target.copy()),
                           anchor_end=('body', self.carry_offset_body))
        self._queue.append(retreat)
        # Do NOT advance visit index; add a full lap for the retry.
        self._retry_offsets = 2 * math.pi
        self._planned = False
        self._advance(t)


class _PlanningError(RuntimeError):
    pass
