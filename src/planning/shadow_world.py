"""Shadow (planning) world: a private DIRECT PyBullet client for
collision checking, IK and FK during motion planning.

The main simulation runs the *real* robot; testing candidate
configurations there with resetJointState would teleport the running
arm.  Instead, this module owns a second physics client containing only
a fixed-base copy of the arm and the (static) tables.  The planner
"thinks" in this shadow world; the real robot is never touched.

All PyBullet calls go through ``sim_lock`` to follow the repo's
thread-safety convention (pass the same lock the subsystems use).
"""

import math
import threading
import numpy as np


class ShadowArmWorld:
    """Planning-only copy of the arm + tables in a DIRECT client.

    Parameters
    ----------
    sim : module
        The ``pybullet`` module handle.
    arm_urdf : str
        URDF path of the arm (resolved through pybullet_data search
        path), e.g. ``"kuka_iiwa/model.urdf"``.
    joint_indices : list[int]
        Indices of the controlled revolute joints (same as main sim).
    ee_index : int
        End-effector link index.
    q_lower, q_upper : array-like (nj,)
        Joint limits.
    tables : list[(pos3, yaw)]
        World poses of the tables to load as static obstacles.
    table_urdf : str
        URDF of the table obstacle.
    default_margin : float
        Required clearance (m) between arm and any table for a
        configuration to count as collision-free.
    sim_lock : threading.Lock or None
        Shared lock guarding PyBullet access.
    """

    def __init__(self, sim, arm_urdf, joint_indices, ee_index,
                 q_lower, q_upper, tables=(),
                 table_urdf="table/table.urdf",
                 default_margin=0.02, sim_lock=None,
                 extra_search_path=None):
        self.p = sim
        self._lock = sim_lock or threading.Lock()
        self.joint_ids = list(joint_indices)
        self.ee_index  = ee_index
        self.nj        = len(self.joint_ids)
        self.q_lower   = np.asarray(q_lower, dtype=float)
        self.q_upper   = np.asarray(q_upper, dtype=float)
        self.default_margin = default_margin

        with self._lock:
            self.cid = sim.connect(sim.DIRECT)
            import pybullet_data
            sim.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                        physicsClientId=self.cid)
            if extra_search_path:
                sim.setAdditionalSearchPath(extra_search_path,
                                            physicsClientId=self.cid)
            self.arm = sim.loadURDF(arm_urdf, [0, 0, 0],
                                    useFixedBase=True,
                                    physicsClientId=self.cid)
            self.table_ids = []
            for pos, yaw in tables:
                tid = sim.loadURDF(
                    table_urdf, list(pos),
                    sim.getQuaternionFromEuler([0, 0, yaw]),
                    useFixedBase=True, physicsClientId=self.cid)
                self.table_ids.append(tid)

            # Map controlled joints -> position in the IK solution
            # vector (IK returns one value per non-fixed joint).
            movable = [
                j for j in range(sim.getNumJoints(self.arm,
                                                  physicsClientId=self.cid))
                if sim.getJointInfo(self.arm, j,
                                    physicsClientId=self.cid)[2]
                != sim.JOINT_FIXED
            ]
            self._ik_map = [movable.index(j) for j in self.joint_ids]

        self._rng = np.random.default_rng(12345)

    # ------------------------------------------------------------------
    # Pose / configuration
    # ------------------------------------------------------------------

    def set_base_pose(self, pos, orn):
        """Move the shadow arm's base to a (predicted) world pose."""
        with self._lock:
            self.p.resetBasePositionAndOrientation(
                self.arm, list(pos), list(orn), physicsClientId=self.cid)

    def get_base_pose(self):
        with self._lock:
            pos, orn = self.p.getBasePositionAndOrientation(
                self.arm, physicsClientId=self.cid)
        return np.array(pos), np.array(orn)

    def _set_config_nolock(self, q):
        for idx, ji in enumerate(self.joint_ids):
            self.p.resetJointState(self.arm, ji, float(q[idx]),
                                   physicsClientId=self.cid)

    # ------------------------------------------------------------------
    # Collision checking (PyBullet closest-point queries — no manual
    # collision geometry)
    # ------------------------------------------------------------------

    def in_collision(self, q, margin=None):
        """True when the arm at configuration q is closer than
        ``margin`` to any table (or penetrating it)."""
        m = self.default_margin if margin is None else margin
        with self._lock:
            self._set_config_nolock(q)
            for tid in self.table_ids:
                pts = self.p.getClosestPoints(
                    self.arm, tid, distance=m, physicsClientId=self.cid)
                if pts:
                    for pt in pts:
                        if pt[8] < m:      # contactDistance
                            return True
        return False

    def make_collision_fn(self, margin=None):
        """Bind a margin into a ``collision_fn(q)`` for the planner."""
        return lambda q: self.in_collision(q, margin=margin)

    def min_clearance(self, q):
        """Smallest arm-to-table distance at configuration q (probes up
        to 1 m; returns 1.0 if farther)."""
        best = 1.0
        with self._lock:
            self._set_config_nolock(q)
            for tid in self.table_ids:
                pts = self.p.getClosestPoints(
                    self.arm, tid, distance=1.0, physicsClientId=self.cid)
                for pt in pts:
                    best = min(best, pt[8])
        return best

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------

    def fk_world(self, q):
        """EE world position (in the shadow world) at configuration q."""
        with self._lock:
            self._set_config_nolock(q)
            ls = self.p.getLinkState(self.arm, self.ee_index,
                                     computeForwardKinematics=True,
                                     physicsClientId=self.cid)
        return np.array(ls[4])

    def fk_world_pose(self, q):
        """EE world pose ``(position, quaternion_xyzw)`` at configuration q."""
        with self._lock:
            self._set_config_nolock(q)
            ls = self.p.getLinkState(self.arm, self.ee_index,
                                     computeForwardKinematics=True,
                                     physicsClientId=self.cid)
        return np.array(ls[4]), np.array(ls[5])

    def fk_local_pose(self, q):
        """EE pose expressed in the arm-base frame as (p_local, R_local)."""
        ee, ee_orn = self.fk_world_pose(q)
        with self._lock:
            pos, orn = self.p.getBasePositionAndOrientation(
                self.arm, physicsClientId=self.cid)
            R_base = np.array(
                self.p.getMatrixFromQuaternion(orn)).reshape(3, 3)
            R_ee = np.array(
                self.p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
        return R_base.T @ (ee - np.array(pos)), R_base.T @ R_ee

    def fk_local(self, q):
        """Backward-compatible local EE position FK."""
        return self.fk_local_pose(q)[0]

    @staticmethod
    def _quat_angle(q_a, q_b):
        """Shortest angular distance between two xyzw quaternions (rad)."""
        qa = np.asarray(q_a, dtype=float)
        qb = np.asarray(q_b, dtype=float)
        qa = qa / max(float(np.linalg.norm(qa)), 1e-12)
        qb = qb / max(float(np.linalg.norm(qb)), 1e-12)
        dot = float(np.clip(abs(np.dot(qa, qb)), 0.0, 1.0))
        return 2.0 * math.acos(dot)

    @staticmethod
    def _matrix_to_quat(R):
        """Convert a 3x3 rotation matrix to a normalized xyzw quaternion."""
        R = np.asarray(R, dtype=float).reshape(3, 3)
        tr = float(np.trace(R))
        if tr > 0.0:
            s = math.sqrt(tr + 1.0) * 2.0
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = math.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1e-15)) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = math.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1e-15)) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1e-15)) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        q = np.array([x, y, z, w], dtype=float)
        return q / max(float(np.linalg.norm(q)), 1e-12)

    def _quat_with_tool_axis(self, reference_orn, target_axis_world,
                             yaw_offset=0.0):
        """Return a pose orientation whose local +Z axis hits ``target_axis``.

        Only the tool-axis direction is prescribed.  Rotation about that axis
        is selected to stay as close as possible to ``reference_orn``; an
        optional ``yaw_offset`` explores alternative redundant wrist yaw.
        This is the key Stage-3 distinction from the old full-quaternion
        constraint.
        """
        R0 = np.asarray(self.p.getMatrixFromQuaternion(
            list(np.asarray(reference_orn, dtype=float).reshape(4))),
            dtype=float).reshape(3, 3)
        z = np.asarray(target_axis_world, dtype=float).reshape(3)
        z /= max(float(np.linalg.norm(z)), 1e-12)

        # Preserve the reference frame's yaw about the new tool axis by
        # projecting its x-axis into the plane normal to z.  If x happens to
        # be nearly parallel to z, use y as the seed instead.
        x_seed = R0[:, 0]
        x = x_seed - z * float(np.dot(z, x_seed))
        if float(np.linalg.norm(x)) < 1e-7:
            y_seed = R0[:, 1]
            x = np.cross(y_seed, z)
        x /= max(float(np.linalg.norm(x)), 1e-12)
        y = np.cross(z, x)
        y /= max(float(np.linalg.norm(y)), 1e-12)

        if abs(float(yaw_offset)) > 1e-12:
            c, s = math.cos(float(yaw_offset)), math.sin(float(yaw_offset))
            x, y = c * x + s * y, -s * x + c * y

        R = np.column_stack((x, y, z))
        return self._matrix_to_quat(R)

    def tool_axis_error(self, orn, target_axis_world):
        """Angle (rad) between EE local +Z and a desired world direction."""
        R = np.asarray(self.p.getMatrixFromQuaternion(
            list(np.asarray(orn, dtype=float).reshape(4))),
            dtype=float).reshape(3, 3)
        a = R[:, 2]
        d = np.asarray(target_axis_world, dtype=float).reshape(3)
        d /= max(float(np.linalg.norm(d)), 1e-12)
        return math.acos(float(np.clip(np.dot(a, d), -1.0, 1.0)))

    def ik_axis(self, target_world, rest_q, target_axis_world,
                tries_per_yaw=4, pos_tol=0.01, collision_margin=None,
                axis_tol=math.radians(8.0),
                yaw_offsets=(0.0, math.pi/4, -math.pi/4,
                             math.pi/2, -math.pi/2)):
        """Pose IK with a *tool-axis* constraint and free wrist yaw.

        PyBullet's native IK accepts a full quaternion, not a direction-only
        constraint.  We therefore generate several full orientations that all
        share the required local +Z direction but differ by rotation about
        that direction, then keep the feasible solution closest to ``rest_q``.
        The result satisfies the requested top-down axis without needlessly
        consuming the seventh joint to lock yaw.
        """
        rest_q = np.asarray(rest_q, dtype=float).reshape(self.nj)
        _, rest_orn = self.fk_world_pose(rest_q)
        best_q, best_score = None, np.inf
        scale = np.maximum(self.q_upper - self.q_lower, 1e-6)
        for yaw in yaw_offsets:
            q_target = self._quat_with_tool_axis(
                rest_orn, target_axis_world, yaw_offset=float(yaw))
            q = self.ik(
                target_world, rest_q=rest_q,
                tries=max(int(tries_per_yaw), 1), pos_tol=pos_tol,
                collision_margin=collision_margin,
                target_orientation_world=q_target,
                orn_tol=min(float(axis_tol), math.radians(8.0)))
            if q is None:
                continue
            fk_pos, fk_orn = self.fk_world_pose(q)
            if float(np.linalg.norm(fk_pos - np.asarray(target_world))) >= pos_tol:
                continue
            if self.tool_axis_error(fk_orn, target_axis_world) >= axis_tol:
                continue
            # Normalize by joint range so a large-range joint does not
            # dominate the continuity preference.
            score = float(np.linalg.norm((np.asarray(q) - rest_q) / scale))
            if score < best_score:
                best_q, best_score = np.asarray(q, dtype=float), score
        return best_q

    def ik(self, target_world, rest_q, tries=12, pos_tol=0.01,
           collision_margin=None, target_orientation_world=None,
           orn_tol=math.radians(7.5)):
        """IK to ``target_world`` near ``rest_q``.

        Retries with jittered rest poses; each candidate is verified by
        FK.  When ``target_orientation_world`` is supplied, the geometric
        EE orientation is constrained and independently verified as well.
        When ``collision_margin`` is given, only configurations with at least
        that much table clearance are accepted.  Oriented grasp IK is strict:
        if no collision-free pose satisfies both tolerances, ``None`` is
        returned.  The historical position-only fallback is preserved for
        legacy callers.
        """
        target = list(np.asarray(target_world, dtype=float))
        target_orn = (None if target_orientation_world is None else
                      np.asarray(target_orientation_world, dtype=float).reshape(4))
        rest_q = np.asarray(rest_q, dtype=float)
        lo = list(self.q_lower)
        hi = list(self.q_upper)
        rng_span = list(self.q_upper - self.q_lower)

        best_q, best_score = None, np.inf        # best regardless of clearance
        for attempt in range(tries):
            if attempt == 0:
                rest = rest_q.copy()
            else:
                jitter = self._rng.normal(0.0, 0.12 * attempt, self.nj)
                rest = np.clip(rest_q + jitter, self.q_lower, self.q_upper)
            with self._lock:
                # Seed the solver state at the rest pose
                self._set_config_nolock(rest)
                ik_kwargs = dict(
                    lowerLimits=lo, upperLimits=hi,
                    jointRanges=rng_span, restPoses=list(rest),
                    maxNumIterations=400, residualThreshold=1e-5,
                    physicsClientId=self.cid)
                if target_orn is None:
                    sol = self.p.calculateInverseKinematics(
                        self.arm, self.ee_index, target, **ik_kwargs)
                else:
                    sol = self.p.calculateInverseKinematics(
                        self.arm, self.ee_index, target,
                        targetOrientation=list(target_orn), **ik_kwargs)
            q = np.array([sol[k] for k in self._ik_map])
            q = np.clip(q, self.q_lower, self.q_upper)
            fk_pos, fk_orn = self.fk_world_pose(q)
            pos_err = float(np.linalg.norm(fk_pos - np.asarray(target_world)))
            orn_err = (0.0 if target_orn is None else
                       self._quat_angle(fk_orn, target_orn))
            # Position remains the dominant term; orientation contributes a
            # modest metric-equivalent penalty only when requested.
            score = pos_err + 0.15 * orn_err
            if score < best_score:
                best_q, best_score = q, score
            if pos_err < pos_tol and orn_err < orn_tol:
                if collision_margin is None:
                    return q
                if not self.in_collision(q, margin=collision_margin):
                    return q
        if best_q is None:
            return None
        fk_pos, fk_orn = self.fk_world_pose(best_q)
        pos_err = float(np.linalg.norm(fk_pos - np.asarray(target_world)))
        orn_err = (0.0 if target_orn is None else
                   self._quat_angle(fk_orn, target_orn))
        # Top-down grasp planning is intentionally strict: if a collision
        # margin was requested, reaching the pose with a colliding candidate is
        # not an acceptable fallback.  Preserve the historical position-only
        # fallback behavior for older planning paths.
        if target_orn is not None and collision_margin is not None:
            return None
        return best_q if (pos_err < pos_tol and orn_err < orn_tol) else None

    # ------------------------------------------------------------------

    def disconnect(self):
        with self._lock:
            try:
                self.p.disconnect(physicsClientId=self.cid)
            except Exception:
                pass
