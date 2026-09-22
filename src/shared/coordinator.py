"""Distributed ADMM coordinator with MCU-style thread architecture.

Mimics a real embedded system with **three microcontrollers**:

    MCU-Base  (Thread-Base)  -- autonomous Husky mobile-base controller
    MCU-Arm   (Thread-Arm)   -- autonomous Kuka 7-DOF arm controller
    Thread-Coordinator       -- CLF-budget projection, trajectory generation,
                                HUD / trail rendering  (runs in main script)

Inter-thread communication
--------------------------
Each MCU has an **inbox** (coordinator -> MCU) and an **outbox**
(MCU -> coordinator).  These ``Mailbox`` objects emulate a
message-passing bus (CAN / SPI / serial) between microcontrollers::

    +----------------+      inbox       +----------+
    |                | --------------->  | MCU-Base |
    |  Coordinator   |                   |          |
    |                | <---------------  |          |
    +----------------+      outbox      +----------+
           |  ^
    inbox  |  | outbox
           v  |
    +----------+
    | MCU-Arm  |
    +----------+

Within each ADMM iteration the coordinator dispatches sub-problems
to **both** MCUs simultaneously.  They solve in parallel and send
results back.  The arm uses the *previous* iteration's base-induced
EE velocity for compensation -- a minor one-iteration lag that
enables true parallel execution without sacrificing convergence.
Symmetrically, the base reads the arm's EE/mount telemetry
(``ee_pos``, ``base_pos``, ``R_base``) from the previous iteration
over the ``arm_to_base`` CAN-bus channel rather than querying it
synchronously -- the same one-iteration lag, applied in the other
direction.

After CLF-ADMM convergence, the coordinator sends "apply" commands to
both MCUs; each MCU drives its own actuators independently.
"""

import threading
import time
import math
import numpy as np

try:
    from .task_space_6d import (
        pose_error_6d_current_minus_desired, task_scale_matrix,
        cap_spatial_twist)
except ImportError:
    from task_space_6d import (
        pose_error_6d_current_minus_desired, task_scale_matrix,
        cap_spatial_twist)

from three_layer_controller import (
    ResidualTaskDecompositionLayer,
    TaskSpaceCapabilityAllocationLayer,
    ShiftedCLFADMMCoordinationLayer,
)
from centralized_clf_qp import CentralizedCLFQPSolver


# ====================================================================
# Mailbox -- thread-safe message channel (mimics inter-MCU comms)
# ====================================================================

class Mailbox:
    """Single-slot, thread-safe message channel.

    Models a point-to-point communication link between two MCUs.
    The sender posts a message (dict); the receiver blocks until
    a message arrives.  Follows a strict request/response protocol
    so a single slot is sufficient.
    """

    def __init__(self):
        self._data  = None
        self._event = threading.Event()
        self._lock  = threading.Lock()

    def send(self, data):
        """Post a message (non-blocking)."""
        with self._lock:
            self._data = data
        self._event.set()

    def recv(self, timeout=None):
        """Block until a message arrives.  Returns *None* on timeout."""
        if not self._event.wait(timeout=timeout):
            return None
        self._event.clear()
        with self._lock:
            data = self._data
            self._data = None
        return data

    def clear(self):
        """Discard any pending message."""
        self._event.clear()
        with self._lock:
            self._data = None


# ====================================================================
# LatestValueChannel -- non-blocking CAN-bus-style broadcast channel
# ====================================================================

class LatestValueChannel:
    """Non-blocking, latest-value-wins broadcast channel.

    Models a CAN bus mailbox where the sender publishes periodic
    updates and the receiver always reads the most recent frame.
    Never blocks -- ``read()`` returns the latest value immediately,
    or *None* if nothing has been published yet.
    """

    def __init__(self):
        self._data = None
        self._lock = threading.Lock()

    def publish(self, data):
        """Overwrite the stored value (non-blocking)."""
        with self._lock:
            self._data = data

    def read(self):
        """Return the latest published value, or *None*."""
        with self._lock:
            return self._data


# ====================================================================
# MCU-Base thread
# ====================================================================

class BaseMCU(threading.Thread):
    """Autonomous Husky mobile-base controller thread.

    Mimics a dedicated microcontroller that:
    1. Waits for an ADMM sub-problem from the coordinator.
    2. Solves the base optimisation (QP proximal operator).
    3. Broadcasts velocity state on the Base->Arm CAN channel.
    4. Sends the ADMM result back to the coordinator.
    5. Waits for an "apply" command, then drives the motors.

    Supported commands (``msg["cmd"]``):
        ``"solve_clf_admm"`` -- run the base CLF-ADMM local QP
        ``"apply"``     -- run ``BaseSubsystem.apply()``
        ``"get_state"`` -- return base XY + yaw
    """

    def __init__(self, base_sub, inbox, outbox, stop_event,
                 base_to_arm=None, arm_from_arm=None, max_peer_age=0.25):
        super().__init__(name="MCU-Base", daemon=True)
        self.sub         = base_sub
        self.inbox       = inbox          # Coordinator -> this MCU
        self.outbox      = outbox         # this MCU -> Coordinator
        # NOTE: must not be named ``_stop`` — that would shadow
        # threading.Thread._stop() and crash Thread.join() on
        # timeout (Python >= 3.12).
        self._stop_evt    = stop_event
        self.base_to_arm  = base_to_arm    # CAN bus broadcast to arm MCU
        self.arm_from_arm = arm_from_arm   # CAN bus telemetry from arm MCU
        self.max_peer_age = float(max_peer_age)
        self._last_v_cmd = 0.0
        self._last_omega_cmd = 0.0
        self._heartbeat_period = 0.01

    @staticmethod
    def _Rz(yaw):
        c, s = math.cos(float(yaw)), math.sin(float(yaw))
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _publish_base_state(self, v_cmd=0.0, omega_cmd=0.0):
        if self.base_to_arm is None:
            return
        pos, yaw = self.sub.get_pose()
        self.base_to_arm.publish({
            "position": pos.copy(), "yaw": float(yaw),
            "v_cmd": float(v_cmd), "omega_cmd": float(omega_cmd),
            "timestamp": time.monotonic(),
        })

    def _read_ee_telemetry(self):
        """Read arm-local kinematics and reconstruct world geometry locally."""
        if self.arm_from_arm is None:
            return None
        data = self.arm_from_arm.read()
        if data is None:
            return None
        age = time.monotonic() - float(data.get("timestamp", 0.0))
        if age > self.max_peer_age:
            return {"error": f"stale arm telemetry ({age:.3f} s)"}
        base_pos, yaw = self.sub.get_pose()
        R_base = self._Rz(yaw)
        r_ee_base = np.asarray(data["r_ee_base"], dtype=float)
        ee_pos = base_pos + R_base @ r_ee_base
        return {"ee_pos": ee_pos, "base_pos": base_pos, "R_base": R_base,
                "age": age}

    def run(self):
        # Break the startup dependency cycle: the arm needs the base frame
        # before it can publish r_ee_base, while the base needs r_ee_base
        # before its first local solve.  Publish the base frame immediately.
        self._publish_base_state(self._last_v_cmd, self._last_omega_cmd)

        while not self._stop_evt.is_set():
            msg = self.inbox.recv(timeout=self._heartbeat_period)
            if msg is None:
                # State exchange is independent of ADMM requests.
                self._publish_base_state(
                    self._last_v_cmd, self._last_omega_cmd)
                continue

            cmd = msg["cmd"]

            if cmd == "solve_clf_admm":
                # During startup or a long planning pause, wait briefly for a
                # fresh arm heartbeat instead of terminating the control loop.
                deadline = time.monotonic() + max(0.5, 2.0 * self.max_peer_age)
                telemetry = self._read_ee_telemetry()
                while (not self._stop_evt.is_set() and
                       (telemetry is None or "error" in telemetry) and
                       time.monotonic() < deadline):
                    self._publish_base_state(
                        self._last_v_cmd, self._last_omega_cmd)
                    time.sleep(0.002)
                    telemetry = self._read_ee_telemetry()

                if telemetry is None:
                    self.outbox.send({"error": "missing arm telemetry after startup wait"})
                    continue
                if "error" in telemetry:
                    self.outbox.send({
                        "error": "Base peer exchange failed: " + telemetry["error"]
                    })
                    continue

                v_cmd, omega_cmd, s_b, u_nom = self.sub.solve_clf_admm(
                    msg["a_clf"], msg["z_clf"], msg["lambda_clf"], msg["rho"],
                    msg["pos_desired"], msg["vel_desired"], msg["dt"],
                    ee_pos=telemetry["ee_pos"],
                    base_pos=telemetry["base_pos"],
                    R_base=telemetry["R_base"],
                    task_share=msg.get("task_share"),
                    path_v_ref=msg.get("path_v_ref", 0.0),
                    path_omega_ref=msg.get("path_omega_ref", 0.0),
                    role_weight=msg.get("role_weight", 1.0),
                    command_lower=msg.get("command_lower"),
                    command_upper=msg.get("command_upper"))
                _, yaw = self.sub.get_state()
                self._last_v_cmd = float(v_cmd)
                self._last_omega_cmd = float(omega_cmd)
                self._publish_base_state(
                    self._last_v_cmd, self._last_omega_cmd)
                self.outbox.send({
                    "v_cmd": v_cmd, "omega_cmd": omega_cmd,
                    "s_clf": s_b, "u_nom": np.asarray(u_nom).copy(),
                    "yaw": yaw,
                    "ee_pos": self.sub._last_ee_pos.copy(),
                    "J_base": self.sub._last_J_ee.copy(),
                    "qp_info": dict(self.sub._last_qp_info),
                    "qp_model": {k: (v.copy() if isinstance(v, np.ndarray) else v)
                                 for k, v in self.sub._last_qp_model.items()},
                    "local_diag": dict(self.sub._last_local_diag),
                })

            elif cmd == "apply":
                self._last_v_cmd = float(msg["v_cmd"])
                self._last_omega_cmd = float(msg["omega_cmd"])
                self.sub.apply(self._last_v_cmd, self._last_omega_cmd, msg["yaw"],
                               dt=msg.get("dt"))
                self._publish_base_state(self._last_v_cmd, self._last_omega_cmd)
                self.outbox.send({"status": "applied"})

            elif cmd == "get_state":
                xy, yaw = self.sub.get_state()
                self.outbox.send({"xy": xy, "yaw": yaw})


# ====================================================================
# MCU-Arm thread
# ====================================================================

class ArmMCU(threading.Thread):
    """Autonomous Kuka 7-DOF arm controller thread.

    Mimics a dedicated microcontroller that:
    1. Reads the latest base velocity from the CAN bus channel.
    2. Waits for an ADMM sub-problem from the coordinator.
    3. Solves the arm optimisation (pure ADMM proximal + DLS IK).
    4. Sends the result back.
    5. Waits for an "apply" command, then drives the joints.

    Supported commands (``msg["cmd"]``):
        ``"solve_clf_admm"`` -- run the arm CLF-ADMM local QP
        ``"apply"`` -- run ``ArmSubsystem.apply()``
    """

    def __init__(self, arm_sub, inbox, outbox, stop_event,
                 base_from_base=None, arm_to_base=None, max_peer_age=0.25):
        super().__init__(name="MCU-Arm", daemon=True)
        self.sub            = arm_sub
        self.inbox          = inbox          # Coordinator -> this MCU
        self.outbox         = outbox         # this MCU -> Coordinator
        # See BaseMCU: ``_stop`` would shadow Thread._stop().
        self._stop_evt      = stop_event
        self.base_from_base = base_from_base  # CAN bus from base MCU
        self.arm_to_base    = arm_to_base     # CAN bus broadcast to base MCU
        self.max_peer_age   = float(max_peer_age)
        self._heartbeat_period = 0.01

    @staticmethod
    def _Rz(yaw):
        c, s = math.cos(float(yaw)), math.sin(float(yaw))
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _read_base_state(self):
        if self.base_from_base is None:
            return None
        data = self.base_from_base.read()
        if data is None:
            return None
        age = time.monotonic() - float(data.get("timestamp", 0.0))
        if age > self.max_peer_age:
            return None
        return data

    def _read_base_velocity(self):
        """Compute base-induced EE velocity using the peer base-state frame."""
        data = self._read_base_state()
        if data is None:
            return np.zeros(6)
        ee_pos, _, _ = self.sub.get_ee_telemetry()
        base_pos = np.asarray(data["position"], dtype=float)
        yaw = float(data["yaw"])
        R_base = self._Rz(yaw)
        v_cmd = float(data.get("v_cmd", 0.0))
        omega_cmd = float(data.get("omega_cmd", 0.0))
        v_lin = v_cmd * R_base[:, 0]
        r_ee = ee_pos - base_pos
        v_ang = np.cross(np.array([0.0, 0.0, omega_cmd]), r_ee)
        return np.r_[v_lin + v_ang,
                     np.array([0.0, 0.0, omega_cmd])]

    def _publish_arm_kinematics(self):
        """Publish the minimum cross-subsystem quantity: EE lever arm in base frame."""
        if self.arm_to_base is None:
            return
        base = self._read_base_state()
        if base is None:
            return
        ee_pos, _, _ = self.sub.get_ee_telemetry()
        base_pos = np.asarray(base["position"], dtype=float)
        R_base = self._Rz(base["yaw"])
        r_ee_base = R_base.T @ (np.asarray(ee_pos) - base_pos)
        self.arm_to_base.publish({
            "r_ee_base": r_ee_base.copy(),
            "timestamp": time.monotonic(),
        })

    def run(self):
        # Wait for the initial base-frame broadcast, then publish arm
        # kinematics before the first ADMM request arrives.
        startup_deadline = time.monotonic() + 2.0
        while (not self._stop_evt.is_set() and
               self._read_base_state() is None and
               time.monotonic() < startup_deadline):
            time.sleep(0.002)
        self._publish_arm_kinematics()

        while not self._stop_evt.is_set():
            msg = self.inbox.recv(timeout=self._heartbeat_period)
            if msg is None:
                # Keep arm-to-base telemetry fresh even while the coordinator
                # is planning or idle.
                self._publish_arm_kinematics()
                continue

            cmd = msg["cmd"]

            if cmd == "solve_clf_admm":
                dq_cmd, s_a, dq_nom = self.sub.solve_clf_admm(
                    msg["a_clf"], msg["z_clf"], msg["lambda_clf"], msg["rho"],
                    msg["pos_desired"], msg["vel_desired"],
                    msg["dt"], msg["step_count"],
                    task_share=msg.get("task_share"),
                    v_compensation=msg.get("v_compensation"),
                    base_path_ee_velocity=msg.get("base_path_ee_velocity"),
                    shareable_residual_velocity=msg.get(
                        "shareable_residual_velocity"),
                    local_pos_desired=msg.get("local_pos_desired"),
                    local_vel_desired=msg.get("local_vel_desired"),
                    local_quat_desired=msg.get("local_quat_desired"),
                    local_rot_desired=msg.get("local_rot_desired"),
                    local_omega_desired=msg.get("local_omega_desired"),
                    role_weight=msg.get("role_weight", 1.0))
                self._publish_arm_kinematics()
                self.outbox.send({
                    "dq_cmd": dq_cmd.copy(), "s_clf": s_a,
                    "dq_nom": np.asarray(dq_nom).copy(),
                    "v_arm": self.sub._last_v_arm.copy(),
                    "v_task": self.sub._last_v_task.copy(),
                    "v_path_ee": self.sub._last_v_path_ee.copy(),
                    "v_residual": self.sub._last_v_residual.copy(),
                    "v_shareable": self.sub._last_v_shareable.copy(),
                    "v_mandatory": self.sub._last_v_mandatory.copy(),
                    "ee_pos": self.sub._last_ee_pos.copy(),
                    "ee_rot": self.sub._last_ee_rot.copy(),
                    "J_arm": self.sub._last_J.copy(),
                    "orientation_error": self.sub._last_orientation_error.copy(),
                    "tool_axis_error": self.sub._last_tool_axis_error.copy(),
                    "omega_arm_local": self.sub._last_omega_arm_local.copy(),
                    "qp_info": dict(self.sub._last_qp_info),
                    "qp_model": {k: (v.copy() if isinstance(v, np.ndarray) else v)
                                 for k, v in self.sub._last_qp_model.items()},
                })

            elif cmd == "apply":
                self.sub.apply(msg["dq_cmd"], msg["dt"])
                self.outbox.send({"status": "applied"})


# ====================================================================
# Distributed ADMM Coordinator
# ====================================================================

class DistributedADMMCoordinator:
    """Modular CLF-ADMM coordinator using nominal-plus-correction control.

    Local controllers independently generate ``u_b^0`` and ``dq_a^0``.
    The coordinator never assigns Cartesian shares.  It only coordinates the
    scalar CLF contributions of the corrections

        s_b = a_b^T (u_b-u_b^0),
        s_a = a_a^T (dq_a-dq_a^0).

    Adaptive ``alpha`` is a role preference: it changes positive local cost
    weights but does not directly scale ``v_parallel`` or ``v_perp``.

    V10.7 uses a nominal-first practical stability supervisor.  The
    coordinator is bypassed when the tracking error is already inside a small
    practical tube, or when the independent nominal commands make
    phi_nom=dV_nom/dt < 0.  Coordination is activated only outside the tube
    when nominal motion no longer decreases V.
    """

    def __init__(self, arm_nj, to_base, from_base, to_arm, from_arm,
                 rho=10.0, max_iter=5, tol=1e-3,
                 rho_adapt=True, rho_min=1.0, rho_max=100.0,
                 dual_decay=1.0, task_share_arm=0.5,
                 clf_rate=1.0, clf_relaxation=0.0,
                 clf_slack_penalty=200.0,
                 finite_iter_tightening=True, task_allocator=None,
                 coordination_mode="admm", task_kp=5.0,
                 task_kp_orientation=1.5, angular_task_scale=0.25,
                 task_vmax=0.45, task_omegamax=0.55,
                 stability_supervisor=True,
                 supervisor_phi_threshold=0.0,
                 supervisor_recovery_rate=0.10,
                 supervisor_error_tolerance=0.005):
        self._arm_nj = int(arm_nj)
        self.rho = float(rho)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.rho_adapt = bool(rho_adapt)
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.dual_decay = float(dual_decay)
        self.role_alpha = float(np.clip(task_share_arm, 0.0, 1.0))
        # Compatibility aliases for old logging/CLI code.  They no longer
        # represent Cartesian task fractions.
        self.task_share_arm = self.role_alpha
        self.task_share_base = 1.0 - self.role_alpha
        self.role_weight_arm = 0.5
        self.role_weight_base = 0.5
        self.task_allocator = task_allocator
        self.task_kp = float(task_kp)
        self.task_kp_orientation = float(task_kp_orientation)
        self.task_vmax = max(float(task_vmax), 1e-6)
        self.task_omegamax = max(float(task_omegamax), 1e-6)
        self.angular_task_scale = float(angular_task_scale)
        if self.angular_task_scale <= 0.0:
            raise ValueError("angular_task_scale must be > 0")
        self.task_scale = task_scale_matrix(self.angular_task_scale)
        self.coordination_mode = str(coordination_mode).strip().lower()
        if self.coordination_mode not in ("admm", "centralized"):
            raise ValueError("coordination_mode must be 'admm' or 'centralized'")

        # V10.7: nominal-first practical stability supervisor.  There are two
        # reasons to leave the local nominal controllers untouched:
        #   (1) the EE tracking error is already inside the accepted tube, or
        #   (2) outside the tube, the nominal controllers already make V fall.
        # Only outside the tube with phi_nom >= threshold is recovery
        # coordination activated.
        self.stability_supervisor = bool(stability_supervisor)
        self.supervisor_phi_threshold = float(supervisor_phi_threshold)
        self.supervisor_recovery_rate = float(supervisor_recovery_rate)
        self.supervisor_error_tolerance = float(supervisor_error_tolerance)
        if self.supervisor_recovery_rate < 0.0:
            raise ValueError("supervisor_recovery_rate must be >= 0")
        if self.supervisor_error_tolerance < 0.0:
            raise ValueError("supervisor_error_tolerance must be >= 0")
        self.supervisor_active = False
        self.supervisor_activations = 0

        self.centralized_solver = CentralizedCLFQPSolver(
            tol=max(min(self.tol*0.1, 1e-6), 1e-9),
            box_tol=1e-9, box_sweeps=300)
        self.clf_rate = float(clf_rate)
        # Revised study formulation uses the hard CLF without optimized slack.
        self.clf_relaxation = 0.0
        self.clf_slack_penalty = float(clf_slack_penalty)
        self.finite_iter_tightening = False
        self.finite_iter_margin = 0.0
        self.P = np.eye(6)

        self.layer1_residual = ResidualTaskDecompositionLayer(pinv_rcond=1e-6)
        self.layer2_allocation = TaskSpaceCapabilityAllocationLayer(
            allocator=self.task_allocator)
        self.layer3_clf_admm = ShiftedCLFADMMCoordinationLayer(
            P=self.P, clf_rate=self.clf_rate)

        self.z_b = self.z_a = 0.0
        self.lambda_b = self.lambda_a = 0.0
        self.primal_res = self.dual_res = 0.0
        self.iters_used = 0
        self.s_b = self.s_a = 0.0
        self.clf_slack = 0.0
        self._to_base = to_base; self._from_base = from_base
        self._to_arm = to_arm; self._from_arm = from_arm

    @property
    def z(self):
        return np.array([self.z_b, self.z_a], dtype=float)

    @property
    def u_b(self):
        return np.array([self.lambda_b], dtype=float)

    @property
    def u_a(self):
        return np.array([self.lambda_a], dtype=float)

    def reset(self, pos_init=None):
        self.z_b = self.z_a = 0.0
        self.lambda_b = self.lambda_a = 0.0
        self.primal_res = self.dual_res = 0.0
        self.iters_used = 0
        self.s_b = self.s_a = 0.0
        self.clf_slack = 0.0
        self.supervisor_active = False
        self.supervisor_activations = 0
        if self.task_allocator is not None and hasattr(self.task_allocator, "reset"):
            self.task_allocator.reset(self.role_alpha)

    @staticmethod
    def _prox_hard_clf(y_b, y_a, b_clf):
        return ShiftedCLFADMMCoordinationLayer.project_hard_halfspace(
            y_b, y_a, b_clf)

    def _arm_request(self, *, a, z, lam, rho, pos_desired, vel_desired,
                     local_pos_desired, local_vel_desired,
                     local_quat_desired, local_rot_desired,
                     local_omega_desired, dt, step_count,
                     role_weight):
        self._to_arm.send({
            "cmd": "solve_clf_admm",
            "a_clf": np.asarray(a, dtype=float).copy(),
            "z_clf": float(z), "lambda_clf": float(lam), "rho": float(rho),
            "pos_desired": np.asarray(pos_desired, dtype=float).copy(),
            "vel_desired": np.asarray(vel_desired, dtype=float).copy(),
            "local_pos_desired": (None if local_pos_desired is None else
                                  np.asarray(local_pos_desired, dtype=float).copy()),
            "local_vel_desired": (None if local_vel_desired is None else
                                  np.asarray(local_vel_desired, dtype=float).copy()),
            "local_quat_desired": (None if local_quat_desired is None else
                                   np.asarray(local_quat_desired, dtype=float).reshape(4).copy()),
            "local_rot_desired": (None if local_rot_desired is None else
                                  np.asarray(local_rot_desired, dtype=float).reshape(3,3).copy()),
            "local_omega_desired": (None if local_omega_desired is None else
                                    np.asarray(local_omega_desired, dtype=float).reshape(3).copy()),
            "dt": float(dt), "step_count": int(step_count),
            "role_weight": float(role_weight),
        })

    def _base_request(self, *, a, z, lam, rho, pos_desired, vel_desired,
                      dt, base_path_v_ref, base_path_omega_ref, role_weight,
                      command_lower=None, command_upper=None):
        self._to_base.send({
            "cmd": "solve_clf_admm",
            "a_clf": np.asarray(a, dtype=float).copy(),
            "z_clf": float(z), "lambda_clf": float(lam), "rho": float(rho),
            "pos_desired": np.asarray(pos_desired, dtype=float).copy(),
            "vel_desired": np.asarray(vel_desired, dtype=float).copy(),
            "dt": float(dt),
            "path_v_ref": float(base_path_v_ref),
            "path_omega_ref": float(base_path_omega_ref),
            "role_weight": float(role_weight),
            "command_lower": (None if command_lower is None else
                              np.asarray(command_lower, dtype=float).reshape(2).copy()),
            "command_upper": (None if command_upper is None else
                              np.asarray(command_upper, dtype=float).reshape(2).copy()),
        })

    @staticmethod
    def _J6_arm(J, n):
        J = np.asarray(J, dtype=float)
        if J.shape == (6, n):
            return J
        if J.shape == (3, n):
            return np.vstack([J, np.zeros((3, n))])
        raise ValueError(f"unexpected arm Jacobian shape {J.shape}")

    @staticmethod
    def _J6_base(J):
        J = np.asarray(J, dtype=float)
        if J.shape == (6, 2):
            return J
        if J.shape == (3, 2):
            return np.vstack([J, np.zeros((3, 2))])
        raise ValueError(f"unexpected base Jacobian shape {J.shape}")

    def _govern_servo_twist(self, twist_desired, e6):
        """Bound the 6D servo twist before residual/capability calculations.

        V10.8 allowed Kp*orientation_error to exceed 5 rad/s while the arm
        local controller itself was capped near 1 rad/s.  Layer 2 then saw an
        impossible residual and the CLF coordinator drove base/arm corrections
        into saturation.  The reference governor keeps the commanded task
        inside the same physical scale as the local controllers.
        """
        tw = np.asarray(twist_desired, dtype=float).reshape(6).copy()
        e6 = np.asarray(e6, dtype=float).reshape(6)
        tw[:3] -= self.task_kp * e6[:3]
        tw[3:] -= self.task_kp_orientation * e6[3:]

        return cap_spatial_twist(
            tw, self.task_vmax, self.task_omegamax)

    def step(self, pos_desired, vel_desired, dt, step_count,
             rot_desired=None, omega_desired=None,
             base_pos_desired=None,
             base_path_v_ref=0.0, base_path_omega_ref=0.0,
             v_compensation=None,
             arm_local_pos_desired=None,
             arm_local_vel_desired=None,
             arm_local_quat_desired=None,
             arm_local_rot_desired=None,
             arm_local_omega_desired=None,
             base_command_lower=None,
             base_command_upper=None):
        """Run one outer control step of the revised architecture."""
        pos_desired = np.asarray(pos_desired, dtype=float).reshape(3)
        vel_desired = np.asarray(vel_desired, dtype=float).reshape(3)
        rot_desired_in = (None if rot_desired is None else
                          np.asarray(rot_desired, dtype=float).reshape(3,3))
        omega_desired = (np.zeros(3) if omega_desired is None else
                         np.asarray(omega_desired, dtype=float).reshape(3))
        u_path = np.array([base_path_v_ref, base_path_omega_ref], dtype=float)

        # --------------------------------------------------------------
        # 1) Independent local nominal controllers (rho=0, no role split)
        # --------------------------------------------------------------
        self._arm_request(
            a=np.zeros(self._arm_nj), z=0.0, lam=0.0, rho=0.0,
            pos_desired=pos_desired, vel_desired=vel_desired,
            local_pos_desired=arm_local_pos_desired,
            local_vel_desired=arm_local_vel_desired,
            local_quat_desired=arm_local_quat_desired,
            local_rot_desired=arm_local_rot_desired,
            local_omega_desired=arm_local_omega_desired,
            dt=dt, step_count=step_count, role_weight=1.0)
        prime_arm = self._from_arm.recv(timeout=5.0)
        if prime_arm is None or "error" in prime_arm:
            raise RuntimeError("Arm MCU did not provide nominal telemetry" if prime_arm is None
                               else prime_arm["error"])

        self._base_request(
            a=np.zeros(2), z=0.0, lam=0.0, rho=0.0,
            pos_desired=pos_desired, vel_desired=vel_desired, dt=dt,
            base_path_v_ref=base_path_v_ref,
            base_path_omega_ref=base_path_omega_ref, role_weight=1.0,
            command_lower=base_command_lower,
            command_upper=base_command_upper)
        prime_base = self._from_base.recv(timeout=5.0)
        if prime_base is None or "error" in prime_base:
            raise RuntimeError("Base MCU did not provide nominal telemetry" if prime_base is None
                               else prime_base["error"])

        ee_pos = np.asarray(prime_arm["ee_pos"], dtype=float).reshape(3)
        ee_rot = np.asarray(prime_arm.get("ee_rot", np.eye(3)),
                            dtype=float).reshape(3, 3)
        J_arm_actual = self._J6_arm(prime_arm["J_arm"], self._arm_nj)
        J_base_actual = self._J6_base(prime_base["J_base"])
        u0 = np.asarray(prime_base["u_nom"], dtype=float).reshape(2)
        dq0 = np.asarray(prime_arm["dq_nom"], dtype=float).reshape(self._arm_nj)

        rot_desired_eff = (ee_rot.copy() if rot_desired_in is None
                           else rot_desired_in.copy())
        e6 = pose_error_6d_current_minus_desired(
            ee_pos, ee_rot, pos_desired, rot_desired_eff)
        twist_desired = np.r_[vel_desired, omega_desired]
        e = self.task_scale @ e6
        twist_desired_metric = self.task_scale @ twist_desired
        J_arm = self.task_scale @ J_arm_actual
        J_base = self.task_scale @ J_base_actual
        task_gain = np.r_[np.full(3, self.task_kp),
                          np.full(3, self.task_kp_orientation)]

        # --------------------------------------------------------------
        # 2) FULL-6D residual task geometry after both nominal controllers
        # --------------------------------------------------------------
        servo_twist_6d = self._govern_servo_twist(twist_desired, e6)
        v_task = self.task_scale @ servo_twist_6d
        layer1 = self.layer1_residual.compute(
            v_task, J_base, u0, J_arm, dq0)

        # --------------------------------------------------------------
        # 3) Directional remaining capability -> alpha -> role weights
        # --------------------------------------------------------------
        base_model = prime_base.get("qp_model", {})
        arm_model = prime_arm.get("qp_model", {})
        allocation = self.layer2_allocation.allocate(
            J_arm, J_base, layer1, dt,
            alpha_fallback=self.role_alpha,
            base_nominal=u0, arm_nominal=dq0,
            base_lower=base_model.get("lower"),
            base_upper=base_model.get("upper"),
            arm_lower=arm_model.get("lower"),
            arm_upper=arm_model.get("upper"))
        self.role_alpha = float(allocation.alpha_arm)
        self.task_share_arm = self.role_alpha  # deprecated compatibility name
        self.task_share_base = 1.0 - self.role_alpha
        self.role_weight_arm = float(allocation.weight_arm)
        self.role_weight_base = float(allocation.weight_base)
        alloc_diag = allocation.diagnostics

        # Re-prime with the selected scalar weights.  The nominal commands do
        # not change under positive scalar rescaling, but this exposes exactly
        # the weighted local quadratic models used by the backend.
        self._arm_request(
            a=np.zeros(self._arm_nj), z=0.0, lam=0.0, rho=0.0,
            pos_desired=pos_desired, vel_desired=vel_desired,
            local_pos_desired=arm_local_pos_desired,
            local_vel_desired=arm_local_vel_desired,
            local_quat_desired=arm_local_quat_desired,
            local_rot_desired=arm_local_rot_desired,
            local_omega_desired=arm_local_omega_desired,
            dt=dt, step_count=step_count, role_weight=self.role_weight_arm)
        self._base_request(
            a=np.zeros(2), z=0.0, lam=0.0, rho=0.0,
            pos_desired=pos_desired, vel_desired=vel_desired, dt=dt,
            base_path_v_ref=base_path_v_ref,
            base_path_omega_ref=base_path_omega_ref,
            role_weight=self.role_weight_base,
            command_lower=base_command_lower,
            command_upper=base_command_upper)
        prime_arm_w = self._from_arm.recv(timeout=5.0)
        prime_base_w = self._from_base.recv(timeout=5.0)
        if prime_arm_w is None or prime_base_w is None:
            raise RuntimeError("Weighted nominal local solve timeout")
        if "error" in prime_arm_w or "error" in prime_base_w:
            raise RuntimeError(prime_arm_w.get("error", prime_base_w.get("error")))
        prime_arm, prime_base = prime_arm_w, prime_base_w
        u0 = np.asarray(prime_base["u_nom"], dtype=float).reshape(2)
        dq0 = np.asarray(prime_arm["dq_nom"], dtype=float).reshape(self._arm_nj)
        ee_pos = np.asarray(prime_arm["ee_pos"], dtype=float).reshape(3)
        ee_rot = np.asarray(prime_arm.get("ee_rot", ee_rot),
                            dtype=float).reshape(3, 3)
        J_arm_actual = self._J6_arm(prime_arm["J_arm"], self._arm_nj)
        J_base_actual = self._J6_base(prime_base["J_base"])
        e6 = pose_error_6d_current_minus_desired(
            ee_pos, ee_rot, pos_desired, rot_desired_eff)
        e = self.task_scale @ e6
        J_arm = self.task_scale @ J_arm_actual
        J_base = self.task_scale @ J_base_actual

        # Re-evaluate residual and CLF around exact commands exposed to ADMM.
        servo_twist_6d = self._govern_servo_twist(twist_desired, e6)
        v_task = self.task_scale @ servo_twist_6d
        layer1 = self.layer1_residual.compute(v_task, J_base, u0, J_arm, dq0)
        layer3 = self.layer3_clf_admm.compute_from_error(
            e, twist_desired_metric, J_base, J_arm, u0, dq0)
        V = layer3.V
        a_b = layer3.a_base
        a_a = layer3.a_arm
        b_clf_exponential = layer3.b_clf

        # --------------------------------------------------------------
        # 4) Practical error-tube stability supervisor (V10.7)
        # --------------------------------------------------------------
        # phi_nominal is exactly dV/dt under the independent nominal
        # controllers.  The supervisor intervenes only if BOTH conditions
        # hold:
        #   (a) ||e|| is outside the accepted tracking tube, and
        #   (b) nominal motion no longer decreases V.
        #
        # Inside the tube, a small positive dV is tolerated because the state
        # is already in the practical tracking set.  Outside the tube, a
        # negative phi_nominal is also accepted because the local controllers
        # are already recovering without global intervention.
        phi_nominal = float(layer3.phi_nominal)
        error_norm = float(np.linalg.norm(e))
        error_tolerance = float(self.supervisor_error_tolerance)
        inside_error_tube = bool(error_norm <= error_tolerance)
        outside_error_tube = not inside_error_tube
        # P=I in the present study, hence V_tol = 1/2 * e_tol^2.
        V_tolerance = float(0.5 * error_tolerance * error_tolerance)
        V_excess = float(max(0.0, V - V_tolerance))

        supervisor_was_active = bool(self.supervisor_active)
        nominal_acceptable = bool(phi_nominal < self.supervisor_phi_threshold)
        supervisor_requested = (
            (not self.stability_supervisor) or
            (outside_error_tube and not nominal_acceptable)
        )
        self.supervisor_active = bool(supervisor_requested)
        if self.supervisor_active and not supervisor_was_active:
            self.supervisor_activations += 1

        # Effective recovery budget.  In practical supervisor mode the decay
        # requirement is applied only to the Lyapunov energy outside the tube:
        #
        #     dV <= -c_s (V - V_tol).
        #
        # Thus the recovery authority naturally fades as the trajectory
        # approaches the accepted error set.  The --always-enforce-clf
        # ablation preserves the original dV <= -c V formulation.
        recovery_rate = (self.supervisor_recovery_rate
                         if self.stability_supervisor else self.clf_rate)
        recovery_energy = (V_excess if self.stability_supervisor else V)
        recovery_bound_active = float(-recovery_rate * recovery_energy)
        b_clf_recovery = float(recovery_bound_active - phi_nominal)
        # In bypass mode no correction problem exists; zero is logged so
        # s_b=s_a=0 has zero correction-space residual.
        b_clf = float(b_clf_recovery if self.supervisor_active else 0.0)

        # Nominal corrections are zero by definition.
        self.s_b = self.s_a = 0.0
        base_result = prime_base
        arm_result = prime_arm
        r_prim = r_dual = 0.0
        central_diag = {
            "multiplier": 0.0, "feasible": True, "active": False,
            "nominal_violation": (max(0.0, -b_clf) if self.supervisor_active else 0.0), "box_solves": 0,
            "solve_time_ms": 0.0, "kkt_residual": float("nan"),
            "objective": float("nan"), "status": "not-centralized",
        }

        coordination_wall_t0 = time.perf_counter()

        # Fast bypass: either the error is already inside the practical tube,
        # or the nominal controllers already make V decrease.  In both cases
        # the defining nominal-first rule is to apply the local commands
        # exactly, identically for ADMM and the centralized comparison backend.
        if not self.supervisor_active:
            u_b_cmd = u0.copy()
            dq_cmd = dq0.copy()
            self.s_b = self.s_a = 0.0
            self.z_b = self.z_a = 0.0
            self.lambda_b = self.lambda_a = 0.0
            self.primal_res = self.dual_res = 0.0
            self.iters_used = 0
            base_result = dict(prime_base)
            arm_result = dict(prime_arm)
            base_result["v_cmd"], base_result["omega_cmd"] = u_b_cmd.tolist()
            arm_result["dq_cmd"] = dq_cmd.copy()
            central_diag.update({
                "active": False, "feasible": True,
                "nominal_violation": 0.0,
                "status": "supervisor-bypass",
            })

        elif self.coordination_mode == "centralized":
            central = self.centralized_solver.solve(
                prime_base.get("qp_model"), prime_arm.get("qp_model"),
                a_b, a_a, b_clf, u0, dq0)
            u_b_cmd = np.asarray(central.u_base, dtype=float).reshape(2)
            dq_cmd = np.asarray(central.dq_arm, dtype=float).reshape(self._arm_nj)
            self.s_b, self.s_a = self.layer3_clf_admm.correction_contributions(
                layer3, u_b_cmd, u0, dq_cmd, dq0)
            self.z_b, self.z_a = self.s_b, self.s_a
            self.lambda_b = self.lambda_a = 0.0
            self.iters_used = 1
            self.primal_res = self.dual_res = 0.0
            central_diag = {
                "multiplier": float(central.multiplier),
                "feasible": bool(central.feasible),
                "active": bool(central.active),
                "nominal_violation": float(central.nominal_violation),
                "box_solves": int(central.box_solves),
                "solve_time_ms": float(central.solve_time_ms),
                "kkt_residual": float(central.kkt_residual),
                "objective": float(central.objective),
                "status": str(central.status),
            }
            base_result = dict(prime_base); arm_result = dict(prime_arm)
            base_result["v_cmd"], base_result["omega_cmd"] = u_b_cmd.tolist()
            arm_result["dq_cmd"] = dq_cmd.copy()
        else:
            # Start each newly activated supervisory episode from a clean
            # dual state.  This prevents stale ADMM memory from generating a
            # correction spike after a long nominal-only interval.
            if not supervisor_was_active:
                self.z_b, self.z_a, _ = self._prox_hard_clf(0.0, 0.0, b_clf)
                self.lambda_b = self.lambda_a = 0.0

            for k in range(self.max_iter):
                self._base_request(
                    a=a_b, z=self.z_b, lam=self.lambda_b, rho=self.rho,
                    pos_desired=pos_desired, vel_desired=vel_desired, dt=dt,
                    base_path_v_ref=base_path_v_ref,
                    base_path_omega_ref=base_path_omega_ref,
                    role_weight=self.role_weight_base,
                    command_lower=base_command_lower,
                    command_upper=base_command_upper)
                self._arm_request(
                    a=a_a, z=self.z_a, lam=self.lambda_a, rho=self.rho,
                    pos_desired=pos_desired, vel_desired=vel_desired,
                    local_pos_desired=arm_local_pos_desired,
                    local_vel_desired=arm_local_vel_desired,
                    local_quat_desired=arm_local_quat_desired,
                    local_rot_desired=arm_local_rot_desired,
                    local_omega_desired=arm_local_omega_desired,
                    dt=dt, step_count=step_count,
                    role_weight=self.role_weight_arm)
                base_result = self._from_base.recv(timeout=5.0)
                arm_result = self._from_arm.recv(timeout=5.0)
                if base_result is None or arm_result is None:
                    raise RuntimeError("CLF-ADMM local solver timeout")
                if "error" in base_result or "error" in arm_result:
                    raise RuntimeError(base_result.get("error", arm_result.get("error")))

                s_b = float(base_result["s_clf"])
                s_a = float(arm_result["s_clf"])
                z_b_old, z_a_old = self.z_b, self.z_a
                self.z_b, self.z_a, _ = self._prox_hard_clf(
                    s_b+self.lambda_b, s_a+self.lambda_a, b_clf)
                self.lambda_b += s_b-self.z_b
                self.lambda_a += s_a-self.z_a
                r_prim = float(np.hypot(s_b-self.z_b, s_a-self.z_a))
                r_dual = float(self.rho*np.hypot(
                    self.z_b-z_b_old, self.z_a-z_a_old))
                self.s_b, self.s_a = s_b, s_a
                if r_prim <= self.tol and r_dual <= self.tol:
                    break

            self.iters_used = k+1
            self.primal_res = r_prim; self.dual_res = r_dual
            if self.rho_adapt:
                mu, tau = 8.0, 1.5
                old_rho = self.rho
                if r_prim > mu*max(r_dual, 1e-12):
                    self.rho = min(self.rho*tau, self.rho_max)
                elif r_dual > mu*max(r_prim, 1e-12):
                    self.rho = max(self.rho/tau, self.rho_min)
                if self.rho != old_rho:
                    scale = old_rho/self.rho
                    self.lambda_b *= scale; self.lambda_a *= scale

            u_b_cmd = np.array([base_result["v_cmd"],
                                base_result["omega_cmd"]], dtype=float)
            dq_cmd = np.asarray(arm_result["dq_cmd"], dtype=float)

        coordination_wall_time_ms = 1e3*(time.perf_counter()-coordination_wall_t0)
        v_cmd, omega_cmd = float(u_b_cmd[0]), float(u_b_cmd[1])

        dV = self.layer3_clf_admm.derivative(
            layer3, J_base, u_b_cmd, J_arm, dq_cmd, twist_desired_metric)
        exponential_residual = float(dV + self.clf_rate*V)
        supervisor_bound = float(recovery_bound_active if self.supervisor_active else 0.0)
        # The practical supervisor has a set-valued bypass condition.  Inside
        # the accepted tube the condition is satisfied by membership itself,
        # so a positive instantaneous dV does not count as a violation.
        if self.supervisor_active:
            supervisor_residual = float(dV - supervisor_bound)
        elif self.stability_supervisor and inside_error_tube:
            supervisor_residual = 0.0
        else:
            supervisor_residual = float(dV - self.supervisor_phi_threshold)
        decomposed_residual = float((self.s_b+self.s_a)-b_clf)
        # Active-mode correction-space identity:
        #   dV - recovery_bound = (s_b+s_a) - b_clf.
        decomposition_error = float(
            (supervisor_residual-decomposed_residual)
            if self.supervisor_active else decomposed_residual)
        finite_violation = max(0.0, decomposed_residual)
        if not self.supervisor_active:
            finite_cert = bool(inside_error_tube or
                               dV < self.supervisor_phi_threshold + self.tol)
        else:
            finite_cert = (bool(central_diag["feasible"])
                           if self.coordination_mode == "centralized"
                           else bool(r_prim <= self.tol+1e-15 and finite_violation <= self.tol))

        # Apply final subsystem commands.
        self._to_base.send({"cmd": "get_state"})
        state = self._from_base.recv(timeout=2.0)
        yaw = state["yaw"] if state else float(base_result.get("yaw", 0.0))
        self._to_base.send({"cmd": "apply", "v_cmd": v_cmd,
                            "omega_cmd": omega_cmd, "yaw": yaw, "dt": dt})
        self._to_arm.send({"cmd": "apply", "dq_cmd": dq_cmd, "dt": dt})
        self._from_base.recv(timeout=2.0); self._from_arm.recv(timeout=2.0)
        self.lambda_b *= self.dual_decay; self.lambda_a *= self.dual_decay

        base_diag = base_result.get("local_diag", {})
        arm_orientation_error = np.asarray(
            arm_result.get("orientation_error", np.zeros(3)), dtype=float).reshape(3)
        arm_tool_axis_error = np.asarray(
            arm_result.get("tool_axis_error", arm_orientation_error),
            dtype=float).reshape(3)
        arm_omega_local_ref = np.asarray(
            arm_result.get("omega_arm_local", np.zeros(3)), dtype=float).reshape(3)
        info = {
            "control_mode": ("centralized" if self.coordination_mode == "centralized"
                             else ("fixed-admm" if getattr(
                                 self.task_allocator, "allocation_kind", "adaptive") == "fixed"
                                 else "adaptive-admm")),
            "coordination_mode": self.coordination_mode,
            "v_cmd": v_cmd, "omega_cmd": omega_cmd,
            "dq_cmd": dq_cmd.copy(),
            "u_b_norm": float(np.linalg.norm(u_b_cmd)),
            "u_a_norm": float(np.linalg.norm(dq_cmd)),
            "primal_residual": float(self.primal_res),
            "dual_residual": float(self.dual_res),
            "iterations": int(self.iters_used),
            "rho": float(self.rho),
            "coordination_wall_time_ms": float(coordination_wall_time_ms),
            "admm_wall_time_ms": (float(coordination_wall_time_ms)
                                  if self.coordination_mode == "admm" else float("nan")),
            "admm_time_per_iteration_ms": (float(coordination_wall_time_ms/max(self.iters_used,1))
                                           if self.coordination_mode == "admm" else float("nan")),
            "centralized_clf_multiplier": float(central_diag["multiplier"]),
            "centralized_clf_feasible": bool(central_diag["feasible"]),
            "centralized_clf_active": bool(central_diag["active"]),
            "centralized_nominal_violation": float(central_diag["nominal_violation"]),
            "centralized_qp_box_solves": int(central_diag["box_solves"]),
            "centralized_qp_time_ms": float(central_diag["solve_time_ms"]),
            "centralized_qp_kkt": float(central_diag["kkt_residual"]),
            "centralized_qp_objective": float(central_diag["objective"]),
            "centralized_qp_status": str(central_diag["status"]),
            # CLF diagnostics
            "clf_enabled": True,
            "clf_V": float(V), "clf_dV": float(dV),
            "task_dim": 6,
            "angular_task_scale": float(self.angular_task_scale),
            "pose_error_6d": e6.copy(),
            "position_error_norm": float(np.linalg.norm(e6[:3])),
            "orientation_error_rad": float(np.linalg.norm(e6[3:])),
            "orientation_error_deg": math.degrees(float(np.linalg.norm(e6[3:]))),
            "ee_rot": ee_rot.copy(),
            "rot_desired": rot_desired_eff.copy(),
            "twist_desired_6d": twist_desired.copy(),
            "servo_twist_6d": servo_twist_6d.copy(),
            "task_vmax": float(self.task_vmax),
            "task_omegamax": float(self.task_omegamax),
            # Effective practical supervisor condition: being inside the
            # accepted error tube is sufficient; outside it, nominal dV<0 is
            # accepted, and active recovery enforces dV<=-c_s(V-V_tol).
            # The raw old exponential residual is logged separately for
            # ablation/diagnosis.
            "clf_bound": float(supervisor_bound),
            "clf_residual": float(supervisor_residual),
            "clf_hard_residual": float(supervisor_residual),
            "clf_soft_bound": float(supervisor_bound),
            "clf_hard_feasible": bool(supervisor_residual <= self.tol),
            "clf_slack": 0.0, "clf_slack_penalty": float(self.clf_slack_penalty),
            "clf_active": bool(self.supervisor_active),
            "clf_feasible": bool(supervisor_residual <= self.tol),
            "clf_correction": float(np.linalg.norm(np.r_[u_b_cmd-u0, dq_cmd-dq0])),
            "stability_supervisor_enabled": bool(self.stability_supervisor),
            "stability_supervisor_active": bool(self.supervisor_active),
            "stability_supervisor_phi_nominal": float(phi_nominal),
            "stability_supervisor_phi_threshold": float(self.supervisor_phi_threshold),
            "stability_supervisor_recovery_rate": float(recovery_rate),
            "stability_supervisor_recovery_bound": float(supervisor_bound),
            "stability_supervisor_nominal_decreasing": bool(phi_nominal < 0.0),
            "stability_supervisor_nominal_acceptable": bool(nominal_acceptable),
            "stability_supervisor_error_norm": float(error_norm),
            "stability_supervisor_error_tolerance": float(error_tolerance),
            "stability_supervisor_inside_error_tube": bool(inside_error_tube),
            "stability_supervisor_outside_error_tube": bool(outside_error_tube),
            "stability_supervisor_V_tolerance": float(V_tolerance),
            "stability_supervisor_V_excess": float(V_excess),
            "stability_supervisor_bypass_inside_tube": bool(
                self.stability_supervisor and inside_error_tube and not self.supervisor_active),
            "stability_supervisor_bypass_nominal_decrease": bool(
                self.stability_supervisor and outside_error_tube and nominal_acceptable
                and not self.supervisor_active),
            "stability_supervisor_activation_count": int(self.supervisor_activations),
            "clf_exponential_bound": float(-self.clf_rate*V),
            "clf_exponential_residual": float(exponential_residual),
            "clf_exponential_feasible": bool(exponential_residual <= self.tol),
            "clf_s_base": float(self.s_b), "clf_s_arm": float(self.s_a),
            "clf_total_contribution": float(self.s_b+self.s_a),
            "clf_budget": float(b_clf),
            "clf_budget_gap": float(b_clf-(self.s_b+self.s_a)),
            "clf_constraint_violation": max(0.0, decomposed_residual),
            "clf_soft_budget": float(b_clf),
            "clf_soft_budget_gap": float(b_clf-(self.s_b+self.s_a)),
            "clf_soft_constraint_violation": max(0.0, decomposed_residual),
            "admm_finite_iter_margin": 0.0,
            "admm_finite_error_bound": 0.0,
            "admm_actual_finite_violation": float(finite_violation),
            "admm_finite_iter_certified": bool(finite_cert),
            "clf_eta_bound": 0.0, "clf_V_ultimate_bound": 0.0,
            "clf_error_ultimate_bound": 0.0,
            "clf_decomposition_error": float(decomposition_error),
            # Layer 1: residual geometry
            "layer1_v_task": layer1.v_task.copy(),
            "layer1_v_nominal": layer1.v_nominal.copy(),
            "layer1_v_path_ee": layer1.v_nominal.copy(),  # compatibility alias
            "layer1_v_residual": layer1.v_residual.copy(),
            "layer1_v_shareable": layer1.v_shareable.copy(),
            "layer1_v_mandatory": layer1.v_mandatory.copy(),
            "layer1_reconstruction_error": float(layer1.reconstruction_error),
            "layer1_projector_idempotence_error": float(layer1.projector_idempotence_error),
            "layer1_mandatory_base_projection_error": float(layer1.mandatory_base_projection_error),
            # Layer 2: preference, not hard allocation
            "layer2_alpha_arm": float(allocation.alpha_arm),
            "layer2_alpha_base": float(allocation.alpha_base),
            "layer2_weight_arm": float(allocation.weight_arm),
            "layer2_weight_base": float(allocation.weight_base),
            "layer2_gamma_arm": float(allocation.gamma_arm),
            "layer2_gamma_base": float(allocation.gamma_base),
            "layer2_v_arm_ref": np.full(6, np.nan),
            "layer2_v_base_correction_ref": np.full(6, np.nan),
            "layer2_allocation_reconstruction_error": float("nan"),
            # Layer 3
            "layer3_a_base": layer3.a_base.copy(),
            "layer3_a_arm": layer3.a_arm.copy(),
            "layer3_clf_budget": float(b_clf),
            "layer3_recovery_clf_budget": float(b_clf_recovery),
            "layer3_exponential_clf_budget": float(b_clf_exponential),
            "layer3_phi_nominal": float(layer3.phi_nominal),
            "layer3_s_base": float(self.s_b),
            "layer3_s_arm": float(self.s_a),
            "layer3_coupling_residual": float(decomposed_residual),
            "layer3_path_clf_contribution": float(layer3.phi_nominal),
            # Role allocator diagnostics and compatibility aliases
            "alpha_task": float(alloc_diag.get("alpha", allocation.alpha_arm)),
            "alpha_raw": float(alloc_diag.get("alpha_raw", allocation.alpha_arm)),
            "alpha_capacity": float(alloc_diag.get("alpha_capacity", allocation.alpha_arm)),
            "alpha_gamma_arm": float(alloc_diag.get("gamma_arm", np.nan)),
            "alpha_gamma_base": float(alloc_diag.get("gamma_base", np.nan)),
            "alpha_weight_arm": float(alloc_diag.get("weight_arm", allocation.weight_arm)),
            "alpha_weight_base": float(alloc_diag.get("weight_base", allocation.weight_base)),
            "alpha_eta_arm": float(alloc_diag.get("eta_arm", np.nan)),
            "alpha_eta_base": float(alloc_diag.get("eta_base", np.nan)),
            "alpha_eta_base_actuator": float(alloc_diag.get("eta_base_actuator", np.nan)),
            "alpha_eta_base_locomotion": float(alloc_diag.get("eta_base_locomotion", np.nan)),
            "alpha_eta_worst": float(alloc_diag.get("eta_worst", np.nan)),
            "alpha_eta_mandatory": float(alloc_diag.get("eta_mandatory", np.nan)),
            "alpha_eta_shareable_full": float(alloc_diag.get("eta_shareable_full", np.nan)),
            "alpha_residual_norm": float(alloc_diag.get("residual_norm", np.nan)),
            "alpha_shareable_norm": float(alloc_diag.get("shareable_norm", np.nan)),
            "alpha_mandatory_norm": float(alloc_diag.get("mandatory_norm", np.nan)),
            "alpha_qdot_full": np.asarray(alloc_diag.get("qdot_full", np.full(self._arm_nj,np.nan))),
            "alpha_qdot_mandatory": np.asarray(alloc_diag.get("qdot_mandatory", np.full(self._arm_nj,np.nan))),
            "alpha_qdot_shareable_full": np.asarray(alloc_diag.get("qdot_shareable_full", np.full(self._arm_nj,np.nan))),
            "alpha_deadband_hold": bool(alloc_diag.get("held_deadband", False)),
            "alpha_capacity_feasible": bool(alloc_diag.get("capacity_feasible", True)),
            "alpha_base_correction_full": np.asarray(alloc_diag.get("base_correction_full", np.full(2,np.nan))),
            "alpha_base_remaining_margin": np.asarray(alloc_diag.get("base_remaining_margin", np.full(2,np.nan))),
            "alpha_path_outside_bounds": bool(alloc_diag.get("path_outside_bounds", False)),
            "alpha_base_path_weight": float(alloc_diag.get("base_path_weight", np.nan)),
            "alpha_base_locomotion_jmax": float(alloc_diag.get("base_locomotion_jmax", np.nan)),
            "base_path_ee_velocity": (J_base@u0).copy(),
            "base_path_clf_contribution": float(layer3.a_base@u0),
            "residual_shareable_velocity": layer1.v_shareable.copy(),
            "residual_mandatory_arm_velocity": layer1.v_mandatory.copy(),
            "arm_residual_task_velocity": layer1.v_residual.copy(),
            "base_residual_task_velocity": layer1.v_residual.copy(),
            "base_correction_cmd": (u_b_cmd-u0).copy(),
            "base_u_path": np.asarray(base_diag.get("u_path", u_path), dtype=float),
            "base_u_nom": u0.copy(),
            "base_ee_cost": float(base_diag.get("ee_cost", 0.0)),
            "base_path_cost": float(base_diag.get("path_cost", 0.0)),
            "base_reg_cost": float(base_diag.get("reg_cost", 0.0)),
            "base_admm_cost": float(base_diag.get("admm_cost", np.nan)),
            "base_path_cmd_gap": float(base_diag.get("path_cmd_gap", np.nan)),
            "base_nom_cmd_gap": float(base_diag.get("nom_cmd_gap", np.nan)),
            "base_path_nom_gap": float(base_diag.get("path_nom_gap", np.nan)),
            "base_v_lower_active": bool(base_diag.get("v_lower_active", False)),
            "base_v_upper_active": bool(base_diag.get("v_upper_active", False)),
            "base_omega_lower_active": bool(base_diag.get("omega_lower_active", False)),
            "base_omega_upper_active": bool(base_diag.get("omega_upper_active", False)),
            "base_path_v_outside_bounds": bool(base_diag.get("path_v_outside_bounds", False)),
            "base_path_omega_outside_bounds": bool(base_diag.get("path_omega_outside_bounds", False)),
            "base_command_lower": np.asarray(
                base_result.get("qp_model", {}).get("lower", np.full(2, np.nan)),
                dtype=float).reshape(2),
            "base_command_upper": np.asarray(
                base_result.get("qp_model", {}).get("upper", np.full(2, np.nan)),
                dtype=float).reshape(2),
            "base_qp_kkt": float(base_result.get("qp_info",{}).get("kkt_residual",np.nan)),
            "base_qp_iterations": int(base_result.get("qp_info",{}).get("iterations",0)),
            "base_qp_converged": bool(base_result.get("qp_info",{}).get("converged",False)),
            "arm_orientation_error": arm_orientation_error.copy(),
            "arm_orientation_error_deg": math.degrees(float(np.linalg.norm(arm_orientation_error))),
            "arm_tool_axis_error": arm_tool_axis_error.copy(),
            "arm_tool_axis_error_deg": math.degrees(float(np.linalg.norm(arm_tool_axis_error))),
            "arm_omega_local_ref": arm_omega_local_ref.copy(),
            "arm_qp_kkt": float(arm_result.get("qp_info",{}).get("kkt_residual",np.nan)),
            "arm_qp_iterations": int(arm_result.get("qp_info",{}).get("iterations",0)),
            "arm_qp_converged": bool(arm_result.get("qp_info",{}).get("converged",False)),
            "arm_accel_fallback": bool(np.any(arm_result.get("qp_info",{}).get(
                "accel_fallback", np.zeros(0,dtype=bool)))),
            "arm_nominal_local_position": np.asarray(
                getattr(self, "_unused", np.full(3,np.nan))),
        }
        return info

    def get_diagnostics(self):
        return {
            "coordination_mode": self.coordination_mode,
            "z": self.z.copy(), "u_b": self.u_b.copy(), "u_a": self.u_a.copy(),
            "rho": float(self.rho), "primal_res": float(self.primal_res),
            "dual_res": float(self.dual_res),
            "agreement": abs((self.s_b+self.s_a)-(self.z_b+self.z_a)),
            "s_base": float(self.s_b), "s_arm": float(self.s_a),
            "alpha_role": float(self.role_alpha),
            "weight_base": float(self.role_weight_base),
            "weight_arm": float(self.role_weight_arm),
            "clf_slack": 0.0,
            "stability_supervisor_enabled": bool(self.stability_supervisor),
            "stability_supervisor_active": bool(self.supervisor_active),
            "stability_supervisor_phi_threshold": float(self.supervisor_phi_threshold),
            "stability_supervisor_recovery_rate": float(self.supervisor_recovery_rate),
            "stability_supervisor_error_tolerance": float(self.supervisor_error_tolerance),
            "stability_supervisor_activation_count": int(self.supervisor_activations),
            "finite_iter_certified": bool(self.primal_res <= self.tol+1e-15),
        }
