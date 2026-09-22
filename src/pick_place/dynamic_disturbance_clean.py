"""Dynamic augmentation layer for PyBullet simulation.

This module adds the effects PyBullet does NOT natively model. All rigid-body
interactions that PyBullet handles via its articulated-body solver (verified
against analytical Lagrangian — see verification suite) are NOT touched here,
to avoid double-counting.

What PyBullet HANDLES NATIVELY (verified, do NOT add here):
  • Linear inertial force on arm from base accel        (Interaction 1)
  • Angular inertial force on arm from base ang accel   (Interaction 2)
  • Centrifugal force on arm                            (Interaction 3)
  • Coriolis force on arm                               (Interaction 4)
  • Link-body gyroscopic on arm                         (Interaction 5)
  • Centripetal (v·ω) on arm                            (Interaction 6)
  • Reaction force/torque on base from arm motion       (Interactions 7-9)
  • CoM-shift and variable yaw inertia                  (Interactions 10-11)
  • Link-body gyroscopic on base                        (Interaction 12)
  • Coriolis/RNEA reactions on base                     (Interactions 14-15)
  • Gravity loading (any tilt)                          (Interaction 16)

What THIS module adds (PyBullet does NOT model):
  • Actuator first-order lag on the base                (real wheel inertia)
  • Wheel-ground friction variation                     (terrain effects)
  • Payload mass attachment/release as a dynamic event  (via changeDynamics)
  • Motor-rotor gyroscopic torque on the base           (Interaction 13)

The MotorRotorAugmentation class is also re-exported here for convenience.
"""

import math
import numpy as np
import threading


# ============================================================
# Motor-rotor gyroscopic augmentation (Interaction 13)
# Standalone class — usable independently of BaseDynamicsAugmentation.
# ============================================================

class MotorRotorAugmentation:
    """Implements Interaction 13: motor-rotor gyroscopic torque on base.

    Motor rotors store angular momentum L_rotor,i = I_r,i · n_i · q̇_i · ẑ_i
    where I_r,i is rotor inertia, n_i is gear ratio, ẑ_i is the rotor spin
    axis (joint axis in body frame), and q̇_i is the joint velocity.

    Under base angular velocity ω_b, this momentum precesses:
        τ_rotor_gyro = ω_b × Σ_i L_rotor,i
    This torque acts on the BASE and is applied as an external torque
    each simulation step.

    Parameters
    ----------
    sim : module
        ``pybullet`` module handle.
    body_id : int
        PyBullet body ID of the arm (or of the rigidly attached base+arm).
    joint_indices : list[int]
        Joint indices whose rotors are considered.
    rotor_inertias : list[float]
        I_r,i per joint (kg·m²). Typical Kuka iiwa: ~1e-5.
    gear_ratios : list[float]
        n_i per joint. Typical harmonic drive: 100.
    joint_axes_body : list[list[float]]
        ẑ_i for each joint, expressed in the body frame at zero base rotation.
        For revolute joints about z: [0, 0, 1].
    target_body_id : int or None
        If set, the torque is applied to this body (e.g., husky_id when arm
        is attached to husky via a constraint). Defaults to body_id.
    target_link_idx : int
        Link index on target_body_id where torque is applied. -1 = base.
    sim_lock : threading.Lock or None
        Lock for thread-safe PyBullet access.
    """

    def __init__(self, sim, body_id, joint_indices,
                 rotor_inertias, gear_ratios, joint_axes_body,
                 target_body_id=None, target_link_idx=-1,
                 sim_lock=None):
        self.p = sim
        self.body_id = body_id
        self.target_body_id = target_body_id if target_body_id is not None \
            else body_id
        self.target_link_idx = target_link_idx
        self.joint_indices = list(joint_indices)
        self.I_r = np.array(rotor_inertias, dtype=float)
        self.n  = np.array(gear_ratios, dtype=float)
        self.axes = [np.array(a, dtype=float) / np.linalg.norm(a)
                     for a in joint_axes_body]
        self._lock = sim_lock or threading.Lock()
        self._last_tau_gyro_body = np.zeros(3)

    def _get_base_angular_velocity_body(self):
        """Return base angular velocity expressed in BODY frame."""
        with self._lock:
            _, ang_world = self.p.getBaseVelocity(self.target_body_id)
            _, orn = self.p.getBasePositionAndOrientation(self.target_body_id)
            R_flat = self.p.getMatrixFromQuaternion(orn)
        R_b2w = np.array(R_flat).reshape(3, 3)
        return R_b2w.T @ np.array(ang_world), R_b2w

    def step(self):
        """Compute and apply rotor-gyroscopic torque. Call each timestep.

        Returns
        -------
        tau_body : np.ndarray(3)
            The torque applied (body frame), useful for logging.
        """
        omega_b_body, R_b2w = self._get_base_angular_velocity_body()

        # Sum rotor angular momenta in body frame
        L_body = np.zeros(3)
        with self._lock:
            for i, jidx in enumerate(self.joint_indices):
                js = self.p.getJointState(self.body_id, jidx)
                q_dot = js[1]
                L_body += self.I_r[i] * self.n[i] * q_dot * self.axes[i]

        tau_gyro_body = np.cross(omega_b_body, L_body)
        tau_gyro_world = R_b2w @ tau_gyro_body

        # Apply in world frame
        with self._lock:
            self.p.applyExternalTorque(
                self.target_body_id, self.target_link_idx,
                tau_gyro_world.tolist(),
                self.p.WORLD_FRAME)

        self._last_tau_gyro_body = tau_gyro_body
        return tau_gyro_body

    def get_last_torque_body(self):
        return self._last_tau_gyro_body.copy()


# ============================================================
# Base dynamics augmentation
# ============================================================

class BaseDynamicsAugmentation:
    """Augments PyBullet base with effects PyBullet does NOT model.

    Specifically:
      1. First-order actuator lag on (v_cmd, omega_cmd) — real wheels have
         non-zero time constant; PyBullet's VELOCITY_CONTROL is near-instant.
      2. Acceleration limiting — bounds peak (de)acceleration.
      3. Optional wheel-friction variation — vary lateralFriction over time
         to simulate terrain changes / slip.
      4. Optional payload attach/detach via changeDynamics — proper mass
         change handled by PyBullet's solver (no fake force needed).

    The lagged (v_actual, omega_actual) outputs of apply_base_dynamics()
    are what should be passed to the actual wheel setJointMotorControl
    calls — NOT v_cmd directly.

    Parameters
    ----------
    sim : pybullet module
    husky_id, kuka_id : int
        Body IDs.
    base_time_constant : float
        First-order lag tau (s). 0 disables lag.
    base_acc_limit, base_ang_acc_limit : float
        Max linear/angular acceleration of the wrapper output.
    wheel_indices : list[int] or None
        Wheel link indices on husky_id (for friction variation).
    ee_index : int
        EE link index on kuka_id (for payload attachment).
    sim_lock : threading.Lock or None
    """

    def __init__(self, sim, husky_id, kuka_id, ee_index,
                 base_time_constant=0.15,
                 base_acc_limit=2.0, base_ang_acc_limit=3.0,
                 wheel_indices=None,
                 nominal_wheel_friction=1.0,
                 sim_lock=None):
        self.p = sim
        self.husky_id = husky_id
        self.kuka_id = kuka_id
        self.ee_index = ee_index
        self._lock = sim_lock or threading.Lock()

        self.base_tau = base_time_constant
        self.base_acc_limit = base_acc_limit
        self.base_ang_acc_limit = base_ang_acc_limit

        self.wheel_indices = wheel_indices or []
        self.nominal_wheel_friction = nominal_wheel_friction
        self._current_friction_scale = 1.0

        # State
        self._v_actual = 0.0
        self._omega_actual = 0.0
        self._is_carrying = False
        self._payload_mass_attached = 0.0

        # Diagnostics
        self._last_v_cmd = 0.0
        self._last_omega_cmd = 0.0

    # ------------- Base velocity lag -------------
    def apply_base_dynamics(self, v_cmd, omega_cmd, dt):
        """First-order lag + acceleration limit on commanded velocity.

        Returns (v_actual, omega_actual) to use for wheel motor commands.
        Caller is responsible for sending these to the wheels.
        """
        self._last_v_cmd = v_cmd
        self._last_omega_cmd = omega_cmd

        if self.base_tau > 0:
            alpha = dt / (self.base_tau + dt)
        else:
            alpha = 1.0

        v_target = alpha * v_cmd + (1.0 - alpha) * self._v_actual
        omega_target = alpha * omega_cmd + (1.0 - alpha) * self._omega_actual

        dv = v_target - self._v_actual
        dv = float(np.clip(dv, -self.base_acc_limit * dt,
                            self.base_acc_limit * dt))
        self._v_actual += dv

        domega = omega_target - self._omega_actual
        domega = float(np.clip(domega,
                                -self.base_ang_acc_limit * dt,
                                self.base_ang_acc_limit * dt))
        self._omega_actual += domega

        return self._v_actual, self._omega_actual

    # ------------- Friction variation -------------
    def set_wheel_friction_scale(self, scale):
        """Multiply nominal friction by `scale` on all configured wheels.

        Use to simulate surface changes (e.g. dry-to-wet, gravel patch).
        Updates lateral, spinning, and rolling friction.
        """
        if scale == self._current_friction_scale:
            return
        with self._lock:
            for w in self.wheel_indices:
                self.p.changeDynamics(
                    self.husky_id, w,
                    lateralFriction=self.nominal_wheel_friction * scale,
                    spinningFriction=0.01 * scale,
                    rollingFriction=0.01 * scale)
        self._current_friction_scale = scale

    # ------------- Payload events -------------
    def attach_payload(self, mass_kg):
        """Increase the EE link's mass to simulate a grasped payload.

        Mass change is permanent until detach_payload(). PyBullet then
        natively handles inertia, gravity, and reaction forces — no fake
        forces needed.
        """
        with self._lock:
            cur = self.p.getDynamicsInfo(self.kuka_id, self.ee_index)
            cur_mass = cur[0]
            new_mass = cur_mass + mass_kg
            self.p.changeDynamics(self.kuka_id, self.ee_index, mass=new_mass)
        self._is_carrying = True
        self._payload_mass_attached = mass_kg

    def detach_payload(self):
        """Reverse attach_payload — restore EE link mass."""
        if not self._is_carrying:
            return
        with self._lock:
            cur = self.p.getDynamicsInfo(self.kuka_id, self.ee_index)
            cur_mass = cur[0]
            new_mass = cur_mass - self._payload_mass_attached
            self.p.changeDynamics(self.kuka_id, self.ee_index, mass=new_mass)
        self._is_carrying = False
        self._payload_mass_attached = 0.0

    # ------------- Diagnostics -------------
    def get_diagnostics(self):
        return {
            "v_cmd": self._last_v_cmd,
            "v_actual": self._v_actual,
            "v_lag": self._last_v_cmd - self._v_actual,
            "omega_cmd": self._last_omega_cmd,
            "omega_actual": self._omega_actual,
            "friction_scale": self._current_friction_scale,
            "is_carrying": self._is_carrying,
            "payload_mass": self._payload_mass_attached,
        }


# ============================================================
# Backwards-compat alias
# ============================================================
# Old code might import `DynamicDisturbance` from this module. The new
# semantics differ (no fake forces) so we deliberately give it a slightly
# different name to force callers to read the docstring above. If you
# need a drop-in alias, uncomment:
# DynamicDisturbance = BaseDynamicsAugmentation
