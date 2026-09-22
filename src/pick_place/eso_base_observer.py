"""Extended State Observer (ESO) for the Husky mobile base.

Replaces the NDOB with the same ESO framework used for the arm,
adapted to the base velocity dynamics.

Plant model (first-order lag with unknown disturbance):

    dv/dt = -v/tau + u/tau + d(t)

where:
    v   = actual base velocity (measured via getBaseVelocity)
    u   = commanded velocity from ADMM
    tau = nominal lag time constant
    d   = total unknown disturbance (velocity-level)

Extended state (augment d as a state x2):

    x1_dot = -x1/tau + u/tau + x2      (velocity)
    x2_dot = h(t)                       (disturbance rate, unknown)
    y      = x1                         (measured)

Observer (2nd-order, one plant state + one disturbance state):

    z1_dot = -z1/tau + u/tau + z2 + beta1 * (y - z1)
    z2_dot = beta2 * (y - z1)

where z1 -> v (estimated velocity), z2 -> d (estimated disturbance).

Bandwidth parameterisation (Gao, 2006) — characteristic polynomial (s+wo)^2:

    beta1 = 2 * omega_o
    beta2 = omega_o^2

Two independent channels run in parallel: linear velocity v and
angular velocity omega.

Compared to the NDOB it replaces:
    - Same plant model assumption (first-order lag)
    - Same disturbance correction formula for ADMM
    - Model-free in the disturbance channel (like the arm ESO)
    - Single tuning knob omega_o (vs separate gain l in NDOB)
    - Reads actual velocity directly from PyBullet physics engine

References
----------
Han, J. (1995). "A class of extended state observers for uncertain
systems." Control and Decision, 10(1), 85-88.

Gao, Z. (2006). "Scaling and bandwidth-parameterization based
controller tuning." ACC, pp. 4989-4996.
"""

import math
import threading
import numpy as np


class ESOBaseObserver:
    """2nd-order ESO for base velocity disturbance estimation.

    Parameters
    ----------
    sim : module
        ``pybullet`` module handle.
    husky_id : int
        PyBullet body ID for the Husky platform.
    tau_v : float
        Nominal linear velocity time constant (s).
    tau_omega : float
        Nominal angular velocity time constant (s).
    omega_o_v : float
        Observer bandwidth for linear channel (rad/s).
    omega_o_omega : float
        Observer bandwidth for angular channel (rad/s).
    warmup_steps : int
        Steps before observer output is used for correction.
    sim_lock : threading.Lock or None
        Lock for thread-safe PyBullet access.
    """

    def __init__(self, sim, husky_id,
                 tau_v=0.3, tau_omega=0.3,
                 omega_o_v=20.0, omega_o_omega=20.0,
                 warmup_steps=50,
                 sim_lock=None):
        self.p        = sim
        self.husky_id = husky_id
        self._lock    = sim_lock or threading.Lock()

        self.tau_v     = max(tau_v,     1e-6)
        self.tau_omega = max(tau_omega, 1e-6)

        # Gains from bandwidth parameterisation: (s + omega_o)^2
        self.beta1_v     = 2.0 * omega_o_v
        self.beta2_v     = omega_o_v ** 2
        self.beta1_omega = 2.0 * omega_o_omega
        self.beta2_omega = omega_o_omega ** 2

        self.warmup_steps = warmup_steps

        # Observer states
        self._z1_v     = 0.0   # estimated linear velocity
        self._z2_v     = 0.0   # estimated linear disturbance
        self._z1_omega = 0.0   # estimated angular velocity
        self._z2_omega = 0.0   # estimated angular disturbance

        self._initialized  = False
        self._step_count   = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_base_velocity(self):
        """Read Husky's actual body-frame velocity from PyBullet physics."""
        with self._lock:
            lin_vel, ang_vel = self.p.getBaseVelocity(self.husky_id)
            _, orn = self.p.getBasePositionAndOrientation(self.husky_id)
            yaw = self.p.getEulerFromQuaternion(orn)[2]
        cy, sy = math.cos(yaw), math.sin(yaw)
        # Project world-frame velocity onto body forward axis
        v_actual     = lin_vel[0] * cy + lin_vel[1] * sy
        omega_actual = ang_vel[2]
        return float(v_actual), float(omega_actual)

    # ------------------------------------------------------------------
    # Public API (mirrors NDBOBaseObserver interface)
    # ------------------------------------------------------------------

    def update(self, v_cmd, omega_cmd, dt):
        """Run one ESO update step.

        Parameters
        ----------
        v_cmd : float
            Commanded linear velocity from ADMM (m/s).
        omega_cmd : float
            Commanded angular velocity from ADMM (rad/s).
        dt : float
            Timestep (s).

        Returns
        -------
        d_hat_v : float
            Estimated linear velocity disturbance (z2_v).
        d_hat_omega : float
            Estimated angular velocity disturbance (z2_omega).
        """
        v_meas, omega_meas = self._get_base_velocity()

        if not self._initialized:
            self._z1_v     = v_meas
            self._z2_v     = 0.0
            self._z1_omega = omega_meas
            self._z2_omega = 0.0
            self._initialized = True
            return 0.0, 0.0

        self._step_count += 1

        # --- Linear velocity channel ---
        e_v      = v_meas - self._z1_v
        z1_dot_v = (-self._z1_v / self.tau_v
                    + v_cmd / self.tau_v
                    + self._z2_v
                    + self.beta1_v * e_v)
        z2_dot_v = self.beta2_v * e_v
        self._z1_v += z1_dot_v * dt
        self._z2_v += z2_dot_v * dt

        # --- Angular velocity channel ---
        e_omega      = omega_meas - self._z1_omega
        z1_dot_omega = (-self._z1_omega / self.tau_omega
                        + omega_cmd / self.tau_omega
                        + self._z2_omega
                        + self.beta1_omega * e_omega)
        z2_dot_omega = self.beta2_omega * e_omega
        self._z1_omega += z1_dot_omega * dt
        self._z2_omega += z2_dot_omega * dt

        return self._z2_v, self._z2_omega

    def get_velocity_correction(self, v_cmd, omega_cmd, dt):
        """Return lag-corrected velocity predictions for ADMM base subproblem.

        The estimated disturbance z2 shifts the steady-state velocity:
            v_actual_ss ≈ v_cmd + tau * z2

        This corrected value is used instead of v_cmd in the ADMM
        base subproblem predicted EE position x_b, making the consensus
        dynamics-aware.

        Returns
        -------
        v_corrected : float
        omega_corrected : float
        """
        if self._step_count < self.warmup_steps:
            return v_cmd, omega_cmd
        v_corrected     = v_cmd     + self._z2_v     * self.tau_v
        omega_corrected = omega_cmd + self._z2_omega * self.tau_omega
        return float(v_corrected), float(omega_corrected)

    def get_estimated_disturbance(self):
        """Return (d_hat_v, d_hat_omega) — estimated disturbances."""
        return self._z2_v, self._z2_omega

    def get_estimated_velocity(self):
        """Return (z1_v, z1_omega) — ESO estimated velocities."""
        return self._z1_v, self._z1_omega

    def get_diagnostics(self):
        """Return diagnostic dict for logging/HUD."""
        return {
            "d_hat_v":     self._z2_v,
            "d_hat_omega": self._z2_omega,
            "d_hat_mag":   abs(self._z2_v) + abs(self._z2_omega),
            "z1_v":        self._z1_v,
            "z1_omega":    self._z1_omega,
        }

    def reset(self):
        """Reset observer state."""
        self._z1_v     = 0.0
        self._z2_v     = 0.0
        self._z1_omega = 0.0
        self._z2_omega = 0.0
        self._initialized = False
        self._step_count  = 0
