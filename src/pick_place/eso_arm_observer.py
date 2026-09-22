"""Extended State Observer (ESO) for the Kuka arm end-effector.

Estimates the total disturbance acting on the EE using the ESO
framework from Active Disturbance Rejection Control (ADRC).

The EE dynamics are modelled as a second-order system with
unknown total disturbance f:

    x1_dot = x2                     (position -> velocity)
    x2_dot = f + b0 * u             (velocity -> acceleration)

where:
    x1 = p_ee          (EE position, measured)
    x2 = v_ee          (EE velocity, estimated)
    f  = total disturbance (estimated as extended state x3)
    u  = ADMM velocity command (known input)
    b0 = nominal input gain

The ESO extends the state to include f:
    x1_dot = x2
    x2_dot = x3 + b0 * u
    x3_dot = h(t)                   (unknown, but bounded)

The observer:
    z1_dot = z2 + beta1 * (x1 - z1)
    z2_dot = z3 + beta2 * (x1 - z1) + b0 * u
    z3_dot = beta3 * (x1 - z1)

where z1 -> x1, z2 -> x2, z3 -> f (disturbance estimate).

Gains are parameterized by observer bandwidth omega_o (Gao, 2006):
    beta1 = 3 * omega_o
    beta2 = 3 * omega_o^2
    beta3 = omega_o^3

One tuning parameter: omega_o.  Higher = faster tracking, more noise.

The estimated total disturbance z3 lumps together ALL uncertainty:
external forces, model mismatch, kinematic errors, and unmodeled
dynamics --- exactly as intended by the ADRC framework.

Key advantage: completely model-free --- no M(q), C(q,qdot), g(q),
no tau_applied needed.  Works with any actuation mode.

References
----------
Han, J. (1995). "A class of extended state observers for uncertain
systems." Control and Decision, 10(1), 85-88.

Gao, Z. (2006). "Scaling and bandwidth-parameterization based
controller tuning." ACC, pp. 4989-4996.
"""

import numpy as np
import threading


class ESOArmObserver:
    """3rd-order Extended State Observer for EE disturbance estimation.

    Runs three independent ESO channels (x, y, z) in Cartesian space.

    Parameters
    ----------
    sim : module
        ``pybullet`` module handle.
    kuka_id : int
        PyBullet body ID for the Kuka arm.
    ee_index : int
        End-effector link index.
    omega_o : float
        Observer bandwidth (rad/s).  Higher = faster but noisier.
        Typical: 15--50 for 240 Hz control.
    b0 : float
        Nominal input gain.  Relates the ADMM velocity command
        to EE acceleration.  Approximate value is fine --- the ESO
        treats model error as part of the disturbance.
    v_comp_max : float
        Maximum compensation velocity magnitude (m/s).
    warmup_steps : int
        Steps before observer output is used.
    sim_lock : threading.Lock or None
        Lock for thread-safe PyBullet access.
    """

    def __init__(self, sim, kuka_id, ee_index,
                 omega_o=30.0, b0=1.0,
                 v_comp_max=0.15, warmup_steps=100,
                 sim_lock=None):
        self.p = sim
        self.kuka_id = kuka_id
        self.ee_index = ee_index
        self._lock = sim_lock or threading.Lock()

        # ESO gains from bandwidth parameterization (Gao, 2006)
        self.beta1 = 3.0 * omega_o
        self.beta2 = 3.0 * omega_o ** 2
        self.beta3 = omega_o ** 3
        self.b0 = b0
        self.omega_o = omega_o

        self.v_comp_max = v_comp_max
        self.warmup_steps = warmup_steps

        # Observer states (3 channels: x, y, z)
        self._z1 = np.zeros(3)   # estimated position
        self._z2 = np.zeros(3)   # estimated velocity
        self._z3 = np.zeros(3)   # estimated total disturbance

        self._initialized = False
        self._step_count = 0

        # Last known input (ADMM velocity applied to EE)
        self._u_prev = np.zeros(3)

    def update(self, dt, u=None):
        """Run one ESO update step.

        Parameters
        ----------
        dt : float
            Timestep (s).
        u : ndarray (3,) or None
            Control input (EE velocity from ADMM).
            If None, uses previous value.

        Returns
        -------
        f_hat : ndarray (3,)
            Estimated total disturbance (acceleration-level).
        """
        # Measure actual EE position
        with self._lock:
            ls = self.p.getLinkState(self.kuka_id, self.ee_index,
                                     computeForwardKinematics=True,
                                     computeLinkVelocity=True)
        x1_meas = np.array(ls[4])   # EE position (measured)

        if u is not None:
            self._u_prev = np.asarray(u, dtype=float)
        u_k = self._u_prev

        if not self._initialized:
            # Initialize observer states to measured values
            self._z1 = x1_meas.copy()
            self._z2 = np.array(ls[6]) if len(ls) > 6 else np.zeros(3)
            self._z3 = np.zeros(3)
            self._initialized = True
            return self._z3.copy()

        self._step_count += 1

        # --- ESO update (Euler integration) ---
        # Innovation: measurement - estimate
        e = x1_meas - self._z1

        # Observer dynamics
        z1_dot = self._z2 + self.beta1 * e
        z2_dot = self._z3 + self.beta2 * e + self.b0 * u_k
        z3_dot = self.beta3 * e

        # Integrate
        self._z1 += z1_dot * dt
        self._z2 += z2_dot * dt
        self._z3 += z3_dot * dt

        return self._z3.copy()

    def get_velocity_compensation(self, dt, m_eff=5.0, comp_gain=1.0):
        """Convert estimated disturbance to velocity compensation.

        The ESO estimates the total disturbance as an acceleration.
        We convert to a velocity correction:
            v_comp = -comp_gain * z3 * dt

        (z3 is acceleration-level, so z3 * dt gives velocity change
        per timestep that we need to counteract.)

        Parameters
        ----------
        dt : float
            Timestep.
        m_eff : float
            Not used (kept for interface compatibility).
        comp_gain : float
            Compensation gain.

        Returns
        -------
        v_comp : ndarray (3,)
            Velocity compensation to add to v_arm.
        """
        if self._step_count < self.warmup_steps:
            return np.zeros(3)

        # z3 is estimated disturbance acceleration
        # Compensate by opposing it: v_comp = -gain * z3 * dt
        v_comp = -comp_gain * self._z3 * dt

        # Clamp
        v_mag = np.linalg.norm(v_comp)
        if v_mag > self.v_comp_max:
            v_comp = v_comp * (self.v_comp_max / v_mag)

        return v_comp

    def get_reference_correction(self, dt, ref_gain=0.5, ref_max=0.03):
        """Shift the ADMM reference target to anticipate disturbance.

        Uses the ESO estimated disturbance (z3, acceleration-level) to
        predict where the EE will drift and pre-correct the target.

        This operates OUTSIDE the ADMM loop --- ADMM tracks the corrected
        target with full authority, no competition.

        Parameters
        ----------
        dt : float
            Timestep.
        ref_gain : float
            Gain on the position correction.
        ref_max : float
            Maximum reference shift magnitude (m).

        Returns
        -------
        dp_ref : ndarray (3,)
            Position offset to ADD to arm_target before ADMM sees it.
        """
        if self._step_count < self.warmup_steps:
            return np.zeros(3)

        # Predict drift: disturbance acceleration causes position drift
        # dp = -gain * z3 * dt^2  (oppose the acceleration)
        dp = -ref_gain * self._z3 * dt ** 2

        # Clamp
        dp_mag = np.linalg.norm(dp)
        if dp_mag > ref_max:
            dp = dp * (ref_max / dp_mag)

        return dp

    def get_cancellation_force(self, m_eff=5.0, force_gain=0.8, force_max=15.0):
        """Compute a force to directly oppose the estimated disturbance.

        Applies Newton's law: F_cancel = -m_eff * z3
        This fights the disturbance at the physics level via
        applyExternalForce, completely bypassing the controller.

        Parameters
        ----------
        m_eff : float
            Effective mass of the EE + payload (kg).
        force_gain : float
            Fraction of estimated disturbance to cancel (0-1).
            < 1.0 for robustness against estimation errors.
        force_max : float
            Maximum cancellation force magnitude (N).

        Returns
        -------
        F_cancel : ndarray (3,)
            Cancellation force in world frame (N).
        """
        if self._step_count < self.warmup_steps:
            return np.zeros(3)

        # F = -m * a  (oppose estimated disturbance acceleration)
        F = -force_gain * m_eff * self._z3

        # Clamp
        F_mag = np.linalg.norm(F)
        if F_mag > force_max:
            F = F * (force_max / F_mag)

        return F

    def get_estimated_velocity(self):
        """Return estimated EE velocity (z2)."""
        return self._z2.copy()

    def get_estimated_disturbance(self):
        """Return estimated total disturbance (z3)."""
        return self._z3.copy()

    def get_diagnostics(self):
        """Return diagnostic info for logging."""
        return {
            "z1": self._z1.copy(),
            "z2": self._z2.copy(),
            "z3_disturbance": self._z3.copy(),
            "z3_mag": float(np.linalg.norm(self._z3)),
        }

    def reset(self):
        """Reset observer state."""
        self._z1 = np.zeros(3)
        self._z2 = np.zeros(3)
        self._z3 = np.zeros(3)
        self._initialized = False
        self._step_count = 0
        self._u_prev = np.zeros(3)
