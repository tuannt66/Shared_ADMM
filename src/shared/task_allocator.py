"""Adaptive role-preference allocator for modular mobile manipulation.

The allocator implements the *preference* architecture used by the revised
controller design:

    v_r -> (v_parallel, v_perp) -> (gamma_b, gamma_a) -> alpha -> weights.

Crucially, ``alpha`` does **not** split the Cartesian task.  The actual base
and arm corrections remain decision variables of the CLF-constrained
coordination problem.  ``alpha`` only changes their relative quadratic costs.

Definitions
-----------
``gamma_b`` is the largest fraction s_b in [0, 1] of the shareable component
that can be added to the nominal base command without violating the current
base bounds.

``gamma_a`` is the largest fraction s_a in [0, 1] of the shareable component
that can be added to the nominal arm command *after* accounting for the
mandatory component v_perp, without violating the current arm bounds.

The role index follows the convention

    alpha -> 0 : base preferred
    alpha -> 1 : arm preferred.

Positive cost weights are normalized to sum to one:

    w_b = eps_w + (1-2 eps_w) alpha
    w_a = 1 - w_b,

so the preferred subsystem receives the smaller correction penalty.
"""

from __future__ import annotations

import numpy as np


class TaskSpaceCapabilityBalancer:
    """Directional remaining-capability role allocator.

    The historical class name is retained so existing scripts continue to
    import successfully.  Its semantics are intentionally different from the
    V9 minimax hard task allocator: ``alpha`` is now a role *preference*, not
    an allocation fraction.
    """

    allocation_kind = "adaptive"

    def __init__(self, max_joint_velocity, base_u_lower, base_u_upper,
                 damping=0.06, alpha_rate=5.0, residual_deadband=0.01,
                 alpha_init=0.5, margin_epsilon=1e-6,
                 role_weight_floor=0.10,
                 # Legacy arguments accepted but no longer used by the
                 # preference formulation.
                 grid_points=201, base_path_weight=1.0,
                 base_locomotion_jmax=0.253645,
                 base_pinv_sigma_floor_rel=0.01,
                 base_pinv_sigma_floor_abs=1e-6,
                 **_ignored):
        qmax = np.asarray(max_joint_velocity, dtype=float)
        if qmax.ndim == 0:
            qmax = qmax.reshape(1)
        if np.any(qmax <= 0.0):
            raise ValueError("max_joint_velocity must be positive")

        self.max_joint_velocity = qmax.copy()
        self.base_u_lower = np.asarray(base_u_lower, dtype=float).reshape(2)
        self.base_u_upper = np.asarray(base_u_upper, dtype=float).reshape(2)
        if np.any(self.base_u_upper <= self.base_u_lower):
            raise ValueError("base_u_upper must be greater than base_u_lower")

        self.damping = float(damping)
        self.alpha_rate = float(alpha_rate)
        self.residual_deadband = float(max(residual_deadband, 0.0))
        self.margin_epsilon = float(max(margin_epsilon, 1e-12))
        self.role_weight_floor = float(role_weight_floor)
        if not (0.0 < self.role_weight_floor < 0.5):
            raise ValueError("role_weight_floor must lie in (0, 0.5)")

        # Retained only for diagnostics/backward-compatible configuration.
        self.grid_points = int(grid_points)
        self.base_path_weight = float(base_path_weight)
        self.base_locomotion_jmax = float(base_locomotion_jmax)
        self.base_pinv_sigma_floor_rel = float(base_pinv_sigma_floor_rel)
        self.base_pinv_sigma_floor_abs = float(base_pinv_sigma_floor_abs)

        self.alpha = float(np.clip(alpha_init, 0.0, 1.0))
        self.alpha_raw = self.alpha
        self.alpha_capacity = self.alpha
        self.weight_base = self._weight_base(self.alpha)
        self.weight_arm = 1.0 - self.weight_base

        self.gamma_base = 1.0
        self.gamma_arm = 1.0
        self.mandatory_feasible = True
        self.held_deadband = False
        self.residual_norm = 0.0
        self.shareable_norm = 0.0
        self.mandatory_norm = 0.0
        self.qdot_mandatory = np.zeros_like(self.max_joint_velocity)
        self.qdot_shareable_full = np.zeros_like(self.max_joint_velocity)
        self.qdot_full = np.zeros_like(self.max_joint_velocity)
        self.base_correction_full = np.zeros(2)
        self.base_remaining_margin = np.zeros(2)
        self.arm_remaining_margin = np.zeros_like(self.max_joint_velocity)
        self.arm_mandatory_tracking_error = 0.0
        self.arm_shareable_tracking_error = 0.0
        self.path_outside_bounds = False

        # Backward-compatible diagnostic aliases.  These are no longer used
        # by the controller to allocate the task.
        self.alpha_geo = self.alpha
        self.delta_err = 0.0
        self.delta_manip = 0.0
        self.delta_admm = 0.0
        self.carry_dist = 0.0
        self.capacity_feasible = True
        self.alpha_feasible_low = 0.0
        self.alpha_feasible_high = 1.0
        self.eta_arm = 0.0
        self.eta_base = 0.0
        self.eta_base_actuator = 0.0
        self.eta_base_locomotion = 0.0
        self.eta_worst = 0.0
        self.eta_mandatory = 0.0
        self.eta_shareable_full = 0.0
        self.eta_arm_at_0 = self.eta_arm_at_05 = self.eta_arm_at_1 = np.nan
        self.eta_base_at_0 = self.eta_base_at_05 = self.eta_base_at_1 = np.nan

    def _qmax_for_n(self, n):
        if self.max_joint_velocity.size == 1:
            return np.full(n, float(self.max_joint_velocity[0]))
        if self.max_joint_velocity.size == n:
            return self.max_joint_velocity.copy()
        raise ValueError("max_joint_velocity length must match arm joints")

    def _weight_base(self, alpha):
        eps = self.role_weight_floor
        return float(eps + (1.0 - 2.0 * eps) * float(alpha))

    @staticmethod
    def _max_feasible_fraction(x0, direction, lower, upper, tol=1e-12):
        """Largest s in [0,1] with lower <= x0+s*direction <= upper."""
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        d = np.asarray(direction, dtype=float).reshape(x0.shape)
        lo = np.asarray(lower, dtype=float).reshape(x0.shape)
        hi = np.asarray(upper, dtype=float).reshape(x0.shape)

        if np.any(x0 < lo - 1e-9) or np.any(x0 > hi + 1e-9):
            return 0.0

        smax = 1.0
        for xi, di, li, ui in zip(x0, d, lo, hi):
            if di > tol:
                smax = min(smax, (ui - xi) / di)
            elif di < -tol:
                smax = min(smax, (xi - li) / (-di))
        return float(np.clip(smax, 0.0, 1.0))

    def reset(self, alpha=None):
        if alpha is not None:
            self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.alpha_raw = self.alpha
        self.alpha_capacity = self.alpha
        self.weight_base = self._weight_base(self.alpha)
        self.weight_arm = 1.0 - self.weight_base
        self.gamma_base = self.gamma_arm = 1.0
        self.held_deadband = False

    def update(self, J_arm, J_base, u_path, v_shareable, v_mandatory, dt,
               *, base_nominal=None, arm_nominal=None,
               base_lower=None, base_upper=None,
               arm_lower=None, arm_upper=None):
        """Update role preference from directional remaining capability.

        Parameters after ``dt`` are optional for backward compatibility.  The
        revised coordinator passes the actual nominal commands and the exact
        local-QP bounds, so gamma uses the same feasible sets as control.
        """
        J_a = np.asarray(J_arm, dtype=float)
        J_b = np.asarray(J_base, dtype=float)
        if J_b.ndim != 2 or J_b.shape[1] != 2:
            raise ValueError("J_base must be task_dim x 2")
        task_dim = J_b.shape[0]
        if J_a.ndim != 2 or J_a.shape[0] != task_dim:
            raise ValueError("J_arm and J_base must have the same task rows")
        v_s = np.asarray(v_shareable, dtype=float).reshape(task_dim)
        v_m = np.asarray(v_mandatory, dtype=float).reshape(task_dim)
        n = J_a.shape[1]
        qmax = self._qmax_for_n(n)

        u0 = (np.asarray(u_path, dtype=float).reshape(2)
              if base_nominal is None else
              np.asarray(base_nominal, dtype=float).reshape(2))
        lb = (self.base_u_lower if base_lower is None else
              np.asarray(base_lower, dtype=float).reshape(2))
        ub = (self.base_u_upper if base_upper is None else
              np.asarray(base_upper, dtype=float).reshape(2))
        dq0 = (np.zeros(n) if arm_nominal is None else
               np.asarray(arm_nominal, dtype=float).reshape(n))
        la = (-qmax if arm_lower is None else
              np.asarray(arm_lower, dtype=float).reshape(n))
        ua = (qmax if arm_upper is None else
              np.asarray(arm_upper, dtype=float).reshape(n))

        self.shareable_norm = float(np.linalg.norm(v_s))
        self.mandatory_norm = float(np.linalg.norm(v_m))
        self.residual_norm = float(np.linalg.norm(v_s + v_m))
        self.held_deadband = self.shareable_norm < self.residual_deadband

        # Base: full correction needed for all of v_parallel.
        Jb_pinv = np.linalg.pinv(J_b, rcond=1e-6)
        self.base_correction_full = Jb_pinv @ v_s
        self.path_outside_bounds = bool(
            np.any(u0 < lb - 1e-9) or np.any(u0 > ub + 1e-9))
        self.gamma_base = self._max_feasible_fraction(
            u0, self.base_correction_full, lb, ub)
        directional_margin_b = np.where(
            self.base_correction_full >= 0.0, ub - u0, u0 - lb)
        self.base_remaining_margin = np.maximum(directional_margin_b, 0.0)

        # Arm capability uses the Moore--Penrose inverse from the Section-4
        # derivation (not the DLS law used by the local nominal controller).
        # This makes weak directions expensive instead of hiding them behind
        # damping.  Range-space residuals explicitly detect directions that
        # the arm cannot realize at the current configuration.
        Ja_pinv = np.linalg.pinv(J_a, rcond=1e-6)
        self.qdot_mandatory = Ja_pinv @ v_m
        self.qdot_shareable_full = Ja_pinv @ v_s
        self.qdot_full = self.qdot_mandatory + self.qdot_shareable_full

        self.arm_mandatory_tracking_error = float(
            np.linalg.norm(J_a @ self.qdot_mandatory - v_m))
        self.arm_shareable_tracking_error = float(
            np.linalg.norm(J_a @ self.qdot_shareable_full - v_s))
        mandatory_tol = 1e-7 + 1e-3 * float(np.linalg.norm(v_m))
        shareable_tol = 1e-7 + 1e-3 * float(np.linalg.norm(v_s))
        mandatory_representable = self.arm_mandatory_tracking_error <= mandatory_tol
        shareable_representable = self.arm_shareable_tracking_error <= shareable_tol

        mandatory_point = dq0 + self.qdot_mandatory
        self.mandatory_feasible = bool(
            mandatory_representable and
            np.all(mandatory_point >= la - 1e-9) and
            np.all(mandatory_point <= ua + 1e-9))
        if self.mandatory_feasible and shareable_representable:
            self.gamma_arm = self._max_feasible_fraction(
                mandatory_point, self.qdot_shareable_full, la, ua)
        else:
            self.gamma_arm = 0.0

        directional_margin_a = np.where(
            self.qdot_shareable_full >= 0.0,
            ua - mandatory_point,
            mandatory_point - la)
        self.arm_remaining_margin = np.maximum(directional_margin_a, 0.0)

        # Capability utilization aliases retained for old plots.  A value near
        # one means little remaining shareable capacity.
        self.eta_base = 1.0 - self.gamma_base
        self.eta_base_actuator = self.eta_base
        self.eta_base_locomotion = np.nan
        self.eta_arm = 1.0 - self.gamma_arm
        self.eta_worst = max(self.eta_arm, self.eta_base)
        self.eta_mandatory = float(np.max(
            np.abs(self.qdot_mandatory) / np.maximum(qmax, 1e-12)))
        self.eta_shareable_full = float(np.max(
            np.abs(self.qdot_shareable_full) / np.maximum(qmax, 1e-12)))

        if self.held_deadband:
            self.alpha_raw = self.alpha
        else:
            denom = self.gamma_arm + self.gamma_base
            if denom <= self.margin_epsilon:
                # No subsystem has useful shareable margin.  Keep the previous
                # preference; the CLF feasibility logic handles the shortage.
                self.alpha_raw = self.alpha
            else:
                self.alpha_raw = float(self.gamma_arm / (denom + self.margin_epsilon))

        max_delta = self.alpha_rate * max(float(dt), 0.0)
        delta = float(np.clip(self.alpha_raw - self.alpha,
                              -max_delta, max_delta))
        self.alpha = float(np.clip(self.alpha + delta, 0.0, 1.0))
        self.alpha_capacity = self.alpha_raw
        self.weight_base = self._weight_base(self.alpha)
        self.weight_arm = 1.0 - self.weight_base
        self.alpha_geo = self.alpha_raw
        self.carry_dist = self.eta_worst
        self.capacity_feasible = bool(self.mandatory_feasible and
                                      (self.gamma_arm > 0.0 or
                                       self.gamma_base > 0.0 or
                                       self.shareable_norm < self.residual_deadband))
        return self.alpha

    @property
    def task_share_arm(self):
        """Backward-compatible alias: now a role preference, not a task share."""
        return self.alpha

    @property
    def task_share_base(self):
        """Backward-compatible complement; not used to split Cartesian task."""
        return 1.0 - self.alpha

    def role_weights(self):
        return float(self.weight_base), float(self.weight_arm)

    def diagnostics(self):
        return {
            "alpha": float(self.alpha),
            "alpha_raw": float(self.alpha_raw),
            "alpha_capacity": float(self.alpha_capacity),
            "gamma_base": float(self.gamma_base),
            "gamma_arm": float(self.gamma_arm),
            "weight_base": float(self.weight_base),
            "weight_arm": float(self.weight_arm),
            "mandatory_feasible": bool(self.mandatory_feasible),
            "residual_norm": float(self.residual_norm),
            "shareable_norm": float(self.shareable_norm),
            "mandatory_norm": float(self.mandatory_norm),
            "qdot_full": self.qdot_full.copy(),
            "qdot_mandatory": self.qdot_mandatory.copy(),
            "qdot_shareable_full": self.qdot_shareable_full.copy(),
            "base_correction_full": self.base_correction_full.copy(),
            "base_remaining_margin": self.base_remaining_margin.copy(),
            "arm_remaining_margin": self.arm_remaining_margin.copy(),
            "arm_mandatory_tracking_error": float(self.arm_mandatory_tracking_error),
            "arm_shareable_tracking_error": float(self.arm_shareable_tracking_error),
            "held_deadband": bool(self.held_deadband),
            "capacity_feasible": bool(self.capacity_feasible),
            "alpha_feasible_low": 0.0,
            "alpha_feasible_high": 1.0,
            "alpha_rate": float(self.alpha_rate),
            "role_weight_floor": float(self.role_weight_floor),
            "eta_arm": float(self.eta_arm),
            "eta_base": float(self.eta_base),
            "eta_base_actuator": float(self.eta_base_actuator),
            "eta_base_locomotion": float(self.eta_base_locomotion),
            "eta_worst": float(self.eta_worst),
            "eta_mandatory": float(self.eta_mandatory),
            "eta_shareable_full": float(self.eta_shareable_full),
            "eta_arm_at_0": float(self.eta_arm_at_0),
            "eta_arm_at_05": float(self.eta_arm_at_05),
            "eta_arm_at_1": float(self.eta_arm_at_1),
            "eta_base_at_0": float(self.eta_base_at_0),
            "eta_base_at_05": float(self.eta_base_at_05),
            "eta_base_at_1": float(self.eta_base_at_1),
            "path_outside_bounds": bool(self.path_outside_bounds),
            "base_path_weight": float(self.base_path_weight),
            "base_locomotion_jmax": float(self.base_locomotion_jmax),
            # Retired V9 diagnostics kept as NaNs for old CSV readers.
            "j_locomotion": float("nan"),
            "j_locomotion_at_0": float("nan"),
            "j_locomotion_at_05": float("nan"),
            "j_locomotion_at_1": float("nan"),
            "base_pinv_sigma_min": float("nan"),
            "base_pinv_sigma_max": float("nan"),
            "base_pinv_sigma_floor": float("nan"),
            "base_pinv_condition_raw": float("nan"),
            "base_pinv_gain_max": float("nan"),
            "base_pinv_regularized": False,
        }


# Research-facing name for new code; legacy imports continue to work.
AdaptiveRolePreferenceAllocator = TaskSpaceCapabilityBalancer
ArmCapabilityAllocator = TaskSpaceCapabilityBalancer


class FixedTaskSpaceCapabilityMonitor(TaskSpaceCapabilityBalancer):
    """Fixed role-preference baseline with the same capability diagnostics."""
    allocation_kind = "fixed"

    def __init__(self, fixed_alpha=0.5, **kwargs):
        self.fixed_alpha = float(np.clip(fixed_alpha, 0.0, 1.0))
        kwargs["alpha_init"] = self.fixed_alpha
        super().__init__(**kwargs)

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self.alpha = self.fixed_alpha
        self.alpha_raw = self.fixed_alpha
        self.alpha_capacity = self.fixed_alpha
        self.weight_base = self._weight_base(self.alpha)
        self.weight_arm = 1.0 - self.weight_base
        self.alpha_geo = self.alpha
        return self.alpha
