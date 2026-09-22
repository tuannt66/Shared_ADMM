"""Pure-numpy layers for the revised modular mobile-manipulator controller.

V10.8 FULL-6D
-------------
The layers are now task-dimension generic.  The production coordinator uses a
normalized 6D task vector

    [v_x, v_y, v_z, l_R*omega_x, l_R*omega_y, l_R*omega_z]

and correspondingly scaled 6xN Jacobians.  The scalar ADMM coupling structure
is unchanged.

Layer 1 -- Residual task geometry
    v_nom = J_b u_b^0 + J_a dq_a^0
    v_r   = v_task - v_nom
    v_parallel = P_b v_r,  v_perp = (I-P_b) v_r.

Layer 2 -- Adaptive role preference
    Directional remaining capability produces gamma_b and gamma_a. Their
    ratio produces alpha, which is converted into positive cost weights.

Layer 3 -- Global CLF coordination
    The CLF is written in correction variables:
        delta_u_b = u_b-u_b^0, delta_dq_a = dq_a-dq_a^0
    and
        a_b^T delta_u_b + a_a^T delta_dq_a <= beta_CLF.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ResidualTaskResult:
    v_task: np.ndarray
    v_nominal: np.ndarray
    u_base_nominal: np.ndarray
    dq_arm_nominal: np.ndarray
    v_residual: np.ndarray
    P_base: np.ndarray
    v_shareable: np.ndarray
    v_mandatory: np.ndarray
    reconstruction_error: float
    projector_idempotence_error: float
    mandatory_base_projection_error: float

    @property
    def v_path_ee(self):
        return self.v_nominal

    @property
    def u_path(self):
        return self.u_base_nominal


class ResidualTaskDecompositionLayer:
    """Decompose the task residual after both local nominal controllers."""

    def __init__(self, pinv_rcond=1e-6):
        self.pinv_rcond = float(pinv_rcond)

    def compute(self, v_task, J_base, u_base_nominal,
                J_arm=None, dq_arm_nominal=None):
        v_task = np.asarray(v_task, dtype=float).reshape(-1)
        dim = v_task.size
        J_b = np.asarray(J_base, dtype=float)
        if J_b.shape != (dim, 2):
            raise ValueError(f"J_base must have shape ({dim},2), got {J_b.shape}")
        u0 = np.asarray(u_base_nominal, dtype=float).reshape(2)

        if J_arm is None or dq_arm_nominal is None:
            v_nom = J_b @ u0
            dq0 = np.zeros(0)
        else:
            J_a = np.asarray(J_arm, dtype=float)
            if J_a.ndim != 2 or J_a.shape[0] != dim:
                raise ValueError("J_arm row dimension must match task dimension")
            dq0 = np.asarray(dq_arm_nominal, dtype=float).reshape(J_a.shape[1])
            v_nom = J_b @ u0 + J_a @ dq0

        v_res = v_task - v_nom
        J_b_pinv = np.linalg.pinv(J_b, rcond=self.pinv_rcond)
        P_b = J_b @ J_b_pinv
        v_share = P_b @ v_res
        v_mand = (np.eye(dim) - P_b) @ v_res

        reconstruction = v_task - (v_nom + v_share + v_mand)
        projector_error = P_b @ P_b - P_b
        mandatory_projection = P_b @ v_mand

        return ResidualTaskResult(
            v_task=v_task.copy(),
            v_nominal=v_nom.copy(),
            u_base_nominal=u0.copy(),
            dq_arm_nominal=dq0.copy(),
            v_residual=v_res.copy(),
            P_base=P_b.copy(),
            v_shareable=v_share.copy(),
            v_mandatory=v_mand.copy(),
            reconstruction_error=float(np.linalg.norm(reconstruction)),
            projector_idempotence_error=float(np.linalg.norm(projector_error)),
            mandatory_base_projection_error=float(np.linalg.norm(mandatory_projection)),
        )


@dataclass
class CapabilityAllocationResult:
    alpha_arm: float
    alpha_base: float
    weight_arm: float
    weight_base: float
    gamma_arm: float
    gamma_base: float
    diagnostics: dict
    task_dim: int = 6

    @property
    def v_arm_ref(self):
        return np.full(self.task_dim, np.nan)

    @property
    def v_base_correction_ref(self):
        return np.full(self.task_dim, np.nan)

    @property
    def allocation_reconstruction_error(self):
        return float("nan")


class TaskSpaceCapabilityAllocationLayer:
    """Convert directional capability into role preference and cost weights."""

    def __init__(self, allocator=None, role_weight_floor=0.10):
        self.allocator = allocator
        self.role_weight_floor = float(role_weight_floor)

    def _weights(self, alpha):
        eps = (self.allocator.role_weight_floor
               if self.allocator is not None and
               hasattr(self.allocator, "role_weight_floor")
               else self.role_weight_floor)
        wb = eps + (1.0 - 2.0 * eps) * float(alpha)
        return float(wb), float(1.0 - wb)

    def allocate(self, J_arm, J_base, residual, dt, alpha_fallback=0.5,
                 *, base_nominal=None, arm_nominal=None,
                 base_lower=None, base_upper=None,
                 arm_lower=None, arm_upper=None):
        task_dim = int(np.asarray(J_base).shape[0])
        if self.allocator is None:
            alpha = float(np.clip(alpha_fallback, 0.0, 1.0))
            wb, wa = self._weights(alpha)
            diag = {
                "alpha": alpha, "alpha_raw": alpha,
                "gamma_arm": float("nan"), "gamma_base": float("nan"),
                "weight_arm": wa, "weight_base": wb,
            }
        else:
            alpha = float(self.allocator.update(
                np.asarray(J_arm, dtype=float),
                np.asarray(J_base, dtype=float),
                residual.u_base_nominal,
                residual.v_shareable,
                residual.v_mandatory,
                dt,
                base_nominal=(residual.u_base_nominal if base_nominal is None
                              else base_nominal),
                arm_nominal=((None if residual.dq_arm_nominal.size == 0
                              else residual.dq_arm_nominal)
                             if arm_nominal is None else arm_nominal),
                base_lower=base_lower, base_upper=base_upper,
                arm_lower=arm_lower, arm_upper=arm_upper,
            ))
            diag = self.allocator.diagnostics()
            wb = float(diag["weight_base"])
            wa = float(diag["weight_arm"])

        return CapabilityAllocationResult(
            alpha_arm=alpha,
            alpha_base=1.0-alpha,
            weight_arm=wa,
            weight_base=wb,
            gamma_arm=float(diag.get("gamma_arm", np.nan)),
            gamma_base=float(diag.get("gamma_base", np.nan)),
            diagnostics=diag,
            task_dim=task_dim,
        )


@dataclass
class ShiftedCLFResult:
    e: np.ndarray
    V: float
    a_base: np.ndarray
    a_arm: np.ndarray
    b_clf: float
    phi_nominal: float
    nominal_velocity: np.ndarray

    @property
    def path_clf_contribution(self):
        return self.phi_nominal


class ShiftedCLFADMMCoordinationLayer:
    """Global CLF geometry around independent nominal commands."""

    def __init__(self, P=None, clf_rate=1.0):
        self.P = None if P is None else np.asarray(P, dtype=float)
        self.clf_rate = float(clf_rate)

    def _P_for_dim(self, dim):
        if self.P is None:
            return np.eye(dim)
        if self.P.shape != (dim, dim):
            raise ValueError(
                f"CLF P has shape {self.P.shape}, task dimension is {dim}")
        return self.P

    def compute_from_error(self, error, vel_desired,
                           J_base, J_arm, u_base_nominal,
                           dq_arm_nominal=None):
        e = np.asarray(error, dtype=float).reshape(-1)
        dim = e.size
        pdot_d = np.asarray(vel_desired, dtype=float).reshape(dim)
        J_b = np.asarray(J_base, dtype=float)
        J_a = np.asarray(J_arm, dtype=float)
        if J_b.shape != (dim, 2):
            raise ValueError(f"J_base must have shape ({dim},2)")
        if J_a.ndim != 2 or J_a.shape[0] != dim:
            raise ValueError("J_arm row dimension must match error dimension")

        u0 = np.asarray(u_base_nominal, dtype=float).reshape(2)
        dq0 = (np.zeros(J_a.shape[1]) if dq_arm_nominal is None else
               np.asarray(dq_arm_nominal, dtype=float).reshape(J_a.shape[1]))
        P = self._P_for_dim(dim)

        Pe = P @ e
        V = 0.5 * float(e @ Pe)
        v_nom = J_b @ u0 + J_a @ dq0
        a_b = J_b.T @ Pe
        a_a = J_a.T @ Pe
        phi_nom = float(e @ P @ (v_nom - pdot_d))
        beta = float(-self.clf_rate * V - phi_nom)

        return ShiftedCLFResult(
            e=e.copy(), V=V,
            a_base=np.asarray(a_b, dtype=float).copy(),
            a_arm=np.asarray(a_a, dtype=float).copy(),
            b_clf=beta,
            phi_nominal=phi_nom,
            nominal_velocity=v_nom.copy(),
        )

    def compute_geometry(self, ee_pos, pos_desired, vel_desired,
                         J_base, J_arm, u_base_nominal,
                         dq_arm_nominal=None):
        """Backward-compatible positional wrapper."""
        p_e = np.asarray(ee_pos, dtype=float).reshape(3)
        p_d = np.asarray(pos_desired, dtype=float).reshape(3)
        return self.compute_from_error(
            p_e-p_d, np.asarray(vel_desired, dtype=float).reshape(3),
            J_base, J_arm, u_base_nominal, dq_arm_nominal)

    @staticmethod
    def project_hard_halfspace(y_base, y_arm, b_clf):
        y_b = float(y_base)
        y_a = float(y_arm)
        b = float(b_clf)
        violation = y_b + y_a - b
        if violation <= 0.0:
            return y_b, y_a, 0.0
        correction = 0.5 * violation
        return y_b-correction, y_a-correction, 0.0

    def derivative(self, geometry, J_base, u_base,
                   J_arm, dq_arm, vel_desired):
        dim = geometry.e.size
        J_b = np.asarray(J_base, dtype=float).reshape(dim, 2)
        J_a = np.asarray(J_arm, dtype=float)
        u_b = np.asarray(u_base, dtype=float).reshape(2)
        dq = np.asarray(dq_arm, dtype=float).reshape(J_a.shape[1])
        pdot_d = np.asarray(vel_desired, dtype=float).reshape(dim)
        P = self._P_for_dim(dim)
        return float(
            geometry.e @ P @ (J_b @ u_b + J_a @ dq - pdot_d))

    @staticmethod
    def correction_contributions(geometry, u_base, u_base_nominal,
                                 dq_arm, dq_arm_nominal=None):
        u_b = np.asarray(u_base, dtype=float).reshape(2)
        u0 = np.asarray(u_base_nominal, dtype=float).reshape(2)
        dq = np.asarray(dq_arm, dtype=float)
        dq0 = (np.zeros_like(dq) if dq_arm_nominal is None else
               np.asarray(dq_arm_nominal, dtype=float).reshape(dq.shape))
        s_b = float(geometry.a_base @ (u_b-u0))
        s_a = float(geometry.a_arm @ (dq-dq0))
        return s_b, s_a
