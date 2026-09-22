"""Centralized hard-CLF QP baseline for the three-layer architecture.

This module intentionally uses the SAME local quadratic models and box bounds
as the distributed base/arm agents.  The only difference is coordination:
all actuator variables are stacked into one vector and the shifted global CLF
constraint is imposed directly.

For x = [u_b, dq_a], the centralized baseline solves

    min  0.5 x^T H x - r^T x
    s.t. lower <= x <= upper
         a_b^T (u_b-u_b^0) + a_a^T (dq_a-dq_a^0) <= beta_clf.

The single linear inequality is handled through its scalar KKT multiplier.
For a fixed multiplier lambda >= 0, the problem is a box QP with linear term
r-lambda*c.  A monotone bisection on lambda therefore recovers the hard-CLF
solution without introducing a new optimisation dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np

try:
    from .box_qp import solve_box_qp
except ImportError:
    from box_qp import solve_box_qp


@dataclass
class CentralizedCLFQPResult:
    u_base: np.ndarray
    dq_arm: np.ndarray
    multiplier: float
    clf_value: float
    clf_bound_absolute: float
    clf_residual: float
    feasible: bool
    active: bool
    nominal_violation: float
    box_solves: int
    solve_time_ms: float
    kkt_residual: float
    objective: float
    status: str


class CentralizedCLFQPSolver:
    """Tiny centralized QP with one hard shifted-CLF inequality."""

    def __init__(self, tol=1e-8, multiplier_tol=1e-10,
                 max_bisection=60, box_tol=1e-9, box_sweeps=300):
        self.tol = float(tol)
        self.multiplier_tol = float(multiplier_tol)
        self.max_bisection = int(max_bisection)
        self.box_tol = float(box_tol)
        self.box_sweeps = int(box_sweeps)

    @staticmethod
    def _copy_model(model, name):
        if model is None:
            raise ValueError(f"missing {name} local QP model")
        required = ("H", "rhs", "lower", "upper")
        for key in required:
            if key not in model:
                raise ValueError(f"{name} QP model missing {key}")
        H = np.asarray(model["H"], dtype=float)
        rhs = np.asarray(model["rhs"], dtype=float).reshape(-1)
        lo = np.asarray(model["lower"], dtype=float).reshape(-1)
        hi = np.asarray(model["upper"], dtype=float).reshape(-1)
        if H.shape != (rhs.size, rhs.size):
            raise ValueError(f"invalid {name} Hessian shape")
        return H, rhs, lo, hi

    @staticmethod
    def _block_diag(A, B):
        out = np.zeros((A.shape[0] + B.shape[0],
                        A.shape[1] + B.shape[1]), dtype=float)
        out[:A.shape[0], :A.shape[1]] = A
        out[A.shape[0]:, A.shape[1]:] = B
        return out

    def solve(self, base_model, arm_model, a_base, a_arm, b_clf,
              u_base_nominal=None, dq_arm_nominal=None, **legacy_kwargs):
        t0 = time.perf_counter()
        Hb, rb, lb, ub = self._copy_model(base_model, "base")
        Ha, ra, la, ua = self._copy_model(arm_model, "arm")
        if rb.size != 2:
            raise ValueError("centralized baseline expects 2 base variables")

        H = self._block_diag(Hb, Ha)
        rhs = np.r_[rb, ra]
        lower = np.r_[lb, la]
        upper = np.r_[ub, ua]

        a_b = np.asarray(a_base, dtype=float).reshape(2)
        a_a = np.asarray(a_arm, dtype=float).reshape(ra.size)
        c = np.r_[a_b, a_a]
        if u_base_nominal is None:
            u_base_nominal = legacy_kwargs.pop("u_path", None)
        if u_base_nominal is None:
            raise ValueError("u_base_nominal is required")
        u0 = np.asarray(u_base_nominal, dtype=float).reshape(2)
        dq0 = (np.zeros(ra.size) if dq_arm_nominal is None else
               np.asarray(dq_arm_nominal, dtype=float).reshape(ra.size))

        # CLF correction constraint in absolute coordinates:
        #   a_b^T u_b + a_a^T dq
        #       <= beta_clf + a_b^T u_b^0 + a_a^T dq_a^0
        beta = float(b_clf + a_b @ u0 + a_a @ dq0)

        box_solves = 0
        x_nom, info_nom = solve_box_qp(
            H, rhs, lower, upper, tol=self.box_tol,
            max_sweeps=self.box_sweeps)
        box_solves += 1
        val_nom = float(c @ x_nom)
        nominal_violation = max(0.0, val_nom - beta)

        if val_nom <= beta + self.tol:
            elapsed = 1e3 * (time.perf_counter() - t0)
            return CentralizedCLFQPResult(
                u_base=x_nom[:2].copy(), dq_arm=x_nom[2:].copy(),
                multiplier=0.0, clf_value=val_nom,
                clf_bound_absolute=beta,
                clf_residual=val_nom - beta,
                feasible=True, active=False,
                nominal_violation=nominal_violation,
                box_solves=box_solves, solve_time_ms=elapsed,
                kkt_residual=float(info_nom["kkt_residual"]),
                objective=float(info_nom["objective"]), status="nominal-feasible")

        # Exact minimum of linear c^T x over a box.  If even this exceeds
        # beta, the hard CLF is infeasible under the current actuator bounds.
        x_min_linear = np.where(c >= 0.0, lower, upper)
        min_clf = float(c @ x_min_linear)
        if min_clf > beta + self.tol:
            # Return the maximum-CLF-reduction feasible command.  This is
            # explicitly flagged infeasible; no hidden slack is introduced.
            obj = float(0.5 * x_min_linear @ H @ x_min_linear - rhs @ x_min_linear)
            elapsed = 1e3 * (time.perf_counter() - t0)
            return CentralizedCLFQPResult(
                u_base=x_min_linear[:2].copy(), dq_arm=x_min_linear[2:].copy(),
                multiplier=float("inf"), clf_value=min_clf,
                clf_bound_absolute=beta,
                clf_residual=min_clf - beta,
                feasible=False, active=True,
                nominal_violation=nominal_violation,
                box_solves=box_solves, solve_time_ms=elapsed,
                kkt_residual=float("nan"), objective=obj,
                status="hard-clf-infeasible")

        # Find an upper multiplier that makes the box-QP CLF-feasible.
        lam_lo = 0.0
        lam_hi = 1.0
        x_hi = x_nom
        info_hi = info_nom
        for _ in range(60):
            x_hi, info_hi = solve_box_qp(
                H, rhs - lam_hi * c, lower, upper,
                x0=x_hi, tol=self.box_tol, max_sweeps=self.box_sweeps)
            box_solves += 1
            if float(c @ x_hi) <= beta:
                break
            lam_hi *= 2.0
        else:
            elapsed = 1e3 * (time.perf_counter() - t0)
            return CentralizedCLFQPResult(
                u_base=x_hi[:2].copy(), dq_arm=x_hi[2:].copy(),
                multiplier=lam_hi, clf_value=float(c @ x_hi),
                clf_bound_absolute=beta,
                clf_residual=float(c @ x_hi - beta),
                feasible=False, active=True,
                nominal_violation=nominal_violation,
                box_solves=box_solves, solve_time_ms=elapsed,
                kkt_residual=float(info_hi.get("kkt_residual", np.nan)),
                objective=float(info_hi.get("objective", np.nan)),
                status="multiplier-bracketing-failed")

        x = x_hi
        info = info_hi
        lam = lam_hi
        for _ in range(self.max_bisection):
            lam_mid = 0.5 * (lam_lo + lam_hi)
            x_mid, info_mid = solve_box_qp(
                H, rhs - lam_mid * c, lower, upper,
                x0=x, tol=self.box_tol, max_sweeps=self.box_sweeps)
            box_solves += 1
            value = float(c @ x_mid)
            x, info, lam = x_mid, info_mid, lam_mid
            if abs(value - beta) <= self.tol or (lam_hi - lam_lo) <= self.multiplier_tol:
                break
            if value > beta:
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid
                x_hi = x_mid.copy()
                info_hi = dict(info_mid)

        clf_value = float(c @ x)
        # If the final bisection point fell microscopically on the violating
        # side, use the last feasible endpoint.
        if clf_value > beta + self.tol:
            x, info, lam = x_hi, info_hi, lam_hi
            clf_value = float(c @ x)

        objective = float(0.5 * x @ H @ x - rhs @ x)
        grad = H @ x - rhs + lam * c
        # A simple projected stationarity diagnostic for the box+active CLF KKT.
        pg = grad.copy()
        at_lo = x <= lower + 1e-8
        at_hi = x >= upper - 1e-8
        pg[at_lo] = np.minimum(pg[at_lo], 0.0)
        pg[at_hi] = np.maximum(pg[at_hi], 0.0)
        kkt = float(max(np.linalg.norm(pg, ord=np.inf),
                        max(0.0, clf_value - beta)))
        elapsed = 1e3 * (time.perf_counter() - t0)
        return CentralizedCLFQPResult(
            u_base=x[:2].copy(), dq_arm=x[2:].copy(),
            multiplier=float(lam), clf_value=clf_value,
            clf_bound_absolute=beta,
            clf_residual=clf_value - beta,
            feasible=bool(clf_value <= beta + self.tol), active=True,
            nominal_violation=nominal_violation,
            box_solves=box_solves, solve_time_ms=elapsed,
            kkt_residual=kkt, objective=objective, status="active-clf")
