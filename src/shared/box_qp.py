"""Small strictly-convex box-QP solver used by the local CLF-ADMM agents.

Solves

    min_x  0.5 x.T H x - r.T x
    s.t.   lower <= x <= upper

for symmetric positive-definite H.  The implementation uses cyclic exact
coordinate minimisation (Gauss-Seidel coordinate descent).  For the tiny local
problems in this project (2 variables for the base, 6/7 for the arm), this is
fast, dependency-free, and converges to the box-constrained minimiser to a
user-selected KKT tolerance.
"""

from __future__ import annotations

import numpy as np


def _projected_gradient(x, grad, lower, upper, active_tol=1e-10):
    """Projected-gradient/KKT residual for box constraints."""
    pg = grad.copy()
    at_lower = x <= lower + active_tol
    at_upper = x >= upper - active_tol
    pg[at_lower] = np.minimum(pg[at_lower], 0.0)
    pg[at_upper] = np.maximum(pg[at_upper], 0.0)
    return pg


def solve_box_qp(H, rhs, lower, upper, x0=None,
                 tol=1e-9, max_sweeps=200):
    """Solve a small SPD quadratic program with simple bounds.

    Parameters
    ----------
    H : ndarray (n,n)
        Symmetric positive-definite Hessian.
    rhs : ndarray (n,)
        Linear right-hand side so the objective is
        ``0.5*x.T@H@x - rhs.T@x``.
    lower, upper : ndarray (n,)
        Componentwise bounds.
    x0 : ndarray (n,), optional
        Warm start.  If omitted, the clipped unconstrained minimiser is used.
    tol : float
        Infinity-norm projected-gradient stopping tolerance.
    max_sweeps : int
        Maximum cyclic coordinate sweeps.

    Returns
    -------
    x : ndarray
        Constrained minimiser to numerical tolerance.
    info : dict
        ``iterations``, ``kkt_residual``, ``converged`` and ``objective``.
    """
    H = np.asarray(H, dtype=float)
    rhs = np.asarray(rhs, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)

    n = rhs.size
    if H.shape != (n, n):
        raise ValueError(f"H must have shape {(n, n)}, got {H.shape}")
    if lower.shape != (n,) or upper.shape != (n,):
        raise ValueError("lower/upper must have shape (n,)")
    if np.any(lower > upper):
        bad = np.where(lower > upper)[0].tolist()
        raise ValueError(f"infeasible box bounds at indices {bad}")

    H = 0.5 * (H + H.T)
    diag = np.diag(H)
    if np.any(diag <= 0.0) or not np.all(np.isfinite(H)):
        raise ValueError("H must be finite with positive diagonal")

    if x0 is None:
        try:
            x = np.linalg.solve(H, rhs)
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(H, rhs, rcond=None)[0]
    else:
        x = np.asarray(x0, dtype=float).reshape(n).copy()
    x = np.clip(x, lower, upper)

    converged = False
    kkt = np.inf
    sweeps_done = 0

    for sweep in range(1, int(max_sweeps) + 1):
        # Exact 1-D minimisation in each coordinate while holding the others
        # fixed.  Updating x in-place yields a Gauss-Seidel sweep.
        for i in range(n):
            # grad_i = H[i,:] @ x - rhs[i].  The unconstrained coordinate
            # minimiser is x_i - grad_i/H_ii.
            grad_i = float(H[i, :] @ x - rhs[i])
            candidate = x[i] - grad_i / diag[i]
            x[i] = np.clip(candidate, lower[i], upper[i])

        grad = H @ x - rhs
        pg = _projected_gradient(x, grad, lower, upper)
        kkt = float(np.linalg.norm(pg, ord=np.inf))
        sweeps_done = sweep
        if kkt <= tol:
            converged = True
            break

    obj = float(0.5 * x @ H @ x - rhs @ x)
    return x, {
        "iterations": sweeps_done,
        "kkt_residual": kkt,
        "converged": converged,
        "objective": obj,
    }
