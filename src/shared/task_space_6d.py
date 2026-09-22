"""SO(3) and normalized 6D task-space helpers.

The 6D controller uses spatial twists ordered as

    xi = [v_x, v_y, v_z, omega_x, omega_y, omega_z].

Linear velocity has units m/s while angular velocity has units rad/s.  To keep
the residual decomposition, capability allocator and CLF dimensionally
well-conditioned, angular components are converted to an equivalent linear
scale through

    S = diag(1,1,1,l_R,l_R,l_R)

where ``l_R`` is a configurable characteristic length (default 0.20 m).

The physical arm/base Jacobians remain unscaled.  Scaling is used only in the
task-space geometry and weighted DLS calculations.
"""

from __future__ import annotations
import numpy as np


def skew(v):
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -z, y],
                     [z, 0.0, -x],
                     [-y, x, 0.0]], dtype=float)


def vee(M):
    M = np.asarray(M, dtype=float).reshape(3, 3)
    return np.array([M[2, 1] - M[1, 2],
                     M[0, 2] - M[2, 0],
                     M[1, 0] - M[0, 1]], dtype=float) * 0.5


def so3_log(R):
    """Rotation-vector logarithm of R in radians.

    Numerically stable for both small angles and rotations close to pi.
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(c))
    if theta < 1e-8:
        return vee(R - R.T)
    if np.pi - theta < 1e-5:
        # Robust axis extraction near pi.
        A = 0.5 * (R + np.eye(3))
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        # Recover signs from the off-diagonal terms.
        if axis[0] >= axis[1] and axis[0] >= axis[2] and axis[0] > 1e-8:
            axis[1] = np.copysign(axis[1], R[0, 1] + R[1, 0])
            axis[2] = np.copysign(axis[2], R[0, 2] + R[2, 0])
        elif axis[1] >= axis[2] and axis[1] > 1e-8:
            axis[0] = np.copysign(axis[0], R[0, 1] + R[1, 0])
            axis[2] = np.copysign(axis[2], R[1, 2] + R[2, 1])
        elif axis[2] > 1e-8:
            axis[0] = np.copysign(axis[0], R[0, 2] + R[2, 0])
            axis[1] = np.copysign(axis[1], R[1, 2] + R[2, 1])
        n = np.linalg.norm(axis)
        if n < 1e-10:
            return np.zeros(3)
        return theta * axis / n
    return (theta / (2.0 * np.sin(theta))) * np.array(
        [R[2, 1] - R[1, 2],
         R[0, 2] - R[2, 0],
         R[1, 0] - R[0, 1]], dtype=float)


def so3_exp(phi):
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    if theta < 1e-10:
        K = skew(phi)
        return np.eye(3) + K + 0.5 * (K @ K)
    a = phi / theta
    K = skew(a)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def interpolate_rotation(R0, R1, s):
    """Geodesic interpolation R(s) = Exp(s Log(R1 R0^T)) R0."""
    w = float(np.clip(s, 0.0, 1.0))
    R0 = np.asarray(R0, dtype=float).reshape(3, 3)
    R1 = np.asarray(R1, dtype=float).reshape(3, 3)
    return so3_exp(w * so3_log(R1 @ R0.T)) @ R0


def orientation_error_current_minus_desired(R_current, R_desired):
    """Spatial/world small-angle error whose derivative is approximately
    omega_current - omega_desired.
    """
    Rc = np.asarray(R_current, dtype=float).reshape(3, 3)
    Rd = np.asarray(R_desired, dtype=float).reshape(3, 3)
    return so3_log(Rc @ Rd.T)


def orientation_error_desired_minus_current(R_current, R_desired):
    """Servo error pointing from current orientation toward desired."""
    Rc = np.asarray(R_current, dtype=float).reshape(3, 3)
    Rd = np.asarray(R_desired, dtype=float).reshape(3, 3)
    return so3_log(Rd @ Rc.T)


def pose_error_6d_current_minus_desired(
        p_current, R_current, p_desired, R_desired):
    return np.r_[
        np.asarray(p_current, dtype=float).reshape(3)
        - np.asarray(p_desired, dtype=float).reshape(3),
        orientation_error_current_minus_desired(R_current, R_desired),
    ]


def task_scale_matrix(angular_scale=0.20):
    lR = float(angular_scale)
    if not np.isfinite(lR) or lR <= 0.0:
        raise ValueError("angular_scale must be finite and > 0")
    return np.diag([1.0, 1.0, 1.0, lR, lR, lR])


def scale_task_vector(x6, angular_scale=0.20):
    return task_scale_matrix(angular_scale) @ np.asarray(x6, dtype=float).reshape(6)


def cap_vector_norm(v, max_norm):
    """Return v with Euclidean norm bounded by max_norm."""
    v = np.asarray(v, dtype=float).copy()
    max_norm = max(float(max_norm), 0.0)
    n = float(np.linalg.norm(v))
    if n > max_norm and n > 1e-15:
        v *= max_norm / n
    return v


def cap_spatial_twist(twist6, vmax, omegamax):
    """Independently bound linear and angular norms of a spatial twist."""
    tw = np.asarray(twist6, dtype=float).reshape(6).copy()
    tw[:3] = cap_vector_norm(tw[:3], vmax)
    tw[3:] = cap_vector_norm(tw[3:], omegamax)
    return tw


def rate_limit_rotation(R_previous, R_target, max_omega, dt):
    """Move R_previous toward R_target by at most max_omega*dt on SO(3).

    Returns
    -------
    R_next : (3,3)
    omega_step : (3,)
        Spatial rotation-vector rate associated with the accepted reference
        step.  This is a reference-generator quantity, not a robot command.
    """
    Rp = np.asarray(R_previous, dtype=float).reshape(3, 3)
    Rt = np.asarray(R_target, dtype=float).reshape(3, 3)
    h = max(float(dt), 0.0)
    if h <= 1e-12:
        return Rp.copy(), np.zeros(3)

    dphi = so3_log(Rt @ Rp.T)
    dphi = cap_vector_norm(dphi, max(float(max_omega), 0.0) * h)
    return so3_exp(dphi) @ Rp, dphi / h
