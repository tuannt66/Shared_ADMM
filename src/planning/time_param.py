"""Progress parameterization of discrete planned paths.

Turns the raw waypoint list from RRT-Connect into a continuous function
``q(s)`` with s in [0, 1], where s is a *progress* variable — NOT wall
clock time.  The executor integrates ds/dt itself (and may hold s
frozen to dwell), so the trajectory never assumes an absolute schedule.

Parameterization is proportional to joint-space arc length, so the
joint-space speed ``|dq/ds|`` is constant along the path (linear
interpolation between waypoints).
"""

import os
import sys
import numpy as np

_SHARED = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'shared'))
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)
from task_space_6d import interpolate_rotation


class ProgressTrajectory:
    """Piecewise-linear q(s), s in [0, 1], arc-length parameterized.

    Parameters
    ----------
    waypoints : sequence of array-like (nj,)
        Path waypoints (at least one).  A single waypoint yields a
        constant trajectory.
    """

    def __init__(self, waypoints):
        pts = [np.asarray(w, dtype=float) for w in waypoints]
        if len(pts) == 0:
            raise ValueError("ProgressTrajectory needs >= 1 waypoint")
        if len(pts) == 1:
            pts = [pts[0], pts[0].copy()]
        self.points = np.array(pts)                       # (m, nj)
        seg_len = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.length = float(seg_len.sum())
        if self.length < 1e-12:
            # Degenerate (start == goal): constant trajectory
            self._s_knots = np.array([0.0, 1.0])
            self.points = self.points[[0, -1]]
        else:
            cum = np.concatenate([[0.0], np.cumsum(seg_len)])
            # Drop zero-length segments (duplicate waypoints)
            keep = np.concatenate([[True], seg_len > 1e-12])
            self.points = self.points[keep]
            cum = cum[keep]
            self._s_knots = cum / cum[-1]

    def q(self, s):
        """Evaluate the trajectory at progress s (clamped to [0, 1])."""
        s = float(np.clip(s, 0.0, 1.0))
        idx = int(np.searchsorted(self._s_knots, s, side="right")) - 1
        idx = max(0, min(idx, len(self._s_knots) - 2))
        s0, s1 = self._s_knots[idx], self._s_knots[idx + 1]
        w = 0.0 if s1 <= s0 else (s - s0) / (s1 - s0)
        return (1.0 - w) * self.points[idx] + w * self.points[idx + 1]

    @property
    def start(self):
        return self.points[0].copy()

    @property
    def goal(self):
        return self.points[-1].copy()

    def sample(self, n):
        """Return (s_values, configs) at n uniformly spaced s in [0, 1]."""
        s_vals = np.linspace(0.0, 1.0, n)
        return s_vals, np.array([self.q(s) for s in s_vals])


class SampledPath3:
    """3-D point path stored as samples over s in [0, 1], with linear
    interpolation.  Used to cache the FK image of a joint trajectory so
    the 240 Hz loop never calls FK."""

    def __init__(self, s_values, points):
        self.s_values = np.asarray(s_values, dtype=float)
        self.points   = np.asarray(points, dtype=float)     # (m, 3)
        assert self.points.shape[0] == self.s_values.shape[0]
        assert np.all(np.diff(self.s_values) >= 0)

    def p(self, s):
        s = float(np.clip(s, 0.0, 1.0))
        return np.array([
            np.interp(s, self.s_values, self.points[:, k])
            for k in range(3)
        ])

    @classmethod
    def from_trajectory(cls, traj, fk_fn, n=60):
        """Sample fk_fn(q(s)) at n uniform s values."""
        s_vals, configs = traj.sample(n)
        pts = np.array([fk_fn(q) for q in configs])
        return cls(s_vals, pts)

class SampledPose6:
    """SE(3) path sampled over progress s.

    Positions are linearly interpolated and orientations are interpolated
    geodesically on SO(3).  This lets the existing joint-space RRT path supply
    a physically reachable full 6D end-effector reference.
    """

    def __init__(self, s_values, positions, rotations):
        self.s_values = np.asarray(s_values, dtype=float)
        self.positions = np.asarray(positions, dtype=float)
        self.rotations = np.asarray(rotations, dtype=float)
        if self.positions.shape != (self.s_values.size, 3):
            raise ValueError("positions must be (N,3)")
        if self.rotations.shape != (self.s_values.size, 3, 3):
            raise ValueError("rotations must be (N,3,3)")

    def pose(self, s):
        s = float(np.clip(s, 0.0, 1.0))
        idx = int(np.searchsorted(self.s_values, s, side="right")) - 1
        idx = max(0, min(idx, len(self.s_values)-2))
        s0, s1 = self.s_values[idx], self.s_values[idx+1]
        w = 0.0 if s1 <= s0 else (s-s0)/(s1-s0)
        p = (1.0-w)*self.positions[idx] + w*self.positions[idx+1]
        R = interpolate_rotation(self.rotations[idx], self.rotations[idx+1], w)
        return p, R

    def p(self, s):
        return self.pose(s)[0]

    def R(self, s):
        return self.pose(s)[1]

    @classmethod
    def from_trajectory(cls, traj, fk_pose_fn, n=60):
        s_vals, configs = traj.sample(n)
        poses = [fk_pose_fn(q) for q in configs]
        pos = np.array([x[0] for x in poses])
        rot = np.array([x[1] for x in poses])
        return cls(s_vals, pos, rot)
