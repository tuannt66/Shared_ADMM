"""Control-Barrier-Function-based reference governor for end-effector
reach safety.

Filters a raw, potentially-unreachable Cartesian end-effector reference
into the closest reference that keeps the EE within a safe working
radius of the arm's own base -- the radius beyond which manipulability
degrades toward a kinematic singularity (measured for
kuka_iiwa/model.urdf: ~0.72 m -> manipulability 0.15, ~0.90 m -> ~0.09,
~1.0 m -> 0.02, ~1.26 m (full extension) -> 0).

Formulation
-----------
Barrier:       h(p) = r_safe^2 - ||p - p_base||^2        (h >= 0 == safe)
CBF condition: hdot >= -alpha * h                          (Ames et al.,
               "Control Barrier Functions: Theory and Applications", 2019)
             = -2(p - p_base)^T v >= -alpha * h
             =  2(p - p_base)^T v <= alpha * h

Safety filter (discrete-time reference governor -- Garone, Di Cairano &
Kolmanovsky, "Reference and Command Governors", 2017 -- with the CBF
condition above as its admissible-set constraint, as in Ames et al.):

    v* = argmin_v  ||v - v_nom||^2   s.t.  2(p - p_base)^T v <= alpha*h(p)
    p_next = p + v* * dt

For a single linear inequality this QP has a closed-form solution (the
orthogonal projection of v_nom onto the constraint half-space), used
here directly rather than invoking a general-purpose QP solver.

Because the filter only overrides v_nom when it would violate the
margin, nominal tracking is exactly recovered everywhere the raw
reference is already safe (v* = v_nom, no distortion); intervention is
minimal, continuous, and grows only as the boundary is approached --
never a post-hoc truncation of an already-computed, unreachable point.

Assumption: the arm base's own velocity is neglected in hdot (treated
as quasi-static over one control tick -- ~1.5 mm at 0.35 m/s / 240 Hz,
negligible next to the reach margin being regulated).
"""

import numpy as np


class ReachSafetyCBF:
    """CBF-QP reference governor keeping ``|target - base| <= r_safe``."""

    def __init__(self, r_safe, alpha=2.0):
        """
        Parameters
        ----------
        r_safe : float
            Safe working radius (m) from the arm base.
        alpha : float
            Class-K gain (1/s) bounding how fast the margin ``h`` may
            shrink.  Larger = the filter intervenes later / more
            aggressively right at the boundary; smaller = it backs off
            earlier and keeps a larger working margin.
        """
        self.r_safe = float(r_safe)
        self.alpha  = float(alpha)

    def filter(self, raw_target, prev_target, base_pos, dt):
        """Return the safety-filtered reference for this tick.

        Parameters
        ----------
        raw_target : ndarray (3,)
            Unfiltered reference from the trajectory generator this tick.
        prev_target : ndarray (3,) or None
            Safety-filtered reference from the previous tick (None to
            bootstrap -- returns ``raw_target`` unfiltered).
        base_pos : ndarray (3,)
            Current arm base world position.
        dt : float
            Tick duration (s).

        Returns
        -------
        ndarray (3,)
            Safe reference for this tick.
        """
        raw_target = np.asarray(raw_target, dtype=float)
        if prev_target is None or dt <= 1e-9:
            return raw_target

        p = np.asarray(prev_target, dtype=float)
        base_pos = np.asarray(base_pos, dtype=float)
        v_nom = (raw_target - p) / dt

        disp = p - base_pos
        h = self.r_safe ** 2 - float(disp @ disp)
        a = 2.0 * disp
        b = self.alpha * h

        av = float(a @ v_nom)
        if av <= b:
            v = v_nom                                  # already safe
        else:
            aa = float(a @ a)
            v = v_nom if aa < 1e-12 else \
                v_nom - a * ((av - b) / aa)             # exact QP solution

        return p + v * dt
