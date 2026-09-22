"""Standalone unit tests for the planning stack (no main sim needed).

Run:
    python src/planning/test_planning_standalone.py

Tests
-----
1. RRT-Connect on a synthetic 2-DOF C-space with a wall obstacle
   (forces a detour; validates tree growth, path validity, shortcut).
2. ProgressTrajectory: endpoints, continuity, monotone arc length,
   dwell semantics (clamping), degenerate paths.
3. Real-geometry test: shadow world with the Kuka + the 3 scene tables,
   IK to hover/grasp points at table T1, RRT-Connect for the segment
   chain carry -> hover -> grasp -> carry.  Asserts every densified
   configuration is inside joint limits and collision-free, and reports
   planning times.
"""

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rrt_connect_joint import RRTConnectJointPlanner, shortcut_path
from time_param import ProgressTrajectory, SampledPath3


def check(name, cond, extra=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name} {extra}")
    if not cond:
        raise AssertionError(name)


# ======================================================================
# 1. Synthetic 2-DOF RRT-Connect test
# ======================================================================

def test_rrt_synthetic():
    print("\n== Test 1: RRT-Connect, synthetic 2-DOF wall ==")
    lo = np.array([-1.0, -1.0])
    hi = np.array([1.0, 1.0])

    # Wall at x in [-0.1, 0.1] with a gap at y > 0.7
    def collide(q):
        return (-0.1 <= q[0] <= 0.1) and (q[1] < 0.7)

    planner = RRTConnectJointPlanner(lo, hi, collide,
                                     step_size=0.15,
                                     edge_resolution=0.03,
                                     max_iters=5000, seed=7)
    q_start = np.array([-0.8, -0.5])
    q_goal  = np.array([0.8, -0.5])
    t0 = time.perf_counter()
    path = planner.plan(q_start, q_goal)
    dt_ms = (time.perf_counter() - t0) * 1e3

    check("path found", path is not None,
          f"({dt_ms:.1f} ms, {planner.last_iters} iters)")
    check("starts at q_start", np.allclose(path[0], q_start))
    check("ends at q_goal", np.allclose(path[-1], q_goal))

    def densify_ok(pth):
        for a, b in zip(pth[:-1], pth[1:]):
            n = max(2, int(np.linalg.norm(b - a) / 0.01))
            for i in range(n + 1):
                q = a + (b - a) * i / n
                if collide(q):
                    return False
                if not planner.within_limits(q):
                    return False
        return True

    check("densified path collision-free & within limits", densify_ok(path))
    # Path must detour through the gap (some waypoint with y > 0.6)
    check("path detours through the gap",
          max(q[1] for q in path) > 0.6)

    short = shortcut_path(path, planner.edge_free, planner.rng)
    check("shortcut path still valid", densify_ok(short),
          f"({len(path)} -> {len(short)} waypoints)")

    # Blocked case: no gap at all -> must return None
    def collide_wall(q):
        return -0.1 <= q[0] <= 0.1

    blocked = RRTConnectJointPlanner(lo, hi, collide_wall,
                                     step_size=0.15, edge_resolution=0.03,
                                     max_iters=300, seed=7)
    check("fully blocked wall returns None",
          blocked.plan(q_start, q_goal) is None,
          f"(failure={blocked.last_failure})")

    # Endpoint in collision -> None with reason
    check("colliding start returns None",
          planner.plan(np.array([0.0, 0.0]), q_goal) is None
          and planner.last_failure == "start")


# ======================================================================
# 2. ProgressTrajectory tests
# ======================================================================

def test_progress_trajectory():
    print("\n== Test 2: ProgressTrajectory ==")
    wps = [np.array([0.0, 0.0]),
           np.array([1.0, 0.0]),
           np.array([1.0, 2.0])]
    traj = ProgressTrajectory(wps)
    check("q(0) == start", np.allclose(traj.q(0.0), wps[0]))
    check("q(1) == goal", np.allclose(traj.q(1.0), wps[-1]))
    check("length == 3.0", abs(traj.length - 3.0) < 1e-12)
    # Arc-length parameterization: first segment (len 1) ends at s=1/3
    check("arc-length knot at s=1/3",
          np.allclose(traj.q(1.0 / 3.0), wps[1], atol=1e-9))

    # Continuity: max step between dense samples ~ length * ds.
    # Steps that straddle a knot are chords across the corner, so they
    # can only be SHORTER than the nominal arc step — never longer.
    s_vals = np.linspace(0, 1, 1001)
    qs = np.array([traj.q(s) for s in s_vals])
    steps = np.linalg.norm(np.diff(qs, axis=0), axis=1)
    step_nom = traj.length / 1000.0
    check("no jump larger than nominal arc step",
          np.all(steps <= step_nom + 1e-9),
          f"(max={steps.max():.5f}, nom={step_nom:.5f})")
    interior = [k for k in range(1000)
                if not np.any((traj._s_knots > s_vals[k] + 1e-12)
                              & (traj._s_knots < s_vals[k + 1] - 1e-12))]
    check("uniform speed |dq/ds| between knots",
          np.allclose(steps[interior], step_nom, atol=1e-9))

    # Clamping (dwell semantics: evaluating at s >= 1 holds the goal)
    check("q(1.5) clamps to goal", np.allclose(traj.q(1.5), wps[-1]))
    check("q(-0.2) clamps to start", np.allclose(traj.q(-0.2), wps[0]))

    # Degenerate: single waypoint / zero length
    t2 = ProgressTrajectory([np.array([0.5, 0.5])])
    check("single-waypoint trajectory constant",
          np.allclose(t2.q(0.3), [0.5, 0.5]))
    t3 = ProgressTrajectory([np.array([1.0, 1.0]), np.array([1.0, 1.0])])
    check("zero-length trajectory constant",
          np.allclose(t3.q(0.7), [1.0, 1.0]))

    # Duplicate waypoints dropped
    t4 = ProgressTrajectory([np.zeros(2), np.zeros(2), np.ones(2)])
    check("duplicate waypoint tolerated",
          np.allclose(t4.q(0.5), [0.5, 0.5], atol=1e-9))

    # SampledPath3
    sp = SampledPath3([0.0, 0.5, 1.0],
                      [[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    check("SampledPath3 interpolates",
          np.allclose(sp.p(0.25), [0.5, 0, 0]))
    check("SampledPath3 clamps", np.allclose(sp.p(2.0), [1, 1, 0]))


# ======================================================================
# 3. Real geometry: Kuka + tables shadow world
# ======================================================================

def test_shadow_world_kuka():
    print("\n== Test 3: shadow world (Kuka + 3 tables, scene geometry) ==")
    import pybullet as p
    from shadow_world import ShadowArmWorld

    # ---- Scene constants (mirror pick_place_cycle.py) ----
    CX, CY   = 3.0, 0.0
    R_CIRCLE = 2.2
    R_TABLE  = 1.3
    TABLE_Z_BASE  = -0.3
    TABLE_SURFACE = TABLE_Z_BASE + 0.625
    OBJECT_HALF   = 0.025
    TABLE_ANGLES  = [-math.pi / 2, math.pi / 6, 5 * math.pi / 6]
    tables = []
    for ang in TABLE_ANGLES:
        cx = CX + R_TABLE * math.cos(ang)
        cy = CY + R_TABLE * math.sin(ang)
        tables.append(([cx, cy, TABLE_Z_BASE], ang + math.pi / 2))

    q_lower = np.array([-2.9671, -2.0944, -2.9671, -2.0944,
                        -2.9671, -2.0944, -3.0543])
    q_upper = -q_lower.copy()
    q_upper[1] = 2.0944; q_upper[3] = 2.0944; q_upper[5] = 2.0944
    q_upper[0] = 2.9671; q_upper[2] = 2.9671; q_upper[4] = 2.9671
    q_upper[6] = 3.0543
    rp = np.array([0, math.pi / 6, 0, -math.pi / 3, 0, math.pi / 4, 0])

    world = ShadowArmWorld(
        sim=p, arm_urdf="kuka_iiwa/model.urdf",
        joint_indices=list(range(7)), ee_index=6,
        q_lower=q_lower, q_upper=q_upper,
        tables=tables, default_margin=0.02)

    # ---- Nominal base pose near T1's item ----
    # Item T1 (corner of outward edge), same formula as the main script
    ang = TABLE_ANGLES[0]
    yaw_t = ang + math.pi / 2
    cx = CX + R_TABLE * math.cos(ang)
    cy = CY + R_TABLE * math.sin(ang)
    lx, ly = 0.75 - 0.05, -(0.50 - 0.05)
    item = np.array([cx + lx * math.cos(yaw_t) - ly * math.sin(yaw_t),
                     cy + lx * math.sin(yaw_t) + ly * math.cos(yaw_t),
                     TABLE_SURFACE + OBJECT_HALF])
    theta_item = math.atan2(item[1] - CY, item[0] - CX)

    # Kuka base on the circle at the item's angle, z as in the main sim
    base_pos = [CX + R_CIRCLE * math.cos(theta_item),
                CY + R_CIRCLE * math.sin(theta_item),
                0.30]
    base_orn = p.getQuaternionFromEuler([0, 0, theta_item + math.pi / 2])
    world.set_base_pose(base_pos, base_orn)

    d_item = np.linalg.norm(item[:2] - np.array(base_pos[:2]))
    print(f"  base at theta={theta_item:.2f}, XY dist to item ="
          f" {d_item:.3f} m")

    # ---- Carry config must be collision-free ----
    check("carry config collision-free (margin 2 cm)",
          not world.in_collision(rp, margin=0.02),
          f"(clearance={world.min_clearance(rp):.3f} m)")

    # ---- IK to hover and grasp points ----
    # Grasp approach z: +5 cm above the item centre.  At the old +2 cm
    # the wrist links (L4-L6) penetrate the table's collision volume by
    # ~11 mm (measured), so the planned approach stops at +5 cm where
    # clearance is >= 19 mm; the grasp constraint still pins the cube
    # 2 cm below the EE exactly as before.
    hover = np.array([item[0], item[1], TABLE_SURFACE + OBJECT_HALF + 0.15])
    grasp = item + np.array([0.0, 0.0, 0.05])

    t0 = time.perf_counter()
    q_hover = world.ik(hover, rest_q=rp, collision_margin=0.02)
    q_grasp = world.ik(grasp, rest_q=q_hover if q_hover is not None else rp,
                       collision_margin=0.012)
    t_ik = (time.perf_counter() - t0) * 1e3
    check("IK hover solved", q_hover is not None, f"({t_ik:.0f} ms both)")
    check("IK grasp solved", q_grasp is not None)
    check("IK hover FK error < 1 cm",
          np.linalg.norm(world.fk_world(q_hover) - hover) < 0.01,
          f"(err={np.linalg.norm(world.fk_world(q_hover)-hover)*1000:.1f} mm)")
    check("IK grasp FK error < 1 cm",
          np.linalg.norm(world.fk_world(q_grasp) - grasp) < 0.01,
          f"(err={np.linalg.norm(world.fk_world(q_grasp)-grasp)*1000:.1f} mm)")
    print(f"  grasp config clearance to tables:"
          f" {world.min_clearance(q_grasp)*1000:.1f} mm")

    # ---- Plan the 3 pick segments ----
    MARGIN_TRANSFER = 0.02   # carry <-> hover
    MARGIN_ACT      = 0.01   # hover <-> grasp (EE works close to surface)

    segs = [("carry->hover", rp,      q_hover, MARGIN_TRANSFER),
            ("hover->grasp", q_hover, q_grasp, MARGIN_ACT),
            ("grasp->carry", q_grasp, rp,      MARGIN_ACT)]
    for name, qa, qb, margin in segs:
        planner = RRTConnectJointPlanner(
            q_lower, q_upper, world.make_collision_fn(margin),
            step_size=0.25, edge_resolution=0.06, max_iters=3000, seed=3)
        t0 = time.perf_counter()
        path = planner.plan(qa, qb)
        dt_ms = (time.perf_counter() - t0) * 1e3
        check(f"plan {name}", path is not None,
              f"({dt_ms:.0f} ms, {len(path) if path else 0} wps)")
        path = shortcut_path(path, planner.edge_free, planner.rng)

        traj = ProgressTrajectory(path)
        check(f"{name}: q(0)/q(1) endpoints",
              np.allclose(traj.q(0), qa) and np.allclose(traj.q(1), qb))

        # Densify and validate every config
        ok = True
        worst = 1.0
        for s in np.linspace(0, 1, 120):
            q = traj.q(s)
            if not (np.all(q >= q_lower - 1e-9)
                    and np.all(q <= q_upper + 1e-9)):
                ok = False
                break
            if world.in_collision(q, margin=margin * 0.99):
                ok = False
                break
            worst = min(worst, world.min_clearance(q))
        check(f"{name}: 120 dense configs valid", ok,
              f"(min clearance={worst*1000:.0f} mm)")

        # FK image path for execution
        sp = SampledPath3.from_trajectory(traj, world.fk_local, n=40)
        check(f"{name}: FK path samples finite",
              np.all(np.isfinite(sp.points)))

    world.disconnect()


if __name__ == "__main__":
    test_rrt_synthetic()
    test_progress_trajectory()
    test_shadow_world_kuka()
    print("\nAll planning tests PASSED.")
