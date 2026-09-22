"""3-Table pick-place-while-moving cycle with subsystem-separated CLF-ADMM.

Case study
----------
Three tables arranged in an equilateral triangle.  Each table holds
one item (cube).  The Husky+Kuka robot drives a **circular path**
around the tables and performs ONE rotation — picking and placing
objects **without stopping**.

    Pick(T1) → Drive → Place(T2)+Pick(T2) → Drive → Place(T3)+Pick(T3)
    → Drive → Place(T1) → DONE

The arm pre-reaches toward the target as the robot approaches each
table, grasps/releases while the base is still moving (speed-modulated
by arm tracking error), and retracts as the robot drives away.

Architecture (3 threads)
------------------------
  MCU-Base  (Thread-Base)   — analytic-circle trajectory tracking,
                              with CLF-coupled ADMM corrections
  MCU-Arm   (Thread-Arm)    — DLS velocity IK with ADMM blending
  Coordinator (Thread-Coord) — scalar CLF-budget ADMM loop, state
                               machine, trajectory generation

Usage
-----
    python pick_place_cycle.py
"""

import pybullet as p
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared'))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
import time
import math
import random
import threading
import atexit
import numpy as np
import pybullet_data

# Batch/headless plotting support.  ``--batch-mode`` and ``--skip-plots``
# are intentionally independent of PyBullet ``--no-gui`` so research runs
# can keep the same simulation mode while suppressing blocking plot windows.
_MPL_NONINTERACTIVE = any(flag in sys.argv for flag in
                          ("--batch-mode", "--skip-plots", "--no-gui"))
if _MPL_NONINTERACTIVE:
    import matplotlib
    matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from base_subsystem import BaseSubsystem
from arm_subsystem import ArmSubsystem
from coordinator import (Mailbox, LatestValueChannel, BaseMCU, ArmMCU,
                         DistributedADMMCoordinator)
from step_logger import StepLogger
from task_allocator import (TaskSpaceCapabilityBalancer,
                            FixedTaskSpaceCapabilityMonitor)
from dynamic_disturbance_clean import (
    BaseDynamicsAugmentation,
    MotorRotorAugmentation,
)
from disturbance_compensator import DisturbanceCompensator
from eso_base_observer import ESOBaseObserver
from eso_arm_observer import ESOArmObserver
# from adaptive_speed import AdaptiveSpeedController  # disabled for this phase

# Motion-planning layer (RRT-Connect in joint space + progress-based
# execution) — replaces the event-driven Phase state machine below.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'planning'))
from shadow_world import ShadowArmWorld
from planned_cycle import PlannedCycleExecutor

# ============================================================
# Command-line flags
# ============================================================
# --no-disturbance    : baseline kinematic (no disturbances, no compensation)
# --no-compensation   : disturbances ON, compensation OFF
# (default)           : disturbances ON, compensation ON
# Manual disturbances disabled — PyBullet physics engine provides natural dynamics.
# Observers (ESO + NDOB) remain active to compensate physics-based effects.
ENABLE_DISTURBANCE  = False
ENABLE_COMPENSATION = "--no-compensation" not in sys.argv
BASE_TYPE = "jackal" if "--base=jackal" in sys.argv else "husky"
ARM_TYPE  = "xarm6"  if "--arm=xarm6"   in sys.argv else "kuka"
NO_GUI    = "--no-gui" in sys.argv
QUICK_MODE = "--quick" in sys.argv
BATCH_MODE = "--batch-mode" in sys.argv
SKIP_PLOTS = BATCH_MODE or ("--skip-plots" in sys.argv)

# V7 supports distributed CLF-ADMM and a centralized hard-CLF QP baseline.
# --clf-rate=X        : exponential CLF decay rate c (default 1.0)
# --clf-relaxation=X  : fixed CLF relaxation offset (default 0.0)
# --clf-slack-penalty=X: quadratic penalty p_delta for online slack delta
#                        (default 200.0; larger => closer to hard CLF)
def _float_flag(prefix, default):
    arg = next((a for a in sys.argv if a.startswith(prefix)), None)
    return float(arg.split("=", 1)[1]) if arg and "=" in arg else float(default)

CLF_RATE = _float_flag("--clf-rate=", 1.0)
CLF_RELAXATION = _float_flag("--clf-relaxation=", 0.0)
CLF_SLACK_PENALTY = _float_flag("--clf-slack-penalty=", 200.0)

# V10.7 nominal-first practical stability supervisor.  Default behavior
# bypasses the coordination backend when the EE tracking error is inside a
# small accepted tube OR when the independent nominal controllers already
# satisfy dV_nom/dt < 0.  ``--always-enforce-clf`` reproduces the V10.5
# always-on exponential-CLF behavior for ablation.
STABILITY_SUPERVISOR = "--always-enforce-clf" not in sys.argv
SUPERVISOR_PHI_THRESHOLD = _float_flag("--supervisor-phi-threshold=", 0.0)
SUPERVISOR_RECOVERY_RATE = _float_flag("--supervisor-recovery-rate=", 0.10)
SUPERVISOR_ERROR_TOLERANCE = _float_flag("--supervisor-error-tol=", 0.005)
if SUPERVISOR_ERROR_TOLERANCE < 0.0:
    raise ValueError("--supervisor-error-tol must be >= 0")

# Base-controller tuning overrides used by the Level-1 validation sweeps.
# They leave the CLF-ADMM coupling itself unchanged.
# --base-alpha=X       : base local proportional-reference coefficient
# --base-track-scale=X : multiplier on the EE-contribution tracking weight
# --base-path-weight=X : explicit base path-tracking weight in the local QP
_base_alpha_arg = next((a for a in sys.argv if a.startswith("--base-alpha=")), None)
BASE_ALPHA_OVERRIDE = (float(_base_alpha_arg.split("=", 1)[1])
                       if _base_alpha_arg else None)
BASE_TRACK_SCALE = _float_flag("--base-track-scale=", 1.0)
BASE_PATH_WEIGHT = _float_flag("--base-path-weight=", 1.0)

# Explicit mobile-base path follower.  ``kanayama`` is the research default;
# ``geometric`` keeps the previous arc/look-ahead controller for ablation.
_path_law_arg = next((a for a in sys.argv if a.startswith("--base-path-law=")), None)
BASE_PATH_LAW = (_path_law_arg.split("=", 1)[1].lower()
                 if _path_law_arg else "kanayama")
if BASE_PATH_LAW not in ("kanayama", "geometric"):
    raise ValueError("--base-path-law must be kanayama or geometric")
BASE_PATH_KX = _float_flag("--base-path-kx=", 1.0)
BASE_PATH_KY = _float_flag("--base-path-ky=", 2.0)
BASE_PATH_KTHETA = _float_flag("--base-path-ktheta=", 2.0)
if min(BASE_PATH_KX, BASE_PATH_KY, BASE_PATH_KTHETA) < 0.0:
    raise ValueError("base path gains must be nonnegative")

# Experimental general-purpose base-speed scheduler.  It is deliberately
# independent of alpha and of phase names.  Disable with
# ``--legacy-phase-speed`` for an A/B baseline run.
ADAPTIVE_BASE_SPEED = "--legacy-phase-speed" not in sys.argv
# Reverse-recovery experiment.  Default: reverse is allowed in both the
# path follower and the local base QP.  Use --forward-only-base to reproduce
# the previous forward-only feasible set.
ALLOW_BASE_REVERSE = "--forward-only-base" not in sys.argv
SPEED_A_LAT_MAX = _float_flag("--speed-a-lat-max=", 0.05)
SPEED_A_DEC_MAX = _float_flag("--speed-a-dec-max=", 0.35)
if SPEED_A_LAT_MAX <= 0.0 or SPEED_A_DEC_MAX <= 0.0:
    raise ValueError("adaptive speed acceleration limits must be > 0")

if BASE_ALPHA_OVERRIDE is not None and BASE_ALPHA_OVERRIDE <= 0.0:
    raise ValueError("--base-alpha must be > 0")
if BASE_TRACK_SCALE <= 0.0:
    raise ValueError("--base-track-scale must be > 0")
if BASE_PATH_WEIGHT < 0.0:
    raise ValueError("--base-path-weight must be >= 0")

# --admm-max-iter=N : force a constant ADMM iteration budget in every phase.
#                     If omitted, the phase-dependent 5/6/8 budget is used.
_admm_iter_arg = next((a for a in sys.argv if a.startswith("--admm-max-iter=")), None)
ADMM_MAX_ITER_OVERRIDE = int(_admm_iter_arg.split("=", 1)[1]) if _admm_iter_arg else None
if ADMM_MAX_ITER_OVERRIDE is not None and ADMM_MAX_ITER_OVERRIDE < 1:
    raise ValueError("--admm-max-iter must be >= 1")

# --config={baseline,observer-only,full}
#   baseline       : CLF-ADMM only, no ESO observers (Config A in paper)
#   observer-only  : ADMM + ESO running but NOT used for compensation (Config B)
#   full           : ADMM + ESO + feedforward compensation (Config C)
_config_arg = next((a for a in sys.argv if a.startswith("--config=")), None)
ABLATION_CONFIG = _config_arg.split("=")[1] if _config_arg else "full"
assert ABLATION_CONFIG in ("baseline", "observer-only", "full"), \
    f"Invalid --config value: {ABLATION_CONFIG}"

# These derived flags replace the bare ENABLE_COMPENSATION.
# observer-only has ESO running but its output is NOT used for compensation.
ESO_OBSERVING    = (ABLATION_CONFIG in ("observer-only", "full"))
ESO_COMPENSATING = (ABLATION_CONFIG == "full")
# ---- Research comparison mode ---------------------------------------------
# Three fair baselines share the same planner, Layer-1 residual decomposition,
# local objectives, actuator bounds, CLF rate and logging.
#
#   adaptive-admm : adaptive role preference + distributed CLF coordination
#   fixed-admm    : fixed role preference + distributed CLF coordination
#   centralized   : adaptive role preference + one centralized hard-CLF QP
#
# Alpha is a cost preference only; no mode hard-splits task-space velocity.
_control_arg = next((a for a in sys.argv if a.startswith("--control-mode=")), None)
CONTROL_MODE = (_control_arg.split("=", 1)[1].strip().lower()
                if _control_arg else "adaptive-admm")
_CONTROL_ALIASES = {
    "adaptive": "adaptive-admm", "admm": "adaptive-admm",
    "fixed": "fixed-admm", "central": "centralized",
}
CONTROL_MODE = _CONTROL_ALIASES.get(CONTROL_MODE, CONTROL_MODE)
if CONTROL_MODE not in ("adaptive-admm", "fixed-admm", "centralized"):
    raise ValueError("--control-mode must be adaptive-admm, fixed-admm, or centralized")

_fixed_arg = next((a for a in sys.argv if a.startswith("--fixed-alpha")), None)
FIXED_ALPHA_VALUE = (float(_fixed_arg.split("=", 1)[1])
                     if _fixed_arg is not None and "=" in _fixed_arg else 0.50)
if not (0.0 <= FIXED_ALPHA_VALUE <= 1.0):
    raise ValueError("--fixed-alpha must be in [0,1]")
# Backward compatibility: explicitly supplying --fixed-alpha without a
# control-mode selects the fixed-ADMM ablation.
if _fixed_arg is not None and _control_arg is None:
    CONTROL_MODE = "fixed-admm"
ALPHA_MODE = "fixed" if CONTROL_MODE == "fixed-admm" else "adaptive"
COORDINATION_MODE = "centralized" if CONTROL_MODE == "centralized" else "admm"

# Controlled capability-stress scenario.  This scales the arm velocity box in
# BOTH the local QP and the capability metric, without changing task geometry.
_scenario_arg = next((a for a in sys.argv if a.startswith("--research-scenario=")), None)
RESEARCH_SCENARIO = (_scenario_arg.split("=", 1)[1].strip().lower()
                     if _scenario_arg else "nominal")
if RESEARCH_SCENARIO not in ("nominal", "arm-velocity-stress"):
    raise ValueError("--research-scenario must be nominal or arm-velocity-stress")
_default_arm_scale = 0.50 if RESEARCH_SCENARIO == "arm-velocity-stress" else 1.0
ARM_CAPABILITY_SCALE = _float_flag("--arm-capability-scale=", _default_arm_scale)
if not (0.05 <= ARM_CAPABILITY_SCALE <= 1.0):
    raise ValueError("--arm-capability-scale must be in [0.05,1.0]")

# --alpha-eta-low/high are intentionally retired in V2: alpha is a direct
# capacity fraction rather than a thresholded heuristic.
ALPHA_RATE = _float_flag("--alpha-rate=", 5.0)
ALPHA_RESIDUAL_DEADBAND = _float_flag("--alpha-residual-deadband=", 0.01)
# Layer-2 V9 locomotion-deviation operating envelope.
# J_loco = delta_u^T W_p delta_u, with W_p identical to the base local-QP
# path metric.  The default 0.253645 is the pooled P99 of
# 2*base_path_cost from the successful nominal run-1001 research baselines.
BASE_LOCOMOTION_JMAX = _float_flag("--base-locomotion-jmax=", 0.253645)
# V9.1 numerical robustness for the Layer-2 base capability estimator only.
# sigma_floor = max(abs, rel * sigma_max); inverse gain is capped at 1/floor.
BASE_CAPABILITY_PINV_SIGMA_FLOOR_REL = _float_flag(
    "--base-capability-pinv-sigma-floor-rel=", 0.01)
BASE_CAPABILITY_PINV_SIGMA_FLOOR_ABS = _float_flag(
    "--base-capability-pinv-sigma-floor-abs=", 1e-6)
# Accept the V8 flag so old command lines fail gracefully rather than silently
# changing meaning.  It is no longer used by the allocator.
_legacy_loco_budget_arg = next((a for a in sys.argv if a.startswith("--base-locomotion-budget=")), None)
if ALPHA_RATE <= 0.0:
    raise ValueError("--alpha-rate must be > 0")
if ALPHA_RESIDUAL_DEADBAND < 0.0:
    raise ValueError("--alpha-residual-deadband must be >= 0")
if not np.isfinite(BASE_LOCOMOTION_JMAX) or BASE_LOCOMOTION_JMAX <= 0.0:
    raise ValueError("--base-locomotion-jmax must be > 0")
if (not np.isfinite(BASE_CAPABILITY_PINV_SIGMA_FLOOR_REL) or
        BASE_CAPABILITY_PINV_SIGMA_FLOOR_REL < 0.0):
    raise ValueError("--base-capability-pinv-damping-rel must be >= 0")
if (not np.isfinite(BASE_CAPABILITY_PINV_SIGMA_FLOOR_ABS) or
        BASE_CAPABILITY_PINV_SIGMA_FLOOR_ABS < 0.0):
    raise ValueError("--base-capability-pinv-damping-abs must be >= 0")

_comp_tag = ABLATION_CONFIG   # baseline / observer-only / full
_alpha_tag = ALPHA_MODE if ALPHA_MODE == "adaptive" else f"fixed{FIXED_ALPHA_VALUE:.2f}"
_control_tag = CONTROL_MODE.replace("-", "_")
_scenario_tag = ("stress" if RESEARCH_SCENARIO == "arm-velocity-stress" else "nominal")
_loco_tag = f"jloco{BASE_LOCOMOTION_JMAX:.3f}"

# --task-radius=X.X  : override circle radius (default 2.2 m)
# --envelope-mode    : switch CSV prefix to "envelope" and include radius in name
_radius_arg   = next((a for a in sys.argv if a.startswith("--task-radius=")), None)
ENVELOPE_MODE = "--envelope-mode" in sys.argv
_R_override   = float(_radius_arg.split("=")[1]) if _radius_arg else None

# --run-id=N : tag this run for multi-run variance analysis (default: no tag)
_run_id_arg = next((a for a in sys.argv if a.startswith("--run-id=")), None)
RUN_ID = int(_run_id_arg.split("=")[1]) if _run_id_arg else None
_run_tag = f"_run{RUN_ID}" if RUN_ID is not None else ""

if ENVELOPE_MODE:
    _radius_tag = f"_R{_R_override:.1f}" if _R_override is not None else "_R2.2"
    CSV_TAG = (f"envelope_{BASE_TYPE}_{ARM_TYPE}{_radius_tag}_"
               f"{_control_tag}_{_scenario_tag}_{_alpha_tag}_{_loco_tag}")
else:
    CSV_TAG = (f"physics_{_comp_tag}_{BASE_TYPE}_{ARM_TYPE}{_run_tag}_"
               f"{_control_tag}_{_scenario_tag}_{_alpha_tag}_{_loco_tag}")

# --max-steps=N : stop after N coordination steps (smoke-test mode)
_max_steps_arg = next((a for a in sys.argv if a.startswith("--max-steps=")), None)
MAX_STEPS = int(_max_steps_arg.split("=")[1]) if _max_steps_arg else None

# ============================================================
# Connect to PyBullet
# ============================================================
clid = p.connect(p.SHARED_MEMORY)
if clid < 0:
    p.connect(p.DIRECT if NO_GUI else p.GUI)

p.setPhysicsEngineParameter(enableConeFriction=0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
if not NO_GUI:
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

# ============================================================
# Scene geometry — equilateral triangle of tables inside a circle
# ============================================================
CX, CY       = 3.0, 0.0        # circle centre
R_CIRCLE      = _R_override if _R_override is not None else 2.3
R_TABLE       = 1.3             # radius for table centres
ITEM_OUTWARD  = 0.42            # item offset toward the robot path

TABLE_Z_BASE  = -0.3            # table URDF base z
TABLE_SURFACE = TABLE_Z_BASE + 0.625
OBJECT_HALF   = 0.025
GRASP_DIST    = 0.08

# Angles: T1 south, T2 NE, T3 NW  (CCW order)
TABLE_ANGLES = [-math.pi / 2, math.pi / 6, 5 * math.pi / 6]
NUM_TABLES   = 3

# ---- Safe-zone parameters -------------------------------------------------
# Table physical half-sizes (from URDF: 1.5 m × 1.0 m)
TABLE_HALF_LONG  = 0.75                    # half-length along table local X
TABLE_HALF_SHORT = 0.50                    # half-width  along table local Y

# Safe zone: small rectangle on the outward portion of each table
# (the side facing the robot path), sized for the Kuka's reach.
SAFE_HALF_LONG   = 0.20                    # tangential half-span  (total 0.40 m)
SAFE_HALF_SHORT  = 0.10                    # radial half-depth     (total 0.20 m)
SAFE_OUTWARD_OFF = 0.30                    # radial offset of zone centre from
                                           # table centre toward robot path

TABLE_CENTERS   = []
ITEM_POSITIONS  = []      # pick spot = middle of outward edge
TABLE_YAWS      = []
for ang in TABLE_ANGLES:
    cx = CX + R_TABLE * math.cos(ang)
    cy = CY + R_TABLE * math.sin(ang)
    TABLE_CENTERS.append([cx, cy, TABLE_Z_BASE])

    # Pick spot: corner of the outward edge
    yaw = ang + math.pi / 2          # table orientation
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    # Local frame: X along long axis, Y along short axis
    lx = TABLE_HALF_LONG - 0.05      # 5 cm inward from the long-axis edge
    ly = -(TABLE_HALF_SHORT - 0.05)  # 5 cm inward from the outward edge
    wx = cx + lx * cos_y - ly * sin_y
    wy = cy + lx * sin_y + ly * cos_y
    ITEM_POSITIONS.append(np.array([wx, wy, TABLE_SURFACE + OBJECT_HALF]))

    # Table orientation: long axis tangent to the circle
    TABLE_YAWS.append(ang + math.pi / 2)


def random_safe_position(table_idx):
    """Return the CENTRE of the safe zone on the given table."""
    cx, cy = TABLE_CENTERS[table_idx][:2]
    yaw = TABLE_YAWS[table_idx]
    # Zone centre in local frame: (0, -SAFE_OUTWARD_OFF)
    lx = 0.0
    ly = -SAFE_OUTWARD_OFF
    # Rotate to world
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    wx = cx + lx * cos_y - ly * sin_y
    wy = cy + lx * sin_y + ly * cos_y
    return np.array([wx, wy, TABLE_SURFACE + OBJECT_HALF])

# ============================================================
# Load environment
# ============================================================
p.loadURDF("plane.urdf", [0, 0, -0.3])

# Robot — start on the circle near T1
start_angle = TABLE_ANGLES[0] - 0.3   # slightly before T1
start_x = CX + R_CIRCLE * math.cos(start_angle)
start_y = CY + R_CIRCLE * math.sin(start_angle)
start_heading = start_angle + math.pi / 2      # tangent (CCW)

start_quat = p.getQuaternionFromEuler([0, 0, start_heading])

if BASE_TYPE == "jackal":
    _base_urdf    = "urdfs/jackal/jackal.urdf"
    _base_spawn_z = -0.07
    _base_wheels  = [7, 8, 9, 10]
    _base_wheel_r = 0.098
else:  # husky
    _base_urdf    = "husky/husky.urdf"
    _base_spawn_z = -0.31
    _base_wheels  = [2, 3, 4, 5]
    _base_wheel_r = 0.165

husky = p.loadURDF(_base_urdf, [start_x, start_y, _base_spawn_z], start_quat)
for w in _base_wheels:
    p.changeDynamics(husky, w, lateralFriction=1.0,
                     spinningFriction=0.01, rollingFriction=0.01)

if ARM_TYPE == "xarm6":
    kukaId = p.loadURDF("xarm/xarm6_robot.urdf",
                        [start_x, start_y, 0.12], start_quat,
                        useFixedBase=False)
    kukaEndEffectorIndex = 6
    numJoints = 6
    _arm_joint_ids = [1, 2, 3, 4, 5, 6]
    q_lower = np.array([-6.2832, -2.059, -3.927, -6.2832, -1.693, -6.2832])
    q_upper = np.array([ 6.2832,  2.0944, 0.192,  6.2832,  3.1416, 6.2832])
    rp = [0.0, 0.0, -math.pi/2, 0.0, math.pi/2, 0.0]
    _constraint_offset = [0., 0., -0.35]
else:  # kuka
    kukaId = p.loadURDF("kuka_iiwa/model.urdf",
                        [start_x, start_y, 0.12], start_quat,
                        useFixedBase=False)
    kukaEndEffectorIndex = 6
    numJoints = 7
    _arm_joint_ids = list(range(7))
    q_lower = np.array([-2.9671, -2.0944, -2.9671, -2.0944,
                        -2.9671, -2.0944, -3.0543])
    q_upper = np.array([ 2.9671,  2.0944,  2.9671,  2.0944,
                         2.9671,  2.0944,  3.0543])
    rp = [0, math.pi/6, 0, -math.pi/3, 0, math.pi/4, 0]
    _constraint_offset = [0., 0., -0.5]

for ji, angle in zip(_arm_joint_ids, rp):
    p.resetJointState(kukaId, ji, angle)

cid = p.createConstraint(husky, -1, kukaId, -1, p.JOINT_FIXED,
                         [0, 0, 0], [0, 0, 0], _constraint_offset,
                         [0, 0, 0, 1])
p.changeConstraint(cid, maxForce=10000)

# Tables
table_ids = []
for i in range(NUM_TABLES):
    tid = p.loadURDF("table/table.urdf", TABLE_CENTERS[i],
                     p.getQuaternionFromEuler([0, 0, TABLE_YAWS[i]]),
                     useFixedBase=True)
    table_ids.append(tid)
    if not NO_GUI:
        p.addUserDebugText(f"T{i+1}",
                           [TABLE_CENTERS[i][0], TABLE_CENTERS[i][1],
                            TABLE_SURFACE + 0.25],
                           textColorRGB=[1, 1, 0], textSize=2.0)

# Items (cubes) — one per table, at the pick spot on the outward edge
cube_colors = [[1, 0.2, 0.2, 1],   # T1: red
               [0.2, 1, 0.2, 1],   # T2: green
               [0.2, 0.2, 1, 1]]   # T3: blue
cube_ids = []
for i in range(NUM_TABLES):
    cid_cube = p.loadURDF("cube_small.urdf", ITEM_POSITIONS[i].tolist(),
                          p.getQuaternionFromEuler([0, 0, 0]))
    p.changeDynamics(cid_cube, -1, mass=0.1, lateralFriction=1.0)
    p.changeVisualShape(cid_cube, -1, rgbaColor=cube_colors[i])
    cube_ids.append(cid_cube)

# ---- Draw safe-zone rectangles on each table -----------------------
def _draw_safe_zone(table_idx, colour=[0, 0.8, 0]):
    cx, cy = TABLE_CENTERS[table_idx][:2]
    yaw = TABLE_YAWS[table_idx]
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    z = TABLE_SURFACE + 0.002  # just above surface
    # Four corners in local frame (zone is offset outward)
    y_lo = -SAFE_OUTWARD_OFF - SAFE_HALF_SHORT
    y_hi = -SAFE_OUTWARD_OFF + SAFE_HALF_SHORT
    corners_local = [
        (-SAFE_HALF_LONG, y_lo),
        ( SAFE_HALF_LONG, y_lo),
        ( SAFE_HALF_LONG, y_hi),
        (-SAFE_HALF_LONG, y_hi),
    ]
    corners_world = []
    for lx, ly in corners_local:
        wx = cx + lx * cos_y - ly * sin_y
        wy = cy + lx * sin_y + ly * cos_y
        corners_world.append([wx, wy, z])
    for i_c in range(4):
        j_c = (i_c + 1) % 4
        p.addUserDebugLine(corners_world[i_c], corners_world[j_c],
                           colour, 2, 0)

if not NO_GUI:
    for _ti in range(NUM_TABLES):
        _draw_safe_zone(_ti)

# Physics
p.setGravity(0, 0, -10)
p.setRealTimeSimulation(1)
dt = 1.0 / 240.0
DONE_HOLD_SEC = 1.5

# ============================================================
# Helpers
# ============================================================


def get_yaw(quat):
    return p.getEulerFromQuaternion(quat)[2]


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def get_ee_pose():
    ls = p.getLinkState(
        kukaId, kukaEndEffectorIndex, computeForwardKinematics=True)
    R_ee = np.array(
        p.getMatrixFromQuaternion(ls[5])).reshape(3, 3)
    return np.array(ls[4]), R_ee


def get_ee_pos():
    return get_ee_pose()[0]


def get_base_state():
    pos, orn = p.getBasePositionAndOrientation(husky)
    return np.array(pos[:2]), get_yaw(orn)


def get_base_pos_3d():
    pos, _ = p.getBasePositionAndOrientation(kukaId)
    return np.array(pos)


def body_to_world(offset_body):
    kuka_pos, kuka_orn = p.getBasePositionAndOrientation(kukaId)
    R = np.array(p.getMatrixFromQuaternion(kuka_orn)).reshape(3, 3)
    return np.array(kuka_pos) + R @ offset_body


def angle_on_circle(xy):
    """Return the angle θ of a point relative to the circle centre."""
    return math.atan2(xy[1] - CY, xy[0] - CX)


# ============================================================
# Phase 0 — Settle
# ============================================================
print("=" * 60)
_settle_arg = next((a for a in sys.argv if a.startswith("--settle=")), None)
_SETTLE_TIME = float(_settle_arg.split("=")[1]) if _settle_arg else 2.0
print(f"Settling ({_SETTLE_TIME:.1f} s) ...")
if NO_GUI:
    # DIRECT mode has no real-time thread: step physics explicitly.
    for _si in range(int(_SETTLE_TIME / dt)):
        for idx, ji in enumerate(_arm_joint_ids):
            p.setJointMotorControl2(kukaId, ji, p.POSITION_CONTROL,
                                    targetPosition=rp[idx], force=500)
        p.stepSimulation()
else:
    t0_settle = time.time()
    while time.time() - t0_settle < _SETTLE_TIME:
        for idx, ji in enumerate(_arm_joint_ids):
            p.setJointMotorControl2(kukaId, ji, p.POSITION_CONTROL,
                                    targetPosition=rp[idx], force=500)
        time.sleep(dt)

# Release wheel default motors so VELOCITY_CONTROL commands take over cleanly
for _w in _base_wheels:
    p.setJointMotorControl2(husky, _w, p.VELOCITY_CONTROL,
                            targetVelocity=0, force=0)

# Measure geometry
ee_init = get_ee_pos()
husky_xy_init, yaw_init = get_base_state()
kuka_pos_init = get_base_pos_3d()
kuka_orn_init = p.getBasePositionAndOrientation(kukaId)[1]
R_init = np.array(p.getMatrixFromQuaternion(kuka_orn_init)).reshape(3, 3)
carry_offset_body = R_init.T @ (ee_init - kuka_pos_init)

# Body-frame XY offset from husky to EE at rest
offset_husky_to_ee_xy = np.array([
    ee_init[0] - husky_xy_init[0],
    ee_init[1] - husky_xy_init[1],
])
# Rotate into body frame at current heading
cy0, sy0 = math.cos(yaw_init), math.sin(yaw_init)
offset_body_xy = np.array([
    cy0 * offset_husky_to_ee_xy[0] + sy0 * offset_husky_to_ee_xy[1],
    -sy0 * offset_husky_to_ee_xy[0] + cy0 * offset_husky_to_ee_xy[1],
])

print(f"  EE start:     {[round(x,3) for x in ee_init]}")
print(f"  Husky XY:     {[round(x,3) for x in husky_xy_init]}")
print(f"  Heading:      {math.degrees(yaw_init):.1f} deg")
print(f"  Carry body:   {[round(x,3) for x in carry_offset_body]}")
print(f"  Offset body:  {[round(x,3) for x in offset_body_xy]}")

# ============================================================
# Analytic circular base trajectory (ACTIVE geometry)
# ============================================================
theta_start = TABLE_ANGLES[0]  # start angle near T1; visualization seed only

# The active PlannedCycleExecutor evaluates the base reference directly from
#   x_d(theta) = CX + R_CIRCLE*cos(theta)
#   y_d(theta) = CY + R_CIRCLE*sin(theta)
#   psi_d(theta) = theta + pi/2
# The dense line samples below are visualization only and never enter control.
_DEBUG_CIRCLE_SAMPLES = 180
for i in range(_DEBUG_CIRCLE_SAMPLES):
    th0 = theta_start + 2.0 * math.pi * i / _DEBUG_CIRCLE_SAMPLES
    th1 = theta_start + 2.0 * math.pi * (i + 1) / _DEBUG_CIRCLE_SAMPLES
    p0 = [CX + R_CIRCLE * math.cos(th0),
          CY + R_CIRCLE * math.sin(th0), 0.01]
    p1 = [CX + R_CIRCLE * math.cos(th1),
          CY + R_CIRCLE * math.sin(th1), 0.01]
    p.addUserDebugLine(p0, p1, [0.4, 0.4, 0.4], 1, 0)

print(f"  Base path: analytic circle  center=({CX:.2f},{CY:.2f}) "
      f"R={R_CIRCLE:.2f} m  (no active waypoints/polyline)")
for i in range(NUM_TABLES):
    print(f"  T{i+1}: centre={[round(x,2) for x in TABLE_CENTERS[i][:2]]}"
          f"  item={[round(x,2) for x in ITEM_POSITIONS[i][:2]]}")

# ============================================================
# Legacy circular waypoint follower (INACTIVE; rollback/reference only)
# ============================================================
class CircularWaypointFollower:
    """Follows waypoints on a circle (wraps around).

    Provides (v_ff, omega_ff, heading_target) for the base MCU so that
    the base naturally follows the circle.  The ADMM P-control in
    BaseSubsystem.solve() adds positioning corrections on top.
    """

    def __init__(self, waypoints, wp_angles, Kp_heading=2.0,
                 wp_threshold=0.4):
        self.waypoints  = waypoints
        self.wp_angles  = wp_angles
        self.wp_index   = 0
        self.n_wp       = len(waypoints)
        self.Kp_h       = Kp_heading
        self.wp_thresh  = wp_threshold

    def compute(self, base_xy, base_yaw, desired_speed):
        """Return (v_ff, omega_ff, heading_target)."""
        idx = self.wp_index % self.n_wp
        wp  = self.waypoints[idx]

        # Advance if close
        if np.linalg.norm(wp - base_xy) < self.wp_thresh:
            self.wp_index += 1
            idx = self.wp_index % self.n_wp
            wp  = self.waypoints[idx]

        diff = wp - base_xy
        target_heading = math.atan2(diff[1], diff[0])
        heading_err = wrap_angle(target_heading - base_yaw)

        v_ff = desired_speed #* max(0.1, math.cos(heading_err))
        v_ff = float(np.clip(v_ff, 0.0, 0.8))

        # Circular omega feedforward: v / R
        omega_circ = desired_speed / R_CIRCLE
        omega_ff = omega_circ + self.Kp_h * heading_err
        omega_ff = float(np.clip(omega_ff, -1.5, 1.5))

        return v_ff, omega_ff, target_heading

    def distance_to_wp(self, base_xy, wp_idx):
        """Arc-aware distance to a specific WP index."""
        idx = wp_idx % self.n_wp
        return np.linalg.norm(self.waypoints[idx] - base_xy)


# ============================================================
# State machine — pick-place-while-moving cycle
# ============================================================
# Arm reach threshold — object must be closer than this to the
# Kuka base (XY) for the arm to attempt a grasp/place.
ARM_REACH     = 0.85
# Angular thresholds on the circle (radians)
PRE_REACH_RAD = 0.55     # arm starts blending toward target
GRASP_RAD     = 0.25     # close enough to attempt grasp/release
# Minimum base speed — the robot NEVER fully stops
MIN_BASE_SPD  = 0.08
CRUISE_SPEED  = 0.35


# ============================================================
# Task allocator
# ============================================================
# Adaptive alpha is implemented by TaskSpaceCapabilityBalancer in
# src/shared/task_allocator.py.  Alpha is a role-preference index derived from
# directional remaining capability.  It only changes positive correction-cost
# weights; it never directly allocates v_parallel or v_perp.


class Phase:
    TRANSIT     = 0     # driving between tables, arm in carry
    PICK_REACH  = 1     # approaching pick target, arm pre-reaching
    PICK_GRASP  = 2     # close enough, arm descending to grasp
    LIFT        = 3     # raising grasped object to carry height
    PLACE_REACH = 4     # approaching place target, arm pre-reaching
    PLACE_DROP  = 5     # close enough, arm descending to release
    PLACE_LIFT  = 6     # raising EE after release
    DONE        = 7     # cycle finished

_P_NAMES = ["TRANSIT", "PCK_RCH", "PCK_GRP", "LIFT",
            "PLC_RCH", "PLC_DRP", "PLC_LFT", "DONE"]


def _obj_angle(pos_xy):
    """Angle of a world XY point on the circle (relative to centre)."""
    return math.atan2(pos_xy[1] - CY, pos_xy[0] - CX)


class PickPlaceCycleSM:
    """Single-round pick-place-while-moving cycle.

        Pick(T1) → Place(T2)+Pick(T2) → Place(T3)+Pick(T3)
        → Place(T1) → DONE

    The base NEVER stops.  The arm pre-reaches toward the object as
    the robot approaches, grasps/releases while the base is still
    moving, and retracts as the robot drives away.  Speed is modulated
    by arm tracking error through the ADMM pipeline.
    """

    def __init__(self):
        """Pick-place state machine (adaptive speed controller disabled)."""
        self.table_items = {i: [cube_ids[i]] for i in range(NUM_TABLES)}
        self.carried     = None
        self.grasp_cid   = None
        self.place_count = 0

        self.target_table  = 0
        self.phase         = Phase.TRANSIT
        self._t0           = 0.0
        self._place_target = None
        self._phase_start_ee = None
        self.finished      = False

        self.hover_z = TABLE_SURFACE + OBJECT_HALF + 0.15
        self._speed_ema = CRUISE_SPEED

    # ---- public ----

    def update(self, t, ee_pos, base_xy, wp_follower):
        """Return (arm_target, arm_vel, base_speed, phase_name)."""
        fn = {
            Phase.TRANSIT:     self._transit,
            Phase.PICK_REACH:  self._pick_reach,
            Phase.PICK_GRASP:  self._pick_grasp,
            Phase.LIFT:        self._lift,
            Phase.PLACE_REACH: self._place_reach,
            Phase.PLACE_DROP:  self._place_drop,
            Phase.PLACE_LIFT:  self._place_lift,
            Phase.DONE:        self._done,
        }.get(self.phase)
        return fn(t, ee_pos, base_xy, wp_follower)

    @property
    def phase_name(self):
        return _P_NAMES[self.phase]

    # ---- internal ----

    def _go(self, new, t, msg=""):
        print(f"  >>> {_P_NAMES[self.phase]} -> {_P_NAMES[new]}  "
              f"table={self.target_table}  {msg}")
        self.phase = new
        self._t0   = t
        self._phase_start_ee = None

    def _item_pos(self, table_idx):
        """Current world position of the ORIGINAL item on the table."""
        items = self.table_items.get(table_idx, [])
        if items:
            cid = items[0]
            pos, _ = p.getBasePositionAndOrientation(cid)
            return np.array(pos)
        return ITEM_POSITIONS[table_idx].copy()

    def _angular_dist_to(self, base_xy, target_xy):
        """Signed angular distance on the circle (positive = target ahead CCW)."""
        robot_angle  = _obj_angle(base_xy)
        target_angle = _obj_angle(target_xy)
        return wrap_angle(target_angle - robot_angle)

    def _adaptive_speed(self, ee_pos, arm_target, nominal, min_speed=None):
        """Error-aware base speed scheduling.

        Keep the base moving, but slow down when arm error grows so the arm can
        recover during reach/grasp/drop phases. Uses smoothstep shaping + EMA to
        avoid abrupt velocity changes that destabilize ADMM coupling.
        """
        arm_err = float(np.linalg.norm(ee_pos - arm_target))
        # Arm-error-priority profile:
        # 20 mm -> no slowdown, 120 mm -> max slowdown
        e_norm = np.clip((arm_err - 0.02) / 0.10, 0.0, 1.0)
        slow = smoothstep(e_norm)
        scheduled = nominal * (1.0 - 0.85 * slow)
        speed_floor = MIN_BASE_SPD if min_speed is None else float(min_speed)
        scheduled = float(np.clip(scheduled, speed_floor, CRUISE_SPEED))
        self._speed_ema = 0.90 * self._speed_ema + 0.10 * scheduled
        return float(np.clip(self._speed_ema, speed_floor, CRUISE_SPEED))

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _transit(self, t, ee_pos, base_xy, wp_follower):
        """Drive along circle, arm in carry.  Transition when approaching
        the target object/place position."""
        carry = body_to_world(carry_offset_body)

        if self.carried is not None:
            # Need to place → check angular distance to place target
            if self._place_target is None:
                self._place_target = random_safe_position(self.target_table)
                print(f"      place target T{self.target_table+1}: "
                      f"{[round(x,3) for x in self._place_target]}")
            ang_dist = abs(self._angular_dist_to(base_xy,
                                                 self._place_target[:2]))
            if ang_dist < PRE_REACH_RAD:
                self._go(Phase.PLACE_REACH, t)
        else:
            # Need to pick → check angular distance to item
            item_pos = self._item_pos(self.target_table)
            ang_dist = abs(self._angular_dist_to(base_xy, item_pos[:2]))
            if ang_dist < PRE_REACH_RAD:
                self._go(Phase.PICK_REACH, t, "(pre-reach)")

        return carry, np.zeros(3), CRUISE_SPEED, "TRANSIT"

    def _pick_reach(self, t, ee_pos, base_xy, _wpf):
        """Arm blends from carry toward hover above the object.
        Base keeps moving, speed modulated by arm error."""
        item_pos = self._item_pos(self.target_table)
        hover_target = np.array([item_pos[0], item_pos[1], self.hover_z])

        # Blend based on angular proximity (closer = more blend)
        ang_dist = abs(self._angular_dist_to(base_xy, item_pos[:2]))
        blend = smoothstep(1.0 - ang_dist / PRE_REACH_RAD)

        carry = body_to_world(carry_offset_body)
        arm_target = (1.0 - blend) * carry + blend * hover_target

        base_speed = self._adaptive_speed(ee_pos, arm_target,
                                          CRUISE_SPEED * 0.7,
                                          min_speed=0.06)

        # Transition to grasp when close enough and arm is within reach
        kuka_base = get_base_pos_3d()
        reach_dist = np.linalg.norm(item_pos[:2] - kuka_base[:2])
        if ang_dist < GRASP_RAD and reach_dist < ARM_REACH:
            self._go(Phase.PICK_GRASP, t)
        elif ang_dist > PRE_REACH_RAD + 0.2:
            # Passed the object without transitioning — give up, keep driving
            self._go(Phase.TRANSIT, t, "(missed pre-reach window)")

        return arm_target, np.zeros(3), base_speed, "PCK_RCH"

    def _pick_grasp(self, t, ee_pos, base_xy, _wpf):
        """Arm descends to object while base keeps moving (slowly).
        Grasps on EE proximity to object."""
        dt_g = t - self._t0
        item_pos = self._item_pos(self.target_table)
        pick_z   = item_pos[2] + 0.02

        # Descend smoothly
        descend = smoothstep(min(dt_g / 1.2, 1.0))
        arm_target = np.array([
            item_pos[0], item_pos[1],
            self.hover_z * (1 - descend) + pick_z * descend
        ])

        base_speed = self._adaptive_speed(ee_pos, arm_target,
                                          CRUISE_SPEED * 0.25,
                                          min_speed=0.03)

        # Attempt grasp when EE is close
        dist = np.linalg.norm(ee_pos - item_pos)
        if dist < GRASP_DIST and self.carried is None:
            self._do_grasp(self.target_table)
            self._go(Phase.LIFT, t, "(grasped!)")
        elif dt_g > 3.5 and self.carried is None:
            # Timeout — force grasp
            self._do_grasp(self.target_table)
            self._go(Phase.LIFT, t, "(timeout grasp)")

        return arm_target, np.zeros(3), base_speed, "PCK_GRP"

    def _lift(self, t, ee_pos, base_xy, _wpf):
        """Raise object to carry height while base accelerates."""
        dt_l = t - self._t0

        carry = body_to_world(carry_offset_body)
        lift  = smoothstep(min(dt_l / 1.0, 1.0))
        arm_target = (1.0 - lift) * ee_pos + lift * carry

        # Accelerate back to cruise as we lift
        base_speed = MIN_BASE_SPD + (CRUISE_SPEED - MIN_BASE_SPD) * lift

        if dt_l > 1.2:
            self.target_table = (self.target_table + 1) % NUM_TABLES
            self._go(Phase.TRANSIT, t,
                     f"(next: T{self.target_table+1})")

        return arm_target, np.zeros(3), base_speed, "LIFT"

    def _place_reach(self, t, ee_pos, base_xy, _wpf):
        """Arm blends from carry toward hover above the place target.
        Base keeps moving, speed modulated."""
        place_xy = self._place_target[:2]
        hover_target = np.array([place_xy[0], place_xy[1], self.hover_z])

        ang_dist = abs(self._angular_dist_to(base_xy, place_xy))
        blend = smoothstep(1.0 - ang_dist / PRE_REACH_RAD)

        carry = body_to_world(carry_offset_body)
        arm_target = (1.0 - blend) * carry + blend * hover_target

        base_speed = self._adaptive_speed(ee_pos, arm_target,
                                          CRUISE_SPEED * 0.7,
                                          min_speed=0.06)

        # Transition to drop when close and arm is within reach
        kuka_base = get_base_pos_3d()
        reach_dist = np.linalg.norm(place_xy - kuka_base[:2])
        if ang_dist < GRASP_RAD and reach_dist < ARM_REACH:
            self._go(Phase.PLACE_DROP, t)
        elif ang_dist > PRE_REACH_RAD + 0.2:
            self._go(Phase.TRANSIT, t, "(missed place window)")

        return arm_target, np.zeros(3), base_speed, "PLC_RCH"

    def _place_drop(self, t, ee_pos, base_xy, _wpf):
        """Arm descends and releases while base keeps moving."""
        dt_d = t - self._t0
        place_xy = self._place_target[:2]
        place_z  = TABLE_SURFACE + OBJECT_HALF + 0.02

        descend = smoothstep(min(dt_d / 1.2, 1.0))
        arm_target = np.array([
            place_xy[0], place_xy[1],
            self.hover_z * (1 - descend) + place_z * descend
        ])

        base_speed = self._adaptive_speed(ee_pos, arm_target,
                                          CRUISE_SPEED * 0.25,
                                          min_speed=0.03)

        # Release when EE is near surface
        if ee_pos[2] < place_z + 0.04 and self.carried is not None:
            self._do_release(self.target_table)
            self._go(Phase.PLACE_LIFT, t, "(released!)")
        elif dt_d > 3.5:
            if self.carried is not None:
                self._do_release(self.target_table)
            self._go(Phase.PLACE_LIFT, t, "(timeout release)")

        return arm_target, np.zeros(3), base_speed, "PLC_DRP"

    def _place_lift(self, t, ee_pos, base_xy, _wpf):
        """Raise EE after release, then transition to pick or done."""
        dt_pl = t - self._t0

        if self._phase_start_ee is None:
            self._phase_start_ee = ee_pos.copy()

        carry = body_to_world(carry_offset_body)
        lift  = smoothstep(min(dt_pl / 0.8, 1.0))
        arm_target = (1.0 - lift) * self._phase_start_ee + lift * carry

        base_speed = MIN_BASE_SPD + (CRUISE_SPEED - MIN_BASE_SPD) * lift

        if dt_pl > 1.0:
            self._place_target = None
            self.place_count += 1
            if self.place_count >= NUM_TABLES:
                self._go(Phase.DONE, t, "(all items placed!)")
            else:
                self._go(Phase.TRANSIT, t, "(next: pick)")

        return arm_target, np.zeros(3), base_speed, "PLC_LFT"

    def _done(self, t, ee_pos, base_xy, _wpf):
        """Cycle finished — hold carry position."""
        if not self.finished:
            print("\n" + "=" * 40)
            print("  CYCLE COMPLETE — all items rotated!")
            print("=" * 40 + "\n")
            self.finished = True
        arm_target = body_to_world(carry_offset_body)
        return arm_target, np.zeros(3), 0.0, "DONE"

    # ---- grasp / release helpers ----

    def _do_grasp(self, table_idx):
        """Grasp the ORIGINAL item on the table."""
        if self.carried is not None:
            return
        items = self.table_items.get(table_idx, [])
        if not items:
            print(f"  !!! No item on T{table_idx+1}!")
            return
        cid_cube = items[0]
        self.grasp_cid = p.createConstraint(
            kukaId, kukaEndEffectorIndex, cid_cube, -1,
            p.JOINT_FIXED, [0, 0, 0], [0, 0, 0.02], [0, 0, 0])
        p.changeConstraint(self.grasp_cid, maxForce=500)
        self.carried = cid_cube
        items.remove(cid_cube)
        if dyn_dist is not None:
            dyn_dist.set_carrying(True)
        print(f"  *** GRASPED cube {cid_cube} from T{table_idx+1} ***")

    def _do_release(self, table_idx):
        """Release item onto the table."""
        if self.carried is None or self.grasp_cid is None:
            return
        p.removeConstraint(self.grasp_cid)
        self.table_items.setdefault(table_idx, []).append(self.carried)
        if dyn_dist is not None:
            dyn_dist.set_carrying(False)
        print(f"  *** RELEASED cube {self.carried} on T{table_idx+1} ***")
        self.carried = None
        self.grasp_cid = None


# ============================================================
# Instantiate subsystems + threads
# ============================================================
sim_lock   = threading.Lock()
stop_event = threading.Event()

if BASE_TYPE == "jackal":
    _base_gains = dict(max_vel=1.5,
                       w_tracking=4.0, alpha=0.03,
                       reg_v=1.2, reg_omega=0.5,
                       path_weight=BASE_PATH_WEIGHT, wheel_force=100.0)
else:  # husky
    _base_gains = dict(max_vel=1.5,
                       w_tracking=6.0, alpha=0.03,
                       reg_v=1.8, reg_omega=0.8,
                       path_weight=BASE_PATH_WEIGHT, wheel_force=500.0)

if BASE_ALPHA_OVERRIDE is not None:
    _base_gains["alpha"] = float(BASE_ALPHA_OVERRIDE)

base_sub = BaseSubsystem(
    sim=p, husky_id=husky, kuka_id=kukaId,
    yaw_init=yaw_init,
    offset_husky_to_ee_xy=offset_body_xy,
    wheels=_base_wheels, wheel_radius=_base_wheel_r,
    sim_lock=sim_lock,
    allow_reverse=ALLOW_BASE_REVERSE,
    **_base_gains,
)

if ARM_TYPE == "xarm6":
    _arm_kwargs = dict(damping=0.07, w_tracking=90.0,
                       alpha=0.13, ema_smooth=0.0,
                       acc_limit=28.0, joint_force=150.0,
                       q_pref=np.array(rp),
                       orientation_tracking_fraction=0.02,
                       orientation_length=0.25,
                       max_omega_ref=0.50,
                       max_linear_ref=0.35,
                       adaptive_damping_sigma=0.06,
                       adaptive_damping_max=0.14)
else:  # kuka
    _arm_kwargs = dict(damping=0.06, w_tracking=100.0,
                       alpha=0.14, ema_smooth=0.0,
                       acc_limit=30.0, joint_force=200.0,
                       q_pref=np.array(rp),
                       orientation_tracking_fraction=0.02,
                       orientation_length=0.25,
                       max_omega_ref=0.50,
                       max_linear_ref=0.35,
                       adaptive_damping_sigma=0.06,
                       adaptive_damping_max=0.14)

arm_sub = ArmSubsystem(
    sim=p, kuka_id=kukaId,
    ee_index=kukaEndEffectorIndex,
    num_joints=numJoints,
    q_lower=q_lower, q_upper=q_upper,
    joint_indices=_arm_joint_ids,
    sim_lock=sim_lock,
    **_arm_kwargs,
)

# Research stress scenario: reduce the physical/local arm velocity authority.
# The Layer-2 allocator is constructed from this same scaled max_dq below, so
# the controller knows the true capability available in the experiment.
arm_sub.max_dq = float(arm_sub.max_dq) * ARM_CAPABILITY_SCALE

# --- Augmentation layer (effects PyBullet does NOT model natively) ---
# Always enabled — these represent real-world physics, not "disturbances".
DYN_BASE_TAU       = 0.15   # first-order actuator lag (s). Typical 0.1–0.3 for Husky-class.
DYN_BASE_ACC_LIMIT = 1.5    # max base linear acceleration (m/s²)
DYN_BASE_ANG_ACC   = 2.0    # max base angular acceleration (rad/s²)

base_aug = BaseDynamicsAugmentation(
    sim=p, husky_id=husky, kuka_id=kukaId,
    ee_index=kukaEndEffectorIndex,
    base_time_constant=DYN_BASE_TAU,
    base_acc_limit=DYN_BASE_ACC_LIMIT,
    base_ang_acc_limit=DYN_BASE_ANG_ACC,
    wheel_indices=_base_wheels,      # _base_wheels defined above with husky/jackal load
    nominal_wheel_friction=1.0,
    sim_lock=sim_lock,
)
# Wire lag into BaseSubsystem.apply() — Change 3
base_sub.set_dynamic_disturbance(base_aug)

# Motor-rotor gyroscopic augmentation (Interaction 13: PyBullet does not model this)
# Typical Kuka iiwa: rotor inertia ~1e-5 kg·m², gear ratio ~100
rotor_I = [1.0e-5] * numJoints
gear_n  = [100.0]  * numJoints
joint_axes = [[0, 0, 1]] * numJoints   # all revolute about z (URDF convention)

motor_aug = MotorRotorAugmentation(
    sim=p, body_id=kukaId, joint_indices=_arm_joint_ids,
    rotor_inertias=rotor_I, gear_ratios=gear_n,
    joint_axes_body=joint_axes,
    target_body_id=husky,   # gyroscopic torque applied on base
    target_link_idx=-1,
    sim_lock=sim_lock,
)

dyn_dist = None   # legacy variable; kept so any stale references don't crash

# --- Observer-based compensation (ESO) ---
if ESO_OBSERVING:
    ndob = ESOBaseObserver(
        sim=p, husky_id=husky,
        tau_v=DYN_BASE_TAU,
        tau_omega=DYN_BASE_TAU,
        omega_o_v=20.0,
        omega_o_omega=20.0,
        warmup_steps=50,
        sim_lock=sim_lock,
    )
    eso = ESOArmObserver(
        sim=p, kuka_id=kukaId,
        ee_index=kukaEndEffectorIndex,
        omega_o=30.0,               # observer bandwidth (rad/s)
        b0=1.0,                     # nominal input gain
        v_comp_max=0.15,            # max compensation velocity (m/s)
        warmup_steps=100,           # warmup steps
        sim_lock=sim_lock,
    )
    dist_comp = None
else:
    ndob = None
    eso = None
    dist_comp = None

# Print startup flags for verification
print(f"  ABLATION_CONFIG={ABLATION_CONFIG}  "
      f"ESO_OBSERVING={ESO_OBSERVING}  ESO_COMPENSATING={ESO_COMPENSATING}  "
      f"NO_GUI={NO_GUI}  QUICK_MODE={QUICK_MODE}")

# Mailboxes (4 channels: 2 per MCU) + CAN bus (Base <-> Arm)
to_base   = Mailbox()
from_base = Mailbox()
to_arm    = Mailbox()
from_arm  = Mailbox()
can_base_to_arm = LatestValueChannel()   # Base -> Arm: v_cmd/omega_cmd
can_arm_to_base = LatestValueChannel()   # Arm -> Base: ee_pos/base_pos/R_base

# Prime both peer-to-peer state channels.  Each side publishes only its
# locally owned measurement.  The arm sends the EE lever arm expressed in
# the base frame; the base sends its pose and commanded twist.
import time as _peer_time
_prime_base_pos, _prime_yaw = base_sub.get_pose()
can_base_to_arm.publish({
    "position": _prime_base_pos.copy(), "yaw": float(_prime_yaw),
    "v_cmd": 0.0, "omega_cmd": 0.0,
    "timestamp": _peer_time.monotonic(),
})
_prime_ee, _, _ = arm_sub.get_ee_telemetry()
_c, _s = np.cos(_prime_yaw), np.sin(_prime_yaw)
_prime_R = np.array([[_c, -_s, 0.0], [_s, _c, 0.0], [0.0, 0.0, 1.0]])
_prime_r_ee_base = _prime_R.T @ (_prime_ee - _prime_base_pos)
can_arm_to_base.publish({
    "r_ee_base": _prime_r_ee_base.copy(),
    "timestamp": _peer_time.monotonic(),
})

# MCU threads
mcu_base = BaseMCU(base_sub, inbox=to_base, outbox=from_base,
                   stop_event=stop_event,
                   base_to_arm=can_base_to_arm,
                   arm_from_arm=can_arm_to_base)
mcu_arm  = ArmMCU(arm_sub, inbox=to_arm, outbox=from_arm,
                  stop_event=stop_event,
                  base_from_base=can_base_to_arm,
                  arm_to_base=can_arm_to_base)
mcu_base.start()
mcu_arm.start()

# Capability-based task allocator.  In adaptive mode the coordinator updates
# it after J_a and v_res are available, before the actual ADMM local solves.
_base_v_lower = -base_sub.max_vel if ALLOW_BASE_REVERSE else 0.0
_allocator_common = dict(
    max_joint_velocity=arm_sub.max_dq,
    base_u_lower=np.array([_base_v_lower, -base_sub.max_vel], dtype=float),
    base_u_upper=np.array([ base_sub.max_vel,  base_sub.max_vel], dtype=float),
    damping=arm_sub.damping,
    alpha_rate=ALPHA_RATE,
    residual_deadband=ALPHA_RESIDUAL_DEADBAND,
    base_path_weight=base_sub.path_weight,
    base_locomotion_jmax=BASE_LOCOMOTION_JMAX,
    base_pinv_sigma_floor_rel=BASE_CAPABILITY_PINV_SIGMA_FLOOR_REL,
    base_pinv_sigma_floor_abs=BASE_CAPABILITY_PINV_SIGMA_FLOOR_ABS,
)
if ALPHA_MODE == "fixed":
    priority_ctrl = FixedTaskSpaceCapabilityMonitor(
        fixed_alpha=FIXED_ALPHA_VALUE, **_allocator_common)
else:
    priority_ctrl = TaskSpaceCapabilityBalancer(
        alpha_init=0.70, **_allocator_common)

# ADMM coordinator
admm = DistributedADMMCoordinator(
    arm_nj=numJoints,
    to_base=to_base, from_base=from_base,
    to_arm=to_arm, from_arm=from_arm,
    rho=6.0, max_iter=5, tol=1e-3,
    rho_adapt=True, rho_min=1.0, rho_max=50.0,
    # 0.98 was tuned when the loop effectively ran at ~60 Hz (wall
    # clock).  At the full 240 Hz the dual accumulation/decay balance
    # lets u_a grow until it exactly cancels the tracking pull during
    # carry (measured: |u_a| ~ 0.32 m holding a 183 mm EE offset with
    # dq = 0).  A stronger per-tick decay keeps that equilibrium small.
    dual_decay=0.90, task_share_arm=(0.7 if ALPHA_MODE == "adaptive" else FIXED_ALPHA_VALUE),
    clf_rate=CLF_RATE,
    clf_relaxation=CLF_RELAXATION,
    clf_slack_penalty=CLF_SLACK_PENALTY,
    task_allocator=priority_ctrl,
    task_kp_orientation=1.5,
    angular_task_scale=0.25,
    task_vmax=0.45,
    task_omegamax=0.55,
    coordination_mode=COORDINATION_MODE,
    stability_supervisor=STABILITY_SUPERVISOR,
    supervisor_phi_threshold=SUPERVISOR_PHI_THRESHOLD,
    supervisor_recovery_rate=SUPERVISOR_RECOVERY_RATE,
    # The historical 5 mm tube was tuned for XYZ only.  In normalized 6D
    # coordinates it would treat only ~1.1 deg of orientation error as
    # outside the tube and activate correction almost continuously.
    supervisor_error_tolerance=max(SUPERVISOR_ERROR_TOLERANCE, 0.020),
)
admm.reset(ee_init)
ADMM_MAX_ITER_BASE = 5
ADMM_MAX_ITER_REACH = 6
ADMM_MAX_ITER_CRITICAL = 8

# No waypoint follower in the active pipeline.  PlannedCycleExecutor owns
# one analytic circle and evaluates its reference directly from theta.
wp_follower = None

# ============================================================
# Planned-trajectory executor (replaces PickPlaceCycleSM)
# ============================================================
# The old event-driven state machine (Phase / PickPlaceCycleSM._go)
# is kept above for reference & rollback, but the cycle is now driven
# by RRT-Connect-planned joint trajectories executed on a progress
# variable s, with a minimal terminal-pose grasp/release gate (see planned_cycle).
GRASP_TOLERANCE     = 0.04   # m — grasp fires only when |ee - target| below this
RELEASE_TOLERANCE   = 0.025  # m — release needs tighter accuracy (placement DoD)
DWELL_TIMEOUT       = _float_flag("--dwell-timeout=", 3.5)
# V10.4 baseline: no phase-specific base hold is required for grasp/release.
# It can still be enabled explicitly for ablation with --release-base-hold.
RELEASE_BASE_HOLD = ("--release-base-hold" in sys.argv
                     and "--no-release-base-hold" not in sys.argv)
RELEASE_BASE_HOLD_V = max(0.0, _float_flag("--release-base-hold-v=", 0.015))
RELEASE_BASE_HOLD_OMEGA = max(0.0, _float_flag("--release-base-hold-omega=", 0.040))
GRASP_SETTLE_TIME    = _float_flag("--grasp-settle=", 0.0)  # legacy/diagnostic only
RELEASE_SETTLE_TIME  = _float_flag("--release-settle=", 0.0)  # legacy/diagnostic only
RELEASE_CLEARANCE    = _float_flag("--release-clearance=", 0.015)
RELEASE_TILT_MAX_DEG = _float_flag("--release-tilt-max=", 15.0)
RELEASE_ANG_STATIC   = _float_flag("--release-ang-static=", 0.35)
# V10.4 baseline has no release re-aim step, so terminal ACTION reference
# shaping is disabled by default.  It can be enabled for ablation.
ACTION_REF_FILTER    = ("--action-ref-filter" in sys.argv
                        and "--no-action-ref-filter" not in sys.argv)
ACTION_REF_WN        = _float_flag("--action-ref-wn=", 35.0)
# V10.4 action decision: trajectory terminal + EE position tolerance only.
# Object-state quantities remain simulation evaluation metrics, not gates.
PLACEMENT_TOLERANCE = 0.03   # m — DoD: final cube pos vs place target
GRASP_Z_OFF         = 0.09   # m above item/place centre for the action point:
                             # at the old +0.02 the wrist links penetrate the
                             # table's collision volume by ~11 mm (measured),
                             # and the suspended cube needs >= ~0.08 of hang
                             # length to clear the wrist flange geometry

_arm_urdf_name = ("xarm/xarm6_robot.urdf" if ARM_TYPE == "xarm6"
                  else "kuka_iiwa/model.urdf")
shadow_world = ShadowArmWorld(
    sim=p, arm_urdf=_arm_urdf_name,
    joint_indices=_arm_joint_ids, ee_index=kukaEndEffectorIndex,
    q_lower=q_lower, q_upper=q_upper,
    tables=list(zip(TABLE_CENTERS, TABLE_YAWS)),
    default_margin=0.02, sim_lock=sim_lock)

sm = PlannedCycleExecutor(
    sim=p, sim_lock=sim_lock, husky_id=husky, kuka_id=kukaId,
    ee_index=kukaEndEffectorIndex, joint_indices=_arm_joint_ids,
    q_lower=q_lower, q_upper=q_upper, shadow=shadow_world,
    cube_ids=cube_ids, circle=(CX, CY, R_CIRCLE),
    table_surface=TABLE_SURFACE, object_half=OBJECT_HALF,
    place_target_fn=random_safe_position,
    carry_offset_body=carry_offset_body,
    cruise_speed=CRUISE_SPEED, min_speed=MIN_BASE_SPD,
    grasp_tolerance=GRASP_TOLERANCE, release_tolerance=RELEASE_TOLERANCE,
    dwell_timeout=DWELL_TIMEOUT, grasp_settle=GRASP_SETTLE_TIME,
    release_settle=RELEASE_SETTLE_TIME,
    release_clearance=RELEASE_CLEARANCE,
    release_tilt_max_deg=RELEASE_TILT_MAX_DEG,
    release_ang_static=RELEASE_ANG_STATIC,
    grasp_z_off=GRASP_Z_OFF, hover_dz=0.15,
    top_down_grasp=True,
    top_down_local_euler=(math.pi, 0.0, 0.0),
    full6d_orientation_rate=0.50,
    full6d_progress_length=0.20,
    full6d_handover_pos_tol=0.045,
    full6d_handover_rot_tol_deg=12.0,
    full6d_action_rot_tol_deg=8.0,
    preferred_grasp_q=np.array(rp, dtype=float),
    rng_seed=RUN_ID if RUN_ID is not None else 0, dyn_dist=dyn_dist,
    adaptive_speed=ADAPTIVE_BASE_SPEED,
    speed_a_lat_max=SPEED_A_LAT_MAX, speed_a_dec_max=SPEED_A_DEC_MAX,
    allow_reverse=ALLOW_BASE_REVERSE,
    # Measured for kuka_iiwa/model.urdf only (see reach_safety_cbf.py);
    # not yet characterized for xArm6, so left unfiltered there.
    reach_safe_radius=0.90 if ARM_TYPE == "kuka" else None,
    action_ref_filter=ACTION_REF_FILTER, action_ref_wn=ACTION_REF_WN)
admm_logger = StepLogger(p, husky, kukaId, kukaEndEffectorIndex, numJoints,
                         admm, base_sub, arm_sub,
                         priority_ctrl=priority_ctrl,
                         out_path=os.path.join(RESULTS_DIR, f"clf_admm_log_{CSV_TAG}.npz"),
                         decimate=1)

print("=" * 60)
print("3-Table Pick-Place-While-Moving -- V10.8 TOPDOWN FULL 6D")
print("  V10.7+Circle: analytic base circle; V10.7 controller/alpha unchanged")
print("  3 threads: MCU-Base, MCU-Arm, Coordinator")
print("  Direct Base->Arm CAN bus for velocity compensation")
print("  Grasp mode: Stage-3 axis-RRT planning + governed FULL-6D execution (smooth yaw)")
print(f"  Control mode: {CONTROL_MODE}")
if STABILITY_SUPERVISOR:
    print("  Stability supervisor: nominal-first practical tube "
          f"(e_tol={1e3*SUPERVISOR_ERROR_TOLERANCE:g} mm, "
          f"activate outside tube when phi_nom >= {SUPERVISOR_PHI_THRESHOLD:g}, "
          f"recovery rate={SUPERVISOR_RECOVERY_RATE:g})")
else:
    print("  Stability supervisor: OFF (always enforce exponential CLF)")
print(f"  Research scenario: {RESEARCH_SCENARIO}  arm_capacity_scale={ARM_CAPABILITY_SCALE:.2f}")
print(f"  Config: {ABLATION_CONFIG}  "
      f"(ESO_OBSERVING={ESO_OBSERVING}, ESO_COMPENSATING={ESO_COMPENSATING})")
print(f"  Hard CLF rate={CLF_RATE:g} relaxation={CLF_RELAXATION:g}")
print("  FULL-6D governor: omega_ref<=0.50 rad/s, global omega<=0.55 rad/s, "
      "combined pose-progress feedback ON")
print("  Global task: FULL 6D pose/twist; angular scale l_R=0.25 m/rad")
print(f"  Terminal ACTION reference filter: {'ON' if ACTION_REF_FILTER else 'OFF'} "
      f"(critical damping, wn={ACTION_REF_WN:g} rad/s)")
if COORDINATION_MODE == "admm":
    if ADMM_MAX_ITER_OVERRIDE is None:
        print(f"  ADMM iteration budget: phase-dependent "
              f"({ADMM_MAX_ITER_BASE}/{ADMM_MAX_ITER_REACH}/{ADMM_MAX_ITER_CRITICAL})")
    else:
        print(f"  ADMM iteration budget: fixed {ADMM_MAX_ITER_OVERRIDE}")
else:
    print("  Layer-3 backend: CENTRALIZED hard-CLF QP (same local objectives/bounds)")
print(f"  Base tuning: alpha={_base_gains['alpha']:.4g} "
      f"ee_track_scale={BASE_TRACK_SCALE:.3g} "
      f"path_weight={BASE_PATH_WEIGHT:.3g} "
      f"path_law={BASE_PATH_LAW} "
      f"gains=({BASE_PATH_KX:.2g},{BASE_PATH_KY:.2g},{BASE_PATH_KTHETA:.2g})")
print(f"  Augmentation: base_tau={DYN_BASE_TAU}s  "
      f"acc_limit={DYN_BASE_ACC_LIMIT}m/s2  motor_rotor=ON")
print(f"  Min speed: {MIN_BASE_SPD} m/s  |  Cruise: {CRUISE_SPEED} m/s")
print(f"  Base speed scheduler: "
      f"{'adaptive(path curvature + stop distance)' if ADAPTIVE_BASE_SPEED else 'legacy phase speed'}  "
      f"a_lat_max={SPEED_A_LAT_MAX:.3g} m/s^2  "
      f"a_dec_max={SPEED_A_DEC_MAX:.3g} m/s^2")
print(f"  Base reverse recovery: {'ENABLED' if ALLOW_BASE_REVERSE else 'DISABLED (forward-only)'}")
print("  Three-layer research architecture: ENABLED")
print("    L1 Residual geometry after both nominal controllers: v_r -> (v_parallel, v_perp)")
if ALPHA_MODE == "adaptive":
    print("    L2 Directional remaining capability: (gamma_b, gamma_a) -> alpha -> cost weights")
    print("       alpha is preference only; NO hard Cartesian task split")
else:
    print(f"    L2 Fixed role preference: alpha={FIXED_ALPHA_VALUE:.2f}")
print(f"    L3 Nominal-shifted CLF coordination: {COORDINATION_MODE.upper()}")
if ALPHA_MODE == "adaptive":
    print(f"  Alpha allocator: DIRECTIONAL CAPABILITY PREFERENCE  "
          f"rate={ALPHA_RATE:.2f}/s  "
          f"shareable-deadband={ALPHA_RESIDUAL_DEADBAND:.3f} m/s")
    print("  Alpha side-effects: cost weights only (no speed/phase/task-share coupling)")
    print(f"  Minimal action gate V10.4+: terminal trajectory + EE position only  "
          f"(grasp<{GRASP_TOLERANCE*1000:.0f} mm, "
          f"release<{RELEASE_TOLERANCE*1000:.0f} mm, "
          f"timeout={DWELL_TIMEOUT:.1f}s)")
    print("  V10.5 grasp geometry: EE-origin suspension pivot + known grasp offset")
    print("  V10.5 place+pick transfer: release -> short clearance -> next pre-grasp")
    print("  No pre-action settle, EE-speed gate, cube-speed gate, cube-pose gate, "
          "cube-tilt gate, or release re-aim")
    print("  Cube pose/velocity/tilt remain LOGGING/EVALUATION ONLY")
    print(f"  Optional release base hold: {'ON' if RELEASE_BASE_HOLD else 'OFF'}  "
          f"|v|<={RELEASE_BASE_HOLD_V:.3f} m/s  "
          f"|omega|<={RELEASE_BASE_HOLD_OMEGA:.3f} rad/s")
else:
    print(f"  Alpha allocator: FIXED alpha={FIXED_ALPHA_VALUE:.3f}")
print(f"  Pre-reach: {math.degrees(PRE_REACH_RAD):.0f} deg  |  "
      f"Grasp zone: {math.degrees(GRASP_RAD):.0f} deg  |  "
      f"Arm reach: {ARM_REACH} m")
print("  Pick(T1) -> Place+Pick(T2) -> Place+Pick(T3) -> Place(T1) -> DONE")
print("=" * 60)


# ============================================================
# Coordination loop
# ============================================================


def coordination_loop():
    t0         = time.time()
    step_count = 0
    hud_id     = -1
    phase_id   = -1
    prevRef    = None
    prevAct    = None
    TRAIL_DUR  = 30

    # ---- Data logging for post-task plots ----
    log_time       = []
    log_alpha      = []
    log_alpha_raw  = []
    log_carry_dist = []
    log_phase      = []          # now stores planned-segment names
    log_arm_err    = []
    log_tool_axis_err_deg = []  # Stage-3 top-down tilt; yaw is free
    log_orientation_err_deg = []  # V10.8 full SO(3) error
    log_base_err   = []          # |base_xy - circle(theta_ref)| per tick
    log_speed      = []
    log_F_dist     = []
    log_v_lag      = []
    log_rho        = []
    # Observer comparison logging
    log_F_true_x   = []
    log_F_true_y   = []
    log_F_true_z   = []
    log_F_est_x    = []
    log_F_est_y    = []
    log_F_est_z    = []
    log_d_num_x    = []
    log_d_num_y    = []
    # New: ADMM + alpha decomposition + base ESO
    log_primal_res   = []
    log_omega_cmd    = []
    log_base_d_v     = []
    log_base_d_omega = []
    log_base_v_est   = []
    log_alpha_geo    = []
    log_delta_err    = []
    log_delta_manip  = []
    log_delta_admm   = []
    # Arm trajectory logs (reference vs actual EE)
    log_ee_ref_x     = []
    log_ee_ref_y     = []
    log_ee_ref_z     = []
    log_ee_act_x     = []
    log_ee_act_y     = []
    log_ee_act_z     = []
    log_d_num_z    = []
    # Extended metrics
    MAX_LOG_JOINTS    = 7
    log_manipulability = []
    log_admm_iters     = []
    log_base_contrib   = []
    log_arm_contrib    = []
    log_q_joints  = [[] for _ in range(MAX_LOG_JOINTS)]
    log_qd_joints = [[] for _ in range(MAX_LOG_JOINTS)]
    # ESO augmentation logs (for ablation .npz output)
    log_z3_x      = []
    log_z3_y      = []
    log_z3_z      = []
    log_z2_v      = []
    log_z2_omega  = []
    # Startup sanity-check counters
    _motor_aug_print_count = 0
    _lag_print_count = 0

    # ---- Safe-save infrastructure ----
    _npz_safe_name = os.path.join(RESULTS_DIR, f"log_{ABLATION_CONFIG}_{BASE_TYPE}_{ARM_TYPE}.npz")

    def save_log_now(reason="unknown"):
        """Write current state of all log_* arrays to .npz (safe mid-run save)."""
        try:
            np.savez(_npz_safe_name,
                     timestamps=np.array(log_time),
                     tracking_err=np.array(log_arm_err),
                     consensus_residual=np.array(log_primal_res),
                     admm_iterations=np.array(log_admm_iters),
                     z3_x=np.array(log_z3_x),
                     z3_y=np.array(log_z3_y),
                     z3_z=np.array(log_z3_z),
                     z2_v=np.array(log_z2_v),
                     z2_omega=np.array(log_z2_omega),
                     v_lag=np.array(log_v_lag),
                     base_speed=np.array(log_speed),
                     )
            print(f"[save] {reason} -> {_npz_safe_name} ({len(log_time)} steps)")
        except Exception as _save_err:
            print(f"[save] WARNING: could not save ({reason}): {_save_err}")

    atexit.register(save_log_now, reason="atexit")

    while not stop_event.is_set():
        # In DIRECT (--no-gui) mode the loop steps physics itself, so
        # simulation time = step_count * dt (decoupled from wall clock).
        t_now = step_count * dt if NO_GUI else time.time() - t0

        try:
            if not p.isConnected():
                print("\nPhysics disconnected.")
                break

            # ---- Hard timeout: abort after 300 s regardless of phase ----
            if t_now > 300.0:
                print(f"\n[TIMEOUT] t={t_now:.1f}s > 300s — forcing exit. "
                      f"phase={sm.phase_name}  place_count={sm.place_count}")
                save_log_now(reason=f"timeout@{t_now:.0f}s")
                stop_event.set()
                break

            # ---- Change 4: Motor-rotor gyroscopic torque (Interaction 13) ----
            tau_gyro = motor_aug.step()
            if _motor_aug_print_count < 10:
                print(f"[sanity] motor_aug.step()[{_motor_aug_print_count}] "
                      f"= {tau_gyro}")
                _motor_aug_print_count += 1

            # ---- 1. Read state ----
            ee_pos, ee_rot = get_ee_pose()
            base_xy, base_yaw = get_base_state()

            # ---- 2. State machine -> arm target + base speed ----
            arm_target, arm_vel, base_speed_ref, phase_name = \
                sm.update(t_now, ee_pos, base_xy, ee_rot=ee_rot)
            speed_diag = sm.adaptive_speed_info
            action_diag = sm.action_diagnostics
            release_base_hold_active = bool(
                RELEASE_BASE_HOLD
                and phase_name == "PLC_DRP"
                and action_diag.get("action_terminal_dwell", False))
            # Geometric mobile-base reference used by base_guidance().
            # Keep this separate from ``base_pos_desired`` below, which is
            # an EE-space local-task reference for the base CLF subproblem.
            base_path_ref_xy = sm.base_reference_xy
            base_track_err_metric = float(sm.base_track_err)

            # ---- 3. Explicit mobile-base path-tracking reference ----
            # base_guidance() already combines geometric feedforward with
            # path-error/heading feedback.  It is the independent nominal
            # base command u_b^0; there is no end-effector-support term in the
            # base-local optimization.
            if phase_name == "DONE":
                v_path_ref, omega_path_ref, heading_target = 0.0, 0.0, base_yaw
            else:
                v_path_ref, omega_path_ref, heading_target = sm.base_guidance(
                    base_xy, base_yaw, law=BASE_PATH_LAW,
                    kx=BASE_PATH_KX, ky=BASE_PATH_KY,
                    ktheta=BASE_PATH_KTHETA)

            # V10.3 release settling: once the PLACE action reaches terminal
            # dwell, stop asking the path tracker to chase its frozen circle
            # reference and shrink the base's *final-command* feasible set.
            # The coordinator is still free to choose any correction inside
            # this set, and gamma_b is computed from these same bounds.
            if release_base_hold_active:
                v_path_ref = 0.0
                omega_path_ref = 0.0
                base_command_lower = np.array(
                    [-RELEASE_BASE_HOLD_V, -RELEASE_BASE_HOLD_OMEGA], dtype=float)
                base_command_upper = np.array(
                    [ RELEASE_BASE_HOLD_V,  RELEASE_BASE_HOLD_OMEGA], dtype=float)
            else:
                # FULL-6D motion can otherwise let CLF corrections use the
                # entire 1.5 m/s, 1.5 rad/s base box when the arm falls behind.
                # These are hard phase-aware actuator limits seen by BOTH the
                # allocator and local base QP, not post-solve clipping.
                if phase_name in ("PCK_GRP", "PLC_DRP"):
                    _vlo = -0.25 if ALLOW_BASE_REVERSE else 0.0
                    base_command_lower = np.array(
                        [_vlo, -0.45], dtype=float)
                    base_command_upper = np.array(
                        [0.45, 0.45], dtype=float)
                elif phase_name in ("LIFT_VERT", "PLC_LFT_VERT"):
                    _vlo = -0.35 if ALLOW_BASE_REVERSE else 0.0
                    base_command_lower = np.array(
                        [_vlo, -0.60], dtype=float)
                    base_command_upper = np.array(
                        [0.55, 0.60], dtype=float)
                elif phase_name in (
                        "PCK_RCH", "PLC_RCH",
                        "LIFT_RRT", "PLC_LFT_RRT"):
                    _vlo = -0.55 if ALLOW_BASE_REVERSE else 0.0
                    base_command_lower = np.array(
                        [_vlo, -0.85], dtype=float)
                    base_command_upper = np.array(
                        [0.80, 0.85], dtype=float)
                else:  # TRANSIT / DONE
                    _vlo = -0.70 if ALLOW_BASE_REVERSE else 0.0
                    base_command_lower = np.array(
                        [_vlo, -1.00], dtype=float)
                    base_command_upper = np.array(
                        [0.90, 1.00], dtype=float)

            # ---- 4. Separate trajectory references ----
            base_pos_desired = body_to_world(carry_offset_body)

            # Compute arm error here so priority controller can use it
            arm_err = np.linalg.norm(ee_pos - arm_target)

            # ---- 4b. Adaptive role preference ------------------------------
            # Alpha is computed INSIDE admm.step() from directional remaining
            # capability after BOTH nominal local controllers are known.  It
            # biases local correction costs; it does not split task velocity.
            if ALPHA_MODE == "fixed":
                admm.role_alpha = float(np.clip(FIXED_ALPHA_VALUE, 0.0, 1.0))
                admm.task_share_arm = admm.role_alpha  # legacy alias only
                admm.task_share_base = 1.0 - admm.role_alpha
                priority_ctrl.alpha = admm.role_alpha
                priority_ctrl.alpha_raw = admm.role_alpha
                priority_ctrl.alpha_geo = admm.role_alpha
                priority_ctrl.eta_arm = float("nan")

            # Local nominal controllers are independent.  Alpha has exactly
            # one role: relative preference of coordination corrections.

            # ---- 4d. Phase-dependent CLF-ADMM iteration budget ----
            # Alpha only changes the positive base/arm correction weights.
            # The scalar CLF coupling is handled by ADMM through s_b, s_a,
            # z_b, z_a.
            # Only max_iter is adjusted per phase.
            if ADMM_MAX_ITER_OVERRIDE is not None:
                admm.max_iter = ADMM_MAX_ITER_OVERRIDE
            elif phase_name in ("PCK_GRP", "PLC_DRP"):
                admm.max_iter = ADMM_MAX_ITER_CRITICAL
            elif phase_name in ("PCK_RCH", "PLC_RCH", "TRANSIT"):
                admm.max_iter = ADMM_MAX_ITER_REACH
            else:
                admm.max_iter = ADMM_MAX_ITER_BASE

            # ---- 4c. Observer-based compensation (NDOB + ESO) ----
            # Strategy 1: Reference correction — shift arm_target
            #   so ADMM tracks a corrected target (no competition).
            # Strategy 2: Force cancellation — apply opposing force
            #   at the physics level (bypasses controller entirely).
            v_lag_pred = None
            omega_lag_pred = None
            arm_target_corrected = arm_target.copy()

            if eso is not None and phase_name != "DONE":
                if ESO_COMPENSATING:
                    # Option 1: shift reference to anticipate disturbance
                    dp_ref = eso.get_reference_correction(
                        dt, ref_gain=0.5, ref_max=0.03)
                    arm_target_corrected = arm_target + dp_ref

                    # Option 4: direct force cancellation at physics level
                    F_cancel = eso.get_cancellation_force(
                        m_eff=5.0, force_gain=0.8, force_max=15.0)
                    if np.linalg.norm(F_cancel) > 0.01:
                        with sim_lock:
                            ee_pos_now = np.array(p.getLinkState(
                                kukaId, kukaEndEffectorIndex)[4])
                            p.applyExternalForce(
                                kukaId, kukaEndEffectorIndex,
                                F_cancel.tolist(), ee_pos_now.tolist(),
                                p.WORLD_FRAME)

            if ndob is not None and phase_name != "DONE" and ESO_COMPENSATING:
                v_lag_pred, omega_lag_pred = ndob.get_velocity_correction(
                    v_path_ref, omega_path_ref, dt)

            # ---- 5. Modular nominal references + CLF-ADMM coordination ----
            # The planner supplies an arm-local reference independently of the
            # base command.  In TRANSIT this is a hold/carry reference in the
            # nominal arm-base frame; in manipulation segments it follows the
            # planned arm motion.  Alpha is NOT used here to split the task.
            (arm_local_target, arm_local_vel,
             arm_local_rot, arm_local_omega) = sm.arm_local_pose_reference(
                arm_target_corrected, dt, phase_name=phase_name)
            arm_rot_target = sm.orientation_reference_world
            arm_omega_target = sm.angular_velocity_reference_world

            # FULL 6D world pose/twist enters the global residual, allocator
            # and practical CLF supervisor.  The scalar ADMM architecture is
            # unchanged.
            info = admm.step(
                pos_desired=arm_target_corrected.copy(),
                vel_desired=arm_vel.copy(),
                rot_desired=arm_rot_target.copy(),
                omega_desired=arm_omega_target.copy(),
                dt=dt,
                step_count=step_count,
                base_pos_desired=base_pos_desired.copy(),
                base_path_v_ref=v_path_ref,
                base_path_omega_ref=omega_path_ref,
                v_compensation=None,
                arm_local_pos_desired=arm_local_target,
                arm_local_vel_desired=arm_local_vel,
                arm_local_rot_desired=arm_local_rot,
                arm_local_omega_desired=arm_local_omega,
                base_command_lower=base_command_lower,
                base_command_upper=base_command_upper,
            )
            # Release-gate diagnostics come from the task executor.
            # V10.3 additionally logs the local base hold constraint used
            # during terminal PLC_DRP settling.
            info["release_base_hold_active"] = bool(release_base_hold_active)
            info["release_base_hold_v"] = float(RELEASE_BASE_HOLD_V)
            info["release_base_hold_omega"] = float(RELEASE_BASE_HOLD_OMEGA)
            info.update(action_diag)
            info.update({
                "planner_full6d_combined_error_m": float(
                    sm.full6d_tracking_diagnostics["combined_error_m"]),
                "planner_full6d_orientation_error_deg": float(
                    sm.full6d_tracking_diagnostics["orientation_error_deg"]),
                "planner_full6d_orientation_rate_limit": float(
                    sm.full6d_tracking_diagnostics["orientation_rate_limit"]),
                "planner_progress_rate": float(
                    speed_diag.get("progress_rate_filtered", np.nan)),
                "planner_progress_rate_raw": float(
                    speed_diag.get("progress_rate_raw", np.nan)),
                "planner_progress_cart_cap": float(
                    speed_diag.get("progress_rate_cart_cap", np.nan)),
                "planner_progress_joint_cap": float(
                    speed_diag.get("progress_rate_joint_cap", np.nan)),
                "planner_progress_tracking_factor": float(
                    speed_diag.get("progress_tracking_factor", np.nan)),
                "planner_progress_paused": bool(
                    speed_diag.get("progress_paused", False)),
                "planner_progress_dpds_m": float(
                    speed_diag.get("progress_dpds_m", np.nan)),
                "planner_segment_min_duration_s": float(
                    speed_diag.get("segment_min_duration_s", np.nan)),
                "base_progress_decoupled": bool(
                    speed_diag.get("base_progress_decoupled", False)),
                "base_reference_theta": float(
                    speed_diag.get("base_reference_theta", np.nan)),
                "base_reference_speed": float(
                    speed_diag.get("base_reference_speed", np.nan)),
                "base_reference_target_speed": float(
                    speed_diag.get("base_reference_target_speed", np.nan)),
                "base_reference_horizon_theta": float(
                    speed_diag.get("base_reference_horizon_theta", np.nan)),
                "base_reference_horizon_distance_m": float(
                    speed_diag.get("base_reference_horizon_distance_m", np.nan)),
                "base_reference_stop_required": bool(
                    speed_diag.get("base_reference_stop_required", False)),
                "base_reference_lead_m": float(
                    speed_diag.get("base_reference_lead_m", np.nan)),
                "base_reference_phase_cap": float(
                    speed_diag.get("base_reference_phase_cap", np.nan)),
            })
            admm_logger.record(
                step_count, t_now=t_now, phase_name=phase_name,
                pos_desired=arm_target_corrected, vel_desired=arm_vel,
                base_pos_desired=base_pos_desired,
                v_ff=v_path_ref, omega_ff=omega_path_ref,
                heading_target=heading_target, info=info, dt=dt,
                base_path_ref_xy=base_path_ref_xy,
                base_track_err=base_track_err_metric,
            )

            # ---- 5c. Update observers (measure actual vs commanded) ----
            if ndob is not None:
                # ESO reads actual velocity from PyBullet physics directly
                ndob.update(info['v_cmd'], info['omega_cmd'], dt)
            if eso is not None:
                # u = trajectory velocity from state machine
                # The ESO lumps all uncertainty (model mismatch,
                # external forces, kinematic errors) into z3.
                eso.update(dt, u=arm_vel)

            # ---- 5d. Grasp-hold verification bookkeeping ----
            sm.post_tick(t_now, ee_pos)

            # Sanity check: print v_lag for first 1s of movement (Change 3 verification)
            v_cmd_now = info['v_cmd']
            if (_lag_print_count < 15
                    and v_cmd_now > 0.05
                    and base_aug._v_actual < v_cmd_now - 0.005):
                aug_diag = base_aug.get_diagnostics()
                print(f"[sanity] lag step={step_count} "
                      f"v_cmd={aug_diag['v_cmd']:.3f}  "
                      f"v_actual={aug_diag['v_actual']:.3f}  "
                      f"v_lag={aug_diag['v_lag']:.3f}")
                _lag_print_count += 1

            # ---- 6. Diagnostics / HUD ----
            if step_count % 48 == 0:
                diag_arm  = arm_sub.get_diagnostics()
                diag_admm = admm.get_diagnostics()
                print(f"[CLF] step={step_count:6d} V={info['clf_V']:.3e} "
                      f"dV={info['clf_dV']:.3e} hard_bound={info['clf_bound']:.3e} "
                      f"delta={info['clf_slack']:.3e} "
                      f"hard_res={info['clf_hard_residual']:.3e} "
                      f"soft_res={info['clf_residual']:.3e} "
                      f"hard_ok={int(info['clf_hard_feasible'])} "
                      f"soft_ok={int(info['clf_feasible'])} "
                      f"fin_cert={int(info['admm_finite_iter_certified'])} "
                      f"eps_fin={info['admm_finite_error_bound']:.2e} "
                      f"e_inf<={info['clf_error_ultimate_bound']:.3e} "
                      f"corr={info['clf_correction']:.3e}")
                w = diag_arm["manipulability"]

                w_col = ([0, .6, 0] if w > 0.05
                         else [.8, .6, 0] if w > 0.02
                         else [1, 0, 0])

                pc = priority_ctrl
                aug_diag = base_aug.get_diagnostics()
                F_mag = 0.0   # no fake forces; use lag as proxy metric
                v_lag = aug_diag["v_lag"]
                # hud_str = (f"v={info['v_cmd']:.2f}  w={info['omega_cmd']:.2f}  | "
                #            f"err={arm_err*1000:.0f}mm  manip={w:.4f}  "
                #            f"rho={diag_admm['rho']:.1f}  | "
                #            f"a={pc.alpha:.2f}(geo={pc.alpha_geo:.2f} "
                #            f"de={pc.delta_err:.2f} dm={pc.delta_manip:.2f} "
                #            f"dr={pc.delta_admm:.2f})  cd={pc.carry_dist:.3f}")
                # if hud_id >= 0:
                #     p.removeUserDebugItem(hud_id)
                # hud_id = p.addUserDebugText(
                #     hud_str, (ee_pos + [0, 0, 0.35]).tolist(),
                #     textColorRGB=w_col, textSize=1.2)

                # g_str = f" [{sm.carried}]" if sm.carried else ""
                # ph_str = (f"{phase_name}{g_str}  T{sm.target_table+1}  "
                #           f"WP={wp_follower.wp_index % wp_follower.n_wp}")
                # if phase_id >= 0:
                #     p.removeUserDebugItem(phase_id)
                # phase_id = p.addUserDebugText(
                #     ph_str, (ee_pos + [0, 0, 0.25]).tolist(),
                #     textColorRGB=[1, 1, 0], textSize=1.2)

                if step_count % 240 == 0:
                    items_str = "  ".join(
                        f"T{i+1}:{sm.table_items.get(i, [])}"
                        for i in range(NUM_TABLES))
                    print(f"[{phase_name:>8s}] t={t_now:5.1f}s  "
                          f"err={arm_err*1000:4.0f}mm  "
                          f"v={info['v_cmd']:.2f}  "
                          f"vs={speed_diag['v_scheduled']:.2f}  "
                          f"vc={speed_diag['v_curve']:.2f}  "
                          f"w={w:.4f}  rho={diag_admm['rho']:.1f}  "
                          f"a={pc.alpha:.2f} eta={pc.eta_arm:.3f}  "
                          f"F={F_mag:.1f}N vlag={v_lag:.3f}  "
                          f"carry={sm.carried}  {items_str}")

                # ---- Diagnostic every 10 s ----
                if step_count % 2400 == 0 and step_count > 0:
                    print(f"[DIAG 10s] t={t_now:.1f}s  phase={phase_name}  "
                          f"place_count={sm.place_count}  step={step_count}  "
                          f"base_speed={info['v_cmd']:.3f} m/s  "
                          f"target_T{sm.target_table+1}  carried={sm.carried}")

            # ---- 7. Trail lines ----
            if not NO_GUI and prevRef is not None:
                p.addUserDebugLine(prevRef, arm_target.tolist(),
                                   [0, 0, 0.3], 1, TRAIL_DUR)
                p.addUserDebugLine(prevAct, ee_pos.tolist(),
                                   [1, 0, 0], 1, TRAIL_DUR)
            prevRef = arm_target.tolist()
            prevAct = ee_pos.tolist()

            # ---- 8. Data logging ----
            log_time.append(t_now)
            log_alpha.append(priority_ctrl.alpha)
            log_alpha_raw.append(priority_ctrl.alpha_raw)
            log_carry_dist.append(priority_ctrl.carry_dist)
            log_phase.append(phase_name)
            log_arm_err.append(arm_err)
            log_tool_axis_err_deg.append(float(
                info.get('arm_tool_axis_error_deg', float('nan'))))
            log_orientation_err_deg.append(float(
                info.get('orientation_error_deg', float('nan'))))
            log_base_err.append(sm.base_track_err)
            log_speed.append(info['v_cmd'])
            log_rho.append(info['rho'])
            # dyn_dist is None (legacy layer disabled); log base_aug lag instead
            _aug_d = base_aug.get_diagnostics()
            log_F_dist.append(0.0)
            log_v_lag.append(_aug_d["v_lag"])
            log_F_true_x.append(0.0)
            log_F_true_y.append(0.0)
            log_F_true_z.append(0.0)

            if eso is not None:
                z3 = eso.get_estimated_disturbance()
                m_eff = 5.0
                log_F_est_x.append(m_eff * z3[0])
                log_F_est_y.append(m_eff * z3[1])
                log_F_est_z.append(m_eff * z3[2])
                log_d_num_x.append(0.0)
                log_d_num_y.append(0.0)
                log_d_num_z.append(0.0)
                # ESO ablation logs
                log_z3_x.append(float(z3[0]))
                log_z3_y.append(float(z3[1]))
                log_z3_z.append(float(z3[2]))
                if ndob is not None:
                    _z2v, _z2w = ndob.get_estimated_disturbance()
                    log_z2_v.append(float(_z2v))
                    log_z2_omega.append(float(_z2w))
                else:
                    log_z2_v.append(0.0)
                    log_z2_omega.append(0.0)
            else:
                log_F_est_x.append(0.0)
                log_F_est_y.append(0.0)
                log_F_est_z.append(0.0)
                log_d_num_x.append(0.0)
                log_d_num_y.append(0.0)
                log_d_num_z.append(0.0)
                log_z3_x.append(0.0)
                log_z3_y.append(0.0)
                log_z3_z.append(0.0)
                log_z2_v.append(0.0)
                log_z2_omega.append(0.0)

            # ---- New logging ----
            log_primal_res.append(info['primal_residual'])
            log_omega_cmd.append(info['omega_cmd'])
            pc = priority_ctrl
            log_alpha_geo.append(pc.alpha_geo)
            log_delta_err.append(pc.delta_err)
            log_delta_manip.append(pc.delta_manip)
            log_delta_admm.append(pc.delta_admm)
            log_ee_ref_x.append(arm_target[0])
            log_ee_ref_y.append(arm_target[1])
            log_ee_ref_z.append(arm_target[2])
            log_ee_act_x.append(ee_pos[0])
            log_ee_act_y.append(ee_pos[1])
            log_ee_act_z.append(ee_pos[2])
            if ndob is not None:
                bd = ndob.get_diagnostics()
                log_base_d_v.append(bd["d_hat_v"])
                log_base_d_omega.append(bd["d_hat_omega"])
                log_base_v_est.append(bd["z1_v"])
            else:
                log_base_d_v.append(0.0)
                log_base_d_omega.append(0.0)
                log_base_v_est.append(info['v_cmd'])

            # ---- Extended metrics ----
            log_manipulability.append(arm_sub.w_current)
            log_admm_iters.append(info['iterations'])

            # Base full EE twist contribution in the 6D task.
            _e_fwd = np.array([math.cos(base_yaw), math.sin(base_yaw), 0.0])
            _r_ee  = ee_pos - get_base_pos_3d()
            _base_lin = (info['v_cmd'] * _e_fwd
                         + info['omega_cmd'] * np.cross(
                             np.array([0., 0., 1.]), _r_ee))
            _base_twist = np.r_[_base_lin,
                                np.array([0., 0., info['omega_cmd']])]
            log_base_contrib.append(float(np.linalg.norm(_base_twist)))

            # Arm full spatial EE twist contribution: ‖J6·q̇‖
            log_arm_contrib.append(
                float(np.linalg.norm(arm_sub._last_J @ arm_sub._last_qd)))

            # Joint angles and velocities (padded to MAX_LOG_JOINTS)
            _q_now  = list(arm_sub._last_q)  + [0.0] * (MAX_LOG_JOINTS - numJoints)
            _qd_now = list(arm_sub._last_qd) + [0.0] * (MAX_LOG_JOINTS - numJoints)
            for _ji in range(MAX_LOG_JOINTS):
                log_q_joints[_ji].append(_q_now[_ji])
                log_qd_joints[_ji].append(_qd_now[_ji])

            step_count += 1

            # ---- Periodic save every 30 s ----
            if step_count % 7200 == 0:
                save_log_now(reason=f"periodic@step{step_count}")

            # Smoke-test early exit
            if MAX_STEPS is not None and step_count >= MAX_STEPS:
                print(f"[smoke-test] Reached {MAX_STEPS} steps — stopping early.")
                stop_event.set()
                break

            # Quick mode: stop after ~2000 steps (~8s of sim time)
            if QUICK_MODE and step_count >= 2000:
                print(f"[quick] Reached 2000 steps — stopping (quick mode).")
                stop_event.set()
                break

            # Auto-stop shortly after DONE to avoid end-of-sim drift/jitter.
            if sm.finished and (t_now - sm._t0) > DONE_HOLD_SEC:
                print(f"\nReached DONE. Holding for {DONE_HOLD_SEC:.1f}s complete; stopping loop.")
                stop_event.set()
                break
            if NO_GUI:
                # DIRECT mode: the loop is the clock — advance physics
                # one dt per coordination step.
                with sim_lock:
                    p.stepSimulation()
            else:
                time.sleep(dt)

        except Exception as e:
            print(f"\nLoop error: {e}")
            break

    admm_logger.save()

    # ================================================================
    # Definition-of-Done summary (planned-trajectory architecture)
    # ================================================================
    # Thresholds:
    #   ARM_RMS_THRESH  — old event-driven run measured RMS = 61.1 mm
    #     (min 0.0 / max 297.8 / p95 93.2 mm, results/log_physics_full_
    #     husky_kuka_adaptive.csv), so the planned architecture must
    #     stay at or below that scale: 65 mm.
    #   BASE_RMS_THRESH — base is expected to track circle(theta_ref)
    #     within ~5 % of the circle radius: 0.12 m.
    ARM_RMS_THRESH  = 0.065
    BASE_RMS_THRESH = 0.12
    GRASP_HOLD_DEV  = 0.15   # cube must stay this close to the EE (m):
                             # hang length GRASP_Z_OFF (~0.09) + link-frame
                             # vs COM-frame offsets + residual swing
    try:
        rep = sm.report()
        arm_rms  = float(np.sqrt(np.mean(np.square(log_arm_err)))) \
            if log_arm_err else float('nan')
        base_rms = float(np.sqrt(np.mean(np.square(log_base_err)))) \
            if log_base_err else float('nan')
        lines = []
        lines.append("=" * 64)
        lines.append("DEFINITION-OF-DONE SUMMARY")
        lines.append("=" * 64)

        # 1. Grasp success (cube follows EE for 2 s after each grasp)
        g_ok = (len(rep['grasps']) > 0
                and all((not g['dropped']) and g['n'] > 0
                        and g['max_dev'] < GRASP_HOLD_DEV
                        for g in rep['grasps']))
        lines.append(f"[{'PASS' if g_ok else 'FAIL'}] 1. Grasps hold "
                     f"({len(rep['grasps'])} grasps)")
        for g in rep['grasps']:
            lines.append(f"        cube {g['cube']} T{g['table']+1}: "
                         f"{g['n']} samples over 2 s, "
                         f"dev mean={g['mean_dev']*1000:.0f} mm "
                         f"max={g['max_dev']*1000:.0f} mm "
                         f"dropped={g['dropped']}")

        # 2. Placement accuracy
        p_ok = (len(rep['placements']) == NUM_TABLES
                and all(pl['err'] < PLACEMENT_TOLERANCE
                        for pl in rep['placements']))
        lines.append(f"[{'PASS' if p_ok else 'FAIL'}] 2. Placements within "
                     f"{PLACEMENT_TOLERANCE*1000:.0f} mm "
                     f"({len(rep['placements'])}/{NUM_TABLES})")
        for pl in rep['placements']:
            _e1 = (f", 1s-after-release={pl['err_1s']*1000:.1f} mm"
                   if pl.get('err_1s') is not None else "")
            _rr = pl.get('release_cube_residual')
            _rv = pl.get('release_cube_speed')
            _rw = pl.get('release_cube_ang_speed')
            _rt = pl.get('release_cube_tilt_deg')
            _ra = pl.get('release_aim_count')
            _gate = (f", release_resid={_rr*1000:.1f} mm"
                     f", aims={_ra}, cube_v={_rv:.3f} m/s"
                     f", tilt={_rt:.1f} deg, omega={_rw:.3f} rad/s"
                     if (_rr is not None and _rv is not None
                         and _rt is not None and _rw is not None) else "")
            lines.append(f"        cube {pl['cube']} T{pl['table']+1}: "
                         f"err={pl['err']*1000:.1f} mm "
                         f"(xy={pl['err_xy']*1000:.1f} mm{_e1}{_gate})")

        # 3. Arm tracking RMS
        a_ok = arm_rms < ARM_RMS_THRESH
        lines.append(f"[{'PASS' if a_ok else 'FAIL'}] 3. Arm tracking RMS "
                     f"{arm_rms*1000:.1f} mm < {ARM_RMS_THRESH*1000:.0f} mm "
                     f"(old-SM baseline: 61.1 mm)")
        if log_arm_err:
            _ae = np.array(log_arm_err)
            lines.append(f"        arm err min={_ae.min()*1000:.1f} "
                         f"max={_ae.max()*1000:.1f} "
                         f"p95={np.percentile(_ae,95)*1000:.1f} mm")
            # Stage-3 diagnostic split: isolate vertical Cartesian lifting
            # from the following tool-axis-constrained RRT retract.  This is
            # intentionally informational, not an extra pass/fail gate.
            _ph = np.asarray(log_phase, dtype=object)
            _ax = np.asarray(log_tool_axis_err_deg, dtype=float)
            for _name in ("PCK_GRP", "LIFT_VERT", "LIFT_RRT",
                          "PLC_DRP", "PLC_LFT_VERT", "PLC_LFT_RRT"):
                _m = (_ph == _name)
                if not np.any(_m):
                    continue
                _err = _ae[_m]
                _axis = _ax[_m]
                _axis = _axis[np.isfinite(_axis)]
                _axis_txt = (f", axis p95={np.percentile(_axis,95):.1f} deg"
                             if _axis.size else "")
                lines.append(
                    f"        {_name}: arm RMS="
                    f"{np.sqrt(np.mean(_err**2))*1000:.1f} mm, "
                    f"max={np.max(_err)*1000:.1f} mm{_axis_txt}")

        # 4. Base tracking RMS
        b_ok = base_rms < BASE_RMS_THRESH
        lines.append(f"[{'PASS' if b_ok else 'FAIL'}] 4. Base tracking RMS "
                     f"{base_rms*1000:.1f} mm < "
                     f"{BASE_RMS_THRESH*1000:.0f} mm")
        if log_base_err:
            _be = np.array(log_base_err)
            lines.append(f"        base err min={_be.min()*1000:.1f} "
                         f"max={_be.max()*1000:.1f} "
                         f"p95={np.percentile(_be,95)*1000:.1f} mm")

        # 5. No timeout / fallback triggered
        t_ok = (len(rep['timeouts']) == 0
                and len(rep['plan_fallbacks']) == 0)
        lines.append(f"[{'PASS' if t_ok else 'FAIL'}] 5. No dwell timeouts "
                     f"({len(rep['timeouts'])}) / planner fallbacks "
                     f"({len(rep['plan_fallbacks'])})")
        for ev in rep['timeouts']:
            lines.append(f"        TIMEOUT t={ev['t']:.1f}s "
                         f"{ev['segment']} {ev['action']} "
                         f"T{ev['table']+1} err={ev['ee_err']*1000:.0f} mm")
        for ev in rep['plan_fallbacks']:
            lines.append(f"        FALLBACK {ev}")

        # 6. Full cycle completed
        c_ok = sm.finished
        lines.append(f"[{'PASS' if c_ok else 'FAIL'}] 6. Full cycle "
                     f"completed (place_count={sm.place_count}/"
                     f"{NUM_TABLES})")
        if rep['plan_times']:
            lines.append(f"        planning: "
                         f"{len(rep['plan_times'])} visits, "
                         f"mean={np.mean(rep['plan_times'])*1000:.0f} ms, "
                         f"max={np.max(rep['plan_times'])*1000:.0f} ms")
        all_ok = g_ok and p_ok and a_ok and b_ok and t_ok and c_ok
        lines.append("-" * 64)
        lines.append(f"OVERALL: {'ALL CRITERIA PASS' if all_ok else 'NOT ALL CRITERIA PASS'}")
        lines.append("=" * 64)
        summary_txt = "\n".join(lines)
        print("\n" + summary_txt)
        _sum_path = os.path.join(RESULTS_DIR, f"dod_summary_{CSV_TAG}.txt")
        with open(_sum_path, "w") as _f:
            _f.write(summary_txt + "\n")
        print(f"[dod] summary saved to {_sum_path}")
    except Exception as _dod_err:
        import traceback
        print(f"[dod] WARNING: summary failed: {_dod_err}")
        traceback.print_exc()

    # ---- Save ablation .npz log (Change 7) ----
    _npz_name = os.path.join(RESULTS_DIR, f"log_{ABLATION_CONFIG}_{BASE_TYPE}_{ARM_TYPE}.npz")
    try:
        np.savez(_npz_name,
                 timestamps=np.array(log_time),
                 tracking_err=np.array(log_arm_err),
                 consensus_residual=np.array(log_primal_res),
                 admm_iterations=np.array(log_admm_iters),
                 z3_x=np.array(log_z3_x),
                 z3_y=np.array(log_z3_y),
                 z3_z=np.array(log_z3_z),
                 z2_v=np.array(log_z2_v),
                 z2_omega=np.array(log_z2_omega),
                 v_lag=np.array(log_v_lag),
                 base_speed=np.array(log_speed),
                 )
        print(f"[ablation] Saved {_npz_name} ({len(log_time)} steps)")
    except Exception as _npz_err:
        print(f"[ablation] WARNING: could not save {_npz_name}: {_npz_err}")

    return (log_time, log_alpha, log_alpha_raw, log_carry_dist,
            log_phase, log_arm_err, log_speed, log_F_dist, log_v_lag,
            log_rho,
            log_F_true_x, log_F_true_y, log_F_true_z,
            log_F_est_x, log_F_est_y, log_F_est_z,
            log_d_num_x, log_d_num_y, log_d_num_z,
            log_primal_res, log_omega_cmd,
            log_base_d_v, log_base_d_omega, log_base_v_est,
            log_alpha_geo, log_delta_err, log_delta_manip, log_delta_admm,
            log_ee_ref_x, log_ee_ref_y, log_ee_ref_z,
            log_ee_act_x, log_ee_act_y, log_ee_act_z,
            # Extended metrics (indices 34–51)
            log_manipulability, log_admm_iters,
            log_base_contrib, log_arm_contrib,
            log_q_joints[0], log_q_joints[1], log_q_joints[2],
            log_q_joints[3], log_q_joints[4], log_q_joints[5], log_q_joints[6],
            log_qd_joints[0], log_qd_joints[1], log_qd_joints[2],
            log_qd_joints[3], log_qd_joints[4], log_qd_joints[5], log_qd_joints[6],
            # New (index 52): base tracking error vs circle(theta_ref)
            log_base_err,
            # Stage 3 (index 53): top-down tool-axis tilt error in degrees.
            log_tool_axis_err_deg,
            # V10.8 (index 54): complete SO(3) orientation error in degrees.
            log_orientation_err_deg)


def plot_results(data, save_path=None):
    """Comprehensive post-simulation plots.

    Panels
    ------
    1. Alpha decomposition  — geo + corrections + smoothed alpha
    2. Arm tracking error   — mm
    3. Base velocity        — v_cmd vs ESO estimated actual (z1_v)
    4. Base ESO disturbance — d_hat_v and d_hat_omega
    5. Arm ESO disturbance  — estimated force magnitude |m_eff * z3|
    6. ADMM diagnostics     — primal residual and rho

    Plus one extra saved figure for arm trajectory diagnostics:
      - XY top-view path (reference vs actual)
      - X/Y/Z components vs time
    """
    if not data or not data[0]:
        print("No data to plot.")
        return

    # Unpack tuple by index
    t              = np.array(data[0])
    log_phase      = data[4]
    err            = np.array(data[5]) * 1000        # arm error  → mm
    v_cmd          = np.array(data[6])               # commanded base speed
    rho            = np.array(data[9])
    F_est_x        = np.array(data[13])
    F_est_y        = np.array(data[14])
    F_est_z        = np.array(data[15])
    primal_res     = np.array(data[19])
    alpha_geo      = np.array(data[24])
    delta_err      = np.array(data[25])
    delta_manip    = np.array(data[26])
    delta_admm     = np.array(data[27])
    ee_ref_x       = np.array(data[28])
    ee_ref_y       = np.array(data[29])
    ee_ref_z       = np.array(data[30])
    ee_act_x       = np.array(data[31])
    ee_act_y       = np.array(data[32])
    ee_act_z       = np.array(data[33])
    base_d_v       = np.array(data[21])
    base_d_omega   = np.array(data[22])
    base_v_est     = np.array(data[23])
    alpha          = np.array(data[1])

    F_arm_est_mag  = np.sqrt(F_est_x**2 + F_est_y**2 + F_est_z**2)

    # NOTE: log_phase stores short names (PCK_RCH, PLC_DRP, ...),
    # so keep color keys aligned with those exact strings.
    phase_colors = {
        "TRANSIT": "#ebebeb",
        "PCK_RCH": "#ffe0b2",
        "PCK_GRP": "#ffcc80",
        "LIFT":    "#a5d6a7",
        "LIFT_VERT": "#c8e6c9",
        "LIFT_RRT":  "#81c784",
        "PLC_RCH": "#b3e5fc",
        "PLC_DRP": "#81d4fa",
        "PLC_LFT": "#c5e1a5",
        "PLC_LFT_VERT": "#dcedc8",
        "PLC_LFT_RRT":  "#aed581",
        "DONE":    "#f5f5f5",
    }

    def shade_phases(ax):
        if not log_phase:
            return
        prev = log_phase[0]
        t0_seg = t[0]
        for i in range(1, len(log_phase)):
            if log_phase[i] != prev or i == len(log_phase) - 1:
                ax.axvspan(t0_seg, t[i], alpha=0.18,
                           color=phase_colors.get(prev, "#f5f5f5"), lw=0)
                t0_seg = t[i]
                prev = log_phase[i]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(6, 1, figsize=(15, 18), sharex=True)
    fig.suptitle(f"CLF-ADMM + Dual ESO Observer — Physics Engine  [{CSV_TAG}]",
                 fontsize=13, fontweight='bold')

    # ---- Panel 1: Alpha decomposition ----
    ax = axes[0]
    shade_phases(ax)
    ax.plot(t, alpha,       'b-',  lw=2.0, label='α (smoothed)')
    ax.plot(t, alpha_geo,   'b--', lw=1.2, alpha=0.7, label='α_raw (capability)')
    ax.plot(t,  delta_err,  'g-',  lw=1.0, alpha=0.8, label='+δ_err')
    ax.plot(t, -delta_manip,'r-',  lw=1.0, alpha=0.8, label='-δ_manip')
    ax.plot(t,  delta_admm, 'c-',  lw=1.0, alpha=0.8, label='+δ_admm')
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.set_ylabel("α (arm weight)")
    ax.set_ylim(-0.25, 1.1)
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    # ---- Panel 2: Arm tracking error ----
    ax = axes[1]
    shade_phases(ax)
    ax.plot(t, err, color='darkgreen', lw=1.2)
    ax.axhline(50, color='r', lw=0.8, ls='--', label='50 mm threshold')
    ax.set_ylabel("Arm error (mm)")
    # Robust cap keeps extreme spikes from flattening the rest of the trace.
    err_p99 = float(np.percentile(err, 99)) if len(err) else 100.0
    ax.set_ylim(0.0, max(60.0, err_p99 * 1.15))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 3: Base velocity — commanded vs ESO estimated ----
    ax = axes[2]
    shade_phases(ax)
    ax.plot(t, v_cmd,      'b-',  lw=1.5, label='v_cmd (ADMM)')
    ax.plot(t, base_v_est, 'orange', lw=1.2, ls='--',
            label='v_est (ESO z₁, actual)')
    ax.set_ylabel("Base velocity (m/s)")
    v_lo = min(float(np.min(v_cmd)), float(np.min(base_v_est)))
    v_hi = max(float(np.max(v_cmd)), float(np.max(base_v_est)))
    ax.set_ylim(v_lo - 0.05, v_hi + 0.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 4: Base ESO disturbance ----
    ax = axes[3]
    shade_phases(ax)
    ax.plot(t, base_d_v,     'm-',  lw=1.2, label='d̂_v  (linear)')
    ax.plot(t, base_d_omega, 'c-',  lw=1.2, label='d̂_ω  (angular)')
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.set_ylabel("Base ESO disturbance")
    bd_max = max(float(np.max(np.abs(base_d_v))), float(np.max(np.abs(base_d_omega))), 0.1)
    ax.set_ylim(-1.1 * bd_max, 1.1 * bd_max)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 5: Arm ESO disturbance force ----
    ax = axes[4]
    shade_phases(ax)
    ax.plot(t, F_arm_est_mag, 'r-', lw=1.2, label='|F̂_arm| = m·|z₃|')
    ax.plot(t, F_est_x, 'r--', lw=0.8, alpha=0.5, label='x')
    ax.plot(t, F_est_y, 'g--', lw=0.8, alpha=0.5, label='y')
    ax.plot(t, F_est_z, 'b--', lw=0.8, alpha=0.5, label='z')
    ax.set_ylabel("Arm ESO force (N)")
    f_p99 = float(np.percentile(F_arm_est_mag, 99)) if len(F_arm_est_mag) else 10.0
    ax.set_ylim(-1.15 * f_p99, 1.15 * f_p99)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # ---- Panel 6: ADMM diagnostics ----
    ax = axes[5]
    shade_phases(ax)
    ax.plot(t, primal_res, 'k-',  lw=1.2, label='primal residual (m)')
    ax2_rho = ax.twinx()
    ax2_rho.plot(t, rho, 'purple', lw=1.0, ls='--', alpha=0.7, label='ρ')
    ax2_rho.set_ylabel("ρ", color='purple')
    ax2_rho.tick_params(axis='y', labelcolor='purple')
    ax.set_ylabel("Primal residual (m)")
    r_p99 = float(np.percentile(primal_res, 99)) if len(primal_res) else 0.01
    ax.set_ylim(0.0, max(0.005, r_p99 * 1.25))
    rho_hi = max(float(np.max(rho)), 1.0)
    ax2_rho.set_ylim(0.0, rho_hi * 1.1)
    ax.set_xlabel("Time (s)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2_rho.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Phase legend ----
    from matplotlib.patches import Patch
    seen = dict.fromkeys(log_phase)
    patches = [Patch(facecolor=phase_colors.get(ph, "#f5f5f5"),
                     alpha=0.6, label=ph) for ph in seen]
    fig.legend(handles=patches, loc="lower center",
               ncol=len(seen), fontsize=8, frameon=False)

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    # Save to file
    out = save_path or os.path.join(RESULTS_DIR, f"results_{CSV_TAG}.png")
    plt.savefig(out, dpi=220, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to {out}")
    if not NO_GUI:
        plt.show()

    # ---- Arm trajectory figure ----
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9), sharex='col')
    fig2.suptitle(f"Arm End-Effector Trajectory  [{CSV_TAG}]",
                  fontsize=12, fontweight='bold')

    ax = axes2[0, 0]
    ax.plot(ee_ref_x, ee_ref_y, 'k--', lw=1.1, alpha=0.8, label='reference XY')
    ax.plot(ee_act_x, ee_act_y, color='tab:red', lw=1.3, alpha=0.9, label='actual XY')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Top View (XY)")
    ax.axis("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes2[0, 1]
    shade_phases(ax)
    ax.plot(t, ee_ref_z, 'k--', lw=1.1, label='z_ref')
    ax.plot(t, ee_act_z, color='tab:red', lw=1.2, label='z_actual')
    ax.set_ylabel("Z (m)")
    ax.set_title("Vertical Motion (Z)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes2[1, 0]
    shade_phases(ax)
    ax.plot(t, ee_ref_x, 'k--', lw=1.0, label='x_ref')
    ax.plot(t, ee_act_x, color='tab:red', lw=1.2, label='x_actual')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("X (m)")
    ax.set_title("X vs Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes2[1, 1]
    shade_phases(ax)
    ax.plot(t, ee_ref_y, 'k--', lw=1.0, label='y_ref')
    ax.plot(t, ee_act_y, color='tab:red', lw=1.2, label='y_actual')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Y vs Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_traj = out.replace(".png", "_arm_traj.png")
    plt.savefig(out_traj, dpi=220, bbox_inches='tight', facecolor='white')
    print(f"Trajectory plot saved to {out_traj}")
    if not NO_GUI:
        plt.show()


# ============================================================
# Launch
# ============================================================
_log_result = [None]  # mutable container to capture thread return

def _coord_wrapper():
    _log_result[0] = coordination_loop()

coord_thread = threading.Thread(target=_coord_wrapper,
                                name="Thread-Coordinator", daemon=True)
coord_thread.start()

print(f"\nThreads: {[t.name for t in threading.enumerate()]}\n")

try:
    while p.isConnected() and not stop_event.is_set():
        time.sleep(0.5)
except KeyboardInterrupt:
    print("\nCtrl+C")

stop_event.set()
coord_thread.join(timeout=3.0)
mcu_base.join(timeout=2.0)
mcu_arm.join(timeout=2.0)
print("Done.")

# ---- Save CSV for comparison plots ----
if _log_result[0] is not None:
    import csv
    data = _log_result[0]
    csv_name = os.path.join(RESULTS_DIR, f"log_{CSV_TAG}.csv")
    with open(csv_name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "alpha", "alpha_raw", "eta_arm", "phase",
                     "arm_err", "speed", "F_dist", "v_lag", "rho",
                     "F_true_x", "F_true_y", "F_true_z",
                     "F_est_x", "F_est_y", "F_est_z",
                     "d_num_x", "d_num_y", "d_num_z",
                     "primal_res", "omega_cmd", "base_d_v", "base_d_omega",
                     "base_v_est", "alpha_raw_alias", "delta_err_unused", "delta_manip_unused",
                     "delta_admm_unused",
                     "ee_ref_x", "ee_ref_y", "ee_ref_z",
                     "ee_act_x", "ee_act_y", "ee_act_z",
                     # Extended columns (indices 34–51)
                     "manipulability", "admm_iters",
                     "base_contrib_norm", "arm_contrib_norm",
                     "q_joint_0", "q_joint_1", "q_joint_2", "q_joint_3",
                     "q_joint_4", "q_joint_5", "q_joint_6",
                     "qdot_joint_0", "qdot_joint_1", "qdot_joint_2", "qdot_joint_3",
                     "qdot_joint_4", "qdot_joint_5", "qdot_joint_6",
                     "base_err", "tool_axis_err_deg",
                     "orientation_err_deg"])
        for i in range(len(data[0])):
            w.writerow([data[j][i] for j in range(len(data))])
    print(f"Saved {len(data[0])} rows to {csv_name}")

    if SKIP_PLOTS:
        print("Plot generation skipped (--batch-mode/--skip-plots).")
    else:
        plot_results(data)
