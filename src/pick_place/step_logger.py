"""Per-step logger for ADMM iteration-map (Subproblem 1a) analysis.

Records, every `decimate` simulation steps, every quantity needed to
reconstruct the affine ADMM iteration map T from `subproblem_1a_draft.md`
§3.3.  All quantities are captured AFTER `admm.step()` returns, so the
snapshot reflects the state at the END of the step (post-dual-decay).
For 1a analysis we want the state at the START of the next step — which
is what this captures.

Usage in pick_place_cycle.py
----------------------------
    from step_logger import StepLogger
    logger = StepLogger(p, husky_id, kuka_id, ee_index, num_joints,
                        admm, base_sub, arm_sub,
                        priority_ctrl=priority_ctrl,
                        out_path="admm_log.npz",
                        decimate=1)
    # ... in the main loop, AFTER admm.step():
    logger.record(step_count, t_now, phase_name,
                  pos_desired=arm_target_corrected,
                  vel_desired=arm_vel,
                  base_pos_desired=base_pos_desired,
                  v_ff=v_ff, omega_ff=omega_ff,
                  heading_target=heading_target,
                  info=info)
    # ... at the end of the simulation:
    logger.save()

Why a separate file
-------------------
* Zero changes to production controller code.
* The Jacobian/forward-kinematics queries here duplicate what the
  subsystems already compute internally — paid only at `decimate`
  rate, configurable.
* Output is a single .npz with structured arrays, easy to load:
      data = np.load("admm_log.npz", allow_pickle=True)
      steps = data["steps"]           # (N,) step indices
      J_all = data["J"]               # (N, 3, n) positional Jacobian
      M_all = data["M"]               # (N, 6, 2) base twist model
      ...
"""

import math
import numpy as np


class StepLogger:
    """Snapshot every input needed to rebuild T at each control step."""

    def __init__(self, sim, husky_id, kuka_id, ee_index, num_joints,
                 admm, base_sub, arm_sub,
                 priority_ctrl=None, out_path="admm_log.npz",
                 decimate=1):
        self.p           = sim
        self.husky_id    = husky_id
        self.kuka_id     = kuka_id
        self.ee_index    = ee_index
        self.nj          = num_joints
        self.admm        = admm
        self.base_sub    = base_sub
        self.arm_sub     = arm_sub
        self.priority    = priority_ctrl
        self.out_path    = out_path
        self.decimate    = max(1, int(decimate))

        # Total non-fixed DOF (xArm6 quirk — copied from ArmSubsystem)
        _FIXED = 4
        self._n_dof = sum(
            1 for i in range(sim.getNumJoints(kuka_id))
            if sim.getJointInfo(kuka_id, i)[2] != _FIXED
        )

        self._records = []

    # ------------------------------------------------------------------
    # Per-step snapshot
    # ------------------------------------------------------------------

    def record(self, step_count, t_now, phase_name,
               pos_desired, vel_desired, base_pos_desired,
               v_ff, omega_ff, heading_target, info, dt=1.0/240.0,
               base_path_ref_xy=None, base_track_err=None):
        """Capture one snapshot.  Cheap if decimated; ~1 ms otherwise."""
        if step_count % self.decimate != 0:
            return

        # ---- Base geometry (M = [d_v | d_w], R_base, ψ, ee, p0) ----
        husky_pos, orn = self.p.getBasePositionAndOrientation(self.husky_id)
        husky_pos = np.asarray(husky_pos, dtype=float)
        psi = self.p.getEulerFromQuaternion(orn)[2]
        cy, sy = math.cos(psi), math.sin(psi)

        ls = self.p.getLinkState(self.kuka_id, self.ee_index,
                                  computeForwardKinematics=True)
        ee = np.array(ls[4])
        base_pos, base_orn = self.p.getBasePositionAndOrientation(self.kuka_id)
        R_base = np.array(
            self.p.getMatrixFromQuaternion(base_orn)).reshape(3, 3)

        r_ee = ee - np.array(base_pos)
        z_axis = np.array([0.0, 0.0, 1.0])
        d_v = np.r_[dt*np.array([cy, sy, 0.0]), np.zeros(3)]
        d_w = np.r_[dt*np.cross(z_axis, r_ee), dt*z_axis]
        M   = np.column_stack([d_v, d_w])      # (6, 2)

        # ---- Arm Jacobian (full spatial 6D, world frame) ----
        js  = [self.p.getJointState(self.kuka_id, j) for j in range(self.nj)]
        q   = np.array([s[0] for s in js])
        qd  = np.array([s[1] for s in js])

        q_pad  = list(q)  + [0.0] * (self._n_dof - self.nj)
        qd_pad = list(qd) + [0.0] * (self._n_dof - self.nj)
        zero_vec = [0.0] * self._n_dof
        jac_lin, jac_ang = self.p.calculateJacobian(
            self.kuka_id, self.ee_index, [0, 0, 0],
            q_pad, qd_pad, zero_vec)
        J = np.vstack([
            R_base @ np.array(jac_lin)[:, :self.nj],
            R_base @ np.array(jac_ang)[:, :self.nj],
        ])   # (6, nj)

        # ---- CLF-ADMM scalar state at end of step ----
        # Legacy NPZ keys z/u_b/u_a are retained, but now contain scalar
        # CLF auxiliary/dual variables rather than Cartesian consensus vectors.
        z   = np.array([self.admm.z_b, self.admm.z_a], dtype=float)
        u_b = np.array([self.admm.lambda_b], dtype=float)
        u_a = np.array([self.admm.lambda_a], dtype=float)
        # dq_prev lives inside ArmSubsystem
        dq_prev = self.arm_sub._dq_prev.copy()

        # ---- ADMM scalar params at this step ----
        rho   = float(self.admm.rho)
        # Revised architecture: alpha is a role preference, while these are
        # the actual positive quadratic cost weights (normalized to sum to 1).
        w_a   = float(getattr(self.admm, "role_weight_arm", np.nan))
        w_b   = float(getattr(self.admm, "role_weight_base", np.nan))
        tol   = float(self.admm.tol)
        max_i = int(self.admm.max_iter)

        # ---- Subsystem weights (post-EMA) ----
        w_track_b   = float(self.base_sub.w_track)
        w_track_a   = float(self.arm_sub.w_track)
        # Legacy CSV fields retained for old analyzers. ``alpha_prox`` is an
        # arm-local tracking fraction, NOT the adaptive role index.  Command
        # smoothing is disabled in the revised optimization, hence beta_ema=0.
        alpha_prox  = float(getattr(self.arm_sub, "tracking_fraction", np.nan))
        beta_ema    = 0.0
        damping     = float(self.arm_sub.damping)

        # ---- Base regularization ----
        reg_v       = float(self.base_sub.reg_v)
        reg_omega   = float(self.base_sub.reg_omega)
        path_weight = float(self.base_sub.path_weight)

        # ---- Optional adaptive-priority signal ----
        if self.priority is not None:
            try:
                alpha_priority = float(getattr(self.priority, "alpha", np.nan))
            except Exception:
                alpha_priority = float("nan")
        else:
            alpha_priority = float("nan")

        # ---- True mobile-base path tracking metric ---------------------
        # ``base_pos_desired`` is retained only as task-level telemetry.  The
        # revised base local controller follows ``base_path_ref_xy`` through
        # its independent path-tracking law.  Log that path reference
        # separately so analyzers never mix frames or controller semantics.
        if base_path_ref_xy is None:
            base_path_ref_xy_arr = np.full(2, np.nan, dtype=float)
        else:
            base_path_ref_xy_arr = np.asarray(base_path_ref_xy, dtype=float).reshape(-1)[:2]
        if base_track_err is None:
            if np.all(np.isfinite(base_path_ref_xy_arr)):
                base_track_err_val = float(np.linalg.norm(
                    husky_pos[:2] - base_path_ref_xy_arr))
            else:
                base_track_err_val = float("nan")
        else:
            base_track_err_val = float(base_track_err)

        # ---- Pack as a flat dict (lists of arrays merged at save time) ----
        rec = {
            "step":         int(step_count),
            "t":            float(t_now),
            "phase":        str(phase_name),
            "dt":           float(dt),
            # geometry
            "psi":          float(psi),
            "p_ee":         ee.astype(np.float64),
            # Mobile-base pose used for path tracking (Husky chassis).
            "p_base":       husky_pos.astype(np.float64),
            "p_base_xy":    husky_pos[:2].astype(np.float64),
            # Manipulator mount/base frame retained for kinematic auditing.
            "p_arm_base":   np.array(base_pos, dtype=np.float64),
            "R_base":       R_base.astype(np.float64),
            "M":            M.astype(np.float64),
            "J":            J.astype(np.float64),
            "q":            q.astype(np.float64),
            "qd":           qd.astype(np.float64),
            # ADMM state (start of next step)
            "z":            z.astype(np.float64),
            "u_b":          u_b.astype(np.float64),
            "u_a":          u_a.astype(np.float64),
            "dq_prev":      dq_prev.astype(np.float64),
            # ADMM scalar params
            "rho":          rho,
            "w_arm":        w_a,
            "w_base":       w_b,
            "tol":          tol,
            "max_iter":     max_i,
            # Subsystem params
            "w_track_a":    w_track_a,
            "w_track_b":    w_track_b,
            "alpha_prox":   alpha_prox,
            "beta_ema":     beta_ema,
            "damping":      damping,
            "reg_v":        reg_v,
            "reg_omega":    reg_omega,
            "base_path_weight": path_weight,
            "alpha_priority": alpha_priority,
            # References at this step
            "pos_des_a":    np.asarray(pos_desired, dtype=np.float64),
            # ``pos_des_b`` is retained for backward compatibility: it is
            # the local Cartesian task reference, not a Husky path point.
            "pos_des_b":    np.asarray(base_pos_desired, dtype=np.float64),
            "base_ref_xy":  base_path_ref_xy_arr.astype(np.float64),
            "base_track_err": base_track_err_val,
            "vel_des":      np.asarray(vel_desired, dtype=np.float64),
            "task_dim":      int(info.get("task_dim", 3)),
            "angular_task_scale": float(info.get("angular_task_scale", np.nan)),
            "pose_error_6d": np.asarray(
                info.get("pose_error_6d", np.full(6, np.nan)), dtype=np.float64),
            "orientation_error_deg": float(
                info.get("orientation_error_deg", np.nan)),
            "twist_desired_6d": np.asarray(
                info.get("twist_desired_6d", np.full(6, np.nan)), dtype=np.float64),
            "servo_twist_6d": np.asarray(
                info.get("servo_twist_6d", np.full(6, np.nan)), dtype=np.float64),
            "task_vmax": float(info.get("task_vmax", np.nan)),
            "task_omegamax": float(info.get("task_omegamax", np.nan)),
            "planner_full6d_combined_error_m": float(
                info.get("planner_full6d_combined_error_m", np.nan)),
            "planner_full6d_orientation_error_deg": float(
                info.get("planner_full6d_orientation_error_deg", np.nan)),
            "planner_full6d_orientation_rate_limit": float(
                info.get("planner_full6d_orientation_rate_limit", np.nan)),
            "planner_progress_rate": float(
                info.get("planner_progress_rate", np.nan)),
            "planner_progress_rate_raw": float(
                info.get("planner_progress_rate_raw", np.nan)),
            "planner_progress_cart_cap": float(
                info.get("planner_progress_cart_cap", np.nan)),
            "planner_progress_joint_cap": float(
                info.get("planner_progress_joint_cap", np.nan)),
            "planner_progress_tracking_factor": float(
                info.get("planner_progress_tracking_factor", np.nan)),
            "planner_progress_paused": bool(
                info.get("planner_progress_paused", False)),
            "planner_progress_dpds_m": float(
                info.get("planner_progress_dpds_m", np.nan)),
            "planner_segment_min_duration_s": float(
                info.get("planner_segment_min_duration_s", np.nan)),
            "base_progress_decoupled": bool(
                info.get("base_progress_decoupled", False)),
            "base_reference_theta": float(
                info.get("base_reference_theta", np.nan)),
            "base_reference_speed": float(
                info.get("base_reference_speed", np.nan)),
            "base_reference_target_speed": float(
                info.get("base_reference_target_speed", np.nan)),
            "base_reference_horizon_theta": float(
                info.get("base_reference_horizon_theta", np.nan)),
            "base_reference_horizon_distance_m": float(
                info.get("base_reference_horizon_distance_m", np.nan)),
            "base_reference_stop_required": bool(
                info.get("base_reference_stop_required", False)),
            "base_reference_lead_m": float(
                info.get("base_reference_lead_m", np.nan)),
            "base_reference_phase_cap": float(
                info.get("base_reference_phase_cap", np.nan)),
            "v_ff":         float(v_ff) if v_ff is not None else float("nan"),
            "omega_ff":     float(omega_ff) if omega_ff is not None else float("nan"),
            "heading_target": float(heading_target) if heading_target is not None else float("nan"),
            # Research comparison backend diagnostics
            "control_mode": str(info.get("control_mode", "unknown")),
            "coordination_mode": str(info.get("coordination_mode", "unknown")),
            "centralized_clf_multiplier": float(info.get("centralized_clf_multiplier", 0.0)),
            "centralized_clf_feasible": bool(info.get("centralized_clf_feasible", True)),
            "centralized_clf_active": bool(info.get("centralized_clf_active", False)),
            "centralized_nominal_violation": float(info.get("centralized_nominal_violation", 0.0)),
            "centralized_qp_box_solves": int(info.get("centralized_qp_box_solves", 0)),
            "centralized_qp_time_ms": float(info.get("centralized_qp_time_ms", 0.0)),
            "centralized_qp_kkt": float(info.get("centralized_qp_kkt", float("nan"))),
            "centralized_qp_objective": float(info.get("centralized_qp_objective", float("nan"))),
            "centralized_qp_status": str(info.get("centralized_qp_status", "not-centralized")),
            "coordination_wall_time_ms": float(info.get("coordination_wall_time_ms", float("nan"))),
            "admm_wall_time_ms": float(info.get("admm_wall_time_ms", float("nan"))),
            "admm_time_per_iteration_ms": float(info.get("admm_time_per_iteration_ms", float("nan"))),
            # Diagnostics from coordinator.step()
            "r_prim":       float(info.get("primal_residual", float("nan"))),
            "r_dual":       float(info.get("dual_residual", float("nan"))),
            "iters_used":   int(info.get("iterations", -1)),
            "v_cmd":        float(info.get("v_cmd", float("nan"))),
            "omega_cmd":    float(info.get("omega_cmd", float("nan"))),
            "u_b_norm":     float(info.get("u_b_norm", float("nan"))),
            "u_a_norm":     float(info.get("u_a_norm", float("nan"))),
            # Optional CLF feasibility-filter diagnostics
            "clf_enabled":    bool(info.get("clf_enabled", False)),
            "clf_V":          float(info.get("clf_V", float("nan"))),
            "clf_dV":         float(info.get("clf_dV", float("nan"))),
            "clf_bound":      float(info.get("clf_bound", float("nan"))),
            "clf_residual":   float(info.get("clf_residual", float("nan"))),
            "clf_hard_residual": float(info.get("clf_hard_residual", float("nan"))),
            "clf_soft_bound": float(info.get("clf_soft_bound", float("nan"))),
            "clf_hard_feasible": bool(info.get("clf_hard_feasible", True)),
            "clf_slack": float(info.get("clf_slack", 0.0)),
            "clf_slack_penalty": float(info.get("clf_slack_penalty", float("nan"))),
            "clf_active":     bool(info.get("clf_active", False)),
            "clf_feasible":   bool(info.get("clf_feasible", True)),
            "clf_correction": float(info.get("clf_correction", float("nan"))),
            # V10.7 nominal-first practical error-tube supervisor diagnostics.
            "stability_supervisor_enabled": bool(info.get("stability_supervisor_enabled", False)),
            "stability_supervisor_active": bool(info.get("stability_supervisor_active", False)),
            "stability_supervisor_phi_nominal": float(info.get("stability_supervisor_phi_nominal", float("nan"))),
            "stability_supervisor_phi_threshold": float(info.get("stability_supervisor_phi_threshold", 0.0)),
            "stability_supervisor_recovery_rate": float(info.get("stability_supervisor_recovery_rate", float("nan"))),
            "stability_supervisor_recovery_bound": float(info.get("stability_supervisor_recovery_bound", float("nan"))),
            "stability_supervisor_nominal_decreasing": bool(info.get("stability_supervisor_nominal_decreasing", False)),
            "stability_supervisor_nominal_acceptable": bool(info.get("stability_supervisor_nominal_acceptable", False)),
            "stability_supervisor_error_norm": float(info.get("stability_supervisor_error_norm", float("nan"))),
            "stability_supervisor_error_tolerance": float(info.get("stability_supervisor_error_tolerance", float("nan"))),
            "stability_supervisor_inside_error_tube": bool(info.get("stability_supervisor_inside_error_tube", False)),
            "stability_supervisor_outside_error_tube": bool(info.get("stability_supervisor_outside_error_tube", False)),
            "stability_supervisor_V_tolerance": float(info.get("stability_supervisor_V_tolerance", float("nan"))),
            "stability_supervisor_V_excess": float(info.get("stability_supervisor_V_excess", float("nan"))),
            "stability_supervisor_bypass_inside_tube": bool(info.get("stability_supervisor_bypass_inside_tube", False)),
            "stability_supervisor_bypass_nominal_decrease": bool(info.get("stability_supervisor_bypass_nominal_decrease", False)),
            "stability_supervisor_activation_count": int(info.get("stability_supervisor_activation_count", 0)),
            "clf_exponential_bound": float(info.get("clf_exponential_bound", float("nan"))),
            "clf_exponential_residual": float(info.get("clf_exponential_residual", float("nan"))),
            "clf_exponential_feasible": bool(info.get("clf_exponential_feasible", False)),
            "clf_s_base": float(info.get("clf_s_base", float("nan"))),
            "clf_s_arm": float(info.get("clf_s_arm", float("nan"))),
            "clf_total_contribution": float(info.get("clf_total_contribution", float("nan"))),
            "clf_budget": float(info.get("clf_budget", float("nan"))),
            "clf_budget_gap": float(info.get("clf_budget_gap", float("nan"))),
            "clf_constraint_violation": float(info.get("clf_constraint_violation", float("nan"))),
            "clf_soft_budget": float(info.get("clf_soft_budget", float("nan"))),
            "clf_soft_budget_gap": float(info.get("clf_soft_budget_gap", float("nan"))),
            "clf_soft_constraint_violation": float(info.get("clf_soft_constraint_violation", float("nan"))),
            # Finite-iteration ADMM / practical CLF certificate
            "admm_finite_iter_margin": float(info.get("admm_finite_iter_margin", float("nan"))),
            "admm_finite_error_bound": float(info.get("admm_finite_error_bound", float("nan"))),
            "admm_actual_finite_violation": float(info.get("admm_actual_finite_violation", float("nan"))),
            "admm_finite_iter_certified": bool(info.get("admm_finite_iter_certified", False)),
            "clf_eta_bound": float(info.get("clf_eta_bound", float("nan"))),
            "clf_V_ultimate_bound": float(info.get("clf_V_ultimate_bound", float("nan"))),
            "clf_error_ultimate_bound": float(info.get("clf_error_ultimate_bound", float("nan"))),
            # Explicit three-layer research architecture diagnostics.
            # Layer 1 -- residual-task decomposition
            "layer1_v_task": np.asarray(info.get("layer1_v_task", [np.nan]*3), dtype=np.float64),
            "layer1_v_path_ee": np.asarray(info.get("layer1_v_path_ee", [np.nan]*3), dtype=np.float64),
            "layer1_v_residual": np.asarray(info.get("layer1_v_residual", [np.nan]*3), dtype=np.float64),
            "layer1_v_shareable": np.asarray(info.get("layer1_v_shareable", [np.nan]*3), dtype=np.float64),
            "layer1_v_mandatory": np.asarray(info.get("layer1_v_mandatory", [np.nan]*3), dtype=np.float64),
            "layer1_reconstruction_error": float(info.get("layer1_reconstruction_error", float("nan"))),
            "layer1_projector_idempotence_error": float(info.get("layer1_projector_idempotence_error", float("nan"))),
            "layer1_mandatory_base_projection_error": float(info.get("layer1_mandatory_base_projection_error", float("nan"))),
            # Layer 2 -- capability-balanced task allocation
            "layer2_alpha_arm": float(info.get("layer2_alpha_arm", float("nan"))),
            "layer2_alpha_base": float(info.get("layer2_alpha_base", float("nan"))),
            "layer2_gamma_arm": float(info.get("layer2_gamma_arm", float("nan"))),
            "layer2_gamma_base": float(info.get("layer2_gamma_base", float("nan"))),
            "layer2_weight_arm": float(info.get("layer2_weight_arm", float("nan"))),
            "layer2_weight_base": float(info.get("layer2_weight_base", float("nan"))),
            "layer2_v_arm_ref": np.asarray(info.get("layer2_v_arm_ref", [np.nan]*3), dtype=np.float64),
            "layer2_v_base_correction_ref": np.asarray(info.get("layer2_v_base_correction_ref", [np.nan]*3), dtype=np.float64),
            "layer2_allocation_reconstruction_error": float(info.get("layer2_allocation_reconstruction_error", float("nan"))),
            # Layer 3 -- shifted CLF-ADMM coordination
            "layer3_a_base": np.asarray(info.get("layer3_a_base", [np.nan]*2), dtype=np.float64),
            "layer3_a_arm": np.asarray(info.get("layer3_a_arm", np.full(self.nj, np.nan)), dtype=np.float64),
            "layer3_clf_budget": float(info.get("layer3_clf_budget", float("nan"))),
            "layer3_recovery_clf_budget": float(info.get("layer3_recovery_clf_budget", float("nan"))),
            "layer3_exponential_clf_budget": float(info.get("layer3_exponential_clf_budget", float("nan"))),
            "layer3_phi_nominal": float(info.get("layer3_phi_nominal", float("nan"))),
            "layer3_s_base": float(info.get("layer3_s_base", float("nan"))),
            "layer3_s_arm": float(info.get("layer3_s_arm", float("nan"))),
            "layer3_coupling_residual": float(info.get("layer3_coupling_residual", float("nan"))),
            "layer3_path_clf_contribution": float(info.get("layer3_path_clf_contribution", float("nan"))),
            # Backward-compatible residual-task decomposition diagnostics
            "base_path_ee_velocity": np.asarray(info.get("base_path_ee_velocity", [np.nan, np.nan, np.nan]), dtype=np.float64),
            "base_path_clf_contribution": float(info.get("base_path_clf_contribution", float("nan"))),
            "clf_decomposition_error": float(info.get("clf_decomposition_error", float("nan"))),
            # Capability-based alpha allocator diagnostics
            "alpha_task": float(info.get("alpha_task", alpha_priority)),
            "alpha_raw_capability": float(info.get("alpha_raw", float("nan"))),
            "alpha_capacity": float(info.get("alpha_capacity", float("nan"))),
            "alpha_eta_arm": float(info.get("alpha_eta_arm", float("nan"))),
            "alpha_eta_base": float(info.get("alpha_eta_base", float("nan"))),
            "alpha_eta_base_actuator": float(info.get("alpha_eta_base_actuator", float("nan"))),
            "alpha_eta_base_locomotion": float(info.get("alpha_eta_base_locomotion", float("nan"))),
            "alpha_eta_worst": float(info.get("alpha_eta_worst", float("nan"))),
            "alpha_eta_arm_at_0": float(info.get("alpha_eta_arm_at_0", float("nan"))),
            "alpha_eta_arm_at_05": float(info.get("alpha_eta_arm_at_05", float("nan"))),
            "alpha_eta_arm_at_1": float(info.get("alpha_eta_arm_at_1", float("nan"))),
            "alpha_eta_base_at_0": float(info.get("alpha_eta_base_at_0", float("nan"))),
            "alpha_eta_base_at_05": float(info.get("alpha_eta_base_at_05", float("nan"))),
            "alpha_eta_base_at_1": float(info.get("alpha_eta_base_at_1", float("nan"))),
            "alpha_eta_mandatory": float(info.get("alpha_eta_mandatory", float("nan"))),
            "alpha_eta_shareable_full": float(info.get("alpha_eta_shareable_full", float("nan"))),
            "alpha_residual_norm": float(info.get("alpha_residual_norm", float("nan"))),
            "alpha_shareable_norm": float(info.get("alpha_shareable_norm", float("nan"))),
            "alpha_mandatory_norm": float(info.get("alpha_mandatory_norm", float("nan"))),
            "alpha_qdot_full": np.asarray(info.get("alpha_qdot_full", np.full(self.nj, np.nan)), dtype=np.float64),
            "alpha_qdot_mandatory": np.asarray(info.get("alpha_qdot_mandatory", np.full(self.nj, np.nan)), dtype=np.float64),
            "alpha_qdot_shareable_full": np.asarray(info.get("alpha_qdot_shareable_full", np.full(self.nj, np.nan)), dtype=np.float64),
            "alpha_deadband_hold": bool(info.get("alpha_deadband_hold", False)),
            "alpha_capacity_feasible": bool(info.get("alpha_capacity_feasible", True)),
            "alpha_base_correction_full": np.asarray(info.get("alpha_base_correction_full", [np.nan, np.nan]), dtype=np.float64),
            "alpha_base_remaining_margin": np.asarray(info.get("alpha_base_remaining_margin", [np.nan, np.nan]), dtype=np.float64),
            "alpha_path_outside_bounds": bool(info.get("alpha_path_outside_bounds", False)),
            "alpha_base_path_weight": float(info.get("alpha_base_path_weight", float("nan"))),
            "alpha_base_locomotion_jmax": float(info.get("alpha_base_locomotion_jmax", float("nan"))),
            "alpha_base_pinv_sigma_min": float(info.get("alpha_base_pinv_sigma_min", float("nan"))),
            "alpha_base_pinv_sigma_max": float(info.get("alpha_base_pinv_sigma_max", float("nan"))),
            "alpha_base_pinv_sigma_floor": float(info.get("alpha_base_pinv_sigma_floor", float("nan"))),
            "alpha_base_pinv_condition_raw": float(info.get("alpha_base_pinv_condition_raw", float("nan"))),
            "alpha_base_pinv_gain_max": float(info.get("alpha_base_pinv_gain_max", float("nan"))),
            "alpha_base_pinv_regularized": bool(info.get("alpha_base_pinv_regularized", False)),
            "alpha_j_locomotion": float(info.get("alpha_j_locomotion", float("nan"))),
            "alpha_j_locomotion_at_0": float(info.get("alpha_j_locomotion_at_0", float("nan"))),
            "alpha_j_locomotion_at_05": float(info.get("alpha_j_locomotion_at_05", float("nan"))),
            "alpha_j_locomotion_at_1": float(info.get("alpha_j_locomotion_at_1", float("nan"))),
            "residual_shareable_velocity": np.asarray(info.get("residual_shareable_velocity", [np.nan, np.nan, np.nan]), dtype=np.float64),
            "residual_mandatory_arm_velocity": np.asarray(info.get("residual_mandatory_arm_velocity", [np.nan, np.nan, np.nan]), dtype=np.float64),
            "arm_residual_task_velocity": np.asarray(info.get("arm_residual_task_velocity", [np.nan, np.nan, np.nan]), dtype=np.float64),
            "base_residual_task_velocity": np.asarray(info.get("base_residual_task_velocity", [np.nan, np.nan, np.nan]), dtype=np.float64),
            "base_correction_cmd": np.asarray(info.get("base_correction_cmd", [np.nan, np.nan]), dtype=np.float64),
            # Local constrained-QP numerical diagnostics
            "base_qp_kkt": float(info.get("base_qp_kkt", float("nan"))),
            "base_qp_iterations": int(info.get("base_qp_iterations", 0)),
            "base_qp_converged": bool(info.get("base_qp_converged", False)),
            # Base bottleneck diagnostics (path objective vs EE/CLF/limits).
            "base_u_path": np.asarray(info.get("base_u_path", [np.nan, np.nan]), dtype=np.float64),
            "base_u_nom": np.asarray(info.get("base_u_nom", [np.nan, np.nan]), dtype=np.float64),
            "base_ee_cost": float(info.get("base_ee_cost", float("nan"))),
            "base_path_cost": float(info.get("base_path_cost", float("nan"))),
            "base_reg_cost": float(info.get("base_reg_cost", float("nan"))),
            "base_admm_cost": float(info.get("base_admm_cost", float("nan"))),
            "base_path_cmd_gap": float(info.get("base_path_cmd_gap", float("nan"))),
            "base_nom_cmd_gap": float(info.get("base_nom_cmd_gap", float("nan"))),
            "base_path_nom_gap": float(info.get("base_path_nom_gap", float("nan"))),
            "base_v_lower_active": bool(info.get("base_v_lower_active", False)),
            "base_v_upper_active": bool(info.get("base_v_upper_active", False)),
            "base_omega_lower_active": bool(info.get("base_omega_lower_active", False)),
            "base_omega_upper_active": bool(info.get("base_omega_upper_active", False)),
            "base_path_v_outside_bounds": bool(info.get("base_path_v_outside_bounds", False)),
            "base_path_omega_outside_bounds": bool(info.get("base_path_omega_outside_bounds", False)),
            "arm_qp_kkt": float(info.get("arm_qp_kkt", float("nan"))),
            "arm_qp_iterations": int(info.get("arm_qp_iterations", 0)),
            "arm_qp_converged": bool(info.get("arm_qp_converged", False)),
            "arm_accel_fallback": bool(info.get("arm_accel_fallback", False)),
            # Stage-3 planner diagnostic: top-down tilt only.
            "arm_tool_axis_error": np.asarray(
                info.get("arm_tool_axis_error", [np.nan, np.nan, np.nan]),
                dtype=np.float64),
            "arm_tool_axis_error_deg": float(
                info.get("arm_tool_axis_error_deg", float("nan"))),
            # V10.8 execution diagnostic: complete SO(3) error, including the
            # wrist-yaw orientation selected by the accepted planner path.
            "arm_full_orientation_error": np.asarray(
                info.get("arm_orientation_error", [np.nan, np.nan, np.nan]),
                dtype=np.float64),
            "arm_full_orientation_error_deg": float(
                info.get("arm_orientation_error_deg", float("nan"))),
            "arm_tool_axis_omega_ref": np.asarray(
                info.get("arm_omega_local_ref", [np.nan, np.nan, np.nan]),
                dtype=np.float64),
            # Release hard-gate diagnostics (task executor; logging only)
            "release_aim_count": int(info.get("release_aim_count", 0)),
            "release_cube_residual": float(info.get("release_cube_residual", float("nan"))),
            "release_cube_residual_xy": np.asarray(
                info.get("release_cube_residual_xy", [np.nan, np.nan]),
                dtype=np.float64),
            "release_cube_speed": float(info.get("release_cube_speed", float("nan"))),
            "release_cube_ang_speed": float(info.get("release_cube_ang_speed", float("nan"))),
            "release_cube_tilt_ang_speed": float(info.get("release_cube_tilt_ang_speed", float("nan"))),
            "release_cube_tilt_deg": float(info.get("release_cube_tilt_deg", float("nan"))),
            "release_cube_aligned": bool(info.get("release_cube_aligned", False)),
            "release_cube_upright": bool(info.get("release_cube_upright", False)),
            "action_terminal_dwell": bool(info.get("action_terminal_dwell", False)),
            "action_ref_filter_active": bool(info.get("action_ref_filter_active", False)),
            "action_ref_filter_error": float(info.get("action_ref_filter_error", 0.0)),
            "action_ref_filter_speed": float(info.get("action_ref_filter_speed", 0.0)),
            "release_base_hold_active": bool(info.get("release_base_hold_active", False)),
            "release_base_hold_v": float(info.get("release_base_hold_v", np.nan)),
            "release_base_hold_omega": float(info.get("release_base_hold_omega", np.nan)),
            "release_gate_ready": bool(info.get("release_gate_ready", False)),
            "release_aim_exhausted": bool(info.get("release_aim_exhausted", False)),
            "action_dwell_elapsed": float(info.get("action_dwell_elapsed", 0.0)),
            "action_settle_elapsed": float(info.get("action_settle_elapsed", 0.0)),
        }
        self._records.append(rec)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        """Dump all records as a single .npz with one array per field."""
        if not self._records:
            print(f"[StepLogger] no records to save (was record() ever called?)")
            return

        # Pivot list-of-dicts → dict-of-arrays
        keys = self._records[0].keys()
        out  = {}
        for k in keys:
            vals = [r[k] for r in self._records]
            if isinstance(vals[0], np.ndarray):
                # Stack along new leading axis
                out[k] = np.stack(vals, axis=0)
            elif isinstance(vals[0], str):
                out[k] = np.array(vals, dtype=object)
            else:
                out[k] = np.array(vals)

        np.savez(self.out_path, **out)
        n = len(self._records)
        print(f"[StepLogger] saved {n} records to {self.out_path}")
        # Also report file size
        try:
            import os
            sz = os.path.getsize(self.out_path)
            print(f"[StepLogger]   file size: {sz/1024:.1f} KB")
        except Exception:
            pass

    def __len__(self):
        return len(self._records)
