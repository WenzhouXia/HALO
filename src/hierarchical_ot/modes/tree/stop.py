from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np

from ...instrumentation.phase_names import COMPONENT_CONVERGENCE_CHECK
from .infeasibility_check import compute_primal_infeas
from .logger import tree_log


def check_tree_convergence(
    solver,
    lp_result: Dict,
    inf_thr: float,
    level_idx: int = 0,
    num_levels: int = 1,
    prev_active_size: int = None,
    curr_active_size: int = None,
    level_s=None,
    level_t=None,
) -> bool:
    del level_idx, num_levels, prev_active_size, curr_active_size
    if not lp_result.get("success"):
        return False

    primal_flag = False
    dual_flag = False
    gap_flag = False

    primal_val = lp_result.get("primal_infeas_ori")
    if primal_val is None and "x" in lp_result and lp_result["x"] is not None and level_s is not None and level_t is not None:
        primal_val = compute_primal_infeas(solver, lp_result["x"], level_s, level_t)
    if primal_val is not None and np.isfinite(primal_val):
        primal_val = float(primal_val)
        primal_flag = abs(primal_val) <= inf_thr
        if not primal_flag:
            tree_log(solver, f"    [Convergence] Primal infeasibility: {primal_val:.2e} > {inf_thr:.2e}")

    dual_val = lp_result.get("dual_feas")
    if dual_val is not None:
        dual_flag = abs(dual_val) <= inf_thr
        if not dual_flag:
            tree_log(solver, f"    [Convergence] Dual feasibility: {dual_val:.2e} > {inf_thr:.2e}")

    gap_val = lp_result.get("gap")
    if gap_val is not None:
        gap_flag = abs(gap_val) <= inf_thr
        if not gap_flag:
            tree_log(solver, f"    [Convergence] Duality gap: {gap_val:.2e} > {inf_thr:.2e}")

    primal_str = f"{primal_val:.2e}" if primal_val is not None else "N/A"
    dual_str = f"{dual_val:.2e}" if dual_val is not None else "N/A"
    gap_str = f"{gap_val:.2e}" if gap_val is not None else "N/A"

    tree_log(solver, f"    [Convergence Check] primal={primal_str}, dual={dual_str}, gap={gap_str}, thr={inf_thr:.2e}")
    tree_log(solver, f"    [Convergence Flags] primal={primal_flag}, dual={dual_flag}, gap={gap_flag}")

    if primal_flag and dual_flag and gap_flag:
        tree_log(solver, "    [Convergence] HALO 风格三重可行性满足!")
        return True

    return False


def should_stop_iteration(
    solver,
    *,
    lp_result: Dict[str, Any],
    state: Dict[str, Any],
    level_s: Any,
    level_t: Any,
    max_inner_iter: int,
    is_coarsest: bool,
) -> bool:
    if is_coarsest:
        tree_log(solver, "  [Coarsest] single LP iteration completed, moving to next level")
        return True

    primal_ori = lp_result.get("primal_infeas_ori")
    if primal_ori is None and lp_result.get("x") is not None:
        primal_ori = compute_primal_infeas(solver, lp_result["x"], level_s, level_t)
    if primal_ori is not None and np.isfinite(primal_ori) and float(primal_ori) > 1.0:
        tree_log(
            solver,
            f"  [Stop] primal infeasibility too large: {float(primal_ori):.2e} > 1.0"
        )
        return True

    converged = check_tree_convergence(
        solver,
        lp_result,
        state["inf_thr"],
        level_idx=state["level_idx"],
        num_levels=solver.hierarchy_s.num_levels,
        prev_active_size=state["prev_active_size"],
        curr_active_size=None,
        level_s=level_s,
        level_t=level_t,
    )
    if converged:
        tree_log(solver, f"  Converged after {state['current_iter'] + 1} iterations")
        return True

    if state["current_iter"] >= max_inner_iter:
        tree_log(solver, f"  Max inner iterations reached ({max_inner_iter})")
        return True
    return False


def should_stop_inner(problem_def, algorithm_state, level_state, step_result) -> bool:
    t0 = time.perf_counter()
    result = should_stop_iteration(
        problem_def.solver,
        lp_result=step_result.data["lp_result"],
        state=level_state.data,
        level_s=level_state.level_s,
        level_t=level_state.level_t,
        max_inner_iter=problem_def.max_inner_iter,
        is_coarsest=(level_state.level_index == algorithm_state.run_state["num_levels"]),
    )
    problem_def.solver._profiler.add_components({COMPONENT_CONVERGENCE_CHECK: time.perf_counter() - t0})
    return bool(result)
