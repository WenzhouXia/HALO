from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np

from ...instrumentation.phase_names import COMPONENT_CONVERGENCE_CHECK


def should_stop_iteration(
    solver,
    level_state: Dict[str, Any],
    run_state: Dict[str, Any],
    max_inner_iter: int,
    step_pack: Dict[str, Any],
) -> bool:
    del run_state
    if level_state.get("is_coarsest"):
        level_state["stop_reason"] = "coarsest_single_lp"
        solver.stop_reasons.append({"level": int(level_state["level_idx"]), "reason": level_state["stop_reason"]})
        return True

    lp_result = step_pack["lp_result"]
    primal_ori = lp_result.get("primal_infeas_ori")
    dual_feas = lp_result.get("dual_feas")
    gap = lp_result.get("gap")
    solver.dual_stats.append(
        {
            "level": int(level_state["level_idx"]),
            "iter": int(level_state["current_iter"]),
            "primal_feas": None if primal_ori is None else float(primal_ori),
            "dual_feas": None if dual_feas is None else float(dual_feas),
            "gap": None if gap is None else float(gap),
        }
    )
    inf_thr = float(level_state["inf_thr"])
    primal_flag = primal_ori is not None and np.isfinite(primal_ori) and abs(float(primal_ori)) <= inf_thr
    dual_flag = dual_feas is not None and np.isfinite(dual_feas) and abs(float(dual_feas)) <= inf_thr
    gap_flag = gap is not None and np.isfinite(gap) and abs(float(gap)) <= inf_thr
    if primal_flag and dual_flag and gap_flag:
        level_state["stop_reason"] = "mgpd_mainline_converged"
        solver.stop_reasons.append({"level": int(level_state["level_idx"]), "reason": level_state["stop_reason"]})
        return True
    if primal_ori is not None and float(primal_ori) > 1.0:
        level_state["stop_reason"] = "primal_infeas_too_large"
        solver.stop_reasons.append({"level": int(level_state["level_idx"]), "reason": level_state["stop_reason"]})
        return True
    if level_state["current_iter"] + 1 >= max_inner_iter:
        level_state["stop_reason"] = "max_inner_iter"
        solver.stop_reasons.append({"level": int(level_state["level_idx"]), "reason": level_state["stop_reason"]})
        return True
    return False


def should_stop_inner(problem_def, algorithm_state, level_state, step_result) -> bool:
    t0 = time.perf_counter()
    result = should_stop_iteration(
        problem_def.solver,
        level_state.data,
        algorithm_state.run_state,
        problem_def.max_inner_iter,
        step_result.data,
    )
    problem_def.solver._profiler.add_components({COMPONENT_CONVERGENCE_CHECK: time.perf_counter() - t0})
    return bool(result)
