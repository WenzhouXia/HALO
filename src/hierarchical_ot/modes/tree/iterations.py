from __future__ import annotations

import time

import numpy as np

from ...core.solver_utils import decode_keep_1d_to_struct, remap_duals_for_warm_start
from ...types.runtime import StepResult
from ...instrumentation.phase_names import (
    COMPONENT_CONVERGENCE_CHECK,
    COMPONENT_KEEP_COORD,
    COMPONENT_KEEP_UNION,
    COMPONENT_PRICING_TOTAL,
    COMPONENT_REMAP_Y,
    COMPONENT_STATE_UPDATE,
    COMPONENT_VIOLATION_CHECK,
    COMPONENT_SHIELDING,
)
from .logger import tree_log
from .shielding import merge_with_use_last
from .solve_lp import tree_solve_lp_pack
from .violation_check import apply_violation_check


def prepare_inner(problem_def, algorithm_state, level_state) -> None:
    solver = problem_def.solver
    if level_state.data.current_iter == 0:
        tree_log(solver, f"  DEBUG: 开始 while 循环, level_idx={level_state.level_index}")
    tree_log(solver, f"  DEBUG: Inner Iteration {level_state.data.current_iter}")


def solve_lp(problem_def, algorithm_state, level_state) -> StepResult:
    step_data = tree_solve_lp_pack(
        problem_def.solver,
        level_idx=level_state.level_index,
        level_s=level_state.level_s,
        level_t=level_state.level_t,
        keep=level_state.mode_state["keep"],
        keep_coord=level_state.mode_state["keep_coord"],
        x_init=level_state.mode_state["x_init"],
        y_init=level_state.mode_state["y_init"],
        stop_thr=level_state.mode_state["stop_thr"],
        inner_iter=level_state.current_iter,
        prev_PrimalFeas=level_state.mode_state.get("prev_PrimalFeas"),
        tree_debug=algorithm_state.run_state["tree_debug"],
        tree_infeas_fallback=algorithm_state.run_state["tree_infeas_fallback"],
    )
    return StepResult(
        success=bool(step_data.get("success", False)),
        data=step_data,
        objective=extract_step_objective(step_data),
    )


def extract_step_objective(step_pack) -> float | None:
    lp_result = step_pack.get("lp_result", {}) if isinstance(step_pack, dict) else {}
    objective = lp_result.get("obj")
    if objective is None:
        return None
    return float(objective)


def _update_active_set(solver, level_state, run_state, step_pack):
    level_idx = level_state["level_idx"]
    if level_idx == run_state["num_levels"]:
        return {"pricing_time": 0.0, "components": {}}

    level_s = level_state["level_s"]
    level_t = level_state["level_t"]
    n_t = len(level_t.points)
    pricing_start = time.perf_counter()
    component_times = {}
    y_last = step_pack["lp_result"].get("y")
    keep_last = level_state["keep"]
    if y_last is None:
        y_keep = np.array([], dtype=np.int64)
        y_vals = np.array([], dtype=np.float32)
    else:
        nonzero_mask = np.abs(y_last) > 1e-20
        y_keep = keep_last[nonzero_mask]
        y_vals = y_last[nonzero_mask]

    t_shield = time.perf_counter()
    update_result = solver.strategy.update_active_support(
        x_solution=level_state["x_solution_last"],
        y_solution_last={"y": y_vals, "keep": y_keep},
        level_s=level_s,
        level_t=level_t,
        hierarchy_s=solver.hierarchy_s,
        hierarchy_t=solver.hierarchy_t,
        build_aux=False,
    )
    shield_total = time.perf_counter() - t_shield
    keep_union_time = 0.0
    if isinstance(update_result, dict):
        timing = update_result.get("timing", {}) or {}
        keep_union_time = float(timing.get("keep_union", 0.0) or 0.0)
    if keep_union_time > 0.0:
        component_times[COMPONENT_KEEP_UNION] = component_times.get(COMPONENT_KEEP_UNION, 0.0) + keep_union_time
    component_times[COMPONENT_SHIELDING] = max(0.0, shield_total - keep_union_time)
    keep_next = np.asarray(update_result["keep"], dtype=np.int64)
    level_state["_trace_keep_after_shield"] = int(len(keep_next))

    t_vcheck = time.perf_counter()
    keep_next = apply_violation_check(
        solver,
        x_dual=level_state["x_solution_last"],
        level_s=level_s,
        level_t=level_t,
        keep=keep_next,
        cost_type=level_state["cost_type"],
        ifcheck=run_state["ifcheck"],
        vd_thr=run_state["vd_thr"],
        check_method=run_state["check_method"],
    )
    component_times[COMPONENT_VIOLATION_CHECK] = time.perf_counter() - t_vcheck
    level_state["_trace_keep_after_check"] = int(len(keep_next))

    t_keep_union = time.perf_counter()
    keep_next = merge_with_use_last(
        solver,
        keep=keep_next,
        use_last=run_state["use_last"],
        use_last_after_inner0=run_state["use_last_after_inner0"],
        inner_iter=level_state["current_iter"] + 1,
        n_t=n_t,
    )
    level_state["_trace_keep_after_uselast"] = int(len(keep_next))
    component_times[COMPONENT_KEEP_UNION] = component_times.get(COMPONENT_KEEP_UNION, 0.0) + (time.perf_counter() - t_keep_union)

    t_keep_coord = time.perf_counter()
    keep_coord_next = decode_keep_1d_to_struct(keep_next, n_t)
    component_times[COMPONENT_KEEP_COORD] = time.perf_counter() - t_keep_coord
    y_init_next = None
    if y_last is not None:
        t_remap = time.perf_counter()
        y_init_next = {"y": remap_duals_for_warm_start({"y": y_vals, "keep": y_keep}, keep_next)}
        component_times[COMPONENT_REMAP_Y] = time.perf_counter() - t_remap

    level_state["keep"] = keep_next
    level_state["keep_coord"] = keep_coord_next
    level_state["x_init"] = level_state["x_solution_last"]
    level_state["y_init"] = y_init_next
    level_state["curr_active_size"] = len(keep_next)
    pricing_time = time.perf_counter() - pricing_start
    component_times[COMPONENT_PRICING_TOTAL] = pricing_time
    return {"pricing_time": pricing_time, "components": component_times}


def finalize_iteration(solver, level_state, run_state, step_pack) -> None:
    lp_result = step_pack["lp_result"]
    level_state["x_solution_last"] = lp_result["x"]
    level_state["y_solution_last"] = {
        "y": lp_result["y"],
        "keep": step_pack["keep"],
    }
    if lp_result.get("primal_feas") is not None:
        level_state["prev_PrimalFeas"] = lp_result["primal_feas"]
    if "level_obj_hist" not in level_state:
        level_state["level_obj_hist"] = []
    if lp_result.get("obj") is not None:
        level_state["level_obj_hist"].append(float(lp_result["obj"]))

    update_pack = _update_active_set(solver, level_state, run_state, step_pack)
    level_state["level_lp_time"] = level_state.get("level_lp_time", 0.0) + float(step_pack.get("lp_time", 0.0))
    level_state["level_pricing_time"] = level_state.get("level_pricing_time", 0.0) + float(update_pack.get("pricing_time", 0.0))
    solver._profiler.add_components(update_pack.get("components"))
    pre_lp_active = level_state.get("_pre_lp_active")
    if pre_lp_active is not None:
        level_state["prev_active_size"] = pre_lp_active
        level_state["last_active_pre_lp"] = pre_lp_active
