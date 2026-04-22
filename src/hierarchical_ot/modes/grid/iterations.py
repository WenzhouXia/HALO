from __future__ import annotations

import numpy as np

from ...types.runtime import StepResult
from ...instrumentation.phase_names import COMPONENT_STATE_UPDATE
from . import initial as grid_initial
from .solve_lp import grid_solve_lp_pack


def prepare_inner(problem_def, algorithm_state, level_state) -> None:
    solver = problem_def.solver
    if level_state.data.current_iter <= 0:
        prep = {
            "keep": level_state.data.get("keep", np.empty(0, dtype=np.int64)),
            "prepare_breakdown": level_state.data.get("prepare_breakdown", {}),
            "prepare_time_total": level_state.data.get("prepare_time_total", 0.0),
        }
    else:
        prep = grid_initial.prepare_inner(
            solver,
            level_state=level_state.data,
            run_state=algorithm_state.run_state,
            inner_iter=level_state.data.current_iter,
        )
        level_state.data.update(prep)
        level_state.curr_active_size = int(len(prep["keep"]))
        level_state.mode_state["last_active_pre_lp"] = int(len(prep["keep"]))
    prepare_time = float(prep.get("prepare_time_total", 0.0))
    if prepare_time > 0.0:
        level_state.level_pricing_time = level_state.level_pricing_time + prepare_time


def solve_iteration_lp(solver, level_state, run_state):
    return grid_solve_lp_pack(solver, level_state, run_state)


def solve_lp(problem_def, algorithm_state, level_state) -> StepResult:
    step_data = solve_iteration_lp(problem_def.solver, level_state.data, algorithm_state.run_state)
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


def finalize_iteration(solver, level_state, _run_state, step_pack) -> None:
    lp_result = step_pack["lp_result"]
    solved_keep = np.asarray(
        step_pack.get("keep", level_state.get("keep", [])),
        dtype=np.int64,
    )
    y_vals = np.asarray(lp_result["y"], dtype=np.float32)
    level_state["x_solution_last"] = lp_result["x"]
    level_state["y_solution_last"] = {
        "y": y_vals,
        "keep": solved_keep,
    }
    if lp_result.get("primal_feas") is not None:
        level_state["prev_PrimalFeas"] = lp_result["primal_feas"]
    elif lp_result.get("primal_infeas_ori") is not None:
        level_state["prev_PrimalFeas"] = lp_result["primal_infeas_ori"]
    level_state["level_obj_hist"].append(float(lp_result.get("obj", 0.0)))
    level_state["level_lp_time"] += step_pack["lp_time"]
    level_state["level_pricing_time"] += float(step_pack.get("pricing_time", 0.0))
    solver._profiler.add_components({COMPONENT_STATE_UPDATE: 0.0})
    pre_lp_active = level_state.get("_pre_lp_active")
    if pre_lp_active is not None:
        level_state["prev_active_size"] = pre_lp_active
        level_state["last_active_pre_lp"] = pre_lp_active
