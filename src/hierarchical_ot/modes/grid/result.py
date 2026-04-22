from __future__ import annotations

import time

import numpy as np

from ...core.result_utils import build_objective_history_by_level


def record_level(problem_def, algorithm_state, level_state) -> None:
    del algorithm_state
    solver = problem_def.solver
    x_solution = level_state.mode_state["x_solution_last"]
    y_solution = level_state.mode_state["y_solution_last"]
    history = list(level_state.mode_state.get("level_obj_hist", []))
    support_final = int(len(np.asarray(y_solution.get("keep", []), dtype=np.int64)))
    solver.solutions[level_state.level_index] = {
        "primal": y_solution,
        "dual": x_solution,
        "history": history,
    }
    solver.level_summaries.append(
        {
            "level": int(level_state.level_index),
            "n_source": int(len(level_state.level_s.points)),
            "n_target": int(len(level_state.level_t.points)),
            "iters": int(level_state.completed_iters),
            "time": float(time.perf_counter() - level_state.t_level_start),
            "lp_time": float(level_state.level_lp_time),
            "pricing_time": float(level_state.level_pricing_time),
            "objective": history[-1] if history else None,
            "support_pre_lp_final": (
                int(level_state.mode_state["last_active_pre_lp"])
                if level_state.mode_state.get("last_active_pre_lp") is not None
                else None
            ),
            "support_final": support_final,
            "stop_reason": level_state.mode_state.get("stop_reason"),
        }
    )


def advance_level(problem_def, algorithm_state, level_state) -> None:
    algorithm_state.run_state["x_solution_last"] = level_state.mode_state.get("x_solution_last")
    algorithm_state.run_state["y_solution_last"] = level_state.mode_state.get("y_solution_last")
    algorithm_state.run_state["keep_last"] = problem_def.solver.keep_last


def package_result(problem_def, algorithm_state):
    del algorithm_state
    solver = problem_def.solver
    final_sol = solver.solutions[0]
    level_summaries = list(getattr(solver, "level_summaries", []))
    return {
        "primal": final_sol.get("primal"),
        "dual": final_sol.get("dual"),
        "all_history": solver.solutions,
        "objective_history_by_level": build_objective_history_by_level(solver.solutions),
        "level_summaries": level_summaries,
        "grid_lp_diags": list(getattr(solver, "grid_lp_diags", [])),
        "grid_iter_snapshots": list(getattr(solver, "grid_iter_snapshots", [])),
        "lp_solve_time_total": solver._sum_level_summary_metric(
            level_summaries,
            "lp_time",
        ),
        "active_support_sizes": list(getattr(solver, "active_support_sizes", [])),
        "dual_stats": list(getattr(solver, "dual_stats", [])),
        "stop_reasons": list(getattr(solver, "stop_reasons", [])),
        "solver_mode": "grid",
    }
