from __future__ import annotations

import numpy as np

from ...core.result_utils import build_objective_history_by_level


def _build_sparse_coupling_from_primal(primal: dict | None, *, n_target: int) -> dict[str, np.ndarray] | None:
    if not isinstance(primal, dict):
        return None
    keep = primal.get("keep")
    values = primal.get("y")
    if keep is None or values is None:
        return None

    keep_arr = np.asarray(keep, dtype=np.int64).ravel()
    values_arr = np.asarray(values, dtype=np.float32).ravel()
    if keep_arr.shape != values_arr.shape:
        raise ValueError(
            "Tree primal keep/y shape mismatch when packaging sparse coupling: "
            f"{keep_arr.shape} != {values_arr.shape}"
        )
    if keep_arr.size == 0:
        return None

    nonzero_mask = np.abs(values_arr) > 1e-12
    if not np.any(nonzero_mask):
        return None

    keep_nz = keep_arr[nonzero_mask]
    values_nz = np.abs(values_arr[nonzero_mask])
    rows = (keep_nz // int(n_target)).astype(np.int32, copy=False)
    cols = (keep_nz % int(n_target)).astype(np.int32, copy=False)
    return {
        "rows": rows,
        "cols": cols,
        "values": values_nz.astype(np.float32, copy=False),
    }


def record_level(problem_def, algorithm_state, level_state) -> None:
    del algorithm_state
    solver = problem_def.solver
    x_solution = level_state.mode_state["x_solution_last"]
    y_solution = level_state.mode_state["y_solution_last"]
    history = list(level_state.mode_state.get("level_obj_hist", []))
    if not hasattr(solver, "level_summaries"):
        solver.level_summaries = []
    solver.solutions[level_state.level_index] = {
        "dual": x_solution,
        "primal": y_solution,
        "history": history,
    }
    solver.level_summaries.append(
        {
            "level": level_state.level_index,
            "n_source": len(level_state.level_s.points),
            "n_target": len(level_state.level_t.points),
            "iters": int(level_state.completed_iters),
            "time": problem_def.solver._time_now() - level_state.t_level_start,
            "lp_time": float(level_state.level_lp_time),
            "pricing_time": float(level_state.level_pricing_time),
            "objective": history[-1] if history else None,
            "support_pre_lp_final": (
                int(level_state.mode_state["last_active_pre_lp"])
                if level_state.mode_state.get("last_active_pre_lp") is not None
                else None
            ),
            "support_final": len(y_solution["y"]) if y_solution else 0,
            "stop_reason": level_state.mode_state.get("stop_reason"),
        }
    )


def advance_level(problem_def, algorithm_state, level_state) -> None:
    algorithm_state.run_state["x_solution_last"] = level_state.mode_state["x_solution_last"]
    algorithm_state.run_state["y_solution_last"] = level_state.mode_state["y_solution_last"]


def package_result(problem_def, algorithm_state):
    del algorithm_state
    solver = problem_def.solver
    final_sol = solver.solutions[0]
    final_obj = 0.0
    level_0 = solver.hierarchy_s.levels[0]
    level_0_t = solver.hierarchy_t.levels[0]
    primal = final_sol.get("primal")
    sparse_coupling = _build_sparse_coupling_from_primal(
        primal,
        n_target=len(level_0_t.points),
    )
    cost_type_raw = str(getattr(solver, "_cost_type", "l2^2")).strip().lower()
    if cost_type_raw in {"l2^2", "sqeuclidean", "l2sq"}:
        cost_type = "L2"
    elif cost_type_raw == "l1":
        cost_type = "L1"
    elif cost_type_raw == "linf":
        cost_type = "LINF"
    else:
        cost_type = "L2"

    if primal is not None and "y" in primal and "keep" in primal:
        y_vals = primal["y"]
        keep = primal["keep"]
        n_t = len(level_0_t.points)
        for idx, y_val in zip(keep, y_vals):
            flow = abs(float(y_val))
            if flow <= 0.0:
                continue
            i = idx // n_t
            j = idx % n_t
            diff = level_0.points[i] - level_0_t.points[j]
            if cost_type == "L2":
                cost = np.sum(diff ** 2)
            elif cost_type == "L1":
                cost = np.sum(np.abs(diff))
            elif cost_type == "LINF":
                cost = np.max(np.abs(diff))
            else:
                cost = np.sum(diff ** 2)
            final_obj += flow * cost

    return {
        "primal": final_sol.get("primal"),
        "dual": final_sol.get("dual"),
        "final_obj": final_obj,
        "all_history": solver.solutions,
        "objective_history_by_level": build_objective_history_by_level(solver.solutions),
        "sparse_coupling": sparse_coupling,
        "level_summaries": solver.level_summaries,
        "lp_solve_time_total": float(
            sum(float(item.get("lp_time", 0.0)) for item in solver.level_summaries if isinstance(item, dict))
        ),
        "tree_lp_diags": getattr(solver, "tree_lp_diags", []),
    }
