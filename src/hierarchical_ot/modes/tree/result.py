from __future__ import annotations

from contextlib import nullcontext
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


def _compute_final_obj_from_sparse_coupling(
    sparse_coupling: dict[str, np.ndarray] | None,
    *,
    source_points: np.ndarray,
    target_points: np.ndarray,
    cost_type: str,
    p: float,
) -> float:
    if not isinstance(sparse_coupling, dict):
        return 0.0

    rows = np.asarray(sparse_coupling.get("rows", []), dtype=np.int64).ravel()
    cols = np.asarray(sparse_coupling.get("cols", []), dtype=np.int64).ravel()
    values = np.asarray(sparse_coupling.get("values", []), dtype=np.float64).ravel()
    if rows.size == 0 or cols.size == 0 or values.size == 0:
        return 0.0
    if rows.shape != cols.shape or rows.shape != values.shape:
        raise ValueError(
            "Tree sparse coupling rows/cols/values shape mismatch when computing final_obj: "
            f"{rows.shape}, {cols.shape}, {values.shape}"
        )

    src = np.asarray(source_points, dtype=np.float64)[rows]
    tgt = np.asarray(target_points, dtype=np.float64)[cols]
    diff = src - tgt

    cost_name = str(cost_type).strip().lower()
    if cost_name in {"l2^2", "sqeuclidean", "l2sq"}:
        costs = np.einsum("ij,ij->i", diff, diff, optimize=True)
    elif cost_name == "l2":
        costs = np.sqrt(np.einsum("ij,ij->i", diff, diff, optimize=True))
    elif cost_name == "l1":
        costs = np.sum(np.abs(diff), axis=1)
    elif cost_name == "linf":
        costs = np.max(np.abs(diff), axis=1)
    elif cost_name == "lp":
        costs = np.sum(np.abs(diff) ** float(p), axis=1)
    else:
        costs = np.einsum("ij,ij->i", diff, diff, optimize=True)

    return float(np.dot(values, costs.astype(np.float64, copy=False)))


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
    trace_collector = getattr(problem_def, "trace_collector", None)
    trace_prefix = str(getattr(problem_def, "trace_prefix", "solve_ot"))

    def _trace_span(name: str, args=None):
        if trace_collector is None:
            return nullcontext()
        return trace_collector.span(
            f"{trace_prefix}.solve.package_result.{name}",
            "solve_ot",
            args=dict(args or {}),
        )

    with _trace_span("final_solution"):
        final_sol = solver.solutions[0]
        level_0 = solver.hierarchy_s.levels[0]
        level_0_t = solver.hierarchy_t.levels[0]
        primal = final_sol.get("primal")

    with _trace_span("sparse_coupling"):
        sparse_coupling = _build_sparse_coupling_from_primal(
            primal,
            n_target=len(level_0_t.points),
        )

    with _trace_span("final_obj"):
        final_obj = _compute_final_obj_from_sparse_coupling(
            sparse_coupling,
            source_points=np.asarray(level_0.points, dtype=np.float64),
            target_points=np.asarray(level_0_t.points, dtype=np.float64),
            cost_type=str(getattr(solver, "_cost_type", "l2^2")),
            p=float(getattr(solver, "_cost_p", 2.0)),
        )

    with _trace_span("objective_history"):
        objective_history_by_level = build_objective_history_by_level(solver.solutions)

    with _trace_span("lp_time_total"):
        lp_solve_time_total = float(
            sum(float(item.get("lp_time", 0.0)) for item in solver.level_summaries if isinstance(item, dict))
        )

    with _trace_span("payload"):
        return {
            "primal": final_sol.get("primal"),
            "dual": final_sol.get("dual"),
            "final_obj": final_obj,
            "all_history": solver.solutions,
            "objective_history_by_level": objective_history_by_level,
            "sparse_coupling": sparse_coupling,
            "level_summaries": solver.level_summaries,
            "lp_solve_time_total": lp_solve_time_total,
            "tree_lp_diags": getattr(solver, "tree_lp_diags", []),
        }
