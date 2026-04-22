from __future__ import annotations

import time
from typing import Any, Dict, Tuple

from scipy.sparse import csc_matrix

from ...core.result_utils import build_objective_history_by_level
from ...types.runtime import LevelState
from . import shielding as grid_shielding


def should_exact_flat(solver) -> bool:
    return solver.hierarchy_s.num_levels <= 1


def solve_exact_flat(solver, tolerance):
    from ..cluster.solve_lp import solve_lp as cluster_solve_lp

    level_s = solver.hierarchy_s.finest_level
    level_t = solver.hierarchy_t.finest_level
    level_s_lp = grid_shielding.grid_level_for_dual_lp(level_s)
    level_t_lp = grid_shielding.grid_level_for_dual_lp(level_t)

    solver.active_support = None
    t_lp_start = time.perf_counter()
    with solver._profiler.timer("lp", gpu_possible=True):
        primal_dense, dual_sol, result, solve_meta = cluster_solve_lp(
            solver,
            level_s_lp,
            level_t_lp,
            tolerance,
            verbose=solver.lp_solver_verbose,
        )
    lp_solve_time_total = float(time.perf_counter() - t_lp_start)
    solver._profiler.add_components((solve_meta or {}).get("components"))
    if not result.success:
        return None

    primal_coo = primal_dense.tocoo()
    mask = primal_coo.data > 1e-12
    primal_sparse = csc_matrix(
        (primal_coo.data[mask], (primal_coo.row[mask], primal_coo.col[mask])),
        shape=primal_dense.shape,
    )
    level_summary = {
        "level": 0,
        "n_source": int(primal_dense.shape[0]),
        "n_target": int(primal_dense.shape[1]),
        "iters": 1,
        "time": lp_solve_time_total,
        "objective": float(result.obj_val),
        "lp_time": lp_solve_time_total,
        "pricing_time": 0.0,
        "support_pre_lp_final": None,
        "support_final": int(primal_sparse.nnz),
        "stop_reason": None,
    }
    solutions = {0: {"history": [result.obj_val]}}
    return {
        "primal": primal_sparse,
        "dual": dual_sol,
        "final_obj": float(result.obj_val),
        "all_history": solutions,
        "objective_history_by_level": build_objective_history_by_level(solutions),
        "sparse_coupling": {
            "rows": primal_coo.row[mask],
            "cols": primal_coo.col[mask],
            "values": primal_coo.data[mask],
        },
        "level_summaries": [level_summary],
        "lp_solve_time_total": lp_solve_time_total,
        "grid_lp_diags": [],
        "grid_iter_snapshots": [],
        "active_support_sizes": [],
        "dual_stats": [],
        "stop_reasons": [],
    }

def prepare_level_tolerances(
    tolerance: Dict[str, float],
    level_idx: int,
) -> Tuple[float, float]:
    if isinstance(tolerance, dict):
        inf_thrs = tolerance.get("inf_thrs", None)
        stop_thrs = tolerance.get("stop_thrs", None)

        if inf_thrs is not None and isinstance(inf_thrs, (list, tuple)):
            inf_thr = inf_thrs[level_idx] if level_idx < len(inf_thrs) else inf_thrs[-1]
        else:
            inf_thr = tolerance.get("primal", 1e-6)

        if stop_thrs is not None and isinstance(stop_thrs, (list, tuple)):
            stop_thr = stop_thrs[level_idx] if level_idx < len(stop_thrs) else stop_thrs[-1]
        else:
            stop_thr = tolerance.get("objective", 1e-6)
    else:
        inf_thr = tolerance if isinstance(tolerance, (int, float)) else 1e-6
        stop_thr = tolerance if isinstance(tolerance, (int, float)) else 1e-6
    return float(inf_thr), float(stop_thr)

def init_run_state(solver, tolerance, **kwargs):
    check_method = str(kwargs.get("grid_check_type", "gpu_exact")).lower()
    if check_method in {"gpu_exact", "gpu", "gpu_full"}:
        check_method = "gpu"
    elif check_method in {"gpu_sampled", "sampled", "approx"}:
        check_method = "gpu_approx"
    elif check_method not in {"cpu", "gpu", "gpu_approx", "auto"}:
        check_method = "gpu"

    solver.solutions = {}
    solver.level_summaries = []
    solver.grid_lp_diags = []
    solver.grid_iter_snapshots = []
    solver.active_support_sizes = []
    solver.dual_stats = []
    solver.stop_reasons = []
    solver.keep_last = None
    solver.active_support = None
    solver._lp_solver_kwargs = {
        "save_info": 2,
        "termination_evaluation_frequency": 200,
    }

    run_state = {
        "coarsest_idx": solver.hierarchy_s.num_levels - 1,
        "tolerance": tolerance,
        "stop_tolerance": kwargs.get("stop_tolerance", tolerance),
        "p": int(kwargs.get("grid_p", 2)),
        "use_last": bool(kwargs.get("use_last", True)),
        "use_last_after_inner0": bool(kwargs.get("use_last_after_inner0", False)),
        "if_check": bool(kwargs.get("if_check", True)),
        "if_shield": bool(kwargs.get("if_shield", True)),
        "coarsest_full_support": bool(kwargs.get("coarsest_full_support", True)),
        "vd_thr": float(kwargs.get("vd_thr", 0.0625)),
        "check_method": check_method,
        "use_primal_feas_ori": bool(kwargs.get("use_primal_feas_ori", True)),
        "adap_primal_tol": bool(kwargs.get("adap_primal_tol", True)),
        "new_mvp": bool(kwargs.get("new_mvp", True)),
        "aty_type": int(kwargs.get("aty_type", 0)),
        "repair_coverage": bool(kwargs.get("repair_coverage", False)),
        "x_solution_last": None,
        "y_solution_last": None,
    }
    run_state["finest_idx"] = 0
    run_state["coarsest_idx"] = int(run_state.get("coarsest_idx", solver.hierarchy_s.num_levels - 1))
    return run_state


def get_level_indices(_solver, run_state):
    return range(run_state["coarsest_idx"], run_state["finest_idx"] - 1, -1)


def get_max_inner_iters(_solver, level_state, max_inner_iter: int) -> int:
    if level_state.get("is_coarsest"):
        return 1
    return max_inner_iter


def prepare_inner(solver, *, level_state: Dict[str, Any], run_state: Dict[str, Any], inner_iter: int) -> Dict[str, Any]:
    if inner_iter == 0:
        return grid_shielding.build_active_set_first_iter(
            solver,
            level_state=level_state,
            run_state=run_state,
        )
    return grid_shielding.build_active_set_subsequent_iter(
        solver,
        level_state=level_state,
        run_state=run_state,
    )


def init_level_state(solver, level_idx: int, run_state: Dict[str, Any]):
    level_s = solver.hierarchy_s.levels[level_idx]
    level_t = solver.hierarchy_t.levels[level_idx]
    n_s = len(level_s.points)
    n_t = len(level_t.points)
    inf_thr, stop_thr = prepare_level_tolerances(
        run_state.get("stop_tolerance", run_state["tolerance"]),
        level_idx,
    )

    state = {
        "level_idx": int(level_idx),
        "is_coarsest": int(level_idx) == int(run_state["coarsest_idx"]),
        "level_s": level_s,
        "level_t": level_t,
        "level_s_lp": grid_shielding.grid_level_for_dual_lp(level_s),
        "level_t_lp": grid_shielding.grid_level_for_dual_lp(level_t),
        "t_level_start": time.perf_counter(),
        "current_iter": 0,
        "completed_iters": 0,
        "level_obj_hist": [],
        "inf_thr": float(inf_thr),
        "stop_thr": float(stop_thr),
        "prev_PrimalFeas": None,
        "x_solution_last": run_state.get("x_solution_last"),
        "y_solution_last": run_state.get("y_solution_last"),
        "curr_active_size": int(n_s * n_t) if n_s > 0 and n_t > 0 else 0,
        "last_active_pre_lp": None,
        "level_lp_time": 0.0,
        "level_pricing_time": 0.0,
        "stop_reason": None,
    }

    prep = prepare_inner(solver, level_state=state, run_state=run_state, inner_iter=0)
    state.update(prep)
    state["curr_active_size"] = int(len(prep["keep"]))
    state["last_active_pre_lp"] = int(len(prep["keep"]))
    return state


def initial(problem_def, algorithm_state, level_index: int):
    level_data = init_level_state(problem_def.solver, level_index, algorithm_state.run_state)
    return LevelState.from_legacy_data(level_index=level_index, max_inner_iter=0, data=level_data)
