from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, Tuple

import numpy as np

from ...core.result_utils import build_objective_history_by_level
from ...core.solver_utils import decode_keep_1d_to_struct, remap_duals_for_warm_start
from ...types.runtime import LevelState
from .logger import tree_log
from .solve_lp import _solve_lp
from .shielding import build_active_set_first_iter, build_active_set_subsequent_iter
from .trace_utils import tree_trace_span


def _build_sparse_coupling(keep: np.ndarray, values: np.ndarray, *, n_target: int) -> Dict[str, np.ndarray] | None:
    keep_arr = np.asarray(keep, dtype=np.int64).ravel()
    values_arr = np.asarray(values, dtype=np.float32).ravel()
    if keep_arr.shape != values_arr.shape:
        raise ValueError(
            "Tree keep/y shape mismatch when packaging exact-flat sparse coupling: "
            f"{keep_arr.shape} != {values_arr.shape}"
        )
    if keep_arr.size == 0:
        return None

    nonzero_mask = np.abs(values_arr) > 1e-12
    if not np.any(nonzero_mask):
        return None

    keep_nz = keep_arr[nonzero_mask]
    values_nz = np.abs(values_arr[nonzero_mask])
    return {
        "rows": (keep_nz // int(n_target)).astype(np.int32, copy=False),
        "cols": (keep_nz % int(n_target)).astype(np.int32, copy=False),
        "values": values_nz.astype(np.float32, copy=False),
    }


def should_exact_flat(solver) -> bool:
    finest_s = solver.hierarchy_s.finest_level
    finest_t = solver.hierarchy_t.finest_level
    return int(len(finest_s.points)) <= 256 and int(len(finest_t.points)) <= 256


def solve_exact_flat(solver, tolerance, *, trace_collector=None, trace_prefix: str = "solve_ot"):
    level_s = solver.hierarchy_s.finest_level
    level_t = solver.hierarchy_t.finest_level
    n_s = int(len(level_s.points))
    n_t = int(len(level_t.points))
    keep = np.arange(n_s * n_t, dtype=np.int64)
    keep_coord = decode_keep_1d_to_struct(keep, n_t)

    solver.keep_last = None
    solver.solutions = {}
    solver.level_summaries = []
    solver.tree_lp_diags = []

    stop_thr = float(tolerance.get("objective", 1e-6)) if isinstance(tolerance, dict) else float(tolerance)
    x_init = np.zeros(n_s + n_t, dtype=np.float64)
    y_init = np.zeros(len(keep), dtype=np.float64)

    with tree_trace_span(
        trace_collector,
        "tree.lp.single_level_exact",
        level_idx=0,
        inner_iter=0,
        stage="single_level_exact",
        args={"n_source": int(n_s), "n_target": int(n_t)},
    ):
        lp_result = _solve_lp(
            solver,
            level_s=level_s,
            level_t=level_t,
            keep_coord=keep_coord,
            x_init=x_init,
            y_init=y_init,
            stop_thr=stop_thr,
            inner_iter=0,
            prev_PrimalFeas=None,
            level_idx=0,
            is_coarsest=False,
            tree_debug=bool(getattr(solver, "tree_debug", False)),
            trace_collector=trace_collector,
            trace_prefix=trace_prefix,
            trace_context={
                "level_idx": 0,
                "inner_iter": 0,
                "stage": "single_level_exact",
            },
        )
    if lp_result.get("diag"):
        solver.tree_lp_diags.append(lp_result["diag"])
    if not lp_result.get("success", False):
        return None

    primal = {
        "y": lp_result["y"],
        "keep": keep,
    }
    sparse_coupling = _build_sparse_coupling(keep, lp_result["y"], n_target=n_t)
    dual = lp_result["x"]
    final_obj = 0.0
    y_vals = primal["y"]
    cost_type_raw = str(getattr(solver, "_cost_type", "l2^2")).strip().lower()
    if cost_type_raw in {"l2^2", "sqeuclidean", "l2sq"}:
        cost_type = "L2"
    elif cost_type_raw == "l1":
        cost_type = "L1"
    elif cost_type_raw == "linf":
        cost_type = "LINF"
    else:
        cost_type = "L2"
    for idx, y_val in zip(keep, y_vals):
        flow = abs(float(y_val))
        if flow <= 0.0:
            continue
        i = idx // n_t
        j = idx % n_t
        diff = level_s.points[i] - level_t.points[j]
        if cost_type == "L2":
            cost = float(np.sum(diff ** 2))
        elif cost_type == "L1":
            cost = float(np.sum(np.abs(diff)))
        elif cost_type == "LINF":
            cost = float(np.max(np.abs(diff)))
        else:
            cost = float(np.sum(diff ** 2))
        final_obj += flow * cost
    level_summary = {
        "level": 0,
        "n_source": n_s,
        "n_target": n_t,
        "iters": 1,
        "time": float(lp_result.get("diag", {}).get("backend_solve_wall_time", 0.0)),
        "lp_time": float(lp_result.get("diag", {}).get("backend_solve_wall_time", 0.0)),
        "pricing_time": 0.0,
        "objective": float(final_obj),
        "support_pre_lp_final": int(len(keep)),
        "support_final": int(len(keep)),
        "stop_reason": "single_level_exact",
    }
    solutions = {
        0: {
            "dual": dual,
            "primal": primal,
            "history": [float(final_obj)],
        }
    }
    solver.solutions = solutions
    solver.level_summaries = [level_summary]
    return {
        "primal": primal,
        "dual": dual,
        "final_obj": float(final_obj),
        "all_history": solutions,
        "objective_history_by_level": build_objective_history_by_level(solutions),
        "sparse_coupling": sparse_coupling,
        "level_summaries": solver.level_summaries,
        "lp_solve_time_total": float(level_summary["lp_time"]),
        "tree_lp_diags": solver.tree_lp_diags,
    }


def parse_bool_flag(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def normalize_tree_infeas_fallback(value: Any) -> str:
    fallback = str(value or "none").strip().lower()
    allowed = {"none"}
    if fallback not in allowed:
        raise ValueError(
            f"Unknown tree_infeas_fallback={value!r}, must be one of {sorted(allowed)}"
        )
    return fallback


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


def init_level_state_impl(
    solver,
    *,
    level_idx: int,
    inf_thr: float,
    stop_thr: float,
    x_solution_last,
    y_solution_last,
) -> Dict[str, Any]:
    solver.keep_last = None
    return {
        "level_idx": level_idx,
        "current_iter": 0,
        "completed_iters": 0,
        "converged": False,
        "inf_thr": inf_thr,
        "stop_thr": stop_thr,
        "prev_active_size": None,
        "prev_PrimalFeas": None,
        "last_active_pre_lp": None,
        "level_obj_hist": [],
        "stop_reason": None,
        "x_solution_last": x_solution_last,
        "y_solution_last": y_solution_last,
    }


def tree_prepare_inner(
    solver,
    *,
    level_idx: int,
    level_s,
    level_t,
    x_solution_last,
    y_solution_last,
    cost_type: str,
    use_last: bool,
    use_last_after_inner0: bool,
    ifcheck: bool,
    vd_thr: float,
    check_method: str,
    inner_iter: int,
    sampled_config=None,
    trace_collector=None,
    trace_prefix: str = "solve_ot",
) -> Dict[str, Any]:
    n_s = len(level_s.points)
    n_t = len(level_t.points)
    num_levels = solver.hierarchy_s.num_levels
    prepare_breakdown: Dict[str, float] = {}
    t_init = solver._time_now()

    if inner_iter == 0:
        init_result = build_active_set_first_iter(
            solver,
            level_idx=level_idx,
            level_s=level_s,
            level_t=level_t,
            x_solution_last=x_solution_last,
            y_solution_last=y_solution_last,
            cost_type=cost_type,
            use_last=use_last,
            ifcheck=ifcheck,
            vd_thr=vd_thr,
            check_method=check_method,
            sampled_config=sampled_config,
            trace_collector=trace_collector,
            trace_prefix=trace_prefix,
            trace_context={
                "level_idx": level_idx,
                "inner_iter": inner_iter,
                "stage": "init_active_set",
            },
        )
    else:
        init_result = build_active_set_subsequent_iter(
            solver,
            level_idx=level_idx,
            level_s=level_s,
            level_t=level_t,
            x_solution_last=x_solution_last,
            y_solution_last=y_solution_last,
            cost_type=cost_type,
            use_last=use_last,
            use_last_after_inner0=use_last_after_inner0,
            ifcheck=ifcheck,
            vd_thr=vd_thr,
            check_method=check_method,
            inner_iter=inner_iter,
            sampled_config=sampled_config,
            trace_collector=trace_collector,
            trace_prefix=trace_prefix,
            trace_context={
                "level_idx": level_idx,
                "inner_iter": inner_iter,
                "stage": "init_active_set",
            },
        )

    if init_result.get("prepare_breakdown"):
        prepare_breakdown.update(init_result["prepare_breakdown"])

    x_init = init_result["x_init"]
    y_init = init_result["y_init"]
    keep = np.asarray(init_result["keep"], dtype=np.int64)
    keep_coord = init_result["keep_coord"]

    if level_idx == num_levels and inner_iter == 0:
        expected = n_s * n_t
        assert len(keep) == expected, (
            f"Coarsest level keep must be full support: got {len(keep)}, expected {expected}"
        )

    curr_active_size = len(keep)
    prepare_time_total = float(solver._time_now() - t_init)
    tree_log(solver, f"  Init time: {prepare_time_total:.3f}s, Active vars: {curr_active_size}")

    return {
        "x_init": x_init,
        "y_init": y_init,
        "keep": keep,
        "keep_coord": keep_coord,
        "curr_active_size": curr_active_size,
        "prepare_breakdown": prepare_breakdown,
        "prepare_time_total": prepare_time_total,
        "trace_keep_after_shield": init_result.get("trace_keep_after_shield"),
        "trace_keep_after_check": init_result.get("trace_keep_after_check"),
        "trace_keep_after_uselast": init_result.get("trace_keep_after_uselast"),
    }


def init_run_state(solver, tolerance, **kwargs):
    tree_debug = bool(kwargs.get("tree_debug", getattr(solver, "tree_debug", False)))
    tree_infeas_fallback = normalize_tree_infeas_fallback(
        kwargs.get("tree_infeas_fallback", getattr(solver, "tree_infeas_fallback", "none"))
    )
    solver.tree_infeas_use_cupy = parse_bool_flag(
        kwargs.get("tree_infeas_use_cupy", getattr(solver, "tree_infeas_use_cupy", True)),
        default=True,
    )
    solver.tree_lp_diags = []

    use_last = parse_bool_flag(kwargs.get("use_last", getattr(solver, "use_last", True)), default=True)
    use_last_after_inner0 = parse_bool_flag(
        kwargs.get("use_last_after_inner0", getattr(solver, "use_last_after_inner0", False)),
        default=False,
    )
    ifcheck = parse_bool_flag(kwargs.get("ifcheck", getattr(solver, "ifcheck", True)), default=True)
    vd_thr = float(kwargs.get("vd_thr", getattr(solver, "vd_thr", 0.25)))
    solver.use_last = use_last
    solver.use_last_after_inner0 = use_last_after_inner0
    solver.ifcheck = ifcheck
    solver.vd_thr = vd_thr
    check_method = str(kwargs.get("check_type", getattr(solver, "check_type", "auto"))).lower()
    if check_method in {"cupy", "gpu_approx", "approx"}:
        check_method = "gpu_approx"
    elif check_method in {"gpu", "gpu_full", "full_gpu"}:
        check_method = "gpu"
    if check_method not in {"cpu", "gpu", "gpu_approx", "auto"}:
        check_method = "auto"
    tree_lp_form = str(kwargs.get("tree_lp_form", getattr(solver, "tree_lp_form", "dual"))).lower()
    if tree_lp_form not in {"primal", "dual"}:
        tree_lp_form = "dual"
    solver.tree_lp_form = tree_lp_form

    solver.keep_last = None
    solver.solutions = {}
    solver.level_summaries = []
    solver.nnz_thr = float(kwargs.get("nnz_thr", getattr(solver, "nnz_thr", 1e-20)))
    solver.check_sampled_config = kwargs.get("sampled_config", getattr(solver, "check_sampled_config", None))
    if hasattr(solver.strategy, "shield_impl") and "shield_impl" in kwargs:
        solver.strategy.shield_impl = str(kwargs.get("shield_impl")).lower()
    if hasattr(solver.strategy, "max_pairs_per_xA") and "max_pairs_per_xA" in kwargs:
        solver.strategy.max_pairs_per_xA = int(kwargs.get("max_pairs_per_xA"))
    if hasattr(solver.strategy, "nnz_thr") and "nnz_thr" in kwargs:
        solver.strategy.nnz_thr = float(kwargs.get("nnz_thr"))
    return {
        "num_levels": solver.hierarchy_s.num_levels,
        "finest_idx": 0,
        "coarsest_idx": solver.hierarchy_s.num_levels,
        "tree_debug": tree_debug,
        "tree_infeas_fallback": tree_infeas_fallback,
        "use_last": use_last,
        "use_last_after_inner0": use_last_after_inner0,
        "ifcheck": ifcheck,
        "vd_thr": vd_thr,
        "check_method": check_method,
        "sampled_config": kwargs.get("sampled_config", getattr(solver, "check_sampled_config", None)),
        "x_solution_last": None,
        "y_solution_last": None,
        "tolerance": tolerance,
        "trace_collector": kwargs.get("trace_collector"),
        "trace_prefix": str(kwargs.get("trace_prefix", "solve_ot")),
    }


def get_level_indices(_solver, run_state):
    return range(run_state["coarsest_idx"], run_state["finest_idx"] - 1, -1)


def get_max_inner_iters(_solver, level_state, max_inner_iter: int) -> int:
    if level_state.get("is_coarsest"):
        return 1
    return max_inner_iter


def init_level_state(solver, level_idx, run_state, tolerance, cost_type, _use_bfs_skeleton):
    level_s = solver.hierarchy_s.levels[level_idx]
    level_t = solver.hierarchy_t.levels[level_idx]
    n_s = len(level_s.points)
    n_t = len(level_t.points)
    tree_log(solver, f"\n{'='*50}")
    tree_log(solver, f"=== Solving Level {level_idx} (Tree Mode) ===")
    tree_log(solver, f"{'='*50}")
    if n_s == 0 or n_t == 0:
        raise ValueError(
            f"Empty tree level detected: level_idx={level_idx}, n_s={n_s}, n_t={n_t}"
        )

    num_levels = run_state["num_levels"]
    inf_thr, stop_thr = prepare_level_tolerances(tolerance, level_idx)
    state = init_level_state_impl(
        solver,
        level_idx=level_idx,
        inf_thr=inf_thr,
        stop_thr=stop_thr,
        x_solution_last=run_state["x_solution_last"],
        y_solution_last=run_state["y_solution_last"],
    )
    state["level_s"] = level_s
    state["level_t"] = level_t
    state["t_level_start"] = solver._time_now()
    state["is_coarsest"] = level_idx == num_levels
    state["cost_type"] = cost_type

    prep0 = tree_prepare_inner(
        solver,
        level_idx=level_idx,
        level_s=level_s,
        level_t=level_t,
        x_solution_last=state["x_solution_last"],
        y_solution_last=state["y_solution_last"],
        cost_type=cost_type,
        use_last=run_state["use_last"],
        use_last_after_inner0=run_state["use_last_after_inner0"],
        ifcheck=run_state["ifcheck"],
        vd_thr=run_state["vd_thr"],
        check_method=run_state["check_method"],
        inner_iter=0,
        sampled_config=run_state.get("sampled_config"),
        trace_collector=run_state.get("trace_collector"),
        trace_prefix=run_state.get("trace_prefix", "solve_ot"),
    )
    state["keep"] = prep0["keep"]
    state["keep_coord"] = prep0["keep_coord"]
    state["x_init"] = prep0["x_init"]
    state["y_init"] = prep0["y_init"]
    state["curr_active_size"] = prep0["curr_active_size"]
    state["_trace_keep_after_shield"] = prep0.get("trace_keep_after_shield")
    state["_trace_keep_after_check"] = prep0.get("trace_keep_after_check")
    state["_trace_keep_after_uselast"] = prep0.get("trace_keep_after_uselast")
    state["level_lp_time"] = 0.0
    state["level_pricing_time"] = float(prep0.get("prepare_time_total", 0.0))
    return state


def initial(problem_def, algorithm_state, level_index: int):
    level_data = init_level_state(
        problem_def.solver,
        level_index,
        algorithm_state.run_state,
        problem_def.tolerance,
        problem_def.cost_type,
        problem_def.use_bfs_skeleton,
    )
    return LevelState.from_legacy_data(level_index=level_index, max_inner_iter=0, data=level_data)
