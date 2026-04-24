from __future__ import annotations

import inspect
import time
from typing import Any, Dict, Optional

import numpy as np

from ...core.solver_utils import build_minus_AT_csc, generate_minus_c
from ...instrumentation.phase_names import (
    COMPONENT_LP_DIAG_EXTRACT,
    COMPONENT_LP_GET_SOLUTION,
    COMPONENT_LP_LOAD_DATA,
    COMPONENT_LP_PARAM_PACK,
    COMPONENT_LP_RESULT_PACK,
    COMPONENT_LP_WARM_START,
)
from .infeasibility_check import compute_primal_infeas
from .logger import tree_log
from .trace_utils import tree_trace_span


def _tree_internal_cost_type(cost_type_raw: str) -> str:
    normalized = str(cost_type_raw).strip().lower()
    if normalized in {"l2^2", "sqeuclidean", "l2sq"}:
        return "L2"
    if normalized in {"l1"}:
        return "L1"
    if normalized in {"linf"}:
        return "LINF"
    if normalized in {"l2", "euclidean"}:
        raise ValueError("tree mode does not support 'l2'; use 'l2^2' for squared Euclidean distance.")
    raise ValueError(f"Unsupported tree cost_type={cost_type_raw!r}")


def _solve_lp(
    solver,
    *,
    level_s,
    level_t,
    keep_coord,
    x_init,
    y_init,
    stop_thr: float,
    inner_iter: int = 0,
    prev_PrimalFeas: float | None = None,
    level_idx: int = -1,
    is_coarsest: bool = False,
    tree_debug: bool = False,
    require_full_coverage: bool = True,
    trace_collector: Optional[Any] = None,
    trace_prefix: str = "solve_ot.lp",
    trace_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    trace_meta = dict(trace_context or {})
    trace_level = trace_meta.get("level_idx", level_idx)
    trace_iter = trace_meta.get("inner_iter", inner_iter)
    trace_stage = trace_meta.get("stage", "lp_solve")

    def _trace_span(name: str, args: Optional[Dict[str, Any]] = None):
        return tree_trace_span(
            trace_collector,
            name,
            level_idx=trace_level,
            inner_iter=trace_iter,
            stage=trace_stage,
            args=args,
        )

    components: Dict[str, float] = {}
    n_s = len(level_s.points)
    n_t = len(level_t.points)
    n_vars = len(keep_coord)
    mass_s = np.asarray(level_s.masses, dtype=np.float64)
    mass_t = np.asarray(level_t.masses, dtype=np.float64)
    sum_s = float(np.sum(mass_s))
    sum_t = float(np.sum(mass_t))

    mass_renormalized = False
    if abs(sum_s - sum_t) > 1e-12:
        if sum_s <= 0.0 or sum_t <= 0.0:
            raise ValueError(f"Invalid masses: sum_s={sum_s}, sum_t={sum_t}")
        mass_s = mass_s / sum_s
        mass_t = mass_t / sum_t
        mass_renormalized = True

    diag: Dict[str, Any] = {
        "level": int(level_idx),
        "is_coarsest": bool(is_coarsest),
        "tree_lp_form": str(getattr(solver, "tree_lp_form", "dual")),
        "n_s": int(n_s),
        "n_t": int(n_t),
        "n_vars": int(n_vars),
        "n_eqs": int(n_s + n_t),
        "sum_source_mass": sum_s,
        "sum_target_mass": sum_t,
        "mass_diff": float(sum_s - sum_t),
        "mass_renormalized": mass_renormalized,
    }

    cost_type = _tree_internal_cost_type(getattr(solver, "_cost_type", "l2^2"))
    p = getattr(solver, "_cost_p", 2.0)
    t_param_pack = time.perf_counter()
    with _trace_span(
        "tree.lp.param_pack",
        args={"n_vars": int(n_vars)},
    ):
        y_init_array = y_init["y"] if isinstance(y_init, dict) else y_init
        lb = np.zeros(n_vars, dtype=np.float64)
        ub = np.full(n_vars, np.inf, dtype=np.float64)
    param_pack_dt = time.perf_counter() - t_param_pack
    if param_pack_dt > 0.0:
        components[COMPONENT_LP_PARAM_PACK] = float(param_pack_dt)

    t_cost = time.perf_counter()
    with _trace_span("tree.lp.cost_build"):
        minus_c = generate_minus_c(keep_coord, level_s.points, level_t.points, cost_type, p)
        c = -minus_c
    cost_build_dt = time.perf_counter() - t_cost
    diag["cost_build_time"] = float(cost_build_dt)
    if cost_build_dt > 0.0:
        components["lp_cost_build"] = float(cost_build_dt)
    diag["c_min"] = float(np.min(c)) if c.size > 0 else 0.0
    diag["c_max"] = float(np.max(c)) if c.size > 0 else 0.0

    if cost_type.upper() in {"L1", "L2", "LINF", "SQEUCLIDEAN"} and c.size > 0:
        if float(np.min(c)) < -1e-10:
            raise ValueError(
                f"Found negative cost entries for cost_type={cost_type}: min(c)={np.min(c)}"
            )

    t_matrix = time.perf_counter()
    with _trace_span("tree.lp.matrix_build"):
        minus_AT = build_minus_AT_csc(keep_coord, n_s, n_t, dtype=np.float64)
        A_eq = minus_AT.T.tocsc()
        A_eq.data *= -1.0
    matrix_build_dt = time.perf_counter() - t_matrix
    diag["matrix_build_time"] = float(matrix_build_dt)
    if matrix_build_dt > 0.0:
        components["lp_matrix_build"] = float(matrix_build_dt)

    t_rhs = time.perf_counter()
    with _trace_span("tree.lp.rhs_bounds_pack"):
        b_eq = np.concatenate([mass_s, mass_t])
    rhs_bounds_dt = time.perf_counter() - t_rhs
    if rhs_bounds_dt > 0.0:
        components["lp_rhs_bounds_pack"] = float(rhs_bounds_dt)
    if A_eq.shape != (n_s + n_t, n_vars):
        raise ValueError(f"Invalid A_eq shape={A_eq.shape}, expected {(n_s + n_t, n_vars)}")

    if n_vars > 0:
        col_nnz = np.diff(A_eq.indptr)
        diag["per_col_nnz_min"] = int(col_nnz.min())
        diag["per_col_nnz_max"] = int(col_nnz.max())
        if not np.all(col_nnz == 2):
            raise ValueError(
                f"Invalid A_eq column nnz profile: min={col_nnz.min()}, max={col_nnz.max()}, expected all 2"
            )
    else:
        diag["per_col_nnz_min"] = 0
        diag["per_col_nnz_max"] = 0

    if A_eq.nnz > 0 and not np.allclose(A_eq.data, 1.0):
        raise ValueError(
            f"Invalid A_eq signs: expected all +1, got min={A_eq.data.min()}, max={A_eq.data.max()}"
        )

    idx_x = keep_coord["idx1"].astype(np.int64, copy=False)
    idx_y = keep_coord["idx2"].astype(np.int64, copy=False)
    row_cov = np.bincount(idx_x, minlength=n_s) if n_s > 0 else np.array([], dtype=np.int64)
    col_cov = np.bincount(idx_y, minlength=n_t) if n_t > 0 else np.array([], dtype=np.int64)
    row_missing = int(np.sum(row_cov == 0))
    col_missing = int(np.sum(col_cov == 0))
    diag["row_cover_ok"] = bool(row_missing == 0)
    diag["col_cover_ok"] = bool(col_missing == 0)
    diag["row_missing"] = row_missing
    diag["col_missing"] = col_missing
    if require_full_coverage and (row_missing > 0 or col_missing > 0):
        raise ValueError(f"Coverage broken: row_missing={row_missing}, col_missing={col_missing}")

    if is_coarsest:
        expected = n_s * n_t
        if n_vars != expected:
            raise AssertionError(f"Coarsest level must use full support: K={n_vars}, expected={expected}")

        total_mass = float(np.sum(mass_s))
        if total_mass <= 0.0:
            raise ValueError(f"Invalid total mass at coarsest level: {total_mass}")
        y_witness = (mass_s[idx_x] * mass_t[idx_y]) / total_mass
        witness_residual = A_eq @ y_witness - b_eq
        witness_inf = float(np.max(np.abs(witness_residual))) if witness_residual.size > 0 else 0.0
        diag["witness_total_mass"] = total_mass
        diag["witness_residual_inf"] = witness_inf
        if witness_inf > 1e-10:
            raise ValueError(
                f"Witness feasibility check failed: ||A*y_witness-b||_inf={witness_inf:.3e}"
            )
    else:
        diag["witness_residual_inf"] = None

    if tree_debug:
        tag = "[COARSEST-LP-DIAG]" if is_coarsest else "[LP-DIAG]"
        tree_log(
            solver,
            f"  {tag} "
            + f"level={level_idx}, K={n_vars}, n_eqs={n_s+n_t}, "
            + f"mass_diff={diag['mass_diff']:+.3e}, row_missing={row_missing}, col_missing={col_missing}, "
            + f"mass_renorm={diag['mass_renormalized']}, "
            + f"c_min={diag['c_min']:.3e}, c_max={diag['c_max']:.3e}"
        )

    if is_coarsest:
        dPrimalTol = stop_thr
    elif inner_iter == 0:
        dPrimalTol = stop_thr * 1e2
    else:
        dPrimalTol = max(prev_PrimalFeas / 10, stop_thr) if prev_PrimalFeas else stop_thr

    if y_init is not None:
        y_vals = np.asarray(y_init)
        if len(y_vals) == len(keep_coord):
            A_y = A_eq @ y_vals
            constraint_error = np.linalg.norm(A_y - b_eq) / np.linalg.norm(b_eq)
            if tree_debug:
                tree_log(solver, f"  [DEBUG] y_init constraint rel-error: {constraint_error:.2e}")

    tree_log(
        solver,
        f"  LP solve: n_vars={n_vars}, n_eqs={n_s+n_t}, "
        f"warm_start_y={y_init is not None}, warm_start_x={x_init is not None}"
    )
    solve_kwargs = dict(
        c=c,
        A_csc=A_eq,
        b_eq=b_eq,
        lb=lb,
        ub=ub,
        n_eqs=n_s + n_t,
        warm_start_primal=y_init_array,
        warm_start_dual=x_init,
        tolerance={"primal": dPrimalTol, "dual": stop_thr, "objective": stop_thr},
    )
    requested_lp_form = str(getattr(solver, "tree_lp_form", "dual"))
    if getattr(solver.solver, "supports_tree_lp_form", False):
        solve_kwargs["lp_form"] = requested_lp_form
        solve_kwargs["dual_form_data"] = {
            "minus_AT": minus_AT,
            "minus_c": minus_c,
            "minus_q": -b_eq,
        }
        solver_params = getattr(solver, "_tree_solver_params", None)
        if isinstance(solver_params, dict) and solver_params:
            solve_kwargs["solver_params"] = dict(solver_params)
    elif requested_lp_form == "dual":
        raise NotImplementedError(
            f"{type(solver.solver).__name__} does not support tree_lp_form='dual'. "
            "Use CuPDLPxSolver or switch tree_lp_form to 'primal'."
        )

    if solver.lp_solver_verbose and hasattr(solver.solver, "solve"):
        try:
            sig = inspect.signature(solver.solver.solve)
            if "verbose" in sig.parameters:
                solve_kwargs["verbose"] = solver.lp_solver_verbose
        except (ValueError, TypeError):
            pass
    t_solve = time.perf_counter()
    with _trace_span("tree.lp.backend_total"):
        result = solver.solver.solve(
            **solve_kwargs,
            trace_collector=trace_collector,
            trace_prefix="tree.lp.backend",
        )
    backend_total_dt = time.perf_counter() - t_solve
    diag["backend_solve_wall_time"] = float(backend_total_dt)
    solve_time_attr = getattr(result, "solve_time", None)
    diag["backend_solve_time"] = None if solve_time_attr is None else float(solve_time_attr)
    t_diag_extract = time.perf_counter()
    with _trace_span("tree.lp.diag_extract"):
        solver_diag = getattr(result, "solver_diag", None)
        if isinstance(solver_diag, dict):
            for key, value in solver_diag.items():
                if value is None:
                    continue
                diag[f"solver_{key}"] = value
            load_data_time = float(solver_diag.get("load_data_time", 0.0) or 0.0)
            warm_start_time = float(solver_diag.get("warm_start_time", 0.0) or 0.0)
            get_solution_time = float(solver_diag.get("get_solution_time", 0.0) or 0.0)
            native_solve_time = float(solver_diag.get("native_solve_wall_time", 0.0) or 0.0)
            construct_time = float(solver_diag.get("construct_time", 0.0) or 0.0)
            if construct_time > 0.0:
                components["lp_backend_construct"] = construct_time
            if load_data_time > 0.0:
                components[COMPONENT_LP_LOAD_DATA] = load_data_time
            if warm_start_time > 0.0:
                components[COMPONENT_LP_WARM_START] = warm_start_time
            if get_solution_time > 0.0:
                components[COMPONENT_LP_GET_SOLUTION] = get_solution_time
            if native_solve_time > 0.0:
                components["lp_backend_solve"] = native_solve_time
    diag_extract_dt = time.perf_counter() - t_diag_extract
    if diag_extract_dt > 0.0:
        components[COMPONENT_LP_DIAG_EXTRACT] = float(diag_extract_dt)

    diag["termination_reason"] = getattr(result, "termination_reason", None)
    lp_payload = None
    if not result.success:
        lp_payload = {"c": c, "A_eq": A_eq, "b_eq": b_eq, "lb": lb, "ub": ub}
    result_pack_dt = 0.0
    if result.success:
        t_result_pack = time.perf_counter()
        with _trace_span("tree.lp.result_pack"):
            pass
        result_pack_dt = time.perf_counter() - t_result_pack
        if result_pack_dt > 0.0:
            components[COMPONENT_LP_RESULT_PACK] = float(result_pack_dt)
    return {
        "success": result.success,
        "x": result.y if result.success else None,
        "y": result.x if result.success else None,
        "obj": result.obj_val if result.success else None,
        "dual_obj": getattr(result, "dual_obj_val", None),
        "iters": getattr(result, "iterations", 0),
        "primal_feas": getattr(result, "primal_feas", None),
        "dual_feas": getattr(result, "dual_feas", None),
        "gap": getattr(result, "gap", None),
        "termination_reason": getattr(result, "termination_reason", None),
        "diag": diag,
        "lp_payload": lp_payload,
        "components": components,
    }


def solve_tree_lp_with_fallback(
    solver,
    *,
    level_idx: int,
    level_s,
    level_t,
    keep: np.ndarray,
    keep_coord,
    x_init: np.ndarray,
    y_init: Optional[np.ndarray],
    stop_thr: float,
    inner_iter: int,
    prev_PrimalFeas: Optional[float],
    tree_debug: bool,
    tree_infeas_fallback: str,
    trace_collector: Optional[Any] = None,
    trace_prefix: str = "solve_ot.lp",
    trace_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    num_levels = solver.hierarchy_s.num_levels
    is_coarsest = level_idx == num_levels
    if tree_infeas_fallback != "none":
        raise RuntimeError(
            "tree_infeas_fallback is no longer supported in the strict execution path. "
            f"Got: {tree_infeas_fallback}"
        )
    lp_result = _solve_lp(
        solver,
        level_s=level_s,
        level_t=level_t,
        keep_coord=keep_coord,
        x_init=x_init,
        y_init=y_init,
        stop_thr=stop_thr,
        inner_iter=inner_iter,
        prev_PrimalFeas=prev_PrimalFeas,
        level_idx=level_idx,
        is_coarsest=is_coarsest,
        tree_debug=tree_debug,
        trace_collector=trace_collector,
        trace_prefix=trace_prefix,
        trace_context=trace_context,
    )

    if lp_result.get("diag"):
        solver.tree_lp_diags.append(lp_result["diag"])

    if lp_result.get("success", False):
        return {"success": True, "lp_result": lp_result, "keep": keep, "keep_coord": keep_coord}

    return {"success": False, "lp_result": lp_result, "keep": keep, "keep_coord": keep_coord}


def tree_solve_lp_pack(
    solver,
    *,
    level_idx: int,
    level_s,
    level_t,
    keep: np.ndarray,
    keep_coord,
    x_init: np.ndarray,
    y_init: Optional[np.ndarray],
    stop_thr: float,
    inner_iter: int,
    prev_PrimalFeas: Optional[float],
    tree_debug: bool,
    tree_infeas_fallback: str,
    trace_collector: Optional[Any] = None,
    trace_prefix: str = "solve_ot.lp",
    trace_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    t_lp = time.perf_counter()
    with solver._profiler.timer("lp", gpu_possible=True):
        lp_pack = solve_tree_lp_with_fallback(
            solver,
            level_idx=level_idx,
            level_s=level_s,
            level_t=level_t,
            keep=keep,
            keep_coord=keep_coord,
            x_init=x_init,
            y_init=(y_init["y"] if y_init else None),
            stop_thr=stop_thr,
            inner_iter=inner_iter,
            prev_PrimalFeas=prev_PrimalFeas,
            tree_debug=tree_debug,
            tree_infeas_fallback=tree_infeas_fallback,
            trace_collector=trace_collector,
            trace_prefix=trace_prefix,
            trace_context=trace_context,
        )
    lp_time = time.perf_counter() - t_lp
    lp_result = lp_pack["lp_result"]
    if lp_pack["success"] and lp_result.get("x") is not None:
        lp_result["primal_infeas_ori"] = compute_primal_infeas(
            solver,
            lp_result["x"],
            level_s,
            level_t,
        )
    else:
        lp_result["primal_infeas_ori"] = None
    tree_log(solver, f"  LP time: {lp_time:.3f}s, Iters: {lp_result.get('iters', 0)}")
    if not lp_pack["success"]:
        raise RuntimeError(
            f"Tree LP failed at level={level_idx}, inner={inner_iter}, "
            f"reason={lp_result.get('termination_reason')}"
        )
    return {
        "success": True,
        "lp_result": lp_result,
        "keep": lp_pack["keep"],
        "keep_coord": lp_pack["keep_coord"],
        "lp_time": float(lp_time),
        "components": lp_result.get("components", {}),
    }
