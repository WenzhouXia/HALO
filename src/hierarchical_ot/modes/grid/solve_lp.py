from __future__ import annotations

import inspect
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ...core.solver_utils import build_minus_AT_csc, generate_minus_c
from ...instrumentation.phase_names import (
    COMPONENT_LP_DIAG_EXTRACT,
    COMPONENT_LP_GET_SOLUTION,
    COMPONENT_LP_LEVEL_VIEW_PREPARE,
    COMPONENT_LP_LOAD_DATA,
    COMPONENT_LP_PARAM_PACK,
    COMPONENT_LP_PRIMAL_INFEAS,
    COMPONENT_LP_RESULT_PACK,
    COMPONENT_LP_SNAPSHOT_MERGE,
    COMPONENT_LP_WARM_START,
)
from . import shielding as grid_shielding
from .infeasibility import compute_primal_infeas

_MGPD_ROOT = Path("/home/xwz/project/MGPD")
logger = logging.getLogger(__name__)


def _load_mgpd_lp_build_ops():
    if not _MGPD_ROOT.exists():
        return None, None, None
    if str(_MGPD_ROOT) not in sys.path:
        sys.path.insert(0, str(_MGPD_ROOT))
    try:
        from operators import dualOT_minus_AT_csc, generate_minus_c, keep2keepcoord  # type: ignore
    except Exception:
        return None, None, None
    return keep2keepcoord, generate_minus_c, dualOT_minus_AT_csc


def _grid_mgpd_cost_type(cost_type: str) -> str:
    normalized = str(cost_type).strip().lower()
    if normalized in {"l2^2", "l2", "sqeuclidean"}:
        return "L2"
    if normalized == "l1":
        return "L1"
    if normalized == "linf":
        return "Linf"
    return "L2"


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


def _tree_log(solver, message: str) -> None:
    if not bool(getattr(solver, "tree_debug", False) or getattr(solver, "ifdebug", False)):
        return
    if hasattr(solver, "_runtime_log"):
        solver._runtime_log("progress", message)
    else:
        logger.debug(message)


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
) -> Dict[str, Any]:
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
    t_cost = time.perf_counter()
    minus_c = generate_minus_c(keep_coord, level_s.points, level_t.points, cost_type, p)
    c = -minus_c
    diag["cost_build_time"] = float(time.perf_counter() - t_cost)
    diag["c_min"] = float(np.min(c)) if c.size > 0 else 0.0
    diag["c_max"] = float(np.max(c)) if c.size > 0 else 0.0

    if cost_type.upper() in {"L1", "L2", "LINF", "SQEUCLIDEAN"} and c.size > 0:
        if float(np.min(c)) < -1e-10:
            raise ValueError(
                f"Found negative cost entries for cost_type={cost_type}: min(c)={np.min(c)}"
            )

    t_matrix = time.perf_counter()
    minus_AT = build_minus_AT_csc(keep_coord, n_s, n_t, dtype=np.float64)
    A_eq = minus_AT.T.tocsc()
    A_eq.data *= -1.0
    diag["matrix_build_time"] = float(time.perf_counter() - t_matrix)

    b_eq = np.concatenate([mass_s, mass_t])
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
        _tree_log(
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

    lb = np.zeros(n_vars, dtype=np.float64)
    ub = np.full(n_vars, np.inf, dtype=np.float64)

    if y_init is not None:
        y_vals = np.asarray(y_init)
        if len(y_vals) == len(keep_coord):
            A_y = A_eq @ y_vals
            constraint_error = np.linalg.norm(A_y - b_eq) / np.linalg.norm(b_eq)
            if tree_debug:
                _tree_log(solver, f"  [DEBUG] y_init constraint rel-error: {constraint_error:.2e}")

    _tree_log(
        solver,
        f"  LP solve: n_vars={n_vars}, n_eqs={n_s+n_t}, "
        f"warm_start_y={y_init is not None}, warm_start_x={x_init is not None}"
    )
    y_init_array = y_init["y"] if isinstance(y_init, dict) else y_init
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
    result = solver.solver.solve(**solve_kwargs)
    diag["backend_solve_wall_time"] = float(time.perf_counter() - t_solve)
    solve_time_attr = getattr(result, "solve_time", None)
    diag["backend_solve_time"] = None if solve_time_attr is None else float(solve_time_attr)
    solver_diag = getattr(result, "solver_diag", None)
    if isinstance(solver_diag, dict):
        for key, value in solver_diag.items():
            if value is None:
                continue
            diag[f"solver_{key}"] = value

    diag["termination_reason"] = getattr(result, "termination_reason", None)
    lp_payload = None
    if not result.success:
        lp_payload = {"c": c, "A_eq": A_eq, "b_eq": b_eq, "lb": lb, "ub": ub}
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
    }


def solve_grid_lp_direct_dual(
    solver,
    *,
    level_s,
    level_t,
    keep: np.ndarray,
    keep_coord: np.ndarray,
    x_init,
    y_init,
    stop_thr: float,
    inner_iter: int,
    prev_PrimalFeas: float | None,
    level_idx: int,
    is_coarsest: bool,
) -> Dict[str, Any]:
    del keep
    n_s = len(level_s.points)
    n_t = len(level_t.points)
    n_vars = len(keep_coord)
    mass_s = np.asarray(level_s.masses, dtype=np.float64)
    mass_t = np.asarray(level_t.masses, dtype=np.float64)
    b_eq = np.concatenate([mass_s, mass_t])
    minus_q = -b_eq
    diag: Dict[str, Any] = {
        "level": int(level_idx),
        "is_coarsest": bool(is_coarsest),
        "tree_lp_form": "dual",
        "n_s": int(n_s),
        "n_t": int(n_t),
        "n_vars": int(n_vars),
        "n_eqs": int(n_s + n_t),
    }

    _, mgpd_generate_minus_c, mgpd_dualOT_minus_AT_csc = _load_mgpd_lp_build_ops()
    if mgpd_generate_minus_c is None or mgpd_dualOT_minus_AT_csc is None:
        raise RuntimeError("MGPD LP build ops are unavailable")

    finest_resolution = int(round(np.sqrt(len(solver.hierarchy_s.levels[0].points))))
    resolution_now = int(round(np.sqrt(n_s)))

    t_cost = time.perf_counter()
    minus_c = mgpd_generate_minus_c(
        int(level_idx),
        int(finest_resolution),
        keep_coord,
        _grid_mgpd_cost_type(getattr(solver, "_cost_type", "l2^2")),
        ifgpu=False,
        print_time=False,
    )
    c = -np.asarray(minus_c, dtype=np.float64)
    diag["cost_build_time"] = float(time.perf_counter() - t_cost)
    diag["c_min"] = float(np.min(c)) if c.size > 0 else 0.0
    diag["c_max"] = float(np.max(c)) if c.size > 0 else 0.0

    t_matrix = time.perf_counter()
    minus_AT = mgpd_dualOT_minus_AT_csc(keep_coord, resolution_now, print_time=False)
    diag["matrix_build_time"] = float(time.perf_counter() - t_matrix)

    idx_x = np.asarray(keep_coord["idx1"], dtype=np.int64)
    idx_y = np.asarray(keep_coord["idx2"], dtype=np.int64)
    row_cov = np.bincount(idx_x, minlength=n_s) if n_s > 0 else np.array([], dtype=np.int64)
    col_cov = np.bincount(idx_y, minlength=n_t) if n_t > 0 else np.array([], dtype=np.int64)
    diag["row_cover_ok"] = bool(np.all(row_cov > 0)) if row_cov.size > 0 else True
    diag["col_cover_ok"] = bool(np.all(col_cov > 0)) if col_cov.size > 0 else True
    diag["row_missing"] = int(np.sum(row_cov == 0)) if row_cov.size > 0 else 0
    diag["col_missing"] = int(np.sum(col_cov == 0)) if col_cov.size > 0 else 0

    if is_coarsest:
        expected = n_s * n_t
        if n_vars != expected:
            raise AssertionError(
                f"Coarsest level must use full support: K={n_vars}, expected={expected}"
            )

    if is_coarsest:
        dPrimalTol = stop_thr
    elif inner_iter == 0:
        dPrimalTol = stop_thr * 1e2
    else:
        dPrimalTol = max(prev_PrimalFeas / 10, stop_thr) if prev_PrimalFeas else stop_thr

    y_init_array = y_init["y"] if isinstance(y_init, dict) else y_init
    solve_kwargs = dict(
        c=np.empty(0, dtype=np.float64),
        A_csc=None,
        b_eq=b_eq,
        lb=None,
        ub=None,
        n_eqs=0,
        warm_start_primal=y_init_array,
        warm_start_dual=x_init,
        tolerance={
            "primal": dPrimalTol,
            "dual": stop_thr,
            "objective": stop_thr,
        },
        lp_form="dual",
        dual_form_data={
            "minus_AT": minus_AT,
            "minus_c": np.asarray(minus_c, dtype=np.float64),
            "minus_q": minus_q,
        },
    )
    solver_params = getattr(solver, "_tree_solver_params", None)
    if isinstance(solver_params, dict) and solver_params:
        solve_kwargs["solver_params"] = dict(solver_params)
    if solver.lp_solver_verbose and hasattr(solver.solver, "solve"):
        try:
            sig = inspect.signature(solver.solver.solve)
            if "verbose" in sig.parameters:
                solve_kwargs["verbose"] = solver.lp_solver_verbose
        except (ValueError, TypeError):
            pass

    t_solve = time.perf_counter()
    result = solver.solver.solve(**solve_kwargs)
    diag["backend_solve_wall_time"] = float(time.perf_counter() - t_solve)
    solve_time_attr = getattr(result, "solve_time", None)
    diag["backend_solve_time"] = None if solve_time_attr is None else float(solve_time_attr)
    solver_diag = getattr(result, "solver_diag", None)
    if isinstance(solver_diag, dict):
        for key, value in solver_diag.items():
            if value is None:
                continue
            diag[f"solver_{key}"] = value

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
        "lp_payload": None,
    }


def grid_solve_lp_pack(
    solver,
    level_state: Dict[str, Any],
    run_state: Dict[str, Any],
) -> Dict[str, Any]:
    components: Dict[str, float] = {}
    t_level_view = time.perf_counter()
    level_s_lp = level_state.get("level_s_lp")
    if level_s_lp is None:
        level_s_lp = grid_shielding.grid_level_for_dual_lp(level_state["level_s"])
        level_state["level_s_lp"] = level_s_lp
    level_t_lp = level_state.get("level_t_lp")
    if level_t_lp is None:
        level_t_lp = grid_shielding.grid_level_for_dual_lp(level_state["level_t"])
        level_state["level_t_lp"] = level_t_lp
    level_view_dt = time.perf_counter() - t_level_view
    if level_view_dt > 0.0:
        components[COMPONENT_LP_LEVEL_VIEW_PREPARE] = level_view_dt
    level_idx = int(level_state["level_idx"])
    inner_iter = int(level_state["current_iter"])
    num_scale = int(run_state.get("coarsest_idx", 0))
    t_param_pack = time.perf_counter()
    solver._tree_solver_params = {
        "use_dual_nnz_gate": int(level_idx != 0),
        "dual_nnz_factor": 2.5,
        "usePrimalFeasOri": "1" if bool(run_state.get("use_primal_feas_ori", True) and level_idx == 0 and num_scale > 0) else "0",
        "newMVP": "1" if bool(run_state.get("new_mvp", True)) else "0",
        "ATy_type": str(int(run_state.get("aty_type", 0))),
        "earlyStop": "1" if level_idx <= 1 and inner_iter == 0 else "0",
        "adapPrimalTol": "1" if bool(run_state.get("adap_primal_tol", True) and level_idx == 0) else "0",
    }
    t0 = time.perf_counter()
    can_use_direct_grid_path = (
        getattr(solver.solver, "supports_tree_lp_form", False)
        and isinstance(level_state.get("keep_coord"), np.ndarray)
        and level_state["keep_coord"].dtype.names is not None
        and "idx_i1" in level_state["keep_coord"].dtype.names
        and "idx_j1" in level_state["keep_coord"].dtype.names
        and "idx_i2" in level_state["keep_coord"].dtype.names
        and "idx_j2" in level_state["keep_coord"].dtype.names
    )
    param_pack_dt = time.perf_counter() - t_param_pack
    if param_pack_dt > 0.0:
        components[COMPONENT_LP_PARAM_PACK] = param_pack_dt
    original_tree_lp_form = getattr(solver, "tree_lp_form", "dual")
    try:
        if can_use_direct_grid_path:
            lp_result = solve_grid_lp_direct_dual(
                solver,
                level_s=level_s_lp,
                level_t=level_t_lp,
                keep=level_state["keep"],
                keep_coord=level_state["keep_coord"],
                x_init=level_state["x_init"],
                y_init=(level_state["y_init"]["y"] if isinstance(level_state.get("y_init"), dict) else level_state.get("y_init")),
                stop_thr=float(level_state["stop_thr"]),
                inner_iter=int(level_state["current_iter"]),
                prev_PrimalFeas=level_state.get("prev_PrimalFeas"),
                level_idx=int(level_state["level_idx"]),
                is_coarsest=bool(level_state.get("is_coarsest", False)),
            )
        else:
            solver.tree_lp_form = "primal"
            lp_result = _solve_lp(
                solver,
                level_s=level_s_lp,
                level_t=level_t_lp,
                keep_coord=level_state["keep_coord"],
                x_init=level_state["x_init"],
                y_init=(level_state["y_init"]["y"] if isinstance(level_state.get("y_init"), dict) else level_state.get("y_init")),
                stop_thr=float(level_state["stop_thr"]),
                inner_iter=int(level_state["current_iter"]),
                prev_PrimalFeas=level_state.get("prev_PrimalFeas"),
                level_idx=int(level_state["level_idx"]),
                is_coarsest=bool(level_state.get("is_coarsest", False)),
                tree_debug=False,
                require_full_coverage=False,
            )
    finally:
        solver.tree_lp_form = original_tree_lp_form
    solver._tree_solver_params = None
    tree_lp_total = time.perf_counter() - t0
    t_diag_extract = time.perf_counter()
    diag_raw = lp_result.get("diag", {})
    diag = diag_raw if isinstance(diag_raw, dict) else {}
    cost_build_time = float(diag.get("cost_build_time", 0.0) or 0.0)
    matrix_build_time = float(diag.get("matrix_build_time", 0.0) or 0.0)
    backend_solve_wall_time = float(diag.get("backend_solve_wall_time", 0.0) or 0.0)
    native_solve_wall_time = float(diag.get("solver_native_solve_wall_time", 0.0) or 0.0)
    load_data_time = float(diag.get("solver_load_data_time", 0.0) or 0.0)
    warm_start_time = float(diag.get("solver_warm_start_time", 0.0) or 0.0)
    get_solution_time = float(diag.get("solver_get_solution_time", 0.0) or 0.0)
    if cost_build_time > 0.0:
        components["lp_cost_build"] = cost_build_time
    if matrix_build_time > 0.0:
        components["lp_matrix_build"] = matrix_build_time
    if load_data_time > 0.0:
        components[COMPONENT_LP_LOAD_DATA] = load_data_time
    if warm_start_time > 0.0:
        components[COMPONENT_LP_WARM_START] = warm_start_time
    if native_solve_wall_time > 0.0:
        components["lp_backend_solve"] = native_solve_wall_time
    elif backend_solve_wall_time > 0.0:
        components["lp_backend_solve"] = backend_solve_wall_time
    if get_solution_time > 0.0:
        components[COMPONENT_LP_GET_SOLUTION] = get_solution_time
    diag_extract_dt = time.perf_counter() - t_diag_extract
    if diag_extract_dt > 0.0:
        components[COMPONENT_LP_DIAG_EXTRACT] = diag_extract_dt
    t_infeas = time.perf_counter()
    if lp_result["success"] and lp_result.get("x") is not None:
        lp_result["primal_infeas_ori"] = compute_primal_infeas(
            solver,
            np.asarray(lp_result["x"], dtype=np.float32),
            level_state["level_s"],
            level_state["level_t"],
        )
    else:
        lp_result["primal_infeas_ori"] = None
    infeas_dt = time.perf_counter() - t_infeas
    if infeas_dt > 0.0:
        components[COMPONENT_LP_PRIMAL_INFEAS] = float(infeas_dt)
    dt = tree_lp_total + infeas_dt
    t_snapshot_merge = time.perf_counter()
    diag.update(
        {
            "level": int(level_state["level_idx"]),
            "iter": int(level_state["current_iter"]),
            "duration": float(dt),
            "active_size": int(len(level_state.get("keep", []))),
        }
    )
    prepare_snapshot = level_state.get("prepare_snapshot")
    if isinstance(prepare_snapshot, dict):
        diag.update(prepare_snapshot)
    snapshot_merge_dt = time.perf_counter() - t_snapshot_merge
    if snapshot_merge_dt > 0.0:
        components[COMPONENT_LP_SNAPSHOT_MERGE] = snapshot_merge_dt
    solver.grid_lp_diags.append(diag)
    if not lp_result["success"]:
        t_result_pack = time.perf_counter()
        level_state["stop_reason"] = str(lp_result.get("termination_reason", "lp_failed"))
        result_pack_dt = time.perf_counter() - t_result_pack
        if result_pack_dt > 0.0:
            components[COMPONENT_LP_RESULT_PACK] = result_pack_dt
        residual = dt - sum(components.values())
        if residual > 1e-9:
            components["lp_wrapper_overhead"] = float(residual)
        return {
            "success": False,
            "lp_result": lp_result,
            "lp_time": float(dt),
            "components": components,
        }
    t_result_pack = time.perf_counter()
    keep_out = level_state["keep"]
    if not isinstance(keep_out, np.ndarray) or keep_out.dtype != np.int64:
        keep_out = np.asarray(keep_out, dtype=np.int64)
    result_pack_dt = time.perf_counter() - t_result_pack
    if result_pack_dt > 0.0:
        components[COMPONENT_LP_RESULT_PACK] = result_pack_dt
    residual = dt - sum(components.values())
    if residual > 1e-9:
        components["lp_wrapper_overhead"] = float(residual)
    return {
        "success": True,
        "lp_result": lp_result,
        "keep": keep_out,
        "keep_coord": level_state["keep_coord"],
        "lp_time": float(dt),
        "components": components,
    }
