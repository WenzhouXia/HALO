from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
try:
    import cupy as cp
except Exception:  
    cp = None

from ..common.utils import decode_keep_1d_to_struct, remap_duals_for_warm_start
from .infeas_grid import dualOT_primal_infeas_grid_auto
from .costs import grid_pairwise_cost
from .operators import prolong_grid_dual
from .shield import build_grid_shield
from ..common.solver_backend import CommonLevelState, CommonRunState, SolverBackend
from ..common.profiling_names import (
    COMPONENT_CONVERGENCE_CHECK,
    COMPONENT_COVERAGE_REPAIR,
    COMPONENT_KEEP_COORD,
    COMPONENT_LP_DIAG_EXTRACT,
    COMPONENT_LP_GET_SOLUTION,
    COMPONENT_LP_LOAD_DATA,
    COMPONENT_LP_LEVEL_VIEW_PREPARE,
    COMPONENT_LP_PARAM_PACK,
    COMPONENT_LP_PRIMAL_INFEAS,
    COMPONENT_LP_RESULT_PACK,
    COMPONENT_LP_SNAPSHOT_MERGE,
    COMPONENT_LP_WARM_START,
    COMPONENT_KEEP_UNION,
    COMPONENT_PACKAGE_FINAL_OBJ,
    COMPONENT_PACKAGE_SORT,
    COMPONENT_PACKAGE_SPARSE_COUPLING,
    COMPONENT_PROLONG_X,
    COMPONENT_REMAP_Y,
    COMPONENT_REFINE_DUALS,
    COMPONENT_SHIELDING,
    COMPONENT_STATE_UPDATE,
    COMPONENT_VIOLATION_CHECK,
)
from .violation_check import (
    check_constraint_violations_gpu_abs_lessGPU,
    check_constraint_violations_gpu_abs_lessGPU_local_2DT,
    grid_violation_candidates,
)

_MGPD_ROOT = None

def _load_mgpd_same_level_ops():
    return None, None, None

def _load_mgpd_lp_build_ops():
    return None, None, None

def _load_mgpd_shield_builder():
    return None

def _grid_mgpd_keepcoord_fields():
    return ["idx_coarse", "idx1", "idx2", "idx_i1", "idx_i2", "idx_j1", "idx_j2"]

def _grid_keep_to_rows_cols(keep: np.ndarray, n_t: int) -> tuple[np.ndarray, np.ndarray]:
    keep_i64 = np.asarray(keep, dtype=np.int64)
    rows = (keep_i64 // int(n_t)).astype(np.int32, copy=False)
    cols = (keep_i64 % int(n_t)).astype(np.int32, copy=False)
    return rows, cols

def _grid_rows_cols_to_keep(rows: np.ndarray, cols: np.ndarray, n_t: int) -> np.ndarray:
    return (
        np.asarray(rows, dtype=np.int64) * np.int64(n_t) + np.asarray(cols, dtype=np.int64)
    ).astype(np.int64, copy=False)

def _grid_compress_solution_keep(
    y_vals: np.ndarray,
    keep: np.ndarray,
    *,
    nnz_thr: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    y_arr = np.asarray(y_vals, dtype=np.float32)
    keep_arr = np.asarray(keep, dtype=np.int64)
    if y_arr.size == 0 or keep_arr.size == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
    mask = np.abs(y_arr) > float(nnz_thr)
    return y_arr[mask], keep_arr[mask]

class GridSolverMixin:
    

    def _grid_nnz_threshold(self) -> float:
        
        
        
        return 1e-12

    def _grid_keepcoord_resolution_last(
        self,
        *,
        level_state: Dict[str, Any],
        inner_iter: int,
    ) -> int:
        resolution_now = int(round(np.sqrt(len(level_state["level_t"].points))))
        if inner_iter == 0 and not bool(level_state.get("is_coarsest", False)):
            return max(resolution_now // 2, 1)
        return resolution_now

    def _grid_decode_keep_to_coord(
        self,
        keep: np.ndarray,
        *,
        resolution_now: int,
        resolution_last: int,
    ) -> np.ndarray:
        keep2keepcoord, _, _ = _load_mgpd_lp_build_ops()
        if keep2keepcoord is not None and resolution_now > 0:
            try:
                return keep2keepcoord(
                    np.asarray(keep, dtype=np.int64),
                    int(resolution_now),
                    int(resolution_last),
                    fields=_grid_mgpd_keepcoord_fields(),
                )
            except Exception:
                pass
        return decode_keep_1d_to_struct(np.asarray(keep, dtype=np.int64), int(resolution_now) * int(resolution_now))

    @staticmethod
    def _grid_mgpd_cost_type(cost_type: str) -> str:
        normalized = str(cost_type).strip().lower()
        if normalized in {"l2^2", "l2", "sqeuclidean"}:
            return "L2"
        if normalized == "l1":
            return "L1"
        if normalized == "linf":
            return "Linf"
        return "L2"

    def _solve_grid_lp_direct_dual(
        self,
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

        finest_resolution = int(round(np.sqrt(len(self.hierarchy_s.levels[0].points))))
        resolution_now = int(round(np.sqrt(n_s)))

        t_cost = time.perf_counter()
        minus_c = mgpd_generate_minus_c(
            int(level_idx),
            int(finest_resolution),
            keep_coord,
            self._grid_mgpd_cost_type(getattr(self, "_cost_type", "l2^2")),
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
        solver_params = getattr(self, "_tree_solver_params", None)
        if isinstance(solver_params, dict) and solver_params:
            solve_kwargs["solver_params"] = dict(solver_params)
        if self.lp_solver_verbose and hasattr(self.solver, "solve"):
            import inspect
            try:
                sig = inspect.signature(self.solver.solve)
                if "verbose" in sig.parameters:
                    solve_kwargs["verbose"] = self.lp_solver_verbose
            except (ValueError, TypeError):
                pass

        t_solve = time.perf_counter()
        result = self.solver.solve(**solve_kwargs)
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

    def _grid_init_run_state(
        self,
        tolerance: Dict[str, float],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        check_method = str(kwargs.get("grid_check_type", "gpu_exact")).lower()
        if check_method in {"gpu_exact", "gpu", "gpu_full"}:
            check_method = "gpu"
        elif check_method in {"gpu_sampled", "sampled", "approx"}:
            check_method = "gpu_approx"
        elif check_method not in {"cpu", "gpu", "gpu_approx", "auto"}:
            check_method = "gpu"

        self.solutions = {}
        self.level_summaries = []
        self.grid_lp_diags = []
        self.grid_iter_snapshots = []
        self.active_support_sizes = []
        self.dual_stats = []
        self.stop_reasons = []
        self.keep_last = None
        self.active_support = None
        self._lp_solver_kwargs = {
            "save_info": 2,
            "termination_evaluation_frequency": 200,
        }
        return {
            "coarsest_idx": self.hierarchy_s.num_levels - 1,
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

    def _grid_init_level_state(
        self,
        level_idx: int,
        run_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        level_s = self.hierarchy_s.levels[level_idx]
        level_t = self.hierarchy_t.levels[level_idx]
        n_s = len(level_s.points)
        n_t = len(level_t.points)
        inf_thr, stop_thr = self._prepare_level_tolerances(
            run_state.get("stop_tolerance", run_state["tolerance"]),
            level_idx,
        )

        state = {
            "level_idx": int(level_idx),
            "is_coarsest": int(level_idx) == int(run_state["coarsest_idx"]),
            "level_s": level_s,
            "level_t": level_t,
            "level_s_lp": self._grid_level_for_dual_lp(level_s),
            "level_t_lp": self._grid_level_for_dual_lp(level_t),
            "t_level_start": time.perf_counter(),
            "inner_iter": 0,
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

        prep = self._grid_prepare_inner(level_state=state, run_state=run_state, inner_iter=0)
        state.update(prep)
        state["curr_active_size"] = int(len(prep["keep"]))
        state["last_active_pre_lp"] = int(len(prep["keep"]))
        return state

    def _grid_prepare_inner(
        self,
        *,
        level_state: Dict[str, Any],
        run_state: Dict[str, Any],
        inner_iter: int,
    ) -> Dict[str, Any]:
        if inner_iter == 0:
            return self._grid_build_active_set_first_iter(level_state=level_state, run_state=run_state)
        return self._grid_build_active_set_subsequent_iter(level_state=level_state, run_state=run_state)

    def _grid_build_active_set_first_iter(
        self,
        *,
        level_state: Dict[str, Any],
        run_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        level_s = level_state["level_s"]
        level_t = level_state["level_t"]
        n_s = len(level_s.points)
        n_t = len(level_t.points)
        prepare_breakdown: Dict[str, float] = {}
        prepare_snapshot: Dict[str, Any] = {"level": int(level_state["level_idx"]), "iter": 0}

        if level_state["is_coarsest"]:
            keep = self._grid_full_keep(n_s, n_t)
            x_init = np.zeros(n_s + n_t, dtype=np.float32)
            y_init = {"y": np.zeros(len(keep), dtype=np.float32)}
            keep_coord = self._grid_decode_keep_to_coord(
                keep,
                resolution_now=int(round(np.sqrt(n_t))),
                resolution_last=int(round(np.sqrt(n_t))),
            )
            prepare_snapshot.update(
                {
                    "keep_refined_len": int(len(keep)),
                    "shield_keep_len": int(len(keep)),
                    "check_keep_len": int(len(keep)),
                    "use_last_merged_len": int(len(keep)),
                    "coverage_added": 0,
                    "y_init_nnz": 0,
                }
            )
            return {
                "x_init": x_init,
                "y_init": y_init,
                "keep": keep,
                "keep_coord": keep_coord,
                "prepare_breakdown": prepare_breakdown,
                "prepare_snapshot": prepare_snapshot,
                "prepare_time_total": float(sum(prepare_breakdown.values())),
            }

        x_last = run_state.get("x_solution_last")
        y_last = run_state.get("y_solution_last")
        if x_last is None or y_last is None:
            raise ValueError("grid mode expects coarse-level dual and primal warm starts.")

        t0 = time.perf_counter()
        x_init = prolong_grid_dual(
            np.asarray(x_last, dtype=np.float32),
            fine_resolution=int(round(np.sqrt(n_s))),
        )
        prepare_breakdown["prolong_x"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        keep_refined, y_refined = self._grid_refine_duals_from_coarse(
            y_solution_last=y_last,
            fine_level_s=level_s,
            fine_level_t=level_t,
        )
        prepare_breakdown["refine_duals"] = time.perf_counter() - t0
        prepare_snapshot["keep_refined_len"] = int(len(keep_refined))
        prepare_snapshot["y_init_nnz"] = int(np.count_nonzero(y_refined))

        resolution_now = int(round(np.sqrt(n_s)))
        resolution_last = max(resolution_now // 2, 1)
        keep_gpu = None
        if (
            cp is not None
            and n_s == n_t
            and resolution_now * resolution_now == n_s
            and resolution_last > 0
        ):
            try:
                keep_gpu = cp.asarray(keep_refined, dtype=cp.int64)
                y_gpu = cp.asarray(y_refined, dtype=cp.float32)
                if run_state.get("if_shield", True):
                    build_shield = _load_mgpd_shield_builder()
                    if build_shield is None:
                        raise RuntimeError("MGPD shield builder is unavailable")
                    t0 = time.perf_counter()
                    keep_gpu = build_shield(
                        int(resolution_now),
                        y_gpu,
                        keep_gpu,
                        return_gpu=True,
                        ifnaive=False,
                        MultiShield=True,
                    )
                    prepare_breakdown["shielding"] = time.perf_counter() - t0
                else:
                    prepare_breakdown["shielding"] = 0.0

                if run_state.get("if_check", True):
                    t0 = time.perf_counter()
                    max_candidates = max(int(float(run_state.get("vd_thr", 0.0625)) * 2 * n_s), 1)
                    if resolution_now >= 512:
                        check_keep_gpu, _ = check_constraint_violations_gpu_abs_lessGPU_local_2DT(
                            x_init,
                            resolution_now,
                            0.0,
                            tile=32,
                            cost_type="L2",
                            max_keep=max_candidates,
                            return_gpu=True,
                        )
                    else:
                        check_keep_gpu, _ = check_constraint_violations_gpu_abs_lessGPU(
                            x_init,
                            resolution_now,
                            0.0,
                            tile=32,
                            cost_type="L2",
                            max_keep=max_candidates,
                            return_gpu=True,
                        )
                    max_keep_valid = np.int64(n_s) * np.int64(n_t)
                    check_keep_gpu = cp.asarray(check_keep_gpu, dtype=cp.int64).reshape(-1)
                    check_keep_gpu = check_keep_gpu[
                        (check_keep_gpu >= 0) & (check_keep_gpu < max_keep_valid)
                    ]
                    keep_gpu = cp.sort(
                        cp.unique(cp.concatenate([cp.asarray(keep_gpu, dtype=cp.int64).reshape(-1), check_keep_gpu]))
                    )
                    prepare_breakdown["violation_check"] = time.perf_counter() - t0
                else:
                    prepare_breakdown["violation_check"] = 0.0

                keep = cp.asnumpy(cp.asarray(keep_gpu, dtype=cp.int64)).astype(np.int64, copy=False)
            except Exception:
                keep_gpu = None

        if keep_gpu is None:
            t0 = time.perf_counter()
            keep = (
                self._grid_expand_keep_local(
                    keep_refined,
                    y_refined,
                    n_s,
                    n_t,
                )
                if run_state.get("if_shield", True)
                else keep_refined
            )
            prepare_breakdown["shielding"] = time.perf_counter() - t0

            if run_state.get("if_check", True):
                t0 = time.perf_counter()
                keep = self._grid_apply_violation_check(
                    x_dual=x_init,
                    level_s=self._grid_level_for_dual_lp(level_s),
                    level_t=self._grid_level_for_dual_lp(level_t),
                    keep=keep,
                    vd_thr=float(run_state.get("vd_thr", 0.0625)),
                    p=int(run_state.get("p", 2)),
                )
                prepare_breakdown["violation_check"] = time.perf_counter() - t0
        prepare_snapshot["shield_keep_len"] = int(len(keep))
        prepare_snapshot["check_keep_len"] = int(len(keep))

        if run_state.get("repair_coverage", False):
            t0 = time.perf_counter()
            keep, coverage_fix = self._grid_repair_keep_coverage(keep, level_s.points, level_t.points)
            prepare_breakdown["coverage_repair"] = time.perf_counter() - t0
        else:
            coverage_fix = {"added": 0, "rows_missing": 0, "cols_missing": 0}
        prepare_snapshot["coverage_added"] = int(coverage_fix["added"])
        prepare_snapshot["rows_missing"] = int(coverage_fix["rows_missing"])
        prepare_snapshot["cols_missing"] = int(coverage_fix["cols_missing"])
        prepare_snapshot["use_last_merged_len"] = int(len(keep))

        t0 = time.perf_counter()
        keep_coord = self._grid_decode_keep_to_coord(
            keep,
            resolution_now=int(round(np.sqrt(n_t))),
            resolution_last=self._grid_keepcoord_resolution_last(level_state=level_state, inner_iter=0),
        )
        prepare_breakdown["keep_coord"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_init_arr = self._grid_build_y_init_exact_from_keep(
            y_solution_last=y_last,
            keep=keep,
            n_s=n_s,
            n_t=n_t,
            fallback={"y": y_refined, "keep": keep_refined},
            resolution_last=max(int(round(np.sqrt(n_t))) // 2, 1),
        )
        y_init = {"y": y_init_arr}
        prepare_breakdown["remap_y"] = time.perf_counter() - t0

        if run_state.get("use_last", True):
            self.keep_last = keep.copy()

        return {
            "x_init": x_init,
            "y_init": y_init,
            "keep": keep,
            "keep_coord": keep_coord,
            "prepare_breakdown": prepare_breakdown,
            "prepare_snapshot": prepare_snapshot,
            "prepare_time_total": float(sum(prepare_breakdown.values())),
        }

    def _grid_build_active_set_subsequent_iter(
        self,
        *,
        level_state: Dict[str, Any],
        run_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        level_s = level_state["level_s"]
        level_t = level_state["level_t"]
        n_t = len(level_t.points)
        x_init = np.asarray(level_state["x_solution_last"], dtype=np.float32)
        keep_last = np.asarray(level_state["y_solution_last"]["keep"], dtype=np.int64)
        keep, y_vals = self._grid_prepare_same_level_support(
            y_solution_last=level_state["y_solution_last"],
            n_s=len(level_s.points),
            n_t=len(level_t.points),
        )
        keep_seed = np.asarray(getattr(self, "_grid_same_level_keep_seed", keep), dtype=np.int64)
        prepare_breakdown: Dict[str, float] = {}
        prepare_snapshot: Dict[str, Any] = {
            "level": int(level_state["level_idx"]),
            "iter": int(level_state["inner_iter"]),
            "keep_refined_len": int(len(keep)),
            "y_init_nnz": int(np.count_nonzero(y_vals)),
        }

        if run_state.get("if_shield", True):
            t0 = time.perf_counter()
            keep = self._grid_expand_keep_local(
                keep,
                y_vals,
                len(level_s.points),
                len(level_t.points),
            )
            prepare_breakdown["shielding"] = time.perf_counter() - t0
        prepare_snapshot["shield_keep_len"] = int(len(keep))

        if run_state.get("if_check", True):
            t0 = time.perf_counter()
            keep = self._grid_apply_violation_check(
                x_dual=x_init,
                level_s=self._grid_level_for_dual_lp(level_s),
                level_t=self._grid_level_for_dual_lp(level_t),
                keep=keep,
                vd_thr=float(run_state.get("vd_thr", 0.0625)),
                p=int(run_state.get("p", 2)),
            )
            prepare_breakdown["violation_check"] = time.perf_counter() - t0
        prepare_snapshot["check_keep_len"] = int(len(keep))

        t0 = time.perf_counter()
        keep = self._merge_with_use_last(
            keep=keep,
            use_last=run_state.get("use_last", True),
            use_last_after_inner0=run_state.get("use_last_after_inner0", False),
            inner_iter=int(level_state["inner_iter"]) + 1,
            n_t=n_t,
        )
        prepare_breakdown["keep_union"] = time.perf_counter() - t0
        prepare_snapshot["use_last_merged_len"] = int(len(keep))

        if run_state.get("repair_coverage", False):
            t0 = time.perf_counter()
            keep, coverage_fix = self._grid_repair_keep_coverage(keep, level_s.points, level_t.points)
            prepare_breakdown["coverage_repair"] = time.perf_counter() - t0
        else:
            coverage_fix = {"added": 0, "rows_missing": 0, "cols_missing": 0}
        prepare_snapshot["coverage_added"] = int(coverage_fix["added"])
        prepare_snapshot["rows_missing"] = int(coverage_fix["rows_missing"])
        prepare_snapshot["cols_missing"] = int(coverage_fix["cols_missing"])

        t0 = time.perf_counter()
        keep_coord = self._grid_decode_keep_to_coord(
            keep,
            resolution_now=int(round(np.sqrt(n_t))),
            resolution_last=self._grid_keepcoord_resolution_last(level_state=level_state, inner_iter=int(level_state["inner_iter"])),
        )
        prepare_breakdown["keep_coord"] = time.perf_counter() - t0

        if np.array_equal(keep, keep_last):
            keep_seed = keep_last
            y_vals = np.asarray(level_state["y_solution_last"]["y"], dtype=np.float32)

        t0 = time.perf_counter()
        y_init = {
            "y": self._grid_build_y_init_exact_from_keep(
                y_solution_last=level_state["y_solution_last"],
                keep=keep,
                n_s=len(level_s.points),
                n_t=len(level_t.points),
                fallback={"y": y_vals, "keep": np.asarray(keep_seed, dtype=np.int64)},
                resolution_last=max(int(round(np.sqrt(len(level_t.points)))), 1),
            )
        }
        prepare_breakdown["remap_y"] = time.perf_counter() - t0

        return {
            "x_init": x_init,
            "y_init": y_init,
            "keep": keep,
            "keep_coord": keep_coord,
            "prepare_breakdown": prepare_breakdown,
            "prepare_snapshot": prepare_snapshot,
            "prepare_time_total": float(sum(prepare_breakdown.values())),
        }

    def _grid_prepare_same_level_support(
        self,
        *,
        y_solution_last: Dict[str, Any],
        n_s: int,
        n_t: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        keep_last = np.asarray(y_solution_last["keep"], dtype=np.int64)
        y_last = np.asarray(y_solution_last["y"], dtype=np.float32)
        nnz_thr = self._grid_nnz_threshold()
        nonzero_mask = np.abs(y_last) > nnz_thr
        keep_seed = keep_last[nonzero_mask]
        y_vals = y_last[nonzero_mask]
        if keep_seed.size == 0:
            self._grid_same_level_keep_seed = np.empty(0, dtype=np.int64)
            return keep_seed, y_vals

        res_s = int(round(np.sqrt(n_s)))
        res_t = int(round(np.sqrt(n_t)))
        if res_s * res_s != n_s or res_t * res_t != n_t or res_s != res_t:
            self._grid_same_level_keep_seed = keep_seed
            return keep_seed, y_vals

        topk_expand_gpu, fine_dualOT_delete_numba, keep2keepcoord = _load_mgpd_same_level_ops()
        if topk_expand_gpu is None or fine_dualOT_delete_numba is None or keep2keepcoord is None:
            self._grid_same_level_keep_seed = keep_seed
            return keep_seed, y_vals

        try:
            resolution_last = max(res_t // 2, 1)
            expanded_keep = topk_expand_gpu(
                y_last,
                keep_last,
                res_s,
                res_t,
                res_s**4,
                return_gpu=False,
            )
            expanded_keep = np.asarray(expanded_keep, dtype=np.int64)
            coord = keep2keepcoord(
                expanded_keep,
                res_s,
                resolution_last,
                fields=["idx_coarse", "idx1", "idx2", "idx_i1", "idx_i2", "idx_j1", "idx_j2"],
            )
            expanded_vals = fine_dualOT_delete_numba(
                y_val=y_last,
                y_keep=keep_last,
                coord_rec=coord,
                resolution_now=res_s,
                resolution_last=resolution_last,
            )
            self._grid_same_level_keep_seed = expanded_keep
            return np.asarray(expanded_keep, dtype=np.int64), np.asarray(expanded_vals, dtype=np.float32)
        except Exception:
            self._grid_same_level_keep_seed = keep_seed
            return keep_seed, y_vals

    def _grid_build_y_init_exact_from_keep(
        self,
        *,
        y_solution_last: Dict[str, Any],
        keep: np.ndarray,
        n_s: int,
        n_t: int,
        fallback: Dict[str, Any],
        resolution_last: int | None = None,
    ) -> np.ndarray:
        keep_arr = np.asarray(keep, dtype=np.int64)
        if keep_arr.size == 0:
            return np.empty(0, dtype=np.float32)

        source_keep = np.asarray(y_solution_last.get("keep", []), dtype=np.int64)
        source_y = np.asarray(y_solution_last.get("y", []), dtype=np.float32)
        if source_keep.size == 0 or source_y.size == 0:
            return np.zeros(keep_arr.shape[0], dtype=np.float32)

        res_s = int(round(np.sqrt(n_s)))
        res_t = int(round(np.sqrt(n_t)))
        topk_expand_gpu, fine_dualOT_delete_numba, keep2keepcoord = _load_mgpd_same_level_ops()
        if (
            topk_expand_gpu is not None
            and fine_dualOT_delete_numba is not None
            and keep2keepcoord is not None
            and res_s == res_t
            and res_s * res_s == n_s
            and res_t * res_t == n_t
        ):
            try:
                resolution_last_eff = int(resolution_last) if resolution_last is not None else max(res_t // 2, 1)
                keep_coord_exact = keep2keepcoord(
                    keep_arr,
                    res_s,
                    resolution_last_eff,
                    fields=_grid_mgpd_keepcoord_fields(),
                )
                return np.asarray(
                    fine_dualOT_delete_numba(
                        y_val=source_y,
                        y_keep=source_keep,
                        coord_rec=keep_coord_exact,
                        resolution_now=res_s,
                        resolution_last=resolution_last_eff,
                    ),
                    dtype=np.float32,
                )
            except Exception:
                pass

        return remap_duals_for_warm_start(
            {"y": np.asarray(fallback["y"], dtype=np.float32), "keep": np.asarray(fallback["keep"], dtype=np.int64)},
            keep_arr,
        )

    def _grid_solve_lp_pack(
        self,
        level_state: Dict[str, Any],
        run_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        components: Dict[str, float] = {}
        t_level_view = time.perf_counter()
        level_s_lp = level_state.get("level_s_lp")
        if level_s_lp is None:
            level_s_lp = self._grid_level_for_dual_lp(level_state["level_s"])
            level_state["level_s_lp"] = level_s_lp
        level_t_lp = level_state.get("level_t_lp")
        if level_t_lp is None:
            level_t_lp = self._grid_level_for_dual_lp(level_state["level_t"])
            level_state["level_t_lp"] = level_t_lp
        level_view_dt = time.perf_counter() - t_level_view
        if level_view_dt > 0.0:
            components[COMPONENT_LP_LEVEL_VIEW_PREPARE] = level_view_dt
        level_idx = int(level_state["level_idx"])
        inner_iter = int(level_state["inner_iter"])
        num_scale = int(run_state.get("coarsest_idx", 0))
        t_param_pack = time.perf_counter()
        self._tree_solver_params = {
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
            getattr(self.solver, "supports_tree_lp_form", False)
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
        if can_use_direct_grid_path:
            lp_result = self._solve_grid_lp_direct_dual(
                level_s=level_s_lp,
                level_t=level_t_lp,
                keep=level_state["keep"],
                keep_coord=level_state["keep_coord"],
                x_init=level_state["x_init"],
                y_init=(level_state["y_init"]["y"] if isinstance(level_state.get("y_init"), dict) else level_state.get("y_init")),
                stop_thr=float(level_state["stop_thr"]),
                inner_iter=int(level_state["inner_iter"]),
                prev_PrimalFeas=level_state.get("prev_PrimalFeas"),
                level_idx=int(level_state["level_idx"]),
                is_coarsest=bool(level_state.get("is_coarsest", False)),
            )
        else:
            lp_result = self._solve_tree_lp(
                level_s=level_s_lp,
                level_t=level_t_lp,
                keep_coord=level_state["keep_coord"],
                x_init=level_state["x_init"],
                y_init=(level_state["y_init"]["y"] if isinstance(level_state.get("y_init"), dict) else level_state.get("y_init")),
                stop_thr=float(level_state["stop_thr"]),
                inner_iter=int(level_state["inner_iter"]),
                prev_PrimalFeas=level_state.get("prev_PrimalFeas"),
                level_idx=int(level_state["level_idx"]),
                is_coarsest=bool(level_state.get("is_coarsest", False)),
                tree_debug=False,
                require_full_coverage=False,
            )
        self._tree_solver_params = None
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
            lp_result["primal_infeas_ori"] = self._grid_compute_primal_infeas(
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
                "iter": int(level_state["inner_iter"]),
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
        self.grid_lp_diags.append(diag)
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

    def _grid_update_active_set(
        self,
        level_state: Dict[str, Any],
        run_state: Dict[str, Any],
        cost_type: str,
        lp_pack: Dict[str, Any],
    ) -> Dict[str, Any]:
        if level_state.get("is_coarsest"):
            return {"pricing_time": 0.0, "components": {}}

        level_s = level_state["level_s"]
        level_t = level_state["level_t"]
        pricing_t0 = time.perf_counter()
        y_last = np.asarray(lp_pack["lp_result"]["y"], dtype=np.float32)
        keep_last = np.asarray(lp_pack["keep"], dtype=np.int64)
        nonzero_mask = np.abs(y_last) > self._grid_nnz_threshold()
        keep = keep_last[nonzero_mask]
        y_vals = y_last[nonzero_mask]
        components: Dict[str, float] = {}
        update_snapshot: Dict[str, Any] = {
            "level": int(level_state["level_idx"]),
            "iter": int(level_state["inner_iter"]),
            "keep_from_solution_len": int(len(keep)),
            "y_solution_nnz": int(np.count_nonzero(y_vals)),
        }

        if run_state.get("if_shield", True):
            t0 = time.perf_counter()
            keep = self._grid_expand_keep_local(
                keep,
                y_vals,
                len(level_s.points),
                len(level_t.points),
            )
            components["shielding"] = time.perf_counter() - t0
        update_snapshot["shield_keep_len"] = int(len(keep))

        if run_state.get("if_check", True):
            t0 = time.perf_counter()
            keep = self._grid_apply_violation_check(
                x_dual=np.asarray(lp_pack["lp_result"]["x"], dtype=np.float32),
                level_s=self._grid_level_for_dual_lp(level_s),
                level_t=self._grid_level_for_dual_lp(level_t),
                keep=keep,
                vd_thr=float(run_state.get("vd_thr", 0.0625)),
                p=int(run_state.get("p", 2)),
            )
            components["violation_check"] = time.perf_counter() - t0
        update_snapshot["check_keep_len"] = int(len(keep))

        t0 = time.perf_counter()
        keep = self._merge_with_use_last(
            keep=keep,
            use_last=run_state.get("use_last", True),
            use_last_after_inner0=run_state.get("use_last_after_inner0", False),
            inner_iter=int(level_state["inner_iter"]) + 1,
            n_t=len(level_t.points),
        )
        components["keep_union"] = time.perf_counter() - t0
        update_snapshot["use_last_merged_len"] = int(len(keep))

        if run_state.get("repair_coverage", False):
            t0 = time.perf_counter()
            keep, coverage_fix = self._grid_repair_keep_coverage(keep, level_s.points, level_t.points)
            components["coverage_repair"] = time.perf_counter() - t0
        else:
            coverage_fix = {"added": 0, "rows_missing": 0, "cols_missing": 0}
        update_snapshot["coverage_added"] = int(coverage_fix["added"])
        update_snapshot["rows_missing"] = int(coverage_fix["rows_missing"])
        update_snapshot["cols_missing"] = int(coverage_fix["cols_missing"])

        t0 = time.perf_counter()
        keep_coord = self._grid_decode_keep_to_coord(
            keep,
            resolution_now=int(round(np.sqrt(len(level_t.points)))),
            resolution_last=int(round(np.sqrt(len(level_t.points)))),
        )
        components["keep_coord"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_init = {"y": remap_duals_for_warm_start({"y": y_vals, "keep": keep_last[nonzero_mask]}, keep)}
        components["remap_y"] = time.perf_counter() - t0

        level_state["keep"] = keep
        level_state["keep_coord"] = keep_coord
        level_state["x_init"] = np.asarray(lp_pack["lp_result"]["x"], dtype=np.float32)
        level_state["y_init"] = y_init
        level_state["curr_active_size"] = int(len(keep))
        level_state["last_active_pre_lp"] = int(len(lp_pack["keep"]))
        self.active_support_sizes.append(
            {"level": int(level_state["level_idx"]), "iter": int(level_state["inner_iter"]), "size": int(len(keep))}
        )
        self.grid_iter_snapshots.append(update_snapshot)
        pricing_time = time.perf_counter() - pricing_t0
        components["pricing"] = pricing_time
        return {"pricing_time": pricing_time, "components": components}

    def _grid_should_stop(
        self,
        level_state: Dict[str, Any],
        run_state: Dict[str, Any],
        max_inner_iter: int,
        step_pack: Dict[str, Any],
    ) -> bool:
        if level_state.get("is_coarsest"):
            level_state["stop_reason"] = "coarsest_single_lp"
            self.stop_reasons.append({"level": int(level_state["level_idx"]), "reason": level_state["stop_reason"]})
            return True

        lp_result = step_pack["lp_result"]
        primal_ori = lp_result.get("primal_infeas_ori")
        dual_feas = lp_result.get("dual_feas")
        gap = lp_result.get("gap")
        self.dual_stats.append(
            {
                "level": int(level_state["level_idx"]),
                "iter": int(level_state["inner_iter"]),
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
            self.stop_reasons.append({"level": int(level_state["level_idx"]), "reason": level_state["stop_reason"]})
            return True
        if primal_ori is not None and float(primal_ori) > 1.0:
            level_state["stop_reason"] = "primal_infeas_too_large"
            self.stop_reasons.append({"level": int(level_state["level_idx"]), "reason": level_state["stop_reason"]})
            return True
        if level_state["inner_iter"] + 1 >= max_inner_iter:
            level_state["stop_reason"] = "max_inner_iter"
            self.stop_reasons.append({"level": int(level_state["level_idx"]), "reason": level_state["stop_reason"]})
            return True
        return False

    def _grid_compute_primal_infeas(
        self,
        x_dual: np.ndarray,
        level_s,
        level_t,
    ) -> float:
        n_s = int(len(level_s.points))
        n_t = int(len(level_t.points))
        if n_s != n_t:
            return float("inf")
        resolution = int(round(np.sqrt(n_s)))
        if resolution * resolution != n_s:
            return float("inf")

        cost_type_raw = str(getattr(self, "_cost_type", "l2^2")).strip().lower()
        if cost_type_raw in {"l2^2", "sqeuclidean", "l2sq"}:
            cost_type = "L2"
        elif cost_type_raw in {"l2", "euclidean"}:
            raise ValueError("grid mode does not support 'l2'; use 'l2^2' for squared Euclidean distance.")
        elif cost_type_raw == "l1":
            cost_type = "L1"
        elif cost_type_raw == "linf":
            cost_type = "Linf"
        else:
            return float("inf")

        try:
            return float(
                dualOT_primal_infeas_grid_auto(
                    x_dual,
                    resolution,
                    cost_type=cost_type,
                    p=int(getattr(self, "_cost_p", 2)),
                    use_cupy=True,
                )
            )
        except Exception:
            return float("inf")

    def _grid_record_level_result(
        self,
        level_state: Dict[str, Any],
    ) -> None:
        level_s = level_state["level_s"]
        level_t = level_state["level_t"]
        y_solution = level_state["y_solution_last"]
        x_solution = level_state["x_solution_last"]
        support_final = int(len(np.asarray(y_solution.get("keep", []), dtype=np.int64)))
        self.solutions[level_state["level_idx"]] = {
            "primal": y_solution,
            "dual": x_solution,
        }
        self.level_summaries.append(
            {
                "level": int(level_state["level_idx"]),
                "n_source": int(len(level_s.points)),
                "n_target": int(len(level_t.points)),
                "iters": int(level_state["inner_iter"]),
                "time": float(time.perf_counter() - level_state["t_level_start"]),
                "lp_time": float(level_state.get("level_lp_time", 0.0)),
                "pricing_time": float(level_state.get("level_pricing_time", 0.0)),
                "support_pre_lp_final": int(level_state.get("last_active_pre_lp") or 0),
                "support_final": support_final,
                "stop_reason": str(level_state.get("stop_reason", "unknown")),
            }
        )

    def _materialize_grid_result_payload(
        self,
        result: Dict[str, Any],
        *,
        need_final_obj: bool,
        need_sparse_coupling: bool,
    ) -> Dict[str, Any]:
        if not result:
            return result
        primal = result.get("primal")
        if not isinstance(primal, dict):
            if need_final_obj and "final_obj" not in result:
                result["final_obj"] = float("nan")
            if need_sparse_coupling and "sparse_coupling" not in result:
                result["sparse_coupling"] = None
            return result

        keep = np.asarray(primal.get("keep", []), dtype=np.int64)
        vals = np.asarray(primal.get("y", []), dtype=np.float32)
        if need_final_obj and "final_obj" not in result:
            final_obj = 0.0
            if keep.size > 0 and vals.size > 0:
                flows = np.abs(vals).astype(np.float32, copy=False)
                mask = flows > 1e-12
                if np.any(mask):
                    costs = self._grid_cost_from_keep(
                        keep=np.asarray(keep[mask], dtype=np.int64),
                        level_s=self.hierarchy_s.levels[0],
                        level_t=self.hierarchy_t.levels[0],
                    )
                    final_obj = float(np.dot(flows[mask].astype(np.float64), costs.astype(np.float64)))
            result["final_obj"] = float(final_obj)
        if need_sparse_coupling and "sparse_coupling" not in result:
            sparse_coupling = None
            if keep.size > 0 and vals.size > 0:
                n_t = len(self.hierarchy_t.levels[0].points)
                rows, cols = _grid_keep_to_rows_cols(keep, n_t)
                flows = np.abs(vals).astype(np.float32, copy=False)
                mask = flows > 1e-12
                sparse_coupling = {
                    "rows": np.asarray(rows[mask], dtype=np.int32),
                    "cols": np.asarray(cols[mask], dtype=np.int32),
                    "values": np.asarray(flows[mask], dtype=np.float32),
                }
            result["sparse_coupling"] = sparse_coupling
        return result

    def _package_grid_result(self) -> Dict[str, Any]:
        final_sol = self.solutions[0]
        return {
            "primal": final_sol.get("primal"),
            "dual": final_sol.get("dual"),
            "all_history": self.solutions,
            "level_summaries": list(getattr(self, "level_summaries", [])),
            "grid_lp_diags": list(getattr(self, "grid_lp_diags", [])),
            "grid_iter_snapshots": list(getattr(self, "grid_iter_snapshots", [])),
            "lp_solve_time_total": self._sum_level_summary_metric(
                getattr(self, "level_summaries", []),
                "lp_time",
            ),
            "active_support_sizes": list(getattr(self, "active_support_sizes", [])),
            "dual_stats": list(getattr(self, "dual_stats", [])),
            "stop_reasons": list(getattr(self, "stop_reasons", [])),
            "solver_mode": "grid",
        }

    def _grid_full_keep(self, n_s: int, n_t: int) -> np.ndarray:
        return np.arange(int(n_s) * int(n_t), dtype=np.int64)

    def _grid_level_for_dual_lp(self, level: Any) -> Any:
        level_points = np.asarray(level.points, dtype=np.float32)
        level_res = int(np.max(level_points)) + 1 if level_points.size > 0 else 1
        scale = max(float(level_res), 1.0)
        return SimpleNamespace(
            points=level_points / scale,
            masses=np.asarray(level.masses, dtype=np.float32),
        )

    def _grid_cost_from_keep(self, keep: np.ndarray, level_s: Any, level_t: Any) -> np.ndarray:
        rows, cols = _grid_keep_to_rows_cols(keep, len(level_t.points))
        s = np.asarray(level_s.points, dtype=np.float32)
        t = np.asarray(level_t.points, dtype=np.float32)
        cost_type = str(getattr(self, "_cost_type", "l2^2")).lower()
        p = int(getattr(self, "_grid_p", 2))
        if cost_type == "l2^2":
            res = float(max(int(np.max(s)) + 1, int(np.max(t)) + 1))
            diff = s[np.asarray(rows, dtype=np.int64)] - t[np.asarray(cols, dtype=np.int64)]
            return (np.sum(diff * diff, axis=1) / (res * res)).astype(np.float32, copy=False)
        return grid_pairwise_cost(s, t, cost_type=cost_type, p=p)[rows, cols].astype(np.float32, copy=False)

    def _grid_refine_duals_from_coarse(
        self,
        *,
        y_solution_last: Dict[str, Any],
        fine_level_s: Any,
        fine_level_t: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        keep_coarse = np.asarray(y_solution_last["keep"], dtype=np.int64)
        vals_coarse = np.asarray(y_solution_last["y"], dtype=np.float32)
        if keep_coarse.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        nonzero_mask = np.abs(vals_coarse) > self._grid_nnz_threshold()
        keep_nonzero = keep_coarse[nonzero_mask]
        vals_nonzero = vals_coarse[nonzero_mask]
        if keep_nonzero.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        fine_n_s = int(len(fine_level_s.points))
        fine_n_t = int(len(fine_level_t.points))
        fine_res_s = int(round(np.sqrt(fine_n_s)))
        fine_res_t = int(round(np.sqrt(fine_n_t)))
        topk_expand_gpu, fine_dualOT_delete_numba, keep2keepcoord = _load_mgpd_same_level_ops()
        if (
            topk_expand_gpu is not None
            and fine_dualOT_delete_numba is not None
            and keep2keepcoord is not None
            and fine_res_s == fine_res_t
            and fine_res_s * fine_res_s == fine_n_s
            and fine_res_t * fine_res_t == fine_n_t
        ):
            try:
                resolution_now = fine_res_s
                resolution_last = max(resolution_now // 2, 1)
                keep_exact = np.asarray(
                    topk_expand_gpu(
                        vals_nonzero,
                        keep_nonzero,
                        resolution_now,
                        resolution_last,
                        resolution_now**4,
                        return_gpu=False,
                    ),
                    dtype=np.int64,
                )
                if keep_exact.size == 0:
                    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
                keep_coord_exact = keep2keepcoord(
                    keep_exact,
                    resolution_now,
                    resolution_last,
                    fields=_grid_mgpd_keepcoord_fields(),
                )
                vals_exact = np.asarray(
                    fine_dualOT_delete_numba(
                        y_val=vals_nonzero,
                        y_keep=keep_nonzero,
                        coord_rec=keep_coord_exact,
                        resolution_now=resolution_now,
                        resolution_last=resolution_last,
                    ),
                    dtype=np.float32,
                )
                return keep_exact.astype(np.int64, copy=False), vals_exact
            except Exception:
                pass

        coarse_level_idx = int(fine_level_s.level_idx) + 1
        child_s = np.asarray(self.hierarchy_s.levels[coarse_level_idx].child_labels, dtype=np.int32)
        child_t = np.asarray(self.hierarchy_t.levels[coarse_level_idx].child_labels, dtype=np.int32)
        children_s = [np.flatnonzero(child_s == idx).astype(np.int32) for idx in range(int(np.max(child_s)) + 1)]
        children_t = [np.flatnonzero(child_t == idx).astype(np.int32) for idx in range(int(np.max(child_t)) + 1)]
        n_t_fine = len(fine_level_t.points)
        coarse_res = len(self.hierarchy_t.levels[coarse_level_idx].points)
        fine_res = len(fine_level_t.points)
        scale = max(int(round(np.sqrt(fine_res // max(coarse_res, 1)))), 1)
        value_scale = np.float32(scale**4)

        keep_parts = []
        val_parts = []
        rows_coarse, cols_coarse = _grid_keep_to_rows_cols(keep_nonzero, len(self.hierarchy_t.levels[coarse_level_idx].points))
        for src_parent, tgt_parent, val in zip(rows_coarse, cols_coarse, vals_nonzero):
            fine_rows = children_s[int(src_parent)]
            fine_cols = children_t[int(tgt_parent)]
            if fine_rows.size == 0 or fine_cols.size == 0:
                continue
            rr = np.repeat(fine_rows, fine_cols.size)
            cc = np.tile(fine_cols, fine_rows.size)
            kk = _grid_rows_cols_to_keep(rr, cc, n_t_fine)
            keep_parts.append(kk)
            val_parts.append(np.full(kk.shape[0], val / value_scale, dtype=np.float32))
        if not keep_parts:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        keep = np.concatenate(keep_parts)
        vals = np.concatenate(val_parts)
        order = np.argsort(keep)
        keep = keep[order]
        vals = vals[order]
        unique_keep, start_idx = np.unique(keep, return_index=True)
        vals_acc = np.add.reduceat(vals, start_idx).astype(np.float32, copy=False)
        return unique_keep.astype(np.int64, copy=False), vals_acc

    def _grid_collect_violation_candidates(
        self,
        *,
        x_dual: np.ndarray,
        level_s: Any,
        level_t: Any,
        vd_thr: float,
        p: int,
    ) -> np.ndarray:
        dual = np.asarray(x_dual, dtype=np.float32)
        n_s = int(len(level_s.points))
        n_t = int(len(level_t.points))
        resolution_s = int(round(np.sqrt(n_s)))
        resolution_t = int(round(np.sqrt(n_t)))
        max_candidates = max(int(vd_thr * 2 * n_s), 1)
        check_keep = None

        if (
            resolution_s == resolution_t
            and resolution_s * resolution_s == n_s
            and resolution_t * resolution_t == n_t
            and str(getattr(self, "_cost_type", "l2^2")).lower() in {"l2^2"}
            and int(p) == 2
        ):
            try:
                if resolution_s >= 512:
                    keep_raw, _ = check_constraint_violations_gpu_abs_lessGPU_local_2DT(
                        dual,
                        resolution_s,
                        0.0,
                        tile=32,
                        cost_type="L2",
                        max_keep=max_candidates,
                        return_gpu=True,
                    )
                else:
                    keep_raw, _ = check_constraint_violations_gpu_abs_lessGPU(
                        dual,
                        resolution_s,
                        0.0,
                        tile=32,
                        cost_type="L2",
                        max_keep=max_candidates,
                        return_gpu=True,
                    )
                import cupy as cp

                max_keep_valid = np.int64(n_s) * np.int64(n_t)
                check_gpu = cp.asarray(keep_raw, dtype=cp.int64).reshape(-1)
                check_gpu = check_gpu[(check_gpu >= 0) & (check_gpu < max_keep_valid)]
                return cp.asnumpy(check_gpu).astype(np.int64, copy=False)
            except Exception:
                check_keep = None

        if check_keep is None:
            check_keep = grid_violation_candidates(
                dual,
                level_s.points,
                level_t.points,
                cost_type=str(getattr(self, "_cost_type", "l2^2")),
                p=p,
                vd_thr=vd_thr,
                max_candidates=max_candidates,
            )
        return np.asarray(check_keep, dtype=np.int64)

    def _grid_apply_violation_check(
        self,
        *,
        x_dual: np.ndarray,
        level_s: Any,
        level_t: Any,
        keep: np.ndarray,
        vd_thr: float,
        p: int,
    ) -> np.ndarray:
        check_keep = self._grid_collect_violation_candidates(
            x_dual=x_dual,
            level_s=level_s,
            level_t=level_t,
            vd_thr=vd_thr,
            p=p,
        )
        if check_keep.size == 0:
            return np.asarray(keep, dtype=np.int64)
        if keep.size == 0:
            return np.asarray(check_keep, dtype=np.int64)
        return np.unique(
            np.concatenate([np.asarray(keep, dtype=np.int64), np.asarray(check_keep, dtype=np.int64)])
        ).astype(np.int64, copy=False)

    def _grid_expand_keep_local(
        self,
        keep: np.ndarray,
        y_vals: np.ndarray,
        n_s: int,
        n_t: int,
    ) -> np.ndarray:
        if keep.size == 0:
            return np.asarray(keep, dtype=np.int64)
        res_s = int(round(np.sqrt(n_s)))
        res_t = int(round(np.sqrt(n_t)))
        if res_s != res_t or res_s * res_s != n_s or res_t * res_t != n_t:
            return np.asarray(keep, dtype=np.int64)
        build_shield = _load_mgpd_shield_builder()
        if build_shield is not None:
            try:
                keep_exact = build_shield(
                    int(res_s),
                    np.asarray(y_vals, dtype=np.float32),
                    np.asarray(keep, dtype=np.int64),
                    return_gpu=False,
                    ifnaive=False,
                    MultiShield=True,
                )
                return np.asarray(keep_exact, dtype=np.int64)
            except Exception:
                pass
        return build_grid_shield(res_s, y_vals, keep)

    def _grid_repair_keep_coverage(
        self,
        keep: np.ndarray,
        points_s: np.ndarray,
        points_t: np.ndarray,
    ) -> tuple[np.ndarray, Dict[str, int]]:
        keep_arr = np.asarray(keep, dtype=np.int64)
        n_s = int(np.asarray(points_s).shape[0])
        n_t = int(np.asarray(points_t).shape[0])
        if keep_arr.size == 0 or n_s == 0 or n_t == 0:
            return keep_arr, {
                "rows_missing": int(n_s),
                "cols_missing": int(n_t),
                "added": 0,
            }
        if n_s != n_t:
            return keep_arr, {
                "rows_missing": 0,
                "cols_missing": 0,
                "added": 0,
            }

        rows, cols = _grid_keep_to_rows_cols(keep_arr, n_t)
        row_mask = np.zeros(n_s, dtype=bool)
        col_mask = np.zeros(n_t, dtype=bool)
        row_mask[rows] = True
        col_mask[cols] = True
        missing_rows = np.flatnonzero(~row_mask)
        missing_cols = np.flatnonzero(~col_mask)
        if missing_rows.size == 0 and missing_cols.size == 0:
            return keep_arr, {
                "rows_missing": 0,
                "cols_missing": 0,
                "added": 0,
            }

        additions = []
        for row in missing_rows:
            additions.append(np.int64(row) * np.int64(n_t) + np.int64(row))
        for col in missing_cols:
            additions.append(np.int64(col))
        keep_fixed = np.unique(
            np.concatenate([keep_arr, np.asarray(additions, dtype=np.int64)])
        ).astype(np.int64, copy=False)
        return keep_fixed, {
            "rows_missing": int(len(missing_rows)),
            "cols_missing": int(len(missing_cols)),
            "added": int(len(keep_fixed) - len(keep_arr)),
        }

class GridSolverBackend(SolverBackend):
    name = "grid"

    def should_exact_flat(self) -> bool:
        return self.solver.hierarchy_s.num_levels <= 1

    def solve_exact_flat(self, tolerance: Dict[str, float]) -> Optional[Dict[str, Any]]:
        return self.solver._solve_exact_flat(tolerance)

    def init_run_state(self, tolerance: Dict[str, float], **kwargs: Any) -> CommonRunState:
        return self.solver._grid_init_run_state(tolerance, **kwargs)

    def get_level_indices(self, run_state: CommonRunState):
        return range(run_state["coarsest_idx"], -1, -1)

    def init_level_state(
        self,
        level_idx: int,
        run_state: CommonRunState,
        tolerance: Dict[str, float],
        cost_type: str,
        use_bfs_skeleton: bool,
    ) -> Optional[CommonLevelState]:
        del tolerance, cost_type, use_bfs_skeleton
        print(f"\n=== Solve Grid Level {level_idx} =====")
        state = self.solver._grid_init_level_state(level_idx, run_state)
        prepare_breakdown = state.get("prepare_breakdown", {})
        if isinstance(prepare_breakdown, dict):
            prepare_cpu_components: Dict[str, float] = {}
            for key, name in (
                ("prolong_x", COMPONENT_PROLONG_X),
                ("refine_duals", COMPONENT_REFINE_DUALS),
                ("keep_union", COMPONENT_KEEP_UNION),
                ("keep_coord", COMPONENT_KEEP_COORD),
                ("remap_y", COMPONENT_REMAP_Y),
                ("shielding", COMPONENT_SHIELDING),
                ("violation_check", COMPONENT_VIOLATION_CHECK),
                ("coverage_repair", COMPONENT_COVERAGE_REPAIR),
            ):
                dt = float(prepare_breakdown.get(key, 0.0))
                if dt > 0.0:
                    prepare_cpu_components[name] = dt
            self.solver._profiler.add_components(prepare_cpu_components)
        return state

    def prepare_iteration_input(
        self,
        level_state: CommonLevelState,
        run_state: CommonRunState,
        inner_iter: int,
    ) -> None:
        solver = self.solver
        if inner_iter <= 0:
            prep = {
                "keep": level_state.get("keep", np.empty(0, dtype=np.int64)),
                "prepare_breakdown": level_state.get("prepare_breakdown", {}),
                "prepare_time_total": level_state.get("prepare_time_total", 0.0),
            }
        else:
            prep = solver._grid_prepare_inner(
                level_state=level_state,
                run_state=run_state,
                inner_iter=inner_iter,
            )
            level_state.update(prep)
            level_state["curr_active_size"] = int(len(prep["keep"]))
            level_state["last_active_pre_lp"] = int(len(prep["keep"]))
        prepare_time = float(prep.get("prepare_time_total", 0.0))
        if prepare_time > 0.0:
            level_state["level_pricing_time"] = (
                level_state.get("level_pricing_time", 0.0) + prepare_time
            )
        prepare_breakdown = prep.get("prepare_breakdown", {})
        if inner_iter > 0 and isinstance(prepare_breakdown, dict):
            prepare_cpu_components: Dict[str, float] = {}
            for key, name in (
                ("prolong_x", COMPONENT_PROLONG_X),
                ("refine_duals", COMPONENT_REFINE_DUALS),
                ("keep_union", COMPONENT_KEEP_UNION),
                ("keep_coord", COMPONENT_KEEP_COORD),
                ("remap_y", COMPONENT_REMAP_Y),
                ("shielding", COMPONENT_SHIELDING),
                ("violation_check", COMPONENT_VIOLATION_CHECK),
                ("coverage_repair", COMPONENT_COVERAGE_REPAIR),
            ):
                dt = float(prepare_breakdown.get(key, 0.0))
                if dt > 0.0:
                    prepare_cpu_components[name] = dt
            solver._profiler.add_components(prepare_cpu_components)

    def solve_iteration_lp(
        self,
        level_state: CommonLevelState,
        run_state: CommonRunState,
    ) -> Dict[str, Any]:
        return self.solver._grid_solve_lp_pack(level_state, run_state)

    def finalize_iteration(
        self,
        level_state: CommonLevelState,
        run_state: CommonRunState,
        step_pack: Dict[str, Any],
        convergence_criterion: str,
        tolerance: Dict[str, float],
    ) -> None:
        del run_state, convergence_criterion, tolerance
        solver = self.solver
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

    def should_stop_iteration(
        self,
        level_state: CommonLevelState,
        run_state: CommonRunState,
        max_inner_iter: int,
        step_pack: Dict[str, Any],
    ) -> bool:
        t0 = time.perf_counter()
        result = self.solver._grid_should_stop(level_state, run_state, max_inner_iter, step_pack)
        self.solver._profiler.add_components({COMPONENT_CONVERGENCE_CHECK: time.perf_counter() - t0})
        return result

    def record_level_result(
        self,
        level_state: CommonLevelState,
        final_refinement_tolerance: Optional[Dict[str, float]],
    ) -> None:
        del final_refinement_tolerance
        self.solver._grid_record_level_result(level_state)

    def advance_to_next_level(
        self,
        level_state: CommonLevelState,
        run_state: CommonRunState,
    ) -> None:
        run_state["x_solution_last"] = level_state.get("x_solution_last")
        run_state["y_solution_last"] = level_state.get("y_solution_last")
        run_state["keep_last"] = self.solver.keep_last

    def package_result(self) -> Dict[str, Any]:
        return self.solver._package_grid_result()

    def extract_step_objective(self, step_pack: Dict[str, Any]) -> Optional[float]:
        objective = step_pack.get("lp_result", {}).get("obj")
        if objective is None:
            return None
        return float(objective)
