import time
import numpy as np
from scipy import sparse
from typing import Dict, Any, Optional, Tuple, List
import logging

from ..common.utils import (
    prolongate_potentials,
    refine_duals,
    decode_keep_1d_to_struct,
    generate_minus_c,
    build_minus_AT_csc,
    remap_duals_for_warm_start,
)
from .violation_check import (
    check_constraint_violations,
)
from .infeas import dualOT_primal_infeas_pointcloud_cupy_auto
from ..common.solver_backend import CommonLevelState, CommonRunState, SolverBackend
from ..common.profiling_names import (
    COMPONENT_CONVERGENCE_CHECK,
    COMPONENT_COVERAGE_REPAIR,
    COMPONENT_KEEP_COORD,
    COMPONENT_KEEP_UNION,
    COMPONENT_PRICING_TOTAL,
    COMPONENT_REFINE_DUALS,
    COMPONENT_REMAP_Y,
    COMPONENT_SHIELD_PICK_T_MAP,
    COMPONENT_SHIELD_SENTINELS,
    COMPONENT_SHIELD_UNION,
    COMPONENT_SHIELD_YHAT,
    COMPONENT_SHIELDING,
    COMPONENT_VIOLATION_CHECK,
)

logger = logging.getLogger(__name__)

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

class TreeSolverMixin:
    

    _TREE_INFEAS_FALLBACKS = {"none", "full_support_retry", "scipy_verify"}

    def _tree_log_enabled(self) -> bool:
        return bool(getattr(self, "tree_debug", False) or getattr(self, "ifdebug", False))

    def _tree_log(self, message: str) -> None:
        if self._tree_log_enabled():
            print(message)

    @staticmethod
    def _parse_bool_flag(value: Any, default: bool = False) -> bool:
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

    def _normalize_tree_infeas_fallback(self, value: Any) -> str:
        fallback = str(value or "none").strip().lower()
        if fallback not in self._TREE_INFEAS_FALLBACKS:
            raise ValueError(
                f"Unknown tree_infeas_fallback={value!r}, "
                f"must be one of {sorted(self._TREE_INFEAS_FALLBACKS)}"
            )
        return fallback

    def _repair_keep_coverage(
        self,
        keep: np.ndarray,
        n_s: int,
        n_t: int,
        points_s: np.ndarray,
        points_t: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        
        keep = np.asarray(keep, dtype=np.int64)
        if keep.size == 0:
            return keep, {"rows_missing": n_s, "cols_missing": n_t, "added": 0}

        idx_x = keep // n_t
        idx_y = keep % n_t
        row_counts = np.bincount(idx_x, minlength=n_s)
        col_counts = np.bincount(idx_y, minlength=n_t)

        missing_rows = np.where(row_counts == 0)[0]
        missing_cols = np.where(col_counts == 0)[0]

        if missing_rows.size == 0 and missing_cols.size == 0:
            return keep, {"rows_missing": 0, "cols_missing": 0, "added": 0}

        extra: List[int] = []
        if n_t > 0:
            for i in missing_rows:
                
                d2 = np.sum((points_t - points_s[i]) ** 2, axis=1)
                j = int(np.argmin(d2))
                extra.append(int(i) * n_t + j)
        if n_s > 0:
            for j in missing_cols:
                d2 = np.sum((points_s - points_t[j]) ** 2, axis=1)
                i = int(np.argmin(d2))
                extra.append(int(i) * n_t + int(j))

        keep_repaired = np.unique(np.concatenate([keep, np.asarray(extra, dtype=np.int64)]))
        return keep_repaired, {
            "rows_missing": int(missing_rows.size),
            "cols_missing": int(missing_cols.size),
            "added": int(keep_repaired.size - keep.size),
        }

    def _verify_lp_with_scipy(
        self,
        c: np.ndarray,
        A_eq: sparse.csc_matrix,
        b_eq: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> Dict[str, Any]:
        
        if c is None or A_eq is None or b_eq is None or lb is None or ub is None:
            return {
                "ran": False,
                "success": False,
                "status": "SKIPPED",
                "message": "Missing LP payload",
            }
        if c.size > 200000:
            return {
                "ran": False,
                "success": False,
                "status": "SKIPPED",
                "message": "Too many variables for scipy_verify",
            }

        try:
            from scipy.optimize import linprog
            bounds = list(zip(lb, ub))
            result = linprog(
                c,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
                options={"presolve": True},
            )
            return {
                "ran": True,
                "success": bool(result.success),
                "status": str(result.status),
                "message": str(result.message),
            }
        except Exception as exc:
            return {
                "ran": True,
                "success": False,
                "status": "ERROR",
                "message": f"scipy_verify failed: {exc}",
            }

    def _solve_tree(
        self,
        max_inner_iter: int,
        tolerance: Dict[str, float],
        cost_type: str,
        lp_solver_verbose: int = 0,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        
        self.lp_solver_verbose = lp_solver_verbose
        return self._solve_hierarchical(
            max_inner_iter=max_inner_iter,
            convergence_criterion="objective",
            tolerance=tolerance,
            final_refinement_tolerance=None,
            cost_type=cost_type,
            use_bfs_skeleton=True,
            mode="tree",
            **kwargs,
        )

    def _prepare_level_tolerances(
        self,
        tolerance: Dict[str, float],
        level_idx: int,
    ) -> Tuple[float, float]:
        
        if isinstance(tolerance, dict):
            inf_thrs = tolerance.get('inf_thrs', None)
            stop_thrs = tolerance.get('stop_thrs', None)

            if inf_thrs is not None and isinstance(inf_thrs, (list, tuple)):
                inf_thr = inf_thrs[level_idx] if level_idx < len(inf_thrs) else inf_thrs[-1]
            else:
                inf_thr = tolerance.get('primal', 1e-6)

            if stop_thrs is not None and isinstance(stop_thrs, (list, tuple)):
                stop_thr = stop_thrs[level_idx] if level_idx < len(stop_thrs) else stop_thrs[-1]
            else:
                stop_thr = tolerance.get('objective', 1e-6)
        else:
            inf_thr = tolerance if isinstance(tolerance, (int, float)) else 1e-6
            stop_thr = tolerance if isinstance(tolerance, (int, float)) else 1e-6
        return float(inf_thr), float(stop_thr)

    def _init_level_state(
        self,
        level_idx: int,
        inf_thr: float,
        stop_thr: float,
        x_solution_last,
        y_solution_last,
    ) -> Dict[str, Any]:
        
        
        self.keep_last = None
        return {
            "level_idx": level_idx,
            "inner_iter": 0,
            "converged": False,
            "inf_thr": inf_thr,
            "stop_thr": stop_thr,
            "prev_active_size": None,
            "prev_PrimalFeas": None,
            "last_active_pre_lp": None,
            "x_solution_last": x_solution_last,
            "y_solution_last": y_solution_last,
        }

    def _solve_tree_single_level(
        self,
        level_idx: int,
        level_s,
        level_t,
        max_inner_iter: int,
        tolerance: Dict[str, float],
        cost_type: str,
        tree_debug: bool,
        tree_infeas_fallback: str,
        use_last: bool,
        use_last_after_inner0: bool,
        ifcheck: bool,
        vd_thr: float,
        check_method: str,
        x_solution_last,
        y_solution_last,
    ) -> Optional[Dict[str, Any]]:
        
        print(f"\n{'='*50}")
        print(f"=== Solving Level {level_idx} (Tree Mode) ===")
        print(f"{'='*50}")

        n_s = len(level_s.points)
        n_t = len(level_t.points)
        if n_s == 0 or n_t == 0:
            return {
                "x_solution_last": x_solution_last,
                "y_solution_last": y_solution_last,
            }

        inf_thr, stop_thr = self._prepare_level_tolerances(tolerance, level_idx)
        state = self._init_level_state(
            level_idx=level_idx,
            inf_thr=inf_thr,
            stop_thr=stop_thr,
            x_solution_last=x_solution_last,
            y_solution_last=y_solution_last,
        )

        t_level_start = time.perf_counter()
        num_levels = self.hierarchy_s.num_levels
        print(f"  DEBUG: start while loop, level_idx={level_idx}")

        while not state["converged"]:
            print(f"  DEBUG: inner iteration {state['inner_iter']}")
            iter_result = self._run_tree_inner_iteration(
                level_idx=level_idx,
                level_s=level_s,
                level_t=level_t,
                state=state,
                cost_type=cost_type,
                tree_debug=tree_debug,
                tree_infeas_fallback=tree_infeas_fallback,
                use_last=use_last,
                use_last_after_inner0=use_last_after_inner0,
                ifcheck=ifcheck,
                vd_thr=vd_thr,
                check_method=check_method,
            )
            if not iter_result["success"]:
                logger.error(
                    f"Tree LP failed irrecoverably at level={level_idx}, inner={state['inner_iter']}"
                )
                return None

            lp_result = iter_result["lp_result"]
            state["x_solution_last"] = iter_result["x_solution_last"]
            state["y_solution_last"] = iter_result["y_solution_last"]

            if lp_result.get('primal_feas') is not None:
                state["prev_PrimalFeas"] = lp_result['primal_feas']

            state["converged"] = self._should_stop_tree_inner(
                lp_result=lp_result,
                state=state,
                level_s=level_s,
                level_t=level_t,
                max_inner_iter=max_inner_iter,
                is_coarsest=(level_idx == num_levels),
            )
            state["prev_active_size"] = iter_result["curr_active_size"]
            state["last_active_pre_lp"] = iter_result["curr_active_size"]
            state["inner_iter"] += 1

        t_level = time.perf_counter() - t_level_start
        self._record_tree_level_result(
            level_idx,
            state["x_solution_last"],
            state["y_solution_last"],
            t_level,
            state["inner_iter"],
            n_s,
            n_t,
            support_pre_lp_final=state.get("last_active_pre_lp"),
        )
        return {
            "x_solution_last": state["x_solution_last"],
            "y_solution_last": state["y_solution_last"],
        }

    def _tree_prepare_inner(
        self,
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
        sampled_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        
        n_s = len(level_s.points)
        n_t = len(level_t.points)
        num_levels = self.hierarchy_s.num_levels
        prepare_breakdown: Dict[str, float] = {}

        t_init = time.perf_counter()
        if inner_iter == 0:
            init_result = self._build_active_set_first_iter(
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
            )
        else:
            init_result = self._build_active_set_subsequent_iter(
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
            )

        if init_result.get("prepare_breakdown"):
            prepare_breakdown.update(init_result["prepare_breakdown"])

        x_init = init_result["x_init"]
        y_init = init_result["y_init"]
        keep = np.asarray(init_result["keep"], dtype=np.int64)
        keep_coord = init_result["keep_coord"]

        keep, coverage_fix = self._repair_keep_coverage(
            keep,
            n_s=n_s,
            n_t=n_t,
            points_s=level_s.points,
            points_t=level_t.points,
        )
        if coverage_fix["added"] > 0:
            print(
                "  [CoverageFix] rows_missing="
                f"{coverage_fix['rows_missing']}, cols_missing={coverage_fix['cols_missing']}, "
                f"added={coverage_fix['added']}"
            )
            if y_init is not None and y_init.get("y") is not None:
                y_init = {
                    "y": remap_duals_for_warm_start(
                        {"y": y_init["y"], "keep": init_result["keep"]},
                        keep,
                    )
                }
            keep_coord = decode_keep_1d_to_struct(keep, n_t)

        if level_idx == num_levels and inner_iter == 0:
            expected = n_s * n_t
            assert len(keep) == expected, (
                f"Coarsest level keep must be full support: "
                f"got {len(keep)}, expected {expected}"
            )

        curr_active_size = len(keep)
        prepare_time_total = time.perf_counter() - t_init
        print(f"  Init time: {prepare_time_total:.3f}s, Active vars: {curr_active_size}")

        return {
            "x_init": x_init,
            "y_init": y_init,
            "keep": keep,
            "keep_coord": keep_coord,
            "curr_active_size": curr_active_size,
            "prepare_breakdown": prepare_breakdown,
            "prepare_time_total": float(prepare_time_total),
            "trace_keep_after_shield": init_result.get("trace_keep_after_shield"),
            "trace_keep_after_check": init_result.get("trace_keep_after_check"),
            "trace_keep_after_uselast": init_result.get("trace_keep_after_uselast"),
        }

    def _tree_solve_lp_pack(
        self,
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
    ) -> Dict[str, Any]:
        
        t_lp = time.perf_counter()
        with self._profiler.timer("lp", gpu_possible=True):
            lp_pack = self._solve_tree_lp_with_fallback(
                level_idx=level_idx,
                level_s=level_s,
                level_t=level_t,
                keep=keep,
                keep_coord=keep_coord,
                x_init=x_init,
                y_init=(y_init['y'] if y_init else None),
                stop_thr=stop_thr,
                inner_iter=inner_iter,
                prev_PrimalFeas=prev_PrimalFeas,
                tree_debug=tree_debug,
                tree_infeas_fallback=tree_infeas_fallback,
            )
        lp_time = time.perf_counter() - t_lp
        lp_result = lp_pack["lp_result"]
        if lp_pack["success"] and lp_result.get("x") is not None:
            lp_result["primal_infeas_ori"] = self._compute_primal_infeas(
                lp_result["x"], level_s, level_t
            )
        else:
            lp_result["primal_infeas_ori"] = None
        print(f"  LP time: {lp_time:.3f}s, Iters: {lp_result.get('iters', 0)}")
        if not lp_pack["success"]:
            print(
                f"  [LP Failure] level={level_idx}, inner={inner_iter}, "
                f"reason={lp_result.get('termination_reason')}"
            )
            return {
                "success": False,
                "lp_result": lp_result,
                "keep": lp_pack["keep"],
                "keep_coord": lp_pack["keep_coord"],
                "lp_time": float(lp_time),
            }
        return {
            "success": True,
            "lp_result": lp_result,
            "keep": lp_pack["keep"],
            "keep_coord": lp_pack["keep_coord"],
            "lp_time": float(lp_time),
        }

    def _run_tree_inner_iteration(
        self,
        level_idx: int,
        level_s,
        level_t,
        state: Dict[str, Any],
        cost_type: str,
        tree_debug: bool,
        tree_infeas_fallback: str,
        use_last: bool,
        use_last_after_inner0: bool,
        ifcheck: bool,
        vd_thr: float,
        check_method: str,
    ) -> Dict[str, Any]:
        
        inner_iter = state["inner_iter"]
        prep = self._tree_prepare_inner(
            level_idx=level_idx,
            level_s=level_s,
            level_t=level_t,
            x_solution_last=state["x_solution_last"],
            y_solution_last=state["y_solution_last"],
            cost_type=cost_type,
            use_last=use_last,
            use_last_after_inner0=use_last_after_inner0,
            ifcheck=ifcheck,
            vd_thr=vd_thr,
            check_method=check_method,
            inner_iter=inner_iter,
            sampled_config=getattr(self, "check_sampled_config", None),
        )
        lp_pack = self._tree_solve_lp_pack(
            level_idx=level_idx,
            level_s=level_s,
            level_t=level_t,
            keep=prep["keep"],
            keep_coord=prep["keep_coord"],
            x_init=prep["x_init"],
            y_init=prep["y_init"],
            stop_thr=state["stop_thr"],
            inner_iter=inner_iter,
            prev_PrimalFeas=state["prev_PrimalFeas"],
            tree_debug=tree_debug,
            tree_infeas_fallback=tree_infeas_fallback,
        )
        if not lp_pack["success"]:
            return {"success": False}
        lp_result = lp_pack["lp_result"]
        keep = lp_pack["keep"]
        return {
            "success": True,
            "lp_result": lp_result,
            "curr_active_size": prep["curr_active_size"],
            "x_solution_last": lp_result["x"],
            "y_solution_last": {
                "y": lp_result["y"],
                "keep": keep,
            },
        }

    def _build_active_set_first_iter(
        self,
        level_idx: int,
        level_s,
        level_t,
        x_solution_last,
        y_solution_last,
        cost_type: str,
        use_last: bool,
        ifcheck: bool,
        vd_thr: float,
        check_method: str,
        sampled_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        
        n_s = len(level_s.points)
        n_t = len(level_t.points)
        prepare_breakdown: Dict[str, float] = {}
        nnz_thr = float(getattr(self, "nnz_thr", 1e-20))

        if level_idx == self.hierarchy_s.num_levels:
            self._tree_log(f"    [Init] coarsest level L{level_idx}, using full-support initialization")
            x_init = np.zeros(n_s + n_t, dtype=np.float32)

            init_result = self.strategy.initialize_support(
                level_s=level_s,
                level_t=level_t,
                x_init=x_init,
                hierarchy_s=self.hierarchy_s,
                hierarchy_t=self.hierarchy_t,
            )
            keep = np.asarray(init_result['keep'], dtype=np.int64)
            keep_coord = init_result['keep_coord']
            y_init = {
                'y': init_result.get('y_init', np.zeros(len(keep), dtype=np.float32))
            }

            self._tree_log(f"    [Init] keep={len(keep)} ({len(keep)/(n_s+n_t):.2f}x n_total)")
            y_vals = y_init['y']
            self._tree_log(f"    [Init] y_init: mean={y_vals.mean():.6f}, std={y_vals.std():.6f}, sum={y_vals.sum():.6f}")
            self._tree_log(f"    [Init] x_init: mean={x_init.mean():.6f}, std={x_init.std():.6f}")
            return {
                'x_init': x_init,
                'y_init': y_init,
                'keep': keep,
                'keep_coord': keep_coord,
                'prepare_breakdown': prepare_breakdown,
                'trace_keep_after_shield': None,
                'trace_keep_after_check': None,
                'trace_keep_after_uselast': None,
            }

        self._tree_log(f"    [Init] initializing L{level_idx} from L{level_idx+1}")
        coarse_level_s = self.hierarchy_s.levels[level_idx + 1]
        coarse_level_t = self.hierarchy_t.levels[level_idx + 1]

        x_init = prolongate_potentials(
            x_solution_last, level_s, level_t, coarse_level_s, coarse_level_t
        )
        t_refine = time.perf_counter()
        y_refined, keep_refined, _ = refine_duals(
            y_solution_last, level_s, level_t, coarse_level_s, coarse_level_t, thr=nnz_thr
        )
        prepare_breakdown["refine_duals"] = time.perf_counter() - t_refine
        self._tree_log(f"    [Init] refine_duals: y_keep={len(keep_refined)}")

        t_shield = time.perf_counter()
        update_result = self.strategy.update_active_support(
            x_solution=x_init,
            y_solution_last={'y': y_refined, 'keep': keep_refined},
            level_s=level_s,
            level_t=level_t,
            hierarchy_s=self.hierarchy_s,
            hierarchy_t=self.hierarchy_t,
            build_aux=False,
        )
        update_time = time.perf_counter() - t_shield
        keep_union_time = 0.0
        if isinstance(update_result, dict):
            update_timing = update_result.get("timing", {}) or {}
            keep_union_time = float(update_timing.get("keep_union", 0.0))
            for key in (
                COMPONENT_SHIELD_PICK_T_MAP,
                COMPONENT_SHIELD_SENTINELS,
                COMPONENT_SHIELD_YHAT,
                COMPONENT_SHIELD_UNION,
            ):
                dt = float(update_timing.get(key, 0.0) or 0.0)
                if dt > 0.0:
                    prepare_breakdown[key] = prepare_breakdown.get(key, 0.0) + dt
        prepare_breakdown["keep_union"] = prepare_breakdown.get("keep_union", 0.0) + keep_union_time
        prepare_breakdown["shielding"] = prepare_breakdown.get("shielding", 0.0) + max(0.0, update_time - keep_union_time)
        keep = np.asarray(update_result['keep'], dtype=np.int64)
        t_vcheck = time.perf_counter()
        keep = self._apply_violation_check(
            x_dual=x_init,
            level_s=level_s,
            level_t=level_t,
            keep=keep,
            cost_type=cost_type,
            ifcheck=ifcheck,
            vd_thr=vd_thr,
            check_method=check_method,
            sampled_config=sampled_config,
        )
        prepare_breakdown["violation_check"] = time.perf_counter() - t_vcheck
        t0 = time.perf_counter()
        keep_coord = decode_keep_1d_to_struct(keep, n_t)
        prepare_breakdown["keep_coord"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        y_init = {
            'y': remap_duals_for_warm_start({'y': y_refined, 'keep': keep_refined}, keep)
        }
        prepare_breakdown["remap_y"] = time.perf_counter() - t0
        if use_last:
            self.keep_last = keep
        return {
            'x_init': x_init,
            'y_init': y_init,
            'keep': keep,
            'keep_coord': keep_coord,
            'prepare_breakdown': prepare_breakdown,
            'trace_keep_after_shield': int(len(update_result['keep'])),
            'trace_keep_after_check': int(len(keep)),
            'trace_keep_after_uselast': int(len(keep)),
        }

    def _build_active_set_subsequent_iter(
        self,
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
        sampled_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        
        n_t = len(level_t.points)
        x_init = x_solution_last
        prepare_breakdown: Dict[str, float] = {}
        nnz_thr = float(getattr(self, "nnz_thr", 1e-20))

        y_last = y_solution_last['y']
        keep_last = y_solution_last['keep']
        nonzero_mask = np.abs(y_last) > nnz_thr
        y_keep = keep_last[nonzero_mask]
        y_vals = y_last[nonzero_mask]
        self._tree_log(f"    [Init] inner_iter>0: y_keep after nonzero_mask={len(y_keep)}")

        t_shield = time.perf_counter()
        update_result = self.strategy.update_active_support(
            x_solution=x_init,
            y_solution_last={'y': y_vals, 'keep': y_keep},
            level_s=level_s,
            level_t=level_t,
            hierarchy_s=self.hierarchy_s,
            hierarchy_t=self.hierarchy_t,
            build_aux=False,
        )
        update_time = time.perf_counter() - t_shield
        keep_union_time = 0.0
        if isinstance(update_result, dict):
            update_timing = update_result.get("timing", {}) or {}
            keep_union_time = float(update_timing.get("keep_union", 0.0))
            for key in (
                COMPONENT_SHIELD_PICK_T_MAP,
                COMPONENT_SHIELD_SENTINELS,
                COMPONENT_SHIELD_YHAT,
                COMPONENT_SHIELD_UNION,
            ):
                dt = float(update_timing.get(key, 0.0) or 0.0)
                if dt > 0.0:
                    prepare_breakdown[key] = prepare_breakdown.get(key, 0.0) + dt
        prepare_breakdown["keep_union"] = prepare_breakdown.get("keep_union", 0.0) + keep_union_time
        prepare_breakdown["shielding"] = prepare_breakdown.get("shielding", 0.0) + max(0.0, update_time - keep_union_time)
        keep = np.asarray(update_result['keep'], dtype=np.int64)
        trace_keep_after_shield = int(len(keep))
        t_vcheck = time.perf_counter()
        keep = self._apply_violation_check(
            x_dual=x_init,
            level_s=level_s,
            level_t=level_t,
            keep=keep,
            cost_type=cost_type,
            ifcheck=ifcheck,
            vd_thr=vd_thr,
            check_method=check_method,
            sampled_config=sampled_config,
        )
        prepare_breakdown["violation_check"] = time.perf_counter() - t_vcheck
        trace_keep_after_check = int(len(keep))
        t0 = time.perf_counter()
        keep = self._merge_with_use_last(
            keep=keep,
            use_last=use_last,
            use_last_after_inner0=use_last_after_inner0,
            inner_iter=inner_iter,
            n_t=n_t,
        )
        prepare_breakdown["keep_union"] = prepare_breakdown.get("keep_union", 0.0) + (time.perf_counter() - t0)
        trace_keep_after_uselast = int(len(keep))
        t0 = time.perf_counter()
        keep_coord = decode_keep_1d_to_struct(keep, n_t)
        prepare_breakdown["keep_coord"] = time.perf_counter() - t0

        if np.array_equal(keep, keep_last):
            y_init = {'y': y_last}
        else:
            t0 = time.perf_counter()
            y_init = {'y': remap_duals_for_warm_start({'y': y_vals, 'keep': y_keep}, keep)}
            prepare_breakdown["remap_y"] = time.perf_counter() - t0
        return {
            'x_init': x_init,
            'y_init': y_init,
            'keep': keep,
            'keep_coord': keep_coord,
            'prepare_breakdown': prepare_breakdown,
            'trace_keep_after_shield': trace_keep_after_shield,
            'trace_keep_after_check': trace_keep_after_check,
            'trace_keep_after_uselast': trace_keep_after_uselast,
        }

    def _apply_violation_check(
        self,
        x_dual: np.ndarray,
        level_s,
        level_t,
        keep: np.ndarray,
        cost_type: str,
        ifcheck: bool,
        vd_thr: float,
        check_method: str,
        sampled_config: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        
        keep = np.asarray(keep, dtype=np.int64)
        if not ifcheck:
            return keep

        n_s = len(level_s.points)
        n_t = len(level_t.points)
        max_keep = int(vd_thr * (n_s + n_t)) if vd_thr is not None else None
        violation_keep, _ = check_constraint_violations(
            x_dual,
            level_s,
            level_t,
            cost_type=cost_type,
            eps=0.0,
            max_keep=max_keep,
            method=check_method,
            sampled_config=sampled_config,
        )
        if violation_keep.size == 0:
            self._tree_log("    [Check] found: 0")
            return keep

        if keep.size > 0:
            new_mask = ~np.isin(violation_keep, keep)
            violation_new = violation_keep[new_mask]
        else:
            violation_new = violation_keep

        if violation_new.size == 0:
            self._tree_log(f"    [Check] found: {len(violation_keep)} (all already in active set)")
            return keep

        keep_new = np.unique(np.concatenate([keep, violation_new.astype(np.int64)]))
        self._tree_log(f"    [Check] found: {len(violation_keep)} ({len(violation_new)} new)")
        self._tree_log(f"    [Check] expanding active set: {len(keep)} -> {len(keep_new)}")
        return keep_new

    def _merge_with_use_last(
        self,
        keep: np.ndarray,
        use_last: bool,
        use_last_after_inner0: bool,
        inner_iter: int,
        n_t: int,
    ) -> np.ndarray:
        
        keep = np.asarray(keep, dtype=np.int64)
        if not use_last:
            return keep

        def _sorted_union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            if a.size == 0:
                return b
            if b.size == 0:
                return a
            if a[-1] < b[0]:
                return np.concatenate((a, b))
            if b[-1] < a[0]:
                return np.concatenate((b, a))
            return np.union1d(a, b)

        if use_last_after_inner0:
            if inner_iter == 1:
                self.keep_last = keep
                self._tree_log(f"    [Init] use_last: inner_iter=1, storing keep_last={len(self.keep_last)}")
                return keep
            if self.keep_last is not None and len(self.keep_last) > 0:
                keep = _sorted_union(keep, self.keep_last)
                self._tree_log(f"    [Init] use_last: merged keep={len(keep)}")
            self.keep_last = keep
            return keep

        if self.keep_last is not None and len(self.keep_last) > 0:
            keep = _sorted_union(keep, self.keep_last)
            self._tree_log(f"    [Init] use_last: merged keep={len(keep)}")
        self.keep_last = keep
        return keep

    def _solve_tree_lp_with_fallback(
        self,
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
    ) -> Dict[str, Any]:
        
        num_levels = self.hierarchy_s.num_levels
        is_coarsest = (level_idx == num_levels)
        n_s = len(level_s.points)
        n_t = len(level_t.points)

        try:
            lp_result = self._solve_tree_lp(
                level_s,
                level_t,
                keep_coord,
                x_init,
                y_init,
                stop_thr,
                inner_iter=inner_iter,
                prev_PrimalFeas=prev_PrimalFeas,
                level_idx=level_idx,
                is_coarsest=is_coarsest,
                tree_debug=tree_debug,
            )
        except Exception as exc:
            lp_result = {
                "success": False,
                "x": None,
                "y": None,
                "obj": None,
                "iters": 0,
                "primal_feas": None,
                "dual_feas": None,
                "gap": None,
                "termination_reason": f"LP_BUILD_ERROR: {exc}",
                "diag": {
                    "level": level_idx,
                    "is_coarsest": is_coarsest,
                    "error": str(exc),
                },
                "lp_payload": None,
            }

        if lp_result.get("diag"):
            self.tree_lp_diags.append(lp_result["diag"])

        if lp_result.get("success", False):
            return {"success": True, "lp_result": lp_result, "keep": keep, "keep_coord": keep_coord}

        if is_coarsest and tree_infeas_fallback in {"full_support_retry", "scipy_verify"}:
            print("  [Fallback] retry coarsest LP with explicit full support")
            keep_retry = np.arange(n_s * n_t, dtype=np.int64)
            keep_coord_retry = decode_keep_1d_to_struct(keep_retry, n_t)
            try:
                retry_result = self._solve_tree_lp(
                    level_s,
                    level_t,
                    keep_coord_retry,
                    x_init,
                    None,
                    stop_thr,
                    inner_iter=inner_iter,
                    prev_PrimalFeas=prev_PrimalFeas,
                    level_idx=level_idx,
                    is_coarsest=True,
                    tree_debug=True,
                )
            except Exception as exc:
                retry_result = {
                    "success": False,
                    "termination_reason": f"LP_BUILD_ERROR: {exc}",
                    "diag": {
                        "level": level_idx,
                        "is_coarsest": True,
                        "error": str(exc),
                    },
                    "lp_payload": None,
                }

            if retry_result.get("diag"):
                self.tree_lp_diags.append(retry_result["diag"])

            if retry_result.get("success", False):
                print("  [Fallback] coarsest full-support retry succeeded")
                return {
                    "success": True,
                    "lp_result": retry_result,
                    "keep": keep_retry,
                    "keep_coord": keep_coord_retry,
                }

            if tree_infeas_fallback == "scipy_verify":
                payload = retry_result.get("lp_payload")
                if payload is None:
                    logger.error("scipy_verify skipped: missing LP payload")
                    payload = {}
                scipy_diag = self._verify_lp_with_scipy(
                    c=payload.get("c"),
                    A_eq=payload.get("A_eq"),
                    b_eq=payload.get("b_eq"),
                    lb=payload.get("lb"),
                    ub=payload.get("ub"),
                )
                print(f"  [Fallback:scipy_verify] {scipy_diag}")
                retry_result["scipy_verify"] = scipy_diag
                return {
                    "success": False,
                    "lp_result": retry_result,
                    "keep": keep_retry,
                    "keep_coord": keep_coord_retry,
                }

        return {"success": False, "lp_result": lp_result, "keep": keep, "keep_coord": keep_coord}

    def _should_stop_tree_inner(
        self,
        lp_result: Dict[str, Any],
        state: Dict[str, Any],
        level_s,
        level_t,
        max_inner_iter: int,
        is_coarsest: bool,
    ) -> bool:
        
        if is_coarsest:
            print("  [Coarsest] single LP iteration completed, moving to next level")
            return True

        primal_ori = lp_result.get("primal_infeas_ori")
        if primal_ori is None and lp_result.get("x") is not None:
            primal_ori = self._compute_primal_infeas(lp_result["x"], level_s, level_t)
        if primal_ori is not None and np.isfinite(primal_ori) and float(primal_ori) > 1.0:
            print(
                f"  [Stop] primal infeasibility too large: {float(primal_ori):.2e} > 1.0"
            )
            return True

        converged = self._check_tree_convergence(
            lp_result,
            state["inf_thr"],
            level_idx=state["level_idx"],
            num_levels=self.hierarchy_s.num_levels,
            prev_active_size=state["prev_active_size"],
            curr_active_size=None,
            level_s=level_s,
            level_t=level_t,
        )
        if converged:
            print(f"  Converged after {state['inner_iter'] + 1} iterations")
            return True

        if state["inner_iter"] >= max_inner_iter:
            print(f"  Max inner iterations reached ({max_inner_iter})")
            return True
        return False

    def _init_tree_level_first(
        self,
        level_idx: int,
        level_s,
        level_t,
        x_solution_last,
        y_solution_last,
        cost_type: str,
        inner_iter: int = 0
    ) -> Dict[str, Any]:
        
        check_method = str(getattr(self, "check_type", "auto")).lower()
        if check_method in {"cupy", "gpu_approx", "approx"}:
            check_method = "gpu_approx"
        elif check_method in {"gpu", "gpu_full", "full_gpu"}:
            check_method = "gpu"
        if check_method not in {"cpu", "gpu", "gpu_approx", "auto"}:
            check_method = "auto"
        return self._build_active_set_first_iter(
            level_idx=level_idx,
            level_s=level_s,
            level_t=level_t,
            x_solution_last=x_solution_last,
            y_solution_last=y_solution_last,
            cost_type=cost_type,
            use_last=getattr(self, 'use_last', True),
            ifcheck=getattr(self, 'ifcheck', True),
            vd_thr=getattr(self, 'vd_thr', 0.25),
            check_method=check_method,
            sampled_config=getattr(self, "check_sampled_config", None),
        )

    def _init_tree_level_subsequent(
        self,
        level_idx: int,
        level_s,
        level_t,
        x_solution_last,
        y_solution_last,
        cost_type: str,
        use_last: bool = True,
        inner_iter: int = 1
    ) -> Dict[str, Any]:
        
        check_method = str(getattr(self, "check_type", "auto")).lower()
        if check_method in {"cupy", "gpu_approx", "approx"}:
            check_method = "gpu_approx"
        elif check_method in {"gpu", "gpu_full", "full_gpu"}:
            check_method = "gpu"
        if check_method not in {"cpu", "gpu", "gpu_approx", "auto"}:
            check_method = "auto"
        return self._build_active_set_subsequent_iter(
            level_idx=level_idx,
            level_s=level_s,
            level_t=level_t,
            x_solution_last=x_solution_last,
            y_solution_last=y_solution_last,
            cost_type=cost_type,
            use_last=use_last,
            use_last_after_inner0=getattr(self, 'use_last_after_inner0', False),
            ifcheck=getattr(self, 'ifcheck', True),
            vd_thr=getattr(self, 'vd_thr', 0.25),
            check_method=check_method,
            inner_iter=inner_iter,
            sampled_config=getattr(self, "check_sampled_config", None),
        )

    def _solve_tree_lp(
        self,
        level_s,
        level_t,
        keep_coord,
        x_init,
        y_init,
        stop_thr: float,
        inner_iter: int = 0,
        prev_PrimalFeas: float = None,
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
                raise ValueError(
                    f"Invalid masses: sum_s={sum_s}, sum_t={sum_t}"
                )
            
            mass_s = mass_s / sum_s
            mass_t = mass_t / sum_t
            mass_renormalized = True

        diag: Dict[str, Any] = {
            "level": int(level_idx),
            "is_coarsest": bool(is_coarsest),
            "tree_lp_form": str(getattr(self, "tree_lp_form", "dual")),
            "n_s": int(n_s),
            "n_t": int(n_t),
            "n_vars": int(n_vars),
            "n_eqs": int(n_s + n_t),
            "sum_source_mass": sum_s,
            "sum_target_mass": sum_t,
            "mass_diff": float(sum_s - sum_t),
            "mass_renormalized": mass_renormalized,
        }

        
        
        cost_type = _tree_internal_cost_type(getattr(self, "_cost_type", "l2^2"))
        p = getattr(self, '_cost_p', 2.0)
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
            raise ValueError(
                f"Invalid A_eq shape={A_eq.shape}, expected {(n_s + n_t, n_vars)}"
            )

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
            raise ValueError(
                f"Coverage broken: row_missing={row_missing}, col_missing={col_missing}"
            )

        if is_coarsest:
            expected = n_s * n_t
            if n_vars != expected:
                raise AssertionError(
                    f"Coarsest level must use full support: K={n_vars}, expected={expected}"
                )

            
            
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
            print(
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
                    print(f"  [DEBUG] y_init constraint rel-error: {constraint_error:.2e}")

        
        print(f"  LP solve: n_vars={n_vars}, n_eqs={n_s+n_t}, warm_start_y={y_init is not None}, warm_start_x={x_init is not None}")
        
        y_init_array = y_init['y'] if isinstance(y_init, dict) else y_init
        solve_kwargs = dict(
            c=c,                    
            A_csc=A_eq,             
            b_eq=b_eq,              
            lb=lb,                  
            ub=ub,                  
            n_eqs=n_s + n_t,        
            warm_start_primal=y_init_array,   
            warm_start_dual=x_init,           
            tolerance={
                'primal': dPrimalTol,
                'dual': stop_thr,
                'objective': stop_thr
            },
        )
        requested_lp_form = str(getattr(self, "tree_lp_form", "dual"))
        if getattr(self.solver, "supports_tree_lp_form", False):
            solve_kwargs["lp_form"] = requested_lp_form
            solve_kwargs["dual_form_data"] = {
                "minus_AT": minus_AT,
                "minus_c": minus_c,
                "minus_q": -b_eq,
            }
            solver_params = getattr(self, "_tree_solver_params", None)
            if isinstance(solver_params, dict) and solver_params:
                solve_kwargs["solver_params"] = dict(solver_params)
        elif requested_lp_form == "dual":
            raise NotImplementedError(
                f"{type(self.solver).__name__} does not support tree_lp_form='dual'. "
                "Use CuPDLPxSolver or switch tree_lp_form to 'primal'."
            )
        
        if self.lp_solver_verbose and hasattr(self.solver, 'solve'):
            
            import inspect
            try:
                sig = inspect.signature(self.solver.solve)
                if 'verbose' in sig.parameters:
                    solve_kwargs['verbose'] = self.lp_solver_verbose
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

        
        diag["termination_reason"] = getattr(result, "termination_reason", None)
        lp_payload = None
        if not result.success:
            lp_payload = {
                'c': c,
                'A_eq': A_eq,
                'b_eq': b_eq,
                'lb': lb,
                'ub': ub,
            }
        return {
            'success': result.success,
            'x': result.y if result.success else None,  
            'y': result.x if result.success else None,  
            'obj': result.obj_val if result.success else None,
            'dual_obj': getattr(result, 'dual_obj_val', None),
            'iters': getattr(result, 'iterations', 0),
            'primal_feas': getattr(result, 'primal_feas', None),
            'dual_feas': getattr(result, 'dual_feas', None),
            'gap': getattr(result, 'gap', None),
            'termination_reason': getattr(result, 'termination_reason', None),
            'diag': diag,
            'lp_payload': lp_payload,
        }

    def _compute_primal_infeas(
        self,
        x_dual: np.ndarray,
        level_s,
        level_t
    ) -> float:
        
        try:
            cost_type = _tree_internal_cost_type(getattr(self, "_cost_type", "l2^2"))
            p = getattr(self, '_cost_p', 2.0)

            
            infeas = dualOT_primal_infeas_pointcloud_cupy_auto(
                x_dual,
                level_s.points,
                level_t.points,
                cost_type=cost_type,
                p=p,
                use_cupy=bool(getattr(self, "tree_infeas_use_cupy", False)),
            )
            return infeas
        except Exception as e:
            logger.warning(f"Primal infeasibility computation failed: {e}")
            return float('inf')

    def _check_tree_convergence(
        self,
        lp_result: Dict,
        inf_thr: float,
        level_idx: int = 0,
        num_levels: int = 1,
        prev_active_size: int = None,
        curr_active_size: int = None,
        level_s=None,
        level_t=None
    ) -> bool:
        
        if not lp_result.get('success'):
            return False

        
        primal_flag = False
        dual_flag = False
        gap_flag = False

        
        primal_val = lp_result.get("primal_infeas_ori")
        if primal_val is None and 'x' in lp_result and lp_result['x'] is not None and level_s is not None and level_t is not None:
            primal_val = self._compute_primal_infeas(lp_result['x'], level_s, level_t)
        if primal_val is not None and np.isfinite(primal_val):
            primal_val = float(primal_val)
            primal_flag = abs(primal_val) <= inf_thr
            if not primal_flag:
                self._tree_log(f"    [Convergence] Primal infeasibility: {primal_val:.2e} > {inf_thr:.2e}")

        
        dual_val = lp_result.get('dual_feas')
        if dual_val is not None:
            dual_flag = abs(dual_val) <= inf_thr
            if not dual_flag:
                self._tree_log(f"    [Convergence] Dual feasibility: {dual_val:.2e} > {inf_thr:.2e}")

        
        gap_val = lp_result.get('gap')
        if gap_val is not None:
            gap_flag = abs(gap_val) <= inf_thr
            if not gap_flag:
                self._tree_log(f"    [Convergence] Duality gap: {gap_val:.2e} > {inf_thr:.2e}")

        
        primal_str = f"{primal_val:.2e}" if primal_val is not None else "N/A"
        dual_str = f"{dual_val:.2e}" if dual_val is not None else "N/A"
        gap_str = f"{gap_val:.2e}" if gap_val is not None else "N/A"

        self._tree_log(f"    [Convergence Check] primal={primal_str}, dual={dual_str}, gap={gap_str}, thr={inf_thr:.2e}")
        self._tree_log(f"    [Convergence Flags] primal={primal_flag}, dual={dual_flag}, gap={gap_flag}")

        
        if primal_flag and dual_flag and gap_flag:
            self._tree_log("    [Convergence] primal, dual, and gap checks all passed")
            return True

        return False

    def _record_tree_level_result(
        self,
        level_idx: int,
        x_solution,
        y_solution,
        total_time: float,
        iters: int,
        n_s: int,
        n_t: int,
        support_pre_lp_final: Optional[int] = None,
        lp_time: float = 0.0,
        pricing_time: float = 0.0,
    ):
        
        if not hasattr(self, 'level_summaries'):
            self.level_summaries = []
        
        self.solutions[level_idx] = {
            'dual': x_solution,
            'primal': y_solution,
        }
        
        self.level_summaries.append({
            'level': level_idx,
            'n_source': n_s,
            'n_target': n_t,
            'iters': iters,
            'time': total_time,
            'lp_time': float(lp_time),
            'pricing_time': float(pricing_time),
            'support_pre_lp_final': int(support_pre_lp_final) if support_pre_lp_final is not None else 0,
            'support_final': len(y_solution['y']) if y_solution else 0,
        })

    def _package_tree_result(self) -> Dict[str, Any]:
        
        final_sol = self.solutions[0]
        
        
        final_obj = 0.0
        level_0 = self.hierarchy_s.levels[0]
        level_0_t = self.hierarchy_t.levels[0]
        primal = final_sol.get('primal')
        
        
        cost_type = _tree_internal_cost_type(getattr(self, "_cost_type", "l2^2"))
        
        if primal is not None and 'y' in primal and 'keep' in primal:
            y_vals = primal['y']
            keep = primal['keep']
            n_t = len(level_0_t.points)
            
            
            for idx, y_val in zip(keep, y_vals):
                flow = abs(float(y_val))
                if flow <= 0.0:
                    continue
                i = idx // n_t
                j = idx % n_t
                diff = level_0.points[i] - level_0_t.points[j]

                if cost_type in ('L2', 'SQEUCLIDEAN'):
                    cost = np.sum(diff**2)
                elif cost_type == 'L1':
                    cost = np.sum(np.abs(diff))
                elif cost_type == 'LINF':
                    cost = np.max(np.abs(diff))
                else:
                    cost = np.sum(diff**2)  

                final_obj += flow * cost
        
        return {
            'primal': final_sol.get('primal'),
            'dual': final_sol.get('dual'),
            'final_obj': final_obj,
            'all_history': self.solutions,
            'sparse_coupling': None,
            'level_summaries': self.level_summaries,
            'lp_solve_time_total': float(
                sum(float(item.get('lp_time', 0.0)) for item in self.level_summaries if isinstance(item, dict))
            ),
            'tree_lp_diags': getattr(self, 'tree_lp_diags', []),
        }

class TreeSolverBackend(SolverBackend):
    name = "tree"

    def init_run_state(self, tolerance: Dict[str, float], **kwargs: Any) -> CommonRunState:
        solver = self.solver
        tree_debug = bool(kwargs.get("tree_debug", getattr(solver, "tree_debug", False)))
        tree_infeas_fallback = solver._normalize_tree_infeas_fallback(
            kwargs.get("tree_infeas_fallback", getattr(solver, "tree_infeas_fallback", "none"))
        )
        solver.tree_infeas_use_cupy = solver._parse_bool_flag(
            kwargs.get("tree_infeas_use_cupy", getattr(solver, "tree_infeas_use_cupy", True)),
            default=True,
        )
        solver.tree_lp_diags = []

        use_last = solver._parse_bool_flag(
            kwargs.get("use_last", getattr(solver, "use_last", True)),
            default=True,
        )
        use_last_after_inner0 = solver._parse_bool_flag(
            kwargs.get("use_last_after_inner0", getattr(solver, "use_last_after_inner0", False)),
            default=False,
        )
        ifcheck = solver._parse_bool_flag(
            kwargs.get("ifcheck", getattr(solver, "ifcheck", True)),
            default=True,
        )
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
            logger.warning("Unknown tree_lp_form=%r, fallback to 'dual'.", tree_lp_form)
            tree_lp_form = "dual"
        solver.tree_lp_form = tree_lp_form

        solver.keep_last = None
        solver.solutions = {}
        solver.level_summaries = []
        solver.nnz_thr = float(kwargs.get("nnz_thr", getattr(solver, "nnz_thr", 1e-20)))
        solver.check_sampled_config = kwargs.get(
            "sampled_config", getattr(solver, "check_sampled_config", None)
        )
        if hasattr(solver.strategy, "shield_impl") and "shield_impl" in kwargs:
            solver.strategy.shield_impl = str(kwargs.get("shield_impl")).lower()
        if hasattr(solver.strategy, "max_pairs_per_xA") and "max_pairs_per_xA" in kwargs:
            solver.strategy.max_pairs_per_xA = int(kwargs.get("max_pairs_per_xA"))
        if hasattr(solver.strategy, "nnz_thr") and "nnz_thr" in kwargs:
            solver.strategy.nnz_thr = float(kwargs.get("nnz_thr"))
        return {
            "num_levels": solver.hierarchy_s.num_levels,
            "tree_debug": tree_debug,
            "tree_infeas_fallback": tree_infeas_fallback,
            "use_last": use_last,
            "use_last_after_inner0": use_last_after_inner0,
            "ifcheck": ifcheck,
            "vd_thr": vd_thr,
            "check_method": check_method,
            "x_solution_last": None,
            "y_solution_last": None,
            "tolerance": tolerance,
        }

    def get_level_indices(self, run_state: CommonRunState):
        return reversed(range(run_state["num_levels"] + 1))

    def init_level_state(
        self,
        level_idx: int,
        run_state: CommonRunState,
        tolerance: Dict[str, float],
        cost_type: str,
        use_bfs_skeleton: bool,
    ) -> Optional[CommonLevelState]:
        del use_bfs_skeleton
        solver = self.solver
        level_s = solver.hierarchy_s.levels[level_idx]
        level_t = solver.hierarchy_t.levels[level_idx]
        n_s = len(level_s.points)
        n_t = len(level_t.points)
        solver._tree_log(f"\n{'='*50}")
        solver._tree_log(f"=== Solving Level {level_idx} (Tree Mode) ===")
        solver._tree_log(f"{'='*50}")
        if n_s == 0 or n_t == 0:
            return None

        num_levels = run_state["num_levels"]
        inf_thr, stop_thr = solver._prepare_level_tolerances(tolerance, level_idx)
        state = solver._init_level_state(
            level_idx=level_idx,
            inf_thr=inf_thr,
            stop_thr=stop_thr,
            x_solution_last=run_state["x_solution_last"],
            y_solution_last=run_state["y_solution_last"],
        )
        state["level_s"] = level_s
        state["level_t"] = level_t
        state["t_level_start"] = time.perf_counter()
        state["is_coarsest"] = level_idx == num_levels
        state["cost_type"] = cost_type

        prep0 = solver._tree_prepare_inner(
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
        )
        state["keep"] = prep0["keep"]
        state["keep_coord"] = prep0["keep_coord"]
        state["x_init"] = prep0["x_init"]
        state["y_init"] = prep0["y_init"]
        state["curr_active_size"] = prep0["curr_active_size"]
        state["_trace_keep_after_shield"] = prep0.get("trace_keep_after_shield")
        state["_trace_keep_after_check"] = prep0.get("trace_keep_after_check")
        state["_trace_keep_after_uselast"] = prep0.get("trace_keep_after_uselast")
        prepare_time = float(prep0.get("prepare_time_total", 0.0))
        prepare_breakdown = prep0.get("prepare_breakdown", {})
        if isinstance(prepare_breakdown, dict):
            prepare_cpu_components: Dict[str, float] = {}
            for key, name in (
                ("refine_duals", COMPONENT_REFINE_DUALS),
                ("shielding", COMPONENT_SHIELDING),
                (COMPONENT_SHIELD_PICK_T_MAP, COMPONENT_SHIELD_PICK_T_MAP),
                (COMPONENT_SHIELD_SENTINELS, COMPONENT_SHIELD_SENTINELS),
                (COMPONENT_SHIELD_YHAT, COMPONENT_SHIELD_YHAT),
                (COMPONENT_SHIELD_UNION, COMPONENT_SHIELD_UNION),
                ("violation_check", COMPONENT_VIOLATION_CHECK),
                ("keep_union", COMPONENT_KEEP_UNION),
                ("keep_coord", COMPONENT_KEEP_COORD),
                ("remap_y", COMPONENT_REMAP_Y),
                ("coverage_repair", COMPONENT_COVERAGE_REPAIR),
            ):
                dt = float(prepare_breakdown.get(key, 0.0))
                if dt > 0.0:
                    prepare_cpu_components[name] = dt
            solver._profiler.add_components(prepare_cpu_components)
        state["level_lp_time"] = 0.0
        state["level_pricing_time"] = prepare_time
        return state

    def prepare_iteration_input(
        self,
        level_state: CommonLevelState,
        run_state: CommonRunState,
        inner_iter: int,
    ) -> None:
        del run_state
        if inner_iter == 0:
            self.solver._tree_log(f"  DEBUG: start while loop, level_idx={level_state['level_idx']}")
        self.solver._tree_log(f"  DEBUG: inner iteration {inner_iter}")

    def solve_iteration_lp(
        self,
        level_state: CommonLevelState,
        run_state: CommonRunState,
    ) -> Dict[str, Any]:
        solver = self.solver
        return solver._tree_solve_lp_pack(
            level_idx=level_state["level_idx"],
            level_s=level_state["level_s"],
            level_t=level_state["level_t"],
            keep=level_state["keep"],
            keep_coord=level_state["keep_coord"],
            x_init=level_state["x_init"],
            y_init=level_state["y_init"],
            stop_thr=level_state["stop_thr"],
            inner_iter=level_state["inner_iter"],
            prev_PrimalFeas=level_state.get("prev_PrimalFeas"),
            tree_debug=run_state["tree_debug"],
            tree_infeas_fallback=run_state["tree_infeas_fallback"],
        )

    def finalize_iteration(
        self,
        level_state: CommonLevelState,
        run_state: CommonRunState,
        step_pack: Dict[str, Any],
        convergence_criterion: str,
        tolerance: Dict[str, float],
    ) -> None:
        del convergence_criterion, tolerance
        solver = self.solver
        lp_result = step_pack["lp_result"]
        level_state["x_solution_last"] = lp_result["x"]
        level_state["y_solution_last"] = {
            "y": lp_result["y"],
            "keep": step_pack["keep"],
        }
        if lp_result.get("primal_feas") is not None:
            level_state["prev_PrimalFeas"] = lp_result["primal_feas"]

        update_pack = self._update_active_set(level_state, run_state, step_pack)
        level_state["level_lp_time"] = level_state.get("level_lp_time", 0.0) + float(
            step_pack.get("lp_time", 0.0)
        )
        level_state["level_pricing_time"] = level_state.get(
            "level_pricing_time", 0.0
        ) + float(update_pack.get("pricing_time", 0.0))
        solver._profiler.add_components(update_pack.get("components"))
        pre_lp_active = level_state.get("_pre_lp_active")
        if pre_lp_active is not None:
            level_state["prev_active_size"] = pre_lp_active
            level_state["last_active_pre_lp"] = pre_lp_active

    def _update_active_set(
        self,
        level_state: CommonLevelState,
        run_state: CommonRunState,
        step_pack: Dict[str, Any],
    ) -> Dict[str, Any]:
        solver = self.solver
        level_idx = level_state["level_idx"]
        if level_idx == run_state["num_levels"]:
            return {"pricing_time": 0.0, "components": {}}

        level_s = level_state["level_s"]
        level_t = level_state["level_t"]
        n_t = len(level_t.points)
        pricing_start = time.perf_counter()
        component_times: Dict[str, float] = {}
        y_last = step_pack["lp_result"].get("y")
        keep_last = level_state["keep"]
        if y_last is None:
            y_keep = np.array([], dtype=np.int64)
            y_vals = np.array([], dtype=np.float32)
        else:
            nonzero_mask = np.abs(y_last) > 1e-20
            y_keep = keep_last[nonzero_mask]
            y_vals = y_last[nonzero_mask]
            if level_state["inner_iter"] > 0:
                solver._tree_log(f"    [Init] inner_iter>0: y_keep after nonzero_mask={len(y_keep)}")

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
            for src_key, component_name in (
                (COMPONENT_SHIELD_PICK_T_MAP, COMPONENT_SHIELD_PICK_T_MAP),
                (COMPONENT_SHIELD_SENTINELS, COMPONENT_SHIELD_SENTINELS),
                (COMPONENT_SHIELD_YHAT, COMPONENT_SHIELD_YHAT),
                (COMPONENT_SHIELD_UNION, COMPONENT_SHIELD_UNION),
            ):
                dt = float(timing.get(src_key, 0.0) or 0.0)
                if dt > 0.0:
                    component_times[component_name] = component_times.get(component_name, 0.0) + dt
        if keep_union_time > 0.0:
            component_times[COMPONENT_KEEP_UNION] = component_times.get(COMPONENT_KEEP_UNION, 0.0) + keep_union_time
        component_times[COMPONENT_SHIELDING] = max(0.0, shield_total - keep_union_time)
        keep_next = np.asarray(update_result["keep"], dtype=np.int64)
        level_state["_trace_keep_after_shield"] = int(len(keep_next))
        t_vcheck = time.perf_counter()
        keep_next = solver._apply_violation_check(
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
        keep_next = solver._merge_with_use_last(
            keep=keep_next,
            use_last=run_state["use_last"],
            use_last_after_inner0=run_state["use_last_after_inner0"],
            inner_iter=level_state["inner_iter"] + 1,
            n_t=n_t,
        )
        level_state["_trace_keep_after_uselast"] = int(len(keep_next))
        component_times[COMPONENT_KEEP_UNION] = component_times.get(COMPONENT_KEEP_UNION, 0.0) + (time.perf_counter() - t_keep_union)
        keep_next, coverage_fix = solver._repair_keep_coverage(
            keep_next,
            n_s=len(level_s.points),
            n_t=n_t,
            points_s=level_s.points,
            points_t=level_t.points,
        )
        if coverage_fix["added"] > 0:
            print(
                "  [CoverageFix] rows_missing="
                f"{coverage_fix['rows_missing']}, cols_missing={coverage_fix['cols_missing']}, "
                f"added={coverage_fix['added']}"
            )
        coverage_dt = float(coverage_fix.get("time", 0.0)) if isinstance(coverage_fix, dict) else 0.0
        if coverage_dt > 0.0:
            component_times[COMPONENT_COVERAGE_REPAIR] = coverage_dt

        t_keep_coord = time.perf_counter()
        keep_coord_next = decode_keep_1d_to_struct(keep_next, n_t)
        component_times[COMPONENT_KEEP_COORD] = time.perf_counter() - t_keep_coord
        y_init_next = None
        if y_last is not None:
            t_remap = time.perf_counter()
            y_init_next = {
                "y": remap_duals_for_warm_start({"y": y_vals, "keep": y_keep}, keep_next)
            }
            component_times[COMPONENT_REMAP_Y] = time.perf_counter() - t_remap

        level_state["keep"] = keep_next
        level_state["keep_coord"] = keep_coord_next
        level_state["x_init"] = level_state["x_solution_last"]
        level_state["y_init"] = y_init_next
        level_state["curr_active_size"] = len(keep_next)
        pricing_time = time.perf_counter() - pricing_start
        component_times[COMPONENT_PRICING_TOTAL] = pricing_time
        return {"pricing_time": pricing_time, "components": component_times}

    def should_stop_iteration(
        self,
        level_state: CommonLevelState,
        run_state: CommonRunState,
        max_inner_iter: int,
        step_pack: Dict[str, Any],
    ) -> bool:
        t0 = time.perf_counter()
        result = self.solver._should_stop_tree_inner(
            lp_result=step_pack["lp_result"],
            state=level_state,
            level_s=level_state["level_s"],
            level_t=level_state["level_t"],
            max_inner_iter=max_inner_iter,
            is_coarsest=(level_state["level_idx"] == run_state["num_levels"]),
        )
        self.solver._profiler.add_components({COMPONENT_CONVERGENCE_CHECK: time.perf_counter() - t0})
        return result

    def record_level_result(
        self,
        level_state: CommonLevelState,
        final_refinement_tolerance: Optional[Dict[str, float]],
    ) -> None:
        del final_refinement_tolerance
        t_level = time.perf_counter() - level_state["t_level_start"]
        level_s = level_state["level_s"]
        level_t = level_state["level_t"]
        self.solver._record_tree_level_result(
            level_state["level_idx"],
            level_state["x_solution_last"],
            level_state["y_solution_last"],
            t_level,
            int(level_state.get("iters_done", level_state["inner_iter"])),
            len(level_s.points),
            len(level_t.points),
            support_pre_lp_final=level_state.get("last_active_pre_lp"),
            lp_time=float(level_state.get("level_lp_time", 0.0)),
            pricing_time=float(level_state.get("level_pricing_time", 0.0)),
        )

    def advance_to_next_level(
        self,
        level_state: CommonLevelState,
        run_state: CommonRunState,
    ) -> None:
        run_state["x_solution_last"] = level_state["x_solution_last"]
        run_state["y_solution_last"] = level_state["y_solution_last"]

    def package_result(self) -> Dict[str, Any]:
        return self.solver._package_tree_result()

    def extract_step_objective(self, step_pack: Dict[str, Any]) -> Optional[float]:
        objective = step_pack.get("lp_result", {}).get("obj")
        if objective is None:
            return None
        return float(objective)
