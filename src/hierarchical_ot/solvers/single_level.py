from __future__ import annotations

import logging
import math
import os
import json
from contextlib import nullcontext
from typing import Any, Dict, Literal, Optional, Sequence, Set, Tuple

import numpy as np
import scipy.sparse as sp

from ..types.base import BaseHierarchy, BaseStrategy, HierarchyLevel
from ..types.config import ConfigType, GridConfig, TreeConfig
from ..instrumentation.runtime_profiler import NoOpProfiler, RuntimeProfiler
from ..core.hierarchical_solver import HierarchicalOTSolver
from ..modes.grid.hierarchy import GridHierarchy
from ..modes.tree.hierarchy import (
    TreeHierarchy,
    align_hierarchy_depths,
    print_hierarchy_info,
    suggest_num_levels,
)
from ..lp_solvers.cupdlpx import CuPDLPxSolver
from ..lp_solvers.hprlp import HPRLPSolver
from ..lp_solvers.scipy import SciPySolver
from ..modes.tree.shielding import ShieldingStrategy
from ..modes.grid.costs import grid_pairwise_cost
from ..modes.grid import shielding as grid_shielding

logger = logging.getLogger(__name__)


class _GridNoOpStrategy:
    def close(self) -> None:
        return None


class _SingleLevelHierarchy(BaseHierarchy):
    """最小 single-level hierarchy 容器，用于 warm-start 路径绕开聚类构建。"""

    def __init__(self, level: HierarchyLevel):
        super().__init__([level])
        self.levels = [level]
        self.num_levels = 1
        self.build_time = 0.0

    def prolongate(
        self,
        coarse_potential: np.ndarray,
        coarse_level_idx: int,
        fine_level_idx: int,
    ) -> np.ndarray:
        if int(coarse_level_idx) != 0 or int(fine_level_idx) != 0:
            raise ValueError("Single-level hierarchy only supports level 0 prolongation.")
        return coarse_potential


def _normalize_cost_type_name(cost_type: str) -> str:
    c = str(cost_type).strip().lower()
    if c in ("lowrank",):
        return "lowrank"
    if c in ("l1",):
        return "l1"
    if c in ("linf", "l_inf"):
        return "linf"
    if c in ("l2", "euclidean"):
        return "l2"
    if c in ("l2^2", "l2sq", "sqeuclidean", "sq_euclidean"):
        return "l2^2"
    if c in ("lp",):
        return "lp"
    raise ValueError(
        f"Unknown cost_type: {cost_type}. Supported names: lowrank, l1, linf, l2, l2^2, lp."
    )


def _sparse_coupling_from_data(
    c_data: Optional[Dict[str, np.ndarray]],
    shape: Tuple[int, int],
) -> sp.coo_matrix:
    if not c_data:
        return sp.coo_matrix(shape, dtype=np.float32)
    return sp.coo_matrix(
        (c_data["values"], (c_data["rows"], c_data["cols"])),
        shape=shape,
        dtype=np.float32,
    )


def _sparse_coupling_from_active_support(
    solver: HierarchicalOTSolver,
    shape: Tuple[int, int],
) -> sp.coo_matrix:
    active = solver.active_support
    if active is None:
        return sp.coo_matrix(shape, dtype=np.float32)
    nonzero_mask = np.asarray(active.x_prev, dtype=np.float32) > 1e-12
    if not np.any(nonzero_mask):
        return sp.coo_matrix(shape, dtype=np.float32)
    return sp.coo_matrix(
        (
            np.asarray(active.x_prev[nonzero_mask], dtype=np.float32),
            (
                np.asarray(active.rows[nonzero_mask], dtype=np.int32),
                np.asarray(active.cols[nonzero_mask], dtype=np.int32),
            ),
        ),
        shape=shape,
        dtype=np.float32,
    )


def _state_from_solver_active_support(
    solver: HierarchicalOTSolver,
    *,
    n_s: int,
    n_t: int,
    dual_uv: Optional[np.ndarray],
    ot_return: Optional[Dict[str, Any]] = None,
    cost_type: str = "l2^2",
    solver_mode: str = "tree",
) -> Dict[str, Any]:
    coupling = _sparse_coupling_from_active_support(solver, (n_s, n_t))
    if coupling.nnz == 0 and ot_return is not None:
        coupling = _sparse_coupling_from_data(ot_return.get("sparse_coupling"), (n_s, n_t))
    coupling = coupling.tocoo(copy=False)
    if dual_uv is None and ot_return is not None:
        dual_uv = ot_return.get("dual")
    dual_arr = None
    if dual_uv is not None:
        dual_arr = np.asarray(dual_uv, dtype=np.float32).copy()
    return {
        "rows": np.asarray(coupling.row, dtype=np.int32).copy(),
        "cols": np.asarray(coupling.col, dtype=np.int32).copy(),
        "x_prev": np.asarray(coupling.data, dtype=np.float32).copy(),
        "dual_uv": dual_arr,
        "n_source": int(n_s),
        "n_target": int(n_t),
        "cost_type": str(cost_type),
        "solver_mode": str(solver_mode),
    }
def _finalize_emd_output(
    *,
    distance: float,
    coupling: sp.coo_matrix,
    state: Optional[Dict[str, Any]],
    dual_source: Optional[np.ndarray],
    level_summaries: Sequence[Dict[str, Any]],
    profiling: Optional[Dict[str, Any]],
    lp_solve_time_total: float,
    elapsed: float,
    extra_log_fields: Optional[Dict[str, Any]] = None,
    chrome_trace: Optional[Dict[str, Any]] = None,
    log: bool,
    return_coupling: bool,
    return_state: bool,
) -> Any:
    if log:
        log_dict = {
            "distance": float(distance),
            "time": float(elapsed),
            "lp_solve_time_total": float(lp_solve_time_total),
            "level_summaries": level_summaries,
            "dual_source": dual_source,
            "sparse_coupling": coupling if return_coupling else None,
            "warm_start_state": state,
        }
        if extra_log_fields:
            log_dict.update(extra_log_fields)
        if profiling is not None:
            log_dict["profiling"] = profiling
        if chrome_trace is not None:
            log_dict["chrome_trace"] = chrome_trace
        return log_dict

    if return_coupling and return_state:
        return float(distance), coupling, state
    if return_coupling:
        return float(distance), coupling
    if return_state:
        return float(distance), state
    return float(distance)


def _build_trace_collector_from_config(
    config: ConfigType,
    *,
    trace_prefix: str = "solve_ot",
) -> Optional[_ChromeTraceCollector]:
    profiling_cfg = config.normalized_profiling()
    if not bool(profiling_cfg.get("write_trace_json", False)):
        return None
    del trace_prefix
    return _ChromeTraceCollector(enabled=True)


def _write_trace_json_if_requested(
    config: ConfigType,
    *,
    chrome_trace: Optional[Dict[str, Any]],
) -> None:
    if chrome_trace is None:
        return
    profiling_cfg = config.normalized_profiling()
    if not bool(profiling_cfg.get("write_trace_json", False)):
        return
    raw_path = profiling_cfg.get("trace_json_path")
    trace_json_path = os.path.abspath(str(raw_path)) if raw_path is not None else os.path.abspath("chrome_trace.json")
    os.makedirs(os.path.dirname(trace_json_path), exist_ok=True)
    with open(trace_json_path, "w", encoding="utf-8") as fh:
        json.dump(chrome_trace, fh)


def _materialize_grid_ot_return(
    solver: HierarchicalOTSolver,
    ot_return: Dict[str, Any],
    *,
    need_final_obj: bool,
    need_sparse_coupling: bool,
) -> Dict[str, Any]:
    if not ot_return:
        return ot_return
    return materialize_grid_result_payload(
        solver,
        ot_return,
        need_final_obj=bool(need_final_obj),
        need_sparse_coupling=bool(need_sparse_coupling),
    )


def grid_cost_from_keep(solver: HierarchicalOTSolver, keep: np.ndarray, level_s: Any, level_t: Any) -> np.ndarray:
    rows, cols = grid_shielding.grid_keep_to_rows_cols(keep, len(level_t.points))
    s = np.asarray(level_s.points, dtype=np.float32)
    t = np.asarray(level_t.points, dtype=np.float32)
    cost_type = str(getattr(solver, "_cost_type", "l2^2")).lower()
    p = int(getattr(solver, "_grid_p", 2))
    if cost_type == "l2^2":
        res = float(max(int(np.max(s)) + 1, int(np.max(t)) + 1))
        diff = s[np.asarray(rows, dtype=np.int64)] - t[np.asarray(cols, dtype=np.int64)]
        return (np.sum(diff * diff, axis=1) / (res * res)).astype(np.float32, copy=False)
    return grid_pairwise_cost(s, t, cost_type=cost_type, p=p)[rows, cols].astype(np.float32, copy=False)


def materialize_grid_result_payload(
    solver: HierarchicalOTSolver,
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
                costs = grid_cost_from_keep(
                    solver,
                    keep=np.asarray(keep[mask], dtype=np.int64),
                    level_s=solver.hierarchy_s.levels[0],
                    level_t=solver.hierarchy_t.levels[0],
                )
                final_obj = float(np.dot(flows[mask].astype(np.float64), costs.astype(np.float64)))
        result["final_obj"] = float(final_obj)
    if need_sparse_coupling and "sparse_coupling" not in result:
        sparse_coupling = None
        if keep.size > 0 and vals.size > 0:
            n_t = len(solver.hierarchy_t.levels[0].points)
            rows, cols = grid_shielding.grid_keep_to_rows_cols(keep, n_t)
            flows = np.abs(vals).astype(np.float32, copy=False)
            mask = flows > 1e-12
            sparse_coupling = {
                "rows": np.asarray(rows[mask], dtype=np.int32),
                "cols": np.asarray(cols[mask], dtype=np.int32),
                "values": np.asarray(flows[mask], dtype=np.float32),
            }
        result["sparse_coupling"] = sparse_coupling
    return result


def _sum_level_summary_metric(
    level_summaries: Sequence[Dict[str, Any]],
    key: str,
) -> float:
    return float(sum(float(item.get(key, 0.0)) for item in level_summaries if isinstance(item, dict)))


def _extract_level0_summary(ot_result: Dict[str, Any]) -> Dict[str, Any]:
    level0 = None
    for item in ot_result.get("level_summaries", []):
        if int(item.get("level", -1)) == 0:
            level0 = item
            break
    if level0 is None:
        return {
            "level0_inner_iterations": -1,
            "level0_lp_time_total": float("nan"),
            "level0_pricing_time_total": float("nan"),
            "level0_total_time": float("nan"),
            "final_active_support_size": 0,
        }
    return {
        "level0_inner_iterations": int(level0.get("iters", -1)),
        "level0_lp_time_total": float(level0.get("lp_time", float("nan"))),
        "level0_pricing_time_total": float(level0.get("pricing_time", float("nan"))),
        "level0_total_time": float(level0.get("time", float("nan"))),
        "final_active_support_size": int(level0.get("support_final", 0)),
    }


def _build_hierarchies(
    source_X: Optional[np.ndarray],
    target_X: Optional[np.ndarray],
    source_mass: Optional[np.ndarray],
    target_mass: Optional[np.ndarray],
    dim: int,
    config: ConfigType,
    cost_type: str,
) -> Dict[str, Any]:
    if isinstance(config, GridConfig):
        del cost_type
        if source_mass is None or target_mass is None:
            raise ValueError("grid mode requires 2D square source_mass and target_mass.")
        src_hist = np.asarray(source_mass, dtype=np.float32, order="C")
        tgt_hist = np.asarray(target_mass, dtype=np.float32, order="C")
        if src_hist.ndim != 2 or src_hist.shape[0] != src_hist.shape[1]:
            raise ValueError("grid mode requires source_mass to be a square 2D array.")
        if tgt_hist.ndim != 2 or tgt_hist.shape[0] != tgt_hist.shape[1]:
            raise ValueError("grid mode requires target_mass to be a square 2D array.")
        if src_hist.shape != tgt_hist.shape:
            raise ValueError(
                "grid mode currently requires source_mass and target_mass to have the same shape."
            )
        src_resolution = int(src_hist.shape[0])
        source_hist = src_hist.reshape(src_resolution, src_resolution)
        target_hist = tgt_hist.reshape(src_resolution, src_resolution)
        hierarchy_s = GridHierarchy(num_scales=config.num_scales)
        hierarchy_s.build(source_hist)
        hierarchy_t = GridHierarchy(num_scales=config.num_scales)
        hierarchy_t.build(target_hist)
        return {
            "hierarchy_s": hierarchy_s,
            "hierarchy_t": hierarchy_t,
        }
    if isinstance(config, TreeConfig):
        if source_X is None or target_X is None:
            raise ValueError("tree mode requires source_X and target_X point clouds.")
        if source_mass is None or target_mass is None:
            raise ValueError("tree mode requires source_mass and target_mass.")
        k_neighbors_tree = 4 * dim
        n_bar = int(np.sqrt(len(source_mass) * len(target_mass)))
        depth_min = suggest_num_levels(n_bar, dim, "2^n")
        depth_max = 100

        hierarchy_s = TreeHierarchy(
            split_mode="2^n",
            build_mode="bucket",
            target_coarse_size=int(config.target_coarse_size),
            max_L0_L1_ratio=float(config.max_L0_L1_ratio),
            k_neighbors=k_neighbors_tree,
            depth_min=depth_min,
            depth_max=depth_max,
            verbose=True,
        )
        hierarchy_s.build(source_X, source_mass)

        hierarchy_t = TreeHierarchy(
            split_mode="2^n",
            build_mode="bucket",
            target_coarse_size=int(config.target_coarse_size),
            max_L0_L1_ratio=float(config.max_L0_L1_ratio),
            k_neighbors=k_neighbors_tree,
            depth_min=depth_min,
            depth_max=depth_max,
            verbose=True,
        )
        hierarchy_t.build(target_X, target_mass)
        align_hierarchy_depths(hierarchy_s, hierarchy_t, verbose=True)
        print_hierarchy_info(hierarchy_s)
        print_hierarchy_info(hierarchy_t)

        return {
            "hierarchy_s": hierarchy_s,
            "hierarchy_t": hierarchy_t,
            "k_neighbors_tree": k_neighbors_tree,
        }

    raise TypeError(f"Unsupported public config type for hierarchy building: {type(config).__name__}")


def _configure_strategy(
    cost_type: str,
    config: ConfigType,
    k_neighbors_tree: Optional[int] = None,
    printing_enabled: bool = True,
) -> Tuple[Any, Any]:
    del printing_enabled
    if isinstance(config, TreeConfig):
        if k_neighbors_tree is None:
            raise ValueError("k_neighbors_tree must be provided for tree mode")
        strategy = ShieldingStrategy(
            k_neighbors=k_neighbors_tree,
            max_pairs_per_xA=int(config.max_pairs_per_xA),
            cost_type="L2" if cost_type == "l2^2" else cost_type.upper(),
            search_method=str(config.search_method).lower(),
            shield_impl=str(config.shield_impl).lower(),
            nnz_thr=float(config.nnz_thr),
        )
        return strategy, None
    if isinstance(config, GridConfig):
        return _GridNoOpStrategy(), None
    raise TypeError(f"Unsupported public config type for strategy configuration: {type(config).__name__}")
def _build_lp_solver(config: ConfigType) -> Any:
    solver_engine = str(config.solver_engine)
    if solver_engine == "cupdlpx":
        return CuPDLPxSolver()
    if solver_engine == "scipy":
        return SciPySolver()
    if solver_engine == "hprlp":
        return HPRLPSolver()
    raise ValueError(f"Unknown solver_engine: {solver_engine}")


def _build_common_solve_kwargs(
    config: ConfigType,
    cost_name: str,
) -> Dict[str, Any]:
    if isinstance(config, TreeConfig):
        mode = "tree"
    elif isinstance(config, GridConfig):
        mode = "grid"
    else:
        raise TypeError("HALO_public config must be an instance of TreeConfig or GridConfig")
    return {
        "max_inner_iter": int(config.max_inner_iter),
        "convergence_criterion": str(config.convergence_criterion),
        "objective_plateau_iters": int(config.objective_plateau_iters),
        "tolerance": config.normalized_tolerance(),
        "final_refinement_tolerance": config.normalized_final_refinement_tolerance(),
        "cost_type": cost_name,
        "mode": mode,
        "enable_profiling": bool(config.normalized_profiling().get("enabled", config.enable_profiling)),
        "printing": config.normalized_printing(),
        "profiling": config.normalized_profiling(),
    }
def _build_tree_solve_kwargs(config: TreeConfig) -> Dict[str, Any]:
    return {
        "tree_debug": bool(config.tree_debug),
        "tree_infeas_fallback": str(config.tree_infeas_fallback),
        "tree_lp_form": str(config.tree_lp_form),
        "check_type": str(config.check_type),
        "use_last": bool(config.use_last),
        "use_last_after_inner0": bool(config.use_last_after_inner0),
        "ifcheck": bool(config.ifcheck),
        "vd_thr": float(config.vd_thr),
        "sampled_config": config.sampled_config,
        "tree_infeas_use_cupy": bool(config.tree_infeas_use_cupy),
        "shield_impl": str(config.shield_impl),
        "max_pairs_per_xA": int(config.max_pairs_per_xA),
        "nnz_thr": float(config.nnz_thr),
    }


def _build_grid_solve_kwargs(config: GridConfig) -> Dict[str, Any]:
    return {
        "grid_p": int(config.p),
        "stop_tolerance": config.normalized_stop_tolerance(),
        "use_last": bool(config.use_last),
        "use_last_after_inner0": bool(config.use_last_after_inner0),
        "if_check": bool(config.if_check),
        "if_shield": bool(config.if_shield),
        "coarsest_full_support": bool(config.coarsest_full_support),
        "vd_thr": float(config.vd_thr),
        "grid_check_type": str(config.check_type),
        "use_primal_feas_ori": bool(config.use_primal_feas_ori),
        "adap_primal_tol": bool(config.adap_primal_tol),
        "new_mvp": bool(config.new_mvp),
        "aty_type": int(config.aty_type),
    }


def _build_solve_kwargs(
    config: ConfigType,
    cost_name: str,
) -> Dict[str, Any]:
    solve_kwargs = _build_common_solve_kwargs(config, cost_name)
    if isinstance(config, TreeConfig):
        solve_kwargs.update(_build_tree_solve_kwargs(config))
        return solve_kwargs
    if isinstance(config, GridConfig):
        solve_kwargs.update(_build_grid_solve_kwargs(config))
        return solve_kwargs
    raise TypeError("HALO_public config must be an instance of TreeConfig or GridConfig")
def _run_emd2_with_config(
    source_X: np.ndarray,
    target_X: np.ndarray,
    source_mass: Optional[np.ndarray],
    target_mass: Optional[np.ndarray],
    log: bool,
    return_coupling: bool,
    return_state: bool,
    config: ConfigType,
    mode: Literal["tree", "grid"],
) -> Any:
    if mode == "grid":
        if source_mass is None or target_mass is None:
            raise ValueError("grid mode requires 2D square source_mass and target_mass.")
        source_mass = np.asarray(source_mass, dtype=np.float32, order="C")
        target_mass = np.asarray(target_mass, dtype=np.float32, order="C")
        if source_mass.ndim != 2 or source_mass.shape[0] != source_mass.shape[1]:
            raise ValueError("grid mode requires source_mass to be a square 2D array.")
        if target_mass.ndim != 2 or target_mass.shape[0] != target_mass.shape[1]:
            raise ValueError("grid mode requires target_mass to be a square 2D array.")
        if source_mass.shape != target_mass.shape:
            raise ValueError("grid mode currently requires source_mass and target_mass to have the same shape.")
        source_mass = source_mass / max(float(source_mass.sum()), 1e-12)
        target_mass = target_mass / max(float(target_mass.sum()), 1e-12)
        n_s = int(source_mass.size)
        n_t = int(target_mass.size)
        dim = 2
    else:
        n_s, dim = source_X.shape
        n_t = target_X.shape[0]
        if source_mass is None:
            source_mass = np.full(n_s, 1.0 / n_s, dtype=np.float32)
        if target_mass is None:
            target_mass = np.full(n_t, 1.0 / n_t, dtype=np.float32)

    cost_name = _normalize_cost_type_name(config.cost_type)

    build_pack = _build_hierarchies(
        source_X=source_X,
        target_X=target_X,
        source_mass=source_mass,
        target_mass=target_mass,
        dim=dim,
        config=config,
        cost_type=cost_name,
    )
    hierarchy_s = build_pack["hierarchy_s"]
    hierarchy_t = build_pack["hierarchy_t"]

    strategy, cleaning_strategy = _configure_strategy(
        cost_type=cost_name,
        config=config,
        k_neighbors_tree=build_pack.get("k_neighbors_tree"),
        printing_enabled=bool(config.normalized_printing().get("enabled", True)),
    )

    lp_solver = _build_lp_solver(config)

    hierarchical_solver = HierarchicalOTSolver(
        hierarchy_s=hierarchy_s,
        hierarchy_t=hierarchy_t,
        strategy=strategy,
        solver=lp_solver,
        cleaning_strategy=cleaning_strategy,
        lp_solver_verbose=bool(config.lp_solver_verbose),
    )

    solve_kwargs = _build_solve_kwargs(config, cost_name)
    ot_return = hierarchical_solver.solve(**solve_kwargs)

    if not ot_return:
        return float("nan")
    if mode == "grid":
        ot_return = _materialize_grid_ot_return(
            hierarchical_solver,
            ot_return,
            need_final_obj=True,
            need_sparse_coupling=bool(return_coupling),
        )

    dist = ot_return.get("final_obj", float("nan"))
    coupling = (
        _sparse_coupling_from_data(ot_return.get("sparse_coupling"), (n_s, n_t))
        if return_coupling
        else sp.coo_matrix((n_s, n_t), dtype=np.float32)
    )
    state = None
    if return_state or log:
        state = _state_from_solver_active_support(
            hierarchical_solver,
            n_s=n_s,
            n_t=n_t,
            dual_uv=ot_return.get("dual"),
            ot_return=ot_return,
            cost_type=cost_name,
            solver_mode=mode,
        )
    extra_log_fields = None
    if mode == "grid":
        extra_log_fields = {
            "grid_lp_diags": ot_return.get("grid_lp_diags", []),
            "grid_iter_snapshots": ot_return.get("grid_iter_snapshots", []),
            "active_support_sizes": ot_return.get("active_support_sizes", []),
            "dual_stats": ot_return.get("dual_stats", []),
            "stop_reasons": ot_return.get("stop_reasons", []),
            "solver_mode": "grid",
        }
    elif mode == "tree":
        extra_log_fields = {
            "tree_lp_diags": ot_return.get("tree_lp_diags", []),
            "solver_mode": "tree",
        }
    return _finalize_emd_output(
        distance=dist,
        coupling=coupling,
        state=state,
        dual_source=ot_return.get("dual", None),
        level_summaries=ot_return.get("level_summaries", []),
        profiling=ot_return.get("profiling"),
        lp_solve_time_total=float(ot_return.get("lp_solve_time_total", _sum_level_summary_metric(ot_return.get("level_summaries", []), "lp_time"))),
        elapsed=sum(item.get("time", 0) for item in ot_return.get("level_summaries", [])),
        extra_log_fields=extra_log_fields,
        log=log,
        return_coupling=return_coupling,
        return_state=return_state,
    )
