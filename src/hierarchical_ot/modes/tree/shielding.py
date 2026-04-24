from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import njit, prange
from numba.typed import List as NumbaList

from ...core.solver_utils import (
    decode_keep_1d_to_struct,
    prolongate_potentials,
    refine_duals,
    remap_duals_for_warm_start,
)
from ...instrumentation.phase_names import (
    COMPONENT_SHIELD_PICK_T_MAP,
    COMPONENT_SHIELD_SENTINELS,
    COMPONENT_SHIELD_UNION,
    COMPONENT_SHIELD_YHAT,
)
from ...types.base import ActiveSupportStrategy, BaseHierarchy, HierarchyLevel
from .logger import tree_log
from .trace_utils import tree_trace_span
from .violation_check import apply_violation_check

logger = logging.getLogger(__name__)

try:
    from HALO.shield import build_shield as halo_build_shield
except Exception:  # pragma: no cover
    halo_build_shield = None


def build_active_set_first_iter(solver, **kwargs):
    """
    CN: 构造某个 tree level 在第一次内迭代前使用的初始 active set。最粗层直接全连接，
    其余层先从 coarse level prolongate/refine warm start，再执行 shielding、violation check
    和 warm-start 重映射，得到该 level 的首轮 LP 支持集。
    EN: Build the initial active set used before the first inner iteration of a tree level.
    The coarsest level uses full support directly; finer levels prolongate/refine the coarse
    warm start, then run shielding, violation checking, and warm-start remapping to produce
    the first LP support for this level.
    """
    level_idx = kwargs["level_idx"]
    level_s = kwargs["level_s"]
    level_t = kwargs["level_t"]
    x_solution_last = kwargs["x_solution_last"]
    y_solution_last = kwargs["y_solution_last"]
    cost_type = kwargs["cost_type"]
    use_last = kwargs["use_last"]
    ifcheck = kwargs["ifcheck"]
    vd_thr = kwargs["vd_thr"]
    check_method = kwargs["check_method"]
    sampled_config = kwargs.get("sampled_config")
    trace_collector = kwargs.get("trace_collector")
    trace_prefix = str(kwargs.get("trace_prefix", "solve_ot"))
    trace_context = dict(kwargs.get("trace_context", {}))

    n_s = len(level_s.points)
    n_t = len(level_t.points)
    prepare_breakdown: Dict[str, float] = {}
    nnz_thr = float(getattr(solver, "nnz_thr", 1e-20))

    if level_idx == solver.hierarchy_s.num_levels:
        tree_log(solver, f"    [Init] 最粗层 L{level_idx}，使用全连接初始化")
        x_init = np.zeros(n_s + n_t, dtype=np.float32)
        init_result = solver.strategy.initialize_support(
            level_s=level_s,
            level_t=level_t,
            x_init=x_init,
            hierarchy_s=solver.hierarchy_s,
            hierarchy_t=solver.hierarchy_t,
        )
        keep = np.asarray(init_result["keep"], dtype=np.int64)
        keep_coord = init_result["keep_coord"]
        y_init = {
            "y": init_result.get("y_init", np.zeros(len(keep), dtype=np.float32))
        }
        tree_log(solver, f"    [Init] keep={len(keep)} ({len(keep)/(n_s+n_t):.2f}x n_total)")
        y_vals = y_init["y"]
        tree_log(
            solver,
            f"    [Init] y_init: mean={y_vals.mean():.6f}, std={y_vals.std():.6f}, sum={y_vals.sum():.6f}",
        )
        tree_log(
            solver,
            f"    [Init] x_init: mean={x_init.mean():.6f}, std={x_init.std():.6f}",
        )
        return {
            "x_init": x_init,
            "y_init": y_init,
            "keep": keep,
            "keep_coord": keep_coord,
            "prepare_breakdown": prepare_breakdown,
            "trace_keep_after_shield": None,
            "trace_keep_after_check": None,
            "trace_keep_after_uselast": None,
        }

    tree_log(solver, f"    [Init] 从 L{level_idx+1} 初始化 L{level_idx}")
    coarse_level_s = solver.hierarchy_s.levels[level_idx + 1]
    coarse_level_t = solver.hierarchy_t.levels[level_idx + 1]
    x_init = prolongate_potentials(
        x_solution_last, level_s, level_t, coarse_level_s, coarse_level_t
    )
    t_refine = time.perf_counter()
    y_refined, keep_refined, _ = refine_duals(
        y_solution_last, level_s, level_t, coarse_level_s, coarse_level_t, thr=nnz_thr
    )
    prepare_breakdown["refine_duals"] = time.perf_counter() - t_refine
    tree_log(solver, f"    [Init] refine_duals: y_keep={len(keep_refined)}")

    t_shield = time.perf_counter()
    update_result = solver.strategy.update_active_support(
        x_solution=x_init,
        y_solution_last={"y": y_refined, "keep": keep_refined},
        level_s=level_s,
        level_t=level_t,
        hierarchy_s=solver.hierarchy_s,
        hierarchy_t=solver.hierarchy_t,
        build_aux=False,
        trace_collector=trace_collector,
        trace_prefix=trace_prefix,
        trace_context=trace_context,
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
    prepare_breakdown["shielding"] = prepare_breakdown.get("shielding", 0.0) + max(
        0.0, update_time - keep_union_time
    )
    keep = np.asarray(update_result["keep"], dtype=np.int64)
    t_vcheck = time.perf_counter()
    keep, _ = apply_violation_check(
        solver,
        x_dual=x_init,
        level_s=level_s,
        level_t=level_t,
        keep=keep,
        cost_type=cost_type,
        ifcheck=ifcheck,
        vd_thr=vd_thr,
        check_method=check_method,
        sampled_config=sampled_config,
        trace_collector=trace_collector,
        trace_prefix=trace_prefix,
        trace_context=trace_context,
        return_meta=True,
    )
    prepare_breakdown["violation_check"] = time.perf_counter() - t_vcheck
    t0 = time.perf_counter()
    keep_coord = decode_keep_1d_to_struct(keep, n_t)
    prepare_breakdown["keep_coord"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_init = {
        "y": remap_duals_for_warm_start({"y": y_refined, "keep": keep_refined}, keep)
    }
    prepare_breakdown["remap_y"] = time.perf_counter() - t0
    if use_last:
        solver.keep_last = keep
    return {
        "x_init": x_init,
        "y_init": y_init,
        "keep": keep,
        "keep_coord": keep_coord,
        "prepare_breakdown": prepare_breakdown,
        "trace_keep_after_shield": int(len(update_result["keep"])),
        "trace_keep_after_check": int(len(keep)),
        "trace_keep_after_uselast": int(len(keep)),
    }


def build_active_set_subsequent_iter(solver, **kwargs):
    """
    CN: 基于上一轮 LP 解更新下一轮内迭代要用的 active set。流程是先筛掉近零流量边，
    再执行 shielding、violation check、use_last 合并和 warm-start 重映射，为下一轮 LP
    提供新的支持集与初始值。
    EN: Update the active set for the next inner iteration based on the previous LP solution.
    The flow first filters near-zero transport entries, then runs shielding, violation
    checking, use-last merging, and warm-start remapping to prepare the next LP support
    and initial values.
    """
    level_s = kwargs["level_s"]
    level_t = kwargs["level_t"]
    x_solution_last = kwargs["x_solution_last"]
    y_solution_last = kwargs["y_solution_last"]
    cost_type = kwargs["cost_type"]
    use_last = kwargs["use_last"]
    use_last_after_inner0 = kwargs["use_last_after_inner0"]
    ifcheck = kwargs["ifcheck"]
    vd_thr = kwargs["vd_thr"]
    check_method = kwargs["check_method"]
    inner_iter = kwargs["inner_iter"]
    sampled_config = kwargs.get("sampled_config")
    trace_collector = kwargs.get("trace_collector")
    trace_prefix = str(kwargs.get("trace_prefix", "solve_ot"))
    trace_context = dict(kwargs.get("trace_context", {}))

    n_t = len(level_t.points)
    x_init = x_solution_last
    prepare_breakdown: Dict[str, float] = {}
    nnz_thr = float(getattr(solver, "nnz_thr", 1e-20))

    y_last = y_solution_last["y"]
    keep_last = y_solution_last["keep"]
    nonzero_mask = np.abs(y_last) > nnz_thr
    y_keep = keep_last[nonzero_mask]
    y_vals = y_last[nonzero_mask]
    tree_log(solver, f"    [Init] inner_iter>0: y_keep after nonzero_mask={len(y_keep)}")

    t_shield = time.perf_counter()
    update_result = solver.strategy.update_active_support(
        x_solution=x_init,
        y_solution_last={"y": y_vals, "keep": y_keep},
        level_s=level_s,
        level_t=level_t,
        hierarchy_s=solver.hierarchy_s,
        hierarchy_t=solver.hierarchy_t,
        build_aux=False,
        trace_collector=trace_collector,
        trace_prefix=trace_prefix,
        trace_context=trace_context,
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
    prepare_breakdown["shielding"] = prepare_breakdown.get("shielding", 0.0) + max(
        0.0, update_time - keep_union_time
    )
    keep = np.asarray(update_result["keep"], dtype=np.int64)
    trace_keep_after_shield = int(len(keep))
    t_vcheck = time.perf_counter()
    keep, _ = apply_violation_check(
        solver,
        x_dual=x_init,
        level_s=level_s,
        level_t=level_t,
        keep=keep,
        cost_type=cost_type,
        ifcheck=ifcheck,
        vd_thr=vd_thr,
        check_method=check_method,
        sampled_config=sampled_config,
        trace_collector=trace_collector,
        trace_prefix=trace_prefix,
        trace_context=trace_context,
        return_meta=True,
    )
    prepare_breakdown["violation_check"] = time.perf_counter() - t_vcheck
    trace_keep_after_check = int(len(keep))
    t0 = time.perf_counter()
    keep = merge_with_use_last(
        solver,
        keep=keep,
        use_last=use_last,
        use_last_after_inner0=use_last_after_inner0,
        inner_iter=inner_iter,
        n_t=n_t,
    )
    prepare_breakdown["keep_union"] = prepare_breakdown.get("keep_union", 0.0) + (
        time.perf_counter() - t0
    )
    trace_keep_after_uselast = int(len(keep))
    t0 = time.perf_counter()
    keep_coord = decode_keep_1d_to_struct(keep, n_t)
    prepare_breakdown["keep_coord"] = time.perf_counter() - t0

    if np.array_equal(keep, keep_last):
        y_init = {"y": y_last}
    else:
        t0 = time.perf_counter()
        y_init = {"y": remap_duals_for_warm_start({"y": y_vals, "keep": y_keep}, keep)}
        prepare_breakdown["remap_y"] = time.perf_counter() - t0
    return {
        "x_init": x_init,
        "y_init": y_init,
        "keep": keep,
        "keep_coord": keep_coord,
        "prepare_breakdown": prepare_breakdown,
        "trace_keep_after_shield": trace_keep_after_shield,
        "trace_keep_after_check": trace_keep_after_check,
        "trace_keep_after_uselast": trace_keep_after_uselast,
    }


def merge_with_use_last(solver, **kwargs):
    """
    CN: 按 `use_last` / `use_last_after_inner0` 规则把当前候选 active set 与上一轮缓存的
    `keep_last` 合并，减少相邻内迭代之间支持集抖动。
    EN: Merge the current candidate active set with the cached `keep_last` according to
    `use_last` / `use_last_after_inner0`, reducing support-set oscillation between adjacent
    inner iterations.
    """
    keep = np.asarray(kwargs["keep"], dtype=np.int64)
    use_last = kwargs["use_last"]
    use_last_after_inner0 = kwargs["use_last_after_inner0"]
    inner_iter = kwargs["inner_iter"]
    del kwargs["n_t"]
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
            solver.keep_last = keep
            tree_log(
                solver,
                f"    [Init] use_last: inner_iter=1, 保存 keep_last={len(solver.keep_last)}",
            )
            return keep
        if solver.keep_last is not None and len(solver.keep_last) > 0:
            keep = _sorted_union(keep, solver.keep_last)
            tree_log(solver, f"    [Init] use_last: 合并后 keep={len(keep)}")
        solver.keep_last = keep
        return keep

    if solver.keep_last is not None and len(solver.keep_last) > 0:
        keep = _sorted_union(keep, solver.keep_last)
        tree_log(solver, f"    [Init] use_last: 合并后 keep={len(keep)}")
    solver.keep_last = keep
    return keep


class ShieldingStrategy(ActiveSupportStrategy):
    """
    基于几何遮蔽的 Active Support 策略 (HALO Shielding)。
    使用 Numba 加速的树搜索寻找未遮蔽对。
    """

    def __init__(
        self,
        k_neighbors: int = 8,
        max_pairs_per_xA: int = 30,
        cost_type: str = "L2",
        cost_p: float = 2.0,
        search_method: str = "tree_numba",
        shield_impl: Optional[str] = None,
        nnz_thr: float = 1e-20,
        verbose: bool = False,
    ):
        self.k_neighbors = k_neighbors
        self.max_pairs_per_xA = max_pairs_per_xA
        self.cost_type = cost_type
        self.cost_p = cost_p
        self.search_method = str(search_method).lower()
        if shield_impl is None:
            shield_impl = "halo"
        self.shield_impl = str(shield_impl).lower()
        if self.shield_impl not in {"local", "halo"}:
            raise ValueError(
                f"Unknown shield_impl={self.shield_impl!r}. Supported values: 'local', 'halo'."
            )
        self.nnz_thr = float(nnz_thr)
        self.verbose = verbose

    def initialize_support(
        self,
        level_s: HierarchyLevel,
        level_t: HierarchyLevel,
        x_init: Optional[np.ndarray] = None,
        hierarchy_s: BaseHierarchy = None,
        hierarchy_t: BaseHierarchy = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        CN: 为最粗层初始化全连接 active set，并在需要时先把层次结构拍平成后续 numba 搜索可用
        的数组表示。
        EN: Initialize the full-support active set for the coarsest level and, when needed,
        flatten the hierarchies into array layouts usable by later numba-based searches.
        """
        n_s = len(level_s.points)
        n_t = len(level_t.points)

        if (
            not hasattr(hierarchy_s, "flat_centers")
            or hierarchy_s.flat_centers is None
            or len(hierarchy_s.flat_centers) == 0
        ) and hasattr(hierarchy_s, "flatten_for_numba"):
            hierarchy_s.flatten_for_numba()
        if (
            not hasattr(hierarchy_t, "flat_centers")
            or hierarchy_t.flat_centers is None
            or len(hierarchy_t.flat_centers) == 0
        ) and hasattr(hierarchy_t, "flatten_for_numba"):
            hierarchy_t.flatten_for_numba()

        idx_s, idx_t = np.meshgrid(
            np.arange(n_s, dtype=np.int32),
            np.arange(n_t, dtype=np.int32),
            indexing="ij",
        )
        keep_1d = idx_s.flatten().astype(np.int64) * n_t + idx_t.flatten().astype(np.int64)

        keep_coord = np.empty(
            len(keep_1d), dtype=[("idx1", np.int32), ("idx2", np.int32)]
        )
        keep_coord["idx1"] = idx_s.flatten()
        keep_coord["idx2"] = idx_t.flatten()

        return {
            "keep": keep_1d,
            "keep_coord": keep_coord,
            "y_init": np.zeros(len(keep_1d), dtype=np.float32),
        }

    def update_active_support(
        self,
        x_solution: np.ndarray,
        y_solution_last: Dict[str, np.ndarray],
        level_s: HierarchyLevel,
        level_t: HierarchyLevel,
        hierarchy_s: BaseHierarchy,
        hierarchy_t: BaseHierarchy,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        CN: 给定当前 dual/transport 支持，执行 tree shielding 主流程：选每个 source 的代表目标
        `t_map`，构造 sentinel，搜索未被屏蔽的 `Yhat`，再与已有支持集合并，产出新的 active set。
        EN: Run the main tree-shielding pipeline from the current dual/transport support:
        pick a representative target `t_map` for each source, build sentinels, search for
        unshielded `Yhat` pairs, and union them with the existing support to produce the
        next active set.
        """
        trace_collector = kwargs.get("trace_collector")
        trace_context = dict(kwargs.get("trace_context", {}))
        trace_level = trace_context.get("level_idx")
        trace_iter = trace_context.get("inner_iter")
        trace_stage = trace_context.get("stage", "post_lp_pricing")

        def _trace_span(name: str, args: Optional[Dict[str, object]] = None):
            return tree_trace_span(
                trace_collector,
                name,
                level_idx=trace_level,
                inner_iter=trace_iter,
                stage=trace_stage,
                args=args,
            )

        build_aux = bool(kwargs.get("build_aux", True))
        n_s = len(level_s.points)
        n_t = len(level_t.points)
        n_total = n_s + n_t

        y_val = y_solution_last["y"]
        y_keep = y_solution_last["keep"]

        if self.nnz_thr > 0.0 and y_val.size > 0:
            nnz_mask = np.abs(y_val) > self.nnz_thr
            if not np.all(nnz_mask):
                y_val = y_val[nnz_mask]
                y_keep = y_keep[nnz_mask]

        if self.search_method != "tree_numba":
            logger.warning(
                "本地 ShieldingStrategy 仅支持 search_method=tree_numba，当前=%s，已回退为 tree_numba",
                self.search_method,
            )

        if self.verbose:
            logger.debug("    [Shielding] n_s=%s, n_t=%s, y_keep=%s", n_s, n_t, len(y_keep))

        timing: Dict[str, float] = {}

        if self.shield_impl == "halo" and halo_build_shield is not None:
            t0 = time.perf_counter()
            detailed_timing: Dict[str, float] = {}
            with _trace_span(
                "tree.shield.build_halo_shield",
                args={"shield_impl": self.shield_impl, "nnz_input": int(len(y_keep))},
            ):
                try:
                    keep_1d, _len_dict = halo_build_shield(
                        level_s,
                        level_t,
                        y_val,
                        y_keep,
                        hierarchy_s,
                        hierarchy_t,
                        return_gpu=False,
                        k_neighbors=self.k_neighbors,
                        cost_type=self.cost_type,
                        p=self.cost_p,
                        search_method=self.search_method,
                        max_pairs_per_xA=self.max_pairs_per_xA,
                        verbose_tree_stats=False,
                        detailed_timing=detailed_timing,
                    )
                except TypeError as exc:
                    if "detailed_timing" not in str(exc):
                        raise
                    detailed_timing.clear()
                    keep_1d, _len_dict = halo_build_shield(
                        level_s,
                        level_t,
                        y_val,
                        y_keep,
                        hierarchy_s,
                        hierarchy_t,
                        return_gpu=False,
                        k_neighbors=self.k_neighbors,
                        cost_type=self.cost_type,
                        p=self.cost_p,
                        search_method=self.search_method,
                        max_pairs_per_xA=self.max_pairs_per_xA,
                        verbose_tree_stats=False,
                    )
            timing["shield_total"] = time.perf_counter() - t0
            if detailed_timing:
                pick_t = float(detailed_timing.get("pick_t_map", 0.0) or 0.0)
                sentinels = float(detailed_timing.get("prepare_sentinels", 0.0) or 0.0)
                yhat = float(detailed_timing.get("build_yhat", 0.0) or 0.0)
                union = float(detailed_timing.get("union", 0.0) or 0.0)
                timing[COMPONENT_SHIELD_PICK_T_MAP] = pick_t
                timing[COMPONENT_SHIELD_SENTINELS] = sentinels
                timing[COMPONENT_SHIELD_YHAT] = yhat
                timing[COMPONENT_SHIELD_UNION] = union
                timing["shield_total"] = float(
                    detailed_timing.get("total", timing["shield_total"])
                    or timing["shield_total"]
                )
            result = {
                "keep": np.asarray(keep_1d, dtype=np.int64),
                "timing": timing,
                "diag": {
                    "shield_impl": self.shield_impl,
                    "nnz_input": int(len(y_keep)),
                    "keep_size": int(len(keep_1d)),
                },
            }
            if build_aux:
                result["keep_coord"] = decode_keep_1d_to_struct(keep_1d, n_t)
                result["y_init"] = remap_duals_for_warm_start(y_solution_last, keep_1d)
            return result

        t0 = time.perf_counter()
        with _trace_span(
            "tree.shield.pick_t_map",
            args={"shield_impl": self.shield_impl, "nnz_input": int(len(y_keep))},
        ):
            best_idx_y, _ = _pick_t_map_arrays(y_keep, y_val, n_s, n_t)
        timing["t_map"] = time.perf_counter() - t0
        timing[COMPONENT_SHIELD_PICK_T_MAP] = timing["t_map"]

        if self.verbose:
            t_map_size = int(np.sum(best_idx_y >= 0))
            logger.debug("  [Shielding] t_map size: %s/%s", t_map_size, n_s)

        t0 = time.perf_counter()
        knn_indices = (
            level_s.knn_indices
            if level_s.knn_indices is not None
            else np.empty((n_s, 0), dtype=np.int32)
        )
        with _trace_span("tree.shield.prepare_sentinels", args={"k_neighbors": int(self.k_neighbors)}):
            sentinels_list, shield_arr = _prepare_sentinels_for_numba_fast(
                nodes_X=level_s.points,
                nodes_Y=level_t.points,
                t_idx=best_idx_y,
                all_knn_indices=knn_indices,
                k_neighbors=self.k_neighbors,
            )
        timing["sentinels"] = time.perf_counter() - t0
        timing[COMPONENT_SHIELD_SENTINELS] = timing["sentinels"]

        if self.verbose:
            logger.debug("    [Shielding] shield_arr=%s", len(shield_arr))

        t0 = time.perf_counter()
        with _trace_span("tree.shield.build_yhat", args={"target_level_idx": int(level_t.level_idx)}):
            if hasattr(hierarchy_t, "flat_centers"):
                yhat_pairs = build_Yhat_tree_numba(
                    nodes_X=level_s.points,
                    hierarchy_Y=hierarchy_t,
                    sentinels_list=sentinels_list,
                    target_level_idx=level_t.level_idx,
                    max_pairs_per_xA=self.max_pairs_per_xA,
                    verbose=self.verbose,
                )
            else:
                yhat_pairs = np.empty((0, 2), dtype=np.int32)
        timing["yhat"] = time.perf_counter() - t0
        timing[COMPONENT_SHIELD_YHAT] = timing["yhat"]

        if self.verbose:
            logger.debug("    [Shielding] yhat_pairs=%s", len(yhat_pairs))

        t0 = time.perf_counter()
        parts_1d = []
        if len(y_keep) > 0:
            parts_1d.append(y_keep.astype(np.int64))
        if len(shield_arr) > 0:
            shield_1d = shield_arr[:, 0].astype(np.int64) * n_t + shield_arr[:, 1].astype(
                np.int64
            )
            parts_1d.append(shield_1d)
        if len(yhat_pairs) > 0:
            yhat_1d = yhat_pairs[:, 0].astype(np.int64) * n_t + yhat_pairs[:, 1].astype(
                np.int64
            )
            parts_1d.append(yhat_1d)

        with _trace_span("tree.shield.union"):
            if parts_1d:
                all_1d = np.concatenate(parts_1d)
                keep_1d = np.unique(all_1d)
            else:
                keep_1d = np.empty(0, dtype=np.int64)
        timing["keep_union"] = time.perf_counter() - t0
        timing[COMPONENT_SHIELD_UNION] = timing["keep_union"]

        if self.verbose:
            logger.debug(
                "    [Shielding] keep_1d=%s (%.2fx n_total)",
                len(keep_1d),
                len(keep_1d) / n_total if n_total > 0 else 0.0,
            )

        result = {
            "keep": keep_1d,
            "timing": timing,
            "diag": {
                "shield_impl": self.shield_impl,
                "nnz_input": int(len(y_keep)),
                "t_map_size": int(np.sum(best_idx_y >= 0)),
                "shield_arr_size": int(len(shield_arr)),
                "yhat_size": int(len(yhat_pairs)),
                "keep_size": int(len(keep_1d)),
            },
        }
        if build_aux:
            result["keep_coord"] = decode_keep_1d_to_struct(keep_1d, n_t)
            result["y_init"] = remap_duals_for_warm_start(y_solution_last, keep_1d)
        return result

    def close(self):
        pass


@njit(cache=True)
def _pick_t_core_numba(idx_x_all, idx_y_all, y_val, n_X):
    """
    CN: 在 numba 内核中为每个 source 行选择流量值最大的目标索引，形成 shielding 所需的
    代表目标映射基础数据。
    EN: In a numba kernel, select the target index with the largest flow for each source
    row, forming the representative target mapping used by shielding.
    """
    best_val = np.empty(n_X, dtype=y_val.dtype)
    best_idx_y = np.empty(n_X, dtype=np.int64)

    for i in range(n_X):
        best_val[i] = -np.inf
        best_idx_y[i] = -1

    for k in range(idx_x_all.shape[0]):
        ix = int(idx_x_all[k])
        v = y_val[k]
        if v > best_val[ix]:
            best_val[ix] = v
            best_idx_y[ix] = int(idx_y_all[k])

    return best_idx_y, best_val


def _pick_t_map_arrays(
    y_keep: np.ndarray, y_val: np.ndarray, n_X: int, n_Y: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CN: 把稀疏 transport 支持 `(keep, y)` 转成按 source 索引对齐的数组版 `t_map`，
    便于后续快速构造 sentinel。
    EN: Convert sparse transport support `(keep, y)` into an array-aligned `t_map` by source
    index, making later sentinel construction faster.
    """
    if y_keep.size == 0:
        return np.full(n_X, -1, dtype=np.int64), np.full(n_X, -np.inf, dtype=np.float32)

    y_keep_i64 = y_keep.astype(np.int64, copy=False)
    idx_x_all = (y_keep_i64 // int(n_Y)).astype(np.int64, copy=False)
    idx_y_all = (y_keep_i64 % int(n_Y)).astype(np.int64, copy=False)

    best_idx_y, best_val = _pick_t_core_numba(idx_x_all, idx_y_all, y_val, n_X)
    return best_idx_y, best_val


def _pick_t_map(y_keep, y_val, n_X, n_Y):
    """
    CN: 生成字典版 `t_map`，主要供本地 shielding 兼容路径使用；语义与数组版相同，但便于
    Python 侧按节点访问。
    EN: Build a dictionary-based `t_map`, mainly for the local shielding compatibility path;
    it has the same meaning as the array version but is convenient for Python-side node lookup.
    """
    t_map = {}
    t_map_values = {}

    if y_keep.size == 0:
        return t_map, t_map_values

    y_keep_i64 = y_keep.astype(np.int64, copy=False)
    idx_x_all = (y_keep_i64 // int(n_Y)).astype(np.int64, copy=False)
    idx_y_all = (y_keep_i64 % int(n_Y)).astype(np.int64, copy=False)

    best_idx_y, best_val = _pick_t_core_numba(idx_x_all, idx_y_all, y_val, n_X)
    valid = best_idx_y >= 0
    x_indices = np.nonzero(valid)[0]

    for i in x_indices:
        t_map[int(i)] = int(best_idx_y[i])
        t_map_values[int(i)] = float(best_val[i])

    return t_map, t_map_values


def _prepare_sentinels_for_numba_fast(
    nodes_X: np.ndarray,
    nodes_Y: np.ndarray,
    t_idx: np.ndarray,
    all_knn_indices: np.ndarray,
    k_neighbors: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    CN: 根据每个 source 节点的近邻和 `t_map`，批量构造 numba 搜索使用的 sentinel 张量，
    同时生成 shielding 必保留的 `(x_A, t_S)` 配对。
    EN: Batch-build the sentinel tensors used by the numba search from per-source neighbors
    and `t_map`, while also producing the shielding pairs `(x_A, t_S)` that must be kept.
    """
    n_X, dim = nodes_X.shape
    if all_knn_indices is None or all_knn_indices.size == 0 or k_neighbors <= 0:
        sentinels_by_A = [np.empty((0, 2, dim), dtype=np.float64) for _ in range(n_X)]
        return sentinels_by_A, np.empty((0, 2), dtype=np.int32)

    k_eff = min(k_neighbors, all_knn_indices.shape[1])
    knn = all_knn_indices[:, :k_eff]

    s_flat = knn.reshape(-1).astype(np.int64, copy=False)
    a_flat = np.repeat(np.arange(n_X, dtype=np.int64), k_eff)

    mask_valid = (s_flat >= 0) & (s_flat < n_X) & (s_flat != a_flat)
    if not np.any(mask_valid):
        sentinels_by_A = [np.empty((0, 2, dim), dtype=np.float64) for _ in range(n_X)]
        return sentinels_by_A, np.empty((0, 2), dtype=np.int32)

    a_valid = a_flat[mask_valid]
    s_valid = s_flat[mask_valid]

    t_s = t_idx[s_valid]
    mask_map = t_s >= 0
    if not np.any(mask_map):
        sentinels_by_A = [np.empty((0, 2, dim), dtype=np.float64) for _ in range(n_X)]
        return sentinels_by_A, np.empty((0, 2), dtype=np.int32)

    a_valid = a_valid[mask_map]
    s_valid = s_valid[mask_map]
    t_s_valid = t_s[mask_map].astype(np.int64, copy=False)

    m = a_valid.shape[0]
    xs_all = nodes_X[s_valid, :]
    x_a_all = nodes_X[a_valid, :]
    ys_all = nodes_Y[t_s_valid, :]

    arr_all = np.empty((m, 2, dim), dtype=np.float64)
    arr_all[:, 0, :] = xs_all - x_a_all
    arr_all[:, 1, :] = ys_all

    keep_shield_pairs = np.empty((m, 2), dtype=np.int32)
    keep_shield_pairs[:, 0] = a_valid.astype(np.int32, copy=False)
    keep_shield_pairs[:, 1] = t_s_valid.astype(np.int32, copy=False)

    if m > 0:
        keep_cont = np.ascontiguousarray(keep_shield_pairs)
        keep_shield_pairs = np.unique(keep_cont.view(f"V{keep_cont.dtype.itemsize * 2}")).view(
            keep_cont.dtype
        ).reshape(-1, 2)
    else:
        keep_shield_pairs = np.empty((0, 2), dtype=np.int32)

    order = np.argsort(a_valid, kind="mergesort")
    a_sorted = a_valid[order]
    arr_sorted = arr_all[order]

    counts = np.bincount(a_sorted, minlength=n_X)
    sentinels_by_A: List[np.ndarray] = []
    start = 0
    for i_A in range(n_X):
        c = counts[i_A]
        if c == 0:
            sentinels_by_A.append(np.empty((0, 2, dim), dtype=np.float64))
        else:
            end = start + c
            sentinels_by_A.append(arr_sorted[start:end])
            start = end

    return sentinels_by_A, keep_shield_pairs


def _build_sentinels(level_X, t_map, knn_indices, k_neighbors, level_Y=None):
    """
    CN: Python 兼容版本的 sentinel 构造逻辑。它逐个 source 节点遍历近邻，拼出 shielding
    需要的向量差和对应目标代表点。
    EN: Python compatibility version of sentinel construction. It iterates source nodes and
    their neighbors to assemble the vector differences and representative target points needed
    by shielding.
    """
    nodes_X = level_X.points
    n_X = len(nodes_X)
    dim = nodes_X.shape[1]
    actual_k = knn_indices.shape[1] if len(knn_indices) > 0 else 0

    sentinels_list = []
    shield_pairs = []

    for i_A in range(n_X):
        x_A = nodes_X[i_A]
        neighbors = (
            knn_indices[i_A] if i_A < len(knn_indices) else np.array([], dtype=np.int32)
        )

        sentinels = []
        for i_S in neighbors[:actual_k]:
            if i_S < 0 or i_S == i_A or i_S not in t_map:
                continue
            x_S = nodes_X[i_S]
            y_tS_idx = t_map[i_S]

            if level_Y is not None and y_tS_idx >= 0:
                y_tS = level_Y.points[y_tS_idx]
            else:
                y_tS = np.zeros(dim, dtype=np.float64)

            sentinels.append((x_S, x_A, y_tS))
            shield_pairs.append((i_A, y_tS_idx))

        if len(sentinels) == 0:
            sentinels_list.append(np.empty((0, 2, dim), dtype=np.float64))
            continue

        arr = np.empty((len(sentinels), 2, dim), dtype=np.float64)
        for k, (x_S, x_A_local, y_tS) in enumerate(sentinels):
            arr[k, 0, :] = x_S - x_A_local
            arr[k, 1, :] = y_tS

        sentinels_list.append(arr)

    if len(shield_pairs) > 0:
        shield_arr = np.array(shield_pairs, dtype=np.int32)
        shield_arr = np.unique(
            np.ascontiguousarray(shield_arr).view(f"V{shield_arr.dtype.itemsize * 2}")
        ).view(shield_arr.dtype).reshape(-1, 2)
    else:
        shield_arr = np.empty((0, 2), dtype=np.int32)

    return sentinels_list, shield_arr


@njit(cache=True)
def _get_exact_inner_njit(
    rep_A: np.ndarray,
    rep_B: np.ndarray,
    candidates: np.ndarray,
) -> float:
    """
    CN: 对单个候选目标点 `rep_B` 与一组 sentinel 精确计算 shielding 判据中的最大内积值，
    用于决定该目标是否会被屏蔽。
    EN: For one candidate target point `rep_B`, compute the maximum inner-product value in the
    shielding criterion against a set of sentinels, deciding whether the target is shielded.
    """
    best = -1e300
    k_eff = candidates.shape[0]
    if k_eff == 0:
        return best

    dim = rep_A.shape[0]
    for k in range(k_eff):
        v_xs_xa = candidates[k, 0, :]
        y_tS = candidates[k, 1, :]

        tmp = 0.0
        for d in range(dim):
            tmp += (rep_B[d] - y_tS[d]) * v_xs_xa[d]

        if tmp > best:
            best = tmp
    return best


@njit(cache=True, fastmath=True, parallel=True)
def _search_all_xA_count(
    nodes_X,
    l0_points_all,
    sentinels_list,
    centers,
    radii,
    levels,
    is_leaf,
    public_idx,
    child_ptr,
    child_indices,
    leaf_point_ptr,
    leaf_points,
    root_indices,
    target_level_idx,
    max_pairs_per_xA=-1,
) -> np.ndarray:
    """
    CN: 在真正写出 `Yhat` 配对之前，先对每个 source 节点统计未被屏蔽的目标数量，
    以便为后续填充阶段分配精确大小的输出缓冲区。
    EN: Before materializing `Yhat` pairs, count the number of unshielded targets for each
    source node so the fill stage can allocate an exact-sized output buffer.
    """
    n_X, dim = nodes_X.shape
    counts = np.zeros(n_X, dtype=np.int64)
    max_stack_size = centers.shape[0] + root_indices.shape[0] + 8
    prune_threshold = 0.0

    for i_A in prange(n_X):
        rep_A = nodes_X[i_A]
        candidates = sentinels_list[i_A]
        k_eff = candidates.shape[0]
        if k_eff == 0:
            continue

        stack = np.empty(max_stack_size, dtype=np.int64)
        cand_norms = np.empty(k_eff, dtype=np.float32)
        for kk in range(k_eff):
            v = candidates[kk, 0, :]
            norm_sq = 0.0
            for d in range(dim):
                norm_sq += v[d] * v[d]
            cand_norms[kk] = np.sqrt(norm_sq)

        local_count = 0
        for r_idx in range(root_indices.shape[0]):
            if max_pairs_per_xA > 0 and local_count >= max_pairs_per_xA:
                break
            top = 0
            stack[top] = int(root_indices[r_idx])
            top += 1

            while top > 0:
                if max_pairs_per_xA > 0 and local_count >= max_pairs_per_xA:
                    break
                top -= 1
                node_idx = stack[top]

                is_pruned = False
                for kk in range(k_eff):
                    v_xs_xa = candidates[kk, 0, :]
                    rep_tS = candidates[kk, 1, :]
                    norm_v = cand_norms[kk]

                    inner = 0.0
                    for d in range(dim):
                        inner += (centers[node_idx, d] - rep_tS[d]) * v_xs_xa[d]

                    bound = inner - norm_v * radii[node_idx]
                    if bound > prune_threshold:
                        is_pruned = True
                        break

                if is_pruned:
                    continue

                node_level = int(levels[node_idx])
                leaf_flag = is_leaf[node_idx] == 1

                if node_level == target_level_idx and target_level_idx > 0:
                    rep_B = centers[node_idx, :]
                    best_inner = _get_exact_inner_njit(rep_A, rep_B, candidates)
                    if not (best_inner > prune_threshold):
                        local_count += 1
                    continue

                if leaf_flag and target_level_idx == 0:
                    start = leaf_point_ptr[node_idx]
                    end = leaf_point_ptr[node_idx + 1]
                    for offs in range(start, end):
                        if max_pairs_per_xA > 0 and local_count >= max_pairs_per_xA:
                            break
                        idx_B_L0 = leaf_points[offs]
                        rep_B = l0_points_all[idx_B_L0, :]
                        best_inner = _get_exact_inner_njit(rep_A, rep_B, candidates)
                        if not (best_inner > prune_threshold):
                            local_count += 1
                    continue

                if leaf_flag and target_level_idx > 0:
                    continue

                c_start = child_ptr[node_idx]
                c_end = child_ptr[node_idx + 1]
                for ci in range(c_start, c_end):
                    child = child_indices[ci]
                    stack[top] = child
                    top += 1

        counts[i_A] = local_count

    return counts


@njit(cache=True, fastmath=True, parallel=True)
def _search_all_xA_fill(
    nodes_X,
    l0_points_all,
    sentinels_list,
    centers,
    radii,
    levels,
    is_leaf,
    public_idx,
    child_ptr,
    child_indices,
    leaf_point_ptr,
    leaf_points,
    root_indices,
    target_level_idx,
    offsets,
    pairs_all,
    max_pairs_per_xA=-1,
):
    """
    CN: 按 `_search_all_xA_count` 给出的偏移量把所有未被屏蔽的 `(x_A, y_B)` 配对写入输出数组，
    形成 `Yhat` 的原始结果。
    EN: Using the offsets computed by `_search_all_xA_count`, write all unshielded
    `(x_A, y_B)` pairs into the output array to form the raw `Yhat` result.
    """
    n_X, dim = nodes_X.shape
    max_stack_size = centers.shape[0] + root_indices.shape[0] + 8
    prune_threshold = 0.0

    for i_A in prange(n_X):
        rep_A = nodes_X[i_A]
        candidates = sentinels_list[i_A]
        k_eff = candidates.shape[0]
        if k_eff == 0:
            continue

        stack = np.empty(max_stack_size, dtype=np.int64)
        cand_norms = np.empty(k_eff, dtype=np.float32)
        for kk in range(k_eff):
            v = candidates[kk, 0, :]
            norm_sq = 0.0
            for d in range(dim):
                norm_sq += v[d] * v[d]
            cand_norms[kk] = np.sqrt(norm_sq)

        base = offsets[i_A]
        limit = offsets[i_A + 1]
        local_pos = 0

        for r_idx in range(root_indices.shape[0]):
            if max_pairs_per_xA > 0 and local_pos >= (limit - base):
                break
            top = 0
            stack[top] = int(root_indices[r_idx])
            top += 1

            while top > 0:
                if max_pairs_per_xA > 0 and local_pos >= (limit - base):
                    break
                top -= 1
                node_idx = stack[top]

                is_pruned = False
                for kk in range(k_eff):
                    v_xs_xa = candidates[kk, 0, :]
                    y_tS = candidates[kk, 1, :]
                    norm_v = cand_norms[kk]

                    inner = 0.0
                    for d in range(dim):
                        inner += (centers[node_idx, d] - y_tS[d]) * v_xs_xa[d]
                    bound = inner - norm_v * radii[node_idx]
                    if bound > prune_threshold:
                        is_pruned = True
                        break

                if is_pruned:
                    continue

                node_level = int(levels[node_idx])
                leaf_flag = is_leaf[node_idx] == 1

                if node_level == target_level_idx and target_level_idx > 0:
                    rep_B = centers[node_idx, :]
                    best_inner = _get_exact_inner_njit(rep_A, rep_B, candidates)
                    if not (best_inner > prune_threshold):
                        if base + local_pos < limit:
                            pairs_all[base + local_pos, 0] = i_A
                            pairs_all[base + local_pos, 1] = public_idx[node_idx]
                            local_pos += 1
                    continue

                if leaf_flag and target_level_idx == 0:
                    start = leaf_point_ptr[node_idx]
                    end = leaf_point_ptr[node_idx + 1]
                    for offs in range(start, end):
                        if base + local_pos >= limit:
                            break
                        idx_B_L0 = leaf_points[offs]
                        rep_B = l0_points_all[idx_B_L0, :]
                        best_inner = _get_exact_inner_njit(rep_A, rep_B, candidates)
                        if not (best_inner > prune_threshold):
                            pairs_all[base + local_pos, 0] = i_A
                            pairs_all[base + local_pos, 1] = idx_B_L0
                            local_pos += 1
                    continue

                if leaf_flag and target_level_idx > 0:
                    continue

                c_start = child_ptr[node_idx]
                c_end = child_ptr[node_idx + 1]
                for ci in range(c_start, c_end):
                    child = child_indices[ci]
                    stack[top] = child
                    top += 1


def build_Yhat_tree_numba(
    nodes_X: np.ndarray,
    hierarchy_Y,
    sentinels_list: List[np.ndarray],
    target_level_idx: int,
    max_pairs_per_xA: int = -1,
    verbose: bool = False,
) -> np.ndarray:
    """
    CN: 基于拍平后的目标层次结构和 sentinel，为每个 source 节点执行树搜索，找出所有未被
    屏蔽的 `Yhat` 候选配对。
    EN: Run the tree search for each source node using the flattened target hierarchy and
    sentinels, producing all unshielded `Yhat` candidate pairs.
    """
    t0 = time.perf_counter()

    n_X = nodes_X.shape[0]
    if n_X == 0:
        return np.empty((0, 2), dtype=np.int32)

    if not hasattr(hierarchy_Y, "flat_centers"):
        raise ValueError("Hierarchy must be flattened first")

    centers = hierarchy_Y.flat_centers
    radii = hierarchy_Y.flat_radii
    levels = hierarchy_Y.flat_levels
    is_leaf = hierarchy_Y.flat_is_leaf
    public_idx = hierarchy_Y.flat_public_idx
    child_ptr = hierarchy_Y.flat_child_ptr
    child_indices = hierarchy_Y.flat_child_indices
    leaf_point_ptr = hierarchy_Y.flat_leaf_point_ptr
    leaf_points = hierarchy_Y.flat_leaf_points
    root_indices = hierarchy_Y.flat_root_indices
    l0_points = hierarchy_Y.levels[0].points.astype(np.float64, copy=False)

    nb_sentinels = NumbaList()
    for s in sentinels_list:
        nb_sentinels.append(s.astype(np.float64, copy=False))

    counts = _search_all_xA_count(
        nodes_X.astype(np.float64, copy=False),
        l0_points,
        nb_sentinels,
        centers,
        radii,
        levels,
        is_leaf,
        public_idx,
        child_ptr,
        child_indices,
        leaf_point_ptr,
        leaf_points,
        root_indices,
        int(target_level_idx),
        max_pairs_per_xA,
    )

    offsets = np.zeros(n_X + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    total_pairs = offsets[-1]
    if total_pairs == 0:
        return np.empty((0, 2), dtype=np.int32)

    pairs_all = np.empty((total_pairs, 2), dtype=np.int64)
    _search_all_xA_fill(
        nodes_X.astype(np.float64, copy=False),
        l0_points,
        nb_sentinels,
        centers,
        radii,
        levels,
        is_leaf,
        public_idx,
        child_ptr,
        child_indices,
        leaf_point_ptr,
        leaf_points,
        root_indices,
        int(target_level_idx),
        offsets,
        pairs_all,
        max_pairs_per_xA,
    )

    pairs_arr = pairs_all.astype(np.int32, copy=False)
    pairs_cont = np.ascontiguousarray(pairs_arr)
    unique_pairs = np.unique(pairs_cont.view(f"V{pairs_cont.dtype.itemsize * 2}")).view(
        pairs_cont.dtype
    ).reshape(-1, 2)

    if verbose:
        print(
            f"  [TreeNumba] Found {len(unique_pairs)} unique pairs in {time.perf_counter() - t0:.3f}s"
        )

    return unique_pairs


__all__ = [
    "ShieldingStrategy",
    "build_active_set_first_iter",
    "build_active_set_subsequent_iter",
    "merge_with_use_last",
]
