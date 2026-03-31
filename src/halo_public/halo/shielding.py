import numpy as np
import logging
import time
from typing import Dict, Optional, List, Tuple
from numba import njit

from ..common.base import ActiveSupportStrategy, HierarchyLevel, BaseHierarchy
from .shielding_numba import build_Yhat_tree_numba

logger = logging.getLogger(__name__)

from ..common.profiling_names import (
    COMPONENT_SHIELD_PICK_T_MAP,
    COMPONENT_SHIELD_SENTINELS,
    COMPONENT_SHIELD_YHAT,
    COMPONENT_SHIELD_UNION,
)

halo_build_shield = None

@njit(cache=True)
def _pick_t_core_numba(idx_x_all, idx_y_all, y_val, n_X):
    
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

def _pick_t_map(y_keep, y_val, n_X, n_Y):
    
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

def _pick_t_map_arrays(y_keep: np.ndarray, y_val: np.ndarray, n_X: int, n_Y: int) -> Tuple[np.ndarray, np.ndarray]:
    
    if y_keep.size == 0:
        return np.full(n_X, -1, dtype=np.int64), np.full(n_X, -np.inf, dtype=np.float32)

    y_keep_i64 = y_keep.astype(np.int64, copy=False)
    idx_x_all = (y_keep_i64 // int(n_Y)).astype(np.int64, copy=False)
    idx_y_all = (y_keep_i64 % int(n_Y)).astype(np.int64, copy=False)

    best_idx_y, best_val = _pick_t_core_numba(idx_x_all, idx_y_all, y_val, n_X)
    return best_idx_y, best_val

def _build_sentinels(level_X, t_map, knn_indices, k_neighbors, level_Y=None):
    
    nodes_X = level_X.points
    n_X = len(nodes_X)
    dim = nodes_X.shape[1]

    actual_k = knn_indices.shape[1] if len(knn_indices) > 0 else 0

    sentinels_list = []
    shield_pairs = []

    for i_A in range(n_X):
        x_A = nodes_X[i_A]
        neighbors = knn_indices[i_A] if i_A < len(knn_indices) else np.array([], dtype=np.int32)

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

def _prepare_sentinels_for_numba_fast(
    nodes_X: np.ndarray,
    nodes_Y: np.ndarray,
    t_idx: np.ndarray,
    all_knn_indices: np.ndarray,
    k_neighbors: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    
    n_X, dim = nodes_X.shape
    if all_knn_indices is None or all_knn_indices.size == 0 or k_neighbors <= 0:
        sentinels_by_A = [np.empty((0, 2, dim), dtype=np.float64) for _ in range(n_X)]
        return sentinels_by_A, np.empty((0, 2), dtype=np.int32)

    k_eff = min(k_neighbors, all_knn_indices.shape[1])
    knn = all_knn_indices[:, :k_eff]

    S_flat = knn.reshape(-1).astype(np.int64, copy=False)
    A_flat = np.repeat(np.arange(n_X, dtype=np.int64), k_eff)

    mask_valid = (S_flat >= 0) & (S_flat < n_X) & (S_flat != A_flat)
    if not np.any(mask_valid):
        sentinels_by_A = [np.empty((0, 2, dim), dtype=np.float64) for _ in range(n_X)]
        return sentinels_by_A, np.empty((0, 2), dtype=np.int32)

    A_valid = A_flat[mask_valid]
    S_valid = S_flat[mask_valid]

    tS = t_idx[S_valid]
    mask_map = tS >= 0
    if not np.any(mask_map):
        sentinels_by_A = [np.empty((0, 2, dim), dtype=np.float64) for _ in range(n_X)]
        return sentinels_by_A, np.empty((0, 2), dtype=np.int32)

    A_valid = A_valid[mask_map]
    S_valid = S_valid[mask_map]
    tS_valid = tS[mask_map].astype(np.int64, copy=False)

    M = A_valid.shape[0]
    xs_all = nodes_X[S_valid, :]
    xA_all = nodes_X[A_valid, :]
    ys_all = nodes_Y[tS_valid, :]

    arr_all = np.empty((M, 2, dim), dtype=np.float64)
    arr_all[:, 0, :] = xs_all - xA_all
    arr_all[:, 1, :] = ys_all

    keep_shield_pairs = np.empty((M, 2), dtype=np.int32)
    keep_shield_pairs[:, 0] = A_valid.astype(np.int32, copy=False)
    keep_shield_pairs[:, 1] = tS_valid.astype(np.int32, copy=False)

    if M > 0:
        keep_cont = np.ascontiguousarray(keep_shield_pairs)
        keep_shield_pairs = np.unique(
            keep_cont.view(f"V{keep_cont.dtype.itemsize * 2}")
        ).view(keep_cont.dtype).reshape(-1, 2)
    else:
        keep_shield_pairs = np.empty((0, 2), dtype=np.int32)

    order = np.argsort(A_valid, kind="mergesort")
    A_sorted = A_valid[order]
    arr_sorted = arr_all[order]

    counts = np.bincount(A_sorted, minlength=n_X)
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

class ShieldingStrategy(ActiveSupportStrategy):
    

    def __init__(
        self,
        k_neighbors: int = 8,  
        max_pairs_per_xA: int = 30,  
        cost_type: str = "L2",
        cost_p: float = 2.0,
        search_method: str = "tree_numba",
        shield_impl: Optional[str] = None,
        halo_shielding: Optional[bool] = None,
        nnz_thr: float = 1e-20,
        verbose: bool = False,
    ):
        self.k_neighbors = k_neighbors
        self.max_pairs_per_xA = max_pairs_per_xA
        self.cost_type = cost_type
        self.cost_p = cost_p
        self.search_method = str(search_method).lower()
        if shield_impl is None:
            if halo_shielding is None:
                shield_impl = "halo"
            else:
                shield_impl = "halo" if halo_shielding else "local"
        self.shield_impl = str(shield_impl).lower()
        if self.shield_impl not in {"local", "halo"}:
            logger.warning("Unknown shield_impl=%r, fallback to 'local'.", self.shield_impl)
            self.shield_impl = "local"
        
        self.halo_shielding = bool(halo_shielding) if halo_shielding is not None else (self.shield_impl == "halo")
        self.nnz_thr = float(nnz_thr)
        self.verbose = verbose

    def initialize_support(
        self,
        level_s: HierarchyLevel,
        level_t: HierarchyLevel,
        x_init: Optional[np.ndarray] = None,
        hierarchy_s: BaseHierarchy = None,
        hierarchy_t: BaseHierarchy = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        
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
            indexing='ij'
        )

        keep_1d = idx_s.flatten().astype(np.int64) * n_t + idx_t.flatten().astype(np.int64)

        
        keep_coord = np.empty(len(keep_1d), dtype=[('idx1', np.int32), ('idx2', np.int32)])
        keep_coord['idx1'] = idx_s.flatten()
        keep_coord['idx2'] = idx_t.flatten()

        return {
            'keep': keep_1d,
            'keep_coord': keep_coord,
            'y_init': np.zeros(len(keep_1d), dtype=np.float32),
        }

    def update_active_support(
        self,
        x_solution: np.ndarray,
        y_solution_last: Dict[str, np.ndarray],
        level_s: HierarchyLevel,
        level_t: HierarchyLevel,
        hierarchy_s: BaseHierarchy,
        hierarchy_t: BaseHierarchy,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        
        build_aux = bool(kwargs.get("build_aux", True))
        n_s = len(level_s.points)
        n_t = len(level_t.points)
        n_total = n_s + n_t

        y_val = y_solution_last['y']
        y_keep = y_solution_last['keep']

        if self.nnz_thr > 0.0 and y_val.size > 0:
            nnz_mask = np.abs(y_val) > self.nnz_thr
            if not np.all(nnz_mask):
                y_val = y_val[nnz_mask]
                y_keep = y_keep[nnz_mask]

        if self.search_method != "tree_numba":
            logger.warning(
                "Local ShieldingStrategy only supports search_method=tree_numba, got=%s; falling back to tree_numba",
                self.search_method,
            )

        if self.verbose:
            print(f"    [Shielding] n_s={n_s}, n_t={n_t}, y_keep={len(y_keep)}")

        timing: Dict[str, float] = {}

        if self.shield_impl == "halo" and halo_build_shield is not None:
            t0 = time.perf_counter()
            detailed_timing: Dict[str, float] = {}
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
                timing["shield_total"] = float(detailed_timing.get("total", timing["shield_total"]) or timing["shield_total"])
            result = {
                'keep': np.asarray(keep_1d, dtype=np.int64),
                'timing': timing,
            }
            if build_aux:
                from ..core.utils import decode_keep_1d_to_struct, remap_duals_for_warm_start

                result['keep_coord'] = decode_keep_1d_to_struct(keep_1d, n_t)
                result['y_init'] = remap_duals_for_warm_start(y_solution_last, keep_1d)
            return result
        
        t0 = time.perf_counter()
        if self.shield_impl == "halo":
            best_idx_y, _ = _pick_t_map_arrays(y_keep, y_val, n_s, n_t)
            t_map = None
        else:
            t_map, _ = _pick_t_map(y_keep, y_val, n_s, n_t)
            best_idx_y = None
        timing["t_map"] = time.perf_counter() - t0
        timing[COMPONENT_SHIELD_PICK_T_MAP] = timing["t_map"]

        if self.verbose:
            t_map_size = int(np.sum(best_idx_y >= 0)) if best_idx_y is not None else len(t_map)
            print(f"  [Shielding] t_map size: {t_map_size}/{n_s}")

        
        t0 = time.perf_counter()
        knn_indices = level_s.knn_indices if level_s.knn_indices is not None else np.empty((n_s, 0), dtype=np.int32)
        if self.shield_impl == "halo":
            sentinels_list, shield_arr = _prepare_sentinels_for_numba_fast(
                nodes_X=level_s.points,
                nodes_Y=level_t.points,
                t_idx=best_idx_y if best_idx_y is not None else np.full(n_s, -1, dtype=np.int64),
                all_knn_indices=knn_indices,
                k_neighbors=self.k_neighbors,
            )
        else:
            sentinels_list, shield_arr = _build_sentinels(
                level_s, t_map, knn_indices, self.k_neighbors, level_Y=level_t
            )
        timing["sentinels"] = time.perf_counter() - t0
        timing[COMPONENT_SHIELD_SENTINELS] = timing["sentinels"]

        if self.verbose:
            print(f"    [Shielding] shield_arr={len(shield_arr)}")

        
        t0 = time.perf_counter()
        if hasattr(hierarchy_t, 'flat_centers'):
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
            print(f"    [Shielding] yhat_pairs={len(yhat_pairs)}")

        
        t0 = time.perf_counter()
        parts_1d = []

        
        if len(y_keep) > 0:
            parts_1d.append(y_keep.astype(np.int64))

        
        if len(shield_arr) > 0:
            shield_1d = shield_arr[:, 0].astype(np.int64) * n_t + shield_arr[:, 1].astype(np.int64)
            parts_1d.append(shield_1d)

        
        if len(yhat_pairs) > 0:
            yhat_1d = yhat_pairs[:, 0].astype(np.int64) * n_t + yhat_pairs[:, 1].astype(np.int64)
            parts_1d.append(yhat_1d)

        
        if parts_1d:
            all_1d = np.concatenate(parts_1d)
            keep_1d = np.unique(all_1d)
        else:
            keep_1d = np.empty(0, dtype=np.int64)
        timing["keep_union"] = time.perf_counter() - t0
        timing[COMPONENT_SHIELD_UNION] = timing["keep_union"]

        if self.verbose:
            print(f"    [Shielding] keep_1d={len(keep_1d)} ({len(keep_1d)/n_total:.2f}x n_total)")

        
        result = {
            'keep': keep_1d,
            'timing': timing,
        }
        if build_aux:
            from ..core.utils import decode_keep_1d_to_struct, remap_duals_for_warm_start

            result['keep_coord'] = decode_keep_1d_to_struct(keep_1d, n_t)
            result['y_init'] = remap_duals_for_warm_start(y_solution_last, keep_1d)
        return result

    def _fallback_yhat(self, level_s, level_t, t_map, knn_indices):
        
        n_s = len(level_s.points)
        pairs = []

        for i_A in range(n_s):
            if i_A < len(knn_indices):
                for i_S in knn_indices[i_A][:self.k_neighbors]:
                    if i_S >= 0 and i_S in t_map:
                        pairs.append((i_A, t_map[i_S]))

        if len(pairs) == 0:
            return np.empty((0, 2), dtype=np.int32)

        arr = np.array(pairs, dtype=np.int32)
        return np.unique(arr.view(f"V{8}")).view(np.int32).reshape(-1, 2)

    def close(self):
        pass
