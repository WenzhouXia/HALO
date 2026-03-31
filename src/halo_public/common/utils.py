

import time
import torch
import gc
import logging
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Optional, Tuple
import numpy as np
from numba import njit, prange
from scipy import sparse

logger = logging.getLogger(__name__)

def log_gpu_memory(tag: str = "") -> None:
    
    if not torch.cuda.is_available():
        return

    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    free_mem, total_mem = torch.cuda.mem_get_info()

    logger.info(f"[{tag}] GPU Mem: Alloc={allocated:.2f}GB | Rsrv={reserved:.2f}GB | "
                f"Max={max_allocated:.2f}GB | Free={free_mem/1024**3:.2f}GB")

def gpu_gc() -> None:
    
    gc.collect()
    torch.cuda.empty_cache()

@contextmanager
def profile_ctx(name: str, stats_dict: Dict[str, float], enabled: bool = True):
    
    if not enabled:
        yield
        return

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    yield

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    dt = time.perf_counter() - t0
    stats_dict[name] += dt

@njit(parallel=True, cache=True)
def _apply_parent_map(vec_fine, vec_coarse, parent_labels):
    
    n_fine = len(vec_fine)
    for i_fine in prange(n_fine):
        i_coarse = parent_labels[i_fine]
        if i_coarse >= 0:
            vec_fine[i_fine] = vec_coarse[i_coarse]

def prolongate_potentials(
    x_coarse: np.ndarray,
    level_fine_X,
    level_fine_Y,
    level_coarse_X,
    level_coarse_Y,
    dtype=np.float32
) -> np.ndarray:
    
    n_X_fine = len(level_fine_X.points)
    n_Y_fine = len(level_fine_Y.points)
    n_X_coarse = len(level_coarse_X.points)
    n_Y_coarse = len(level_coarse_Y.points)

    x_init_fine = np.zeros(n_X_fine + n_Y_fine, dtype=dtype)

    
    u_coarse = x_coarse[:n_X_coarse]
    v_coarse = x_coarse[n_X_coarse:]

    
    parent_labels_X = level_fine_X.parent_labels
    if parent_labels_X is None:
        raise ValueError(f"Level {level_fine_X.level_idx} (X) has no parent_labels")
    _apply_parent_map(x_init_fine[:n_X_fine], u_coarse, parent_labels_X)

    
    parent_labels_Y = level_fine_Y.parent_labels
    if parent_labels_Y is None:
        raise ValueError(f"Level {level_fine_Y.level_idx} (Y) has no parent_labels")
    _apply_parent_map(x_init_fine[n_X_fine:], v_coarse, parent_labels_Y)

    return x_init_fine

@njit(parallel=True, cache=True)
def _refine_expand_numba(
    idx_X_coarse: np.ndarray,
    idx_Y_coarse: np.ndarray,
    y_coarse_val: np.ndarray,
    children_X_indices: np.ndarray,
    children_X_offsets: np.ndarray,
    children_Y_indices: np.ndarray,
    children_Y_offsets: np.ndarray,
    n_Y_fine: int,
):
    
    Kc = idx_X_coarse.shape[0]
    
    
    block_sizes = np.empty(Kc, dtype=np.int64)
    for i in range(Kc):
        ix = idx_X_coarse[i]
        iy = idx_Y_coarse[i]
        x_start = children_X_offsets[ix]
        x_end = children_X_offsets[ix + 1]
        y_start = children_Y_offsets[iy]
        y_end = children_Y_offsets[iy + 1]
        m = x_end - x_start
        n = y_end - y_start
        block_sizes[i] = m * n
    
    
    offsets = np.empty(Kc + 1, dtype=np.int64)
    offsets[0] = 0
    for i in range(Kc):
        offsets[i + 1] = offsets[i] + block_sizes[i]
    
    total_Kf = offsets[Kc]
    
    
    y_init_y = np.empty(total_Kf, dtype=y_coarse_val.dtype)
    y_init_keep_1d = np.empty(total_Kf, dtype=np.int64)
    y_init_keep_tuples = np.empty((total_Kf, 2), dtype=np.int32)
    
    
    for i in prange(Kc):
        size = block_sizes[i]
        if size == 0:
            continue
        
        ix = idx_X_coarse[i]
        iy = idx_Y_coarse[i]
        yv = y_coarse_val[i]
        
        x_start = children_X_offsets[ix]
        x_end = children_X_offsets[ix + 1]
        y_start = children_Y_offsets[iy]
        y_end = children_Y_offsets[iy + 1]
        
        m = x_end - x_start
        n = y_end - y_start
        if m == 0 or n == 0:
            continue

        base = offsets[i]
        pos = base
        
        for a in range(m):
            xf = children_X_indices[x_start + a]
            for b in range(n):
                yf = children_Y_indices[y_start + b]

                y_init_keep_tuples[pos, 0] = xf
                y_init_keep_tuples[pos, 1] = yf
                
                y_init_y[pos] = yv
                y_init_keep_1d[pos] = (np.int64(xf) * np.int64(n_Y_fine) + np.int64(yf))
                pos += 1
    
    return y_init_y, y_init_keep_1d, y_init_keep_tuples

def refine_duals(
    y_solution_coarse: Dict,
    level_fine_X,
    level_fine_Y,
    level_coarse_X,
    level_coarse_Y,
    dtype=np.float32,
    thr: float = 1e-20,
    timing: bool = False,  
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    t_start = time.perf_counter()

    y_coarse_val_all = y_solution_coarse["y"]
    keep_coarse_1d_all = y_solution_coarse["keep"]

    
    nz_indices = np.where(np.abs(y_coarse_val_all) > thr)[0]
    y_coarse_val = y_coarse_val_all[nz_indices].astype(dtype, copy=False)
    keep_coarse_1d = keep_coarse_1d_all[nz_indices]

    K_coarse_nz = len(keep_coarse_1d)
    if K_coarse_nz == 0:
        return (
            np.array([], dtype=dtype),
            np.array([], dtype=np.int64),
            np.empty((0, 2), dtype=np.int32)
        )

    if timing:
        print(f"      [RefineDuals-hier] K_coarse_nz={K_coarse_nz}")

    
    n_Y_coarse = len(level_coarse_Y.points)
    idx_X_coarse = (keep_coarse_1d // n_Y_coarse).astype(np.int32)
    idx_Y_coarse = (keep_coarse_1d % n_Y_coarse).astype(np.int32)

    if timing:
        print(f"      [RefineDuals-hier] idx_X_coarse range: [{idx_X_coarse.min()}, {idx_X_coarse.max()}]")
        print(f"      [RefineDuals-hier] idx_Y_coarse range: [{idx_Y_coarse.min()}, {idx_Y_coarse.max()}]")
        print(f"      [RefineDuals-hier] level_coarse_X has {len(level_coarse_X.points)} nodes")
        print(f"      [RefineDuals-hier] level_coarse_Y has {len(level_coarse_Y.points)} nodes")

    
    children_X_indices = level_coarse_X.children_indices
    children_X_offsets = level_coarse_X.children_offsets
    children_Y_indices = level_coarse_Y.children_indices
    children_Y_offsets = level_coarse_Y.children_offsets

    if children_X_indices is None or children_Y_indices is None:
        raise ValueError("Level missing children CSR data")

    if timing:
        print(f"      [RefineDuals-hier] children_X_offsets length: {len(children_X_offsets)}")
        print(f"      [RefineDuals-hier] children_Y_offsets length: {len(children_Y_offsets)}")

    
    n_Y_fine = len(level_fine_Y.points)
    y_init_y, y_init_keep_1d, y_init_keep_tuples = _refine_expand_numba(
        idx_X_coarse,
        idx_Y_coarse,
        y_coarse_val,
        children_X_indices,
        children_X_offsets,
        children_Y_indices,
        children_Y_offsets,
        n_Y_fine,
    )

    if timing:
        t_end = time.perf_counter()
        print(f"      [RefineDuals-hier] result: K_fine={len(y_init_keep_1d)}, time={t_end-t_start:.3f}s")

    return y_init_y, y_init_keep_1d, y_init_keep_tuples

def decode_keep_1d_to_struct(keep_1d: np.ndarray, n_Y: int) -> np.ndarray:
    
    n_keep = len(keep_1d)
    keep_coord_struct = np.empty(n_keep, dtype=[('idx1', np.int64), ('idx2', np.int64)])
    
    if n_keep > 0:
        idx_x = (keep_1d // n_Y).astype(np.int64, copy=False)
        idx_y = (keep_1d % n_Y).astype(np.int64, copy=False)
        keep_coord_struct['idx1'] = idx_x
        keep_coord_struct['idx2'] = idx_y
    
    return keep_coord_struct

@njit(parallel=True, cache=True)
def _calculate_costs_l2(rhs, idx_x_all, idx_y_all, points_X, points_Y):
    
    n_keep = len(idx_x_all)
    dim = points_X.shape[1]
    
    for i in prange(n_keep):
        idx_x = idx_x_all[i]
        idx_y = idx_y_all[i]
        
        cost_sq = 0.0
        for d in range(dim):
            diff = points_X[idx_x, d] - points_Y[idx_y, d]
            cost_sq += diff * diff
        
        rhs[i] = -cost_sq

@njit(parallel=True, cache=True)
def _calculate_costs_l1(rhs, idx_x_all, idx_y_all, points_X, points_Y):
    
    n_keep = len(idx_x_all)
    dim = points_X.shape[1]
    
    for i in prange(n_keep):
        idx_x = idx_x_all[i]
        idx_y = idx_y_all[i]
        
        cost_1 = 0.0
        for d in range(dim):
            diff = points_X[idx_x, d] - points_Y[idx_y, d]
            cost_1 += abs(diff)
        
        rhs[i] = -cost_1

@njit(parallel=True, cache=True)
def _calculate_costs_linf(rhs, idx_x_all, idx_y_all, points_X, points_Y):
    
    n_keep = len(idx_x_all)
    dim = points_X.shape[1]
    
    for i in prange(n_keep):
        idx_x = idx_x_all[i]
        idx_y = idx_y_all[i]
        
        max_abs = 0.0
        for d in range(dim):
            diff = points_X[idx_x, d] - points_Y[idx_y, d]
            ad = abs(diff)
            if ad > max_abs:
                max_abs = ad
        
        rhs[i] = -max_abs

def generate_minus_c(
    keep_coord_struct: np.ndarray,
    nodes_X: np.ndarray,
    nodes_Y: np.ndarray,
    cost_type: str = "L2",
    p: float = 2.0
) -> np.ndarray:
    
    n_keep = len(keep_coord_struct)
    rhs = np.empty(n_keep, dtype=np.float64)
    
    idx_x_all = keep_coord_struct['idx1'].astype(np.int64, copy=False)
    idx_y_all = keep_coord_struct['idx2'].astype(np.int64, copy=False)
    
    
    cost_type_upper = cost_type.upper()
    
    if cost_type_upper in ("L2", "SQEUCLIDEAN"):
        _calculate_costs_l2(rhs, idx_x_all, idx_y_all, nodes_X, nodes_Y)
    elif cost_type_upper == "L1":
        _calculate_costs_l1(rhs, idx_x_all, idx_y_all, nodes_X, nodes_Y)
    elif cost_type_upper == "LINF":
        _calculate_costs_linf(rhs, idx_x_all, idx_y_all, nodes_X, nodes_Y)
    else:
        raise NotImplementedError(f"Unsupported cost_type={cost_type}")
    
    return rhs

def build_minus_AT_csc(
    keep_coord_struct: np.ndarray,
    n_nodes_X: int,
    n_nodes_Y: int,
    dtype=np.float64
) -> sparse.csc_matrix:
    
    nRows = len(keep_coord_struct)
    nCols = n_nodes_X + n_nodes_Y
    
    row_idx = np.repeat(np.arange(nRows, dtype=np.int64), 2)
    col_idx = np.empty(2 * nRows, dtype=np.int64)
    
    col_idx[0::2] = keep_coord_struct['idx1']
    col_idx[1::2] = keep_coord_struct['idx2'] + n_nodes_X
    
    data = np.full(2 * nRows, -1.0, dtype=dtype)
    
    return sparse.csc_matrix((data, (row_idx, col_idx)), shape=(nRows, nCols))

def remap_duals_for_warm_start(
    y_solution_last: Dict,
    keep_add_new: np.ndarray,
    dtype=np.float32
) -> np.ndarray:
    
    y_last_vals = np.asarray(y_solution_last['y'], dtype=dtype)
    y_last_keep = np.asarray(y_solution_last['keep'])
    keep_add_new = np.asarray(keep_add_new)
    
    if y_last_keep.size == 0 or keep_add_new.size == 0:
        return np.zeros(keep_add_new.shape[0], dtype=dtype)
    
    
    order = np.argsort(y_last_keep)
    keep_sorted = y_last_keep[order]
    vals_sorted = y_last_vals[order]
    
    pos = np.searchsorted(keep_sorted, keep_add_new)
    
    in_bound = (pos < keep_sorted.size)
    valid_mask = np.zeros(keep_add_new.shape[0], dtype=bool)
    
    if np.any(in_bound):
        valid_mask[in_bound] = (
            keep_sorted[pos[in_bound]] == keep_add_new[in_bound]
        )
    
    y_init_new = np.zeros(keep_add_new.shape[0], dtype=dtype)
    if np.any(valid_mask):
        y_init_new[valid_mask] = vals_sorted[pos[valid_mask]]
    
    return y_init_new
