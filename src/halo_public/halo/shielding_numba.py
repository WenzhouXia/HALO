import numpy as np
from numba import njit, prange
from numba.typed import List as NumbaList
import time
from typing import Tuple, List

@njit(cache=True)
def _dot_njit(a: np.ndarray, b: np.ndarray) -> float:
    
    s = 0.0
    for i in range(a.shape[0]):
        s += a[i] * b[i]
    return s

@njit(cache=True)
def _get_exact_inner_njit(
    rep_A: np.ndarray,
    rep_B: np.ndarray,
    candidates: np.ndarray  
) -> float:
    
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
    nodes_X, l0_points_all, sentinels_list,
    centers, radii, levels, is_leaf, public_idx,
    child_ptr, child_indices,
    leaf_point_ptr, leaf_points,
    root_indices,
    target_level_idx,
    max_pairs_per_xA=-1,
) -> np.ndarray:
    
    n_X, dim = nodes_X.shape
    counts = np.zeros(n_X, dtype=np.int64)
    max_stack_size = centers.shape[0] + root_indices.shape[0] + 8
    PRUNE_THRESHOLD = 0.0
    
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
                    if bound > PRUNE_THRESHOLD:
                        is_pruned = True
                        break
                
                if is_pruned:
                    continue
                
                node_level = int(levels[node_idx])
                leaf_flag = (is_leaf[node_idx] == 1)
                
                
                if node_level == target_level_idx and target_level_idx > 0:
                    rep_B = centers[node_idx, :]
                    best_inner = _get_exact_inner_njit(rep_A, rep_B, candidates)
                    if not (best_inner > PRUNE_THRESHOLD):
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
                        if not (best_inner > PRUNE_THRESHOLD):
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
    nodes_X, l0_points_all, sentinels_list,
    centers, radii, levels, is_leaf, public_idx,
    child_ptr, child_indices,
    leaf_point_ptr, leaf_points,
    root_indices,
    target_level_idx,
    offsets,
    pairs_all,
    max_pairs_per_xA=-1,
):
    
    n_X, dim = nodes_X.shape
    max_stack_size = centers.shape[0] + root_indices.shape[0] + 8
    PRUNE_THRESHOLD = 0.0
    
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
                    if bound > PRUNE_THRESHOLD:
                        is_pruned = True
                        break
                
                if is_pruned:
                    continue
                
                node_level = int(levels[node_idx])
                leaf_flag = (is_leaf[node_idx] == 1)
                
                
                if node_level == target_level_idx and target_level_idx > 0:
                    rep_B = centers[node_idx, :]
                    best_inner = _get_exact_inner_njit(rep_A, rep_B, candidates)
                    if not (best_inner > PRUNE_THRESHOLD):
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
                        if not (best_inner > PRUNE_THRESHOLD):
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
    
    t0 = time.perf_counter()
    
    n_X = nodes_X.shape[0]
    if n_X == 0:
        return np.empty((0, 2), dtype=np.int32)
    
    
    if not hasattr(hierarchy_Y, 'flat_centers'):
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
        centers, radii, levels, is_leaf, public_idx,
        child_ptr, child_indices,
        leaf_point_ptr, leaf_points,
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
        centers, radii, levels, is_leaf, public_idx,
        child_ptr, child_indices,
        leaf_point_ptr, leaf_points,
        root_indices,
        int(target_level_idx),
        offsets,
        pairs_all,
        max_pairs_per_xA,
    )
    
    
    pairs_arr = pairs_all.astype(np.int32, copy=False)
    pairs_cont = np.ascontiguousarray(pairs_arr)
    unique_pairs = np.unique(
        pairs_cont.view(f"V{pairs_cont.dtype.itemsize * 2}")
    ).view(pairs_cont.dtype).reshape(-1, 2)
    
    if verbose:
        print(f"  [TreeNumba] Found {len(unique_pairs)} unique pairs in {time.perf_counter()-t0:.3f}s")
    
    return unique_pairs
