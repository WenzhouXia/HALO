from ..halo.shielding import ShieldingStrategy
from ..mgpd.costs import grid_pairwise_cost
from .utils import (
    profile_ctx,
    prolongate_potentials,
    refine_duals,
    decode_keep_1d_to_struct,
    generate_minus_c,
    build_minus_AT_csc,
    remap_duals_for_warm_start,
)
from typing import Literal, Optional, Dict, Tuple, Any, Union
from scipy.spatial.distance import cdist
from scipy.sparse import csc_matrix, coo_matrix
import numba
from numba import types
try:
    import torch
except Exception:
    torch = None
import sys
import time
import numpy as np
import logging
from scipy import sparse
from typing import Tuple, Dict, Any, Optional, List
from .base import BaseHierarchy, BaseStrategy, ActiveSupport, ActiveSupportStrategy
from ..lp_solvers.wrapper import LPSolver
from ..mgpd.hierarchy import GridHierarchy
from ..halo.hierarchy import TreeHierarchy
from .solver_backend import SolverBackend
from ..mgpd.solver_grid import GridSolverMixin, GridSolverBackend
from ..halo.solver_tree import TreeSolverMixin, TreeSolverBackend
from .profiling import NoOpProfiler, RuntimeProfiler
from .profiling_names import (
    PHASE_ADVANCE_TO_NEXT_LEVEL,
    PHASE_CALLBACK,
    PHASE_FINALIZE_ITERATION,
    PHASE_INIT_LEVEL_STATE,
    PHASE_INIT_RUN_STATE,
    PHASE_PACKAGE_RESULT,
    PHASE_PREPARE_ITERATION_INPUT,
    PHASE_PROFILING_OUTPUT,
    PHASE_RECORD_LEVEL_RESULT,
    PHASE_SHOULD_STOP_ITERATION,
    PHASE_SOLVE_ITERATION_LP,
)

logger = logging.getLogger(__name__)

@numba.jit(types.Tuple((types.int32[:], types.int32[:]))(types.float64[:], types.float64[:]), nopython=True, cache=True)
def _northwest_corner_numba(p_in, q_in):
    
    
    p = p_in.copy()
    q = q_in.copy()

    m = p.shape[0]
    n = q.shape[0]

    
    
    max_edges = m + n + 1
    rows = np.empty(max_edges, dtype=np.int32)
    cols = np.empty(max_edges, dtype=np.int32)

    i = 0
    j = 0
    count = 0

    
    tol = 1e-20

    while i < m and j < n:
        
        rows[count] = i
        cols[count] = j
        count += 1

        val = min(p[i], q[j])
        p[i] -= val
        q[j] -= val

        
        p_empty = p[i] < tol
        q_empty = q[j] < tol

        if p_empty and q_empty:
            
            
            
            i += 1
            
            
        elif p_empty:
            i += 1
        else:
            j += 1

    
    return rows[:count], cols[:count]

fill_x_prev_sig = types.float32[:](
    types.int32[:],  
    types.int32[:],  
    types.float32[:],  
    types.int64[:],  
    types.int64[:],  
    types.float32[:],  
    types.int64     
)

@numba.jit(fill_x_prev_sig, nopython=True, fastmath=True, parallel=True, cache=True)
def _fill_x_prev_numba(primal_rows, primal_cols, primal_data,
                       sorted_inc_keys, sorted_inc_pos, x_prev_filled, num_cols):
    
    num_primal = len(primal_rows)
    
    for i in numba.prange(num_primal):
        
        key_to_find = np.int64(primal_rows[i]) * num_cols + primal_cols[i]

        
        index = np.searchsorted(sorted_inc_keys, key_to_find)

        
        if index < len(sorted_inc_keys) and sorted_inc_keys[index] == key_to_find:
            
            pos_to_update = sorted_inc_pos[index]
            value_to_assign = primal_data[i]
            x_prev_filled[pos_to_update] = value_to_assign

    return x_prev_filled

_expand_kernel_sig = types.void(
    types.int32[:],   
    types.int32[:],   
    types.float64[:],  
    types.int32[:],   
    types.int32[:],   
    types.float64[:],  
    types.int32[:],   
    types.int32[:],   
    types.float64[:],  
    types.int64[:],   
    types.int32[:],   
    types.int32[:],   
    types.float32[:]  
)

@numba.njit(_expand_kernel_sig, parallel=True, cache=True)
def _numba_expand_kernel(
    coarse_rows, coarse_cols, coarse_data,
    s_ptr, s_indices, s_weights,
    t_ptr, t_indices, t_weights,
    edge_offsets,
    out_rows, out_cols, out_data
):
    n_edges = len(coarse_data)
    for k in numba.prange(n_edges):
        c_r = coarse_rows[k]
        c_c = coarse_cols[k]
        val = coarse_data[k]

        s_start, s_end = s_ptr[c_r], s_ptr[c_r+1]
        t_start, t_end = t_ptr[c_c], t_ptr[c_c+1]

        write_idx = edge_offsets[k]

        for i in range(s_start, s_end):
            f_r = s_indices[i]
            w_r = s_weights[i]
            for j in range(t_start, t_end):
                f_c = t_indices[j]
                w_c = t_weights[j]

                out_rows[write_idx] = f_r
                out_cols[write_idx] = f_c
                out_data[write_idx] = val * w_r * w_c
                write_idx += 1

@numba.njit(types.int64[:](types.int32[:], types.int32[:], types.int64[:], types.int64[:]), cache=True)
def _precalc_offsets(coarse_rows, coarse_cols, s_counts, t_counts):
    n = len(coarse_rows)
    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = 0
    total = 0
    for i in range(n):
        
        size = s_counts[coarse_rows[i]] * t_counts[coarse_cols[i]]
        offsets[i] = total
        total += size
    offsets[n] = total
    return offsets

def _to_pinned_numpy(array: np.ndarray) -> np.ndarray:
    
    
    if torch is None:
        return np.asarray(array, dtype=np.float32, order="C")
    tensor = torch.empty(array.shape, dtype=torch.float32, pin_memory=True)
    tensor.copy_(torch.from_numpy(array))
    return tensor.numpy()

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
    raise ValueError(f"Unknown cost_type: {cost_type}")

def _prepare_level_cost_cache_sqeuclidean(points_s: np.ndarray, points_t: np.ndarray) -> Dict[str, Any]:
    S = np.asarray(points_s, dtype=np.float32, order='C')
    T = np.asarray(points_t, dtype=np.float32, order='C')
    s2 = np.einsum('ij,ij->i', S, S)
    t2 = np.einsum('ij,ij->i', T, T)
    return {'S': S, 'T': T, 's2': s2, 't2': t2, 'cost_type': 'l2^2'}

def _prepare_level_cost_cache_euclidean(points_s: np.ndarray, points_t: np.ndarray) -> Dict[str, Any]:
    
    S = np.asarray(points_s, dtype=np.float32, order='C')
    T = np.asarray(points_t, dtype=np.float32, order='C')
    return {'S': S, 'T': T, 'cost_type': 'l2'}

def _prepare_level_cost_cache_lowrank(F: np.ndarray, G: np.ndarray, a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    F_mat = _to_pinned_numpy(F)
    G_mat = _to_pinned_numpy(G)
    a_vec = _to_pinned_numpy(a)
    b_vec = _to_pinned_numpy(b)

    
    return {'F': F_mat, 'G': G_mat, 'a': a_vec, 'b': b_vec, 'cost_type': 'lowrank'}

def _prepare_level_cost_cache_l1(points_s: np.ndarray, points_t: np.ndarray) -> Dict[str, Any]:
    
    S = np.asarray(points_s, dtype=np.float32, order='C')
    T = np.asarray(points_t, dtype=np.float32, order='C')
    return {'S': S, 'T': T, 'cost_type': 'l1'}

def _prepare_level_cost_cache_linf(points_s: np.ndarray, points_t: np.ndarray) -> Dict[str, Any]:
    
    S = np.asarray(points_s, dtype=np.float32, order='C')
    T = np.asarray(points_t, dtype=np.float32, order='C')
    return {'S': S, 'T': T, 'cost_type': 'linf'}

def _cost_pairs_l1_batched(
    level_cache,
    rows: np.ndarray,
    cols: np.ndarray,
    batch_size: int = 65536,
    free_after: bool = True
) -> np.ndarray:
    
    S, T = level_cache['S'], level_cache['T']
    n = int(rows.shape[0])
    if n == 0:
        return np.array([], dtype=S.dtype)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = int(S.shape[1])
    if device.type == 'cpu':
        
        return np.sum(np.abs(S[rows] - T[cols]), axis=1).astype(S.dtype, copy=False)

    
    S_gpu = torch.from_numpy(np.ascontiguousarray(S)).to(
        device=device, dtype=torch.float32, non_blocking=False)
    T_gpu = torch.from_numpy(np.ascontiguousarray(T)).to(
        device=device, dtype=torch.float32, non_blocking=False)

    out = np.empty(n, dtype=np.float32)
    i = 0
    while i < n:
        j = min(i + batch_size, n)
        rows_b = rows[i:j]
        cols_b = cols[i:j]

        rows_t = torch.from_numpy(rows_b.astype(np.int64, copy=False)).to(
            device=device, dtype=torch.long, non_blocking=False)
        cols_t = torch.from_numpy(cols_b.astype(np.int64, copy=False)).to(
            device=device, dtype=torch.long, non_blocking=False)

        Si = S_gpu.index_select(0, rows_t)  
        Tj = T_gpu.index_select(0, cols_t)  
        l1_dist = torch.sum(torch.abs(Si - Tj), dim=1)  
        
        out[i:j] = l1_dist.detach().cpu().numpy()
        i = j
        del rows_t, cols_t, Si, Tj, l1_dist

    if free_after and device.type == 'cuda':
        del S_gpu, T_gpu
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return out.astype(S.dtype, copy=False)

def _cost_pairs_linf_batched(
    level_cache,
    rows: np.ndarray,
    cols: np.ndarray,
    batch_size: int = 65536,
    free_after: bool = True
) -> np.ndarray:
    
    S, T = level_cache['S'], level_cache['T']
    n = int(rows.shape[0])
    if n == 0:
        return np.array([], dtype=S.dtype)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = int(S.shape[1])
    if device.type == 'cpu':
        
        return np.max(np.abs(S[rows] - T[cols]), axis=1).astype(S.dtype, copy=False)

    
    S_gpu = torch.from_numpy(np.ascontiguousarray(S)).to(
        device=device, dtype=torch.float32, non_blocking=False)
    T_gpu = torch.from_numpy(np.ascontiguousarray(T)).to(
        device=device, dtype=torch.float32, non_blocking=False)

    out = np.empty(n, dtype=np.float32)
    i = 0
    while i < n:
        j = min(i + batch_size, n)
        rows_b = rows[i:j]
        cols_b = cols[i:j]

        rows_t = torch.from_numpy(rows_b.astype(np.int64, copy=False)).to(
            device=device, dtype=torch.long, non_blocking=False)
        cols_t = torch.from_numpy(cols_b.astype(np.int64, copy=False)).to(
            device=device, dtype=torch.long, non_blocking=False)

        Si = S_gpu.index_select(0, rows_t)  
        Tj = T_gpu.index_select(0, cols_t)  
        linf_dist = torch.max(torch.abs(Si - Tj), dim=1)[0]  
        
        out[i:j] = linf_dist.detach().cpu().numpy()
        i = j
        del rows_t, cols_t, Si, Tj, linf_dist

    if free_after and device.type == 'cuda':
        del S_gpu, T_gpu
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return out.astype(S.dtype, copy=False)

def _cost_pairs_lowrank_batched(
    level_cache,
    rows: np.ndarray,
    cols: np.ndarray,
    batch_size: int = 65536,
    d_chunk: int = None,
    free_after: bool = True
) -> np.ndarray:
    
    F, G, a, b = level_cache['F'], level_cache['G'], level_cache['a'], level_cache['b']
    n = int(rows.shape[0])
    if n == 0:
        return np.array([], dtype=F.dtype)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    d = int(F.shape[1])
    if device.type == 'cpu':
        c_add = a[rows] + b[cols]
        dots = np.einsum('nd,nd->n', F[rows], G[cols])
        return (c_add - dots).astype(F.dtype, copy=False)

    
    F_gpu = torch.from_numpy(np.ascontiguousarray(F)).to(
        device=device, dtype=torch.float32, non_blocking=False)
    G_gpu = torch.from_numpy(np.ascontiguousarray(G)).to(
        device=device, dtype=torch.float32, non_blocking=False)
    a_gpu = torch.from_numpy(np.ascontiguousarray(a)).to(
        device=device, dtype=torch.float32, non_blocking=False)
    b_gpu = torch.from_numpy(np.ascontiguousarray(b)).to(
        device=device, dtype=torch.float32, non_blocking=False)

    
    out = np.empty(n, dtype=np.float32)
    i = 0
    while i < n:
        j = min(i + batch_size, n)
        rows_b = rows[i:j]
        cols_b = cols[i:j]

        rows_t = torch.from_numpy(rows_b.astype(np.int64, copy=False)).to(
            device=device, dtype=torch.long, non_blocking=False)
        cols_t = torch.from_numpy(cols_b.astype(np.int64, copy=False)).to(
            device=device, dtype=torch.long, non_blocking=False)

        
        if d_chunk is None or d_chunk >= d:
            Fi = F_gpu.index_select(0, rows_t)        
            Gj = G_gpu.index_select(0, cols_t)        
            dots = (Fi * Gj).sum(dim=1)               
        else:
            B = rows_t.shape[0]
            acc = torch.zeros(B, dtype=torch.float32, device=device)
            for k in range(0, d, d_chunk):
                kk = min(k + d_chunk, d)
                Fi = F_gpu.index_select(0, rows_t)[:, k:kk]  
                Gj = G_gpu.index_select(0, cols_t)[:, k:kk]  
                acc += (Fi * Gj).sum(dim=1)
            dots = acc

        c_add = a_gpu.index_select(0, rows_t) +            b_gpu.index_select(0, cols_t) - dots
        out[i:j] = c_add.detach().cpu().numpy()
        i = j

    
    if free_after and device.type == 'cuda':
        del F_gpu, G_gpu, a_gpu, b_gpu, rows_t, cols_t
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return out.astype(F.dtype, copy=False)

def _cost_pairs_sqeuclidean_batched(
    level_cache,
    rows: np.ndarray,
    cols: np.ndarray,
    batch_size: int = 65536,
    d_chunk: int = None,
    free_after: bool = True
) -> np.ndarray:
    
    S, T, s2, t2 = level_cache['S'], level_cache['T'], level_cache['s2'], level_cache['t2']
    n = int(rows.shape[0])
    if n == 0:
        return np.array([], dtype=S.dtype)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = int(S.shape[1])
    if device.type == 'cpu':
        c_add = s2[rows] + t2[cols]
        dots = np.einsum('nd,nd->n', S[rows], T[cols])
        return (c_add - 2.0 * dots).astype(S.dtype, copy=False)

    
    S_gpu = torch.from_numpy(np.ascontiguousarray(S)).to(
        device=device, dtype=torch.float32, non_blocking=False)
    T_gpu = torch.from_numpy(np.ascontiguousarray(T)).to(
        device=device, dtype=torch.float32, non_blocking=False)
    s2_gpu = torch.from_numpy(np.ascontiguousarray(s2)).to(
        device=device, dtype=torch.float32, non_blocking=False)
    t2_gpu = torch.from_numpy(np.ascontiguousarray(t2)).to(
        device=device, dtype=torch.float32, non_blocking=False)

    
    out = np.empty(n, dtype=np.float32)
    i = 0
    while i < n:
        j = min(i + batch_size, n)
        rows_b = rows[i:j]
        cols_b = cols[i:j]

        rows_t = torch.from_numpy(rows_b.astype(np.int64, copy=False)).to(
            device=device, dtype=torch.long, non_blocking=False)
        cols_t = torch.from_numpy(cols_b.astype(np.int64, copy=False)).to(
            device=device, dtype=torch.long, non_blocking=False)

        if d_chunk is None or d_chunk >= d:
            Si = S_gpu.index_select(0, rows_t)        
            Tj = T_gpu.index_select(0, cols_t)        
            dots = (Si * Tj).sum(dim=1)               
        else:
            B = rows_t.shape[0]
            acc = torch.zeros(B, dtype=torch.float32, device=device)
            for k in range(0, d, d_chunk):
                kk = min(k + d_chunk, d)
                Si = S_gpu.index_select(0, rows_t)[:, k:kk]
                Tj = T_gpu.index_select(0, cols_t)[:, k:kk]
                acc += (Si * Tj).sum(dim=1)
            dots = acc

        c_add = s2_gpu.index_select(0, rows_t) +            t2_gpu.index_select(0, cols_t) - 2.0 * dots
        out[i:j] = c_add.detach().cpu().numpy()
        i = j

    if free_after and device.type == 'cuda':
        del S_gpu, T_gpu, s2_gpu, t2_gpu, rows_t, cols_t
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return out.astype(S.dtype, copy=False)

def _cost_pairs_euclidean_batched(
    level_cache,
    rows: np.ndarray,
    cols: np.ndarray,
    batch_size: int = 65536,
    free_after: bool = True
) -> np.ndarray:
    
    S, T = level_cache['S'], level_cache['T']
    n = int(rows.shape[0])
    if n == 0:
        return np.array([], dtype=S.dtype)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        diff = S[rows] - T[cols]
        return np.sqrt(np.sum(diff * diff, axis=1)).astype(S.dtype, copy=False)

    S_gpu = torch.from_numpy(np.ascontiguousarray(S)).to(
        device=device, dtype=torch.float32, non_blocking=False)
    T_gpu = torch.from_numpy(np.ascontiguousarray(T)).to(
        device=device, dtype=torch.float32, non_blocking=False)

    out = np.empty(n, dtype=np.float32)
    i = 0
    while i < n:
        j = min(i + batch_size, n)
        rows_b = rows[i:j]
        cols_b = cols[i:j]

        rows_t = torch.from_numpy(rows_b.astype(np.int64, copy=False)).to(
            device=device, dtype=torch.long, non_blocking=False)
        cols_t = torch.from_numpy(cols_b.astype(np.int64, copy=False)).to(
            device=device, dtype=torch.long, non_blocking=False)

        Si = S_gpu.index_select(0, rows_t)
        Tj = T_gpu.index_select(0, cols_t)
        sq = torch.sum((Si - Tj) * (Si - Tj), dim=1)
        c_add = torch.sqrt(torch.clamp(sq, min=0.0) + 1e-12)
        out[i:j] = c_add.detach().cpu().numpy()
        i = j
        del rows_t, cols_t, Si, Tj, sq, c_add

    if free_after and device.type == 'cuda':
        del S_gpu, T_gpu
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return out.astype(S.dtype, copy=False)

class HierarchicalOTSolver(TreeSolverMixin, GridSolverMixin):
    def __init__(
        self,
        hierarchy_s: BaseHierarchy,
        hierarchy_t: BaseHierarchy,
        strategy: BaseStrategy,
        solver: LPSolver,
        lp_solver_verbose: int = 0,
    ):
        self.hierarchy_s = hierarchy_s
        self.hierarchy_t = hierarchy_t
        self.solver = solver
        self.strategy = strategy
        self.lp_solver_verbose = lp_solver_verbose

        self.solutions: Dict[int, Dict[str, Any]] = {}
        self.active_support: Optional[ActiveSupport] = None

        self.build_time = hierarchy_s.build_time + hierarchy_t.build_time
        
        self.check_type = "auto"
        self.tree_lp_form = "primal"
        self.use_last = True
        self.use_last_after_inner0 = False
        self.ifcheck = True
        self.vd_thr = 0.25
        self.tree_debug = False
        self.tree_infeas_fallback = "none"
        self.tree_infeas_use_cupy = True
        self.nnz_thr = 1e-20
        self.check_sampled_config = None
        self._profiler = NoOpProfiler()
        self._profiling_enabled = False
        self._solve_wall_time = 0.0
        self._grid_p = 2
        self._lp_solver_kwargs: Dict[str, Any] = {}
        self._runtime_logging: Dict[str, Any] = {
            "enabled": True,
            "progress": True,
            "profile_iter": True,
            "profile_level": True,
            "profile_run": True,
            "warm_start": True,
            "iter_interval": 10,
        }

    def _runtime_log_enabled(self, category: str) -> bool:
        cfg = getattr(self, "_runtime_logging", None)
        if not isinstance(cfg, dict):
            return True
        if not bool(cfg.get("enabled", True)):
            return False
        return bool(cfg.get(category, True))

    def _runtime_log(self, category: str, message: str, *, flush: bool = False) -> None:
        if self._runtime_log_enabled(category):
            print(message, flush=flush)

    @staticmethod
    def _format_profile_components(
        components: Dict[str, float],
        max_items: int = 6,
        base_total: Optional[float] = None,
    ) -> str:
        if not components:
            return "-"
        items = [(k, float(v)) for k, v in components.items() if float(v) > 0.0]
        if not items:
            return "-"
        items.sort(key=lambda kv: kv[1], reverse=True)
        if base_total is None:
            base_total = sum(v for _, v in items)
        base = max(float(base_total), 1e-12)
        return ", ".join(f"{k}={v:.2f}s({(v/base)*100.0:.1f}%)" for k, v in items[:max_items])

    @staticmethod
    def _fmt_optional_sci(value: Any, digits: int = 3) -> str:
        if value is None:
            return "n/a"
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "n/a"
        if not np.isfinite(v):
            return "n/a"
        return f"{v:.{digits}e}"

    @staticmethod
    def _fmt_optional_fixed(value: Any, digits: int = 6) -> str:
        if value is None:
            return "n/a"
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "n/a"
        if not np.isfinite(v):
            return "n/a"
        return f"{v:.{digits}f}"

    def _extract_lp_profile_summary(self, step_pack: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(step_pack, dict):
            return None
        if not bool(step_pack.get("success", False)):
            return None

        if step_pack.get("res") is not None:
            res = step_pack["res"]
            return {
                "time": getattr(res, "duration", None),
                "iters": getattr(res, "iterations", None),
                "primal_obj": getattr(res, "obj_val", None),
                "dual_obj": getattr(res, "dual_obj_val", None),
                "primal_feas": getattr(res, "primal_feas", None),
                "dual_feas": getattr(res, "dual_feas", None),
                "gap": getattr(res, "gap", None),
            }

        lp_result = step_pack.get("lp_result")
        if isinstance(lp_result, dict):
            diag = lp_result.get("diag", {}) if isinstance(lp_result.get("diag"), dict) else {}
            return {
                "time": diag.get("backend_solve_wall_time"),
                "iters": lp_result.get("iters"),
                "primal_obj": lp_result.get("obj"),
                "dual_obj": lp_result.get("dual_obj"),
                "primal_feas": lp_result.get("primal_feas"),
                "dual_feas": lp_result.get("dual_feas"),
                "gap": lp_result.get("gap"),
            }
        return None

    def _print_profile_lp(
        self,
        level_idx: int,
        inner_iter: int,
        step_pack: Optional[Dict[str, Any]],
    ) -> None:
        if not self._profiling_enabled or not self._runtime_log_enabled("profile_iter"):
            return
        summary = self._extract_lp_profile_summary(step_pack)
        if not summary:
            return
        self._runtime_log(
            "profile_iter",
            f"[Profile][L{level_idx}][I{inner_iter + 1}][lp] "
            f"time={self._fmt_optional_fixed(summary.get('time'), digits=2)}s, "
            f"iter={summary.get('iters', 'n/a')}, "
            f"obj={self._fmt_optional_fixed(summary.get('primal_obj'))} / {self._fmt_optional_fixed(summary.get('dual_obj'))}, "
            f"RelRes={self._fmt_optional_sci(summary.get('primal_feas'))} / {self._fmt_optional_sci(summary.get('dual_feas'))}, "
            f"Gap={self._fmt_optional_sci(summary.get('gap'), digits=4)}"
        )

    def _print_profile_pricing(
        self,
        level_idx: int,
        inner_iter: int,
        step_pack: Optional[Dict[str, Any]],
    ) -> None:
        if not self._profiling_enabled or not self._runtime_log_enabled("profile_iter"):
            return
        if not isinstance(step_pack, dict):
            return
        info = step_pack.get("pricing_info")
        if not isinstance(info, dict):
            return
        self._runtime_log(
            "profile_iter",
            f"[Profile][L{level_idx}][I{inner_iter + 1}][pricing] "
            f"time={self._fmt_optional_fixed(info.get('time'), digits=2)}s, "
            f"active={int(info.get('active_before', 0)):,}, "
            f"found={int(info.get('found', 0)):,}, "
            f"added={int(info.get('added', 0)):,}"
        )

    def _print_profile_convergence_check(
        self,
        level_idx: int,
        inner_iter: int,
        step_pack: Optional[Dict[str, Any]],
    ) -> None:
        if not self._profiling_enabled or not self._runtime_log_enabled("profile_iter"):
            return
        if not isinstance(step_pack, dict):
            return
        info = step_pack.get("convergence_info")
        if not isinstance(info, dict):
            return
        criterion = info.get("criterion", "n/a")
        plateau_counter = info.get("plateau_counter", "n/a")
        required_plateau = info.get("required_plateau", "n/a")
        objective_tol = self._fmt_optional_sci(info.get("objective_tol"))
        self._runtime_log(
            "profile_iter",
            f"[Profile][L{level_idx}][I{inner_iter + 1}][convergence_check] "
            f"signed_rel_obj_change={self._fmt_optional_sci(info.get('signed_rel_obj_change'))}, "
            f"converged={bool(info.get('is_converged', False))}, "
            f"criterion={criterion}, "
            f"plateau={plateau_counter}/{required_plateau}, "
            f"tol={objective_tol}"
        )

    def _print_profile_inner(
        self,
        level_idx: int,
        inner_iter: int,
        iter_profile: Optional[Dict[str, Any]],
        step_pack: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._profiling_enabled or not self._runtime_log_enabled("profile_iter"):
            return
        if not iter_profile:
            return
        iter_wall = float(iter_profile.get("time", 0.0))
        phase_times = iter_profile.get("phase_times", {})
        if not isinstance(phase_times, dict):
            phase_times = {}
        phases = iter_profile.get("phases", {})
        if not isinstance(phases, dict):
            phases = {}
        solve_t = float(phase_times.get(PHASE_SOLVE_ITERATION_LP, 0.0))
        finalize_t = float(phase_times.get(PHASE_FINALIZE_ITERATION, 0.0))
        finalize_components = phases.get(PHASE_FINALIZE_ITERATION, {})
        if isinstance(finalize_components, dict):
            finalize_components = finalize_components.get("components", {})
        else:
            finalize_components = {}
        base = max(iter_wall, 1e-12)
        else_t = max(iter_wall - solve_t - finalize_t, 0.0)
        finalize_component_total = sum(float(v) for v in finalize_components.values() if float(v) > 0.0)
        finalize_component_base = max(finalize_component_total, 1e-12)
        self._print_profile_lp(level_idx, inner_iter, step_pack)
        self._print_profile_pricing(level_idx, inner_iter, step_pack)
        self._print_profile_convergence_check(level_idx, inner_iter, step_pack)
        self._runtime_log(
            "profile_iter",
            f"[Profile][L{level_idx}][I{inner_iter + 1}][finalize] "
            f"{self._format_profile_components(finalize_components, max_items=4, base_total=finalize_component_base)}"
        )
        self._runtime_log(
            "profile_iter",
            f"[Profile][L{level_idx}][I{inner_iter + 1}] "
            f"total={iter_wall:.2f}s "
            f"solve={solve_t:.2f}s({(solve_t/base)*100.0:.1f}%) "
            f"finalize={finalize_t:.2f}s({(finalize_t/base)*100.0:.1f}%) "
            f"else={else_t:.2f}s({(else_t/base)*100.0:.1f}%)"
        )
        self._runtime_log("profile_iter", "---")

    def _print_profile_level(
        self,
        level_idx: int,
        level_state: Dict[str, Any],
        level_profile: Optional[Dict[str, Any]],
    ) -> None:
        if not self._profiling_enabled or not self._runtime_log_enabled("profile_level"):
            return
        if not level_profile:
            return
        level_wall = float(level_profile.get("time", 0.0))
        phase_times = level_profile.get("phase_times", {})
        if not isinstance(phase_times, dict):
            phase_times = {}
        iters = int(level_profile.get("num_iters", 0))
        if iters <= 0:
            iters = int(level_state.get("iters_done", level_state.get("inner_iter", 0)))
        active = level_state.get("curr_active_size")
        active_text = f" active={int(active)}" if active is not None else ""
        lp_time = float(level_state.get("level_lp_time", 0.0))
        pricing_time = float(level_state.get("level_pricing_time", 0.0))
        base = max(level_wall, 1e-12)
        else_t = max(level_wall - lp_time - pricing_time, 0.0)
        self._runtime_log("profile_level", f"=== Summary level {level_idx} ===")
        self._runtime_log(
            "profile_level",
            f"[Profile][L{level_idx}] total={level_wall:.2f}s iters={iters} "
            f"lp={lp_time:.2f}s({(lp_time/base)*100.0:.1f}%) "
            f"pricing={pricing_time:.2f}s({(pricing_time/base)*100.0:.1f}%) "
            f"else={else_t:.2f}s({(else_t/base)*100.0:.1f}%){active_text}"
        )

    def _print_profile_init_level(
        self,
        level_idx: int,
        components: Dict[str, float],
        *,
        primal_nnz: int,
        active_size: int,
        bfs_added: int = 0,
    ) -> None:
        if not self._runtime_log_enabled("profile_level"):
            return
        total = sum(float(v) for v in components.values() if float(v) > 0.0)
        base = max(total, 1e-12)
        self._runtime_log(
            "profile_level",
            f"[Profile][L{level_idx}][Init] total={total:.2f}s "
            f"primal_nnz={int(primal_nnz):,} active={int(active_size):,} bfs_added={int(bfs_added):,}"
        )
        self._runtime_log(
            "profile_level",
            f"[Profile][L{level_idx}][Init][components] "
            f"{self._format_profile_components(components, max_items=6, base_total=base)}"
        )
        self._runtime_log("profile_level", "---")

    def _print_profile_run(
        self,
        profiling: Optional[Dict[str, Any]],
        *,
        include_level_breakdown: bool = True,
    ) -> None:
        if not self._profiling_enabled or not profiling or not self._runtime_log_enabled("profile_run"):
            return
        solve_time = float(profiling.get("solve_time", 0.0))
        build_time = float(profiling.get("build_time", 0.0))
        total_time = float(profiling.get("total_time", 0.0))
        run_components = profiling.get("components", {})
        if not isinstance(run_components, dict):
            run_components = {}
        lp_time = float(run_components.get("lp", 0.0))
        pricing_time = float(run_components.get("pricing", 0.0))
        else_t = max(solve_time - lp_time - pricing_time, 0.0)
        solve_base = max(float(solve_time), 1e-12)
        self._runtime_log(
            "profile_run",
            f"[Profile][Run] total={total_time:.2f}s solve={solve_time:.2f}s build={build_time:.2f}s | "
            f"lp={lp_time:.2f}s({(lp_time/solve_base)*100.0:.1f}%), "
            f"pricing={pricing_time:.2f}s({(pricing_time/solve_base)*100.0:.1f}%), "
            f"else={else_t:.2f}s({(else_t/solve_base)*100.0:.1f}%)"
        )
        if not include_level_breakdown:
            return
        self._print_profile_run_level_breakdown(profiling)

    def _print_profile_run_level_breakdown(self, profiling: Optional[Dict[str, Any]]) -> None:
        if not self._profiling_enabled or not profiling or not self._runtime_log_enabled("profile_run"):
            return
        run = profiling.get("run", {})
        if not isinstance(run, dict):
            run = {}
        levels = run.get("levels", profiling.get("levels", []))
        if not isinstance(levels, list) or not levels:
            return
        self._runtime_log("profile_run", "=== Overall Breakdown ===")
        for row in levels:
            if not isinstance(row, dict):
                continue
            level = int(row.get("level_idx", row.get("level", -1)))
            iters = int(row.get("num_iters", row.get("iters", 0)))
            t_level = float(row.get("time", 0.0))
            components = row.get("components", {})
            if not isinstance(components, dict):
                components = {}
            level_base = max(t_level, 1e-12)
            level_lp = float(components.get("lp", 0.0))
            level_pricing = float(components.get("pricing", 0.0))
            level_else = max(t_level - level_lp - level_pricing, 0.0)
            self._runtime_log(
                "profile_run",
                f"[Profile][Run][L{level}] total={t_level:.2f}s iters={iters} "
                f"(lp={(level_lp/level_base)*100.0:.1f}%, "
                f"pricing={(level_pricing/level_base)*100.0:.1f}%, "
                f"else={(level_else/level_base)*100.0:.1f}%)"
            )

    def solve(
        self,
        max_inner_iter: int = 100,
        convergence_criterion: Literal['strict', 'objective',
                                       'objective_and_violation'] = 'objective',
        objective_plateau_iters: int = 1,
        tolerance: Dict[str, float] = {
            'objective': 1e-6, 'primal': 1e-6, 'dual': 1e-6},
        final_refinement_tolerance: Optional[Dict[str, float]] = {
            'objective': 1e-8, 'primal': 1e-8, 'dual': 1e-8},
        cost_type: str = 'l2^2',  
        use_bfs_skeleton: bool = True,
        use_faiss_backend: bool = True,
        enable_profiling: bool = True,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        
        cost_type = _normalize_cost_type_name(cost_type)
        
        self._cost_type = cost_type
        self._profiling_enabled = bool(enable_profiling)
        self._profiler = RuntimeProfiler() if self._profiling_enabled else NoOpProfiler()
        self._inner_iteration_callback = kwargs.get("inner_iteration_callback")
        self._grid_p = int(kwargs.get("grid_p", getattr(self, "_grid_p", 2)))
        self._lp_solver_kwargs = dict(kwargs.get("lp_solver_kwargs", {}))
        self._objective_plateau_iters = int(objective_plateau_iters)
        if self._objective_plateau_iters <= 0:
            raise ValueError("objective_plateau_iters must be > 0")
        runtime_logging = kwargs.get("runtime_logging")
        if isinstance(runtime_logging, dict):
            self._runtime_logging = dict(runtime_logging)
        self._profiler.start_run()

        return self._solve_hierarchical(
            max_inner_iter=max_inner_iter,
            convergence_criterion=convergence_criterion,
            tolerance=tolerance,
            final_refinement_tolerance=final_refinement_tolerance,
            cost_type=cost_type,
            use_bfs_skeleton=use_bfs_skeleton,
            **kwargs,
        )

    def _emit_inner_iteration_callback(
        self,
        *,
        backend: SolverBackend,
        level_state: Dict[str, Any],
        step_pack: Dict[str, Any],
    ) -> None:
        callback = getattr(self, "_inner_iteration_callback", None)
        if callback is None:
            return

        objective = backend.extract_step_objective(step_pack)
        if objective is None:
            return

        callback(
            {
                "mode": backend.name,
                "level_idx": int(level_state.get("level_idx", -1)),
                "inner_iter": int(level_state.get("inner_iter", -1)),
                "objective": float(objective),
                "is_coarsest": bool(level_state.get("is_coarsest", False)),
            }
        )

    def _select_backend(self, **kwargs: Any) -> SolverBackend:
        if isinstance(self.hierarchy_s, GridHierarchy):
            default_mode = "grid"
        elif isinstance(self.hierarchy_s, TreeHierarchy):
            default_mode = "tree"
        else:
            raise ValueError("Unsupported hierarchy type for public solver.")
        mode = str(kwargs.get("mode", default_mode)).lower()
        if mode == "tree":
            return TreeSolverBackend(self)
        if mode == "grid":
            return GridSolverBackend(self)
        raise ValueError(f"Unknown mode: {mode}. Supported modes: tree, grid.")

    def _solve_hierarchical(
        self,
        max_inner_iter: int,
        convergence_criterion: str,
        tolerance: Dict[str, float],
        final_refinement_tolerance: Optional[Dict[str, float]],
        cost_type: str,
        use_bfs_skeleton: bool,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        t_solve_start = time.perf_counter()
        backend = self._select_backend(**kwargs)
        mode_name = getattr(backend, "name", "unknown")
        if backend.should_exact_flat():
            self._runtime_log("progress", f"[HierOT] mode={mode_name} single-level exact solve")
            result = backend.solve_exact_flat(tolerance)
            self._solve_wall_time = time.perf_counter() - t_solve_start
            self._profiler.end_run()
            if result is not None and self._profiling_enabled:
                profiling = self._profiler.summary(
                    build_time=self.build_time,
                    solve_time=self._solve_wall_time,
                )
                result["profiling"] = profiling
                self._print_profile_run(profiling)
            return result

        self._profiler.start_phase(PHASE_INIT_RUN_STATE)
        try:
            run_state = backend.init_run_state(tolerance, **kwargs)
        finally:
            self._profiler.end_phase()
        level_indices = list(backend.get_level_indices(run_state))
        self._runtime_log(
            "progress",
            f"[HierOT] mode={mode_name} start hierarchical solve with "
            f"{len(level_indices)} levels, max_inner_iter={max_inner_iter}"
        )
        deferred_final_level: Optional[Dict[str, Any]] = None
        for level_idx in level_indices:
            self._profiler.start_level(level_idx)
            self._profiler.start_phase(PHASE_INIT_LEVEL_STATE)
            try:
                level_state = backend.init_level_state(
                    level_idx=level_idx,
                    run_state=run_state,
                    tolerance=tolerance,
                    cost_type=cost_type,
                    use_bfs_skeleton=use_bfs_skeleton,
                )
            finally:
                self._profiler.end_phase()
            if level_state is None:
                self._profiler.end_level(level_idx, 0)
                continue

            max_iters = backend.get_max_inner_iters(level_state, max_inner_iter)
            active_size = level_state.get("curr_active_size")
            active_text = "" if active_size is None else f", active_size={int(active_size):,}"
            self._runtime_log(
                "progress",
                f"[HierOT] enter level={level_idx}, "
                f"max_iters={int(max_iters)}{active_text}"
            )
            iters_done = 0
            for inner_iter in range(max_iters):
                level_state["inner_iter"] = inner_iter
                self._profiler.start_iteration(level_idx, inner_iter)
                step_pack: Dict[str, Any] = {"success": False}
                try:
                    self._profiler.start_phase(PHASE_PREPARE_ITERATION_INPUT)
                    backend.prepare_iteration_input(level_state, run_state, inner_iter)
                    self._profiler.end_phase()

                    self._profiler.start_phase(PHASE_SOLVE_ITERATION_LP)
                    level_state["_pre_lp_active"] = level_state.get("curr_active_size")
                    step_pack = backend.solve_iteration_lp(level_state, run_state)
                    if isinstance(step_pack.get("components"), dict):
                        self._profiler.add_components(step_pack.get("components"))
                    self._profiler.end_phase()
                    iters_done = inner_iter + 1
                    if not step_pack.get("success"):
                        self._profiler.end_level(level_idx, int(level_state.get("inner_iter", 0)))
                        self._solve_wall_time = time.perf_counter() - t_solve_start
                        self._profiler.end_run()
                        return None

                    self._profiler.start_phase(PHASE_FINALIZE_ITERATION, gpu_possible=True)
                    backend.finalize_iteration(
                        level_state=level_state,
                        run_state=run_state,
                        step_pack=step_pack,
                        convergence_criterion=convergence_criterion,
                        tolerance=tolerance,
                    )
                    self._profiler.end_phase()

                    self._profiler.start_phase(PHASE_CALLBACK)
                    self._emit_inner_iteration_callback(
                        backend=backend,
                        level_state=level_state,
                        step_pack=step_pack,
                    )
                    self._profiler.end_phase()

                    self._profiler.start_phase(PHASE_SHOULD_STOP_ITERATION)
                    should_stop = backend.should_stop_iteration(
                        level_state=level_state,
                        run_state=run_state,
                        max_inner_iter=max_inner_iter,
                        step_pack=step_pack,
                    )
                    self._profiler.end_phase()
                    if should_stop:
                        self._runtime_log(
                            "progress",
                            f"[HierOT] level={level_idx} stop at iter={inner_iter + 1}/{int(max_iters)}"
                        )
                        break
                finally:
                    self._profiler.end_phase()
                    self._profiler.start_phase(PHASE_PROFILING_OUTPUT)
                    iter_profile = self._profiler.current_iteration_snapshot()
                    self._print_profile_inner(
                        level_idx=level_idx,
                        inner_iter=inner_iter,
                        iter_profile=iter_profile,
                        step_pack=step_pack,
                    )
                    self._profiler.end_phase()
                    self._profiler.end_iteration()

            level_state["iters_done"] = iters_done
            level_state["inner_iter"] = iters_done
            self._profiler.start_phase(PHASE_RECORD_LEVEL_RESULT)
            backend.record_level_result(level_state, final_refinement_tolerance)
            self._profiler.end_phase()
            self._profiler.start_phase(PHASE_ADVANCE_TO_NEXT_LEVEL)
            backend.advance_to_next_level(level_state, run_state)
            self._profiler.end_phase()
            self._profiler.start_phase(PHASE_PROFILING_OUTPUT)
            level_profile = self._profiler.current_level_snapshot()
            if level_idx == 0:
                deferred_final_level = {
                    "level_idx": level_idx,
                    "level_state": dict(level_state),
                    "level_profile": level_profile,
                }
            else:
                self._print_profile_level(level_idx, level_state, level_profile)
            self._profiler.end_phase()
            self._profiler.end_level(level_idx, int(level_state.get("inner_iter", 0)))
            level_obj = float("nan")
            history = self.solutions.get(level_idx, {}).get("history")
            if isinstance(history, list) and history:
                level_obj = float(history[-1])
            if level_idx == 0:
                if deferred_final_level is not None:
                    deferred_final_level["level_obj"] = level_obj
            else:
                self._runtime_log(
                    "progress",
                    f"[HierOT] finish level={level_idx}, "
                    f"iters={int(level_state.get('iters_done', 0))}, obj={level_obj:.6e}"
                )

        self._profiler.start_phase(PHASE_PACKAGE_RESULT)
        try:
            result = backend.package_result()
        finally:
            self._profiler.end_phase()
        self._solve_wall_time = time.perf_counter() - t_solve_start
        self._profiler.end_run()
        if self._profiling_enabled:
            profiling = self._profiler.summary(
                build_time=self.build_time,
                solve_time=self._solve_wall_time,
            )
            result["profiling"] = profiling
            self._print_profile_run(profiling, include_level_breakdown=False)
        if deferred_final_level is not None:
            self._runtime_log("profile_level", "")
            self._print_profile_level(
                deferred_final_level["level_idx"],
                deferred_final_level["level_state"],
                deferred_final_level["level_profile"],
            )
            self._runtime_log(
                "progress",
                f"[HierOT] finish level={deferred_final_level['level_idx']}, "
                f"iters={int(deferred_final_level['level_state'].get('iters_done', 0))}, "
                f"obj={float(deferred_final_level.get('level_obj', float('nan'))):.6e}"
            )
        if self._profiling_enabled and "profiling" in result:
            self._runtime_log("profile_run", "")
            self._print_profile_run_level_breakdown(result["profiling"])
        self._runtime_log("progress", f"[HierOT] solve complete in {self._solve_wall_time:.3f}s")
        self._runtime_log("progress", "================ solve complete ================")
        return result

    @staticmethod
    def _fmt_size_ratio(count: int, n_s: int, n_t: int) -> str:
        denom = n_s + n_t
        ratio = (count / denom) if denom > 0 else 0.0
        return f"{count:,} (x{ratio:.2f})"

    def _initialize_level(
        self,
        level_idx: int,
        cost_type: str,
        use_bfs_skeleton: bool
    ) -> Tuple[csc_matrix, np.ndarray]:
        
        fine_lvl_s = self.hierarchy_s.levels[level_idx]
        fine_lvl_t = self.hierarchy_t.levels[level_idx]
        coarse_sol = self.solutions[level_idx + 1]

        n_s = len(fine_lvl_s.points)
        n_t = len(fine_lvl_t.points)
        init_components: Dict[str, float] = {}

        
        t0 = time.perf_counter()
        primal_curr, dual_curr = self._refine_solution(
            coarse_sol,
            self.hierarchy_s.levels[level_idx + 1],
            self.hierarchy_t.levels[level_idx + 1],
            fine_lvl_s, fine_lvl_t
        )
        init_components["refine_solution"] = time.perf_counter() - t0
        logger.debug(
            f"  [Level {level_idx}] Init: primal nnz={primal_curr.nnz}")

        cost_type = _normalize_cost_type_name(cost_type)
        
        t0 = time.perf_counter()
        if cost_type == 'lowrank':
            level_cache = _prepare_level_cost_cache_lowrank(
                fine_lvl_s.points, fine_lvl_t.points, fine_lvl_s.cost_vec, fine_lvl_t.cost_vec)
        elif cost_type == 'l1':
            level_cache = _prepare_level_cost_cache_l1(
                fine_lvl_s.points, fine_lvl_t.points)
        elif cost_type == 'linf':
            level_cache = _prepare_level_cost_cache_linf(
                fine_lvl_s.points, fine_lvl_t.points)
        elif cost_type == 'l2':
            level_cache = _prepare_level_cost_cache_euclidean(
                fine_lvl_s.points, fine_lvl_t.points)
        else:  
            level_cache = _prepare_level_cost_cache_sqeuclidean(
                fine_lvl_s.points, fine_lvl_t.points)
        init_components["build_level_cache"] = time.perf_counter() - t0

        
        t0 = time.perf_counter()
        self.active_support = ActiveSupport(level_cache, track_creation=False)
        init_components["init_active_support"] = time.perf_counter() - t0

        
        t0 = time.perf_counter()
        init_args = (
            primal_curr, (dual_curr[:n_s], dual_curr[n_s:]), level_cache)
        init_cands = self.strategy.generate(
            *init_args,
            level_idx=int(level_idx),
            inner_iter=-1,
        )
        init_components["initial_pricing"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self._append_new_pairs_arrays(init_cands[0], init_cands[1], 0)
        init_components["append_initial_candidates"] = time.perf_counter() - t0

        
        bfs_added = 0
        if use_bfs_skeleton:
            t0 = time.perf_counter()
            bfs_rows, bfs_cols = _northwest_corner_numba(
                fine_lvl_s.masses, fine_lvl_t.masses)
            self._append_new_pairs_arrays(bfs_rows, bfs_cols, -1)
            bfs_added = int(len(bfs_rows))
            init_components["bfs_skeleton"] = time.perf_counter() - t0
            logger.debug(
                f"  [Level {level_idx}] Init BFS added {len(bfs_rows)} edges.")

        
        t0 = time.perf_counter()
        primal_coo = primal_curr.tocoo()
        inc_keys = self.active_support.keys
        sorter = np.argsort(inc_keys)
        sorted_inc_keys = inc_keys[sorter]
        sorted_inc_pos = np.arange(len(inc_keys))[sorter]

        self.active_support.x_prev = _fill_x_prev_numba(
            primal_coo.row, primal_coo.col, primal_coo.data,
            sorted_inc_keys, sorted_inc_pos, self.active_support.x_prev, n_t
        )
        init_components["warm_start_fill"] = time.perf_counter() - t0

        self._print_profile_init_level(
            level_idx,
            init_components,
            primal_nnz=primal_curr.nnz,
            active_size=self.active_support.size,
            bfs_added=bfs_added,
        )

        return primal_curr, dual_curr

    def _solve_exact_flat(self, tolerance: Dict[str, float]) -> Dict[str, Any]:
        
        logger.info("Single level detected. Running exact solver.")
        lvl_s = self.hierarchy_s.finest_level
        lvl_t = self.hierarchy_t.finest_level

        t_lp_start = time.perf_counter()
        with self._profiler.timer("lp", gpu_possible=True):
            primal_dense, dual_sol, result, _ = self._solve_lp(
                lvl_s, lvl_t, tolerance,verbose=self.lp_solver_verbose)
        lp_solve_time_total = float(time.perf_counter() - t_lp_start)
        if not result.success:
            return None

        primal_coo = primal_dense.tocoo()
        mask = primal_coo.data > 1e-12
        primal_sparse = csc_matrix(
            (primal_coo.data[mask], (primal_coo.row[mask], primal_coo.col[mask])), shape=primal_dense.shape)
        level_summary = {
            'level': 0,
            'n_source': int(primal_dense.shape[0]),
            'n_target': int(primal_dense.shape[1]),
            'iters': 1,
            'time': lp_solve_time_total,
            'objective': float(result.obj_val),
            'lp_time': lp_solve_time_total,
            'pricing_time': 0.0,
            'support_final': int(primal_sparse.nnz),
        }

        return {
            'primal': primal_sparse,
            'dual': dual_sol,
            'final_obj': result.obj_val,
            'all_history': {0: {'history': [result.obj_val]}},
            'sparse_coupling': {'rows': primal_coo.row[mask], 'cols': primal_coo.col[mask], 'values': primal_coo.data[mask]},
            'level_summaries': [level_summary],
            'lp_solve_time_total': lp_solve_time_total,
        }

    def _check_convergence(
        self,
        history: List[float],
        criterion: str,
        tolerance: Dict[str, float],
        plateau_counter: int
    ) -> Tuple[bool, int]:
        
        if len(history) < 2:
            return False, plateau_counter

        is_converged = False
        if criterion == 'objective':
            diff = abs(history[-2] - history[-1]) / (abs(history[-2]) + 1e-9)
            if diff < tolerance['objective']:
                plateau_counter += 1
            else:
                plateau_counter = 0

            if plateau_counter >= int(getattr(self, "_objective_plateau_iters", 1)):
                is_converged = True

        return is_converged, plateau_counter

    def _perform_cleaning(self, duals: np.ndarray, n_s: int, n_t: int) -> None:
        del duals, n_s, n_t
        return None

    def _record_level_result(self, level, primal, dual, history, total_time, lp_time, price_time, iters):
        
        if not hasattr(self, 'level_summaries'):
            self.level_summaries = []

        self.solutions[level] = {'primal': primal,
                                 'dual': dual, 'history': history}

        self.level_summaries.append({
            "level": level,
            "n_source": primal.shape[0],
            "n_target": primal.shape[1],
            "iters": iters,
            "time": total_time,
            "objective": history[-1],
            "lp_time": lp_time,
            "pricing_time": price_time,
            "support_final": primal.nnz
        })

    @staticmethod
    def _sum_level_summary_metric(
        level_summaries: List[Dict[str, Any]],
        key: str,
    ) -> float:
        return float(
            sum(float(item.get(key, 0.0)) for item in level_summaries if isinstance(item, dict))
        )

    def _package_final_result(self) -> Dict[str, Any]:
        
        final_sol = self.solutions[0]
        nonzero_mask = self.active_support.x_prev > 1e-12
        level_summaries = getattr(self, 'level_summaries', [])
        sparse_coupling = {
            'rows': self.active_support.rows[nonzero_mask],
            'cols': self.active_support.cols[nonzero_mask],
            'values': self.active_support.x_prev[nonzero_mask]
        }
        return {
            'primal': final_sol['primal'],
            'dual': final_sol['dual'],
            'final_obj': final_sol['history'][-1],
            'all_history': self.solutions,
            'sparse_coupling': sparse_coupling,
            'level_summaries': level_summaries,
            'lp_solve_time_total': self._sum_level_summary_metric(level_summaries, 'lp_time'),
        }

    def _append_new_pairs_arrays(self, rows_add: np.ndarray, cols_add: np.ndarray, current_inner_iter: int):
        if len(rows_add) == 0:
            return

        
        level_cache = self.active_support.level_cache
        cost_type = _normalize_cost_type_name(level_cache.get('cost_type', 'l2^2'))
        if cost_type == 'lowrank':
            c_add = _cost_pairs_lowrank_batched(
                level_cache, rows_add, cols_add)
        elif cost_type == 'l1':
            c_add = _cost_pairs_l1_batched(
                level_cache, rows_add, cols_add)
        elif cost_type == 'linf':
            c_add = _cost_pairs_linf_batched(
                level_cache, rows_add, cols_add)
        elif cost_type == 'l2':
            c_add = _cost_pairs_euclidean_batched(
                level_cache, rows_add, cols_add)
        else:
            c_add = _cost_pairs_sqeuclidean_batched(
                level_cache, rows_add, cols_add)

        
        self.active_support.add_pairs(
            rows_add, cols_add, c_add, current_inner_iter)

        logger.debug(f"  [_append] Added {len(rows_add)} pairs.")

    def _solve_lp(self, lvl_s, lvl_t, tolerance, warm_start_dual=None, verbose=0):
        use_incremental = (
            self.active_support is not None and self.active_support.size > 0)
        n_source, n_target = len(lvl_s.points), len(lvl_t.points)

        if use_incremental:
            rows, cols, c = self.active_support.rows, self.active_support.cols, self.active_support.c_vec
            warm_start_primal = self.active_support.x_prev
        else:
            
            rows, cols = np.indices((n_source, n_target))
            rows, cols = rows.ravel(), cols.ravel()
            
            cost_type = _normalize_cost_type_name(getattr(self, '_cost_type', 'l2^2'))
            if cost_type == 'lowrank':
                F, G, a, b = lvl_s.points, lvl_t.points, lvl_s.cost_vec, lvl_t.cost_vec
                c = (a[:, None] + b[None, :] - F @ G.T).ravel()
            elif cost_type == 'l1':
                c = cdist(lvl_s.points, lvl_t.points, 'cityblock').ravel()
            elif cost_type == 'linf':
                c = cdist(lvl_s.points, lvl_t.points, 'chebyshev').ravel()
            elif cost_type == 'l2':
                c = cdist(lvl_s.points, lvl_t.points, 'euclidean').ravel()
            elif cost_type == 'lp':
                c = grid_pairwise_cost(
                    lvl_s.points,
                    lvl_t.points,
                    cost_type='lp',
                    p=int(getattr(self, '_grid_p', 2)),
                ).ravel()
            else:  
                c = cdist(lvl_s.points, lvl_t.points, 'sqeuclidean').ravel()
            warm_start_primal = None

        n_vars = c.size
        n_constraints = n_source + n_target

        
        data = np.ones(2 * n_vars, dtype=np.float32)
        row_indices = np.empty(2 * n_vars, dtype=np.int32)
        row_indices[0::2] = rows
        row_indices[1::2] = n_source + cols
        indptr = np.arange(0, 2 * n_vars + 1, 2, dtype=np.int32)
        A_csc = csc_matrix((data, row_indices, indptr),
                           shape=(n_constraints, n_vars))

        b_eq = np.concatenate([lvl_s.masses, lvl_t.masses])
        lb, ub = np.zeros(n_vars), np.full(n_vars, np.inf)

        result = self.solver.solve(
            c, A_csc, b_eq, lb, ub, n_constraints,
            warm_start_primal=warm_start_primal,
            warm_start_dual=warm_start_dual,
            tolerance=tolerance,
            verbose=verbose,
            **getattr(self, "_lp_solver_kwargs", {}),
        )

        primal_sol = None
        dual_sol = None
        if result.success:
            primal_sol = csc_matrix(
                (result.x, (rows, cols)), shape=(n_source, n_target))
            dual_sol = result.y

        return primal_sol, dual_sol, result, {}

    def _refine_solution(self, coarse_sol, coarse_lvl_s, coarse_lvl_t, fine_lvl_s, fine_lvl_t):
        primal_coarse, duals_coarse = coarse_sol['primal'], coarse_sol['dual']

        
        f_coarse, g_coarse = duals_coarse[:len(
            coarse_lvl_s.points)], duals_coarse[len(coarse_lvl_s.points):]
        duals_fine = np.concatenate(
            (f_coarse[fine_lvl_s.child_labels], g_coarse[fine_lvl_t.child_labels]))

        
        p_s, p_t = fine_lvl_s.child_labels, fine_lvl_t.child_labels
        n_fine_s, n_fine_t = len(p_s), len(p_t)
        n_coarse_s, n_coarse_t = primal_coarse.shape

        s_counts = np.bincount(p_s, minlength=n_coarse_s)
        t_counts = np.bincount(p_t, minlength=n_coarse_t)

        primal_coarse_coo = primal_coarse.tocoo()
        coarse_data = np.asarray(primal_coarse_coo.data, dtype=np.float64)

        offsets = _precalc_offsets(
            primal_coarse_coo.row, primal_coarse_coo.col, s_counts, t_counts)
        total_nnz = offsets[-1]
        out_rows = np.empty(total_nnz, dtype=np.int32)
        out_cols = np.empty(total_nnz, dtype=np.int32)
        out_data = np.empty(total_nnz, dtype=np.float32)

        
        w_s = 1.0 / np.maximum(s_counts[p_s], 1.0)
        w_t = 1.0 / np.maximum(t_counts[p_t], 1.0)

        map_s = coo_matrix((w_s, (p_s, np.arange(n_fine_s))),
                           shape=(n_coarse_s, n_fine_s)).tocsr()
        map_t = coo_matrix((w_t, (p_t, np.arange(n_fine_t))),
                           shape=(n_coarse_t, n_fine_t)).tocsr()

        _numba_expand_kernel(
            primal_coarse_coo.row, primal_coarse_coo.col, coarse_data,
            map_s.indptr, map_s.indices, map_s.data,
            map_t.indptr, map_t.indices, map_t.data,
            offsets, out_rows, out_cols, out_data
        )

        refined_primal = coo_matrix(
            (out_data, (out_rows, out_cols)), shape=(n_fine_s, n_fine_t))
        return refined_primal, duals_fine
