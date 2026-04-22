from __future__ import annotations

import sys
import time
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None

from ...core.solver_utils import decode_keep_1d_to_struct, remap_duals_for_warm_start
from .operators import prolong_grid_dual
from . import violation_check as grid_violation_check

_MGPD_ROOT = Path("/home/xwz/project/MGPD")


def _load_mgpd_same_level_ops():
    if not _MGPD_ROOT.exists():
        return None, None, None
    if str(_MGPD_ROOT) not in sys.path:
        sys.path.insert(0, str(_MGPD_ROOT))
    try:
        from init import topk_expand_gpu  # type: ignore
        from operators import fine_dualOT_delete_numba, keep2keepcoord  # type: ignore
    except Exception:
        return None, None, None
    return topk_expand_gpu, fine_dualOT_delete_numba, keep2keepcoord


def _load_mgpd_shield_builder():
    if not _MGPD_ROOT.exists():
        return None
    if str(_MGPD_ROOT) not in sys.path:
        sys.path.insert(0, str(_MGPD_ROOT))
    try:
        from shield import build_shield  # type: ignore
    except Exception:
        return None
    return build_shield


def grid_mgpd_keepcoord_fields():
    return ["idx_coarse", "idx1", "idx2", "idx_i1", "idx_i2", "idx_j1", "idx_j2"]


def grid_keep_to_rows_cols(keep: np.ndarray, n_t: int) -> tuple[np.ndarray, np.ndarray]:
    keep_i64 = np.asarray(keep, dtype=np.int64)
    rows = (keep_i64 // int(n_t)).astype(np.int32, copy=False)
    cols = (keep_i64 % int(n_t)).astype(np.int32, copy=False)
    return rows, cols


def grid_rows_cols_to_keep(rows: np.ndarray, cols: np.ndarray, n_t: int) -> np.ndarray:
    return (
        np.asarray(rows, dtype=np.int64) * np.int64(n_t) + np.asarray(cols, dtype=np.int64)
    ).astype(np.int64, copy=False)


def grid_nnz_threshold() -> float:
    return 1e-12


def grid_keepcoord_resolution_last(*, level_state: Dict[str, Any], inner_iter: int) -> int:
    resolution_now = int(round(np.sqrt(len(level_state["level_t"].points))))
    if inner_iter == 0 and not bool(level_state.get("is_coarsest", False)):
        return max(resolution_now // 2, 1)
    return resolution_now


def grid_full_keep(n_s: int, n_t: int) -> np.ndarray:
    return np.arange(int(n_s) * int(n_t), dtype=np.int64)


def grid_level_for_dual_lp(level: Any) -> Any:
    level_points = np.asarray(level.points, dtype=np.float32)
    level_res = int(np.max(level_points)) + 1 if level_points.size > 0 else 1
    scale = max(float(level_res), 1.0)
    return SimpleNamespace(
        points=level_points / scale,
        masses=np.asarray(level.masses, dtype=np.float32),
    )


def grid_decode_keep_to_coord(
    keep: np.ndarray,
    *,
    resolution_now: int,
    resolution_last: int,
) -> np.ndarray:
    from .solve_lp import _load_mgpd_lp_build_ops

    keep2keepcoord, _, _ = _load_mgpd_lp_build_ops()
    if keep2keepcoord is not None and resolution_now > 0:
        try:
            return keep2keepcoord(
                np.asarray(keep, dtype=np.int64),
                int(resolution_now),
                int(resolution_last),
                fields=grid_mgpd_keepcoord_fields(),
            )
        except Exception:
            pass
    return decode_keep_1d_to_struct(
        np.asarray(keep, dtype=np.int64),
        int(resolution_now) * int(resolution_now),
    )


CUDA_SRC_ARGMAX_KEY = r'''
extern "C" __global__
void pick_argmax_key(const long long* __restrict__ y_keep,
                     const float* __restrict__ y_val,
                     const long long K,
                     const int r,
                     unsigned long long* __restrict__ best_key)
{
    const long long tid = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (tid >= K) return;

    long long enc = y_keep[tid];
    int j_t =  enc % r;   enc /= r;
    int i_t =  enc % r;   enc /= r;
    int j_s =  enc % r;   enc /= r;
    int i_s =  enc;
    int src = i_s * r + j_s;

    float v          = y_val[tid];
    unsigned int vb  = __float_as_uint(v);
    unsigned int kb  = 0xFFFFFFFFu - (unsigned int)tid;
    unsigned long long key = ((unsigned long long)vb << 32) | (unsigned long long)kb;

    atomicMax(&best_key[src], key);
}
'''

CUDA_SRC_DECODE_KEY = r'''
extern "C" __global__
void decode_from_key(const long long* __restrict__ y_keep,
                     const unsigned long long* __restrict__ best_key,
                     const int        r,
                     int*             __restrict__ t_row,
                     int*             __restrict__ t_col)
{
    const int src = blockDim.x * blockIdx.x + threadIdx.x;
    const int r2  = r * r;
    if (src >= r2) return;

    unsigned long long key = best_key[src];
    if (key == 0ULL) return;

    unsigned int kb = (unsigned int)(key & 0xFFFFFFFFULL);
    int tid = 0xFFFFFFFFu - kb;

    long long enc = y_keep[tid];
    int j_t = enc % r; enc /= r;
    int i_t = enc % r;

    t_row[src] = i_t;
    t_col[src] = j_t;
}
'''


@lru_cache(None)
def _ker_pick_argmax_key():
    if cp is None:  # pragma: no cover
        raise RuntimeError("cupy is required for grid shielding")
    return cp.RawKernel(CUDA_SRC_ARGMAX_KEY, "pick_argmax_key")


@lru_cache(None)
def _ker_decode_from_key():
    if cp is None:  # pragma: no cover
        raise RuntimeError("cupy is required for grid shielding")
    return cp.RawKernel(CUDA_SRC_DECODE_KEY, "decode_from_key")


def _pick_t_cupy(r: int, y_keep: np.ndarray, y_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if cp is None:  # pragma: no cover
        raise RuntimeError("cupy is required for grid shielding")
    yk = cp.asarray(y_keep, dtype=cp.int64)
    yv = cp.asarray(y_val, dtype=cp.float32)

    r2 = r * r
    k = int(yk.size)
    thr = 256
    blk_k = (k + thr - 1) // thr
    blk_r = (r2 + thr - 1) // thr

    best_key = cp.zeros(r2, dtype=cp.uint64)
    t_row = cp.arange(r2, dtype=cp.int32) // r
    t_col = cp.arange(r2, dtype=cp.int32) % r

    _ker_pick_argmax_key()((blk_k,), (thr,), (yk, yv, k, r, best_key))
    _ker_decode_from_key()((blk_r,), (thr,), (yk, best_key, r, t_row, t_col))
    return cp.asnumpy(t_row), cp.asnumpy(t_col)


def _build_keep_shield_np_8(r: int, t_row: np.ndarray, t_col: np.ndarray) -> np.ndarray:
    idx = np.arange(r * r, dtype=np.int64)
    ii = idx // r
    jj = idx % r
    parts = []
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        mask = (ii + di >= 0) & (ii + di < r) & (jj + dj >= 0) & (jj + dj < r)
        if not np.any(mask):
            continue
        idx_a = idx[mask]
        idx_s = idx_a + np.int64(di * r + dj)
        parts.append(
            (
                (idx_a * np.int64(r) + t_row[idx_s].astype(np.int64)) * np.int64(r)
                + t_col[idx_s].astype(np.int64)
            )
        )
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(parts).astype(np.int64, copy=False))


def _gpu_unique_inplace(arr):
    if cp is None:  # pragma: no cover
        raise RuntimeError("cupy is required for grid shielding")
    arr = arr.ravel()
    max_val = int(arr.max().item()) if arr.size else 0
    if arr.dtype == cp.int64 and max_val < np.iinfo(np.int32).max:
        arr = arr.astype(cp.int32)
    arr.sort()
    if arr.size == 0:
        return arr.astype(cp.int64, copy=False)
    diff = arr[1:] != arr[:-1]
    unique_count = int(cp.count_nonzero(diff).item()) + 1
    out = cp.empty(unique_count, dtype=arr.dtype)
    out[0] = arr[0]
    out[1:] = arr[1:][diff]
    return out.astype(cp.int64, copy=False)


_SRC_COUNT8 = r'''
extern "C" __global__
void count8(const int *rowL,const int *rowH,const int *colL,const int *colH,
            const int *mu,const int *md,const int *ml,const int *mr,
            const int *mul,const int *mur,const int *mdl,const int *mdr,
            const int *iA,const int *jA,
            const int *up,const int *down,const int *left,const int *right,
            const int *ul,const int *ur,const int *dl,const int *dr,
            const int *trow,const int *tcol,
            long long *counts,
            const int r,const long long r2,const long long N)
{
    long long a = blockDim.x*blockIdx.x + threadIdx.x;
    if (a >= N) return;

    int rl = rowL[a], rh = rowH[a], cl = colL[a], ch = colH[a];
    int ia = iA[a], ja = jA[a];

    unsigned long long cnt = 0ULL;

    for (int bi = rl; bi <= rh; ++bi) {
        for (int bj = cl; bj <= ch; ++bj) {
            bool shield = false;
            if (!shield && mu[a]) {
                int isp = trow[ up[a] ], jsp = tcol[ up[a] ];
                if ((-1)*(bi-isp) + 0*(bj-jsp) > 0) shield = true;
            }
            if (!shield && md[a]) {
                int isp = trow[ down[a] ], jsp = tcol[ down[a] ];
                if ((+1)*(bi-isp) + 0*(bj-jsp) > 0) shield = true;
            }
            if (!shield && ml[a]) {
                int isp = trow[ left[a] ], jsp = tcol[ left[a] ];
                if (0*(bi-isp) + (-1)*(bj-jsp) > 0) shield = true;
            }
            if (!shield && mr[a]) {
                int isp = trow[ right[a] ], jsp = tcol[ right[a] ];
                if (0*(bi-isp) + (+1)*(bj-jsp) > 0) shield = true;
            }
            if (!shield && mul[a]) {
                int isp = trow[ ul[a] ], jsp = tcol[ ul[a] ];
                if ((-1)*(bi-isp) + (-1)*(bj-jsp) > 0) shield = true;
            }
            if (!shield && mur[a]) {
                int isp = trow[ ur[a] ], jsp = tcol[ ur[a] ];
                if ((-1)*(bi-isp) + (+1)*(bj-jsp) > 0) shield = true;
            }
            if (!shield && mdl[a]) {
                int isp = trow[ dl[a] ], jsp = tcol[ dl[a] ];
                if ((+1)*(bi-isp) + (-1)*(bj-jsp) > 0) shield = true;
            }
            if (!shield && mdr[a]) {
                int isp = trow[ dr[a] ], jsp = tcol[ dr[a] ];
                if ((+1)*(bi-isp) + (+1)*(bj-jsp) > 0) shield = true;
            }
            if (!shield) ++cnt;
        }
    }
    counts[a] = (long long)cnt;
}
'''

_SRC_FILL8 = r'''
extern "C" __global__
void fill8(const int *rowL,const int *rowH,const int *colL,const int *colH,
           const int *mu,const int *md,const int *ml,const int *mr,
           const int *mul,const int *mur,const int *mdl,const int *mdr,
           const int *iA,const int *jA,
           const int *up,const int *down,const int *left,const int *right,
           const int *ul,const int *ur,const int *dl,const int *dr,
           const int *trow,const int *tcol,
           const long long *pref,long long *out,
           const int r,const long long r2,const long long N)
{
    long long a = blockDim.x*blockIdx.x + threadIdx.x;
    if (a >= N) return;

    int rl = rowL[a], rh = rowH[a], cl = colL[a], ch = colH[a];
    long long pos = pref[a];

    for (int bi = rl; bi <= rh; ++bi) {
        long long base = (long long)bi * r;
        for (int bj = cl; bj <= ch; ++bj) {
            bool shield = false;
            if (!shield && mu[a]) {
                int isp = trow[ up[a] ], jsp = tcol[ up[a] ];
                if ((-1)*(bi-isp) + 0*(bj-jsp) > 0) shield = true;
            }
            if (!shield && md[a]) {
                int isp = trow[ down[a] ], jsp = tcol[ down[a] ];
                if ((+1)*(bi-isp) + 0*(bj-jsp) > 0) shield = true;
            }
            if (!shield && ml[a]) {
                int isp = trow[ left[a] ], jsp = tcol[ left[a] ];
                if (0*(bi-isp) + (-1)*(bj-jsp) > 0) shield = true;
            }
            if (!shield && mr[a]) {
                int isp = trow[ right[a] ], jsp = tcol[ right[a] ];
                if (0*(bi-isp) + (+1)*(bj-jsp) > 0) shield = true;
            }
            if (!shield && mul[a]) {
                int isp = trow[ ul[a] ], jsp = tcol[ ul[a] ];
                if ((-1)*(bi-isp) + (-1)*(bj-jsp) > 0) shield = true;
            }
            if (!shield && mur[a]) {
                int isp = trow[ ur[a] ], jsp = tcol[ ur[a] ];
                if ((-1)*(bi-isp) + (+1)*(bj-jsp) > 0) shield = true;
            }
            if (!shield && mdl[a]) {
                int isp = trow[ dl[a] ], jsp = tcol[ dl[a] ];
                if ((+1)*(bi-isp) + (-1)*(bj-jsp) > 0) shield = true;
            }
            if (!shield && mdr[a]) {
                int isp = trow[ dr[a] ], jsp = tcol[ dr[a] ];
                if ((+1)*(bi-isp) + (+1)*(bj-jsp) > 0) shield = true;
            }
            if (!shield) {
                long long b = base + bj;
                out[pos++] = a * r2 + b;
            }
        }
    }
}
'''


@lru_cache(None)
def _kernel_count_8():
    if cp is None:  # pragma: no cover
        raise RuntimeError("cupy is required for grid shielding")
    return cp.RawKernel(_SRC_COUNT8, "count8")


@lru_cache(None)
def _kernel_fill_8():
    if cp is None:  # pragma: no cover
        raise RuntimeError("cupy is required for grid shielding")
    return cp.RawKernel(_SRC_FILL8, "fill8")


def _build_yhat_grid_gpu_8(r: int, t_row: np.ndarray, t_col: np.ndarray) -> np.ndarray:
    if cp is None:  # pragma: no cover
        raise RuntimeError("cupy is required for grid shielding")
    t_row_cp = cp.asarray(t_row, dtype=cp.int32)
    t_col_cp = cp.asarray(t_col, dtype=cp.int32)
    r2 = r * r
    idx = cp.arange(r2, dtype=cp.int32)
    i_a = idx // r
    j_a = idx - i_a * r

    mu = (i_a > 0).astype(cp.int32)
    md = (i_a < r - 1).astype(cp.int32)
    ml = (j_a > 0).astype(cp.int32)
    mr = (j_a < r - 1).astype(cp.int32)
    up, down = idx - r, idx + r
    left, right = idx - 1, idx + 1

    mul = ((i_a > 0) & (j_a > 0)).astype(cp.int32)
    mur = ((i_a > 0) & (j_a < r - 1)).astype(cp.int32)
    mdl = ((i_a < r - 1) & (j_a > 0)).astype(cp.int32)
    mdr = ((i_a < r - 1) & (j_a < r - 1)).astype(cp.int32)
    ul = idx - r - 1
    ur = idx - r + 1
    dl = idx + r - 1
    dr = idx + r + 1

    row_low = cp.zeros(r2, dtype=cp.int32)
    row_high = cp.full(r2, r - 1, dtype=cp.int32)
    col_low = cp.zeros(r2, dtype=cp.int32)
    col_high = cp.full(r2, r - 1, dtype=cp.int32)
    row_low[mu == 1] = t_row_cp[up][mu == 1]
    row_high[md == 1] = t_row_cp[down][md == 1]
    col_low[ml == 1] = t_col_cp[left][ml == 1]
    col_high[mr == 1] = t_col_cp[right][mr == 1]

    counts = cp.empty(r2, dtype=cp.int64)
    threads = 128
    blocks = (r2 + threads - 1) // threads
    _kernel_count_8()(
        (blocks,),
        (threads,),
        (
            row_low, row_high, col_low, col_high,
            mu, md, ml, mr,
            mul, mur, mdl, mdr,
            i_a, j_a,
            up, down, left, right,
            ul, ur, dl, dr,
            t_row_cp, t_col_cp,
            counts,
            np.int32(r), np.int64(r2), np.int64(r2),
        ),
    )
    prefix = cp.empty(r2 + 1, dtype=cp.int64)
    prefix[0] = 0
    cp.cumsum(counts, out=prefix[1:])
    total = int(prefix[-1].item())
    out = cp.empty(total, dtype=cp.int64)
    _kernel_fill_8()(
        (blocks,),
        (threads,),
        (
            row_low, row_high, col_low, col_high,
            mu, md, ml, mr,
            mul, mur, mdl, mdr,
            i_a, j_a,
            up, down, left, right,
            ul, ur, dl, dr,
            t_row_cp, t_col_cp,
            prefix, out,
            np.int32(r), np.int64(r2), np.int64(r2),
        ),
    )
    out.sort()
    return cp.asnumpy(_gpu_unique_inplace(out))


def build_grid_shield(r_now: int, y_value_last: np.ndarray, y_keep_last: np.ndarray) -> np.ndarray:
    keep = np.asarray(y_keep_last, dtype=np.int64)
    vals = np.asarray(y_value_last, dtype=np.float32)
    if keep.size == 0 or vals.size == 0:
        return np.empty(0, dtype=np.int64)
    t_row, t_col = _pick_t_cupy(int(r_now), keep, vals)
    keep_shield = _build_keep_shield_np_8(int(r_now), t_row, t_col)
    keep_hat = _build_yhat_grid_gpu_8(int(r_now), t_row, t_col)
    return np.unique(np.concatenate([keep, keep_shield, keep_hat])).astype(np.int64, copy=False)


def grid_refine_duals_from_coarse(
    solver,
    *,
    y_solution_last: Dict[str, Any],
    fine_level_s: Any,
    fine_level_t: Any,
) -> tuple[np.ndarray, np.ndarray]:
    keep_coarse = np.asarray(y_solution_last["keep"], dtype=np.int64)
    vals_coarse = np.asarray(y_solution_last["y"], dtype=np.float32)
    if keep_coarse.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    nonzero_mask = np.abs(vals_coarse) > grid_nnz_threshold()
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
                fields=grid_mgpd_keepcoord_fields(),
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
    child_s = np.asarray(solver.hierarchy_s.levels[coarse_level_idx].child_labels, dtype=np.int32)
    child_t = np.asarray(solver.hierarchy_t.levels[coarse_level_idx].child_labels, dtype=np.int32)
    children_s = [np.flatnonzero(child_s == idx).astype(np.int32) for idx in range(int(np.max(child_s)) + 1)]
    children_t = [np.flatnonzero(child_t == idx).astype(np.int32) for idx in range(int(np.max(child_t)) + 1)]
    n_t_fine = len(fine_level_t.points)
    coarse_res = len(solver.hierarchy_t.levels[coarse_level_idx].points)
    fine_res = len(fine_level_t.points)
    scale = max(int(round(np.sqrt(fine_res // max(coarse_res, 1)))), 1)
    value_scale = np.float32(scale**4)

    keep_parts = []
    val_parts = []
    rows_coarse, cols_coarse = grid_keep_to_rows_cols(
        keep_nonzero,
        len(solver.hierarchy_t.levels[coarse_level_idx].points),
    )
    for src_parent, tgt_parent, val in zip(rows_coarse, cols_coarse, vals_nonzero):
        fine_rows = children_s[int(src_parent)]
        fine_cols = children_t[int(tgt_parent)]
        if fine_rows.size == 0 or fine_cols.size == 0:
            continue
        rr = np.repeat(fine_rows, fine_cols.size)
        cc = np.tile(fine_cols, fine_rows.size)
        kk = grid_rows_cols_to_keep(rr, cc, n_t_fine)
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


def build_active_set_first_iter(solver, *, level_state, run_state):
    level_s = level_state["level_s"]
    level_t = level_state["level_t"]
    n_s = len(level_s.points)
    n_t = len(level_t.points)
    prepare_breakdown: Dict[str, float] = {}
    prepare_snapshot: Dict[str, Any] = {"level": int(level_state["level_idx"]), "iter": 0}

    if level_state["is_coarsest"]:
        keep = grid_full_keep(n_s, n_t)
        x_init = np.zeros(n_s + n_t, dtype=np.float32)
        y_init = {"y": np.zeros(len(keep), dtype=np.float32)}
        keep_coord = grid_decode_keep_to_coord(
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
    keep_refined, y_refined = grid_refine_duals_from_coarse(
        solver,
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
                check_keep_gpu = grid_violation_check.collect_violation_candidates(
                    solver,
                    x_dual=x_init,
                    level_s=grid_level_for_dual_lp(level_s),
                    level_t=grid_level_for_dual_lp(level_t),
                    vd_thr=float(run_state.get("vd_thr", 0.0625)),
                    p=int(run_state.get("p", 2)),
                )
                check_keep_gpu = cp.asarray(check_keep_gpu, dtype=cp.int64).reshape(-1)
                max_keep_valid = np.int64(n_s) * np.int64(n_t)
                check_keep_gpu = check_keep_gpu[
                    (check_keep_gpu >= 0) & (check_keep_gpu < max_keep_valid)
                ]
                if check_keep_gpu.size > max_candidates:
                    check_keep_gpu = check_keep_gpu[:max_candidates]
                keep_gpu = cp.sort(
                    cp.unique(
                        cp.concatenate([cp.asarray(keep_gpu, dtype=cp.int64).reshape(-1), check_keep_gpu])
                    )
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
            expand_keep_local(solver, keep_refined, y_refined, n_s, n_t)
            if run_state.get("if_shield", True)
            else keep_refined
        )
        prepare_breakdown["shielding"] = time.perf_counter() - t0

        if run_state.get("if_check", True):
            t0 = time.perf_counter()
            keep = grid_violation_check.apply_violation_check(
                solver,
                x_dual=x_init,
                level_s=grid_level_for_dual_lp(level_s),
                level_t=grid_level_for_dual_lp(level_t),
                keep=keep,
                vd_thr=float(run_state.get("vd_thr", 0.0625)),
                p=int(run_state.get("p", 2)),
            )
            prepare_breakdown["violation_check"] = time.perf_counter() - t0
    prepare_snapshot["shield_keep_len"] = int(len(keep))
    prepare_snapshot["check_keep_len"] = int(len(keep))

    if run_state.get("repair_coverage", False):
        t0 = time.perf_counter()
        keep, coverage_fix = repair_keep_coverage(solver, keep, level_s.points, level_t.points)
        prepare_breakdown["coverage_repair"] = time.perf_counter() - t0
    else:
        coverage_fix = {"added": 0, "rows_missing": 0, "cols_missing": 0}
    prepare_snapshot["coverage_added"] = int(coverage_fix["added"])
    prepare_snapshot["rows_missing"] = int(coverage_fix["rows_missing"])
    prepare_snapshot["cols_missing"] = int(coverage_fix["cols_missing"])
    prepare_snapshot["use_last_merged_len"] = int(len(keep))

    t0 = time.perf_counter()
    keep_coord = grid_decode_keep_to_coord(
        keep,
        resolution_now=int(round(np.sqrt(n_t))),
        resolution_last=grid_keepcoord_resolution_last(level_state=level_state, inner_iter=0),
    )
    prepare_breakdown["keep_coord"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_init_arr = build_y_init_exact_from_keep(
        solver,
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
        solver.keep_last = keep.copy()

    return {
        "x_init": x_init,
        "y_init": y_init,
        "keep": keep,
        "keep_coord": keep_coord,
        "prepare_breakdown": prepare_breakdown,
        "prepare_snapshot": prepare_snapshot,
        "prepare_time_total": float(sum(prepare_breakdown.values())),
    }


def build_active_set_subsequent_iter(solver, *, level_state, run_state):
    level_s = level_state["level_s"]
    level_t = level_state["level_t"]
    n_t = len(level_t.points)
    x_init = np.asarray(level_state["x_solution_last"], dtype=np.float32)
    keep_last = np.asarray(level_state["y_solution_last"]["keep"], dtype=np.int64)
    keep, y_vals = prepare_same_level_support(
        solver,
        y_solution_last=level_state["y_solution_last"],
        n_s=len(level_s.points),
        n_t=len(level_t.points),
    )
    keep_seed = np.asarray(getattr(solver, "_grid_same_level_keep_seed", keep), dtype=np.int64)
    prepare_breakdown: Dict[str, float] = {}
    prepare_snapshot: Dict[str, Any] = {
        "level": int(level_state["level_idx"]),
        "iter": int(level_state["current_iter"]),
        "keep_refined_len": int(len(keep)),
        "y_init_nnz": int(np.count_nonzero(y_vals)),
    }

    if run_state.get("if_shield", True):
        t0 = time.perf_counter()
        keep = expand_keep_local(
            solver,
            keep,
            y_vals,
            len(level_s.points),
            len(level_t.points),
        )
        prepare_breakdown["shielding"] = time.perf_counter() - t0
    prepare_snapshot["shield_keep_len"] = int(len(keep))

    if run_state.get("if_check", True):
        t0 = time.perf_counter()
        keep = grid_violation_check.apply_violation_check(
            solver,
            x_dual=x_init,
            level_s=grid_level_for_dual_lp(level_s),
            level_t=grid_level_for_dual_lp(level_t),
            keep=keep,
            vd_thr=float(run_state.get("vd_thr", 0.0625)),
            p=int(run_state.get("p", 2)),
        )
        prepare_breakdown["violation_check"] = time.perf_counter() - t0
    prepare_snapshot["check_keep_len"] = int(len(keep))

    t0 = time.perf_counter()
    keep = merge_with_use_last(
        solver,
        keep=keep,
        use_last=run_state.get("use_last", True),
        use_last_after_inner0=run_state.get("use_last_after_inner0", False),
        inner_iter=int(level_state["current_iter"]) + 1,
        n_t=n_t,
    )
    prepare_breakdown["keep_union"] = time.perf_counter() - t0
    prepare_snapshot["use_last_merged_len"] = int(len(keep))

    if run_state.get("repair_coverage", False):
        t0 = time.perf_counter()
        keep, coverage_fix = repair_keep_coverage(solver, keep, level_s.points, level_t.points)
        prepare_breakdown["coverage_repair"] = time.perf_counter() - t0
    else:
        coverage_fix = {"added": 0, "rows_missing": 0, "cols_missing": 0}
    prepare_snapshot["coverage_added"] = int(coverage_fix["added"])
    prepare_snapshot["rows_missing"] = int(coverage_fix["rows_missing"])
    prepare_snapshot["cols_missing"] = int(coverage_fix["cols_missing"])

    t0 = time.perf_counter()
    keep_coord = grid_decode_keep_to_coord(
        keep,
        resolution_now=int(round(np.sqrt(n_t))),
        resolution_last=grid_keepcoord_resolution_last(
            level_state=level_state,
            inner_iter=int(level_state["current_iter"]),
        ),
    )
    prepare_breakdown["keep_coord"] = time.perf_counter() - t0

    if np.array_equal(keep, keep_last):
        keep_seed = keep_last
        y_vals = np.asarray(level_state["y_solution_last"]["y"], dtype=np.float32)

    t0 = time.perf_counter()
    y_init = {
        "y": build_y_init_exact_from_keep(
            solver,
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


def merge_with_use_last(solver, **kwargs):
    keep = np.asarray(kwargs["keep"], dtype=np.int64)
    use_last = bool(kwargs["use_last"])
    use_last_after_inner0 = bool(kwargs["use_last_after_inner0"])
    inner_iter = int(kwargs["inner_iter"])
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
            solver.keep_last = keep.copy()
            return keep
        if solver.keep_last is not None and len(solver.keep_last) > 0:
            keep = _sorted_union(keep, np.asarray(solver.keep_last, dtype=np.int64))
        solver.keep_last = keep.copy()
        return keep

    if solver.keep_last is not None and len(solver.keep_last) > 0:
        keep = _sorted_union(keep, np.asarray(solver.keep_last, dtype=np.int64))
    solver.keep_last = keep.copy()
    return keep


def prepare_same_level_support(solver, *, y_solution_last, n_s: int, n_t: int):
    keep_last = np.asarray(y_solution_last["keep"], dtype=np.int64)
    y_last = np.asarray(y_solution_last["y"], dtype=np.float32)
    nnz_thr = grid_nnz_threshold()
    nonzero_mask = np.abs(y_last) > nnz_thr
    keep_seed = keep_last[nonzero_mask]
    y_vals = y_last[nonzero_mask]
    if keep_seed.size == 0:
        solver._grid_same_level_keep_seed = np.empty(0, dtype=np.int64)
        return keep_seed, y_vals

    res_s = int(round(np.sqrt(n_s)))
    res_t = int(round(np.sqrt(n_t)))
    if res_s * res_s != n_s or res_t * res_t != n_t or res_s != res_t:
        solver._grid_same_level_keep_seed = keep_seed
        return keep_seed, y_vals

    topk_expand_gpu, fine_dualOT_delete_numba, keep2keepcoord = _load_mgpd_same_level_ops()
    if topk_expand_gpu is None or fine_dualOT_delete_numba is None or keep2keepcoord is None:
        solver._grid_same_level_keep_seed = keep_seed
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
            fields=grid_mgpd_keepcoord_fields(),
        )
        expanded_vals = fine_dualOT_delete_numba(
            y_val=y_last,
            y_keep=keep_last,
            coord_rec=coord,
            resolution_now=res_s,
            resolution_last=resolution_last,
        )
        solver._grid_same_level_keep_seed = expanded_keep
        return np.asarray(expanded_keep, dtype=np.int64), np.asarray(expanded_vals, dtype=np.float32)
    except Exception:
        solver._grid_same_level_keep_seed = keep_seed
        return keep_seed, y_vals


def build_y_init_exact_from_keep(
    solver,
    *,
    y_solution_last,
    keep,
    n_s: int,
    n_t: int,
    fallback,
    resolution_last=None,
):
    del solver
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
                fields=grid_mgpd_keepcoord_fields(),
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
        {
            "y": np.asarray(fallback["y"], dtype=np.float32),
            "keep": np.asarray(fallback["keep"], dtype=np.int64),
        },
        keep_arr,
    )


def expand_keep_local(solver, keep, y_vals, n_s: int, n_t: int):
    del solver
    keep_arr = np.asarray(keep, dtype=np.int64)
    if keep_arr.size == 0:
        return keep_arr
    res_s = int(round(np.sqrt(n_s)))
    res_t = int(round(np.sqrt(n_t)))
    if res_s != res_t or res_s * res_s != n_s or res_t * res_t != n_t:
        return keep_arr
    build_shield = _load_mgpd_shield_builder()
    if build_shield is not None:
        try:
            keep_exact = build_shield(
                int(res_s),
                np.asarray(y_vals, dtype=np.float32),
                keep_arr,
                return_gpu=False,
                ifnaive=False,
                MultiShield=True,
            )
            return np.asarray(keep_exact, dtype=np.int64)
        except Exception:
            pass
    return build_grid_shield(res_s, y_vals, keep_arr)


def repair_keep_coverage(solver, keep, points_s, points_t):
    del solver
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

    rows, cols = grid_keep_to_rows_cols(keep_arr, n_t)
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
