from __future__ import annotations

from functools import lru_cache

import numpy as np

try:
    import cupy as cp
except Exception:  
    cp = None

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
    if cp is None:  
        raise RuntimeError("cupy is required for grid shielding")
    return cp.RawKernel(CUDA_SRC_ARGMAX_KEY, "pick_argmax_key")

@lru_cache(None)
def _ker_decode_from_key():
    if cp is None:  
        raise RuntimeError("cupy is required for grid shielding")
    return cp.RawKernel(CUDA_SRC_DECODE_KEY, "decode_from_key")

def _pick_t_cupy(r: int, y_keep: np.ndarray, y_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if cp is None:  
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
        parts.append(((idx_a * np.int64(r) + t_row[idx_s].astype(np.int64)) * np.int64(r) + t_col[idx_s].astype(np.int64)))
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(parts).astype(np.int64, copy=False))

def _gpu_unique_inplace(arr):
    if cp is None:  
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
    if cp is None:  
        raise RuntimeError("cupy is required for grid shielding")
    return cp.RawKernel(_SRC_COUNT8, "count8")

@lru_cache(None)
def _kernel_fill_8():
    if cp is None:  
        raise RuntimeError("cupy is required for grid shielding")
    return cp.RawKernel(_SRC_FILL8, "fill8")

def _build_yhat_grid_gpu_8(r: int, t_row: np.ndarray, t_col: np.ndarray) -> np.ndarray:
    if cp is None:  
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
