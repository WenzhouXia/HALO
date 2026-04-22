from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .costs import _grid_resolution_from_points

try:
    import cupy as cp
except Exception:  # pragma: no cover - imported lazily by callers
    cp = None


def _require_cupy():
    """
    CN: 确保当前环境可用 CuPy，否则立即报错。
    EN: Ensure CuPy is available in the current environment, otherwise raise immediately.
    """
    if cp is None:  # pragma: no cover
        raise RuntimeError("CuPy is required for grid violation checking")

def launch_timed(kernel, grid, block, args, *, shared_mem=0):
    """
    CN: 启动一个 CuPy kernel 并返回同步后的执行耗时。
    EN: Launch a CuPy kernel and return its synchronized execution time.
    """
    start_evt = cp.cuda.Event()
    end_evt = cp.cuda.Event()
    start_evt.record()
    kernel(grid, block, args, shared_mem=shared_mem)
    end_evt.record()
    end_evt.synchronize()
    return cp.cuda.get_elapsed_time(start_evt, end_evt) / 1000.0


def _build_check_kernel_abs_fp32(
    cost_type: str = "L2",
    p: int = 2,
    T: int = 8,
    K: int = 32,
):
    """
    CN: 构建基于块遍历的 FP32 violation 检查 RawKernel。
    EN: Build the FP32 blockwise RawKernel used for violation checking.
    """
    if cost_type == "L2":
        cost_expr = "(di + dj + 1e-8f)"
    elif cost_type == "L1":
        cost_expr = "(fabsf(d0) + fabsf(d1) + 1e-8f)"
    elif cost_type == "Linf":
        cost_expr = "(fmaxf(fabsf(d0), fabsf(d1)) + 1e-8f)"
    elif cost_type == "Lp":
        cost_expr = f"(powf(fabsf(d0), {p}) + powf(fabsf(d1), {p}) + 1e-8f)"
    else:  # pragma: no cover - validated by caller
        raise ValueError(cost_type)

    code = rf"""
    extern "C" __global__
    void check_abs(const float* __restrict__ u,
                   const float* __restrict__ v,
                   const int    r,
                   const float  eps,
                   const float  inv_scale,
                   long long*   out_idx,
                   float*       out_slack,
                   const unsigned long long max_keep,
                   const unsigned int write,
                   unsigned long long* counter)
    {{
        const int T = {T}, K = {K};
        int bi = blockIdx.y, bj = blockIdx.x;
        int ti = threadIdx.y, tj = threadIdx.x;
        int i1 = bi * T + ti, j1 = bj * T + tj;
        if (i1 >= r || j1 >= r) return;

        float uij = u[i1 * r + j1];
        unsigned int local_cnt = 0;
        extern __shared__ float vsh[];

        for (int i2t = 0; i2t < r; i2t += K) {{
            for (int j2t = 0; j2t < r; j2t += K) {{
                for (int li = ti; li < K; li += T)
                    for (int lj = tj; lj < K; lj += T) {{
                        int gi = i2t + li, gj = j2t + lj;
                        vsh[li * K + lj] = (gi < r && gj < r) ? v[gi * r + gj] : 0.f;
                    }}
                __syncthreads();
                for (int li = 0; li < K; ++li) {{
                    float d0 = float(i1 - (i2t + li));
                    float di = d0 * d0;
                    for (int lj = 0; lj < K; ++lj) {{
                        float d1 = float(j1 - (j2t + lj));
                        float dj = d1 * d1;
                        float cost = {cost_expr} * inv_scale;
                        float val = uij + vsh[li * K + lj] - cost;
                        if (val > eps) {{
                            if (!write) {{
                                ++local_cnt;
                            }} else {{
                                unsigned long long pos = atomicAdd(counter, 1ULL);
                                if (pos < max_keep) {{
                                    long long idx =
                                      (long long)(i1 * r + j1) * (r * r)
                                      + (long long)(i2t + li) * r
                                      + (long long)(j2t + lj);
                                    out_idx[pos] = idx;
                                    out_slack[pos] = val;
                                }}
                            }}
                        }}
                    }}
                }}
                __syncthreads();
            }}
        }}

        if (!write) {{
            __shared__ unsigned int buf[{T * T}];
            int tid = ti * T + tj;
            buf[tid] = local_cnt;
            __syncthreads();
            for (int s = {T * T} / 2; s > 0; s >>= 1) {{
                if (tid < s) buf[tid] += buf[tid + s];
                __syncthreads();
            }}
            if (tid == 0) {{
                unsigned long long total =
                  atomicAdd(counter, (unsigned long long)buf[0]) + buf[0];
                if (total > max_keep) *counter = max_keep + 1ULL;
            }}
        }}
    }}
    """
    shared = K * K * 4
    ker = cp.RawKernel(
        code,
        "check_abs",
        options=("-std=c++17", "-use_fast_math"),
        backend="nvrtc",
    )
    return ker, (T, T), shared


@lru_cache(maxsize=None)
def _get_kernel_cached(cost_type: str, p: int, tile: int, K: int, type: str = "abs"):
    """
    CN: 按参数缓存并返回对应的 GPU violation kernel。
    EN: Cache and return the GPU violation kernel for the given parameter set.
    """
    if type == "abs":
        return _build_check_kernel_abs_fp32(cost_type, p, tile, K)
    raise ValueError(type)


_DT1D_VAL_IDX_CODE = r'''
extern "C" __global__
void dt1d_val_idx(const float* __restrict__ f2d,
                  const int    n,
                  const float  a,
                  float*       g2d,
                  int*         arg2d)
{
    extern __shared__ float sh[];
    float* z = sh;
    int*   v = (int*)&z[n+1];

    int row = blockIdx.x;
    const float* f = f2d + row * n;
    float*       g = g2d + row * n;
    int*         arg = arg2d + row * n;

    int k = 0;
    v[0] = 0;
    z[0] = -1e30f;
    z[1] = +1e30f;
    for (int q = 1; q < n; ++q) {
        float fq = f[q] + a * q * q;
        float s = (fq - (f[v[k]] + a * v[k] * v[k])) / (2 * a * (q - v[k]));
        while (s <= z[k]) {
            --k;
            s = (fq - (f[v[k]] + a * v[k] * v[k])) / (2 * a * (q - v[k]));
        }
        ++k;
        v[k] = q;
        z[k] = s;
        z[k+1] = +1e30f;
    }

    k = 0;
    for (int i = 0; i < n; ++i) {
        while (z[k+1] < i) ++k;
        arg[i] = v[k];
        int d = i - v[k];
        g[i] = f[v[k]] + a * d * d;
    }
}
'''


_dt1d_val_idx_kernel = (
    cp.RawKernel(_DT1D_VAL_IDX_CODE, "dt1d_val_idx", options=("-std=c++17",), backend="nvrtc")
    if cp is not None
    else None
)


def dt2d_val_idx(v_mat: "cp.ndarray", inv_r2: float):
    """
    CN: 对二维势函数做距离变换并返回每个位置对应的最优索引映射。
    EN: Run a 2D distance transform on the potential and return the argmin index maps.
    """
    r = v_mat.shape[0]
    a = inv_r2
    shmem = (r + 1) * 4 + r * 4

    f2d = -v_mat
    g1 = cp.empty_like(f2d)
    arg1 = cp.empty((r, r), dtype=cp.int32)
    _dt1d_val_idx_kernel(
        (r,),
        (1,),
        (f2d.reshape(-1), np.int32(r), np.float32(a), g1.reshape(-1), arg1.reshape(-1)),
        shared_mem=shmem,
    )

    g1T = g1.T
    g2T = cp.empty_like(g1T)
    arg2T = cp.empty((r, r), dtype=cp.int32)
    _dt1d_val_idx_kernel(
        (r,),
        (1,),
        (g1T.reshape(-1), np.int32(r), np.float32(a), g2T.reshape(-1), arg2T.reshape(-1)),
        shared_mem=shmem,
    )
    arg2 = arg2T.T
    k_map = arg2
    j_idx = cp.arange(r, dtype=cp.int32)[None, :]
    l_map = arg1[k_map, j_idx]
    return k_map, l_map


def _build_check_kernel_abs_dynctr_chunked(cost_type: str, p: int, T: int, R: int):
    """
    CN: 构建围绕局部中心搜索的分块 GPU violation 检查 kernel。
    EN: Build the chunked GPU violation kernel that searches around local center candidates.
    """
    if cost_type == "L2":
        cost_expr = "(d0*d0 + d1*d1 + 1e-8f)"
    elif cost_type == "L1":
        cost_expr = "(fabsf(d0) + fabsf(d1) + 1e-8f)"
    elif cost_type == "Linf":
        cost_expr = "(fmaxf(fabsf(d0), fabsf(d1)) + 1e-8f)"
    else:
        cost_expr = f"(powf(fabsf(d0), {p}) + powf(fabsf(d1), {p}) + 1e-8f)"

    code = fr'''
    extern "C" __global__
    void check_abs_dynctr_chunked(
        const float* __restrict__ u,
        const float* __restrict__ v,
        const int     r,
        const int*    __restrict__ k_map,
        const int*    __restrict__ l_map,
        const int     R,
        const float   eps,
        const float   inv_scale,
        unsigned long long* __restrict__ out_idx,
        float*               __restrict__ out_slack,
        const unsigned long long max_keep,
        const unsigned int  write,
        unsigned long long* counter,
        const int     row_offset)
    {{
        const int T = {T};
        int bi = blockIdx.y, bj = blockIdx.x;
        int ti = threadIdx.y, tj = threadIdx.x;
        int i1 = row_offset + bi * T + ti;
        int j1 = bj * T + tj;
        if (i1 < 0 || i1 >= r || j1 < 0 || j1 >= r) return;

        int idx_u = i1 * r + j1;
        int kc = k_map[idx_u];
        int lc = l_map[idx_u];

        float uij = u[idx_u];
        unsigned int local_cnt = 0;

        for (int dk = -R; dk <= R; ++dk) {{
            int kk = kc + dk;
            if (kk < 0 || kk >= r) continue;
            float d0 = float(i1 - kk);
            for (int dl = -R; dl <= R; ++dl) {{
                int ll = lc + dl;
                if (ll < 0 || ll >= r) continue;
                float d1 = float(j1 - ll);
                float slack = uij + v[kk*r + ll] - ({cost_expr}) * inv_scale;
                if (slack > eps) {{
                    if (!write) {{
                        ++local_cnt;
                    }} else {{
                        unsigned long long pos = atomicAdd(counter, 1ULL);
                        if (pos < max_keep) {{
                            unsigned long long idx =
                              (unsigned long long)idx_u * (unsigned long long)(r*r)
                              + (unsigned long long)kk * (unsigned long long)r
                              + (unsigned long long)ll;
                            out_idx[pos] = idx;
                            out_slack[pos] = slack;
                        }}
                    }}
                }}
            }}
        }}

        if (!write) {{
            extern __shared__ unsigned int buf[];
            int tid = ti * T + tj;
            buf[tid] = local_cnt;
            __syncthreads();
            for (int s = (T*T)>>1; s > 0; s >>= 1) {{
                if (tid < s) buf[tid] += buf[tid+s];
                __syncthreads();
            }}
            if (tid == 0) {{
                atomicAdd(counter, (unsigned long long)buf[0]);
            }}
        }}
    }}
    '''
    shared = T * T * np.dtype(np.uint32).itemsize
    ker = cp.RawKernel(
        code,
        "check_abs_dynctr_chunked",
        options=("-std=c++17",),
        backend="nvrtc",
    )
    return ker, (T, T), shared


def check_constraint_violations_gpu_abs_lessGPU(
    x_flat_np: np.ndarray,
    resolution: int,
    eps: float,
    *,
    cost_type: str = "L2",
    p: int = 2,
    tile: int = 8,
    K: int = 32,
    max_keep: int | None = None,
    return_gpu: bool = False,
    rows_per_pass: int | None = None,
    mem_budget_mb: int = 1500,
):
    """
    CN: 用全局分块 GPU 路径检测网格约束违反并返回 top violations。
    EN: Detect grid constraint violations with the global blockwise GPU path and return top violations.
    """
    _require_cupy()
    if max_keep is None:
        max_keep = np.iinfo(np.int32).max
    r = int(resolution)
    N = r * r
    inv_f = (
        np.float32(1.0 / (r * r)) if cost_type == "L2"
        else np.float32(1.0 / r) if cost_type in ("L1", "Linf")
        else np.float32(1.0 / (r ** p))
    )
    ker, blk, shm = _get_kernel_cached(cost_type, p, tile, K, type="abs")
    gx = (r + tile - 1) // tile

    if rows_per_pass is None:
        rows_per_pass = max(tile, min(r, int((mem_budget_mb * 1e6) / (12 * r * r))))
        rows_per_pass = math.ceil(rows_per_pass / tile) * tile

    x_gpu = cp.asarray(x_flat_np, dtype=cp.float32)
    u_gpu, v_gpu = x_gpu[:N], x_gpu[N:]
    g_idx = cp.empty(0, cp.uint64)
    g_slk = cp.empty(0, cp.float32)
    thr = -cp.inf

    for i_start in range(0, r, rows_per_pass):
        i_end = min(i_start + rows_per_pass, r)
        rows_this = i_end - i_start
        gy_batch = (rows_this + tile - 1) // tile
        i_off_blk = i_start // tile

        ctr = cp.zeros(1, dtype=cp.uint64)
        launch_timed(
            ker,
            (gx, gy_batch),
            blk,
            (
                u_gpu,
                v_gpu,
                r,
                np.float32(eps),
                inv_f,
                None,
                None,
                cp.uint64(0xFFFFFFFFFFFFFFFF),
                cp.uint32(0),
                ctr,
                cp.int32(i_off_blk),
            ),
            shared_mem=shm,
        )
        tot = int(ctr.get().item())
        if tot == 0:
            continue

        idx_gpu = cp.empty(tot, dtype=cp.uint64)
        slk_gpu = cp.empty(tot, dtype=cp.float32)
        ctr.fill(0)
        launch_timed(
            ker,
            (gx, gy_batch),
            blk,
            (
                u_gpu,
                v_gpu,
                r,
                np.float32(eps),
                inv_f,
                idx_gpu,
                slk_gpu,
                cp.uint64(tot),
                cp.uint32(1),
                ctr,
                cp.int32(i_off_blk),
            ),
            shared_mem=shm,
        )

        if thr != -cp.inf:
            mask = slk_gpu > thr
            if not mask.all():
                idx_gpu = idx_gpu[mask]
                slk_gpu = slk_gpu[mask]
        g_idx = cp.concatenate((g_idx, idx_gpu))
        g_slk = cp.concatenate((g_slk, slk_gpu))

        if g_slk.size > max_keep:
            order_glb = cp.lexsort(cp.stack((g_idx, -g_slk)))
            sel = order_glb[:max_keep]
            g_idx = g_idx[sel]
            g_slk = g_slk[sel]
            thr = g_slk[-1]
        else:
            thr = -cp.inf

    if g_slk.size == 0:
        empty = cp.empty(0, cp.int64) if return_gpu else np.empty(0, np.int64)
        return empty, np.float32(0.0)

    order = cp.lexsort(cp.stack((g_idx, -g_slk)))
    g_idx = g_idx[order]
    g_slk = g_slk[order]
    uniq_idx, inv = cp.unique(g_idx, return_inverse=True)
    uniq_slk = cp.zeros_like(uniq_idx, dtype=g_slk.dtype)
    cp.maximum.at(uniq_slk, inv, g_slk)
    if uniq_slk.size > max_keep:
        top = cp.lexsort(cp.stack((uniq_idx, -uniq_slk)))[:max_keep]
        uniq_idx = uniq_idx[top]
        uniq_slk = uniq_slk[top]
    final_ord = cp.lexsort(cp.stack((uniq_idx, -uniq_slk)))
    uniq_idx = uniq_idx[final_ord]
    uniq_slk = uniq_slk[final_ord]

    idx_out = uniq_idx.view(cp.int64) if return_gpu else cp.asnumpy(uniq_idx.view(cp.int64))
    crit_slack = np.float32(cp.asnumpy(uniq_slk[-1]))
    return idx_out, crit_slack


def check_constraint_violations_gpu_abs_lessGPU_local_2DT(
    x_flat_np: np.ndarray,
    resolution: int,
    eps: float,
    *,
    cost_type: str = "L2",
    p: int = 2,
    tile: int = 8,
    max_keep: int | None = None,
    return_gpu: bool = False,
    rows_per_pass: int | None = None,
    mem_budget_mb: int = 1500,
):
    """
    CN: 用 2D 距离变换引导的局部 GPU 路径检测网格约束违反。
    EN: Detect grid constraint violations with the 2D-transform-guided local GPU path.
    """
    _require_cupy()
    if max_keep is None:
        max_keep = np.iinfo(np.int32).max
    r = int(resolution)
    N = r * r
    inv_f = (
        np.float32(1.0 / (r * r)) if cost_type == "L2"
        else np.float32(1.0 / r) if cost_type in ("L1", "Linf")
        else np.float32(1.0 / (r ** p))
    )
    R = int(math.ceil(math.sqrt(r)))

    x_gpu = cp.asarray(x_flat_np, dtype=cp.float32)
    u_gpu, v_gpu = x_gpu[:N], x_gpu[N:]
    v_mat = v_gpu.reshape(r, r)
    k_map, l_map = dt2d_val_idx(v_mat, float(inv_f))
    kc_gpu = cp.asarray(k_map.reshape(-1), dtype=cp.int32)
    lc_gpu = cp.asarray(l_map.reshape(-1), dtype=cp.int32)

    ker, blk, shm = _build_check_kernel_abs_dynctr_chunked(cost_type, p, tile, R)
    gx = (r + tile - 1) // tile

    if rows_per_pass is None:
        rows_per_pass = max(tile, min(r, int((mem_budget_mb * 1e6) / (12 * r * r))))
        rows_per_pass = math.ceil(rows_per_pass / tile) * tile

    g_idx = cp.empty(0, cp.uint64)
    g_slk = cp.empty(0, cp.float32)
    thr = -cp.inf

    for i0 in range(0, r, rows_per_pass):
        rows = min(rows_per_pass, r - i0)
        gy = (rows + tile - 1) // tile

        ctr1 = cp.zeros(1, dtype=cp.uint64)
        ker(
            (gx, gy),
            blk,
            (
                u_gpu,
                v_gpu,
                np.int32(r),
                kc_gpu,
                lc_gpu,
                np.int32(R),
                np.float32(eps),
                inv_f,
                None,
                None,
                cp.uint64(max_keep),
                cp.uint32(0),
                ctr1,
                np.int32(i0),
            ),
            shared_mem=shm,
        )
        cp.cuda.Stream.null.synchronize()
        tot = int(ctr1.get())
        if tot == 0:
            continue

        ctr2 = cp.zeros(1, dtype=cp.uint64)
        idx_all = cp.empty(tot, dtype=cp.uint64)
        slk_all = cp.empty(tot, dtype=cp.float32)
        ker(
            (gx, gy),
            blk,
            (
                u_gpu,
                v_gpu,
                np.int32(r),
                kc_gpu,
                lc_gpu,
                np.int32(R),
                np.float32(eps),
                inv_f,
                idx_all,
                slk_all,
                cp.uint64(tot),
                cp.uint32(1),
                ctr2,
                np.int32(i0),
            ),
            shared_mem=shm,
        )
        cp.cuda.Stream.null.synchronize()

        if tot > max_keep:
            order_blk = cp.lexsort(cp.stack((idx_all, -slk_all)))
            sel = order_blk[:max_keep]
            idx_block = idx_all[sel]
            slk_block = slk_all[sel]
        else:
            idx_block = idx_all
            slk_block = slk_all

        if thr != -cp.inf:
            mask = slk_block > thr
            idx_block = idx_block[mask]
            slk_block = slk_block[mask]

        g_idx = cp.concatenate((g_idx, idx_block))
        g_slk = cp.concatenate((g_slk, slk_block))

        if g_slk.size > max_keep:
            order_glb = cp.lexsort(cp.stack((g_idx, -g_slk)))
            sel = order_glb[:max_keep]
            g_idx = g_idx[sel]
            g_slk = g_slk[sel]
            thr = g_slk[-1]
        else:
            thr = -cp.inf

    order = cp.lexsort(cp.stack((g_idx, -g_slk)))
    g_idx, g_slk = g_idx[order], g_slk[order]
    uniq, inv = cp.unique(g_idx, return_inverse=True)
    slk_u = cp.zeros_like(uniq, dtype=g_slk.dtype)
    cp.maximum.at(slk_u, inv, g_slk)
    if slk_u.size > max_keep:
        sel = cp.argpartition(-slk_u, max_keep - 1)[:max_keep]
        uniq, slk_u = uniq[sel], slk_u[sel]
    fo = cp.lexsort(cp.stack((uniq, -slk_u)))
    uniq, slk_u = uniq[fo], slk_u[fo]

    idx_out = uniq.view(cp.int64) if return_gpu else cp.asnumpy(uniq.view(cp.int64))
    crit = np.float32(cp.asnumpy(slk_u[-1]))
    return idx_out, crit


def grid_violation_candidates(
    dual_uv: np.ndarray,
    points_s: np.ndarray,
    points_t: np.ndarray,
    *,
    cost_type: str,
    p: int = 2,
    vd_thr: float = 0.25,
    max_candidates: int | None = None,
) -> np.ndarray:
    """
    CN: 为规则网格 dual 解选择 violation 候选，并在不满足 GPU 条件时直接报错。
    EN: Select violation candidates for a regular-grid dual solution and fail fast when GPU requirements are not met.
    """
    _require_cupy()
    dual = np.asarray(dual_uv, dtype=np.float32)
    n_s = int(np.asarray(points_s).shape[0])
    n_t = int(np.asarray(points_t).shape[0])
    if dual.ndim != 1 or dual.shape[0] != n_s + n_t:
        raise ValueError("dual_uv must have length n_source + n_target")

    resolution_sq_s = int(round(np.sqrt(n_s)))
    resolution_sq_t = int(round(np.sqrt(n_t)))
    resolution_s = _grid_resolution_from_points(points_s)
    resolution_t = _grid_resolution_from_points(points_t)
    if (
        n_s == n_t
        and resolution_sq_s == resolution_sq_t
        and resolution_sq_s * resolution_sq_s == n_s
        and resolution_s == resolution_t
        and str(cost_type).lower() in {"l2^2", "l2", "sqeuclidean", "l2sq"}
        and int(p) == 2
        and max_candidates is not None
    ):
        eps = 0.0
        r = int(resolution_sq_s)
        max_keep = int(max_candidates)
        if cp is not None:
            if r >= 512:
                keep, _ = check_constraint_violations_gpu_abs_lessGPU_local_2DT(
                    dual,
                    r,
                    eps,
                    tile=32,
                    cost_type="L2",
                    max_keep=max_keep,
                    return_gpu=False,
                )
            else:
                keep, _ = check_constraint_violations_gpu_abs_lessGPU(
                    dual,
                    r,
                    eps,
                    tile=32,
                    cost_type="L2",
                    max_keep=max_keep,
                    return_gpu=False,
                )
            keep_arr = np.asarray(keep, dtype=np.int64).reshape(-1)
            max_keep_valid = np.int64(n_s) * np.int64(n_t)
            keep_arr = keep_arr[(keep_arr >= 0) & (keep_arr < max_keep_valid)]
            return np.asarray(np.unique(keep_arr), dtype=np.int64)
    raise RuntimeError(
        "Grid violation checking no longer supports CPU fallback. "
        "Expected square source/target grids with matching resolution, L2^2 cost, p=2, "
        "and an explicit max_candidates budget."
    )
