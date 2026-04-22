from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None


def _cupy_available() -> bool:
    return cp is not None and bool(cp.is_available())


@lru_cache(maxsize=16)
def _build_grid_infeas_kernel_fp32(
    cost_type: str = "L2",
    p: int = 2,
    tile: int = 8,
    block_tile: int = 32,
):
    if cp is None:
        raise RuntimeError("CuPy is not available")

    if cost_type == "L2":
        cost_expr = "(di + dj + 1e-8f)"
    elif cost_type == "L1":
        cost_expr = "(fabsf(d0) + fabsf(d1) + 1e-8f)"
    elif cost_type == "Linf":
        cost_expr = "(fmaxf(fabsf(d0), fabsf(d1)) + 1e-8f)"
    elif cost_type == "Lp":
        cost_expr = f"(powf(fabsf(d0), {p}) + powf(fabsf(d1), {p}) + 1e-8f)"
    else:
        raise ValueError(f"Unknown cost_type={cost_type}")

    code = rf"""
    extern "C" __global__
    void infeas_fp32(
        const float* __restrict__ u,
        const float* __restrict__ v,
        const int r,
        const float inv_scale,
        float* diff_sum)
    {{
        const int T = {tile};
        const int K = {block_tile};
        int bi = blockIdx.y, bj = blockIdx.x;
        int ti = threadIdx.y, tj = threadIdx.x;
        int i1 = bi * T + ti;
        int j1 = bj * T + tj;
        if (i1 >= r || j1 >= r) return;
        float uij = u[i1 * r + j1];
        float local = 0.0f;
        extern __shared__ float vsh[];
        for (int i2t = 0; i2t < r; i2t += K) {{
            for (int j2t = 0; j2t < r; j2t += K) {{
                for (int li = ti; li < K; li += T) {{
                    for (int lj = tj; lj < K; lj += T) {{
                        int gi = i2t + li;
                        int gj = j2t + lj;
                        vsh[li * K + lj] = (gi < r && gj < r) ? v[gi * r + gj] : 0.0f;
                    }}
                }}
                __syncthreads();
                for (int li = 0; li < K; ++li) {{
                    int gi = i2t + li;
                    if (gi >= r) break;
                    float d0 = float(i1 - gi);
                    float di = d0 * d0;
                    for (int lj = 0; lj < K; ++lj) {{
                        int gj = j2t + lj;
                        if (gj >= r) break;
                        float d1 = float(j1 - gj);
                        float dj = d1 * d1;
                        float cost = {cost_expr} * inv_scale;
                        float tmp = uij + vsh[li * K + lj] - cost;
                        if (tmp > 0.0f) local += tmp * tmp;
                    }}
                }}
                __syncthreads();
            }}
        }}
        __shared__ float buf[{tile * tile}];
        int tid = ti * T + tj;
        buf[tid] = local;
        __syncthreads();
        for (int s = {tile * tile}/2; s > 0; s >>= 1) {{
            if (tid < s) buf[tid] += buf[tid + s];
            __syncthreads();
        }}
        if (tid == 0) atomicAdd(diff_sum, buf[0]);
    }}
    """
    shared_sz = block_tile * block_tile * 4
    kernel = cp.RawKernel(code, "infeas_fp32", options=("-std=c++17",), backend="nvrtc")
    return kernel, (tile, tile), shared_sz


def grid_c_norm_direct(r: int, cost_type: str = "L2") -> float:
    if cost_type == "L2":
        table = {
            2: 1.224744871392,
            4: 6.442049363363,
            8: 27.303845882952,
            16: 110.749717832598,
            32: 444.532900919606,
            64: 1779.665558468781,
            128: 7120.196170050373,
            256: 28482.318611728224,
            512: 113930.80837727779,
            1024: 455724.7674391376,
            2048: 1822900.6036979337,
        }
    elif cost_type == "L1":
        table = {
            2: 2.449489742783,
            4: 11.401754250991,
            8: 47.180504448342,
            16: 190.28925350634,
            32: 762.722754347869,
            64: 3052.456387894838,
            128: 12211.390829876833,
            256: 48847.12857476885,
            512: 195390.0795485789,
        }
    elif cost_type == "Linf":
        table = {
            2: 1.732050807569,
            4: 7.937253933194,
            8: 32.726136343907,
            16: 131.874940758281,
            32: 528.468542110124,
            64: 2114.842547330652,
            128: 8460.338468406568,
            256: 33842.32212777368,
            512: 135370.25675900892,
        }
    else:
        raise NotImplementedError(f"{cost_type} is not supported in grid_c_norm_direct")
    if r not in table:
        raise ValueError(f"r={r} is not supported in grid_c_norm_direct")
    return float(table[r])


def dualOT_primal_infeas_grid_cpu(
    x: np.ndarray,
    r: int,
    *,
    cost_type: str = "L2",
    p: int = 2,
) -> float:
    n = int(r) * int(r)
    x64 = np.asarray(x, dtype=np.float64)
    u = x64[:n].reshape(r, r)
    v = x64[n:].reshape(r, r)

    if cost_type == "L2":
        inv_scale = 1.0 / float(r * r)
    elif cost_type in {"L1", "Linf"}:
        inv_scale = 1.0 / float(r)
    elif cost_type == "Lp":
        inv_scale = 1.0 / float(r**p)
    else:
        raise NotImplementedError(f"Unknown cost_type={cost_type}")

    diff_sum = 0.0
    for i1 in range(r):
        for j1 in range(r):
            uij = u[i1, j1]
            for i2 in range(r):
                di = float((i1 - i2) * (i1 - i2))
                for j2 in range(r):
                    if cost_type == "L2":
                        cost = (di + float((j1 - j2) * (j1 - j2)) + 1e-8) * inv_scale
                    elif cost_type == "L1":
                        cost = (abs(i1 - i2) + abs(j1 - j2) + 1e-8) * inv_scale
                    elif cost_type == "Linf":
                        cost = (max(abs(i1 - i2), abs(j1 - j2)) + 1e-8) * inv_scale
                    else:
                        cost = (
                            abs(i1 - i2) ** p + abs(j1 - j2) ** p + 1e-8
                        ) * inv_scale
                    tmp = uij + v[i2, j2] - cost
                    if tmp > 0.0:
                        diff_sum += tmp * tmp
    return math.sqrt(diff_sum) / (1.0 + grid_c_norm_direct(r, cost_type))


def dualOT_primal_infeas_grid_gpu_fp32(
    x: np.ndarray,
    r: int,
    *,
    cost_type: str = "L2",
    p: int = 2,
    tile: int = 8,
    block_tile: int = 32,
) -> float:
    if not _cupy_available():
        raise RuntimeError("CuPy is not available")

    kernel, block, shared_mem = _build_grid_infeas_kernel_fp32(cost_type, p, tile, block_tile)
    n = int(r) * int(r)
    x32 = np.asarray(x, dtype=np.float32)
    u_gpu = cp.asarray(x32[:n])
    v_gpu = cp.asarray(x32[n:])
    diff_gpu = cp.zeros(1, dtype=cp.float32)

    if cost_type == "L2":
        inv_scale = np.float32(1.0 / float(r * r))
    elif cost_type in {"L1", "Linf"}:
        inv_scale = np.float32(1.0 / float(r))
    elif cost_type == "Lp":
        inv_scale = np.float32(1.0 / float(r**p))
    else:
        raise ValueError(f"Unknown cost_type={cost_type}")

    gx = (int(r) + tile - 1) // tile
    gy = (int(r) + tile - 1) // tile
    kernel((gx, gy), block, (u_gpu, v_gpu, int(r), inv_scale, diff_gpu), shared_mem=shared_mem)
    cp.cuda.Stream.null.synchronize()
    diff = float(diff_gpu.get()[0])
    if diff < 0.0:
        raise ValueError(f"grid primal infeas kernel returned negative diff={diff}")
    return math.sqrt(diff) / (1.0 + grid_c_norm_direct(r, cost_type))


def dualOT_primal_infeas_grid_auto(
    x: np.ndarray,
    r: int,
    *,
    cost_type: str = "L2",
    p: int = 2,
    use_cupy: bool = True,
) -> float:
    if use_cupy and _cupy_available() and int(r) >= 8:
        try:
            return dualOT_primal_infeas_grid_gpu_fp32(
                x,
                r,
                cost_type=cost_type,
                p=p,
            )
        except Exception:
            pass
    return dualOT_primal_infeas_grid_cpu(x, r, cost_type=cost_type, p=p)


def compute_primal_infeas(solver, x_dual: np.ndarray, level_s, level_t) -> float:
    n_s = int(len(level_s.points))
    n_t = int(len(level_t.points))
    if n_s != n_t:
        return float("inf")
    resolution = int(round(np.sqrt(n_s)))
    if resolution * resolution != n_s:
        return float("inf")

    cost_type_raw = str(getattr(solver, "_cost_type", "l2^2")).strip().lower()
    if cost_type_raw in {"l2^2", "sqeuclidean", "l2sq"}:
        cost_type = "L2"
    elif cost_type_raw in {"l2", "euclidean"}:
        raise ValueError("grid mode does not support 'l2'; use 'l2^2' for squared Euclidean distance.")
    elif cost_type_raw == "l1":
        cost_type = "L1"
    elif cost_type_raw == "linf":
        cost_type = "Linf"
    else:
        return float("inf")

    try:
        return float(
            dualOT_primal_infeas_grid_auto(
                x_dual,
                resolution,
                cost_type=cost_type,
                p=int(getattr(solver, "_cost_p", 2)),
                use_cupy=True,
            )
        )
    except Exception:
        return float("inf")
