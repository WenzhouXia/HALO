from __future__ import annotations

import numpy as np


def coarsen_histogram2d(hist: np.ndarray) -> np.ndarray:
    arr = np.asarray(hist, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("coarsen_histogram2d only supports square 2D histograms.")
    res = int(arr.shape[0])
    if res % 2 != 0:
        raise ValueError("resolution must be even for 2x2 coarsening.")
    return arr.reshape(res // 2, 2, res // 2, 2).sum(axis=(1, 3))


def prolong_grid_dual(coarse_dual: np.ndarray, fine_resolution: int) -> np.ndarray:
    fine_res = int(fine_resolution)
    coarse_res = fine_res // 2
    dual = np.asarray(coarse_dual, dtype=np.float32)
    n_coarse = coarse_res * coarse_res
    if dual.ndim != 1 or dual.shape[0] != 2 * n_coarse:
        raise ValueError("coarse_dual must have length 2 * (fine_resolution/2)^2")
    u = dual[:n_coarse].reshape(coarse_res, coarse_res)
    v = dual[n_coarse:].reshape(coarse_res, coarse_res)
    u_fine = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1).reshape(-1)
    v_fine = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1).reshape(-1)
    return np.concatenate([u_fine, v_fine]).astype(np.float32, copy=False)
