from __future__ import annotations

import numpy as np

def _grid_resolution_from_points(points: np.ndarray) -> int:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("grid points must have shape (N, 2)")
    axis0 = np.unique(np.asarray(pts[:, 0], dtype=np.float32))
    axis1 = np.unique(np.asarray(pts[:, 1], dtype=np.float32))
    if axis0.size != axis1.size:
        raise ValueError("grid points must form a square grid")
    if axis0.size == 0:
        raise ValueError("grid points must be non-empty")

    
    
    
    
    return int(axis0.size)

def grid_pairwise_cost(
    points_s: np.ndarray,
    points_t: np.ndarray,
    *,
    cost_type: str,
    p: int = 2,
) -> np.ndarray:
    s = np.asarray(points_s, dtype=np.float32)
    t = np.asarray(points_t, dtype=np.float32)
    di = s[:, None, 0] - t[None, :, 0]
    dj = s[:, None, 1] - t[None, :, 1]
    res = max(_grid_resolution_from_points(s), _grid_resolution_from_points(t))
    cost_name = str(cost_type).lower()

    if cost_name == "l2^2":
        return ((di * di + dj * dj) / float(res * res)).astype(np.float32, copy=False)
    if cost_name == "l1":
        return ((np.abs(di) + np.abs(dj)) / float(res)).astype(np.float32, copy=False)
    if cost_name == "linf":
        return (np.maximum(np.abs(di), np.abs(dj)) / float(res)).astype(np.float32, copy=False)
    if cost_name == "lp":
        p_int = int(p)
        scale = float(res ** p_int)
        return ((np.abs(di) ** p_int + np.abs(dj) ** p_int) / scale).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported grid cost_type: {cost_type}")
