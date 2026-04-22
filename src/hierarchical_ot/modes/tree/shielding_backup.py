from __future__ import annotations

"""
Archived tree shielding helpers kept only as backup during the refactor.

This module is not part of the supported runtime path, is not expected to be
covered by tests, and should not be imported by the main tree solver flow.
"""

from typing import Dict

import numpy as np
from numba import njit


@njit(cache=True)
def _dot_njit(a: np.ndarray, b: np.ndarray) -> float:
    s = 0.0
    for i in range(a.shape[0]):
        s += a[i] * b[i]
    return s


def _pick_t_map(y_keep, y_val, n_X, n_Y):
    from .shielding import _pick_t_core_numba

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


def _build_sentinels(level_X, t_map, knn_indices, k_neighbors, level_Y=None):
    nodes_X = level_X.points
    n_X = len(nodes_X)
    dim = nodes_X.shape[1]
    actual_k = knn_indices.shape[1] if len(knn_indices) > 0 else 0

    sentinels_list = []
    shield_pairs = []

    for i_A in range(n_X):
        x_A = nodes_X[i_A]
        neighbors = (
            knn_indices[i_A] if i_A < len(knn_indices) else np.array([], dtype=np.int32)
        )

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


def _fallback_yhat(k_neighbors, level_s, t_map, knn_indices):
    n_s = len(level_s.points)
    pairs = []

    for i_A in range(n_s):
        if i_A < len(knn_indices):
            for i_S in knn_indices[i_A][:k_neighbors]:
                if i_S >= 0 and i_S in t_map:
                    pairs.append((i_A, t_map[i_S]))

    if len(pairs) == 0:
        return np.empty((0, 2), dtype=np.int32)

    arr = np.array(pairs, dtype=np.int32)
    return np.unique(arr.view("V8")).view(np.int32).reshape(-1, 2)


def repair_keep_coverage(
    keep: np.ndarray,
    *,
    n_s: int,
    n_t: int,
    points_s: np.ndarray,
    points_t: np.ndarray,
):
    keep = np.asarray(keep, dtype=np.int64)
    if keep.size == 0:
        return keep, {"rows_missing": n_s, "cols_missing": n_t, "added": 0}

    idx_x = keep // n_t
    idx_y = keep % n_t
    row_counts = np.bincount(idx_x, minlength=n_s)
    col_counts = np.bincount(idx_y, minlength=n_t)

    missing_rows = np.where(row_counts == 0)[0]
    missing_cols = np.where(col_counts == 0)[0]
    if missing_rows.size == 0 and missing_cols.size == 0:
        return keep, {"rows_missing": 0, "cols_missing": 0, "added": 0}

    extra = []
    if n_t > 0:
        for i in missing_rows:
            d2 = np.sum((points_t - points_s[i]) ** 2, axis=1)
            j = int(np.argmin(d2))
            extra.append(int(i) * n_t + j)
    if n_s > 0:
        for j in missing_cols:
            d2 = np.sum((points_s - points_t[j]) ** 2, axis=1)
            i = int(np.argmin(d2))
            extra.append(int(i) * n_t + int(j))

    keep_repaired = np.unique(np.concatenate([keep, np.asarray(extra, dtype=np.int64)]))
    return keep_repaired, {
        "rows_missing": int(missing_rows.size),
        "cols_missing": int(missing_cols.size),
        "added": int(keep_repaired.size - keep.size),
    }


try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def build_Yhat_direct_gpu(
    level_X,
    level_Y,
    t_map: Dict[int, int],
    knn_indices: np.ndarray,
    k_neighbors: int = 10,
    cost_type: str = "L2",
    p: float = 2.0,
    batch_size_X: int = 1024,
    batch_size_Y: int = 4096,
) -> np.ndarray:
    if not HAS_CUPY:
        raise RuntimeError("CuPy is required for GPU Direct Shielding")

    nodes_X = level_X.points
    nodes_Y = level_Y.points
    n_X = len(nodes_X)
    n_Y = len(nodes_Y)

    cost_type_upper = cost_type.upper()
    use_L2_fast = (cost_type_upper == "L2") or (
        cost_type_upper == "LP" and abs(p - 2.0) < 1e-8
    )

    X_gpu = cp.asarray(nodes_X, dtype=cp.float32)
    Y_gpu = cp.asarray(nodes_Y, dtype=cp.float32)

    sentinels = []
    for i_A in range(n_X):
        sentinels_i = []
        if i_A in t_map:
            y_idx = t_map[i_A]
            if 0 <= y_idx < n_Y:
                sentinels_i.append((nodes_Y[y_idx], y_idx))

        if i_A < len(knn_indices):
            for k_idx in knn_indices[i_A][:k_neighbors]:
                if k_idx >= 0 and k_idx not in [s[1] for s in sentinels_i]:
                    if k_idx in t_map:
                        y_t = t_map[k_idx]
                        if 0 <= y_t < n_Y:
                            sentinels_i.append((nodes_Y[y_t], y_t))

        sentinels.append(sentinels_i)

    all_pairs = []
    for x_start in range(0, n_X, batch_size_X):
        x_end = min(x_start + batch_size_X, n_X)
        X_batch = X_gpu[x_start:x_end]

        for y_start in range(0, n_Y, batch_size_Y):
            y_end = min(y_start + batch_size_Y, n_Y)
            Y_batch = Y_gpu[y_start:y_end]

            diff = X_batch[:, None, :] - Y_batch[None, :, :]
            if use_L2_fast or cost_type_upper == "L2":
                C_batch = cp.sum(diff**2, axis=2)
            elif cost_type_upper == "L1":
                C_batch = cp.sum(cp.abs(diff), axis=2)
            elif cost_type_upper == "LINF":
                C_batch = cp.max(cp.abs(diff), axis=2)
            elif cost_type_upper == "LP":
                C_batch = cp.sum(cp.abs(diff) ** p, axis=2)
            else:
                C_batch = cp.sum(diff**2, axis=2)

            for i_A in range(x_start, x_end):
                x_A = nodes_X[i_A]
                sentinels_i = sentinels[i_A]
                if len(sentinels_i) == 0:
                    y_indices = cp.arange(y_start, y_end)
                    for y_B in range(y_end - y_start):
                        all_pairs.append((i_A, int(y_indices[y_B].get())))
                    continue

                i_batch = i_A - x_start
                c_xA_yB = C_batch[i_batch, :]
                for y_B_local in range(y_end - y_start):
                    y_B = y_start + y_B_local
                    shielded = False
                    for x_S, y_S in sentinels_i:
                        if use_L2_fast or cost_type_upper == "L2":
                            c_xA_xS = np.sum((x_A - x_S) ** 2)
                            c_yB_yS = (
                                np.sum((nodes_Y[y_B] - nodes_Y[y_S]) ** 2) if y_S < n_Y else 0
                            )
                        elif cost_type_upper == "L1":
                            c_xA_xS = np.sum(np.abs(x_A - x_S))
                            c_yB_yS = (
                                np.sum(np.abs(nodes_Y[y_B] - nodes_Y[y_S])) if y_S < n_Y else 0
                            )
                        elif cost_type_upper == "LINF":
                            c_xA_xS = np.max(np.abs(x_A - x_S))
                            c_yB_yS = (
                                np.max(np.abs(nodes_Y[y_B] - nodes_Y[y_S])) if y_S < n_Y else 0
                            )
                        else:
                            c_xA_xS = np.sum(np.abs(x_A - x_S) ** p)
                            c_yB_yS = (
                                np.sum(np.abs(nodes_Y[y_B] - nodes_Y[y_S]) ** p) if y_S < n_Y else 0
                            )

                        c_xS_yS = 0
                        if c_xA_yB[y_B_local] > c_xA_xS - c_yB_yS + 2 * c_xS_yS:
                            shielded = True
                            break

                    if not shielded:
                        all_pairs.append((i_A, y_B))

    if all_pairs:
        return np.array(all_pairs, dtype=np.int32)
    return np.empty((0, 2), dtype=np.int32)


def build_Yhat_simple_cpu(
    level_X,
    level_Y,
    t_map: Dict[int, int],
    knn_indices: np.ndarray,
    k_neighbors: int = 10,
    cost_type: str = "L2",
    p: float = 2.0,
    max_pairs: int = 100000,
) -> np.ndarray:
    nodes_X = level_X.points
    nodes_Y = level_Y.points
    n_X = len(nodes_X)
    n_Y = len(nodes_Y)
    dim = nodes_X.shape[1]
    cost_type_upper = cost_type.upper()

    all_pairs = []
    count = 0
    for i_A in range(n_X):
        if count >= max_pairs:
            break

        x_A = nodes_X[i_A]
        sentinels_i = []
        if i_A in t_map:
            y_idx = t_map[i_A]
            if 0 <= y_idx < n_Y:
                sentinels_i.append((nodes_Y[y_idx], y_idx))

        if i_A < len(knn_indices):
            for k_idx in knn_indices[i_A][:k_neighbors]:
                if k_idx >= 0 and k_idx not in [s[1] for s in sentinels_i]:
                    if k_idx in t_map:
                        y_t = t_map[k_idx]
                        if 0 <= y_t < n_Y:
                            sentinels_i.append((nodes_Y[y_t], y_t))

        for y_B in range(n_Y):
            if count >= max_pairs:
                break

            diff = x_A - nodes_Y[y_B]
            if cost_type_upper == "L2":
                c_xA_yB = np.sum(diff**2)
            elif cost_type_upper == "L1":
                c_xA_yB = np.sum(np.abs(diff))
            elif cost_type_upper == "LINF":
                c_xA_yB = np.max(np.abs(diff))
            else:
                c_xA_yB = np.sum(np.abs(diff) ** p)

            shielded = False
            for x_S, y_S in sentinels_i:
                diff_x = x_A - x_S
                diff_y = nodes_Y[y_B] - nodes_Y[y_S] if y_S < n_Y else np.zeros(dim)

                if cost_type_upper == "L2":
                    c_xA_xS = np.sum(diff_x**2)
                    c_yB_yS = np.sum(diff_y**2)
                elif cost_type_upper == "L1":
                    c_xA_xS = np.sum(np.abs(diff_x))
                    c_yB_yS = np.sum(np.abs(diff_y))
                elif cost_type_upper == "LINF":
                    c_xA_xS = np.max(np.abs(diff_x))
                    c_yB_yS = np.max(np.abs(diff_y))
                else:
                    c_xA_xS = np.sum(np.abs(diff_x) ** p)
                    c_yB_yS = np.sum(np.abs(diff_y) ** p)

                if c_xA_yB > c_xA_xS + c_yB_yS:
                    shielded = True
                    break

            if not shielded:
                all_pairs.append((i_A, y_B))
                count += 1

    if all_pairs:
        return np.array(all_pairs, dtype=np.int32)
    return np.empty((0, 2), dtype=np.int32)
