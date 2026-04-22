from __future__ import annotations

from typing import Any

import numpy as np

from .violation_kernels import (
    check_constraint_violations_gpu_abs_lessGPU,
    check_constraint_violations_gpu_abs_lessGPU_local_2DT,
    grid_violation_candidates,
)


def collect_violation_candidates(
    solver,
    *,
    x_dual: np.ndarray,
    level_s: Any,
    level_t: Any,
    vd_thr: float,
    p: int,
) -> np.ndarray:
    """
    CN: 根据当前 dual 解调用网格 violation checker，收集需要加入 active set 的候选边。
    EN: Collect candidate violating edges from the current dual solution with the grid violation checker.
    """
    n_s = int(len(level_s.points))
    max_candidates = max(int(vd_thr * 2 * n_s), 1)
    check_keep = grid_violation_candidates(
        np.asarray(x_dual, dtype=np.float32),
        level_s.points,
        level_t.points,
        cost_type=str(getattr(solver, "_cost_type", "l2^2")),
        p=p,
        vd_thr=vd_thr,
        max_candidates=max_candidates,
    )
    return np.asarray(check_keep, dtype=np.int64)


def apply_violation_check(solver, *, x_dual, level_s, level_t, keep, vd_thr: float, p: int):
    """
    CN: 将新检测到的 violation 候选与现有 keep 合并并去重。
    EN: Merge newly detected violation candidates into the existing keep set and deduplicate them.
    """
    check_keep = collect_violation_candidates(
        solver,
        x_dual=x_dual,
        level_s=level_s,
        level_t=level_t,
        vd_thr=vd_thr,
        p=p,
    )
    if check_keep.size == 0:
        return np.asarray(keep, dtype=np.int64)
    if np.asarray(keep).size == 0:
        return np.asarray(check_keep, dtype=np.int64)
    return np.unique(
        np.concatenate([np.asarray(keep, dtype=np.int64), np.asarray(check_keep, dtype=np.int64)])
    ).astype(np.int64, copy=False)


__all__ = [
    "check_constraint_violations_gpu_abs_lessGPU",
    "check_constraint_violations_gpu_abs_lessGPU_local_2DT",
    "grid_violation_candidates",
    "collect_violation_candidates",
    "apply_violation_check",
]
