from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from ..common.base import ArrayType, BaseHierarchy, HierarchyLevel

def _grid_points(resolution: int) -> np.ndarray:
    idx = np.arange(int(resolution), dtype=np.float32)
    ii, jj = np.meshgrid(idx, idx, indexing="ij")
    return np.stack([ii.reshape(-1), jj.reshape(-1)], axis=1)

def _coarsen_histogram(hist: np.ndarray) -> np.ndarray:
    res = int(hist.shape[0])
    if res % 2 != 0:
        raise ValueError("GridHierarchy currently requires even resolution at every coarsening step.")
    return hist.reshape(res // 2, 2, res // 2, 2).sum(axis=(1, 3))

def _child_labels_for_resolution(resolution: int) -> np.ndarray:
    res = int(resolution)
    fine_idx = np.arange(res * res, dtype=np.int32)
    fine_i = fine_idx // res
    fine_j = fine_idx % res
    coarse_res = res // 2
    return ((fine_i // 2) * coarse_res + (fine_j // 2)).astype(np.int32, copy=False)

class GridHierarchy(BaseHierarchy):
    

    def __init__(self, num_scales: Optional[int] = None):
        super().__init__([])
        self.num_scales = num_scales
        self.build_time = 0.0
        self.original_shape: Optional[Tuple[int, int]] = None

    def build(self, histogram: np.ndarray) -> None:
        t0 = time.perf_counter()
        hist = np.asarray(histogram, dtype=np.float32)
        if hist.ndim != 2 or hist.shape[0] != hist.shape[1]:
            raise ValueError("GridHierarchy only supports square 2D histograms.")
        if hist.shape[0] <= 0:
            raise ValueError("histogram resolution must be positive.")

        self.original_shape = tuple(hist.shape)
        levels_hist: List[np.ndarray] = [hist]
        max_scales = 0
        curr = hist
        while curr.shape[0] > 1 and curr.shape[0] % 2 == 0:
            if self.num_scales is not None and max_scales >= int(self.num_scales):
                break
            curr = _coarsen_histogram(curr)
            levels_hist.append(curr)
            max_scales += 1

        self.levels = []
        for level_idx, level_hist in enumerate(levels_hist):
            res = int(level_hist.shape[0])
            child_labels = None
            if level_idx > 0:
                prev_res = int(levels_hist[level_idx - 1].shape[0])
                child_labels = _child_labels_for_resolution(prev_res)
            self.levels.append(
                HierarchyLevel(
                    level_idx=level_idx,
                    points=_grid_points(res),
                    masses=level_hist.reshape(-1).astype(np.float32, copy=False),
                    cost_vec=None,
                    child_labels=child_labels,
                )
            )

        self.num_levels = len(self.levels)
        self.build_time = time.perf_counter() - t0

    def prolongate(
        self,
        coarse_potential: ArrayType,
        coarse_level_idx: int,
        fine_level_idx: int,
    ) -> ArrayType:
        fine_level = self.levels[int(fine_level_idx)]
        if fine_level.child_labels is None:
            raise ValueError(f"Level {fine_level_idx} has no child_labels.")
        return coarse_potential[np.asarray(fine_level.child_labels, dtype=np.int32)]
