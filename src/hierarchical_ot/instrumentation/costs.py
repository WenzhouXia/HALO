from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

from ..instrumentation.trace import _ChromeTraceCollector
from ..core.hierarchical_solver import _cost_pairs_lowrank_batched


def _extract_level_shape_from_cache(level_cache: Dict[str, Any]) -> Tuple[int, int, int]:
    if "F" in level_cache and "G" in level_cache:
        src = np.asarray(level_cache["F"])
        tgt = np.asarray(level_cache["G"])
        return int(src.shape[0]), int(tgt.shape[0]), int(src.shape[1])
    if "S" in level_cache and "T" in level_cache:
        src = np.asarray(level_cache["S"])
        tgt = np.asarray(level_cache["T"])
        return int(src.shape[0]), int(tgt.shape[0]), int(src.shape[1])
    raise ValueError("Unsupported level_cache layout for nodewise_auto pricing")


def _preprocess_sqeuclidean_to_lowrank(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    scaled = np.array(points, dtype=np.float32, order="C", copy=True)
    cost_vec = np.einsum("ij,ij->i", scaled, scaled).astype(np.float32, copy=False)
    scaled *= np.float32(math.sqrt(2.0))
    return scaled, cost_vec


def _compute_lowrank_cost_vec_from_pairs_chunked(
    level_cache: Dict[str, Any],
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    chunk_size: int = 65536,
    progress_tag: str = "WarmStartSeed",
    verbose: bool = True,
    tracer: _ChromeTraceCollector | None = None,
    trace_args: Dict[str, Any] | None = None,
) -> np.ndarray:
    total = int(rows.shape[0])
    if total == 0:
        return np.empty(0, dtype=np.float32)

    num_chunks = (total + chunk_size - 1) // chunk_size
    if verbose:
        print(
            f"[{progress_tag}] start standard lowrank batched c_vec build: "
            f"edges={total:,}, batch_size={chunk_size}, chunks={num_chunks}",
            flush=True,
        )
    out = _cost_pairs_lowrank_batched(
        level_cache,
        rows,
        cols,
        batch_size=chunk_size,
        free_after=True,
        tracer=tracer,
        trace_args=trace_args,
    ).astype(np.float32, copy=False)
    if verbose:
        print(f"[{progress_tag}] finish chunked c_vec build", flush=True)
    return out
