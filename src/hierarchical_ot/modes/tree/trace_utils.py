from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, Optional


def tree_trace_args(
    *,
    level_idx: Optional[int] = None,
    inner_iter: Optional[int] = None,
    stage: Optional[str] = None,
    args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    if level_idx is not None:
        merged["level"] = int(level_idx)
    if inner_iter is not None:
        merged["iter"] = int(inner_iter)
    if stage is not None:
        merged["stage"] = str(stage)
    if args:
        merged.update(args)
    return merged


def tree_trace_span(
    trace_collector,
    name: str,
    *,
    level_idx: Optional[int] = None,
    inner_iter: Optional[int] = None,
    stage: Optional[str] = None,
    args: Optional[Dict[str, Any]] = None,
):
    if trace_collector is None:
        return nullcontext()
    merged_args = tree_trace_args(
        level_idx=level_idx,
        inner_iter=inner_iter,
        stage=stage,
        args=args,
    )
    return trace_collector.span(name, "solve_ot", args=merged_args)
