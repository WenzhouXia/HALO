from __future__ import annotations

import logging

import numpy as np

from .legacy_infeasibility import dualOT_primal_infeas_pointcloud_cupy_auto
from .violation_check import apply_violation_check as _apply_violation_check
from .violation_check import check_constraint_violations

logger = logging.getLogger(__name__)


def _tree_internal_cost_type(cost_type_raw: str) -> str:
    normalized = str(cost_type_raw).strip().lower()
    if normalized in {"l2^2", "sqeuclidean", "l2sq"}:
        return "L2"
    if normalized in {"l1"}:
        return "L1"
    if normalized in {"linf"}:
        return "LINF"
    if normalized in {"l2", "euclidean"}:
        raise ValueError("tree mode does not support 'l2'; use 'l2^2' for squared Euclidean distance.")
    raise ValueError(f"Unsupported tree cost_type={cost_type_raw!r}")


def apply_violation_check(solver, **kwargs):
    return _apply_violation_check(solver, **kwargs)


def compute_primal_infeas(solver, x_dual, level_s, level_t):
    try:
        cost_type = _tree_internal_cost_type(getattr(solver, "_cost_type", "l2^2"))
        p = getattr(solver, "_cost_p", 2.0)
        infeas = dualOT_primal_infeas_pointcloud_cupy_auto(
            x_dual,
            level_s.points,
            level_t.points,
            cost_type=cost_type,
            p=p,
            use_cupy=bool(getattr(solver, "tree_infeas_use_cupy", False)),
        )
        return infeas
    except Exception as exc:
        logger.warning(f"Primal infeasibility computation failed: {exc}")
        return float("inf")
def repair_keep_coverage(solver, *args, **kwargs):
    from .shielding_backup import repair_keep_coverage as _repair_keep_coverage

    del solver
    return _repair_keep_coverage(*args, **kwargs)


def solve_tree_lp_with_fallback(solver, **kwargs):
    from .solve_lp import solve_tree_lp_with_fallback as _solve

    return _solve(solver, **kwargs)


__all__ = [
    "check_constraint_violations",
    "apply_violation_check",
    "compute_primal_infeas",
    "repair_keep_coverage",
    "solve_tree_lp_with_fallback",
]
