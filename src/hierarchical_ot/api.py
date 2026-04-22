"""Deprecated bridge module for direct imports of ``hierarchical_ot.api`` as a file."""

from .api import _build_lp_solver_by_mode, _configure_strategy_by_mode, emd2, emd2_grid

__all__ = [
    "emd2",
    "emd2_grid",
    "_build_lp_solver_by_mode",
    "_configure_strategy_by_mode",
]
