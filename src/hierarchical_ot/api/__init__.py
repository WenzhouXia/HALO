from __future__ import annotations

from ..types.config import GridConfig, TreeConfig
from ..solvers.single_level import _build_lp_solver, _configure_strategy
from .emd import emd2, emd2_grid

__all__ = [
    "emd2",
    "emd2_grid",
    "_build_lp_solver_by_mode",
    "_configure_strategy_by_mode",
]


def _configure_strategy_by_mode(*, mode: str, cost_type: str, config, k_neighbors_tree=None):
    mode = str(mode).lower()
    if mode not in {"grid", "tree"}:
        raise ValueError("HALO_public only supports modes: tree, grid.")
    return _configure_strategy(
        cost_type=cost_type,
        config=config,
        k_neighbors_tree=k_neighbors_tree,
    )


def _build_lp_solver_by_mode(solver_engine: str, mode: str):
    mode = str(mode).lower()
    if mode == "grid":
        config = GridConfig(solver_engine=str(solver_engine))
    elif mode == "tree":
        config = TreeConfig(solver_engine=str(solver_engine))
    else:
        raise ValueError("HALO_public only supports modes: tree, grid.")
    return _build_lp_solver(config)
