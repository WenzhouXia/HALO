"""Core runtime package for hierarchical OT execution."""

from .hierarchical_solver import HierarchicalOTSolver
from .mode_dispatch import detect_mode
from .multilevel_flow import run_multilevel_flow
from ..types.base import ActiveSupport, ActiveSupportStrategy, ArrayType, BaseHierarchy, BaseStrategy, HierarchyLevel
from ..types.config import ConfigType, GridConfig, SolverConfig, TreeConfig, create_config

__all__ = [
    "BaseHierarchy",
    "HierarchyLevel",
    "BaseStrategy",
    "ActiveSupportStrategy",
    "ActiveSupport",
    "ArrayType",
    "SolverConfig",
    "GridConfig",
    "TreeConfig",
    "ConfigType",
    "create_config",
    "HierarchicalOTSolver",
    "detect_mode",
    "run_multilevel_flow",
]
