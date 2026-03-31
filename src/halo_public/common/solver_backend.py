from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional, TypedDict

class CommonRunState(TypedDict, total=False):
    tolerance: Dict[str, float]
    num_levels: int
    coarsest_idx: int

class CommonLevelState(TypedDict, total=False):
    level_idx: int
    is_coarsest: bool
    t_level_start: float
    inner_iter: int
    iters_done: int
    level_lp_time: float
    level_pricing_time: float

class SolverBackend(ABC):
    

    name = "unknown"

    def __init__(self, solver: Any):
        self.solver = solver

    def should_exact_flat(self) -> bool:
        return False

    def solve_exact_flat(self, tolerance: Dict[str, float]) -> Optional[Dict[str, Any]]:
        raise NotImplementedError(f"{type(self).__name__} does not implement solve_exact_flat")

    @abstractmethod
    def init_run_state(self, tolerance: Dict[str, float], **kwargs: Any) -> CommonRunState:
        raise NotImplementedError

    @abstractmethod
    def get_level_indices(self, run_state: CommonRunState) -> Iterable[int]:
        raise NotImplementedError

    @abstractmethod
    def init_level_state(
        self,
        level_idx: int,
        run_state: CommonRunState,
        tolerance: Dict[str, float],
        cost_type: str,
        use_bfs_skeleton: bool,
    ) -> Optional[CommonLevelState]:
        raise NotImplementedError

    def get_max_inner_iters(
        self,
        level_state: CommonLevelState,
        max_inner_iter: int,
    ) -> int:
        if level_state.get("is_coarsest"):
            return 1
        return max_inner_iter

    def prepare_iteration_input(
        self,
        level_state: CommonLevelState,
        run_state: CommonRunState,
        inner_iter: int,
    ) -> None:
        return None

    @abstractmethod
    def solve_iteration_lp(
        self,
        level_state: CommonLevelState,
        run_state: CommonRunState,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def finalize_iteration(
        self,
        level_state: CommonLevelState,
        run_state: CommonRunState,
        step_pack: Dict[str, Any],
        convergence_criterion: str,
        tolerance: Dict[str, float],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def should_stop_iteration(
        self,
        level_state: CommonLevelState,
        run_state: CommonRunState,
        max_inner_iter: int,
        step_pack: Dict[str, Any],
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def record_level_result(
        self,
        level_state: CommonLevelState,
        final_refinement_tolerance: Optional[Dict[str, float]],
    ) -> None:
        raise NotImplementedError

    def advance_to_next_level(
        self,
        level_state: CommonLevelState,
        run_state: CommonRunState,
    ) -> None:
        return None

    @abstractmethod
    def package_result(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def extract_step_objective(
        self,
        step_pack: Dict[str, Any],
    ) -> Optional[float]:
        raise NotImplementedError
