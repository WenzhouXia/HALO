from __future__ import annotations

from typing import Any

from ..modes.grid import initial as grid_initial
from ..modes.grid import iterations as grid_iterations
from ..modes.grid import result as grid_result
from ..modes.grid import stop as grid_stop
from ..modes.grid.hierarchy import GridHierarchy
from ..modes.tree import initial as tree_initial
from ..modes.tree import iterations as tree_iterations
from ..modes.tree import result as tree_result
from ..modes.tree import stop as tree_stop
from ..modes.tree.hierarchy import TreeHierarchy
from ..types.runtime import LevelState


def detect_mode(solver: Any, mode_override: Any = None) -> str:
    if isinstance(solver.hierarchy_s, GridHierarchy):
        default_mode = "grid"
    elif isinstance(solver.hierarchy_s, TreeHierarchy):
        default_mode = "tree"
    else:
        raise ValueError("HALO_public only supports tree/grid hierarchies.")
    return str(mode_override or default_mode).lower()


def should_exact_flat_by_mode(problem_def) -> bool:
    if problem_def.mode == "tree":
        return bool(tree_initial.should_exact_flat(problem_def.solver))
    if problem_def.mode == "grid":
        return bool(grid_initial.should_exact_flat(problem_def.solver))
    raise ValueError(f"Unknown mode: {problem_def.mode}")


def solve_exact_flat_by_mode(problem_def):
    spec = problem_def.solve_spec
    if problem_def.mode == "grid":
        return grid_initial.solve_exact_flat(problem_def.solver, spec.tolerance)
    raise ValueError(f"Mode {problem_def.mode!r} does not support exact-flat solve.")


def init_run_state_by_mode(problem_def):
    spec = problem_def.solve_spec
    if problem_def.mode == "tree":
        return tree_initial.init_run_state(problem_def.solver, spec.tolerance, **problem_def.extra_kwargs)
    if problem_def.mode == "grid":
        return grid_initial.init_run_state(problem_def.solver, spec.tolerance, **problem_def.extra_kwargs)
    raise ValueError(f"Unknown mode: {problem_def.mode}")


def get_max_inner_iters_by_mode(problem_def, level_state) -> int:
    if level_state.data.is_coarsest:
        return 1
    return int(problem_def.solve_spec.max_inner_iter)


def extract_step_objective_by_mode(problem_def, step_pack):
    if problem_def.mode == "tree":
        return tree_iterations.extract_step_objective(step_pack)
    if problem_def.mode == "grid":
        return grid_iterations.extract_step_objective(step_pack)
    raise ValueError(f"Unknown mode: {problem_def.mode}")


def initial_by_mode(problem_def, algorithm_state, level_index) -> LevelState:
    if problem_def.mode == "tree":
        return tree_initial.initial(problem_def, algorithm_state, level_index)
    if problem_def.mode == "grid":
        return grid_initial.initial(problem_def, algorithm_state, level_index)
    raise ValueError(f"Unknown mode: {problem_def.mode}")


def prepare_inner_by_mode(problem_def, algorithm_state, level_state) -> None:
    if problem_def.mode == "tree":
        return tree_iterations.prepare_inner(problem_def, algorithm_state, level_state)
    if problem_def.mode == "grid":
        return grid_iterations.prepare_inner(problem_def, algorithm_state, level_state)
    raise ValueError(f"Unknown mode: {problem_def.mode}")


def solve_lp_by_mode(problem_def, algorithm_state, level_state):
    if problem_def.mode == "tree":
        return tree_iterations.solve_lp(problem_def, algorithm_state, level_state)
    if problem_def.mode == "grid":
        return grid_iterations.solve_lp(problem_def, algorithm_state, level_state)
    raise ValueError(f"Unknown mode: {problem_def.mode}")


def finalize_iteration_by_mode(problem_def, algorithm_state, level_state, step_result) -> None:
    if problem_def.mode == "tree":
        return tree_iterations.finalize_iteration(
            problem_def.solver,
            level_state.data,
            algorithm_state.run_state,
            step_result.data,
        )
    if problem_def.mode == "grid":
        return grid_iterations.finalize_iteration(
            problem_def.solver,
            level_state.data,
            algorithm_state.run_state,
            step_result.data,
        )
    raise ValueError(f"Unknown mode: {problem_def.mode}")


def should_stop_inner_by_mode(problem_def, algorithm_state, level_state, step_result) -> bool:
    if problem_def.mode == "tree":
        return bool(tree_stop.should_stop_inner(problem_def, algorithm_state, level_state, step_result))
    if problem_def.mode == "grid":
        return bool(grid_stop.should_stop_inner(problem_def, algorithm_state, level_state, step_result))
    raise ValueError(f"Unknown mode: {problem_def.mode}")


def record_level_by_mode(problem_def, algorithm_state, level_state) -> None:
    if problem_def.mode == "tree":
        return tree_result.record_level(problem_def, algorithm_state, level_state)
    if problem_def.mode == "grid":
        return grid_result.record_level(problem_def, algorithm_state, level_state)
    raise ValueError(f"Unknown mode: {problem_def.mode}")


def advance_level_by_mode(problem_def, algorithm_state, level_state) -> None:
    if problem_def.mode == "tree":
        return tree_result.advance_level(problem_def, algorithm_state, level_state)
    if problem_def.mode == "grid":
        return grid_result.advance_level(problem_def, algorithm_state, level_state)
    raise ValueError(f"Unknown mode: {problem_def.mode}")


def package_result_by_mode(problem_def, algorithm_state):
    if problem_def.mode == "tree":
        return tree_result.package_result(problem_def, algorithm_state)
    if problem_def.mode == "grid":
        return grid_result.package_result(problem_def, algorithm_state)
    raise ValueError(f"Unknown mode: {problem_def.mode}")
