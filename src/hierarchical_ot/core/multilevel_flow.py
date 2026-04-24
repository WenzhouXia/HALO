from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

from ..instrumentation.phase_names import (
    PHASE_ADVANCE_TO_NEXT_LEVEL,
    PHASE_CALLBACK,
    PHASE_FINALIZE_ITERATION,
    PHASE_INIT_LEVEL_STATE,
    PHASE_INIT_RUN_STATE,
    PHASE_PACKAGE_RESULT,
    PHASE_PREPARE_ITERATION_INPUT,
    PHASE_PROFILING_OUTPUT,
    PHASE_RECORD_LEVEL_RESULT,
    PHASE_SHOULD_STOP_ITERATION,
    PHASE_SOLVE_ITERATION_LP,
)
from ..instrumentation.reporting import (
    finalize_run_profiling,
    report_iteration_summaries,
    report_level_summary,
    report_run_completion,
)
from ..types.runtime import AlgorithmState, ProblemDef
from .mode_dispatch import (
    advance_level_by_mode,
    finalize_iteration_by_mode,
    get_max_inner_iters_by_mode,
    initial_by_mode,
    init_run_state_by_mode,
    package_result_by_mode,
    prepare_inner_by_mode,
    record_level_by_mode,
    should_exact_flat_by_mode,
    solve_lp_by_mode,
    solve_exact_flat_by_mode,
    should_stop_inner_by_mode,
)
from ..instrumentation.preflight import validate_problem


@contextmanager
def _profile_phase(profiler, phase_name: str, *, gpu_possible: bool = False):
    profiler.start_phase(phase_name, gpu_possible=gpu_possible)
    try:
        yield
    finally:
        profiler.end_phase()


@contextmanager
def _profile_level(profiler, level_index: int, *, get_iters):
    profiler.start_level(level_index)
    try:
        yield
    finally:
        profiler.end_level(level_index, int(get_iters()))


@contextmanager
def _profile_iteration(profiler, level_index: int, inner_iter: int):
    profiler.start_iteration(level_index, inner_iter)
    try:
        yield
    finally:
        profiler.end_iteration()


def _init_algorithm_state(problem_def: ProblemDef) -> AlgorithmState:
    run_state = init_run_state_by_mode(problem_def)
    return AlgorithmState(
        run_state=run_state,
        level_indices=list(
            range(
                run_state["coarsest_idx"],
                run_state["finest_idx"] - 1,
                -1,
            )
        ),
    )


@contextmanager
def _trace_span(problem_def: ProblemDef, name: str, *, args: Optional[Dict[str, Any]] = None):
    tracer = getattr(problem_def, "trace_collector", None)
    if tracer is None:
        yield
        return
    trace_prefix = str(getattr(problem_def, "trace_prefix", "solve_ot"))
    with tracer.span(f"{trace_prefix}.solve.{name}", "solve_ot", args=dict(args or {})):
        yield


def run_multilevel_flow(problem_def: ProblemDef) -> Optional[Dict[str, Any]]:
    validate_problem(problem_def)
    solver = problem_def.solver
    spec = problem_def.solve_spec
    profiler = problem_def.profiler
    reporter = problem_def.reporter
    solve_start = time.perf_counter()

    # CN: 特殊情况，单层求解
    # EN: special case, single-level exact solve
    if should_exact_flat_by_mode(problem_def):
        reporter.report_progress(f"[HierOT] mode={problem_def.mode} single-level exact solve")
        with _trace_span(problem_def, "single_level_exact"):
            result = solve_exact_flat_by_mode(problem_def)
        solver._solve_wall_time = time.perf_counter() - solve_start
        profiler.end_run()
        if result is not None and bool(problem_def.profiling_options.enabled):
            profiling = profiler.summary(build_time=solver.build_time, solve_time=solver._solve_wall_time)
            result["profiling"] = profiling
            solver._print_profile_run(profiling)
        return result

    # CN: 初始化运行状态
    # EN: initialize algorithm state
    with _profile_phase(profiler, PHASE_INIT_RUN_STATE):
        with _trace_span(problem_def, PHASE_INIT_RUN_STATE):
            algorithm_state = _init_algorithm_state(problem_def)

    reporter.report_progress(
        f"[HierOT] mode={problem_def.mode} start hierarchical solve with "
        f"{len(algorithm_state.level_indices)} levels, max_inner_iter={spec.max_inner_iter}"
    )

    for level_index in algorithm_state.level_indices:
        level_iters_done = 0
        with _profile_level(profiler, level_index, get_iters=lambda: level_iters_done):
            with _trace_span(problem_def, "level", args={"level": int(level_index)}):
                with _profile_phase(profiler, PHASE_INIT_LEVEL_STATE):
                    with _trace_span(problem_def, PHASE_INIT_LEVEL_STATE, args={"level": int(level_index)}):
                        level_state = initial_by_mode(problem_def, algorithm_state, level_index)

                level_state.max_inner_iter = get_max_inner_iters_by_mode(problem_def, level_state)
                active_size = level_state.data.curr_active_size
                active_text = "" if active_size is None else f", active_size={int(active_size):,}"
                reporter.report_progress(
                    f"[HierOT] enter level={level_index}, max_iters={int(level_state.max_inner_iter)}{active_text}"
                )

                # CN: 外迭代，用 coarse level 给 fine level 提供warmstart
                # EN: coarse level provides warmstart to fine level
                for inner_iter in range(level_state.max_inner_iter):
                    level_state.data.current_iter = inner_iter
                    step_data: Dict[str, Any] = {"success": False}
                    iter_args = {"level": int(level_index), "iter": int(inner_iter)}
                    with _profile_iteration(profiler, level_index, inner_iter):
                        with _trace_span(problem_def, "iteration", args=iter_args):
                            with _profile_phase(profiler, PHASE_PREPARE_ITERATION_INPUT):
                                with _trace_span(problem_def, PHASE_PREPARE_ITERATION_INPUT, args=iter_args):
                                    prepare_inner_by_mode(problem_def, algorithm_state, level_state)

                            with _profile_phase(profiler, PHASE_SOLVE_ITERATION_LP):
                                with _trace_span(problem_def, PHASE_SOLVE_ITERATION_LP, args=iter_args):
                                    level_state.mode_state["_pre_lp_active"] = level_state.curr_active_size
                                    step_result = solve_lp_by_mode(problem_def, algorithm_state, level_state)
                            step_data = step_result.data
                            if isinstance(step_data.get("components"), dict):
                                profiler.add_components(step_data.get("components"))
                            level_state.data.completed_iters = inner_iter + 1
                            level_iters_done = level_state.data.completed_iters

                            if not step_result.success:
                                level_iters_done = int(level_state.data.current_iter)
                                solver._solve_wall_time = time.perf_counter() - solve_start
                                profiler.end_run()
                                return None

                            with _profile_phase(profiler, PHASE_FINALIZE_ITERATION, gpu_possible=True):
                                with _trace_span(problem_def, PHASE_FINALIZE_ITERATION, args=iter_args):
                                    finalize_iteration_by_mode(problem_def, algorithm_state, level_state, step_result)

                            with _profile_phase(profiler, PHASE_CALLBACK):
                                with _trace_span(problem_def, PHASE_CALLBACK, args=iter_args):
                                    solver._emit_inner_iteration_callback(
                                        callback=problem_def.inner_iteration_callback,
                                        problem_def=problem_def,
                                        level_state=level_state.data,
                                        step_pack=step_data,
                                    )

                            with _profile_phase(profiler, PHASE_SHOULD_STOP_ITERATION):
                                with _trace_span(problem_def, PHASE_SHOULD_STOP_ITERATION, args=iter_args):
                                    should_stop = should_stop_inner_by_mode(problem_def, algorithm_state, level_state, step_result)

                            with _profile_phase(profiler, PHASE_PROFILING_OUTPUT):
                                with _trace_span(problem_def, PHASE_PROFILING_OUTPUT, args=iter_args):
                                    iter_profile = profiler.current_iteration_snapshot()
                                    report_iteration_summaries(
                                        reporter,
                                        level_index=level_index,
                                        inner_iter=inner_iter,
                                        step_pack=step_data,
                                        iter_profile=iter_profile,
                                    )

                            if should_stop:
                                reporter.report_progress(
                                    f"[HierOT] level={level_index} stop at iter={inner_iter + 1}/{int(level_state.max_inner_iter)}"
                                )
                                break

                level_iters_done = int(level_state.data.completed_iters)
                level_args = {"level": int(level_index), "iters_done": int(level_iters_done)}
                with _profile_phase(profiler, PHASE_RECORD_LEVEL_RESULT):
                    with _trace_span(problem_def, PHASE_RECORD_LEVEL_RESULT, args=level_args):
                        record_level_by_mode(problem_def, algorithm_state, level_state)
                
                with _profile_phase(profiler, PHASE_ADVANCE_TO_NEXT_LEVEL):
                    with _trace_span(problem_def, PHASE_ADVANCE_TO_NEXT_LEVEL, args=level_args):
                        advance_level_by_mode(problem_def, algorithm_state, level_state)

                with _profile_phase(profiler, PHASE_PROFILING_OUTPUT):
                    with _trace_span(problem_def, "level_summary", args=level_args):
                        level_profile = profiler.current_level_snapshot()
                        level_summary = report_level_summary(
                            reporter,
                            level_index=level_index,
                            level_state=level_state.data,
                            level_profile=level_profile,
                        )
                        if level_index == 0:
                            final_level_summary = level_summary

    # CN: 打包结果
    # EN: Package results
    with _profile_phase(profiler, PHASE_PACKAGE_RESULT):
        with _trace_span(problem_def, PHASE_PACKAGE_RESULT):
            result = package_result_by_mode(problem_def, algorithm_state)
    solver._solve_wall_time = time.perf_counter() - solve_start
    profiler.end_run()
    finalize_run_profiling(problem_def, result)
    report_run_completion(problem_def, level_index=0, level_summary=final_level_summary, result=result)
    algorithm_state.result = result
    return result
