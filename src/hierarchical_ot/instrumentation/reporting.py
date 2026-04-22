from __future__ import annotations

from typing import Any, Dict, Optional

from ..instrumentation.phase_names import (
    PHASE_FINALIZE_ITERATION,
    PHASE_SOLVE_ITERATION_LP,
)
from ..types.runtime import (
    ConvergenceSummary,
    FinalizeSummary,
    IterationSummary,
    LevelSummary,
    LPSummary,
    PricingSummary,
)


def extract_lp_summary(level_index: int, inner_iter: int, step_pack: Dict[str, Any]) -> Optional[LPSummary]:
    diag = {}
    primal_obj = None
    dual_obj = None
    primal_feas = None
    dual_feas = None
    gap = None
    time_seconds = None
    iterations = "n/a"
    n_vars = None
    n_constraints = None

    if "res" in step_pack:
        res = step_pack.get("res")
        primal_obj = getattr(res, "obj_val", None)
        dual_obj = getattr(res, "dual_obj_val", None)
        primal_feas = getattr(res, "primal_feas", None)
        dual_feas = getattr(res, "dual_feas", None)
        gap = getattr(res, "gap", None)
        iterations = getattr(res, "iterations", "n/a")
        diag = step_pack.get("diag", {}) if isinstance(step_pack.get("diag"), dict) else {}
        time_seconds = step_pack.get("lp_time")
        x_val = getattr(res, "x", None)
        y_val = getattr(res, "y", None)
        n_vars = diag.get("n_vars", None if x_val is None else getattr(x_val, "size", None))
        n_constraints = diag.get("n_constraints", None if y_val is None else getattr(y_val, "size", None))
    elif "lp_result" in step_pack:
        lp_result = step_pack.get("lp_result", {}) if isinstance(step_pack.get("lp_result"), dict) else {}
        diag = lp_result.get("diag", {}) if isinstance(lp_result.get("diag"), dict) else {}
        primal_obj = lp_result.get("obj")
        dual_obj = lp_result.get("dual_obj")
        primal_feas = lp_result.get("primal_feas")
        dual_feas = lp_result.get("dual_feas")
        gap = lp_result.get("gap")
        iterations = lp_result.get("iters", "n/a")
        time_seconds = diag.get("backend_solve_wall_time", step_pack.get("lp_time"))
        n_vars = diag.get("n_vars")
        n_constraints = diag.get("n_constraints")
    else:
        return None

    return LPSummary(
        level_index=level_index,
        inner_iter=inner_iter,
        time_seconds=time_seconds,
        iterations=iterations,
        num_variables=n_vars,
        num_constraints=n_constraints,
        primal_objective=primal_obj,
        dual_objective=dual_obj,
        primal_residual=primal_feas,
        dual_residual=dual_feas,
        gap=gap,
    )


def extract_pricing_summary(level_index: int, inner_iter: int, step_pack: Dict[str, Any]) -> Optional[PricingSummary]:
    info = step_pack.get("pricing_info")
    if not isinstance(info, dict):
        return None
    return PricingSummary(
        level_index=level_index,
        inner_iter=inner_iter,
        time_seconds=info.get("time"),
        active_before=int(info.get("active_before", 0)),
        found=int(info.get("found", 0)),
        added=int(info.get("added", 0)),
    )


def extract_convergence_summary(
    level_index: int,
    inner_iter: int,
    step_pack: Dict[str, Any],
) -> Optional[ConvergenceSummary]:
    info = step_pack.get("convergence_info")
    if not isinstance(info, dict):
        return None
    return ConvergenceSummary(
        level_index=level_index,
        inner_iter=inner_iter,
        signed_rel_obj_change=info.get("signed_rel_obj_change"),
        converged=bool(info.get("is_converged", False)),
        criterion=str(info.get("criterion", "n/a")),
        plateau_counter=info.get("plateau_counter", "n/a"),
        required_plateau=info.get("required_plateau", "n/a"),
        tolerance=info.get("objective_tol"),
    )


def extract_finalize_summary(
    level_index: int,
    inner_iter: int,
    iter_profile: Optional[Dict[str, Any]],
) -> FinalizeSummary:
    components: Dict[str, float] = {}
    if iter_profile and isinstance(iter_profile.get("phases"), dict):
        phase_info = iter_profile["phases"].get(PHASE_FINALIZE_ITERATION, {})
        if isinstance(phase_info, dict) and isinstance(phase_info.get("components"), dict):
            components = {key: float(value) for key, value in phase_info["components"].items()}
    return FinalizeSummary(level_index=level_index, inner_iter=inner_iter, components=components)


def extract_iteration_summary(
    level_index: int,
    inner_iter: int,
    iter_profile: Optional[Dict[str, Any]],
) -> Optional[IterationSummary]:
    if not iter_profile:
        return None
    phase_times = iter_profile.get("phase_times", {})
    if not isinstance(phase_times, dict):
        phase_times = {}
    total_seconds = float(iter_profile.get("time", 0.0))
    solve_seconds = float(phase_times.get(PHASE_SOLVE_ITERATION_LP, 0.0))
    finalize_seconds = float(phase_times.get(PHASE_FINALIZE_ITERATION, 0.0))
    other_seconds = max(total_seconds - solve_seconds - finalize_seconds, 0.0)
    return IterationSummary(
        level_index=level_index,
        inner_iter=inner_iter,
        total_seconds=total_seconds,
        solve_seconds=solve_seconds,
        finalize_seconds=finalize_seconds,
        other_seconds=other_seconds,
    )


def extract_level_summary(
    level_index: int,
    level_state: Dict[str, Any],
    level_profile: Optional[Dict[str, Any]],
) -> Optional[LevelSummary]:
    if not level_profile:
        return None
    total_seconds = float(level_profile.get("time", 0.0))
    iterations = int(level_profile.get("num_iters", 0))
    if iterations <= 0:
        iterations = int(level_state.get("completed_iters", level_state.get("current_iter", 0)))
    lp_seconds = float(level_state.get("level_lp_time", 0.0))
    pricing_seconds = float(level_state.get("level_pricing_time", 0.0))
    other_seconds = max(total_seconds - lp_seconds - pricing_seconds, 0.0)
    active_size = level_state.get("curr_active_size")
    return LevelSummary(
        level_index=level_index,
        total_seconds=total_seconds,
        iterations=iterations,
        lp_seconds=lp_seconds,
        pricing_seconds=pricing_seconds,
        other_seconds=other_seconds,
        active_size=None if active_size is None else int(active_size),
    )


def report_iteration_summaries(
    reporter,
    *,
    level_index: int,
    inner_iter: int,
    step_pack: Dict[str, Any],
    iter_profile: Optional[Dict[str, Any]],
) -> None:
    lp_summary = extract_lp_summary(level_index, inner_iter, step_pack)
    if lp_summary is not None:
        reporter.report_lp(lp_summary)

    pricing_summary = extract_pricing_summary(level_index, inner_iter, step_pack)
    if pricing_summary is not None:
        reporter.report_pricing(pricing_summary)

    convergence_summary = extract_convergence_summary(level_index, inner_iter, step_pack)
    if convergence_summary is not None:
        reporter.report_convergence(convergence_summary)

    reporter.report_finalize(extract_finalize_summary(level_index, inner_iter, iter_profile))

    iteration_summary = extract_iteration_summary(level_index, inner_iter, iter_profile)
    if iteration_summary is not None:
        reporter.report_iteration_total(iteration_summary)


def report_level_summary(
    reporter,
    *,
    level_index: int,
    level_state: Dict[str, Any],
    level_profile: Optional[Dict[str, Any]],
) -> Optional[LevelSummary]:
    level_summary = extract_level_summary(level_index, level_state, level_profile)
    if level_summary is not None:
        reporter.report_level(level_summary)
    return level_summary


def finalize_run_profiling(problem_def, result: Dict[str, Any]) -> None:
    if not bool(problem_def.profiling_options.enabled):
        return
    profiling = problem_def.profiler.summary(
        build_time=problem_def.solver.build_time,
        solve_time=problem_def.solver._solve_wall_time,
    )
    result["profiling"] = profiling
    problem_def.solver._print_profile_run(profiling, include_level_breakdown=False)
    problem_def.solver._runtime_log("profile_run", "")
    problem_def.solver._print_profile_run_level_breakdown(profiling)


def report_run_completion(problem_def, level_index: int, level_summary: Optional[LevelSummary], result: Dict[str, Any]) -> None:
    reporter = problem_def.reporter
    solver = problem_def.solver
    if level_summary is not None:
        reporter.report_progress(
            f"[HierOT] finish level={level_index}, "
            f"iters={int(level_summary.iterations)}, "
            f"obj={float(result.get('final_obj', float('nan'))):.6e}"
        )
    reporter.report_progress(f"[HierOT] solve complete in {solver._solve_wall_time:.3f}s")
    reporter.report_progress("================ solve complete ================")
