from __future__ import annotations

from typing import Any, Dict

from .printing import PrintingOptions


def _fmt_optional_sci(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if value != value:
        return "n/a"
    return f"{value:.{digits}e}"


def _fmt_optional_fixed(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if value != value:
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_components(components: Dict[str, float]) -> str:
    positive = [(key, float(value)) for key, value in components.items() if float(value) > 0.0]
    if not positive:
        return "-"
    total = max(sum(value for _, value in positive), 1e-12)
    positive.sort(key=lambda item: item[1], reverse=True)
    return ", ".join(
        f"{key}={value:.2f}s({(value / total) * 100.0:.1f}%)"
        for key, value in positive[:4]
    )


class ConsoleReporter:
    def __init__(self, solver: Any, printing_options: PrintingOptions) -> None:
        self._solver = solver
        self._printing_options = printing_options

    def _log(self, category: str, message: str) -> None:
        if hasattr(self._solver, "_runtime_log"):
            self._solver._runtime_log(category, message)
        else:
            print(message)

    def report_progress(self, message: str) -> None:
        if self._printing_options.enabled and self._printing_options.progress:
            self._log("progress", message)

    def report_lp(self, summary) -> None:
        if not (self._printing_options.enabled and self._printing_options.profile_iter):
            return
        dims = "vars=n/a, constraints=n/a"
        if summary.num_variables is not None and summary.num_constraints is not None:
            dims = (
                f"vars={int(summary.num_variables):,}, "
                f"constraints={int(summary.num_constraints):,}, "
                f"A=({int(summary.num_constraints):,}, {int(summary.num_variables):,})"
            )
        self._log(
            "profile_iter",
            f"[Profile][L{summary.level_index}][I{summary.inner_iter + 1}][lp] "
            f"time={_fmt_optional_fixed(summary.time_seconds, digits=2)}s, "
            f"iter={summary.iterations}, {dims}, "
            f"obj={_fmt_optional_fixed(summary.primal_objective)} / {_fmt_optional_fixed(summary.dual_objective)}, "
            f"RelRes={_fmt_optional_sci(summary.primal_residual)} / {_fmt_optional_sci(summary.dual_residual)}, "
            f"Gap={_fmt_optional_sci(summary.gap, digits=4)}"
        )

    def report_pricing(self, summary) -> None:
        if not (self._printing_options.enabled and self._printing_options.profile_iter):
            return
        self._log(
            "profile_iter",
            f"[Profile][L{summary.level_index}][I{summary.inner_iter + 1}][pricing] "
            f"time={_fmt_optional_fixed(summary.time_seconds, digits=2)}s, "
            f"active={int(summary.active_before):,}, "
            f"found={int(summary.found):,}, "
            f"added={int(summary.added):,}"
        )

    def report_convergence(self, summary) -> None:
        if not (self._printing_options.enabled and self._printing_options.profile_iter):
            return
        self._log(
            "profile_iter",
            f"[Profile][L{summary.level_index}][I{summary.inner_iter + 1}][convergence_check] "
            f"signed_rel_obj_change={_fmt_optional_sci(summary.signed_rel_obj_change)}, "
            f"converged={bool(summary.converged)}, "
            f"criterion={summary.criterion}, "
            f"plateau={summary.plateau_counter}/{summary.required_plateau}, "
            f"tol={_fmt_optional_sci(summary.tolerance)}"
        )

    def report_finalize(self, summary) -> None:
        if not (self._printing_options.enabled and self._printing_options.profile_iter):
            return
        self._log(
            "profile_iter",
            f"[Profile][L{summary.level_index}][I{summary.inner_iter + 1}][finalize] "
            f"{_fmt_components(summary.components)}"
        )

    def report_iteration_total(self, summary) -> None:
        if not (self._printing_options.enabled and self._printing_options.profile_iter):
            return
        base = max(float(summary.total_seconds), 1e-12)
        self._log(
            "profile_iter",
            f"[Profile][L{summary.level_index}][I{summary.inner_iter + 1}] "
            f"total={float(summary.total_seconds):.2f}s "
            f"solve={float(summary.solve_seconds):.2f}s({(float(summary.solve_seconds) / base) * 100.0:.1f}%) "
            f"finalize={float(summary.finalize_seconds):.2f}s({(float(summary.finalize_seconds) / base) * 100.0:.1f}%) "
            f"else={float(summary.other_seconds):.2f}s({(float(summary.other_seconds) / base) * 100.0:.1f}%)"
        )
        self._log("profile_iter", "---")

    def report_level(self, summary) -> None:
        if not (self._printing_options.enabled and self._printing_options.profile_level):
            return
        base = max(float(summary.total_seconds), 1e-12)
        active_text = "" if summary.active_size is None else f" active={int(summary.active_size)}"
        self._log("profile_level", f"=== Summary level {summary.level_index} ===")
        self._log(
            "profile_level",
            f"[Profile][L{summary.level_index}] total={float(summary.total_seconds):.2f}s "
            f"iters={int(summary.iterations)} "
            f"lp={float(summary.lp_seconds):.2f}s({(float(summary.lp_seconds) / base) * 100.0:.1f}%) "
            f"pricing={float(summary.pricing_seconds):.2f}s({(float(summary.pricing_seconds) / base) * 100.0:.1f}%) "
            f"else={float(summary.other_seconds):.2f}s({(float(summary.other_seconds) / base) * 100.0:.1f}%){active_text}"
        )
