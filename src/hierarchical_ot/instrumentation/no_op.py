from __future__ import annotations


class NoOpReporter:
    def report_progress(self, _message: str) -> None:
        return None

    def report_lp(self, _summary) -> None:
        return None

    def report_pricing(self, _summary) -> None:
        return None

    def report_convergence(self, _summary) -> None:
        return None

    def report_finalize(self, _summary) -> None:
        return None

    def report_iteration_total(self, _summary) -> None:
        return None

    def report_level(self, _summary) -> None:
        return None
