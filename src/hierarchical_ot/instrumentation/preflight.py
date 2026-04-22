from __future__ import annotations

from typing import Any


def validate_problem(problem_def: Any) -> None:
    mode = str(problem_def.mode).lower()
    if mode not in {"cluster", "grid", "tree"}:
        raise ValueError(f"Unknown mode: {problem_def.mode}")

    if getattr(problem_def, "backend", None) is None:
        raise RuntimeError("Missing backend in ProblemDef.")

    solver = getattr(problem_def, "solver", None)
    if solver is None:
        raise RuntimeError("Missing solver in ProblemDef.")

    if mode == "tree":
        fallback = str(getattr(solver, "tree_infeas_fallback", "none")).strip().lower()
        if fallback != "none":
            raise RuntimeError(
                "tree_infeas_fallback is not allowed in the new strict execution path. "
                f"Got: {fallback}"
            )
