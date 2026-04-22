from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def tree_log_enabled(solver) -> bool:
    return bool(getattr(solver, "tree_debug", False) or getattr(solver, "ifdebug", False))


def tree_log(solver, message: str) -> None:
    if tree_log_enabled(solver):
        if hasattr(solver, "_runtime_log"):
            solver._runtime_log("progress", message)
        else:
            logger.debug(message)


__all__ = ["tree_log", "tree_log_enabled"]
