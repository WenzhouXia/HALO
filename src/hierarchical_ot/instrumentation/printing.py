from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class PrintingOptions:
    enabled: bool = True
    progress: bool = True
    profile_iter: bool = True
    profile_level: bool = True
    profile_run: bool = True
    warm_start: bool = True
    iter_interval: int = 10
    verbosity: str = "compact"


def build_printing_options(printing: Optional[Dict[str, Any]]) -> PrintingOptions:
    printing = dict(printing or {})
    return PrintingOptions(
        enabled=bool(printing.get("enabled", True)),
        progress=bool(printing.get("progress", True)),
        profile_iter=bool(printing.get("profile_iter", True)),
        profile_level=bool(printing.get("profile_level", True)),
        profile_run=bool(printing.get("profile_run", True)),
        warm_start=bool(printing.get("warm_start", True)),
        iter_interval=int(printing.get("iter_interval", 10)),
        verbosity=str(printing.get("verbosity", "compact")),
    )
