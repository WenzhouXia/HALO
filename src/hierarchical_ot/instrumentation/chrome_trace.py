from __future__ import annotations

from typing import Optional

from .trace import _ChromeTraceCollector


def build_trace_collector(enabled: bool) -> Optional[_ChromeTraceCollector]:
    if not enabled:
        return None
    return _ChromeTraceCollector()
