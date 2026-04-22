from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .runtime_profiler import NoOpProfiler, RuntimeProfiler
from .printing import PrintingOptions


@dataclass
class ProfilingOptions:
    enabled: bool = True
    write_trace_json: bool = False
    trace_json_path: Optional[str] = None
    capture_component_breakdown: bool = True


def build_profiling_options(
    *,
    enabled: bool,
    config: Optional[Dict[str, Any]] = None,
    write_trace_json: bool = False,
    trace_json_path: Optional[str] = None,
    capture_component_breakdown: bool = True,
) -> ProfilingOptions:
    raw = dict(config or {})
    return ProfilingOptions(
        enabled=bool(raw.get("enabled", enabled)),
        write_trace_json=bool(raw.get("write_trace_json", write_trace_json)),
        trace_json_path=raw.get("trace_json_path", trace_json_path),
        capture_component_breakdown=bool(
            raw.get("capture_component_breakdown", capture_component_breakdown)
        ),
    )


def build_profiler(
    profiling_options: ProfilingOptions,
    printing_options: PrintingOptions,
):
    should_time = bool(profiling_options.enabled or printing_options.profile_iter or printing_options.profile_level or printing_options.profile_run)
    return RuntimeProfiler() if should_time else NoOpProfiler()
