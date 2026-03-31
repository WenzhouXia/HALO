

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import DefaultDict, Dict, Iterable, Iterator, List, Optional

import torch

@dataclass
class _TimerHandle:
    elapsed: float = 0.0

class NoOpProfiler:
    

    enabled = False

    @contextmanager
    def timer(self, _name: str, *, gpu_possible: bool = False) -> Iterator[_TimerHandle]:
        del gpu_possible
        yield _TimerHandle(0.0)

    def start_run(self) -> None:
        return None

    def end_run(self) -> None:
        return None

    def start_level(self, _level_idx: int) -> None:
        return None

    def end_level(self, _level_idx: int, _iters: int) -> None:
        return None

    def start_iteration(self, _level_idx: int, _inner_iter: int) -> None:
        return None

    def end_iteration(self) -> None:
        return None

    def start_phase(self, _name: str, *, gpu_possible: bool = False) -> None:
        del gpu_possible
        return None

    def end_phase(self) -> None:
        return None

    def add_components(self, _components: Optional[Dict[str, float]]) -> None:
        return None

    def add_phase_time(self, _dt: float) -> None:
        return None

    def current_iteration_snapshot(self) -> Optional[Dict[str, object]]:
        return None

    def current_level_snapshot(self) -> Optional[Dict[str, object]]:
        return None

    def summary(self, *, build_time: float, solve_time: float) -> Optional[Dict[str, object]]:
        del build_time, solve_time
        return None

class RuntimeProfiler:
    

    enabled = True

    def __init__(self) -> None:
        self._cuda_available = bool(torch.cuda.is_available())
        self._run_components: DefaultDict[str, float] = defaultdict(float)
        self._run_phase_times: DefaultDict[str, float] = defaultdict(float)
        self._run_phases: Dict[str, object] = {}
        self._levels: Dict[int, Dict[str, object]] = {}
        self._level_order: List[int] = []
        self._current_level: Optional[Dict[str, object]] = None
        self._current_iteration: Optional[Dict[str, object]] = None
        self._current_phase: Optional[Dict[str, object]] = None
        self._run_t0 = time.perf_counter()
        self._run_time = 0.0

    def _new_phase(self, name: str, *, gpu_possible: bool = False) -> Dict[str, object]:
        return {
            "name": str(name),
            "time": 0.0,
            "_t0": time.perf_counter(),
            "_gpu_possible": bool(gpu_possible),
            "components": defaultdict(float),
        }

    def _new_level(self, level_idx: int) -> Dict[str, object]:
        return {
            "level_idx": int(level_idx),
            "num_iters": 0,
            "time": 0.0,
            "_t0": 0.0,
            "components": defaultdict(float),
            "phase_times": defaultdict(float),
            "phases": {},
            "iterations": [],
        }

    def _new_iteration(self, iter_idx: int) -> Dict[str, object]:
        return {
            "iter_idx": int(iter_idx),
            "time": 0.0,
            "_t0": time.perf_counter(),
            "components": defaultdict(float),
            "phase_times": defaultdict(float),
            "phases": {},
        }

    @staticmethod
    def _finalize_component_map(data: DefaultDict[str, float]) -> Dict[str, float]:
        return {k: float(v) for k, v in sorted(data.items()) if float(v) > 0.0}

    @staticmethod
    def _finalize_phase_map(data: DefaultDict[str, float]) -> Dict[str, float]:
        return {k: float(v) for k, v in sorted(data.items()) if float(v) >= 0.0}

    def _iter_component_buckets(self) -> Iterable[DefaultDict[str, float]]:
        yield self._run_components
        if self._current_level is not None:
            yield self._current_level["components"]  
        if self._current_iteration is not None:
            yield self._current_iteration["components"]  
        if self._current_phase is not None:
            yield self._current_phase["components"]  

    def _add_component(self, name: str, dt: float) -> None:
        if dt <= 0.0:
            return
        for bucket in self._iter_component_buckets():
            bucket[name] += float(dt)

    def start_run(self) -> None:
        self._run_t0 = time.perf_counter()
        self._run_time = 0.0

    def end_run(self) -> None:
        self._run_time = float(time.perf_counter() - self._run_t0)

    def start_level(self, level_idx: int) -> None:
        level = self._levels.get(level_idx)
        if level is None:
            level = self._new_level(level_idx)
            self._levels[level_idx] = level
            self._level_order.append(level_idx)
        level["_t0"] = time.perf_counter()
        self._current_level = level
        self._current_iteration = None
        self._current_phase = None

    def end_level(self, level_idx: int, iters: int) -> None:
        level = self._levels.get(level_idx)
        if level is not None:
            level["num_iters"] = int(iters)
            t0 = float(level.get("_t0", 0.0))
            if t0 > 0.0:
                level["time"] = float(time.perf_counter() - t0)
        self._current_phase = None
        self._current_iteration = None
        self._current_level = None

    def start_iteration(self, level_idx: int, inner_iter: int) -> None:
        level = self._levels.get(level_idx)
        if level is None:
            self.start_level(level_idx)
            level = self._levels[level_idx]
        iteration = self._new_iteration(inner_iter)
        level["iterations"].append(iteration)  
        self._current_level = level
        self._current_iteration = iteration
        self._current_phase = None

    def end_iteration(self) -> None:
        if self._current_iteration is not None:
            t0 = float(self._current_iteration.get("_t0", 0.0))
            if t0 > 0.0:
                self._current_iteration["time"] = float(time.perf_counter() - t0)
        self._current_phase = None
        self._current_iteration = None

    def start_phase(self, name: str, *, gpu_possible: bool = False) -> None:
        should_sync = self._cuda_available and gpu_possible
        if should_sync:
            torch.cuda.synchronize()
        phase = self._new_phase(name, gpu_possible=gpu_possible)
        if self._current_iteration is not None:
            self._current_iteration["phases"][name] = phase  
        elif self._current_level is not None:
            self._current_level["phases"][name] = phase  
        else:
            self._run_phases[name] = phase
        self._current_phase = phase

    def end_phase(self) -> None:
        if self._current_phase is None:
            return
        phase_name = str(self._current_phase["name"])
        should_sync = self._cuda_available and bool(self._current_phase.get("_gpu_possible", False))
        if should_sync:
            torch.cuda.synchronize()
        t0 = float(self._current_phase.get("_t0", 0.0))
        dt = float(time.perf_counter() - t0) if t0 > 0.0 else 0.0
        dt += float(self._current_phase.get("_extra_time", 0.0))
        self._current_phase["time"] = dt
        if self._current_iteration is not None:
            if self._current_level is not None:
                self._current_level["phase_times"][phase_name] += dt  
            self._current_iteration["phase_times"][phase_name] += dt  
        elif self._current_level is not None:
            self._current_level["phase_times"][phase_name] += dt  
        else:
            self._run_phase_times[phase_name] += dt
        self._current_phase = None

    @contextmanager
    def timer(self, name: str, *, gpu_possible: bool = False) -> Iterator[_TimerHandle]:
        should_sync = self._cuda_available and gpu_possible
        if should_sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        handle = _TimerHandle(0.0)
        try:
            yield handle
        finally:
            if should_sync:
                torch.cuda.synchronize()
            dt = float(time.perf_counter() - t0)
            handle.elapsed = dt
            self._add_component(name, dt)

    def add_components(self, components: Optional[Dict[str, float]]) -> None:
        if not components:
            return
        for name, value in components.items():
            dt = float(value)
            if dt > 0.0:
                self._add_component(name, dt)

    def add_phase_time(self, dt: float) -> None:
        if self._current_phase is None:
            return
        extra = float(dt)
        if extra <= 0.0:
            return
        self._current_phase["_extra_time"] = float(self._current_phase.get("_extra_time", 0.0)) + extra

    def _snapshot_phase_dict(self, phases: Dict[str, object]) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for name, phase in phases.items():
            phase_dict = phase  
            t0 = float(phase_dict.get("_t0", 0.0))
            dt = float(phase_dict.get("time", 0.0))
            if dt <= 0.0 and t0 > 0.0:
                dt = float(time.perf_counter() - t0)
                dt += float(phase_dict.get("_extra_time", 0.0))
            out[str(name)] = {
                "time": dt,
                "components": self._finalize_component_map(phase_dict["components"]),  
            }
        return out

    def _snapshot_iteration(self, iteration: Dict[str, object]) -> Dict[str, object]:
        iter_time = float(iteration.get("time", 0.0))
        t0 = float(iteration.get("_t0", 0.0))
        if iter_time <= 0.0 and t0 > 0.0:
            iter_time = float(time.perf_counter() - t0)
        return {
            "iter_idx": int(iteration.get("iter_idx", -1)),
            "time": iter_time,
            "components": self._finalize_component_map(iteration["components"]),  
            "phase_times": self._finalize_phase_map(iteration["phase_times"]),  
            "phases": self._snapshot_phase_dict(iteration["phases"]),  
        }

    def _snapshot_level(self, level: Dict[str, object]) -> Dict[str, object]:
        iterations = [
            self._snapshot_iteration(iteration)
            for iteration in level["iterations"]  
        ]
        level_time = float(level.get("time", 0.0))
        t0 = float(level.get("_t0", 0.0))
        if level_time <= 0.0 and t0 > 0.0:
            level_time = float(time.perf_counter() - t0)
        return {
            "level_idx": int(level.get("level_idx", -1)),
            "time": level_time,
            "num_iters": int(level.get("num_iters", 0)),
            "components": self._finalize_component_map(level["components"]),  
            "phase_times": self._finalize_phase_map(level["phase_times"]),  
            "phases": self._snapshot_phase_dict(level["phases"]),  
            "iterations": iterations,
        }

    def current_iteration_snapshot(self) -> Optional[Dict[str, object]]:
        if self._current_iteration is None:
            return None
        return self._snapshot_iteration(self._current_iteration)

    def current_level_snapshot(self) -> Optional[Dict[str, object]]:
        if self._current_level is None:
            return None
        return self._snapshot_level(self._current_level)

    def summary(self, *, build_time: float, solve_time: float) -> Optional[Dict[str, object]]:
        levels_out: List[Dict[str, object]] = []
        for level_idx in self._level_order:
            levels_out.append(self._snapshot_level(self._levels[level_idx]))

        run = {
            "time": float(self._run_time if self._run_time > 0.0 else solve_time),
            "components": self._finalize_component_map(self._run_components),
            "phase_times": self._finalize_phase_map(self._run_phase_times),
            "phases": self._snapshot_phase_dict(self._run_phases),
            "levels": levels_out,
        }

        alias_levels: List[Dict[str, object]] = []
        for level in levels_out:
            alias_levels.append(
                {
                    "level": int(level["level_idx"]),
                    "iters": int(level["num_iters"]),
                    "time": float(level["time"]),
                    "components": dict(level["components"]),
                    "inner_iters": [
                        {
                            "iter": int(iteration["iter_idx"]),
                            "time": float(iteration["time"]),
                            "components": dict(iteration["components"]),
                        }
                        for iteration in level["iterations"]  
                    ],
                }
            )

        return {
            "enabled": True,
            "schema_version": 2,
            "cuda_available": bool(self._cuda_available),
            "cuda_sync_mode": "component_only_minimal",
            "build_time": float(build_time),
            "solve_time": float(solve_time),
            "total_time": float(build_time + solve_time),
            "run": run,
            "components": dict(run["components"]),
            "levels": alias_levels,
        }
