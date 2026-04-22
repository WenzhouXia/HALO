"""Parameter configuration classes for the public tree/grid HALO subset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple, Union
import warnings


ToleranceInput = Union[float, Dict[str, float]]
RuntimeLoggingInput = Optional[Dict[str, Any]]
PrintingInput = Optional[Dict[str, Any]]
ProfilingInput = Optional[Dict[str, Any]]


def _parse_bool_flag(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _normalize_cost_type_name(cost_type: str) -> str:
    c = str(cost_type).strip().lower()
    if c in ("l1",):
        return "l1"
    if c in ("linf", "l_inf"):
        return "linf"
    if c in ("l2", "euclidean"):
        return "l2"
    if c in ("l2^2", "l2sq", "sqeuclidean", "sq_euclidean"):
        return "l2^2"
    if c in ("lp",):
        return "lp"
    raise ValueError(
        f"Unknown cost_type: {cost_type}. Supported names: l1, linf, l2, l2^2, lp."
    )


def _normalize_printing_config(
    printing: PrintingInput,
    runtime_logging: RuntimeLoggingInput,
) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "enabled": True,
        "progress": True,
        "profile_iter": True,
        "profile_level": True,
        "profile_run": True,
        "warm_start": True,
        "iter_interval": 10,
    }
    if printing is not None and runtime_logging is not None:
        if dict(printing) == dict(runtime_logging):
            return dict(printing)
        raise ValueError("Use either printing or runtime_logging, not both.")

    raw_config = printing if printing is not None else runtime_logging
    if raw_config is None:
        return dict(defaults)
    if not isinstance(raw_config, dict):
        config_name = "printing" if printing is not None else "runtime_logging"
        raise ValueError(f"{config_name} must be a dict when provided")

    allowed_keys = set(defaults) | {"verbosity"}
    unknown = set(raw_config) - allowed_keys
    if unknown:
        config_name = "printing" if printing is not None else "runtime_logging"
        raise ValueError(
            f"{config_name} only supports keys: "
            f"{sorted(allowed_keys)}; got unknown keys={sorted(unknown)}"
        )

    if runtime_logging is not None and printing is None:
        warnings.warn(
            "runtime_logging is deprecated; use printing instead.",
            DeprecationWarning,
            stacklevel=3,
        )

    normalized = dict(defaults)
    for key in ("enabled", "progress", "profile_iter", "profile_level", "profile_run", "warm_start"):
        if key in raw_config:
            normalized[key] = _parse_bool_flag(raw_config.get(key), default=bool(defaults[key]))
    if "iter_interval" in raw_config:
        normalized["iter_interval"] = int(raw_config["iter_interval"])
    if int(normalized["iter_interval"]) <= 0:
        raise ValueError("printing.iter_interval must be > 0")
    if "verbosity" in raw_config:
        normalized["verbosity"] = str(raw_config["verbosity"])
    return normalized


def _normalize_profiling_config(
    profiling: ProfilingInput,
    *,
    enable_profiling: bool,
) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "enabled": bool(enable_profiling),
        "write_trace_json": False,
        "trace_json_path": None,
        "capture_component_breakdown": True,
    }
    if profiling is None:
        return defaults
    if not isinstance(profiling, dict):
        raise ValueError("profiling must be a dict when provided")

    unknown = set(profiling) - set(defaults)
    if unknown:
        raise ValueError(
            "profiling only supports keys: "
            f"{sorted(defaults)}; got unknown keys={sorted(unknown)}"
        )

    normalized = dict(defaults)
    for key in ("enabled", "write_trace_json", "capture_component_breakdown"):
        if key in profiling:
            normalized[key] = _parse_bool_flag(profiling.get(key), default=bool(defaults[key]))
    if "trace_json_path" in profiling:
        trace_json_path = profiling.get("trace_json_path")
        normalized["trace_json_path"] = None if trace_json_path is None else str(trace_json_path)
    return normalized


@dataclass
class SolverConfig:
    cost_type: str = "l2^2"
    solver_engine: Literal["cupdlpx", "scipy", "hprlp"] = "cupdlpx"
    tolerance: ToleranceInput = 1e-6
    convergence_tolerance: Optional[ToleranceInput] = None
    lp_solver_verbose: bool = False
    enable_profiling: bool = True
    printing: PrintingInput = None
    profiling: ProfilingInput = None
    runtime_logging: RuntimeLoggingInput = None
    max_inner_iter: int = 100
    convergence_criterion: Literal[
        "strict", "objective", "objective_and_violation"
    ] = "objective"
    objective_plateau_iters: int = 1
    final_refinement_tolerance: Optional[ToleranceInput] = field(
        default_factory=lambda: {"objective": 1e-8, "primal": 1e-8, "dual": 1e-8}
    )

    def normalized_tolerance(self) -> Dict[str, Any]:
        return self._normalize_tolerance_input(self.tolerance)

    def normalized_convergence_tolerance(self) -> Dict[str, Any]:
        if self.convergence_tolerance is None:
            return self.normalized_tolerance()
        return self._normalize_tolerance_input(self.convergence_tolerance)

    def normalized_printing(self) -> Dict[str, Any]:
        return _normalize_printing_config(self.printing, self.runtime_logging)

    def normalized_profiling(self) -> Dict[str, Any]:
        return _normalize_profiling_config(
            self.profiling,
            enable_profiling=_parse_bool_flag(self.enable_profiling, default=True),
        )

    def normalized_runtime_logging(self) -> Dict[str, Any]:
        return self.normalized_printing()

    def normalized_final_refinement_tolerance(self) -> Optional[Dict[str, Any]]:
        if self.final_refinement_tolerance is None:
            return None
        return self._normalize_tolerance_input(self.final_refinement_tolerance)

    def validate_common(self) -> None:
        self.cost_type = _normalize_cost_type_name(self.cost_type)
        if self.solver_engine not in {"cupdlpx", "scipy", "hprlp"}:
            raise ValueError(f"Unknown solver_engine: {self.solver_engine}")
        self.lp_solver_verbose = _parse_bool_flag(self.lp_solver_verbose, default=False)
        self.enable_profiling = _parse_bool_flag(self.enable_profiling, default=False)
        self.printing = self.normalized_printing()
        self.runtime_logging = dict(self.printing)
        self.profiling = self.normalized_profiling()
        self.enable_profiling = bool(self.profiling["enabled"])
        if int(self.max_inner_iter) <= 0:
            raise ValueError("max_inner_iter must be > 0")
        if self.convergence_criterion not in {
            "strict",
            "objective",
            "objective_and_violation",
        }:
            raise ValueError(
                "convergence_criterion must be one of: strict, objective, objective_and_violation"
            )
        if int(self.objective_plateau_iters) <= 0:
            raise ValueError("objective_plateau_iters must be > 0")
        self.objective_plateau_iters = int(self.objective_plateau_iters)
        _ = self.normalized_tolerance()
        _ = self.normalized_convergence_tolerance()
        _ = self.normalized_final_refinement_tolerance()

    @staticmethod
    def _normalize_tolerance_input(tol: ToleranceInput) -> Dict[str, Any]:
        if isinstance(tol, dict):
            out: Dict[str, Any] = {
                "objective": float(tol.get("objective", 1e-6)),
                "primal": float(tol.get("primal", 1e-6)),
                "dual": float(tol.get("dual", 1e-6)),
            }
            if "inf_thrs" in tol:
                out["inf_thrs"] = tol["inf_thrs"]
            if "stop_thrs" in tol:
                out["stop_thrs"] = tol["stop_thrs"]
            return out

        scalar = float(tol)
        return {"objective": scalar, "primal": scalar, "dual": scalar}


@dataclass
class TreeConfig(SolverConfig):
    max_inner_iter: int = 10
    search_method: str = "tree_numba"
    shield_impl: Literal["local", "halo"] = "halo"
    target_coarse_size: int = 128
    max_L0_L1_ratio: float = 2.5
    use_last: bool = True
    use_last_after_inner0: bool = True
    ifcheck: bool = True
    vd_thr: float = 0.25
    max_pairs_per_xA: int = 30
    nnz_thr: float = 1e-20
    tree_debug: bool = False
    tree_infeas_fallback: Literal["none"] = "none"
    tree_lp_form: Literal["primal", "dual"] = "dual"
    check_type: Literal["cpu", "gpu", "gpu_approx", "auto"] = "gpu_approx"
    sampled_config: Optional[Dict[str, Any]] = None
    tree_infeas_use_cupy: bool = True

    def validate(self) -> None:
        self.validate_common()
        if self.cost_type not in {"l2^2", "l1", "linf"}:
            raise ValueError(
                "tree mode currently supports: l2^2, l1, linf; "
                f"got {self.cost_type}"
            )
        self.search_method = str(self.search_method).lower()
        self.shield_impl = str(self.shield_impl).lower()  # type: ignore[assignment]
        if self.shield_impl not in {"local", "halo"}:
            raise ValueError("shield_impl must be 'local' or 'halo'")
        if int(self.target_coarse_size) <= 0:
            raise ValueError("target_coarse_size must be > 0")
        self.target_coarse_size = int(self.target_coarse_size)
        self.max_L0_L1_ratio = float(self.max_L0_L1_ratio)
        if self.max_L0_L1_ratio <= 0.0:
            raise ValueError("max_L0_L1_ratio must be > 0")
        self.use_last = _parse_bool_flag(self.use_last, default=True)
        self.use_last_after_inner0 = _parse_bool_flag(self.use_last_after_inner0, default=True)
        self.ifcheck = _parse_bool_flag(self.ifcheck, default=True)
        self.vd_thr = float(self.vd_thr)
        if int(self.max_pairs_per_xA) <= 0:
            raise ValueError("max_pairs_per_xA must be > 0")
        self.nnz_thr = float(self.nnz_thr)
        self.tree_debug = _parse_bool_flag(self.tree_debug, default=False)
        self.tree_lp_form = str(self.tree_lp_form).lower()  # type: ignore[assignment]
        if self.tree_lp_form not in {"primal", "dual"}:
            raise ValueError("tree_lp_form must be 'primal' or 'dual'")
        self.check_type = str(self.check_type).lower()  # type: ignore[assignment]
        if self.check_type in {"cupy", "approx"}:
            self.check_type = "gpu_approx"
        elif self.check_type in {"gpu_full", "full_gpu"}:
            self.check_type = "gpu"
        if self.check_type not in {"cpu", "gpu", "gpu_approx", "auto"}:
            raise ValueError("check_type must be one of: cpu, gpu, gpu_approx, auto")
        self.tree_infeas_fallback = str(self.tree_infeas_fallback).lower()  # type: ignore[assignment]
        if self.tree_infeas_fallback != "none":
            raise ValueError("tree_infeas_fallback must be 'none' in the strict tree execution path")
        self.tree_infeas_use_cupy = _parse_bool_flag(self.tree_infeas_use_cupy, default=True)


@dataclass
class GridConfig(SolverConfig):
    cost_type: str = "l2^2"
    p: int = 2
    solver_engine: Literal["cupdlpx", "scipy", "hprlp"] = "cupdlpx"
    stop_tolerance: Optional[ToleranceInput] = None
    num_scales: Optional[int] = None
    max_inner_iter: int = 5
    use_last: bool = True
    use_last_after_inner0: bool = False
    check_type: Literal["cpu", "gpu_exact", "gpu_sampled"] = "gpu_exact"
    vd_thr: float = 0.0625
    if_shield: bool = True
    if_check: bool = True
    use_primal_feas_ori: bool = True
    adap_primal_tol: bool = True
    new_mvp: bool = True
    aty_type: int = 0
    coarsest_full_support: bool = True

    def normalized_stop_tolerance(self) -> Dict[str, Any]:
        if self.stop_tolerance is None:
            return self.normalized_tolerance()
        return self._normalize_tolerance_input(self.stop_tolerance)

    def validate(self) -> None:
        self.validate_common()
        if self.cost_type != "l2^2":
            raise ValueError(
                "grid mode mainline currently only supports: l2^2; "
                f"got {self.cost_type}"
            )
        self.p = int(self.p)
        if self.p < 1:
            raise ValueError("GridConfig.p must be >= 1")
        if self.num_scales is not None and int(self.num_scales) < 0:
            raise ValueError("num_scales must be >= 0 when provided")
        if self.solver_engine not in {"cupdlpx", "scipy"}:
            raise ValueError("grid mode mainline currently supports solver_engine: cupdlpx, scipy")
        self.use_last = _parse_bool_flag(self.use_last, default=True)
        self.use_last_after_inner0 = _parse_bool_flag(self.use_last_after_inner0, default=False)
        self.if_shield = _parse_bool_flag(self.if_shield, default=True)
        self.if_check = _parse_bool_flag(self.if_check, default=True)
        self.use_primal_feas_ori = _parse_bool_flag(self.use_primal_feas_ori, default=True)
        self.adap_primal_tol = _parse_bool_flag(self.adap_primal_tol, default=True)
        self.new_mvp = _parse_bool_flag(self.new_mvp, default=True)
        self.aty_type = int(self.aty_type)
        self.coarsest_full_support = _parse_bool_flag(self.coarsest_full_support, default=True)
        self.vd_thr = float(self.vd_thr)
        self.check_type = str(self.check_type).lower()  # type: ignore[assignment]
        if self.check_type in {"gpu", "exact"}:
            self.check_type = "gpu_exact"
        elif self.check_type in {"sampled", "approx"}:
            self.check_type = "gpu_sampled"
        if self.check_type not in {"cpu", "gpu_exact"}:
            raise ValueError("grid mode mainline currently supports check_type: cpu, gpu_exact")
        _ = self.normalized_stop_tolerance()


ConfigType = Union[TreeConfig, GridConfig]


def _normalize_config_for_mode(
    config: ConfigType,
    n_s: int,
    n_t: int,
) -> Tuple[ConfigType, Literal["tree", "grid"]]:
    del n_s, n_t
    if isinstance(config, TreeConfig):
        config.validate()
        return config, "tree"
    if isinstance(config, GridConfig):
        config.validate()
        return config, "grid"
    raise TypeError("HALO_public config must be an instance of TreeConfig or GridConfig")


def create_config(method: Literal["tree", "grid"], **kwargs: Any) -> ConfigType:
    if method == "tree":
        config: ConfigType = TreeConfig(**kwargs)
    elif method == "grid":
        config = GridConfig(**kwargs)
    else:
        raise ValueError("HALO_public only supports methods: tree, grid")

    config.validate()
    return config


__all__ = [
    "ToleranceInput",
    "RuntimeLoggingInput",
    "PrintingInput",
    "ProfilingInput",
    "SolverConfig",
    "TreeConfig",
    "GridConfig",
    "ConfigType",
    "_normalize_config_for_mode",
    "create_config",
]
