from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Union

ToleranceInput = Union[float, Dict[str, float]]


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


def _normalize_tolerance_input(tol: ToleranceInput) -> Dict[str, float]:
    if isinstance(tol, dict):
        return {
            "objective": float(tol.get("objective", 1e-6)),
            "primal": float(tol.get("primal", 1e-6)),
            "dual": float(tol.get("dual", 1e-6)),
        }
    scalar = float(tol)
    return {"objective": scalar, "primal": scalar, "dual": scalar}


@dataclass
class BaseConfig:
    solver_engine: Literal["scipy", "cupdlpx"] = "scipy"
    tolerance: ToleranceInput = 1e-6
    lp_solver_verbose: bool = False
    enable_profiling: bool = False
    max_inner_iter: int = 10
    convergence_criterion: Literal["strict", "objective", "objective_and_violation"] = "objective"
    objective_plateau_iters: int = 1
    final_refinement_tolerance: Optional[ToleranceInput] = field(default_factory=lambda: {"objective": 1e-8, "primal": 1e-8, "dual": 1e-8})

    def normalized_tolerance(self) -> Dict[str, float]:
        return _normalize_tolerance_input(self.tolerance)

    def normalized_final_refinement_tolerance(self) -> Optional[Dict[str, float]]:
        if self.final_refinement_tolerance is None:
            return None
        return _normalize_tolerance_input(self.final_refinement_tolerance)

    def validate_common(self) -> None:
        if self.solver_engine not in {"scipy", "cupdlpx"}:
            raise ValueError("solver_engine must be one of: scipy, cupdlpx")
        self.lp_solver_verbose = _parse_bool_flag(self.lp_solver_verbose, default=False)
        self.enable_profiling = _parse_bool_flag(self.enable_profiling, default=False)
        self.max_inner_iter = int(self.max_inner_iter)
        if self.max_inner_iter <= 0:
            raise ValueError("max_inner_iter must be > 0")
        self.objective_plateau_iters = int(self.objective_plateau_iters)
        if self.objective_plateau_iters <= 0:
            raise ValueError("objective_plateau_iters must be > 0")
        _normalize_tolerance_input(self.tolerance)
        self.normalized_final_refinement_tolerance()


@dataclass
class MGPDConfig(BaseConfig):
    cost_type: Literal["l2^2"] = "l2^2"
    p: int = 2
    stop_tolerance: Optional[ToleranceInput] = None
    num_scales: Optional[int] = None
    max_inner_iter: int = 5
    use_last: bool = True
    use_last_after_inner0: bool = False
    check_type: Literal["cpu", "gpu_exact", "gpu_sampled"] = "gpu_exact"
    vd_thr: float = 0.25
    if_shield: bool = True
    if_check: bool = True
    use_primal_feas_ori: bool = True
    adap_primal_tol: bool = True
    new_mvp: bool = True
    aty_type: int = 0
    coarsest_full_support: bool = True

    def normalized_stop_tolerance(self) -> Dict[str, float]:
        if self.stop_tolerance is None:
            return self.normalized_tolerance()
        return _normalize_tolerance_input(self.stop_tolerance)

    def validate(self) -> None:
        self.validate_common()
        if self.cost_type != "l2^2":
            raise ValueError("MGPDConfig only supports l2^2 in the public package.")
        self.p = int(self.p)
        if self.p < 1:
            raise ValueError("p must be >= 1")
        if self.num_scales is not None and int(self.num_scales) < 0:
            raise ValueError("num_scales must be >= 0")


@dataclass
class HALOConfig(BaseConfig):
    cost_type: Literal["l2^2", "l1", "linf"] = "l2^2"
    max_inner_iter: int = 10
    search_method: str = "tree_numba"
    shield_impl: Literal["local", "halo"] = "local"
    target_coarse_size: int = 128
    max_L0_L1_ratio: float = 2.5
    use_last: bool = True
    use_last_after_inner0: bool = True
    ifcheck: bool = True
    vd_thr: float = 0.25
    max_pairs_per_xA: int = 30
    nnz_thr: float = 1e-20
    tree_debug: bool = False
    tree_infeas_fallback: Literal["none", "full_support_retry", "scipy_verify"] = "none"
    tree_lp_form: Literal["primal", "dual"] = "primal"
    check_type: Literal["cpu", "gpu", "gpu_approx", "auto"] = "auto"
    sampled_config: Optional[Dict[str, Any]] = None
    tree_infeas_use_cupy: bool = True

    def validate(self) -> None:
        self.validate_common()
        if self.cost_type not in {"l2^2", "l1", "linf"}:
            raise ValueError("HALOConfig supports l2^2, l1, linf only.")
        self.target_coarse_size = int(self.target_coarse_size)
        if self.target_coarse_size <= 0:
            raise ValueError("target_coarse_size must be > 0")
        self.max_pairs_per_xA = int(self.max_pairs_per_xA)
        if self.max_pairs_per_xA <= 0:
            raise ValueError("max_pairs_per_xA must be > 0")
