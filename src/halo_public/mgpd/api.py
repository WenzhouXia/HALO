from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy import sparse

from ..common.config import MGPDConfig
from ..common.solver import HierarchicalOTSolver
from ..lp_solvers import build_solver
from .hierarchy import GridHierarchy


def _coupling_from_payload(payload: Dict[str, np.ndarray], shape: tuple[int, int]) -> sparse.coo_matrix:
    if not payload:
        return sparse.coo_matrix(shape, dtype=np.float32)
    return sparse.coo_matrix((payload["values"], (payload["rows"], payload["cols"])), shape=shape, dtype=np.float32)


class _GridNoOpStrategy:
    def close(self) -> None:
        return None


def solve(source_hist: np.ndarray, target_hist: np.ndarray, config: MGPDConfig | None = None) -> Dict[str, Any]:
    cfg = MGPDConfig() if config is None else config
    cfg.validate()
    source = np.asarray(source_hist, dtype=np.float32, order="C")
    target = np.asarray(target_hist, dtype=np.float32, order="C")
    if source.ndim != 2 or source.shape[0] != source.shape[1]:
        raise ValueError("source_hist must be a square 2D array")
    if target.ndim != 2 or target.shape[0] != target.shape[1]:
        raise ValueError("target_hist must be a square 2D array")
    if source.shape != target.shape:
        raise ValueError("source_hist and target_hist must have the same shape")
    source = source / max(float(source.sum()), 1e-12)
    target = target / max(float(target.sum()), 1e-12)

    hierarchy_s = GridHierarchy(num_scales=cfg.num_scales)
    hierarchy_t = GridHierarchy(num_scales=cfg.num_scales)
    hierarchy_s.build(source)
    hierarchy_t.build(target)

    solver = HierarchicalOTSolver(
        hierarchy_s=hierarchy_s,
        hierarchy_t=hierarchy_t,
        strategy=_GridNoOpStrategy(),
        solver=build_solver(cfg.solver_engine),
        lp_solver_verbose=bool(cfg.lp_solver_verbose),
    )
    result = solver.solve(
        mode="grid",
        cost_type="l2^2",
        max_inner_iter=int(cfg.max_inner_iter),
        convergence_criterion=str(cfg.convergence_criterion),
        objective_plateau_iters=int(cfg.objective_plateau_iters),
        tolerance=cfg.normalized_tolerance(),
        final_refinement_tolerance=cfg.normalized_final_refinement_tolerance(),
        enable_profiling=bool(cfg.enable_profiling),
        grid_p=int(cfg.p),
        stop_tolerance=cfg.normalized_stop_tolerance(),
        use_last=bool(cfg.use_last),
        use_last_after_inner0=bool(cfg.use_last_after_inner0),
        if_check=bool(cfg.if_check),
        if_shield=bool(cfg.if_shield),
        coarsest_full_support=bool(cfg.coarsest_full_support),
        vd_thr=float(cfg.vd_thr),
        grid_check_type=str(cfg.check_type),
        use_primal_feas_ori=bool(cfg.use_primal_feas_ori),
        adap_primal_tol=bool(cfg.adap_primal_tol),
        new_mvp=bool(cfg.new_mvp),
        aty_type=int(cfg.aty_type),
    )
    result = solver._materialize_grid_result_payload(result, need_final_obj=True, need_sparse_coupling=True)
    coupling = _coupling_from_payload(result.get("sparse_coupling"), (source.size, target.size))
    return {
        "distance": float(result.get("final_obj", float("nan"))),
        "objective": float(result.get("final_obj", float("nan"))),
        "level_summaries": result.get("level_summaries", []),
        "grid_lp_diags": result.get("grid_lp_diags", []),
        "dual": result.get("dual"),
        "coupling": coupling,
        "solver_mode": "grid",
    }
