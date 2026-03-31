from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy import sparse

from ..common.config import HALOConfig
from ..common.solver import HierarchicalOTSolver
from ..lp_solvers import build_solver
from .hierarchy import TreeHierarchy, align_hierarchy_depths
from .shielding import ShieldingStrategy


def _coupling_from_payload(payload: Optional[Dict[str, np.ndarray]], shape: tuple[int, int]) -> sparse.coo_matrix:
    if not payload:
        return sparse.coo_matrix(shape, dtype=np.float32)
    return sparse.coo_matrix((payload["values"], (payload["rows"], payload["cols"])), shape=shape, dtype=np.float32)


def _payload_from_tree_result(result: Dict[str, Any], n_t: int) -> Optional[Dict[str, np.ndarray]]:
    primal = result.get("primal")
    if not isinstance(primal, dict):
        return None
    keep = np.asarray(primal.get("keep", []), dtype=np.int64)
    vals = np.asarray(primal.get("y", []), dtype=np.float32)
    if keep.size == 0 or vals.size == 0:
        return None
    rows = (keep // int(n_t)).astype(np.int32, copy=False)
    cols = (keep % int(n_t)).astype(np.int32, copy=False)
    flow = np.abs(vals).astype(np.float32, copy=False)
    mask = flow > 1e-12
    return {"rows": rows[mask], "cols": cols[mask], "values": flow[mask]}


def solve(
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_mass: Optional[np.ndarray] = None,
    target_mass: Optional[np.ndarray] = None,
    config: HALOConfig | None = None,
) -> Dict[str, Any]:
    cfg = HALOConfig() if config is None else config
    cfg.validate()
    source = np.asarray(source_points, dtype=np.float32, order="C")
    target = np.asarray(target_points, dtype=np.float32, order="C")
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source_points and target_points must be 2D arrays")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source_points and target_points must share the same feature dimension")
    n_s = int(source.shape[0])
    n_t = int(target.shape[0])
    if source_mass is None:
        source_mass = np.full(n_s, 1.0 / max(n_s, 1), dtype=np.float32)
    if target_mass is None:
        target_mass = np.full(n_t, 1.0 / max(n_t, 1), dtype=np.float32)
    source_mass = np.asarray(source_mass, dtype=np.float32, order="C")
    target_mass = np.asarray(target_mass, dtype=np.float32, order="C")

    hierarchy_s = TreeHierarchy(
        target_coarse_size=int(cfg.target_coarse_size),
        max_L0_L1_ratio=float(cfg.max_L0_L1_ratio),
    )
    hierarchy_t = TreeHierarchy(
        target_coarse_size=int(cfg.target_coarse_size),
        max_L0_L1_ratio=float(cfg.max_L0_L1_ratio),
    )
    hierarchy_s.build(source, source_mass)
    hierarchy_t.build(target, target_mass)
    align_hierarchy_depths(hierarchy_s, hierarchy_t)

    strategy = ShieldingStrategy(shield_impl=str(cfg.shield_impl), max_pairs_per_xA=int(cfg.max_pairs_per_xA), nnz_thr=float(cfg.nnz_thr))
    solver = HierarchicalOTSolver(
        hierarchy_s=hierarchy_s,
        hierarchy_t=hierarchy_t,
        strategy=strategy,
        solver=build_solver(cfg.solver_engine),
        lp_solver_verbose=bool(cfg.lp_solver_verbose),
    )
    result = solver.solve(
        mode="tree",
        cost_type=str(cfg.cost_type),
        max_inner_iter=int(cfg.max_inner_iter),
        convergence_criterion=str(cfg.convergence_criterion),
        objective_plateau_iters=int(cfg.objective_plateau_iters),
        tolerance=cfg.normalized_tolerance(),
        final_refinement_tolerance=cfg.normalized_final_refinement_tolerance(),
        enable_profiling=bool(cfg.enable_profiling),
        tree_debug=bool(cfg.tree_debug),
        tree_infeas_fallback=str(cfg.tree_infeas_fallback),
        tree_lp_form=str(cfg.tree_lp_form),
        check_type=str(cfg.check_type),
        use_last=bool(cfg.use_last),
        use_last_after_inner0=bool(cfg.use_last_after_inner0),
        ifcheck=bool(cfg.ifcheck),
        vd_thr=float(cfg.vd_thr),
        sampled_config=cfg.sampled_config,
        tree_infeas_use_cupy=bool(cfg.tree_infeas_use_cupy),
        shield_impl=str(cfg.shield_impl),
        max_pairs_per_xA=int(cfg.max_pairs_per_xA),
        nnz_thr=float(cfg.nnz_thr),
    )
    coupling_payload = result.get("sparse_coupling") or _payload_from_tree_result(result, n_t)
    coupling = _coupling_from_payload(coupling_payload, (n_s, n_t))
    return {
        "distance": float(result.get("final_obj", float("nan"))),
        "objective": float(result.get("final_obj", float("nan"))),
        "level_summaries": result.get("level_summaries", []),
        "tree_lp_diags": result.get("tree_lp_diags", []),
        "dual": result.get("dual"),
        "coupling": coupling,
        "solver_mode": "tree",
    }
