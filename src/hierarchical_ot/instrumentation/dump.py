from __future__ import annotations

import datetime
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse import save_npz

from ..lp_solvers.wrapper import SolverResult


DEFAULT_CUPDLPX_DUMP_ROOT = Path("/home/xwz/project/HELLO/debug_cupdlpx/dumps")


def dump_cupdlpx_lp(
    *,
    c: np.ndarray,
    A_csc: sp.csc_matrix,
    b_eq: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    n_eqs: int,
    warm_start_primal: Optional[np.ndarray],
    warm_start_dual: Optional[np.ndarray],
    tolerance: Dict[str, float],
    params: Dict[str, Any],
    result_dict: Dict[str, Any],
    sol_dict: Optional[Dict[str, Any]],
    dump_root: Path = DEFAULT_CUPDLPX_DUMP_ROOT,
) -> Path:
    """Dump a failed CuPDLPx LP instance for offline debugging."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dump_dir = Path(dump_root) / f"debug_lp_dump_{timestamp}"
    dump_dir.mkdir(parents=True, exist_ok=False)

    np.save(dump_dir / "c_cost_vector.npy", np.asarray(c))
    np.save(dump_dir / "b_eq_rhs.npy", np.asarray(b_eq))
    np.save(dump_dir / "lb_lower_bounds.npy", np.asarray(lb))
    np.save(dump_dir / "ub_upper_bounds.npy", np.asarray(ub))
    save_npz(dump_dir / "A_constraint_matrix.npz", A_csc)

    np.save(
        dump_dir / "warm_start_primal.npy",
        np.asarray(warm_start_primal) if warm_start_primal is not None else np.array([]),
    )
    np.save(
        dump_dir / "warm_start_dual.npy",
        np.asarray(warm_start_dual) if warm_start_dual is not None else np.array([]),
    )
    np.save(dump_dir / "tolerance.npy", tolerance, allow_pickle=True)

    meta = {
        "n_vars": int(len(c)),
        "n_eqs": int(n_eqs),
        "nnz": int(A_csc.nnz),
        "shape": [int(A_csc.shape[0]), int(A_csc.shape[1])],
        "termination_reason": result_dict.get("termination_reason"),
        "runtime_sec": result_dict.get("runtime_sec"),
        "iterations": result_dict.get("iterations"),
        "time_sec_limit": params.get("time_sec_limit"),
    }
    (dump_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with open(dump_dir / "failed_result_dict.pkl", "wb") as f:
        pickle.dump(dict(result_dict), f)
    with open(dump_dir / "failed_solution_dict.pkl", "wb") as f:
        pickle.dump(sol_dict, f)

    return dump_dir


def solve_cupdlpx_lp_with_dump(
    *,
    c: np.ndarray,
    A_csc: sp.csc_matrix,
    b_eq: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    n_eqs: int,
    tolerance: Optional[Dict[str, float]] = None,
    warm_start_primal: Optional[np.ndarray] = None,
    warm_start_dual: Optional[np.ndarray] = None,
    verbose: int = 0,
    time_sec_limit: float = 60.0,
    dump_root: Path = DEFAULT_CUPDLPX_DUMP_ROOT,
) -> SolverResult:
    """Solve one CuPDLPx LP and dump the instance if the solver does not report OPTIMAL."""
    import pycupdlpx

    tol = dict(tolerance or {"objective": 1e-6, "primal": 1e-6, "dual": 1e-6})
    solver = pycupdlpx.cupdlpx()
    solver.loadData(A=A_csc, c=c, rhs=b_eq, lb=lb, ub=ub, nEqs=n_eqs)

    if (warm_start_primal is not None and len(warm_start_primal) == len(c)) or (
        warm_start_dual is not None and len(warm_start_dual) == n_eqs
    ):
        solver.setInitSol(warm_start_primal, warm_start_dual)

    params = pycupdlpx.make_default_params()
    params.update(
        {
            "verbose": verbose,
            "verbose_time": 1,
            "step_size_method": 3,
            "l_inf_ruiz_iterations": 0,
            "bound_objective_rescaling": False,
            "has_pock_chambolle_alpha": True,
            "time_sec_limit": float(time_sec_limit),
            "eps_optimal_relative": tol.get("objective"),
            "eps_feasible_relative_primal": tol.get("primal"),
            "eps_feasible_relative_dual": tol.get("dual"),
        }
    )

    result_dict = solver.solve(params)
    try:
        sol_dict = solver.getSolution()
    except Exception:
        sol_dict = None

    is_optimal = result_dict.get("termination_reason") == "OPTIMAL"
    if not is_optimal:
        dump_dir = dump_cupdlpx_lp(
            c=c,
            A_csc=A_csc,
            b_eq=b_eq,
            lb=lb,
            ub=ub,
            n_eqs=n_eqs,
            warm_start_primal=warm_start_primal,
            warm_start_dual=warm_start_dual,
            tolerance=tol,
            params=params,
            result_dict=result_dict,
            sol_dict=sol_dict,
            dump_root=Path(dump_root),
        )
        print(
            f"[cupdlpx dump] termination={result_dict.get('termination_reason')} "
            f"dumped_to={dump_dir}"
        )

    save_info = sol_dict.get("SaveInfo") if isinstance(sol_dict, dict) else None
    if isinstance(sol_dict, dict):
        if save_info == 2:
            primal_feas = sol_dict.get("PrimalFeasAvgRel", None)
            dual_feas = sol_dict.get("DualFeasAvgRel", None)
            gap = sol_dict.get("RelObjGapAverage", None)
        else:
            primal_feas = sol_dict.get("PrimalFeasRel", None)
            dual_feas = sol_dict.get("DualFeasRel", None)
            gap = sol_dict.get("RelObjGap", None)
    else:
        primal_feas = result_dict.get("PrimalFeasRel", None)
        dual_feas = result_dict.get("DualFeasRel", None)
        gap = result_dict.get("RelObjGap", None)

    return SolverResult(
        success=is_optimal,
        x=(
            sol_dict.get("x").copy()
            if is_optimal and isinstance(sol_dict, dict) and sol_dict.get("x") is not None
            else (
                result_dict.get("x").copy()
                if is_optimal and result_dict.get("x") is not None
                else None
            )
        ),
        y=(
            sol_dict.get("y").copy()
            if is_optimal and isinstance(sol_dict, dict) and sol_dict.get("y") is not None
            else (
                result_dict.get("y").copy()
                if is_optimal and result_dict.get("y") is not None
                else None
            )
        ),
        obj_val=(
            sol_dict.get("PrimalObj")
            if is_optimal and isinstance(sol_dict, dict) and sol_dict.get("PrimalObj") is not None
            else (result_dict.get("primal_objective") if is_optimal else 0.0)
        ),
        duration=result_dict.get("runtime_sec", 0.0),
        iterations=(
            sol_dict.get("iters")
            if is_optimal and isinstance(sol_dict, dict) and sol_dict.get("iters") is not None
            else (result_dict.get("iterations") if is_optimal else 0)
        ),
        peak_mem=result_dict.get("peak_gpu_mem_mib", 0.0),
        termination_reason=result_dict.get("termination_reason"),
        primal_feas=primal_feas,
        dual_feas=dual_feas,
        gap=gap,
    )


__all__ = [
    "DEFAULT_CUPDLPX_DUMP_ROOT",
    "dump_cupdlpx_lp",
    "solve_cupdlpx_lp_with_dump",
]
