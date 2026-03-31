from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import numpy as np
from scipy import sparse as sp

from ..lp_solvers.wrapper import LPSolver, SolverResult

try:
    import pycupdlpx

    CUPDLPX_AVAILABLE = True
except ImportError:
    CUPDLPX_AVAILABLE = False

logger = logging.getLogger(__name__)


class CuPDLPxSolver(LPSolver):

    supports_tree_lp_form = True

    @staticmethod
    def _copy_if_present(mapping: Optional[Dict[str, Any]], key: str):
        if isinstance(mapping, dict) and mapping.get(key) is not None:
            return mapping[key]
        return None

    @staticmethod
    def _extract_metrics(
        result_dict: Dict[str, Any],
        sol_dict: Optional[Dict[str, Any]],
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        save_info = sol_dict.get("SaveInfo") if isinstance(sol_dict, dict) else None
        if isinstance(sol_dict, dict):
            if save_info == 2:
                primal_feas = sol_dict.get("PrimalFeasAvgRel")
                dual_feas = sol_dict.get("DualFeasAvgRel")
                gap = sol_dict.get("RelObjGapAverage")
            else:
                primal_feas = sol_dict.get("PrimalFeasRel")
                dual_feas = sol_dict.get("DualFeasRel")
                gap = sol_dict.get("RelObjGap")
        else:
            primal_feas = result_dict.get("PrimalFeasRel")
            dual_feas = result_dict.get("DualFeasRel")
            gap = result_dict.get("RelObjGap")
        return primal_feas, dual_feas, gap

    @staticmethod
    def _build_params(
        tolerance: Dict[str, float],
        verbose: int,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params = {
            "verbose": verbose,
            # Disable native timing summary unless explicitly verbose to avoid
            # uncontrollable C++ stdout noise and timing jitter.
            "verbose_time": 1 if int(verbose) else 0,
            "print_summary": bool(verbose),
            "termination_evaluation_frequency": 200,
            "step_size_method": 3,
            "eps_optimal_relative": tolerance.get("objective"),
            "eps_feasible_relative_primal": tolerance.get("primal"),
            "eps_feasible_relative_dual": tolerance.get("dual"),
            "l_inf_ruiz_iterations": 0,
            "bound_objective_rescaling": False,
            "has_pock_chambolle_alpha": True,
            "hybrid_refine_iterations": 100,
            "stepsize_power_reference": False,
            "stepsize_reference_max_iterations": 5000,
        }
        if isinstance(extra_params, dict):
            params.update(extra_params)
        return params

    @staticmethod
    def _maybe_set_init_sol_primal_layout(
        solver,
        warm_start_primal,
        warm_start_dual,
        n_primal: int,
        n_dual: int,
    ) -> None:
        primal_ok = warm_start_primal is not None and len(warm_start_primal) == n_primal
        dual_ok = warm_start_dual is not None and len(warm_start_dual) == n_dual
        if not (primal_ok or dual_ok):
            return
        x0 = np.asarray(warm_start_primal, dtype=np.float32) if primal_ok else np.zeros(n_primal, dtype=np.float32)
        y0 = np.asarray(warm_start_dual, dtype=np.float32) if dual_ok else np.zeros(n_dual, dtype=np.float32)
        solver.setInitSol(x0, y0)

    @staticmethod
    def _maybe_set_init_sol_dual_layout(
        solver,
        warm_start_primal,
        warm_start_dual,
        n_dual: int,
        n_primal: int,
    ) -> None:
        dual_ok = warm_start_dual is not None and len(warm_start_dual) == n_dual
        primal_ok = warm_start_primal is not None and len(warm_start_primal) == n_primal
        if not (dual_ok or primal_ok):
            return
        x0 = np.asarray(warm_start_dual, dtype=np.float32) if dual_ok else np.zeros(n_dual, dtype=np.float32)
        y0 = np.asarray(warm_start_primal, dtype=np.float32) if primal_ok else np.zeros(n_primal, dtype=np.float32)
        solver.setInitSol(x0, y0)

    def solve(
        self,
        c,
        A_csc,
        b_eq,
        lb,
        ub,
        n_eqs,
        tolerance: Dict[str, float] = {"objective": 1e-6, "primal": 1e-6, "dual": 1e-6},
        warm_start_primal=None,
        warm_start_dual=None,
        verbose=0,
        lp_form: str = "primal",
        dual_form_data: Optional[Dict[str, Any]] = None,
        solver_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SolverResult:
        if not CUPDLPX_AVAILABLE:
            raise ImportError("pycupdlpx module not installed.")

        form = str(lp_form).strip().lower()
        if form not in {"primal", "dual"}:
            raise ValueError(f"Unknown lp_form: {lp_form!r}, expected 'primal' or 'dual'")

        solver = pycupdlpx.cupdlpx()
        solver_diag: Dict[str, Any] = {"lp_form": form}

        def _as_float32_or_none(arr):
            if arr is None:
                return None
            return np.asarray(arr, dtype=np.float32)

        if form == "primal":
            t0 = time.perf_counter()
            solver.loadData(A=A_csc, c=c, rhs=b_eq, lb=lb, ub=ub, nEqs=n_eqs)
            solver_diag["load_data_time"] = float(time.perf_counter() - t0)
            t0 = time.perf_counter()
            self._maybe_set_init_sol_primal_layout(
                solver,
                warm_start_primal=warm_start_primal,
                warm_start_dual=warm_start_dual,
                n_primal=len(c),
                n_dual=n_eqs,
            )
            solver_diag["warm_start_time"] = float(time.perf_counter() - t0)
        else:
            payload = dual_form_data or {}
            minus_AT = payload.get("minus_AT")
            neg_minus_AT = payload.get("neg_minus_AT")
            minus_c = payload.get("minus_c")
            minus_q = payload.get("minus_q")

            if minus_AT is None or minus_c is None:
                raise ValueError("dual_form_data must contain 'minus_AT' and 'minus_c'")

            if neg_minus_AT is None:
                minus_AT_csc = minus_AT if sp.isspmatrix_csc(minus_AT) else minus_AT.tocsc()
                A_load = -minus_AT_csc
            else:
                A_load = neg_minus_AT if sp.isspmatrix_csc(neg_minus_AT) else neg_minus_AT.tocsc()
                minus_AT_csc = minus_AT if sp.isspmatrix_csc(minus_AT) else minus_AT.tocsc()

            minus_c = np.asarray(minus_c, dtype=np.float64)
            if minus_q is None:
                if b_eq is None:
                    raise ValueError("dual_form_data missing 'minus_q' and fallback b_eq is None")
                minus_q = -np.asarray(b_eq, dtype=np.float64)
            else:
                minus_q = np.asarray(minus_q, dtype=np.float64)

            if minus_AT_csc.shape[0] != minus_c.shape[0]:
                raise ValueError(
                    f"dual form shape mismatch: minus_AT rows={minus_AT_csc.shape[0]} vs minus_c={minus_c.shape[0]}"
                )
            if minus_AT_csc.shape[1] != minus_q.shape[0]:
                raise ValueError(
                    f"dual form shape mismatch: minus_AT cols={minus_AT_csc.shape[1]} vs minus_q={minus_q.shape[0]}"
                )

            n_dual = minus_q.shape[0]
            n_primal = minus_AT_csc.shape[0]
            lb_dual = np.full(n_dual, -np.inf, dtype=np.float64)
            ub_dual = np.full(n_dual, np.inf, dtype=np.float64)

            t0 = time.perf_counter()
            solver.loadData(
                A=A_load,
                c=minus_q,
                rhs=-minus_c,
                lb=lb_dual,
                ub=ub_dual,
                nEqs=0,
            )
            solver_diag["load_data_time"] = float(time.perf_counter() - t0)
            t0 = time.perf_counter()
            self._maybe_set_init_sol_dual_layout(
                solver,
                warm_start_primal=warm_start_primal,
                warm_start_dual=warm_start_dual,
                n_dual=n_dual,
                n_primal=n_primal,
            )
            solver_diag["warm_start_time"] = float(time.perf_counter() - t0)

        params = self._build_params(
            tolerance=tolerance,
            verbose=verbose,
            extra_params=solver_params,
        )
        logger.info(f"  Solver tolerance: {tolerance}")
        logger.info(f"  Calling cuPDLPx (lp_form={form}, n_vars={len(c)})...")

        t0 = time.perf_counter()
        result_dict = solver.solve(params)
        solver_diag["native_solve_wall_time"] = float(time.perf_counter() - t0)

        try:
            t0 = time.perf_counter()
            sol_dict = solver.getSolution()
            solver_diag["get_solution_time"] = float(time.perf_counter() - t0)
        except Exception:
            sol_dict = None
            solver_diag["get_solution_time"] = 0.0

        peak_mem = result_dict.get("peak_gpu_mem_mib", 0.0)
        duration = result_dict.get("runtime_sec", 0.0)
        iterations = result_dict.get("iterations", 0)
        is_optimal = result_dict.get("termination_reason") == "OPTIMAL"
        primal_feas, dual_feas, gap = self._extract_metrics(result_dict, sol_dict)

        logger.info(f"  [GPU Mem] C++ Solver Peak Usage: {peak_mem:.2f} MiB")
        if duration:
            logger.info(
                f"  Solve Done, Time = {duration:.2f}, nIter = {iterations:,}, iter/sec = {iterations / duration:.2f}"
            )
        else:
            logger.info(f"  Solve Done, Time = {duration:.2f}, nIter = {iterations:,}")
        if not is_optimal:
            logger.warning(f"  Solve failed or did not reach optimality. Reason: {result_dict.get('termination_reason')}")

        if form == "primal":
            primal_sol = self._copy_if_present(sol_dict, "x") if is_optimal else None
            if primal_sol is None and is_optimal:
                primal_sol = self._copy_if_present(result_dict, "x")
            dual_sol = self._copy_if_present(sol_dict, "y") if is_optimal else None
            if dual_sol is None and is_optimal:
                dual_sol = self._copy_if_present(result_dict, "y")
        else:
            primal_sol = self._copy_if_present(sol_dict, "y") if is_optimal else None
            if primal_sol is None and is_optimal:
                primal_sol = self._copy_if_present(result_dict, "y")
            dual_sol = self._copy_if_present(sol_dict, "x") if is_optimal else None
            if dual_sol is None and is_optimal:
                dual_sol = self._copy_if_present(result_dict, "x")

        t0 = time.perf_counter()
        primal_sol = _as_float32_or_none(primal_sol)
        dual_sol = _as_float32_or_none(dual_sol)
        solver_diag["extract_solution_time"] = float(time.perf_counter() - t0)
        solver_diag["runtime_sec"] = float(duration)

        obj_val = (
            sol_dict.get("PrimalObj")
            if is_optimal and isinstance(sol_dict, dict) and sol_dict.get("PrimalObj") is not None
            else (result_dict.get("primal_objective") if is_optimal else 0.0)
        )
        dual_obj_val = (
            sol_dict.get("DualObj")
            if is_optimal and isinstance(sol_dict, dict) and sol_dict.get("DualObj") is not None
            else (result_dict.get("dual_objective") if is_optimal else None)
        )
        iter_count = (
            sol_dict.get("iters")
            if is_optimal and isinstance(sol_dict, dict) and sol_dict.get("iters") is not None
            else (result_dict.get("iterations") if is_optimal else 0)
        )

        return SolverResult(
            success=is_optimal,
            x=primal_sol,
            y=dual_sol,
            obj_val=obj_val,
            dual_obj_val=dual_obj_val,
            duration=duration,
            iterations=iter_count,
            peak_mem=peak_mem,
            termination_reason=result_dict.get("termination_reason"),
            primal_feas=primal_feas,
            dual_feas=dual_feas,
            gap=gap,
            solver_diag=solver_diag,
        )
