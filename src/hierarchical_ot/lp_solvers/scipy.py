from .wrapper import LPSolver, SolverResult
from scipy.optimize import linprog
from contextlib import nullcontext
import time
import logging
logger = logging.getLogger(__name__)


class SciPySolver(LPSolver):
    verbose = False  # SciPySolver 不支持 verbose，但保留属性以便 isinstance 检查

    def solve(
        self,
        c,
        A_csc,
        b_eq,
        lb,
        ub,
        n_eqs,
        tolerance: dict = {'objective': 1e-6, 'primal': 1e-6, 'dual': 1e-6},
        warm_start_primal=None,
        warm_start_dual=None,
        verbose: int = 0,
        trace_collector=None,
        trace_prefix: str = "lp_backend",
        **kwargs,
    ) -> SolverResult:
        def _trace_span(name: str):
            if trace_collector is None:
                return nullcontext()
            return trace_collector.span(name, "solve_ot")

        bounds = list(zip(lb, ub))
        options = {'dual_feasibility_tolerance': tolerance.get('dual'), 'primal_feasibility_tolerance': tolerance.get(
            'primal'), 'ipm_optimality_tolerance': tolerance.get('objective')}
        logger.info(f"  使用容差值: {tolerance}")
        logger.info(f"  调用 scipy.optimize.linprog (n_vars={len(c)})...")
        start_time = time.time()
        with _trace_span(f"{trace_prefix}.native_solve"):
            result_obj = linprog(c, A_eq=A_csc, b_eq=b_eq,
                                 bounds=bounds, method='highs', options=options)
        duration = time.time() - start_time
        logger.info(f"  求解完成, 耗时: {duration:.4f} 秒。")

        if not result_obj.success:
            logger.warning(f"  警告: 求解失败。状态: {result_obj.message}")
        y = result_obj.eqlin.marginals if result_obj.success else None

        return SolverResult(
            success=result_obj.success,
            x=result_obj.x if result_obj.success else None,
            y=y,
            obj_val=result_obj.fun if result_obj.success else 0.0,
            duration=duration,
            termination_reason=str(result_obj.message),
            solver_diag={"native_solve_wall_time": float(duration)},
        )
