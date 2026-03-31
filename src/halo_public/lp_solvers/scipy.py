from .wrapper import LPSolver, SolverResult
from scipy.optimize import linprog
import time
import logging
logger = logging.getLogger(__name__)

class SciPySolver(LPSolver):
    verbose = False  

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
        **kwargs,
    ) -> SolverResult:
        bounds = list(zip(lb, ub))
        options = {'dual_feasibility_tolerance': tolerance.get('dual'), 'primal_feasibility_tolerance': tolerance.get(
            'primal'), 'ipm_optimality_tolerance': tolerance.get('objective')}
        logger.info(f"  Solver tolerance: {tolerance}")
        logger.info(f"  Calling scipy.optimize.linprog (n_vars={len(c)})...")
        start_time = time.time()
        result_obj = linprog(c, A_eq=A_csc, b_eq=b_eq,
                             bounds=bounds, method='highs', options=options)
        duration = time.time() - start_time
        logger.info(f"  Solve finished in {duration:.4f}s.")

        if not result_obj.success:
            logger.warning(f"  Solve failed. Status: {result_obj.message}")
        y = result_obj.eqlin.marginals if result_obj.success else None

        return SolverResult(
            success=result_obj.success,
            x=result_obj.x if result_obj.success else None,
            y=y,
            obj_val=result_obj.fun if result_obj.success else 0.0,
            duration=duration,
            termination_reason=str(result_obj.message),
        )
