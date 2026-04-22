from ..lp_solvers.wrapper import LPSolver, SolverResult
import time
import numpy as np
from contextlib import nullcontext
import ctypes
from pathlib import Path
try:
    import hprlp
    HPRLP_AVAILABLE = True
except ImportError:
    HPRLP_AVAILABLE = False
    print("警告: 未找到 hprlp 模块。HPRLPSolver 将不可用。")
import logging
logger = logging.getLogger(__name__)


def _ensure_hprlp_runtime():
    if not HPRLP_AVAILABLE:
        raise ImportError("hprlp module not installed.")

    pkg_dir = Path(hprlp.__file__).resolve().parent
    candidate_libs = (
        pkg_dir / "libhprlp.so.0",
        pkg_dir / "libhprlp.so",
    )
    for lib_path in candidate_libs:
        if not lib_path.exists():
            continue
        try:
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
            return hprlp
        except OSError as exc:
            logger.warning("Failed to preload HPRLP runtime %s: %s", lib_path, exc)

    raise ImportError(
        f"Unable to preload HPRLP runtime library from {pkg_dir}. "
        "Expected libhprlp.so.0 or libhprlp.so to be present and loadable."
    )


class HPRLPSolver(LPSolver):
    def solve(
        self,
        c,
        A_csc,
        b_eq,
        lb,
        ub,
        n_eqs,
        warm_start_primal=None,
        warm_start_dual=None,
        tolerance: dict = {'objective': 1e-6, 'primal': 1e-6, 'dual': 1e-6},
        verbose=1,
        trace_collector=None,
        trace_prefix: str = "lp_backend",
        **kwargs,
    ) -> SolverResult:
        hprlp_module = _ensure_hprlp_runtime()
        del kwargs

        def _trace_span(name: str):
            if trace_collector is None:
                return nullcontext()
            return trace_collector.span(name, "solve_ot")

        # 1. 准备数据
        # HPRLP 示例建议使用 CSR 格式，这里将 CSC 转为 CSR 以确保兼容性
        A = A_csc.tocsr()

        # 对应等式约束 Ax = b，设置 AL = b, AU = b
        AL = b_eq
        AU = b_eq

        n_vars = len(c)

        # 2. 处理 Warm Start (严格遵循：None 则创建全 0 数组)
        if warm_start_primal is None:
            x_init = np.zeros(n_vars)
        else:
            x_init = np.array(warm_start_primal)

        if warm_start_dual is None:
            y_init = np.zeros(n_eqs)
        else:
            y_init = np.array(warm_start_dual)

        if verbose:
            logger.info(f" 调用 HPRLP (n_vars={n_vars})...")

        # 3. 创建模型
        # hprlp.Model.from_arrays(A, AL, AU, l, u, c, x_init, y_init)
        t_start_model = time.perf_counter()
        with _trace_span(f"{trace_prefix}.construct_solver"):
            model = hprlp_module.Model.from_arrays(A, AL, AU, lb, ub, c, x_init, y_init)
        construct_time = float(time.perf_counter() - t_start_model)
        if verbose:
            logger.info(
                f" Model created in {construct_time:.4f} seconds")

        # 4. 设置参数
        params = hprlp_module.Parameters()
        # 注意：使用修正后的参数名 primal_tol
        params.primal_tol = tolerance.get('primal', 1e-6)
        params.dual_tol = tolerance.get('dual', 1e-6)
        params.gap_tol = tolerance.get('objective', 1e-6)

        # 根据示例和旧代码风格，显式关闭 scaling
        params.use_bc_scaling = False
        params.use_Ruiz_scaling = False

        # 5. 求解
        t_solve = time.perf_counter()
        with _trace_span(f"{trace_prefix}.native_solve"):
            result = model.solve(params)
        native_solve_time = float(time.perf_counter() - t_solve)

        # 6. 结果提取
        # 由于未提供 result.status 的具体枚举映射，暂默认 success=True，并记录状态
        # 根据你的需求，从 result 对象中提取特定字段
        duration = result.time
        iterations = result.iter
        peak_mem = getattr(result, 'peak_mem', 0.0)
        if verbose:
            logger.info(
                f" Solve Done, Time = {duration:.2f}, nIter = {iterations}, Status = {result.status}")
        logger.info(
            f" [GPU Mem] HPRLP Peak Usage: {peak_mem:.2f} MiB")  # 新增日志

        # 7. 释放模型内存
        model.free()

        return SolverResult(
            success=True,  # 暂定为 True，可根据 result.status 进一步细化
            x=result.x,
            y=result.y,
            obj_val=result.primal_obj,
            duration=duration,
            iterations=iterations,
            peak_mem=peak_mem   # HPRLP 结果中未包含内存峰值信息，置为 0
            ,
            solver_diag={
                "construct_time": construct_time,
                "native_solve_wall_time": native_solve_time,
                "solver_name": "hprlp",
            },
        )
