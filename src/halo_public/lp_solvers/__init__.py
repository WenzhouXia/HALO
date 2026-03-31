from .scipy import SciPySolver

try:
    from .cupdlpx import CuPDLPxSolver
except Exception:
    CuPDLPxSolver = None

__all__ = ["SciPySolver", "CuPDLPxSolver", "build_solver"]


def build_solver(solver_engine: str):
    engine = str(solver_engine).strip().lower()
    if engine == "scipy":
        return SciPySolver()
    if engine == "cupdlpx":
        if CuPDLPxSolver is None:
            raise ImportError("CuPDLPxSolver is unavailable. Ensure pycupdlpx is installed.")
        return CuPDLPxSolver()
    raise ValueError(f"Unknown solver_engine: {solver_engine}")
