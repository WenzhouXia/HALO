from __future__ import annotations

from .cupdlpx import CuPDLPxSolver


class TreeCuPDLPxSolver(CuPDLPxSolver):
    """兼容旧导入路径的别名类。

    新代码应直接使用 `CuPDLPxSolver`。
    """

    pass
