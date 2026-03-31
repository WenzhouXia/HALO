from ..common.config import HALOConfig

__all__ = ["HALOConfig", "solve"]


def solve(*args, **kwargs):
    from .api import solve as _solve

    return _solve(*args, **kwargs)
