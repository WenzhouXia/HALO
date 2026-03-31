from ..common.config import MGPDConfig

__all__ = ["MGPDConfig", "solve"]


def solve(*args, **kwargs):
    from .api import solve as _solve

    return _solve(*args, **kwargs)
