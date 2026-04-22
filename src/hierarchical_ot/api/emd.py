from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..solvers.single_level import _run_emd2_with_config as _run_emd2_with_config
from ..types.config import ConfigType, GridConfig, _normalize_config_for_mode


def _normalize_grid_masses(
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    *,
    source_name: str,
    target_name: str,
    prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    src_mass = np.asarray(source_mass, dtype=np.float32, order="C")
    tgt_mass = np.asarray(target_mass, dtype=np.float32, order="C")
    if src_mass.ndim != 2 or src_mass.shape[0] != src_mass.shape[1]:
        raise ValueError(f"{prefix} requires {source_name} to be a square 2D array.")
    if tgt_mass.ndim != 2 or tgt_mass.shape[0] != tgt_mass.shape[1]:
        raise ValueError(f"{prefix} requires {target_name} to be a square 2D array.")
    if src_mass.shape != tgt_mass.shape:
        raise ValueError(f"{prefix} requires {source_name} and {target_name} to have the same shape.")
    return src_mass, tgt_mass


def emd2(
    source_X: np.ndarray,
    target_X: np.ndarray,
    source_mass: Optional[np.ndarray] = None,
    target_mass: Optional[np.ndarray] = None,
    log: bool = False,
    return_coupling: bool = False,
    return_state: bool = False,
    config: Optional[ConfigType] = None,
) -> Any:
    if config is None:
        raise TypeError("emd2 now requires 'config'.")

    if isinstance(config, GridConfig):
        if source_mass is None or target_mass is None:
            raise TypeError(
                "grid mode requires source_mass and target_mass as square 2D arrays; "
                "do not pass grid masses via source_X/target_X."
            )
        src_mass, tgt_mass = _normalize_grid_masses(
            source_mass,
            target_mass,
            source_name="source_mass",
            target_name="target_mass",
            prefix="grid mode",
        )
        n_s = int(src_mass.size)
        n_t = int(tgt_mass.size)
        source_mass = src_mass
        target_mass = tgt_mass
    else:
        source_arr = np.asarray(source_X)
        target_arr = np.asarray(target_X)
        n_s = int(source_arr.shape[0])
        n_t = int(target_arr.shape[0])

    cfg, mode = _normalize_config_for_mode(config=config, n_s=n_s, n_t=n_t)
    return _run_emd2_with_config(
        source_X=source_X,
        target_X=target_X,
        source_mass=source_mass,
        target_mass=target_mass,
        log=log,
        return_coupling=return_coupling,
        return_state=return_state,
        config=cfg,
        mode=mode,
    )


def emd2_grid(
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    *,
    config: GridConfig,
    log: bool = False,
    return_coupling: bool = False,
    return_state: bool = False,
) -> Any:
    src_mass, tgt_mass = _normalize_grid_masses(
        source_mass,
        target_mass,
        source_name="source_mass",
        target_name="target_mass",
        prefix="emd2_grid",
    )
    cfg, mode = _normalize_config_for_mode(
        config=config,
        n_s=int(src_mass.size),
        n_t=int(tgt_mass.size),
    )
    if mode != "grid":
        raise TypeError("emd2_grid requires config to be a GridConfig instance.")
    return _run_emd2_with_config(
        source_X=None,
        target_X=None,
        source_mass=src_mass,
        target_mass=tgt_mass,
        log=log,
        return_coupling=return_coupling,
        return_state=return_state,
        config=cfg,
        mode="grid",
    )
