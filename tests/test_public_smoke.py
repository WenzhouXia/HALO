from __future__ import annotations

import numpy as np

from halo_public import HALOConfig, MGPDConfig, solve_halo, solve_mgpd
from halo_public.halo import solve as halo_solve
from halo_public.mgpd.hierarchy import GridHierarchy
from halo_public.mgpd import solve as mgpd_solve

try:
    import pycupdlpx  # noqa: F401
    HAS_CUPDLPX = True
except Exception:
    HAS_CUPDLPX = False


def _make_hist(seed: int, resolution: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hist = rng.random((resolution, resolution), dtype=np.float32)
    hist /= hist.sum()
    return hist.astype(np.float32, copy=False)


def test_package_exports_match_submodule_exports() -> None:
    hs = _make_hist(1)
    ht = _make_hist(2)
    out_a = solve_mgpd(hs, ht, MGPDConfig(max_inner_iter=2, num_scales=1))
    out_b = mgpd_solve(hs, ht, MGPDConfig(max_inner_iter=2, num_scales=1))
    assert np.isfinite(float(out_a["distance"]))
    assert abs(float(out_a["distance"]) - float(out_b["distance"])) < 1e-6


def test_mgpd_smoke() -> None:
    hs = _make_hist(3, resolution=8)
    ht = _make_hist(4, resolution=8)
    out = solve_mgpd(hs, ht, MGPDConfig(max_inner_iter=2, num_scales=2))
    assert np.isfinite(float(out["distance"]))
    assert out["level_summaries"]
    assert out["solver_mode"] == "grid"


def test_mgpd_adaptive_default_stops_at_16x16() -> None:
    hist = _make_hist(10, resolution=256)
    hierarchy = GridHierarchy(num_scales=None)
    hierarchy.build(hist)
    assert hierarchy.num_levels == 5
    assert hierarchy.levels[-1].points.shape[0] == 16 * 16


def test_halo_smoke_and_l1() -> None:
    rng = np.random.default_rng(5)
    xs = rng.normal(size=(16, 2)).astype(np.float32)
    xt = rng.normal(size=(16, 2)).astype(np.float32)
    out_l2 = solve_halo(xs, xt, config=HALOConfig(max_inner_iter=2, cost_type="l2^2"))
    out_l1 = halo_solve(xs, xt, config=HALOConfig(max_inner_iter=2, cost_type="l1"))
    assert np.isfinite(float(out_l2["distance"]))
    assert np.isfinite(float(out_l1["distance"]))
    assert out_l2["level_summaries"]
    assert out_l2["solver_mode"] == "tree"


def test_cupdlpx_smoke_if_available() -> None:
    if not HAS_CUPDLPX:
        return
    hs = _make_hist(7, resolution=4)
    ht = _make_hist(8, resolution=4)
    out_grid = solve_mgpd(hs, ht, MGPDConfig(max_inner_iter=1, num_scales=1, solver_engine="cupdlpx"))
    rng = np.random.default_rng(9)
    xs = rng.normal(size=(8, 2)).astype(np.float32)
    xt = rng.normal(size=(8, 2)).astype(np.float32)
    out_tree = solve_halo(xs, xt, config=HALOConfig(max_inner_iter=1, solver_engine="cupdlpx", tree_lp_form="primal"))
    assert np.isfinite(float(out_grid["distance"]))
    assert np.isfinite(float(out_tree["distance"]))
