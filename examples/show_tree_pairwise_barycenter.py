#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ot

EXAMPLE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
ASSET_DIR = EXAMPLE_ROOT / "assets"
OUTPUT_ROOT = EXAMPLE_ROOT / "outputs" / "show_tree_pairwise_barycenter"
OUTPUT_DIR_2D = OUTPUT_ROOT / "2d"
OUTPUT_DIR_3D = OUTPUT_ROOT / "3d"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hierarchical_ot import TreeConfig, emd2


def _normalize_weights(weights: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(weights, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative.")
    total = float(arr.sum())
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError(f"{name} must have positive finite total mass.")
    return arr / total


def _as_coupling(coupling: Any) -> np.ndarray:
    if hasattr(coupling, "toarray"):
        return np.asarray(coupling.toarray(), dtype=np.float64)
    return np.asarray(coupling, dtype=np.float64)


def _solve_ot_tree(
    Xs: np.ndarray,
    ws: np.ndarray,
    Xt: np.ndarray,
    wt: np.ndarray,
    *,
    tree_config: TreeConfig,
) -> np.ndarray:
    _, coupling = emd2(
        Xs,
        Xt,
        source_mass=np.asarray(ws, dtype=np.float32),
        target_mass=np.asarray(wt, dtype=np.float32),
        return_coupling=True,
        config=TreeConfig(**vars(tree_config)),
    )
    return _as_coupling(coupling)


def _solve_ot_emd(
    Xs: np.ndarray,
    ws: np.ndarray,
    Xt: np.ndarray,
    wt: np.ndarray,
) -> np.ndarray:
    cost = ot.dist(
        np.asarray(Xs, dtype=np.float64),
        np.asarray(Xt, dtype=np.float64),
        metric="sqeuclidean",
    )
    return np.asarray(
        ot.emd(
            _normalize_weights(np.asarray(ws, dtype=np.float64), name="ws"),
            _normalize_weights(np.asarray(wt, dtype=np.float64), name="wt"),
            cost,
        ),
        dtype=np.float64,
    )


def _pairwise_one_step_barycenter(
    X_list: list[np.ndarray],
    a_list: list[np.ndarray],
    *,
    lam: np.ndarray,
    solve_ot: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    n_measures = len(X_list)
    couplings: list[list[np.ndarray | None]] = [[None for _ in range(n_measures)] for _ in range(n_measures)]
    for i in range(n_measures):
        couplings[i][i] = np.diag(a_list[i]).astype(np.float64, copy=False)
        for j in range(i + 1, n_measures):
            coupling_arr = solve_ot(X_list[i], a_list[i], X_list[j], a_list[j])
            couplings[i][j] = coupling_arr
            couplings[j][i] = coupling_arr.T

    Y_blocks: list[np.ndarray] = []
    b_blocks: list[np.ndarray] = []
    for i, (Xi, ai) in enumerate(zip(X_list, a_list)):
        Yi = np.zeros_like(Xi, dtype=np.float64)
        for j, Xj in enumerate(X_list):
            Pij = couplings[i][j]
            if Pij is None:
                raise RuntimeError("Missing pairwise coupling.")
            Yi += lam[j] * (Pij @ Xj)
        Y_blocks.append(Yi / ai[:, None])
        b_blocks.append(lam[i] * ai)
    return np.concatenate(Y_blocks, axis=0), np.concatenate(b_blocks, axis=0)


def _reference_one_step_barycenter(
    X_list: list[np.ndarray],
    a_list: list[np.ndarray],
    *,
    lam: np.ndarray,
    ref_idx: int,
    solve_ot: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 <= int(ref_idx) < len(X_list):
        raise ValueError("ref_idx is out of range.")
    ref_idx = int(ref_idx)
    Xr = X_list[ref_idx]
    ar = a_list[ref_idx]
    Y = np.zeros_like(Xr, dtype=np.float64)
    for idx, (Xi, ai) in enumerate(zip(X_list, a_list)):
        if idx == ref_idx:
            Pi = np.diag(ar).astype(np.float64, copy=False)
        else:
            Pi = solve_ot(Xr, ar, Xi, ai)
        Y += lam[idx] * (Pi @ Xi)
    return Y / ar[:, None], ar.copy()


def _exact_barycenter_objective(
    X: np.ndarray,
    b: np.ndarray,
    measures_locations: list[np.ndarray],
    measures_weights: list[np.ndarray],
    lam: np.ndarray,
) -> float:
    total = 0.0
    for lambda_i, Xi, ai in zip(lam, measures_locations, measures_weights):
        cost = ot.dist(
            np.asarray(X, dtype=np.float64),
            np.asarray(Xi, dtype=np.float64),
            metric="sqeuclidean",
        )
        total += float(lambda_i) * float(
            ot.emd2(
                np.asarray(b, dtype=np.float64),
                np.asarray(ai, dtype=np.float64),
                cost,
            )
        )
    return total


def _load_2d_measures() -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    red = plt.imread(ASSET_DIR / "red_cross.png").astype(np.float64)[::4, ::4, :3]
    blue = plt.imread(ASSET_DIR / "blue_circle.png").astype(np.float64)[::4, ::4, :3]

    sz = blue.shape[0]
    xx, yy = np.meshgrid(np.arange(sz), np.arange(sz))
    red_mask = (red[..., 0] > 0.5) & (red[..., 1] < 0.2) & (red[..., 2] < 0.2)
    blue_mask = (blue[..., 2] > 0.5) & (blue[..., 0] < 0.2) & (blue[..., 1] < 0.5)

    x1 = np.stack((xx[red_mask], yy[red_mask]), axis=1).astype(np.float64)
    x2 = np.stack((xx[blue_mask] + 80, -yy[blue_mask] + 32), axis=1).astype(np.float64)
    a1 = _normalize_weights(np.ones(x1.shape[0], dtype=np.float64), name="a1")
    a2 = _normalize_weights(np.ones(x2.shape[0], dtype=np.float64), name="a2")
    return [x1, x2], [a1, a2], ["red cross", "blue circle"]


def _subsample_points(points: np.ndarray, num_points: int, *, seed: int) -> np.ndarray:
    if num_points <= 0:
        raise ValueError("num_points must be > 0.")
    if points.shape[0] < num_points:
        raise ValueError(
            f"Requested {num_points} points, but asset only has {points.shape[0]} points."
        )
    if points.shape[0] == num_points:
        return np.asarray(points, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(points.shape[0], size=num_points, replace=False))
    return np.asarray(points[idx], dtype=np.float64)


def _separate_point_clouds_along_x(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    gap: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    x1 = np.asarray(source_points, dtype=np.float64).copy()
    x2 = np.asarray(target_points, dtype=np.float64).copy()
    max_x1 = float(np.max(x1[:, 0]))
    min_x2 = float(np.min(x2[:, 0]))
    shift_x = max_x1 - min_x2 + float(gap)
    x2[:, 0] += shift_x
    return x1, x2


def _load_3d_measures(num_points: int) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    chair = np.load(ASSET_DIR / "chair_0001_points_65536.npz")["points"]
    toilet = np.load(ASSET_DIR / "toilet_0001_points_65536.npz")["points"]
    x1 = _subsample_points(np.asarray(chair, dtype=np.float32), num_points, seed=0)
    x2 = _subsample_points(np.asarray(toilet, dtype=np.float32), num_points, seed=1)
    x1, x2 = _separate_point_clouds_along_x(x1, x2, gap=3.0)
    a1 = _normalize_weights(np.ones(x1.shape[0], dtype=np.float64), name="a1")
    a2 = _normalize_weights(np.ones(x2.shape[0], dtype=np.float64), name="a2")
    return [x1, x2], [a1, a2], ["chair_0001", "toilet_0001"]


def _plot_distributions_2d(
    measures_locations: list[np.ndarray],
    labels: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.scatter(measures_locations[0][:, 0], measures_locations[0][:, 1], alpha=0.5, label=labels[0])
    ax.scatter(measures_locations[1][:, 0], measures_locations[1][:, 1], alpha=0.5, label=labels[1])
    ax.set_title("Input distributions")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_barycenter_2d(
    measures_locations: list[np.ndarray],
    labels: list[str],
    X_bar: np.ndarray,
    b_bar: np.ndarray,
    output_path: Path,
    *,
    title: str,
    label: str,
    objective_value: float | None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.scatter(measures_locations[0][:, 0], measures_locations[0][:, 1], alpha=0.35, label=labels[0])
    ax.scatter(measures_locations[1][:, 0], measures_locations[1][:, 1], alpha=0.35, label=labels[1])
    ax.scatter(X_bar[:, 0], X_bar[:, 1], s=np.asarray(b_bar) * 800.0, marker="s", label=label)
    if objective_value is None:
        ax.set_title(title)
    else:
        ax.set_title(f"{title}\nexact objective={objective_value:.6f}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_distributions_3d(
    measures_locations: list[np.ndarray],
    labels: list[str],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(12, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        measures_locations[0][:, 0],
        measures_locations[0][:, 1],
        measures_locations[0][:, 2],
        s=0.6,
        alpha=0.08,
        label=labels[0],
    )
    ax.scatter(
        measures_locations[1][:, 0],
        measures_locations[1][:, 1],
        measures_locations[1][:, 2],
        s=0.6,
        alpha=0.08,
        label=labels[1],
    )
    _configure_3d_axes(ax, measures_locations)
    ax.set_title("Input point clouds")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_barycenter_3d(
    measures_locations: list[np.ndarray],
    labels: list[str],
    X_bar: np.ndarray,
    output_path: Path,
    *,
    title: str,
) -> None:
    fig = plt.figure(figsize=(9, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        measures_locations[0][:, 0],
        measures_locations[0][:, 1],
        measures_locations[0][:, 2],
        s=0.6,
        alpha=0.08,
        label=labels[0],
    )
    ax.scatter(
        measures_locations[1][:, 0],
        measures_locations[1][:, 1],
        measures_locations[1][:, 2],
        s=0.6,
        alpha=0.08,
        label=labels[1],
    )
    ax.scatter(X_bar[:, 0], X_bar[:, 1], X_bar[:, 2], s=0.6, alpha=0.08, label="wasserstein barycenter")
    _configure_3d_axes(ax, [*measures_locations, X_bar])
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _configure_3d_axes(ax: Any, clouds: list[np.ndarray]) -> None:
    all_points = np.concatenate([np.asarray(cloud, dtype=np.float64) for cloud in clouds], axis=0)
    mins = np.min(all_points, axis=0)
    maxs = np.max(all_points, axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    centers = 0.5 * (mins + maxs)
    padded_spans = spans * 1.08
    ax.set_xlim(centers[0] - 0.5 * padded_spans[0], centers[0] + 0.5 * padded_spans[0])
    ax.set_ylim(centers[1] - 0.5 * padded_spans[1], centers[1] + 0.5 * padded_spans[1])
    ax.set_zlim(centers[2] - 0.5 * padded_spans[2], centers[2] + 0.5 * padded_spans[2])
    # Compress the display width of x so the intentional source-target gap
    # does not visually squash all three point clouds.
    ax.set_box_aspect((padded_spans[0], padded_spans[1], padded_spans[2]))
    ax.view_init(elev=18, azim=-62)
    ax.grid(False)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def _load_measures(
    dimension: int,
    *,
    sample_points_3d: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    if dimension == 2:
        return _load_2d_measures()
    if dimension == 3:
        return _load_3d_measures(sample_points_3d)
    raise ValueError(f"Unsupported dimension: {dimension}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a one-step barycenter using hierarchical_ot tree mode."
    )
    parser.add_argument("--solver-engine", choices=("cupdlpx", "scipy"), default="cupdlpx")
    parser.add_argument("--dimension", choices=(2, 3), type=int, default=2)
    parser.add_argument("--sample-points-3d", type=int, default=8192)
    parser.add_argument("--method", choices=("reference", "pairwise"), default="reference")
    parser.add_argument("--ref-idx", type=int, default=0)
    args = parser.parse_args()

    measures_locations, measures_weights, labels = _load_measures(
        int(args.dimension),
        sample_points_3d=int(args.sample_points_3d),
    )
    lam = np.full(len(measures_locations), 1.0 / len(measures_locations), dtype=np.float64)
    tree_config = TreeConfig(
        cost_type="l2^2",
        solver_engine=str(args.solver_engine),
        max_inner_iter=4,
        tolerance={"objective": 1e-5, "primal": 1e-5, "dual": 1e-5},
        target_coarse_size=64 if int(args.dimension) == 2 else 256,
        enable_profiling=False,
        printing={"enabled": False},
        tree_infeas_use_cupy=False,
        check_type="cpu",
        tree_lp_form="primal",
    )

    method_name = str(args.method)
    tree_solver = lambda Xs, ws, Xt, wt: _solve_ot_tree(Xs, ws, Xt, wt, tree_config=tree_config)
    barycenter_fn = (
        (lambda solve_ot: _reference_one_step_barycenter(
            measures_locations,
            measures_weights,
            lam=lam,
            ref_idx=int(args.ref_idx),
            solve_ot=solve_ot,
        ))
        if method_name == "reference"
        else (lambda solve_ot: _pairwise_one_step_barycenter(
            measures_locations,
            measures_weights,
            lam=lam,
            solve_ot=solve_ot,
        ))
    )

    t0 = time.perf_counter()
    X_pair_tree, b_pair_tree = barycenter_fn(tree_solver)
    elapsed_tree = time.perf_counter() - t0

    objective_tree: float | None
    objective_emd: float | None
    elapsed_emd: float | None
    X_pair_emd: np.ndarray | None
    b_pair_emd: np.ndarray | None

    if int(args.dimension) == 2:
        objective_tree = _exact_barycenter_objective(
            X_pair_tree,
            b_pair_tree,
            measures_locations,
            measures_weights,
            lam,
        )
        t0 = time.perf_counter()
        X_pair_emd, b_pair_emd = barycenter_fn(_solve_ot_emd)
        elapsed_emd = time.perf_counter() - t0
        objective_emd = _exact_barycenter_objective(
            X_pair_emd,
            b_pair_emd,
            measures_locations,
            measures_weights,
            lam,
        )
    else:
        objective_tree = None
        objective_emd = None
        elapsed_emd = None
        X_pair_emd = None
        b_pair_emd = None

    output_dir = OUTPUT_DIR_2D if int(args.dimension) == 2 else OUTPUT_DIR_3D
    output_dir.mkdir(parents=True, exist_ok=True)
    distributions_path = output_dir / "distributions.png"
    tree_barycenter_path = output_dir / f"tree_{method_name}_barycenter.png"
    metadata_path = output_dir / "metadata.json"

    outputs = {
        "distributions": str(distributions_path),
        "tree_barycenter": str(tree_barycenter_path),
    }

    if int(args.dimension) == 2:
        emd_barycenter_path = output_dir / f"emd_{method_name}_barycenter.png"
        outputs["emd_barycenter"] = str(emd_barycenter_path)
        _plot_distributions_2d(measures_locations, labels, distributions_path)
        _plot_barycenter_2d(
            measures_locations,
            labels,
            X_pair_tree,
            b_pair_tree,
            tree_barycenter_path,
            title=f"Tree-mode {method_name} one-step barycenter",
            label=f"tree {method_name} barycenter",
            objective_value=objective_tree,
        )
        _plot_barycenter_2d(
            measures_locations,
            labels,
            X_pair_emd,
            b_pair_emd,
            emd_barycenter_path,
            title=f"EMD {method_name} one-step barycenter",
            label=f"emd {method_name} barycenter",
            objective_value=objective_emd,
        )
    else:
        _plot_distributions_3d(measures_locations, labels, distributions_path)
        _plot_barycenter_3d(
            measures_locations,
            labels,
            X_pair_tree,
            tree_barycenter_path,
            title=f"Tree-mode 3D {method_name} barycenter ({int(args.sample_points_3d)} points / shape)",
        )

    metadata = {
        "solver_engine": str(args.solver_engine),
        "dimension": int(args.dimension),
        "method": method_name,
        "ref_idx": int(args.ref_idx) if method_name == "reference" else None,
        "tree_mode": True,
        "support_size": int(X_pair_tree.shape[0]),
        "sample_points_3d": int(args.sample_points_3d) if int(args.dimension) == 3 else None,
        "tree": {
            "objective": None if objective_tree is None else float(objective_tree),
            "time_sec": float(elapsed_tree),
        },
        "emd_baseline": (
            None
            if objective_emd is None or elapsed_emd is None
            else {
                "objective": float(objective_emd),
                "time_sec": float(elapsed_emd),
            }
        ),
        "outputs": outputs,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("wasserstein barycenter example")
    print(f"dimension={int(args.dimension)}")
    print(f"method={method_name}")
    print(f"support_size={int(X_pair_tree.shape[0])}")
    if int(args.dimension) == 3:
        print(f"sample_points_3d={int(args.sample_points_3d)}")
    if objective_tree is not None:
        print(f"tree_objective={float(objective_tree):.9f}")
    print(f"tree_time_sec={float(elapsed_tree):.4f}")
    if objective_emd is not None and elapsed_emd is not None:
        print(f"emd_objective={float(objective_emd):.9f}")
        print(f"emd_time_sec={float(elapsed_emd):.4f}")
    else:
        print("emd_baseline=skipped")
    for path_text in outputs.values():
        print(f"saved to {path_text}")
    print(f"saved to {metadata_path}")


if __name__ == "__main__":
    main()
