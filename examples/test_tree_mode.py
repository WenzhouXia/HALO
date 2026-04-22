#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

EXAMPLE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
OUTPUT_DIR = EXAMPLE_ROOT / "outputs" / "test_tree_mode"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hierarchical_ot import TreeConfig, emd2


def _make_case(n: int, d: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    source = rng.normal(loc=-0.5, scale=0.8, size=(int(n), int(d))).astype(np.float32)
    target = rng.normal(loc=0.5, scale=0.8, size=(int(n), int(d))).astype(np.float32)
    source_mass = np.full(int(n), 1.0 / float(n), dtype=np.float32)
    target_mass = np.full(int(n), 1.0 / float(n), dtype=np.float32)
    return source, target, source_mass, target_mass


def _level_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in result.get("level_summaries", []):
        rows.append(
            {
                "level": int(row["level"]),
                "iters": int(row["iters"]),
                "support_pre_lp_final": int(row.get("support_pre_lp_final", 0) or 0),
                "support_final": int(row.get("support_final", 0) or 0),
                "stop_reason": row.get("stop_reason"),
            }
        )
    return rows


def _print_report(*, n: int, d: int, seed: int, result: dict[str, Any], elapsed: float) -> None:
    print("tree mode example")
    print(f"n={n}, d={d}, seed={seed}")
    print(f"distance={float(result['distance']):.9f}")
    print(f"time_sec={float(elapsed):.4f}")
    print()
    print(f"{'level':>6} | {'iters':>6} | {'pre_lp':>10} | {'final':>10} | {'stop_reason':>18}")
    print("-" * 62)
    for row in _level_rows(result):
        print(
            f"{int(row['level']):>6} | "
            f"{int(row['iters']):>6} | "
            f"{int(row['support_pre_lp_final']):>10} | "
            f"{int(row['support_final']):>10} | "
            f"{str(row.get('stop_reason')):>18}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight public tree-mode example.")
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--solver-engine", choices=("cupdlpx", "scipy"), default="scipy")
    args = parser.parse_args()

    source, target, source_mass, target_mass = _make_case(args.n, args.d, args.seed)
    config = TreeConfig(
        cost_type="l2^2",
        solver_engine=str(args.solver_engine),
        max_inner_iter=4,
        tolerance={"objective": 1e-5, "primal": 1e-5, "dual": 1e-5},
        target_coarse_size=64,
        enable_profiling=False,
        printing={"enabled": False},
        tree_infeas_use_cupy=False,
        check_type="cpu",
        tree_lp_form="primal",
    )

    t0 = time.perf_counter()
    result = emd2(
        source,
        target,
        source_mass=source_mass,
        target_mass=target_mass,
        log=True,
        config=config,
    )
    elapsed = time.perf_counter() - t0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "n": int(args.n),
        "d": int(args.d),
        "seed": int(args.seed),
        "solver_engine": str(args.solver_engine),
        "distance": float(result["distance"]),
        "time_sec": float(elapsed),
        "level_summaries": _level_rows(result),
    }
    (OUTPUT_DIR / "report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    _print_report(
        n=int(args.n),
        d=int(args.d),
        seed=int(args.seed),
        result=result,
        elapsed=elapsed,
    )
    print()
    print(f"saved={OUTPUT_DIR / 'report.json'}")


if __name__ == "__main__":
    main()
