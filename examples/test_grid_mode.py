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
OUTPUT_DIR = EXAMPLE_ROOT / "outputs" / "test_grid_mode"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hierarchical_ot import GridConfig, emd2_grid


def _make_case(resolution: int) -> tuple[np.ndarray, np.ndarray]:
    res = int(resolution)
    if res < 8 or res % 2 != 0:
        raise ValueError("resolution must be an even integer >= 8")

    source = np.zeros((res, res), dtype=np.float32)
    target = np.zeros((res, res), dtype=np.float32)

    offset = max(1, res // 8)
    source[offset, offset] = 0.35
    source[res // 2, res // 2] = 0.30
    source[res - offset - 1, res - offset - 1] = 0.35

    target[offset, res - offset - 1] = 0.30
    target[res // 2, res // 2] = 0.20
    target[res - offset - 1, offset] = 0.50

    source /= max(float(source.sum()), 1e-12)
    target /= max(float(target.sum()), 1e-12)
    return source, target


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


def _print_report(*, resolution: int, num_scales: int | None, result: dict[str, Any], elapsed: float) -> None:
    print("grid mode example")
    print(f"resolution={resolution}, num_scales={num_scales}")
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
    parser = argparse.ArgumentParser(description="Run a lightweight public grid-mode example.")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--num-scales", type=int, default=1)
    parser.add_argument("--solver-engine", choices=("cupdlpx", "scipy"), default="scipy")
    args = parser.parse_args()

    source, target = _make_case(args.resolution)
    config = GridConfig(
        solver_engine=str(args.solver_engine),
        num_scales=int(args.num_scales),
        max_inner_iter=3,
        enable_profiling=False,
        printing={"enabled": False},
        check_type="cpu",
        if_check=False,
    )

    t0 = time.perf_counter()
    result = emd2_grid(source, target, config=config, log=True)
    elapsed = time.perf_counter() - t0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "resolution": int(args.resolution),
        "num_scales": int(args.num_scales),
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
        resolution=int(args.resolution),
        num_scales=int(args.num_scales),
        result=result,
        elapsed=elapsed,
    )
    print()
    print(f"saved={OUTPUT_DIR / 'report.json'}")


if __name__ == "__main__":
    main()
