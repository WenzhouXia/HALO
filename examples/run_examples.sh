#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[examples] running grid mode example"
"$PYTHON_BIN" "$ROOT_DIR/examples/test_grid_mode.py"

echo "[examples] running tree mode example"
"$PYTHON_BIN" "$ROOT_DIR/examples/test_tree_mode.py"

echo "[examples] running 2D tree reference barycenter example"
"$PYTHON_BIN" "$ROOT_DIR/examples/show_tree_pairwise_barycenter.py" --dimension 2

echo "[examples] running 3D tree reference barycenter example"
"$PYTHON_BIN" "$ROOT_DIR/examples/show_tree_pairwise_barycenter.py" --dimension 3

echo "[examples] completed"
