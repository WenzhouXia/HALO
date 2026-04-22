# HALO_public

[English](README.md) | [简体中文](README.zh.md)

`HALO_public` is the public tree/grid subset rebuilt from `hierarchical_ot`.

It exposes the installable runtime for hierarchical optimal transport together with a small set of runnable examples.

## Core Solver Flow

The main multilevel algorithm is implemented in `src/hierarchical_ot/core/multilevel_flow.py`.

At a high level, the solver runs the following loop:

1. Validate the problem definition and initialize run-level state.
2. Build a coarse-to-fine list of hierarchy levels.
3. For each level, initialize mode-specific state and active support.
4. Repeatedly prepare the current subproblem, solve one LP step, finalize the iteration, and check stopping criteria.
5. Record the current level result and propagate warm-start information to the next finer level.
6. After the finest level finishes, package the final transport result and profiling output.

This is the core idea behind the library: solve a sequence of progressively finer OT problems, using the coarse solution to warm-start the fine one.

## Modes

`grid` mode is for 2D Cartesian grids.
Typical use case: image-like histograms on regular 2D lattices.

`tree` mode is for low-dimensional point clouds, especially 2D and 3D point clouds.
It is also the mode used by the provided free-support Wasserstein barycenter examples.

## Examples

Run:

```bash
bash examples/run_examples.sh
```

This script runs the public examples directly:

- `examples/test_grid_mode.py`: a basic 2D grid-mode example
- `examples/test_tree_mode.py`: a basic tree-mode example
- `examples/show_tree_pairwise_barycenter.py --dimension 2`: 2D free-support Wasserstein barycenter
- `examples/show_tree_pairwise_barycenter.py --dimension 3`: 3D free-support Wasserstein barycenter

These barycenter scripts are intended as downstream applications built on top of the exported OT solver.

## Export Policy

- Keep the installable tree/grid runtime and APIs
- Exclude cluster mode, dual-assignment, gromov, and low-rank research paths
- Overwrite shared entrypoints with public-only overlays so the package imports cleanly
- Export runnable public examples under `examples/`

The export script also writes:

- `EXPORT_MANIFEST.json`: copied files, overlays, and exclusion policy
- `EXPORT_AUDIT.md`: forbidden-path / forbidden-text audit results

## Example-Only Dependencies

- `matplotlib` and `POT` are needed for the barycenter visualization example
- The 3D barycenter example uses pre-sampled ModelNet10 chair/toilet assets bundled under `examples/assets/`
