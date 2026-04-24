# HALO_public

[English](README.md) | [简体中文](README.zh.md)

`HALO_public` is the official implementation of the ICLR 2026 paper **A Memory-Efficient Hierarchical Algorithm for Large-scale Optimal Transport Problems**. 
It provides an O(n)-memory, GPU-friendly solver for large-scale OT problems, especially suitable for low-dimensional settings such as 2D grids and 3D point clouds.

## Core Idea

The solver uses coarse levels in the hierarchy to provide a warm-start for the original OT problem, then iteratively refines the solution layer by layer to efficiently obtain the final transport map.

## Modes

- `grid` mode: suitable for 2D Cartesian grids, typical use case is image-like histograms on regular 2D lattices.  
- `tree` mode: suitable for low-dimensional point clouds, especially 2D and 3D. Free-support Wasserstein barycenter examples also use this mode.  
- The code in this repository is designed for squared Euclidean distance OT problems.

## Examples

Run:

```bash
bash examples/run_examples.sh
```

This automatically runs four basic examples, including 2D/3D grid and tree-mode barycenter experiments.
