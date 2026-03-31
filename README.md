# HALO_public

`HALO_public` is a minimal public package extracted from `hierarchical_ot` with two public algorithm entry points:

- `MGPD`: grid-based OT, exposed as `halo_public.mgpd.solve`
- `HALO`: point-cloud OT, exposed as `halo_public.halo.solve`

The repository ships with the `scipy` LP backend by default to keep the package runnable out of the box. GPU-related dependencies are provided separately through optional extras.

## Install

```bash
pip install -e .
```



## Usage

Below are typical usage examples for the two public algorithm entry points built on top of the unified framework.

MGPD:

```python
import numpy as np
from halo_public import MGPDConfig, solve_mgpd

source = np.ones((8, 8), dtype=np.float32)
target = np.eye(8, dtype=np.float32)
out = solve_mgpd(source, target, MGPDConfig(num_scales=2, max_inner_iter=2))
print(out["distance"])
```

HALO:

```python
import numpy as np
from halo_public import HALOConfig, solve_halo

rng = np.random.default_rng(0)
xs = rng.normal(size=(32, 2)).astype(np.float32)
xt = rng.normal(size=(32, 2)).astype(np.float32)
out = solve_halo(xs, xt, config=HALOConfig(max_inner_iter=2, cost_type="l2^2"))
print(out["distance"])
```

## Public API

- `halo_public.MGPDConfig`
- `halo_public.solve_mgpd`
- `halo_public.HALOConfig`
- `halo_public.solve_halo`

## Tests

```bash
python -m pytest tests/test_public_smoke.py
```
