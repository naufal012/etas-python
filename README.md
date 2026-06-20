# ETAS Python — Truncated & Renormalized 3D ETAS

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-GPL--3.0-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-23%2F23%20passed-brightgreen)](tests/)

A pure-Python / NumPy implementation of the **Epidemic-Type Aftershock Sequence
(ETAS)** model for seismicity, with a **truncated & renormalized** extension
that makes it scalable to large earthquake catalogues ($N > 10^5$ events).

Based on the original [ETAS R package](https://CRAN.R-project.org/package=etas)
by A. Jalilian, extended with:

- **Analytic renormalization** of the truncated Omori-Utsu (temporal) and
  power-law (spatial) kernels — the pruned kernels remain valid probability
  densities.
- **KDTree-based O(N log N) neighbour pruning** replacing the O(N²)
  pairwise-mask bottleneck.
- **3D hypocentral depth kernel** (`h(u;v)` Beta-density form) for
  depth-dependent triggering.

## Quick Start

```bash
pip install -r requirements.txt
```

```python
import numpy as np
from etas import catalog, etas, rates, search_isc

# Fetch data from ISC
df = search_isc(start_year=2020, start_month=1, start_day=1,
                end_year=2020, end_month=12, end_day=31,
                searchshape="RECT",
                lat_bot=30, lat_top=45,
                long_left=130, long_right=145,
                mag_min=4.5)

# Build catalog
cat = catalog(data=df, time_begin="2020-01-01",
              study_start="2020-03-01", study_length=250,
              mag_threshold=4.5, dist_unit="degree")

# Fit with truncation + renormalization (fast path)
fit = etas(cat, no_itr=10, mver=1, eps_t=1e-4, eps_s=1e-4, verbose=True)

# Or fit the full (un-pruned) model
fit_baseline = etas(cat, no_itr=10, mver=1, verbose=True)

# Background probability map
bkgd_probs = rates(fit, cat, bkgd_only=True)

# 3D fit (with depth kernel)
cat_3d = catalog(data=df, ..., depth_col="depth")
fit_3d = etas(cat_3d, is_3d=True, Z_max=30.0, no_itr=10, mver=1)
```

## Performance

On a synthetic catalog of 288 events (3 EM iterations):

| Model | Time | Log-lik | \|ΔLL\| |
|---|---|---|---|
| Baseline (no pruning) | 236 s | −237.58 | — |
| Renormalized, ε=10⁻⁵ | 179 s | −237.71 | 0.13 |
| Renormalized, ε=10⁻³ | 180 s | −238.69 | 1.10 |

Speedups reach **10–50×** for catalogues above 10,000 events.
See [MODEL.md](MODEL.md) §10 for full details.

## Features

| Feature | Status |
|---|---|
| Standard ETAS (mver=1,2) | ✅ |
| Truncated + Renormalized kernels | ✅ |
| KDTree neighbour pruning | ✅ |
| 3D depth kernel (Beta density) | ✅ |
| Analytic gradient (DFP quasi-Newton) | ✅ |
| GPU support (CuPy backend) | ✅ |
| ISC catalog search | ✅ |
| Stochastic declustering (EM) | ✅ |
| Background rate estimation | ✅ |
| Residual analysis | ✅ |

## Mathematical Documentation

Full model description, derivations, and algorithm pseudocode are in
**[MODEL.md](MODEL.md)**.  Covers:

1. Kernel definitions (Omori-Utsu, power-law, Gaussian, Beta-depth)
2. Truncation cutoffs and closed-form renormalization constants
3. Total-derivative gradients (with the proof that partial ≠ total)
4. KDTree O(N log N) lookup design
5. EM algorithm and DFP optimisation
6. Sqrt-parameterisation chain rule

## Running Tests

```bash
pytest tests/ -v
```

23 tests covering:
- Renormalization constants vs numerical quadrature
- Gradient finite-difference verification (2D, 3D, with renorm)
- KDTree neighbour lookup vs brute-force masks
- 3D ↔ 2D consistency
- Pruning accuracy
- Full pipeline (catalog → fit → parameters)

## Requirements

- Python ≥ 3.9
- NumPy, SciPy, Pandas
- (Optional) CuPy for GPU acceleration

## References

- Ogata, Y. (1998). Space-time point-process models for earthquake occurrences.
  *Ann. Inst. Stat. Math.*, 50(2), 379–402.
- Zhuang, J., Ogata, Y., & Vere-Jones, D. (2002). Stochastic declustering.
  *JASA*, 97(458), 369–380.
- Jalilian, A. (2019). ETAS: an R package for fitting the space-time ETAS model.

## License

GPL-3.0 — see [LICENSE](LICENSE).
