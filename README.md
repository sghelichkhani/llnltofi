# llnltofi

A Python package for tomographic filtering of seismic mantle structure ($v_S$ or $v_P$) using the resolution matrix $R$ of the LLNL-G3D-JPS model by Simmons et al. (2015). Given a velocity model on the LLNL grid, `llnltofi` applies the resolution operator via a single sparse matrix-vector multiply $Rm = m'$ and returns the filtered result.

Author: Sia Ghelichkhan (sia.ghelichkhani@anu.edu.au)

## Installation

```bash
pip install -e .
```

Dependencies are just `numpy` and `scipy`. For running the example scripts you will also need `matplotlib`, `seaborn`, `pandas`, `xarray`, and `netCDF4`, which can be installed with:

```bash
pip install -e ".[examples]"
```

## Quick start

```python
import llnltofi

model = llnltofi.ResolutionModel()

# Get grid coordinates as (lon, gc_lat, depth_km)
coords = model.coordinates_in_lonlatdepth  # shape (1_003_608, 3)

# Or as Cartesian (x, y, z) in metres
xyz = model.coordinates_in_xyz  # shape (1_003_608, 3)

# Evaluate your model at these points and compute slowness perturbation
# du = 1/v_3D - 1/v_1D at each grid point
model.values = ...  # shape (1_003_608,)

# Apply the resolution matrix
du_filtered = model.apply()  # shape (1_003_608,)
```

An interpolation helper is also provided for projecting an arbitrary point cloud onto the LLNL grid:

```python
from llnltofi.interpolation import project_onto_grid

du = project_onto_grid(source_points, source_values, model)  # shape (1_003_608,)
```

## Correctness note

This package is a modern reimplementation of [LLNL_ToFi](https://gitlab.lrz.de/bschuberth/LLNL_ToFi) by Bernhard Schuberth (LMU Munich). The original code performed the filtering by looping over the sparse matrix entries element by element, converting each 1-based global column index back to a (layer, coordinate) pair using integer division and modulo arithmetic:

```python
c_index = np.mod(cj, np_UM_TZ) - 1
l_index = cj // np_UM_TZ
```

This mapping has a boundary bug. Whenever `cj` is an exact multiple of the layer size (40,962 for upper-mantle/transition-zone layers, 10,242 for lower-mantle layers), the modulo returns zero, `c_index` becomes $-1$, and `l_index` is off by one. Python silently wraps the $-1$ to the last array element, so the code reads a value from the wrong depth layer without raising any error. Cross-validation against the original text files shows that 2,357 out of 75,508,775 R-matrix entries fall on these boundary columns, producing errors in 1,107 output grid points (0.11 % of the model vector) with absolute differences up to $\approx 0.15$.

`llnltofi` avoids the problem entirely by pre-converting the 44 layer text files into a single `scipy.sparse` CSR matrix with straightforward 0-based indexing (`col = cj - 1`). The filtering step becomes a single call to `scipy.sparse.csr_matrix @ numpy.ndarray`, delegating all index arithmetic to well-tested library code. A cross-validation test suite (`tests/test_equivalence.py`) confirms that the bundled `R.npz` is bit-for-bit identical to one freshly assembled from the original text files and that `model.apply()` reproduces the correct matrix-vector product.

## Data

All required data files, including the grid coordinates and the resolution matrix, are hosted on S3 and downloaded automatically on first use. There is nothing to configure — just `pip install` the package and everything is fetched lazily when you first access `model.R` or the coordinate properties. Downloaded files are cached locally so subsequent runs are instant.

## References

Simmons, N. A., Myers, S. C., Johannesson, G., Matzel, E., & Grand, S. P. (2015). Evidence for long-lived subduction of an ancient tectonic plate beneath the southern Indian Ocean. *Geophysical Research Letters, 42*(21), 9270-9278. https://doi.org/10.1002/2015GL066237

Simmons, N. A., Schuberth, B. S. A., Myers, S. C., & Knapp, D. R. (2019). Resolution and covariance of the LLNL-G3D-JPS global seismic tomography model: applications to travel time uncertainty and tomographic filtering of geodynamic models. *Geophysical Journal International, 217*(3), 1543-1557. https://doi.org/10.1093/gji/ggz102

## License

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](LICENSE)
