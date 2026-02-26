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

grid = llnltofi.Grid()

# Get grid coordinates as (lon, gc_lat, depth_km)
coords = grid.coordinates_in_lonlatdepth  # shape (1_003_608, 3)

# Or as Cartesian (x, y, z) in metres
xyz = grid.coordinates_in_xyz  # shape (1_003_608, 3)

# Evaluate your model at these points and compute slowness perturbation
# du = 1/v_3D - 1/v_1D at each grid point
du = ...  # shape (1_003_608,)

# Apply the resolution matrix
du_filtered = grid.filter_slowness_perturbation(du)  # shape (1_003_608,)
```

An interpolation helper is also provided for projecting an arbitrary point cloud onto the LLNL grid:

```python
from llnltofi.interpolation import project_onto_grid

du = project_onto_grid(source_points, source_values, grid)  # shape (1_003_608,)
```

## Setting up the resolution matrix

The resolution matrix is distributed by LLNL as 44 text files in COO format. These can be obtained from the [LLNL server](https://gs.llnl.gov/nuclear-threat-reduction/nuclear-explosion-monitoring/global-3d-seismic-tomography) or by email request to simmons27@llnl.gov. Once you have the text files, convert them to a bundled npz that the package will pick up automatically:

```python
from llnltofi import convert_text_to_npz

convert_text_to_npz("./path/to/text/files/")
```

This writes `R.npz` into the package's `data/` directory. After that, `load_resolution_matrix()` works without any path argument.

## Data

The package bundles the LLNL-G3D-JPS tessellated grid coordinates, layer depth information, and (after conversion) the resolution matrix as a sparse npz file.

## References

Simmons, N. A., Myers, S. C., Johannesson, G., Matzel, E., & Grand, S. P. (2015). Evidence for long-lived subduction of an ancient tectonic plate beneath the southern Indian Ocean. *Geophysical Research Letters, 42*(21), 9270-9278. https://doi.org/10.1002/2015GL066237

Simmons, N. A., Schuberth, B. S. A., Myers, S. C., & Knapp, D. R. (2019). Resolution and covariance of the LLNL-G3D-JPS global seismic tomography model: applications to travel time uncertainty and tomographic filtering of geodynamic models. *Geophysical Journal International, 217*(3), 1543-1557. https://doi.org/10.1093/gji/ggz102

## License

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](LICENSE)
