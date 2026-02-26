"""Project an arbitrary point cloud onto the LLNL-G3D-JPS grid."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ._constants import (
    R_EARTH_KM,
    N_LAYERS_UM_TZ,
    N_POINTS_UM_TZ,
    N_POINTS_LM,
    N_MODEL,
    GRID_SPACING_UM_TZ_KM,
    GRID_SPACING_LM_KM,
)
from ._grid import Grid
from ._spherical import geo2sph, sph2cart


def project_onto_grid(
    source_points: np.ndarray,
    source_values: np.ndarray,
    grid: Grid,
    *,
    k: int = 1000,
    weighting: str = "inverse_distance",
) -> np.ndarray:
    """Interpolate a source field onto every layer of the LLNL grid.

    Parameters
    ----------
    source_points : ndarray, shape (N, 3)
        Source locations as ``(gc_lat_deg, lon_deg, radius_km)``.
    source_values : ndarray, shape (N,)
        Field values at the source locations.
    grid : Grid
        An ``llnltofi.Grid`` instance.
    k : int
        Number of nearest neighbours queried per target point.
    weighting : str
        ``"inverse_distance"`` (default) or ``"uniform"``.

    Returns
    -------
    ndarray, shape (n_model,)
        Flat array of interpolated values on the LLNL grid.
    """
    if source_points.ndim != 2 or source_points.shape[1] != 3:
        raise ValueError("source_points must have shape (N, 3)")
    if source_values.ndim != 1 or source_values.shape[0] != source_points.shape[0]:
        raise ValueError("source_values must have shape (N,)")

    # Pre-compute Cartesian coordinates and radii of source points (normalised)
    src_radii = source_points[:, 2] / R_EARTH_KM
    src_cart = sph2cart(
        geo2sph(
            np.column_stack(
                (
                    source_points[:, 2] / R_EARTH_KM,
                    source_points[:, 1],
                    source_points[:, 0],
                )
            )
        )
    )

    result = np.empty(N_MODEL, dtype="float64")

    for layer in range(grid.n_layers):
        n = N_POINTS_UM_TZ if layer < N_LAYERS_UM_TZ else N_POINTS_LM
        off = grid._layer_offset(layer)
        radii = grid.layer_radius(layer)
        depth_avg = grid._depth_avg[layer]
        radius_avg = R_EARTH_KM - depth_avg

        grid_spacing = (
            GRID_SPACING_UM_TZ_KM if layer < N_LAYERS_UM_TZ else GRID_SPACING_LM_KM
        )

        radius_avg_norm = radius_avg / R_EARTH_KM
        r_min = radii["min"] / R_EARTH_KM
        r_max = radii["max"] / R_EARTH_KM

        # Ensure non-zero radial thickness
        if r_min == radius_avg_norm:
            r_min = radius_avg_norm - 10.0 / R_EARTH_KM
        if r_max == radius_avg_norm:
            r_max = radius_avg_norm + 10.0 / R_EARTH_KM

        # Select source points within the radial shell
        mask = (src_radii >= r_min) & (src_radii <= r_max)

        # Broaden search radius if no source points fall within the shell
        thickness = r_max - r_min
        while np.count_nonzero(mask) == 0:
            r_min -= thickness / 4
            r_max += thickness / 4
            mask = (src_radii >= r_min) & (src_radii <= r_max)

        subset_cart = src_cart[mask]
        subset_vals = source_values[mask]

        gc_lat = grid._geocentric_latitude[:n]
        lon = grid._longitude[:n]

        # Target Cartesian coordinates (normalised by R_EARTH_KM)
        target_cart = sph2cart(
            geo2sph(np.column_stack((np.full(n, radius_avg_norm), lon, gc_lat)))
        )

        tree = cKDTree(subset_cart)
        dists, inds = tree.query(target_cart, k=min(k, len(subset_cart)))

        # If k=1, query returns 1-D arrays — make them 2-D for uniform code
        if dists.ndim == 1:
            dists = dists[:, np.newaxis]
            inds = inds[:, np.newaxis]

        # Mask out neighbours beyond the grid spacing
        within = dists < (grid_spacing / R_EARTH_KM)
        dists[~within] = 1e10

        if weighting == "inverse_distance":
            weights = 1.0 / dists
            values = np.sum(weights * subset_vals[inds], axis=1) / np.sum(
                weights, axis=1
            )
        else:
            masked_vals = np.where(within, subset_vals[inds], 0.0)
            counts = np.sum(within, axis=1).clip(min=1)
            values = np.sum(masked_vals, axis=1) / counts

        result[off : off + n] = values

    return result
