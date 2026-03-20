"""Project an arbitrary point cloud onto / from the LLNL-G3D-JPS grid."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.spatial import cKDTree

from ._constants import (
    R_EARTH_KM,
    N_LAYERS,
    N_LAYERS_UM_TZ,
    N_POINTS_UM_TZ,
    N_POINTS_LM,
    N_MODEL,
    GRID_SPACING_UM_TZ_KM,
    GRID_SPACING_LM_KM,
)
from ._grid import ResolutionModel
from ._spherical import geo2sph, sph2cart


def project_onto_grid(
    source_points: np.ndarray,
    source_values: np.ndarray,
    model: ResolutionModel,
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
    model : ResolutionModel
        An ``llnltofi.ResolutionModel`` instance.
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

    for layer in range(model.n_layers):
        n = N_POINTS_UM_TZ if layer < N_LAYERS_UM_TZ else N_POINTS_LM
        off = model._layer_offset(layer)
        radii = model.layer_radius(layer)
        depth_avg = model._depth_avg[layer]
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

        gc_lat = model._geocentric_latitude[:n]
        lon = model._longitude[:n]

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
            dists = np.maximum(dists, 1e-10)
            weights = 1.0 / dists**2
            values = np.sum(weights * subset_vals[inds], axis=1) / np.sum(
                weights, axis=1
            )
        else:
            masked_vals = np.where(within, subset_vals[inds], 0.0)
            counts = np.sum(within, axis=1).clip(min=1)
            values = np.sum(masked_vals, axis=1) / counts

        result[off : off + n] = values

    return result


def _build_unit_tree(model: ResolutionModel, n_points: int) -> cKDTree:
    """Build a KD-tree of unit-sphere Cartesian vectors for an angular grid."""
    geo = np.column_stack((
        np.ones(n_points),
        model._longitude[:n_points],
        model._geocentric_latitude[:n_points],
    ))
    return cKDTree(sph2cart(geo2sph(geo)))


def _lateral_idw(tree: cKDTree, query_unit: np.ndarray, k: int):
    """Query a unit-sphere tree and return IDW weights and indices.

    Returns
    -------
    weights : ndarray, shape (N, k)
        Normalised 1/d² weights.
    idx : ndarray, shape (N, k)
        Neighbour indices into the angular grid.
    """
    dists, idx = tree.query(query_unit, k=k, workers=-1)
    if dists.ndim == 1:
        dists = dists[:, np.newaxis]
        idx = idx[:, np.newaxis]
    dists = np.maximum(dists, 1e-10)
    weights = 1.0 / dists**2
    weights /= weights.sum(axis=1, keepdims=True)
    return weights, idx


def _vectorised_offsets(layer_indices: np.ndarray) -> np.ndarray:
    """Compute layer offsets for an array of layer indices without a Python loop."""
    offsets = np.empty_like(layer_indices, dtype=np.int64)
    is_umtz = layer_indices < N_LAYERS_UM_TZ
    offsets[is_umtz] = layer_indices[is_umtz] * N_POINTS_UM_TZ
    offsets[~is_umtz] = (
        N_LAYERS_UM_TZ * N_POINTS_UM_TZ
        + (layer_indices[~is_umtz] - N_LAYERS_UM_TZ) * N_POINTS_LM
    )
    return offsets


def _interpolate_batch(
    grid_values: np.ndarray,
    query_idxs: np.ndarray,
    weights: np.ndarray,
    lateral_idx: np.ndarray,
    layer_below: np.ndarray,
    layer_above: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Lateral IDW + radial linear interpolation for a batch of query points."""
    off_below = _vectorised_offsets(layer_below[query_idxs])
    off_above = _vectorised_offsets(layer_above[query_idxs])
    vals_below = grid_values[(off_below[:, None] + lateral_idx)]
    vals_above = grid_values[(off_above[:, None] + lateral_idx)]
    interp_below = np.einsum("ij,ij->i", weights, vals_below)
    interp_above = np.einsum("ij,ij->i", weights, vals_above)
    t_batch = t[query_idxs]
    return (1.0 - t_batch) * interp_below + t_batch * interp_above


def project_from_grid(
    grid_values: np.ndarray,
    query_points: np.ndarray,
    model: ResolutionModel,
    *,
    k: int = 6,
) -> np.ndarray:
    """Interpolate from the LLNL grid to arbitrary query locations.

    Uses layered interpolation: lateral IDW (1/d²) on the unit sphere within
    each bracketing layer, then linear radial interpolation between layers.

    Query points outside the radial extent of the model (above the shallowest
    layer or below the deepest layer) are clamped to the nearest layer and a
    warning is issued.

    Parameters
    ----------
    grid_values : ndarray, shape (n_model,)
        Field values on the LLNL grid (e.g. from ``project_onto_grid``).
    query_points : ndarray, shape (N, 3)
        Query locations as ``(gc_lat_deg, lon_deg, radius_km)``.
        Geocentric latitude in [-90, 90], longitude in [-180, 180].
    model : ResolutionModel
        An ``llnltofi.ResolutionModel`` instance.
    k : int
        Number of lateral nearest neighbours for IDW (default 6).

    Returns
    -------
    ndarray, shape (N,)
        Interpolated values at the query locations.
    """
    if grid_values.ndim != 1 or grid_values.shape[0] != N_MODEL:
        raise ValueError(f"grid_values must have shape ({N_MODEL},)")
    if query_points.ndim != 2 or query_points.shape[1] != 3:
        raise ValueError("query_points must have shape (N, 3)")

    n_nan = np.count_nonzero(np.isnan(grid_values))
    if n_nan > 0:
        warnings.warn(
            f"grid_values contains {n_nan} NaN(s); these will propagate "
            f"into the interpolated result.",
            stacklevel=2,
        )

    n_query = query_points.shape[0]
    query_r = query_points[:, 2]

    # Build unit vectors for query points (radius=1, same geo2sph convention)
    query_geo = np.column_stack((
        np.ones(n_query),
        query_points[:, 1],
        query_points[:, 0],
    ))
    query_unit = sph2cart(geo2sph(query_geo))

    # Layer radii sorted ascending (CMB first, surface last)
    layer_radii = np.array([
        R_EARTH_KM - model._depth_avg[l] for l in range(N_LAYERS)
    ])
    sort_idx = np.argsort(layer_radii)
    radii_sorted = layer_radii[sort_idx]

    # Find bracketing layers via searchsorted
    i_above = np.searchsorted(radii_sorted, query_r, side="right")
    i_below = i_above - 1

    # Warn and clamp points outside the model's radial extent
    n_outside = int(np.count_nonzero(i_below < 0) + np.count_nonzero(i_above >= N_LAYERS))
    if n_outside > 0:
        warnings.warn(
            f"{n_outside} query point(s) lie outside the model's radial "
            f"extent ({radii_sorted[0]:.1f}–{radii_sorted[-1]:.1f} km) "
            f"and will be clamped to the nearest layer.",
            stacklevel=2,
        )
    i_below = np.clip(i_below, 0, N_LAYERS - 1)
    i_above = np.clip(i_above, 0, N_LAYERS - 1)

    # Map sorted indices back to actual layer numbers
    layer_below = sort_idx[i_below]
    layer_above = sort_idx[i_above]

    # Radial interpolation parameter t
    r_below = layer_radii[layer_below]
    r_above = layer_radii[layer_above]
    dr = r_above - r_below
    # When both layers are the same (clamped boundary), dr=0, t doesn't matter
    safe_dr = np.where(dr == 0.0, 1.0, dr)
    t = np.clip((query_r - r_below) / safe_dr, 0.0, 1.0)

    # Classify each query point by which angular grids its bracketing layers use
    below_is_umtz = layer_below < N_LAYERS_UM_TZ
    above_is_umtz = layer_above < N_LAYERS_UM_TZ

    mask_both_umtz = below_is_umtz & above_is_umtz
    mask_both_lm = ~below_is_umtz & ~above_is_umtz
    mask_mixed = ~mask_both_umtz & ~mask_both_lm

    # Only build trees that are actually needed
    need_umtz = mask_both_umtz.any() or mask_mixed.any()
    need_lm = mask_both_lm.any() or mask_mixed.any()
    tree_umtz = _build_unit_tree(model, N_POINTS_UM_TZ) if need_umtz else None
    tree_lm = _build_unit_tree(model, N_POINTS_LM) if need_lm else None

    result = np.empty(n_query, dtype="float64")

    # --- Batch A: both layers UM/TZ ---
    idxs_a = np.where(mask_both_umtz)[0]
    if len(idxs_a) > 0:
        w, li = _lateral_idw(tree_umtz, query_unit[idxs_a], k)
        result[idxs_a] = _interpolate_batch(
            grid_values, idxs_a, w, li, layer_below, layer_above, t,
        )

    # --- Batch B: both layers LM ---
    idxs_b = np.where(mask_both_lm)[0]
    if len(idxs_b) > 0:
        w, li = _lateral_idw(tree_lm, query_unit[idxs_b], k)
        result[idxs_b] = _interpolate_batch(
            grid_values, idxs_b, w, li, layer_below, layer_above, t,
        )

    # --- Batch C: mixed (one UM/TZ, one LM) ---
    idxs_c = np.where(mask_mixed)[0]
    if len(idxs_c) > 0:
        below_umtz_c = below_is_umtz[idxs_c]

        w_umtz, li_umtz = _lateral_idw(tree_umtz, query_unit[idxs_c], k)
        w_lm, li_lm = _lateral_idw(tree_lm, query_unit[idxs_c], k)

        off_below = _vectorised_offsets(layer_below[idxs_c])
        off_above = _vectorised_offsets(layer_above[idxs_c])

        w_below = np.where(below_umtz_c[:, None], w_umtz, w_lm)
        li_below = np.where(below_umtz_c[:, None], li_umtz, li_lm)
        w_above = np.where(below_umtz_c[:, None], w_lm, w_umtz)
        li_above = np.where(below_umtz_c[:, None], li_lm, li_umtz)

        vals_below = grid_values[(off_below[:, None] + li_below)]
        vals_above = grid_values[(off_above[:, None] + li_above)]
        interp_below = np.einsum("ij,ij->i", w_below, vals_below)
        interp_above = np.einsum("ij,ij->i", w_above, vals_above)
        t_c = t[idxs_c]
        result[idxs_c] = (1.0 - t_c) * interp_below + t_c * interp_above

    return result
