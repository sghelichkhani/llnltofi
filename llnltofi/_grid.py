from __future__ import annotations

import numpy as np

from ._constants import (
    R_EARTH_KM,
    N_LAYERS,
    N_LAYERS_UM_TZ,
    N_POINTS_UM_TZ,
    N_POINTS_LM,
    N_MODEL,
)
from ._download import ensure_data
from ._spherical import geo2sph, sph2cart


class Grid:
    """LLNL-G3D-JPS tessellated grid.

    Loads the bundled coordinate and depth files on construction and provides
    flat coordinate arrays and a resolution-matrix filter for the full model
    vector.

    Layers are 0-based: 0 = crust, 43 = D''/CMB.
    """

    def __init__(self) -> None:
        data = np.load(ensure_data("grid_data.npz"))
        coords = data["coordinates"]
        depths = data["layer_depths"]

        self._geocentric_latitude = coords[:, 2]
        self._longitude = coords[:, 1]

        self._depth_min = depths[:, 0]
        self._depth_avg = depths[:, 1]
        self._depth_max = depths[:, 2]

        self._coordinates_in_lonlatdepth = None
        self._coordinates_in_xyz = None
        self._R = None

    # -- scalar properties ---------------------------------------------------

    @property
    def n_layers(self) -> int:
        return N_LAYERS

    @property
    def n_model(self) -> int:
        return N_MODEL

    # -- per-layer queries ---------------------------------------------------

    def n_points(self, layer: int) -> int:
        self._check_layer(layer)
        return N_POINTS_UM_TZ if layer < N_LAYERS_UM_TZ else N_POINTS_LM

    def layer_depth(self, layer: int) -> dict[str, float]:
        self._check_layer(layer)
        return {
            "min": float(self._depth_min[layer]),
            "avg": float(self._depth_avg[layer]),
            "max": float(self._depth_max[layer]),
        }

    def layer_radius(self, layer: int) -> dict[str, float]:
        self._check_layer(layer)
        return {
            "min": R_EARTH_KM - float(self._depth_max[layer]),
            "avg": R_EARTH_KM - float(self._depth_avg[layer]),
            "max": R_EARTH_KM - float(self._depth_min[layer]),
        }

    # -- flat coordinate arrays ----------------------------------------------

    @property
    def coordinates_in_lonlatdepth(self) -> np.ndarray:
        """Flat ``(n_model, 3)`` array of ``(lon_deg, gc_lat_deg, depth_km)``."""
        if self._coordinates_in_lonlatdepth is None:
            out = np.empty((N_MODEL, 3), dtype="float64")
            for layer in range(N_LAYERS):
                n = N_POINTS_UM_TZ if layer < N_LAYERS_UM_TZ else N_POINTS_LM
                off = self._layer_offset(layer)
                out[off : off + n, 0] = self._longitude[:n]
                out[off : off + n, 1] = self._geocentric_latitude[:n]
                out[off : off + n, 2] = self._depth_avg[layer]
            self._coordinates_in_lonlatdepth = out
        return self._coordinates_in_lonlatdepth

    @property
    def coordinates_in_xyz(self) -> np.ndarray:
        """Flat ``(n_model, 3)`` array of ``(x, y, z)`` in metres."""
        if self._coordinates_in_xyz is None:
            out = np.empty((N_MODEL, 3), dtype="float64")
            for layer in range(N_LAYERS):
                n = N_POINTS_UM_TZ if layer < N_LAYERS_UM_TZ else N_POINTS_LM
                off = self._layer_offset(layer)
                radius_km = R_EARTH_KM - self._depth_avg[layer]
                geo = np.column_stack(
                    (
                        np.full(n, radius_km),
                        self._longitude[:n],
                        self._geocentric_latitude[:n],
                    )
                )
                cart = sph2cart(geo2sph(geo))
                out[off : off + n] = cart * 1000.0
            self._coordinates_in_xyz = out
        return self._coordinates_in_xyz

    # -- resolution-matrix filter --------------------------------------------

    def filter_slowness_perturbation(self, du: np.ndarray) -> np.ndarray:
        """Apply the resolution matrix to a slowness-perturbation vector.

        Parameters
        ----------
        du : ndarray, shape (n_model,)
            Slowness perturbation at each grid point.

        Returns
        -------
        ndarray, shape (n_model,)
            Filtered slowness perturbation ``R @ du``.
        """
        if du.shape != (N_MODEL,):
            raise ValueError(f"Expected shape ({N_MODEL},), got {du.shape}")
        if self._R is None:
            from ._resolution_matrix import load_resolution_matrix

            self._R = load_resolution_matrix()
        return self._R @ du

    # -- internal ------------------------------------------------------------

    def _layer_offset(self, layer: int) -> int:
        """Starting index of *layer* in the flat model vector (0-based)."""
        if layer < N_LAYERS_UM_TZ:
            return layer * N_POINTS_UM_TZ
        return N_LAYERS_UM_TZ * N_POINTS_UM_TZ + (layer - N_LAYERS_UM_TZ) * N_POINTS_LM

    @staticmethod
    def _check_layer(layer: int) -> None:
        if not (0 <= layer < N_LAYERS):
            raise ValueError(f"Layer must be 0..{N_LAYERS - 1}, got {layer}")
