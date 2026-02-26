import numpy as np
import pytest

from llnltofi import Grid
from llnltofi._constants import (
    N_LAYERS,
    N_LAYERS_UM_TZ,
    N_MODEL,
    N_POINTS_UM_TZ,
    N_POINTS_LM,
    R_EARTH_KM,
)


@pytest.fixture
def grid():
    return Grid()


def test_n_layers(grid):
    assert grid.n_layers == N_LAYERS


def test_n_model(grid):
    assert grid.n_model == N_MODEL


def test_n_points_um(grid):
    for layer in range(N_LAYERS_UM_TZ):
        assert grid.n_points(layer) == N_POINTS_UM_TZ


def test_n_points_lm(grid):
    for layer in range(N_LAYERS_UM_TZ, N_LAYERS):
        assert grid.n_points(layer) == N_POINTS_LM


def test_layer_radius_vs_depth(grid):
    for layer in range(N_LAYERS):
        r = grid.layer_radius(layer)
        d = grid.layer_depth(layer)
        assert abs(r["avg"] + d["avg"] - R_EARTH_KM) < 0.01


def test_invalid_layer_raises(grid):
    with pytest.raises(ValueError):
        grid.n_points(-1)
    with pytest.raises(ValueError):
        grid.n_points(N_LAYERS)


def test_coordinates_in_lonlatdepth_shape(grid):
    coords = grid.coordinates_in_lonlatdepth
    assert coords.shape == (N_MODEL, 3)


def test_coordinates_in_lonlatdepth_depth_values(grid):
    coords = grid.coordinates_in_lonlatdepth
    depths = coords[:, 2]
    assert depths.min() >= 0.0
    assert depths.max() <= 2891.0


def test_coordinates_in_xyz_shape(grid):
    xyz = grid.coordinates_in_xyz
    assert xyz.shape == (N_MODEL, 3)


def test_coordinates_in_xyz_radius(grid):
    xyz = grid.coordinates_in_xyz
    radii_m = np.linalg.norm(xyz, axis=1)
    radii_km = radii_m / 1000.0
    assert radii_km.min() >= 3480.0
    assert radii_km.max() <= 6371.0


def test_filter_slowness_perturbation_zeros(grid):
    du = np.zeros(N_MODEL)
    result = grid.filter_slowness_perturbation(du)
    np.testing.assert_array_equal(result, 0.0)


def test_filter_shape(grid):
    du = np.zeros(N_MODEL)
    result = grid.filter_slowness_perturbation(du)
    assert result.shape == (N_MODEL,)


def test_filter_wrong_shape_raises(grid):
    with pytest.raises(ValueError):
        grid.filter_slowness_perturbation(np.zeros(100))
