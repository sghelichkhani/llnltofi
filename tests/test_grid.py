import numpy as np
import pytest
import scipy.sparse

from llnltofi import ResolutionModel
from llnltofi._constants import (
    N_LAYERS,
    N_LAYERS_UM_TZ,
    N_MODEL,
    N_POINTS_UM_TZ,
    N_POINTS_LM,
    R_EARTH_KM,
)


@pytest.fixture
def model():
    return ResolutionModel()


def test_n_layers(model):
    assert model.n_layers == N_LAYERS


def test_n_model(model):
    assert model.n_model == N_MODEL


def test_n_points_um(model):
    for layer in range(N_LAYERS_UM_TZ):
        assert model.n_points(layer) == N_POINTS_UM_TZ


def test_n_points_lm(model):
    for layer in range(N_LAYERS_UM_TZ, N_LAYERS):
        assert model.n_points(layer) == N_POINTS_LM


def test_layer_radius_vs_depth(model):
    for layer in range(N_LAYERS):
        r = model.layer_radius(layer)
        d = model.layer_depth(layer)
        assert abs(r["avg"] + d["avg"] - R_EARTH_KM) < 0.01


def test_invalid_layer_raises(model):
    with pytest.raises(ValueError):
        model.n_points(-1)
    with pytest.raises(ValueError):
        model.n_points(N_LAYERS)


def test_coordinates_in_lonlatdepth_shape(model):
    coords = model.coordinates_in_lonlatdepth
    assert coords.shape == (N_MODEL, 3)


def test_coordinates_in_lonlatdepth_depth_values(model):
    coords = model.coordinates_in_lonlatdepth
    depths = coords[:, 2]
    assert depths.min() >= 0.0
    assert depths.max() <= 2891.0


def test_coordinates_in_xyz_shape(model):
    xyz = model.coordinates_in_xyz
    assert xyz.shape == (N_MODEL, 3)


def test_coordinates_in_xyz_radius(model):
    xyz = model.coordinates_in_xyz
    radii_m = np.linalg.norm(xyz, axis=1)
    radii_km = radii_m / 1000.0
    assert radii_km.min() >= 3480.0
    assert radii_km.max() <= 6371.0


def test_values_default_none(model):
    assert model.values is None


def test_apply_zeros(model):
    model.values = np.zeros(N_MODEL)
    result = model.apply()
    np.testing.assert_array_equal(result, 0.0)


def test_apply_shape(model):
    model.values = np.zeros(N_MODEL)
    result = model.apply()
    assert result.shape == (N_MODEL,)


def test_values_wrong_shape_raises(model):
    with pytest.raises(ValueError):
        model.values = np.zeros(100)


def test_apply_without_values_raises(model):
    with pytest.raises(RuntimeError):
        model.apply()


def test_R_property_shape(model):
    assert model.R.shape == (N_MODEL, N_MODEL)


def test_R_setter_validates_shape(model):
    with pytest.raises(ValueError):
        model.R = scipy.sparse.eye(10)
