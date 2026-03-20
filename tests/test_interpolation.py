import warnings

import numpy as np
import pytest

from llnltofi import ResolutionModel
from llnltofi._constants import (
    N_MODEL,
    R_EARTH_KM,
    N_LAYERS,
    N_LAYERS_UM_TZ,
    N_POINTS_UM_TZ,
    N_POINTS_LM,
)
from llnltofi.interpolation import project_onto_grid, project_from_grid


@pytest.fixture
def model():
    return ResolutionModel()


# ── project_onto_grid tests ──────────────────────────────────────────────────


def test_constant_field_single_layer(model):
    """A spatially constant source field should interpolate to that constant."""
    radii = model.layer_radius(0)
    n = model.n_points(0)

    # Build a dense cloud of source points within the first layer's radial shell
    n_src = 5000
    rng = np.random.default_rng(99)
    src_lat = rng.uniform(-90, 90, n_src)
    src_lon = rng.uniform(-180, 180, n_src)
    src_r = rng.uniform(radii["min"], radii["max"], n_src)

    source_points = np.column_stack((src_lat, src_lon, src_r))
    source_values = np.full(n_src, 3.14)

    result = project_onto_grid(source_points, source_values, model, k=50)
    # Only check layer 0 (other layers may not have source points)
    np.testing.assert_allclose(result[:n], 3.14, atol=1e-6)


def test_output_shape(model):
    """project_onto_grid should return a flat (n_model,) array."""
    rng = np.random.default_rng(42)
    n_src = 20000
    src_lat = rng.uniform(-90, 90, n_src)
    src_lon = rng.uniform(-180, 180, n_src)
    src_r = rng.uniform(3500, 6370, n_src)

    source_points = np.column_stack((src_lat, src_lon, src_r))
    source_values = rng.standard_normal(n_src)

    result = project_onto_grid(source_points, source_values, model, k=20)
    assert result.shape == (N_MODEL,)


def test_bad_input_shape():
    model = ResolutionModel()
    with pytest.raises(ValueError):
        project_onto_grid(np.zeros((10, 2)), np.zeros(10), model)
    with pytest.raises(ValueError):
        project_onto_grid(np.zeros((10, 3)), np.zeros(5), model)


# ── project_from_grid tests ──────────────────────────────────────────────────


def test_back_constant_field(model):
    """Back-projecting a constant grid vector should return that constant."""
    grid_values = np.full(N_MODEL, 42.0)
    query_points = np.array([
        [0.0, 0.0, 5500.0],
        [45.0, 90.0, 4500.0],
        [-30.0, -120.0, 6000.0],
        [89.0, 0.0, 5000.0],
        [-89.0, 180.0, 3600.0],
    ])
    result = project_from_grid(grid_values, query_points, model)
    np.testing.assert_allclose(result, 42.0, atol=1e-6)


def test_back_output_shape(model):
    """project_from_grid should return shape (N,)."""
    grid_values = np.ones(N_MODEL)
    query_points = np.column_stack([
        np.linspace(-80, 80, 50),
        np.linspace(-170, 170, 50),
        np.linspace(3600, 6300, 50),
    ])
    result = project_from_grid(grid_values, query_points, model)
    assert result.shape == (50,)


def test_back_linear_in_depth_lm(model):
    """A field linear in radius should interpolate exactly (LM layers)."""
    layer_radii = np.array([
        R_EARTH_KM - model._depth_avg[l] for l in range(N_LAYERS)
    ])
    grid_values = np.empty(N_MODEL)
    for layer in range(N_LAYERS):
        n = model.n_points(layer)
        off = model._layer_offset(layer)
        grid_values[off:off + n] = layer_radii[layer]

    query_pts = []
    for l in range(N_LAYERS_UM_TZ + 1, N_LAYERS - 1):
        r_mid = 0.5 * (layer_radii[l] + layer_radii[l + 1])
        query_pts.append([0.0, 0.0, r_mid])
    query_pts = np.array(query_pts)
    expected = query_pts[:, 2]

    result = project_from_grid(grid_values, query_pts, model)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_back_linear_in_depth_umtz(model):
    """A field linear in radius should interpolate exactly (UM/TZ layers)."""
    layer_radii = np.array([
        R_EARTH_KM - model._depth_avg[l] for l in range(N_LAYERS)
    ])
    grid_values = np.empty(N_MODEL)
    for layer in range(N_LAYERS):
        n = model.n_points(layer)
        off = model._layer_offset(layer)
        grid_values[off:off + n] = layer_radii[layer]

    query_pts = []
    for l in range(0, N_LAYERS_UM_TZ - 2):
        r_mid = 0.5 * (layer_radii[l] + layer_radii[l + 1])
        query_pts.append([0.0, 0.0, r_mid])
    query_pts = np.array(query_pts)
    expected = query_pts[:, 2]

    result = project_from_grid(grid_values, query_pts, model)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_back_660km_boundary(model):
    """Query points spanning the 660 km UM/TZ-to-LM transition."""
    grid_values = np.ones(N_MODEL) * 5000.0
    depths_km = np.linspace(620, 720, 20)
    query_points = np.column_stack([
        np.zeros(20),
        np.zeros(20),
        R_EARTH_KM - depths_km,
    ])
    result = project_from_grid(grid_values, query_points, model)
    np.testing.assert_allclose(result, 5000.0, atol=1e-6)


def test_back_660km_varying_field(model):
    """A spatially varying field should interpolate smoothly across 660 km."""
    layer_radii = np.array([
        R_EARTH_KM - model._depth_avg[l] for l in range(N_LAYERS)
    ])
    grid_values = np.empty(N_MODEL)
    for layer in range(N_LAYERS):
        n = model.n_points(layer)
        off = model._layer_offset(layer)
        grid_values[off:off + n] = layer_radii[layer]

    depths_km = np.linspace(630, 700, 30)
    radii_km = R_EARTH_KM - depths_km
    query_points = np.column_stack([
        np.full(30, 10.0),
        np.full(30, 45.0),
        radii_km,
    ])
    result = project_from_grid(grid_values, query_points, model)
    # Result should be monotonically decreasing (radius decreases with depth)
    assert np.all(np.diff(result) < 0), "Expected monotonic decrease across 660 km"


def test_back_lateral_variation(model):
    """A field varying laterally should be interpolated, not smeared out."""
    grid_values = np.empty(N_MODEL)
    for layer in range(N_LAYERS):
        n = model.n_points(layer)
        off = model._layer_offset(layer)
        lat = model._geocentric_latitude[:n]
        grid_values[off:off + n] = np.sin(np.radians(lat * 2))

    # Query at several latitudes at a mid-LM depth
    lats = np.array([-60.0, -30.0, 0.0, 30.0, 60.0])
    query_points = np.column_stack([
        lats,
        np.zeros(5),
        np.full(5, 5000.0),
    ])
    result = project_from_grid(grid_values, query_points, model)
    expected = np.sin(np.radians(lats * 2))
    np.testing.assert_allclose(result, expected, atol=0.05)


def test_back_lateral_variation_umtz(model):
    """Lateral variation test in the UM/TZ region."""
    grid_values = np.empty(N_MODEL)
    for layer in range(N_LAYERS):
        n = model.n_points(layer)
        off = model._layer_offset(layer)
        lon = model._longitude[:n]
        grid_values[off:off + n] = np.cos(np.radians(lon))

    lons = np.array([-90.0, 0.0, 90.0, 180.0])
    query_points = np.column_stack([
        np.zeros(4),
        lons,
        np.full(4, 6100.0),
    ])
    result = project_from_grid(grid_values, query_points, model)
    expected = np.cos(np.radians(lons))
    np.testing.assert_allclose(result, expected, atol=0.05)


def test_back_poles(model):
    """Query points at the geographic poles where longitude is degenerate."""
    grid_values = np.full(N_MODEL, 99.0)
    query_points = np.array([
        [90.0, 0.0, 5500.0],
        [90.0, 90.0, 5500.0],
        [90.0, -180.0, 5500.0],
        [-90.0, 0.0, 5500.0],
        [-90.0, 45.0, 4000.0],
    ])
    result = project_from_grid(grid_values, query_points, model)
    np.testing.assert_allclose(result, 99.0, atol=1e-6)


def test_back_boundary_clamp(model):
    """Query above shallowest and below deepest layer should clamp with warning."""
    grid_values = np.full(N_MODEL, 7.0)
    query_points = np.array([
        [0.0, 0.0, R_EARTH_KM + 100.0],
        [0.0, 0.0, 3000.0],
    ])
    with pytest.warns(UserWarning, match="outside the model's radial extent"):
        result = project_from_grid(grid_values, query_points, model)
    np.testing.assert_allclose(result, 7.0, atol=1e-6)


def test_back_nan_warning(model):
    """Grid values with NaN should trigger a warning."""
    grid_values = np.ones(N_MODEL)
    grid_values[100] = np.nan
    query_points = np.array([[0.0, 0.0, 5500.0]])
    with pytest.warns(UserWarning, match="NaN"):
        project_from_grid(grid_values, query_points, model)


def test_back_bad_input_shape():
    model = ResolutionModel()
    with pytest.raises(ValueError):
        project_from_grid(np.zeros(10), np.zeros((5, 3)), model)
    with pytest.raises(ValueError):
        project_from_grid(np.zeros(N_MODEL), np.zeros((5, 2)), model)
