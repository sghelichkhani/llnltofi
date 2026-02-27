import numpy as np
import pytest

from llnltofi import ResolutionModel
from llnltofi._constants import N_MODEL
from llnltofi.interpolation import project_onto_grid


@pytest.fixture
def model():
    return ResolutionModel()


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
