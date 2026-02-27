"""Cross-validation between modern llnltofi and original LLNL_ToFi.

Verifies that both packages produce identical results when given the same
input model vector. Requires the original DATA/ directory containing the
44 R-matrix text files, the coordinate file, and the depth file.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse

from llnltofi import ResolutionModel, load_resolution_matrix
from llnltofi._constants import (
    N_LAYERS,
    N_LAYERS_UM_TZ,
    N_MODEL,
    N_POINTS_UM_TZ,
    N_POINTS_LM,
)
from llnltofi._download import ensure_data

# ---------------------------------------------------------------------------
# Path to the original LLNL-G3D-JPS data files.
# Override with the LLNL_TOFI_DATA environment variable when running in CI.
# ---------------------------------------------------------------------------
DATA_DIR = Path(
    os.environ.get("LLNL_TOFI_DATA", "")
    or str(Path(__file__).resolve().parents[2] / "DATA")
)

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "R_Matrix_TomoFilt_Layer_1.txt").exists(),
    reason=f"Original DATA/ directory not found at {DATA_DIR}",
)


# ---------------------------------------------------------------------------
# Original code reproductions (verbatim from LLNL_ToFi_3/utils.py)
# ---------------------------------------------------------------------------

_NL_UM_TZ = 18
_NP_UM_TZ = 40962
_NP_LM = 10242
_N_M = 1_003_608


def _row_index_offset(ilyr):
    """Row index offset for 1-based layer number (utils.py:249)."""
    if ilyr <= _NL_UM_TZ:
        return (ilyr - 1) * _NP_UM_TZ
    return _NL_UM_TZ * _NP_UM_TZ + (ilyr - _NL_UM_TZ - 1) * _NP_LM


def _calculate_coord_index(cj):
    """Map 1-based global column to (c_index, l_index) (utils.py:262)."""
    ntot_UM_TZ = _NL_UM_TZ * _NP_UM_TZ
    if cj <= ntot_UM_TZ:
        c_index = np.mod(cj, _NP_UM_TZ) - 1
        l_index = cj // _NP_UM_TZ
    else:
        cj_rem = cj - ntot_UM_TZ
        c_index = np.mod(cj_rem, _NP_LM) - 1
        l_index = _NL_UM_TZ + cj_rem // _NP_LM
    if cj == _N_M:
        l_index = 43
    return c_index, l_index


def _calculate_coord_index_vectorized(cj_array):
    """Vectorized _calculate_coord_index for arrays of 1-based indices."""
    cj = np.asarray(cj_array)
    ntot_UM_TZ = _NL_UM_TZ * _NP_UM_TZ

    c_index = np.empty(len(cj), dtype=int)
    l_index = np.empty(len(cj), dtype=int)

    um_mask = cj <= ntot_UM_TZ
    c_index[um_mask] = np.mod(cj[um_mask], _NP_UM_TZ) - 1
    l_index[um_mask] = cj[um_mask] // _NP_UM_TZ

    lm_mask = ~um_mask
    cj_rem = cj[lm_mask] - ntot_UM_TZ
    c_index[lm_mask] = np.mod(cj_rem, _NP_LM) - 1
    l_index[lm_mask] = _NL_UM_TZ + cj_rem // _NP_LM

    last_mask = cj == _N_M
    l_index[last_mask] = 43

    return c_index, l_index


def _original_indices_to_flat(c_indices, l_indices):
    """Convert (c_index, l_index) pairs from the original mapping to flat indices.

    Handles the original code's negative c_index values via Python-style modulo.
    """
    flat = np.empty(len(c_indices), dtype=int)

    um = l_indices < _NL_UM_TZ
    flat[um] = l_indices[um] * _NP_UM_TZ + (c_indices[um] % _NP_UM_TZ)

    lm = ~um
    flat[lm] = (
        _NL_UM_TZ * _NP_UM_TZ
        + (l_indices[lm] - _NL_UM_TZ) * _NP_LM
        + (c_indices[lm] % _NP_LM)
    )

    return flat


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def modern_model():
    return ResolutionModel()


@pytest.fixture(scope="session")
def layer_R_data():
    """Read all 44 R-matrix text files once. Returns list of (row, col, val) tuples (1-based)."""
    layers = []
    for ilyr in range(1, N_LAYERS + 1):
        path = DATA_DIR / f"R_Matrix_TomoFilt_Layer_{ilyr}.txt"
        data = np.loadtxt(path)
        layers.append((data[:, 0].astype(int), data[:, 1].astype(int), data[:, 2]))
    return layers


@pytest.fixture(scope="session")
def R_from_text(layer_R_data):
    """R matrix built from text files with correct 0-based indexing."""
    rows = np.concatenate([entry[0] for entry in layer_R_data]) - 1
    cols = np.concatenate([entry[1] for entry in layer_R_data]) - 1
    vals = np.concatenate([entry[2] for entry in layer_R_data])
    return scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(N_MODEL, N_MODEL)
    ).tocsr()


@pytest.fixture(scope="session")
def R_bundled():
    """R matrix loaded from the modern package's bundled R.npz."""
    return load_resolution_matrix()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_R_matrix_equivalence(R_bundled, R_from_text):
    """Bundled R.npz must be bit-for-bit identical to R built from text files."""
    diff = R_bundled - R_from_text
    assert diff.nnz == 0, f"R matrices differ in {diff.nnz} entries"


def test_coordinates_match():
    """Grid coordinates in grid_data.npz must match the original text file."""
    original = np.loadtxt(
        DATA_DIR / "LLNL_G3D_JPS.Tessellated.Coordinates.txt"
    )
    modern = np.load(ensure_data("grid_data.npz"))["coordinates"]
    np.testing.assert_array_almost_equal(modern, original)


def test_layer_depths_match():
    """Layer depths in grid_data.npz must match the original text file."""
    original = np.loadtxt(
        DATA_DIR / "LLNL_G3D_JPS.Layer_Depths_min_avg_max.txt"
    )
    modern = np.load(ensure_data("grid_data.npz"))["layer_depths"]
    np.testing.assert_array_almost_equal(modern, original)


def test_apply_matches_direct_text_R(R_from_text, modern_model):
    """model.apply() must equal R_from_text @ values for random input."""
    rng = np.random.default_rng(42)
    values = rng.standard_normal(N_MODEL)

    modern_model.values = values
    result_modern = modern_model.apply()
    result_text = R_from_text @ values

    np.testing.assert_allclose(result_modern, result_text, rtol=1e-12)


@pytest.mark.xfail(
    reason=(
        "The original calculate_coord_index has a boundary bug: when the "
        "global column index is an exact multiple of the layer size, modular "
        "arithmetic yields c_index=-1 and l_index is off by one, reading a "
        "value from the wrong depth layer.  2,357 / 75M R entries are "
        "affected, causing 1,107 / 1,003,608 output points to differ."
    ),
    strict=True,
)
def test_original_loop_vs_modern(layer_R_data, modern_model):
    """Replicate the original element-by-element R multiplication and compare.

    This test faithfully reproduces the loop from LLNL_ToFi.py (lines 222-237),
    vectorized per-layer for performance. It uses the original
    calculate_coord_index mapping to look up model values, exactly as the
    original code does.
    """
    rng = np.random.default_rng(42)
    flat_values = rng.standard_normal(N_MODEL)

    # Apply using the original layer-by-layer loop (vectorized within each layer)
    result_original = np.zeros(N_MODEL)

    for ilyr in range(1, N_LAYERS + 1):
        row_i, column_j, R_ij = layer_R_data[ilyr - 1]

        n_pts = N_POINTS_UM_TZ if ilyr <= N_LAYERS_UM_TZ else N_POINTS_LM
        offset = _row_index_offset(ilyr)

        # Local row index within the current layer (same as cri in the original)
        cri = row_i - offset - 1

        # Use the original calculate_coord_index mapping to find values
        c_indices, l_indices = _calculate_coord_index_vectorized(column_j)
        flat_col = _original_indices_to_flat(c_indices, l_indices)
        looked_up = flat_values[flat_col]

        # Accumulate (equivalent to original's m_prime[cri] += R_ij * val)
        m_prime = np.zeros(n_pts)
        np.add.at(m_prime, cri, R_ij * looked_up)

        layer_off = _row_index_offset(ilyr)
        result_original[layer_off : layer_off + n_pts] = m_prime

    # Apply using the modern scipy sparse multiply
    modern_model.values = flat_values
    result_modern = modern_model.apply()

    np.testing.assert_allclose(result_original, result_modern, rtol=1e-12)


# ---------------------------------------------------------------------------
# Boundary bug characterisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cj, expected_flat",
    [
        (1, 0),  # first point — should be correct
        (2, 1),  # second point — should be correct
        (40962, 40961),  # last point of UM layer 0 — boundary
        (40963, 40962),  # first point of UM layer 1 — should be correct
        (81924, 81923),  # last point of UM layer 1 — boundary
        (737316, 737315),  # last point of UM layer 17 — boundary
        (737317, 737316),  # first point of LM layer 0 — should be correct
        (747558, 747557),  # last point of LM layer 0 — boundary
        (1003608, 1003607),  # last point overall — patched in original
    ],
)
def test_calculate_coord_index_boundary(cj, expected_flat):
    """Document the boundary-case behaviour of the original calculate_coord_index.

    For column indices that are exact multiples of np_UM_TZ (or np_LM),
    the original code's modular arithmetic yields c_index = -1 and l_index
    one too high, causing it to look up the last element of the next layer
    rather than the current one.
    """
    c_index, l_index = _calculate_coord_index(cj)

    if l_index < _NL_UM_TZ:
        flat_original = l_index * _NP_UM_TZ + (c_index % _NP_UM_TZ)
    else:
        flat_original = (
            _NL_UM_TZ * _NP_UM_TZ
            + (l_index - _NL_UM_TZ) * _NP_LM
            + (c_index % _NP_LM)
        )

    if flat_original != expected_flat:
        pytest.xfail(
            f"Known boundary issue: cj={cj} maps to flat={flat_original} "
            f"instead of expected {expected_flat}"
        )
    assert flat_original == expected_flat


def test_original_vs_modern_discrepancy_count(layer_R_data):
    """Count how many R-matrix entries are affected by the boundary issue."""
    boundary_cj = set()
    for k in range(1, _NL_UM_TZ + 1):
        boundary_cj.add(k * _NP_UM_TZ)
    ntot_UM_TZ = _NL_UM_TZ * _NP_UM_TZ
    for k in range(1, 27):
        boundary_cj.add(ntot_UM_TZ + k * _NP_LM)

    total_affected = 0
    total_entries = 0
    for row_i, col_j, R_ij in layer_R_data:
        total_entries += len(col_j)
        total_affected += np.isin(col_j, list(boundary_cj)).sum()

    print(
        f"\nR entries at boundary columns: {total_affected} / {total_entries}"
    )
