import numpy as np
import pytest
import scipy.sparse

from llnltofi.convert import convert_text_to_npz
from llnltofi._constants import N_LAYERS


@pytest.fixture
def fake_data_dir(tmp_path):
    """Create a directory with tiny fake R-matrix layer files (1-based indexing)."""
    for ilyr in range(1, N_LAYERS + 1):
        rows = np.array([1, 2, 3])
        cols = np.array([1, 2, 3])
        vals = np.array([0.5, 0.3, 0.2])
        data = np.column_stack((rows, cols, vals))
        np.savetxt(
            tmp_path / f"R_Matrix_TomoFilt_Layer_{ilyr}.txt",
            data,
            fmt=["%d", "%d", "%.6f"],
        )
    return tmp_path


def test_convert_creates_npz(fake_data_dir, tmp_path):
    out = tmp_path / "R.npz"
    result = convert_text_to_npz(fake_data_dir, out)
    assert result.exists()
    R = scipy.sparse.load_npz(out)
    assert R.shape[0] == R.shape[1]
    # All entries should be 0-based (indices 0, 1, 2), so max index < 3
    coo = R.tocoo()
    assert coo.row.max() <= 2
    assert coo.col.max() <= 2


def test_convert_zero_based_indexing(fake_data_dir, tmp_path):
    """The text files use 1-based indices. The npz should be 0-based."""
    out = tmp_path / "R.npz"
    convert_text_to_npz(fake_data_dir, out)
    R = scipy.sparse.load_npz(out).tocoo()
    # Original text files have indices 1,2,3 → should become 0,1,2
    assert 0 in R.row
    assert 0 in R.col
