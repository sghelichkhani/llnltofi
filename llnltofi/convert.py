"""Convert LLNL-G3D-JPS resolution-matrix text files to a single sparse ``.npz``."""

from __future__ import annotations

import importlib.resources
from pathlib import Path

import numpy as np
import scipy.sparse

from ._constants import N_LAYERS, N_MODEL

_DATA_DIR = importlib.resources.files("llnltofi") / "data"


def _read_layer_text(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a single COO-format layer file (row_i, column_j, R_ij)."""
    data = np.loadtxt(path)
    row_i = data[:, 0].astype(int)
    col_j = data[:, 1].astype(int)
    vals = data[:, 2]
    return row_i, col_j, vals


def convert_text_to_npz(
    data_dir: str | Path,
    output: str | Path | None = None,
    prefix: str = "R_Matrix_TomoFilt_Layer",
) -> Path:
    """Read 44 COO text files and save as a single sparse CSR ``.npz``.

    The text files use 1-based indexing; this function converts to 0-based.

    Parameters
    ----------
    data_dir : path
        Directory containing the layer text files.
    output : path, optional
        Output ``.npz`` file path.  Defaults to ``llnltofi/data/R.npz``
        inside the installed package, so that ``load_resolution_matrix()``
        picks it up automatically.
    prefix : str
        Filename prefix for the layer files (files are ``{prefix}_{i}.txt``
        for i in 1..44).

    Returns
    -------
    Path to the written ``.npz`` file.
    """
    data_dir = Path(data_dir)
    if output is None:
        output = Path(str(_DATA_DIR)) / "R.npz"
    else:
        output = Path(output)

    all_rows: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    all_vals: list[np.ndarray] = []

    for ilyr in range(1, N_LAYERS + 1):
        layer_file = data_dir / f"{prefix}_{ilyr}.txt"
        row_i, col_j, vals = _read_layer_text(layer_file)
        # Convert from 1-based to 0-based indexing
        all_rows.append(row_i - 1)
        all_cols.append(col_j - 1)
        all_vals.append(vals)

    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    vals = np.concatenate(all_vals)

    R = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(N_MODEL, N_MODEL))
    R = R.tocsr()

    scipy.sparse.save_npz(output, R)
    return output
