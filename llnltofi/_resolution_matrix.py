from __future__ import annotations

from pathlib import Path

import scipy.sparse

from ._constants import N_MODEL
from ._download import ensure_data


def load_resolution_matrix(path: str | Path | None = None) -> scipy.sparse.csr_matrix:
    """Load the resolution matrix from an ``.npz`` file.

    When called without arguments, loads ``R.npz`` from the package's
    ``data/`` directory (downloading it first if needed).  Pass an explicit
    *path* to load from elsewhere.

    Returns a ``scipy.sparse.csr_matrix`` of shape ``(N_MODEL, N_MODEL)``.
    """
    if path is None:
        try:
            path = ensure_data("R.npz")
        except Exception as exc:
            raise FileNotFoundError(
                "R.npz is not available and could not be downloaded. "
                "Data files are normally fetched during installation "
                "(pip install -e .). If you set PYTHONPATH manually, "
                "run: python -c 'from llnltofi._download import download_all; download_all()'"
            ) from exc
    R = scipy.sparse.load_npz(path)
    if R.shape != (N_MODEL, N_MODEL):
        raise ValueError(f"Expected shape ({N_MODEL}, {N_MODEL}), got {R.shape}")
    return R
