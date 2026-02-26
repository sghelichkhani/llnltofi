from __future__ import annotations

import importlib.resources
from pathlib import Path

import scipy.sparse

from ._constants import N_MODEL

_DEFAULT_PATH = importlib.resources.files("llnltofi") / "data" / "R.npz"


def load_resolution_matrix(path: str | Path | None = None) -> scipy.sparse.csr_matrix:
    """Load the resolution matrix from an ``.npz`` file.

    When called without arguments, loads the bundled ``R.npz`` from the
    package's ``data/`` directory.  Pass an explicit *path* to load from
    elsewhere.

    Returns a ``scipy.sparse.csr_matrix`` of shape ``(N_MODEL, N_MODEL)``.
    """
    if path is None:
        path = _DEFAULT_PATH
    R = scipy.sparse.load_npz(path)
    if R.shape != (N_MODEL, N_MODEL):
        raise ValueError(f"Expected shape ({N_MODEL}, {N_MODEL}), got {R.shape}")
    return R
