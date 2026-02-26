"""Tomographic filtering of a synthetic Gaussian anomaly.

This example builds a synthetic slowness perturbation field — a Gaussian blob
centred beneath a chosen location — and passes it through the LLNL-G3D-JPS
resolution operator. The input and filtered fields are then compared in a
simple histogram to show how the resolution matrix smears and attenuates the
original signal.

No external model data is needed; the anomaly is generated directly on the
LLNL grid points.
"""

import numpy as np
import matplotlib.pyplot as plt

import llnltofi

# ── 1. Set up the grid ──────────────────────────────────────────────────────
grid = llnltofi.Grid()
coords = grid.coordinates_in_lonlatdepth  # (n_model, 3): lon, gc_lat, depth_km
print(f"Grid points: {grid.n_model:,}")

# ── 2. Build a synthetic slowness perturbation ──────────────────────────────
# Place a Gaussian blob at (lon=120°, lat=-25°, depth=600 km) — roughly
# beneath the Indonesian slab.
lon0, lat0, depth0 = 120.0, -25.0, 600.0
sigma_h, sigma_v = 10.0, 50.0  # horizontal (degrees) and vertical (km) widths

dlon = coords[:, 0] - lon0
dlat = coords[:, 1] - lat0
ddep = coords[:, 2] - depth0

du = 0.01 * np.exp(-0.5 * ((dlon / sigma_h)**2 + (dlat / sigma_h)**2 + (ddep / sigma_v)**2))
print(f"Peak input du: {du.max():.4e} s/km")

# ── 3. Apply the resolution matrix ─────────────────────────────────────────
du_filtered = grid.filter_slowness_perturbation(du)
print(f"Peak filtered du: {du_filtered.max():.4e} s/km")

# ── 4. Quick comparison ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")

axes[0].hist(du[du > 0], bins=60)
axes[0].set_xlabel("du (s/km)")
axes[0].set_title("Input anomaly")

axes[1].hist(du_filtered[du_filtered > 0], bins=60)
axes[1].set_xlabel("du (s/km)")
axes[1].set_title("After resolution filter")

plt.show()
