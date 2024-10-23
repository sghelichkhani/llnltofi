import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RBFInterpolator
from pathlib import Path

from utils import (R_EARTH_KM, R_CMB_KM, LLNL_PATH, LLNL_COORD_FILE, LLNL_DEPTH_FILE,
                   LLNL_R_FILE_PREFIX, nl_UM_TZ, np_UM_TZ, np_LM, n_m,
                   OUTPUT_PATH, OUTFILE_FILT_PREFIX, OUTFILE_PARM_PREFIX)

tofi_files = list(Path(OUTPUT_PATH).glob(f"{OUTFILE_FILT_PREFIX}*.txt"))
tofi_files.sort(key=lambda filename: int(str(filename).split("_")[6]))

nr, nlat, nlon = 90, 91, 180

radii = np.linspace(R_CMB_KM, R_EARTH_KM, nr) * 1e3 # normalised radius for RBF stability
lats = np.linspace(-90, 90, nlat) # put lat-lon in radians for RBF stability
lons = np.linspace(-180, 179, nlon)
grid = np.array(np.meshgrid(radii, lats, lons, indexing='ij'))
grid_flat = grid.reshape(3,-1).T
tofi_data = []
for tofi_file in tofi_files:
    depth = float(str(tofi_file).split("_")[7][1:-6])
    radius = (R_EARTH_KM - depth) * 1e3
    data = pd.read_csv(tofi_file, sep="\s+", skiprows=1, header=None, names=["lon", "lat", "dVp"])
    # data["lat"] = np.deg2rad(data["lat"])
    # data["lon"] = np.deg2rad(data["lon"])
    data["r"] = radius
    data = data[["r", "lat", "lon", "dVp"]]
    tofi_data += data.values.tolist()

tofi_data = np.array(tofi_data)
print("Making interpolator")
rbf = RBFInterpolator(tofi_data[:,:-1], tofi_data[:,-1], neighbors=64, kernel="linear")
print("Interpolating")
tofi_data = rbf(grid_flat).T.reshape(nr, nlat, nlon)

# # lat-lon back to degrees
# lats, lons = np.rad2deg(lats), np.rad2deg(lons)
# # un-normalise radius
# radii *= R_EARTH_KM * 1e3

# set up DataArrays for primary coordinates
r = xr.DataArray(
    radii,
    dims="r",
    attrs={
        "long_name": "radius",
        "units": "\metre",
        "positive": "up"
    }
)
lat = xr.DataArray(
    lats,
    dims="lat",
    attrs={
        "long_name": "latitude",
        "units": "\degree"
    }
)
lon = xr.DataArray(
    lons,
    dims="lon",
    attrs={
        "long_name": "longitude",
        "units": "\degree",
        "convention": "bipolar"
    }
)

# create dataset
tofi_data = np.array(tofi_data) * 100 # convert to percent
tofi_data = xr.Dataset(
    data_vars={"dVp_percent": (("r", "lat", "lon"), tofi_data)},
    coords={"r": r, "lat": lat, "lon": lon, "depth": ("r", depth)},
    attrs={
        "id": "Hall2002 ToFi"
    }
)

# assign attributes to depth
tofi_data["depth"] = tofi_data["depth"].assign_attrs({
    "long_name": "depth",
    "units": "\kilo\metre",
    "positive": "down"
})
# assign attributes to data
tofi_data["dVp_percent"] = tofi_data["dVp_percent"].assign_attrs({
    "long_name": "Body wave velocity perturbation",
    "units": "\percent"
})

# write to disk
write_path = Path.home() / Path("OneDrive/phd/firedrake-models/Hall2002_ToFi.nc")
tofi_data.to_netcdf(write_path)