import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata
from pathlib import Path

from utils import (R_EARTH_KM, LLNL_PATH, LLNL_COORD_FILE, LLNL_DEPTH_FILE,
                   LLNL_R_FILE_PREFIX, nl_UM_TZ, np_UM_TZ, np_LM, n_m,
                   OUTPUT_PATH, OUTFILE_FILT_PREFIX, OUTFILE_PARM_PREFIX)

tofi_files = list(Path(OUTPUT_PATH).glob(f"{OUTFILE_FILT_PREFIX}*.txt"))
tofi_files.sort(key=lambda filename: int(str(filename).split("_")[6]))

lats = np.linspace(-90, 90, 181)
lons = np.linspace(-180, 179, 360)
grid_lon, grid_lat = np.meshgrid(lons, lats)
depths = []
tofi_data = []
for tofi_file in tofi_files:
    depth = float(str(tofi_file).split("_")[7][1:-6])
    depths.append(depth)
    data = pd.read_csv(tofi_file, sep="\s+", skiprows=1, header=None, names=["lon", "lat", "V"])
    gdata = griddata(data[["lon", "lat"]].to_numpy(), data["V"].to_numpy(), (grid_lon, grid_lat), method="cubic")
    tofi_data.append(gdata)

depths = np.array(depths)
radii = (R_EARTH_KM - depths) * 1e3

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
    coords={"r": r, "lat": lat, "lon": lon, "depth": ("r", depths)},
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