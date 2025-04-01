import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata
from pathlib import Path

from utils import (R_EARTH_KM, LLNL_PATH, LLNL_COORD_FILE, LLNL_DEPTH_FILE,
                   LLNL_R_FILE_PREFIX, nl_UM_TZ, np_UM_TZ, np_LM, n_m,
                   OUTPUT_PATH, OUTFILE_FILT_PREFIX, OUTFILE_PARM_PREFIX)


def grid_llnl_from_txt(path, grid_lon, grid_lat):
    depths = []
    V = []
    files = sorted(path.iterdir(), key=lambda filename: int(
        str(filename).split("_")[-2]))
    for file in files:
        data = pd.read_csv(file, sep="\s+", skiprows=1,
                           header=None, names=["lon", "lat", "V"])
        data = griddata(
            data[["lon", "lat"]].to_numpy(),
            data["V"].to_numpy(), (grid_lon, grid_lat), method="cubic"
        )
        V.append(data)
        depth = float(str(file).split("_")[-1][1:-6])
        depths.append(depth)
    return np.array(V), np.array(depths)


lats = np.linspace(-90, 90, 181)
lons = np.linspace(-180, 179, 360)
grid_lon, grid_lat = np.meshgrid(lons, lats)

reconstruction = "Hall2002"
root_path = Path(
    f"/Volumes/Grey/phd/ojp-collision_cg/{reconstruction}/LLNL_ToFi_3")
reparam_path = root_path / Path("reparam")
dVp_reparam_path = reparam_path / Path("dVp")
dVs_reparam_path = reparam_path / Path("dVs")
tofi_path = root_path / Path("ToFi")
dVp_tofi_path = tofi_path / Path("dVp")
dVs_tofi_path = tofi_path / Path("dVs")
paths = [dVp_reparam_path, dVp_tofi_path, dVs_reparam_path, dVs_tofi_path]
names = ["dVp_reparam_percent", "dVp_tofi_percent",
         "dVs_reparam_percent", "dVs_tofi_percent"]

Vs = []
for path in paths:
    V, depths = grid_llnl_from_txt(path, grid_lon, grid_lat)
    Vs.append(V)

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
ds = xr.Dataset(
    coords={"r": r, "lat": lat, "lon": lon, "depth": ("r", depths)},
    attrs={
        "id": f"{reconstruction} LLNL ToFi"
    }
)

for i in range(len(Vs)):
    Vs[i] *= 100  # convert to percent
    # assign attributes to depth
    ds["depth"] = ds["depth"].assign_attrs({
        "long_name": "depth",
        "units": "\kilo\metre",
        "positive": "down"
    })
    ds[f"{names[i]}"] = (
        ds.dims,
        Vs[i]
    )

    # assign attributes to data
    ds[f"{names[i]}"] = ds[f"{names[i]}"].assign_attrs({
        "long_name": "Velocity perturbation",
        "units": "\percent"
    })

# write to disk
write_path = Path.home() / \
    Path(f"OneDrive/phd/firedrake-models/{reconstruction}_LLNL_ToFi_cg.nc")
ds.to_netcdf(write_path)
