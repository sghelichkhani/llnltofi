import numpy as np
import pandas as pd
import pyvista as pv
import xarray as xr
import gdrift
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
cm = 1/2.54

from llnltofi._constants import R_EARTH_KM

OUTPUT_PATH = './OUTPUT_FILES/'
OUTFILE_PARM_PREFIX = 'LLNL_G3D_JPS_Parm_layer'
OUTFILE_FILT_PREFIX = 'LLNL_G3D_JPS_ToFi_layer'
FIREDRAKE_PATH = '/Volumes/Grey/phd'


# fractional dvp from gdrift
model_path = Path(FIREDRAKE_PATH) / Path("Mann2004/Stage_28_Gplates") / Path("output_4.pvtu")
model = pv.read(model_path)
model = model.clean()
model.points /= 2.22 # normalise the model for rbf interpolation
model.point_data['T'] = model['FullTemperature'] * 3700 + 300
model.point_data['dT'] = model['DeltaT'] * (np.max(model['T']) - np.min(model['T']))
model.point_data['T_av'] = model['T'] - model['dT']
model.point_data['radii'] = np.linalg.norm(model.points, axis=1)

model.point_data['depth'] = (R_EARTH_KM - model["radii"]) * 1.0e3

slb_pyrolite = gdrift.ThermodynamicModel('SLB_16', 'pyrolite')

model.point_data['vp'] = slb_pyrolite.temperature_to_vp(temperature=np.array(model['T']), depth=np.array(model['depth']))
model.point_data['vp_av'] = slb_pyrolite.temperature_to_vp(temperature=np.array(model['T_av']), depth=np.array(model['depth']))
model.point_data["dvp"] = (model["vp"] - model["vp_av"]) / model["vp_av"]

model.point_data['vs'] = slb_pyrolite.temperature_to_vs(temperature=np.array(model['T']), depth=np.array(model['depth']))
model.point_data['vs_av'] = slb_pyrolite.temperature_to_vs(temperature=np.array(model['T_av']), depth=np.array(model['depth']))


# fractional dvp after downsampling to LLNL grid
llnl_files = list(Path(OUTPUT_PATH).glob(f"{OUTFILE_PARM_PREFIX}*.txt"))
llnl_files.sort(key=lambda filename: int(str(filename).split("_")[6]))

llnl_dvp = []
for llnl_file in llnl_files:
    data = pd.read_csv(llnl_file, sep="\s+", skiprows=1, header=None, names=["lon", "lat", "du"])
    du = data["du"].to_numpy()

    with open(llnl_file) as f:
        v_1D = float(f.readline()[46:])

    v_3D = np.reciprocal(du + np.reciprocal(v_1D))
    dv = (v_3D - v_1D) / v_1D
    llnl_dvp += list(dv)

llnl_dvp = np.array(llnl_dvp)

llnl2_files = list(Path(OUTPUT_PATH).glob(f"{OUTFILE_FILT_PREFIX}*.txt"))
llnl2_files.sort(key=lambda filename: int(str(filename).split("_")[6]))


# fractional dvp after resolution operator
llnl2_dvp = []
for llnl2_file in llnl2_files:
    data = pd.read_csv(llnl2_file, sep="\s+", skiprows=1, header=None, names=["lon", "lat", "dvp"])
    dvp = data["dvp"].to_list()
    llnl2_dvp += dvp

llnl2_dvp = np.array(llnl_dvp)


# fractional dvp after upsampling to lld grid
llnl3_path = Path.home() / Path("OneDrive/phd/firedrake-models/Mann2004_ToFi.nc")
llnl3 = xr.open_dataset(llnl3_path)


dvps = [model["dvp"], llnl_dvp, llnl2_dvp, llnl3["dVp_percent"].data.flatten()/100]
labels = ["firedrake", "downsampled to LLNL", "downsampled and filtered", "filtered and resampled to lat-lon-depth"]

fig, axs = plt.subplots(2, 2, figsize=(20*cm, 20*cm), layout="constrained")
axs = axs.flatten()
for i in range(4):
    axs[i].hist(dvps[i], bins=20)
    axs[i].set_title(labels[i])

plt.show()
