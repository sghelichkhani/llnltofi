#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.11.9

"""
Author:   Bernhard Schuberth, LMU Munich, Germany (bernhard.schuberth@lmu.de)
Date:     2019-02-15
Modified: Tom New, The University of Sydney, Australia (tom.new@sydney.edu.au)
Date:     2024-08-12

LLNL_ToFi

Example routines for determining the values of a seismic velocity model on the 
grid points of the LLNL-G3D-JPS model.

    Original work Copyright (C) 2019 Bernhard Schuberth (bernhard.schuberth@lmu.de)
    Modified work Copyright (C) 2024 Tom New (tom.new@sydney.edu.au)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    
"""
# -----------------------------------------------------------------------------

from mpi4py import MPI
import numpy as np
from numpy.linalg import LinAlgError
import pyvista as pv
import gdrift
import spherical
from scipy.interpolate import RBFInterpolator, CubicSpline
from pathlib import Path
import sys

import ctypes as C


from utils import (R_EARTH_KM, LLNL_PATH, LLNL_COORD_FILE, LLNL_DEPTH_FILE,
                   LLNL_R_FILE_PREFIX, nl_UM_TZ, np_UM_TZ, np_LM, n_m,
                   OUTPUT_PATH, OUTFILE_FILT_PREFIX, OUTFILE_PARM_PREFIX,
                   FIREDRAKE_PATH)

import utils


# --------------------------------------------------------------------------
def init_model_parallel(comm=0):

    myrank = comm.Get_rank()
    num_procs = comm.Get_size()

    comm.barrier()

    snd_model = None
    if myrank == 0:
        snd_model = read_model(comm)

    keys = None
    if myrank == 0:
        keys = list(snd_model.keys())
    keys = comm.bcast(keys, root=0)

    rcv_model = {}

    for key in keys:
        snd_array = None
        if myrank == 0:
            snd_array = snd_model[key]
        snd_array = comm.bcast(snd_array, root=0)

        rcv_model[key] = snd_array

    model = pv.UnstructuredGrid(
        rcv_model["cells"], rcv_model["celltypes"], rcv_model["points"])
    model.point_data["du"] = rcv_model["du"]
    model.point_data["v_1D"] = rcv_model["v_1D"]

    comm.barrier()

    return model

# --------------------------------------------------------------------------


def read_model(comm):

    # USER MODIFICATION REQUIRED
    # Please provide the code to read in your model
    myrank = comm.Get_rank()
    print(f"Reading model on process {myrank}")
    model_path = Path(FIREDRAKE_PATH) / Path("ojp-collision_cg/Hall2002") / \
        Path("output_0.pvtu")
    model = pv.read(model_path)
    model = model.clean()  # prune duplicate mesh points
    model.points /= 2.208  # normalise the model
    # drop unneeded arrays
    for array_name in model.point_data.keys():
        if array_name not in ["FullTemperature_CG", "Temperature_Deviation_CG"]:
            del model.point_data[array_name]
    # calculate T and T_av, dropping arrays after they become unneeded
    model.point_data["T"] = model["FullTemperature_CG"] * 3700 + 300
    model.point_data["dT"] = model["Temperature_Deviation_CG"] * \
        (np.max(model["T"]) - np.min(model["T"]))
    model.point_data["T_av"] = model["T"] - model["dT"]
    model.point_data["depth"] = (
        1 - np.linalg.norm(model.points, axis=1)) * R_EARTH_KM * 1.0e3

    slb_pyrolite = gdrift.ThermodynamicModel(
        "SLB_16", "pyrolite", temps=np.linspace(300, 4000), depths=np.linspace(0, 2890e3))
    cammarano_q_model = "Q4"  # choose model from cammarano et al., 2003
    anelasticity = gdrift.CammaranoAnelasticityModel.from_q_profile(
        cammarano_q_model)  # Instantiate the anelasticity model
    anelastic_slb_pyrolite = gdrift.apply_anelastic_correction(
        slb_pyrolite, anelasticity)  # Apply anelastic correction to the thermodynamic model

    # A temperautre profile representing the mantle average temperature
    # This is used to anchor the regularised thermodynamic table (we make sure the seismic speeds are the same at those temperature for the regularised and unregularised table)
    temperature_spline = gdrift.SplineProfile(
        depth=np.asarray([0., 500e3, 2700e3, 3000e3]),
        value=np.asarray([300, 1000, 3000, 4000])
    )

    # Regularising the table
    # Regularisation works by saturating the minimum and maximum of variable gradients with respect to temperature.
    # Default values are between -inf and 0.0; which essentialy prohibits phase jumps that would otherwise render
    # v_s/v_p/rho versus temperature non-unique.
    linear_slb_pyrolite = gdrift.mineralogy.regularise_thermodynamic_table(
        slb_pyrolite, temperature_spline,
        regular_range={"v_s": [-0.5, 0], "v_p": [-0.5, 0.], "rho": [-0.5, 0.]}
    )

    # Regularising the table
    linear_anelastic_slb_pyrolite = gdrift.apply_anelastic_correction(
        linear_slb_pyrolite, anelasticity
    )

    v = "vs"  # choose vp or vs
    if v == "vp":
        temperature_to_v = linear_anelastic_slb_pyrolite.temperature_to_vp
    elif v == "vs":
        temperature_to_v = linear_anelastic_slb_pyrolite.temperature_to_vs
    else:
        raise ValueError("v must be 'vp' or 'vs'")

    model.point_data["v_3D"] = temperature_to_v(
        temperature=np.array(model['T']), depth=np.array(model['depth']))
    model.point_data["v_1D"] = temperature_to_v(
        temperature=np.array(model['T_av']), depth=np.array(model['depth']))
    model.point_data["du"] = 1/model["v_3D"] - 1 / \
        model["v_1D"]  # calculate slowness perturbation

    # drop unneeded point_data arrays
    for array_name in model.point_data.keys():
        if array_name not in ["du", "v_1D"]:
            del model.point_data[array_name]

    model = {
        "cells": np.array(model.cells),
        "celltypes": np.array(model.celltypes),
        "points": np.array(model.points),
        "du": np.array(model["du"]),
        "v_1D": np.array(model["v_1D"])
    }

    print(f"Model loaded on process {myrank}")

    # END USER MODIFICATION REQUIRED

    return model

# --------------------------------------------------------------------------


def project_slowness_3D(model, radius_avg, lat, lon, radius_min, radius_max, grid_spacing):

    # This is a dummy routine that needs to be modified by the user.

    # Please modify the code to obtain 3-D slowness perturbations (i.e., the absolute difference between
    # 3-D and 1-D slowness at the current point) for your model.
    # Depending on your model parametrization (i.e., coarser or finer than LLNL-G3D-JPS), you will need
    # to either perform an interpolation (e.g., from the nearest neighbor's on your grid to the current
    # point in the LLNL-G3D-JPS grid if coarser), or you will have to compute an average value in the
    # volume given by "radius_min" and "radius_max" in vertical direction and grid_spacing as
    # search radius in lateral direction (e.g., by an inverse-distance weighting algorithm).

    # NOTE: Perturbation in slowness is du = (1/v_3D - 1/v_1D), and du is approximately -(v - v_1D)/(v_1D**2)
    #       => du = -dv/v_1D^2 = -dln(v)/v_1D

    # USER MODIFICATION REQUIRED
    radius_avg /= R_EARTH_KM
    spherical_coord = spherical.geo2sph([radius_avg, lon, lat])
    cart_coord = spherical.sph2cart(spherical_coord)
    point = pv.PolyData([cart_coord])
    try:
        du = point.sample(model)["du"][0]
    except Exception as e:
        print(f"Error with coordinate {cart_coord}: {e}")
    del point
    # END USER MODIFICATION REQUIRED

    return du

# --------------------------------------------------------------------------


def model_1D(model, radius):

    # This is a dummy routine that needs to be modified by the user

    # Please modify the code to obtain the 1-D seismic velocity value for the given radius.

    # USER MODIFICATION REQUIRED
    radius /= R_EARTH_KM
    point = pv.PolyData([[radius, 0., 0.]])
    v_1D = point.sample(model)["v_1D"][0]
    del point
    # END USER MODIFICATION REQUIRED

    return v_1D

# --------------------------------------------------------------------------


def get_slowness_layer(model, radius_in, lat, lon, grid_spacing):

    # This is a dummy routine that illustrates how to get values of a seismic velocity
    # model in terms of slowness perturbation du = 1/v_3D - 1/v_1D onto the grid
    # of the LLNL-G3D-JPS tomographic model.
    # Note: dv = -du*v_1D^2 => dv/v_1D = dln(v) = -du*v_1D; du = -dln(v)/v_1D

    # USER MODIFICATION REQUIRED
    # This routine expects radius to be given in km.
    # Thus, normalize the radii if necessary (uncomment the line below if applicable).
    # r_norm = R_EARTH_KM
    r_norm = 1.  # no radius normalization by default
    # END USER MODIFICATION REQUIRED

    # turn input radius into a vector if not already provided in this form
    if np.size(radius_in["avg"]) != np.size(lat):
        radius_avg = np.ones(len(lat)) * radius_in["avg"] / r_norm
        radius_min = np.ones(len(lat)) * radius_in["min"] / r_norm
        radius_max = np.ones(len(lat)) * radius_in["max"] / r_norm
    else:
        radius_avg = radius_in["avg"] / r_norm
        radius_min = radius_in["min"] / r_norm
        radius_max = radius_in["max"] / r_norm

    # Get 1-D seismic velocity for that layer
    v_1D = model_1D(model, radius_avg[0])

    slowness_perturbation = np.ones(len(lat))
    # Loop over all points
    for ip in np.arange(len(lat)):

        # Get 3-D (absolute) slowness perturbation du = 1/v_3D - 1/v_1D
        slowness_perturbation[ip] = project_slowness_3D(
            model, radius_avg[ip], lat[ip], lon[ip], radius_min[ip], radius_max[ip], grid_spacing)

    return slowness_perturbation, v_1D


# --------------------------------------------------------------------------
def reparam(comm, radii, gc_lat, lon, reparam):

    myrank = comm.Get_rank()
    num_procs = comm.Get_size()

    # Get number of layers
    nl = len(radii)

    slowness_perturbation = {}

    if reparam:
        # USER MODIFICATION REQUIRED
        # Initialize the seismic model (if necessary)
        model = init_model_parallel(comm)
        # END USER MODIFICATION REQUIRED

    v_1D = np.zeros(nl)

    for ilyr in range(1, nl+1):

        # Initialize model vectors for that layer
        if ilyr <= nl_UM_TZ:
            cnp = np_UM_TZ
            # nominal grid spacing is 1 degree in the upper mantle and transition zone
            grid_spacing = 111.
        else:
            cnp = np_LM
            # nominal grid spacing is 2 degree in the lower mantle
            grid_spacing = 222.

        slowness_perturbation[ilyr-1] = np.zeros(cnp, dtype='float64')

        if reparam:

            if myrank == 0:
                if ilyr == 1:
                    print('#')
                    print('# reparametrising the model...')
                    print('#       ... layer %2d ...' % ilyr)
                elif ilyr == nl:
                    print('#       ... layer %2d' % ilyr)
                else:
                    print('#       ... layer %2d ...' % ilyr)

            # Distribute work load on all processors
            [cnp_sub, my_ib, my_ie] = utils.parallelize(myrank, num_procs, cnp)

            m_true = np.zeros(cnp, dtype='float64')
            tmp = np.zeros(cnp, dtype='float64')

            # Get slowness and 1-D velocity at current location
            [tmp[my_ib:my_ie], v_1D_tmp] = get_slowness_layer(model, radii[ilyr-1], gc_lat[my_ib:my_ie], lon[my_ib:my_ie],
                                                              grid_spacing)

            v_1D[ilyr-1] = v_1D_tmp

            comm.Allreduce([tmp, MPI.DOUBLE], [
                           slowness_perturbation[ilyr-1], MPI.DOUBLE], op=MPI.SUM)

            if myrank == 0:
                # Note: dv = -du*v_1D^2 => dv/v_1D = dln(v) = -du*v_1D
                # reparametrised model (dln(v))
                m_true = -1. * slowness_perturbation[ilyr-1] * v_1D[ilyr-1]

                # Output reparametrised model
                header = '# v1D: %12.7f ' % v_1D[ilyr-1]
                utils.write_layer(
                    ilyr, m_true, radii[ilyr-1]["avg"], lon, gc_lat, OUTFILE_PARM_PREFIX, string=header)

        else:

            if myrank == 0:
                if ilyr == 1:
                    print('#')
                    print('# Reading the reparametrised model...')
                    print('#       ... layer %2d ...' % ilyr)
                elif ilyr == nl:
                    print('#       ... layer %2d' % ilyr)
                else:
                    print('#       ... layer %2d ...' % ilyr)

            m_true = []
            header = ''
            if myrank == 0:
                # reparametrised model
                [lon_in, gc_lat_in, m_true, header] = utils.read_layer(
                    ilyr, radii[ilyr-1]["avg"], OUTFILE_PARM_PREFIX)

            m_true = comm.bcast(m_true, root=0)
            header = comm.bcast(header, root=0)

            v_1D[ilyr-1] = header[-1]

            # Convert velocity to slowness perturbation du
            # dv = -du*v_1D^2 => dv/v_1D = dln(v) = -du*v_1D, du = -dln(v)/v_1D
            slowness_perturbation[ilyr-1] = -1. * m_true / v_1D[ilyr-1]

    return slowness_perturbation, v_1D
