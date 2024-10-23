#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.11.9

"""
Program LLNL_ToFi

Author:   Bernhard Schuberth, LMU Munich, Germany (bernhard.schuberth@lmu.de)
Date:     2019-02-15
Modified: Tom New, The University of Sydney, Australia (tom.new@sydney.edu.au)
Date:     2024-08-12

This program performs the matrix-vector multiplication R*m=m' to obtain the 
tomographically filtered version m' of a seismic model m for the resolution 
operator R of the tomographic model LLNL-GD3-JPS. To apply R to the input 
model, it is necessary to project the model first onto the parametrization
of LLNL-G3D-JPS. The information on depth layers, grid point coordinates 
and the values of the R matrix are read from files stored in the directory 
"./DATA" (see README.md for details).

Options:
The flag "-n|--no-reparam" can be used to skip the reparametrisation
(e.g., in case one runs the filtering several times for one model that
already has been reparametrised).


    Original work Copyright (C) 2019 Bernhard Schuberth (bernhard.schuberth@lmu.de)
    Modified work Copyright (C) 2024 Tom New (tom.new@sydney.edu.au)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.


"""
#-----------------------------------------------------------------------------

from mpi4py import MPI
import numpy as np
from pathlib import Path
import sys
import getopt

import ctypes as C


from utils import (R_EARTH_KM, LLNL_PATH, LLNL_COORD_FILE, LLNL_DEPTH_FILE,
                   LLNL_R_FILE_PREFIX, nl_UM_TZ, np_UM_TZ, np_LM, n_m,
                   OUTPUT_PATH, OUTFILE_FILT_PREFIX, OUTFILE_PARM_PREFIX)

import utils

import model

#-----------------------------------------------------------------------------

# Make sure OUTPUT_PATH exists
Path(OUTPUT_PATH).mkdir(exist_ok=True)

def usage():

    print('')
    print('Usage: python '+sys.argv[0]+' [-n|--no-reparam]"' )
    print('       or' )
    print('       mpirun -np {number of processes} python '+sys.argv[0]+' [-n|--no-reparam]"' )
    print('')
    print('Options:')
    print('         -n | --no-reparam       This flag can be used to skip the reparametrisation')
    print('                                 (e.g., in case one runs the filtering several times')
    print('                                 for one model that already has been reparametrised).')
    print('')

    return 0

#--------------------------------------------------------------------------    
def main(argv):
    
    # Set defaults
    reparam = True

    try:                                
        opts, args = getopt.getopt(argv, "hn", [ "help", "no-reparam" ])
    except getopt.GetoptError:          
        usage()                         
        sys.exit(2)                     
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-n", "--no-reparam"):
            reparam = False


    # Initialize MPI
    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
    num_procs = comm.Get_size()

    gd_lat     = []
    lon        = []
    gc_lat     = []
    top_radius = []
    radii     = {}

    if myrank == 0:
        # Read coordinates of LLNL-GD3-JPS
        [ gd_lat,lon,gc_lat,top_radius ] = utils.read_coords()
        # Read radii of LLNL-GD3-JPS
        radii = utils.read_radii()


    gd_lat     = comm.bcast(gd_lat, root=0)
    lon        = comm.bcast(lon, root=0)
    gc_lat     = comm.bcast(gc_lat, root=0)
    top_radius = comm.bcast(top_radius, root=0)
    radii      = comm.bcast(radii, root=0)

    # Get number of layers
    nl = len(radii)

    # Get model on LLNL grid
    [slowness_perturbation, v_1D] = model.reparam(comm,radii,gc_lat,lon,reparam)

    for ilyr in range(1,nl+1):

        n        = []
        row_i    = []
        column_j = []
        R_ij     = []
        
        if myrank == 0:
            if ilyr == 1:
                print('#')
                print('# Filtering the model...')
                print('#       ... layer %2d ...' % ilyr)
            elif ilyr == nl:
                print('#       ... layer %2d' % ilyr)
            else:
                print('#       ... layer %2d ...' % ilyr)

            [row_i,column_j,R_ij] = utils.read_layer_R(ilyr)

            # Get number of points in this layer
            n = len(R_ij)

        # Broadcast the information to all processors
        n        = comm.bcast(n, root=0)
        row_i    = comm.bcast(row_i,root=0)
        column_j = comm.bcast(column_j,root=0)
        R_ij     = comm.bcast(R_ij,root=0)

        # Distribute work load on all processors
        [ n_sub, my_ib, my_ie ] = utils.parallelize(myrank,num_procs,n)


        # Initialize model vectors for that layer
        if ilyr <= nl_UM_TZ:
            m_prime = np.zeros(np_UM_TZ)
            my_m_prime = np.zeros(np_UM_TZ)
        else:
            m_prime = np.zeros(np_LM)
            my_m_prime = np.zeros(np_LM)


        # We only use chunks of the model vector limited to the current layer. Therefore, we
        # need to calculate and offset of the index for the current layer
        offset = utils.row_index_offset(ilyr)

        # Loop over entries in file and multiply with corresponding element of the
        # model vector
        for ip in range(my_ib,my_ie+1):
            
            # Compute indices for current point
            [ c_index, l_index ]  = utils.calculate_coord_index(column_j[ip])

            cri = row_i[ip] - offset - 1 # calculate current row index in the part of the model vector for the current layer

            # Tomographically filtered model
            my_m_prime[cri] = my_m_prime[cri] + R_ij[ip] * slowness_perturbation[l_index][c_index]


        # Now collect all contributions on the master proc
        comm.Reduce([my_m_prime, MPI.DOUBLE], [m_prime, MPI.DOUBLE], op = MPI.SUM, root = 0)

        # Output filtered model only on master proc
        if myrank == 0:
            # Convert slowness perturbation to relative velocity perturbation for filtered model
            m_prime = m_prime * -v_1D[ilyr-1] # dv = du*v_1D^2 => dv/v_1D = du*v_1D

            # Now output the filtered model
            utils.write_layer(ilyr,m_prime,radii[ilyr-1]["avg"],lon,gc_lat,OUTFILE_FILT_PREFIX)
    


#-----------------------------------------------------------------------------

if __name__ == "__main__":
    main(sys.argv[1:])

