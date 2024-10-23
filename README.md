# LLNL_ToFi

LLNL\_ToFi is a small Python program for tomographic filtering of hypothetical seismic mantle structure ($v_S$ or $v_P$) using the resolution matrix of the LLNL-G3D-JPS model by Simmons et al. (2015). The routine *LLNL_ToFi.py* performs the matrix-vector multiplication $Rm=m'$ to obtain the filtered version $m'$ of the given seismic model $m$. To be able to perform this operation, $m$ needs to be given in the parametrization of the LLNL-G3D-JPS model.

Original author: Bernhard Schuberth (Geophysics, LMU Munich, Germany, bernhard.schuberth@lmu.de)  
Contributing author: Tom New (EarthByte, School of Geosciences, The University of Sydney, Australia, tom.new@sydney.edu.au)

## License

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](LICENSE)

## Development

Development is hosted on [GitHub](https://github.com/tom-new/LLNL_ToFi_3/).

## Data
Input data required by *LLNL_ToFi.py* are located on the [LLNL server](https://gs.llnl.gov/nuclear-threat-reduction/nuclear-explosion-monitoring/global-3d-seismic-tomography) or by email request to <simmons27@llnl.gov>.

## Documentation

  * Information on the resolution matrix and the parametrization of the LLNL-G3D-JPS tomographic model can be found in *LLNL-G3D-JPS\_R\_Matrix\_TomoFilter\_README.pdf*

  * The LLNL-G3D-JPS model is described in  
    Simmons, N. A., Myers, S. C., Johannesson, G., Matzel, E., & Grand, S. P. (2015). Evidence for long-lived subduction of an ancient tectonic plate beneath the southern Indian Ocean. *Geophysical Research Letters, 42*(21), 9270–9278.  
    <https://doi.org/10.1002/2015GL066237>

  * An example of applying the resolution matrix $R$ to a geodynamic model is described in  
    Simmons, N. A., Schuberth, B. S. A., Myers, S. C., & Knapp, D. R. (2019). Resolution and covariance of the LLNL-G3D-JPS global seismic tomography model: applications to travel time uncertainty and tomographic filtering of geodynamic models. *Geophysical Journal International, 217*(3), 1543–1557.  
    <https://doi.org/10.1093/gji/ggz102>

  * Running the code:
    1. To run the code, please first get the necessary input data (i.e., the resolution matrix files) from the source given above and put them into the directory *./DATA*.

    2. Edit the file *model.py* such that the dummy function `project_model_3D` returns the values of your specific seismic velocity model at the given coordinates (radius, lat, lon).

    3. Run the code in a Python 3.x environment use `python3 LLNL_ToFi.py` for serial processing, or `mpirun -n {number of processes} python3 LLNL_Tofi.py` for parallel processing.
       
       Output files will be stored in the directory *./OUTPUT\_FILES*. Files containing the reparametrized model will be named according to the variable `OUTFILE_PARM_PREFIX` [default: `LLNL_G3D_JPS_Parm_layer`] and the tomographically filtered model will be stored according to the variable `OUTFILE_FILT_PREFIX` [default: `LLNL_G3D_JPS_ToFi_layer`].

       Specify the option `-n|--no-reparam` if you run the code the several times and you do not want to perform the reparametrization again. This assumes that the reparametrized version of your seismic model (i.e., on the parametrization of the LLNL-G3D-JPS tomographic model) is already stored in the directory *./OUTPUT\_FILES*.


