This folder contains the files necessary to reproduce the results in the paper "??".
A copy of this paper is included as `dudt2023omnigenity.pdf`.
In order to run the following code, checkout version `8fcf0d13187b2b9ff8f9fcba3a7d795df35a9380` of DESC.

The included files are as follows:

- `name.py` is a Python script that runs the optimization in DESC.
- `name.h5` is the DESC HDF5 output file of the Equilibrium class representing the solution.
- `wout_name.nc` is the VMEC NetCDF output file representing the solution.
- `in_booz.name` is the BOOZ_XFORM input file (required for running NEO).
- `neo_in.name` is the NEO input file.
- `name.npy` is a NumPy array with the effective ripple values from the NEO output.
- `plotter.py` is a Python script that generates the figures in the paper.

`name` in the above filenames corresponds to any of the following six examples:

- `poloidal` M=0, N=1 omnigentiy
- `poloidal_qs` M=0, N=1 quasi-symmetry
- `helical` M=1, N=5 omnigentiy
- `helical_qs` M=1, N=5 quasi-symmetry
- `toroidal` M=1, N=0 omnigentiy
- `toroidal_qs` M=1, N=0 quasi-symmetry
