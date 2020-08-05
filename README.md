# DESC
Stellarator Equilibrium Solver

This is a lightweight version of the DESC code in Python. 

The main script is DESC.py, which allows the user to specify the spectral resolution, boundary conditions, pressure and rotational transform profiles, runs the solver, then plots the results.

* backend.py - set of core functions and jax/numpy compatibility layer
* boundary_conditions.py - functions for calculating boundary error
* force_balance.py - functions for evaluating co- and contra-variant bases, magnetic fields, and force balance errors
* gfile_helpers.py - functions for reading/writing gfiles for tokamak GS equilibria
* init_guess.py - functions for generating initial guesses for solution
* plotting.py - routines for plotting solutions
* zernike.py - Zernike/Fourier transforms and basis functions

TODO:
* figure out asymptotics for contravariant basis at axis
* expanding solution to higher resolution
* expanding basis fn matrices to higher spectral/real space resolution without full recompute
* R_zz, Z_zz minimization instead of force
* autodiff hessian
* stellarator symmetry stuff -> maybe just linear constraint setting half coeffs to zero
* squareness
* default node setups
* better plotting for stellarator stuff


* I/O compatibiltiy with VMEC
* command line interface
* documentation
* clean up backend implementation
