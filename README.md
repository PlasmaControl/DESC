# DESC
Stellarator Equilibrium Solver

This is a lightweight version of the DESC code in Python. 

The main script is DESC.py, which allows the user to specify the spectral resolution, boundary conditions, pressure and rotational transform profiles, runs the solver, then plots the results.

* backend.py - set of core functions and jax/numpy compatibility layer
* boundary_conditions.py - functions for calculating boundary error
* example_inputs.py - some default scenarios to use for testing (D shaped tokamak, heliotron, etc)
* force_balance.py - functions for evaluating co- and contra-variant bases, magnetic fields, and force balance errors
* gfile_helpers.py - functions for reading/writing gfiles for tokamak GS equilibria
* init_guess.py - functions for generating initial guesses for solution
* plotting.py - routines for plotting solutions
* zernike.py - Zernike/Fourier transforms and basis functions
* nodes.py - generate collocation nodes

TODO:
* memory management for gpu
* what needs jit, which devices
* precompute SVD for fitting

* force balance
    - project back to get R_tt,Z_tt
    - incompressibility constraint

* symmetry, symmetric nodes, best way to enforce and reduce size
    - masking
    
* avoiding recompilation
    - masking?
    - comparing masked arrays for bdry

* figure out asymptotics for contravariant basis at axis
* zernike transform using FFT
* function for quasi-symmetry
* allow option for square system
* use "root" algorithms instead of "least_squares" (and compare results)
* autodiff hessian
* option for 3/2 rule to prevent spectral aliasing in BC error function
* add node option to avoid rational surfaces
* better plotting for stellarator stuff
* I/O compatibiltiy with VMEC, GS solvers
* command line interface
* documentation
* clean up backend implementation
