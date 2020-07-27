# DESC
Stellarator Equilibrium Solver

This is a lightweight version of the DESC code in Python. 

The main script is DESC.py, which allows the user to specify the spectral resolution, boundary conditions, pressure and rotational transform profiles, runs the solver, then plots the results.

* boundary_conditions.py - functions for calculating boundary error
* force_balance.py - functions for evaluating co- and contra-variant bases, magnetic fields, and force balance errors
* init_guess.py - functions for generating initial guesses for solution
* utils.py - wrappers for simple operations, plotting, etc
* zernike.py - Zernike/Fourier transforms and basis functions
