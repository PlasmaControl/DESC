# DESC
Stellarator Equilibrium Solver

This is the primary version of the DESC code in Python.

## Contents

### Python files

DESC.py is the main script which calls the following supplemental scripts:
- backend.py - set of core functions and jax/numpy compatibility layer
- boundary_conditions.py - functions for calculating boundary errors
- field_components.py - functions for calculating B and J components
- gfile_helpers.py - functions for reading/writing gfiles for tokamak GS equilibria
- init_guess.py - functions for generating initial guesses for the solution
- input_output.py - functions for reading/writing input/output
- nodes.py - functions for generating collocation nodes
- objective_funs.py - assembles the objective functions to be minimized
- plotting.py - routines for plotting solutions and their errors
- zernike.py - Zernike/Fourier transforms and basis functions

### Benchmarks

Sample VMEC input files are contained in [benchmarks/VMEC/inputs](https://github.com/ddudt/DESC/tree/python/benchmarks/VMEC/inputs), 
and their corresponding wout files are in [benchmarks/VMEC/outputs](https://github.com/ddudt/DESC/tree/python/benchmarks/VMEC/outputs).
The equivalent DESC input files are contained in [benchmarks/DESC/inputs](https://github.com/ddudt/DESC/tree/python/benchmarks/DESC/inputs), 
and the solution outputs are saved in [benchmarks/DESC/outputs](https://github.com/ddudt/DESC/tree/python/benchmarks/DESC/outputs).

### Documentation

Additional documentation on specific parts of the code can be found in [documentation](https://github.com/ddudt/DESC/tree/python/documentation).

## TODO List

### Priorities

- continuation method
- memory management for gpu
- what needs jit, which devices
- precompute SVD for fitting
- force balance
    - project back to get `R_tt`,`Z_tt`
    - incompressibility constraint
- symmetry, symmetric nodes, best way to enforce and reduce size
    - masking
- avoiding recompilation
    - masking?
    - comparing masked arrays for bdry
- profiling to find speed bottlenecks

### Known Issues

- vmec input conversion cannot handle multi-line inputs
- `bdry_ratio` does not work with `compute_bc_err_RZ` function

### Features to Add

- figure out asymptotics for contravariant basis at axis
- zernike transform using FFT
- QS function specifying M/N
- allow option for square system
- use `root` algorithms instead of `least_squares` (and compare results)
- autodiff hessian
- add node option to avoid rational surfaces
- I/O compatibiltiy with VMEC, GS solvers
- command line interface
- documentation
- clean up backend implementation
