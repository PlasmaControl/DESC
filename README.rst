###################################
DESC Stellarator Equilibrium Solver
###################################
|License|

This is the primary version of the DESC code in Python.

Repository Contents
===================
Python files
************
``DESC.py`` is the main script which computes and plots equilibrium solutions.
The following files contain supplemental functions:

- ``backend.py`` - set of core functions and jax/numpy compatibility layer
- ``boundary_conditions.py`` - functions for calculating boundary errors
- ``continuation.py`` - calls the optimization routine and functions for perturbing solutions
- ``field_components.py`` - functions for calculating B and J components
- ``gfile_helpers.py`` - functions for reading/writing gfiles for tokamak GS equilibria
- ``init_guess.py`` - functions for generating initial guesses for the solution
- ``input_output.py`` - functions for reading/writing input/output
- ``nodes.py`` - functions for generating collocation nodes
- ``objective_funs.py`` - assembles the objective functions to be minimized
- ``plotting.py`` - functions for plotting solutions and their errors
- ``zernike.py`` - Zernike/Fourier transforms and basis functions

Benchmarks
**********
Sample VMEC input files and their corresponding wout files are contained in `benchmarks/VMEC <https://github.com/ddudt/DESC/tree/master/benchmarks/VMEC>`_.
The equivalent DESC input files are contained in `benchmarks/DESC <https://github.com/ddudt/DESC/tree/master/benchmarks/DESC>`_.

Contribute
==========

- Issue Tracker: `<https://github.com/ddudt/DESC/issues>`_
- Source Code: `<https://github.com/ddudt/DESC/>`_
- Documentation: `<https://desc-docs.readthedocs.io/>`_



.. |License| image:: https://img.shields.io/github/license/ddudt/DESC
    :target: https://github.com/ddudt/DESC/blob/master/LICENSE
    :alt: License: MIT
