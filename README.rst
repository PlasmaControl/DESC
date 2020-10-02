###################################
DESC Stellarator Equilibrium Solver
###################################
|License| |Docs| |Travis| |Codecov|

DESC computes 3D MHD equilibria by solving the force balance equations.
It can also be used for perturbation analysis and sensitivity studies to see how the equilibria change as input parameters are varied.

The theoretical approach and numerical methods used by DESC are presented in this paper_ [1]_ 
and documented at Theory_. 
Please cite our work if you use DESC! 

.. [1] Dudt, D. & Kolemen, E. (2020). DESC: A Stellarator Equilibrium Solver. *Physics of Plasmas*. 
.. _paper: https://github.com/ddudt/DESC/blob/master/docs/Dudt_Kolemen_PoP_2020.pdf
.. _Theory: https://desc-docs.readthedocs.io/en/latest/theory.html

Quick Start
===========

.. role:: bash(code)
   :language: bash

For instructions on installing DESC and its dependencies, see Installation_. 
DESC requires an input file to specify the equilibrium and solver options, and can also accept VMEC input files. 
Refer to Inputs_ for documentation on how to format the input file. 
The code is run using the syntax :bash:`python desc/DESC.py <path/to/input_file>` and the full list of command line options are given in Usage_. 
The equilibrium solution is output in both an ASCII text file and a HDF5 binary file, whose formats are detailed in Outputs_. 

.. _Installation: https://desc-docs.readthedocs.io/en/latest/installation.html
.. _Inputs: https://desc-docs.readthedocs.io/en/latest/input.html
.. _Usage: https://desc-docs.readthedocs.io/en/latest/usage.html
.. _Outputs: https://desc-docs.readthedocs.io/en/latest/output.html

Repository Contents
===================

- desc_ contains the source code including the main script ``DESC.py`` and supplemental files. 
Refer to the API_ documentation for details on all of the available functions. 
- docs_ contains the documentation files. 
- examples contains example input files along with corresponding VMEC solutions. 
- tests_ contains routines for automatic testing. 

.. _desc: https://github.com/ddudt/DESC/tree/master/desc
.. _docs: https://github.com/ddudt/DESC/tree/master/docs
.. _examples: https://github.com/ddudt/DESC/tree/master/examples
.. _tests: https://github.com/ddudt/DESC/tree/master/tests
.. _API: https://desc-docs.readthedocs.io/en/latest/api.html

Contribute
==========
- Contributing guidelines: `<https://github.com/ddudt/DESC/blob/master/CONTRIBUTING.md>`_
- Issue Tracker: `<https://github.com/ddudt/DESC/issues>`_
- Source Code: `<https://github.com/ddudt/DESC/>`_
- Documentation: `<https://desc-docs.readthedocs.io/>`_

.. |License| image:: https://img.shields.io/github/license/ddudt/DESC
    :target: https://github.com/ddudt/DESC/blob/master/LICENSE
    :alt: License

.. |Docs| image:: https://readthedocs.org/projects/desc-docs/badge/?version=latest
    :target: https://desc-docs.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation

.. |Travis| image:: https://travis-ci.org/ddudt/DESC.svg?branch=master
    :target: https://travis-ci.org/ddudt/DESC
    :alt: Build

.. |Codecov| image:: https://codecov.io/gh/ddudt/DESC/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/ddudt/DESC
    :alt: Coverage
