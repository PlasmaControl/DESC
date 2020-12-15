.. image:: docs/_static/images/logo_med_clear.png

.. inclusion-marker-do-not-remove
	   
##############################
Stellarator Equilibrium Solver
##############################
|License| |Docs| |Travis| |Codecov|

DESC computes 3D MHD equilibria by solving the force balance equations.
It can also be used for perturbation analysis and sensitivity studies to see how the equilibria change as input parameters are varied.

The theoretical approach and numerical methods used by DESC are presented in this paper_ [1]_ 
and documented at Theory_. 
Please cite our work if you use DESC! 

.. [1] Dudt, D. & Kolemen, E. (2020). DESC: A Stellarator Equilibrium Solver. *Physics of Plasmas*. 
.. _paper: https://github.com/ddudt/DESC/blob/master/docs/Dudt_Kolemen_PoP_2020.pdf
.. _Theory: https://desc-apc524.readthedocs.io/en/latest/theory_general.html

Quick Start
===========

.. role:: bash(code)
   :language: bash

For instructions on installing DESC and its dependencies, see Installation_. 
The code is run using the syntax :bash:`python -m desc <path/to/input_file>` and the full list of command line options are given in `Command Line Interface`_. 
DESC requires an input file to specify the equilibrium and solver options, and can also accept VMEC input files.
Refer to Inputs_ for documentation on how to format the input file.
The equilibrium solution is output in both an ASCII text file and a HDF5 binary file, whose formats are detailed in Outputs_. 

As an example usage, to use DESC to solve for the equilibrium of the high-beta, D-shaped plasma described with the DSHAPE input file, the command from the :bash:'desc' directory is 
:bash:`python -u -m desc -p examples/DESC/DSHAPE`
Where the :bash:`-u` flag is so Python prints the output of the optimization in real time as opposed to storing in a buffer, and the :bash:`-p` flag tells DESC to plot the results once it finishes.

.. _Installation: https://desc-apc524.readthedocs.io/en/latest/installation.html
.. _Command Line Interface: https://desc-apc524.readthedocs.io/en/latest/command_line.html
.. _Inputs: https://desc-apc524.readthedocs.io/en/latest/input.html
.. _Outputs: https://desc-apc524.readthedocs.io/en/latest/output.html

Repository Contents
===================

- desc_ contains the source code including the main script and supplemental files. Refer to the API_ documentation for details on all of the available functions. 
- docs_ contains the documentation files. 
- examples_ contains example input files along with corresponding VMEC solutions. 
- tests_ contains routines for automatic testing. 

.. _desc: https://github.com/dpanici/DESC/tree/master/desc
.. _docs: https://github.com/dpanici/DESC/tree/master/docs
.. _examples: https://github.com/dpanici/DESC/tree/master/examples
.. _tests: https://github.com/dpanici/DESC/tree/master/tests
.. _API: https://desc-apc524.readthedocs.io/en/latest/api.html

Contribute
==========
 
- `Contributing guidelines <https://github.com/dpanici/DESC/blob/master/CONTRIBUTING.rst>`_
- `Issue Tracker <https://github.com/dpanici/DESC/issues>`_
- `Source Code <https://github.com/dpanici/DESC/>`_
- `Documentation <https://desc-apc524.readthedocs.io/>`_

.. |License| image:: https://img.shields.io/github/license/ddudt/desc?color=blue&logo=open-source-initiative&logoColor=white
    :target: https://github.com/ddudt/DESC/blob/master/LICENSE
    :alt: License

.. |Docs| image:: https://img.shields.io/readthedocs/desc-apc524?logo=Read-the-Docs
    :target: https://desc-apc524.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation

.. |Travis| image:: https://img.shields.io/travis/dpanici/DESC?logo=travis   
    :target: https://travis-ci.org/dpanici/DESC.svg?branch=master
    :alt: Build

.. |Codecov| image:: https://codecov.io/gh/dpanici/DESC/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/dpanici/DESC
    :alt: Coverage





