.. image:: https://raw.githubusercontent.com/PlasmaControl/DESC/master/docs/_static/images/logo_med_clear.png

.. inclusion-marker-do-not-remove

##############################
Stellarator Equilibrium Solver
##############################
|License| |DOI| |Docs| |Tests| |Codecov|
|Issues| |Pypi|

DESC computes 3D MHD equilibria by solving the force balance equations.
It can also be used for perturbation analysis and sensitivity studies to see how the equilibria change as input parameters are varied.

The theoretical approach and numerical methods used by DESC are presented in this paper_ [1]_
and documented at Theory_.
Please cite our work if you use DESC!

.. [1] Dudt, D. & Kolemen, E. (2020). DESC: A Stellarator Equilibrium Solver. *Physics of Plasmas*.
.. _paper: https://github.com/PlasmaControl/DESC/blob/master/docs/Dudt_Kolemen_PoP_2020.pdf
.. _Theory: https://desc-docs.readthedocs.io/en/latest/theory_general.html

Quick Start
===========

.. role:: bash(code)
   :language: bash

The easiest way to install DESC is from pypi: :bash:`pip install desc-opt`

For more detailed instructions on installing DESC and its dependencies, see Installation_.
The code is run using the syntax :bash:`desc <path/to/input_file>` and the full list of command line options are given in `Command Line Interface`_. (Note that if you may have to prepend the command with :bash:`python -m`)
DESC requires an input file to specify the equilibrium and solver options, and can also accept VMEC input files.
Refer to Inputs_ for documentation on how to format the input file.
The equilibrium solution is output in a HDF5 binary file, whose format is detailed in Outputs_.

As an example usage, to use DESC to solve for the equilibrium of the high-beta, D-shaped plasma described with the DSHAPE input file, the command from the :bash:`DESC` directory is
:bash:`desc -p examples/DESC/DSHAPE`
Where the :bash:`-p` flag tells DESC to plot the results once it finishes.

.. _Installation: https://desc-docs.readthedocs.io/en/latest/installation.html
.. _Command Line Interface: https://desc-docs.readthedocs.io/en/latest/command_line.html
.. _Inputs: https://desc-docs.readthedocs.io/en/latest/input.html
.. _Outputs: https://desc-docs.readthedocs.io/en/latest/output.html

Repository Contents
===================

- desc_ contains the source code including the main script and supplemental files. Refer to the API_ documentation for details on all of the available functions and classes.
- docs_ contains the documentation files.
- examples_ contains example input files along with corresponding VMEC solutions.
- tests_ contains routines for automatic testing.

.. _desc: https://github.com/PlasmaControl/DESC/tree/master/desc
.. _docs: https://github.com/PlasmaControl/DESC/tree/master/docs
.. _examples: https://github.com/PlasmaControl/DESC/tree/master/examples
.. _tests: https://github.com/PlasmaControl/DESC/tree/master/tests
.. _API: https://desc-docs.readthedocs.io/en/latest/api.html

Contribute
==========
 
- `Contributing guidelines <https://github.com/PlasmaControl/DESC/blob/master/CONTRIBUTING.rst>`_
- `Issue Tracker <https://github.com/PlasmaControl/DESC/issues>`_
- `Source Code <https://github.com/PlasmaControl/DESC/>`_
- `Documentation <https://desc-docs.readthedocs.io/>`_

.. |License| image:: https://img.shields.io/github/license/PlasmaControl/desc?color=blue&logo=open-source-initiative&logoColor=white
    :target: https://github.com/PlasmaControl/DESC/blob/master/LICENSE
    :alt: License

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4876504.svg
   :target: https://doi.org/10.5281/zenodo.4876504 
   :alt: DOI
   
.. |Docs| image:: https://img.shields.io/readthedocs/desc-docs?logo=Read-the-Docs
    :target: https://desc-docs.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation

.. |Tests| image:: https://github.com/PlasmaControl/DESC/actions/workflows/pytest.yml/badge.svg
    :target: https://github.com/PlasmaControl/DESC/actions/workflows/pytest.yml
    :alt: Tests

.. |Codecov| image:: https://codecov.io/gh/PlasmaControl/DESC/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/PlasmaControl/DESC
    :alt: Coverage

.. |Issues| image:: https://img.shields.io/github/issues/PlasmaControl/DESC
    :target: https://github.com/PlasmaControl/DESC/issues
    :alt: GitHub issues

.. |Pypi| image:: https://img.shields.io/pypi/v/desc-opt
    :target: https://pypi.org/project/desc-opt/
    :alt: Pypi
