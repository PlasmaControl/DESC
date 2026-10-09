.. image:: https://raw.githubusercontent.com/PlasmaControl/DESC/master/docs/_static/images/logo_med_clear.png

.. inclusion-marker-do-not-remove

################################
Stellarator Optimization Package
################################
|License| |DOI| |Issues| |Pypi|

|Docs| |UnitTests| |RegressionTests| |Codecov|

DESC solves for and optimizes 3D MHD equilibria using pseudo-spectral numerical methods
and automatic differentiation.

The theoretical approach and implementation details used by DESC are presented in the
following papers and documented at Theory_. Please cite our work if you use DESC!

- Dudt, D. & Kolemen, E. (2020). DESC: A Stellarator Equilibrium Solver.
  [`Physics of Plasmas <https://doi.org/10.1063/5.0020743>`__]
  [`pdf <https://github.com/PlasmaControl/DESC/blob/master/publications/dudt2020/dudt2020desc.pdf>`__]
- Panici, D. et al (2023). The DESC Stellarator Code Suite Part I: Quick and accurate equilibria computations.
  [`Journal of Plasma Physics <https://doi.org/10.1017/S0022377823000272>`__]
  [`pdf <https://github.com/PlasmaControl/DESC/blob/master/publications/panici2022/Panici_DESC_Stellarator_suite_part_I_quick_accurate_equilibria.pdf>`__]
- Conlin, R. et al. (2023). The DESC Stellarator Code Suite Part II: Perturbation and continuation methods.
  [`Journal of Plasma Physics <https://doi.org/10.1017/S0022377823000399>`__]
  [`pdf <https://github.com/PlasmaControl/DESC/blob/master/publications/conlin2022/conlin2022perturbations.pdf>`__]
- Dudt, D. et al. (2023). The DESC Stellarator Code Suite Part III: Quasi-symmetry optimization.
  [`Journal of Plasma Physics <https://doi.org/10.1017/S0022377823000235>`__]
  [`pdf <https://github.com/PlasmaControl/DESC/blob/master/publications/dudt2022/dudt2022optimization.pdf>`__]

A list of papers which feature DESC can be found here_.

.. _Theory: https://desc-docs.readthedocs.io/en/latest/theory_general.html
.. _here: https://desc-docs.readthedocs.io/en/latest/pubs_list.html



Quick Start
===========

The easiest way to install DESC is from PyPI: ``pip install desc-opt``

For more detailed instructions on installing DESC and its dependencies, see Installation_.

The best place to start learning about DESC is our tutorials:

- `Basic fixed boundary equilibrium`_: running from a VMEC input, creating an equilibrium from scratch
- `Advanced equilibrium`_: continuation and perturbation methods.
- `Free boundary equilibrium`_: vacuum and or finite beta with external field.
- `Using DESC outputs`_: analysis, plotting, saving to VMEC format.
- `Basic optimization`_: specifying objectives, fixing degrees of freedom.
- `Advanced optimization`_: advanced constraints, precise quasi-symmetry, constrained optimization.
- `Near axis constraints`_: loading solutions from QSC/QIC and fixing near axis expansion.
- `Coil optimization`_: "second stage" optimization of magnetic coils.

For details on the various objectives, constraints, optimizable objects and more, see
the full `api documentation`_.

If all you need is an equilibrium solution, the simplest method is through the command
line by giving an input file specifying the equilibrium and solver options, this
way can also can also accept VMEC input files.

The code is run using the syntax ``desc <path/to/inputfile>`` and the full list
of command line options are given in `Command Line Interface`_. (Note that you may have
to prepend the command with ``python -m``)

Refer to `Inputs`_ for documentation on how to format the input file.

The equilibrium solution is output in a HDF5 binary file, whose format is detailed in `Outputs`_.

.. _Installation: https://desc-docs.readthedocs.io/en/latest/installation.html
.. _Command Line Interface: https://desc-docs.readthedocs.io/en/latest/command_line.html
.. _Inputs: https://desc-docs.readthedocs.io/en/latest/input.html
.. _Outputs: https://desc-docs.readthedocs.io/en/latest/output.html
.. _Basic fixed boundary equilibrium: https://desc-docs.readthedocs.io/en/latest/notebooks/tutorials/basic_equilibrium.html
.. _Advanced equilibrium: https://desc-docs.readthedocs.io/en/latest/notebooks/tutorials/advanced_equilibrium_continuation.html
.. _Free boundary equilibrium: https://desc-docs.readthedocs.io/en/latest/notebooks/tutorials/free_boundary_equilibrium.html
.. _Using DESC outputs: https://desc-docs.readthedocs.io/en/latest/notebooks/tutorials/use_outputs.html
.. _Basic optimization: https://desc-docs.readthedocs.io/en/latest/notebooks/tutorials/basic_optimization.html
.. _Advanced optimization: https://desc-docs.readthedocs.io/en/latest/notebooks/tutorials/advanced_optimization.html
.. _Near axis constraints: https://desc-docs.readthedocs.io/en/latest/notebooks/tutorials/nae_constraint.html
.. _Coil optimization: https://desc-docs.readthedocs.io/en/latest/notebooks/tutorials/coil_stage_two_optimization.html
.. _api documentation: https://desc-docs.readthedocs.io/en/latest/api.html

Repository Contents
===================

- desc_ contains the source code including the main script and supplemental files. Refer to the API_ documentation for details on all of the available functions and classes.
- docs_ contains the documentation files.
- tests_ contains routines for automatic testing.
- publications_ contains PDFs of publications by the DESC group, as well as scripts and data to reproduce the results of these papers.

.. _desc: https://github.com/PlasmaControl/DESC/tree/master/desc
.. _docs: https://github.com/PlasmaControl/DESC/tree/master/docs
.. _tests: https://github.com/PlasmaControl/DESC/tree/master/tests
.. _publications: https://github.com/PlasmaControl/DESC/tree/master/publications
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

.. |UnitTests| image:: https://github.com/PlasmaControl/DESC/actions/workflows/unit_tests.yml/badge.svg
    :target: https://github.com/PlasmaControl/DESC/actions/workflows/unit_tests.yml
    :alt: UnitTests

.. |RegressionTests| image:: https://github.com/PlasmaControl/DESC/actions/workflows/regression_tests.yml/badge.svg
    :target: https://github.com/PlasmaControl/DESC/actions/workflows/regression_tests.yml
    :alt: RegressionTests

.. |Codecov| image:: https://codecov.io/gh/PlasmaControl/DESC/branch/master/graph/badge.svg?token=5LDR4B1O7Z
    :target: https://codecov.io/github/PlasmaControl/DESC
    :alt: Coverage

.. |Issues| image:: https://img.shields.io/github/issues/PlasmaControl/DESC
    :target: https://github.com/PlasmaControl/DESC/issues
    :alt: GitHub issues

.. |Pypi| image:: https://img.shields.io/pypi/v/desc-opt
    :target: https://pypi.org/project/desc-opt/
    :alt: Pypi
