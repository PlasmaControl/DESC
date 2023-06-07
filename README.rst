.. image:: https://raw.githubusercontent.com/PlasmaControl/DESC/master/docs/_static/images/logo_med_clear.png

.. inclusion-marker-do-not-remove

################################
Stellarator Optimization Package
################################
|License| |DOI| |Issues| |Pypi|

|Docs| |UnitTests| |RegressionTests| |Codecov|

DESC solves for and optimizes 3D MHD equilibria using pseudo-spectral numerical methods and automatic differentiation.

The theoretical approach and implementation details used by DESC are presented in these papers [1]_ [2]_ [3]_ [4]_ and documented at Theory_.
Please cite our work if you use DESC!

.. [1] Dudt, D. & Kolemen, E. (2020). DESC: A Stellarator Equilibrium Solver. [`Physics of Plasmas <https://aip.scitation.org/doi/full/10.1063/5.0020743>`__]    [`pdf <https://github.com/PlasmaControl/DESC/blob/master/docs/Dudt_Kolemen_PoP_2020.pdf>`__]
.. [2] Panici, D. et al (2022). The DESC Stellarator Code Suite Part I: Quick and accurate equilibria computations. [`arxiv <https://arxiv.org/abs/2203.17173>`__]    [`pdf <https://github.com/PlasmaControl/DESC/blob/c669f5f28bf32042c76597c254d85c92f62f078a/publications/panici2022/panici2022equilibria.pdf>`__]
.. [3] Conlin, R. et al. (2022). The DESC Stellarator Code Suite Part II: Perturbation and continuation methods. [`arxiv <https://arxiv.org/abs/2203.15927>`__]    [`pdf <https://github.com/PlasmaControl/DESC/blob/c669f5f28bf32042c76597c254d85c92f62f078a/publications/conlin2022/conlin2022perturbations.pdf>`__]
.. [4] Dudt, D. et al. (2022). The DESC Stellarator Code Suite Part III: Quasi-symmetry optimization. [`arxiv <https://arxiv.org/abs/2204.00078>`__]    [`pdf <https://github.com/PlasmaControl/DESC/blob/c669f5f28bf32042c76597c254d85c92f62f078a/publications/dudt2022/dudt2022optimization.pdf>`__]
.. _Theory: https://desc-docs.readthedocs.io/en/latest/theory_general.html

Quick Start
===========

.. role:: console(code)
   :language: console

The easiest way to install DESC is from pypi: :console:`pip install desc-opt`

For more detailed instructions on installing DESC and its dependencies, see Installation_.
The code is run using the syntax :console:`desc <path/to/input_file>` and the full list of command line options are given in `Command Line Interface`_. (Note that you may have to prepend the command with :console:`python -m`)
DESC requires an input file to specify the equilibrium and solver options, and can also accept VMEC input files.
Refer to Inputs_ for documentation on how to format the input file.
The equilibrium solution is output in a HDF5 binary file, whose format is detailed in Outputs_.

As an example usage, to use DESC to solve for the equilibrium of the high-beta, D-shaped plasma described with the DSHAPE input file, the command from the :console:`DESC` directory is
:console:`desc -p examples/DESC/DSHAPE`
Where the :console:`-p` flag tells DESC to plot the results once it finishes.

.. _Installation: https://desc-docs.readthedocs.io/en/latest/installation.html
.. _Command Line Interface: https://desc-docs.readthedocs.io/en/latest/command_line.html
.. _Inputs: https://desc-docs.readthedocs.io/en/latest/input.html
.. _Outputs: https://desc-docs.readthedocs.io/en/latest/output.html

Repository Contents
===================

- desc_ contains the source code including the main script and supplemental files. Refer to the API_ documentation for details on all of the available functions and classes.
- docs_ contains the documentation files.
- examples_ contains example input files along with corresponding DESC solutions, which are also accessible using the `desc.examples.get` function.
- tests_ contains routines for automatic testing.

.. _desc: https://github.com/PlasmaControl/DESC/tree/master/desc
.. _docs: https://github.com/PlasmaControl/DESC/tree/master/docs
.. _examples: https://github.com/PlasmaControl/DESC/tree/master/desc/examples
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

.. |UnitTests| image:: https://github.com/PlasmaControl/DESC/actions/workflows/unittest.yml/badge.svg
    :target: https://github.com/PlasmaControl/DESC/actions/workflows/unittest.yml
    :alt: UnitTests

.. |RegressionTests| image:: https://github.com/PlasmaControl/DESC/actions/workflows/regression_test.yml/badge.svg
    :target: https://github.com/PlasmaControl/DESC/actions/workflows/regression_test.yml
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
