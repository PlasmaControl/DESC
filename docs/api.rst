=================
API Documentation
=================

Basis
*****

.. autosummary::
    :toctree: _api/basis/
    :recursive:
    :template: class.rst

    desc.basis.PowerSeries
    desc.basis.FourierSeries
    desc.basis.DoubleFourierSeries
    desc.basis.ZernikePolynomial
    desc.basis.FourierZernikeBasis

Boundary Conditions
*******************

.. autosummary:: 
    :toctree: _api/boundary_conditions/
    :recursive:
    :template: class.rst

    desc.boundary_conditions.LCFSConstraint
    desc.boundary_conditions.PoincareConstraint
    desc.boundary_conditions.UmbilicConstraint

Derivatives
***********
Note that the ``derivative`` module also exposes the ``Derivative`` class, which is an alias for ``AutoDiffDerivative`` if JAX is installed, or ``FiniteDiffDerivative`` if not.

.. autosummary::
    :toctree: _api/derivatives
    :recursive:
    :template: class.rst

    desc.derivatives.AutoDiffDerivative
    desc.derivatives.FiniteDiffDerivative


Equilibrium
***********

.. autosummary:: 
    :toctree: _api/equilibrium
    :recursive:
    :template: class.rst

    desc.equilibrium.Equilibrium
    desc.equilibrium.EquilibriaFamily

Grid
****

.. autosummary::
    :toctree: _api/grid/
    :recursive:
    :template: class.rst

    desc.grid.Grid
    desc.grid.LinearGrid
    desc.grid.QuadratureGrid
    desc.grid.ConcentricGrid

IO
***

.. autosummary::
    :toctree: _api/io/
    :recursive:
    :template: class.rst

    desc.io.InputReader
    desc.io.load

Objective Functions
*******************

.. autosummary::
    :toctree: _api/objective_funs
    :recursive:
    :template: class.rst

    desc.objective_funs.get_objective_function
    desc.objective_funs.ForceErrorNodes
    desc.objective_funs.ForceErrorGalerkin
    desc.objective_funs.EnergyVolIntegral
    desc.objective_funs.QuasisymmetryTripleProduct
    desc.objective_funs.QuasisymmetryFluxFunction
	       
Optimize
********

.. autosummary:: 
   :toctree: _api/optimize
   :recursive:
   :template: class.rst

   desc.optimize.Optimizer
   desc.optimize.fmintr
   desc.optimize.lsqtr

Perturbations
*************

.. autosummary:: 
    :toctree: _api/perturbations
    :recursive:

    desc.perturbations.perturb
    desc.perturbations.optimal_perturb

Plotting
********

.. autosummary:: 
    :toctree: _api/plotting
    :recursive:

    desc.plotting.plot_1d
    desc.plotting.plot_2d
    desc.plotting.plot_3d
    desc.plotting.plot_surfaces
    desc.plotting.plot_section
    desc.plotting.plot_grid
    desc.plotting.plot_basis
    desc.plotting.plot_logo
    desc.plotting.plot_field_lines_sfl

Profiles
********

.. autosummary::
    :toctree: _api/profiles
    :recursive:
    :template: class.rst
	       
    desc.profiles.PowerSeriesProfile
    desc.profiles.SplineProfile
    desc.profiles.MTanhProfile
    
Transform
*********

.. autosummary::
   :toctree: _api/transform/
   :recursive:
   :template: class.rst

   desc.transform.Transform

VMEC
****

.. autosummary:: 
    :toctree: _api/vmec/
    :recursive:
    :template: class.rst

    desc.vmec.VMECIO
