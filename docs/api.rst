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

Geometry
********

.. autosummary::
   :toctree: _api/geometry/
   :recursive:
   :template: class.rst

    desc.geometry.FourierRZCurve
    desc.geometry.FourierXYZCurve
    desc.geometry.FourierPlanarCurve
    desc.geometry.FourierRZToroidalSurface
    desc.geometry.ZernikeRZToroidalSection
    
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
    :toctree: _api/objectives
    :recursive:
    :template: class.rst

    desc.objectives.AspectRatio
    desc.objectives.CurrentDensity
    desc.objectives.Energy
    desc.objectives.FixBoundaryR
    desc.objectives.FixBoundaryZ
    desc.objectives.FixIota    
    desc.objectives.FixPressure
    desc.objectives.FixPsi
    desc.objectives.ForceBalance
    desc.objectives.GenericObjective    
    desc.objectives.get_fixed_boundary_constraints
    desc.objectives.get_equilibrium_objective
    desc.objectives.HelicalForceBalance
    desc.objectives.LambdaGauge
    desc.objectives.ObjectiveFunction
    desc.objectives.PoincareLambda    
    desc.objectives.QuasisymmetryBoozer
    desc.objectives.QuasisymmetryTwoTerm
    desc.objectives.QuasisymmetryTripleProduct
    desc.objectives.RadialForceBalance
    desc.objectives.TargetIota
    desc.objectives.ToroidalCurrent
    desc.objectives.Volume


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
    desc.plotting.plot_basis
    desc.plotting.plot_boozer_modes
    desc.plotting.plot_boozer_surface
    desc.plotting.plot_coefficients	   
    desc.plotting.plot_comparison
    desc.plotting.plot_fsa
    desc.plotting.plot_grid
    desc.plotting.plot_logo
    desc.plotting.plot_qs_error
    desc.plotting.plot_section
    desc.plotting.plot_surfaces

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
