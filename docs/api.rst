====================
Alphabetical Listing
====================

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
    desc.basis.ChebyshevDoubleFourierBasis
    desc.basis.FourierZernikeBasis

Coils
*****

.. autosummary::
    :toctree: _api/coils/
    :recursive:
    :template: class.rst

    desc.coils.CoilSet
    desc.coils.FourierRZCoil
    desc.coils.FourierXYZCoil
    desc.coils.FourierPlanarCoil
    desc.coils.MixedCoilSet
    desc.coils.SplineXYZCoil

Compatibility
*************

.. autosummary::
    :toctree: _api/compat
    :recursive:

    desc.compat.ensure_positive_jacobian
    desc.compat.flip_helicity
    desc.compat.rescale

Continuation
************
.. autosummary::
    :toctree: _api/continuation
    :recursive:

    desc.continuation.solve_continuation
    desc.continuation.solve_continuation_automatic

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

Examples
********

.. autosummary::
    :toctree: _api/examples
    :recursive:

    desc.examples.get
    desc.examples.listall

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
    desc.geometry.SplineXYZCurve
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
    desc.grid.find_least_rational_surfaces
    desc.grid.find_most_rational_surfaces

IO
***

.. autosummary::
    :toctree: _api/io/
    :recursive:
    :template: class.rst

    desc.io.load

Magnetic Fields
***************

.. autosummary::
    :toctree: _api/magnetic_fields
    :recursive:
    :template: class.rst

    desc.magnetic_fields.CurrentPotentialField
    desc.magnetic_fields.FourierCurrentPotentialField
    desc.magnetic_fields.DommaschkPotentialField
    desc.magnetic_fields.OmnigenousField
    desc.magnetic_fields.PoloidalMagneticField
    desc.magnetic_fields.ScalarPotentialField
    desc.magnetic_fields.ScaledMagneticField
    desc.magnetic_fields.SplineMagneticField
    desc.magnetic_fields.SumMagneticField
    desc.magnetic_fields.ToroidalMagneticField
    desc.magnetic_fields.VerticalMagneticField
    desc.magnetic_fields.field_line_integrate
    desc.magnetic_fields.read_BNORM_file

Objective Functions
*******************

.. autosummary::
    :toctree: _api/objectives
    :recursive:
    :template: class.rst

    desc.objectives.AspectRatio
    desc.objectives.BootstrapRedlConsistency
    desc.objectives.BoundaryError
    desc.objectives.BScaleLength
    desc.objectives.CurrentDensity
    desc.objectives.Elongation
    desc.objectives.Energy
    desc.objectives.FixAtomicNumber
    desc.objectives.FixAxisR
    desc.objectives.FixAxisZ
    desc.objectives.FixBoundaryR
    desc.objectives.FixBoundaryZ
    desc.objectives.FixCurrent
    desc.objectives.FixElectronDensity
    desc.objectives.FixElectronTemperature
    desc.objectives.FixIonTemperature
    desc.objectives.FixIota
    desc.objectives.FixModeR
    desc.objectives.FixModeZ
    desc.objectives.FixOmniBmax
    desc.objectives.FixOmniMap
    desc.objectives.FixOmniWell
    desc.objectives.FixParameter
    desc.objectives.FixPressure
    desc.objectives.FixPsi
    desc.objectives.FixSumModesR
    desc.objectives.FixSumModesZ
    desc.objectives.FixThetaSFL
    desc.objectives.ForceBalance
    desc.objectives.ForceBalanceAnisotropic
    desc.objectives.GenericObjective
    desc.objectives.get_equilibrium_objective
    desc.objectives.get_fixed_axis_constraints
    desc.objectives.get_fixed_boundary_constraints
    desc.objectives.get_NAE_constraints
    desc.objectives.GoodCoordinates
    desc.objectives.HelicalForceBalance
    desc.objectives.Isodynamicity
    desc.objectives.LinearObjectiveFromUser
    desc.objectives.MagneticWell
    desc.objectives.MeanCurvature
    desc.objectives.MercierStability
    desc.objectives.ObjectiveFromUser
    desc.objectives.ObjectiveFunction
    desc.objectives.Omnigenity
    desc.objectives.PlasmaVesselDistance
    desc.objectives.Pressure
    desc.objectives.PrincipalCurvature
    desc.objectives.QuasisymmetryBoozer
    desc.objectives.QuasisymmetryTwoTerm
    desc.objectives.QuasisymmetryTripleProduct
    desc.objectives.RadialForceBalance
    desc.objectives.RotationalTransform
    desc.objectives.Shear
    desc.objectives.ToroidalCurrent
    desc.objectives.VacuumBoundaryError
    desc.objectives.Volume

Optimize
********

.. autosummary::
   :toctree: _api/optimize
   :recursive:
   :template: class.rst

   desc.optimize.Optimizer
   desc.optimize.fmin_auglag
   desc.optimize.fmintr
   desc.optimize.lsq_auglag
   desc.optimize.lsqtr
   desc.optimize.register_optimizer
   desc.optimize.sgd

Perturbations
*************

.. autosummary::
    :toctree: _api/perturbations
    :recursive:

    desc.perturbations.get_deltas
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
    desc.plotting.plot_boundaries
    desc.plotting.plot_boundary
    desc.plotting.plot_coefficients
    desc.plotting.plot_coils
    desc.plotting.plot_comparison
    desc.plotting.plot_field_lines_sfl
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
    desc.profiles.ScaledProfile
    desc.profiles.SumProfile
    desc.profiles.ProductProfile

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
