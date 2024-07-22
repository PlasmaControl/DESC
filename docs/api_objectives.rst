=================================
Objectives, Optimizers, and Grids
=================================

Objectives and Constraints
**************************
``desc.objectives`` defines a number of different metrics for stellarator optimization
problems, which can be used as either objectives or constraints depending on how they
are passed to the optimizer. Individual objectives are combined into an ``ObjectiveFunction``
that is then passed to the ``Optimizer``.


ObjectiveFunction
-----------------
.. autosummary::
    :toctree: _api/objectives
    :recursive:
    :template: class.rst

    desc.objectives.ObjectiveFunction


Equilibrium
-----------
.. autosummary::
    :toctree: _api/objectives
    :recursive:
    :template: class.rst

    desc.objectives.ForceBalance
    desc.objectives.ForceBalanceAnisotropic
    desc.objectives.Energy
    desc.objectives.CurrentDensity
    desc.objectives.RadialForceBalance
    desc.objectives.HelicalForceBalance


Geometry
--------
.. autosummary::
    :toctree: _api/objectives
    :recursive:
    :template: class.rst

    desc.objectives.AspectRatio
    desc.objectives.Elongation
    desc.objectives.Volume
    desc.objectives.MeanCurvature
    desc.objectives.PrincipalCurvature
    desc.objectives.PlasmaVesselDistance
    desc.objectives.BScaleLength
    desc.objectives.GoodCoordinates


Omnigenity
----------
.. autosummary::
    :toctree: _api/objectives
    :recursive:
    :template: class.rst

    desc.objectives.QuasisymmetryTwoTerm
    desc.objectives.QuasisymmetryTripleProduct
    desc.objectives.QuasisymmetryBoozer
    desc.objectives.Omnigenity
    desc.objectives.Isodynamicity


Stability
---------
.. autosummary::
    :toctree: _api/objectives
    :recursive:
    :template: class.rst

    desc.objectives.MagneticWell
    desc.objectives.MercierStability


Free boundary / Single stage optimization
-----------------------------------------
.. autosummary::
    :toctree: _api/objectives
    :recursive:
    :template: class.rst

    desc.objectives.BoundaryError
    desc.objectives.VacuumBoundaryError


Coil Optimization
-----------------
.. autosummary::
    :toctree: _api/objectives
    :recursive:
    :template: class.rst


    desc.objectives.QuadraticFlux
    desc.objectives.CoilLength
    desc.objectives.CoilCurvature
    desc.objectives.CoilTorsion
    desc.objectives.CoilSetMinDistance
    desc.objectives.PlasmaCoilSetMinDistance
    desc.objectives.CoilCurrentLength
    desc.objectives.ToroidalFlux


Profiles
--------
.. autosummary::
    :toctree: _api/objectives
    :recursive:
    :template: class.rst

    desc.objectives.RotationalTransform
    desc.objectives.Shear
    desc.objectives.ToroidalCurrent
    desc.objectives.Pressure
    desc.objectives.BootstrapRedlConsistency


Fixing degrees of freedom
-------------------------
.. autosummary::
    :toctree: _api/objectives
    :recursive:
    :template: class.rst

    desc.objectives.FixBoundaryR
    desc.objectives.FixBoundaryZ
    desc.objectives.FixAxisR
    desc.objectives.FixAxisZ
    desc.objectives.FixPsi
    desc.objectives.FixPressure
    desc.objectives.FixIota
    desc.objectives.FixCurrent
    desc.objectives.FixAtomicNumber
    desc.objectives.FixElectronDensity
    desc.objectives.FixElectronTemperature
    desc.objectives.FixIonTemperature
    desc.objectives.FixAnisotropy
    desc.objectives.FixModeR
    desc.objectives.FixModeZ
    desc.objectives.FixSumModesR
    desc.objectives.FixSumModesZ
    desc.objectives.FixThetaSFL
    desc.objectives.FixCoilCurrent
    desc.objectives.FixSumCoilCurrent
    desc.objectives.FixParameters


User defined objectives
-----------------------
.. autosummary::
    :toctree: _api/objectives
    :recursive:
    :template: class.rst


    desc.objectives.GenericObjective
    desc.objectives.ObjectiveFromUser
    desc.objectives.LinearObjectiveFromUser


Utilities for getting common groups of constraints
--------------------------------------------------
.. autosummary::
    :toctree: _api/objectives
    :recursive:
    :template: class.rst

    desc.objectives.get_fixed_boundary_constraints
    desc.objectives.get_NAE_constraints
    desc.objectives.get_fixed_axis_constraints
    desc.objectives.get_equilibrium_objective


Optimization
************
``desc.optimize.Optimizer`` is the primary interface, it contains wrappers for a number
of different methods listed in `Optimizers Supported <https://desc-docs.readthedocs.io/en/latest/optimizers.html>`_.

.. autosummary::
   :toctree: _api/optimize
   :recursive:
   :template: class.rst

   desc.optimize.Optimizer

There are also a number of optimizers written specifically for DESC that we also offer
with a direct interface similar to ``scipy.optimize.minimize``:

.. autosummary::
   :toctree: _api/optimize
   :recursive:
   :template: class.rst

   desc.optimize.lsqtr
   desc.optimize.fmintr
   desc.optimize.fmin_auglag
   desc.optimize.lsq_auglag
   desc.optimize.sgd

DESC also allows you to use custom optimizers by creating a wrapper function and
registering it using ``desc.optimize.register_optimizer``. See `Adding optimizers <https://desc-docs.readthedocs.io/en/stable/adding_optimizers.html>`_
for details

.. autosummary::
   :toctree: _api/optimize
   :recursive:
   :template: class.rst

   desc.optimize.register_optimizer


Grids
*****
A grid defines a set of collocation nodes in computational coordinates where physics
quantities are to be evaluated. DESC offers a number of options with different patterns
and spacing. Each objective generally has a default grid that works for most cases, but
often it is desired to specify where particular objectives should be targeted, such as
targeting quasi-symmetry on particular surfaces. For this a user defined grid can be
created and passed to the corresponding objective.

.. autosummary::
    :toctree: _api/grid/
    :recursive:
    :template: class.rst

    desc.grid.Grid
    desc.grid.LinearGrid
    desc.grid.QuadratureGrid
    desc.grid.ConcentricGrid

``desc.grid`` also contains utilities for finding the most and least rational surfaces
for a given iota profile, for either avoiding or analyzing rational surfaces.

.. autosummary::
    :toctree: _api/grid/
    :recursive:
    :template: class.rst

    desc.grid.find_least_rational_surfaces
    desc.grid.find_most_rational_surfaces
