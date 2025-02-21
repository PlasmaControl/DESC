Changelog
=========

New Features

- Enables tracking multiple fieldlines in ``Bounce2D``.
- Bounce integral methods with ``desc.integrals.Bounce2D``.
- Effective ripple ``desc.objectives.EffectiveRipple`` and Gamma_c ``desc.objectives.Gamma_c`` optimization objectives.
- See GitHub pull requests [#1003](https://github.com/PlasmaControl/DESC/pull/1003), [#1042](https://github.com/PlasmaControl/DESC/pull/1042), [#1119](https://github.com/PlasmaControl/DESC/pull/1119), and [#1290](https://github.com/PlasmaControl/DESC/pull/1290) for more details.
- Many new compute quantities for partial derivatives in different coordinate systems.
- Adds a new profile class ``PowerProfile`` for raising profiles to a power.
- Adds ``desc.objectives.LinkingCurrentConsistency`` for ensuring that coils in a stage 2 or single stage optimization provide the required linking current for a given equilibrium.
- Adds an option ``scaled_termination`` (defaults to True) to all the desc optimizers to measure the norms for ``xtol`` and ``gtol`` in the scaled norm provided by ``x_scale`` (which defaults to using an adaptive scaling based on the Jacobian or Hessian). This should make things more robust when optimizing parameters with widely different magnitudes. The old behavior can be recovered by passing ``options={"scaled_termination": False}``.
- ``desc.objectives.Omnigenity`` is now vectorized and able to optimize multiple surfaces at the same time. Previously it was required to use a different objective for each surface.
- Adds a new objective ``desc.objectives.MirrorRatio`` for targeting a particular mirror ratio on each flux surface, for either an ``Equilibrium`` or ``OmnigenousField``.
- Adds the output quantities ``wb`` and ``wp`` to ``VMECIO.save``.
- Changes implementation of Dommaschk potentials to use recursive algorithm and symbolic integration.
- Changes hessian computation to use chunked ``jacfwd`` and ``jacrev``, allowing ``jac_chunk_size`` to now reduce hessian memory usage as well.
- Adds an option to ``VMECIO.save`` to specify the grid resolution in real space.
- Adds a new objective ``desc.objectives.CoilIntegratedCurvature`` for targeting convex coils.
- `eq.solve` and `eq.perturb` now accept `LinearConstraintProjection` as objective. This option must be used without any constraints.
- Adds the example "reactor_QA", which is similar to "precise_QA" but with self-consistent bootstrap current at finite beta.
- Allows non-proximal optimizers to  handle optimizing more than one ``Equilibrium`` object simultaneously.
- Adds batching feature to singular integrals.
- ``desc.objectives.CoilSetMinDistance`` and ``desc.objectives.PlasmaCoilSetMinDistance`` now include the option to use a softmin which can give smoother gradients. They also both now have a ``dist_chunk_size`` option to break up the distance calculation into smaller pieces to save memory
- Adds a new function ``desc.coils.initialize_helical_coils`` for creating an initial guess for stage 2 helical coil optimization.
- Adds ``desc.vmec_utils.make_boozmn_output `` for writing boozmn.nc style output files
for compatibility with other codes which expect such files from the Booz_Xform code.
- Renames compute quantity ``sqrt(g)_B`` to ``sqrt(g)_Boozer_DESC`` to more accurately reflect what the quantiy is (the jacobian from (rho,theta_B,zeta_B) to (rho,theta,zeta)), and adds a new function to compute ``sqrt(g)_Boozer`` which is the jacobian from (rho,theta_B,zeta_B) to (R,phi,Z).
- Allows specification of Nyquist spectrum maximum modenumbers when using ``VMECIO.save`` to save a DESC .h5 file as a VMEC-format wout file
- Adds a new objective ``desc.objectives.ExternalObjective`` for wrapping external codes with finite differences.
- DESC/JAX version and device info is no longer printed by default, but can be accessed with the function `desc.backend.print_backend_info()`.

Speed Improvements

- A number of minor improvements to basis function evaluation and spectral transforms to improve speed. These will also enable future improvements for larger gains.

Bug Fixes

- Small bug fix to use the correct normalization length ``a`` in the BallooningStability objective.
- Fixes I/O bug when saving/loading ``_Profile`` classes that do not have a ``_params`` attribute.
- Minor bugs described in [#1323](https://github.com/PlasmaControl/DESC/pull/1323).
- Corrects basis vectors computations made on surface objects [#1175](https://github.com/PlasmaControl/DESC/pull/1175).
- Allows keyword arguments to be passed to ``GenericObjective`` and ``ObjectiveFromUser``.
- Fixes bug where ``save_in_makegrid_format`` function did not correctly account for ``CoilSet`` objects with NFP>1 or sym=True attributes, and so would not save all the coils.
- Fixes issue with interpolator for singular integrals [#1522](https://github.com/PlasmaControl/DESC/issues/1522) and additional checks [1519](https://github.com/PlasmaControl/DESC/issues/1519).
- Fixes the coil currents in ``desc.coils.initialize_modular_coils`` to now give the correct expected linking current.
- ``desc.objectives.PlasmaVesselDistance`` now correctly accounts for multiple field periods on both the equilibrium and the vessel surface. Previously it only considered distances within a single field period.
- Sets ``os.environ["JAX_PLATFORMS"] = "cpu"`` instead of ``os.environ["JAX_PLATFORM_NAME"] = "cpu"`` when doing ``set_device("cpu")``.
- Fixes bug in ``desc.input_reader.desc_output_to_input`` utility function for asymmetric equilibria profiles, where the full profile resolution was not being saved.

Performance Improvements

- `proximal-` optimizers use a single `LinearConstraintProjection` and this makes the optimization faster for high resolution cases where taking the SVD (for null-space and inverse) of constraint matrix takes significant time.

v0.13.0
-------

New Features

- Adds ``from_input_file`` method to ``Equilibrium`` class to generate an ``Equilibrium`` object with boundary, profiles, resolution and flux specified in a given DESC or VMEC input file
- Adds function ``solve_regularized_surface_current`` to ``desc.magnetic_fields`` module that implements the REGCOIL algorithm (Landreman, (2017)) for surface current normal field optimization
    * Can specify the tuple ``current_helicity=(M_coil, N_coil)`` to determine if resulting contours correspond to helical topology (both ``(M_coil, N_coil)`` not equal to 0), modular (``N_coil`` equal to 0 and ``M_coil`` nonzero) or windowpane/saddle (``M_coil`` and ``N_coil`` both zero)
    * ``M_coil`` is the number of poloidal transits a coil makes before returning to itself, while ``N_coil`` is the number of toroidal transits a coil makes before returning to itself (this is sort of like the QS ``helicity``)
    * if multiple values of the regularization parameter are input, will return a family of surface current fields (as a list) corresponding to the solution at each regularization value
- Adds method ``to_CoilSet`` to ``FourierCurrentPotentialField`` which implements a coil cutting algorithm to discretize the surface current into coils
    * works for both modular and helical coils
- Adds a new objective ``SurfaceCurrentRegularization`` (which minimizes ``w*|K|``, the regularization term from surface current in the REGCOIL algorithm, with `w` being the objective weight which act as the regularization parameter)
    * use of both this and the ``QuadraticFlux`` objective allows for REGCOIL solutions to be obtained through the optimization framework, and combined with other objectives as well.
- Changes local area weighting of Bn in QuadraticFlux objective to be the square root of the local area element (Note that any existing optimizations using this objective may need different weights to achieve the same result now.)
- Adds a new tutorial showing how to use``REGCOIL`` features.
- Adds an ``NFP`` attribute to ``ScalarPotentialField``, ``VectorPotentialField`` and ``DommaschkPotentialField``, to allow ``SplineMagneticField.from_field`` and ``MagneticField.save_mgrid`` to efficiently take advantage of the discrete toroidal symmetry of these fields, if present.
- Adds ``SurfaceQuadraticFlux`` objective which minimizes the quadratic magnetic flux through a ``FourierRZToroidalSurface`` object, allowing for optimizing for Quadratic flux minimizing (QFM) surfaces.
- Allows ``ToroidalFlux`` objective to accept ``FourierRZToroidalSurface`` so it can be used to specify the toroidal flux through a QFM surface.
- Adds ``eq_fixed`` flag to ``ToroidalFlux`` to allow for the equilibrium/QFM surface to vary during optimization, useful for single-stage optimizations.
- Adds tutorial notebook showcasing QFM surface capability.
- Add ``desc.coils.initialize_modular_coils`` and ``desc.coils.initialize_saddle_coils`` for creating an initial guess for stage 2 optimization.
- Adds ``rotate_zeta`` function to ``desc.compat`` to rotate an ``Equilibrium`` around Z axis.


Bug Fixes

- Fixes bug that occurs when taking the gradient of ``root`` and ``root_scalar`` with newer versions of JAX (>=0.4.34) and unpins the JAX version.
- Changes ``FixLambdaGauge`` constraint to now enforce zero flux surface average for lambda, instead of enforcing lambda(rho,0,0)=0 as it was incorrectly doing before.
- Fixes bug in ``softmin/softmax`` implementation.
- Fixes bug that occured when using ``ProximalProjection`` with a scalar optimization algorithm.

v0.12.3
-------

[Github Commits](https://github.com/PlasmaControl/DESC/compare/v0.12.2...v0.12.3)

New Features

- Add infinite-n ideal-ballooning stability solver implemented as a part of the ``BallooningStability`` Objective. DESC can use reverse-mode AD to now optimize equilibria against infinite-n ideal ballooning modes.
- Add ``jac_chunk_size`` to ``ObjectiveFunction`` and ``_Objective`` to control the above chunk size for the ``fwd`` mode Jacobian calculation
  - if ``None``, the chunk size is equal to ``dim_x``, so no chunking is done
  - if an ``int``, this is the chunk size to be used.
  - if ``"auto"`` for the ``ObjectiveFunction``, will use a heuristic for the maximum ``jac_chunk_size`` needed to fit the jacobian calculation on the available device memory, according to the formula: ``max_jac_chunk_size = (desc_config.get("avail_mem") / estimated_memory_usage - 0.22)  / 0.85  * self.dim_x`` with ``estimated_memory_usage = 2.4e-7 * self.dim_f * self.dim_x + 1``
- the ``ObjectiveFunction`` ``jac_chunk_size`` is used if ``deriv_mode="batched"``, and the ``_Objective`` ``jac_chunk_size`` will be used if ``deriv_mode="blocked"``
- Make naming of grids kwargs among free boundary objectives more uniform
- Add kwarg options to plot 3d without any axis visible
- Pin jax version temporarily to avoid JAX-related bug

Bug Fixes

- Fix error that can occur when `get_NAE_constraints` is called for only fixing the axis
- Bug fix for `most_rational` with negative arguments
- Fix bug in `FixOmniBMax`

Deprecations

- ``deriv_mode="looped"`` in ``ObjectiveFunction`` is deprecated and will be removed in a future version in favored of ``deriv_mode="batched"`` with ``jac_chunk_size=1``,




v0.12.2
-------

[Github Commits](https://github.com/PlasmaControl/DESC/compare/v0.12.1...v0.12.2)

- Add Vector Potential Calculation to `Coil` classes and Most `MagneticField` Classes
- Add automatic intersection checking to `CoilSet` objects, and a method `is_self_intersecting` which check if the coils in the `CoilSet` intersect one another.
- Add `flip_theta` compatibility function to switch the zero-point of the poloidal angle between the inboard/outboard side of the plasma.
- Change field line integration to use `diffrax` package instead of the deprecated `jax.experimental.odeint` function, allowing for specifying the integration method, the step-size used, and more. See the documentation of [`field_line_integrate`](https://desc-docs.readthedocs.io/en/latest/_api/magnetic_fields/desc.magnetic_fields.field_line_integrate.html#desc.magnetic_fields.field_line_integrate) and [`diffrax`](https://docs.kidger.site/diffrax/api/diffeqsolve/) for more details.
- Add `use_signed_distance` keyword to `PlasmaVesselDistance` objective to allow for specifying the desired relative position of the plasma and surface.
- Vectorize Boozer transform over multiple surfaces, to allow for calculation of Boozer-related quantities on grids that contain multiple radial surfaces.
- Optimizer now automatically scales linearly-constrained optimization parameters to be of roughly the same magnitude, to improve optimization when parameter values range many orders of magnitude
- Add `HermiteSplineProfile` class, which allows for profile derivative information to be specified along with profile value information.
- Add installation instructions for RAVEN cluster at IPP to the docs
- Change optimizer printed output to be easier to read
- Add `HeatingPower` and `FusionPower` objectives
- Reduce `QuadratureGrid` number of radial points to match its intended functionality
- Fix some plotting issues that arose when NFP differs from 1 for objects, or when passed-in phi exceeds 2pi/nfp
- Update `VMECIO` to allow specification of Nyquist spectrum and fix some bugs with asymmetric wout files
- The code no longer mods non-periodic angles (such as the field line label $\alpha$) by $2\pi$, as in field-line-following contexts, functions may not be periodic in these angles.


v0.12.1
-------

[Github Commits](https://github.com/PlasmaControl/DESC/compare/v0.12.0...v0.12.1)

- Optimizers now default to use QR factorization for least squares which is much faster
especially on GPU.
- Fix bug when reading VMEC input ZAXIS as ZAXIS_CS
- Some fixes/improvements for computing quantities along a fieldline.
- Adds compute quantities for PEST coordinate system basis vectors
- Many init methods now default to running on CPU, even when GPU is enabled, as CPU was found
to be much faster for these cases.
- New objectives `desc.objectives.FixNearAxis{R,Z,Lambda}` for fixing near axis behavior.
- Adds ``from_values`` method that was present in ``FourierRZCurve`` but missing in ``FourierRZCoil``
- Adds new ``from_values`` method for ``FourierPlanarCurve`` and ``FourierPlanarCoil``


v0.12.0
-------

[Github Commits](https://github.com/PlasmaControl/DESC/compare/v0.11.1...v0.12.0)

New Features

- Coil optimization is now possible in DESC using various filamentary coils. This includes
a number of new objectives:
    - ``desc.objectives.QuadraticFlux``
    - ``desc.objectives.ToroidalFlux``
    - ``desc.objectives.CoilLength``
    - ``desc.objectives.CoilCurvature``
    - ``desc.objectives.CoilTorsion``
    - ``desc.objectives.CoilCurrentLength``
    - ``desc.objectives.CoilSetMinDistance``
    - ``desc.objectives.PlasmaCoilSetMinDistance``
    - ``desc.objectives.FixCoilCurrent``
    - ``desc.objectives.FixSumCoilCurrent``
- Add Normal Field Error ``"B*n"`` as a plot quantity to ``desc.plotting.{plot_2d, plot_3d}``.
- New function ``desc.plotting.poincare_plot`` for creating Poincare plots by tracing
field lines from coils or other external fields.
- New profile type ``desc.profiles.TwoPowerProfile``.
- Add ``desc.geometry.FourierRZCurve.from_values`` method to fit curve with data.
- Add ``desc.geometry.FourierRZToroidalSurface.from_shape_parameters`` method for generating a surface
with specified elongation, triangularity, squareness, etc.
- New class ``desc.magnetic_fields.MagneticFieldFromUser`` for user defined B(R,phi,Z).
- All vector variables are now computed in toroidal (R,phi,Z) coordinates by default.
Cartesian (X,Y,Z) coordinates can be requested with the compute keyword ``basis='xyz'``.
- Add method ``desc.coils.CoilSet.is_self_intersecting``, which checks if any coils
intersect each other in the coilset.

Minor changes

- Improved heuristic initial guess for ``Equilibrium.map_coordinates``.
- Add documentation for default grid and target/bounds for objectives.
- Add documentation for compute function keyword arguments.
- Loading a coilset from a MAKEGRID file will now return a nested ``MixedCoilSet`` if there
are coil groups present in the MAKEGRID file.
- Users must now pass in spacing/weights to custom ``Grid``s (the previous defaults were
often wrong, leading to incorrect results)
- The ``normal`` and ``center`` parameters of a ``FourierPlanarCurve`` can now be specified
in either cartesian or cylindrical coordinates, as determined by the ``basis`` parameter.
- Misc small changes to reduce compile time and memory consumption (more coming soon!)
- Linear constraint factorization has been refactored to improve efficiency and reduce
floating point error.
- ``desc.objectives.{GenericObjective, ObjectiveFromUser}`` can now work with other objects
besides an ``Equilibrium`` (such as surfaces, curves, etc.)
- Improve warning for missing attributes when loading desc objects.

Bug Fixes

- Several small fixes to ensure things that should be ``int``s are ``int``s
- Fix incorrect toroidal components of surface basis vectors.
- Fix a regression in performance in evaluating Zernike polynomials.
- Fix errors in ``Equilibrium.map_coordinates`` for prescribed current equilibria.
- Fix definition of ``b0`` in VMEC output.
- Fix a bug where calling ``Equilibrium.compute(..., data=data)`` would lead to excessive
recalculation and potentially wrong results.
- Fixes a bug causing NaN in reverse mode AD for ``Omnigenity`` objective.
- Fix a bug where ``"A(z)"`` would be zero if the grid doesn't contain nodes at rho=1.


v0.11.1
-------
[Github Commits](https://github.com/PlasmaControl/DESC/compare/v0.11.0...v0.11.1)

- Change default symmetry to ``"sin"`` for current potential fields created with the ``from_surface`` method when the surface is stellarator symmetric
- Add objectives for coil length, curvature and torsion
- Improve Dommaschk potential magnetic field fitting by adding NFP option to only use potentials with the desired periodicity
- Fix incorrect Jacobian when bounds constraints are used by adding explicit Jacobian of ``compute_scaled_error``
- Fix bug in Dommaschk potentials that arose when evaluating the potential at Z=0
- Fix bug in Dommaschk potential fitting when symmetry is set to true
- Bump black version from 22.10.0 to 24.3.0


v0.11.0
-------

[Github Commits](https://github.com/PlasmaControl/DESC/compare/v0.10.4...v0.11.0)

New Features

- Adds functionality to optimize for omnigenity. This includes the ``OmnigenousField``
  magnetic field class, the ``Omnigenity`` objective function, and an accompanying tutorial.
- Adds new objectives for free boundary equilibria: ``BoundaryError`` and
``VacuumBoundaryError``, along with a new tutorial notebook demonstrating their usage.
- Objectives ``Volume``, ``AspectRatio``, ``Elongation`` now work for
``FourierRZToroidalSurface`` objects as well as ``Equilibrium``.
- ``MagneticField`` objects now have a method ``save_mgrid`` for saving field data
in the MAKEGRID format for use with other codes.
- ``SplineMagneticField.from_mgrid`` now defaults to using ``extcur`` from the mgrid file.
- When converting a near axis solution from QSC/QIC to a desc ``Equilibrium``, the
least squares fit is now weighted inversely with the distance from the axis to improve
the accuracy for low aspect ratio.
- Adds a bounding box to the `field_line_integrate` defined by `bounds_R` and `bounds_Z`
keyword arguments, which form a hollow cylindrical bounding box. If the field line
trajectory exits these bounds, the RHS will be multiplied by an exponentially decaying
function of the distance to the box to stop the trajectory and prevent tracing the field
line out to infinity, which is both costly and unnecessary when making a Poincare plot,
the principle purpose of the function.
- Adds a new class ``DommaschkPotentialField`` which allows creation of magnetic fields based
off of the vacuum potentials detailed in Representations for Vacuum Potentials in Stellarators
https://doi.org/10.1016/0010-4655(86)90109-8.

Speed Improvements

- ``CoilSet`` is now more efficient when stellarator or field period symmetry is used.
- Improves the efficiency of ``proximal`` optimizers by reducing the number of objective
derivative evaluations. Optimization steps should now be 2-5x faster.
- Improved performance of Zernike polynomial evaluation.
- Adds a bounding box to the `field_line_integrate` defined by `bounds_R` and `bounds_Z`
keyword arguments, which form a hollow cylindrical bounding box. If the field line
trajectory exits these bounds, the RHS will be multiplied by an exponentially decaying
function of the distance to the box to stop the trajectory and prevent tracing the
field line out to infinity, which is both costly and unnecessary when making a Poincare
plot, the principle purpose of the function.

Bug Fixes

- Fix bug causing NaN in ``ForceBalance`` objective when the grid contained nodes at
the magnetic axis.
- When saving VMEC output, ``buco`` and ``bvco`` are now correctly saved on the half
mesh. Previously they were saved on the full mesh.
- Fixed a bug where hdf5 files were not properly closed after reading.
- Fixed bugs relating to `Curve` objects not being optimizable.
- Fixed incorrect rotation matrix for `FourierPlanarCurve`.
- Fixed bug where ``plot_boundaries`` with a single ``phi`` value would return an
empty plot.

Breaking Changes

- Renames the method for comparing equivalence between DESC objects from `eq` to `equiv`
to avoid confusion with the common shorthand for `Equilibrium`.
- Minimum Python version is now 3.9

v0.10.4
-------

[Github Commits](https://github.com/PlasmaControl/DESC/compare/v0.10.3...v0.10.4)

- `Equilibrium.map_coordinates` is now differentiable.
- Removes method `Equilibrium.compute_flux_coordinates` as it is now redundant with the
more general `Equilibrium.map_coordinates`.
- Allows certain objectives to target ``FourierRZToroidalSurface`` objects as well as
``Equilibrium`` objects, such as ``MeanCurvature``, ``PrincipalCurvature``, and ``Volume``.
- Allow optimizations where the only object being optimized is not an ``Equilibrium``
object e.g. optimizing only a ``FourierRZToroidalSurface`` object to have a certain
``Volume``.
- Many functions from ``desc.plotting`` now also work for plotting quantities from
``Curve`` and ``Surface`` classes.
- Adds method ``FourierRZToroidalSurface.constant_offset_surface`` which creates
a  surface with a specified constant offset from the base surface.
- Adds method ``FourierRZToroidalSurface.from_values``  to create a surface by fitting
(R,phi,Z) points, along with a user-defined poloidal angle theta which sets the poloidal
angle for the created surface
- Adds new objective ``LinearObjectiveFromUser`` for custom linear constraints.
- `elongation` is now computed as a function of zeta rather than a single global scalar.
- Adds `beta_vol` and `betaxis` to VMEC output.
- Reorder steps in `solve_continuation_automatic` to avoid finite pressure tokamak with
zero current.
- Fix error in lambda o(rho) constraint for near axis behavior.
- Fix bug when optimizing with only a single constraint.
- Fix some bugs causing NaN in reverse mode AD for some objectives.
- Fix incompatible array shapes when user supplies initial guess for lagrange multipliers
for augmented lagrangian optimizers.
- Fix a bug caused when optimizing multiple objects at the same time and the order of
the objects gets mixed up.


v0.10.3
-------

[Github Commits](https://github.com/PlasmaControl/DESC/compare/v0.10.2...v0.10.3)

- Adds ``deriv_mode`` keyword argument to all ``Objective``s for specifying whether to
use forward or reverse mode automatic differentiation.
- Adds ``desc.compat.rescale`` for rescaling equilibria to a specified size and field
strength.
 - Adds new keyword ``surface_fixed`` to ``PlasmaVesselDistance`` objective which says
whether or not the surface comparing the distance from the plasma to is fixed or not.
If True, then the surface coordinates can be precomputed, saving on computation during
optimization. Set to False by default.
- Adds objective function `desc.objectives.GoodCoordinates` for finding "good" (ie,
non-singular, non-degenerate) coordinate mappings for initial guesses. This is applied
automatically when creating a new `Equilibrium` if the default initial guess of scaling
the boundary surface produces self-intersecting surfaces. This can be disabled by
passing `ensure_nested=False` when constructing the `Equilibrium`.
- Adds `loss_function` argument to all `Objective`s for applying one of min/max/mean
to objective function values (for targeting the average value of a profile, etc).
- `Equilibrium.get_profile` now allows user to choose a profile type (power series, spline, etc)
- Fixes a bug preventing linear objectives like `FixPressure` from being used as bounds.
- Updates to tutorials and example scripts
- `desc.interpolate` module has been deprecated in favor of the `interpax` package.
- Utility functions like `desc.objectives.get_fixed_boundary_constraints` now no longer
require the user to specify which profiles the equilibrium has, they will instead be
inferred from the equilibrium argument.


v0.10.2
-------

[Github Commits](https://github.com/PlasmaControl/DESC/compare/v0.10.1...v0.10.2)

- Updates `desc.examples`:
    * `NCSX` now has a fixed current profile. Previously it used a fixed iota based on a
    fit, but this was somewhat inaccurate.
    * `QAS` has been removed as it is now redundant with `NCSX`.
    * `ARIES-CS` has been scaled to the correct size and field strength.
    * `WISTELL-A` is now a true vacuum solution, previously it approximated the vacuum
    solution with fixed rotational transform.
    * Flips sign of iota for `W7-X` and `ATF` to account for positive jacobian.
    * new example for `HSX`.
- Adds new compute quantities `"iota current"` and `"iota vacuum"` to compute the
rotational transform contributions from the toroidal current and background field.
- Adds ability to compute equilibria with anisotropic pressure. This includes a new
profile, ``Equilibrium.anisotropy``, new compute quantity ``F_anisotropic``, and a new
objective ``ForceBalanceAnisotropic``.
- `plot_3d` and `plot_coils` have been updated to use Plotly as a backend instead of
Matplotlib, since Matplotlib isn't great for 3d plots, especially ones with multiple
overlapping objects in the scene. Main API differences:
    * Plotly doesn't have "axes" like Matplotlib does, just figures. So the `ax`
    argument has been replaced by `fig` for `plot_3d` and `plot_coils`, and they no
    longer return `ax`.
    * Names of colormaps, line patterns, etc are different, so use caution when
    specifying those using `kwargs`. Thankfully the error messages Plotly generates are
    usually pretty informative and list the available options.
- Adds zeroth and first order NAE constraints on the poloidal stream function lambda,
accessible by passing in ``fix_lambda=True`` to the ``get_NAE_constraint`` getter function.
- Implements `CurrentPotentialField` and `FourierCurrentPotentialField` classes,
which allow for computation of the magnetic field from a surface current density
given by `K = n x grad(Phi)` where `Phi` is a surface current potential.
    * `CurrentPotentialField` allows for an arbitrary current potential function `Phi`
    * `FourierCurrentPotentialField` assumes the current potential function to
    be of the form of a periodic potential (represented by a `DoubleFourierSeries`)
    and two secular terms, one each linear in the poloidal and in the toroidal angle.


v0.10.1
-------

[Github Commits](https://github.com/PlasmaControl/DESC/compare/v0.10.0...v0.10.1)

Improvements
- Adds second derivatives of contravariant basis vectors to the list of quantities we can compute.
- Refactors most of the optimizer subproblems to use JAX control flow, allowing them
to run more efficiently on the GPU.
- Adds ``'shear'`` as a compute quantity and ``Shear`` as an objective function.
- Adds a new objective ``Pressure`` to target a pressure profile as a function of rho instead
of spectral coefficients like ``FixPressure``. Can also be used when optimizing kinetic equilibria.
- Allows all profile objectives to have callable bounds and targets.
- All objective function values should now be approximately independent of the grid
resolution. Previously this was only true when objectives had `normalize=True`
- `Objective.print_value` Now prints max/min/avg for most objectives, and it should be
clear whether it is printing the actual value of the quantity or the error between
the objective and its target.
- Adds new options to `plot_boozer_modes` to plot only symmetry breaking modes (when
helicity is supplied) or only the pointwise maximum of the symmetry breaking modes.
- Changes default ``Grid`` sorting to False, to avoid unintentional sorting of
passed-in nodes. Must explicitly specify `sort=True` to ``Grid`` object to sort now.

Breaking Changes
- Removes ``grid`` attribute from ``Profile`` classes, ``grid`` should now be passed
in when calling ``Profile.compute``.

Bug Fixes
- Fix bug where running DESC through the command line interface with the `-g` flag
failed to properly utilize the GPU


v0.10.0
-------

[Github Commits](https://github.com/PlasmaControl/DESC/compare/v0.9.2...v0.10.0)

Major Changes
- Removes the various ``compute_*`` methods from ``Surface`` and ``Curve`` classes in
favor of a unified ``compute`` method, similar to ``Equilibrium.compute``. The method
takes as arguments strings containing the desired data. A full list of available options
is at https://desc-docs.readthedocs.io/en/stable/variables.html
- Analytic limits at the magnetic axis of all quantities have now been implemented.
- New functions ``desc.random.random_surface`` and ``desc.random.random_pressure`` for
generating pseudo-random toroidal surfaces and monotonic profiles.
- Adds new curve parameterization `` desc.geometry.SplineXYZCurve`` and corresponding
coil ``desc.coils.SplineXYZCoil`` that use a local spline of points in real space.
- New methods ``CoilSet.from_makegrid_coilfile`` and ``CoilSet.save_in_makegrid_format``
for creating a ``CoilSet`` of ``SplineXYZCoil`` from a MAKEGRID style text file or saving
coil data in the format expected by MAKEGRID.
- New function ``desc.magnetic_fields.read_BNORM_file`` for reading the Bnormal distribution
on a surface from a BNORM code output file.
- New methods ``compute_Bnormal`` and ``save_BNORM_file`` for all magnetic field classes
to compute the normal component of the field on a given surface and save the data in the
same format as the BNORM code.

Minor Changes
- Increases default radial resolution for stability objectives to be consistent with
other objectives.
- Creating ``Equilibrium`` objects or calling ``change_resolution`` on objects that have
it should now be significantly faster.
- ``Grid`` and ``Transform`` objects can now be created within the context of ``jit``,
by passing ``jitable=True`` to the constructor.
- Added support for newer JAX versions, up to v0.4.14. Newer versions likely work as well
but are not automatically tested.
- Adds ability to compute curvatures of constant theta and constant zeta surfaces.
- Fixes definition of derivatives of co- and contra-variant basis vectors to properly
account for the chain rule derivatives of the cylindrical basis vectors as well.
- Adds calculation of ``A(r)``, the approximate cross sectional area as a function of rho.
- Adds method ``desc.io.InputReader.descout_to_input`` to create a text input file for
DESC from a saved hdf5 output.

Bug Fixes
* Fixes bug in saving nested dicts/lists.
* Removes default node at rho=1 for ``BootstrapRedlConsistency`` objective to avoid
dividing by zero where profiles may be zero.
- Fixes bug causing ``QuasisymmetryBoozer`` to fail when compiling due to JAX issues.
- Fixes incorrect implementation of derivatives of contravariant metric tensor elements
(these were unused at present so shouldn't have caused any issues.)
* Fixes bug where bounds for profile objectives were not scaled correctly when used
as an inequality constraint.
- Fixes a bug where calculating elongation would return NaN for near-circular cross sections.


v0.9.2
------

[Github Commits](https://github.com/PlasmaControl/DESC/compare/v0.9.1...v0.9.2)

Improvements
- Improves robustness and speed of ``map_coordinates``.
- Adds axis parameters ``Ra_n`` and ``Za_n`` as optimizable DoF when using standard
constrained optimization methods (ie, not ``ProximalProjection``).
- Adds method to convert ``FourierRZCurve`` to ``FourierXYZCurve``.
- Makes DESC classes compatible with JAX pytrees.
- Adds Chebyshev polynomials as basis function (for future use).

Breaking changes
- Renames ``theta_sfl`` to ``theta_PEST`` in compute functions to avoid confusion with
other straight field line coordinate systems.
- Makes plotting kwargs a bit more uniform. ``zeta``, ``nzeta``, ``nphi`` have all been
superseded by ``phi`` which can be an integer for equally spaced angles or a float or
array of float to specify angles manually.

Bug fixes
- Avoids accidentally overwriting equilibria during automatic continuation method.

New Contributors

- \@rahulgaur104 made their first contribution in https://github.com/PlasmaControl/DESC/pull/576


v0.9.1
------

[Github Commits](https://github.com/PlasmaControl/DESC/compare/v0.9.0...v0.9.1)

Deprecations
- Creating an ``Objective`` without specifying the ``Equilibrium`` or other object to be
 optimized is deprecated, and in the future will raise an error.
 - Passing in an ``Equilibrium`` when creating an ``Objective`` no longer builds the
 objective immediately.
 - ``Objective.build`` can now be called without arguments, assuming the object to be
 optimized was specified when the objective was created.
- Removes ``Equilibrium.solved`` attribute as it was generally unused and occasionally
caused issues when saving to VMEC format.

New Features
- Adds ``deriv_mode="looped"`` option to ``desc.objectives.ObjectiveFunction`` for
computing derivative matrices. This is slightly slower but much more memory efficient
than the default ``"batched"`` option.
- Adds BFGS option for augmented Lagrangian optimizers.
- Adds utility functions for computing line integrals, vector valued integrals, and
integral transforms in ``desc.compute.utils``.
- Adds ``method="monotonic-0"`` to ``desc.interpolate.interp1d``, which enforces
monotonicity and zero slope at the endpoints.
- Adds ``rho`` argument to ``desc.plotting.plot_boozer_surface`` to specify the desired
surface, rather than having to create custom grids. Also adds a ``fieldlines`` argument
for overlaying magnetic field lines on the Boozer strength plot.
- DESC objects with a ``change_resolution`` method now allow changing symmetry type.

Minor Changes
- Augmented Lagrangian methods now use a default starting Lagrange multiplier of 0, rather
than the least squares estimate which can be a bad approximation if the starting point
is far from optimal. The old behavior can be recovered by passing
``"initial_multipliers": "least_squares"`` as part of ``options`` when calling ``optimize``.
- Enforces periodicity convention for ``alpha`` and ``theta_sfl`` - They are both now
defined to be between 0 and 2pi.

Bug Fixes
- Flips sign of current loading/saving VMEC data to account for negative Jacobian.
- Increases default resolution for computing magnetic field harmonics in ``desc.plotting.plot_qs_error``
to avoid aliasing.
- Ensures that ``Basis.L`` , ``Basis.M``, ``Basis.N`` are all integers.
- Removes duplicated entries in the ``data_index``
- Fixes a bug in the normalization of the radial unit vector.


v0.9.0
------

[Github Commits](https://github.com/PlasmaControl/DESC/compare/v0.8.2...v0.9.0)

New Features
- Implements a new limit API to correctly evaluate a number of quantities at the
coordinate singularity at $\rho=0$ rather than returning NaN. Currently only quantities
related to rotational transform and magnetic field strength are implemented, though in
the future all quantities should evaluate correctly at the magnetic axis. Note that
evaluating quantities at the axis generally requires higher order derivatives and so
can be much more expensive than evaluating at nonsingular points, so during optimization
it is not recommended to include a grid point at the axis. Generally a small finite value
such as ``rho = 1e-6`` will avoid the singularity with a negligible loss in accuracy for
analytic quantities.
- Adds new optimizers ``fmin-auglag`` and ``lsq-auglag`` for performing constrained
optimization using the augmented Lagrangian method. These generally perform much better
than constrained algorithms from scipy.
- Adds interfaces to ``trust-constr`` and ``SLSQP`` methods of ``scipy.optimize.minimize``.
These methods can handle general nonlinear constraints, though their performance on
badly scaled problems like those encountered in stellarator optimization isn't great.
- Adds calculation of the PEST straight field line coordinate jacobian, which is now
used to check for nestedness in ``Equilibrium.is_nested``. Previously the non-straight
field line jacobian was used, which would not detect if SFL theta contours overlapped.
- Introduces a new function/method ``Equilibrium.map_coordinates`` which generalizes
the existing methods ``compute_theta_coordinates`` and ``compute_flux_coordinates``,
but allows mapping between arbitrary coordinates.
- Adds calculation of $\nabla \mathbf{B}$ tensor and corresponding $L_{\nabla B}$ metric
- Adds objective ``BScaleLength`` for penalizing strong magnetic field curvature.
- Adds objective ``ObjectiveFromUser`` for wrapping an arbitrary user defined function.
- Adds utilities ``desc.grid.find_least_rational_surfaces`` and
``desc.grid.find_most_rational_surfaces`` for finding the least/most rational surfaces
for a given rotational transform profile.

Breaking changes
- ``Objective`` and ``ObjectiveFunction`` compute methods have now been separated into
``compute_unscaled`` which returns the raw physics value of the objective,
``compute_scaled``, which returns the normalized value, and ``compute_scaled_error``
which returns the normalized difference between the physics value and the target/bounds.
Similarly, ``jac_scaled`` and ``jac_unscaled`` are the relevant derivatives (note that
``jac_scaled`` is equivalent to ``jac_scaled_error`` as the constant target drops out).
``grad`` and ``hess`` methods still correspond to ``compute_scalar`` which returns the
 sum of squares of ``compute_scaled_error``
- renames ``zeta`` -> ``phi`` in many places in ``desc.plotting`` to be consistent with
when we mean the computational coordinate $\zeta$ vs the physical coordinate $\phi$
- Replaces ``nfev`` with ``maxiter`` in many places in the code. For most optimizers
in scipy and all DESC optimizers, one iteration means one accepted step, which may
require more than 1 function evaluation due to line searches or trust region subproblems.
However, derivatives are generally only evaluated once per iteration, and are usually the
most significant cost, so the iteration count is generally a better proxy for wall time
than number of function evaluations.

Minor changes
- Minor updates to work with newer versions of JAX. Minimum ``jax`` version  is now
``0.3.2``, as some functions used in the constrained optimizers aren't present in
previous versions. Maximum ``jax`` version is now ``0.4.11``, the latest as of 6/13/23.
- Adds new ``ObjectiveFunction`` attributes ``target_scaled`` and ``bounds_scaled``
which return vectors of the scaled values from each sub-objective.
- Adds automatic scaling of variables for ``scipy.optimize.minimize`` methods, using
the hessian at the initial point.
- Ensures objectives don't have both bounds and target set at the same time. This
occasionally caused issues if one of them had a default value without the user realizing.
- Adds documentation on how to interface with new optimizers.
- Adds a documentation notebook with a simple example of using constrained optimizers
- Adds a table of optimizer info to docs
- Adds capability of ``InputReader`` to read VMEC files that have inputs spanning
multiple lines
- Reduces default initial trust region radius for most optimizers to be more conservative.
- Adds a "softmin" option to ``PlasmaVesselDistance`` objective which is smoother and
usually provides a better optimization landscape compared to the standard hard min.

Bug Fixes
- Fixes orientation of theta in VMEC output. (Previously we flipped theta for the base
quantities such as $R$, $Z$, and $\lambda$, but not derived quantities such as $B$ and $J$).
- Fixes VMEC utility bug that would cause ``xn``, ``xm`` to be empty if the Fourier
modes given had only sine symmetry.
- Fixes bug when VMEC input file has duplicated lines, DESC now will just use the
last duplicated line (which is what VMEC does)

New Contributors

- \@pkim1818 made their first contribution in https://github.com/PlasmaControl/DESC/pull/503


v0.8.2
------

[Github Commits](https://github.com/PlasmaControl/DESC/compare/v0.8.1...v0.8.2)

New Features
- New compute functions for derivatives of contravariant metric tensor elements, eg, for Laplace's equation.
- New objective `Isodynamic` for penalizing *local* radial drifts, not just the bounce average as in QI
- `solve_continuation_automatic` now uses adaptive step sizing for perturbations in the event of a solution becoming unnested.

Minor changes
- Now uses `jnp` for perturbations rather than `np` which should be significantly faster, especially on GPU
- The `fixed_boundary` flag has been removed from `FixBoundaryR` and `FixBoundaryZ` constraints. It is now detected automatically based on the objective and optimizer.
- Plotting normalized force error now always uses the gradient of magnetic pressure to normalize, even at finite beta. The old behavior can be recovered by passing `norm_name="<|grad(p)|>_vol"` to the relevant plotting function.

Bug Fixes
- Fixed minor bug with symmetric grids that caused end points to be double counted
- Fixed bug causing `NFP` of curves to not be updated correctly when the equilibrium changed
- Fixed issue when converting `pyQIC` solutions to `DESC` equilibria related to offset toroidal grid


v0.8.1
------

[Github Commits](https://github.com/PlasmaControl/DESC/compare/v0.8.0...v0.8.1)

Minor Changes
* Include near-axis verification checks in NAE-constrained equilibrium example notebook

Bug Fixes
* Fix read-the-docs build error
* Add missing classes to API docs
* fix error in fix axis util function
* Add missing attributes to new classes added in `v0.8.0`


v0.8.0
------

[Github Commits](https://github.com/PlasmaControl/DESC/compare/v0.7.2...v0.8.0)

New Features
- Add profiles for kinetic quantities to `Equilibrium`
- Add compute functions and objectives for the bootstrap current for stellarators near quasisymmetry.
- Added ability to solve equilibria with the axis held fixed, or by constraining the O(rho) behavior from a near axis expansion.
- New objective for penalizing plasma-vessel distance
- All objectives now have a `bounds` argument, and the loss will be zero if within bounds.
- Added compute functions for field line quantities such as field line label `alpha`, unit vector `b`, field line curvature `kappa` etc.
- Add compute functions for covariant components of current.
- `Equilibrium.compute` will now automatically use the correct grids for surface and volume averages.
- Added a number of fields to VMEC output from DESC equilibria.

Minor Changes
- Improved handling of indefinite hessian matrices in `fmintr`

Bug Fixes
- Fix issue with composite profiles parsing parameters incorrectly
- Loading an equilibrium from VMEC now uses spline profiles to ensure consistency, as VMEC does not always save the input profile


v0.7.2
------

[Github Commits](https://github.com/PlasmaControl/DESC/compare/v0.7.1...v0.7.2)

What's Changed
* Fix bug in QS Boozer metric where non-symmetric modes were sometimes counted as
symmetric due to different Fourier series conventions.
* Improve speed of functions for converting between VMEC and DESC Fourier representations.
* Add objectives for penalizing strong shaping.
    - `MeanCurvature` targets values for the mean curvature of the flux surfaces
    (average of principal curvatures)
    - `PrincipalCurvature` penalizes the largest magnitude of the principal curvatures
* Improve default tolerances when converting input file from VMEC to DESC


v0.7.1
------

[Github Commits](https://github.com/PlasmaControl/DESC/compare/v0.7.0...v0.7.1)

What's Changed
- Allow targets for `RotationalTransform` and `ToroidalCurrent` objectives to be Profile
  objects rather than just arrays.
- Document how to add new compute functions and objectives.
- Add objective for targeting elongation.
- Fix bug in `plot_qs_error` causing the same value to be plotted for each value of rho.
- Fix bug sometimes causing the wrong equilibrium to be returned after optimization.
- Improve numerical stability of perturbations and optimization.


v0.7.0
------

[Github Commits](https://github.com/PlasmaControl/DESC/compare/v0.6.4...v0.7.0)

New Features
- Add implementation of `Surface.compute_curvature`.
- Add `return_data` flag to plotting functions to return dictionary of plotted data.
- Add computation of beta, average B, surface curvature, magnetic field derivatives, and others.
  Full list at https://desc-docs.readthedocs.io/en/stable/variables.html
- Modify `Surface` classes to always orient surfaces correctly to give the equilibrium a positive Jacobian. This means DESC will always use a right handed coordinate system in both cylindrical and flux coordinates. Previously the orientation of the flux coordinate system would depend on the boundary parameterization.
- New utility function `desc.compat.ensure_positive_jacobian` for converting previously saved equilibria to a right handed coordinate system.

Major Changes
- Refactored backend `compute` functions to calculate dependencies recursively. This should make it much easier to add new quantities, and reduces compilation times by 50-70%

Minor Changes
- Refactor wrapping of scipy optimizers.
- Add check for incompatible constraints in optimization.
- Improvements to `plot_1d` to correctly plot flux surface average quantities.
- Speed up calculation of Boozer transform.
- Have stability objectives print their max, min and average value at end of solve.

Bug Fixes
- Fix issue where if fixing specific boundary modes, the indices used to index the target do not match up with the indices of the A matrix.
- Fix weighting of duplicate nodes for periodic domains.
- Ensure transforms build correctly even with empty grid.
- Ensure transforms always have 0,0,0 derivative.
- Change normalization for poloidal field to avoid having a 0 normalization factor.


v0.6.4
------

[Github Commits](https://github.com/PlasmaControl/DESC/compare/v0.6.3...v0.6.4)

Major Changes
- All objectives now have a `normalize` argument that when true will nondimensionalize
the physics value and scale to be approximately ~O(1) in magnitude. This should make it
easier to tune weights when doing multiobjective optimization.
- New objective `RotationalTransform` for targeting a particular iota profile in real
space.
- New function `plot_boundaries` to plot comparisons between boundary shapes.

Minor Changes
- Maximum JAX version is now `0.4.1` (latest version as of release date). Minimum
version is still `0.2.11` but this will likely change in the future.

Bug fixes
- Fix indexing bug in biot-savart for coils that caused the output to have the wrong shape
- Fix a bug occasionally preventing the optimizer from restarting correctly after
trying a bad step


v0.6.3
------

[Github Commits](https://github.com/PlasmaControl/DESC/compare/v0.6.2...v0.6.3)

Major Changes
- Adds new function ``desc.continuation.solve_continuation`` which is a functional interface to the ``EquilibriaFamily.solve_continuation`` method
- Adds new function ``desc.continuation.solve_continuation_automatic`` which uses conservative default settings in a continuation method for solving complicated equilibria.
- Adds method ``Objective.xs(eq)`` for getting needed arguments for an objective. For example, ``objective.compute(*objective.xs(eq))``.
- Adds utility ``desc.perturbations.get_deltas`` for finding the differences between surfaces and profiles for perturbations.

Minor Changes
- ``EquilibriaFamily`` can now be created with one or more ``Equilibrium`` objects, or no arguments to create an empty family
- ``SplineMagneticField`` can now interpolate axisymmetric fields more efficiently.

Bug Fixes
- Fix bug preventing ``lsqtr`` from terminating when ``maxiter`` is zero.
- Fix bug when converting profiles to ``FourierZernikeProfile``.
- Fix bug where a ``FixBoundary`` constraint with only 1 mode constrained would throw an error during ``objective.build``


v0.6.2
------

[Github Commits](https://github.com/PlasmaControl/DESC/compare/v0.6.1...v0.6.2)

Minor Changes:
- Remove ``parent`` and ``children`` from ``Equilibrium`` - this was generally unused and caused memory leaks in long optimization runs.
- Allow targeting current on multiple surfaces with ``ToroidalCurrent`` objective.
- Refactored optimizer backends to get rid of unused code and use more standard BFGS implementation.
- Added ``sgd`` optimizer for performing fixed step gradient descent, with or without momentum. In the future this will be upgraded to include other step size rules and line searches.
- Allow selecting profile when loading VMEC equilibrium, as previously fixed iota was always assumed.
- Ensure equilibrium and surface have same symmetry.

Bug fixes:
- Fix floating point comparison when recovering solution after optimization, occasionally leading to the wrong iteration being returned as "optimal"
- Fix plotting iota of a current-constrained with ``plot_1d`` function
- Fix bug where having iota specified in an input file along with vacuum objective lead to error. Now specifying vacuum objective will ignore all profile inputs


v0.6.1
------

[Github Commits](https://github.com/PlasmaControl/DESC/compare/v0.6.0...v0.6.1)

New Features
- `plot_boundary` function to plot boundary surfaces and multiple toroidal angles together in a single plot. This is a popular plot format in stellarator optimization papers when comparing boundary shapes.

Bug Fixes
- Fix bugs for vacuum solve
  - Allow for constraining arguments that aren't used in the objective (i.e pressure when minimizing current density)
  - fix bug in `CurrentDensity` where only jacobi grid was being used
- Fixes to wrapped objective and jit compiling
  - optimizer with wrapped objective would sometimes return the equilibrium after the final attempted step, even if that step was rejected for not lowering the objective, resulting in incorrect "optimal" result
  - Fixes a bug where the `use_jit` arg passed to `objective.build` would override any previously set value for `use_jit` (such as in the class constructor)
- Grid spacing bugs fixed
  - fixed a bug where setting nodes with a linear spaced array versus asking for `N` linearly spaced nodes would result in different weights despite being the same nodes


v0.6.0
------

[Github Commits](https://github.com/PlasmaControl/DESC/compare/v0.5.2...v0.6.0)

Major changes

-   Can now solve equilibria with fixed toroidal current, as opposed to
    fixed rotational transform.
    -   input file now accepts `c` parameter for toroidal current
        profile (in Amps - note it should be an even polynomial and 0 on
        axis)
    -   `Equilibrium` now has attribute `current` which can be set to
        any `Profile` type (or `None` if using rotational transform)
    -   Default `Equilibrium` is now fixed zero current rather than zero
        rotational transform.
    -   For equilibria with both `iota` and `current` assigned, which to
        fix should be specified manually by using either `FixIota` or
        `FixCurrent` constraints
    -   Note that computing `iota` from fixed current requires flux
        surface averages, so more oversampling in real space may be
        required to get correct values.
-   Near axis interface:
    -   `Equilibrium.from_near_axis` allows users to load in a solution
        from `pyQSC` or `pyQIC`.
    -   `FourierRZToroidalSurface.from_near_axis` allows users to create
        boundary surfaces that are approximately QP/QI based on an
        unpublished analytic model shared by Matt Landreman.

Minor changes

-   Plotting:
    -   Document kwargs, according to [matplotlib-style
        documentation](https://stackoverflow.com/questions/62511086/how-to-document-kwargs-according-to-numpy-style-docstring)
    -   Add kwargs to plotting functions missing sensible/useful kwargs
    -   Add check for unused kwargs to most plotting functions
    -   Add `norm_F` option to `plot_fsa`
-   Transform:
    -   Modifies `Transform.fit` to use the inverse of the forward
        method transform method (ie, `direct1`, `direct2`, `fft`) rather
        than always the full matrix inverse as in `direct1`
    -   Removes weighting from `transform.fit`, to ensure that the
        inverse transform is the actual inverse of the forward
        transform.
-   Profiles:
    -   Add methods and classes for adding, subtracting, multiplying,
        scaling profiles
    -   Add class for anisotropic profiles using Fourier-Zernike basis
        (though the compute functions don\'t make use of the anisotropy
        yet)
-   Input/Output:
    -   VMEC input conversion now allows for:
        -   comma-separated lists of numbers, such as:
            `AC = 1.0, 0.5, 0.2`
        -   non-stellarator symmetric axis initial guesses using the
            inputs `RAXIS_CS` and `ZAXIS_CC`
    -   Add the Boozer currents `I` and `G` and the Mercier stability
        fields to VMEC outputs.
    -   Make DESC input file reader agnostic to the case of the input
        options (i.e. `spectral_indexing=ANSI` in the input file will
        work now and register as `ansi` internally)
-   Misc:
    -   Allow applying boundary conditions on interior surfaces: Adds a
        `surface_label` arg to `FixBoundaryR` and `FixBoundaryZ`,
        defaulting to the label of the given surface. That surface is
        fixed, instead of always the rho=1 surface.
    -   Remove `use_jit` from `Derivative` class in favor of `jit`ing
        attributes of `ObjectiveFunction`
    -   Add `jit` to `ObjectiveFunction.jvp`, to hopefully speed up
        perturbations a bit
    -   Enforce odd number of theta nodes for `ConcentricGrid`, to
        ensure correct flux surface averages
    -   Remove `ConcentricGrid` `rotation` option, as this was generally
        unused and caused some issues with surface averages.

Bug fixes

-   Fix bug in derivative of abs(`sqrt(g)`) (thanks to Matt Landreman
    for reporting). Affected quantities are `V_rr(r)`, `D_well`,
    `D_Mercier`, `magnetic well`
-   Fix a bug with using `Transform.fit()` for double Fourier series on
    multiple surfaces simultaneously. Performing the fit one surface at
    a time corrects this, but there could be room for speed
    improvements.
-   Rescale the Jacobian saved as `gmnc` & `gmns` when saving a VMEC
    output to reflect the VMEC radial coordinate convention of
    `s = rho^2`.


v0.5.2
------

[Github Commits](https://github.com/PlasmaControl/DESC/compare/v0.5.1...v0.5.2)

Major Changes

-   New objectives for `MercierStability` and `MagneticWell`
-   Change `LinearGrid` API to be more consistent with other `Grid`
    classes:
    -   L, M, N now correspond to the grid spectral resolution, rather
        than the number of grid points
    -   rho, theta, zeta can be passed as integers to specify the number
        of grid points (functionality that used to belong to L, M, N)
    -   rho, theta, zeta still retain their functionality of specifying
        coordinate values if they are not integers
    -   Other code that depends on `LinearGrid` was updated accordingly
        to use the new syntax
-   Poloidal grid points are now shifted when `sym=True` to give correct
    averages over a flux surface.
-   Added default continuation steps to converted VMEC input files

Minor Changes

-   add option to `plot_comparison` and `plot_surfaces` to not plot
    vartheta contours
-   Add better warnings for gpu and jax issues
-   add volume avg force and pressure gradient to the compute functions
-   change `is_nested` function to use jacobian sign instead of looking
    for intersections between surfaces
-   Allow alternate computation of multi-objective derivatives,
    computing individual jacobians and blocking together rather than
    computing all at once.

Bug Fixes

-   Fix nfev=1 and Some scalar solver Issues
-   Fix some formula errors in second derivatives of certain magnetic
    field components, caused by some hanging expressions.
-   fix bug where node pattern is always jacobi when force is used as
    objective
-   Allow hdf5 to store None attributes correctly
-   Fix profile parity and Z axis coefficients in `VMECIO.save`
-   Ensure axis coefficients are updated correctly after solving
    equilibrium

New Contributors

-   \@unalmis made their first contribution in
    <https://github.com/PlasmaControl/DESC/pull/247>


v0.5.1
------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.5.0...v0.5.1)

Major Changes

-   Add ability to change NFP after creation for equilibria, curves, and
    surfaces.
-   Fix rounding errors when building high resolution grids.
-   Rename `LambdaGauge` constraint to `FixLambdaGauge` for consistency.
-   Documentation updates to hdf5 output, VMEC conversion, plotting,
    etc.
-   Change default spectral indexing to \"ansi\".
-   Adds `desc.examples.get` to load boundaries, profiles, and full
    solutions from a number of example equilibria.
-   Default grid resolution is now 2x spectral resolution.
-   New surface method `FourierRZToroidalSurface.from_input_file` to
    create a surface from boundary coefficients in a DESC or VMEC input
    file.
-   Adds new tutorial notebooks demonstrating VMEC io, continuation,
    plotting, perturbations, optimization etc.
-   New documentation on perturbation theory and evaluating Zernike
    polynomials.
-   Fix bug preventing vacuum solutions from solving properly.


v0.5.0
------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.4.12...v0.5.0)

Major Changes

-   New API for building objectives and solving/optimizing equilibria. A
    brief explainer can be found in the
    [documentation](https://desc-docs.readthedocs.io/en/stable/notebooks/hands_on.html)
-   The Equilibrium class no longer \"owns\" an optimizer or objective,
    and does not build its own transforms.
-   The ObjectiveFunction class is a \"super-objective\" that combines
    multiple \"sub-objectives\" which follow the ABC \_Objective class.
-   Each sub-objective function can be used as either an \"objective\"
    (minimized during optimization) or a \"constraint\" (enforced
    exactly).
-   Each sub-objective function takes unique inputs that can be
    specified by a grid/surface/etc. or use default values from an
    Equilibrium.
-   Each sub-objective is responsible for building its own transforms or
    other constant matrices, and is also responsible for computing its
    own derivatives.
-   The super-objective dynamically builds the state vector with the
    independent variables from each sub-objective, and also combines the
    function values and derivatives by combining the outputs from each
    sub-objective.
-   The super-objective only takes a single argument (the state vector x
    or y). Perturbations are now performed wrt the full state vector y,
    which contains all of the individual parameters.
-   Adds ability to optimize physics quantities under equilibrium
    constraint using wide array of scipy and custom optimizers.
-   New objective for solving vacuum equilibria


v0.4.13
-------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.4.12...v0.4.13)

Major Changes

-   Updates JAX dependencies to take advantage of new functionality and
    faster compile times:
-   Minimum `jax` version is now 0.2.11
-   Minimum `jaxlib` version is now 0.1.69
-   Pressure and Iota perturbations now weight mode numbers by
    `L**2 + M**2` to avoid high frequency noise.
-   Custom weighting also allowed by passing `weights` array to
    `perturb` functions
-   Refactor `basis.get_idx` to use a lookup table rather than
    `np.where`. This means it works under JIT and AD, but only allows
    scalar inputs. To get multiple indices, call the method multiple
    times
-   `ConcentricGrid` now accepts a rotation argument to rotate the grid
    for either `'sin'` or `'cos'` symmetry, or `False` for no symmetry.
    This is independent of the `sym` argument, which eliminates nodes
    with theta \> pi.
-   Derivative operators for spline based profile and magnetic field
    classes are now precomputed, giving roughly 30-40% speedup in
    evaluation.

Bug Fixes

-   Fixed a bug where some properties were not copied correctly when
    doing `obj.copy(deepcopy=True)`
-   Fixed sign convention on poloidal quantities when saving to VMEC
    format
-   Fixed bugs in `Curve` and `Surface` that would fail when setting
    coefficients in JAX arrays

Testing

-   Add tests for Heliotron example
-   Adds timing benchmarks for standard equilibrium solves

Examples

-   Fix sign convention issue with Heliotron boundary modes to be
    consistent with VMEC
-   Add example for Simsopt QA stellarator from A. Bader et al. 2021

Miscellaneous

-   renamed `opsindex` to `Index` for consistency with JAX
-   Move `sign` function from `utils` to `backend`, as it now needs JAX
-   lots of minor formatting changes in docstrings


v0.4.12
-------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.4.11...v0.4.12)

New Features:

-   New function `plot_comparison` to plot comparison between multiple
    DESC equilibria
-   `plot_surfaces` now has a more intuitive API - instead of specifying
    grids, the user specifies the specific rho/theta contours to plot
-   `equil.is_nested` now checks more toroidal planes for
    non-axisymmetric equilibria by default
-   Updates `Equilibrium` to make creating them more straightforward.
    -   Instead of a dictionary of arrays and values, init method now
        takes individual arguments. These can either be objects of the
        correct type (ie `Surface` objects for boundary conditions,
        `Profile` for pressure and iota etc,) or ndarrays which will get
        parsed into objects of the correct type (for backwards
        compatibility)
    -   Also introduces more options for generating initial guesses, and
        a new dedicated method `equilibrium.set_initial_guess()`. The
        default is to scale the boundary surface that is assigned to the
        equilibrium, but another surface (and axis) can be supplied as
        an argument to the function to use that surface instead for the
        initial guess. It also accepts another `Equilibrium` instance,
        or a path to a saved DESC or VMEC equilibrium which will be
        loaded and its flux surfaces will be used as the initial guess.
    -   Command line interface updated to allow for initial guesses from
        DESC or VMEC solutions using `--guess=path` (this also replaces
        the old `--vmec` flag)
-   Adds classes for representing various types of magnetic fields
    -   Base class for all magnetic field types defining the
        `compute_magnetic_field` API and methods for combining fields
    -   `SplineMagneticField` for dealing with mgrid files and splining
        expensive to compute fields
    -   `ScalarPotentialField` for vacuum fields that can be written as
        B=grad(Phi)
    -   basic field types for testing, such as toroidal, poloidal,
        vertical
    -   field line integration function for tracing field lines in
        R,phi,Z, using JAX for differentiability


v0.4.11
-------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.4.10...v0.4.11)

Bug fixes:

-   Transforms used in the profile class weren\'t built by default,
    causing them to be built when first called which is under jit,
    meaning they would be recomputed every time instead of caching the
    transform as expected. Updated to now build transforms by default.

New Features:

-   DESC version number is now saved in hdf5 output files as
    `__version__` field.
-   Added straight field line method for plotting field line traces from
    a solved equilibrium.
-   A new method has been implemented that uses identities for the
    zernike polynomials in terms of jacobi polynomials, and a stable
    iterative evaluation for the jacobi polynomials and binomial
    coefficients. Accuracy seems on par or better than the old method
    using extended precision, at least for a given amount of computation
    time. There is some overhead from JIT compilation, but seems to pay
    off well for high resolution
-   Added new \"unique\" option for `basis.evaluate` to first reduce the
    work by finding unique combos of nodes/modes. Previously this was
    done inside each basis function evaluation, but doing it on the
    outside should be more efficient and makes the underlying functions
    differentiable.
-   Refactored fourier series evaluation to shift the arguments for
    evaluating derivatives rather than using recursion and conditionals.


v0.4.10
-------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.4.9...v0.4.10)

Bug Fixes:

-   Reordered import statements to ensure user requests to use GPU are
    handled correctly

New Features:

-   Adds several new classes to represent different
    types/parameterizations of curves and surfaces, for plasma
    boundaries, coordinate surfaces, coils, magnetic axis etc
-   New classes also have several new methods that will be made into
    objectives in the future, such as area, length, curvature, etc.
-   Surfaces can be used as boundary conditions via
    surface.get\_constraint method
-   Added new plot method to trace field lines and plot them in real
    space (R, phi, Z)


v0.4.9
------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.4.8...v0.4.9)

Bug Fixes:

-   Fix a major bug in the least squares routine that set the initial
    regularization parameter to np.nan, meaning that the optimizer would
    stall as soon as it can no longer take full newton steps.

New Features:

-   Adds a Cholesky factorization option for solving the least squares
    trust region problem. This can be faster, but less numerically
    stable due to squaring the condition number of the Jacobian. Often
    still produces good results since the trust region itself
    regularizes the solution enough to overcome the poor conditioning.
-   Methods that take Grid objects now also accept an ndarray of nodes
    or an integer specifying the number of nodes in each direction.
-   Added repr methods for string representations of more objects.


v0.4.8
------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.4.7...v0.4.8)

Bug fixes:

-   Fixed array comparison in `eq` method to return `False` for
    differently sized arrays rather than throwing an error
-   Misc errors fixed in `VMECIO.save()`
-   Fixed indexing issue with m=0, n=0 modes when transforming
    `FourierSeries` basis
-   Fixed sign error in computations of MHD energy

Changes:

-   2nd-order optimal perturbation capability added
-   Quasi-symmetry objective functions have been validated against
    STELLOPT benchmarks
-   Additional data added to the VMEC-like NetCDF output generated by
    `VMECIO.save()` for compatibility with other legacy codes
-   Added equilibrium methods for calculating cross sectional area,
    aspect ratio, major and minor radii
-   Grid weights are now scaled to always sum to 4pi\^2 even for
    symmetric grids so that volume and area will be calculated correctly
    for symmetric equilibria

Tests:

-   Added tests for `VMECIO.save()`
-   Added tests for `FourierSeries` transform bug


v0.4.7
------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.4.6...v0.4.7)

Bug fixes:

-   Fixes the magnetic axis initial guess error raised in Issue \#92

Tests:

-   Added a test to check the magnetic axis guess is used properly
-   Updated the \"Dummy Stellarator\" parameters, which gets used for
    several of the tests


v0.4.6
------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.4.5...v0.4.6)

Bug fixes:

-   Plots of straight field line vartheta contours are now actually of
    straight field line vartheta, previously they were only
    approximations.

Backend:

-   New method `equil.compute_theta_coords` finds the geometric angle
    theta that maps to a given straight field line angle vartheta


v0.4.5
------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.4.4...v0.4.5)

Bug fixes:

-   Fix bug in pickle IO that prevented objects with jitted attributes
    from being saved, pickling now only saves essential information.

Changes:

-   Added generic load function for loading objects without knowing what
    class they are
-   Removed usage of \"==\" operator between DESC objects in favor of
    `obj1.eq(obj2)`. Equivalence is defined as \"if saved and loaded,
    the two objects would be the same,\" so it ignores equality in
    trivially recomputeable attributes and focuses on the actual physics
    of the objects being compared.
-   Concentric grids are now up-down symmetric when symmetry is not
    enforced

Backend:

-   Remove `object_lib` from io, instead, now use built in dynamic
    importing to import the correct classes at runtime
-   Avoids needing to import classes in lots of files just so they can
    be in the `object_lib`, makes adding new stuff a lot easier.
-   Changed name in io stuff to class to avoid conflicts with actual
    name attributes


v0.4.4
------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.4.3...v0.4.4)

Bug Fixes:

-   Fixed key error in hdf5io that prevented some solutions from being
    loaded properly
-   Updated requirements with correct version of flatbuffers to work
    with JAX

Documentation:

-   Updated installation instructions
-   Updated hands on example and other notebooks with recent changes
-   Fixed bug where docs wouldn\'t build on RTD

New functionality:

-   Added new method equilibrium.compute\_flux\_coords to find the flux
    coordinates (rho, theta, zeta) corresponding to a set of real space
    coordinates (R,phi,Z), useful for computing synthetic diagnostics.

Backend:

-   Added wrappers for more control flow operators, which will be needed
    for future development
-   Added interpolation module with 1d, 2d, and 3d interpolation using
    linear or various cubic splines. These will primarily be needed for
    planned work on equilibrium reconstruction.


v0.4.3
------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.4.2...v0.4.3)

Major changes:

-   New transform method `direct2` that uses DFT instead of FFT to
    handle general toroidal spacing and number of planes
-   Plotting now quite a bit faster due to not having to oversample or
    use direct1 method
-   Removed `zeta_ratio` as it generally didn\'t give good results and
    is quite a bit slower than standard boundary perturbations
-   Zernike evaluation now done with higher precision for L\>24
-   Updated ASCII output format
-   Refactored how jacobian is calculated to hopefully use less memory
    on GPUs
-   New abbreviated syntax for continuation parameter arrays (see docs
    for more details)


v0.4.2
------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.4.1...v0.4.2)

Major changes:

-   New concentric grid pattern [ocs]{.title-ref}, designed to reduce
    the condition number of the interpolation matrix for fitting data to
    a zernike basis.
-   Fixed bug in poloidal resolution for concentric grids with \"ansi\"
    indexing, where only M+1 points were used instead of the correct
    2\*M+1
-   Rotated concentric grids by 2pi/3M to avoid symmetry plane at
    theta=0,pi. Previously, for stellarator symmetric cases, the nodes at
    theta=0 did not contribute to helical force balance.
-   Added [L\_grid]{.title-ref} parameter to specify radial resolution
    of grid nodes directly and making the API more consistent.


v0.4.1
------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.4.0...v0.4.1)

Major Changes:

-   GPU allocation should work correctly now, previously JAX would grab
    all GPU memory even if told to only run on CPU
-   Updated I/O to work with h5py version 3, no longer support h5py
    version 2


v0.4.0
------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.3.28...v0.4.0)


v0.3.28
-------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.3.27...v0.3.28)

Major changes:

-   better normalization for QS\_TP


v0.3.27
-------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.3.26...v0.3.27)

Major changes:

-   Update equilibriafamily to reuse objectives if possible


v0.3.26
-------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.3.25...v0.3.26)

Major changes:

-   Quasisymmetry metric finished and checked
    -   Quasisymmetry compute function is finished. This computes the
        triple product metric of quasisymmetry, denoted \'QS\_TP\'.
    -   The flux function metric \'QS\_FF\' is also computed, but has
        singularities.
    -   Appropriate references to quasisymmetry are added to
        Configuration and the plotting routines.
    -   Extensive testing functions were added to verify that the
        magnetic field and magnitude components agree with finite
        difference calculations.
    -   A \"dummy stellarator\" example was added to the test suite.
        This configuration is not in equilibrium, and gets used to test
        the compute functions.


v0.3.25
-------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.3.24...v0.3.25)

Major changes:

-   Add 3rd order perturbations
    -   seems like they\'re not that great, error is usually worse than
        2nd order but a bit better than 1st.
    -   also they take a long time (4x longer than 2nd order)
    -   might still be useful


v0.3.24
-------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.3.23...v0.3.24)

Major changes:

-   Add method to convert between coordinates
    -   Going from sfl -\> boundary representation is trivial because
        the sfl coords are valid bdry coords
    -   Going the other way is hard
    -   Added a method to configuration to transform to sfl by least
        squares fitting the flux surfaces using lambda shift
    -   Surfaces look ok after transforming, but error is a bit high
        around the edges, so we might want to revisit it in the future
        to see if we can find a better way to do it (field line
        integration?)


v0.3.23
-------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.3.22...v0.3.23)

Major changes:

-   Update handling of gpu backend
    -   Previously, telling it to run on the gpu didn\'t actually work
        and most of the computation would still be done on the cpu
    -   refactored the old method to handle the gpu properly
    -   new function for setting device that should be called before
        importing anything from backend (or anything that imports
        backend)
    -   new packages required to parse gpu and cpu info, so make sure to
        update with [pip install -r requirements.txt]{.title-ref}


v0.3.22
-------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.3.21...v0.3.22)

Major changes:

-   Added an ABC BoundaryCondition class, which inherits from
    LinearEqualityConstraint. Concrete BC\'s such as LCFSConstraint and
    PoincareConstraint are children of BoundaryCondition.
-   Added ZernikePolynomial as a Basis type. This is used for Rb\_basis
    and Zb\_basis when bdry\_mode=\"poincare\".
-   Equilibrium now has a constraint property to represent the BC. This
    must be set before setting the equilibrium\'s objective.

Minor changes:

-   Updated tests to work with changes.
-   Changed definition of beta to be e\^theta-iota\*e\^zeta. This makes
    F\_rho and F\_beta have the same units (N/m\^2).
-   Default spectral indexing set to \"fringe\" (instead of \"ansi\") in
    Basis object constructors.
-   Renamed Rb\_mn and Zb\_mn to Rb\_lmn and Zb\_lmn to reflect more
    general usage.
-   Documentation updates to meet NumPy documentation style
    requirements.


v0.3.21
-------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.3.20...v0.3.21)

This update addresses 2 major issues: objectives/optimizers not being
saved, and objectives getting compiled more often than necessary

Major Changes:

-   Changes to Equilibrium/EquilibriaFamily:
    -   general switching to using properties rather than direct
        attributes when referencing things (ie, `eq.foo`, not
        `eq._foo`). This allows getter methods to have safeguards if
        things weren\'t defined or loaded correctly for some reason
    -   Add `node_pattern` property to equilibrium
    -   Add public `transforms` property to equilibrium (public
        interface to old `_transforms` dict)
    -   When assigning objective function to equilibrium, it now checks
        if the new one is equivalent to the old one, if they are it
        skips the update. This prevents needless recompilation if
        nothing really changed.
    -   optimizer and objective attributes now assigned to
        `equilibrium.initial`
-   Changes to objective functions:
    -   object lib is now set correctly for saving/loading
    -   init method can now properly handle loading from file
    -   moved most of the derivative setup/jit/etc to its own method
        that is automatically called after the main init. The function
        `set_derivatives` can also be called manually to change jit
        settings or devices to compile to.
    -   compiling is now done on the objective rather than the
        optimizer, again, a way to prevent needless recompilation. This
        is done with a new `compile` method that takes the generic
        function arguments to call the objective, plus a \"mode\"
        argument to tell it which derivatives to compile (ie, for scalar
        vs least squares optimization)
    -   new `eq` method for comparing different objective functions.
        Effectively the same way we\'ve been doing a custom `__eq__`,
        but we can\'t do that for the objectives because it breaks the
        hashing the jax uses when jitting the objective. So instead of
        doing `objective1 == objective2`, do `objective1.eq(objective2)`
    -   Removed init methods from `ForceErrorNodes` and
        `ForceConstraintNodes`, since the default one from
        `ObjectiveFunction` now handles everything.
    -   Init for Galerkin and Energy remains but just calls super init
        and then warns if the grid is not quadrature grid
    -   new method to make sure the transforms have the correct
        derivatives for the objective and recomputing them if not
-   Changes to optimizer:
    -   io attributes now set, inheritance from IOAble and refactored
        init to work with io stuff
    -   objective no longer passed in at init, just the method
    -   instead, objective is now passed as an argument to
        `optimizer.optimize()`
    -   removed compile method in favor of compiling the objective
        directly (which is automatically done in optimizer.optimize)
    -   added equality checking for optimizers


v0.3.20
-------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.3.19...v0.3.20)

Major Changes:

-   added `ForceErrorGalerkin` objective function
    -   Returns the Galerkin equations (spectral coefficients of the
        residual), computed using Gaussian integration
    -   \"galerkin\" objective option in the input file
    -   Must use with `quad` node pattern


v0.3.19
-------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.3.18...v0.3.19)

Major Changes:

-   Added missing arg for scaling in equilibrium optimize/solve methods
-   Now checks for nestedness after perturbing but before solving to
    avoid needless computation if the perturbation throws you way off


v0.3.18
-------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.3.17...v0.3.18)

Major Changes:

-   added compute functions for magnetic pressure gradient and magnetic
    tension
-   added `norm_F` option to `plot_2d` and `plot_section`, which will
    normalize F by gradP or grad(B\^2/2mu0), depending on if the
    equilibrium is a pressure or vacuum equilibrium.


v0.3.17
-------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.3.16...v0.3.17)

Major Changes:

-   Update perturbations with trust region
    -   Method of perturbations implicitly assumes an asymptotic
        ordering of the terms in the series, but sometimes the 2nd order
        term would be much larger than the first order and the result
        would be super wrong.
    -   Perturbations are now done using a trust region approach, where
        the error is minimized subject to a bound on the step size, and
        the bound is inversely proportional to the order of the
        perturbation.
    -   trust region ratio can be varied, default of 0.1 seems ok.
    -   2nd order perturbations for BC seem to work fine now
    -   2nd order for pressure still works, though visually they look a
        bit worse despite the new method resulting in lower force error.


v0.3.16
-------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.3.15...v0.3.16)

Major Changes:

-   Updated \"put\" test to avoid deprecated usage


v0.3.15
-------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.3.14...v0.3.15)

Major Changes:

-   Update plotting
    -   removed `Plot` class in favor of individual functions (class
        wasn\'t really doing anything and just led to extra typing)
    -   Fixed bug that caused things to be plotted against the wrong
        axes (with fft node sorting things should be reshaped as (M,L,N)
        order=\'F\')
    -   `plot_surfaces` and `plot_section` now plot multiple sections
        for non-axisymmetric cases by default
    -   Made 3d plot show all field periods by default
    -   Fixed aspect ratio on 3d plots so that the axes are equal
    -   Changed method for section plotting from `tricontourf` to
        regular `contourf` so it can plot non-convex shapes correctly
    -   Added tests for 3d plotting and plotting vs different grids
    -   Updated baseline images for all tests


v0.3.14
-------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.3.13...v0.3.14)

Major Changes:

-   Fix bug with boundary perturbations
    -   Changing the resolution before perturbation was changing the BC
        coeffs as well, so the delta was zero
    -   Now only change the resolution.
    -   Also added some logic to avoid recomputing stuff when not needed


v0.3.13
-------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.3.12...v0.3.13)


v0.3.12
-------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.3.11...v0.3.12)

Major Changes:

-   Update configuration - make private
-   Configuration now inherits from ABC
-   Replaced references to configuration in other code with reference to
    Equilibrium


v0.3.11
-------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.3.10...v0.3.11)

Major Changes:

-   `perturb` function uses jvp and has 1st-order testing
    -   perturb method now uses jvp instead of full jacobians for
        1st-order perturbations
    -   test\_perturbations.py is updated to include testing for the new
        syntax with a linear test function
    -   added Equilibrium.perturb() and ObjectiveFunction.jvp() methods

Minor changes:

-   added zeta\_ratio getter method to Configuration
-   added compute method to Equilibrium
-   bug fix in ObjectiveFunction.derivative for int argnums
-   updated documentation


v0.3.10
-------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.3.9...v0.3.10)

Major Changes:

-   Add blocked derivative
    -   AutoDiffDerivative now takes keyword args to compute
        jacobian/hessian in smaller blocks to save memory
    -   Still need to find sensible defaults or come up with some way to
        automatically select block size based on hardware and memory


v0.3.9
------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.3.8...v0.3.9)

Major Changes:

-   Improved testing of SOLOVEV results
    -   Changed SOLOVEV input file to use same resolution as VMEC
        results
    -   Added a test to check that SOLOVEV solution matches VMEC results
    -   Created temporary directory to store misc testing files
    -   Fixed IO bug in Configuration


v0.3.8
------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.3.7...v0.3.8)

Major Changes:

-   Fix issue with jax and zero sized arrays
    -   Computing the pseudoinverse of a zero sized array caused jax to
        crash
    -   Now have a check to only compute pinv if array has data,
        otherwise its just zeros.
    -   Jax now seems to work fine in all cases with the new coordinates


v0.3.7
------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.3.6...v0.3.7)

Major Changes:

-   Update setup.py and \_\_main\_\_.py with version info


v0.3.6
------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.3.5...v0.3.6)

Major Changes:

-   Add colorama and termcolor to requirements.txt


v0.3.5
------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.3.4...v0.3.5)

-   initial work on VMEC IO
-   Added VMECIO class to handle loading and saving to/from VMEC netCDF
    file formats.
-   Removed check for nested flux surfaces.
-   Minor documentation changes.


v0.3.4
------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.3.3...v0.3.4)

Major Changes:

-   Update BC to work with perturbations
-   Objective functions now know about bc constraint and how to convert
    between full and reduced form of x
-   LinearEqualityConstraint class now exposes A,Ainv,Z etc for other
    uses, bypassing methods of the class when we want to differentiate
    through them


v0.3.3
------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.3.2...v0.3.3)

Major Changes:

-   Updates to files that depend on compute functions
-   Configuration now overloads all available compute functions.
-   Updated Plot class to use new compute functions.
-   Removed unused objective functions (some of these may need to be
    rewritten)
-   Updated ForceErrorNodes to use the new compute functions.
-   Minor documentation changes to the compute functions.


v0.3.2
------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.3.1...v0.3.2)

Major Changes:

-   Add solve method to Equilibrium
-   Configuration now has attributes for continuation params
    (\*\_ratios)
-   Equilibrium now has solve method which takes an Optimizer and
    Objective function and does it\'s thing.


v0.3.1
------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.3.0...v0.3.1)

Major Changes:

-   revised compute functions for new polar coordinates
-   `compute_polar_coords` now handles the transforms from spectral to
    real space.
-   `compute_toroidal_coords` then converts the polar coordinates (R0,
    Z0, r, lambda) to the toroidal coordinates (R,phi,Z).
-   `compute_magnetic_field` was modified to use the non-sfl coordinate
    system.
-   Started segregating functions to only handle specific objective
    functions (force balance vs quasi-symmetry, etc).


v0.3.0
------

[Github
Commits](https://github.com/PlasmaControl/DESC/compare/v0.2.0...v0.3.0)

Major Changes:

-   Refactored all code to be object oriented
