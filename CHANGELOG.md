Changelog
=========

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
        correct type (ie `Surface` objects for boundary condiitons,
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
    theta=0,pi. Previously, for stellarator symmetic cases, the nodes at
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

-   Changes to Equilibium/EquilibriaFamily:
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
