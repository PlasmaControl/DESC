Changelog
=========

v0.5.1
#######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.5.0...v0.5.1>`_

Major Changes

- Add ability to change NFP after creation for equilibria, curves, and surfaces.
- Fix rounding errors when building high resolution grids.
- Rename ``LambdaGauge`` constraint to ``FixLambdaGauge`` for consistency.
- Documentation updates to hdf5 output, VMEC conversion, plotting, etc.
- Change default spectral indexing to "ansi".
- Adds ``desc.examples.get`` to load boundaries, profiles, and full solutions from a number of example equilibria.
- Default grid resolution is now 2x spectral resolution.
- New surface method ``FourierRZToroidalSurface.from_input_file`` to create a surface from boundary coefficients in a DESC or VMEC input file.
- Adds new tutorial notebooks demonstrating VMEC io, continuation, plotting, perturbations, optimization etc.
- New documentation on perturbation theory and evaluating Zernike polynomials.
- Fix bug preventing vacuum solutions from solving properly.  

v0.5.0
#######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.4.12...v0.5.0>`_

Major Changes

- New API for building objectives and solving/optimizing equilibria. A brief explainer can be found in the `documentation <https://desc-docs.readthedocs.io/en/stable/notebooks/hands_on.html>`_
- The Equilibrium class no longer "owns" an optimizer or objective, and does not build its own transforms.
- The ObjectiveFunction class is a "super-objective" that combines multiple "sub-objectives" which follow the ABC _Objective class.
- Each sub-objective function can be used as either an "objective" (minimized during optimization) or a "constraint" (enforced exactly).
- Each sub-objective function takes unique inputs that can be specified by a grid/surface/etc. or use default values from an Equilibrium.
- Each sub-objective is responsible for building its own transforms or other constant matrices, and is also responsible for computing its own derivatives.
- The super-objective dynamically builds the state vector with the independent variables from each sub-objective, and also combines the function values and derivatives by combining the outputs from each sub-objective.
- The super-objective only takes a single argument (the state vector x or y). Perturbations are now performed wrt the full state vector y, which contains all of the individual parameters.
- Adds ability to optimize physics quantities under equilibrium constraint using wide array of scipy and custom optimizers.
- New objective for solving vacuum equilibria


v0.4.13
#######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.4.12...v0.4.13>`_

Major Changes

- Updates JAX dependencies to take advantage of new functionality and faster compile times:
- Minimum ``jax`` version is now 0.2.11
- Minimum ``jaxlib`` version is now 0.1.69
- Pressure and Iota perturbations now weight mode numbers by ``L**2 + M**2`` to avoid high frequency noise.
- Custom weighting also allowed by passing ``weights`` array to ``perturb`` functions
- Refactor ``basis.get_idx`` to use a lookup table rather than ``np.where``. This means it works under JIT and AD, but only allows scalar inputs. To get multiple indices, call the method multiple times
- ``ConcentricGrid`` now accepts a rotation argument to rotate the grid for either ``'sin'`` or ``'cos'`` symmetry, or ``False`` for no symmetry. This is independent of the ``sym`` argument, which eliminates nodes with theta > pi.
- Derivative operators for spline based profile and magnetic field classes are now precomputed, giving roughly 30-40% speedup in evaluation.

Bug Fixes

- Fixed a bug where some properties were not copied correctly when doing ``obj.copy(deepcopy=True)``
- Fixed sign convention on poloidal quantities when saving to VMEC format
- Fixed bugs in ``Curve`` and ``Surface`` that would fail when setting coefficients in JAX arrays

Testing

- Add tests for Heliotron example
- Adds timing benchmarks for standard equilibrium solves 

Examples

- Fix sign convention issue with Heliotron boundary modes to be consistent with VMEC
- Add example for Simsopt QA stellarator from A. Bader et al. 2021

Miscellaneous

- renamed ``opsindex`` to ``Index`` for consistency with JAX
- Move ``sign`` function from ``utils`` to ``backend``, as it now needs JAX
- lots of minor formatting changes in docstrings


v0.4.12
#######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.4.11...v0.4.12>`_

New Features:

- New function ``plot_comparison`` to plot comparison between multiple DESC equilibria
- ``plot_surfaces`` now has a more intuitive API - instead of specifying grids, the user specifies the specific rho/theta contours to plot
- ``equil.is_nested`` now checks more toroidal planes for non-axisymmetric equilibria by default
- Updates ``Equilibrium`` to make creating them more straightforward.

  - Instead of a dictionary of arrays and values, init method now takes individual arguments. These can either be objects of the correct type (ie ``Surface`` objects for boundary condiitons, ``Profile`` for pressure and iota etc,) or ndarrays which will get parsed into objects of the correct type (for backwards compatibility)
  - Also introduces more options for generating initial guesses, and a new dedicated method ``equilibrium.set_initial_guess()``. The default is to scale the boundary surface that is assigned to the equilibrium, but another surface (and axis) can be supplied as an argument to the function to use that surface instead for the initial guess. It also accepts another ``Equilibrium`` instance, or a path to a saved DESC or VMEC equilibrium which will be loaded and its flux surfaces will be used as the initial guess.
  - Command line interface updated to allow for initial guesses from DESC or VMEC solutions using ``--guess=path`` (this also replaces the old ``--vmec`` flag)

- Adds classes for representing various types of magnetic fields

  - Base class for all magnetic field types defining the ``compute_magnetic_field`` API and methods for combining fields
  - ``SplineMagneticField`` for dealing with mgrid files and splining expensive to compute fields
  - ``ScalarPotentialField`` for vacuum fields that can be written as B=grad(Phi)
  - basic field types for testing, such as toroidal, poloidal, vertical
  - field line integration function for tracing field lines in R,phi,Z, using JAX for differentiability


v0.4.11
#######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.4.10...v0.4.11>`_

Bug fixes:

- Transforms used in the profile class weren't built by default, causing them to be built when first called which is under jit, meaning they would be recomputed every time instead of caching the transform as expected. Updated to now build transforms by default.

New Features:

- DESC version number is now saved in hdf5 output files as ``__version__`` field.
- Added straight field line method for plotting field line traces from a solved equilibrium.
- A new method has been implemented that uses identities for the zernike polynomials in terms of jacobi polynomials, and a stable iterative evaluation for the jacobi polynomials and binomial coefficients. Accuracy seems on par or better than the old method using extended precision, at least for a given amount of computation time. There is some overhead from JIT compilation, but seems to pay off well for high resolution
- Added new "unique" option for ``basis.evaluate`` to first reduce the work by finding unique combos of nodes/modes. Previously this was done inside each basis function evaluation, but doing it on the outside should be more efficient and makes the underlying functions differentiable.
- Refactored fourier series evaluation to shift the arguments for evaluating derivatives rather than using recursion and conditionals.


v0.4.10
#######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.4.9...v0.4.10>`_

Bug Fixes:

- Reordered import statements to ensure user requests to use GPU are handled correctly

New Features:

- Adds several new classes to represent different types/parameterizations of curves and surfaces, for plasma boundaries, coordinate surfaces, coils, magnetic axis etc
- New classes also have several new methods that will be made into objectives in the future, such as area, length, curvature, etc.
- Surfaces can be used as boundary conditions via surface.get_constraint method
- Added new plot method to trace field lines and plot them in real space (R, phi, Z)


v0.4.9
######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.4.8...v0.4.9>`_

Bug Fixes:

- Fix a major bug in the least squares routine that set the initial regularization parameter to np.nan, meaning that the optimizer would stall as soon as it can no longer take full newton steps.

New Features:

- Adds a Cholesky factorization option for solving the least squares trust region problem. This can be faster, but less numerically stable due to squaring the condition number of the Jacobian. Often still produces good results since the trust region itself regularizes the solution enough to overcome the poor conditioning.
- Methods that take Grid objects now also accept an ndarray of nodes or an integer specifying the number of nodes in each direction.
- Added repr methods for string representations of more objects.


v0.4.8
######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.4.7...v0.4.8>`_

Bug fixes:

- Fixed array comparison in ``eq`` method to return ``False`` for differently sized arrays rather than throwing an error
- Misc errors fixed in ``VMECIO.save()``
- Fixed indexing issue with m=0, n=0 modes when transforming ``FourierSeries`` basis
- Fixed sign error in computations of MHD energy

Changes:

- 2nd-order optimal perturbation capability added
- Quasi-symmetry objective functions have been validated against STELLOPT benchmarks
- Additional data added to the VMEC-like NetCDF output generated by ``VMECIO.save()`` for compatibility with other legacy codes
- Added equilibrium methods for calculating cross sectional area, aspect ratio, major and minor radii
- Grid weights are now scaled to always sum to 4pi^2 even for symmetric grids so that volume and area will be calculated correctly for symmetric equilibria

Tests:

- Added tests for ``VMECIO.save()``
- Added tests for ``FourierSeries`` transform bug


v0.4.7
######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.4.6...v0.4.7>`_

Bug fixes:

- Fixes the magnetic axis initial guess error raised in Issue #92

Tests:

- Added a test to check the magnetic axis guess is used properly
- Updated the "Dummy Stellarator" parameters, which gets used for several of the tests

  
v0.4.6
######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.4.5...v0.4.6>`_

Bug fixes:

- Plots of straight field line vartheta contours are now actually of straight field line vartheta, previously they were only approximations.

Backend:

- New method ``equil.compute_theta_coords`` finds the geometric angle theta that maps to a given straight field line angle vartheta


v0.4.5
######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.4.4...v0.4.5>`_

Bug fixes:

- Fix bug in pickle IO that prevented objects with jitted attributes from being saved, pickling now only saves essential information.

Changes:

- Added generic load function for loading objects without knowing what class they are
- Removed usage of "==" operator between DESC objects in favor of ``obj1.eq(obj2)``. Equivalence is defined as "if saved and loaded, the two objects would be the same," so it ignores equality in trivially recomputeable attributes and focuses on the actual physics of the objects being compared.
- Concentric grids are now up-down symmetric when symmetry is not enforced

Backend:

- Remove ``object_lib`` from io, instead, now use built in dynamic importing to import the correct classes at runtime
- Avoids needing to import classes in lots of files just so they can be in the ``object_lib``, makes adding new stuff a lot easier.
- Changed name in io stuff to class to avoid conflicts with actual name attributes

  
v0.4.4
######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.4.3...v0.4.4>`_
  
Bug Fixes:

- Fixed key error in hdf5io that prevented some solutions from being loaded properly
- Updated requirements with correct version of flatbuffers to work with JAX

Documentation:

- Updated installation instructions
- Updated hands on example and other notebooks with recent changes
- Fixed bug where docs wouldn't build on RTD

New functionality:

- Added new method equilibrium.compute_flux_coords to find the flux coordinates (rho, theta, zeta) corresponding to a set of real space coordinates (R,phi,Z), useful for computing synthetic diagnostics.

Backend:

- Added wrappers for more control flow operators, which will be needed for future development
- Added interpolation module with 1d, 2d, and 3d interpolation using linear or various cubic splines. These will primarily be needed for planned work on equilibrium reconstruction.


v0.4.3
######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.4.2...v0.4.3>`_

Major changes:

- New transform method ``direct2`` that uses DFT instead of FFT to handle general toroidal spacing and number of planes
- Plotting now quite a bit faster due to not having to oversample or use direct1 method
- Removed ``zeta_ratio`` as it generally didn't give good results and is quite a bit slower than standard boundary perturbations
- Zernike evaluation now done with higher precision for L>24
- Updated ASCII output format
- Refactored how jacobian is calculated to hopefully use less memory on GPUs
- New abbreviated syntax for continuation parameter arrays (see docs for more details)


v0.4.2
######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.4.1...v0.4.2>`_

Major changes:

- New concentric grid pattern `ocs`, designed to reduce the condition number of the interpolation matrix for fitting data to a zernike basis.
- Fixed bug in poloidal resolution for concentric grids with "ansi" indexing, where only M+1 points were used instead of the correct 2*M+1
- Rotated concentric grids by 2pi/3M to avoid symmetry plane at theta=0,pi. Previously, for stellarator symmetic cases, the nodes at theta=0 did not contribute to helical force balance.
- Added `L_grid` parameter to specify radial resolution of grid nodes directly and making the API more consistent.


v0.4.1
######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.4.0...v0.4.1>`_

Major Changes:

- GPU allocation should work correctly now, previously JAX would grab all GPU memory even if told to only run on CPU
- Updated I/O to work with h5py version 3, no longer support h5py version 2


v0.4.0
######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.3.28...v0.4.0>`_


v0.3.28
#######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.3.27...v0.3.28>`_

Major changes:

- better normalization for QS_TP


v0.3.27
#######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.3.26...v0.3.27>`_

Major changes:

- Update equilibriafamily to reuse objectives if possible
  

v0.3.26
#######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.3.25...v0.3.26>`_

Major changes:

- Quasisymmetry metric finished and checked

  - Quasisymmetry compute function is finished. This computes the triple product metric of quasisymmetry, denoted 'QS_TP'.
  - The flux function metric 'QS_FF' is also computed, but has singularities.
  - Appropriate references to quasisymmetry are added to Configuration and the plotting routines.
  - Extensive testing functions were added to verify that the magnetic field and magnitude components agree with finite difference calculations.
  - A "dummy stellarator" example was added to the test suite. This configuration is not in equilibrium, and gets used to test the compute functions.


v0.3.25
#######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.3.24...v0.3.25>`_

Major changes:

- Add 3rd order perturbations

  - seems like they're not that great, error is usually worse than 2nd order but a bit better than 1st.
  - also they take a long time (4x longer than 2nd order)
  - might still be useful


v0.3.24
#######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.3.23...v0.3.24>`_

Major changes:

- Add method to convert between coordinates

  - Going from sfl -> boundary representation is trivial because the sfl coords are valid bdry coords
  - Going the other way is hard
  - Added a method to configuration to transform to sfl by least squares fitting the flux surfaces using lambda shift
  - Surfaces look ok after transforming, but error is a bit high around the edges, so we might want to revisit it in the future to see if we can find a better way to do it (field line integration?)


v0.3.23
#######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.3.22...v0.3.23>`_

Major changes:

- Update handling of gpu backend

  - Previously, telling it to run on the gpu didn't actually work and most of the computation would still be done on the cpu
  - refactored the old method to handle the gpu properly
  - new function for setting device that should be called before importing anything from backend (or anything that imports backend)
  - new packages required to parse gpu and cpu info, so make sure to update with `pip install -r requirements.txt`


v0.3.22
#######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.3.21...v0.3.22>`_

Major changes:

- Added an ABC BoundaryCondition class, which inherits from LinearEqualityConstraint.  Concrete BC's such as LCFSConstraint and PoincareConstraint are children of BoundaryCondition.
- Added ZernikePolynomial as a Basis type. This is used for Rb_basis and Zb_basis when bdry_mode="poincare".
- Equilibrium now has a constraint property to represent the BC. This must be set before setting the equilibrium's objective.

Minor changes:

- Updated tests to work with changes.
- Changed definition of beta to be e^theta-iota*e^zeta.  This makes F_rho and F_beta have the same units (N/m^2).
- Default spectral indexing set to "fringe" (instead of "ansi") in Basis object constructors.
- Renamed Rb_mn and Zb_mn to Rb_lmn and Zb_lmn to reflect more general usage.
- Documentation updates to meet NumPy documentation style requirements.


v0.3.21
#######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.3.20...v0.3.21>`_

This update addresses 2 major issues: objectives/optimizers not being saved, and objectives getting compiled more often than necessary

Major Changes:

- Changes to Equilibium/EquilibriaFamily:

  - general switching to using properties rather than direct attributes when referencing things (ie, ``eq.foo``, not ``eq._foo``). This allows getter methods to have safeguards if things weren't defined or loaded correctly for some reason
  - Add ``node_pattern`` property to equilibrium
  - Add public ``transforms`` property to equilibrium (public interface to old ``_transforms`` dict)
  - When assigning objective function to equilibrium, it now checks if the new one is equivalent to the old one, if they are it skips the update. This prevents needless recompilation if nothing really changed.
  - optimizer and objective attributes now assigned to ``equilibrium.initial``

- Changes to objective functions:
  
  - object lib is now set correctly for saving/loading
  - init method can now properly handle loading from file
  - moved most of the derivative setup/jit/etc to its own method that is automatically called after the main init. The function ``set_derivatives`` can also be called manually to change jit settings or devices to compile to.
  - compiling is now done on the objective rather than the optimizer, again, a way to prevent needless recompilation. This is done with a new ``compile`` method that takes the generic function arguments to call the objective, plus a "mode" argument to tell it which derivatives to compile (ie, for scalar vs least squares optimization)
  - new ``eq`` method for comparing different objective functions. Effectively the same way we've been doing a custom ``__eq__``, but we can't do that for the objectives because it breaks the hashing the jax uses when jitting the objective. So instead of doing ``objective1 == objective2``, do ``objective1.eq(objective2)``
  - Removed init methods from ``ForceErrorNodes`` and ``ForceConstraintNodes``, since the default one from ``ObjectiveFunction`` now handles everything.
  - Init for Galerkin and Energy remains but just calls super init and then warns if the grid is not quadrature grid
  - new method to make sure the transforms have the correct derivatives for the objective and recomputing them if not

- Changes to optimizer:
  
  - io attributes now set, inheritance from IOAble and refactored init to work with io stuff
  - objective no longer passed in at init, just the method
  - instead, objective is now passed as an argument to ``optimizer.optimize()``
  - removed compile method in favor of compiling the objective directly (which is automatically done in optimizer.optimize)
  - added equality checking for optimizers


v0.3.20
#######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.3.19...v0.3.20>`_

Major Changes:

- added ``ForceErrorGalerkin`` objective function

  - Returns the Galerkin equations (spectral coefficients of the residual), computed using Gaussian integration
  - "galerkin" objective option in the input file
  - Must use with ``quad`` node pattern


v0.3.19
#######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.3.18...v0.3.19>`_

Major Changes:

- Added missing arg for scaling in equilibrium optimize/solve methods
- Now checks for nestedness after perturbing but before solving to avoid needless computation if the perturbation throws you way off


v0.3.18
#######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.3.17...v0.3.18>`_

Major Changes:

- added compute functions for magnetic pressure gradient and magnetic tension
- added ``norm_F`` option to ``plot_2d`` and ``plot_section``, which will normalize F by gradP or grad(B^2/2mu0), depending on if the equilibrium is a pressure or vacuum equilibrium.


v0.3.17
#######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.3.16...v0.3.17>`_

Major Changes:

- Update perturbations with trust region

  - Method of perturbations implicitly assumes an asymptotic ordering of the terms in the series, but sometimes the 2nd order term would be much larger than the first order and the result would be super wrong.
  - Perturbations are now done using a trust region approach, where the error is minimized subject to a bound on the step size, and the bound is inversely proportional to the order of the perturbation.
  - trust region ratio can be varied, default of 0.1 seems ok.
  - 2nd order perturbations for BC seem to work fine now
  - 2nd order for pressure still works, though visually they look a bit worse despite the new method resulting in lower force error.


v0.3.16
#######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.3.15...v0.3.16>`_

Major Changes:

- Updated "put" test to avoid deprecated usage

  
v0.3.15
#######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.3.14...v0.3.15>`_

Major Changes:

- Update plotting

  - removed ``Plot`` class in favor of individual functions (class wasn't really doing anything and just led to extra typing)
  - Fixed bug that caused things to be plotted against the wrong axes (with fft node sorting things should be reshaped as (M,L,N) order='F')
  - ``plot_surfaces`` and ``plot_section`` now plot multiple sections for non-axisymmetric cases by default
  - Made 3d plot show all field periods by default
  - Fixed aspect ratio on 3d plots so that the axes are equal
  - Changed method for section plotting from ``tricontourf`` to regular ``contourf`` so it can plot non-convex shapes correctly
  - Added tests for 3d plotting and plotting vs different grids
  - Updated baseline images for all tests


v0.3.14
#######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.3.13...v0.3.14>`_

Major Changes:

- Fix bug with boundary perturbations

  - Changing the resolution before perturbation was changing the BC coeffs as well, so the delta was zero
  - Now only change the resolution.
  - Also added some logic to avoid recomputing stuff when not needed


v0.3.13
#######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.3.12...v0.3.13>`_


v0.3.12
#######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.3.11...v0.3.12>`_

Major Changes:

- Update configuration - make private
- Configuration now inherits from ABC
- Replaced references to configuration in other code with reference to Equilibrium


v0.3.11
#######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.3.10...v0.3.11>`_

Major Changes:

- ``perturb`` function uses jvp and has 1st-order testing

  - perturb method now uses jvp instead of full jacobians for 1st-order perturbations
  - test_perturbations.py is updated to include testing for the new syntax with a linear test function
  - added Equilibrium.perturb() and ObjectiveFunction.jvp() methods

Minor changes:

- added zeta_ratio getter method to Configuration
- added compute method to Equilibrium
- bug fix in ObjectiveFunction.derivative for int argnums
- updated documentation


v0.3.10
#######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.3.9...v0.3.10>`_

Major Changes:

- Add blocked derivative
  
  - AutoDiffDerivative now takes keyword args to compute jacobian/hessian in smaller blocks to save memory
  - Still need to find sensible defaults or come up with some way to automatically select block size based on hardware and memory


v0.3.9
######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.3.8...v0.3.9>`_

Major Changes:

- Improved testing of SOLOVEV results
  
  - Changed SOLOVEV input file to use same resolution as VMEC results
  - Added a test to check that SOLOVEV solution matches VMEC results
  - Created temporary directory to store misc testing files
  - Fixed IO bug in Configuration


v0.3.8
######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.3.7...v0.3.8>`_

Major Changes:

- Fix issue with jax and zero sized arrays
  
  - Computing the pseudoinverse of a zero sized array caused jax to crash
  - Now have a check to only compute pinv if array has data, otherwise its just zeros.
  - Jax now seems to work fine in all cases with the new coordinates


v0.3.7
######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.3.6...v0.3.7>`_

Major Changes:

- Update setup.py and __main__.py with version info

  
v0.3.6
######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.3.5...v0.3.6>`_

Major Changes:

- Add colorama and termcolor to requirements.txt

  
v0.3.5
######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.3.4...v0.3.5>`_

- initial work on VMEC IO
- Added VMECIO class to handle loading and saving to/from VMEC netCDF file formats.
- Removed check for nested flux surfaces.
- Minor documentation changes.


v0.3.4
######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.3.3...v0.3.4>`_

Major Changes:

- Update BC to work with perturbations
- Objective functions now know about bc constraint and how to convert between full and reduced form of x
- LinearEqualityConstraint class now exposes A,Ainv,Z etc for other uses, bypassing methods of the class when we want to differentiate through them


v0.3.3
######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.3.2...v0.3.3>`_

Major Changes:

- Updates to files that depend on compute functions
- Configuration now overloads all available compute functions.
- Updated Plot class to use new compute functions.
- Removed unused objective functions (some of these may need to be rewritten)
- Updated ForceErrorNodes to use the new compute functions.
- Minor documentation changes to the compute functions.


v0.3.2
######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.3.1...v0.3.2>`_

Major Changes:

- Add solve method to Equilibrium
- Configuration now has attributes for continuation params (*_ratios)
- Equilibrium now has solve method which takes an Optimizer and Objective function and does it's thing.


v0.3.1
######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.3.0...v0.3.1>`_

Major Changes:

- revised compute functions for new polar coordinates
- ``compute_polar_coords`` now handles the transforms from spectral to real space.
- ``compute_toroidal_coords`` then converts the polar coordinates (R0, Z0, r, lambda) to the toroidal coordinates (R,phi,Z).
- ``compute_magnetic_field`` was modified to use the non-sfl coordinate system.
- Started segregating functions to only handle specific objective functions (force balance vs quasi-symmetry, etc).


v0.3.0
######

`Github Commits <https://github.com/PlasmaControl/DESC/compare/v0.2.0...v0.3.0>`_

Major Changes:

- Refactored all code to be object oriented
