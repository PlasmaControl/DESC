==============================
Adding new objective functions
==============================

This guide walks through creating a new objective to optimize using Quasi-symmetry and mirror ratio as
an example. The primary methods needed for a new objective are ``__init__``, ``build``,
and ``compute``. The base class ``_Objective`` provides a number of other methods that
generally do not need to be re-implemented for subclasses.

``__init__`` should generally just assign attributes and store inputs. It should not do
any expensive calculations, these should be in ``build`` or ``compute``. The main
arguments are summarized in the example below.

``build`` is called before optimization with the ``Equilibrium`` to be optimized. It is
used to precompute things like transform matrices that convert between spectral
coefficients and real space values. In the build method, we first ensure that a ``Grid``
is assigned, using default values from the equilibrium if necessary. The grid defines
the points in flux coordinates where we evaluate the residuals. Next, we define the
physics quantities we need to evaluate the objective (``_data_keys``), and the number of
residuals that will be returned by ``compute`` (``_dim_f``). Next, we use some helper
functions to build the required ``Transform`` and ``Profile`` objects needed to compute
the desired physics quantities. These ``transforms`` and ``profiles`` are then packaged
into ``constants``, which will be passed to the ``compute`` method. Other "constant"
values that are needed to compute the given quantity such as hyperparameters or other
objects that will not be optimized should also be included in ``constants``. Finally, we
call the base class ``build`` method to do some checking of array sizes and other
miscellaneous stuff.

``compute`` is where the actual calculation of the residual takes place. Objectives
generally return a vector of residuals that are minimized in a least squares sense,
though the exact method will depend on the optimization algorithm. The main thing here
is calling ``compute_fun`` to get physics quantities, and then performing any
post-processing we want such as averaging, combining, etc. ``compute`` always must return
a 1-D array.

A full example objective with comments describing the key points is given below:
::

    from desc.objectives.objective_funs import _Objective
    from desc.objectives.normalization import compute_scaling_factors
    from desc.compute import get_profiles, get_transforms
    from desc.compute.utils import _compute as compute_fun
    from desc.grid import LinearGrid


    class QuasisymmetryTripleProduct(_Objective):  # need to subclass from ``desc.objectives._Objective``
        """Give a description of what it is and what it's useful for.

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium that will be optimized to satisfy the Objective.
        grid : Grid, optional
            Collocation grid containing the nodes to evaluate at.

        """
        # Most of the documentation is shared among all objectives, so we just inherit
        # the docstring from the base class and add a few details specific to this objective.
        # See the documentation of `collect_docs` for more details.
        __doc__ = __doc__.rstrip() + collect_docs(
            target_default="``target=0``.", bounds_default="``target=0``."
        )

        _coordinates = "rtz"    # What coordinates is this objective a function of, with r=rho, t=theta, z=zeta?
                                # i.e. if only a profile, it is "r" , while if all 3 coordinates it is "rtz"
        _units = "(T^4/m^2)"    # units of the output
        _print_value_fmt = "Quasi-symmetry error: "    # string with the name of the printed value, used when showing results of an optimization

        def __init__(
            self,
            eq,
            target=None,
            bounds=None,
            weight=1,
            normalize=True,
            normalize_target=True,
            grid=None,
            name="QS triple product",
            jac_chunk_size=None,
        ):
            # we don't have to do much here, mostly just call ``super().__init__()``
            if target is None and bounds is None:
                target = 0 # default target value
            self._grid = grid
            super().__init__(
                things=[eq], # things is a list of things that will be optimized, in this case just the equilibrium
                target=target,
                bounds=bounds,
                weight=weight,
                normalize=normalize,
                normalize_target=normalize_target,
                name=name,
                jac_chunk_size=jac_chunk_size
            )

        def build(self, use_jit=True, verbose=1):
            """Build constant arrays.

            Parameters
            ----------
            use_jit : bool, optional
                Whether to just-in-time compile the objective and derivatives.
            verbose : int, optional
                Level of output.

            """
            # things is the list of things that will be optimized,
            # we assigned things to be just eq in the init, so we know that the
            # first (and only) element of things is the equilibrium
            eq = self.things[0]
            # need some sensible default grid
            if self._grid is None:
                grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
            else:
                grid = self._grid
            # dim_f = size of the output vector returned by self.compute
            # usually the same as self.grid.num_nodes, unless you're doing some downsampling
            # or averaging etc.
            self._dim_f = self.grid.num_nodes
            # What data from desc.compute is needed? Here we want the QS triple product.
            self._data_keys = ["f_T"]

            # some helper code for profiling and logging
            timer = Timer()
            if verbose > 0:
                print("Precomputing transforms")
            timer.start("Precomputing transforms")

            # helper functions for building transforms etc to compute given
            # quantities. Alternatively, these can be created manually based on the
            # equilibrium, though in most cases that isn't necessary.
            profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
            transforms = get_transforms(self._data_keys, obj=eq, grid=grid)
            self._constants = {
                "transforms": transforms,
                "profiles": profiles,
            }

            timer.stop("Precomputing transforms")
            if verbose > 1:
                timer.disp("Precomputing transforms")


            # We try to normalize things to order(1) by dividing things by some
            # characteristic scale for a given quantity.
            # See ``desc.objectives.compute_scaling_factors`` for examples.
            if self._normalize:
                scales = compute_scaling_factors(eq)
                # since the objective has units of T^4/m^2, the normalization here is
                # based on a characteristic field strength and minor radius.
                self._normalization = (
                    scales["B"] ** 4 / scales["a"] ** 2
                )

            # finally, call ``super.build()``
            super().build(use_jit=use_jit, verbose=verbose)

        def compute(self, params, constants=None):
            """Signature should take params (or possibly multiple params, one for each thing in self.things),
               which is the params_dict of the expected thing(s) to be optimized.
               It also takes in constants, which is a dictionary of any other constant data needed to compute
               the objective, and is usually none by default so the self.constants are used.

            Parameters
            ----------
            params : dict
                Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
            constants : dict
                Dictionary of constant data, eg transforms, profiles etc. Defaults to
                self.constants

            Returns
            -------
            f : ndarray
                Quasi-symmetry flux function error at each node (T^4/m^2).

            """
            if constants is None:
                constants = self.constants

            # here we get the physics quantities from ``desc.compute.utils._compute``
            data = compute_fun(
                "desc.equilibrium.equilibrium.Equilibrium",
                self._data_keys,                 # quantities we want
                params=params,                   # params from input containing the equilibrium R_lmn, Z_lmn, etc
                transforms=self._transforms,     # transforms and profiles from self.build
                profiles=self._profiles,
            )
            # next we do any additional processing, such as combining things,
            # averaging, etc. Here we just return the QS triple product f_T evaluated at each
            # node in the grid.
            f = data["f_T"]
            # this is all we need to do here. Applying objective weights/targets/bounds
            # is handled by the base _Objective class, as well as the normalizations to be unitless
            # and to make the objective value independent of grid resolution.
            return f

    An example that is slightly more complex is shown below for computing the mirror ratio
    on each flux surface in the passed-in grid for an Equilibrium. (Some of the redundant comments
    from above are not repeated here)

::
    from desc.objectives.objective_funs import _Objective
    from desc.compute import get_profiles, get_transforms
    from desc.compute.utils import _compute as compute_fun
    from desc.grid import LinearGrid
    from desc.integrals.surface_integral import (
    surface_max,
    surface_min,
    )
    class MirrorRatio(_Objective):
        """Target a particular value mirror ratio.

        The mirror ratio is defined as:

        (Bₘₐₓ - Bₘᵢₙ) / (Bₘₐₓ + Bₘᵢₙ)

        Where Bₘₐₓ and Bₘᵢₙ are the maximum and minimum values of ||B|| on a given surface.
        Returns one value for each surface in ``grid``.

        Parameters
        ----------
        eq : Equilibrium or OmnigenousField
            Equilibrium or OmnigenousField that will be optimized to satisfy the Objective.
        grid : Grid, optional
            Collocation grid containing the nodes to evaluate at. Defaults to
            ``LinearGrid(M=eq.M_grid, N=eq.N_grid)`` for ``Equilibrium``
            or ``LinearGrid(theta=2*eq.M_B, N=2*eq.N_x)`` for ``OmnigenousField``.

        """

        __doc__ = __doc__.rstrip() + collect_docs(
            target_default="``target=0.2``.",
            bounds_default="``target=0.2``.",
        )

        _coordinates = "r"  # Because the mirror ratio is a function of flux surface (rho) alone, we set
                            # _coordinates="r"
        _units = "(dimensionless)"
        _print_value_fmt = "Mirror ratio: "

        def __init__(
            self,
            eq,
            *, # this just means all kwargs after this must be passed as kwargs, not as positional arguments
            grid=None,
            target=None,
            bounds=None,
            weight=1,
            normalize=True,
            normalize_target=True,
            loss_function=None,
            deriv_mode="auto",
            name="mirror ratio",
            jac_chunk_size=None,
        ):
            if target is None and bounds is None:
                target = 0.2 # default target value
            self._grid = grid
            super().__init__(
                things=eq,
                target=target,
                bounds=bounds,
                weight=weight,
                normalize=normalize,
                normalize_target=normalize_target,
                loss_function=loss_function,
                deriv_mode=deriv_mode,
                name=name,
                jac_chunk_size=jac_chunk_size,
            )

        def build(self, use_jit=True, verbose=1):
            """Build constant arrays.

            Parameters
            ----------
            use_jit : bool, optional
                Whether to just-in-time compile the objective and derivatives.
            verbose : int, optional
                Level of output.

            """
            eq = self.things[0]
            from desc.equilibrium import Equilibrium
            from desc.magnetic_fields import OmnigenousField

            # set defaults if grid is not passed in
            # Note that the grid has resolution in all three coordinates because
            # in order to compute mirror ratio (a flux surface quantity), we require
            # computation of |B| across the entire volume (i.e. poloidally and toroidally
            # on each flux surface) so that we can take the necessary min/maxes.
            if self._grid is None and isinstance(eq, Equilibrium):
                grid = LinearGrid( # default grid here only has rho=1.0 so a single flux surface, but the objective is written generally for arbitrary number of surfaces
                    M=eq.M_grid,
                    N=eq.N_grid,
                    NFP=eq.NFP,
                    sym=eq.sym,
                )
            elif self._grid is None and isinstance(eq, OmnigenousField):
                grid = LinearGrid( # we have a different default grid when an OmnigenousField is the object being optimized
                    theta=2 * eq.M_B,
                    N=2 * eq.N_x,
                    NFP=eq.NFP,
                )
            else:
                grid = self._grid

            # because the mirror ratio is a flux-surface quantity, we will
            # in the end only be returning an array of size grid.num_rho, which is
            # the number of flux surfaces in our grid (i.e. the number of unique rho
            # values in the grid)
            self._dim_f = grid.num_rho

            # we will only need "|B|" to compute this quantity. For this example objective
            # we compute mirror ratio manually in the objective compute method, but one may look at the List of Variables docs
            # and see that you could also compute "mirror ratio" as a key directly, but
            # for the sake of demonstrating functionality we compute it manually here
            self._data_keys = ["|B|"]

            timer = Timer()
            if verbose > 0:
                print("Precomputing transforms")
            timer.start("Precomputing transforms")

            profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
            transforms = get_transforms(self._data_keys, obj=eq, grid=grid)
            self._constants = {
                "transforms": transforms,
                "profiles": profiles,
            }

            timer.stop("Precomputing transforms")
            if verbose > 1:
                timer.disp("Precomputing transforms")

            super().build(use_jit=use_jit, verbose=verbose)

        def compute(self, params, constants=None):
            """Compute mirror ratio.

            Parameters
            ----------
            params : dict
                Dictionary of equilibrium or field degrees of freedom,
                eg Equilibrium.params_dict
            constants : dict
                Dictionary of constant data, eg transforms, profiles etc. Defaults to
                self.constants

            Returns
            -------
            M : ndarray
                Mirror ratio on each surface.

            """
            if constants is None:
                constants = self.constants

            # we use this compute_fun to compute quantities inside of our
            # objective functions, as opposed to using `eq.compute`, because
            # we jit our objective compute functions and some logic inside of
            # `eq.compute` does not work under jit.
            data = compute_fun(
                # parameterization is the object (or type of object) that we are computing quantities with i.e. ``desc.equilibrium.equilibrium.Equilibrium`` or just the Equilibrium object directly
                parameterization=self.things[0],
                # names is the names of the data (see List of Variables doc page) we want to compute
                names=self._data_keys,
                # params is the dict of the DOFs of the parameterization, and are used to compute these quantities, i.e. for Equilibrium this contains R_lmn, Z_lmn, etc.
                params=params,
                # transforms and profiles pre-computed from the build method
                transforms=constants["transforms"],
                profiles=constants["profiles"],
            )
            # now data is a dictionary containing the key "|B|", the magnetic field evaluated at our grid
            # which is accessible through constants["transforms"]["grid"]


            # compute max and min of |B| on each flux surface in the grid
            ## we have utility functions which, given a quantity computed on a grid,
            ## return the max or min of that quantity on each coordinate surface
            max_tz_B = surface_max(
                grid=constants["transforms"]["grid"], x=data["|B|"], surface_label="rho"
            )  # also can be computed directly as "max_tz |B|" in our list of variables
            min_tz_B = surface_min(
                grid=constants["transforms"]["grid"], x=data["|B|"], surface_label="rho"
            )  # also can be computed directly as "min_tz |B|" in our list of variables

            # to avoid issues with array shapes, these two arrays (max_tz_B and min_tz_B)
            # are still the same shape as data["|B|"] i.e. still are 1-D arrays of length grid.num_nodes,
            # (i.e. there are data corresponding to nodes (rho, theta, zeta) = (1.0, 0, 0) and (1.0, pi, 0),
            # which have the same max_tz_B). This is useful if we, for instance, wanted to  multiply
            # a flux-surface quantity like max_tz_B with a non-flux-surface quantity like sqrt(g)
            # without needing to worry about shape mismatches.

            # However, since we only need these quantities on each flux surface from
            # here on out, we can use the grid.compress function to reduce these quantities
            # down to just the values at each unique rho surface
            max_tz_B = constants["transforms"]["grid"].compress(
                max_tz_B, surface_label="rho"
            )
            min_tz_B = constants["transforms"]["grid"].compress(
                min_tz_B, surface_label="rho"
            )
            # now max_tz_B and min_tz_B are just 1-D arrays of size grid.num_rho

            # Finally, compute the mirror ratio using the above max/min on each flux surface
            mirror_ratio = (max_tz_B - min_tz_B) / (min_tz_B + max_tz_B)

            return mirror_ratio # return the value of the objective



Converting to Cartesian coordinates
-----------------------------------

The above example of quasi-symmetry is a scalar quantity that is independent of the
coordinate system. ``desc.compute.utils._compute`` always returns all vector quantities
in toroidal coordinates :math:`(R,\phi,Z)`. If you would prefer to work in Cartesian
coordinates :math:`(X,Y,Z)` for any intermediate computations within your new objective,
you will have to manually convert these vectors using the geometry utility functions
``rpz2xyz`` and/or ``rpz2xyz_vec``. See the ``PlasmaVesselDistance`` objective for an
example of this.

Adapting Existing Objectives with Different Loss Functions
----------------------------------------------------------

If your desired objective is already implemented in DESC, but not in the correct form,
a few different loss functions are available through the ``loss_function`` kwarg
when instantiating an Objective, to modify the objective cost in order to adapt
the objective to your desired purpose. For example, the DESC ``RotationalTransform``
objective with ``target=iota_target`` by default forms the residual by taking the target
and subtracting it from the profile at the points in the grid, resulting in a residual
of the form :math:`\iota_{err} = \sum_{i} (\iota_i - iota_{target})^2`, i.e. the residual
is the sum of squared pointwise error between the current rotational transform profile
and the target passed into the objective. If the desired objective instead is to
optimize to target an average rotational transform of `iota_target`, we can adapt the
``RotationalTransform`` object by passing in ``loss_function="mean"``. The options
available for the ``loss_function`` kwarg are ``[None,"mean","min","max"]``, with
``None`` meaning using the usual default objective cost, while ``"mean"`` takes the
average of the raw objective values (before subtracting the target/bounds or
normalization), ``"min"`` takes the minimum, and ``"max"`` takes the maximum.
