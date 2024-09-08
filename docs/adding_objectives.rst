Adding new objective functions
------------------------------

This guide walks through creating a new objective to optimize using Quasi-symmetry as
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
post-processing we want such as averaging, combining, etc.

A full example objective with comments describing the key points is given below:
::

    from desc.objectives.objective_funs import _Objective
    from desc.objectives.normalization import compute_scaling_factors
    from desc.compute import get_profiles, get_transforms
    from desc.compute.utils import _compute as compute_fun


    class QuasisymmetryTripleProduct(_Objective):  # need to subclass from ``desc.objectives._Objective``
        """Give a description of what it is and what it's useful for.

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium that will be optimized to satisfy the Objective.
        target : {float, ndarray}, optional
            Target value(s) of the objective. Only used if bounds is None.
            Must be broadcastable to Objective.dim_f.
        bounds : tuple of {float, ndarray}, optional
            Lower and upper bounds on the objective. Overrides target.
            Both bounds must be broadcastable to Objective.dim_f
        weight : {float, ndarray}, optional
            Weighting to apply to the Objective, relative to other Objectives.
            Must be broadcastable to Objective.dim_f
        normalize : bool, optional
            Whether to compute the error in physical units or non-dimensionalize.
        normalize_target : bool, optional
            Whether target and bounds should be normalized before comparing to computed
            values. If `normalize` is `True` and the target is in physical units,
            this should also be set to True.
        loss_function : {None, 'mean', 'min', 'max'}, optional
            Loss function to apply to the objective values once computed. This loss function
            is called on the raw compute value, before any shifting, scaling, or
            normalization.
        grid : Grid, optional
            Collocation grid containing the nodes to evaluate at.
        name : str, optional
            Name of the objective function.

        """

        _coordinates = "rtz"    # What coordinates is this objective a function of, with r=rho, t=theta, z=zeta?
                                # i.e. if only a profile, it is "r" , while if all 3 coordinates it is "rtz"
        _units = "(T^4/m^2)"    # units of the output
        _print_value_fmt = "Quasi-symmetry error: "    # string with python string formatting for printing the value

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
a few different loss functions are available through the the ``loss_function`` kwarg
when instantiating an Objective objective to modify the objective cost in order to adapt
the objective to your desired purpose. For example, the DESC ``RotationalTransform``
objective with ``target=iota_target`` by default forms the residual by taking the target
and subtracting it from the profile at the points in the grid, resulting in a residual
of the form :math:`\iota_{err} = \sum_{i} (\iota_i - iota_target)^2`, i.e. the residual
is the sum of squared pointwise error between the current rotational transform profile
and the target passed into the objective. If the desired objective instead is to
optimize to target an average rotational transform of `iota_target`, we can adapt the
``RotationalTransform`` object by passing in ``loss_function="mean"``. The options
available for the ``loss_function`` kwarg are ``[None,"mean","min","max"]``, with
``None`` meaning using the usual default objective cost, while ``"mean"`` takes the
average of the raw objective values (before subtracting the target/bounds or
normalization), ``"min"`` takes the minimum, and ``"max"`` takes the maximum.
