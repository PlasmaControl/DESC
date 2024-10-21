==============================
Adding new objective functions
==============================

This guide walks through creating a new objective to optimize using Quasi-symmetry as
an example. The primary methods needed for a new objective are ``__init__``, ``build``,
and ``compute``. The base class ``_Objective`` provides a number of other methods that
generally do not need to be re-implemented for subclasses.

``__init__`` should generally just assign attributes and store inputs. It should not do
any expensive calculations, these should be in ``build`` or ``compute``. The main arguments
are summarized in the example below.

``build`` is called before optimization with the ``Equilibrium`` to be optimized.
It is used to precompute things like transform matrices that convert between spectral
coefficients and real space values.
In the build method, we first ensure that a ``Grid`` is assigned, using default values
from the equilibrium if necessary. The grid defines the points in flux coordinates where
we evaluate the residuals.
Next, we define the physics quantities we need to evaluate the objective (``_data_keys``),
and the number of residuals that will be returned by ``compute`` (``_dim_f``).
Next, we use some helper functions to build the required ``Tranform`` and ``Profile``
objects needed to compute the desired physics quantities.
Finally, we call the base class ``build`` method to do some checking of array sizes and
other misc. stuff.

``compute`` is where the actual calculation of the residual takes place. Objectives
generally return a vector of residuals that are minimized in a least squares sense, though
the exact method will depend on the optimization algorithm. The main thing here is
calling ``compute_fun`` to get physics quantities, and then performing any post-processing
we want such as averaging, combining, etc. The final step is to call ``self._shift_scale``
which subtracts out the target and applies weighting and normalizations.

A full example objective with comments describing key points is given below:
::

    from desc.objectives.objective_funs import _Objective
    from desc.objectives.normalization import compute_scaling_factors
    from desc.compute.utils import get_params, get_profiles, get_transforms
    from desc.compute import compute as compute_fun


    class QuasisymmetryTripleProduct(_Objective):  # need to subclass from ``desc.objectives._Objective``
        """Give a description of what it is and what it's useful for.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        target : float, ndarray, optional
            Target value(s) of the objective.
            len(target) must be equal to Objective.dim_f
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            len(weight) must be equal to Objective.dim_f
        normalize : bool
            Whether to compute the error in physical units or non-dimensionalize.
        normalize_target : bool
            Whether target should be normalized before comparing to computed values.
            if `normalize` is `True` and the target is in physical units, this should also
            be set to True.
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.
        name : str
            Name of the objective function.

        """

        _scalar = False         # does self.compute return a scalar or vector?
        _linear = False         # is self.compute a linear function of its parameters?
        _units = "(T^4/m^2)"    # units of the output
        _print_value_fmt = "Quasi-symmetry error: {:10.3e} "  # string with python string formatting for printing the value

        def __init__(
            self,
            eq=None,
            target=0,
            weight=1,
            normalize=True,
            normalize_target=True,
            grid=None,
            name="QS triple product",
        ):

            # we don't have to do much here, mostly just call ``super().__init__()``
            # to inherit common initialization logic from ``desc.objectives._Objective``
            self.grid = grid
            super().__init__(
                eq=eq,
                target=target,
                weight=weight,
                normalize=normalize,
                normalize_target=normalize_target,
                name=name,
            )

        def build(self, eq, use_jit=True, verbose=1):
            """Build constant arrays.

            Parameters
            ----------
            eq : Equilibrium, optional
                Equilibrium that will be optimized to satisfy the Objective.
            use_jit : bool, optional
                Whether to just-in-time compile the objective and derivatives.
            verbose : int, optional
                Level of output.

            """
            # need some sensible default grid
            if self.grid is None:
                self.grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)

            # dim_f = size of the output vector returned by self.compute
            # self.compute refers to the objective's own compute method
            # Typically an objective returns the output of a quantity computed in
            # ``desc.compute``, with some additional scale factor.
            # In these cases dim_f should match the size of the quantity calculated in
            # ``desc.compute`` (for example self.grid.num_nodes).
            # If the objective does post-processing on the quantity, like downsampling or
            # averaging, then dim_f should be changed accordingly.
            # What data from desc.compute is needed? Here we want the QS triple product.
            self._data_keys = ["f_T"]
            # what arguments should be passed to self.compute
            self._args = get_params(self._data_keys)

            # some helper code for profiling and logging
            timer = Timer()
            if verbose > 0:
                print("Precomputing transforms")
            timer.start("Precomputing transforms")

            # helper functions for building transforms etc to compute given
            # quantities. Alternatively, these can be created manually based on the
            # equilibrium, though in most cases that isn't necessary.
            self._profiles = get_profiles(self._data_keys, eq=eq, grid=self.grid)
            self._transforms = get_transforms(self._data_keys, eq=eq, grid=self.grid)

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
                # we also divide by the square root of number of residuals to keep
                # things roughly independent of the grid resolution.
                self._normalization = (
                    scales["B"] ** 4 / scales["a"] ** 2 / jnp.sqrt(self._dim_f)
                )

            # finally, call ``super.build()``
            super().build(eq=eq, use_jit=use_jit, verbose=verbose)

        def compute(self, *args, **kwargs):
            """Signature should only take args and kwargs, but you can use the Parameters
            block below to specify what these should be.

            Parameters
            ----------
            R_lmn : ndarray
                Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
            Z_lmn : ndarray
                Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).
            L_lmn : ndarray
                Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
            i_l : ndarray
                Spectral coefficients of iota(rho) -- rotational transform profile.
            c_l : ndarray
                Spectral coefficients of I(rho) -- toroidal current profile.
            Psi : float
                Total toroidal magnetic flux within the last closed flux surface (Wb).

            Returns
            -------
            f : ndarray
                Quasi-symmetry flux function error at each node (T^4/m^2).

            """
            # this parses the inputs into a dictionary expected by ``desc.compute.compute``
            params = self._parse_args(*args, **kwargs)

            # here we get the physics quantities from ``desc.compute.compute``
            data = compute_fun(
                self._data_keys,                 # quantities we want
                params=params,                   # params from previous line
                transforms=self._transforms,     # transforms and profiles from self.build
                profiles=self._profiles,
            )
            # next we do any additional processing, such as combining things,
            # averaging, etc. Here we just scale things by the quadrature weights from
            # the grid to make things roughly independent of the grid resolution.
            f = data["f_T"] * self.grid.weights

            # finally, we call ``self._shift_scale`` here to subtract out the target and
            # apply weighing and normalizations.
            return self._shift_scale(f)
