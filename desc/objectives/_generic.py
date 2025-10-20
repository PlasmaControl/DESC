"""Generic objectives that don't belong anywhere else."""

import re

import numpy as np

from desc.backend import jnp, tree_flatten, tree_leaves, tree_unflatten
from desc.compute import data_index
from desc.compute.utils import _compute as compute_fun
from desc.compute.utils import _parse_parameterization, get_profiles, get_transforms
from desc.grid import QuadratureGrid
from desc.optimizable import OptimizableCollection
from desc.utils import errorif, getsource, jaxify, parse_argname_change, setdefault

from .linear_objectives import _FixedObjective
from .objective_funs import _Objective, collect_docs


class ExternalObjective(_Objective):
    """Wrap an external code.

    Similar to ``ObjectiveFromUser``, except derivatives of the objective function are
    computed with finite differences instead of AD. The function does not need to be
    JAX transformable.

    The user supplied function must take an Equilibrium or a list of Equilibria as its
    only positional argument, but can take additional keyword arguments.
    It must return a single 1D array of floats.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    fun : callable
        External objective function. It must take an Equilibrium or list of Equilibria
        as its only positional argument, but can take additional keyword arguments.
        It does not need to be JAX transformable.
    dim_f : int
        Dimension of the output of ``fun``.
    fun_kwargs : dict, optional
        Keyword arguments that are passed as inputs to ``fun``.
    vectorized : bool
        Set to False if ``fun`` takes a single Equilibrium as its positional argument.
        Set to True if ``fun`` instead takes a list of Equilibria.
    abs_step : float, optional
        Absolute finite difference step size. Default = 1e-4.
        Total step size is ``abs_step + rel_step * mean(abs(x))``.
    rel_step : float, optional
        Relative finite difference step size. Default = 0.
        Total step size is ``abs_step + rel_step * mean(abs(x))``.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )
    __doc__ += """
    Examples
    --------
    .. code-block:: python

        from desc.io import load

        def myfun(eq, path=""):
            # This will return the compute quantity '<beta>_vol',
            # but uses I/O operations that are not JAX transformable.
            eq.save(path)
            eq = load(path)
            data = eq.compute("<beta>_vol")
            return data["<beta>_vol"]

        myobj = ExternalObjective(
            eq=eq, fun=myfun, dim_f=1, fun_kwargs={"path": "temp.h5"}, vectorized=False,
        )

    """

    _units = "(Unknown)"
    _print_value_fmt = "External objective value: "
    _static_attrs = _Objective._static_attrs + [
        "_fun",
        "_fun_kwargs",
        "_fun_wrapped",
        "_vectorized",
    ]

    def __init__(
        self,
        eq,
        *,
        fun,
        dim_f,
        fun_kwargs={},
        vectorized=False,
        abs_step=1e-4,
        rel_step=0,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        loss_function=None,
        name="external",
    ):
        if target is None and bounds is None:
            target = 0
        self._eq = eq.copy()
        self._fun = fun
        self._fun_kwargs = fun_kwargs
        self._dim_f = dim_f
        self._vectorized = vectorized
        self._abs_step = abs_step
        self._rel_step = rel_step
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode="fwd",
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
        self._scalar = self._dim_f == 1
        self._constants = {"quad_weights": 1.0}

        def fun_wrapped(params):
            """Wrap external function with possibly vectorized params."""
            # number of equilibria for vectorized computations
            param_shape = params["Psi"].shape
            num_eq = param_shape[0] if len(param_shape) > 1 else 1

            # convert params to list of equilibria
            eqs = [self._eq.copy() for _ in range(num_eq)]
            for k, eq in enumerate(eqs):
                # update equilibria with new params
                for param_key in self._eq.optimizable_params:
                    param_value = np.atleast_2d(params[param_key])[k, :]
                    if len(param_value):
                        setattr(eq, param_key, param_value)

            # call external function on equilibrium or list of equilibria
            if not self._vectorized:
                eqs = eqs[0]
            return self._fun(eqs, **self._fun_kwargs)

        # wrap external function to work with JAX
        abstract_eval = lambda *args, **kwargs: jnp.empty(self._dim_f)
        self._fun_wrapped = jaxify(
            fun_wrapped,
            abstract_eval,
            vectorized=self._vectorized,
            abs_step=self._abs_step,
            rel_step=self._rel_step,
        )

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute the quantity.

        Parameters
        ----------
        params : list of dict
            List of dictionaries of degrees of freedom, eg CoilSet.params_dict
        constants : dict
            Unused by this Objective.

        Returns
        -------
        f : ndarray
            Computed quantity.

        """
        f = self._fun_wrapped(params)
        return f


class GenericObjective(_Objective):
    """A generic objective that can compute any quantity from the `data_index`.

    Note that the grid passed-in should be the grid that is required to compute
    the quantity. For example, "mirror ratio" is a flux-surface quantity which
    depends on the max and min field strength on the flux surface. This requires
    knowledge of the magnetic field magnitude on the whole flux surface, so the
    passed-in grid must have poloidal and toroidal resolution, in addition to
    whatever radial surfaces are desired. The same thing applies for quantities
    needing flux surface averages, such as anything depending on iota
    for a current-constrained equilibrium.

    Also note that if a quantity is only a function of flux surface rho
    (like "mirror ratio"), ``GenericObjective`` will detect this and only return
    the unique values, one per flux surface, instead of the values corresponding to
    every node of the passed-in grid (which may be 3-D as explained above).

    Finally, this objective is intended for quantities computed in native DESC flux
    coordinates (rho, theta, zeta). For quantities which require more complicated
    transformations and calculations in other coordinate systems
    (such as "Gamma_c"), this objective cannot be used and it is recommended to use
    the dedicated objectives for those quantities.

    Parameters
    ----------
    f : str
        Name of the quantity to compute.
    thing : Optimizable
        Object that will be optimized to satisfy the Objective.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at. Defaults to
        ``QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid)`` if thing is an Equilibrium.
    compute_kwargs : dict
        Optional keyword arguments passed to core compute function, eg ``helicity``.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )
    __doc__ += """
    Examples
    --------

    For examples, see the Advanced QS Optimization notebook in the documentation, or
    the documentation page for adding new objective functions.

    """

    _print_value_fmt = "Generic objective value: "
    _static_attrs = _Objective._static_attrs + ["_compute_kwargs", "f", "_p"]

    def __init__(
        self,
        f,
        thing,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="Generic",
        jac_chunk_size=None,
        compute_kwargs=None,
        **kwargs,
    ):
        errorif(
            isinstance(thing, OptimizableCollection),
            NotImplementedError,
            "thing must be of type Optimizable and not OptimizableCollection.",
        )
        thing = parse_argname_change(thing, kwargs, "eq", "thing")
        if target is None and bounds is None:
            target = 0
        self.f = f
        self._grid = grid
        self._compute_kwargs = setdefault(compute_kwargs, {})
        super().__init__(
            things=thing,
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
        self._print_value_fmt = f"{name} objective value: "
        self._p = _parse_parameterization(thing)
        self._scalar = not bool(data_index[self._p][self.f]["dim"])
        self._coordinates = data_index[self._p][self.f]["coordinates"]
        self._units = "(" + data_index[self._p][self.f]["units"] + ")"

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        thing = self.things[0]
        if self._grid is None:
            errorif(
                self._p != "desc.equilibrium.equilibrium.Equilibrium",
                ValueError,
                "grid must be supplied for things besides an Equilibrium.",
            )
            grid = QuadratureGrid(thing.L_grid, thing.M_grid, thing.N_grid, thing.NFP)
        else:
            grid = self._grid

        if data_index[self._p][self.f]["dim"] == 0:
            self._dim_f = 1
        elif data_index[self._p][self.f]["coordinates"] == "r":
            self._dim_f = grid.num_rho
        else:
            self._dim_f = grid.num_nodes * np.prod(data_index[self._p][self.f]["dim"])
        profiles = get_profiles(self.f, obj=thing, grid=grid)
        transforms = get_transforms(
            self.f, obj=thing, grid=grid, **self._compute_kwargs
        )
        self._constants = {
            "transforms": transforms,
            "profiles": profiles,
        }
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute the quantity.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc.
            Defaults to self.constants

        Returns
        -------
        f : ndarray
            Computed quantity.

        """
        if constants is None:
            constants = self.constants
        data = compute_fun(
            self._p,
            self.f,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
            **self._compute_kwargs,
        )
        f = data[self.f]
        if self._coordinates == "r":
            f = constants["transforms"]["grid"].compress(f, surface_label="rho")
        return f.flatten(order="F")  # so that default quad weights line up correctly.


class LinearObjectiveFromUser(_FixedObjective):
    """Wrap a user defined linear objective function.

    The user supplied function should take one argument, ``params``, which is a
    dictionary of parameters of an Optimizable "thing".

    The function should be JAX traceable and differentiable, and should return a single
    JAX array. The source code of the function must be visible to the ``inspect`` module
    for parsing.

    Parameters
    ----------
    fun : callable
        Custom objective function.
    thing : Optimizable
        Object whose degrees of freedom are being constrained.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )
    __doc__ += """
    Examples
    --------

    For example use, see the Omnigenity Optimization notebook,
    the Advanced QS Optimization notebook, or
    the documentation page for adding new objective functions.

    """

    _static_attrs = _Objective._static_attrs + ["_fun"]

    _scalar = False
    _linear = True
    _fixed = True
    _units = "(Unknown)"
    _print_value_fmt = "Custom linear objective value: "

    def __init__(
        self,
        fun,
        thing,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        name="custom linear",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0
        self._fun = fun
        super().__init__(
            things=thing,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
            jac_chunk_size=jac_chunk_size,
        )

    def build(self, use_jit=False, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        thing = self.things[0]

        import jax

        self._dim_f = jax.eval_shape(self._fun, thing.params_dict).size

        # check that fun is linear
        params, params_tree = tree_flatten(thing.params_dict)
        for param in params:
            param += np.random.rand(param.size) * 10
        params = tree_unflatten(params_tree, params)
        J1 = jax.jacrev(self._fun)(thing.params_dict)
        J2 = jax.jacrev(self._fun)(params)
        for j1, j2 in zip(tree_leaves(J1), tree_leaves(J2)):
            assert np.all(j1 == j2), "Function must be linear!"

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute fixed degree of freedom errors.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg thing.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Fixed degree of freedom errors.

        """
        f = self._fun(params)
        return f


class ObjectiveFromUser(_Objective):
    """Wrap a user defined objective function.

    The user supplied function should take two arguments: ``grid`` and ``data``.

    ``grid`` is the Grid object containing the nodes where the data is computed.

    ``data`` is a dictionary of values with keys from the list of `variables`_. Values
    will be the given data evaluated at ``grid``.

    The function should be JAX traceable and differentiable, and should return a single
    JAX array. The source code of the function must be visible to the ``inspect`` module
    for parsing.

    .. _variables: https://desc-docs.readthedocs.io/en/stable/variables.html

    Parameters
    ----------
    fun : callable
        Custom objective function.
    thing : Optimizable
        Object that will be optimized to satisfy the Objective.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at. Defaults to
        ``QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid)`` if thing is an Equilibrium.
    compute_kwargs : dict
        Optional keyword arguments passed to core compute function, eg ``helicity``.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )
    __doc__ += """
    Examples
    --------
    .. code-block:: python

        from desc.integrals.surface_integral import surface_averages

        def myfun(grid, data):
            # This will compute the flux surface average of the function
            # R*B_T from the Grad-Shafranov equation
            f = data['R'] * data['B_phi']
            # q here is the kwarg for "quantity" as in, quantity to be averaged
            f_fsa = surface_averages(grid, q=f, sqrt_g=data['sqrt(g)'])
            # this has the FSA values on the full grid,
            # but we just want the unique values:
            return grid.compress(f_fsa)

        myobj = ObjectiveFromUser(fun=myfun, thing=eq)

    """

    _units = "(Unknown)"
    _print_value_fmt = "Custom objective value: "
    _static_attrs = _Objective._static_attrs + [
        "_compute_kwargs",
        "_fun",
        "_fun_wrapped",
        "_p",
    ]

    def __init__(
        self,
        fun,
        thing,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="Custom",
        jac_chunk_size=None,
        compute_kwargs=None,
        **kwargs,
    ):
        errorif(
            isinstance(thing, OptimizableCollection),
            NotImplementedError,
            "thing must be of type Optimizable and not OptimizableCollection.",
        )
        thing = parse_argname_change(thing, kwargs, "eq", "thing")
        if target is None and bounds is None:
            target = 0
        self._fun = fun
        self._grid = grid
        self._print_value_fmt = f"{name} objective value: "

        self._compute_kwargs = setdefault(compute_kwargs, {})
        super().__init__(
            things=thing,
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
        self._p = _parse_parameterization(thing)

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        import jax

        thing = self.things[0]
        if self._grid is None:
            errorif(
                self._p != "desc.equilibrium.equilibrium.Equilibrium",
                ValueError,
                "grid must be supplied for things besides an Equilibrium.",
            )
            grid = QuadratureGrid(thing.L_grid, thing.M_grid, thing.N_grid, thing.NFP)
        else:
            grid = self._grid

        def get_vars(fun):
            pattern = r"data\[(.*?)\]"
            src = getsource(fun)
            variables = re.findall(pattern, src)
            variables = list({s.strip().strip("'").strip('"') for s in variables})
            return variables

        self._data_keys = get_vars(self._fun)
        dummy_data = {}
        for key in self._data_keys:
            assert key in data_index[self._p], f"Don't know how to compute {key}."
            if data_index[self._p][key]["dim"] == 0:
                dummy_data[key] = jnp.array(0.0)
            else:
                dummy_data[key] = jnp.empty(
                    (grid.num_nodes, data_index[self._p][key]["dim"])
                ).squeeze()

        self._fun_wrapped = lambda data: self._fun(grid, data)

        self._dim_f = jax.eval_shape(self._fun_wrapped, dummy_data).size
        self._scalar = self._dim_f == 1
        profiles = get_profiles(self._data_keys, obj=thing, grid=grid)
        transforms = get_transforms(
            self._data_keys, obj=thing, grid=grid, **self._compute_kwargs
        )
        self._constants = {
            "transforms": transforms,
            "profiles": profiles,
            "quad_weights": 1.0,
        }

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute the quantity.

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
            Computed quantity.

        """
        if constants is None:
            constants = self.constants
        data = compute_fun(
            self._p,
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
            **self._compute_kwargs,
        )
        f = self._fun_wrapped(data)
        return f
