"""Generic objectives that don't belong anywhere else."""

import functools
import inspect
import multiprocessing
import os
import re
from abc import ABC

import numpy as np

from desc.backend import jnp
from desc.compute import data_index
from desc.compute.utils import _compute as compute_fun
from desc.compute.utils import get_profiles, get_transforms
from desc.grid import QuadratureGrid

from .linear_objectives import _FixedObjective
from .objective_funs import _Objective


class _ExternalObjective(_Objective, ABC):
    """Wrap an external code.

    Similar to ``ObjectiveFromUser``, except derivatives of the objective function are
    computed with finite differences instead of AD.

    The user supplied function must take an Equilibrium as its only positional argument,
    but can take additional keyword arguments.

    # TODO: add Parameters documentation

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    fun : callable
        Custom objective function.
    dim_f : int
        Dimension of the output of fun.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f. Defaults to ``target=0``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f.
        Defaults to ``target=0``.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
        Has no effect for this objective.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This loss function
        is called on the raw compute value, before any shifting, scaling, or
        normalization.
    name : str, optional
        Name of the objective function.

    # TODO: add example

    """

    _units = "(Unknown)"
    _print_value_fmt = "External objective value: {:10.3e}"

    def __init__(
        self,
        eq,
        fun,
        dim_f,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        loss_function=None,
        fd_step=1e-4,  # TODO: generalize this to allow a vector of different scales
        vectorized=False,
        name="external",
        **kwargs,
    ):
        if target is None and bounds is None:
            target = 0
        self._eq = eq.copy()
        self._fun = fun
        self._dim_f = dim_f
        self._fd_step = fd_step
        self._vectorized = vectorized
        self._kwargs = kwargs
        if self._vectorized:
            try:  # spawn a new environment so the backend can be set to numpy
                multiprocessing.set_start_method("spawn")
            except RuntimeError:  # context can only be set once
                pass
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

            if self._vectorized and num_eq > 1:
                # convert params to list of equilibria
                eqs = [self._eq.copy() for _ in range(num_eq)]
                for k, eq in enumerate(eqs):
                    # update equilibria with new params
                    for param_key in self._eq.optimizable_params:
                        param_value = np.array(params[param_key][k, :])
                        if len(param_value):
                            setattr(eq, param_key, param_value)
                # parallelize calls to external function
                with multiprocessing.Pool(
                    processes=min(os.cpu_count(), num_eq)
                ) as pool:
                    results = pool.map(
                        functools.partial(self._fun, **self._kwargs), eqs
                    )
                    pool.join()
                    return jnp.vstack(results, dtype=float)
            else:  # no vectorization
                # update equilibrium with new params
                for param_key in self._eq.optimizable_params:
                    param_value = params[param_key]
                    if len(param_value):
                        setattr(self._eq, param_key, param_value)
                return self._fun(self._eq, **self._kwargs)

        # wrap external function to work with JAX
        abstract_eval = lambda *args, **kwargs: jnp.empty(self._dim_f)
        self._fun_wrapped = self._jaxify(fun_wrapped, abstract_eval)

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute the quantity.

        Parameters
        ----------
        params : list of dict
            List of dictionaries of degrees of freedom, eg CoilSet.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Computed quantity.

        """
        f = self._fun_wrapped(params)
        return f

    def _jaxify(self, func, abstract_eval):
        """Make an external (python) function work with JAX.

        Positional arguments to func can be differentiated,
        use keyword args for static values and non-differentiable stuff.

        Note: Only forward mode differentiation is supported currently.

        Parameters
        ----------
        func : callable
            Function to wrap. Should be a "pure" function, in that it has no side
            effects and doesn't maintain state. Does not need to be JAX transformable.
        abstract_eval : callable
            Auxilliary function that computes the output shape and dtype of func.
            **Must be JAX transformable**. Should be of the form

                abstract_eval(*args, **kwargs) -> Pytree with same shape and dtype as
                func(*args, **kwargs)

            For example, if func always returns a scalar:

                abstract_eval = lambda *args, **kwargs: jnp.array(1.)

            Or if func takes an array of shape(n) and returns a dict of arrays of
            shape(n-2):

                abstract_eval = lambda arr, **kwargs:
                {"out1": jnp.empty(arr.size-2), "out2": jnp.empty(arr.size-2)}

        Returns
        -------
        func : callable
            New function that behaves as func but works with jit/vmap/jacfwd etc.

        """
        import jax

        def wrap_pure_callback(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                result_shape_dtype = abstract_eval(*args, **kwargs)
                return jax.pure_callback(
                    func,
                    result_shape_dtype,
                    *args,
                    vectorized=self._vectorized,
                    **kwargs,
                )

            return wrapper

        def define_fd_jvp(func):
            func = jax.custom_jvp(func)

            @func.defjvp
            def func_jvp(primals, tangents):
                primal_out = func(*primals)

                # flatten everything into 1D vectors for easier finite differences
                y, unflaty = jax.flatten_util.ravel_pytree(primal_out)
                x, unflatx = jax.flatten_util.ravel_pytree(primals)
                v, _______ = jax.flatten_util.ravel_pytree(tangents)
                # scale to unit norm if nonzero
                normv = jnp.linalg.norm(v)
                vh = jnp.where(normv == 0, v, v / normv)

                def f(x):
                    return jax.flatten_util.ravel_pytree(func(*unflatx(x)))[0]

                tangent_out = (f(x + self._fd_step * vh) - y) / self._fd_step * normv
                tangent_out = unflaty(tangent_out)

                return primal_out, tangent_out

            return func

        return define_fd_jvp(wrap_pure_callback(func))


class GenericObjective(_Objective):
    """A generic objective that can compute any quantity from the `data_index`.

    Parameters
    ----------
    f : str
        Name of the quantity to compute.
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f. Defaults to ``target=0``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f.
        Defaults to ``target=0``.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f.
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
        Has no effect for this objective
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True. Note: Has no effect on this objective.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This loss function
        is called on the raw compute value, before any shifting, scaling, or
        normalization.
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid)``.
    name : str, optional
        Name of the objective function.

    """

    _print_value_fmt = "Generic objective value: {:10.3e} "

    def __init__(
        self,
        f,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="generic",
    ):
        if target is None and bounds is None:
            target = 0
        self.f = f
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
        )
        self._scalar = not bool(
            data_index["desc.equilibrium.equilibrium.Equilibrium"][self.f]["dim"]
        )
        self._coordinates = data_index["desc.equilibrium.equilibrium.Equilibrium"][
            self.f
        ]["coordinates"]
        self._units = (
            "("
            + data_index["desc.equilibrium.equilibrium.Equilibrium"][self.f]["units"]
            + ")"
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
        if self._grid is None:
            grid = QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid, eq.NFP)
        else:
            grid = self._grid

        p = "desc.equilibrium.equilibrium.Equilibrium"
        if data_index[p][self.f]["dim"] == 0:
            self._dim_f = 1
        elif data_index[p][self.f]["coordinates"] == "r":
            self._dim_f = grid.num_rho
        else:
            self._dim_f = grid.num_nodes * np.prod(data_index[p][self.f]["dim"])
        profiles = get_profiles(self.f, obj=eq, grid=grid)
        transforms = get_transforms(self.f, obj=eq, grid=grid)
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
            "desc.equilibrium.equilibrium.Equilibrium",
            self.f,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
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
    target : dict of {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f. Defaults to ``target=0``.
    bounds : tuple of dict {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f.
        Defaults to ``target=0``.
    weight : dict of {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Should be a scalar or have the same tree structure as thing.params.
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
        Has no effect for this objective.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True. Has no effect for this objective.
    name : str, optional
        Name of the objective function.

    """

    _scalar = False
    _linear = True
    _fixed = True
    _units = "(Unknown)"
    _print_value_fmt = "Custom linear objective value: {:10.3e}"

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
        J1 = jax.jacrev(self._fun)(thing.params_dict)
        params = thing.params_dict.copy()
        for key, value in params.items():
            params[key] = value + np.random.rand(value.size) * 10
        J2 = jax.jacrev(self._fun)(params)
        for key in J1.keys():
            assert np.all(J1[key] == J2[key]), "Function must be linear!"

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
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f. Defaults to ``target=0``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f.
        Defaults to ``target=0``.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
        Has no effect for this objective.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This loss function
        is called on the raw compute value, before any shifting, scaling, or
        normalization.
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid)``.
    name : str, optional
        Name of the objective function.


    Examples
    --------
    .. code-block:: python

        from desc.compute.utils import surface_averages
        def myfun(grid, data):
            # This will compute the flux surface average of the function
            # R*B_T from the Grad-Shafranov equation
            f = data['R'] * data['B_phi']
            f_fsa = surface_averages(grid, f, sqrt_g=data['sqrt_g'])
            # this is the FSA on the full grid, but we only want the unique values:
            return grid.compress(f_fsa)

        myobj = ObjectiveFromUser(myfun)

    """

    _units = "(Unknown)"
    _print_value_fmt = "Custom objective value: {:10.3e}"

    def __init__(
        self,
        fun,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="custom",
    ):
        if target is None and bounds is None:
            target = 0
        self._fun = fun
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
        import jax

        eq = self.things[0]
        if self._grid is None:
            grid = QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid, eq.NFP)
        else:
            grid = self._grid

        def get_vars(fun):
            pattern = r"data\[(.*?)\]"
            src = inspect.getsource(fun)
            variables = re.findall(pattern, src)
            variables = list({s.strip().strip("'").strip('"') for s in variables})
            return variables

        self._data_keys = get_vars(self._fun)
        dummy_data = {}
        p = "desc.equilibrium.equilibrium.Equilibrium"
        for key in self._data_keys:
            assert key in data_index[p], f"Don't know how to compute {key}"
            if data_index[p][key]["dim"] == 0:
                dummy_data[key] = jnp.array(0.0)
            else:
                dummy_data[key] = jnp.empty(
                    (grid.num_nodes, data_index[p][key]["dim"])
                ).squeeze()

        self._fun_wrapped = lambda data: self._fun(grid, data)

        self._dim_f = jax.eval_shape(self._fun_wrapped, dummy_data).size
        self._scalar = self._dim_f == 1
        profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
        transforms = get_transforms(self._data_keys, obj=eq, grid=grid)
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
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
        )
        f = self._fun_wrapped(data)
        return f
