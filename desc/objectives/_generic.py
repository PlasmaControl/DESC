"""Generic objectives that don't belong anywhere else."""

import re

import numpy as np

from desc.backend import (
    jnp,
    tree_flatten,
    tree_leaves,
    tree_map,
    tree_structure,
    tree_unflatten,
)
from desc.compute import data_index
from desc.compute.utils import _compute as compute_fun
from desc.compute.utils import _parse_parameterization, get_profiles, get_transforms
from desc.grid import QuadratureGrid
from desc.optimizable import OptimizableCollection
from desc.utils import (
    broadcast_tree,
    errorif,
    getsource,
    jaxify,
    parse_argname_change,
    setdefault,
)

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
            # needs to return a 1d array, not a scalar
            return jnp.atleast_1d(data["<beta>_vol"])

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


class DeflationOperator(_Objective):
    r"""Deflation wrapper to be added to or to wrap objective to find new solutions.

    If DeflationOperator is created while passing in an objective, the cost will be M*f
    where f is the objective's computed value.
    If DeflationOperator is created without passing in an objective, the cost will be
    only M

    Deflation is done on the passed-in list of parameters. This objective
    value will be large if the current state is close to one of the already-found
    states given by `things_to_deflate`, thus enabling new solutions to be found,
    and guarantees that old solutions are not found (as the objective increases
    without bound as an already-found solution is approached)

    The deflation operator is defined as:

    M(x;xₖ)=(||x−xₖ||₂)⁻ᵖ + σ

    (if `deflation_type="power"`)

    or

    M(𝐱;𝐱₁*) = exp(1/||𝐱−𝐱₁*||₂) + σ

    (if `deflation_type="exp"`)

    where x is the state and xₖ the passed-in known state.
    If multiple known states are used for deflation, then M is computed
    for each deflated state, then either multipled or added together (depending
    on if `multiple_deflation_type="prod"` or `"sum"`) to form the final cost.
    If an objective was passed in, this will then be multiplied by that objective's
    compute.

    Parameters
    ----------
    thing : Optimizable
        Optimizable that will be optimized to satisfy the Objective.
    things_to_deflate: list containing elements of type {Optimizable, None}
        list of objects to use in deflation operator. Should be same type
        as thing. Can also contain None elements, in which case those will be ignored.
        The utility of allowing the None element and ignoring them is if one is using
        this objective in a loop with a pre-determined number of iterations and adding
        each result of the loop iterate to the things_to_deflate, it may trigger
        recompilation of the objective's compute and jac/grad functions each time,
        which is wasteful. You can instead pass in a list containing None elements
        padding the list out to the max length it will attain. In this way, no
        recompilations will be triggered, and the entire loop will be completed
        much more quickly.
        If all things_to_deflate are None, this objective has zero cost (if not
        wrapping another objective) or simply returns the wrapped objective's
        cost (if wrapping another objective)
    params_to_deflate_with : nested list of dicts, optional
        Dict keys are the names of parameters to deflate (str), and dict values are the
        indices to deflate with for each corresponding parameter (int array).
        Use True (False) instead of an int array to deflate all (none) of the indices
        for that parameter.
        Must have the same pytree structure as thing.params_dict.
        The default is to deflate all indices of all parameters.
    objective: _Objective, optional
        Objective to wrap with the DeflationOperator. If not None, the cost will
        be M(x;xₖ)f(x) where f(x) is the Objective's cost. If None, then the cost
        returned will be M(x;xₖ). The objective must accept only one optimizable
        thing, and it must be the same as the thing passed to the DeflationOperator
    sigma: float, optional
        shift parameter in deflation operator.
    power: float, optional
        power parameter in deflation operator, ignored if `deflation_type="exp"`.
    deflation_type: {"power","exp"}
        What type of deflation to use. If `"power"`, uses the form
        pioneered by Farrell where M(𝐱;𝐱₁*) = ||𝐱−𝐱₁*||⁻ᵖ₂ + σ
        while `"exp"` uses the form from Riley 2024, where
        M(𝐱;𝐱₁*) = exp(1/||𝐱−𝐱₁*||₂) + σ. Defaults to "power".
    multiple_deflation_type: {"prod","sum"}
        When deflating multiple states, how to reduce the individual deflation
        terms Mᵢ(𝐱;𝐱ᵢ*). `"prod"` will multiply each individual deflation term
        together, while `"sum"` will add each individual term.
    single_shift: bool,
        Whether to use a single shift or include the shift in each individual
        deflation term. i.e. whether to use M = σ + prod(||𝐱−𝐱_i*||⁻ᵖ₂) (if True)
        or to use M = prod( σ + ||𝐱−𝐱_i*||⁻ᵖ₂). Defaults to False.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )
    _static_attrs = _Objective._static_attrs + [
        "_deflation_type",
        "_multiple_deflation_type",
        "_single_shift",
        "_params_to_deflate_with",
    ]

    _coordinates = "rtz"
    _units = "~"
    _print_value_fmt = "Deflation error: "

    def __init__(
        self,
        thing,
        things_to_deflate,
        params_to_deflate_with=None,
        objective=None,
        sigma=1.0,
        power=2,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        name="Deflation",
        jac_chunk_size=None,
        deflation_type="power",
        multiple_deflation_type="prod",
        single_shift=False,
    ):
        if target is None and bounds is None:
            target = 0
        assert np.all(
            [
                (isinstance(t, type(thing)) and t != thing) or t is None
                for t in things_to_deflate
            ]
        )
        self._things_to_deflate = things_to_deflate.copy()
        self._sigma = sigma
        self._power = power
        self._params_to_deflate_with = params_to_deflate_with
        assert deflation_type in ["power", "exp"]
        self._deflation_type = deflation_type
        assert multiple_deflation_type in ["prod", "sum"]
        self._multiple_deflation_type = multiple_deflation_type
        self._single_shift = single_shift
        self._objective = objective
        if self._objective is not None:
            assert isinstance(
                self._objective, _Objective
            ), "objective passed in must be an _Objective!"
            assert len(objective.things) == 1
            assert objective.things[0] == thing
            name = "Deflated " + self._objective._name
            self._units = self._objective._units
            self._scalar = self._objective._scalar
            self._coordinates = self._objective._coordinates
            self._print_value_fmt = "Deflated " + self._objective._print_value_fmt

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

        # default params
        default_params = tree_map(lambda dim: np.arange(dim), thing.dimensions)
        self._params_to_deflate_with = setdefault(
            self._params_to_deflate_with, default_params
        )
        self._params_to_deflate_with = broadcast_tree(
            self._params_to_deflate_with, default_params
        )
        self._indices = tree_leaves(self._params_to_deflate_with)
        assert tree_structure(self._params_to_deflate_with) == tree_structure(
            default_params
        )

        if self._objective is not None:
            if not self._objective.built:
                self._objective.build()
            self._dim_f = self._objective._dim_f
            self._normalization = self._objective._normalization
            self._constants = self._objective._constants
        else:
            self._dim_f = 1

        self._is_none_mask = []
        self._is_not_none_mask = []
        self._not_all_things_to_deflate_are_None = not np.all(
            [t is None for t in self._things_to_deflate]
        )

        for i, t in enumerate(self._things_to_deflate):
            if t is None:
                self._is_none_mask.append(1.0)
                self._is_not_none_mask.append(0.0)
                self._things_to_deflate[i] = thing
            else:
                self._is_none_mask.append(0.0)
                self._is_not_none_mask.append(1.0)

        self._is_none_mask = np.array(self._is_none_mask)
        self._is_not_none_mask = np.array(self._is_not_none_mask)

        if (
            self._objective is None and self._bounds is not None
        ):  # if being used as constraint/obj, min value should be sigma
            lower_bound_min = (
                self._sigma
                if self._single_shift
                else self._sigma * np.sum(self._is_not_none_mask)
            )
            assert np.all(self._bounds[0] <= lower_bound_min), (
                "Provided lower bound for deflation operator is too high compared "
                f"to the minimum value of {lower_bound_min} it can take based off "
                "of sigma, use a smaller lower bound"
            )

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute deflation error.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : scalar
            Deflation error.

        """
        this_thing_params = jnp.concatenate(
            [
                jnp.atleast_1d(param[idx])
                for param, idx in zip(tree_leaves(params), self._indices)
            ]
        )
        diffs = [
            this_thing_params
            - self._is_not_none_mask[i]
            * jnp.concatenate(
                [
                    jnp.atleast_1d(param[idx])
                    for param, idx in zip(tree_leaves(t.params_dict), self._indices)
                ]
            )
            for i, t in enumerate(self._things_to_deflate)
        ]
        # to avoid division by zero if the states are the exact same
        eps = 1e2 * jnp.finfo(diffs[0].dtype).eps
        diffs = jnp.vstack(diffs)
        if self._deflation_type == "power":
            M_i = 1 / (
                jnp.linalg.norm(diffs, axis=1) + eps
            ) ** self._power + self._sigma * (not self._single_shift)
        else:
            M_i = jnp.exp(1 / (jnp.linalg.norm(diffs, axis=1) + eps)) + self._sigma * (
                not self._single_shift
            )

        # we use the where= to only count the non-None things in things_to_deflate
        if self._multiple_deflation_type == "prod":
            deflation_parameter = jnp.prod(
                M_i, initial=1.0, where=self._is_not_none_mask
            ) + self._sigma * (self._single_shift)
        else:
            deflation_parameter = jnp.sum(
                M_i, where=self._is_not_none_mask, initial=0.0
            ) + self._sigma * (self._single_shift)

        # enforce deflation paremeter 0 here if every thing_to_deflate is None
        deflation_parameter = (
            deflation_parameter * self._not_all_things_to_deflate_are_None
        )

        if self._objective is not None:
            f = self._objective.compute(params)
            # if wrapping an objective, but all things are None, make deflation do
            # nothing when multiplying f, so here we add 1 to it as it is 0 right now
            deflation_parameter += float(
                jnp.invert(self._not_all_things_to_deflate_are_None)
            )
        else:
            f = 1.0

        return deflation_parameter * f
