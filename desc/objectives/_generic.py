"""Generic objectives that don't belong anywhere else."""

import inspect
import re

import numpy as np

from desc.backend import jnp, tree_flatten, tree_leaves, tree_unflatten
from desc.compute import data_index
from desc.compute.utils import _compute as compute_fun
from desc.compute.utils import _parse_parameterization, get_profiles, get_transforms
from desc.grid import QuadratureGrid
from desc.optimizable import OptimizableCollection
from desc.utils import errorif, parse_argname_change

from .linear_objectives import _FixedObjective
from .objective_funs import _Objective, collect_docs


class GenericObjective(_Objective):
    """A generic objective that can compute any quantity from the `data_index`.

    Parameters
    ----------
    f : str
        Name of the quantity to compute.
    thing : Optimizable
        Object that will be optimized to satisfy the Objective.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at. Defaults to
        ``QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid)`` if thing is an Equilibrium.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )

    _print_value_fmt = "Generic objective value: "

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
        name="generic",
        jac_chunk_size=None,
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
            grid = QuadratureGrid(
                thing.L_grid,
                thing.M_grid,
                thing.N_grid,
                thing.NFP,
            )
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
        transforms = get_transforms(self.f, obj=thing, grid=grid)
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
            f = data['R']*data['B_phi']
            # q here is the kwarg for "quantity" as in, quantity to be averaged
            f_fsa = surface_averages(grid, q=f, sqrt_g=data['sqrt(g)'])
            # this has the FSA values on the full grid, but we just want
            # the unique values:
            return grid.compress(f_fsa)

        myobj = ObjectiveFromUser(fun=myfun, thing=eq)

    """

    _units = "(Unknown)"
    _print_value_fmt = "Custom objective value: "

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
        name="custom",
        jac_chunk_size=None,
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
            src = inspect.getsource(fun)
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
        import jax

        self._dim_f = jax.eval_shape(self._fun_wrapped, dummy_data).size
        self._scalar = self._dim_f == 1
        profiles = get_profiles(self._data_keys, obj=thing, grid=grid)
        transforms = get_transforms(self._data_keys, obj=thing, grid=grid)
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
        )
        f = self._fun_wrapped(data)
        return f
