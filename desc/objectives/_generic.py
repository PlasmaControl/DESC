"""Generic objectives that don't belong anywhere else."""

import inspect
import re

from desc.backend import jnp
from desc.compute import compute as compute_fun
from desc.compute import data_index
from desc.compute.utils import get_params, get_profiles, get_transforms
from desc.grid import QuadratureGrid

from .objective_funs import _Objective


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
        Must be broadcastable to Objective.dim_f.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
        Has no effect for this objective.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True. Has no effect for this objective.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
    name : str, optional
        Name of the objective function.


    Examples
    --------
    .. code-block:: python

        from desc.compute.utils import surface_averages
        def myfun(grid, data):
            # This will compute the flux surface average of the function
            # R*B_T from the Grad-Shafranov equation
            f = data['R']*data['B_phi']
            f_fsa = surface_averages(grid, f, sqrt_g=data['sqrt_g'])
            # this has the FSA values on the full grid, but we just want
            # the unique values:
            return grid.compress(f_fsa)

        myobj = ObjectiveFromUser(myfun)

    """

    _units = "(Unknown)"
    _print_value_fmt = "Custom Objective value: {:10.3e} "

    def __init__(
        self,
        fun,
        eq=None,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        grid=None,
        name="custom",
    ):
        if target is None and bounds is None:
            target = 0
        self._fun = fun
        self._grid = grid
        super().__init__(
            eq=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

    def build(self, eq=None, use_jit=True, verbose=1):
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
        eq = eq or self._eq
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
                dummy_data[key] = jnp.empty((grid.num_nodes, data_index[p][key]["dim"]))

        self._fun_wrapped = lambda data: self._fun(grid, data)
        import jax

        self._dim_f = jax.eval_shape(self._fun_wrapped, dummy_data).size
        self._scalar = self._dim_f == 1
        self._args = get_params(
            self._data_keys,
            obj="desc.equilibrium.equilibrium.Equilibrium",
            has_axis=grid.axis.size,
        )
        profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
        transforms = get_transforms(self._data_keys, obj=eq, grid=grid)
        self._constants = {
            "transforms": transforms,
            "profiles": profiles,
        }

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, *args, **kwargs):
        """Compute the quantity.

        Parameters
        ----------
        args : ndarray
            Parameters given by self.args.

        Returns
        -------
        f : ndarray
            Computed quantity.

        """
        params, constants = self._parse_args(*args, **kwargs)
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
        Must be broadcastable to Objective.dim_f.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
        Has no effect for this objective
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True. Has no effect for this objective.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
    name : str, optional
        Name of the objective function.

    """

    _print_value_fmt = "GenericObjective value: {:10.3e} "

    def __init__(
        self,
        f,
        eq=None,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        grid=None,
        name="generic",
    ):
        if target is None and bounds is None:
            target = 0
        self.f = f
        self._grid = grid
        super().__init__(
            eq=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
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

    def build(self, eq=None, use_jit=True, verbose=1):
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
        eq = eq or self._eq
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
            self._dim_f = grid.num_nodes * data_index[p][self.f]["dim"]
        self._args = get_params(
            self.f,
            obj="desc.equilibrium.equilibrium.Equilibrium",
            has_axis=grid.axis.size,
        )
        profiles = get_profiles(self.f, obj=eq, grid=grid)
        transforms = get_transforms(self.f, obj=eq, grid=grid)
        self._constants = {
            "transforms": transforms,
            "profiles": profiles,
        }

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, *args, **kwargs):
        """Compute the quantity.

        Parameters
        ----------
        args : ndarray
            Parameters given by self.args.

        Returns
        -------
        f : ndarray
            Computed quantity.

        """
        params, constants = self._parse_args(*args, **kwargs)
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
