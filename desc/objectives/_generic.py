"""Generic objectives that don't belong anywhere else."""
import inspect
import re

from desc.backend import jnp
from desc.compute import compute as compute_fun
from desc.compute import data_index
from desc.compute.utils import compress, get_params, get_profiles, get_transforms
from desc.grid import LinearGrid, QuadratureGrid
from desc.profiles import Profile
from desc.utils import Timer

from .normalization import compute_scaling_factors
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
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective. Only used if bounds is None.
        len(target) must be equal to Objective.dim_f
    bounds : tuple, optional
        Lower and upper bounds on the objective. Overrides target.
        len(bounds[0]) and len(bounds[1]) must be equal to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
        Note: has no effect for this objective.
    normalize_target : bool
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.


    Examples
    --------
    .. code-block:: python

        from desc.compute.utils import surface_averages, compress
        def myfun(grid, data):
            # This will compute the flux surface average of the function
            # R*B_T from the Grad-Shafranov equation
            f = data['R']*data['B_phi']
            f_fsa = surface_averages(grid, f, sqrt_g=data['sqrt_g'])
            # this has the FSA values on the full grid, but we just want
            # the unique values:
            return compress(grid, f_fsa)

        myobj = ObjectiveFromUser(myfun)

    """

    _scalar = False
    _linear = False
    _units = "(Unknown)"
    _print_value_fmt = "Custom Objective Residual: {:10.3e} "

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

        def getvars(fun):
            pattern = r"data\[(.*?)\]"
            src = inspect.getsource(fun)
            variables = re.findall(pattern, src)
            variables = [s.replace("'", "").replace('"', "") for s in variables]
            return variables

        self._data_keys = getvars(self._fun)
        dummy_data = {}
        for key in self._data_keys:
            assert key in data_index, f"Don't know how to compute {key}"
            if data_index[key]["dim"] == 0:
                dummy_data[key] = jnp.array(0.0)
            else:
                dummy_data[key] = jnp.empty((grid.num_nodes, data_index[key]["dim"]))

        self._fun_wrapped = lambda data: self._fun(grid, data)
        import jax

        self._dim_f = jax.eval_shape(self._fun_wrapped, dummy_data).size
        self._scalar = self._dim_f == 1
        self._args = get_params(self._data_keys, has_axis=grid.axis.size)
        self._profiles = get_profiles(self._data_keys, eq=eq, grid=grid)
        self._transforms = get_transforms(self._data_keys, eq=eq, grid=grid)
        self._constants = {
            "transforms": self._transforms,
            "profiles": self._profiles,
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
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective. Only used if bounds is None.
        len(target) must be equal to Objective.dim_f
    bounds : tuple, optional
        Lower and upper bounds on the objective. Overrides target.
        len(bounds[0]) and len(bounds[1]) must be equal to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
        Note: has no effect for this objective.
    normalize_target : bool
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = False
    _units = "(Unknown)"
    _print_value_fmt = "Residual: {:10.3e} "

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
        self._units = "(" + data_index[self.f]["units"] + ")"

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

        self._data_keys = [self.f]
        if data_index[self.f]["dim"] == 0:
            self._dim_f = 1
            self._scalar = True
        else:
            self._dim_f = grid.num_nodes * data_index[self.f]["dim"]
            self._scalar = False
        self._args = get_params(self.f, has_axis=grid.axis.size)
        self._profiles = get_profiles(self.f, eq=eq, grid=grid)
        self._transforms = get_transforms(self.f, eq=eq, grid=grid)
        self._constants = {
            "transforms": self._transforms,
            "profiles": self._profiles,
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
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
        )
        f = data[self.f]
        if not self.scalar:
            f = (f.T * constants["transforms"]["grid"].weights).flatten()
        return f


class ToroidalCurrent(_Objective):
    """Target toroidal current profile.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective. Only used if bounds is None.
        len(target) must be equal to Objective.dim_f
    bounds : tuple, optional
        Lower and upper bounds on the objective. Overrides target.
        len(bounds[0]) and len(bounds[1]) must be equal to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = False
    _units = "(A)"
    _print_value_fmt = "Toroidal current: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        grid=None,
        name="toroidal current",
    ):
        if target is None and bounds is None:
            target = 0
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
            grid = LinearGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
            )
        else:
            grid = self._grid

        if isinstance(self._target, Profile):
            self._target = self._target(grid.nodes[grid.unique_rho_idx])

        self._dim_f = grid.num_rho
        self._data_keys = ["current"]
        self._args = get_params(self._data_keys, has_axis=grid.axis.size)

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._profiles = get_profiles(self._data_keys, eq=eq, grid=grid)
        self._transforms = get_transforms(self._data_keys, eq=eq, grid=grid)
        self._constants = {
            "transforms": self._transforms,
            "profiles": self._profiles,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["I"]

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, *args, **kwargs):
        """Compute toroidal current.

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
        current : ndarray
            Toroidal current (A) through specified surfaces.

        """
        params, constants = self._parse_args(*args, **kwargs)
        if constants is None:
            constants = self.constants
        data = compute_fun(
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
        )
        return compress(
            constants["transforms"]["grid"], data["current"], surface_label="rho"
        )

    def _scale(self, *args, **kwargs):
        """Compute and apply the target/bounds, weighting, and normalization."""
        constants = kwargs.get("constants", None)
        if constants is None:
            constants = self.constants
        w = compress(
            constants["transforms"]["grid"],
            constants["transforms"]["grid"].spacing[:, 0],
            surface_label="rho",
        )
        return super()._scale(*args, **kwargs) * jnp.sqrt(w)

    def print_value(self, *args, **kwargs):
        """Print the value of the objective."""
        f = self.compute(*args, **kwargs)
        print("Maximum " + self._print_value_fmt.format(jnp.max(f)) + self._units)
        print("Minimum " + self._print_value_fmt.format(jnp.min(f)) + self._units)
        print("Average " + self._print_value_fmt.format(jnp.mean(f)) + self._units)

        if self._normalize:
            print(
                "Maximum "
                + self._print_value_fmt.format(jnp.max(f / self.normalization))
                + "(normalized)"
            )
            print(
                "Minimum "
                + self._print_value_fmt.format(jnp.min(f / self.normalization))
                + "(normalized)"
            )
            print(
                "Average "
                + self._print_value_fmt.format(jnp.mean(f / self.normalization))
                + "(normalized)"
            )


class RotationalTransform(_Objective):
    """Targets a rotational transform profile.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective. Only used if bounds is None.
        len(target) must be equal to Objective.dim_f
    bounds : tuple, optional
        Lower and upper bounds on the objective. Overrides target.
        len(bounds[0]) and len(bounds[1]) must be equal to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
        Note: has no effect for this objective.
    normalize_target : bool
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
        Note: has no effect for this objective.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = False
    _units = "(dimensionless)"
    _print_value_fmt = "Rotational transform: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        grid=None,
        name="rotational transform",
    ):
        if target is None and bounds is None:
            target = 0
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
            grid = LinearGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
            )
        else:
            grid = self._grid

        if isinstance(self._target, Profile):
            self._target = self._target(grid.nodes[grid.unique_rho_idx])

        self._dim_f = grid.num_rho
        self._data_keys = ["iota"]
        self._args = get_params(self._data_keys, has_axis=grid.axis.size)

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._profiles = get_profiles(self._data_keys, eq=eq, grid=grid)
        self._transforms = get_transforms(self._data_keys, eq=eq, grid=grid)
        self._constants = {
            "transforms": self._transforms,
            "profiles": self._profiles,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, *args, **kwargs):
        """Compute rotational transform profile errors.

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
        iota : ndarray
            rotational transform on specified flux surfaces.

        """
        params, constants = self._parse_args(*args, **kwargs)
        if constants is None:
            constants = self.constants
        data = compute_fun(
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
        )
        return compress(
            constants["transforms"]["grid"], data["iota"], surface_label="rho"
        )

    def _scale(self, *args, **kwargs):
        """Compute and apply the target/bounds, weighting, and normalization."""
        constants = kwargs.get("constants", None)
        if constants is None:
            constants = self.constants
        w = compress(
            constants["transforms"]["grid"],
            constants["transforms"]["grid"].spacing[:, 0],
            surface_label="rho",
        )
        return super()._scale(*args, **kwargs) * jnp.sqrt(w)

    def print_value(self, *args, **kwargs):
        """Print the value of the objective."""
        f = self.compute(*args, **kwargs)
        print("Maximum " + self._print_value_fmt.format(jnp.max(f)) + self._units)
        print("Minimum " + self._print_value_fmt.format(jnp.min(f)) + self._units)
        print("Average " + self._print_value_fmt.format(jnp.mean(f)) + self._units)
