"""Objectives for targeting profiles."""

import numpy as np

from desc.compute import compute as compute_fun
from desc.compute.utils import get_profiles, get_transforms
from desc.grid import LinearGrid
from desc.utils import Timer, setdefault

from .normalization import compute_scaling_factors
from .objective_funs import _Objective


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

    _coordinates = "r"
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
            things=eq,
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
        self.things = setdefault(eq, self.things)
        eq = self.things[0]
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

        if callable(self._target):
            self._target = self._target(grid.nodes[grid.unique_rho_idx])

        self._dim_f = grid.num_rho
        self._data_keys = ["iota"]

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

        super().build(things=eq, use_jit=use_jit, verbose=verbose)

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
        return constants["transforms"]["grid"].compress(data["iota"])


class Shear(_Objective):
    """Targets a shear profile (normalized derivative of rotational transform).

    f = -dι/dρ * (ρ/ι)

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

    _coordinates = "r"
    _units = "(dimensionless)"
    _print_value_fmt = "Shear: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        grid=None,
        name="shear",
    ):
        if target is None and bounds is None:
            bounds = (-np.inf, 0)
        self._grid = grid
        super().__init__(
            things=eq,
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
        self.things = setdefault(eq, self.things)
        eq = self.things[0]
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

        if callable(self._target):
            self._target = self._target(grid.nodes[grid.unique_rho_idx])

        self._dim_f = grid.num_rho
        self._data_keys = ["shear"]

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

        super().build(things=eq, use_jit=use_jit, verbose=verbose)

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
        return constants["transforms"]["grid"].compress(data["shear"])


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

    _coordinates = "r"
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
            things=eq,
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
        self.things = setdefault(eq, self.things)
        eq = self.things[0]
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

        if callable(self._target):
            self._target = self._target(grid.nodes[grid.unique_rho_idx])

        self._dim_f = grid.num_rho
        self._data_keys = ["current"]

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

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["I"]

        super().build(things=eq, use_jit=use_jit, verbose=verbose)

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
        return constants["transforms"]["grid"].compress(data["current"])
