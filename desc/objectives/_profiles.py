"""Objectives for targeting profiles."""

import numpy as np

from desc.backend import jnp
from desc.compute import compute as compute_fun
from desc.compute.utils import get_params, get_profiles, get_transforms
from desc.grid import LinearGrid
from desc.utils import Timer

from .normalization import compute_scaling_factors
from .objective_funs import _Objective
from .utils import _parse_callable_target_bounds


class Pressure(_Objective):
    """Target pressure profile.

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
    _linear = False  # in general, ie for kinetic equilibria
    _units = "(Pa)"
    _print_value_fmt = "Pressure: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        grid=None,
        name="pressure",
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
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
            )
        else:
            grid = self._grid

        self._target, self._bounds = _parse_callable_target_bounds(
            self._target, self._bounds, grid.nodes[grid.unique_rho_idx]
        )

        self._dim_f = grid.num_rho
        self._data_keys = ["p"]
        self._args = get_params(
            self._data_keys,
            obj="desc.equilibrium.equilibrium.Equilibrium",
            has_axis=grid.axis.size,
        )

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
        self._transforms = get_transforms(self._data_keys, obj=eq, grid=grid)
        self._constants = {
            "transforms": self._transforms,
            "profiles": self._profiles,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["p"]

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, *args, **kwargs):
        """Compute plasma pressure.

        Returns
        -------
        pressure : ndarray
            Plasma pressure at specified points.

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
        return constants["transforms"]["grid"].compress(data["p"])

    def _scale(self, *args, **kwargs):
        """Compute and apply the target/bounds, weighting, and normalization."""
        constants = kwargs.get("constants", None)
        if constants is None:
            constants = self.constants
        w = constants["transforms"]["grid"].compress(
            constants["transforms"]["grid"].spacing[:, 0]
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

        self._target, self._bounds = _parse_callable_target_bounds(
            self._target, self._bounds, grid.nodes[grid.unique_rho_idx]
        )

        self._dim_f = grid.num_rho
        self._data_keys = ["iota"]
        self._args = get_params(
            self._data_keys,
            obj="desc.equilibrium.equilibrium.Equilibrium",
            has_axis=grid.axis.size,
        )

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
        self._transforms = get_transforms(self._data_keys, obj=eq, grid=grid)
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
            Rotational transform on specified flux surfaces.

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
        return constants["transforms"]["grid"].compress(data["iota"])

    def _scale(self, *args, **kwargs):
        """Compute and apply the target/bounds, weighting, and normalization."""
        constants = kwargs.get("constants", None)
        if constants is None:
            constants = self.constants
        w = constants["transforms"]["grid"].compress(
            constants["transforms"]["grid"].spacing[:, 0]
        )
        return super()._scale(*args, **kwargs) * jnp.sqrt(w)

    def print_value(self, *args, **kwargs):
        """Print the value of the objective."""
        f = self.compute(*args, **kwargs)
        print("Maximum " + self._print_value_fmt.format(jnp.max(f)) + self._units)
        print("Minimum " + self._print_value_fmt.format(jnp.min(f)) + self._units)
        print("Average " + self._print_value_fmt.format(jnp.mean(f)) + self._units)


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

    _scalar = False
    _linear = False
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

        self._target, self._bounds = _parse_callable_target_bounds(
            self._target, self._bounds, grid.nodes[grid.unique_rho_idx]
        )

        self._dim_f = grid.num_rho
        self._data_keys = ["shear"]
        self._args = get_params(
            self._data_keys,
            obj="desc.equilibrium.equilibrium.Equilibrium",
            has_axis=grid.axis.size,
        )

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
        self._transforms = get_transforms(self._data_keys, obj=eq, grid=grid)
        self._constants = {
            "transforms": self._transforms,
            "profiles": self._profiles,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, *args, **kwargs):
        """Compute shear profile errors.

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
        shear : ndarray
            Normalized radial derivative of the rotational transform.

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
        return constants["transforms"]["grid"].compress(data["shear"])

    def _scale(self, *args, **kwargs):
        """Compute and apply the target/bounds, weighting, and normalization."""
        constants = kwargs.get("constants", None)
        if constants is None:
            constants = self.constants
        w = constants["transforms"]["grid"].compress(
            constants["transforms"]["grid"].spacing[:, 0]
        )
        return super()._scale(*args, **kwargs) * jnp.sqrt(w)

    def print_value(self, *args, **kwargs):
        """Print the value of the objective."""
        f = self.compute(*args, **kwargs)
        print("Maximum " + self._print_value_fmt.format(jnp.max(f)) + self._units)
        print("Minimum " + self._print_value_fmt.format(jnp.min(f)) + self._units)
        print("Average " + self._print_value_fmt.format(jnp.mean(f)) + self._units)


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

        self._target, self._bounds = _parse_callable_target_bounds(
            self._target, self._bounds, grid.nodes[grid.unique_rho_idx]
        )

        self._dim_f = grid.num_rho
        self._data_keys = ["current"]
        self._args = get_params(
            self._data_keys,
            obj="desc.equilibrium.equilibrium.Equilibrium",
            has_axis=grid.axis.size,
        )

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
        self._transforms = get_transforms(self._data_keys, obj=eq, grid=grid)
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
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
        )
        return constants["transforms"]["grid"].compress(data["current"])

    def _scale(self, *args, **kwargs):
        """Compute and apply the target/bounds, weighting, and normalization."""
        constants = kwargs.get("constants", None)
        if constants is None:
            constants = self.constants
        w = constants["transforms"]["grid"].compress(
            constants["transforms"]["grid"].spacing[:, 0]
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
