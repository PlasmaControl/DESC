"""Objectives for targeting profiles."""

import numpy as np

from desc.compute.utils import _compute as compute_fun
from desc.compute.utils import get_profiles, get_transforms
from desc.grid import LinearGrid
from desc.utils import Timer, warnif

from .normalization import compute_scaling_factors
from .objective_funs import _Objective, collect_docs
from .utils import _parse_callable_target_bounds

profile_overwrite = {
    "target": """
    target : {float, ndarray, callable}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to ``Objective.dim_f``. If a callable, should take a
        single argument `rho` and return the desired value of the profile at those
        locations. Defaults to ``target=0``.
        """,
    "bounds": """
    bounds : tuple of {float, ndarray, callable}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to ``Objective.dim_f``
        If a callable, each should take a single argument `rho` and return the
        desired bound (lower or upper) of the profile at those locations.
        Defaults to ``target=0``.
        """,
}


class Pressure(_Objective):
    """Target pressure profile.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(L=eq.L_grid)``.

    """

    __doc__ = __doc__.rstrip() + collect_docs(overwrite=profile_overwrite)

    _coordinates = "r"
    _units = "(Pa)"
    _print_value_fmt = "Pressure: "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="pressure",
        jac_chunk_size=None,
        device_id=0,
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
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            name=name,
            jac_chunk_size=jac_chunk_size,
            device_id=device_id,
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
            self._normalization = scales["p"]

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
        return constants["transforms"]["grid"].compress(data["p"])


class RotationalTransform(_Objective):
    """Targets a rotational transform profile.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid)``. Note that
        it should have poloidal and toroidal resolution, as flux surface averages
        are required.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        overwrite=profile_overwrite,
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
    )

    _coordinates = "r"
    _units = "(dimensionless)"
    _print_value_fmt = "Rotational transform: "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="rotational transform",
        jac_chunk_size=None,
        device_id=0,
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
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            name=name,
            jac_chunk_size=jac_chunk_size,
            device_id=device_id,
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

        warnif(
            (eq.iota is None) and ((grid.num_theta * (1 + eq.sym)) < 2 * eq.M),
            RuntimeWarning,
            "RotationalTransform objective grid requires poloidal "
            "resolution for surface averages",
        )
        warnif(
            (eq.iota is None) and (grid.num_zeta < 2 * eq.N),
            RuntimeWarning,
            "RotationalTransform objective grid requires toroidal "
            "resolution for surface averages",
        )

        self._target, self._bounds = _parse_callable_target_bounds(
            self._target, self._bounds, grid.nodes[grid.unique_rho_idx]
        )

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
        return constants["transforms"]["grid"].compress(data["iota"])


class Shear(_Objective):
    """Targets a shear profile (normalized derivative of rotational transform).

    f = -dι/dρ * (ρ/ι)

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid)``. Note that
        it should have poloidal and toroidal resolution, as flux surface averages
        are required.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        overwrite=profile_overwrite,
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
    )

    _coordinates = "r"
    _units = "(dimensionless)"
    _print_value_fmt = "Shear: "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="shear",
        jac_chunk_size=None,
        device_id=0,
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
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            name=name,
            jac_chunk_size=jac_chunk_size,
            device_id=device_id,
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

        warnif(
            (eq.iota is None) and ((grid.num_theta * (1 + eq.sym)) < 2 * eq.M),
            RuntimeWarning,
            "Shear objective grid requires poloidal " "resolution for surface averages",
        )
        warnif(
            (eq.iota is None) and (grid.num_zeta < 2 * eq.N),
            RuntimeWarning,
            "Shear objective grid requires toroidal " "resolution for surface averages",
        )

        self._target, self._bounds = _parse_callable_target_bounds(
            self._target, self._bounds, grid.nodes[grid.unique_rho_idx]
        )

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
        return constants["transforms"]["grid"].compress(data["shear"])


class ToroidalCurrent(_Objective):
    """Target toroidal current profile.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid)``. Note that
        it should have poloidal and toroidal resolution, as flux surface averages
        are required.

    """

    __doc__ = __doc__.rstrip() + collect_docs(overwrite=profile_overwrite)

    _coordinates = "r"
    _units = "(A)"
    _print_value_fmt = "Toroidal current: "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="toroidal current",
        jac_chunk_size=None,
        device_id=0,
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
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            name=name,
            jac_chunk_size=jac_chunk_size,
            device_id=device_id,
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

        warnif(
            (eq.current is None) and ((grid.num_theta * (1 + eq.sym)) < 2 * eq.M),
            RuntimeWarning,
            "ToroidalCurrent objective grid requires poloidal "
            "resolution for surface averages",
        )
        warnif(
            (eq.current is None) and (grid.num_zeta < 2 * eq.N),
            RuntimeWarning,
            "ToroidalCurrent objective grid requires toroidal "
            "resolution for surface averages",
        )

        self._target, self._bounds = _parse_callable_target_bounds(
            self._target, self._bounds, grid.nodes[grid.unique_rho_idx]
        )

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
        return constants["transforms"]["grid"].compress(data["current"])
