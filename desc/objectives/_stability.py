"""Objectives for targeting MHD stability."""

import numpy as np

from desc.backend import jnp
from desc.compute import compute as compute_fun
from desc.compute import get_params, get_profiles, get_transforms
from desc.grid import LinearGrid
from desc.utils import Timer

from .normalization import compute_scaling_factors
from .objective_funs import _Objective


class MercierStability(_Objective):
    """The Mercier criterion is a fast proxy for MHD stability.

    This makes it a useful figure of merit for stellarator operation.
    Systems with D_Mercier > 0 are favorable for stability.

    See equation 4.16 in
    Landreman, M., & Jorge, R. (2020). Magnetic well and Mercier stability of
    stellarators near the magnetic axis. Journal of Plasma Physics, 86(5), 905860510.
    doi:10.1017/S002237782000121X.

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
    _units = "(Wb^-2)"
    _print_value_fmt = "Mercier Stability: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        grid=None,
        name="Mercier Stability",
    ):
        if target is None and bounds is None:
            bounds = (0, np.inf)
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

        self._dim_f = grid.num_rho
        self._data_keys = ["D_Mercier"]
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
            self._normalization = 1 / scales["Psi"] ** 2

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, *args, **kwargs):
        """Compute the Mercier stability criterion.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).
        L_lmn : ndarray
            Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
        p_l : ndarray
            Spectral coefficients of p(rho) -- pressure profile (Pa).
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.
        c_l : ndarray
            Spectral coefficients of I(rho) -- toroidal current profile (A).
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).
        Te_l : ndarray
            Spectral coefficients of Te(rho) -- electron temperature profile (eV).
        ne_l : ndarray
            Spectral coefficients of ne(rho) -- electron density profile (1/m^3).
        Ti_l : ndarray
            Spectral coefficients of Ti(rho) -- ion temperature profile (eV).
        Zeff_l : ndarray
            Spectral coefficients of Zeff(rho) -- effective atomic number profile.

        Returns
        -------
        D_Mercier : ndarray
            Mercier stability criterion.

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
        return constants["transforms"]["grid"].compress(data["D_Mercier"])

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


class MagneticWell(_Objective):
    """The magnetic well is a fast proxy for MHD stability.

    This makes it a useful figure of merit for stellarator operation.
    Systems with magnetic well > 0 are favorable for stability.

    This objective uses the magnetic well parameter defined in equation 3.2 of
    Landreman, M., & Jorge, R. (2020). Magnetic well and Mercier stability of
    stellarators near the magnetic axis. Journal of Plasma Physics, 86(5), 905860510.
    doi:10.1017/S002237782000121X.

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
        Note: Has no effect for this objective.
    normalize_target : bool
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True. Note: Has no effect for this objective.
    grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = False
    _units = "(dimensionless)"
    _print_value_fmt = "Magnetic Well: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        grid=None,
        name="Magnetic Well",
    ):
        if target is None and bounds is None:
            bounds = (0, np.inf)
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

        self._dim_f = grid.num_rho
        self._data_keys = ["magnetic well"]
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
        """Compute a magnetic well parameter.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).
        L_lmn : ndarray
            Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
        p_l : ndarray
            Spectral coefficients of p(rho) -- pressure profile (Pa).
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.
        c_l : ndarray
            Spectral coefficients of I(rho) -- toroidal current profile (A).
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).
        Te_l : ndarray
            Spectral coefficients of Te(rho) -- electron temperature profile (eV).
        ne_l : ndarray
            Spectral coefficients of ne(rho) -- electron density profile (1/m^3).
        Ti_l : ndarray
            Spectral coefficients of Ti(rho) -- ion temperature profile (eV).
        Zeff_l : ndarray
            Spectral coefficients of Zeff(rho) -- effective atomic number profile.

        Returns
        -------
        magnetic_well : ndarray
            Magnetic well parameter.

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
        return constants["transforms"]["grid"].compress(data["magnetic well"])

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
