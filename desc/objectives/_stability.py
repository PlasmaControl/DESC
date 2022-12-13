"""Objectives for targeting MHD stability."""

import numpy as np

from desc.backend import jnp
from desc.compute import compute_magnetic_well, compute_mercier_stability, data_index
from desc.compute.utils import compress
from desc.grid import LinearGrid
from desc.transform import Transform
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
        Target value(s) of the objective.
        len(target) must be equal to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
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
        target=0,
        weight=1,
        normalize=True,
        normalize_target=True,
        grid=None,
        name="Mercier Stability",
    ):
        self.grid = grid
        super().__init__(
            eq=eq,
            target=target,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

    def build(self, eq, use_jit=True, verbose=1):
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
        if self.grid is None:
            self.grid = LinearGrid(
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                rho=np.linspace(1 / 5, 1, 5),
            )

        self._dim_f = self.grid.num_rho

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._pressure = eq.pressure.copy()
        self._pressure.grid = self.grid
        if eq.iota is not None:
            self._iota = eq.iota.copy()
            self._iota.grid = self.grid
            self._current = None
        else:
            self._current = eq.current.copy()
            self._current.grid = self.grid
            self._iota = None

        self._R_transform = Transform(
            self.grid,
            eq.R_basis,
            derivs=data_index["D_Mercier"]["R_derivs"],
            build=True,
        )
        self._Z_transform = Transform(
            self.grid,
            eq.Z_basis,
            derivs=data_index["D_Mercier"]["R_derivs"],
            build=True,
        )
        self._L_transform = Transform(
            self.grid,
            eq.L_basis,
            derivs=data_index["D_Mercier"]["L_derivs"],
            build=True,
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = 1 / scales["Psi"] ** 2 / jnp.sqrt(self._dim_f)

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, R_lmn, Z_lmn, L_lmn, p_l, i_l, c_l, Psi, **kwargs):
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
            Spectral coefficients of p(rho) -- pressure profile.
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.
        c_l : ndarray
            Spectral coefficients of I(rho) -- toroidal current profile.
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        D_Mercier : ndarray
            Mercier stability criterion.

        """
        data = compute_mercier_stability(
            R_lmn,
            Z_lmn,
            L_lmn,
            p_l,
            i_l,
            c_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._pressure,
            self._iota,
            self._current,
        )
        f = compress(self.grid, data["D_Mercier"], surface_label="rho")
        w = compress(self.grid, self.grid.spacing[:, 0], surface_label="rho")
        return self._shift_scale(f) * w


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
        Target value(s) of the objective.
        len(target) must be equal to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
        Note: has no effect for this objective.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
        Note: has no effect for this objective.
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
        target=0,
        weight=1,
        normalize=False,
        normalize_target=False,
        grid=None,
        name="Magnetic Well",
    ):
        self.grid = grid
        super().__init__(
            eq=eq,
            target=target,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

    def build(self, eq, use_jit=True, verbose=1):
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
        if self.grid is None:
            self.grid = LinearGrid(
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                rho=np.linspace(1 / 5, 1, 5),
            )

        self._dim_f = self.grid.num_rho

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._pressure = eq.pressure.copy()
        self._pressure.grid = self.grid
        if eq.iota is not None:
            self._iota = eq.iota.copy()
            self._iota.grid = self.grid
            self._current = None
        else:
            self._current = eq.current.copy()
            self._current.grid = self.grid
            self._iota = None

        self._R_transform = Transform(
            self.grid,
            eq.R_basis,
            derivs=data_index["magnetic well"]["R_derivs"],
            build=True,
        )
        self._Z_transform = Transform(
            self.grid,
            eq.Z_basis,
            derivs=data_index["magnetic well"]["R_derivs"],
            build=True,
        )
        self._L_transform = Transform(
            self.grid,
            eq.L_basis,
            derivs=data_index["magnetic well"]["L_derivs"],
            build=True,
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, R_lmn, Z_lmn, L_lmn, p_l, i_l, c_l, Psi, **kwargs):
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
            Spectral coefficients of p(rho) -- pressure profile.
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.
        c_l : ndarray
            Spectral coefficients of I(rho) -- toroidal current profile.
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        magnetic_well : ndarray
            Magnetic well parameter.

        """
        data = compute_magnetic_well(
            R_lmn,
            Z_lmn,
            L_lmn,
            p_l,
            i_l,
            c_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._pressure,
            self._iota,
            self._current,
        )
        f = compress(self.grid, data["magnetic well"], surface_label="rho")
        w = compress(self.grid, self.grid.spacing[:, 0], surface_label="rho")
        return self._shift_scale(f) * w
