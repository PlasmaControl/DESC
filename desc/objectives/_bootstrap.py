"""Objectives related to the bootstrap current profile."""

import numpy as np

from desc.backend import jnp
from desc.compute import (
    compute_flux_coords,
    compute_J_dot_B_Redl,
    compute_contravariant_current_density,
    data_index,
)
from desc.compute.utils import compress
from desc.grid import LinearGrid
from desc.transform import Transform
from desc.utils import Timer

from .objective_funs import _Objective


class BootstrapRedlConsistency(_Objective):
    r"""Promote consistency of the bootstrap current for axisymmetry or quasisymmetry.

    The scalar objective is defined as in eq (15) of
    Landreman, Buller, & Drevlak, Physics of Plasmas 29, 082501 (2022)
    https://doi.org/10.1063/5.0098166
    slightly generalized to allow different radial weighting, and with a factor of
    1/2 for consistency with desc conventions.

    f_{boot} = numerator / denominator

    where

    numerator = (1/2) \int_0^1 d\rho \rho^p [<J \cdot B>_{MHD} - <J \cdot B>_{Redl}]^2

    denominator = \int_0^1 d\rho \rho^p [<J \cdot B>_{MHD} + <J \cdot B>_{Redl}]^2

    <J \cdot B>_{MHD} is the parallel current profile of the MHD equilibrium, and

    <J \cdot B>_{Redl} is the parallel current profile from drift-kinetic physics

    The denominator serves as a normalization so f_{boot} is dimensionless, and
    f_{boot} = 1/2 when either <J \cdot B>_{MHD} or <J \cdot B>_{Redl} vanishes. Note that the
    scalar objective is approximately independent of grid resolution.

    The objective is treated as a sum of Nr least-squares terms, where Nr is the number
    of rho grid points. In other words, the contribution to the numerator from each rho
    grid point is returned as a separate entry in the returned vector of residuals,
    each weighted by the square root of the denominator.

    Parameters
    ----------
    helicity_N : int
        Toroidal mode number of quasisymmetry, used for evaluating the Redl bootstrap current
        formula. Set to 0 for axisymmetry or quasi-axisymmetry; set to +/- NFP for
        quasi-helical symmetry.
    ne : Profile
        Electron density profile, in units of meter^{-3}
    Te : Profile
        Electron temperature profile, in units of eV
    Ti : Profile
        Ion temperature profile, in units of eV
    Zeff : Profile or float, optional
        Effective impurity charge
    rho_exponent: float
        Exponent p acting on rho in the numerator and denominator above.
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
    _print_value_fmt = "Bootstrap current self-consistency: {:10.3e} "

    def __init__(
        self,
        helicity_N=0,
        ne=None,
        Te=None,
        Ti=None,
        Zeff=1,
        rho_exponent=1,
        eq=None,
        target=0,
        weight=1,
        normalize=False,
        normalize_target=False,
        grid=None,
        name="Bootstrap current self-consistency (Redl)",
    ):
        self.helicity_N = helicity_N
        self.ne = ne
        self.Te = Te
        self.Ti = Ti
        self.Zeff = Zeff
        self.rho_exponent = rho_exponent
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
            derivs=data_index["<J dot B>"]["R_derivs"],
            build=True,
        )
        self._Z_transform = Transform(
            self.grid,
            eq.Z_basis,
            derivs=data_index["<J dot B>"]["R_derivs"],
            build=True,
        )
        self._L_transform = Transform(
            self.grid,
            eq.L_basis,
            derivs=data_index["<J dot B>"]["L_derivs"],
            build=True,
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, c_l, Psi, **kwargs):
        """Compute the bootstrap current self-consistency objective.

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
        obj : ndarray
            Bootstrap current self-consistency residual on each rho grid point.

        """
        data = compute_contravariant_current_density(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            c_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
            self._current,
        )
        data = compute_J_dot_B_Redl(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            c_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
            self._current,
            helicity_N=self.helicity_N,
            ne=self.ne,
            Te=self.Te,
            Ti=self.Ti,
            Zeff=self.Zeff,
            data=data,
        )
        data = compute_flux_coords(grid=self.grid, data=data)

        fourpi2 = 4 * jnp.pi * jnp.pi

        # Consider moving the following calculation of rho_weights to
        # grid.py:Grid._count_nodes()?

        # Get the number of (theta, zeta) grid points for each rho
        # grid point. The result is a 1D array of size num_rho.
        num_surf_points, _ = jnp.histogram(
            self.grid.inverse_rho_idx, bins=self.grid.num_rho
        )
        # Get the weights for integrating in rho without also integrating in (theta, zeta):
        rho_weights = compress(self.grid, self.grid.weights) * num_surf_points / fourpi2

        denominator = (
            jnp.sum(
                (data["<J dot B>"] + data["<J dot B> Redl"]) ** 2
                * (data["rho"] ** self.rho_exponent)
                * self.grid.weights
            )
            / fourpi2
        )

        residuals = compress(
            self.grid,
            (data["<J dot B>"] - data["<J dot B> Redl"])
            * jnp.sqrt(data["rho"] ** self.rho_exponent),
        ) * jnp.sqrt(rho_weights / denominator)

        return self._shift_scale(residuals)
