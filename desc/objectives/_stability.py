from desc.backend import jnp
from desc.compute import (
    data_index,
    compute_dmerc,
    compute_magnetic_well,
)
from desc.compute.utils import compress
from desc.grid import LinearGrid
from desc.transform import Transform
from desc.utils import Timer
from .objective_funs import _Objective


class MercierStability(_Objective):
    """The Mercier criterion is a fast proxy for MHD stability.

    This makes it a useful figure of merit for stellarator operation.
    Systems with DMerc > 0 are favorable for stability.

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
    grid : LinearGrid, ConcentricGrid, QuadratureGrid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.
    """

    _scalar = False
    _linear = False

    def __init__(
        self, eq=None, target=0, weight=1, grid=None, name="Mercier Stability"
    ):
        self.grid = grid
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        self._callback_fmt = "Mercier Stability: {:10.3e}"

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
                rho=jnp.linspace(1 / 5, 1, 5),
            )

        self._dim_f = 1

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._pressure = eq.pressure.copy()
        self._pressure.grid = self.grid
        self._iota = eq.iota.copy()
        self._iota.grid = self.grid

        self._R_transform = Transform(
            self.grid,
            eq.R_basis,
            derivs=data_index["Mercier DMerc"]["R_derivs"],
            build=True,
        )
        self._Z_transform = Transform(
            self.grid,
            eq.Z_basis,
            derivs=data_index["Mercier DMerc"]["R_derivs"],
            build=True,
        )
        self._L_transform = Transform(
            self.grid,
            eq.L_basis,
            derivs=data_index["Mercier DMerc"]["L_derivs"],
            build=True,
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def compute(self, R_lmn, Z_lmn, L_lmn, p_l, i_l, Psi, **kwargs):
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
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        DMerc : ndarray
            Mercier stability criterion.
        """
        data = compute_dmerc(
            R_lmn,
            Z_lmn,
            L_lmn,
            p_l,
            i_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._pressure,
            self._iota,
        )
        return self._shift_scale(compress(self.grid, data["Mercier DMerc"]))


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
    grid : LinearGrid, ConcentricGrid, QuadratureGrid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.
    """

    _scalar = False
    _linear = False

    def __init__(self, eq=None, target=0, weight=1, grid=None, name="Magnetic Well"):
        self.grid = grid
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        self._callback_fmt = "Magnetic Well: {:10.3e}"

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
                rho=jnp.linspace(1 / 5, 1, 5),
            )

        self._dim_f = 1

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._pressure = eq.pressure.copy()
        self._pressure.grid = self.grid
        self._iota = eq.iota.copy()
        self._iota.grid = self.grid

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

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def compute(self, R_lmn, Z_lmn, L_lmn, p_l, i_l, Psi, **kwargs):
        """Compute the magnetic well parameter.

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
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        ndarray
            Magnetic well parameter.
        """
        data = compute_magnetic_well(
            R_lmn,
            Z_lmn,
            L_lmn,
            p_l,
            i_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._pressure,
            self._iota,
        )
        return self._shift_scale(compress(self.grid, data["magnetic well"]))
