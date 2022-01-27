import numpy as np
from scipy.constants import mu_0

from desc.backend import jnp
from desc.utils import Timer
from desc.grid import QuadratureGrid, ConcentricGrid, LinearGrid
from desc.basis import DoubleFourierSeries
from desc.transform import Transform
from desc.compute import (
    data_index,
    compute_covariant_metric_coefficients,
    compute_magnetic_field_magnitude,
    compute_contravariant_current_density,
    compute_force_error,
    compute_boozer_coords,
    compute_quasisymmetry_error,
    compute_energy,
    compute_geometry,
)
from .objective_funs import _Objective


# scalar nonlinear objectives


class Volume(_Objective):
    """Plasma volume."""

    def __init__(self, eq=None, target=0, weight=1, grid=None):
        """Initialize a Volume Objective.

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
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.

        """
        self.grid = grid
        super().__init__(eq=eq, target=target, weight=weight)

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
            self.grid = QuadratureGrid(L=eq.L, M=eq.M, N=eq.N, NFP=eq.NFP)

        self._dim_f = 1

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=data_index["V"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["V"]["R_derivs"], build=True
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, R_lmn, Z_lmn):
        data = compute_geometry(R_lmn, Z_lmn, self._R_transform, self._Z_transform)
        return data["V"]

    def compute(self, R_lmn, Z_lmn, **kwargs):
        """Compute plasma volume.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).

        Returns
        -------
        V : float
            Plasma volume (m^3).

        """
        V = self._compute(R_lmn, Z_lmn)
        return (jnp.atleast_1d(V) - self.target) * self.weight

    def callback(self, R_lmn, Z_lmn, **kwargs):
        """Print plamsa volume.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).

        """
        V = self._compute(R_lmn, Z_lmn)
        print("Plasma volume: {:10.3e} (m^3)".format(V))
        return None

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return True

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return False

    @property
    def name(self):
        """Name of objective function (str)."""
        return "volume"


class AspectRatio(_Objective):
    """Aspect ratio = major radius / minor radius."""

    def __init__(self, eq=None, target=2, weight=1, grid=None):
        """Initialize an AspectRatio Objective.

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
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.

        """
        self.grid = grid
        super().__init__(eq=eq, target=target, weight=weight)

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
            self.grid = QuadratureGrid(L=eq.L, M=eq.M, N=eq.N, NFP=eq.NFP)

        self._dim_f = 1

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=data_index["V"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["V"]["R_derivs"], build=True
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, R_lmn, Z_lmn):
        data = compute_geometry(R_lmn, Z_lmn, self._R_transform, self._Z_transform)
        return data["R0/a"]

    def compute(self, R_lmn, Z_lmn, **kwargs):
        """Compute aspect ratio.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).

        Returns
        -------
        AR : float
            Aspect ratio, dimensionless.

        """
        AR = self._compute(R_lmn, Z_lmn)
        return (jnp.atleast_1d(AR) - self.target) * self.weight

    def callback(self, R_lmn, Z_lmn, **kwargs):
        """Print aspect ratio.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).

        """
        AR = self._compute(R_lmn, Z_lmn)
        print("Aspect ratio: {:10.3e} (dimensionless)".format(AR))
        return None

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return True

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return False

    @property
    def name(self):
        """Name of objective function (str)."""
        return "aspect ratio"


class Energy(_Objective):
    """MHD energy.

    W = integral( B^2 / (2*mu0) + p / (gamma - 1) ) dV  (J)

    """

    _io_attrs_ = _Objective._io_attrs_ + ["gamma"]

    def __init__(self, eq=None, target=0, weight=1, grid=None, gamma=0):
        """Initialize an Energy Objective.

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
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.
        gamma : float, optional
            Adiabatic (compressional) index. Default = 0.

        """
        self.grid = grid
        self.gamma = gamma
        super().__init__(eq=eq, target=target, weight=weight)

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
            self.grid = QuadratureGrid(L=eq.L, M=eq.M, N=eq.N, NFP=eq.NFP)

        self._dim_f = 1

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._iota = eq.iota.copy()
        self._pressure = eq.pressure.copy()
        self._iota.grid = self.grid
        self._pressure.grid = self.grid

        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=data_index["W"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["W"]["R_derivs"], build=True
        )
        self._L_transform = Transform(
            self.grid, eq.L_basis, derivs=data_index["W"]["L_derivs"], build=True
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi):
        data = compute_energy(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            p_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
            self._pressure,
            self._gamma,
        )
        return data["W"], data["W_B"], data["W_p"]

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi, **kwargs):
        """Compute MHD energy.

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
        p_l : ndarray
            Spectral coefficients of p(rho) -- pressure profile.
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        W : float
            Total MHD energy in the plasma volume (J).

        """
        W, W_B, W_p = self._compute(R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi)
        return (jnp.atleast_1d(W) - self.target) * self.weight

    def callback(self, R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi, **kwargs):
        """Print MHD energy.

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
        p_l : ndarray
            Spectral coefficients of p(rho) -- pressure profile.
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        """
        W, W_B, W_p = self._compute(R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi)
        print(
            "Total MHD energy: {:10.3e}, ".format(W)
            + "Magnetic Energy: {:10.3e}, ".format(W_B)
            + "Pressure Energy: {:10.3e} ".format(W_p)
            + "(J)"
        )
        return None

    @property
    def gamma(self):
        """float: Adiabatic (compressional) index."""
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return True

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return False

    @property
    def name(self):
        """Name of objective function (str)."""
        return "energy"


class ToroidalCurrent(_Objective):
    """Toroidal current encolsed by a surface."""

    def __init__(self, eq=None, target=0, weight=1, grid=None):
        """Initialize a ToroidalCurrent Objective.

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
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.

        """
        self.grid = grid
        super().__init__(eq=eq, target=target, weight=weight)

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
                L=1,
                M=2 * eq.M_grid + 1,
                N=2 * eq.N_grid + 1,
                NFP=eq.NFP,
                sym=eq.sym,
                rho=1,
            )

        self._dim_f = 1

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._iota = eq.iota.copy()
        self._iota.grid = self.grid

        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=data_index["I"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["I"]["R_derivs"], build=True
        )
        self._L_transform = Transform(
            self.grid, eq.L_basis, derivs=data_index["I"]["L_derivs"], build=True
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, R_lmn, Z_lmn, L_lmn, i_l, Psi):
        data = compute_quasisymmetry_error(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
        )
        return 2 * np.pi / mu_0 * data["I"]

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, Psi, **kwargs):
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
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        I : float
            Toroidal current (A).

        """
        I = self._compute(R_lmn, Z_lmn, L_lmn, i_l, Psi)
        return (I - self.target) * self.weight

    def callback(self, R_lmn, Z_lmn, L_lmn, i_l, Psi, **kwargs):
        """Print toroidal current.

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
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        """
        I = self._compute(R_lmn, Z_lmn, L_lmn, i_l, Psi)
        print("Toroidal current: {:10.3e} ".format(I) + "(A)")
        return None

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return True

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return False

    @property
    def name(self):
        """Name of objective function (str)."""
        return "toroidal current"


# non-scalar nonlinear objectives


class RadialForceBalance(_Objective):
    """Radial MHD force balance.

    F_rho = sqrt(g) (B^zeta J^theta - B^theta J^zeta) - grad(p)
    f_rho = F_rho |grad(rho)| dV  (N)

    """

    def __init__(self, eq=None, target=0, weight=1, grid=None, norm=False):
        """Initialize a RadialForceBalance Objective.

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
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.
        norm : bool, optional
            Whether to normalize the objective values (make dimensionless).

        """
        self.grid = grid
        self.norm = norm
        super().__init__(eq=eq, target=target, weight=weight)

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
            self.grid = ConcentricGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
                rotation="cos",
                node_pattern=eq.node_pattern,
            )

        self._dim_f = self.grid.num_nodes

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._pressure = eq.pressure.copy()
        self._iota = eq.iota.copy()
        self._pressure.grid = self.grid
        self._iota.grid = self.grid

        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=data_index["F_rho"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["F_rho"]["R_derivs"], build=True
        )
        self._L_transform = Transform(
            self.grid, eq.L_basis, derivs=data_index["F_rho"]["L_derivs"], build=True
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi):
        data = compute_force_error(
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
        f = data["F_rho"] * data["|grad(rho)|"]
        if self.norm:
            f = f / data["|grad(p)|"]
        f = f * data["sqrt(g)"] * self.grid.weights
        # XXX: when normalized this has units of m^3 ?
        return f

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi, **kwargs):
        """Compute radial MHD force balance errors.

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
        p_l : ndarray
            Spectral coefficients of p(rho) -- pressure profile.
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        f_rho : ndarray
            Radial MHD force balance error at each node (N).

        """
        f = self._compute(R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi)
        return (f - self.target) * self.weight

    def callback(self, R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi, **kwargs):
        """Print radial MHD force balance error.

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
        p_l : ndarray
            Spectral coefficients of p(rho) -- pressure profile.
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        """
        f = self._compute(R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi)
        if self.norm:
            units = "(normalized)"
        else:
            units = "(N)"
        print("Radial force: {:10.3e}, ".format(jnp.linalg.norm(f)) + units)
        return None

    @property
    def norm(self):
        """bool: Whether the objective values are normalized."""
        return self._norm

    @norm.setter
    def norm(self, norm):
        self._norm = norm

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return False

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return False

    @property
    def name(self):
        """Name of objective function (str)."""
        return "radial force"


class HelicalForceBalance(_Objective):
    """Helical MHD force balance.

    F_beta = sqrt(g) J^rho
    beta = -B^zeta grad(theta) + B^theta grad(zeta)
    f_beta = F_beta |beta| dV  (N)

    """

    def __init__(self, eq=None, target=0, weight=1, grid=None, norm=False):
        """Initialize a HelicalForceBalance Objective.

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
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.
        norm : bool, optional
            Whether to normalize the objective values (make dimensionless).

        """
        self.grid = grid
        self.norm = norm
        super().__init__(eq=eq, target=target, weight=weight)

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
            self.grid = ConcentricGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
                rotation="sin",
                node_pattern=eq.node_pattern,
            )

        self._dim_f = self.grid.num_nodes

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._iota = eq.iota.copy()
        self._pressure = eq.pressure.copy()
        self._iota.grid = self.grid
        self._pressure.grid = self.grid

        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=data_index["F_beta"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["F_beta"]["R_derivs"], build=True
        )
        self._L_transform = Transform(
            self.grid, eq.L_basis, derivs=data_index["F_beta"]["L_derivs"], build=True
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi):
        data = compute_force_error(
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
        f = data["F_beta"] * data["|beta|"]
        if self.norm:
            f = f / data["|grad(p)|"]
        f = f * data["sqrt(g)"] * self.grid.weights
        # XXX: when normalized this has units of m^3 ?
        return f

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi, **kwargs):
        """Compute helical MHD force balance errors.

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
        p_l : ndarray
            Spectral coefficients of p(rho) -- pressure profile.
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        f : ndarray
            Helical MHD force balance error at each node (N).

        """
        f = self._compute(R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi)
        return (f - self.target) * self.weight

    def callback(self, R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi, **kwargs):
        """Print helical MHD force balance error.

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
        p_l : ndarray
            Spectral coefficients of p(rho) -- pressure profile.
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        """
        f = self._compute(R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi)
        if self.norm:
            units = "(normalized)"
        else:
            units = "(N)"
        print("Helical force: {:10.3e}, ".format(jnp.linalg.norm(f)) + units)
        return None

    @property
    def norm(self):
        """bool: Whether the objective values are normalized."""
        return self._norm

    @norm.setter
    def norm(self, norm):
        self._norm = norm

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return False

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return False

    @property
    def name(self):
        """Name of objective function (str)."""
        return "helical force"


class RadialCurrentDensity(_Objective):
    """Radial current density."""

    def __init__(self, eq=None, target=0, weight=1, grid=None, norm=False):
        """Initialize a RadialCurrentDensity Objective.

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
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.
        norm : bool, optional
            Whether to normalize the objective values (make dimensionless).

        """
        self.grid = grid
        self.norm = norm
        super().__init__(eq=eq, target=target, weight=weight)

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
            self.grid = ConcentricGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
                rotation="sin",
                node_pattern=eq.node_pattern,
            )

        self._dim_f = self.grid.num_nodes

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._iota = eq.iota.copy()
        self._iota.grid = self.grid

        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=data_index["J^rho"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["J^rho"]["R_derivs"], build=True
        )
        self._L_transform = Transform(
            self.grid, eq.L_basis, derivs=data_index["J^rho"]["L_derivs"], build=True
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, R_lmn, Z_lmn, L_lmn, i_l, Psi):
        data = compute_contravariant_current_density(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
        )
        data = compute_covariant_metric_coefficients(
            R_lmn, Z_lmn, self._R_transform, self._Z_transform, data=data
        )
        f = data["J^rho"] * jnp.sqrt(data["g_rr"])
        if self.norm:
            data = compute_magnetic_field_magnitude(
                R_lmn,
                Z_lmn,
                L_lmn,
                i_l,
                Psi,
                self._R_transform,
                self._Z_transform,
                self._L_transform,
                self._iota,
            )
            B = jnp.mean(data["|B|"] * data["sqrt(g)"]) / jnp.mean(data["sqrt(g)"])
            R = jnp.mean(data["R"] * data["sqrt(g)"]) / jnp.mean(data["sqrt(g)"])
            f = f * mu_0 / (B * R ** 2)
        f = f * data["sqrt(g)"] * self.grid.weights
        # XXX: when normalized this has units of m^3 ?
        return f

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, Psi, **kwargs):
        """Compute radial current density.

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
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        f : ndarray
            Radial current at each node (A*m).

        """
        f = self._compute(R_lmn, Z_lmn, L_lmn, i_l, Psi)
        return (f - self.target) * self.weight

    def callback(self, R_lmn, Z_lmn, L_lmn, i_l, Psi, **kwargs):
        """Print radial current density.

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
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        """
        f = self._compute(R_lmn, Z_lmn, L_lmn, i_l, Psi)
        if self.norm:
            units = "(normalized)"
        else:
            units = "(A*m)"
        print("Radial current: {:10.3e} ".format(jnp.linalg.norm(f)) + units)
        return None

    @property
    def norm(self):
        """bool: Whether the objective values are normalized."""
        return self._norm

    @norm.setter
    def norm(self, norm):
        self._norm = norm

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return False

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return False

    @property
    def name(self):
        """Name of objective function (str)."""
        return "radial current density"


class PoloidalCurrentDensity(_Objective):
    """Poloidal current density."""

    def __init__(self, eq=None, target=0, weight=1, grid=None, norm=False):
        """Initialize a PoloidalCurrentDensity Objective.

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
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.
        norm : bool, optional
            Whether to normalize the objective values (make dimensionless).

        """
        self.grid = grid
        self.norm = norm
        super().__init__(eq=eq, target=target, weight=weight)

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
            self.grid = ConcentricGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
                rotation="cos",
                node_pattern=eq.node_pattern,
            )

        self._dim_f = self.grid.num_nodes

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._iota = eq.iota.copy()
        self._iota.grid = self.grid

        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=data_index["J^theta"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["J^theta"]["R_derivs"], build=True
        )
        self._L_transform = Transform(
            self.grid, eq.L_basis, derivs=data_index["J^theta"]["L_derivs"], build=True
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, R_lmn, Z_lmn, L_lmn, i_l, Psi):
        data = compute_contravariant_current_density(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
        )
        data = compute_covariant_metric_coefficients(
            R_lmn, Z_lmn, self._R_transform, self._Z_transform, data=data
        )
        f = data["J^theta"] * jnp.sqrt(data["g_tt"])
        if self.norm:
            data = compute_magnetic_field_magnitude(
                R_lmn,
                Z_lmn,
                L_lmn,
                i_l,
                Psi,
                self._R_transform,
                self._Z_transform,
                self._L_transform,
                self._iota,
            )
            B = jnp.mean(data["|B|"] * data["sqrt(g)"]) / jnp.mean(data["sqrt(g)"])
            R = jnp.mean(data["R"] * data["sqrt(g)"]) / jnp.mean(data["sqrt(g)"])
            f = f * mu_0 / (B * R ** 2)
        f = f * data["sqrt(g)"] * self.grid.weights
        # XXX: when normalized this has units of m^3 ?
        return f

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, Psi, **kwargs):
        """Compute poloidal current density.

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
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        f : ndarray
            Poloidal current at each node (A*m).

        """
        f = self._compute(R_lmn, Z_lmn, L_lmn, i_l, Psi)
        return (f - self.target) * self.weight

    def callback(self, R_lmn, Z_lmn, L_lmn, i_l, Psi, **kwargs):
        """Print poloidal current density.

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
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        """
        f = self._compute(R_lmn, Z_lmn, L_lmn, i_l, Psi)
        if self.norm:
            units = "(normalized)"
        else:
            units = "(A*m)"
        print("Poloidal current: {:10.3e} ".format(jnp.linalg.norm(f)) + units)
        return None

    @property
    def norm(self):
        """bool: Whether the objective values are normalized."""
        return self._norm

    @norm.setter
    def norm(self, norm):
        self._norm = norm

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return False

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return False

    @property
    def name(self):
        """Name of objective function (str)."""
        return "poloidal current"


class ToroidalCurrentDensity(_Objective):
    """Toroidal current density."""

    def __init__(self, eq=None, target=0, weight=1, grid=None, norm=False):
        """Initialize a ToroidalCurrentDensity Objective.

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
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.
        norm : bool, optional
            Whether to normalize the objective values (make dimensionless).

        """
        self.grid = grid
        self.norm = norm
        super().__init__(eq=eq, target=target, weight=weight)

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
            self.grid = ConcentricGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
                rotation="cos",
                node_pattern=eq.node_pattern,
            )

        self._dim_f = self.grid.num_nodes

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._iota = eq.iota.copy()
        self._iota.grid = self.grid

        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=data_index["J^zeta"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["J^zeta"]["R_derivs"], build=True
        )
        self._L_transform = Transform(
            self.grid, eq.L_basis, derivs=data_index["J^zeta"]["L_derivs"], build=True
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, R_lmn, Z_lmn, L_lmn, i_l, Psi):
        data = compute_contravariant_current_density(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
        )
        data = compute_covariant_metric_coefficients(
            R_lmn, Z_lmn, self._R_transform, self._Z_transform, data=data
        )
        f = data["J^zeta"] * jnp.sqrt(data["g_zz"])
        if self.norm:
            data = compute_magnetic_field_magnitude(
                R_lmn,
                Z_lmn,
                L_lmn,
                i_l,
                Psi,
                self._R_transform,
                self._Z_transform,
                self._L_transform,
                self._iota,
            )
            B = jnp.mean(data["|B|"] * data["sqrt(g)"]) / jnp.mean(data["sqrt(g)"])
            R = jnp.mean(data["R"] * data["sqrt(g)"]) / jnp.mean(data["sqrt(g)"])
            f = f * mu_0 / (B * R ** 2)
        f = f * data["sqrt(g)"] * self.grid.weights
        # XXX: when normalized this has units of m^3 ?
        return f

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, Psi, **kwargs):
        """Compute toroidal current density.

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
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        f : ndarray
            Toroidal current at each node (A*m).

        """
        f = self._compute(R_lmn, Z_lmn, L_lmn, i_l, Psi)
        return (f - self.target) * self.weight

    def callback(self, R_lmn, Z_lmn, L_lmn, i_l, Psi, **kwargs):
        """Print toroidal current density.

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
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        """
        f = self._compute(R_lmn, Z_lmn, L_lmn, i_l, Psi)
        if self.norm:
            units = "(normalized)"
        else:
            units = "(A*m)"
        print("Toroidal current: {:10.3e} ".format(jnp.linalg.norm(f)) + units)
        return None

    @property
    def norm(self):
        """bool: Whether the objective values are normalized."""
        return self._norm

    @norm.setter
    def norm(self, norm):
        self._norm = norm

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return False

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return False

    @property
    def name(self):
        """Name of objective function (str)."""
        return "toroidal current"


class QuasisymmetryBoozer(_Objective):
    """Quasi-symmetry Boozer harmonics error."""

    def __init__(
        self,
        eq=None,
        target=0,
        weight=1,
        grid=None,
        helicity=(1, 0),
        M_booz=None,
        N_booz=None,
        norm=False,
    ):
        """Initialize a QuasisymmetryBoozer Objective.

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
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.
        helicity : tuple, optional
            Type of quasi-symmetry (M, N). Default = quasi-axisymmetry (1, 0).
        M_booz : int, optional
            Poloidal resolution of Boozer transformation. Default = 2 * eq.M.
        N_booz : int, optional
            Toroidal resolution of Boozer transformation. Default = 2 * eq.N.
        norm : bool, optional
            Whether to normalize the objective values (make dimensionless).

        """
        self.grid = grid
        self.helicity = helicity
        self.M_booz = M_booz
        self.N_booz = N_booz
        self.norm = norm
        super().__init__(eq=eq, target=target, weight=weight)

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
                L=1,
                M=4 * eq.M_grid + 1,
                N=4 * eq.N_grid + 1,
                NFP=eq.NFP,
                sym=False,
                rho=1,
            )
        if self.M_booz is None:
            self.M_booz = 2 * eq.M
        if self.N_booz is None:
            self.N_booz = 2 * eq.N

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._iota = eq.iota.copy()
        self._iota.grid = self.grid

        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=data_index["|B|_mn"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["|B|_mn"]["R_derivs"], build=True
        )
        self._L_transform = Transform(
            self.grid, eq.L_basis, derivs=data_index["|B|_mn"]["L_derivs"], build=True
        )
        self._B_transform = Transform(
            self.grid,
            DoubleFourierSeries(
                M=self.M_booz, N=self.N_booz, NFP=eq.NFP, sym=eq.R_basis.sym
            ),
            derivs=data_index["|B|_mn"]["R_derivs"],
            build=True,
            build_pinv=True,
        )
        self._w_transform = Transform(
            self.grid,
            DoubleFourierSeries(
                M=self.M_booz, N=self.N_booz, NFP=eq.NFP, sym=eq.Z_basis.sym
            ),
            derivs=data_index["|B|_mn"]["L_derivs"],
            build=True,
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        M = self.helicity[0]
        N = self.helicity[1] / eq.NFP
        self._idx_00 = np.where(
            (self._B_transform.basis.modes == [0, 0, 0]).all(axis=1)
        )[0]
        if N == 0:
            self._idx_MN = np.where(self._B_transform.basis.modes[:, 2] == 0)[0]
        else:
            self._idx_MN = np.where(
                self._B_transform.basis.modes[:, 1]
                / self._B_transform.basis.modes[:, 2]
                == M / N
            )[0]
        self._idx = np.ones((self._B_transform.basis.num_modes,), bool)
        self._idx[self._idx_00] = False
        self._idx[self._idx_MN] = False

        self._dim_f = np.sum(self._idx)

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, R_lmn, Z_lmn, L_lmn, i_l, Psi):
        data = compute_boozer_coords(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._B_transform,
            self._w_transform,
            self._iota,
        )
        b_mn = data["|B|_mn"]
        if self.norm:
            b_mn = b_mn / jnp.sqrt(jnp.sum(b_mn ** 2))
        return b_mn[self._idx]

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, Psi, **kwargs):
        """Compute quasi-symmetry Boozer harmonics error.

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
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        f : ndarray
            Quasi-symmetry flux function error at each node (T^3).

        """
        f = self._compute(R_lmn, Z_lmn, L_lmn, i_l, Psi)
        return (f - self.target) * self.weight

    def callback(self, R_lmn, Z_lmn, L_lmn, i_l, Psi, **kwargs):
        """Print quasi-symmetry Boozer harmonics error.

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
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        """
        f = self._compute(R_lmn, Z_lmn, L_lmn, i_l, Psi)
        if self.norm:
            units = "(normalized)"
        else:
            units = "(T)"
        print(
            "Quasi-symmetry ({},{}) Boozer error: {:10.3e} ".format(
                self.helicity[0], self.helicity[1], jnp.linalg.norm(f)
            )
            + units
        )
        return None

    @property
    def helicity(self):
        """tuple: Type of quasi-symmetry (M, N)."""
        return self._helicity

    @helicity.setter
    def helicity(self, helicity):
        self._helicity = helicity

    @property
    def norm(self):
        """bool: Whether the objective values are normalized."""
        return self._norm

    @norm.setter
    def norm(self, norm):
        self._norm = norm

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return False

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return False

    @property
    def name(self):
        """Name of objective function (str)."""
        return "QS Boozer"


class QuasisymmetryFluxFunction(_Objective):
    """Quasi-symmetry flux function error."""

    def __init__(
        self, eq=None, target=0, weight=1, grid=None, helicity=(1, 0), norm=False
    ):
        """Initialize a QuasisymmetryFluxFunction Objective.

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
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.
        helicity : tuple, optional
            Type of quasi-symmetry (M, N).
        norm : bool, optional
            Whether to normalize the objective values (make dimensionless).

        """
        self.grid = grid
        self.helicity = helicity
        self.norm = norm
        super().__init__(eq=eq, target=target, weight=weight)

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
                L=1,
                M=2 * eq.M_grid + 1,
                N=2 * eq.N_grid + 1,
                NFP=eq.NFP,
                sym=eq.sym,
                rho=1,
            )

        self._dim_f = self.grid.num_nodes

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._iota = eq.iota.copy()
        self._iota.grid = self.grid

        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=data_index["f_C"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["f_C"]["R_derivs"], build=True
        )
        self._L_transform = Transform(
            self.grid, eq.L_basis, derivs=data_index["f_C"]["L_derivs"], build=True
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, R_lmn, Z_lmn, L_lmn, i_l, Psi):
        data = compute_quasisymmetry_error(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
            self._helicity,
        )
        f = data["f_C"] * self.grid.weights
        if self.norm:
            B = jnp.mean(data["|B|"] * data["sqrt(g)"]) / jnp.mean(data["sqrt(g)"])
            f = f / B ** 3
        return f

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, Psi, **kwargs):
        """Compute quasi-symmetry flux function errors.

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
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        f : ndarray
            Quasi-symmetry flux function error at each node (T^3).

        """
        f = self._compute(R_lmn, Z_lmn, L_lmn, i_l, Psi)
        return (f - self.target) * self.weight

    def callback(self, R_lmn, Z_lmn, L_lmn, i_l, Psi, **kwargs):
        """Print quasi-symmetry flux function error.

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
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        """
        f = self._compute(R_lmn, Z_lmn, L_lmn, i_l, Psi)
        if self.norm:
            units = "(normalized)"
        else:
            units = "(T^3)"
        print(
            "Quasi-symmetry ({},{}) error: {:10.3e} ".format(
                self.helicity[0], self.helicity[1], jnp.linalg.norm(f)
            )
            + units
        )
        return None

    @property
    def helicity(self):
        """tuple: Type of quasi-symmetry (M, N)."""
        return self._helicity

    @helicity.setter
    def helicity(self, helicity):
        self._helicity = helicity

    @property
    def norm(self):
        """bool: Whether the objective values are normalized."""
        return self._norm

    @norm.setter
    def norm(self, norm):
        self._norm = norm

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return False

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return False

    @property
    def name(self):
        """Name of objective function (str)."""
        return "QS flux function"


class QuasisymmetryTripleProduct(_Objective):
    """Quasi-symmetry triple product error."""

    def __init__(self, eq=None, target=0, weight=1, grid=None, norm=False):
        """Initialize a QuasisymmetryTripleProduct Objective.

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
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.
        norm : bool, optional
            Whether to normalize the objective values (make dimensionless).

        """
        self.grid = grid
        self.norm = norm
        super().__init__(eq=eq, target=target, weight=weight)

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
                L=1,
                M=2 * eq.M_grid + 1,
                N=2 * eq.N_grid + 1,
                NFP=eq.NFP,
                sym=eq.sym,
                rho=1,
            )

        self._dim_f = self.grid.num_nodes

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._iota = eq.iota.copy()
        self._iota.grid = self.grid

        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=data_index["f_T"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["f_T"]["R_derivs"], build=True
        )
        self._L_transform = Transform(
            self.grid, eq.L_basis, derivs=data_index["f_T"]["L_derivs"], build=True
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, R_lmn, Z_lmn, L_lmn, i_l, Psi):
        data = compute_quasisymmetry_error(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
        )
        f = data["f_T"] * self.grid.weights
        if self.norm:
            B = jnp.mean(data["|B|"] * data["sqrt(g)"]) / jnp.mean(data["sqrt(g)"])
            R = jnp.mean(data["R"] * data["sqrt(g)"]) / jnp.mean(data["sqrt(g)"])
            f = f * R ** 2 / B ** 4
        return f

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, Psi, **kwargs):
        """Compute quasi-symmetry triple product errors.

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
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        f : ndarray
            Quasi-symmetry flux function error at each node (T^4/m^2).

        """
        f = self._compute(R_lmn, Z_lmn, L_lmn, i_l, Psi)
        return (f - self.target) * self.weight

    def callback(self, R_lmn, Z_lmn, L_lmn, i_l, Psi, **kwargs):
        """Print quasi-symmetry triple product error.

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
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        """
        f = self._compute(R_lmn, Z_lmn, L_lmn, i_l, Psi)
        if self.norm:
            units = "(normalized)"
        else:
            units = "(T^4/m^2)"
        print("Quasi-symmetry error: {:10.3e} ".format(jnp.linalg.norm(f)) + units)
        return None

    @property
    def norm(self):
        """bool: Whether the objective values are normalized."""
        return self._norm

    @norm.setter
    def norm(self, norm):
        self._norm = norm

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return False

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return False

    @property
    def name(self):
        """Name of objective function (str)."""
        return "QS triple product"
