"""Objectives for solving equilibrium problems."""

import warnings

from termcolor import colored

from desc.backend import jnp
from desc.compute import (
    compute_contravariant_current_density,
    compute_energy,
    compute_force_error,
    data_index,
)
from desc.grid import ConcentricGrid, QuadratureGrid
from desc.transform import Transform
from desc.utils import Timer

from .objective_funs import _Objective


class ForceBalance(_Objective):
    """Radial and helical MHD force balance.

    F_rho = sqrt(g) (B^zeta J^theta - B^theta J^zeta) - grad(p)
    f_rho = F_rho |grad(rho)| dV  (N)

    F_beta = sqrt(g) J^rho
    beta = -B^zeta grad(theta) + B^theta grad(zeta)
    f_beta = F_beta |beta| dV  (N)

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
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = False

    def __init__(self, eq=None, target=0, weight=1, grid=None, name="force"):

        self.grid = grid
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        units = "(N)"
        self._print_value_fmt = "Total force: {:10.3e} " + units

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
            if eq.node_pattern is None or eq.node_pattern in [
                "jacobi",
                "cheb1",
                "cheb2",
                "ocs",
                "linear",
            ]:
                self.grid = ConcentricGrid(
                    L=eq.L_grid,
                    M=eq.M_grid,
                    N=eq.N_grid,
                    NFP=eq.NFP,
                    sym=eq.sym,
                    axis=False,
                    node_pattern=eq.node_pattern,
                )
            elif eq.node_pattern == "quad":
                self.grid = QuadratureGrid(
                    L=eq.L_grid,
                    M=eq.M_grid,
                    N=eq.N_grid,
                    NFP=eq.NFP,
                )

        self._dim_f = 2 * self.grid.num_nodes

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

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, R_lmn, Z_lmn, L_lmn, p_l, i_l, c_l, Psi, **kwargs):
        """Compute MHD force balance errors.

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
        f : ndarray
            MHD force balance error at each node (N).

        """
        data = compute_force_error(
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
        fr = data["F_rho"] * data["|grad(rho)|"]
        fr = fr * data["sqrt(g)"] * self.grid.weights

        fb = data["F_beta"] * data["|beta|"]
        fb = fb * data["sqrt(g)"] * self.grid.weights

        f = jnp.concatenate([fr, fb])
        return self._shift_scale(f)


class RadialForceBalance(_Objective):
    """Radial MHD force balance.

    F_rho = sqrt(g) (B^zeta J^theta - B^theta J^zeta) - grad(p)
    f_rho = F_rho |grad(rho)| dV  (N)

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
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = False

    def __init__(self, eq=None, target=0, weight=1, grid=None, name="radial force"):

        self.grid = grid
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        units = "(N)"
        self._print_value_fmt = "Radial force: {:10.3e} " + units

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
            if eq.node_pattern is None or eq.node_pattern in [
                "jacobi",
                "cheb1",
                "cheb2",
                "ocs",
                "linear",
            ]:
                self.grid = ConcentricGrid(
                    L=eq.L_grid,
                    M=eq.M_grid,
                    N=eq.N_grid,
                    NFP=eq.NFP,
                    sym=eq.sym,
                    axis=False,
                    node_pattern=eq.node_pattern,
                )
            elif eq.node_pattern == "quad":
                self.grid = QuadratureGrid(
                    L=eq.L_grid,
                    M=eq.M_grid,
                    N=eq.N_grid,
                    NFP=eq.NFP,
                )

        self._dim_f = self.grid.num_nodes

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

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, R_lmn, Z_lmn, L_lmn, p_l, i_l, c_l, Psi, **kwargs):
        """Compute radial MHD force balance errors.

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
        f_rho : ndarray
            Radial MHD force balance error at each node (N).

        """
        data = compute_force_error(
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
        f = data["F_rho"] * data["|grad(rho)|"]
        f = f * data["sqrt(g)"] * self.grid.weights

        return self._shift_scale(f)


class HelicalForceBalance(_Objective):
    """Helical MHD force balance.

    F_beta = sqrt(g) J^rho
    beta = -B^zeta grad(theta) + B^theta grad(zeta)
    f_beta = F_beta |beta| dV  (N)

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
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = False

    def __init__(self, eq=None, target=0, weight=1, grid=None, name="helical force"):

        self.grid = grid
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        units = "(N)"
        self._print_value_fmt = "Helical force: {:10.3e}, " + units

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
            if eq.node_pattern is None or eq.node_pattern in [
                "jacobi",
                "cheb1",
                "cheb2",
                "ocs",
                "linear",
            ]:
                self.grid = ConcentricGrid(
                    L=eq.L_grid,
                    M=eq.M_grid,
                    N=eq.N_grid,
                    NFP=eq.NFP,
                    sym=eq.sym,
                    axis=False,
                    node_pattern=eq.node_pattern,
                )
            elif eq.node_pattern == "quad":
                self.grid = QuadratureGrid(
                    L=eq.L_grid,
                    M=eq.M_grid,
                    N=eq.N_grid,
                    NFP=eq.NFP,
                )

        self._dim_f = self.grid.num_nodes

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

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, R_lmn, Z_lmn, L_lmn, p_l, i_l, c_l, Psi, **kwargs):
        """Compute helical MHD force balance errors.

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
        f : ndarray
            Helical MHD force balance error at each node (N).

        """
        data = compute_force_error(
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
        f = data["F_beta"] * data["|beta|"]
        f = f * data["sqrt(g)"] * self.grid.weights

        return self._shift_scale(f)


class Energy(_Objective):
    """MHD energy.

    W = integral( B^2 / (2*mu0) + p / (gamma - 1) ) dV  (J)

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
        This will default to a QuadratureGrid
    gamma : float, optional
        Adiabatic (compressional) index. Default = 0.
    name : str
        Name of the objective function.

    """

    _io_attrs_ = _Objective._io_attrs_ + ["gamma"]
    _scalar = True
    _linear = False

    def __init__(self, eq=None, target=0, weight=1, grid=None, gamma=0, name="energy"):

        self.grid = grid
        self.gamma = gamma
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        self._print_value_fmt = "Total MHD energy: {:10.3e} (J)"

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
            if eq.node_pattern in [
                "jacobi",
                "cheb1",
                "cheb2",
                "ocs",
                "linear",
            ]:
                warnings.warn(
                    colored(
                        "Energy objective built using grid "
                        + "that is not the quadrature grid! "
                        + "This is not recommended and may result in poor convergence. "
                        "yellow",
                    )
                )
                self.grid = ConcentricGrid(
                    L=eq.L_grid,
                    M=eq.M_grid,
                    N=eq.N_grid,
                    NFP=eq.NFP,
                    sym=eq.sym,
                    axis=False,
                    node_pattern=eq.node_pattern,
                )

            else:
                self.grid = QuadratureGrid(
                    L=eq.L_grid,
                    M=eq.M_grid,
                    N=eq.N_grid,
                    NFP=eq.NFP,
                )

        self._dim_f = 1

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

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, R_lmn, Z_lmn, L_lmn, p_l, i_l, c_l, Psi, **kwargs):
        """Compute MHD energy.

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
        W : float
            Total MHD energy in the plasma volume (J).

        """
        data = compute_energy(
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
            self._gamma,
        )
        return self._shift_scale(data["W"])

    @property
    def gamma(self):
        """float: Adiabatic (compressional) index."""
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma


class CurrentDensity(_Objective):
    """Radial, poloidal, and toroidal current density.

    Useful for solving vacuum equilibria.

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
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = False

    def __init__(
        self,
        eq=None,
        target=0,
        weight=1,
        grid=None,
        name="current density",
    ):

        self.grid = grid
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        units = "(A/m^2)"
        self._print_value_fmt = "Total current density: {:10.3e} " + units

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
            if eq.node_pattern is None or eq.node_pattern in [
                "jacobi",
                "cheb1",
                "cheb2",
                "ocs",
                "linear",
            ]:
                self.grid = ConcentricGrid(
                    L=eq.L_grid,
                    M=eq.M_grid,
                    N=eq.N_grid,
                    NFP=eq.NFP,
                    sym=eq.sym,
                    axis=False,
                    node_pattern=eq.node_pattern,
                )
            elif eq.node_pattern == "quad":
                self.grid = QuadratureGrid(
                    L=eq.L_grid,
                    M=eq.M_grid,
                    N=eq.N_grid,
                    NFP=eq.NFP,
                )

        self._dim_f = 3 * self.grid.num_nodes

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
            self.grid, eq.R_basis, derivs=data_index["J"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["J"]["R_derivs"], build=True
        )
        self._L_transform = Transform(
            self.grid, eq.L_basis, derivs=data_index["J"]["L_derivs"], build=True
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, c_l, Psi, **kwargs):
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
        c_l : ndarray
            Spectral coefficients of I(rho) -- toroidal current profile.
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        f : ndarray
            Toroidal current at each node (A*m).

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
        jr = data["J^rho"] * data["sqrt(g)"] * self.grid.weights
        jt = data["J^theta"] * data["sqrt(g)"] * self.grid.weights
        jz = data["J^zeta"] * data["sqrt(g)"] * self.grid.weights

        f = jnp.concatenate([jr, jt, jz])
        return self._shift_scale(f)
