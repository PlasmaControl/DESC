import numpy as np
from desc.backend import jnp
from desc.utils import Timer
from desc.grid import LinearGrid
from desc.basis import DoubleFourierSeries
from desc.transform import Transform
from desc.compute import (
    data_index,
    compute_boozer_coords,
    compute_quasisymmetry_error,
)
from .objective_funs import _Objective


class QuasisymmetryBoozer(_Objective):
    """Quasi-symmetry Boozer harmonics error.

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
        helicity=(1, 0),
        M_booz=None,
        N_booz=None,
        name="QS Boozer",
    ):

        self.grid = grid
        self.helicity = helicity
        self.M_booz = M_booz
        self.N_booz = N_booz
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        units = "(T)"
        self._callback_fmt = (
            "Quasi-symmetry ({},{}) Boozer error: ".format(
                self.helicity[0], self.helicity[1]
            )
            + "{:10.3e} "
            + units
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
        if self.M_booz is None:
            self.M_booz = 2 * eq.M
        if self.N_booz is None:
            self.N_booz = 2 * eq.N
        if self.grid is None:
            self.grid = LinearGrid(
                L=1,
                M=3 * self.M_booz + 1,
                N=3 * self.N_booz + 1,
                NFP=eq.NFP,
                sym=False,
                rho=1,
            )

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
        b_mn = b_mn[self._idx]

        return self._shift_scale(b_mn)

    @property
    def helicity(self):
        """tuple: Type of quasi-symmetry (M, N)."""
        return self._helicity

    @helicity.setter
    def helicity(self, helicity):
        assert (
            (len(helicity) == 2)
            and (int(helicity[0]) == helicity[0])
            and (int(helicity[1]) == helicity[1])
        )
        self._helicity = helicity
        if hasattr(self, "_callback_fmt"):
            units = "(T)"
            self._callback_fmt = (
                "Quasi-symmetry ({},{}) Boozer error: ".format(
                    self.helicity[0], self.helicity[1]
                )
                + "{:10.3e} "
                + units
            )


class QuasisymmetryTwoTerm(_Objective):
    """Quasi-symmetry two-term error.

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
        helicity=(1, 0),
        name="QS two-term",
    ):

        self.grid = grid
        self.helicity = helicity
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        units = "(T^3)"
        self._callback_fmt = (
            "Quasi-symmetry ({},{}) error: ".format(self.helicity[0], self.helicity[1])
            + "{:10.3e} "
            + units
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

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, Psi, **kwargs):
        """Compute quasi-symmetry two-term errors.

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

        return self._shift_scale(f)

    @property
    def helicity(self):
        """tuple: Type of quasi-symmetry (M, N)."""
        return self._helicity

    @helicity.setter
    def helicity(self, helicity):
        assert (
            (len(helicity) == 2)
            and (int(helicity[0]) == helicity[0])
            and (int(helicity[1]) == helicity[1])
        )
        self._helicity = helicity
        if hasattr(self, "_callback_fmt"):
            units = "(T^3)"
            self._callback_fmt = (
                "Quasi-symmetry ({},{}) error: ".format(
                    self.helicity[0], self.helicity[1]
                )
                + "{:10.3e} "
                + units
            )


class QuasisymmetryTripleProduct(_Objective):
    """Quasi-symmetry triple product error.

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
        name="QS triple product",
    ):

        self.grid = grid
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        units = "(T^4/m^2)"
        self._callback_fmt = "Quasi-symmetry error: {:10.3e} " + units

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

        return self._shift_scale(f)
