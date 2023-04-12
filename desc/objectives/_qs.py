"""Objectives for targeting quasisymmetry."""

import warnings

import numpy as np

from desc.backend import jnp
from desc.basis import DoubleFourierSeries
from desc.compute import compute as compute_fun
from desc.compute import get_params, get_profiles, get_transforms
from desc.grid import LinearGrid
from desc.interpolate import interp1d
from desc.utils import Timer
from desc.vmec_utils import ptolemy_linear_transform

from .normalization import compute_scaling_factors
from .objective_funs import _Objective


class QuasisymmetryBoozer(_Objective):
    """Quasi-symmetry Boozer harmonics error.

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
        Must be a LinearGrid with a single flux surface and sym=False.
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
    _units = "(T)"
    _print_value_fmt = "Quasi-symmetry Boozer error: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=0,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        grid=None,
        helicity=(1, 0),
        M_booz=None,
        N_booz=None,
        name="QS Boozer",
    ):

        assert len(helicity) == 2
        assert (int(helicity[0]) == helicity[0]) and (int(helicity[1]) == helicity[1])
        self.grid = grid
        self.helicity = helicity
        self.M_booz = M_booz
        self.N_booz = N_booz
        super().__init__(
            eq=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

        self._print_value_fmt = (
            "Quasi-symmetry ({},{}) Boozer error: ".format(
                self.helicity[0], self.helicity[1]
            )
            + "{:10.3e} "
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
                M=2 * self.M_booz, N=2 * self.N_booz, NFP=eq.NFP, sym=False
            )
        if self.M_booz is None:
            self.M_booz = 2 * eq.M
        if self.N_booz is None:
            self.N_booz = 2 * eq.N

        self._data_keys = ["|B|_mn"]
        self._args = get_params(self._data_keys)

        assert self.grid.sym is False
        assert self.grid.num_rho == 1

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._profiles = get_profiles(self._data_keys, eq=eq, grid=self.grid)
        self._transforms = get_transforms(
            self._data_keys,
            eq=eq,
            grid=self.grid,
            M_booz=self.M_booz,
            N_booz=self.N_booz,
        )
        self._matrix, self._modes, self._idx = ptolemy_linear_transform(
            self._transforms["B"].basis.modes,
            helicity=self.helicity,
            NFP=self._transforms["B"].basis.NFP,
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._dim_f = np.sum(self._idx)

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["B"]

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, *args, **kwargs):
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
        c_l : ndarray
            Spectral coefficients of I(rho) -- toroidal current profile.
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        f : ndarray
            Quasi-symmetry flux function error at each node (T^3).

        """
        params = self._parse_args(*args, **kwargs)
        data = compute_fun(
            self._data_keys,
            params=params,
            transforms=self._transforms,
            profiles=self._profiles,
        )
        B_mn = self._matrix @ data["|B|_mn"]
        return B_mn[self._idx]

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
        if hasattr(self, "_helicity") and self._helicity != helicity:
            self._built = False
        self._helicity = helicity
        if hasattr(self, "_print_value_fmt"):
            units = "(T)"
            self._print_value_fmt = (
                "Quasi-symmetry ({},{}) Boozer error: ".format(
                    self.helicity[0], self.helicity[1]
                )
                + "{:10.3e} "
                + units
            )
        warnings.warn("Re-build objective after changing the helicity!")


class QuasisymmetryTwoTerm(_Objective):
    """Quasi-symmetry two-term error.

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
    helicity : tuple, optional
        Type of quasi-symmetry (M, N).
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = False
    _units = "(T^3)"
    _print_value_fmt = "Quasi-symmetry two-term error: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=0,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        grid=None,
        helicity=(1, 0),
        name="QS two-term",
    ):

        self.grid = grid
        self.helicity = helicity
        super().__init__(
            eq=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

        self._print_value_fmt = (
            "Quasi-symmetry ({},{}) two-term error: ".format(
                self.helicity[0], self.helicity[1]
            )
            + "{:10.3e} "
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
            self.grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)

        self._dim_f = self.grid.num_nodes
        self._data_keys = ["f_C"]
        self._args = get_params(self._data_keys)

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._profiles = get_profiles(self._data_keys, eq=eq, grid=self.grid)
        self._transforms = get_transforms(self._data_keys, eq=eq, grid=self.grid)

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["B"] ** 3 / jnp.sqrt(self._dim_f)

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, *args, **kwargs):
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
        c_l : ndarray
            Spectral coefficients of I(rho) -- toroidal current profile.
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        f : ndarray
            Quasi-symmetry flux function error at each node (T^3).

        """
        params = self._parse_args(*args, **kwargs)
        data = compute_fun(
            self._data_keys,
            params=params,
            transforms=self._transforms,
            profiles=self._profiles,
            helicity=self.helicity,
        )
        return data["f_C"] * self.grid.weights

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
        if hasattr(self, "_helicity") and self._helicity != helicity:
            self._built = False
        self._helicity = helicity
        if hasattr(self, "_print_value_fmt"):
            units = "(T^3)"
            self._print_value_fmt = (
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
    _units = "(T^4/m^2)"
    _print_value_fmt = "Quasi-symmetry error: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=0,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        grid=None,
        name="QS triple product",
    ):

        self.grid = grid
        super().__init__(
            eq=eq,
            target=target,
            bounds=bounds,
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
            self.grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)

        self._dim_f = self.grid.num_nodes
        self._data_keys = ["f_T"]
        self._args = get_params(self._data_keys)

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._profiles = get_profiles(self._data_keys, eq=eq, grid=self.grid)
        self._transforms = get_transforms(self._data_keys, eq=eq, grid=self.grid)

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = (
                scales["B"] ** 4 / scales["a"] ** 2 / jnp.sqrt(self._dim_f)
            )

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, *args, **kwargs):
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
        c_l : ndarray
            Spectral coefficients of I(rho) -- toroidal current profile.
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        f : ndarray
            Quasi-symmetry flux function error at each node (T^4/m^2).

        """
        params = self._parse_args(*args, **kwargs)
        data = compute_fun(
            self._data_keys,
            params=params,
            transforms=self._transforms,
            profiles=self._profiles,
        )
        return data["f_T"] * self.grid.weights


class QuasiIsodynamic(_Objective):
    """Quasi-Isodynamic error.

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
    helicity : tuple, optional
        Type of omnigenity (M, N). Default = quasi-isodynamic (0, 1).
    L_QI : int
        Size of QI_l parameter. Default = 3.
    M_QI : int
        Poloidal resolution of QI_mn parameter. Default = 1.
    N_QI : int
        Toroidal resolution of QI_mn parameter. Default = 1.
    M_booz : int, optional
        Poloidal resolution of Boozer transformation. Default = 2 * eq.M.
    N_booz : int, optional
        Toroidal resolution of Boozer transformation. Default = 2 * eq.N.
    QI_l : ndarray, optional
        Initial parameters for QI well shape.
    QI_mn : ndarray, optional
        Initial parameters for QI well shift.
    well_weight : float, optional
        Weight applied to the bottom of the magnetic well (B_min) relative to the top
        of the magnetic well (B_max). Default = 1, which weights all points equally.
    name : str
        Name of the objective function.

    """

    _io_attrs_ = _Objective._io_attrs_ + ["QI_l", "QI_mn"]

    _scalar = False
    _linear = False
    _units = "(T)"
    _print_value_fmt = "Quasi-isodynamic error: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=0,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        grid=None,
        helicity=(1, 0),
        L_QI=3,
        M_QI=1,
        N_QI=1,
        M_booz=None,
        N_booz=None,
        QI_l=None,
        QI_mn=None,
        well_weight=1,
        name="QI",
    ):

        assert len(helicity) == 2
        assert (int(helicity[0]) == helicity[0]) and (int(helicity[1]) == helicity[1])
        self.grid = grid
        self.helicity = helicity
        self.L_QI = L_QI
        self.M_QI = M_QI
        self.N_QI = N_QI
        self.M_booz = M_booz
        self.N_booz = N_booz
        self.QI_l = QI_l
        self.QI_mn = QI_mn
        self.well_weight = well_weight
        super().__init__(
            eq=eq,
            target=target,
            bounds=bounds,
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
        if self.M_booz is None:
            self.M_booz = 2 * eq.M
        if self.N_booz is None:
            self.N_booz = 2 * eq.N
        if self.grid is None:
            self.grid = LinearGrid(
                M=2 * self.M_booz, N=2 * self.N_booz, NFP=eq.NFP, sym=False
            )
        if self.QI_l is None:
            data = eq.compute(["mirror ratio"], grid=self.grid)
            self._QI_l = np.linspace(
                min(data["min_tz |B|"]), max(data["max_tz |B|"]), num=self.L_QI
            )
        if self.QI_mn is None:
            self._QI_mn = np.zeros(((2 * self.M_QI + 1) * self.N_QI,))

        self._dim_f = self.grid.num_nodes
        self._data_keys = ["f_QI"]
        self._args = get_params(self._data_keys)
        self._args = list(
            map(
                lambda arg: arg.replace("QI_l", "QI_l {}".format(self.name)), self._args
            )
        )
        self._args = list(
            map(
                lambda arg: arg.replace("QI_mn", "QI_mn {}".format(self.name)),
                self._args,
            )
        )

        assert self.grid.sym is False
        assert self.grid.num_rho == 1

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._profiles = get_profiles(self._data_keys, eq=eq, grid=self.grid)
        self._transforms = get_transforms(
            self._data_keys,
            eq=eq,
            grid=self.grid,
            M_booz=self.M_booz,
            N_booz=self.N_booz,
            M_QI=self.M_QI,
            N_QI=self.N_QI,
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            self._normalization = jnp.mean(self.QI_l) / jnp.sqrt(self._dim_f)

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, *args, **kwargs):
        """Compute quasi-isodynamic errors.

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
        QI_l : ndarray
            Magnetic well shaping parameters.
            Values of |B| on a linearly spaced grid zeta_bar=[0,pi/2].
        QI_mn : ndarray
            Magnetic well shifting parameters.
            Fourier coefficients of zeta_Boozer(theta_Boozer,zeta_bar).

        Returns
        -------
        f : ndarray
            Quasi-isodynamic error at each node (T).

        """
        params = self._parse_args(*args, **kwargs)
        data = compute_fun(
            self._data_keys,
            params=params,
            transforms=self._transforms,
            profiles=self._profiles,
        )
        weights = (self.well_weight + 1) / 2 + (self.well_weight - 1) / 2 * jnp.cos(
            data["eta"]
        )
        return data["f_QI"] * weights

    def _set_dimensions(self, eq):
        """Set state vector component dimensions."""
        super()._set_dimensions(eq)
        self._dimensions["QI_l {}".format(self.name)] = self.L_QI
        self._dimensions["QI_mn {}".format(self.name)] = (2 * self.M_QI + 1) * self.N_QI

    def _parse_args(self, *args, **kwargs):
        params = super()._parse_args(*args, **kwargs)
        params["QI_l"] = params.pop("QI_l {}".format(self.name))
        params["QI_mn"] = params.pop("QI_mn {}".format(self.name))
        return params

    def xs(self, eq):
        """Return a tuple of args required by this objective from the Equilibrium eq."""
        args = []
        for arg in self.args:
            arg_name = arg.split(" ")[0]
            if arg_name not in ["QI_l", "QI_mn"]:
                args.append(getattr(eq, arg_name))
            else:
                args.append(getattr(self, arg_name))
        return tuple(args)

    def change_resolution(self, L_QI, M_QI, N_QI):
        """Change resolution of QI params. Returns QI_l and QI_mn at new resolution."""
        if L_QI != self.L_QI:
            old_z = np.linspace(0, np.pi / 2, num=self.L_QI)
            new_z = np.linspace(0, np.pi / 2, num=L_QI)
            B = np.sort(self.QI_l)
            self.QI_l = interp1d(new_z, old_z, B, method="monotonic-0")
            self.L_QI = L_QI
        if M_QI != self.M_QI or N_QI != self.N_QI:
            old_modes = DoubleFourierSeries(
                M=self.M_QI, N=self.N_QI, sym="cos(z)"
            ).modes
            new_modes = DoubleFourierSeries(M=M_QI, N=N_QI, sym="cos(z)").modes
            idx = np.nonzero((new_modes == old_modes[:, None]).all(-1))[1]
            QI_mn = np.zeros(((2 * M_QI + 1) * N_QI,))
            QI_mn[idx[: (2 * self.M_QI + 1) * self.N_QI]] = self.QI_mn
            self.QI_mn = QI_mn
            self.M_QI = M_QI
            self.N_QI = N_QI
        return self.QI_l, self.QI_mn

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
        if hasattr(self, "_helicity") and self._helicity != helicity:
            self._built = False
        self._helicity = helicity
        if hasattr(self, "_print_value_fmt"):
            units = "(T)"
            self._print_value_fmt = (
                "Quasi-symmetry ({},{}) Boozer error: ".format(
                    self.helicity[0], self.helicity[1]
                )
                + "{:10.3e} "
                + units
            )
        warnings.warn("Re-build objective after changing the helicity!")

    @property
    def QI_l(self):
        """Magnetic well shaping parameters."""
        return self._QI_l

    @QI_l.setter
    def QI_l(self, QI_l):
        self._QI_l = QI_l

    @property
    def QI_mn(self):
        """Magnetic well shifting parameters."""
        return self._QI_mn

    @QI_mn.setter
    def QI_mn(self, QI_mn):
        self._QI_mn = QI_mn
