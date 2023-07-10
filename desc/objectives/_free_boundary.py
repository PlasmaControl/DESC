"""Objectives for solving free boundary equilibria."""

import warnings

import numpy as np
from scipy.constants import mu_0

from desc.backend import jnp
from desc.compute import compute as compute_fun
from desc.compute import get_params, get_profiles, get_transforms
from desc.geometry.utils import xyz2rpz_vec
from desc.grid import LinearGrid, QuadratureGrid
from desc.nestor import Nestor
from desc.objectives.objective_funs import _Objective
from desc.singularities import DFTInterpolator, FFTInterpolator, singular_integral
from desc.utils import Timer

from .normalization import compute_scaling_factors


class BoundaryErrorBIEST(_Objective):
    """Target B*n = 0 and B^2_plasma + p - B^2_ext = 0 on LCFS.

    Uses virtual casing to find plasma component of B and penalizes
    (B_coil + B_plasma)*n and jump in total pressure

    Parameters
    ----------
    ext_field : MagneticField
        External field produced by coils.
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
    s, q : integer
        Hyperparameters for singular integration scheme, s is roughly equal to the size
        of the local singular grid with respect to the global grid, q is the order of
        integration on the local grid
    src_grid, eval_grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at for source terms and where
        to evaluate errors.
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = False
    _print_value_fmt = "Boundary Error: {:10.3e} "
    _units = "(T)"

    def __init__(
        self,
        ext_field,
        eq=None,
        target=0,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        s=None,
        q=None,
        src_grid=None,
        eval_grid=None,
        name="Boundary error BIEST",
    ):
        self._src_grid = src_grid
        self._eval_grid = eval_grid
        self._s = s
        self._q = q
        self._ext_field = ext_field
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
        if self._src_grid is None:
            src_grid = LinearGrid(
                rho=np.array([1.0]),
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=int(eq.NFP),
                sym=False,
            )
        else:
            src_grid = self._src_grid

        if self._eval_grid is None:
            eval_grid = LinearGrid(
                rho=np.array([1.0]),
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=int(eq.NFP),
                sym=False,
            )
        else:
            src_grid = self._src_grid
        if self._s is None:
            k = min(src_grid.num_theta, src_grid.num_zeta)
            self._s = k // 2 + int(np.sqrt(k))
        if self._q is None:
            k = min(src_grid.num_theta, src_grid.num_zeta)
            self._q = k // 2 + int(np.sqrt(k))

        try:
            self._interpolator = FFTInterpolator(eval_grid, src_grid, self._s, self._q)
        except AssertionError as e:
            warnings.warn(
                "Could not built fft interpolator, switching to dft method which is"
                " much slower. Reason: " + str(e)
            )
            self._interpolator = DFTInterpolator(eval_grid, src_grid, self._s, self._q)

        self._data_keys = [
            "K_vc",
            "|B|^2",
            "B",
            "R",
            "zeta",
            "Z",
            "e^rho",
            "n_rho",
            "|e_theta x e_zeta|",
            "p",
        ]
        self._args = get_params(
            self._data_keys,
            obj="desc.equilibrium.equilibrium.Equilibrium",
            has_axis=False,
        )

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._src_profiles = get_profiles(self._data_keys, obj=eq, grid=src_grid)
        self._src_transforms = get_transforms(self._data_keys, obj=eq, grid=src_grid)
        self._eval_profiles = get_profiles(self._data_keys, obj=eq, grid=eval_grid)
        self._eval_transforms = get_transforms(self._data_keys, obj=eq, grid=eval_grid)

        self._constants = {
            "eval_transforms": self._eval_transforms,
            "eval_profiles": self._eval_profiles,
            "src_transforms": self._src_transforms,
            "src_profiles": self._src_profiles,
            "interpolator": self._interpolator,
            "ext_field": self._ext_field,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._dim_f = eval_grid.num_nodes

        if self._normalize:
            scales = compute_scaling_factors(eq)
            # local quantity, want to divide by number of nodes
            self._normalization = scales["B"] / jnp.sqrt(self._dim_f)

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, *args, **kwargs):
        """Compute boundary force error.

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
        f : ndarray
            Boundary force error (N).

        """
        params, constants = self._parse_args(*args, **kwargs)
        if constants is None:
            constants = self.constants
        src_data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=constants["src_transforms"],
            profiles=constants["src_profiles"],
        )
        eval_data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=constants["eval_transforms"],
            profiles=constants["eval_profiles"],
        )
        # this is in cartesian
        Bplasma = -singular_integral(
            eval_data,
            constants["eval_transforms"]["grid"],
            src_data,
            constants["src_transforms"]["grid"],
            self._s,
            self._q,
            "biot_savart",
            constants["interpolator"],
        )
        # need extra factor of B/2 bc we're evaluating on plasma surface
        Bplasma = xyz2rpz_vec(Bplasma, phi=eval_data["zeta"]) + eval_data["B"] / 2
        x = jnp.array([eval_data["R"], eval_data["zeta"], eval_data["Z"]]).T
        Bext = constants["ext_field"].compute_magnetic_field(x)
        Bex_total = Bext + Bplasma
        Bn = jnp.sum(Bex_total * eval_data["n_rho"], axis=-1)

        bsq_out = jnp.sum(Bex_total * Bex_total, axis=-1)
        bsq_in = eval_data["|B|^2"]
        w = constants["eval_transforms"]["grid"].weights
        g = eval_data["|e_theta x e_zeta|"]
        Bn_err = Bn * w * g
        Bsq_err = (bsq_in + eval_data["p"] * (2 * mu_0) - bsq_out) * w * g
        return jnp.concatenate([Bn_err, Bsq_err])


class QuadraticFlux(_Objective):
    """Target B*n = 0 on LCFS.

    Uses virtual casing to find plasma component of B and penalizes
    (B_coil + B_plasma)*n

    Parameters
    ----------
    ext_field : MagneticField
        External field produced by coils.
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
    s, q : integer
        Hyperparameters for singular integration scheme, s is roughly equal to the size
        of the local singular grid with respect to the global grid, q is the order of
        integration on the local grid
    src_grid, eval_grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at for source terms and where
        to evaluate errors.
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = False
    _print_value_fmt = "Boundary normal field Error: {:10.3e} "
    _units = "(T)"

    def __init__(
        self,
        ext_field,
        eq=None,
        target=0,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        s=None,
        q=None,
        src_grid=None,
        eval_grid=None,
        name="Quadratic flux",
    ):
        self._src_grid = src_grid
        self._eval_grid = eval_grid
        self._s = s
        self._q = q
        self._ext_field = ext_field
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
        if self._src_grid is None:
            src_grid = LinearGrid(
                rho=np.array([1.0]),
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=int(eq.NFP),
                sym=False,
            )
        else:
            src_grid = self._src_grid

        if self._eval_grid is None:
            eval_grid = LinearGrid(
                rho=np.array([1.0]),
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=int(eq.NFP),
                sym=False,
            )
        else:
            src_grid = self._src_grid
        if self._s is None:
            k = min(src_grid.num_theta, src_grid.num_zeta)
            self._s = k // 2 + int(np.sqrt(k))
        if self._q is None:
            k = min(src_grid.num_theta, src_grid.num_zeta)
            self._q = k // 2 + int(np.sqrt(k))

        try:
            self._interpolator = FFTInterpolator(eval_grid, src_grid, self._s, self._q)
        except AssertionError as e:
            warnings.warn(
                "Could not built fft interpolator, switching to dft method which is"
                " much slower. Reason: " + str(e)
            )
            self._interpolator = DFTInterpolator(eval_grid, src_grid, self._s, self._q)

        self._data_keys = [
            "K_vc",
            "R",
            "zeta",
            "Z",
            "e^rho",
            "n_rho",
            "|e_theta x e_zeta|",
        ]
        self._args = get_params(
            self._data_keys,
            obj="desc.equilibrium.equilibrium.Equilibrium",
            has_axis=False,
        )

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._src_profiles = get_profiles(self._data_keys, obj=eq, grid=src_grid)
        self._src_transforms = get_transforms(self._data_keys, obj=eq, grid=src_grid)
        self._eval_profiles = get_profiles(self._data_keys, obj=eq, grid=eval_grid)
        self._eval_transforms = get_transforms(self._data_keys, obj=eq, grid=eval_grid)

        self._constants = {
            "eval_transforms": self._eval_transforms,
            "eval_profiles": self._eval_profiles,
            "src_transforms": self._src_transforms,
            "src_profiles": self._src_profiles,
            "interpolator": self._interpolator,
            "ext_field": self._ext_field,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._dim_f = eval_grid.num_nodes

        if self._normalize:
            scales = compute_scaling_factors(eq)
            # local quantity, want to divide by number of nodes
            self._normalization = scales["B"] / jnp.sqrt(self._dim_f)

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, *args, **kwargs):
        """Compute boundary force error.

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
        f : ndarray
            Boundary force error (N).

        """
        params, constants = self._parse_args(*args, **kwargs)
        if constants is None:
            constants = self.constants
        src_data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=constants["src_transforms"],
            profiles=constants["src_profiles"],
        )
        eval_data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=constants["eval_transforms"],
            profiles=constants["eval_profiles"],
        )
        # this is in cartesian
        Bplasma = -singular_integral(
            eval_data,
            constants["eval_transforms"]["grid"],
            src_data,
            constants["src_transforms"]["grid"],
            self._s,
            self._q,
            "biot_savart",
            constants["interpolator"],
        )
        # don't need extra B/2 since we only care about normal component
        Bplasma = xyz2rpz_vec(Bplasma, phi=eval_data["zeta"])
        x = jnp.array([eval_data["R"], eval_data["zeta"], eval_data["Z"]]).T
        Bext = constants["ext_field"].compute_magnetic_field(x)
        return jnp.sum((Bext + Bplasma) * eval_data["n_rho"], axis=-1)


class ToroidalFlux(_Objective):
    """Target toroidal flux through the torus.

    Parameters
    ----------
    ext_field : MagneticField
        External field produced by coils.
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
    _print_value_fmt = "Toroidal flux Error: {:10.3e} "
    _units = "(Wb)"

    def __init__(
        self,
        ext_field,
        eq=None,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        grid=None,
        name="Toroidal flux",
    ):
        self._grid = grid
        self._ext_field = ext_field
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
            grid = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=0)
        else:
            grid = self._grid

        self._data_keys = ["R", "zeta", "Z", "e^zeta", "|e_rho x e_theta|"]
        self._args = get_params(
            self._data_keys,
            obj="desc.equilibrium.equilibrium.Equilibrium",
            has_axis=False,
        )

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
        self._transforms = get_transforms(self._data_keys, obj=eq, grid=grid)

        self._constants = {
            "profiles": self._profiles,
            "transforms": self._transforms,
            "ext_field": self._ext_field,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._dim_f = 1
        if self.target is None:
            self.target = eq.Psi

        if self._normalize:
            scales = compute_scaling_factors(eq)
            # local quantity, want to divide by number of nodes
            self._normalization = scales["B"] * scales["A"] / jnp.sqrt(self._dim_f)

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, *args, **kwargs):
        """Compute boundary force error.

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
        f : ndarray
            Boundary force error (N).

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
        x = jnp.array([data["R"], data["zeta"], data["Z"]]).T
        Bext = constants["ext_field"].compute_magnetic_field(x, basis="rpz")
        n = data["e^zeta"] / jnp.linalg.norm(data["e^zeta"], axis=-1)[:, None]
        Bn = jnp.sum(Bext * n, axis=-1)
        return jnp.sum(
            Bn
            * data["|e_rho x e_theta|"]
            * constants["transforms"]["grid"].spacing[:, :2].prod(axis=-1)
        )


class BoundaryErrorNESTOR(_Objective):
    """Pressure balance across LCFS.

    Uses NESTOR algorithm to compute B_vac such that (B_vac + B_coil)*n=0,
    then calculates the pressure mismatch across the boundary:

        1/2mu0*(B_vac + B_coil)^2 - 1/2mu0*B_plasma^2 - p


    Parameters
    ----------
    ext_field : MagneticField
        External field produced by coils.
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
    mf, nf : integer
        maximum poloidal and toroidal mode numbers to use for NESTOR scalar potential.
    ntheta, nzeta : int
        number of grid points in poloidal, toroidal directions to use in NESTOR.
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
    _print_value_fmt = "Boundary Force Error: {:10.3e} "
    _units = "(N)"

    def __init__(
        self,
        ext_field,
        eq=None,
        target=0,
        bounds=None,
        weight=1,
        mf=None,
        nf=None,
        ntheta=None,
        nzeta=None,
        normalize=True,
        normalize_target=True,
        name="NESTOR Boundary",
    ):
        self.mf = mf
        self.nf = nf
        self.ntheta = ntheta
        self.nzeta = nzeta
        self.ext_field = ext_field
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
        self.mf = eq.M + 1 if self.mf is None else self.mf
        self.nf = eq.N if self.nf is None else self.nf
        self.ntheta = 4 * eq.M + 1 if self.ntheta is None else self.ntheta
        self.nzeta = 4 * eq.N + 1 if self.nzeta is None else self.nzeta

        eq._sym = False
        self.nest = Nestor(
            eq, self.ext_field, self.mf, self.nf, self.ntheta, self.nzeta
        )
        eq._sym = True
        self.grid = LinearGrid(rho=1, theta=self.ntheta, zeta=self.nzeta, NFP=eq.NFP)
        self._data_keys = ["current", "|B|^2", "p", "|e_theta x e_zeta|"]
        self._args = get_params(
            self._data_keys,
            obj="desc.equilibrium.equilibrium.Equilibrium",
            has_axis=False,
        )

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._profiles = get_profiles(self._data_keys, obj=eq, grid=self.grid)
        self._transforms = get_transforms(self._data_keys, obj=eq, grid=self.grid)

        self._constants = {
            "profiles": self._profiles,
            "transforms": self._transforms,
            "ext_field": self.ext_field,
            "nestor": self.nest,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._dim_f = self.grid.num_nodes

        if self._normalize:
            scales = compute_scaling_factors(eq)
            # local quantity, want to divide by number of nodes
            self._normalization = scales["p"] / jnp.sqrt(self._dim_f)

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, *args, **kwargs):
        """Compute boundary force error.

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
        f : ndarray
            Boundary force error (N).

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

        ctor = jnp.mean(data["current"])
        out = constants["nestor"].compute(params["R_lmn"], params["Z_lmn"], ctor)
        grid = constants["nestor"]._Rb_transform.grid
        bsq = out[1]["|B|^2"].reshape((grid.num_zeta, grid.num_theta)).T.flatten()
        bv = bsq / (2 * mu_0)

        bp = data["|B|^2"] / (2 * mu_0)
        w = self.grid.weights
        g = data["|e_theta x e_zeta|"]
        return (bv - bp - data["p"]) * w * g
