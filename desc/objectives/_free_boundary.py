"""Objectives for solving free boundary equilibria."""

import warnings

import numpy as np
from scipy.constants import mu_0

from desc.backend import jnp
from desc.compute import compute as compute_fun
from desc.compute import get_params, get_profiles, get_transforms, xyz2rpz_vec
from desc.grid import LinearGrid
from desc.nestor import Nestor
from desc.objectives.objective_funs import _Objective
from desc.singularities import (
    DFTInterpolator,
    FFTInterpolator,
    virtual_casing_biot_savart,
)
from desc.utils import Timer, errorif, warnif

from .normalization import compute_scaling_factors


class VacuumBoundaryError(_Objective):
    """Target B*n = 0 and B^2_in - B^2_out = 0 on LCFS.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    ext_field : MagneticField
        External field produced by coils.
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
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate error at.
    field_grid : Grid, optional
        Grid used to discretize ext_field.
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = False
    _print_value_fmt = "Boundary Error: {:10.3e} "
    _units = "(T)"
    _coordinates = "rtz"

    def __init__(
        self,
        eq,
        ext_field,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        grid=None,
        field_grid=None,
        name="Vacuum boundary error",
    ):
        if target is None and bounds is None:
            target = 0
        self._grid = grid
        self._ext_field = ext_field
        self._field_grid = field_grid
        things = [eq]

        super().__init__(
            things=things,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
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
                rho=np.array([1.0]),
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=int(eq.NFP),
                sym=False,
            )
        else:
            grid = self._grid

        pres = np.max(np.abs(eq.compute("p")["p"]))
        curr = np.max(np.abs(eq.compute("current")["current"]))
        warnif(
            pres > 1e-8,
            UserWarning,
            f"Pressure is non-zero (max {pres} Pa), "
            + "VacuumBoundaryError will be incorrect.",
        )
        warnif(
            curr > 1e-8,
            UserWarning,
            f"Current is non-zero (max {curr} A), "
            + "VacuumBoundaryError will be incorrect.",
        )

        self._eq_data_keys = [
            "B",
            "R",
            "zeta",
            "Z",
            "n_rho",
            "|e_theta x e_zeta|",
        ]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        profiles = get_profiles(self._eq_data_keys, obj=eq, grid=grid)
        transforms = get_transforms(self._eq_data_keys, obj=eq, grid=grid)

        self._constants = {
            "transforms": transforms,
            "profiles": profiles,
            "ext_field": self._ext_field,
            "quad_weights": np.sqrt(np.tile(transforms["grid"].weights, 2)),
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._dim_f = 2 * grid.num_nodes

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["B"] * scales["R0"] * scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, eq_params, constants=None):
        """Compute boundary force error.

        Parameters
        ----------
        eq_params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Boundary force error (N).

        """
        if constants is None:
            constants = self.constants
        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._eq_data_keys,
            params=eq_params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
        )
        x = jnp.array([data["R"], data["zeta"], data["Z"]]).T
        Bext = constants["ext_field"].compute_magnetic_field(
            x, grid=self._field_grid, basis="rpz"
        )
        Bex_total = Bext
        Bin_total = data["B"]
        Bn = jnp.sum(Bex_total * data["n_rho"], axis=-1)

        bsq_out = jnp.sum(Bex_total * Bex_total, axis=-1)
        bsq_in = jnp.sum(Bin_total * Bin_total, axis=-1)

        g = data["|e_theta x e_zeta|"]
        Bn_err = Bn * g
        Bsq_err = (bsq_in - bsq_out) * g
        return jnp.concatenate([Bn_err, Bsq_err])

    def print_value(self, *args, **kwargs):
        """Print the value of the objective."""
        # this objective is really 3 residuals concatenated so its helpful to print
        # them individually
        f = self.compute_unscaled(*args, **kwargs)
        # try to do weighted mean if possible
        constants = kwargs.get("constants", self.constants)
        if constants is None:
            w = jnp.ones_like(f)
        else:
            w = constants["quad_weights"]

        abserr = jnp.all(self.target == 0)

        def _print(fmt, fmax, fmin, fmean, units):
            print(
                "Maximum " + ("absolute " if abserr else "") + fmt.format(fmax) + units
            )
            print(
                "Minimum " + ("absolute " if abserr else "") + fmt.format(fmin) + units
            )
            print(
                "Average " + ("absolute " if abserr else "") + fmt.format(fmean) + units
            )

            if self._normalize and units != "(dimensionless)":
                print(
                    "Maximum "
                    + ("absolute " if abserr else "")
                    + fmt.format(fmax / self.normalization)
                    + "(normalized)"
                )
                print(
                    "Minimum "
                    + ("absolute " if abserr else "")
                    + fmt.format(fmin / self.normalization)
                    + "(normalized)"
                )
                print(
                    "Average "
                    + ("absolute " if abserr else "")
                    + fmt.format(fmean / self.normalization)
                    + "(normalized)"
                )

        formats = [
            "Boundary normal field error: {:10.3e} ",
            "Boundary magnetic pressure error: {:10.3e} ",
        ]
        units = ["(T)", "(T^2)"]
        nn = f.size // 2
        for i, (fmt, unit) in enumerate(zip(formats, units)):
            fi = f[i * nn : (i + 1) * nn]
            # target == 0 probably indicates f is some sort of error metric,
            # mean abs makes more sense than mean
            fi = jnp.abs(fi) if abserr else fi
            wi = w[i * nn : (i + 1) * nn]
            fmax = jnp.max(fi)
            fmin = jnp.min(fi)
            fmean = jnp.mean(fi * wi) / jnp.mean(wi)
            _print(fmt, fmax, fmin, fmean, unit)


class BoundaryError(_Objective):
    """Target B*n = 0 and B^2_plasma + p - B^2_ext = 0 on LCFS.

    Uses virtual casing to find plasma component of B and penalizes
    (B_coil + B_plasma)*n and jump in total pressure.

    Optionally includes sheet current term for finite beta equilibria.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    ext_field : MagneticField
        External field produced by coils.
    sheet_current : FourierCurrentPotentialField
        Sheet current on the last closed flux surface. Not required for vacuum fields,
        but generally needed to correctly solve at finite beta/current. Geometry will
        be automatically constrained to be the same as the equilibrium LCFS.
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
    src_grid, eval_grid : Grid, optional
        Collocation grid containing the nodes to evaluate at for source terms and where
        to evaluate errors.
    field_grid : Grid, optional
        Grid used to discretize ext_field.
    loop : bool
        If True, evaluate integral using loops, as opposed to vmap. Slower, but uses
        less memory.
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = False
    _print_value_fmt = "Boundary Error: {:10.3e} "
    _units = "(T)"
    _coordinates = "rtz"

    def __init__(
        self,
        eq,
        ext_field,
        sheet_current=None,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        s=None,
        q=None,
        src_grid=None,
        eval_grid=None,
        field_grid=None,
        loop=True,
        name="Boundary error",
    ):
        if target is None and bounds is None:
            target = 0
        self._src_grid = src_grid
        self._eval_grid = eval_grid
        self._s = s
        self._q = q
        self._ext_field = ext_field
        self._field_grid = field_grid
        self._loop = loop
        self._sheet_current = None
        things = [eq]
        if sheet_current:
            things += [sheet_current]
            self._sheet_current = True

        super().__init__(
            things=things,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
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
        M_grid = eq.M_grid
        N_grid = eq.N_grid if eq.N > 0 else M_grid
        if self._src_grid is None:
            src_grid = LinearGrid(
                rho=np.array([1.0]),
                M=M_grid,
                N=N_grid,
                NFP=int(eq.NFP),
                sym=False,
            )
        else:
            src_grid = self._src_grid

        if self._eval_grid is None:
            eval_grid = LinearGrid(
                rho=np.array([1.0]),
                M=M_grid,
                N=N_grid,
                NFP=int(eq.NFP),
                sym=False,
            )
        else:
            eval_grid = self._eval_grid

        errorif(
            src_grid.sym,
            ValueError,
            "Source grids for singular integrals must be non-symmetric",
        )
        if self._s is None:
            k = min(src_grid.num_theta, src_grid.num_zeta)
            self._s = k // 2 + int(np.sqrt(k))
        if self._q is None:
            k = min(src_grid.num_theta, src_grid.num_zeta)
            self._q = k // 2 + int(np.sqrt(k))

        try:
            interpolator = FFTInterpolator(eval_grid, src_grid, self._s, self._q)
        except AssertionError as e:
            warnings.warn(
                "Could not built fft interpolator, switching to dft method which is"
                " much slower. Reason: " + str(e)
            )
            interpolator = DFTInterpolator(eval_grid, src_grid, self._s, self._q)

        edge_pres = np.max(np.abs(eq.compute("p", grid=eval_grid)["p"]))
        warnif(
            (edge_pres > 1e-6) and not self._sheet_current,
            UserWarning,
            f"Boundary pressure is nonzero (max {edge_pres} Pa), "
            + "a sheet current should be included.",
        )

        self._eq_data_keys = [
            "K_vc",
            "B",
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

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        src_profiles = get_profiles(self._eq_data_keys, obj=eq, grid=src_grid)
        src_transforms = get_transforms(self._eq_data_keys, obj=eq, grid=src_grid)
        eval_profiles = get_profiles(self._eq_data_keys, obj=eq, grid=eval_grid)
        eval_transforms = get_transforms(self._eq_data_keys, obj=eq, grid=eval_grid)

        neq = 3 if self._sheet_current else 2  # number of equations we're using

        self._constants = {
            "eval_transforms": eval_transforms,
            "eval_profiles": eval_profiles,
            "src_transforms": src_transforms,
            "src_profiles": src_profiles,
            "interpolator": interpolator,
            "ext_field": self._ext_field,
            "quad_weights": np.sqrt(np.tile(eval_transforms["grid"].weights, neq)),
        }

        if self._sheet_current:
            sheet_current = self.things[1]
            assert (
                (sheet_current.M == eq.surface.M)
                and (sheet_current.N == eq.surface.N)
                and (sheet_current.NFP == eq.surface.NFP)
                and (sheet_current.sym == eq.surface.sym)
            ), "sheet current must have same resolution as eq.surface"
            self._sheet_data_keys = ["K"]
            sheet_eval_transforms = get_transforms(
                self._sheet_data_keys, obj=sheet_current, grid=eval_grid
            )
            sheet_src_transforms = get_transforms(
                self._sheet_data_keys, obj=sheet_current, grid=src_grid
            )
            self._constants["sheet_eval_transforms"] = sheet_eval_transforms
            self._constants["sheet_src_transforms"] = sheet_src_transforms

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._dim_f = neq * eval_grid.num_nodes

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["B"] * scales["R0"] * scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, eq_params, sheet_params=None, constants=None):
        """Compute boundary force error.

        Parameters
        ----------
        eq_params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        sheet_params : dict
            Dictionary of sheet current degrees of freedom.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Boundary force error (N).

        """
        if constants is None:
            constants = self.constants
        src_data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._eq_data_keys,
            params=eq_params,
            transforms=constants["src_transforms"],
            profiles=constants["src_profiles"],
        )
        eval_data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._eq_data_keys,
            params=eq_params,
            transforms=constants["eval_transforms"],
            profiles=constants["eval_profiles"],
        )
        if self._sheet_current:
            assert sheet_params is not None
            # enforce that they're the same surface
            sheet_params["R_lmn"] = eq_params["Rb_lmn"]
            sheet_params["Z_lmn"] = eq_params["Zb_lmn"]
            sheet_src_data = compute_fun(
                "desc.magnetic_fields.FourierCurrentPotentialField",
                self._sheet_data_keys,
                params=sheet_params,
                transforms=constants["sheet_src_transforms"],
                profiles={},
            )
            sheet_eval_data = compute_fun(
                "desc.magnetic_fields.FourierCurrentPotentialField",
                self._sheet_data_keys,
                params=sheet_params,
                transforms=constants["sheet_eval_transforms"],
                profiles={},
            )
            src_data["K_vc"] += sheet_src_data["K"]

        # this is in cartesian
        Bplasma = -virtual_casing_biot_savart(
            eval_data,
            constants["eval_transforms"]["grid"],
            src_data,
            constants["src_transforms"]["grid"],
            constants["interpolator"],
            loop=self._loop,
        )
        # need extra factor of B/2 bc we're evaluating on plasma surface
        Bplasma = xyz2rpz_vec(Bplasma, phi=eval_data["zeta"]) + eval_data["B"] / 2
        x = jnp.array([eval_data["R"], eval_data["zeta"], eval_data["Z"]]).T
        Bext = constants["ext_field"].compute_magnetic_field(
            x, grid=self._field_grid, basis="rpz"
        )
        Bex_total = Bext + Bplasma
        Bin_total = eval_data["B"]
        Bn = jnp.sum(Bex_total * eval_data["n_rho"], axis=-1)

        bsq_out = jnp.sum(Bex_total * Bex_total, axis=-1)
        bsq_in = jnp.sum(Bin_total * Bin_total, axis=-1)

        g = eval_data["|e_theta x e_zeta|"]
        Bn_err = Bn * g
        Bsq_err = (bsq_in + eval_data["p"] * (2 * mu_0) - bsq_out) * g
        Bjump = Bex_total - Bin_total
        if self._sheet_current:
            Kerr = mu_0 * sheet_eval_data["K"] - jnp.cross(eval_data["n_rho"], Bjump)
            Kerr = jnp.linalg.norm(Kerr, axis=-1) * g
            return jnp.concatenate([Bn_err, Bsq_err, Kerr])
        else:
            return jnp.concatenate([Bn_err, Bsq_err])

    def print_value(self, *args, **kwargs):
        """Print the value of the objective."""
        # this objective is really 3 residuals concatenated so its helpful to print
        # them individually
        f = self.compute_unscaled(*args, **kwargs)
        # try to do weighted mean if possible
        constants = kwargs.get("constants", self.constants)
        if constants is None:
            w = jnp.ones_like(f)
        else:
            w = constants["quad_weights"]

        abserr = jnp.all(self.target == 0)

        def _print(fmt, fmax, fmin, fmean, units):
            print(
                "Maximum " + ("absolute " if abserr else "") + fmt.format(fmax) + units
            )
            print(
                "Minimum " + ("absolute " if abserr else "") + fmt.format(fmin) + units
            )
            print(
                "Average " + ("absolute " if abserr else "") + fmt.format(fmean) + units
            )

            if self._normalize and units != "(dimensionless)":
                print(
                    "Maximum "
                    + ("absolute " if abserr else "")
                    + fmt.format(fmax / self.normalization)
                    + "(normalized)"
                )
                print(
                    "Minimum "
                    + ("absolute " if abserr else "")
                    + fmt.format(fmin / self.normalization)
                    + "(normalized)"
                )
                print(
                    "Average "
                    + ("absolute " if abserr else "")
                    + fmt.format(fmean / self.normalization)
                    + "(normalized)"
                )

        formats = [
            "Boundary normal field error: {:10.3e} ",
            "Boundary magnetic pressure error: {:10.3e} ",
            "Boundary field jump error: {:10.3e} ",
        ]
        units = ["(T)", "(T^2)", "(T)"]
        if self._sheet_current:
            nn = f.size // 3
        else:
            nn = f.size // 2
            formats = formats[:-1]
            units = units[:-1]
        for i, (fmt, unit) in enumerate(zip(formats, units)):
            fi = f[i * nn : (i + 1) * nn]
            # target == 0 probably indicates f is some sort of error metric,
            # mean abs makes more sense than mean
            fi = jnp.abs(fi) if abserr else fi
            wi = w[i * nn : (i + 1) * nn]
            fmax = jnp.max(fi)
            fmin = jnp.min(fi)
            fmean = jnp.mean(fi * wi) / jnp.mean(wi)
            _print(fmt, fmax, fmin, fmean, unit)


class BoundaryErrorNESTOR(_Objective):
    """Pressure balance across LCFS.

    Uses NESTOR algorithm to compute B_vac such that (B_vac + B_coil)*n=0,
    then calculates the pressure mismatch across the boundary:

        1/2mu0*(B_vac + B_coil)^2 - 1/2mu0*B_plasma^2 - p


    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    ext_field : MagneticField
        External field produced by coils.
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
    field_grid : Grid, optional
        Grid used to discretize ext_field.
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = False
    _print_value_fmt = "Boundary magnetic pressure error: {:10.3e} "
    _units = "(T^2)"
    _coordinates = "rtz"

    def __init__(
        self,
        eq,
        ext_field,
        target=None,
        bounds=None,
        weight=1,
        mf=None,
        nf=None,
        ntheta=None,
        nzeta=None,
        field_grid=None,
        normalize=True,
        normalize_target=True,
        name="NESTOR Boundary",
    ):
        if target is None and bounds is None:
            target = 0
        self.mf = mf
        self.nf = nf
        self.ntheta = ntheta
        self.nzeta = nzeta
        self.ext_field = ext_field
        self.field_grid = field_grid
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
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
        self.mf = eq.M + 1 if self.mf is None else self.mf
        self.nf = eq.N if self.nf is None else self.nf
        self.ntheta = 4 * eq.M + 1 if self.ntheta is None else self.ntheta
        self.nzeta = 4 * eq.N + 1 if self.nzeta is None else self.nzeta

        nest = Nestor(
            eq,
            self.ext_field,
            self.mf,
            self.nf,
            self.ntheta,
            self.nzeta,
            self.field_grid,
        )
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

        profiles = get_profiles(self._data_keys, obj=eq, grid=self.grid)
        transforms = get_transforms(self._data_keys, obj=eq, grid=self.grid)

        self._constants = {
            "profiles": profiles,
            "transforms": transforms,
            "ext_field": self.ext_field,
            "nestor": nest,
            "quad_weights": np.sqrt(transforms["grid"].weights),
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._dim_f = self.grid.num_nodes

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["B"] * scales["R0"] * scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute boundary magnetic pressure error.

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
            Boundary magnetic pressure error (T^2).

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

        ctor = jnp.mean(data["current"])
        out = constants["nestor"].compute(params["R_lmn"], params["Z_lmn"], ctor)
        grid = constants["nestor"]._Rb_transform.grid
        bsq = out[1]["|B|^2"].reshape((grid.num_zeta, grid.num_theta)).T.flatten()
        bv = bsq

        bp = data["|B|^2"]
        g = data["|e_theta x e_zeta|"]
        return (bv - bp - data["p"] * (2 * mu_0)) * g


class BConsistencyError(_Objective):
    """Target B_{nxB}=B_{eq} on target surfaces.

    Uses virtual casing to find K such that Bn=0 on a
    given surface, then penalizes the difference between
    the B field computed from biot savart from that K and
    the B from the equilibrium on that surface.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    ext_field : MagneticField
        External field produced by coils.
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
    src_grid, eval_grid : Grid, optional
        Collocation grid containing the nodes to evaluate at for source terms and where
        to evaluate errors.
    field_grid : Grid, optional
        Grid used to discretize ext_field.
    loop : bool
        If True, evaluate integral using loops, as opposed to vmap. Slower, but uses
        less memory.
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = False
    _print_value_fmt = "B Consistency Error: {:10.3e} "
    _units = "(T)"
    _coordinates = "rtz"

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        s=None,
        q=None,
        src_grid=None,
        eval_grid=None,
        field_grid=None,
        loop=True,
        name="B consistency error",
    ):
        if target is None and bounds is None:
            target = 0
        self._src_grid = src_grid
        self._eval_grid = eval_grid
        self._s = s
        self._q = q
        self._field_grid = field_grid
        self._loop = loop
        things = eq

        super().__init__(
            things=things,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
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
        M_grid = eq.M_grid
        N_grid = eq.N_grid if eq.N > 0 else M_grid
        if self._src_grid is None:
            src_grid = LinearGrid(
                rho=np.array([1.0]),
                M=M_grid,
                N=N_grid,
                NFP=int(eq.NFP),
                sym=False,
            )
        else:
            src_grid = self._src_grid

        if self._eval_grid is None:
            eval_grid = LinearGrid(
                rho=np.array([1.0]),
                M=M_grid,
                N=N_grid,
                NFP=int(eq.NFP),
                sym=False,
            )
        else:
            eval_grid = self._eval_grid

        errorif(
            src_grid.sym,
            ValueError,
            "Source grids for singular integrals must be non-symmetric",
        )
        if self._s is None:
            k = min(src_grid.num_theta, src_grid.num_zeta)
            self._s = k // 2 + int(np.sqrt(k))
        if self._q is None:
            k = min(src_grid.num_theta, src_grid.num_zeta)
            self._q = k // 2 + int(np.sqrt(k))

        try:
            interpolator = FFTInterpolator(eval_grid, src_grid, self._s, self._q)
        except AssertionError as e:
            warnings.warn(
                "Could not built fft interpolator, switching to dft method which is"
                " much slower. Reason: " + str(e)
            )
            interpolator = DFTInterpolator(eval_grid, src_grid, self._s, self._q)

        self._eq_data_keys = [
            "K_vc",
            "B",
            "R",
            "zeta",
            "Z",
            "e^rho",
            "n_rho",
            "|e_theta x e_zeta|",
        ]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        src_profiles = get_profiles(self._eq_data_keys, obj=eq, grid=src_grid)
        src_transforms = get_transforms(self._eq_data_keys, obj=eq, grid=src_grid)
        eval_profiles = get_profiles(self._eq_data_keys, obj=eq, grid=eval_grid)
        eval_transforms = get_transforms(self._eq_data_keys, obj=eq, grid=eval_grid)

        self._constants = {
            "eval_transforms": eval_transforms,
            "eval_profiles": eval_profiles,
            "src_transforms": src_transforms,
            "src_profiles": src_profiles,
            "interpolator": interpolator,
            "quad_weights": np.sqrt(np.tile(eval_transforms["grid"].weights, 3)),
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")
        # the 3 components of B
        self._dim_f = 3 * eval_grid.num_nodes

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["B"] * scales["R0"] * scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, eq_params, constants=None):
        """Compute B consistency error.

        Parameters
        ----------
        eq_params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Magnetic Field vector difference (T).

        """
        if constants is None:
            constants = self.constants
        src_data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._eq_data_keys,
            params=eq_params,
            transforms=constants["src_transforms"],
            profiles=constants["src_profiles"],
        )
        eval_data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._eq_data_keys,
            params=eq_params,
            transforms=constants["eval_transforms"],
            profiles=constants["eval_profiles"],
        )

        # this is in cartesian
        Bplasma = -virtual_casing_biot_savart(
            eval_data,
            constants["eval_transforms"]["grid"],
            src_data,
            constants["src_transforms"]["grid"],
            constants["interpolator"],
            loop=self._loop,
        )

        # need extra factor of B/2 bc we're evaluating on the surface itself
        B_nxB = xyz2rpz_vec(Bplasma, phi=eval_data["zeta"]) + eval_data["B"] / 2

        B_eq = eval_data["B"]

        g = eval_data["|e_theta x e_zeta|"]
        B_err = (g * (B_eq - B_nxB).T).T.flatten(order="F")

        return B_err

    def print_value(self, *args, **kwargs):
        """Print the value of the objective."""
        # this objective is really 3 residuals concatenated so its helpful to print
        # them individually
        f = self.compute_unscaled(*args, **kwargs)
        # try to do weighted mean if possible
        constants = kwargs.get("constants", self.constants)
        if constants is None:
            w = jnp.ones_like(f)
        else:
            w = constants["quad_weights"]

        abserr = jnp.all(self.target == 0)

        def _print(fmt, fmax, fmin, fmean, units):
            print(
                "Maximum " + ("absolute " if abserr else "") + fmt.format(fmax) + units
            )
            print(
                "Minimum " + ("absolute " if abserr else "") + fmt.format(fmin) + units
            )
            print(
                "Average " + ("absolute " if abserr else "") + fmt.format(fmean) + units
            )

            if self._normalize and units != "(dimensionless)":
                print(
                    "Maximum "
                    + ("absolute " if abserr else "")
                    + fmt.format(fmax / self.normalization)
                    + "(normalized)"
                )
                print(
                    "Minimum "
                    + ("absolute " if abserr else "")
                    + fmt.format(fmin / self.normalization)
                    + "(normalized)"
                )
                print(
                    "Average "
                    + ("absolute " if abserr else "")
                    + fmt.format(fmean / self.normalization)
                    + "(normalized)"
                )

        formats = [
            "B_R error: {:10.3e} ",
            "B_phi error: {:10.3e} ",
            "B_Z error: {:10.3e} ",
        ]
        units = ["(T)", "(T)", "(T)"]
        nn = f.size // 3

        for i, (fmt, unit) in enumerate(zip(formats, units)):
            fi = f[i * nn : (i + 1) * nn]
            # target == 0 probably indicates f is some sort of error metric,
            # mean abs makes more sense than mean
            fi = jnp.abs(fi) if abserr else fi
            wi = w[i * nn : (i + 1) * nn]
            fmax = jnp.max(fi)
            fmin = jnp.min(fi)
            fmean = jnp.mean(fi * wi) / jnp.mean(wi)
            _print(fmt, fmax, fmin, fmean, unit)
