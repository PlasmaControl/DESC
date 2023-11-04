"""Objectives for solving free boundary equilibria."""

import warnings

import numpy as np
from scipy.constants import mu_0

from desc.backend import jnp
from desc.compute import compute as compute_fun
from desc.compute import get_params, get_profiles, get_transforms, xyz2rpz_vec
from desc.grid import LinearGrid, QuadratureGrid
from desc.nestor import Nestor
from desc.objectives.objective_funs import _Objective
from desc.singularities import (
    DFTInterpolator,
    FFTInterpolator,
    virtual_casing_biot_savart,
)
from desc.utils import Timer

from .normalization import compute_scaling_factors


class BoundaryErrorBIESTSC(_Objective):
    """Target B*n = 0 and B^2_plasma + p - B^2_ext = 0 on LCFS.

    Uses virtual casing to find plasma component of B and penalizes
    (B_coil + B_plasma)*n and jump in total pressure.

    Includes sheet current term.

    Parameters
    ----------
    ext_field : MagneticField
        External field produced by coils.
    eq : Equilibrium
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
    src_grid, eval_grid : Grid, optional
        Collocation grid containing the nodes to evaluate at for source terms and where
        to evaluate errors.
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
        ext_field,
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
        name="Boundary error BIEST (SC)",
    ):
        if target is None and bounds is None:
            target = 0
        self._src_grid = src_grid
        self._eval_grid = eval_grid
        self._s = s
        self._q = q
        self._ext_field = ext_field
        self._field_grid = field_grid
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
            interpolator = FFTInterpolator(eval_grid, src_grid, self._s, self._q)
        except AssertionError as e:
            warnings.warn(
                "Could not built fft interpolator, switching to dft method which is"
                " much slower. Reason: " + str(e)
            )
            interpolator = DFTInterpolator(eval_grid, src_grid, self._s, self._q)

        self._data_keys = [
            "K_vc",
            "K_sc",
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
        self._args = get_params(
            self._data_keys,
            obj="desc.equilibrium.equilibrium.Equilibrium",
            has_axis=False,
        )

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        src_profiles = get_profiles(self._data_keys, obj=eq, grid=src_grid)
        src_transforms = get_transforms(self._data_keys, obj=eq, grid=src_grid)
        eval_profiles = get_profiles(self._data_keys, obj=eq, grid=eval_grid)
        eval_transforms = get_transforms(self._data_keys, obj=eq, grid=eval_grid)

        self._constants = {
            "eval_transforms": eval_transforms,
            "eval_profiles": eval_profiles,
            "src_transforms": src_transforms,
            "src_profiles": src_profiles,
            "interpolator": interpolator,
            "ext_field": self._ext_field,
            "quad_weights": np.tile(eval_transforms["grid"].weights, 3),
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._dim_f = 3 * eval_grid.num_nodes

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["B"] * scales["R0"] * scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute boundary force error.

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
            Boundary force error (N).

        """
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
        src_data["K_vc"] += src_data["K_sc"]
        # this is in cartesian
        Bplasma = -virtual_casing_biot_savart(
            eval_data,
            constants["eval_transforms"]["grid"],
            src_data,
            constants["src_transforms"]["grid"],
            constants["interpolator"],
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
        Kerr = eval_data["K_sc"] - jnp.cross(eval_data["n_rho"], Bjump)
        Kerr = jnp.sum(Kerr * Kerr, axis=-1) * g
        return jnp.concatenate([Bn_err, Bsq_err, Kerr])


class BoundaryErrorBIEST(_Objective):
    """Target B*n = 0 and B^2_plasma + p - B^2_ext = 0 on LCFS.

    Uses virtual casing to find plasma component of B and penalizes
    (B_coil + B_plasma)*n and jump in total pressure

    Parameters
    ----------
    ext_field : MagneticField
        External field produced by coils.
    eq : Equilibrium
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
    src_grid, eval_grid : Grid, optional
        Collocation grid containing the nodes to evaluate at for source terms and where
        to evaluate errors.
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
        ext_field,
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
        name="Boundary error BIEST",
    ):
        if target is None and bounds is None:
            target = 0
        self._src_grid = src_grid
        self._eval_grid = eval_grid
        self._s = s
        self._q = q
        self._ext_field = ext_field
        self._field_grid = field_grid
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
            interpolator = FFTInterpolator(eval_grid, src_grid, self._s, self._q)
        except AssertionError as e:
            warnings.warn(
                "Could not built fft interpolator, switching to dft method which is"
                " much slower. Reason: " + str(e)
            )
            interpolator = DFTInterpolator(eval_grid, src_grid, self._s, self._q)

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

        src_profiles = get_profiles(self._data_keys, obj=eq, grid=src_grid)
        src_transforms = get_transforms(self._data_keys, obj=eq, grid=src_grid)
        eval_profiles = get_profiles(self._data_keys, obj=eq, grid=eval_grid)
        eval_transforms = get_transforms(self._data_keys, obj=eq, grid=eval_grid)

        self._constants = {
            "eval_transforms": eval_transforms,
            "eval_profiles": eval_profiles,
            "src_transforms": src_transforms,
            "src_profiles": src_profiles,
            "interpolator": interpolator,
            "ext_field": self._ext_field,
            "quad_weights": np.tile(eval_transforms["grid"].weights, 2),
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._dim_f = 2 * eval_grid.num_nodes

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["B"] * scales["R0"] * scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute boundary force error.

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
            Boundary force error (N).

        """
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
        Bplasma = -virtual_casing_biot_savart(
            eval_data,
            constants["eval_transforms"]["grid"],
            src_data,
            constants["src_transforms"]["grid"],
            constants["interpolator"],
        )
        # need extra factor of B/2 bc we're evaluating on plasma surface
        Bplasma = xyz2rpz_vec(Bplasma, phi=eval_data["zeta"]) + eval_data["B"] / 2
        x = jnp.array([eval_data["R"], eval_data["zeta"], eval_data["Z"]]).T
        Bext = constants["ext_field"].compute_magnetic_field(
            x, grid=self._field_grid, basis="rpz"
        )
        Bex_total = Bext + Bplasma
        Bn = jnp.sum(Bex_total * eval_data["n_rho"], axis=-1)

        bsq_out = jnp.sum(Bex_total * Bex_total, axis=-1)
        bsq_in = eval_data["|B|^2"]
        g = eval_data["|e_theta x e_zeta|"]
        Bn_err = Bn * g
        Bsq_err = (bsq_in + eval_data["p"] * (2 * mu_0) - bsq_out) * g
        return jnp.concatenate([Bn_err, Bsq_err])


class QuadraticFlux(_Objective):
    """Target B*n = 0 on LCFS.

    Uses virtual casing to find plasma component of B and penalizes
    (B_coil + B_plasma)*n

    Parameters
    ----------
    ext_field : MagneticField
        External field produced by coils.
    eq : Equilibrium
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
    src_grid, eval_grid : Grid, optional
        Collocation grid containing the nodes to evaluate at for source terms and where
        to evaluate errors.
    field_grid : Grid, optional
        Grid used to discretize ext_field.
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = False
    _print_value_fmt = "Boundary normal field Error: {:10.3e} "
    _units = "(T)"
    _coordinates = "rtz"

    def __init__(
        self,
        ext_field,
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
        name="Quadratic flux",
    ):
        if target is None and bounds is None:
            target = 0
        self._src_grid = src_grid
        self._eval_grid = eval_grid
        self._s = s
        self._q = q
        self._ext_field = ext_field
        self._field_grid = field_grid
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
            interpolator = FFTInterpolator(eval_grid, src_grid, self._s, self._q)
        except AssertionError as e:
            warnings.warn(
                "Could not built fft interpolator, switching to dft method which is"
                " much slower. Reason: " + str(e)
            )
            interpolator = DFTInterpolator(eval_grid, src_grid, self._s, self._q)

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

        src_profiles = get_profiles(self._data_keys, obj=eq, grid=src_grid)
        src_transforms = get_transforms(self._data_keys, obj=eq, grid=src_grid)
        eval_profiles = get_profiles(self._data_keys, obj=eq, grid=eval_grid)
        eval_transforms = get_transforms(self._data_keys, obj=eq, grid=eval_grid)

        self._constants = {
            "eval_transforms": eval_transforms,
            "eval_profiles": eval_profiles,
            "src_transforms": src_transforms,
            "src_profiles": src_profiles,
            "interpolator": interpolator,
            "ext_field": self._ext_field,
            "quad_weights": eval_transforms["grid"].weights,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._dim_f = eval_grid.num_nodes

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["B"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute boundary force error.

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
            Boundary force error (N).

        """
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
        Bplasma = -virtual_casing_biot_savart(
            eval_data,
            constants["eval_transforms"]["grid"],
            src_data,
            constants["src_transforms"]["grid"],
            constants["interpolator"],
        )
        # don't need extra B/2 since we only care about normal component
        Bplasma = xyz2rpz_vec(Bplasma, phi=eval_data["zeta"])
        x = jnp.array([eval_data["R"], eval_data["zeta"], eval_data["Z"]]).T
        Bext = constants["ext_field"].compute_magnetic_field(
            x, grid=self._field_grid, basis="rpz"
        )
        return jnp.sum((Bext + Bplasma) * eval_data["n_rho"], axis=-1)


class ToroidalFlux(_Objective):
    """Target toroidal flux through the torus.

    Parameters
    ----------
    ext_field : MagneticField
        External field produced by coils.
    eq : Equilibrium
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
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
    field_grid : Grid, optional
        Grid used to discretize ext_field.
    name : str
        Name of the objective function.

    """

    _scalar = False
    _linear = False
    _print_value_fmt = "Toroidal flux Error: {:10.3e} "
    _units = "(Wb)"
    _coordinates = ""

    def __init__(
        self,
        ext_field,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        grid=None,
        field_grid=None,
        name="Toroidal flux",
    ):
        if target is None and bounds is None:
            target = 0
        self._grid = grid
        self._ext_field = ext_field
        self._field_grid = field_grid
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

        profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
        transforms = get_transforms(self._data_keys, obj=eq, grid=grid)

        self._constants = {
            "profiles": profiles,
            "transforms": transforms,
            "ext_field": self._ext_field,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._dim_f = 1

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["B"] * scales["A"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute boundary force error.

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
            Boundary force error (N).

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
        x = jnp.array([data["R"], data["zeta"], data["Z"]]).T
        Bext = constants["ext_field"].compute_magnetic_field(
            x, grid=self._field_grid, basis="rpz"
        )
        n = data["e^zeta"] / jnp.linalg.norm(data["e^zeta"], axis=-1)[:, None]
        Bn = jnp.sum(Bext * n, axis=-1)
        return (
            jnp.sum(
                Bn
                * data["|e_rho x e_theta|"]
                * constants["transforms"]["grid"].spacing[:, :2].prod(axis=-1)
            )
            - params["Psi"]
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
    eq : Equilibrium
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
    _print_value_fmt = "Boundary Force Error: {:10.3e} "
    _units = "(N)"
    _coordinates = "rtz"

    def __init__(
        self,
        ext_field,
        eq,
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

        eq._sym = False
        nest = Nestor(
            eq,
            self.ext_field,
            self.mf,
            self.nf,
            self.ntheta,
            self.nzeta,
            self.field_grid,
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

        profiles = get_profiles(self._data_keys, obj=eq, grid=self.grid)
        transforms = get_transforms(self._data_keys, obj=eq, grid=self.grid)

        self._constants = {
            "profiles": profiles,
            "transforms": transforms,
            "ext_field": self.ext_field,
            "nestor": nest,
            "quad_weights": transforms["grid"].weights,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._dim_f = self.grid.num_nodes

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["B"] ** 2 * scales["R0"] * scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute boundary force error.

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
            Boundary force error (N).

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
        bv = bsq / (2 * mu_0)

        bp = data["|B|^2"] / (2 * mu_0)
        g = data["|e_theta x e_zeta|"]
        return (bv - bp - data["p"]) * g * mu_0
