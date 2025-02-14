"""Objectives for solving free boundary equilibria."""

import numpy as np
from scipy.constants import mu_0

from desc.backend import jnp
from desc.compute import get_params, get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.integrals import DFTInterpolator, FFTInterpolator, virtual_casing_biot_savart
from desc.nestor import Nestor
from desc.objectives.objective_funs import _Objective, collect_docs
from desc.utils import (
    PRINT_WIDTH,
    Timer,
    errorif,
    parse_argname_change,
    setdefault,
    warnif,
)

from ..integrals.singularities import _get_default_params
from .normalization import compute_scaling_factors


class VacuumBoundaryError(_Objective):
    """Target for free boundary conditions on LCFS for vacuum equilibrium.

    Computes the residuals of the following:

    𝐁ₒᵤₜ ⋅ 𝐧 = 0
    𝐁ₒᵤₜ² - 𝐁ᵢₙ² = 0

    Where 𝐁ᵢₙ is the total field inside the LCFS (from fixed boundary calculation)
    𝐁ₒᵤₜ is the total field outside the LCFS (from coils), and 𝐧 is the outward surface
    normal. All residuals are weighted by the local area element ||𝐞_θ × 𝐞_ζ|| Δθ Δζ

    (Technically for vacuum equilibria the second condition is redundant with the first,
    but including it makes things more robust).

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    field : MagneticField
        External field produced by coils or other sources outside the plasma.
    eval_grid : Grid, optional
        Collocation grid containing the nodes to evaluate error at. Should be at rho=1.
        Defaults to ``LinearGrid(M=eq.M_grid, N=eq.N_grid)``
    field_grid : Grid, optional
        Grid used to discretize field. Defaults to the default grid for given field.
    field_fixed : bool
        Whether to assume the field is fixed. For free boundary solve, should
        be fixed. For single stage optimization, should be False (default).

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )

    _scalar = False
    _linear = False
    _print_value_fmt = "Boundary Error: "
    _units = "(T*m^2, T^2*m^2)"
    _coordinates = "rtz"

    def __init__(
        self,
        eq,
        field,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        eval_grid=None,
        field_grid=None,
        field_fixed=False,
        name="Vacuum boundary error",
        jac_chunk_size=None,
        device_id=0,
        **kwargs,
    ):
        eval_grid = parse_argname_change(eval_grid, kwargs, "grid", "eval_grid")
        if target is None and bounds is None:
            target = 0
        self._eval_grid = eval_grid
        self._eq = eq
        self._field = field
        self._field_grid = field_grid
        self._field_fixed = field_fixed
        if field_fixed:
            things = [eq]
        else:
            things = [eq, field]
        super().__init__(
            things=things,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            name=name,
            jac_chunk_size=jac_chunk_size,
            device_id=device_id,
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
        if self._eval_grid is None:
            grid = LinearGrid(
                rho=np.array([1.0]), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False
            )
        else:
            grid = self._eval_grid

        data = eq.compute(["p", "current"])
        pres = np.max(np.abs(data["p"]))
        curr = np.max(np.abs(data["current"]))
        errorif(
            not np.all(grid.nodes[:, 0] == 1.0),
            ValueError,
            "grid contains nodes not on rho=1",
        )
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
            "phi",
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
            "field": self._field,
            "quad_weights": np.sqrt(np.tile(transforms["grid"].weights, 2)),
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._dim_f = 2 * grid.num_nodes

        if self._normalize:
            scales = compute_scaling_factors(eq)
            Bn_norm = np.ones(grid.num_nodes) * scales["B"] * scales["R0"] * scales["a"]
            B2_norm = (
                np.ones(grid.num_nodes) * scales["B"] ** 2 * scales["R0"] * scales["a"]
            )
            self._normalization = np.concatenate([Bn_norm, B2_norm])

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, eq_params, field_params=None, constants=None):
        """Compute boundary force error.

        Parameters
        ----------
        eq_params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        field_params : dict
            Dictionary of field parameters, if field is not fixed.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Boundary error. First half is √g𝐁⋅𝐧 in T*m^2, second half is
            √g[[B²]] in T^2*m^2.

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
        x = jnp.array([data["R"], data["phi"], data["Z"]]).T
        # can always pass in field params. If they're None, it just uses the
        # defaults for the given field.
        Bext = constants["field"].compute_magnetic_field(
            x, source_grid=self._field_grid, basis="rpz", params=field_params
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

    def print_value(self, args, args0=None, **kwargs):
        """Print the value of the objective."""
        # this objective is really 2 residuals concatenated so its helpful to print
        # them individually
        f = self.compute_unscaled(*args, **kwargs)
        f0 = self.compute_unscaled(*args0, **kwargs) if args0 is not None else f
        # try to do weighted mean if possible
        constants = kwargs.get("constants", self.constants)
        if constants is None:
            w = jnp.ones_like(f)
        else:
            w = constants["quad_weights"]

        abserr = jnp.all(self.target == 0)
        pre_width = len("Maximum absolute ") if abserr else len("Maximum ")

        def _print(fmt, fmax, fmin, fmean, f0max, f0min, f0mean, norm, units):

            print(
                "Maximum "
                + ("absolute " if abserr else "")
                + fmt.format(f0max, fmax)
                + units
            )
            print(
                "Minimum "
                + ("absolute " if abserr else "")
                + fmt.format(f0min, fmin)
                + units
            )
            print(
                "Average "
                + ("absolute " if abserr else "")
                + fmt.format(f0mean, fmean)
                + units
            )

            if self._normalize and units != "(dimensionless)":
                print(
                    "Maximum "
                    + ("absolute " if abserr else "")
                    + fmt.format(f0max / norm, fmax / norm)
                    + "(normalized)"
                )
                print(
                    "Minimum "
                    + ("absolute " if abserr else "")
                    + fmt.format(f0min / norm, fmin / norm)
                    + "(normalized)"
                )
                print(
                    "Average "
                    + ("absolute " if abserr else "")
                    + fmt.format(f0mean / norm, fmean / norm)
                    + "(normalized)"
                )

        formats = [
            "Boundary normal field error: ",
            "Boundary magnetic pressure error: ",
        ]
        units = ["(T*m^2)", "(T^2*m^2)"]
        nn = f.size // 2
        norms = [self.normalization[0], self.normalization[nn]]
        for i, (fmt, norm, units) in enumerate(zip(formats, norms, units)):
            fi = f[i * nn : (i + 1) * nn]
            f0i = f0[i * nn : (i + 1) * nn]
            # target == 0 probably indicates f is some sort of error metric,
            # mean abs makes more sense than mean
            fi = jnp.abs(fi) if abserr else fi
            f0i = jnp.abs(f0i) if abserr else f0i
            wi = w[i * nn : (i + 1) * nn]
            fmax = jnp.max(fi)
            fmin = jnp.min(fi)
            fmean = jnp.mean(fi * wi) / jnp.mean(wi)

            f0max = jnp.max(f0i)
            f0min = jnp.min(f0i)
            f0mean = jnp.mean(f0i * wi) / jnp.mean(wi)
            fmt = (
                f"{fmt:<{PRINT_WIDTH-pre_width}}" + "{:10.3e}  -->  {:10.3e} "
                if args0 is not None
                else f"{fmt:<{PRINT_WIDTH-pre_width}}" + "{:10.3e} "
            )
            _print(fmt, fmax, fmin, fmean, f0max, f0min, f0mean, norm, units)


class BoundaryError(_Objective):
    """Target for free boundary conditions on LCFS for finite beta equilibrium.

    Computes the residual of the following:

    𝐁ₒᵤₜ ⋅ 𝐧 = 0
    𝐁ₒᵤₜ² - 𝐁ᵢₙ² - p = 0
    μ₀∇Φ − 𝐧 × [𝐁ₒᵤₜ − 𝐁ᵢₙ]

    Where 𝐁ᵢₙ is the total field inside the LCFS (from fixed boundary calculation)
    𝐁ₒᵤₜ is the total field outside the LCFS (from coils and virtual casing principle),
    𝐧 is the outward surface normal, p is the plasma pressure, and Φ is the surface
    current potential on the LCFS. All residuals are weighted by the local area
    element ||𝐞_θ × 𝐞_ζ|| Δθ Δζ

    The third equation is only included if a sheet current is supplied by making
    the ``equilibrium.surface`` object a FourierCurrentPotentialField, otherwise it
    is trivially satisfied. If it is known that the external field accurately reproduces
    the target equilibrium with low normal field error and pressure at the edge is zero,
    then the sheet current will generally be negligible and can be omitted to save
    effort.

    This objective also works for vacuum equilibria, though in that case
    VacuumBoundaryError will be much faster as it avoids the singular virtual casing
    integral.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    field : MagneticField
        External field produced by coils.
    s : int
        Hyperparameter for the singular integration scheme.
        ``s`` is roughly equal to the size of the local singular grid with
        respect to the global grid.
    q : int
        Order of integration on the local singular grid.
    source_grid, eval_grid : Grid, optional
        Collocation grid containing the nodes to evaluate at for source terms for Biot-
        Savart integral and where to evaluate errors. ``source_grid`` should not be
        stellarator symmetric, and both should be at rho=1.
        Defaults to ``LinearGrid(M=eq.M_grid, N=eq.N_grid)`` for both.
    field_grid : Grid, optional
        Grid used to discretize field. Defaults to default grid for given field.
    field_fixed : bool
        Whether to assume the field is fixed. For free boundary solve, should
        be fixed. For single stage optimization, should be False (default).
    chunk_size : int or None
        Size to split computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``1``.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )
    __doc__ += """
    Examples
    --------
    Assigning a surface current to the equilibrium:

    .. code-block:: python

        from desc.magnetic_fields import FourierCurrentPotentialField
        # turn the regular FourierRZToroidalSurface into a current potential on the
        # last closed flux surface
        eq.surface = FourierCurrentPotentialField.from_surface(eq.surface,
                                                              M_Phi=eq.M,
                                                              N_Phi=eq.N,
                                                              )
        objective = BoundaryError(eq, field)

    """

    _scalar = False
    _linear = False
    _print_value_fmt = "Boundary Error: "
    _units = "(T*m^2, T^2*m^2, T*m^2)"

    _coordinates = "rtz"

    def __init__(
        self,
        eq,
        field,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        s=None,
        q=None,
        source_grid=None,
        eval_grid=None,
        field_grid=None,
        field_fixed=False,
        chunk_size=1,
        name="Boundary error",
        jac_chunk_size=None,
        device_id=0,
        **kwargs,
    ):
        if target is None and bounds is None:
            target = 0
        self._source_grid = source_grid
        self._eval_grid = eval_grid
        self._eval_grid_is_source_grid = source_grid == eval_grid
        self._st = s
        self._sz = kwargs.get("sz", s)
        self._q = q
        self._field = field
        self._field_grid = field_grid
        self._chunk_size = parse_argname_change(
            chunk_size, kwargs, "loop", "chunk_size"
        )
        if self._chunk_size == 0:
            self._chunk_size = None
        self._sheet_current = hasattr(eq.surface, "Phi_mn")
        if field_fixed:
            things = [eq]
        else:
            things = [eq, field]

        super().__init__(
            things=things,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            name=name,
            jac_chunk_size=jac_chunk_size,
            device_id=device_id,
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

        if self._source_grid is None:
            # for axisymmetry we still need to know about toroidal effects, so its
            # cheapest to pretend there are extra field periods
            source_grid = LinearGrid(
                rho=np.array([1.0]),
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP if eq.N > 0 else 64,
                sym=False,
            )
        else:
            source_grid = self._source_grid

        if self._eval_grid is None:
            eval_grid = source_grid
        else:
            eval_grid = self._eval_grid

        self._eval_grid_is_source_grid = eval_grid.equiv(source_grid)

        errorif(
            not np.all(source_grid.nodes[:, 0] == 1.0),
            ValueError,
            "source_grid contains nodes not on rho=1",
        )
        errorif(
            not np.all(eval_grid.nodes[:, 0] == 1.0),
            ValueError,
            "eval_grid contains nodes not on rho=1",
        )
        errorif(
            source_grid.sym,
            ValueError,
            "Source grids for singular integrals must be non-symmetric",
        )
        st, sz, q = _get_default_params(source_grid)
        self._st = setdefault(self._st, st)
        self._sz = setdefault(self._sz, sz)
        self._q = setdefault(self._q, q)

        try:
            interpolator = FFTInterpolator(
                eval_grid, source_grid, self._st, self._sz, self._q
            )
        except AssertionError as e:
            warnif(
                True,
                msg="Could not build fft interpolator, switching to dft which is slow."
                "\nReason: " + str(e),
            )
            interpolator = DFTInterpolator(
                eval_grid, source_grid, self._st, self._sz, self._q
            )

        edge_pres = np.max(np.abs(eq.compute("p", grid=eval_grid)["p"]))
        warnif(
            (edge_pres * mu_0 > 1e-6) and not self._sheet_current,
            UserWarning,
            f"Boundary pressure is nonzero (max {edge_pres} Pa), "
            + "a sheet current should be included.",
        )

        self._eq_data_keys = [
            "K_vc",
            "B",
            "|B|^2",
            "R",
            "phi",
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

        source_profiles = get_profiles(self._eq_data_keys, obj=eq, grid=source_grid)
        source_transforms = get_transforms(self._eq_data_keys, obj=eq, grid=source_grid)
        if self._eval_grid_is_source_grid:
            eval_profiles = source_profiles
            eval_transforms = source_transforms
        else:
            eval_profiles = get_profiles(self._eq_data_keys, obj=eq, grid=eval_grid)
            eval_transforms = get_transforms(self._eq_data_keys, obj=eq, grid=eval_grid)

        neq = 3 if self._sheet_current else 2  # number of equations we're using

        self._constants = {
            "eval_transforms": eval_transforms,
            "eval_profiles": eval_profiles,
            "source_transforms": source_transforms,
            "source_profiles": source_profiles,
            "interpolator": interpolator,
            "field": self._field,
            "quad_weights": np.sqrt(np.tile(eval_transforms["grid"].weights, neq)),
        }

        if self._sheet_current:
            self._sheet_data_keys = ["K"]
            self._constants["sheet_source_transforms"] = get_transforms(
                self._sheet_data_keys, obj=eq.surface, grid=source_grid
            )
            self._constants["sheet_eval_transforms"] = (
                self._constants["sheet_source_transforms"]
                if self._eval_grid_is_source_grid
                else get_transforms(
                    self._sheet_data_keys, obj=eq.surface, grid=eval_grid
                )
            )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._dim_f = neq * eval_grid.num_nodes

        if self._normalize:
            scales = compute_scaling_factors(eq)
            Bn_norm = (
                np.ones(eval_grid.num_nodes) * scales["B"] * scales["R0"] * scales["a"]
            )
            B2_norm = (
                np.ones(eval_grid.num_nodes)
                * scales["B"] ** 2
                * scales["R0"]
                * scales["a"]
            )
            self._normalization = np.concatenate([Bn_norm, B2_norm])
            if self._sheet_current:
                self._normalization = np.concatenate([self._normalization, Bn_norm])

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, eq_params, field_params=None, constants=None):
        """Compute boundary force error.

        Parameters
        ----------
        eq_params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        field_params : dict
            Dictionary of field parameters, if field is not fixed.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Boundary error. First half is √g𝐁⋅𝐧 in T*m^2, second half is
            √g[[B² + 2μ₀p]] in T^2*m^2. If sheet current is included, third half is
            √g||μ₀𝐊 − 𝐧 × [𝐁]|| in T*m^2

        """
        if constants is None:
            constants = self.constants
        source_data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._eq_data_keys,
            params=eq_params,
            transforms=constants["source_transforms"],
            profiles=constants["source_profiles"],
        )
        eval_data = (
            source_data
            if self._eval_grid_is_source_grid
            else compute_fun(
                "desc.equilibrium.equilibrium.Equilibrium",
                self._eq_data_keys,
                params=eq_params,
                transforms=constants["eval_transforms"],
                profiles=constants["eval_profiles"],
            )
        )
        if self._sheet_current:
            p = "desc.magnetic_fields._current_potential.FourierCurrentPotentialField"
            sheet_params = {
                "R_lmn": eq_params["Rb_lmn"],
                "Z_lmn": eq_params["Zb_lmn"],
                "I": eq_params["I"],
                "G": eq_params["G"],
                "Phi_mn": eq_params["Phi_mn"],
            }
            sheet_source_data = compute_fun(
                p,
                self._sheet_data_keys,
                params=sheet_params,
                transforms=constants["sheet_source_transforms"],
                profiles={},
            )
            sheet_eval_data = (
                sheet_source_data
                if self._eval_grid_is_source_grid
                else compute_fun(
                    p,
                    self._sheet_data_keys,
                    params=sheet_params,
                    transforms=constants["sheet_eval_transforms"],
                    profiles={},
                )
            )
            source_data["K_vc"] += sheet_source_data["K"]

        Bplasma = virtual_casing_biot_savart(
            eval_data,
            source_data,
            constants["interpolator"],
            chunk_size=self._chunk_size,
        )
        # need extra factor of B/2 bc we're evaluating on plasma surface
        Bplasma = Bplasma + eval_data["B"] / 2
        x = jnp.array([eval_data["R"], eval_data["phi"], eval_data["Z"]]).T
        # can always pass in field params. If they're None, it just uses the
        # defaults for the given field.
        Bext = constants["field"].compute_magnetic_field(
            x, source_grid=self._field_grid, basis="rpz", params=field_params
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

    def print_value(self, args, args0=None, **kwargs):
        """Print the value of the objective."""
        # this objective is really 3 residuals concatenated so its helpful to print
        # them individually
        f = self.compute_unscaled(*args, **kwargs)
        f0 = self.compute_unscaled(*args0, **kwargs) if args0 is not None else f
        # try to do weighted mean if possible
        constants = kwargs.get("constants", self.constants)
        if constants is None:
            w = jnp.ones_like(f)
        else:
            w = constants["quad_weights"]

        abserr = jnp.all(self.target == 0)
        pre_width = len("Maximum absolute ") if abserr else len("Maximum ")

        def _print(fmt, fmax, fmin, fmean, f0max, f0min, f0mean, norm, unit):

            print(
                "Maximum "
                + ("absolute " if abserr else "")
                + fmt.format(f0max, fmax)
                + unit
            )
            print(
                "Minimum "
                + ("absolute " if abserr else "")
                + fmt.format(f0min, fmin)
                + unit
            )
            print(
                "Average "
                + ("absolute " if abserr else "")
                + fmt.format(f0mean, fmean)
                + unit
            )

            if self._normalize and units != "(dimensionless)":
                print(
                    "Maximum "
                    + ("absolute " if abserr else "")
                    + fmt.format(f0max / norm, fmax / norm)
                    + "(normalized)"
                )
                print(
                    "Minimum "
                    + ("absolute " if abserr else "")
                    + fmt.format(f0min / norm, fmin / norm)
                    + "(normalized)"
                )
                print(
                    "Average "
                    + ("absolute " if abserr else "")
                    + fmt.format(f0mean / norm, fmean / norm)
                    + "(normalized)"
                )

        formats = [
            "Boundary normal field error: ",
            "Boundary magnetic pressure error: ",
            "Boundary field jump error: ",
        ]
        units = ["(T*m^2)", "(T^2*m^2)", "(T*m^2)"]
        if self._sheet_current:
            nn = f.size // 3
            norms = [
                self.normalization[0],
                self.normalization[nn],
                self.normalization[-1],
            ]
        else:
            nn = f.size // 2
            formats = formats[:-1]
            units = units[:-1]
            norms = [self.normalization[0], self.normalization[nn]]
        for i, (fmt, norm, unit) in enumerate(zip(formats, norms, units)):
            fi = f[i * nn : (i + 1) * nn]
            f0i = f0[i * nn : (i + 1) * nn]
            # target == 0 probably indicates f is some sort of error metric,
            # mean abs makes more sense than mean
            fi = jnp.abs(fi) if abserr else fi
            f0i = jnp.abs(f0i) if abserr else fi
            wi = w[i * nn : (i + 1) * nn]
            fmax = jnp.max(fi)
            fmin = jnp.min(fi)
            fmean = jnp.mean(fi * wi) / jnp.mean(wi)

            f0max = jnp.max(f0i)
            f0min = jnp.min(f0i)
            f0mean = jnp.mean(f0i * wi) / jnp.mean(wi)
            fmt = (
                f"{fmt:<{PRINT_WIDTH-pre_width}}" + "{:10.3e}  -->  {:10.3e} "
                if args0 is not None
                else f"{fmt:<{PRINT_WIDTH-pre_width}}" + "{:10.3e} "
            )
            _print(fmt, fmax, fmin, fmean, f0max, f0min, f0mean, norm, unit)


class BoundaryErrorNESTOR(_Objective):
    """Pressure balance across LCFS.

    Uses NESTOR algorithm to compute B_vac such that (B_vac + B_coil)*n=0,
    then calculates the pressure mismatch across the boundary:

        1/2mu0*(B_vac + B_coil)^2 - 1/2mu0*B_plasma^2 - p

    Residuals are weighted by the local area element ||𝐞_θ × 𝐞_ζ|| Δθ Δζ

    Note: This objective is still experimental and may not work in all cases.
    Recommend using ``BoundaryError`` or ``VacuumBoundaryError``

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    field : MagneticField
        External field produced by coils.
    mf, nf : integer
        maximum poloidal and toroidal mode numbers to use for NESTOR scalar potential.
    ntheta, nzeta : int
        number of grid points in poloidal, toroidal directions to use in NESTOR.
    field_grid : Grid, optional
        Grid used to discretize field.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )

    _scalar = False
    _linear = False
    _print_value_fmt = "Boundary magnetic pressure error: "
    _units = "(T^2*m^2)"
    _coordinates = "rtz"

    def __init__(
        self,
        eq,
        field,
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
        loss_function=None,
        deriv_mode="auto",
        name="NESTOR Boundary",
        jac_chunk_size=None,
        device_id=0,
    ):
        if target is None and bounds is None:
            target = 0
        self.mf = mf
        self.nf = nf
        self.ntheta = ntheta
        self.nzeta = nzeta
        self.field = field
        self.field_grid = field_grid
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            name=name,
            jac_chunk_size=jac_chunk_size,
            device_id=device_id,
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
            self.field,
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
            "field": self.field,
            "nestor": nest,
            "quad_weights": np.sqrt(transforms["grid"].weights),
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
            Boundary magnetic pressure error (T^2*m^2).

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
        bsq = out[1]["|B|^2"].reshape(grid.num_zeta, grid.num_theta).T.flatten()
        bv = bsq

        bp = data["|B|^2"]
        g = data["|e_theta x e_zeta|"]
        return (bv - bp - data["p"] * (2 * mu_0)) * g
