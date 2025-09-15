"""Objectives for solving free boundary equilibria."""

import numpy as np
from scipy.constants import mu_0

from desc.backend import jnp
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.compute.utils import _compute_RpZ_data as compute_fun_rpz
from desc.grid import LinearGrid
from desc.integrals import get_interpolator, virtual_casing_biot_savart
from desc.nestor import Nestor
from desc.objectives.objective_funs import _Objective, collect_docs
from desc.utils import (
    PRINT_WIDTH,
    Timer,
    cross,
    dot,
    errorif,
    parse_argname_change,
    setdefault,
    warnif,
)

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
    bs_chunk_size : int or None
        Size to split Biot-Savart computation into chunks of evaluation points.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )

    _static_attrs = _Objective._static_attrs + [
        "_bs_chunk_size",
        "_eq_data_keys",
        "_field_fixed",
    ]

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
        *,
        bs_chunk_size=None,
        **kwargs,
    ):
        eval_grid = parse_argname_change(eval_grid, kwargs, "grid", "eval_grid")
        if target is None and bounds is None:
            target = 0
        self._eval_grid = eval_grid
        self._eq = eq
        self._field = [field] if not isinstance(field, list) else field
        self._field_grid = field_grid
        self._field_fixed = field_fixed
        self._bs_chunk_size = bs_chunk_size
        things = [eq]
        if not field_fixed:
            things.append(self._field)
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
        from desc.magnetic_fields import SumMagneticField

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
            "field": SumMagneticField(self._field),
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

    def compute(self, eq_params, *field_params, constants=None):
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
        if field_params == ():  # common case for field_fixed=True
            field_params = None
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
            x,
            source_grid=self._field_grid,
            basis="rpz",
            params=field_params,
            chunk_size=self._bs_chunk_size,
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
        """Print the value of the objective and return a dict of values."""
        out = {}
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
        for i, (fmti, norm, units) in enumerate(zip(formats, norms, units)):
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
            out[fmti] = {
                "f_max": fmax,
                "f_min": fmin,
                "f_mean": fmean,
                "f_max_norm": fmax / norm,
                "f_min_norm": fmin / norm,
                "f_mean_norm": fmean / norm,
            }
            if args0 is not None:
                out[fmti]["f0_max"] = f0max
                out[fmti]["f0_min"] = f0min
                out[fmti]["f0_mean"] = f0mean
                out[fmti]["f0_max_norm"] = f0max / norm
                out[fmti]["f0_min_norm"] = f0min / norm
                out[fmti]["f0_mean_norm"] = f0mean / norm
            fmt = (
                f"{fmti:<{PRINT_WIDTH-pre_width}}" + "{:10.3e}  -->  {:10.3e} "
                if args0 is not None
                else f"{fmti:<{PRINT_WIDTH-pre_width}}" + "{:10.3e} "
            )
            _print(fmt, fmax, fmin, fmean, f0max, f0min, f0mean, norm, units)
        return out


class BoundaryError(_Objective):
    """Target for free boundary conditions on LCFS for finite beta equilibrium.

    Computes the residual of the following:

    𝐁ₒᵤₜ ⋅ 𝐧 = 0
    𝐁ₒᵤₜ² - 𝐁ᵢₙ² - 2μ₀p = 0
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
    s : int or tuple[int]
        Hyperparameter for the singular integration scheme.
        ``s`` is roughly equal to the size of the local singular grid with
        respect to the global grid. More precisely the local singular grid
        is an ``st`` × ``sz`` subset of the full domain (θ,ζ) ∈ [0, 2π)² of
        ``source_grid``. That is a subset of
        ``source_grid.num_theta`` × ``source_grid.num_zeta*source_grid.NFP``.
        If given an integer then ``st=s``, ``sz=s``, otherwise ``st=s[0]``, ``sz=s[1]``.
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
    bs_chunk_size : int or None
        Size to split Biot-Savart computation into chunks of evaluation points.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.
    B_plasma_chunk_size : int or None
        Size to split singular integral computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``bs_chunk_size``.

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

    _static_attrs = _Objective._static_attrs + [
        "_B_plasma_chunk_size",
        "_bs_chunk_size",
        "_eq_data_keys",
        "_field_fixed",
        "_q",
        "_sheet_current",
        "_sheet_data_keys",
        "_use_same_grid",
    ]

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
        name="Boundary error",
        jac_chunk_size=None,
        *,
        bs_chunk_size=None,
        B_plasma_chunk_size=None,
        **kwargs,
    ):
        if target is None and bounds is None:
            target = 0
        self._source_grid = source_grid
        self._eval_grid = eval_grid
        self._st, self._sz = s if isinstance(s, (tuple, list)) else (s, s)
        self._q = q
        self._field = [field] if not isinstance(field, list) else field
        self._field_grid = field_grid
        self._bs_chunk_size = bs_chunk_size
        B_plasma_chunk_size = parse_argname_change(
            B_plasma_chunk_size, kwargs, "loop", "B_plasma_chunk_size"
        )
        if B_plasma_chunk_size == 0:
            B_plasma_chunk_size = None
        self._B_plasma_chunk_size = B_plasma_chunk_size
        self._sheet_current = hasattr(eq.surface, "Phi_mn")
        things = [eq]
        if not field_fixed:
            things.append(self._field)
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
        from desc.magnetic_fields import SumMagneticField

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

        eval_grid = setdefault(self._eval_grid, source_grid)
        self._use_same_grid = eval_grid.equiv(source_grid)

        ratio_data = (
            eq.compute(["|e_theta x e_zeta|", "e_theta", "e_zeta"], grid=source_grid)
            if (self._st is None or self._sz is None or self._q is None)
            else {}
        )
        interpolator = get_interpolator(
            eval_grid, source_grid, ratio_data, st=self._st, sz=self._sz, q=self._q
        )
        del self._st
        del self._sz
        del self._q

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
        if self._use_same_grid:
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
            "field": SumMagneticField(self._field),
            "quad_weights": np.sqrt(np.tile(eval_transforms["grid"].weights, neq)),
        }

        if self._sheet_current:
            self._sheet_data_keys = ["K"]
            self._constants["sheet_source_transforms"] = get_transforms(
                self._sheet_data_keys, obj=eq.surface, grid=source_grid
            )
            self._constants["sheet_eval_transforms"] = (
                self._constants["sheet_source_transforms"]
                if self._use_same_grid
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

    def compute(self, eq_params, *field_params, constants=None):
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
        if field_params == ():  # common case for field_fixed=True
            field_params = None
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
            if self._use_same_grid
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
                if self._use_same_grid
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
            chunk_size=self._B_plasma_chunk_size,
        )
        # need extra factor of B/2 bc we're evaluating on plasma surface
        Bplasma = Bplasma + eval_data["B"] / 2
        x = jnp.array([eval_data["R"], eval_data["phi"], eval_data["Z"]]).T
        # can always pass in field params. If they're None, it just uses the
        # defaults for the given field.
        Bext = constants["field"].compute_magnetic_field(
            x,
            source_grid=self._field_grid,
            basis="rpz",
            params=field_params,
            chunk_size=self._bs_chunk_size,
        )
        Bex_total = Bext + Bplasma
        Bin_total = eval_data["B"]
        Bn = jnp.sum(Bex_total * eval_data["n_rho"], axis=-1)

        bsq_out = jnp.sum(Bex_total * Bex_total, axis=-1)
        bsq_in = jnp.sum(Bin_total * Bin_total, axis=-1)

        g = eval_data["|e_theta x e_zeta|"]
        Bn_err = Bn * g
        Bsq_err = jnp.where(
            eval_data["p"] == 0,
            (bsq_in - bsq_out) * g,
            (bsq_in - bsq_out + eval_data["p"] * 2 * mu_0) * g,
        )
        Bjump = Bex_total - Bin_total
        if self._sheet_current:
            Kerr = mu_0 * sheet_eval_data["K"] - cross(eval_data["n_rho"], Bjump)
            Kerr = jnp.linalg.norm(Kerr, axis=-1) * g
            return jnp.concatenate([Bn_err, Bsq_err, Kerr])
        else:
            return jnp.concatenate([Bn_err, Bsq_err])

    def print_value(self, args, args0=None, **kwargs):
        """Print the value of the objective and return a dict of values."""
        out = {}
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
        for i, (fmti, norm, unit) in enumerate(zip(formats, norms, units)):
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
            out[fmti] = {
                "f_max": fmax,
                "f_min": fmin,
                "f_mean": fmean,
                "f_max_norm": fmax / norm,
                "f_min_norm": fmin / norm,
                "f_mean_norm": fmean / norm,
            }
            if args0 is not None:
                out[fmti]["f0_max"] = f0max
                out[fmti]["f0_min"] = f0min
                out[fmti]["f0_mean"] = f0mean
                out[fmti]["f0_max_norm"] = f0max / norm
                out[fmti]["f0_min_norm"] = f0min / norm
                out[fmti]["f0_mean_norm"] = f0mean / norm
            fmt = (
                f"{fmti:<{PRINT_WIDTH-pre_width}}" + "{:10.3e}  -->  {:10.3e} "
                if args0 is not None
                else f"{fmti:<{PRINT_WIDTH-pre_width}}" + "{:10.3e} "
            )
            _print(fmt, fmax, fmin, fmean, f0max, f0min, f0mean, norm, unit)
        return out


class FreeSurfaceError(_Objective):
    """Target for free surface ideal MHD equilirium conditions on LCFS.

    References
    ----------
        [1] Unalmis et al. New high-order accurate free surface stellarator
            equilibria optimization and boundary integral methods in DESC.

    Warnings
    --------
    Bugs with the optimizer may cause the optimizer to stall in the optimization
    landscape of this objective. See GitHub issue #1208 to resolve.
    Also, if ``field`` is an instance of ``FreeSurfaceOuterField``, then
    ``field._B_coil`` should be smooth and divergence free until GitHub issue #1796
    is resolved.

    Parameters
    ----------
    eq : Equilibrium
        ``Equilibrium`` to be optimized.
    field : FreeSurfaceOuterField or SourceFreeField
        Laplace solver object. If is an instance of ``FreeSurfaceOuterField``
        assumes ``field._B_coil`` is the magnetic field due to coils.
        If is an instance of ``SourceFreeField`` then assumes ``field._B0`` is
        the magnetic field due to coils.
    eval_grid : Grid
        Evaluation points on boundary to evaluate objective error.
        Default is ``grid``, but it is likely best to use a grid
        with much lower resolution.
    grid : Grid
        Grid for the integral transforms.
        Tensor-product grid in (θ, ζ) with uniformly spaced nodes
        (θ, ζ) ∈ [0, 2π) × [0, 2π/NFP) on the boundary.
        Default is ``LinearGrid(M=eq.M_grid,N=eq.N_grid,NFP=eq.NFP)``.
    coil_grid : Grid, optional
        Source grid used to discretize coil magnetic field computation.
        Default is default grid of coil magnetic field.
    q : int
        Order of integration on the local singular grid.
    xtol : float
        Stopping tolerance for fixed point method. Default is ``1e-7``.
    maxiter : int
        Maximum number of iterations for fixed point method.
        If non-positive then the linear operator will be inverted instead.
        If positive, then performs that many fixed point iterations until ``maxiter``
        or an error tolerance of ``xtol`` is reached. For reference, ``20`` yields an
        error of ``1e-5`` as illustrated in [1]. Default is ``25``.
    chunk_size : int or None
        Size to split integral computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.  Default is ``None``.
        Recommend to verify computation with ``chunk_size`` set to a
        small number due to bugs in JAX or XLA.
    B_coil_chunk_size : int or None
        Size to split coil integral computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.  Default is ``None``.
    I_sheet : float
        Net toroidal sheet current determining a circulation of Φ.
        Default is zero.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
    )

    _scalar = False
    _print_value_fmt = "Free surface Error: "
    _units = "T^2 m^2"

    _static_attrs = _Objective._static_attrs + [
        "_is_neumann",
        "_field",
        "_B_coil",
        "_use_same_grid",
        "_q",
        "_xtol",
        "_maxiter",
        "_chunk_size",
        "_B_coil_chunk_size",
        "_grad_keys",
        "_inner_keys",
        "_reuseable_keys",
    ]

    _coordinates = "rtz"

    def __init__(
        self,
        eq,
        field,
        *,
        eval_grid=None,
        grid=None,
        coil_grid=None,
        q=None,
        xtol=1e-7,
        maxiter=25,
        chunk_size=None,
        B_coil_chunk_size=None,
        I_sheet=0.0,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        jac_chunk_size=None,
        name="Free surface error",
        **kwargs,
    ):
        if target is None and bounds is None:
            target = 0.0

        if grid is None:
            grid = LinearGrid(
                rho=np.array([1.0]),
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP if eq.N > 0 else 64,
                sym=False,
            )
        assert grid.can_fft2
        errorif(field.M_Phi > grid.M, msg=f"M_Phi = {field.M_Phi} > {grid.M} = grid.M.")
        errorif(field.N_Phi > grid.N, msg=f"N_Phi = {field.N_Phi} > {grid.N} = grid.N.")

        self._is_neumann = not hasattr(field, "M_Phi_coil")
        errorif(
            not self._is_neumann and field.M_Phi_coil > grid.M,
            msg=f"M_Phi_coil = {getattr(field, 'M_Phi_coil', 0)} > {grid.M} = grid.M.",
        )
        errorif(
            not self._is_neumann and field.N_Phi_coil > grid.N,
            msg=f"N_Phi_coil = {getattr(field, 'N_Phi_coil', 0)} > {grid.N} = grid.N.",
        )
        eval_grid = setdefault(eval_grid, grid)

        self._field = field
        self._B_coil = field._B0 if self._is_neumann else field._B_coil
        self._grid = grid
        self._eval_grid = eval_grid
        self._coil_grid = coil_grid
        self._use_same_grid = grid.equiv(eval_grid)
        self._q = q
        self._xtol = xtol
        self._maxiter = maxiter
        self._chunk_size = chunk_size
        self._B_coil_chunk_size = B_coil_chunk_size
        self._grad_keys = ["grad(theta)", "grad(zeta)", "n_rho"]
        self._inner_keys = [
            "|B|^2",
            "p",
            "I",
            "|e_theta x e_zeta|",
            "R",
            "phi",
            "zeta",
            "omega",
            "Z",
        ] + self._grad_keys
        self._reuseable_keys = [
            "0",
            "R",
            "Z",
            "phi",
            "zeta",
            "omega",
            "R_t",
            "R_z",
            "Z_t",
            "Z_z",
            "e_theta",
            "e_theta x e_zeta",
            "e_zeta",
            "n_rho",
            "omega_t",
            "omega_z",
            "|e_theta x e_zeta|",
        ]
        self._I_sheet = I_sheet

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

        eq_transforms = get_transforms(self._inner_keys, eq, grid=self._eval_grid)
        eval_transforms = get_transforms("|K_vc|^2", self._field, grid=self._eval_grid)
        if self._use_same_grid:
            source_transforms = eval_transforms
            grad_transforms = eq_transforms
        else:
            source_transforms = get_transforms("Phi_mn", self._field, grid=self._grid)
            grad_transforms = get_transforms(self._grad_keys, eq, grid=self._grid)

        data, _ = self._field.compute(
            ["interpolator"] if self._is_neumann else ["interpolator", "Y_coil"],
            grid=self._grid,
            q=self._q,
            transforms=source_transforms,
            B_coil=self._B_coil,
            B_coil_chunk_size=self._B_coil_chunk_size,
        )
        # No net poloidal current in equation 4.13 of [1].
        self._field.Y = 0.0 if self._is_neumann else data["Y_coil"]

        self._constants = {
            "interpolator": data["interpolator"],
            "eq_transforms": eq_transforms,
            "grad_transforms": grad_transforms,
            "eval_transforms": eval_transforms,
            "source_transforms": source_transforms,
            "profiles": get_profiles(self._inner_keys, eq, grid=self._eval_grid),
            "quad_weights": np.sqrt(eval_transforms["grid"].weights),
        }
        self._dim_f = self._eval_grid.num_nodes

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = (
                np.ones(self._eval_grid.num_nodes)
                * scales["B"] ** 2
                * scales["R0"]
                * scales["a"]
            )

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute boundary error.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, e.g.
            ``Equilibrium.params_dict``.
        constants : dict
            Dictionary of constant data, e.g. transforms, profiles etc.
            Defaults to ``self.constants``.

        Returns
        -------
        f : ndarray
            Boundary error [[B² + 2μ₀p]]*area Jacobian in T² m².

        """
        if constants is None:
            constants = self.constants
        eq = self.things[0]

        inner = compute_fun(
            eq,
            self._inner_keys,
            params,
            constants["eq_transforms"],
            constants["profiles"],
        )
        field_params = {
            "R_lmn": params["Rb_lmn"],
            "Z_lmn": params["Zb_lmn"],
            # This is I_plasma + I_sheet.
            "I": inner["I"][self._eval_grid.unique_rho_idx[-1]] + self._I_sheet,
            "Y": self._field.Y,
        }
        outer = {key: inner[key] for key in self._reuseable_keys}

        if self._is_neumann:
            outer = compute_fun(
                self._field,
                "B_coil",
                field_params,
                constants["eval_transforms"],
                constants["profiles"],
                data=outer,
                B_coil_chunk_size=self._B_coil_chunk_size,
                B_coil=self._B_coil,
                field_grid=self._coil_grid,
            )

        problem = {"problem": "exterior Neumann"} if self._is_neumann else {}

        if self._use_same_grid:
            outer["interpolator"] = constants["interpolator"]
            outer["B0*n"] = self._phi_sec_dot_n(field_params, inner)
            if self._is_neumann:
                outer["B0*n"] += dot(outer["B_coil"], inner["n_rho"])
        else:
            grads = compute_fun(
                eq,
                self._grad_keys,
                params,
                constants["grad_transforms"],
                constants["profiles"],
            )
            data = {key: grads[key] for key in self._reuseable_keys}
            data["interpolator"] = constants["interpolator"]
            data["B0*n"] = self._phi_sec_dot_n(field_params, grads)
            if self._is_neumann:
                data = compute_fun(
                    self._field,
                    "B_coil",
                    field_params,
                    constants["source_transforms"],
                    constants["profiles"],
                    data=data,
                    B_coil_chunk_size=self._B_coil_chunk_size,
                    B_coil=self._B_coil,
                    field_grid=self._coil_grid,
                )
                data["B0*n"] += dot(data["B_coil"], data["n_rho"])

            outer["Phi_mn"] = compute_fun(
                self._field,
                "Phi_mn",
                field_params,
                constants["source_transforms"],
                constants["profiles"],
                data=data,
                xtol=self._xtol,
                maxiter=self._maxiter,
                chunk_size=self._chunk_size,
                B_coil_chunk_size=self._B_coil_chunk_size,
                B_coil=self._B_coil,
                field_grid=self._coil_grid,
                **problem,
            )["Phi_mn"]

        outer = compute_fun(
            self._field,
            (
                ["K_vc", "n_rho x B_coil", "K_vc (periodic)"]
                if self._is_neumann
                else "|K_vc|^2"
            ),
            field_params,
            constants["eval_transforms"],
            constants["profiles"],
            data=outer,
            xtol=self._xtol,
            maxiter=self._maxiter,
            chunk_size=self._chunk_size,
            B_coil_chunk_size=self._B_coil_chunk_size,
            B_coil=self._B_coil,
            field_grid=self._coil_grid,
            **problem,
        )

        outer_rpz = compute_fun_rpz(
            self._field,
            ["B"],
            field_params,
            constants["eval_transforms"],
            constants["profiles"],
            data=outer,
            maxiter=self._maxiter,
            chunk_size=self._chunk_size,
            B_coil_chunk_size=self._B_coil_chunk_size,
            B_coil=self._B_coil,
            field_grid=self._coil_grid,
            RpZ_data={"R": outer["R"], "Z": outer["Z"], "phi": outer["phi"]},
            RpZ_grid=constants["eval_transforms"]["grid"],
            eval_interpolator=outer["interpolator"],
            B0=self._field._B0,
            on_boundary=True,
            **problem,
        )

        if self._is_neumann:
            # Compute from equation 4.26 instead of the equation
            # sandwhiched between 4.23 and 4.24 due to discretization error
            # in quadrature messing up the optimizer.
            outer["|B|^2"] = dot(outer_rpz["B"], outer_rpz["B"])

        return (outer["|B|^2"] - inner["|B|^2"] - 2 * mu_0 * inner["p"]) * inner[
            "|e_theta x e_zeta|"
        ]

    @staticmethod
    def _phi_sec_dot_n(params, grads):
        return dot(
            params["I"] * grads["grad(theta)"] + params["Y"] * grads["grad(zeta)"],
            grads["n_rho"],
        )


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
    ):
        if target is None and bounds is None:
            target = 0
        self._mf = mf
        self._nf = nf
        self._ntheta = ntheta
        self._nzeta = nzeta
        self._field = [field] if not isinstance(field, list) else field
        self._field_grid = field_grid
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
        from desc.magnetic_fields import SumMagneticField

        eq = self.things[0]
        self._mf = eq.M + 1 if self._mf is None else self._mf
        self._nf = eq.N if self._nf is None else self._nf
        self._ntheta = 4 * eq.M + 1 if self._ntheta is None else self._ntheta
        self._nzeta = 4 * eq.N + 1 if self._nzeta is None else self._nzeta

        nest = Nestor(
            eq,
            SumMagneticField(self._field),
            self._mf,
            self._nf,
            self._ntheta,
            self._nzeta,
            self._field_grid,
        )
        grid = LinearGrid(rho=1, theta=self._ntheta, zeta=self._nzeta, NFP=eq.NFP)
        self._data_keys = ["current", "|B|^2", "p", "|e_theta x e_zeta|"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
        transforms = get_transforms(self._data_keys, obj=eq, grid=grid)

        self._constants = {
            "transforms": transforms,
            "profiles": profiles,
            "nestor": nest,
            "quad_weights": np.sqrt(transforms["grid"].weights),
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._dim_f = grid.num_nodes

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
