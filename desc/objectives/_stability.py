"""Objectives for targeting MHD stability."""

import os

import numpy as np

from desc.backend import jax, jnp
from desc.basis import DoubleFourierSeries
from desc.compute import get_profiles, get_transforms
from desc.compute.data_index import data_index
from desc.compute.utils import _compute as compute_fun
from desc.grid import Grid, LinearGrid, QuadratureGrid
from desc.utils import ResolutionWarning, Timer, errorif, setdefault, warnif

from .normalization import compute_scaling_factors
from .objective_funs import _Objective, collect_docs
from .utils import _parse_callable_target_bounds

overwrite_stability = {
    "target": """
    target : {float, ndarray, callable}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to ``Objective.dim_f``. If a callable, should take a
        single argument ``rho`` and return the desired value of the profile at those
        locations. Defaults to ``bounds=(0, np.inf)``
    """,
    "bounds": """
    bounds : tuple of {float, ndarray, callable}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to ``Objective.dim_f``
        If a callable, each should take a single argument ``rho`` and return the
        desired bound (lower or upper) of the profile at those locations.
        Defaults to ``bounds=(0, np.inf)``
    """,
}


class MercierStability(_Objective):
    """The Mercier criterion is a fast proxy for MHD stability.

    This makes it a useful figure of merit for stellarator operation.
    Systems with D_Mercier > 0 are favorable for stability.

    See equation 4.16 in
    Landreman, M., & Jorge, R. (2020). Magnetic well and Mercier stability of
    stellarators near the magnetic axis. Journal of Plasma Physics, 86(5), 905860510.
    doi:10.1017/S002237782000121X.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid)``. Note that
        it should have poloidal and toroidal resolution, as flux surface averages
        are required.

    """

    __doc__ = __doc__.rstrip() + collect_docs(overwrite=overwrite_stability)

    _coordinates = "r"
    _units = "(Wb^-2)"
    _print_value_fmt = "Mercier Stability: "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="Mercier Stability",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            bounds = (0, np.inf)
        self._grid = grid
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
        if self._grid is None:
            grid = LinearGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
            )
        else:
            grid = self._grid

        warnif(
            (grid.num_theta * (1 + eq.sym)) < 2 * eq.M,
            ResolutionWarning,
            "MercierStability objective grid requires poloidal "
            "resolution for surface averages",
        )
        warnif(
            grid.num_zeta < 2 * eq.N,
            ResolutionWarning,
            "MercierStability objective grid requires toroidal "
            "resolution for surface averages",
        )
        errorif(
            grid.axis.size,
            ValueError,
            "MercierStability objective grid should not contain axis, "
            "as its on-axis limit does not exist",
        )

        self._target, self._bounds = _parse_callable_target_bounds(
            self._target, self._bounds, grid.nodes[grid.unique_rho_idx]
        )

        self._dim_f = grid.num_rho
        self._data_keys = ["D_Mercier"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
        transforms = get_transforms(self._data_keys, obj=eq, grid=grid)
        self._constants = {
            "transforms": transforms,
            "profiles": profiles,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = 1 / scales["Psi"] ** 2

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute the Mercier stability criterion.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants. (Deprecated)

        Returns
        -------
        D_Mercier : ndarray
            Mercier stability criterion.

        """
        constants = self._get_deprecated_constants(constants)
        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
        )
        return constants["transforms"]["grid"].compress(data["D_Mercier"])


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
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, axis=False)``.
        Note that it should have poloidal and toroidal resolution, as flux surface
        averages are required, and on-axis magnetic well is always zero, so
        it is not necessary to include a point on-axis.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        overwrite=overwrite_stability,
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
    )

    _coordinates = "r"
    _units = "(dimensionless)"
    _print_value_fmt = "Magnetic Well: "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="Magnetic Well",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            bounds = (0, np.inf)
        self._grid = grid
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
        if self._grid is None:
            grid = LinearGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
            )
        else:
            grid = self._grid

        warnif(
            (grid.num_theta * (1 + eq.sym)) < 2 * eq.M,
            ResolutionWarning,
            "MagneticWell objective grid requires poloidal "
            "resolution for surface averages",
        )
        warnif(
            grid.num_zeta < 2 * eq.N,
            ResolutionWarning,
            "MagneticWell objective grid requires toroidal "
            "resolution for surface averages",
        )

        self._target, self._bounds = _parse_callable_target_bounds(
            self._target, self._bounds, grid.nodes[grid.unique_rho_idx]
        )

        self._dim_f = grid.num_rho
        self._data_keys = ["magnetic well"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        profiles = get_profiles(
            self._data_keys, obj=eq, grid=grid, has_axis=grid.axis.size
        )
        transforms = get_transforms(
            self._data_keys, obj=eq, grid=grid, has_axis=grid.axis.size
        )
        self._constants = {
            "transforms": transforms,
            "profiles": profiles,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute a magnetic well parameter.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants. (Deprecated)

        Returns
        -------
        magnetic_well : ndarray
            Magnetic well parameter.

        """
        constants = self._get_deprecated_constants(constants)
        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
        )
        return constants["transforms"]["grid"].compress(data["magnetic well"])


class BallooningStability(_Objective):
    """A type of ideal MHD instability.

    Infinite-n ideal MHD ballooning modes are of significant interest.
    These instabilities are also related to smaller-scale kinetic instabilities.
    With this class, we optimize MHD equilibria against the ideal ballooning mode.

    Targets the following metric:

    f = w₀ sum(ReLU(λ-λ₀)) + w₁ max(ReLU(λ-λ₀))

    where λ is the negative squared growth rate for each field line (such that λ>0 is
    unstable), λ₀ is a cutoff, and w₀ and w₁ are weights.

    Parameters
    ----------
    eq : Equilibrium
        ``Equilibrium`` to be optimized.
    rho : float
        Flux surface to optimize on. Instabilities often peak near the middle.
    alpha : float, ndarray
        Field line labels to optimize. Values should be in [0, 2π). Default is
        ``alpha=0`` for axisymmetric equilibria, or 8 field lines linearly spaced
        in [0, π] for non-axisymmetric cases.
    nturns : int
        Number of toroidal transits of a field line to consider. Field line
        will run from -π*``nturns`` to π*``nturns``. Default 3.
    nzetaperturn : int
        Number of points along the field line per toroidal transit. Total number of
        points is ``nturns*nzetaperturn``. Default 100.
    zeta0 : array-like
        Points of vanishing integrated local shear to scan over.
        Default 15 points in [-π/2,π/2].
        The values ``zeta0`` correspond to values of ι ζ₀ and not ζ₀.
    Neigvals : int
        Number of top eigenvalues to select.
        Default is 1.
    diffmat: DiffMat
        DiffMat object.
        Default uses the finite-difference solver in ``ideal ballooning lambda``.
    lambda0 : float
        Threshold for penalizing growth rates in metric above.
    w0, w1 : float
        Weights for sum and max terms in metric above.
    name : str, optional
        Name of the objective function.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
    )

    _static_attrs = _Objective._static_attrs + [
        "_iota_keys",
        "_Neigvals",
        "_diffmat",
        "_nturns",
        "_nzetaperturn",
        "_add_lcfs",
    ]

    _coordinates = "r"
    _units = "~"
    _print_value_fmt = "Ideal ballooning lambda: "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        rho=np.array([0.5]),
        alpha=None,
        nturns=3,
        nzetaperturn=200,
        zeta0=None,
        Neigvals=1,
        diffmat=None,
        lambda0=0.0,
        w0=1.0,
        w1=10.0,
        name="ideal ballooning lambda",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0

        self._nturns = nturns
        self._nzetaperturn = nzetaperturn
        self._Neigvals = Neigvals
        self._diffmat = diffmat
        self._lambda0 = lambda0
        self._w0 = w0
        self._w1 = w1
        self._rho = np.atleast_1d(rho)
        self._add_lcfs = np.all(self._rho < 0.97)
        self._alpha = setdefault(
            alpha,
            (
                jnp.linspace(0, (1 + eq.sym) * jnp.pi, (1 + eq.sym) * 8)
                if eq.N
                else jnp.array([0])
            ),
        )
        self._zeta0 = setdefault(zeta0, jnp.linspace(-0.5 * np.pi, 0.5 * np.pi, 15))

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
        self._iota_keys = ["iota", "iota_r", "shear", "a"]

        eq = self.things[0]
        iota_grid = LinearGrid(
            # to compute length scale quantities correctly
            rho=np.append(self._rho, 1) if self._add_lcfs else self._rho,
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP,
            sym=eq.sym,
        )
        assert not iota_grid.axis.size
        self._dim_f = iota_grid.num_rho - self._add_lcfs
        transforms = get_transforms(self._iota_keys, eq, iota_grid)
        profiles = get_profiles(
            self._iota_keys + ["ideal ballooning lambda"], eq, iota_grid
        )
        self._constants = {
            "diffmat": self._diffmat,
            "lambda0": self._lambda0,
            "w0": self._w0,
            "w1": self._w1,
            "rho": self._rho,
            "alpha": self._alpha,
            "zeta": jnp.linspace(
                -self._nturns * jnp.pi,
                +self._nturns * jnp.pi,
                +self._nturns * self._nzetaperturn,
            ),
            "zeta0": self._zeta0,
            "iota_transforms": transforms,
            "profiles": profiles,
            "quad_weights": 1.0,
        }
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute the ballooning stability growth rate.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, e.g.
            ``Equilibrium.params_dict``.
        constants : dict
            Dictionary of constant data, e.g. transforms, profiles etc.
            Defaults to ``self.constants``. (Deprecated)

        Returns
        -------
        lam : ndarray
            Ideal ballooning growth rate.

        """
        constants = self._get_deprecated_constants(constants)
        eq = self.things[0]
        iota_data = compute_fun(
            eq,
            self._iota_keys,
            params,
            constants["iota_transforms"],
            constants["profiles"],
        )
        iota_grid = constants["iota_transforms"]["grid"]

        def get(key):
            x = iota_grid.compress(iota_data[key])
            return x[:-1] if self._add_lcfs else x

        iota = get("iota")
        # TODO(#1243): Upgrade this to use _map_clebsch_coordinates once
        #  the note in _L_partial_sum method is resolved.
        grid = eq._get_rtz_grid(
            constants["rho"],
            constants["alpha"],
            constants["zeta"],
            coordinates="raz",
            iota=iota,
            params=params,
        )
        data = {
            key: grid.expand(get(key))
            for key in self._iota_keys
            if (key != "iota" and key != "a")
        }
        data["iota"] = grid.expand(iota)
        data["a"] = iota_data["a"]
        data = compute_fun(
            eq,
            ["ideal ballooning lambda"],
            params,
            transforms=get_transforms(
                ["ideal ballooning lambda"],
                eq,
                grid,
                diffmat=constants["diffmat"],
                jitable=True,
            ),
            profiles=constants["profiles"],
            data=data,
            zeta0=constants["zeta0"],
            Neigvals=self._Neigvals,
        )
        lam = data["ideal ballooning lambda"]
        lambda0, w0, w1 = constants["lambda0"], constants["w0"], constants["w1"]
        # shifted ReLU
        lam = (lam - lambda0) * (lam >= lambda0)
        lam = w0 * lam.sum(axis=(-1, -2, -3)) + w1 * lam.max(axis=(-1, -2, -3))
        return lam


class FinitenStability(_Objective):
    """Ideal MHD instability.

    Finite-n ideal MHD modes are of significant interest.
    With this class, we optimize MHD equilibria against the finite-n unstable modes.

    ``compute`` evaluates ``finite-n lambda3 rayleigh``: the Rayleigh quotient
    ``lambda_R = v^T A(p) v / v^T v`` where ``v`` is eigensolved from ``A(p)`` at that
    same ``p``. The eigensolve sits behind a ``custom_vjp`` with zero cotangents,
    so AD reduces the derivative to the Hellmann-Feynman contraction
    ``v^T (dA/dp) v / v^T v``. ``eigensolver`` picks the solve: ``eigsh_callback``
    (dense ARPACK on the host, default), ``jax_lanczos`` (dense shift-invert on
    device) or ``pcg_deflated`` (matrix-free Jacobi-Davidson; with ``coarse_grid``
    and ``AGNI_COARSE_DEFL=1`` its preconditioner is deflated by coarse modes).

    The eigenvector is deliberately NOT cached. ProximalProjection re-solves the
    equilibrium before every objective evaluation; ``L_lmn`` then moves, ``theta``
    moves with it, and a 7e-5 mesh shift already sends the Rayleigh residual to ~4800
    and flips lambda_R's sign.

    Parameters
    ----------
    eq : Equilibrium
        ``Equilibrium`` to be optimized.
    grid : Grid
        PEST grid used for the finite-n operator.
    diffmat: DiffMat
        Differentiation matrices for the PEST grid.
    v_guess : ndarray, optional
        Cached full eigenfunction. Updated by ``update_state``.
    v_fixed : ndarray, optional
        Skip the eigensolve and use this as the eigenvector, so only the
        Rayleigh quotient is evaluated. Valid ONLY when it came from a call at
        the same ``x``; reusing it after the equilibrium moves is silently
        wrong. Not for use by an optimizer or line search. Default None.
    lambda_guess : float, optional
        Eigenvalue estimate. ``sigma = sigma_factor * lambda_guess`` at
        construction; must sit below the spectrum.
    sigma_factor : float
        Multiplier for the fixed shift. Default 1.3.
    adapt : bool
        pcg_deflated only. Each solve stores its lambda and eigenvector; the next
        solve uses ``sigma = sigma_factor * lambda`` and starts from that vector.
        The first solve uses ``sigma_factor * lambda_guess`` and the coarse seed.
    eigensolver : {"eigsh_callback", "jax_lanczos", "pcg_deflated"}, optional
        Eigensolver. None falls back to ``AGNI_EIGENSOLVER``, then eigsh_callback.
    state_solver : {"dense_eigsh", "matfree"}
        Solve used by ``update_state``.
    coarse_grid, coarse_diffmat, coarse_density : optional
        Coarse PEST level whose modes seed and deflate ``pcg_deflated``.
    num_matvecs, k_defl, jd_outer, jd_inner, jd_maxdim, jd_keep, jd_tol, jd_theta_tol
        Solver options forwarded to ``finite-n lambda3 rayleigh`` when not None
        (see its registration); None falls back to the environment, then default.
    lambda0 : float
        Threshold for ``metric="shifted_relu"``.
    w0 : float
        Weight for ``metric="shifted_relu"``.
    metric : {"raw", "shifted_relu"}
        Objective metric. ``"raw"`` returns lambda directly.
    name : str, optional
        Name of the objective function.
    free_boundary : bool, optional
        Compute the vacuum-response operator ``phi_matrix`` from the current
        boundary geometry (differentiably) and forward it to ``"finite-n
        lambda3 rayleigh"``/``"finite-n lambda3"``, activating their
        free-boundary term. Default False (fixed boundary, ``xi^rho=0`` at
        the plasma edge). When True and a ``coarse_grid`` is also given, the
        coarse-space deflation basis is built free-boundary too -- a
        fixed-boundary coarse basis can fail to represent an
        external-kink-type fine eigenmode.
    phi_chunk_size : int, optional
        Chunk size for the singular-integral computation behind
        ``phi_matrix``. Default 1. Unchunked (``None``) fuses the whole
        boundary singular-integral computation into one GPU kernel, which
        can take XLA pathologically long to compile.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
    )

    _static_attrs = _Objective._static_attrs + [
        # Lists of compute-key strings. They must be declared static or the jitted
        # ObjectiveFunction tries to interpret each str as an abstract array.
        "_flux_keys",
        "_zero_d_keys",
        # The DiffMat carries `zernike_penalty_alpha`, which `_get_zernike_penalty`
        # reads as a concrete Python float to decide whether to build the penalty
        # projector at all. If the DiffMat is a traced pytree that read raises
        # TracerBoolConversionError. BallooningStability marks `_diffmat` static
        # for the same reason.
        "_diffmat",
        "_axisym",
        "_n_mode_axisym",
        "_gamma",
        # `_v_guess` and `_lambda_guess` are static, and are therefore INVARIANTS:
        # nothing may ever rebind them after construction. A static attr lives in
        # the jit signature, so rebinding one mints a new treedef and discards the
        # compiled objective. `update_state` used to rebind both on every refresh,
        # i.e. recompile the whole graph once per outer step. Measured at 12x16x8:
        # a repeat objective call costs 0.002 s with `_v_guess=None` and 1.3 s once
        # a real eigenvector is installed -- 650x, on calls that should be free.
        # `_v_guess` now stays None forever and `_lambda_guess` keeps its
        # construction value (it only ever sets the fixed eigsh shift). The live
        # eigenpair lives in `_constants`, which is traced. `_density` is likewise
        # a static ndarray and is safe only because it is never rebound.
        "_v_guess",
        # `_v_fixed` is deliberately absent: it takes a different value every
        # call, so it must stay a dynamic pytree leaf. Marking it static made
        # DESC's generic flatten embed the array as compile-time aux_data -- a
        # recompile per value, with v baked in as an HLO constant. That OOM'd at
        # 32x48x16 matfree asking for 80.46 GiB. This bool is the static flag.
        "_use_v_fixed",
        "_lambda_guess",
        "_state_solver",
        "_sigma_factor",
        # adapt: a flag and a host callback (a function, hashed by identity)
        "_adapt",
        "_store_guess",
        "_eigensolver",
        "_num_matvecs",
        "_k_defl",
        "_jd_outer",
        "_jd_inner",
        "_jd_maxdim",
        "_jd_keep",
        "_jd_tol",
        "_jd_theta_tol",
        "_eigsh_tol",
        "_coupled_rt",
        "_n_rho_coupled",
        "_n_theta_coupled",
        "_incompressible",
        "_density",
        "_metric",
        # 1D rho nodes of each level, as tuples of Python floats. Static because
        # `compute_data` passes them into the compute function, where an ordinary
        # pytree leaf arrives as a tracer and cannot be used to build the
        # prolongation operator.
        #
        # DECLARED HERE, at class level, NOT appended in build(). `_static_attrs`
        # is itself a member of `_static_attrs`, so rebinding it per instance
        # changes that objective's treedef relative to the class default;
        # ProximalProjection flattens and unflattens the objective across its
        # blocked JVP and that mismatch surfaced as
        #   TypeError: No constant handler for type: DynamicJaxprTracer
        # in, after a forward evaluation that had succeeded.
        # Every other _static_attrs in DESC is composed at class definition.
        "_fine_rho1d",
        "_coarse_rho1d",
        # The coarse level's grid, DiffMat and density are CONSTRUCTION-TIME
        # constants, exactly like their fine counterparts `_diffmat`/`_density`
        # above. Declared static for the same reason.
        #
        # Omitting them was the bug behind
        #   TypeError: No constant handler for type: DynamicJaxprTracer
        # . Left as ordinary leaves, the
        # coarse DiffMat is a TRACER inside the compute -- measured, against
        # `transforms['diffmat']` which is concrete -- so it gets closed into the
        # `_v_primal` custom_vjp, lands in that jaxpr's CONSTANTS, and MLIR
        # lowering dies in `ir_constant`. It fails at LOWERING, not tracing,
        # which is why `jax.eval_shape` reported the trace clean.
        "_coarse_grid",
        "_coarse_diffmat",
        "_coarse_density",
        # Free-boundary (phi_matrix) option. `_free_boundary` gates whether
        # `compute_data`/`update_state` build/forward phi_matrix at all.
        # `_phi_chunk_size` is a plain solver knob, same
        # treatment as `_eigsh_tol` etc. above. Everything else here
        # (`_phi_pest_grid`, `_phi_surf_spacing`/`_phi_surf_weights`,
        # `_phi_st`/`_phi_sz`/`_phi_q`, `_phi_interpolator`, and their
        # coarse_ counterparts) is BUILD-TIME-CONSTANT scaffolding from
        # `_build_phi_scaffolding` -- same category as `_diffmat`/
        # `_coarse_grid` above, and for the same reason: a Grid (or anything
        # holding one) stored as an ordinary, non-static attribute becomes a
        # DYNAMIC pytree leaf, so accessing it under jit yields a TRACER
        # rather than the concrete object, and `_BIESTInterpolator`'s own
        # `assert source_grid.nodes[0, 0] == eval_grid.nodes[0, 0]` (a
        # Python-level bool) then raises `TracerBoolConversionError` even
        # though the grid's VALUE never changes. Marking these static makes
        # them compile-time constants instead -- exactly why `_phi_matrix`
        # also reuses `_phi_interpolator` outright (via a prefilled
        # "interpolator_pest" in `data=`) rather than ever re-invoking
        # `get_interpolator`/`_BIESTInterpolator.__init__` under trace at
        # all: the interpolator depends only on this fixed scaffolding, not
        # on the moving boundary, so there is nothing to recompute.
        "_free_boundary",
        "_phi_chunk_size",
        "_phi_pest_grid",
        "_phi_surf_spacing",
        "_phi_surf_weights",
        "_phi_st",
        "_phi_sz",
        "_phi_q",
        "_phi_interpolator",
        # Equilibrium's own Phi_basis, capped to what phi_pest_grid can
        # resolve (min(eq.Phi_basis.M, phi_pest_grid.M), same for N) and
        # passed to "phi_matrix_pest" as an explicit `Phi_basis=` override.
        # A no-op cap whenever the grid already resolves eq.Phi_basis; only
        # engages for a grid coarser than the equilibrium's own (fixed,
        # file-level) Phi_basis resolution -- otherwise
        # `_lsmr_compute_phi_matrix`'s `assert basis.M <= potential_grid.M`
        # fails, since eq.Phi_basis has nothing to do with this grid's
        # resolution.
        "_phi_basis",
        # Quadrature resolution of the singular integral, decoupled from the
        # evaluation grid. `phi_n_theta`/`phi_n_zeta` size the SOURCE grid;
        # the EVAL grid stays the stability boundary shell, so `phi_matrix`
        # keeps its (n_theta*n_zeta)^2 shape and drops into the operator
        # unchanged. None = inherit the level grid (N_source == N_eval, the
        # original behaviour). The source grid, its PEST nodes and its
        # spacing/weights are resolution-only, hence static like the rest of
        # the scaffolding above.
        "_phi_n_theta",
        "_phi_n_zeta",
        "_phi_upscaled",
        "_phi_src_pest_grid",
        "_phi_src_nodes",
        "_phi_src_spacing",
        "_phi_src_weights",
        "_coarse_phi_upscaled",
        "_coarse_phi_src_pest_grid",
        "_coarse_phi_src_nodes",
        "_coarse_phi_src_spacing",
        "_coarse_phi_src_weights",
        "_coarse_phi_pest_grid",
        "_coarse_phi_surf_spacing",
        "_coarse_phi_surf_weights",
        "_coarse_phi_st",
        "_coarse_phi_sz",
        "_coarse_phi_q",
        "_coarse_phi_interpolator",
        "_coarse_phi_basis",
    ]

    _coordinates = "r"
    _units = "~"
    _print_value_fmt = "Finite-n lambda: "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        v_guess=None,
        lambda_guess=None,
        grid=None,
        axisym=None,
        gamma=0.0,
        n_mode_axisym=1,
        incompressible=False,
        density=None,
        diffmat=None,
        coarse_grid=None,
        coarse_diffmat=None,
        coarse_density=None,
        state_solver="dense_eigsh",
        sigma_factor=1.3,
        adapt=False,
        eigensolver=None,
        num_matvecs=None,
        k_defl=None,
        jd_outer=None,
        jd_inner=None,
        jd_maxdim=None,
        jd_keep=None,
        jd_tol=None,
        jd_theta_tol=None,
        eigsh_tol=1e-8,
        coupled_rt=False,
        n_rho_coupled=None,
        n_theta_coupled=None,
        metric="raw",
        lambda0=0.0,
        w0=1.0,
        name="finite-n lambda3 rayleigh",
        jac_chunk_size=None,
        v_fixed=None,
        free_boundary=False,
        phi_chunk_size=1,
        phi_n_theta=None,
        phi_n_zeta=None,
    ):
        if target is None and bounds is None:
            target = 0

        self._axisym = axisym
        self._v_guess = v_guess
        self._v_fixed = None if v_fixed is None else jnp.asarray(v_fixed)
        self._use_v_fixed = v_fixed is not None
        self._lambda_guess = setdefault(lambda_guess, -1e-1)
        self._gamma = gamma
        self._n_mode_axisym = n_mode_axisym
        self._incompressible = incompressible
        self._density = density
        self._diffmat = diffmat
        self._grid = grid
        # OPTIONAL SECOND LEVEL, for coarse-space deflation (AGNI_COARSE_DEFL).
        # A coarse PEST grid + DiffMat whose generalized modes are prolonged to
        # supply the fine solve's seed and deflation basis. Unset -> nothing
        # changes.
        self._coarse_grid = coarse_grid
        self._coarse_diffmat = coarse_diffmat
        self._coarse_density = coarse_density
        # Free boundary: compute phi_matrix (the vacuum-response operator) and
        # forward it to "finite-n lambda3 rayleigh"/"finite-n lambda3". See
        # `_build_phi_scaffolding`/`_phi_matrix`. `_phi_st`/`_phi_sz`/`_phi_q`
        # (and the coarse_ counterparts) are set in `build()`.
        self._free_boundary = free_boundary
        self._phi_chunk_size = phi_chunk_size
        self._phi_n_theta = phi_n_theta
        self._phi_n_zeta = phi_n_zeta
        self._state_solver = state_solver
        self._sigma_factor = sigma_factor
        self._adapt = adapt
        self._store_guess = None
        self._eigensolver = eigensolver
        self._num_matvecs = num_matvecs
        self._k_defl = k_defl
        self._jd_outer = jd_outer
        self._jd_inner = jd_inner
        self._jd_maxdim = jd_maxdim
        self._jd_keep = jd_keep
        self._jd_tol = jd_tol
        self._jd_theta_tol = jd_theta_tol
        self._eigsh_tol = eigsh_tol
        self._coupled_rt = coupled_rt
        self._n_rho_coupled = n_rho_coupled
        self._n_theta_coupled = n_theta_coupled
        self._metric = metric
        self._lambda0 = lambda0
        self._w0 = w0

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
        errorif(
            self._grid is None,
            ValueError,
            "FinitenStability requires a PEST grid. Pass the same source grid used "
            "to build the finite-n lambda3 DiffMat.",
        )
        errorif(
            self._diffmat is None,
            ValueError,
            "FinitenStability requires diffmat for finite-n lambda3 matfree.",
        )

        self._dim_f = 1
        grid_PEST = self._grid
        # Flux functions (coordinates="r"). These are constant on a rho surface, so
        # computing them on a LinearGrid over the PEST rho values and copying them
        # onto the AGNI nodes reproduces what eq.compute(override_grid=True) does
        # for its internal `grid1dr` (equilibrium.py). This is the same pattern
        # BallooningStability uses, and it is required here: the AGNI grid's nodes
        # are traced (they come from map_coordinates(params=params)), so DESC's own
        # override_grid machinery cannot build its internal grids under AD.
        # These live on `self`, not in `_constants`: `_constants` is traversed as a
        # JAX pytree by the jitted ObjectiveFunction, and a list of str in there
        # raises "Error interpreting argument to ... as an abstract array".
        # BallooningStability keeps `_iota_keys` on self for the same reason.
        self._flux_keys = flux_keys = [
            "iota",
            "iota_r",
            "iota_den",
            "iota_den_r",
            "iota_num",
            "iota_num_r",
            "iota_num current",
            "iota_num_r current",
            "iota_num vacuum",
            "iota_num_r vacuum",
            "psi_r",
            "psi_rr",
            "p",
            "p_r",
        ]
        # `a` is 0-D (coordinates=""), NOT a flux function, and it must not ride
        # along on flux_grid. eq.compute(override_grid=True) -- i.e. the dense
        # finite-n lambda3 reference that produces the eigenpair -- computes 0-D
        # quantities on a QuadratureGrid, where `A` is a direct area integral of
        # |e_rho x e_theta|. On any other grid, `_compute_A_of_z` instead takes a
        # boundary line-integral branch that differs by ~3.8% for this QH case.
        # `a` is the whole non-dimensionalization (B_N = |Psi|/(pi a^2), and the
        # operator's terms carry a^2, a^3 and a^4), so a mismatched `a` gives the
        # Rayleigh quotient a different operator than the one eigsh diagonalized.
        # QuadratureGrid is built from static resolutions only, so it holds no
        # traced nodes and is safe to use inside AD.
        self._zero_d_keys = zero_d_keys = ["a"]
        quad_grid = QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid, eq.NFP)
        quad_transforms = get_transforms(zero_d_keys, obj=eq, grid=quad_grid)
        quad_profiles = get_profiles(zero_d_keys, eq, quad_grid)

        rho_nodes = np.asarray(grid_PEST.nodes[:, 0])
        rho_unique = np.unique(rho_nodes)
        flux_grid = LinearGrid(
            rho=rho_unique,
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP,
            sym=eq.sym,
        )
        assert not flux_grid.axis.size
        flux_transforms = get_transforms(flux_keys, obj=eq, grid=flux_grid)
        flux_profiles = get_profiles(flux_keys, eq, flux_grid)
        n_rho = grid_PEST.num_rho
        n_theta = grid_PEST.num_theta
        n_zeta = grid_PEST.num_zeta
        PEST_nodes = jnp.reshape(
            grid_PEST.meshgrid_reshape(grid_PEST.nodes, order="rtz"),
            (n_rho * n_theta * n_zeta, 3),
        )
        # rho is invariant under the PEST->DESC map, so the rho compress/expand
        # indices of the mapped grid are exactly those of PEST_nodes. Precomputing
        # them here (concretely, on the reshaped rho-major ordering that the operator
        # actually uses) is what lets the mapped grid be rebuilt from *traced* nodes
        # inside AD -- Grid(jitable=True) accepts them instead of rediscovering them
        # with NumPy. This is the same trick get_rtz_grid uses for BallooningStability.
        rho_PEST = np.asarray(PEST_nodes[:, 0])
        _, unique_rho_idx, inverse_rho_idx = np.unique(
            rho_PEST, return_index=True, return_inverse=True
        )
        # 1D radial nodes for the prolongation operator, kept as a concrete
        # numpy array on `self`. They cannot be recovered inside `compute_data`:
        # the mapped grid's nodes are built from params, and `constants` cross
        # the jit boundary as arguments, so BOTH are traced there. build() is
        # the last place they are unambiguously concrete.
        # Stored as TUPLES OF PYTHON FLOATS and registered STATIC. Two separate
        # trace boundaries have to be crossed and each needs its own measure:
        #
        #  - `self` is a pytree, so an ordinary attribute becomes a traced leaf
        #    when it is flattened for jit. That is why these read as tracers
        #    inside compute_data despite being concrete ndarrays on the object
        # -- the same failure as DiffMat.zernike_penalty_alpha.
        #    `_static_attrs` keeps them out of the flattening.
        #  - kwargs ARRAYS are traced crossing into the compute function
        # ; Python scalars are not. Hence tuples, not ndarrays.
        #
        # 32 and 16 values, so the transport costs nothing.
        self._fine_rho1d = tuple(float(_x) for _x in rho_PEST.reshape(n_rho, -1)[:, 0])
        if not hasattr(self, "_coarse_rho1d"):
            self._coarse_rho1d = None
        v_guess = self._v_guess
        if v_guess is None:
            # adapt: zeros mean "no previous vector", so the coarse seed is used
            v_guess = (np.zeros if self._adapt else np.ones)(
                3 * n_rho * n_theta * n_zeta
            )
        lambda_guess = setdefault(self._lambda_guess, -1e-1)

        # Coarse level, mirroring the fine one. rho is invariant under the
        # PEST->DESC map on either grid, so the same precomputed-index trick makes
        # the coarse mapped grid rebuildable from traced nodes.
        coarse_constants = {}
        if self._coarse_grid is not None:
            cg = self._coarse_grid
            c_nodes = jnp.reshape(
                cg.meshgrid_reshape(cg.nodes, order="rtz"),
                (cg.num_rho * cg.num_theta * cg.num_zeta, 3),
            )
            c_rho = np.asarray(c_nodes[:, 0])
            _, c_uidx, c_iidx = np.unique(c_rho, return_index=True, return_inverse=True)
            self._coarse_rho1d = tuple(
                float(_x) for _x in c_rho.reshape(cg.num_rho, -1)[:, 0]
            )
            # The flux functions are constant on a rho surface, so they need their
            # OWN LinearGrid over the coarse rho values -- the fine one's rho set
            # is different and copying across would be silently wrong.
            c_flux_grid = LinearGrid(
                rho=np.unique(c_rho), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False
            )
            coarse_constants = {
                "coarse_PEST_nodes": c_nodes,
                "coarse_unique_rho_idx": jnp.asarray(c_uidx),
                "coarse_inverse_rho_idx": jnp.asarray(c_iidx),
                "coarse_flux_transforms": get_transforms(
                    flux_keys, obj=eq, grid=c_flux_grid
                ),
                "coarse_flux_profiles": get_profiles(flux_keys, eq, c_flux_grid),
            }
            if self._free_boundary:
                self._build_phi_scaffolding(cg, "coarse_")

        self._constants = {
            "PEST_nodes": PEST_nodes,
            "flux_transforms": flux_transforms,
            "flux_profiles": flux_profiles,
            "quad_transforms": quad_transforms,
            "quad_profiles": quad_profiles,
            "unique_rho_idx": jnp.asarray(unique_rho_idx),
            "inverse_rho_idx": jnp.asarray(inverse_rho_idx),
            "quad_weights": 1.0,
            "lambda0": self._lambda0,
            "w0": self._w0,
            "v_guess": jnp.asarray(v_guess).reshape(-1),
            "lambda_guess": jnp.asarray(lambda_guess),
            **coarse_constants,
        }
        if self._free_boundary:
            self._build_phi_scaffolding(grid_PEST, "")
        if self._adapt:
            # Called from inside the jitted objective with each solve's result.
            # It writes into THIS object's `_constants`, which the next call reads
            # as traced values, so nothing recompiles.
            def _store_guess(lam, v):
                lam, v = np.asarray(lam).reshape(-1), np.asarray(v).reshape(-1)
                c = self._constants
                if lam.size == 1 and v.shape == c["v_guess"].shape:
                    if np.isfinite(lam[0]) and np.all(np.isfinite(v)):
                        c["lambda_guess"] = np.asarray(lam[0], dtype=float)
                        c["v_guess"] = v.astype(c["v_guess"].dtype)

            self._store_guess = _store_guess
        super().build(use_jit=use_jit, verbose=verbose)

    def _build_phi_scaffolding(self, level_grid, pre):
        """Static, resolution-only free-boundary scaffolding for one level.

        ``pre`` is ``""`` for fine, ``"coarse_"`` for coarse, matching the
        existing fine/coarse naming convention (``_mapped_grid``,
        ``_flux_data``). Builds the PEST-space grid `phi_matrix` is evaluated
        on, its spacing/weights (reordered to the BIEST convention), and the
        singular-integral INTERPOLATOR itself (frozen -- see below).

        Stores everything as plain, STATIC attributes on ``self`` (added to
        ``_static_attrs``), not in ``self._constants``: `_constants` values
        are DYNAMIC pytree leaves, so accessing e.g. a `Grid` stored there
        from inside a jitted `compute_data` yields a TRACER standing in for
        it, not the concrete object -- even though its value never changes.
        `_BIESTInterpolator.__init__` does a Python-level
        ``assert source_grid.nodes[0, 0] == eval_grid.nodes[0, 0]``, which
        raises ``TracerBoolConversionError`` on such a tracer. Static attrs
        are compile-time constants instead, so this never happens.

        This also means the interpolator can, and must, be reused outright
        rather than rebuilt under trace: `get_interpolator`/
        `_BIESTInterpolator.__init__` depend only on the grids and (st, sz,
        q) -- never on the moving boundary geometry (`source_data` is read
        only by the `_best_params`/`_best_ratio` heuristic that picks
        (st, sz, q) when they are not given, which we bypass here by fixing
        them once). So `_phi_matrix` never calls `get_interpolator` again --
        it prefills the already-built interpolator into `data=` and DESC's
        dependency resolution skips recomputing "interpolator_pest"
        entirely, sidestepping both the tracer-unsafe comparison above and
        the wasted recomputation. This is the one deliberate approximation
        in an otherwise fully differentiable ``phi_matrix``: the quadrature
        *topology* doesn't get its own gradient, only the boundary data
        being integrated does -- exactly the tradeoff
        ``BoundaryError``/``FreeSurfaceError``
        (``desc/objectives/_free_boundary.py``) already make for their own
        frozen interpolators.
        """
        eq = self.things[0]
        n_theta, n_zeta = level_grid.num_theta, level_grid.num_zeta
        n_surf = n_theta * n_zeta
        surf_grid_NFP = 128 if self._axisym else level_grid.NFP
        phi_pest_grid = LinearGrid(
            rho=1.0,
            theta=level_grid.unique_theta,
            zeta=level_grid.unique_zeta,
            NFP=surf_grid_NFP,
            sym=False,
        )
        setattr(self, f"_{pre}phi_pest_grid", phi_pest_grid)
        # Cap the equilibrium's own (fixed, file-level) Phi_basis to what
        # this level's grid can actually resolve. `eq.Phi_basis` (set when
        # the equilibrium's boundary was wrapped in a SourceFreeField, e.g.
        # `SourceFreeField(surface, M=..., N=...)`) has nothing to do with
        # `grid_PEST`'s own resolution, and if it needs more theta/zeta
        # samples than this (possibly deliberately coarse) AGNI grid
        # provides, `_lsmr_compute_phi_matrix`'s
        # `assert basis.M <= potential_grid.M` fails. `min(...)` only ever
        # REDUCES resolution relative to what was asked for, so this is a
        # no-op whenever the grid already resolves eq.Phi_basis fine.
        setattr(
            self,
            f"_{pre}phi_basis",
            DoubleFourierSeries(
                M=phi_pest_grid.M,  # min(eq_phi_basis.M, phi_pest_grid.M),
                N=phi_pest_grid.N,  # min(eq_phi_basis.N, phi_pest_grid.N),
                NFP=phi_pest_grid.NFP,
                sym=phi_pest_grid.sym,
            ),
        )
        # AGNI (theta outer, zeta fastest) -> BIEST (zeta outer, theta
        # fastest). Reuses LinearGrid's own already-correct spacing/weights
        # instead of re-deriving them for the traced surf_grid built by
        # `_phi_matrix` on every call. Kept as plain NumPy (not jnp): these
        # are static attrs, and a `jax.Array` stored there gets silently
        # converted (with a warning) by the pytree machinery anyway.
        setattr(
            self,
            f"_{pre}phi_surf_spacing",
            np.asarray(phi_pest_grid.spacing)
            .reshape(n_theta, n_zeta, 3)
            .transpose(1, 0, 2)
            .reshape(n_surf, 3),
        )
        setattr(
            self,
            f"_{pre}phi_surf_weights",
            np.asarray(phi_pest_grid.weights)
            .reshape(n_theta, n_zeta)
            .transpose(1, 0)
            .reshape(n_surf),
        )

        # SOURCE grid: where the singular integral is actually quadratured.
        # Defaults to the eval grid (`phi_pest_grid`, the stability boundary
        # shell), which is the original N_source == N_eval behaviour. Raising it
        # resolves the integral -- which is what makes the discrete operator
        # self-adjoint in the surface measure, and hence the vacuum term a
        # proper energy -- WITHOUT touching the mode count or the size of the
        # returned matrix: `_lsmr_compute_phi_matrix` returns (N_eval, N_eval)
        # either way, and `basis` is still capped by the eval grid above.
        n_theta_src = int(setdefault(self._phi_n_theta, n_theta))
        n_zeta_src = int(setdefault(self._phi_n_zeta, n_zeta))
        upscaled = (n_theta_src != n_theta) or (n_zeta_src != n_zeta)
        setattr(self, f"_{pre}phi_upscaled", upscaled)
        n_surf_src = n_theta_src * n_zeta_src

        if upscaled:
            # Span the SAME angles as the eval grid, just sampled more finely.
            zeta_eval = np.asarray(phi_pest_grid.nodes[:, 2])
            zeta_span = (
                n_zeta * float(np.diff(np.unique(zeta_eval))[0])
                if n_zeta > 1
                else 2 * np.pi / max(int(surf_grid_NFP), 1)
            )
            src_pest_grid = LinearGrid(
                rho=1.0,
                theta=np.linspace(0.0, 2 * np.pi, n_theta_src, endpoint=False),
                zeta=np.linspace(0.0, zeta_span, n_zeta_src, endpoint=False),
                NFP=surf_grid_NFP,
                sym=False,
            )
            setattr(self, f"_{pre}phi_src_pest_grid", src_pest_grid)
            setattr(
                self,
                f"_{pre}phi_src_spacing",
                np.asarray(src_pest_grid.spacing)
                .reshape(n_theta_src, n_zeta_src, 3)
                .transpose(1, 0, 2)
                .reshape(n_surf_src, 3),
            )
            setattr(
                self,
                f"_{pre}phi_src_weights",
                np.asarray(src_pest_grid.weights)
                .reshape(n_theta_src, n_zeta_src)
                .transpose(1, 0)
                .reshape(n_surf_src),
            )
        else:
            src_pest_grid = phi_pest_grid
            for _a in ("phi_src_pest_grid", "phi_src_spacing", "phi_src_weights"):
                setattr(self, f"_{pre}{_a}", None)

        # One-time, EAGER, concrete build of the interpolator (picks (st, sz,
        # q) itself via the default heuristic, since we have real geometry
        # data to base it on here -- unlike inside a traced `_phi_matrix`
        # call). Reused as-is by every later `_phi_matrix` call, at any
        # params: see the docstring above for why that is exact, not an
        # approximation of convenience.
        #
        # Built on the SOURCE grid, with the eval grid handed over as
        # `potential_grid`. When the two differ, `_interpolator_pest` also
        # rfft-interpolates the boundary geometry onto the eval grid and
        # publishes it as `data["potential data"]`. Expect
        # `singularities.py`'s "Frequency spectrum of FFT interpolation will be
        # truncated" warning once N_eval < N_source//2 + 1: that is the
        # intended regime here (the eval grid only ever needs its own
        # resolution), not a defect.
        nodes0 = np.reshape(
            np.asarray(src_pest_grid.meshgrid_reshape(src_pest_grid.nodes, "rtz")),
            (n_surf_src, 3),
        )
        setattr(self, f"_{pre}phi_src_nodes", nodes0 if upscaled else None)
        rtz0 = np.asarray(
            eq.map_coordinates(
                nodes0,
                inbasis=("rho", "theta_PEST", "zeta"),
                outbasis=("rho", "theta", "zeta"),
                period=(np.inf, 2 * np.pi, np.inf),
                tol=1e-12,
                maxiter=50,
                params=eq.params_dict,
            )
        )
        surf_nodes0 = rtz0.reshape(n_theta_src, n_zeta_src, 3).transpose(1, 0, 2)
        surf_grid0 = Grid(surf_nodes0.reshape(n_surf_src, 3), NFP=surf_grid_NFP)
        interp0 = eq.compute(
            ["interpolator_pest"],
            grid=surf_grid0,
            pest_grid=src_pest_grid,
            potential_grid=phi_pest_grid,
            problem="exterior Neumann",
            chunk_size=self._phi_chunk_size,
            params=eq.params_dict,
        )["interpolator_pest"]
        setattr(self, f"_{pre}phi_st", int(interp0.st))
        setattr(self, f"_{pre}phi_sz", int(interp0.sz))
        setattr(self, f"_{pre}phi_q", int(interp0.q))
        setattr(self, f"_{pre}phi_interpolator", interp0)

    def _phi_matrix(self, params, grid, level="fine"):
        """Free-boundary vacuum-response operator, differentiable in params.

        Rebuilt fresh from the CURRENT boundary geometry on every call
        (unlike the scaffolding from ``_build_phi_scaffolding``, all of
        which is fixed), by slicing the boundary (rho=1) shell out of the
        already-mapped ``grid`` rather than a second ``map_coordinates``
        call. ``level`` selects fine vs. coarse scaffolding/resolution.
        """
        eq = self.things[0]
        pre = "" if level == "fine" else "coarse_"
        level_grid = self._grid if level == "fine" else self._coarse_grid
        n_theta, n_zeta = level_grid.num_theta, level_grid.num_zeta
        n_surf = n_theta * n_zeta

        # DIAGNOSTIC ONLY: replace the vacuum response with c*I. Returns before any
        # mapping or eq.compute, so the real phi_matrix is not built at all.
        #
        # A diagonal is permutation-invariant (Pi^T (c I) Pi == c I),
        # so this is completely insensitive to any BIEST<->AGNI ordering or index
        # misalignment. And because the diagonal measures commute, the vacuum term
        # collapses to a LOCAL nodal operator on the boundary shell,
        #     A_vac = -c * L^H diag(m1*m2) L,
        # with m1 = W_surf*psi_r**3 > 0, m2 = psi_r/|e_theta x e_zeta| > 0. So if the
        # grid-scale oscillation survives this, it is not coming from the vacuum
        # RESPONSE -- look at L_bp/W_surf/b_idx and the boundary DOF gating instead.
        #
        # SIGN: xi^H A_vac xi = -c * ||(m1 m2)^(1/2) L xi||^2. So c > 0 is
        # DESTABILIZING (lambda -> -inf); the penalty that drives
        # B_p.grad xi^rho -> 0 on the boundary, i.e. the fixed-boundary limit, is
        # c < 0. `[vacuum energy]` must come out POSITIVE for the penalizing sign.
        #
        # The penalty forces L xi^rho -> 0, i.e. xi^rho constant along field lines on
        # the boundary (so constant, for irrational lines) -- not xi^rho = 0. It
        # approaches fixed boundary up to a uniform xi^rho on that shell.
        _phi_diag = os.environ.get("AGNI_PHI_DIAG")
        if _phi_diag is not None:
            _c = float(_phi_diag)
            print(
                f"[phi] *** DIAGNOSTIC: phi_matrix = {_c:g} * I "
                f"({n_surf}x{n_surf}), level={level}. NOT A PHYSICAL RUN. ***",
                flush=True,
            )
            return _c * jnp.eye(n_surf)

        phi_pest_grid = getattr(self, f"_{pre}phi_pest_grid")
        upscaled = bool(getattr(self, f"_{pre}phi_upscaled", False))

        if upscaled:
            # The quadrature grid is finer than the stability boundary, so its
            # nodes are not in `grid` and have to be mapped themselves. Same
            # pattern as `_mapped_grid`: traced nodes, still differentiable in
            # `params`, just on a surface instead of a volume.
            src_pest_grid = getattr(self, f"_{pre}phi_src_pest_grid")
            n_theta_s = src_pest_grid.num_theta
            n_zeta_s = src_pest_grid.num_zeta
            rtz = eq.map_coordinates(
                jnp.asarray(getattr(self, f"_{pre}phi_src_nodes")),
                inbasis=("rho", "theta_PEST", "zeta"),
                outbasis=("rho", "theta", "zeta"),
                period=(jnp.inf, 2 * jnp.pi, jnp.inf),
                tol=1e-12,
                maxiter=50,
                params=params,
            )
            surf_nodes = jnp.transpose(
                rtz.reshape(n_theta_s, n_zeta_s, 3), (1, 0, 2)
            ).reshape(n_theta_s * n_zeta_s, 3)
            spacing = getattr(self, f"_{pre}phi_src_spacing")
            weights = getattr(self, f"_{pre}phi_src_weights")
        else:
            src_pest_grid = phi_pest_grid
            n_theta_s, n_zeta_s = n_theta, n_zeta
            bnd_nodes = grid.nodes[-n_surf:]  # AGNI order, rho=1 shell
            surf_nodes = jnp.transpose(
                bnd_nodes.reshape(n_theta, n_zeta, 3), (1, 0, 2)
            ).reshape(
                n_surf, 3
            )  # -> BIEST order (zeta outer, theta fastest)
            spacing = getattr(self, f"_{pre}phi_surf_spacing")
            weights = getattr(self, f"_{pre}phi_surf_weights")

        n_surf_s = n_theta_s * n_zeta_s
        surf_grid = Grid(
            nodes=surf_nodes,
            jitable=True,
            is_meshgrid=True,
            spacing=jnp.asarray(spacing),
            weights=jnp.asarray(weights),
            NFP=phi_pest_grid.NFP,
            _unique_rho_idx=jnp.array([0]),
            _unique_poloidal_idx=jnp.arange(n_theta_s),
            _unique_zeta_idx=jnp.arange(n_zeta_s) * n_theta_s,
            _inverse_rho_idx=jnp.zeros(n_surf_s, dtype=int),
            _inverse_poloidal_idx=jnp.tile(jnp.arange(n_theta_s), n_zeta_s),
            _inverse_zeta_idx=jnp.repeat(jnp.arange(n_zeta_s), n_theta_s),
        )

        # Prefill the frozen interpolator so DESC's dependency resolution
        # skips recomputing "interpolator_pest" entirely -- see
        # `_build_phi_scaffolding`'s docstring for why that is exact, not an
        # approximation, and why rebuilding it here would be tracer-unsafe.
        # `pest_grid`/`potential_grid` MUST match what `_build_phi_scaffolding` passed
        # when it built the interpolator being prefilled below, or the three disagree
        # about which grid is which. `pest_grid` is the SOURCE side -- it also selects
        # the grid for transforms["Phi_PEST"], whose Vandermonde
        # `_lsmr_compute_phi_matrix` uses as `Phi_src`, shape (N_source, N_modes), and
        # assigns to `source_data["B0*n"]`. Passing the eval grid here made that
        # eval-sized, and left "potential data pest" on the source grid via its
        # equal-grid branch. Both only bite when the quadrature is refined
        # (`upscaled`); with source == eval the two grids are the same object and
        # nothing distinguishes them.
        #
        # The Phi basis resolution is NOT affected: `_{pre}phi_basis` was built from
        # `phi_pest_grid.M/N`, so `assert basis.M <= eval_grid.M` still holds.
        data_phi = eq.compute(
            ["phi_matrix_pest"],
            grid=surf_grid,
            pest_grid=src_pest_grid,
            potential_grid=phi_pest_grid,
            problem="exterior Neumann",
            chunk_size=self._phi_chunk_size,
            Phi_basis=getattr(self, f"_{pre}phi_basis"),
            data={"interpolator_pest": getattr(self, f"_{pre}phi_interpolator")},
            params=params,
        )
        phi_matrix = data_phi["phi_matrix_pest"]
        # BIEST -> AGNI ordering, on BOTH axes, to match that level's own
        # boundary-shell node order (what _agni3_assemble/
        # _agni3_matfree_operator expect for the (n_per_shell, n_per_shell)
        # phi_matrix block).
        return jnp.transpose(
            phi_matrix.reshape(n_zeta, n_theta, n_zeta, n_theta), (1, 0, 3, 2)
        ).reshape(n_surf, n_surf)

    def _mapped_grid(self, params, constants, level="fine"):
        """Map the PEST nodes to DESC coordinates at THESE parameters.

        The PEST grid is fixed in PEST coordinates, but the DESC coordinates of
        those nodes move as the equilibrium changes, so this must be rebuilt on
        every call -- including inside AD. Caching it would freeze the nodes and
        silently drop the node-motion contribution to dlambda/dp.

        Two things make that possible under a trace, both taken from
        ``get_rtz_grid`` (the ``BallooningStability`` path):

        - ``jitable=True``: skips ``_find_axis``/``_find_unique_inverse_nodes``,
          which call into NumPy and cannot see traced nodes. This is what raised
          ``ConcretizationTypeError``/``TracerArrayConversionError`` when this was
          a bare ``Grid(DESC_nodes)``.
        - supplying the rho indices: rho is invariant under the PEST->DESC map, so
          the mapped grid's rho compress/expand indices are just the PEST grid's,
          precomputed concretely in ``build``. The grid never has to rediscover
          them from traced values.
        """
        eq = self.things[0]
        pre = "" if level == "fine" else "coarse_"
        errorif(
            level != "fine" and (pre + "PEST_nodes") not in constants,
            ValueError,
            "_mapped_grid(level='coarse') needs a coarse_grid at construction.",
        )
        DESC_nodes = eq.map_coordinates(
            constants[pre + "PEST_nodes"],  # (ρ,θ_PEST,ζ)
            inbasis=("rho", "theta_PEST", "zeta"),
            outbasis=("rho", "theta", "zeta"),
            period=(jnp.inf, 2 * jnp.pi, jnp.inf),
            tol=1e-12,
            maxiter=50,
            params=params,
        )
        return Grid(
            nodes=DESC_nodes,
            coordinates="rtz",
            sort=False,
            jitable=True,
            _unique_rho_idx=constants[pre + "unique_rho_idx"],
            _inverse_rho_idx=constants[pre + "inverse_rho_idx"],
        )

    def _flux_data(self, params, constants, grid, level="fine"):
        """Prefill the quantities that must not be computed on the AGNI grid.

        Two distinct kinds, on two distinct grids, matching what
        ``eq.compute(override_grid=True)`` does for the dense reference:

        - 0-D (``a``) on a ``QuadratureGrid``.
        - 1-D flux functions on a ``LinearGrid`` over the PEST rho values, copied
          onto the AGNI nodes.

        Everything else is a pointwise profile evaluation (``ne``, ``Ti``, ``rho``,
        ...) which is grid-insensitive and correct on the AGNI grid directly.
        """
        eq = self.things[0]
        # The 0-D quantities live on a QuadratureGrid and are grid-independent, so
        # both levels share them. The 1-D flux functions are constant on a rho
        # surface and must come from THIS level's rho set -- the coarse grid has
        # different rho values, so reusing the fine transforms would be silently
        # wrong rather than an error.
        _pre = "" if level == "fine" else "coarse_"
        # 0-D first, then seed it into the flux compute, mirroring the
        # `data=data1dr_seed | data0d_seed` ordering in Equilibrium.compute.
        zero_d_data = compute_fun(
            eq,
            self._zero_d_keys,
            params=params,
            transforms=constants["quad_transforms"],
            profiles=constants["quad_profiles"],
        )
        data = {key: jnp.asarray(zero_d_data[key]) for key in self._zero_d_keys}

        flux_data = compute_fun(
            eq,
            self._flux_keys,
            params=params,
            transforms=constants[_pre + "flux_transforms"],
            profiles=constants[_pre + "flux_profiles"],
            data=dict(data),
        )
        flux_grid = constants[_pre + "flux_transforms"]["grid"]
        for key in self._flux_keys:
            data[key] = grid.copy_data_from_other(
                jnp.asarray(flux_data[key]), flux_grid, surface_label="rho"
            )
        return data

    def compute_data(self, params, constants=None, solve=False):
        """Evaluate ``finite-n lambda3 rayleigh`` at ``params``.

        Every call eigensolves ``A(params)`` with the configured ``eigensolver``
        (dense ARPACK, dense shift-invert Lanczos, or matrix-free Jacobi-Davidson)
        and returns the Rayleigh quotient with that eigenvector held fixed for AD.
        ``solve`` is accepted only for backwards compatibility and must be False.
        """
        errorif(
            solve,
            ValueError,
            "FinitenStability.compute_data(solve=True) is not supported; every call "
            "already eigensolves at params.",
        )
        constants = self._constants if constants is None else constants
        eq = self.things[0]

        grid = self._mapped_grid(params, constants)

        # COARSE-SPACE DEFLATION (AGNI_COARSE_DEFL=1 and a coarse_grid supplied).
        #
        # The coarse level is rebuilt at THIS evaluation's parameters, because H_c
        # is assembled from the equilibrium geometry and a space built at an
        # earlier boundary is invalid once the boundary moves.
        #
        # stop_gradient is not an approximation here. Z and v0 only affect how
        # fast the fine eigenvector is found, never its value at the primal point,
        # so they carry no derivative -- the same reason the eigensolve itself sits
        # outside AD behind custom_vjp. dlambda/dp flows solely through the FINE
        # matrix-free contraction v'Ax(v)/v'v with v frozen.
        coarse_opts = {}
        # Skipped under v_fixed: the coarse space is consumed only by
        # `_eigensolve_pcg`'s deflation, which v_fixed bypasses. Building it
        # anyway was not merely wasted -- with no custom_vjp boundary around v to
        # stop XLA tracing across it, it OOM'd at 80.46 GiB on 32x48x16 matfree.
        if (
            not self._use_v_fixed
            and self._coarse_grid is not None
            and os.environ.get("AGNI_COARSE_DEFL", "0").lower()
            not in {"0", "false", "no", ""}
        ):
            _pc = jax.lax.stop_gradient(params)
            _grid_c = self._mapped_grid(_pc, constants, level="coarse")
            # The coarse operator needs the same GEOMETRY quantities the fine one
            # does (`sqrt(g)_PEST`, the metric components, ...), evaluated on the
            # COARSE grid. `_flux_data` supplies only the 0-D and flux-function
            # prefill; DESC fills the geometry from the key's declared `data=[...]`
            # deps, and it only does that for the grid it was called with. So the
            # coarse level gets its own compute over exactly that dependency list.
            _ckeys = data_index["desc.equilibrium.equilibrium.Equilibrium"][
                "finite-n lambda3 rayleigh"
            ]["dependencies"]["data"]
            _cdata = eq.compute(
                _ckeys,
                grid=_grid_c,
                diffmat=self._coarse_diffmat,
                params=_pc,
                data=self._flux_data(_pc, constants, _grid_c, "coarse"),
                override_grid=False,
            )
            _cg0 = self._coarse_grid
            coarse_opts = {
                "coarse_grid": _grid_c,
                "coarse_diffmat": self._coarse_diffmat,
                "coarse_data": _cdata,
                "coarse_params": _pc,
                # The coupled Zernike operator reshapes by n_rho_coupled /
                # n_theta_coupled. Inheriting the FINE values reshapes the coarse
                # arrays into the fine shape and raises. Taken from the PEST grid,
                # whose counts are concrete, not from the traced mapped grid.
                "coarse_res": (_cg0.num_rho, _cg0.num_theta, _cg0.num_zeta),
                # Radial nodes for the prolongation operator, taken from the
                # PEST nodes rather than from the MAPPED grid. `_mapped_grid`
                # builds its nodes with `eq.map_coordinates(params=...)`, so
                # `grid.nodes` is traced and `np.asarray` on it raises
                # (TracerArrayConversionError float64[6144]).
                # rho is invariant under the PEST->DESC map -- see
                # `_mapped_grid.__doc__`, which relies on the same fact for the
                # rho compress/expand indices -- so these ARE the mapped grid's
                # rho values, just obtained concretely.
                # Already tuples of Python floats, built in `build()`. Do NOT
                # re-convert here: that reintroduces float() on what may be a
                # traced leaf.
                "coarse_rho": self._coarse_rho1d,
                "fine_rho": self._fine_rho1d,
            }
            if self._coarse_density is not None:
                coarse_opts["coarse_density"] = self._coarse_density
            if self._free_boundary:
                # A fixed-boundary coarse deflation basis can fail to
                # represent an external-kink-type free-boundary fine
                # eigenmode at all, so the coarse level needs its own
                # phi_matrix too. Built at the stop_gradient'd _pc, like the
                # rest of this branch: the coarse level is a solver aid and
                # carries no derivative.
                coarse_opts["coarse_phi_matrix"] = self._phi_matrix(
                    _pc, _grid_c, level="coarse"
                )

        options = {
            "axisym": self._axisym,
            "n_mode_axisym": self._n_mode_axisym,
            "gamma": self._gamma,
            "incompressible": self._incompressible,
            "coupled_rt": self._coupled_rt,
            "n_rho_coupled": self._n_rho_coupled,
            "n_theta_coupled": self._n_theta_coupled,
            # No v_guess. `finite-n lambda3 rayleigh` eigensolves A(p) itself, at
            # this p, so the eigenvector is always the primal point's. Caching one
            # here is what made the optimizer minimize a stale-vector quotient:
            # ProximalProjection re-solves the equilibrium before every evaluation,
            # L_lmn moves, theta moves with it, and a 7e-5 mesh shift already sends
            # the Rayleigh residual to ~4800.
            # Fixed shift from the construction-time lambda_guess. Under the
            # matrix-free JD solve it only shifts the preconditioner.
            "sigma": self._sigma_factor * self._lambda_guess,
            "eigsh_tol": self._eigsh_tol,
        }
        if self._adapt:
            _lg = constants["lambda_guess"]
            options["sigma"] = self._sigma_factor * jnp.where(
                _lg < 0, _lg, self._lambda_guess
            )
            options["v_guess"] = constants["v_guess"]

        # Solver options, forwarded as KWARGS -- but ONLY the ones the caller
        # actually set. They used to be stored on `self` and never passed, so the
        # compute function fell through to its environment fallbacks.
        #
        # Forwarding them unconditionally is WRONG even though it looks tidier:
        # an unset parameter would send its own default down, and since the kwarg
        # now beats the environment, that default would silently override every
        # AGNI_* environment variable.
        #
        # None therefore means "not set": fall through to the environment, then
        # to the compute function's own default.
        for _key, _val in (
            ("eigensolver", self._eigensolver),
            ("num_matvecs", self._num_matvecs),
            ("k_defl", self._k_defl),
            ("jd_outer", self._jd_outer),
            ("jd_inner", self._jd_inner),
            ("jd_maxdim", self._jd_maxdim),
            ("jd_keep", self._jd_keep),
            ("jd_tol", self._jd_tol),
            ("jd_theta_tol", self._jd_theta_tol),
        ):
            if _val is not None:
                options[_key] = _val
        # Gated on the static flag, not folded into the loop above: `_v_fixed` is
        # a dynamic leaf, so this fixes the compiled structure once per instance
        # while letting the array's contents change without retracing.
        if self._use_v_fixed:
            options["v_fixed"] = self._v_fixed
        if self._density is not None:
            options["density"] = self._density
        if self._free_boundary:
            options["phi_matrix"] = self._phi_matrix(params, grid)
        options.update(coarse_opts)

        data = eq.compute(
            "finite-n lambda3 rayleigh",
            grid=grid,
            diffmat=self._diffmat,
            params=params,
            data=self._flux_data(params, constants, grid),
            override_grid=False,
            **options,
        )
        if self._adapt and not self._use_v_fixed:
            jax.debug.callback(
                self._store_guess,
                jax.lax.stop_gradient(data["finite-n lambda3 rayleigh"]),
                jax.lax.stop_gradient(data["finite-n eigenfunction3 rayleigh"]),
            )
        return data

    def metric(self, lam, constants=None):
        """Apply the requested scalar metric to the finite-n eigenvalue."""
        constants = self._constants if constants is None else constants
        if self._metric == "raw":
            return lam
        errorif(
            self._metric != "shifted_relu",
            ValueError,
            "Unknown finite-n stability metric: expected 'raw' or 'shifted_relu'.",
        )
        return (
            constants["w0"]
            * (lam - constants["lambda0"])
            * (lam >= constants["lambda0"])
        )

    def update_state(self, params, constants=None):
        """Refresh and cache the finite-n eigenpair.

        Call this before each one-step DESC optimization solve. ``compute`` then
        returns the fixed-mode Rayleigh quotient, so DESC computes the gradient in
        its usual way while the expensive eigensolve stays outside AD.
        ``state_solver="dense_eigsh"`` (default) refreshes with dense
        ``finite-n lambda3`` and SciPy ARPACK ``eigsh``; ``"matfree"`` runs
        ``compute_data`` (the configured eigensolver) and refreshes only
        ``lambda_guess``.
        """
        constants = self._constants if constants is None else constants
        state_solver = str(self._state_solver).lower()
        if state_solver == "matfree":
            # MATRIX-FREE REFRESH. Required above ~32x32x12: the dense branch
            # below assembles the full fine matrix, which is 10.4 GB at
            # 32x32x12 and 53.5 GB at 48x48x12 --
            # both died there, in exactly this call.
            #
            # WHAT THIS GIVES UP, and why it is affordable here:
            #   * `finite-n lambda3` -- there is no dense eigenvalue, so callers
            #     lose the independent ARPACK cross-check. `finite-n lambda3
            #     rayleigh` is returned instead and callers must handle the
            #     absence of the dense key.
            #   * `v_guess` -- the matrix-free path exposes no eigenvector as a
            #     data key, so the cached start vector is LEFT UNCHANGED. That is
            #     harmless with coarse deflation, which overrides the seed with
            #     the prolonged coarse mode (`_seed_v0`) anyway, and merely
            #     un-warm-started without it.
            #
            # `lambda_guess` IS refreshed in `_constants`.
            #
            # Cost: one extra eigensolve per OUTER step -- not per objective
            # call -- so it amortises over the optimizer's inner evaluations.
            # solve=False: `compute_data` rejects solve=True, and it does not
            # need it. With AGNI_EIGENSOLVER=pcg_deflated the eigensolve happens
            # INSIDE the `finite-n lambda3 rayleigh` compute chain on every call,
            # so this already performs a full matrix-free solve.
            data = self.compute_data(params, constants=constants, solve=False)
            lam = jnp.asarray(data["finite-n lambda3 rayleigh"]).reshape(-1)[0]
            if hasattr(self, "_constants"):
                self._constants["lambda_guess"] = lam
            return data
        elif state_solver == "dense_eigsh":
            eq = self.things[0]
            grid = self._mapped_grid(params, constants)
            options = {
                "axisym": self._axisym,
                "n_mode_axisym": self._n_mode_axisym,
                "gamma": self._gamma,
                "incompressible": self._incompressible,
                "coupled_rt": self._coupled_rt,
                "n_rho_coupled": self._n_rho_coupled,
                "n_theta_coupled": self._n_theta_coupled,
                # sigma is a FIXED shift built from the immutable `_lambda_guess`
                # supplied at construction -- never from the last refresh. See the
                # note below on why nothing here may be rebound.
                "sigma": self._sigma_factor * self._lambda_guess,
                # No v_guess: the dense eigsh cold-starts every time. A warm start
                # from the previous step's eigenvector is only as good as that
                # vector is fresh; if a boundary step moved the equilibrium, ARPACK
                # is handed a stale v0 and can grind or converge to a different
                # mode. Cold-starting is slower per solve but cannot mislead.
                "eigsh_tol": self._eigsh_tol,
            }
            if self._density is not None:
                options["density"] = self._density
            if self._free_boundary:
                # Required for correctness, not just consistency: `v` below
                # must come from the same (free- or fixed-boundary) operator
                # "finite-n lambda3 rayleigh" assembles in `compute_data`, or
                # the cached eigenvector no longer matches the matrix being
                # differentiated.
                options["phi_matrix"] = self._phi_matrix(params, grid)
            data = eq.compute(
                "finite-n lambda3",
                grid=grid,
                diffmat=self._diffmat,
                params=params,
                **options,
            )
            lam = jnp.asarray(data["finite-n lambda3"]).reshape(-1)[0]
            v = jnp.asarray(data["finite-n eigenfunction3"]).reshape(-1)
        else:
            raise ValueError(
                "Unknown finite-n state_solver: expected 'matfree' or "
                f"'dense_eigsh', got {self._state_solver!r}."
            )
        # `self._v_guess` and `self._lambda_guess` are declared in `_static_attrs`,
        # so they live in the jit signature. Rebinding a static attr mints a new
        # treedef and throws away the compiled objective -- and update_state runs
        # once per outer step, so rebinding here means recompiling the whole graph
        # every refresh. Measured at 12x16x8: a repeat objective call costs 0.002 s
        # with `_v_guess=None` and 1.3 s once a real eigenvector is installed.
        # They are therefore left at their construction values FOREVER. The refreshed
        # eigenpair goes only into `_constants`, which is traced: writing a new array
        # of the same shape/dtype there changes a value, not a signature, so nothing
        # recompiles. `compute_data` reads `constants["v_guess"]`, never self.
        if hasattr(self, "_constants"):
            self._constants["v_guess"] = v
            self._constants["lambda_guess"] = lam
        return data

    def compute(self, params, constants=None):
        """Compute the finite-n stability eigenvalue.

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
        lam : ndarray
            Finite-n instability eigenvalue.

        """
        constants = self._constants if constants is None else constants
        data = self.compute_data(params, constants=constants, solve=False)
        lam = data["finite-n lambda3 rayleigh"]
        return self.metric(lam, constants=constants)
