"""Objectives for targeting MHD stability."""

import numpy as np

from desc.backend import jnp
from desc.compute import get_profiles, get_transforms
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


def _agni_sigma_shift(obj, constants):
    """Shift-invert sigma for the finite-n eigensolve. Fixed unless tracking is on.

    DEFAULT (AGNI_SIGMA_MODE=fixed): `sigma_factor * self._lambda_guess`, both
    static -- the historical behaviour, byte-for-byte. Nothing below runs.

    TRACKING (AGNI_SIGMA_MODE=track): sigma is re-based on `constants["lambda_guess"]`,
    which `update_state` refreshes every outer step from the TRUSTED dense eigsh
    (see the note at the end of update_state). That entry is a traced array, so
    writing a new value of the same shape/dtype changes a VALUE, not a signature --
    no recompile. This is the mechanism the file already documents and already
    maintains; the only thing that was missing is that sigma read the STATIC
    attribute instead of this traced one.

    WHY tracking is wanted: the near-zero eigenvalue cluster sits at a FIXED
    mu = 1/|sigma|, while the wanted mode's mu = 1/(lambda - sigma) moves as the
    optimizer drives lambda -> 0. With sigma pinned, the wanted mode slides INTO
    the cluster: separation 1.150 at the start, 1.023 at the observed drift point,
    1.0046 at job 56105697. At separation 1.0134 the objective read +2.14e-02 on a
    genuinely UNSTABLE equilibrium. See BENCHMARKING.md section 10.

    SCOPE (BENCHMARKING.md 10.8): this re-bases sigma from the value refreshed once
    per OUTER step. It does NOT track sigma per trial point inside a line search --
    that remains blocked, and would also break objective purity.

    !! FACTOR WARNING !! `_sigma_factor` defaults to 10, which is correct for the
    CONSTRUCTION-time convention (callers pass lambda_guess = lambda/10, so
    10 * lambda/10 = lambda). Once `lambda_guess` holds the TRUE lambda, a factor of
    10 puts sigma at 10x lambda and separation gets WORSE than the fixed default
    (1.111 vs 1.150). BENCHMARKING.md 10.4(b) measured the sweet spot at
    sigma ~ 2-3 x lambda: closer than that and the LU conditioning
    (|lam - sigma| = 1.95e-5 against ||A|| = 2.8e7) costs accuracy instead.
    Set AGNI_SIGMA_FACTOR=2.5 when enabling tracking.
    """
    import os as _os

    import jax as _jax  # not imported at module scope in this file

    # AGNI_SIGMA_MODE=track only. 'fixed' and 'adapt' both leave sigma alone here --
    # 'adapt' re-shifts INSIDE the eigensolve instead, which needs no state.
    # "track" or "track+adapt" both track here; "adapt" alone leaves sigma fixed at
    # this level and does its re-shift inside the eigensolve instead.
    if "track" not in _os.environ.get("AGNI_SIGMA_MODE", "fixed").lower():
        return obj._sigma_factor * obj._lambda_guess

    factor = float(_os.environ.get("AGNI_SIGMA_FACTOR", "2.5"))
    lam_g = None if constants is None else constants.get("lambda_guess", None)
    if lam_g is None:  # pre-first-refresh: nothing to track yet
        return obj._sigma_factor * obj._lambda_guess
    # stop_gradient: sigma is a SOLVER SETTING, not a physical parameter. It comes
    # from the PREVIOUS outer step, so it is constant w.r.t. the current params --
    # but JAX must be told, or a spurious dsigma/dp path appears. Hellmann-Feynman
    # contains no sigma at all, so this cannot bias the gradient's form; a poor
    # sigma degrades the ACCURACY of v, never the formula.
    return factor * _jax.lax.stop_gradient(jnp.asarray(lam_g))


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
    """A type of ideal MHD instability.

    Finite-n ideal MHD ballooning modes are of significant interest.
    With this class, we optimize MHD equilibria against the finite-n unstable modes.

    ``compute`` evaluates ``finite-n lambda3 rayleigh``: the Rayleigh quotient
    ``lambda_R = v^T A(p) v / v^T v`` where ``v`` is eigensolved from ``A(p)`` at that
    same ``p`` (ARPACK on the host, via ``jax.pure_callback``). Because a callback
    output carries no tangent, AD reduces the derivative to the Hellmann-Feynman
    contraction ``v^T (dA/dp) v / v^T v`` automatically.

    The eigenvector is deliberately NOT cached. ProximalProjection re-solves the
    equilibrium before every objective evaluation; ``L_lmn`` then moves, ``theta``
    moves with it, and a 7e-5 mesh shift already sends the Rayleigh residual to ~4800
    and flips lambda_R's sign. See WHY_V_CANNOT_BE_CACHED.md. The matrix-free solver
    is not used anywhere in this objective. Historical note: ``update_state``
    between accepted optimization steps to refresh the cached eigenvalue/eigenfunction
    by running the warm-started matrix-free eigensolver.

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
    lambda_guess : float, optional
        Cached eigenvalue. Updated by ``update_state`` and used to set the
        shift-invert sigma.
    lambda0 : float
        Threshold for ``metric="shifted_relu"``.
    w0 : float
        Weight for ``metric="shifted_relu"``.
    metric : {"raw", "shifted_relu"}
        Objective metric. ``"raw"`` returns lambda directly.
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
        "_lambda_guess",
        "_state_solver",
        "_matfree_solver",
        "_sigma_factor",
        "_num_matvecs",
        "_cg_tol",
        "_cg_maxiter",
        "_eigsh_tol",
        "_coupled_rt",
        "_n_rho_coupled",
        "_n_theta_coupled",
        "_incompressible",
        "_density",
        "_metric",
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
        state_solver="dense_eigsh",
        matfree_solver=None,
        sigma_factor=1.3,
        num_matvecs=64,
        cg_tol=1e-6,
        cg_maxiter=30000,
        eigsh_tol=1e-8,
        coupled_rt=False,
        n_rho_coupled=None,
        n_theta_coupled=None,
        metric="raw",
        lambda0=0.0,
        w0=1.0,
        name="finite-n lambda3 rayleigh",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0

        self._axisym = axisym
        self._v_guess = v_guess
        self._lambda_guess = setdefault(lambda_guess, -1e-1)
        self._gamma = gamma
        self._n_mode_axisym = n_mode_axisym
        self._incompressible = incompressible
        self._density = density
        self._diffmat = diffmat
        self._grid = grid
        self._state_solver = state_solver
        self._matfree_solver = matfree_solver
        self._sigma_factor = sigma_factor
        self._num_matvecs = num_matvecs
        self._cg_tol = cg_tol
        self._cg_maxiter = cg_maxiter
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
        v_guess = self._v_guess
        if v_guess is None:
            v_guess = np.ones(3 * n_rho * n_theta * n_zeta)
        lambda_guess = setdefault(self._lambda_guess, -1e-1)

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
        }
        super().build(use_jit=use_jit, verbose=verbose)

    def _mapped_grid(self, params, constants):
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
        DESC_nodes = eq.map_coordinates(
            constants["PEST_nodes"],  # (ρ,θ_PEST,ζ)
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
            _unique_rho_idx=constants["unique_rho_idx"],
            _inverse_rho_idx=constants["inverse_rho_idx"],
        )

    def _flux_data(self, params, constants, grid):
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
            transforms=constants["flux_transforms"],
            profiles=constants["flux_profiles"],
            data=dict(data),
        )
        flux_grid = constants["flux_transforms"]["grid"]
        for key in self._flux_keys:
            data[key] = grid.copy_data_from_other(
                jnp.asarray(flux_data[key]), flux_grid, surface_label="rho"
            )
        return data

    def compute_data(self, params, constants=None, solve=False):
        """Evaluate the fixed-vector Rayleigh quotient of the finite-n operator.

        ``solve`` is accepted only for backwards compatibility and must be False.
        This objective never solves an eigenproblem here: the eigenpair comes from
        dense ``finite-n lambda3`` + eigsh in ``update_state``, and ``compute``
        differentiates the Rayleigh quotient with that eigenvector held fixed.
        """
        errorif(
            solve,
            ValueError,
            "FinitenStability.compute_data(solve=True) is not supported: it would "
            "run the matrix-free eigensolver. Refresh the eigenpair with "
            "update_state(state_solver='dense_eigsh') instead.",
        )
        constants = self._constants if constants is None else constants
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
            # No v_guess. `finite-n lambda3 rayleigh` eigensolves A(p) itself, at
            # this p, so the eigenvector is always the primal point's. Caching one
            # here is what made the optimizer minimize a stale-vector quotient:
            # ProximalProjection re-solves the equilibrium before every evaluation,
            # L_lmn moves, theta moves with it, and a 7e-5 mesh shift already sends
            # the Rayleigh residual to ~4800. See WHY_V_CANNOT_BE_CACHED.md.
            # Fixed by default; AGNI_SIGMA_MODE=track re-bases it on the traced
            # constants["lambda_guess"]. See _agni_sigma_shift -- including the
            # FACTOR WARNING (set AGNI_SIGMA_FACTOR=2.5 when tracking).
            "sigma": _agni_sigma_shift(self, constants),
            "eigsh_tol": self._eigsh_tol,
        }
        if self._density is not None:
            options["density"] = self._density

        return eq.compute(
            "finite-n lambda3 rayleigh",
            grid=grid,
            diffmat=self._diffmat,
            params=params,
            data=self._flux_data(params, constants, grid),
            override_grid=False,
            **options,
        )

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
        its usual way while the expensive eigensolve stays outside AD. By default
        this preserves the original warm-started matrix-free refresh. If
        ``state_solver="dense_eigsh"``, the cached eigenpair is instead refreshed
        with dense ``finite-n lambda3`` and SciPy ARPACK ``eigsh``.
        """
        constants = self._constants if constants is None else constants
        state_solver = str(self._state_solver).lower()
        if state_solver in {"matfree", "shiftinvert_cg"}:
            raise ValueError(
                "state_solver='matfree' is not supported: it runs the matrix-free "
                "eigensolver. Use state_solver='dense_eigsh'."
            )
        elif state_solver in {"dense", "dense_eigsh", "eigsh"}:
            eq = self.things[0]
            grid = self._mapped_grid(params, constants)
            if bool(__import__("os").environ.get("AGNI_OBJECTIVE_DEBUG", "")):
                _sig = self._sigma_factor * constants.get(
                    "lambda_guess", self._lambda_guess
                )
                print(
                    "[FinitenStability dense_eigsh] "
                    f"sigma={_sig} "
                    f"eigsh_tol={self._eigsh_tol} coupled_rt={self._coupled_rt} "
                    f"n_rho_coupled={self._n_rho_coupled} "
                    f"n_theta_coupled={self._n_theta_coupled} grid=bare",
                    flush=True,
                )
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
