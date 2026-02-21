"""Objectives for neoclassical transport."""

import warnings

import numpy as np
from orthax.legendre import leggauss
from termcolor import colored

from desc.backend import jnp
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import Grid, LinearGrid
from desc.integrals._interp_utils import bijection_from_disc, cheb_pts, fourier_pts
from desc.utils import parse_argname_change, setdefault

from .objective_funs import _Objective, collect_docs
from .utils import _parse_callable_target_bounds

from ..integrals.quad_utils import (
    automorphism_sin,
    chebgauss2,
    get_quadrature,
    grad_automorphism_sin,
)
from desc.utils import Timer

_bounce_overwrite = {
    "deriv_mode": """
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute Jacobian matrix, either forward mode or reverse mode AD.
        ``auto`` selects forward or reverse mode based on the size of the input and
        output of the objective. Has no effect on ``self.grad`` or ``self.hess`` which
        always use reverse mode and forward over reverse mode respectively.

        Unless ``fwd`` is specified, ``jac_chunk_size=1`` is recommended to reduce
        memory consumption. In ``rev`` mode, reducing the pitch angle parameter
        ``pitch_batch_size`` does not reduce memory consumption, so it is recommended
        to retain the default for that.
        """
}


class EffectiveRipple(_Objective):
    """Proxy for neoclassical transport in the banana regime.

    A 3D stellarator magnetic field admits ripple wells that lead to enhanced
    radial drift of trapped particles. In the banana regime, neoclassical (thermal)
    transport from ripple wells can become the dominant transport channel.
    The effective ripple (ε) proxy estimates the neoclassical transport
    coefficients in the banana regime. To ensure low neoclassical transport,
    a stellarator is typically optimized so that ε < 0.02.

    References
    ----------
    https://doi.org/10.1063/1.873749.
    Evaluation of 1/ν neoclassical transport in stellarators.
    V. V. Nemov, S. V. Kasilov, W. Kernbichler, M. F. Heyn.
    Phys. Plasmas 1 December 1999; 6 (12): 4622–4632.

    Notes
    -----
    Performance will improve significantly by resolving these GitHub issues.
      * https://github.com/jax-ml/jax/issues/30627
      * ``1303`` Patch for differentiable code with dynamic shapes
      * ``1206`` Upsample data above midplane to full grid assuming stellarator symmetry
      * ``1034`` Optimizers/objectives with auxiliary output


    Parameters
    ----------
    eq : Equilibrium
        ``Equilibrium`` to be optimized.
    grid : Grid
        Tensor-product grid in (ρ, θ, ζ) with uniformly spaced nodes
        (θ, ζ) ∈ [0, 2π) × [0, 2π/NFP).
        Number of poloidal and toroidal nodes preferably rounded down to powers of two.
        Determines the flux surfaces to compute on and resolution of FFTs.
        Default grid samples the boundary surface at ρ=1.
    X : int
        Poloidal Fourier grid resolution to interpolate the poloidal coordinate.
        Preferably rounded down to power of 2.
    Y : int
        Toroidal Chebyshev grid resolution to interpolate the poloidal coordinate.
        Preferably rounded down to power of 2.
    Y_B : int
        Desired resolution for algorithm to compute bounce points.
        Default is double ``Y``. Something like 100 is usually sufficient.
        Currently, this is the number of knots per toroidal transit over
        to approximate B with cubic splines.
    alpha : np.ndarray
        Shape (num alpha, ).
        Starting field line poloidal labels.
        Default is single field line. To compute a surface average
        on a rational surface, it is necessary to average over multiple
        field lines until the surface is covered sufficiently.
    num_transit : int
        Number of toroidal transits to follow field line.
        In an axisymmetric device, field line integration over a single poloidal
        transit is sufficient to capture a surface average. For a 3D
        configuration, more transits will approximate surface averages on an
        irrational magnetic surface better, with diminishing returns.
    num_well : int
        Maximum number of wells to detect for each pitch and field line.
        Giving ``None`` will detect all wells but due to current limitations in
        JAX this will have worse performance.
        Specifying a number that tightly upper bounds the number of wells will
        increase performance. In general, an upper bound on the number of wells
        per toroidal transit is ``Aι+B`` where ``A``, ``B`` are the poloidal and
        toroidal Fourier resolution of B, respectively, in straight-field line
        PEST coordinates, and ι is the rotational transform normalized by 2π.
        A tighter upper bound than ``num_well=(Aι+B)*num_transit`` is preferable.
        The ``check_points`` or ``plot`` methods in ``desc.integrals.Bounce2D``
        are useful to select a reasonable value.

        This is the most important paramter to specify for performance.
    num_quad : int
        Resolution for quadrature of bounce integrals. Default is 32.
    num_pitch : int
        Resolution for quadrature over velocity coordinate. Default is 51.
    pitch_batch_size : int
        Number of pitch values with which to compute simultaneously.
        If given ``None``, then ``pitch_batch_size`` is ``num_pitch``.
        Default is ``num_pitch``.
    surf_batch_size : int
        Number of flux surfaces with which to compute simultaneously.
        If given ``None``, then ``surf_batch_size`` is ``grid.num_rho``.
        Default is ``1``. Only consider increasing if ``pitch_batch_size`` is ``None``.
    nufft_eps : float
        Precision requested for interpolation with non-uniform fast Fourier
        transform (NUFFT). If less than ``1e-14`` then NUFFT will not be used.
    use_bounce1d : bool
        Set to ``True`` to use ``Bounce1D`` instead of ``Bounce2D``,
        basically replacing some pseudo-spectral methods with local splines.
        This can be efficient if ``num_transit`` and ``alpha.size`` are small,
        depending on hardware and hardware features used by the JIT compiler.
        If ``True``, then parameters ``X``, ``Y``, ``nufft_eps`` are ignored.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
        overwrite=_bounce_overwrite,
    )

    _static_attrs = _Objective._static_attrs + [
        "_hyperparam",
        "_keys_1dr",
        "_use_bounce1d",
    ]

    _coordinates = "r"
    _units = "~"
    _print_value_fmt = "Effective ripple ε: "

    def __init__(
        self,
        eq,
        *,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        jac_chunk_size=None,
        name="Effective ripple",
        grid=None,
        X=16,
        Y=32,
        Y_B=None,
        alpha=np.array([0.0]),
        num_transit=20,
        num_well=None,
        num_quad=32,
        num_pitch=51,
        pitch_batch_size=None,
        surf_batch_size=1,
        nufft_eps=1e-6,
        use_bounce1d=False,
        **kwargs,
    ):
        try:
            import jax_finufft  # noqa: F401
        except ImportError:
            warnings.warn(
                colored(
                    "\njax-finufft is not installed.\n"
                    "Setting parameter nufft_eps to zero.\n"
                    "Performance will deteriorate significantly.\n"
                    "yellow",
                )
            )
            nufft_eps = 0.0

        if target is None and bounds is None:
            target = 0.0

        self._use_bounce1d = parse_argname_change(
            use_bounce1d, kwargs, "spline", "use_bounce1d"
        )
        self._grid = grid
        self._constants = {
            "quad_weights": 1.0,
            "alpha": alpha,
            "X": fourier_pts(X),
            "Y": cheb_pts(Y, (0, 2 * np.pi))[::-1],
        }
        Y_B = setdefault(Y_B, 2 * Y)
        self._hyperparam = {
            "Y_B": Y_B,
            "num_transit": num_transit,
            "num_well": setdefault(num_well, Y_B * num_transit),
            "num_quad": num_quad,
            "num_pitch": num_pitch,
            "pitch_batch_size": pitch_batch_size,
            "surf_batch_size": surf_batch_size,
            "nufft_eps": nufft_eps,
        }

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
        if self._use_bounce1d:
            return self._build_bounce1d(use_jit, verbose)

        eq = self.things[0]
        if self._grid is None:
            self._grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
        assert self._grid.can_fft2

        rho = self._grid.compress(self._grid.nodes[:, 0])
        x, w = leggauss(self._hyperparam["Y_B"] // 2)
        self._constants["_vander"] = _get_vander(self, x)
        self._constants["fieldline quad"] = (x, w)
        self._constants["quad"] = chebgauss2(self._hyperparam.pop("num_quad"))
        self._constants["profiles"] = get_profiles(
            "effective ripple", eq, grid=self._grid
        )
        self._constants["transforms"] = get_transforms(
            "effective ripple", eq, grid=self._grid
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Unequal number of field periods")
            # TODO(#1243): Set grid.sym=eq.sym once basis is padded for partial sum
            self._constants["lambda"] = get_transforms(
                "lambda",
                eq,
                grid=LinearGrid(rho=rho, M=eq.L_basis.M, zeta=self._constants["Y"]),
            )["L"]
        assert self._constants["lambda"].basis.NFP == eq.NFP

        self._dim_f = self._grid.num_rho
        self._target, self._bounds = _parse_callable_target_bounds(
            self._target, self._bounds, rho
        )
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute the effective ripple.

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
        epsilon : ndarray
            Effective ripple as a function of the flux surface label.

        """
        if self._use_bounce1d:
            return self._compute_bounce1d(params, constants)

        if constants is None:
            constants = self.constants
        eq = self.things[0]

        data = compute_fun(
            eq, "iota", params, constants["transforms"], constants["profiles"]
        )
        theta = eq._map_clebsch_coordinates(
            iota=constants["transforms"]["grid"].compress(data["iota"]),
            alpha=constants["X"],
            zeta=constants["Y"],
            L_lmn=params["L_lmn"],
            lmbda=constants["lambda"],
            # TODO (#1034): Use old theta values as initial guess.
            tol=1e-7,
        )[..., ::-1]

        data = compute_fun(
            eq,
            "effective ripple",
            params,
            constants["transforms"],
            constants["profiles"],
            data,
            theta=theta,
            alpha=constants["alpha"],
            fieldline_quad=constants["fieldline quad"],
            quad=constants["quad"],
            _vander=constants["_vander"],
            **self._hyperparam,
        )
        return constants["transforms"]["grid"].compress(data["effective ripple"])

    def _build_bounce1d(self, use_jit=True, verbose=1):
        Y_B = self._hyperparam.pop("Y_B")
        num_transit = self._hyperparam.pop("num_transit")
        num_quad = self._hyperparam.pop("num_quad")
        self._hyperparam.pop("nufft_eps")
        del self._constants["X"]
        self._constants["Y"] = np.linspace(
            0, 2 * np.pi * num_transit, Y_B * num_transit
        )
        self._keys_1dr = [
            "iota",
            "iota_r",
            "<|grad(rho)|>",
            "min_tz |B|",
            "max_tz |B|",
            "R0",
        ]

        eq = self.things[0]
        if self._grid is None:
            self._grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
        assert self._grid.is_meshgrid and eq.sym == self._grid.sym

        rho = self._grid.compress(self._grid.nodes[:, 0])
        self._constants["rho"] = rho
        self._constants["quad"] = chebgauss2(num_quad)
        self._constants["profiles"] = get_profiles(
            self._keys_1dr + ["old effective ripple"], eq, self._grid
        )
        self._constants["transforms_1dr"] = get_transforms(
            self._keys_1dr, eq, self._grid
        )

        self._dim_f = self._grid.num_rho
        self._target, self._bounds = _parse_callable_target_bounds(
            self._target, self._bounds, rho
        )
        super().build(use_jit=use_jit, verbose=verbose)

    def _compute_bounce1d(self, params, constants=None):
        if constants is None:
            constants = self.constants
        eq = self.things[0]

        data = compute_fun(
            eq,
            self._keys_1dr,
            params,
            constants["transforms_1dr"],
            constants["profiles"],
        )
        # TODO(#1243): Upgrade this to use _map_clebsch_coordinates once
        #  the note in _L_partial_sum method is resolved.
        grid = eq._get_rtz_grid(
            constants["rho"],
            constants["alpha"],
            constants["Y"],
            coordinates="raz",
            iota=self._grid.compress(data["iota"]),
            params=params,
        )
        data = {
            key: (
                grid.copy_data_from_other(data[key], self._grid)
                if key != "R0"
                else data[key]
            )
            for key in self._keys_1dr
        }
        data = compute_fun(
            eq,
            "old effective ripple",
            params,
            transforms=get_transforms("old effective ripple", eq, grid, jitable=True),
            profiles=constants["profiles"],
            data=data,
            quad=constants["quad"],
            **self._hyperparam,
        )
        return grid.compress(data["old effective ripple"])


def _get_vander(obj, x):
    eq = obj.things[0]
    Y_B = ((obj._hyperparam["Y_B"] + eq.NFP - 1) // eq.NFP) * eq.NFP
    return {
        "dct cfl": _vander_dct_cfl(x, obj._constants["Y"].size),
        "dft cfl": _vander_dft_cfl(x, obj._grid),
        "dct spline": _vander_dct_cfl(
            jnp.linspace(-1, 1, Y_B, endpoint=False), obj._constants["Y"].size
        ),
    }


def _vander_dft_cfl(x, grid):
    modes = jnp.fft.fftfreq(grid.num_zeta, 1 / (grid.NFP * grid.num_zeta))
    zeta = bijection_from_disc(x, 0, 2 * jnp.pi)
    return jnp.exp(1j * modes * zeta[:, jnp.newaxis])[..., jnp.newaxis]


def _vander_dct_cfl(x, Y):
    return jnp.cos(jnp.arange(Y) * jnp.arccos(x)[:, jnp.newaxis])



def _build_eta_grid(eq, rhos, alpha_per_rho, zeta, iotas, params):
    """Build a DESC grid with per-rho alpha values derived from uniform eta.

    Creates a meshgrid-like grid where each rho surface has its own alpha
    values (computed from uniformly spaced eta), maps it to DESC (rho, theta,
    zeta) coordinates, and returns the resulting grid with the original
    (rho, alpha, zeta) grid stored as ``source_grid``.

    Parameters
    ----------
    eq : Equilibrium
    rhos : jnp.ndarray, shape (num_rho,)
    alpha_per_rho : jnp.ndarray, shape (num_rho, num_eta)
        Alpha values for each rho surface, derived from uniform eta.
    zeta : jnp.ndarray, shape (num_zeta,)
    iotas : jnp.ndarray, shape (num_rho,)
    params : dict
        Equilibrium parameters.

    Returns
    -------
    grid : Grid
        DESC grid in (rho, theta, zeta) with ``source_grid`` in (rho, alpha, zeta).
    """
    from desc.equilibrium.coords import map_coordinates

    num_rho = len(rhos)
    num_eta = alpha_per_rho.shape[1]
    num_zeta = len(zeta)

    # Build raz nodes in meshgrid order: alpha fastest, rho middle, zeta slowest
    # (matching the Fortran-order layout used by Grid.create_meshgrid)
    _, rr, zz = jnp.meshgrid(jnp.arange(num_eta), rhos, zeta, indexing="ij")
    alpha_arr = jnp.broadcast_to(
        alpha_per_rho.T[:, :, jnp.newaxis], (num_eta, num_rho, num_zeta)
    )
    raz_nodes = jnp.column_stack([
        rr.flatten(order="F"),
        alpha_arr.flatten(order="F"),
        zz.flatten(order="F"),
    ])

    unique_rho_idx = jnp.arange(num_rho) * num_eta
    unique_poloidal_idx = jnp.arange(num_eta)
    unique_zeta_idx = jnp.arange(num_zeta) * num_rho * num_eta
    inverse_rho_idx = jnp.tile(
        jnp.repeat(jnp.arange(num_rho), num_eta), num_zeta
    )
    inverse_poloidal_idx = jnp.tile(jnp.arange(num_eta), num_rho * num_zeta)
    inverse_zeta_idx = jnp.repeat(jnp.arange(num_zeta), num_rho * num_eta)

    raz_grid = Grid(
        nodes=raz_nodes,
        coordinates="raz",
        period=(jnp.inf, jnp.inf, jnp.inf),
        sort=False,
        is_meshgrid=True,
        jitable=True,
        _unique_rho_idx=unique_rho_idx,
        _unique_poloidal_idx=unique_poloidal_idx,
        _unique_zeta_idx=unique_zeta_idx,
        _inverse_rho_idx=inverse_rho_idx,
        _inverse_poloidal_idx=inverse_poloidal_idx,
        _inverse_zeta_idx=inverse_zeta_idx,
    )

    iota_expanded = raz_grid.expand(jnp.atleast_1d(jnp.asarray(iotas)))
    rtz_nodes = map_coordinates(
        eq,
        raz_grid.nodes,
        inbasis=["rho", "alpha", "zeta"],
        outbasis=("rho", "theta", "zeta"),
        period=(jnp.inf, jnp.inf, jnp.inf),
        iota=iota_expanded,
        params=params,
    )

    desc_grid = Grid(
        nodes=rtz_nodes,
        coordinates="rtz",
        source_grid=raz_grid,
        sort=False,
        jitable=True,
        _unique_rho_idx=unique_rho_idx,
        _inverse_rho_idx=inverse_rho_idx,
    )
    return desc_grid


# New resonance objective from John Anthony Labbate
class TrappedResonance(_Objective):
    """Trapped energetic particle resonance penalty.

    Creates bump function about a specified number of lowest order resonances
    (m/n) for trapped energetic particle motion. Vicinity to these rational
    values is penalized.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    grid : Grid, optional
        Meshgrid (e.g. ``LinearGrid``) whose rho nodes define the flux
        surfaces to evaluate on.  If provided, rho is extracted via
        ``grid.compress(grid.nodes[:, 0])``, matching the pattern used by
        ``EffectiveRipple``.  Takes precedence over the ``rho`` parameter.
    rho : ndarray, optional
        Unique flux surface labels.  Ignored when ``grid`` is given.
        Default is ``np.linspace(0.1, 0.9, 3)``.
    num_eta : int, optional
        Number of uniformly spaced eta points in [0, 2*pi).
        Alpha values are derived per rho surface via
        ``alpha = eta * (N*nfp - iota*M) / nfp``.
        Default is 10.

    """

    _scalar = True
    _coordinates = "" # "rtz" if need all three coordinates
    _units = "(s^-2)"
    _print_value_fmt = "Trapped EP Resonance Penalty: "

    _static_attrs = _Objective._static_attrs + ["_hyperparameters", "_keys_1dr", "_key"]

    def __init__(
        self,
        eq,
        grid=None,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        rho=None,
        num_eta=10,
        KE_frac=np.array([1]),
        *,
        num_transit=5,
        knots_per_transit=100,
        num_quad=32,
        num_pitch=16,
        pitch_method=1,
        batch=True,
        num_well=None,
        Nemov=True,
        name="TrappedResonance",
        jac_chunk_size=None,
        pitch_invs=None,
        N=0,
        M=1,
        m_max=10,
        n_max=10,
        res_range_min=-4,
        res_range_max=4,
        DEBUG=False,
        verbose=False,
        pitch_batch_size=1,
        surf_batch_size=1,
        f_q_conservative=False,
    ):
        if target is None and bounds is None:
            target = 1e-8
        self._grid = grid
        self._rho = np.atleast_1d(rho) if rho is not None else None
        self._num_eta = int(num_eta)
        self._constants = {
            "quad_weights": 1,
            "zeta": np.linspace(
                0, 2 * np.pi * num_transit, knots_per_transit * num_transit
            ),
        }

        self._hyperparameters = {
            "num_quad": num_quad,
            "num_pitch": num_pitch,
            "batch": batch,
            "num_well": num_well,
            "KE_frac": KE_frac,
            "pitch_invs": pitch_invs,
            "N": N,
            "M": M,
            "m_max": m_max,
            "n_max": n_max,
            "res_range_min": res_range_min,
            "res_range_max": res_range_max,
            "DEBUG": DEBUG,
            "verbose": verbose,
            "pitch_batch_size": pitch_batch_size,
            "surf_batch_size": surf_batch_size,
            "num_transit": num_transit,
            "pitch_method": pitch_method,
            "f_q_conservative": f_q_conservative,
        }
        self._keys_1dr = ["iota", "iota_r", "min_tz |B|", "max_tz |B|", "Psi"]
        self._key = "f_tr2"

        super().__init__(
            things=[eq],
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
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

        # Resolve rho: grid takes precedence, then explicit array, then default.
        if self._grid is not None:
            assert self._grid.is_meshgrid, (
                "Provided grid must be a meshgrid (e.g. LinearGrid)."
            )
            rho = self._grid.compress(self._grid.nodes[:, 0])
            if self._rho is not None:
                warnings.warn(
                    "Both `grid` and `rho` were provided. "
                    "Using rho values from `grid`."
                )
        elif self._rho is not None:
            rho = self._rho
        else:
            rho = np.linspace(0.1, 0.9, 3)

        self._constants["rho"] = rho
        self._dim_f = rho.size

        self._grid_1dr = LinearGrid(
            rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym
        )
        self._constants["quad"] = get_quadrature(
            leggauss(self._hyperparameters.pop("num_quad")),
            (automorphism_sin, grad_automorphism_sin),
        )
        self._params2 = {
            "rho_res": (rho[-1] - rho[0]) / (len(rho) - 1),
        }
        self._target, self._bounds = _parse_callable_target_bounds(
            self._target, self._bounds, rho
        )

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._constants["transforms_1dr"] = get_transforms(
            self._keys_1dr, eq, self._grid_1dr
        )
        self._constants["profiles"] = get_profiles(
            self._keys_1dr + [self._key], eq, self._grid_1dr
        )

        # Setup rational array
        m_max = self._hyperparameters["m_max"]
        n_max = self._hyperparameters["n_max"]
        res_range_min = self._hyperparameters["res_range_min"]
        res_range_max = self._hyperparameters["res_range_max"]

        res_arr = np.full((2*n_max+1)*(m_max+1), jnp.pi) # maximum possible size of array of resonances, including the zero resonance and negative resonances
        n_arr = np.full((2*n_max+1)*(m_max+1), 1)
        m_arr = np.full((2*n_max+1)*(m_max+1), 1)
        res_arr_set = 0

        for m in range(0,m_max+1):
            for n in range(1,n_max+1):
                condition = np.logical_and(m/n >= res_range_min, m/n <= res_range_max)
                if condition:
                    res_arr[res_arr_set] = m/n
                    n_arr[res_arr_set] = n
                    m_arr[res_arr_set] = m
                    res_arr_set+=1
                    if m != 0:
                        res_arr[res_arr_set] = -m/n
                        n_arr[res_arr_set] = n
                        m_arr[res_arr_set] = m
                        res_arr_set+=1

        self._hyperparameters['q_arr'] = n_arr
        self._hyperparameters['res_arr'] = res_arr
        self._hyperparameters['p_arr'] = m_arr
        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute TrappedResonance objective.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, e.g.
            ``Equilibrium.params_dict``
        constants : dict
            Dictionary of constant data, e.g. transforms, profiles etc.
            Defaults to ``self.constants``.

        Returns
        -------
        result : ndarray
            Γ_c as a function of the flux surface label.

        """
        if constants is None:
            constants = self._constants
        eq = self.things[0]

        data = compute_fun(
            eq,
            self._keys_1dr,
            params,
            constants["transforms_1dr"],
            constants["profiles"],
            
        )
        # Build grid with per-rho alpha derived from uniformly spaced eta
        iotas = self._grid_1dr.compress(data["iota"])
        rhos = constants["rho"]
        zeta = constants["zeta"]
        num_eta = self._num_eta
        nfp = eq.NFP
        N_mode = self._hyperparameters["N"]
        M_mode = self._hyperparameters["M"]

        eta_vals = jnp.linspace(0, 2 * jnp.pi, num_eta, endpoint=False)
        ft_denom = N_mode * nfp - iotas * M_mode
        alpha_per_rho = eta_vals[None, :] * ft_denom[:, None] / nfp

        grid = _build_eta_grid(eq, rhos, alpha_per_rho, zeta, iotas, params)
        data = {
            key: grid.copy_data_from_other(data[key], self._grid_1dr)
            for key in self._keys_1dr
        }
        quad2 = {}
        if "quad2" in constants:
            quad2["quad2"] = constants["quad2"]

        data = compute_fun(
            eq,
            self._key,
            params,
            get_transforms(self._key, eq, grid, jitable=True),
            constants["profiles"],
            data=data,
            quad=constants["quad"],
            nfp=eq.NFP,
            eta_vals=eta_vals,
            **quad2,
            **self._hyperparameters,
            **self._params2,
        )
        # return grid.compress(data[self._key]) # return the value of the objective function evaluated at each point on the grid

        return data[self._key]