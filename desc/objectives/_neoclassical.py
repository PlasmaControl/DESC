"""Objectives for neoclassical transport."""

import warnings

import numpy as np
from orthax.legendre import leggauss

from desc.backend import jnp
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.integrals._interp_utils import bijection_from_disc, cheb_pts, fourier_pts
from desc.utils import parse_argname_change, setdefault, warnif

from ..integrals.quad_utils import chebgauss2
from .objective_funs import _Objective, collect_docs
from .utils import _parse_callable_target_bounds

_bounce_overwrite = {
    "deriv_mode": """
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute Jacobian matrix, either forward mode or reverse mode AD.
        ``auto`` selects forward or reverse mode based on the size of the input and
        output of the objective. Has no effect on ``self.grad`` or ``self.hess`` which
        always use reverse mode and forward over reverse mode respectively.

        In ``rev`` mode, reducing the parameter ``pitch_batch_size`` does not
        reduce memory consumption, so it is recommended to retain the default for that.
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

        This is the most important parameter to specify for performance.
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
            warnif(
                nufft_eps >= 1e-14,
                msg="\njax-finufft is not installed.\n"
                "Setting parameter nufft_eps to zero.\n"
                "Performance will deteriorate significantly.\n",
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
