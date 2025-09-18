"""Objectives for fast ion confinement."""

import warnings

import numpy as np
from orthax.legendre import leggauss

from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.integrals._interp_utils import cheb_pts, fourier_pts
from desc.utils import parse_argname_change, setdefault, warnif

from ..integrals.quad_utils import (
    automorphism_sin,
    get_quadrature,
    grad_automorphism_sin,
)
from ._neoclassical import _bounce_overwrite, _get_vander
from .objective_funs import _Objective, collect_docs
from .utils import _parse_callable_target_bounds


class GammaC(_Objective):
    """Proxy for fast ion confinement.

    A 3D stellarator magnetic field admits ripple wells that lead to enhanced
    radial drift of trapped particles. The energetic particle confinement
    metric γ_c quantifies whether the contours of the second adiabatic invariant
    close on the flux surfaces. In the limit where the poloidal drift velocity
    majorizes the radial drift velocity, the contours lie parallel to flux
    surfaces. The optimization metric Γ_c averages γ_c² over the distribution
    of trapped particles on each flux surface.

    The radial electric field has a negligible effect, since fast particles
    have high energy with collisionless orbits, so it is assumed to be zero.

    References
    ----------
    Poloidal motion of trapped particle orbits in real-space coordinates.
    V. V. Nemov, S. V. Kasilov, W. Kernbichler, G. O. Leitold.
    Phys. Plasmas 1 May 2008; 15 (5): 052501.
    https://doi.org/10.1063/1.2912456.
    Equation 61.

    A model for the fast evaluation of prompt losses of energetic ions in stellarators.
    J.L. Velasco et al. 2021 Nucl. Fusion 61 116059.
    https://doi.org/10.1088/1741-4326/ac2994.
    Equation 16.

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
        Resolution for quadrature over velocity coordinate. Default is 64.
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
    Nemov : bool
        Whether to use the Γ_c as defined by Nemov et al. or Velasco et al.
        Default is Nemov. Set to ``False`` to use Velascos's.

        Nemov's Γ_c converges to a finite nonzero value in the infinity limit
        of the number of toroidal transits. Velasco's expression has a secular
        term that drives the result to zero as the number of toroidal transits
        increases if the secular term is not averaged out from the singular
        integrals. Currently, an optimization using Velasco's metric may need
        to be evaluated by measuring decrease in Γ_c at a fixed number of toroidal
        transits.

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
        "_key",
        "_keys_1dr",
        "_use_bounce1d",
    ]

    _coordinates = "r"
    _units = "~"
    _print_value_fmt = "Γ_c: "

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
        name="Gamma_c",
        grid=None,
        X=16,
        Y=32,
        Y_B=None,
        alpha=np.array([0.0]),
        num_transit=20,
        num_well=None,
        num_quad=32,
        num_pitch=64,
        pitch_batch_size=None,
        surf_batch_size=1,
        nufft_eps=1e-7,
        use_bounce1d=False,
        Nemov=True,
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
        self._key = "Gamma_c" if Nemov else "Gamma_c Velasco"

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
        self._constants["quad"] = get_quadrature(
            leggauss(self._hyperparam.pop("num_quad")),
            (automorphism_sin, grad_automorphism_sin),
        )
        self._constants["profiles"] = get_profiles(self._key, eq, grid=self._grid)
        self._constants["transforms"] = get_transforms(self._key, eq, grid=self._grid)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Unequal number of field periods")
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
        """Compute Γ_c.

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
        Gamma_c : ndarray
            Γ_c as a function of the flux surface label.

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
            self._key,
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
        return constants["transforms"]["grid"].compress(data[self._key])

    def _build_bounce1d(self, use_jit=True, verbose=1):
        Y_B = self._hyperparam.pop("Y_B")
        num_transit = self._hyperparam.pop("num_transit")
        num_quad = self._hyperparam.pop("num_quad")
        self._hyperparam.pop("nufft_eps")
        del self._constants["X"]
        self._constants["Y"] = np.linspace(
            0, 2 * np.pi * num_transit, Y_B * num_transit
        )
        self._keys_1dr = ["iota", "iota_r", "min_tz |B|", "max_tz |B|"]
        self._key = "old " + self._key

        eq = self.things[0]
        if self._grid is None:
            self._grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
        assert self._grid.is_meshgrid and eq.sym == self._grid.sym

        rho = self._grid.compress(self._grid.nodes[:, 0])
        self._constants["rho"] = rho
        self._constants["quad"] = get_quadrature(
            leggauss(num_quad), (automorphism_sin, grad_automorphism_sin)
        )
        self._constants["profiles"] = get_profiles(
            self._keys_1dr + [self._key], eq, self._grid
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
            key: grid.copy_data_from_other(data[key], self._grid)
            for key in self._keys_1dr
        }
        data = compute_fun(
            eq,
            self._key,
            params,
            transforms=get_transforms(self._key, eq, grid, jitable=True),
            profiles=constants["profiles"],
            data=data,
            quad=constants["quad"],
            **self._hyperparam,
        )
        return grid.compress(data[self._key])
