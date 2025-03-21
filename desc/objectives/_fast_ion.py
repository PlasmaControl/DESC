"""Objectives for fast ion confinement."""

import numpy as np
from orthax.legendre import leggauss

from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.utils import setdefault

from ..integrals import Bounce2D
from ..integrals.basis import FourierChebyshevSeries
from ..integrals.quad_utils import (
    automorphism_sin,
    get_quadrature,
    grad_automorphism_sin,
)
from ._neoclassical import _bounce_overwrite
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
      * ``1154`` Improve coordinate mapping performance
      * ``1294`` Nonuniform fast transforms
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
    spline : bool
        Set to ``True`` to replace pseudo-spectral methods with local splines.
        This can be efficient if ``num_transit`` and ``alpha.size`` are small,
        depending on hardware and hardware features used by the JIT compiler.
        If ``True``, then parameters ``X`` and ``Y`` are ignored.
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
        # Y_B is expensive to increase if one does not fix num well per transit.
        Y_B=None,
        alpha=np.array([0.0]),
        num_transit=20,
        num_well=None,
        num_quad=32,
        num_pitch=64,
        pitch_batch_size=None,
        surf_batch_size=1,
        spline=False,
        Nemov=True,
    ):
        if target is None and bounds is None:
            target = 0.0

        self._spline = spline
        self._grid = grid
        self._constants = {"quad_weights": 1.0, "alpha": alpha}
        self._X = X
        self._Y = Y
        Y_B = setdefault(Y_B, 2 * Y)
        self._hyperparam = {
            "Y_B": Y_B,
            "num_transit": num_transit,
            "num_well": setdefault(num_well, Y_B * num_transit),
            "num_quad": num_quad,
            "num_pitch": num_pitch,
            "pitch_batch_size": pitch_batch_size,
            "surf_batch_size": surf_batch_size,
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
        if self._spline:
            return self._build_spline(use_jit, verbose)

        eq = self.things[0]
        if self._grid is None:
            self._grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
        assert self._grid.can_fft2
        self._constants["clebsch"] = FourierChebyshevSeries.nodes(
            self._X,
            self._Y,
            self._grid.compress(self._grid.nodes[:, 0]),
            domain=(0, 2 * np.pi),
        )
        self._constants["fieldline quad"] = leggauss(self._hyperparam["Y_B"] // 2)
        self._constants["quad"] = get_quadrature(
            leggauss(self._hyperparam.pop("num_quad")),
            (automorphism_sin, grad_automorphism_sin),
        )
        self._dim_f = self._grid.num_rho
        self._target, self._bounds = _parse_callable_target_bounds(
            self._target, self._bounds, self._grid.compress(self._grid.nodes[:, 0])
        )
        self._constants["transforms"] = get_transforms(self._key, eq, grid=self._grid)
        self._constants["profiles"] = get_profiles(self._key, eq, grid=self._grid)
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
        if self._spline:
            return self._compute_spline(params, constants)

        if constants is None:
            constants = self.constants
        eq = self.things[0]
        data = compute_fun(
            eq, "iota", params, constants["transforms"], constants["profiles"]
        )
        # TODO (#1034): Use old theta values as initial guess.
        theta = Bounce2D.compute_theta(
            eq,
            self._X,
            self._Y,
            iota=constants["transforms"]["grid"].compress(data["iota"]),
            clebsch=constants["clebsch"],
            # Pass in params so that root finding is done with the new
            # perturbed λ coefficients and not the original equilibrium's.
            params=params,
        )
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
            **self._hyperparam,
        )
        return constants["transforms"]["grid"].compress(data[self._key])

    def _build_spline(self, use_jit=True, verbose=1):
        self._keys_1dr = ["iota", "iota_r", "min_tz |B|", "max_tz |B|"]
        self._key = "old " + self._key
        num_transit = self._hyperparam.pop("num_transit")
        Y_B = self._hyperparam.pop("Y_B")

        eq = self.things[0]
        if self._grid is None:
            self._grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
        assert self._grid.is_meshgrid and eq.sym == self._grid.sym
        self._constants["rho"] = self._grid.compress(self._grid.nodes[:, 0])
        self._constants["zeta"] = np.linspace(
            0, 2 * np.pi * num_transit, Y_B * num_transit
        )
        self._constants["quad"] = get_quadrature(
            leggauss(self._hyperparam.pop("num_quad")),
            (automorphism_sin, grad_automorphism_sin),
        )
        self._dim_f = self._grid.num_rho
        self._target, self._bounds = _parse_callable_target_bounds(
            self._target, self._bounds, self._constants["rho"]
        )
        self._constants["transforms_1dr"] = get_transforms(
            self._keys_1dr, eq, self._grid
        )
        self._constants["profiles"] = get_profiles(
            self._keys_1dr + [self._key], eq, self._grid
        )
        super().build(use_jit=use_jit, verbose=verbose)

    def _compute_spline(self, params, constants=None):
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
        grid = eq._get_rtz_grid(
            constants["rho"],
            constants["alpha"],
            constants["zeta"],
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
