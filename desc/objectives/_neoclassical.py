"""Objectives for targeting neoclassical transport.

Notes
-----
Performance will improve significantly by resolving these GitHub issues.
  * ``1154`` Improve coordinate mapping performance
  * ``1294`` Nonuniform fast transforms
  * ``1303`` Patch for differentiable code with dynamic shapes
  * ``1206`` Upsample data above midplane to full grid assuming stellarator symmetry
  * ``1034`` Optimizers/objectives with auxiliary output
"""

import numpy as np
from orthax.legendre import leggauss

from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.utils import Timer, setdefault

from ..integrals import Bounce2D
from ..integrals._quad_utils import (
    automorphism_sin,
    chebgauss2,
    get_quadrature,
    grad_automorphism_sin,
)
from ..integrals.basis import FourierChebyshevSeries
from .objective_funs import _Objective, collect_docs
from .utils import _parse_callable_target_bounds


class EffectiveRipple(_Objective):
    """The effective ripple is a proxy for neoclassical transport.

    The 3D geometry of the magnetic field in stellarators produces local magnetic
    wells that lead to bad confinement properties with enhanced radial drift,
    especially for trapped particles. Neoclassical (thermal) transport can become the
    dominant transport channel in stellarators which are not optimized to reduce it.
    The effective ripple is a proxy, measuring the effective modulation amplitude of the
    magnetic field averaged along a magnetic surface, which can be used to optimize for
    stellarators with improved confinement. It is targeted as a flux surface function.

    References
    ----------
    https://doi.org/10.1063/1.873749.
    Evaluation of 1/ν neoclassical transport in stellarators.
    V. V. Nemov, S. V. Kasilov, W. Kernbichler, M. F. Heyn.
    Phys. Plasmas 1 December 1999; 6 (12): 4622–4632.

    Parameters
    ----------
    eq : Equilibrium
        ``Equilibrium`` to be optimized.
    grid : Grid
        Optional, tensor-product grid in (ρ, θ, ζ) with uniformly spaced nodes
        (θ, ζ) ∈ [0, 2π) × [0, 2π/NFP). Powers of two are preferable.
    X : int
        Grid resolution in poloidal direction for Clebsch coordinate grid.
        Preferably power of 2.
    Y : int
        Grid resolution in toroidal direction for Clebsch coordinate grid.
        Preferably power of 2.
    Y_B : int
        Desired resolution for |B| along field lines to compute bounce points.
        Default is double ``Y``.
    num_transit : int
        Number of toroidal transits to follow field line.
        For axisymmetric devices, one poloidal transit is sufficient. Otherwise,
        assuming the surface is not near rational, more transits will
        approximate surface averages better, with diminishing returns.
    num_quad : int
        Resolution for quadrature of bounce integrals. Default is 32.
    num_pitch : int
        Resolution for quadrature over velocity coordinate. Default is 64.
    num_well : int
        Maximum number of wells to detect for each pitch and field line.
        Giving ``None`` will detect all wells but due to current limitations in
        JAX this will have worse performance.
        Specifying a number that tightly upper bounds the number of wells will
        increase performance. In general, an upper bound on the number of wells
        per toroidal transit is ``Aι+B`` where ``A``,``B`` are the poloidal and
        toroidal Fourier resolution of |B|, respectively, in straight-field line
        PEST coordinates, and ι is the rotational transform normalized by 2π.
        A tighter upper bound than ``num_well=(Aι+B)*num_transit`` is preferable.
        The ``check_points`` or ``plot`` methods in ``desc.integrals.Bounce2D``
        are useful to select a reasonable value.
    batch_size : int
        Number of pitch values with which to compute simultaneously.
        If given ``None``, then ``batch_size`` defaults to ``num_pitch``.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
    )

    _coordinates = "r"
    _units = "~"
    _print_value_fmt = "Effective ripple ε: "

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
        name="Effective ripple",
        jac_chunk_size=None,
        *,
        X=16,  # X is cheap to increase.
        Y=32,
        # Y_B is expensive to increase if one does not fix num well per transit.
        Y_B=None,
        num_transit=20,
        num_quad=32,
        num_pitch=50,
        num_well=None,
        batch_size=None,
    ):
        if target is None and bounds is None:
            target = 0.0

        self._grid = grid
        self._constants = {"quad_weights": 1}
        self._X = X
        self._Y = Y
        Y_B = setdefault(Y_B, 2 * Y)
        self._hyperparam = {
            "Y_B": Y_B,
            "num_transit": num_transit,
            "num_quad": num_quad,
            "num_pitch": num_pitch,
            "num_well": setdefault(num_well, Y_B * num_transit),
            "batch_size": batch_size,
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
        eq = self.things[0]
        if self._grid is None:
            self._grid = LinearGrid(
                theta=eq.M_grid, zeta=eq.N_grid, NFP=eq.NFP, sym=False
            )
        assert self._grid.can_fft
        self._constants["clebsch"] = FourierChebyshevSeries.nodes(
            self._X,
            self._Y,
            self._grid.compress(self._grid.nodes[:, 0]),
            domain=(0, 2 * np.pi),
        )
        self._constants["fieldline_quad"] = leggauss(self._hyperparam["Y_B"] // 2)
        self._constants["quad"] = chebgauss2(self._hyperparam.pop("num_quad"))

        self._dim_f = self._grid.num_rho
        self._target, self._bounds = _parse_callable_target_bounds(
            self._target, self._bounds, self._grid.compress(self._grid.nodes[:, 0])
        )

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")
        self._constants["transforms"] = get_transforms(
            "effective ripple", eq, grid=self._grid
        )
        self._constants["profiles"] = get_profiles(
            "effective ripple", eq, grid=self._grid
        )
        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

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
        eps_eff : ndarray
            Effective ripple as a function of the flux surface label.

        """
        # TODO: GitHub pull request #1094.
        if constants is None:
            constants = self.constants
        eq = self.things[0]
        data = compute_fun(
            eq, "iota", params, constants["transforms"], constants["profiles"]
        )
        # TODO: GitHub issue #1034. Use old theta values as initial guess.
        data = compute_fun(
            eq,
            "effective ripple",
            params,
            constants["transforms"],
            constants["profiles"],
            data,
            theta=Bounce2D.compute_theta(
                eq,
                self._X,
                self._Y,
                iota=constants["transforms"]["grid"].compress(data["iota"]),
                clebsch=constants["clebsch"],
                # Pass in params so that root finding is done with the new
                # perturbed λ coefficients and not the original equilibrium's.
                params=params,
            ),
            fieldline_quad=constants["fieldline_quad"],
            quad=constants["quad"],
            **self._hyperparam,
        )
        return constants["transforms"]["grid"].compress(data["effective ripple"])


class GammaC(_Objective):
    """Γ_c is a proxy for measuring energetic ion confinement.

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

    Parameters
    ----------
    eq : Equilibrium
        ``Equilibrium`` to be optimized.
    grid : Grid
        Optional, tensor-product grid in (ρ, θ, ζ) with uniformly spaced nodes
        (θ, ζ) ∈ [0, 2π) × [0, 2π/NFP). Powers of two are preferable.
    X : int
        Grid resolution in poloidal direction for Clebsch coordinate grid.
        Preferably power of 2.
    Y : int
        Grid resolution in toroidal direction for Clebsch coordinate grid.
        Preferably power of 2.
    Y_B : int
        Desired resolution for |B| along field lines to compute bounce points.
        Default is double ``Y``.
    num_transit : int
        Number of toroidal transits to follow field line.
        For axisymmetric devices, one poloidal transit is sufficient. Otherwise,
        assuming the surface is not near rational, more transits will
        approximate surface averages better, with diminishing returns.
    num_quad : int
        Resolution for quadrature of bounce integrals. Default is 32.
    num_pitch : int
        Resolution for quadrature over velocity coordinate. Default is 64.
    num_well : int
        Maximum number of wells to detect for each pitch and field line.
        Giving ``None`` will detect all wells but due to current limitations in
        JAX this will have worse performance.
        Specifying a number that tightly upper bounds the number of wells will
        increase performance. In general, an upper bound on the number of wells
        per toroidal transit is ``Aι+B`` where ``A``,``B`` are the poloidal and
        toroidal Fourier resolution of |B|, respectively, in straight-field line
        PEST coordinates, and ι is the rotational transform normalized by 2π.
        A tighter upper bound than ``num_well=(Aι+B)*num_transit`` is preferable.
        The ``check_points`` or ``plot`` methods in ``desc.integrals.Bounce2D``
        are useful to select a reasonable value.
    batch_size : int
        Number of pitch values with which to compute simultaneously.
        If given ``None``, then ``batch_size`` defaults to ``num_pitch``.
    Nemov : bool
        Whether to use the Γ_c as defined by Nemov et al. or Velasco et al.
        Default is Nemov. Set to ``False`` to use Velascos's.

        Note that Nemov's Γ_c converges to a finite nonzero value in the
        infinity limit of the number of toroidal transits.
        Velasco's expression has a secular term that will drive the result
        to zero as the number of toroidal transits increases unless the
        secular term is averaged out from all the singular integrals.
        Therefore, an optimization using Velasco's metric should be evaluated by
        measuring decrease in Γ_c at a fixed number of toroidal transits until
        unless an adaptive quadrature is used.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
    )

    _coordinates = "r"
    _units = "~"
    _print_value_fmt = "Γ_c: "

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
        name="Gamma_c",
        jac_chunk_size=None,
        *,
        X=16,  # X is cheap to increase.
        Y=32,
        # Y_B is expensive to increase if one does not fix num well per transit.
        Y_B=None,
        num_transit=20,
        num_quad=32,
        num_pitch=64,
        num_well=None,
        batch_size=None,
        Nemov=True,
    ):
        if target is None and bounds is None:
            target = 0.0

        self._grid = grid
        self._constants = {"quad_weights": 1}
        self._X = X
        self._Y = Y
        Y_B = setdefault(Y_B, 2 * Y)
        self._hyperparam = {
            "Y_B": Y_B,
            "num_transit": num_transit,
            "num_quad": num_quad,
            "num_pitch": num_pitch,
            "num_well": setdefault(num_well, Y_B * num_transit),
            "batch_size": batch_size,
        }
        if Nemov:
            self._key = "Gamma_c"
        else:
            self._key = "Gamma_c Velasco"
            raise NotImplementedError

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
            self._grid = LinearGrid(
                theta=eq.M_grid, zeta=eq.N_grid, NFP=eq.NFP, sym=False
            )
        assert self._grid.can_fft
        self._constants["clebsch"] = FourierChebyshevSeries.nodes(
            self._X,
            self._Y,
            self._grid.compress(self._grid.nodes[:, 0]),
            domain=(0, 2 * np.pi),
        )
        self._constants["fieldline_quad"] = leggauss(self._hyperparam["Y_B"] // 2)
        num_quad = self._hyperparam.pop("num_quad")
        self._constants["quad"] = get_quadrature(
            leggauss(num_quad),
            (automorphism_sin, grad_automorphism_sin),
        )
        self._constants["quad2"] = chebgauss2(num_quad)

        self._dim_f = self._grid.num_rho
        self._target, self._bounds = _parse_callable_target_bounds(
            self._target, self._bounds, self._grid.compress(self._grid.nodes[:, 0])
        )

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")
        self._constants["transforms"] = get_transforms(self._key, eq, grid=self._grid)
        self._constants["profiles"] = get_profiles(self._key, eq, grid=self._grid)
        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

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
        if constants is None:
            constants = self.constants
        if "quad2" in constants:
            self._hyperparam["quad2"] = constants["quad2"]

        eq = self.things[0]
        data = compute_fun(
            eq, "iota", params, constants["transforms"], constants["profiles"]
        )
        # TODO: GitHub issue #1034. Use old theta values as initial guess.
        data = compute_fun(
            eq,
            self._key,
            params,
            constants["transforms"],
            constants["profiles"],
            data,
            theta=Bounce2D.compute_theta(
                eq,
                self._X,
                self._Y,
                iota=constants["transforms"]["grid"].compress(data["iota"]),
                clebsch=constants["clebsch"],
                # Pass in params so that root finding is done with the new
                # perturbed λ coefficients and not the original equilibrium's.
                params=params,
            ),
            fieldline_quad=constants["fieldline_quad"],
            quad=constants["quad"],
            **self._hyperparam,
        )
        return constants["transforms"]["grid"].compress(data[self._key])
