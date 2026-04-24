"""Methods for computing bounce integrals (singular or otherwise)."""

import warnings
from abc import ABC, abstractmethod
from typing import NamedTuple

import equinox as eqx
from interpax import CubicHermiteSpline, PPoly
from interpax_fft import (
    FourierChebyshevSeries,
    PiecewiseChebyshevSeries,
    cheb_from_dct,
    cheb_pts,
    fourier_pts,
    idct_mmt,
    ifft_mmt,
    irfft2_mmt_pos,
    irfft_mmt_pos,
    rfft2_modes,
)
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from orthax.chebyshev import chebvander
from orthax.legendre import leggauss

from desc.backend import jax, jnp, rfft2
from desc.batching import batch_map
from desc.derivatives import sparse_pullback
from desc.grid import LinearGrid
from desc.integrals._bounce_utils import (
    _sentinel,
    argmin,
    bounce_points,
    broadcast_for_bounce,
    check_bounce_points,
    check_interp,
    fast_chebyshev,
    fast_cubic_spline,
    get_mins,
    mmt_for_bounce,
    move,
    plot_ppoly,
    regular_points,
    set_default_plot_kwargs,
    theta_on_fieldlines,
    truncate_rule,
)
from desc.integrals._interp_utils import (
    _JF_BUG,
    _eps,
    interp1d_Hermite_vec,
    interp1d_vec,
    nufft2d2r,
)
from desc.integrals.quad_utils import (
    automorphism_sin,
    bijection_from_disc,
    chebgauss1,
    chebgauss2,
    get_quadrature,
    grad_automorphism_sin,
    grad_bijection_from_disc,
    simpson2,
    uniform,
)
from desc.utils import (
    apply,
    atleast_nd,
    errorif,
    flatten_mat,
    parse_argname_change,
    setdefault,
    warnif,
)


class Bounce(eqx.Module, ABC):
    """Abstract class for bounce integrals."""

    @staticmethod
    def pitch_quad(min_B, max_B, num_pitch, **kwargs):
        """Return 1/λ values and weights for quadrature between ``min_B`` and ``max_B``.

        Parameters
        ----------
        min_B : jnp.ndarray
            Minimum B value.
        max_B : jnp.ndarray
            Maximum B value.
        num_pitch : int or tuple[jnp.ndarray]
            If given an integer, this is interpreted as the resolution for
            a quadrature using Simpson’s 1/3 in the interior completed by an
            open midpoint scheme near the boundary.
            If given a tuple, then this is interpreted as the quadrature
            points xₖ and weights wₖ for the approximation of the integral
            ∫₋₁¹ f(x) dx ≈ ∑ₖ wₖ f(xₖ). Then this method simply rescales
            the quadrature for integration between ``min_B`` and ``max_B``.

        Returns
        -------
        pitch_inv, weight : tuple[jnp.ndarray]
            Shape (min_B.shape, num pitch).
            1/λ values and weights.

        """
        if isinstance(num_pitch, int):
            errorif(
                num_pitch > 1e5,
                msg="Floating point error impedes detection of bounce points "
                f"near global extrema. Choose {num_pitch} < 1e5.",
            )
            simp = kwargs.get("simp", True)

            num_pitch = simpson2(num_pitch) if simp else uniform(num_pitch)
        x, w = num_pitch

        if jnp.ndim(min_B):
            min_B = min_B[..., None]
            max_B = max_B[..., None]

        x = bijection_from_disc(x, min_B, max_B)
        w = w * grad_bijection_from_disc(min_B, max_B)
        return x, w

    get_pitch_inv_quad = pitch_quad
    """Alias to ``pitch_quad``."""

    @abstractmethod
    def points(self, pitch_inv, num_well=None):
        """Compute bounce points."""

    @abstractmethod
    def check_points(self, points, pitch_inv, *, plot=True):
        """Check that bounce points are computed correctly."""

    @abstractmethod
    def integrate(
        self,
        integrand,
        pitch_inv,
        data=None,
        names=None,
        points=None,
        *,
        num_well=None,
        quad=None,
    ):
        """Bounce integrate ∫ f(ρ,α,λ,ℓ) dℓ."""

    @abstractmethod
    def interp_to_argmin(self, f, points):
        """Interpolate ``f`` to the deepest point pⱼ in magnetic well j.

        Let E = {ζ ∣ ζ₁ < ζ < ζ₂} and A ∈ argmin_E B(ζ). Returns f(A).
        """

    @abstractmethod
    def plot(self, l, m, pitch_inv=None, **kwargs):
        """Plot B and bounce points on the specified field line."""


class Bounce2D(Bounce):
    """Computes bounce integrals using pseudo-spectral methods.

    The bounce integral is defined as ∫ f(ρ,α,λ,ℓ) dℓ where

    * dℓ parametrizes the distance along the field line in meters.
    * f(ρ,α,λ,ℓ) is the quantity to integrate along the field line.
    * The boundaries of the integral are bounce points ℓ₁, ℓ₂ s.t. λB(ρ,α,ℓᵢ) = 1.
    * λ is a constant defining the integral proportional to the magnetic moment
      over energy.
    * B is the norm of the magnetic field.

    For a particle with fixed λ, bounce points are defined to be the location on the
    field line such that the particle's velocity parallel to the magnetic field is zero.
    The bounce integral is defined up to a sign. We choose the sign that corresponds to
    the particle's guiding center trajectory traveling in the direction of increasing
    field-line-following coordinate ζ.

    Refrences
    ---------
    Spectrally accurate, reverse-mode differentiable bounce-averaging algorithm
    and its applications. Kaya Unalmis et al. Journal of Plasma Physics.

    Examples
    --------
      * ``tests/test_integrals.py::TestBounce2D::test_bounce2d_checks``
      * ``desc/compute/_fast_ion.py::_little_gamma_c_Nemov``
      * ``desc/compute/_neoclassical.py::_epsilon_32``
      * ``desc/objectives/_fast_ion.py::GammaC``
      * ``desc/objectives/_neoclassical.py::EffectiveRipple``

    See Also
    --------
    Bounce1D
        Some comments comparing ``Bounce1D`` to ``Bounce2D`` are given below.
        ``Bounce1D`` uses lower order accurate, one-dimensional splines.
        ``Bounce2D`` is superior for optimization objectives in DESC as it solves the
        moving grid interpolation problem, avoids recomputing 3D Fourier-Zernike
        series on a time-dependent grid, and is able to compute the derivative
        matrix relevant to optimzation with a compact sparse pullback.

    Parameters
    ----------
    grid : Grid
        Tensor-product grid in (ρ, θ, ζ) with uniformly spaced nodes
        (θ, ζ) ∈ [0, 2π) × [0, 2π/NFP).
        Number of poloidal and toroidal nodes preferably rounded down to powers of two.
        Determines the flux surfaces to compute on and resolution of FFTs.
        The ζ coordinates (the unique values prior to taking the tensor-product)
        must be strictly increasing.
    data : dict[str, jnp.ndarray]
        Data evaluated on ``grid``.
        Must include names in ``Bounce2D.required_names``.
        If the input is not a real-valued array, then it
        is assumed that the Fourier transform as returned by ``Bounce2D.fourier``
        was given instead.
    angle : jnp.ndarray
        Shape (num ρ, X, Y).
        Angle returned by ``Bounce2D.angle``.
    Y_B : int
        Desired resolution for algorithm to compute bounce points.
        If the option ``spline`` is ``True``, the bounce points are found with
        8th order accuracy in this parameter. If the option ``spline`` is ``False``,
        then the bounce points are found with spectral accuracy in this parameter.
        A reference value is ``(grid.num_theta+grid.num_zeta)//2``.

        An error of ε in a bounce point manifests
        𝒪(ε¹ᐧ⁵) error in bounce integrals with (v_∥)¹ and
        𝒪(ε⁰ᐧ⁵) error in bounce integrals with (v_∥)⁻¹.
    alpha : jnp.ndarray
        Shape (num α, ).
        Starting field line poloidal labels.
        Default is single field line. To compute a surface average
        on a rational surface, it is necessary to average over multiple
        field lines until the surface is covered sufficiently.
    num_field_periods : int
        Number of field periods to follow field line.
        In an axisymmetric device, field line integration over a single poloidal
        transit is sufficient to capture a surface average. For a 3D
        configuration, more transits will approximate surface averages on an
        irrational magnetic surface better, with diminishing returns.
    quad : tuple[jnp.ndarray]
        Quadrature points xₖ and weights wₖ for the approximation of an
        integral ∫₋₁¹ g(x) dx = ∑ₖ wₖ g(xₖ). Default is 32 points.
    automorphism : tuple[Callable] or None
        The first callable should be an automorphism of the real interval [-1, 1].
        The second callable should be the derivative of the first. This map defines
        a change of variable for the bounce integral. The choice made for the
        automorphism will affect the performance of the quadrature.
    nufft_eps : float
        Precision requested for interpolation with non-uniform fast Fourier
        transform (NUFFT). If less than ``1e-14`` then NUFFT will not be used.
    spline : bool
        Whether to use cubic splines to compute initial guess for bounce points
        instead of Chebyshev series. Default is ``True``. It can be preferable
        to set to ``False`` on equilibria with high ``NFP``, (such cases make
        smaller ``Y_B`` feasible), or on GPUs where eigenvalue solves are fast.
    Bref : float
        Optional. Reference magnetic field strength for normalization.
    Lref : float
        Optional. Reference length scale for normalization.
    check : bool
        Flag for debugging. Must be false for JAX transformations.

    """

    required_names = ["B^zeta", "|B|", "iota"]  # TODO(#2152)
    """Required keys in the ``data`` dictionary given to the ``__init__`` method."""

    _quad: tuple[jax.Array]
    _NFP: int
    _num_t: int
    _modes_z: jax.Array
    _modes_t: jax.Array
    _c: dict[str, jax.Array]
    _theta: PiecewiseChebyshevSeries
    _nufft_eps: float = eqx.field(static=True)

    def __init__(
        self,
        grid,
        data,
        angle,
        Y_B=None,
        alpha=jnp.array([0.0]),
        num_field_periods=20,
        quad=None,
        *,
        automorphism=None,
        nufft_eps=1e-6,
        spline=True,
        Bref=1.0,
        Lref=1.0,
        check=False,
        vander=None,
        **kwargs,
    ):
        """Returns an object to compute bounce integrals."""
        assert grid.can_fft2

        if quad is None:
            quad = jax.lax.stop_gradient(
                get_quadrature(leggauss(32), (automorphism_sin, grad_automorphism_sin))
            )
        else:
            quad = get_quadrature(quad, automorphism)
        self._quad = quad

        self._NFP = grid.NFP
        self._num_t = grid.num_theta
        self._modes_z, self._modes_t = rfft2_modes(
            grid.num_zeta, grid.num_theta, (0, 2 * jnp.pi / grid.NFP)
        )

        # Figure out if input is split into batches or needs a Fourier transform.
        is_real = jnp.isrealobj(data["|B|"])
        s = data["|B|"].shape
        is_reshaped = (
            len(s) > 1
            and s[-2] == grid.num_zeta
            and s[-1] == (grid.num_theta if is_real else (grid.num_theta // 2 + 1))
        )
        errorif(
            is_reshaped and jnp.size(data["iota"]) != (s[0] if (len(s) > 2) else 1),
            msg="You forgot to call grid.compress(data['iota'])",
        )

        self._c = {"|B|": data["|B|"] / Bref, "B^zeta": data["B^zeta"] * Lref / Bref}
        if not is_reshaped:
            self._c["|B|"] = Bounce2D.reshape(grid, self._c["|B|"])
            self._c["B^zeta"] = Bounce2D.reshape(grid, self._c["B^zeta"])
        if is_real:
            self._c["|B|"] = Bounce2D.fourier(self._c["|B|"])
            self._c["B^zeta"] = Bounce2D.fourier(self._c["B^zeta"])

        angle = parse_argname_change(angle, kwargs, "theta", "angle")
        iota = data["iota"] if is_reshaped else grid.compress(data["iota"])
        iota, alpha = jnp.atleast_1d(iota, alpha)
        self._theta = theta_on_fieldlines(
            angle, iota, alpha, num_field_periods, grid.NFP
        )

        self._nufft_eps = float(nufft_eps)

        if Y_B is None:
            Y_B = Options._guess_Y_B(grid)
        if spline:
            self._c["B(z)"], self._c["knots"] = fast_cubic_spline(
                self._theta,
                self._c["|B|"],
                Y_B,
                self._modes_t,
                self._modes_z,
                self._nufft_eps,
                vander_t=None if vander is None else vander.get("dct spline", None),
                check=check,
            )
        else:
            warnif(
                Y_B > grid.num_theta + grid.num_zeta,
                msg="Unnecessarily high resolution for Y_B with spline=False.",
            )
            self._c["B(z)"] = fast_chebyshev(
                self._theta,
                self._c["|B|"],
                Y_B,
                self._modes_t,
                self._modes_z,
            )

    @staticmethod
    def batch(fun, fun_data, desc_data, angle, grid, surf_batch_size=1, sparse=True):
        """Compute function ``fun`` over phase space in batches.

        This is a utility method to compute some function of bounce integrals
        over the phase space efficiently. You may want to also JIT compile your
        code which calls this utility method.

        Examples
        --------
          * ``desc/compute/_fast_ion.py::_little_gamma_c_Nemov``
          * ``desc/compute/_neoclassical.py::_epsilon_32``

        Parameters
        ----------
        fun : callable
            A function  which takes a single argument ``fun_data`` and computes
            bounce integrals assuming ``fun_data`` holds all required quantities
            to construct a ``Bounce2D`` operator as well as call its methods.
        fun_data : dict[str, jnp.ndarray]
            Data to reshape, interpolate, and pass to ``fun``.
            The structure of the data should match the structure
            returned by the registered compute functions in ``desc.compute``.
            Note this dictionary will be modified.
        desc_data : dict[str, jnp.ndarray]
            Data dictionary with the same structure as the data returned by the
            functions in ``desc.compute``.
        angle : jnp.ndarray
            Shape (num rho, X, Y).
            Angle returned by ``Bounce2D.angle``.
        grid : Grid
            Grid on which ``fun_data`` and ``desc_data`` were computed.
        surf_batch_size : int
            Number of flux surfaces with which to compute simultaneously.
            Default is ``1``.
        sparse : bool
            Whether to differentiate with sparsity preserving pullbacks.
            Default is ``True``, which makes the most sense if the output has
            shape (num_rho, ). Otherwise, if the output shape is larger, and
            the final objective of interest is a lower dimensional quantity
            than the output, it may be preferable to delay the vjp
            by setting to ``False``.

        Returns
        -------
        The output ``fun(fun_data)``.

        """
        for name in Bounce2D.required_names:
            fun_data[name] = desc_data[name]
        fun_data.pop("iota", None)
        for name in fun_data:
            fun_data[name] = Bounce2D.fourier(Bounce2D.reshape(grid, fun_data[name]))
        fun_data["iota"] = grid.compress(desc_data["iota"])
        fun_data["min_tz |B|"] = grid.compress(desc_data["min_tz |B|"])
        fun_data["max_tz |B|"] = grid.compress(desc_data["max_tz |B|"])
        fun_data["angle"] = angle

        if sparse:
            return sparse_pullback(fun, fun_data, surf_batch_size, strip_dim0=True)

        return batch_map(fun, fun_data, surf_batch_size, strip_dim0=True)

    @staticmethod
    def reshape(grid, f):
        """Reshape arrays for acceptable input to ``integrate``.

        Parameters
        ----------
        grid : Grid
            Tensor-product grid in (ρ, θ, ζ).
        f : jnp.ndarray
            Data evaluated on grid.

        Returns
        -------
        f : jnp.ndarray
            Shape (num ρ, num ζ, num θ).
            Reshaped data which may be given to ``integrate``.

        """
        return grid.meshgrid_reshape(f, "rzt")

    @staticmethod
    def fourier(f):
        """Transform to DESC spectral domain.

        Parameters
        ----------
        f : jnp.ndarray
            Shape (..., num ζ, num θ).
            Real scalar-valued periodic function evaluated on tensor-product grid
            with uniformly spaced nodes (θ, ζ) ∈ [0, 2π) × [0, 2π/NFP).

        Returns
        -------
        a : jnp.ndarray
            Shape is (..., 1, num ζ, num θ // 2 + 1).
            Complex coefficients of 2D real FFT of ``f``.

        """
        i = (0, -1) if (f.shape[-1] % 2 == 0) else 0
        # Due to the structure of the problem, often when evaluating the series the
        # number of ζ coordinates at which to compute the toroidal basis functions is
        # less than the number of θ coordinates at which to compute the poloidal basis.
        # Hence, it more efficient to compute the real transform in the poloidal angle.
        # Likewise to perform partial summation in this application, the real transform
        # must be done in the poloidal angle and the complex transform in the toroidal.
        f = rfft2(f, norm="forward").at[..., i].divide(2) * 2
        return f[..., None, :, :]

    @staticmethod
    def compute_theta(
        eq,
        X=32,
        Y=32,
        rho=jnp.array([1.0]),
        iota=None,
        params=None,
        profiles=None,
        tol=1e-8,
        maxiter=30,
        **kwargs,
    ):
        """Method has been deprecated in favor of Bounce2D.angle."""
        warnings.warn("Please use Bounce2D.angle instead.", DeprecationWarning)
        return Bounce2D.angle(
            eq, X, Y, rho, iota, params, profiles, tol, maxiter, **kwargs
        )

    @staticmethod
    def angle(
        eq,
        X=32,
        Y=32,
        rho=jnp.array([1.0]),
        iota=None,
        params=None,
        profiles=None,
        tol=1e-8,
        maxiter=30,
        **kwargs,
    ):
        """Return the angle for mapping boundary coordinates to field line coordinates.

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium to use defining the coordinate mapping.
        X : int
            Poloidal Fourier grid resolution to interpolate the angle.
            Preferably rounded down to power of 2.
            Default is 32.
        Y : int
            Toroidal Chebyshev grid resolution over a single field period
            to interpolate the angle.
            Preferably rounded down to power of 2.
            Default is 32.
        rho : float or jnp.ndarray
            Shape (num ρ, ).
            Flux surfaces labels in [0, 1] on which to compute.
        iota : float or jnp.ndarray
            Shape (num ρ, ).
            Optional, rotational transform on the flux surfaces to compute on.
        params : dict[str,jnp.ndarray]
            Parameters from the equilibrium, such as R_lmn, Z_lmn, i_l, p_l, etc.
            Defaults to ``eq.params_dict``.
        profiles
            Optional profiles.
        tol : float
            Stopping tolerance for root finding.
            Default is ``1e-8``.
        maxiter : int
            Maximum number of Newton iterations.

        Returns
        -------
        angle : jnp.ndarray
            Shape (num ρ, X, Y).
            Angle that maps boundary coordinates to field line coordinates.

        """
        from desc.compute.utils import get_transforms

        params = setdefault(params, eq.params_dict)

        name = kwargs.pop("name", "delta")
        if name == "lambda":
            errorif(not kwargs.pop("ignore_lambda_guard", False))

            in_name = "vartheta"
            zeta = fourier_pts(Y, (0, 2 * jnp.pi / eq.NFP))
            grid = LinearGrid(rho=rho, M=eq.L_basis.M, zeta=zeta.size, NFP=eq.NFP)
            if iota is None:
                iota = 0.0

        elif name == "delta":
            in_name = "alpha"
            zeta = cheb_pts(Y, (0, 2 * jnp.pi / eq.NFP))[::-1]
            grid = LinearGrid(rho=rho, M=eq.L_basis.M, zeta=zeta, NFP=eq.NFP)
            if iota is None:
                iota = eq._compute_iota_under_jit(rho, params, profiles, **kwargs)

        angle = eq._map_poloidal_coordinates(
            jnp.atleast_1d(iota),
            fourier_pts(X),
            zeta,
            params["L_lmn"],
            get_transforms("lambda", eq, grid)["L"],
            inbasis=in_name,
            outbasis=name,
            tol=tol,
            maxiter=maxiter,
        )
        return angle if (name == "lambda") else angle[..., ::-1]

    @property
    def _num_z(self):
        return self._modes_z.size

    def _swap_pitch(self, pitch_inv):
        """Transpose to simplify broadcasting.

        Parameters
        ----------
        pitch_inv : jnp.ndarray
            Shape broadcasts with (num ρ, num pitch).

        Returns
        -------
        pitch_inv : jnp.ndarray
            Shape broadcasts with (num pitch, *self._theta.cheb.shape[:-2])

        """
        return atleast_nd(self._theta.cheb.ndim - 1, pitch_inv).swapaxes(0, -1)

    def points(self, pitch_inv, num_well=None):
        """Compute bounce points.

        Parameters
        ----------
        pitch_inv : jnp.ndarray
            Shape (num ρ, num pitch).
            1/λ values to compute the bounce integrals. 1/λ(ρ) is specified by
            ``pitch_inv[ρ]`` where in the latter the labels are interpreted
            as the indices that correspond to that field line.
        num_well : int
            Specify to return the first ``num_well`` pairs of bounce points for each
            pitch and field line. Choosing ``-1`` will detect all wells, but due
            to current limitations in JAX this will have worse performance.
            Specifying a number that tightly upper bounds the number of wells will
            increase performance. In general, an upper bound on the number of wells
            per toroidal transit is ``Aι+C`` where ``A``, ``C`` are the poloidal and
            toroidal Fourier resolution of B, respectively, in straight-field line
            PEST coordinates, and ι is the rotational transform normalized by 2π.
            A tighter upper bound than ``num_well=(Aι+C)*num_transit`` is preferable.
            The ``check_points`` or ``plot`` method is useful to select a reasonable
            value.

            This is the most important parameter to specify for performance.

            If there were fewer wells detected along a field line than the size of the
            last axis of the returned arrays, then that axis is padded with zero.

        Returns
        -------
        z1, z2 : tuple[jnp.ndarray]
            Shape (num ρ, num α, num pitch, num well).
            Tuple of length two (z1, z2) that stores ζ coordinates of bounce points.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of B.

            If there were less than ``num_well`` wells detected along a field line,
            then the last axis, which enumerates bounce points for a particular field
            line and pitch, is padded with zero.

        """
        num_field_periods = self._theta.X

        if isinstance(self._c["B(z)"], PiecewiseChebyshevSeries):
            if num_well is None:
                num_well = Options._guess_num_well(
                    num_field_periods,
                    self._NFP,
                    self._c["B(z)"].Y // 2,
                )
            # Skip Newton update since these points are exponentially accurate.
            z1, z2 = self._c["B(z)"].intersect1d(
                self._swap_pitch(pitch_inv), num_intersect=num_well, eps=_eps
            )
            z1 = move(z1)
            z2 = move(z2)
            return z1, z2

        if num_well is None:
            num_well = Options._guess_num_well(num_field_periods, self._NFP)
        pitch_inv = broadcast_for_bounce(pitch_inv)
        if self._nufft_eps < 1e-14:
            # FIXME: Newton update has only been implemented for nuffts; contributions
            # welcome : copy logic in desc/equilibrium/coords._map_poloidal_coordinates.
            return bounce_points(pitch_inv, self._c["knots"], self._c["B(z)"], num_well)

        return regular_points(self, pitch_inv, num_well)

    def check_points(self, points, pitch_inv, *, plot=True, **kwargs):
        """Check that bounce points are computed correctly.

        Parameters
        ----------
        points : tuple[jnp.ndarray]
            Shape (num ρ, num α, num pitch, num well).
            Output of method ``self.points``.
            Tuple of length two (z1, z2) that stores ζ coordinates of bounce points.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of B.
        pitch_inv : jnp.ndarray
            Shape (num ρ, num pitch).
            1/λ values to compute the bounce integrals. 1/λ(ρ) is specified by
            ``pitch_inv[ρ]`` where in the latter the labels are interpreted
            as the indices that correspond to that field line.
        plot : bool
            Whether to plot the field lines and bounce points of the given pitch angles.
        kwargs : dict
            Keyword arguments into
            ``desc/integrals/_bounce_utils.py::plot_ppoly``.

        Returns
        -------
        plots : list
            Matplotlib (fig, ax) tuples for the 1D plot of each field line.

        """
        kwargs = set_default_plot_kwargs(kwargs)
        if isinstance(self._c["B(z)"], PiecewiseChebyshevSeries):
            z1, z2 = points
            return self._c["B(z)"].check_intersect1d(
                move(z1, False),
                move(z2, False),
                self._swap_pitch(pitch_inv),
                plot=plot,
                **kwargs,
            )
        return check_bounce_points(
            *points,
            pitch_inv,
            self._c["knots"],
            self._c["B(z)"],
            plot=plot,
            **kwargs,
        )

    def integrate(
        self,
        integrand,
        pitch_inv,
        data=None,
        names=None,
        points=None,
        *,
        num_well=None,
        nufft_eps=1e-6,
        loop=False,
        quad=None,
        check=False,
        plot=False,
        **kwargs,
    ):
        """Bounce integrate ∫ f(ρ,α,λ,ℓ) dℓ.

        Computes the bounce integral ∫ f(ρ,α,λ,ℓ) dℓ for every field line and pitch.

        Warnings
        --------
        Make sure to replace √(1−λB) with √|1−λB| or clip the radicand
        to some value near machine precision when defining ``integrand``
        to account for imperfect computation of bounce points.

        Parameters
        ----------
        integrand : callable or list[callable]
            The composition operator on the set of functions in ``data``
            that determines ``f`` in ∫ f(ρ,α,λ,ℓ) dℓ. It should accept a dictionary
            which stores the interpolated data and the arguments ``B`` and ``pitch``.
        pitch_inv : jnp.ndarray
            Shape (num ρ, num pitch).
            1/λ values to compute the bounce integrals. 1/λ(ρ) is specified by
            ``pitch_inv[ρ]`` where in the latter the labels are interpreted
            as the indices that correspond to that field line.
        data : dict[str, jnp.ndarray]
            Shape (num ρ, num ζ, num θ).
            Real scalar-valued periodic functions in (θ, ζ) ∈ [0, 2π) × [0, 2π/NFP)
            evaluated on the ``grid`` supplied to construct this object.
            Use the method ``Bounce2D.reshape`` to reshape the data into the
            expected shape. If the input is not a real-valued array, then it
            is assumed that the Fourier transform as returned by ``Bounce2D.fourier``
            was given instead.
        names : str or list[str]
            Names in ``data`` to interpolate. Default is all keys in ``data``.
        points : tuple[jnp.ndarray]
            Shape (num ρ, num α, num pitch, num well).
            Optional, output of method ``self.points``.
            Tuple of length two (z1, z2) that stores ζ coordinates of bounce points.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of B.
        num_well : int
            See ``self.points`` for the description of this parameter.
        nufft_eps : float
            Precision requested for interpolation with non-uniform fast Fourier
            transform (NUFFT). If less than ``1e-14`` then NUFFT will not be used.
        loop : bool
            Whether to use loops to compute sums where a loop option is implemented.
            This is slower to differentiate with JAX.
            For best performance, one should only use this option if batching
            is already being done via ``Bounce2D.batch``  with ``surf_batch_size=1``.
        quad : tuple[jnp.ndarray]
            Optional quadrature points and weights. If given this overrides
            the quadrature chosen when this object was made.
        check : bool
            Flag for debugging. Must be false for JAX transformations.
        plot : bool
            Whether to plot the quantities in the integrand interpolated to the
            quadrature points of each integral. Ignored if ``check`` is false.

        Returns
        -------
        result : jnp.ndarray or list[jnp.ndarray]
            Shape (num ρ, num α, num pitch, num well).
            Last axis enumerates the bounce integrals for a given field line
            and pitch value.

        """
        x, w = setdefault(quad, self._quad)

        if not isinstance(integrand, (list, tuple)):
            integrand = [integrand]

        data = apply(
            data,
            _fourier_if_real,
            subset=names,
            exclude=("|B|", "B^zeta", "|e_zeta|r,a|", "zeta", "theta"),
        )

        if points is None:
            points = self.points(pitch_inv, num_well)

        pitch = 1 / pitch_inv
        # to broadcast with (..., num pitch, num well, num quad)
        if jnp.ndim(pitch) == 1:
            pitch = pitch[..., None, None]
        elif jnp.ndim(pitch) > 1:
            pitch = pitch[:, None, :, None, None]

        if nufft_eps < 1e-14:
            data = self._nummt(x, *points, data, loop)
        else:
            data = self._nufft(x, *points, data, loop, nufft_eps, pitch_inv)
        data["|e_zeta|r,a|"] = data["|B|"] / jnp.abs(data["B^zeta"])

        # Strictly increasing ζ knots enforces dζ > 0.
        # To retain dℓ = |B|/(B⋅∇ζ) dζ > 0 after fixing dζ > 0, we require
        # B⋅∇ζ > 0. This is equivalent to changing the sign of ∇ζ
        # or (∂ℓ/∂ζ)|ρ,a. Recall dζ = ∇ζ⋅dR ⇔ 1 = ∇ζ⋅(e_ζ|ρ,a).
        cov = grad_bijection_from_disc(*points)
        result = [
            (f(data, data["|B|"], pitch) * data["|e_zeta|r,a|"]).dot(w) * cov
            for f in integrand
        ]

        if check:
            check_interp(
                data["zeta"],
                jnp.reciprocal(data["|e_zeta|r,a|"]),
                data["|B|"],
                [data[k] for k in data if k not in ("zeta", "|e_zeta|r,a|", "|B|")],
                result,
                plot=plot,
            )

        return result[0] if len(result) == 1 else result

    def _nufft(self, x, z1, z2, data, loop, eps, pitch_inv):
        shape = (*z1.shape, x.size)

        z = flatten_mat(bijection_from_disc(x, z1[..., None], z2[..., None]), 3)
        t = flatten_mat(self._theta.eval1d(z, loop=loop))
        z = flatten_mat(z)
        # t and z have shape (num rho surfaces, num points on each surface)
        #            or just (                  num points on each surface).

        if _JF_BUG:
            mask = fill_value = None
        else:
            mask = flatten_mat(jnp.broadcast_to((z1 < z2)[..., None], shape), 4)
            fill_value = 0.5 * jnp.min(pitch_inv)

        c = nufft2d2r(
            z,
            t,
            jnp.concatenate([*data.values(), self._c["B^zeta"], self._c["|B|"]], -3),
            (0, 2 * jnp.pi / self._NFP),
            vec=True,
            eps=eps,
            mask=mask,
            fill_value=fill_value,
        )
        c = (
            c.reshape(len(data) + 2, *shape)
            if c.ndim == 2
            # reshape before swap to avoid memory copy
            else c.reshape(shape[0], len(data) + 2, *shape[1:]).swapaxes(0, 1)
        )

        data = dict(zip([*data.keys(), "B^zeta", "|B|"], c))
        data["zeta"] = z.reshape(shape)
        return data

    def _nummt(self, x, z1, z2, data, loop):
        shape = (*z1.shape, x.size)

        zeta = bijection_from_disc(x, z1[..., None], z2[..., None])
        z = flatten_mat(zeta, 3)
        t = self._theta.eval1d(z, loop=loop).reshape(*shape, 1)

        if isinstance(self._c["B(z)"], PiecewiseChebyshevSeries):
            # Using the same |B| that gave bounce points increases correlation
            # in discretization error in an open neighboorhood around the
            # current point in the optimization landscape, and hence removes
            # noise from optimization derivatives. For example, the difference
            # between the auto derivative and a 4 point finite difference
            # stencil at the optimal step size is reduced from 10³ to 10⁻⁶ for
            # Γ_c (which has the singular weight 1/v_∥).
            # Also uses less memory due to the dimension reduction.
            B = self._c["B(z)"].eval1d(z, loop=loop).reshape(shape)

        z = zeta[..., None]
        z = jnp.exp(1j * self._modes_z * z)
        t = jnp.exp(1j * self._modes_t * t)
        data = {name: mmt_for_bounce(z, t, c) for name, c in data.items()}
        data["B^zeta"] = mmt_for_bounce(z, t, self._c["B^zeta"])
        if isinstance(self._c["B(z)"], PiecewiseChebyshevSeries):
            data["|B|"] = B
        else:
            data["|B|"] = mmt_for_bounce(z, t, self._c["|B|"])
        data["zeta"] = zeta

        return data

    def interp_to_argmin(self, f, points, *, nufft_eps=1e-6, **kwargs):
        """Interpolate ``f`` to the deepest point pⱼ in magnetic well j.

        Let E = {ζ ∣ ζ₁ < ζ < ζ₂} and A ∈ argmin_E B(ζ). Returns f(A).

        Parameters
        ----------
        f : jnp.ndarray
            Shape (num ρ, num ζ, num θ).
            Real scalar-valued periodic function in (θ, ζ) ∈ [0, 2π) × [0, 2π/NFP)
            evaluated on the ``grid`` supplied to construct this object.
            Use the method ``Bounce2D.reshape`` to reshape the data into the
            expected shape. If the input is not a real-valued array, then it
            is assumed that the Fourier transform as returned by ``Bounce2D.fourier``
            was given instead.
        points : tuple[jnp.ndarray]
            Shape (num ρ, num α, num pitch, num well).
            Optional, output of method ``self.points``.
            Tuple of length two (z1, z2) that stores ζ coordinates of bounce points.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of B.
        nufft_eps : float
            Precision requested for interpolation with non-uniform fast Fourier
            transform (NUFFT). If less than ``1e-14`` then NUFFT will not be used.

        Returns
        -------
        f_j : jnp.ndarray
            Shape (num ρ, num α, num pitch, num well).
            ``f`` interpolated to the deepest point between ``points``.

        """
        f = _fourier_if_real(f)

        num_mins = kwargs.get("num_mins", -1)
        if isinstance(self._c["B(z)"], PiecewiseChebyshevSeries):
            bound = self._c["B(z)"].X * (self._c["B(z)"].Y // 2)
            num_mins = bound if (num_mins < 0) else min(num_mins, bound)

        # We set fill value to 0 since we chose our coordinates
        # such that all bounce points are at ζ >= 0; and therefore,
        # junk values in B_mins cannot be selected in argmin.
        mins, B_mins = (
            self._c["B(z)"].extrema1d(1, num_mins, fill_value=0.0, eps=_eps)
            if isinstance(self._c["B(z)"], PiecewiseChebyshevSeries)
            else get_mins(self._c["knots"], self._c["B(z)"], num_mins, fill_value=0.0)
        )
        t = self._theta.eval1d(mins)

        if nufft_eps < 1e-14:
            f = irfft2_mmt_pos(
                mins,
                t,
                f[..., None, :, :],
                self._num_z,
                self._num_t,
                (0, 2 * jnp.pi / self._NFP),
            )
        else:
            shape = (*mins.shape[:-2], -1)
            t = t.reshape(shape)
            f = nufft2d2r(
                mins.reshape(shape),
                t,
                f.squeeze(-3),
                (0, 2 * jnp.pi / self._NFP),
                eps=nufft_eps,
            ).reshape(mins.shape)

        return argmin(*points, f, mins, B_mins)

    def compute_fieldline_length(self, quad=None):
        """Compute the (mean) proper length of the field line ∫ dℓ / B.

        Parameters
        ----------
        quad : tuple[jnp.ndarray]
            Quadrature points xₖ and weights wₖ for the
            approximation of the integral ∫₋₁¹ f(x) dx ≈ ∑ₖ wₖ f(xₖ).
            Default is Gauss-Legendre quadrature on each field period along
            the field line.

        Returns
        -------
        length : jnp.ndarray
            Shape (num ρ, ).

        """
        warnings.warn(
            "This result will converge to "
            "(num field periods / 2π) * ∬_Ω abs(𝐁⋅∇ζ)⁻¹ dα dζ, "
            "where (α,ζ) ∈ Ω = [0, 2π/NFP)².\n"
            "This can be computed more efficiently as "
            '(num field periods / 2π) * eq.compute("V_psi") / eq.NFP.\n',
            DeprecationWarning,
        )

        if quad is None:
            deg = max(
                (
                    self._c["B(z)"].Y
                    if isinstance(self._c["B(z)"], PiecewiseChebyshevSeries)
                    else self._theta.Y
                ),
                8,
            )
            quad = leggauss(deg)
        x, w = quad

        # Let m, n denote the poloidal and toroidal Fourier resolution. We need to
        # compute a set of 2D Fourier series each on non-uniform tensor product grids
        # of size
        # |𝛉|×|𝛇| where |𝛉| = num α × num field periods × deg/z_eff and |𝛇| = z_eff.
        # Partial summation is more efficient than direct evaluation when
        # mn|𝛉||𝛇| > mn|𝛇| + m|𝛉||𝛇| or equivalently n|𝛉| > n + |𝛉|.

        if self._c["B^zeta"].shape[-2] == 1:  # axisymmetric
            z_eff = jnp.array([0.0], ndmin=2)
        else:
            z_eff = bijection_from_disc(x, *self._theta.domain)[:, None]

        B_sup_z = ifft_mmt(
            z_eff,
            self._c["B^zeta"],
            (0, 2 * jnp.pi / self._NFP),
            axis=-2,
            modes=self._modes_z,
        )[..., None, None, :, :]
        B_sup_z = irfft_mmt_pos(
            idct_mmt(x, self._theta.cheb[..., None, :]),
            B_sup_z,
            self._num_t,
            modes=self._modes_t,
        )

        # B⋅∇ζ never vanishes, so it has the same sign over a surface.
        # Simple mean over α because when ζ extends beyond one transit we need
        # to weight all field lines uniformly regardless of their area wrt α.
        dz_dx = jnp.pi / self._NFP
        return jnp.abs(jnp.reciprocal(B_sup_z).dot(w).sum(-1).mean(-1)) * dz_dx

    def plot(self, l, m, pitch_inv=None, **kwargs):
        """Plot B and bounce points on the specified field line.

        Parameters
        ----------
        l : int
            Index into the nodes of the grid supplied to make this object.
            The rho value corresponds to
            ``rho=grid.compress(grid.nodes[:,0])[l]``.
        m : int
            Index into the ``alpha`` array supplied to make this object.
            The alpha value corresponds to ``alpha[m]``.
        pitch_inv : jnp.ndarray
            Shape (num pitch, ).
            Optional, 1/λ values whose corresponding bounce points on the field line
            specified by Clebsch coordinate ρ(l), α(m) will be plotted.

        Returns
        -------
        fig, ax
            Matplotlib (fig, ax) tuple.

        """
        errorif(
            pitch_inv is not None and jnp.ndim(pitch_inv) > 1,
            msg=f"Got pitch_inv.ndim={jnp.ndim(pitch_inv)}, but expected 1.",
        )
        kwargs = set_default_plot_kwargs(kwargs, l, m)

        B = self._c["B(z)"]
        if isinstance(B, PiecewiseChebyshevSeries):
            domain = B.domain
            B = B.cheb
            if B.ndim == 4:
                B = B[l]
            if B.ndim == 3:
                B = B[m]
            B = PiecewiseChebyshevSeries(B, domain)
            if pitch_inv is not None:
                kwargs["z1"], kwargs["z2"] = B.intersect1d(pitch_inv, eps=_eps)
                kwargs["k"] = pitch_inv
            return B.plot1d(B.cheb, **kwargs)

        if B.ndim == 4:
            B = B[l]
        if B.ndim == 3:
            B = B[m]
        if pitch_inv is not None:
            kwargs["z1"], kwargs["z2"] = bounce_points(pitch_inv, self._c["knots"], B)
            kwargs["k"] = pitch_inv
        return plot_ppoly(PPoly(B.T, self._c["knots"]), **kwargs)

    def plot_theta(self, l, m, **kwargs):
        """Plot θ on the specified field line.

        Parameters
        ----------
        l : int
            Index into the nodes of the grid supplied to make this object.
            The rho value corresponds to
            ``rho=grid.compress(grid.nodes[:,0])[l]``.
        m : int
            Index into the ``alpha`` array supplied to make this object.
            The alpha value corresponds to ``alpha[m]``.

        Returns
        -------
        fig, ax
            Matplotlib (fig, ax) tuple.

        """
        domain = self._theta.domain
        theta = self._theta.cheb
        if theta.ndim == 4:
            theta = theta[l]
        if theta.ndim == 3:
            theta = theta[m]
        theta = PiecewiseChebyshevSeries(theta, domain)
        kwargs.setdefault(
            "title",
            rf"$\theta \text{{ mod }} (2 \pi)$ "
            rf"on field line $(\rho_{{l={l}}}, \alpha_{{m={m}}})$",
        )
        kwargs.setdefault("vlabel", r"$\theta \text{ mod } (2 \pi)$")
        return theta.plot1d(theta.cheb, **set_default_plot_kwargs(kwargs, l, m))

    @staticmethod
    def plot_angle_spectrum(
        angle,
        l,
        *,
        truncate=0,
        norm=LogNorm(1e-7),
        h_ax_numticks=None,
        v_ax_numticks=None,
        **kwargs,
    ):
        """Plot frequency spectrum of the given inverse stream map.

        Parameters
        ----------
        angle : jnp.ndarray
            Shape (num ρ, X, Y).
            Angle returned by ``Bounce2D.angle``.
        l : int
            Index into first axis of ``angle``.
        truncate : int
            Index at which to truncate any Chebyshev series.
            This will remove aliasing error at the shortest wavelengths where the signal
            to noise ratio is lowest. The default value is zero which is interpreted as
            no truncation.
        norm : str
            The normalization method used for the color scale.
            See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html.
            Default is logarithmic scale with cutoff at ``1e-7``.
        h_ax_numticks : int
            If given, labels at most ``h_ax_numticks`` marks on the horizontal axis.
        v_ax_numticks : int
            If given, labels at most ``v_ax_numticks`` marks on the vertical axis.
        kwargs
            Keyword arguments to pass to ``matplotlib``.

        Returns
        -------
        fig, ax
            Matplotlib (fig, ax) tuple.

        """
        kwargs = kwargs.copy()
        kwargs.setdefault("fignum", 0)
        kwargs.setdefault("cmap", "turbo")
        fig, ax = plt.subplots()

        angle = angle[l]
        X, Y = angle.shape

        name = kwargs.pop("name", "delta")
        if name == "delta":
            title = kwargs.pop(
                "title",
                r"Projection of "
                r"$\alpha, \zeta \mapsto \theta - \alpha$ onto "
                r"$\{e^{i x \alpha} T_y(N_{\text{FP}} \zeta / \pi - 1)\}$"
                r"$_{\text{Fourier-Chebyshev}}$ "
                rf"on $\rho_{{l={l}}}$",
            )

            c = FourierChebyshevSeries(angle, (jnp.nan, jnp.nan), truncate=truncate)._c
            c = cheb_from_dct(
                c.at[..., (0, -1) if (X % 2 == 0) else 0, :].divide(2) * 2
            )

        elif name == "lambda":
            title = kwargs.pop(
                "title",
                "Projection of "
                r"$\vartheta, \zeta \mapsto \theta - \alpha - \iota \zeta$ onto "
                r"$\{e^{i x \alpha} e^{i y N_{\text{FP}} \zeta}\}_{\text{Fourier}}$ "
                rf"on $\rho_{{l={l}}}$",
            )
            ax.set_xticks(
                jnp.arange(Y),
                jnp.fft.fftshift(jnp.fft.fftfreq(Y, 1 / Y).astype(int)),
            )

            c = Bounce2D.fourier(-angle.T).squeeze(0).T
            c = jnp.fft.fftshift(c, -1)

        c = jnp.abs(c)

        ax.set(xlabel=kwargs.pop("xlabel", r"$y$"), ylabel=kwargs.pop("ylabel", r"$x$"))
        ax.set_title(title, pad=kwargs.pop("pad", 20))
        plt.matshow(c, norm=norm, **kwargs)
        cbar = plt.colorbar(orientation="horizontal")
        cbar.ax.invert_xaxis()

        if h_ax_numticks is not None:
            ax.xaxis.set_major_locator(MaxNLocator(h_ax_numticks, integer=True))
        if v_ax_numticks is not None:
            ax.yaxis.set_major_locator(MaxNLocator(v_ax_numticks, integer=True))

        return fig


def _fourier_if_real(thing):
    return Bounce2D.fourier(thing) if jnp.isrealobj(thing) else thing


class Bounce1D(Bounce):
    """Computes bounce integrals using one-dimensional local spline methods.

    The bounce integral is defined as ∫ f(ρ,α,λ,ℓ) dℓ where

    * dℓ parametrizes the distance along the field line in meters.
    * f(ρ,α,λ,ℓ) is the quantity to integrate along the field line.
    * The boundaries of the integral are bounce points ℓ₁, ℓ₂ s.t. λB(ρ,α,ℓᵢ) = 1.
    * λ is a constant defining the integral proportional to the magnetic moment
      over energy.
    * B is the norm of the magnetic field.

    For a particle with fixed λ, bounce points are defined to be the location on the
    field line such that the particle's velocity parallel to the magnetic field is zero.
    The bounce integral is defined up to a sign. We choose the sign that corresponds to
    the particle's guiding center trajectory traveling in the direction of increasing
    field-line-following coordinate ζ.

    Examples
    --------
      * ``tests/test_integrals.py::TestBounce::test_bounce1d_checks``
      * ``desc/compute/_old.py::_epsilon_32_1D``
      * ``desc/compute/_old.py::_Gamma_c_1D``

    See Also
    --------
    Bounce2D
        ``Bounce2D`` uses 2D pseudo-spectral methods for the same task.
        The function approximation in ``Bounce1D`` is ignorant
        that the objects to approximate are defined on a bounded subset of ℝ².
        The domain is projected to ℝ, where information sampled about the function
        at infinity cannot support reconstruction of the function near the origin.
        As the functions of interest do not vanish at infinity, pseudo-spectral
        techniques are not used. Instead, function approximation is done with local
        splines. This is useful if one can efficiently obtain data along field lines
        and the number of toroidal transits to follow a field line is not large.

    Parameters
    ----------
    grid : Grid
        Tensor-product grid in (ρ, α, ζ) Clebsch coordinates.
        The ζ coordinates (the unique values prior to taking the tensor-product)
        must be strictly increasing and preferably uniformly spaced. These are used
        as knots to construct splines. A reference knot density is 100 knots per
        toroidal transit. Also, the minimum value of the zeta coordinate must be
        greater than the sentinel value of ``-1e5``. If this requirement is limiting
        make a GitHub issue requesting to lower this value.
    data : dict[str, jnp.ndarray]
        Data evaluated on ``grid``.
        Must include names in ``Bounce1D.required_names``.
    quad : tuple[jnp.ndarray]
        Quadrature points xₖ and weights wₖ for the approximation of an
        integral ∫₋₁¹ g(x) dx = ∑ₖ wₖ g(xₖ). Default is 32 points.
    automorphism : tuple[Callable] or None
        The first callable should be an automorphism of the real interval [-1, 1].
        The second callable should be the derivative of the first. This map defines
        a change of variable for the bounce integral. The choice made for the
        automorphism will affect the performance of the quadrature.
    Bref : float
        Optional. Reference magnetic field strength for normalization.
    Lref : float
        Optional. Reference length scale for normalization.
    check : bool
        Flag for debugging. Must be false for JAX transformations.

    """

    required_names = ["B^zeta", "B^zeta_z|r,a", "|B|", "|B|_z|r,a"]
    """Required keys in the ``data`` dictionary given to the ``__init__`` method."""

    _quad: tuple[jax.Array]
    _data: dict[str, jax.Array]
    _zeta: jax.Array
    _B: jax.Array

    def __init__(
        self,
        grid,
        data,
        quad=None,
        *,
        automorphism=None,
        Bref=1.0,
        Lref=1.0,
        check=False,
        **kwargs,
    ):
        """Returns an object to compute bounce integrals."""
        assert grid.is_meshgrid

        if quad is None:
            quad = jax.lax.stop_gradient(
                get_quadrature(leggauss(32), (automorphism_sin, grad_automorphism_sin))
            )
        else:
            quad = get_quadrature(quad, automorphism)
        self._quad = quad

        self._data = {
            "|b^zeta|": jnp.abs(data["B^zeta"]) * Lref / data["|B|"],
            "|B|": data["|B|"] / Bref,
            "|B|_z|r,a": data["|B|_z|r,a"] / Bref,
        }
        self._data["|b^zeta|_z|r,a"] = (
            data["B^zeta_z|r,a"] * jnp.sign(data["B^zeta"]) * Lref
            - self._data["|b^zeta|"] * data["|B|_z|r,a"]
        ) / data["|B|"]

        # Figure out if input is split into batches.
        s = data["|B|"].shape
        is_reshaped = len(s) > 1 and s[-2] == grid.num_alpha and s[-1] == grid.num_zeta
        if not is_reshaped:
            for name in self._data:
                self._data[name] = Bounce1D.reshape(grid, self._data[name])

        self._zeta = jnp.asarray(grid.compress(grid.nodes[:, 2], surface_label="zeta"))
        self._B = jnp.moveaxis(
            CubicHermiteSpline(
                x=self._zeta,
                y=self._data["|B|"],
                dydx=self._data["|B|_z|r,a"],
                axis=-1,
                check=check,
            ).c,
            (0, 1),
            (-1, -2),
        )

    @staticmethod
    def batch(fun, fun_data, desc_data, grid, surf_batch_size=1, sparse=True):
        """Compute function ``fun`` over phase space in batches.

        This is a utility method to compute some function of bounce integrals
        over the phase space efficiently. You may want to also JIT compile your
        code which calls this utility method.

        Examples
        --------
          * ``desc/compute/_old.py::_epsilon_32_1D``
          * ``desc/compute/_old.py::_Gamma_c_1D``

        Parameters
        ----------
        fun : callable
            A function  which takes a single argument ``fun_data`` and computes
            bounce integrals assuming ``fun_data`` holds all required quantities
            to construct a ``Bounce1D`` operator as well as call its methods.
        fun_data : dict[str, jnp.ndarray]
            Data to reshape, interpolate, and pass to ``fun``.
            The structure of the data should match the structure
            returned by the registered compute functions in ``desc.compute``.
            Note this dictionary will be modified.
        desc_data : dict[str, jnp.ndarray]
            Data dictionary with the same structure as the data returned by the
            functions in ``desc.compute``.
        grid : Grid
            Grid on which ``fun_data`` and ``desc_data`` were computed.
        surf_batch_size : int
            Number of flux surfaces with which to compute simultaneously.
            Default is ``1``.
        sparse : bool
            Whether to differentiate with sparsity preserving pullbacks.
            Default is ``True``, which makes the most sense if the output has
            shape (num_rho, ). Otherwise, if the output shape is larger, and
            the final objective of interest is a lower dimensional quantity
            than the output, it may be preferable to delay the vjp
            by setting to ``False``.

        Returns
        -------
        The output ``fun(fun_data)``.

        """
        for name in Bounce1D.required_names:
            fun_data[name] = desc_data[name]
        for name in fun_data:
            fun_data[name] = Bounce1D.reshape(grid, fun_data[name])
        fun_data["min_tz |B|"] = grid.compress(desc_data["min_tz |B|"])
        fun_data["max_tz |B|"] = grid.compress(desc_data["max_tz |B|"])

        if sparse:
            return sparse_pullback(fun, fun_data, surf_batch_size, strip_dim0=True)

        return batch_map(fun, fun_data, surf_batch_size, strip_dim0=True)

    @staticmethod
    def reshape(grid, f):
        """Reshape arrays for acceptable input to ``integrate``.

        Parameters
        ----------
        grid : Grid
            Tensor-product grid in (ρ, α, ζ) Clebsch coordinates.
        f : jnp.ndarray
            Data evaluated on grid.

        Returns
        -------
        f : jnp.ndarray
            Shape (num ρ, num α, num ζ).
            Reshaped data which may be given to ``integrate``.

        """
        return grid.meshgrid_reshape(f, "raz")

    def points(self, pitch_inv, num_well=None):
        """Compute bounce points.

        Parameters
        ----------
        pitch_inv : jnp.ndarray
            Shape (num ρ, num pitch).
            1/λ values to compute the bounce integrals. 1/λ(ρ) is specified by
            ``pitch_inv[ρ]`` where in the latter the labels are interpreted
            as the indices that correspond to that field line.
        num_well : int
            Specify to return the first ``num_well`` pairs of bounce points for each
            pitch and field line. Choosing ``-1`` will detect all wells, but due
            to current limitations in JAX this will have worse performance.
            Specifying a number that tightly upper bounds the number of wells will
            increase performance. In general, an upper bound on the number of wells
            per toroidal transit is ``Aι+C`` where ``A``, ``C`` are the poloidal and
            toroidal Fourier resolution of B, respectively, in straight-field line
            PEST coordinates, and ι is the rotational transform normalized by 2π.
            A tighter upper bound than ``num_well=(Aι+C)*num_transit`` is preferable.
            The ``check_points`` or ``plot`` method is useful to select a reasonable
            value.

            If there were fewer wells detected along a field line than the size of the
            last axis of the returned arrays, then that axis is padded with zero.

        Returns
        -------
        z1, z2 : tuple[jnp.ndarray]
            Shape (num ρ, num α, num pitch, num well).
            Tuple of length two (z1, z2) that stores ζ coordinates of bounce points.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of B.

            If there were less than ``num_well`` wells detected along a field line,
            then the last axis, which enumerates bounce points for a particular field
            line and pitch, is padded with zero.

        """
        return bounce_points(
            broadcast_for_bounce(pitch_inv), self._zeta, self._B, num_well
        )

    def check_points(self, points, pitch_inv, *, plot=True, **kwargs):
        """Check that bounce points are computed correctly.

        Parameters
        ----------
        points : tuple[jnp.ndarray]
            Shape (num ρ, num α, num pitch, num well).
            Output of method ``self.points``.
            Tuple of length two (z1, z2) that stores ζ coordinates of bounce points.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of B.
        pitch_inv : jnp.ndarray
            Shape (num ρ, num pitch).
            1/λ values to compute the bounce integrals. 1/λ(ρ) is specified by
            ``pitch_inv[ρ]`` where in the latter the labels are interpreted
            as the indices that correspond to that field line.
        plot : bool
            Whether to plot the field lines and bounce points of the given pitch angles.
        kwargs
            Keyword arguments into ``desc/integrals/_bounce_utils.py::plot_ppoly``.

        Returns
        -------
        plots : list
            Matplotlib (fig, ax) tuples for the 1D plot of each field line.

        """
        return check_bounce_points(
            *points, pitch_inv, self._zeta, self._B, plot=plot, **kwargs
        )

    def integrate(
        self,
        integrand,
        pitch_inv,
        data=None,
        names=None,
        points=None,
        *,
        num_well=None,
        method="cubic",
        quad=None,
        check=False,
        plot=False,
        **kwargs,
    ):
        """Bounce integrate ∫ f(ρ,α,λ,ℓ) dℓ.

        Computes the bounce integral ∫ f(ρ,α,λ,ℓ) dℓ for every field line and pitch.

        Warnings
        --------
        Make sure to replace √(1−λB) with √|1−λB| or clip the radicand
        to some value near machine precision when defining ``integrand``
        to account for imperfect computation of bounce points.

        Parameters
        ----------
        integrand : callable or list[callable]
            The composition operator on the set of functions in ``data``
            that determines ``f`` in ∫ f(ρ,α,λ,ℓ) dℓ. It should accept a dictionary
            which stores the interpolated data and the arguments ``B`` and ``pitch``.
        pitch_inv : jnp.ndarray
            Shape (num ρ, num pitch).
            1/λ values to compute the bounce integrals. 1/λ(ρ) is specified by
            ``pitch_inv[ρ]`` where in the latter the labels are interpreted
            as the indices that correspond to that field line.
        data : dict[str, jnp.ndarray]
            Shape (num ρ, num α, num ζ).
            Real scalar-valued functions evaluated on the ``grid`` supplied to
            construct this object. Use the method ``Bounce1D.reshape`` to reshape
            the data into the expected shape.
        names : str or list[str]
            Names in ``data`` to interpolate. Default is all keys in ``data``.
        points : tuple[jnp.ndarray]
            Shape (num ρ, num α, num pitch, num well).
            Optional, output of method ``self.points``.
            Tuple of length two (z1, z2) that stores ζ coordinates of bounce points.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of B.
        num_well : int or None
            See ``self.points`` for the description of this parameter.
        method : str
            Method of interpolation.
            See https://interpax.readthedocs.io/en/latest/_api/interpax.interp1d.html.
            Default is cubic C1 local spline.
        quad : tuple[jnp.ndarray]
            Optional quadrature points and weights. If given this overrides
            the quadrature chosen when this object was made.
        check : bool
            Flag for debugging. Must be false for JAX transformations.
        plot : bool
            Whether to plot the quantities in the integrand interpolated to the
            quadrature points of each integral. Ignored if ``check`` is false.

        Returns
        -------
        result : jnp.ndarray or list[jnp.ndarray]
            Shape (num ρ, num α, num pitch, num well).
            Last axis enumerates the bounce integrals for a given field line
            and pitch value.

        """
        x, w = setdefault(quad, self._quad)

        if not isinstance(integrand, (list, tuple)):
            integrand = [integrand]

        data = apply(data, subset=names, exclude=("|B|",))

        if points is None:
            points = self.points(pitch_inv, num_well)
        z1, z2 = points

        pitch = broadcast_for_bounce(1 / pitch_inv)[..., None, None]

        shape = (*z1.shape, x.size)  # (..., num pitch, num well, num quad)

        z = flatten_mat(bijection_from_disc(x, z1[..., None], z2[..., None]), 3)

        b_sup_z = interp1d_Hermite_vec(
            z,
            self._zeta,
            self._data["|b^zeta|"],
            self._data["|b^zeta|_z|r,a"],
        ).reshape(shape)
        B = interp1d_Hermite_vec(
            z,
            self._zeta,
            self._data["|B|"],
            self._data["|B|_z|r,a"],
        ).reshape(shape)
        data = {
            k: interp1d_vec(z, self._zeta, v, method=method).reshape(shape)
            for k, v in data.items()
        }

        # Strictly increasing ζ knots enforces dζ > 0.
        # To retain dℓ = |B|/(B⋅∇ζ) dζ > 0 after fixing dζ > 0, we require
        # B⋅∇ζ > 0. This is equivalent to changing the sign of ∇ζ
        # or (∂ℓ/∂ζ)|ρ,a. Recall dζ = ∇ζ⋅dR ⇔ 1 = ∇ζ⋅(e_ζ|ρ,a).
        cov = grad_bijection_from_disc(z1, z2)
        result = [(f(data, B, pitch) / b_sup_z).dot(w) * cov for f in integrand]

        if check:
            check_interp(
                z.reshape(shape),
                b_sup_z,
                B,
                data.values(),
                result,
                plot=plot,
            )

        return result[0] if len(result) == 1 else result

    def interp_to_argmin(self, f, points, *, method="cubic"):
        """Interpolate ``f`` to the deepest point pⱼ in magnetic well j.

        Let E = {ζ ∣ ζ₁ < ζ < ζ₂} and A ∈ argmin_E B(ζ). Returns f(A).

        Parameters
        ----------
        f : jnp.ndarray
            Shape (num ρ, num α, num ζ).
            Real scalar-valued functions evaluated on the ``grid`` supplied to
            construct this object. Use the method ``Bounce1D.reshape`` to
            reshape the data into the expected shape.
        points : tuple[jnp.ndarray]
            Shape (num ρ, num α, num pitch, num well).
            Optional, output of method ``self.points``.
            Tuple of length two (z1, z2) that stores ζ coordinates of bounce points.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of B.
        method : str
            Method of interpolation.
            See https://interpax.readthedocs.io/en/latest/_api/interpax.interp1d.html.
            Default is cubic C1 local spline.

        Returns
        -------
        f_j : jnp.ndarray
            Shape (num ρ, num α, num pitch, num well).
            ``f`` interpolated to the deepest point between ``points``.

        """
        # We set fill value to sentinel since all bounce points are at
        # ζ > sentinel (as documented in Bounce1D docstring); and
        # therefore, junk values in B_mins cannot be selected in argmin.
        mins, B_mins = get_mins(self._zeta, self._B, fill_value=_sentinel)
        return argmin(
            *points, interp1d_vec(mins, self._zeta, f, method=method), mins, B_mins
        )

    def plot(self, l, m, pitch_inv=None, **kwargs):
        """Plot B and bounce points on the specified field line.

        Parameters
        ----------
        l, m : int
            Indices into the nodes of the grid supplied to make this object.
            ``rho,alpha=Bounce1D.reshape(grid,grid.nodes[:,:2])[l,m,0]``.
        pitch_inv : jnp.ndarray
            Shape (num pitch, ).
            Optional, 1/λ values whose corresponding bounce points on the field line
            specified by Clebsch coordinate ρ(l), α(m) will be plotted.
        kwargs
            Keyword arguments into ``desc/integrals/_bounce_utils.py::plot_ppoly``.

        Returns
        -------
        fig, ax
            Matplotlib (fig, ax) tuple.

        """
        B = self._B
        if B.ndim == 4:
            B = B[l]
        if B.ndim == 3:
            B = B[m]
        if pitch_inv is not None:
            errorif(
                jnp.ndim(pitch_inv) > 1,
                msg=f"Got pitch_inv.ndim={jnp.ndim(pitch_inv)}, but expected 1.",
            )
            kwargs["z1"], kwargs["z2"] = bounce_points(pitch_inv, self._zeta, B)
            kwargs["k"] = pitch_inv
        fig, ax = plot_ppoly(
            PPoly(B.T, self._zeta), **set_default_plot_kwargs(kwargs, l, m)
        )
        return fig, ax


class Options(NamedTuple):
    """Parameter container for Bounce2D."""

    # TODO(#2152): Consider instead of having users pass in 10 kwargs
    #    have them pass in this object, e.g. eq.compute(bounce_opts=opts).
    #    Reasons: 1) eq.compute kwarg namespace is polluted;
    #    e.g. some other compute function can't use the kwarg spline.
    #    2) Long kwarg descriptions aren't readable in the public list
    #       of variables docs.
    _doc = {
        "angle": """jnp.ndarray :
            Shape (num rho, X, Y).
            Angle returned by ``Bounce2D.angle``.
            """,
        "Y_B": """int :
            Desired resolution for algorithm to compute bounce points.
            If the option ``spline`` is ``True``, the bounce points are found with
            8th order accuracy in this parameter. If the option ``spline`` is ``False``,
            then the bounce points are found with spectral accuracy in this parameter.
            A reference value is ``(grid.num_theta+grid.num_zeta)//2``.

            An error of ε in a bounce point manifests
            𝒪(ε¹ᐧ⁵) error in bounce integrals with (v_∥)¹ and
            𝒪(ε⁰ᐧ⁵) error in bounce integrals with (v_∥)⁻¹.
            """,
        "alpha": """jnp.ndarray :
            Shape (num alpha, ).
            Starting field line poloidal labels.
            Default is single field line. To compute a surface average
            on a rational surface, it is necessary to average over multiple
            field lines until the surface is covered sufficiently.
            """,
        "num_field_periods": """int :
            Number of field periods to follow field line.
            In an axisymmetric device, field line integration over a single poloidal
            transit is sufficient to capture a surface average. For a 3D
            configuration, more transits will approximate surface averages on an
            irrational magnetic surface better, with diminishing returns.
            """,
        "num_well": """int :
            Maximum number of wells to detect for each pitch and field line.
            Giving ``-1`` will detect all wells but due to current limitations in
            JAX this will have worse performance.
            Specifying a number that tightly upper bounds the number of wells will
            increase performance. In general, an upper bound on the number of wells
            per toroidal transit is ``Aι+C`` where ``A``, ``C`` are the poloidal and
            toroidal Fourier resolution of B, respectively, in straight-field line
            PEST coordinates, and ι is the rotational transform normalized by 2π.
            A tighter upper bound than ``num_well=(Aι+C)*num_transit`` is preferable.
            The ``check_points`` or ``plot`` methods in ``desc.integrals.Bounce2D``
            are useful to select a reasonable value.

            This is the most important parameter to specify for performance.
            """,
        "num_quad": """int :
            Resolution for quadrature of bounce integrals.
            Default is 32. This parameter is ignored if given ``quad``.
            """,
        "num_pitch": """int :
            Resolution for quadrature over velocity coordinate.
            """,
        "pitch_batch_size": """int :
            Number of pitch values with which to compute simultaneously.
            If given ``None``, then ``pitch_batch_size`` is ``num_pitch``.
            Default is ``num_pitch``.
            """,
        "surf_batch_size": """int :
            Number of flux surfaces with which to compute simultaneously.
            If given ``None``, then ``surf_batch_size`` is ``grid.num_rho``.
            Default is ``1``.
            Only consider increasing if ``pitch_batch_size`` is ``None``.
            """,
        "nufft_eps": """float :
            Precision requested for interpolation with non-uniform fast Fourier
            transform (NUFFT). If less than ``1e-14`` then NUFFT will not be used.
            """,
        "spline": """bool :
            Whether to use cubic splines to compute initial guess for bounce points
            instead of Chebyshev series. Default is ``True``. It can be preferable
            to set to ``False`` on equilibria with high ``NFP``, (such cases make
            smaller ``Y_B`` feasible), or on GPUs where eigenvalue solves are fast.
            """,
        "quad": """tuple[jnp.ndarray] :
            Used to compute bounce integrals.
            Quadrature points xₖ and weights wₖ for the
            approximation of the integral ∫₋₁¹ f(x) dx ≈ ∑ₖ wₖ f(xₖ).
            """,
        "_vander": """dict[str,jnp.ndarray] :
            Precomputed transform matrix "dct spline".
            This private parameter is intended to be used only by
            developers for objectives.
            """,
        "theta": "",
    }

    _static_argnames = (
        "nufft_eps",
        "num_field_periods",
        "num_pitch",
        "num_quad",
        "num_well",
        "pitch_batch_size",
        "spline",
        "surf_batch_size",
        "Y_B",
    )

    alpha: jnp.ndarray
    loop: bool
    nufft_eps: float
    num_field_periods: int
    num_well: int
    pitch_batch_size: int
    pitch_quad: tuple[jnp.ndarray]
    quad: tuple[jnp.ndarray]
    spline: bool
    surf_batch_size: int
    vander: tuple[jnp.ndarray]
    Y_B: int

    @classmethod
    def guess(cls, eta, grid, **kwargs):
        """Guess parameters based on eta and grid if not given in kwargs.

        Parameters
        ----------
        eta : int
            The number η ∈ {−1, 1} denoting which factor (v_∥)^η matches the
            behavior of the integrand near the bounce points. If η ∉ {-1, 1},
            then a quadrature that works for all η ∈ {−1, 0, 1} will be used.
        grid : Grid
            Tensor-product grid in (ρ, θ, ζ) with uniformly spaced nodes
            (θ, ζ) ∈ [0, 2π) × [0, 2π/NFP).

        """
        pitch_batch_size = kwargs.get("pitch_batch_size", None)
        surf_batch_size = kwargs.get("surf_batch_size", 1)
        errorif(
            (surf_batch_size > 1) and (pitch_batch_size is not None),
            msg=f"Expected pitch_batch_size to be None, got {pitch_batch_size}.",
        )

        alpha = kwargs.get("alpha", jnp.array([0.0]))
        num_field_periods = kwargs.get("num_field_periods", 20)

        quad = kwargs.get("quad", None)
        if quad is None:
            quad = Options._quad(eta, kwargs.get("num_quad", 32))
        quad = jax.lax.stop_gradient(quad)

        if eta == 1:
            nufft_eps = kwargs.get("nufft_eps", 1e-6)
            num_pitch = kwargs.get("num_pitch", 51)
        else:
            nufft_eps = kwargs.get("nufft_eps", 1e-7)
            num_pitch = kwargs.get("num_pitch", 65)
        pitch_quad = jax.lax.stop_gradient(simpson2(num_pitch))

        spline = kwargs.get("spline", True)
        Y_B = kwargs.get("Y_B", Options._guess_Y_B(grid))
        num_well = kwargs.get(
            "num_well",
            Options._guess_num_well(
                num_field_periods,
                grid.NFP,
                Y_B if spline else (Y_B // 2),
            ),
        )

        return cls(
            alpha=alpha,
            loop=kwargs.get("loop", False),
            nufft_eps=nufft_eps,
            num_field_periods=num_field_periods,
            num_well=num_well,
            pitch_batch_size=pitch_batch_size,
            pitch_quad=pitch_quad,
            quad=quad,
            spline=spline,
            surf_batch_size=surf_batch_size,
            vander=kwargs.get("_vander", None),
            Y_B=Y_B,
        )

    def keys(self):
        """Names of elements in tuple."""
        return self._fields

    def __getitem__(self, key):
        """Lookup by string or index."""
        return getattr(self, key) if isinstance(key, str) else tuple.__getitem__(key)

    @staticmethod
    def _quad(eta, num_quad):
        if eta == 1:
            return chebgauss2(num_quad)
        if eta == -1:
            return chebgauss1(num_quad)
        return get_quadrature(
            leggauss(num_quad), (automorphism_sin, grad_automorphism_sin)
        )

    @staticmethod
    def _guess_num_well(num_field_periods, NFP, mins_per_field_period=None):
        """Guess upper bound for number of wells based on spectrum.

        Parameters
        ----------
        num_field_periods : int
            Number of field periods to follow field line.
        NFP : int
            Number of field periods per toroidal transit.
        mins_per_field_period : int
            An upper bound for the number of minima of B, (and hence number of wells),
            per field period. For splines this is the number of knots per field period.
            For Chebyshev series, this is the max degree floor division by 2.

        Returns
        -------
        num_well : int
            A guess for the max number of wells that exist for any pitch angle
            or field line after following it for the specified length.
            The guess will ideally be more conservative than
            ``num_field_periods*mins_per_field_period`` to enhance performance
            (due to limitations in JAX), yet sill remain loose enough that all
            wells are always detected.

        """
        # e.g. heliotron with nfp 19 needs num field periods * 2
        num_well = round(num_field_periods * (1 + 20/NFP))
        return (
            num_well
            if mins_per_field_period is None
            else min(num_well, num_field_periods * mins_per_field_period)
        )

    @staticmethod
    def _guess_Y_B(grid):
        """Guess Y_B from grid resolution.

        Parameters
        ----------
        grid : Grid
            Tensor-product grid in (ρ, θ, ζ) with uniformly spaced nodes
            (θ, ζ) ∈ [0, 2π) × [0, 2π/NFP).

        """
        return (grid.num_theta + grid.num_zeta) // 2

    @staticmethod
    def _build_objective(o, names, eta):
        """Builds the objective, selecting default values if they were not specified.

        Examples
        --------
          * ``desc/objectives/_fast_ion.py::GammaC``
          * ``desc/objectives/_neoclassical.py::EffectiveRipple``

        Parameters
        ----------
        o : _Objective
            The objective instance.
        names : str
            Builds profiles and transforms for the compute quantities registered
            with these names.
        eta : int
            The number η ∈ {−1, 1} denoting which factor (v_∥)^η matches the
            behavior of the integrand near the bounce points. If η ∉ {-1, 1},
            then a quadrature that works for all η ∈ {−1, 0, 1} will be used.

        """
        from desc.compute import get_profiles, get_transforms
        from desc.objectives.utils import _parse_callable_target_bounds

        eq = o.things[0]
        if o._grid is None:
            o._grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
        assert o._grid.can_fft2

        X = o._hyperparam.pop("X")
        Y = o._hyperparam.pop("Y")
        o._constants["x"] = fourier_pts(X)
        o._constants["y"] = cheb_pts(Y, (0, 2 * jnp.pi / eq.NFP))[::-1]

        Y_B = o._hyperparam["Y_B"]
        if Y_B is None:
            o._hyperparam["Y_B"] = Y_B = Options._guess_Y_B(o._grid)
        if o._hyperparam["num_well"] is None:
            o._hyperparam["num_well"] = Options._guess_num_well(
                o._hyperparam["num_field_periods"],
                eq.NFP,
                Y_B if o._hyperparam["spline"] else (Y_B // 2),
            )

        o._constants["_vander"] = (
            {
                "dct spline": chebvander(
                    jnp.linspace(-1, 1, Y_B, endpoint=False), truncate_rule(Y) - 1
                )
            }
            if o._hyperparam["spline"]
            else {}
        )
        o._constants["quad"] = Options._quad(eta, o._hyperparam.pop("num_quad"))

        rho = o._grid.compress(o._grid.nodes[:, 0])
        o._constants["lambda"] = get_transforms(
            "lambda",
            eq,
            grid=LinearGrid(
                rho=rho, M=eq.L_basis.M, zeta=o._constants["y"], NFP=eq.NFP
            ),
        )["L"]
        o._constants["profiles"] = get_profiles(names, eq, grid=o._grid)
        o._constants["transforms"] = get_transforms(names, eq, grid=o._grid)
        o._dim_f = o._grid.num_rho
        o._target, o._bounds = _parse_callable_target_bounds(o._target, o._bounds, rho)
