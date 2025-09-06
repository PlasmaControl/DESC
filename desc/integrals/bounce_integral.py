"""Methods for computing bounce integrals (singular or otherwise)."""

import warnings
from abc import ABC, abstractmethod

from interpax import CubicHermiteSpline, PPoly
from orthax.legendre import leggauss

from desc.backend import jnp, rfft2
from desc.batching import batch_map
from desc.grid import LinearGrid
from desc.integrals._bounce_utils import (
    _broadcast_for_bounce,
    _check_bounce_points,
    _check_interp,
    _mmt_for_bounce,
    _move,
    _set_default_plot_kwargs,
    argmin,
    bounce_points,
    fast_chebyshev,
    fast_cubic_spline,
    fourier_chebyshev,
    get_extrema,
    plot_ppoly,
)
from desc.integrals._interp_utils import (
    _irfft2_mmt,
    cheb_pts,
    fourier_pts,
    idct_mmt,
    ifft_mmt,
    interp1d_Hermite_vec,
    interp1d_vec,
    irfft_mmt,
    nufft2d2r,
    polyder_vec,
    rfft2_modes,
    rfft2_vander,
)
from desc.integrals.basis import PiecewiseChebyshevSeries
from desc.integrals.quad_utils import (
    automorphism_sin,
    bijection_from_disc,
    get_quadrature,
    grad_automorphism_sin,
    grad_bijection_from_disc,
    simpson2,
    uniform,
)
from desc.io import IOAble
from desc.utils import apply, atleast_nd, errorif, flatten_mat, setdefault


class Bounce(IOAble, ABC):
    """Abstract class for bounce integrals."""

    @staticmethod
    def get_pitch_inv_quad(min_B, max_B, num_pitch, simp=False):
        """Return 1/λ values and weights for quadrature between ``min_B`` and ``max_B``.

        Parameters
        ----------
        min_B : jnp.ndarray
            Minimum B value.
        max_B : jnp.ndarray
            Maximum B value.
        num_pitch : int
            Number of values.
        simp : bool
            Whether to use an open Simpson rule instead of uniform weights.

        Returns
        -------
        x, w : tuple[jnp.ndarray]
            Shape (min_B.shape, num pitch).
            1/λ values and weights.

        """
        errorif(
            num_pitch > 1e5,
            msg="Floating point error impedes detection of bounce points "
            f"near global extrema. Choose {num_pitch} < 1e5.",
        )
        # Samples should be uniformly spaced in |B| and not λ.
        # Important to do an open quadrature since the bounce integrals at the
        # global maxima of |B| are not computable even ignoring precision issues.
        x, w = simpson2(num_pitch) if simp else uniform(num_pitch)
        x = bijection_from_disc(x, min_B[..., None], max_B[..., None])
        w = w * grad_bijection_from_disc(min_B, max_B)[..., None]
        return x, w

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
        """Interpolate ``f`` to the deepest point pⱼ in magnetic well j."""

    @abstractmethod
    def plot(self, l, m, pitch_inv=None, **kwargs):
        """Plot B and bounce points on the specified field line."""


default_quad = get_quadrature(
    leggauss(32),
    (automorphism_sin, grad_automorphism_sin),
)


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

    Notes
    -----
    Magnetic field line with label α, defined by B = ∇ψ × ∇α, is determined from
      α : ρ, θ, ζ ↦ θ + Λ(ρ,θ,ζ) − ι(ρ) [ζ + ω(ρ,θ,ζ)]
    Interpolate Fourier-Chebyshev series to DESC poloidal coordinate.
      θ : ρ, α, ζ ↦ tₘₙ(ρ) exp(jmα) Tₙ(ζ)
    Compute bounce points.
      λ B(ζₖ) = 1
    Interpolate smooth periodic parts of integrand with FFTs.
      G : ρ, α, ζ ↦ gₘₙ(ρ) exp(j [m θ(ρ,α,ζ) + n ζ])
    Perform Gaussian quadrature after removing singularities.
      Fᵢ : ρ, α, λ, ζ₁, ζ₂ ↦  ∫ᵢ f(ρ,α,λ,ζ,{Gⱼ}) dζ

    If the map G is multivalued at a physical location, then it is still
    permissible if separable into periodic and secular parts.
    In that case, supply the periodic part, which will be interpolated
    with FFTs, and use the provided coordinates θ,ζ ∈ ℝ to compose G.

    Examples
    --------
    See ``tests/test_integrals.py::TestBounce2D::test_bounce2d_checks``.

    See Also
    --------
    Bounce1D
        ``Bounce1D`` uses one-dimensional splines for the same task.
        ``Bounce2D`` solves the dominant cost of optimization objectives in DESC
        relying on ``Bounce1D``: Computing a dense optimization-step dependent
        grid along field lines and interpolating 3D FourierZernike series to this grid.
        The function approximation done here requires FourierZernike series on a
        smaller fixed grid and uses FFTs to compute the map between coordinate systems.
        2D interpolation enables tracing the field line for more toroidal transits.
        Performance will improve significantly by resolving GitHub issue ``1303``:
        Patch for differentiable code with dynamic shapes.

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
    theta : jnp.ndarray
        Shape (num rho, X, Y).
        DESC coordinates θ from ``Bounce2D.compute_theta``.
        ``X`` and ``Y`` are preferably rounded down to powers of two.
    Y_B : int
        Desired resolution for algorithm to compute bounce points.
        A reference value is 100. Default is double ``Y``.
    alpha : jnp.ndarray
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
    quad : tuple[jnp.ndarray]
        Quadrature points xₖ and weights wₖ for the approximate evaluation of an
        integral ∫₋₁¹ g(x) dx = ∑ₖ wₖ g(xₖ). Default is 32 points.
    automorphism : tuple[Callable] or None
        The first callable should be an automorphism of the real interval [-1, 1].
        The second callable should be the derivative of the first. This map defines
        a change of variable for the bounce integral. The choice made for the
        automorphism will affect the performance of the quadrature.
    nufft_eps : float
        Precision requested for interpolation with non-uniform fast Fourier
        transform (NUFFT). If less than ``1e-14`` then NUFFT will not be used.
    is_reshaped : bool
        Whether the arrays in ``data`` are already reshaped to the expected form of
        shape (..., num zeta, num theta) or (num rho, num zeta, num theta).
        This option can be used to iteratively compute bounce integrals one flux
        surface at a time, reducing memory usage.
        To do so, set to ``True`` and provide only those chunks of the reshaped data.
        If set to ``True``, then it is assumed that ``data["iota"]`` has shape
        ``(grid.num_rho,)`` or is a scalar.
    is_fourier : bool
        If true, then it is assumed that ``data`` holds Fourier transforms
        as returned by ``Bounce2D.fourier`` and ``data["iota"]`` has shape
        ``(grid.num_rho,)`` or is a scalar. Default is false.
    Bref : float
        Optional. Reference magnetic field strength for normalization.
    Lref : float
        Optional. Reference length scale for normalization.
    spline : bool
        Whether to use cubic splines to compute bounce points instead of
        Chebyshev series. Note the algorithm for efficient root-finding on
        Chebyshev series has not been implemented.
    check : bool
        Flag for debugging. Must be false for JAX transformations.

    """

    # For applications which reduce to computing a nonlinear function of distance
    # along field lines between bounce points, it is required to identify these
    # points with field-line-following coordinates. (In the special case of a linear
    # function summing integrals between bounce points over a flux surface, arbitrary
    # coordinate systems may be used as that task reduces to a surface integral,
    # which is invariant to the order of summation).
    #
    # The DESC coordinate system maps to field-line-following coordinate
    # systems with a relation whose solution is best found with Newton iteration.
    # In contrast, Newton iteration is not a globally convergent algorithm to
    # find the real roots of r : ζ ↦ B(ζ) − 1/λ where ζ is a field-line-following
    # coordinate. For this, function approximation of B is necessary.
    #
    # The frequency transform of a map under the chosen basis must be concentrated
    # at low frequencies for the series to converge fast. For periodic (non-periodic)
    # maps, the standard choice for the basis is a Fourier (Chebyshev) series.
    # Both converge exponentially, but the larger region of convergence in the
    # complex plane of Fourier series makes it preferable to choose coordinate
    # systems such that the function to approximate is periodic. A Fourier-Chebyshev
    # series is chosen to interpolate θ(α,ζ). Using Chebyshev series with the
    # Kosloff and Tal-Ezer almost-equispaced grid does not show worthwhile improvement.
    #
    # Function approximation in (α, ζ) coordinates demands particular interpolation
    # points in that coordinate system because there is no transformation that converts
    # series coefficients in periodic coordinates, e.g. (ϑ, ϕ), to a low order
    # polynomial basis in non-periodic coordinates. For example, one can obtain series
    # coefficients in (α, ϕ) coordinates from those in (ϑ, ϕ) as follows
    #   g : ϑ, ϕ ↦ ∑ₘₙ aₘₙ exp(j [mϑ + nϕ])
    #
    #   g : α, ϕ ↦ ∑ₘₙ aₘₙ exp(j [mα + (m ι + n)ϕ])
    # The basis for the latter are trigonometric functions with
    # irrational frequencies, courtesy of the irrational rotational transform.
    # The denominator of a close rational could be absorbed into the coordinate ϕ,
    # but this balloons the frequency, and hence degree of the series.
    #
    # Quadrature is chosen over Runge-Kutta methods of the form
    #     ∂Fᵢ/∂ζ = f(ρ,α,λ,ζ,{Gⱼ}) subject to Fᵢ(ζ₁) = 0
    # A fourth order Runge-Kutta method is equivalent to a quadrature
    # with Simpson's rule. The quadratures resolve these integrals more efficiently.

    required_names = ["B^zeta", "|B|", "iota"]

    def __init__(
        self,
        grid,
        data,
        theta,
        Y_B=None,
        alpha=jnp.array([0.0]),
        num_transit=20,
        quad=None,
        *,
        automorphism=None,
        nufft_eps=1e-6,
        is_reshaped=False,
        is_fourier=False,
        Bref=1.0,
        Lref=1.0,
        spline=True,
        check=False,
        vander=None,
    ):
        """Returns an object to compute bounce integrals."""
        assert grid.can_fft2
        is_reshaped = is_reshaped or is_fourier
        vander = setdefault(vander, {})
        quad = setdefault(quad, default_quad)

        self._x, self._w = get_quadrature(quad, automorphism)
        self._NFP = grid.NFP
        self._num_theta = grid.num_theta
        self._n_modes, self._m_modes = rfft2_modes(
            grid.num_zeta, grid.num_theta, (0, 2 * jnp.pi / grid.NFP)
        )

        self._c = {
            "|B|": data["|B|"] / Bref,
            "B^zeta": data["B^zeta"] * Lref / Bref,
            "T(z)": fourier_chebyshev(
                theta,
                data["iota"] if is_reshaped else grid.compress(data["iota"]),
                alpha,
                num_transit,
            ),
        }
        if not is_reshaped:
            self._c["|B|"] = Bounce2D.reshape(grid, self._c["|B|"])
            self._c["B^zeta"] = Bounce2D.reshape(grid, self._c["B^zeta"])
        if not is_fourier:
            self._c["|B|"] = Bounce2D.fourier(self._c["|B|"])
            self._c["B^zeta"] = Bounce2D.fourier(self._c["B^zeta"])

        Y_B = setdefault(Y_B, theta.shape[-1] * 2)
        if spline:
            self._c["B(z)"], self._c["knots"] = fast_cubic_spline(
                self._c["T(z)"],
                self._c["|B|"],
                Y_B,
                self._num_theta,
                self._m_modes,
                self._n_modes,
                self._NFP,
                nufft_eps,
                vander_theta=vander.get("dct spline", None),
                check=check,
            )
        else:
            self._c["B(z)"] = fast_chebyshev(
                self._c["T(z)"],
                self._c["|B|"],
                Y_B,
                self._num_theta,
                self._m_modes,
                self._n_modes,
                self._NFP,
            )

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
            Shape (num rho, num zeta, num theta).
            Reshaped data which may be given to ``integrate``.

        """
        return grid.meshgrid_reshape(f, "rzt")

    @staticmethod
    def fourier(f):
        """Transform to DESC spectral domain.

        Parameters
        ----------
        f : jnp.ndarray
            Shape (..., num zeta, num theta).
            Real scalar-valued periodic function evaluated on tensor-product grid
            with uniformly spaced nodes (θ, ζ) ∈ [0, 2π) × [0, 2π/NFP).

        Returns
        -------
        a : jnp.ndarray
            Shape is (..., 1, num zeta, num theta // 2 + 1).
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
        X=16,
        Y=32,
        rho=jnp.array([1.0]),
        iota=None,
        params=None,
        profiles=None,
        tol=1e-7,
        **kwargs,
    ):
        """Return DESC coordinates θ of (α,ζ) Fourier Chebyshev basis nodes.

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium to use defining the coordinate mapping.
        X : int
            Poloidal Fourier grid resolution to interpolate the poloidal coordinate.
            Preferably rounded down to power of 2.
        Y : int
            Toroidal Chebyshev grid resolution to interpolate the poloidal coordinate.
            Preferably rounded down to power of 2.
        rho : float or jnp.ndarray
            Shape (num rho, ).
            Flux surfaces labels in [0, 1] on which to compute.
        iota : float or jnp.ndarray
            Shape (num rho, ).
            Optional, rotational transform on the flux surfaces to compute on.
        params : dict[str,jnp.ndarray]
            Parameters from the equilibrium, such as R_lmn, Z_lmn, i_l, p_l, etc
            Defaults to ``eq.params_dict``.
        profiles
            Optional profiles.
        tol : float
            Stopping tolerance for root finding.
            Default is ``1e-7``.
        kwargs
            Additional parameters to supply to the coordinate mapping function.
            See ``desc.equilibrium.Equilibrium.map_coordinates``.

        Returns
        -------
        theta : jnp.ndarray
            Shape (num rho, X, Y).
            DESC coordinates θ.

        """
        from desc.compute.utils import get_transforms

        params = setdefault(params, eq.params_dict)
        if iota is None:
            iota = eq._compute_iota_under_jit(rho, params, profiles, **kwargs)
        iota = jnp.atleast_1d(iota)

        zeta = cheb_pts(Y, (0, 2 * jnp.pi))[::-1]
        lmbda = kwargs.get("lmbda", None)
        if lmbda is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Unequal number of field periods")
                lmbda = get_transforms(
                    "lambda", eq, grid=LinearGrid(rho=rho, M=eq.L_basis.M, zeta=zeta)
                )["L"]
        assert lmbda.basis.NFP == eq.NFP

        return eq._map_clebsch_coordinates(
            iota=iota,
            alpha=fourier_pts(X),
            zeta=zeta,
            L_lmn=params["L_lmn"],
            lmbda=lmbda,
            tol=tol,
        )[..., ::-1]

    @property
    def _num_zeta(self):
        return self._n_modes.size

    def _swap_pitch(self, pitch_inv):
        """Transpose to simplify broadcasting.

        Parameters
        ----------
        pitch_inv : jnp.ndarray
            Shape broadcasts with (num rho, num pitch).

        Returns
        -------
        pitch_inv : jnp.ndarray
            Shape broadcasts with (num pitch, *self._c["T(z)"].cheb.shape[:-2])

        """
        return atleast_nd(self._c["T(z)"].cheb.ndim - 1, pitch_inv).swapaxes(0, -1)

    def points(self, pitch_inv, num_well=None):
        """Compute bounce points.

        Parameters
        ----------
        pitch_inv : jnp.ndarray
            Shape (num rho, num pitch).
            1/λ values to compute the bounce integrals. 1/λ(ρ) is specified by
            ``pitch_inv[ρ]`` where in the latter the labels are interpreted
            as the indices that correspond to that field line.
        num_well : int or None
            Specify to return the first ``num_well`` pairs of bounce points for each
            pitch and field line. Default is ``None``, which will detect all wells,
            but due to current limitations in JAX this will have worse performance.
            Specifying a number that tightly upper bounds the number of wells will
            increase performance. In general, an upper bound on the number of wells
            per toroidal transit is ``Aι+B`` where ``A``, ``B`` are the poloidal and
            toroidal Fourier resolution of B, respectively, in straight-field line
            PEST coordinates, and ι is the rotational transform normalized by 2π.
            A tighter upper bound than ``num_well=(Aι+B)*num_transit`` is preferable.
            The ``check_points`` or ``plot`` method is useful to select a reasonable
            value.

            If there were fewer wells detected along a field line than the size of the
            last axis of the returned arrays, then that axis is padded with zero.

        Returns
        -------
        z1, z2 : tuple[jnp.ndarray]
            Shape (num rho, num alpha, num pitch, num well).
            Tuple of length two (z1, z2) that stores ζ coordinates of bounce points.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of B.

            If there were less than ``num_well`` wells detected along a field line,
            then the last axis, which enumerates bounce points for a particular field
            line and pitch, is padded with zero.

        """
        if isinstance(self._c["B(z)"], PiecewiseChebyshevSeries):
            z1, z2 = self._c["B(z)"].intersect1d(self._swap_pitch(pitch_inv), num_well)
            z1 = _move(z1)
            z2 = _move(z2)
            return z1, z2

        return bounce_points(
            _broadcast_for_bounce(pitch_inv),
            self._c["knots"],
            self._c["B(z)"],
            polyder_vec(self._c["B(z)"]),
            num_well,
        )

    def _polish_points(self, points, pitch_inv):
        # TODO after (#1243): One application of Newton on Fourier series |B|-1/λ.
        #  Need Fourier coefficients of lambda, but that is already known.
        #  Then can use less resolution for the global root finding algorithm
        #  and rely on the local one once good neighbourhood is found.
        #  For now, we integrate with √|1−λB| as justified in doi.org/10.1063/5.0160282.
        raise NotImplementedError

    def check_points(self, points, pitch_inv, *, plot=True, **kwargs):
        """Check that bounce points are computed correctly.

        Parameters
        ----------
        points : tuple[jnp.ndarray]
            Shape (num rho, num alpha, num pitch, num well).
            Output of method ``self.points``.
            Tuple of length two (z1, z2) that stores ζ coordinates of bounce points.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of B.
        pitch_inv : jnp.ndarray
            Shape (num rho, num pitch).
            1/λ values to compute the bounce integrals. 1/λ(ρ) is specified by
            ``pitch_inv[ρ]`` where in the latter the labels are interpreted
            as the indices that correspond to that field line.
        plot : bool
            Whether to plot the field lines and bounce points of the given pitch angles.
        kwargs : dict
            Keyword arguments into
            ``desc/integrals/basis.py::PiecewiseChebyshevSeries.plot1d`` or
            ``desc/integrals/_bounce_utils.py::plot_ppoly``.

        Returns
        -------
        plots : list
            Matplotlib (fig, ax) tuples for the 1D plot of each field line.

        """
        kwargs = _set_default_plot_kwargs(kwargs)
        if isinstance(self._c["B(z)"], PiecewiseChebyshevSeries):
            z1, z2 = points
            return self._c["B(z)"].check_intersect1d(
                _move(z1, False),
                _move(z2, False),
                self._swap_pitch(pitch_inv),
                plot=plot,
                **kwargs,
            )
        return _check_bounce_points(
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
        is_fourier=False,
        quad=None,
        check=False,
        plot=False,
    ):
        """Bounce integrate ∫ f(ρ,α,λ,ℓ) dℓ.

        Computes the bounce integral ∫ f(ρ,α,λ,ℓ) dℓ for every field line and pitch.

        Notes
        -----
        Make sure to replace √(1−λB) with √|1−λB| in ``integrand`` to account
        for imperfect computation of bounce points.

        Parameters
        ----------
        integrand : callable or list[callable]
            The composition operator on the set of functions in ``data``
            that determines ``f`` in ∫ f(ρ,α,λ,ℓ) dℓ. It should accept a dictionary
            which stores the interpolated data and the arguments ``B`` and ``pitch``.
        pitch_inv : jnp.ndarray
            Shape (num rho, num pitch).
            1/λ values to compute the bounce integrals. 1/λ(ρ) is specified by
            ``pitch_inv[ρ]`` where in the latter the labels are interpreted
            as the indices that correspond to that field line.
        data : dict[str, jnp.ndarray]
            Shape (num rho, num zeta, num theta).
            Real scalar-valued periodic functions in (θ, ζ) ∈ [0, 2π) × [0, 2π/NFP)
            evaluated on the ``grid`` supplied to construct this object.
            Use the method ``Bounce2D.reshape`` to reshape the data into the
            expected shape.
        names : str or list[str]
            Names in ``data`` to interpolate. Default is all keys in ``data``.
        points : tuple[jnp.ndarray]
            Shape (num rho, num alpha, num pitch, num well).
            Optional, output of method ``self.points``.
            Tuple of length two (z1, z2) that stores ζ coordinates of bounce points.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of B.
        num_well : int or None
            See ``self.points`` for the description of this parameter.
        nufft_eps : float
            Precision requested for interpolation with non-uniform fast Fourier
            transform (NUFFT). If less than ``1e-14`` then NUFFT will not be used.
        is_fourier : bool
            If true, then it is assumed that ``data`` holds Fourier transforms
            as returned by ``Bounce2D.fourier``. Default is false.
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
            Shape (num rho, num alpha, num pitch, num well).
            Last axis enumerates the bounce integrals for a given field line
            and pitch value.

        """
        x, w = self._x, self._w if quad is None else quad
        if not isinstance(integrand, (list, tuple)):
            integrand = [integrand]
        if isinstance(names, str):
            names = [names]

        exclude = ("|B|", "B^zeta", "|e_zeta|r,a|", "zeta", "theta")
        data = setdefault(data, {})
        if is_fourier:
            data = apply(data, subset=names, exclude=exclude)
        else:
            data = apply(data, Bounce2D.fourier, names, exclude)

        if points is None:
            points = self.points(pitch_inv, num_well)
        z1, z2 = points

        pitch = 1 / pitch_inv
        # to broadcast with (..., num pitch, num well, num quad)
        if jnp.ndim(pitch) == 1:
            pitch = pitch[..., None, None]
        elif jnp.ndim(pitch) > 1:
            pitch = pitch[:, None, :, None, None]

        zeta = bijection_from_disc(x, z1[..., None], z2[..., None])
        theta = self._c["T(z)"].eval1d(flatten_mat(zeta, 3)).reshape(zeta.shape)

        if nufft_eps < 1e-14:
            data = self._nummt(zeta, theta, data)
        else:
            data = self._nufft(zeta, theta, data, nufft_eps)
        data["|e_zeta|r,a|"] = data["|B|"] / jnp.abs(data["B^zeta"])
        data["zeta"] = zeta
        data["theta"] = theta

        # Strictly increasing ζ knots enforces dζ > 0.
        # To retain dℓ = |B|/(B⋅∇ζ) dζ > 0 after fixing dζ > 0, we require
        # B⋅∇ζ > 0. This is equivalent to changing the sign of ∇ζ
        # or (∂ℓ/∂ζ)|ρ,a. Recall dζ = ∇ζ⋅dR ⇔ 1 = ∇ζ⋅(e_ζ|ρ,a).
        cov = grad_bijection_from_disc(z1, z2)
        result = [
            (f(data, data["|B|"], pitch) * data["|e_zeta|r,a|"]).dot(w) * cov
            for f in integrand
        ]

        if check:
            _check_interp(
                data["zeta"],
                jnp.reciprocal(data["|e_zeta|r,a|"]),
                data["|B|"],
                [data[k] for k in data if k not in ("zeta", "|e_zeta|r,a|", "|B|")],
                result,
                plot=plot,
            )

        return result[0] if len(result) == 1 else result

    def _nufft(self, zeta, theta, data, eps):
        shape = zeta.shape
        c = nufft2d2r(
            flatten_mat(zeta, 4),
            flatten_mat(theta, 4),
            jnp.concatenate([*data.values(), self._c["B^zeta"], self._c["|B|"]], -3),
            (0, 2 * jnp.pi / self._NFP),
            vec=True,
            eps=eps,
        )
        c = c.swapaxes(0, -2).reshape(len(data) + 2, *shape)
        return dict(zip([*data.keys(), "B^zeta", "|B|"], c))

    def _nummt(self, zeta, theta, data):
        v = rfft2_vander(zeta, theta, self._n_modes, self._m_modes)
        data = {name: _mmt_for_bounce(v, c) for name, c in data.items()}
        data["B^zeta"] = _mmt_for_bounce(v, self._c["B^zeta"])
        data["|B|"] = _mmt_for_bounce(v, self._c["|B|"])
        return data

    def interp_to_argmin(self, f, points, *, nufft_eps=1e-6, is_fourier=False):
        """Interpolate ``f`` to the deepest point pⱼ in magnetic well j.

        Parameters
        ----------
        f : jnp.ndarray
            Shape (num rho, num zeta, num theta).
            Real scalar-valued periodic function in (θ, ζ) ∈ [0, 2π) × [0, 2π/NFP)
            evaluated on the ``grid`` supplied to construct this object.
            Use the method ``Bounce2D.reshape`` to reshape the data into the
            expected shape.
        points : tuple[jnp.ndarray]
            Shape (num rho, num alpha, num pitch, num well).
            Optional, output of method ``self.points``.
            Tuple of length two (z1, z2) that stores ζ coordinates of bounce points.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of B.
        nufft_eps : float
            Precision requested for interpolation with non-uniform fast Fourier
            transform (NUFFT). If less than ``1e-14`` then NUFFT will not be used.
        is_fourier : bool
            If true, then it is assumed that ``f`` is the Fourier transforms
            as returned by ``Bounce2D.fourier``. Default is false.

        Returns
        -------
        f_j : jnp.ndarray
            Shape (num rho, num alpha, num pitch, num well).
            ``f`` interpolated to the deepest point between ``points``.

        """
        errorif(
            isinstance(self._c["B(z)"], PiecewiseChebyshevSeries),
            NotImplementedError,
            "Must choose Bounce2D(spline=True) for this feature.",
        )
        if not is_fourier:
            f = Bounce2D.fourier(f)

        ext, B_ext = get_extrema(
            self._c["knots"],
            self._c["B(z)"],
            polyder_vec(self._c["B(z)"]),
            sentinel=0.0,
        )
        theta = self._c["T(z)"].eval1d(ext)

        if nufft_eps < 1e-14:
            f = _irfft2_mmt(
                ext,
                theta,
                f[..., None, :, :],
                self._num_zeta,
                self._num_theta,
                (0, 2 * jnp.pi / self._NFP),
            )
        else:
            shape = (*ext.shape[:-2], -1)
            theta = theta.reshape(shape)
            f = nufft2d2r(
                ext.reshape(shape),
                theta,
                f.squeeze(-3),
                (0, 2 * jnp.pi / self._NFP),
                eps=nufft_eps,
            ).reshape(ext.shape)

        return argmin(*points, f, ext, B_ext)

    def compute_fieldline_length(self, quad=None, vander=None):
        """Compute the (mean) proper length of the field line ∫ dℓ / B.

        Computes mean_A ∫ dℓ / B where A is the set of field line labels
        given when making this object.

        Parameters
        ----------
        quad : tuple[jnp.ndarray]
            Quadrature points xₖ and weights wₖ for the
            approximate evaluation of the integral ∫₋₁¹ f(x) dx ≈ ∑ₖ wₖ f(xₖ).
            Default is Gauss-Legendre quadrature at resolution ``Y_B//2``
            on each toroidal transit.
        vander : dict[str,jnp.ndarray]
            Optional precomputed Vandermonde matrices for interpolation.

        Returns
        -------
        length : jnp.ndarray
            Shape (num rho, ).

        """
        if quad is None:
            # Integrating an analytic oscillatory map so a high order quadrature
            # is ideal. Difficult to pick the right frequency for Filon quadrature
            # in general, which would work best at high NFP. Gauss-Legendre is
            # superior to Clenshaw-Curtis for smooth oscillatory maps. Prolate
            # spheroidal wave function quadrature would be an improvement.
            deg = (
                self._c["B(z)"].Y
                if isinstance(self._c["B(z)"], PiecewiseChebyshevSeries)
                else (self._c["knots"].size // self._c["T(z)"].X)
            )
            quad = leggauss(deg // 2)
        x, w = quad
        vander = setdefault(vander, {})

        B_sup_zeta = irfft_mmt(
            idct_mmt(
                x,
                self._c["T(z)"].cheb[..., None, :],
                vander=vander.get("dct cfl", None),
            ),
            self._partial_sum_cfl(x, vander.get("dft cfl", None)),
            self._num_theta,
            _modes=self._m_modes,
        )

        # B⋅∇ζ never vanishes, so it has the same sign over a surface.
        # Simple mean over α because when ζ extends beyond one transit we need
        # to weight all field lines uniformly regardless of their area wrt α.
        dz_dx = jnp.pi
        return jnp.abs(jnp.reciprocal(B_sup_zeta).dot(w).sum(-1).mean(-1)) * dz_dx

    def _partial_sum_cfl(self, x, vander):
        # Let m, n denote the poloidal and toroidal Fourier resolution. We need to
        # compute a set of 2D Fourier series each on non-uniform tensor product grids
        # of size |𝛉|×|𝛇| where |𝛉| = num alpha × num transit and |𝛇| = x.size.
        # Partial summation is more efficient than direct evaluation when
        # mn|𝛉||𝛇| > mn|𝛇| + m|𝛉||𝛇| or equivalently n|𝛉| > n + |𝛉|.

        return ifft_mmt(
            bijection_from_disc(x, 0, 2 * jnp.pi)[:, None] if vander is None else None,
            self._c["B^zeta"],
            (0, 2 * jnp.pi / self._NFP),
            axis=-2,
            modes=self._n_modes,
            vander=vander,
        )[..., None, None, :, :]

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
        kwargs
            Keyword arguments into
            ``desc/integrals/basis.py::PiecewiseChebyshevSeries.plot1d``.

        Returns
        -------
        fig, ax
            Matplotlib (fig, ax) tuple.

        """
        errorif(
            pitch_inv is not None and jnp.ndim(pitch_inv) > 1,
            msg=f"Got pitch_inv.ndim={jnp.ndim(pitch_inv)}, but expected 1.",
        )
        kwargs = _set_default_plot_kwargs(kwargs, l, m)

        B = self._c["B(z)"]
        if isinstance(B, PiecewiseChebyshevSeries):
            if B.cheb.ndim == 4:
                B = PiecewiseChebyshevSeries(B.cheb[l, m], B.domain)
            elif B.cheb.ndim == 3:
                B = PiecewiseChebyshevSeries(B.cheb[m], B.domain)
            if pitch_inv is not None:
                z1, z2 = B.intersect1d(pitch_inv)
                kwargs["z1"] = z1
                kwargs["z2"] = z2
                kwargs["k"] = pitch_inv
            return B.plot1d(B.cheb, **kwargs)

        if B.ndim == 4:
            B = B[l]
        if B.ndim == 3:
            B = B[m]
        if pitch_inv is not None:
            z1, z2 = bounce_points(pitch_inv, self._c["knots"], B, polyder_vec(B))
            kwargs["z1"] = z1
            kwargs["z2"] = z2
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
        kwargs
            Keyword arguments into
            ``desc/integrals/basis.py::PiecewiseChebyshevSeries.plot1d``.

        Returns
        -------
        fig, ax
            Matplotlib (fig, ax) tuple.

        """
        T = self._c["T(z)"]
        if T.cheb.ndim == 4:
            T = PiecewiseChebyshevSeries(T.cheb[l, m], T.domain)
        elif T.cheb.ndim == 3:
            T = PiecewiseChebyshevSeries(T.cheb[m], T.domain)
        kwargs.setdefault(
            "title",
            rf"Poloidal angle $\theta$ on field line $\rho(l={l})$, $\alpha(m={m})$",
        )
        kwargs.setdefault("vlabel", r"$\theta$")
        return T.plot1d(T.cheb, **_set_default_plot_kwargs(kwargs, l, m))


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
    See ``tests/test_integrals.py::TestBounce::test_bounce1d_checks``.

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
        Quadrature points xₖ and weights wₖ for the approximate evaluation of an
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
    is_reshaped : bool
        Whether the arrays in ``data`` are already reshaped to the expected form of
        shape (..., num zeta) or (..., num alpha, num zeta) or
        (num rho, num alpha, num zeta). This option can be used to iteratively
        compute bounce integrals one flux surface or one field line at a time,
        respectively, reducing memory usage.
        To do so, set to ``True`` and provide only those chunks of the reshaped data.
    check : bool
        Flag for debugging. Must be false for JAX transformations.

    """

    required_names = ["B^zeta", "B^zeta_z|r,a", "|B|", "|B|_z|r,a"]

    def __init__(
        self,
        grid,
        data,
        quad=None,
        *,
        automorphism=None,
        Bref=1.0,
        Lref=1.0,
        is_reshaped=False,
        check=False,
    ):
        """Returns an object to compute bounce integrals."""
        assert grid.is_meshgrid
        quad = setdefault(quad, default_quad)

        self._x, self._w = get_quadrature(quad, automorphism)
        self._data = {
            "|b^zeta|": jnp.abs(data["B^zeta"]) * Lref / data["|B|"],
            "|B|": data["|B|"] / Bref,
            "|B|_z|r,a": data["|B|_z|r,a"] / Bref,
        }
        self._data["|b^zeta|_z|r,a"] = (
            data["B^zeta_z|r,a"] * jnp.sign(data["B^zeta"]) * Lref
            - self._data["|b^zeta|"] * data["|B|_z|r,a"]
        ) / data["|B|"]

        if not is_reshaped:
            for name in self._data:
                self._data[name] = Bounce1D.reshape(grid, self._data[name])

        self._zeta = grid.compress(grid.nodes[:, 2], surface_label="zeta")
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
            Shape (num rho, num alpha, num zeta).
            Reshaped data which may be given to ``integrate``.

        """
        return grid.meshgrid_reshape(f, "raz")

    def points(self, pitch_inv, num_well=None):
        """Compute bounce points.

        Parameters
        ----------
        pitch_inv : jnp.ndarray
            Shape (num rho, num pitch).
            1/λ values to compute the bounce integrals. 1/λ(ρ) is specified by
            ``pitch_inv[ρ]`` where in the latter the labels are interpreted
            as the indices that correspond to that field line.
        num_well : int or None
            Specify to return the first ``num_well`` pairs of bounce points for each
            pitch and field line. Default is ``None``, which will detect all wells,
            but due to current limitations in JAX this will have worse performance.
            Specifying a number that tightly upper bounds the number of wells will
            increase performance. In general, an upper bound on the number of wells
            per toroidal transit is ``Aι+B`` where ``A``, ``B`` are the poloidal and
            toroidal Fourier resolution of B, respectively, in straight-field line
            PEST coordinates, and ι is the rotational transform normalized by 2π.
            A tighter upper bound than ``num_well=(Aι+B)*num_transit`` is preferable.
            The ``check_points`` or ``plot`` method is useful to select a reasonable
            value.

            If there were fewer wells detected along a field line than the size of the
            last axis of the returned arrays, then that axis is padded with zero.

        Returns
        -------
        z1, z2 : tuple[jnp.ndarray]
            Shape (num rho, num alpha, num pitch, num well).
            Tuple of length two (z1, z2) that stores ζ coordinates of bounce points.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of B.

            If there were less than ``num_well`` wells detected along a field line,
            then the last axis, which enumerates bounce points for a particular field
            line and pitch, is padded with zero.

        """
        return bounce_points(
            _broadcast_for_bounce(pitch_inv),
            self._zeta,
            self._B,
            polyder_vec(self._B),
            num_well,
        )

    def check_points(self, points, pitch_inv, *, plot=True, **kwargs):
        """Check that bounce points are computed correctly.

        Parameters
        ----------
        points : tuple[jnp.ndarray]
            Shape (num rho, num alpha, num pitch, num well).
            Output of method ``self.points``.
            Tuple of length two (z1, z2) that stores ζ coordinates of bounce points.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of B.
        pitch_inv : jnp.ndarray
            Shape (num rho, num pitch).
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
        return _check_bounce_points(
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

        Parameters
        ----------
        integrand : callable or list[callable]
            The composition operator on the set of functions in ``data``
            that determines ``f`` in ∫ f(ρ,α,λ,ℓ) dℓ. It should accept a dictionary
            which stores the interpolated data and the arguments ``B`` and ``pitch``.
        pitch_inv : jnp.ndarray
            Shape (num rho, num pitch).
            1/λ values to compute the bounce integrals. 1/λ(ρ) is specified by
            ``pitch_inv[ρ]`` where in the latter the labels are interpreted
            as the indices that correspond to that field line.
        data : dict[str, jnp.ndarray]
            Shape (num rho, num alpha, num zeta).
            Real scalar-valued functions evaluated on the ``grid`` supplied to
            construct this object. Use the method ``Bounce1D.reshape`` to reshape
            the data into the expected shape.
        names : str or list[str]
            Names in ``data`` to interpolate. Default is all keys in ``data``.
        points : tuple[jnp.ndarray]
            Shape (num rho, num alpha, num pitch, num well).
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
            Shape (num rho, num alpha, num pitch, num well).
            Last axis enumerates the bounce integrals for a given field line
            and pitch value.

        """
        x, w = self._x, self._w if quad is None else quad
        if not isinstance(integrand, (list, tuple)):
            integrand = [integrand]
        if isinstance(names, str):
            names = [names]

        data = apply(setdefault(data, {}), subset=names, exclude=("|B|",))

        if points is None:
            points = self.points(pitch_inv, num_well)
        pitch = jnp.atleast_1d(1 / _broadcast_for_bounce(pitch_inv))[..., None]

        if kwargs.get("batch", True):
            pitch = pitch[..., None]
            result = self._integrate(
                x,
                w,
                integrand,
                pitch,
                data,
                *points,
                method,
                check,
                plot,
                batch=True,
            )
        else:

            def loop(points):
                """Integrate one well at a time."""
                return self._integrate(
                    x,
                    w,
                    integrand,
                    pitch,
                    data,
                    *points,
                    method,
                    check=False,
                    plot=False,
                    batch=False,
                )

            result = batch_map(loop, [jnp.moveaxis(z, -1, 0) for z in points], 1)
            result = [jnp.moveaxis(r, 0, -1) for r in result]

        return result[0] if len(result) == 1 else result

    def _integrate(
        self, x, w, integrand, pitch, data, z1, z2, method, check, plot, batch
    ):
        shape = (*z1.shape, x.size)  # (..., num pitch, num well, num quad)

        zeta = flatten_mat(
            bijection_from_disc(x, z1[..., None], z2[..., None]), 2 + batch
        )

        b_sup_z = interp1d_Hermite_vec(
            zeta,
            self._zeta,
            self._data["|b^zeta|"],
            self._data["|b^zeta|_z|r,a"],
        ).reshape(shape)
        B = interp1d_Hermite_vec(
            zeta,
            self._zeta,
            self._data["|B|"],
            self._data["|B|_z|r,a"],
        ).reshape(shape)
        data = {
            k: interp1d_vec(zeta, self._zeta, v, method=method).reshape(shape)
            for k, v in data.items()
        }

        # Strictly increasing ζ knots enforces dζ > 0.
        # To retain dℓ = |B|/(B⋅∇ζ) dζ > 0 after fixing dζ > 0, we require
        # B⋅∇ζ > 0. This is equivalent to changing the sign of ∇ζ
        # or (∂ℓ/∂ζ)|ρ,a. Recall dζ = ∇ζ⋅dR ⇔ 1 = ∇ζ⋅(e_ζ|ρ,a).
        cov = grad_bijection_from_disc(z1, z2)
        result = [(f(data, B, pitch) / b_sup_z).dot(w) * cov for f in integrand]

        if check:
            _check_interp(
                zeta.reshape(shape),
                b_sup_z,
                B,
                data.values(),
                result,
                plot=plot,
            )

        return result

    def interp_to_argmin(self, f, points, *, method="cubic"):
        """Interpolate ``f`` to the deepest point pⱼ in magnetic well j.

        Parameters
        ----------
        f : jnp.ndarray
            Shape (num rho, num alpha, num zeta).
            Real scalar-valued functions evaluated on the ``grid`` supplied to
            construct this object. Use the method ``Bounce1D.reshape`` to
            reshape the data into the expected shape.
        points : tuple[jnp.ndarray]
            Shape (num rho, num alpha, num pitch, num well).
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
            Shape (num rho, num alpha, num pitch, num well).
            ``f`` interpolated to the deepest point between ``points``.

        """
        ext, g_ext = get_extrema(
            self._zeta, self._B, polyder_vec(self._B), sentinel=0.0
        )
        return argmin(
            *points, interp1d_vec(ext, self._zeta, f, method=method), ext, g_ext
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
            z1, z2 = bounce_points(pitch_inv, self._zeta, B, polyder_vec(B))
            kwargs["z1"] = z1
            kwargs["z2"] = z2
            kwargs["k"] = pitch_inv
        fig, ax = plot_ppoly(
            PPoly(B.T, self._zeta), **_set_default_plot_kwargs(kwargs, l, m)
        )
        return fig, ax
