"""Methods for computing bounce integrals (singular or otherwise)."""

from abc import ABC, abstractmethod

from interpax import CubicHermiteSpline, PPoly
from orthax.legendre import leggauss

from desc.backend import jnp, rfft2
from desc.batching import batch_map
from desc.integrals._bounce_utils import (
    _check_bounce_points,
    _check_interp,
    _set_default_plot_kwargs,
    bounce_points,
    chebyshev,
    cubic_spline,
    fourier_chebyshev,
    interp_fft_to_argmin,
    interp_to_argmin,
    plot_ppoly,
)
from desc.integrals._interp_utils import (
    idct_non_uniform,
    ifft_non_uniform,
    interp1d_Hermite_vec,
    interp1d_vec,
    irfft_non_uniform,
    polyder_vec,
    rfft2_modes,
    rfft2_vander,
)
from desc.integrals.basis import FourierChebyshevSeries, PiecewiseChebyshevSeries
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
from desc.utils import atleast_nd, errorif, flatten_matrix, setdefault


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
        x = bijection_from_disc(x, min_B[..., jnp.newaxis], max_B[..., jnp.newaxis])
        w = w * grad_bijection_from_disc(min_B, max_B)[..., jnp.newaxis]
        return x, w

    @abstractmethod
    def points(self, pitch_inv, num_well=None):
        """Compute bounce points."""

    @abstractmethod
    def check_points(self, points, pitch_inv, *, plot=True):
        """Check that bounce points are computed correctly."""

    @abstractmethod
    def integrate(
        self, integrand, pitch_inv, data=None, names=None, points=None, *, quad=None
    ):
        """Bounce integrate ∫ f(ρ,α,λ,ℓ) dℓ."""

    @abstractmethod
    def interp_to_argmin(self, f, points):
        """Interpolate ``f`` to the deepest point pⱼ in magnetic well j."""

    @abstractmethod
    def plot(self, l, m, pitch_inv=None, **kwargs):
        """Plot B and bounce points on the specified field line."""


def _swap_shape(f):
    """Use to swap between the following shapes.

    The LHS shape enables the simplest broadcasting so it is used internally,
    while the RHS shape is the returned shape which enables simplest to use API
    for computing various quantities.
    (num pitch, num alpha, num rho, -1) <-> (num rho, num alpha, num pitch, -1)
    (num pitch, num alpha,          -1) <-> (num alpha, num pitch,          -1)
    (num pitch,                     -1) <-> (num pitch,                     -1)
    """
    assert f.ndim <= 4
    return jnp.swapaxes(f, 0, -2)


default_quad = get_quadrature(
    leggauss(32),
    (automorphism_sin, grad_automorphism_sin),
)


class Bounce2D(Bounce):
    """Computes bounce integrals using pseudo-spectral methods.

    The bounce integral is defined as ∫ f(ρ,α,λ,ℓ) dℓ where

    * dℓ parameterizes the distance along the field line in meters.
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
      α : ρ, θ, ζ ↦ θ + λ(ρ,θ,ζ) − ι(ρ) [ζ + ω(ρ,θ,ζ)]
    Interpolate Fourier-Chebyshev series to DESC poloidal coordinate.
      θ : ρ, α, ζ ↦ tₘₙ(ρ) exp(jmα) Tₙ(ζ)
    Compute bounce points.
      λ B(ζₖ) = 1
    Interpolate smooth periodic components of integrand with FFTs.
      G : ρ, α, ζ ↦ gₘₙ(ρ) exp(j [m θ(ρ,α,ζ) + n ζ])
    Perform Gaussian quadrature after removing singularities.
      Fᵢ : ρ, α, λ, ζ₁, ζ₂ ↦  ∫ᵢ f(ρ,α,λ,ζ,{Gⱼ}) dζ

    If the map G is multivalued at a physical location, then it is still
    permissible if separable into periodic and secular components.
    In that case, supply the periodic component, which will be interpolated
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
        The drawback is that evaluating a Fourier series with resolution F at Q
        non-uniform quadrature points takes 𝒪(-(F+Q) log(F) log(ε)) time
        whereas cubic splines take 𝒪(C Q) time. However, as NFP increases,
        F decreases whereas C increases. Also, Q >> F and Q >> C.

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
        DESC coordinates θ sourced from the Clebsch coordinates
        ``FourierChebyshevSeries.nodes(X,Y,rho,domain=(0,2*jnp.pi))``.
        Use the ``Bounce2D.compute_theta`` method to obtain this.
        ``X`` and ``Y`` are preferably rounded down to powers of two.
    Y_B : int
        Desired resolution for algorithm to compute bounce points.
        Default is double ``Y``.
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
    Bref : float
        Optional. Reference magnetic field strength for normalization.
    Lref : float
        Optional. Reference length scale for normalization.
    is_reshaped : bool
        Whether the arrays in ``data`` are already reshaped to the expected form of
        shape (..., num zeta, num theta) or (num rho, num zeta, num theta).
        This option can be used to iteratively compute bounce integrals one flux
        surface at a time, reducing memory usage. To do so, set to true and provide
        only those chunks of the reshaped data.
        If true, then it is assumed that ``data["iota"]`` has shape
        ``(grid.num_rho,)`` or is a scalar. Default is false.
    is_fourier : bool
        If true, then it is assumed that ``data`` holds Fourier transforms
        as returned by ``Bounce2D.fourier`` and ``data["iota"]`` has shape
        ``(grid.num_rho,)`` or is a scalar. Default is false.
    check : bool
        Flag for debugging. Must be false for JAX transformations.
    spline : bool
        Whether to use cubic splines to compute bounce points.
        Default is true, because the algorithm for efficient root-finding on
        Chebyshev series is not yet implemented.

    """

    # For applications which reduce to computing a nonlinear function of distance
    # along field lines between bounce points, it is required to identify these
    # points with field-line-following coordinates. (In the special case of a linear
    # function summing integrals between bounce points over a flux surface, arbitrary
    # coordinate systems may be used as that task reduces to a surface integral,
    # which is invariant to the order of summation).
    #
    # The DESC coordinate system is related to field-line-following coordinate
    # systems by a relation whose solution is best found with Newton iteration
    # since this solution is unique. Newton iteration is not a globally
    # convergent algorithm to find the real roots of r : ζ ↦ B(ζ) − 1/λ where
    # ζ is a field-line-following coordinate. For this, function approximation
    # of B is necessary.
    #
    # The frequency transform of a map under the chosen basis must be concentrated
    # at low frequencies for the series to converge fast. For periodic
    # (non-periodic) maps, the standard choice for the basis is a Fourier (Chebyshev)
    # series. Both converge exponentially, but the larger region of convergence in the
    # complex plane of Fourier series makes it preferable to choose coordinate
    # systems such that the function to approximate is periodic. One reason Chebyshev
    # polynomials are preferred to other orthogonal polynomial series is
    # fast discrete polynomial transforms (DPT) are implemented via fast transform
    # to Chebyshev then DCT. Therefore, a Fourier-Chebyshev series is chosen
    # to interpolate θ(α,ζ). Using Chebyshev series with the Kosloff and Tal-Ezer
    # almost-equispaced grid does not really show much improvement.
    # Alternative approaches include using filtered Fourier series, Fourier
    # continuation methods, or (preferably) prolate spheroidal wave functions.
    #
    # Function approximation in (α, ζ) coordinates demands particular interpolation
    # points in that coordinate system because there is no transformation that converts
    # series coefficients in periodic coordinates, e.g. (ϑ, ϕ), to a low order
    # polynomial basis in non-periodic coordinates. For example, one can obtain series
    # coefficients in (α, ϕ) coordinates from those in (ϑ, ϕ) as follows
    #   g : ϑ, ϕ ↦ ∑ₘₙ aₘₙ exp(j [mϑ + nϕ])
    #
    #   g : α, ϕ ↦ ∑ₘₙ aₘₙ exp(j [mα + (m ι + n)ϕ])
    # However, the basis for the latter are trigonometric functions with
    # irrational frequencies, courtesy of the irrational rotational transform.
    # Globally convergent root-finding schemes for that basis (at fixed α) are
    # not efficient. The denominator of a close rational could be absorbed into the
    # coordinate ϕ, but this balloons the frequency, and hence degree of the series.
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
        quad=default_quad,
        *,
        automorphism=None,
        Bref=1.0,
        Lref=1.0,
        is_reshaped=False,
        is_fourier=False,
        check=False,
        spline=True,
    ):
        """Returns an object to compute bounce integrals."""
        assert grid.can_fft2
        is_reshaped = is_reshaped or is_fourier
        self._NFP = grid.NFP
        self._m = grid.num_theta
        self._n = grid.num_zeta
        self._n_modes, self._m_modes = rfft2_modes(
            self._n, self._m, domain_fft=(0, 2 * jnp.pi / grid.NFP)
        )
        self._x, self._w = get_quadrature(quad, automorphism)

        self._c = {
            "|B|": data["|B|"] / Bref,
            "B^zeta": data["B^zeta"] * Lref / Bref,
            "T(z)": fourier_chebyshev(
                theta,
                data["iota"] if is_reshaped else grid.compress(data["iota"]),
                jnp.atleast_1d(alpha),
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
            self._c["B(z)"], self._c["knots"] = cubic_spline(
                self._c["T(z)"],
                self._c["|B|"],
                Y_B,
                self._m,
                self._m_modes,
                self._n_modes,
                self._NFP,
                check=check,
            )
        else:
            self._c["B(z)"] = chebyshev(
                self._c["T(z)"],
                self._c["|B|"],
                Y_B,
                self._m,
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
        if (f.shape[-1] % 2) == 0:
            i = (0, -1)
        else:
            i = 0
        # Due to the structure of the problem, often when evaluating the series the
        # number of ζ coordinates at which to compute the toroidal basis functions is
        # less than the number of θ coordinates at which to compute the poloidal basis.
        # Hence, it more efficient to compute the real transform in the poloidal angle.
        # Likewise to perform partial summation in this application, the real transform
        # must be done in the poloidal angle and the complex transform in the toroidal.
        a = rfft2(f, norm="forward").at[..., i].divide(2) * 2
        return a[..., jnp.newaxis, :, :]

    # TODO (#1034): Pass in the previous
    #  θ(α, ζ) coordinates as an initial guess for the next coordinate mapping.
    #  Might be possible to perturb the coefficients of the
    #  θ(α, ζ) since these are related to lambda.

    @staticmethod
    def compute_theta(eq, X=16, Y=32, rho=1.0, iota=None, clebsch=None, **kwargs):
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
        clebsch : jnp.ndarray
            Shape (num rho * X * Y, 3).
            Optional, precomputed Clebsch coordinate tensor-product grid (ρ, α, ζ).
            ``FourierChebyshevSeries.nodes(X,Y,rho,domain=(0,2*jnp.pi))``.
            If supplied ``rho`` is ignored.
        kwargs
            Additional parameters to supply to the coordinate mapping function.
            See ``desc.equilibrium.Equilibrium.map_coordinates``.

        Returns
        -------
        theta : jnp.ndarray
            Shape (num rho, X, Y).
            DESC coordinates θ sourced from the Clebsch coordinates
            ``FourierChebyshevSeries.nodes(X,Y,rho,domain=(0,2*jnp.pi))``.

        """
        if clebsch is None:
            clebsch = FourierChebyshevSeries.nodes(X, Y, rho, domain=(0, 2 * jnp.pi))
        if iota is not None:
            iota = jnp.atleast_1d(iota)
            kwargs["iota"] = jnp.broadcast_to(iota, shape=(X * Y, iota.size)).T.ravel()
        return eq.map_coordinates(
            coords=clebsch,
            inbasis=("rho", "alpha", "zeta"),
            period=(jnp.inf, jnp.inf, jnp.inf),
            tol=kwargs.pop("tol", 1e-7),
            maxiter=kwargs.pop("maxiter", 40),
            **kwargs,
        ).reshape(-1, X, Y, 3)[..., 1]

    def _swap_pitch(self, pitch_inv):
        # Move num pitch axis to front so that the num rho axis broadcasts with
        # the spectral coefficients of the Fourier series defined on that surface.
        # Shape is (num pitch, 1, num rho) or (num pitch, num rho) or (num pitch, ).
        return jnp.moveaxis(atleast_nd(self._c["T(z)"].cheb.ndim - 1, pitch_inv), -1, 0)

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
            z1 = _swap_shape(z1)
            z2 = _swap_shape(z2)
        else:
            z1, z2 = bounce_points(
                pitch_inv,
                self._c["knots"],
                self._c["B(z)"],
                polyder_vec(self._c["B(z)"]),
                num_well,
            )
            if z1.ndim == 4:
                # move rho axis to 0 and alpha axis to 1
                z1 = jnp.swapaxes(z1, 0, 1)
                z2 = jnp.swapaxes(z2, 0, 1)
        return z1, z2

    def _polish_points(self, points, pitch_inv):
        # TODO after (#1243): One application of Newton on Fourier series B - 1/λ.
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
            return self._c["B(z)"].check_intersect1d(
                z1=_swap_shape(points[0]),
                z2=_swap_shape(points[1]),
                k=self._swap_pitch(pitch_inv),
                plot=plot,
                **kwargs,
            )
        else:
            B = self._c["B(z)"]
            if B.ndim == 4:
                # move rho axis to 0 and alpha axis to 1
                B = jnp.swapaxes(B, 0, 1)
            return _check_bounce_points(
                z1=points[0],
                z2=points[1],
                pitch_inv=pitch_inv,
                knots=self._c["knots"],
                B=B,
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
        is_fourier=False,
        check=False,
        plot=False,
        quad=None,
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
            Do not include ``|B|`` or ``B^zeta``.
        points : tuple[jnp.ndarray]
            Shape (num rho, num alpha, num pitch, num well).
            Optional, output of method ``self.points``.
            Tuple of length two (z1, z2) that stores ζ coordinates of bounce points.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of B.
        is_fourier : bool
            If true, then it is assumed that ``data`` holds Fourier transforms
            as returned by ``Bounce2D.fourier``. Default is false.
        check : bool
            Flag for debugging. Must be false for JAX transformations.
        plot : bool
            Whether to plot the quantities in the integrand interpolated to the
            quadrature points of each integral. Ignored if ``check`` is false.
        quad : tuple[jnp.ndarray]
            Optional quadrature points and weights. If given this overrides
            the quadrature chosen when this object was made.

        Returns
        -------
        result : jnp.ndarray or list[jnp.ndarray]
            Shape (num rho, num alpha, num pitch, num well).
            Last axis enumerates the bounce integrals for a given field line
            and pitch value.

        """
        if not isinstance(integrand, (list, tuple)):
            integrand = [integrand]
        data = setdefault(data, {})
        if names is None:
            names = data.keys()
        elif isinstance(names, str):
            names = [names]
        x, w = self._x, self._w if quad is None else quad

        if not is_fourier:
            data = {name: Bounce2D.fourier(data[name]) for name in names}

        if points is None:
            points = self.points(pitch_inv)

        result = self._integrate(
            x,
            w,
            integrand,
            # add axis to broadcast against quadrature points
            self._swap_pitch(1 / pitch_inv)[..., jnp.newaxis],
            data,
            names,
            _swap_shape(points[0]),
            _swap_shape(points[1]),
            check,
            plot,
        )
        return result[0] if len(result) == 1 else result

    # TODO: Singularity subtraction quadrature enables more efficient algorithms.
    #  To compute
    #    ∫ fh dζ where e.g. h = (1−λ|B|)⁰ᐧ⁵
    #  Taylor expand the singular part. For example, to first order
    #    g₁ = f(ζ₁) [−λ [∂|B|/∂ζ|ρ,α](ζ₁)]⁰ᐧ⁵ (ζ − ζ₁)⁰ᐧ⁵
    #    g₂ = f(ζ₂) [+λ [∂|B|/∂ζ|ρ,α](ζ₂)]⁰ᐧ⁵ (ζ₂ − ζ)⁰ᐧ⁵
    #  Then compute with uniform quadrature (analytically) the first (second) integral.
    #    ∫ fh dζ = ∫ fh-(g₁+g₂) dζ + ∫ g₁+g₂ dζ
    #  1. The quadrature points to interpolate to are now θ(α, ζ) and ζ for uniform
    #     ζ ∈ [0, 2π/NFP]. For weakly singular integrals the integrand is still
    #     periodic after the singularity subtraction so this will converge fast.
    #  2. The interpolated values are reused for each integral, so the number of
    #     points to interpolate to is reduced by a factor of ``num_pitch*NFP``.
    #  3. Longer bounce orbits merit more quadrature points than short ones.
    #     This is now possible.
    #  4. Uiform FFT can be used in toroidal direction. Combined with partial
    #     summation the interpolation becomes cheap.
    #     (Same code as ``desc/integrals/_bounce_utils.py::cubic_spline``).
    #  5. The quadrature points are no longer functions of the solutions
    #     to the nonlinear equation λB = 1. In particular, all the ζ values
    #     are constants throughout optimization. This makes AD cheaper.
    #  6. Because the interpolation is now purely a function of ``num_transit``
    #     and θ, rather than the apriori unknown number of bounce points, the
    #     expensive JAX limitation in GitHub issue #1303 is avoided.

    # TODO (#1303).
    def _integrate(self, x, w, integrand, pitch, data, names, z1, z2, check, plot):
        # TODO (#1294): Use non-uniform fast transforms here.
        #  Compare to Cubic-Fourier spline (3x up-sampled with poloidal FFT)
        #  done once in ``desc/_compute/_neoclassical.py::_compute``.
        # num pitch, num alpha, num rho, num well, num quad
        shape = [*z1.shape, x.size]
        # ζ ∈ ℝ and θ ∈ ℝ coordinates of quadrature points
        zeta = flatten_matrix(
            bijection_from_disc(x, z1[..., jnp.newaxis], z2[..., jnp.newaxis])
        )
        theta = self._c["T(z)"].eval1d(zeta)

        # Goal is to reuse the same Vandermonde array to interpolate. This took
        # some care to appease JIT to fuse the operations. Be careful if editing
        # as what usually qualifies as a cosmetic change may cause memory leaks.
        vander = rfft2_vander(zeta, theta, self._n_modes, self._m_modes)
        data = {name: (vander * data[name]).real.sum((-2, -1)) for name in names}
        data["B^zeta"] = (vander * self._c["B^zeta"]).real.sum((-2, -1))
        B = (vander * self._c["|B|"]).real.sum((-2, -1))
        data["theta"] = theta
        data["zeta"] = zeta

        # Strictly increasing zeta knots enforces dζ > 0.
        # To retain dℓ = |B|/(B⋅∇ζ) dζ > 0 after fixing dζ > 0, we require
        # B⋅∇ζ > 0. This is equivalent to changing the sign of ∇ζ
        # or (∂ℓ/∂ζ)|ρ,a. Recall dζ = ∇ζ⋅dR ⇔ 1 = ∇ζ⋅(e_ζ|ρ,a).
        dl_dz = B / jnp.abs(data["B^zeta"])
        cov = grad_bijection_from_disc(z1, z2)
        result = [
            _swap_shape((f(data, B, pitch) * dl_dz).reshape(shape).dot(w) * cov)
            for f in integrand
        ]

        if check:
            shape[-3], shape[0] = shape[0], shape[-3]
            _check_interp(
                shape,
                *map(_swap_shape, (zeta, 1 / dl_dz, B)),
                [_swap_shape(data[name]) for name in names],
                result,
                plot=plot,
            )

        return result

    def interp_to_argmin(self, f, points, *, is_fourier=False):
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
            msg="Set spline to true until implemented.",
        )
        return _swap_shape(
            interp_fft_to_argmin(
                self._c["T(z)"],
                f if is_fourier else Bounce2D.fourier(f),
                map(_swap_shape, points),
                self._c["knots"],
                self._c["B(z)"],
                polyder_vec(self._c["B(z)"]),
                m=self._m,
                n=self._n,
                NFP=self._NFP,
            )
        )

    def compute_fieldline_length(self, quad=None):
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

        Returns
        -------
        length : jnp.ndarray
            Shape (num rho, ).

        """
        # Integrating an analytic oscillatory map so a high order quadrature is ideal.
        # Difficult to pick the right frequency for Filon quadrature in general, which
        # would work best, especially at high NFP. Gauss-Legendre is superior to
        # Clenshaw-Curtis for smooth oscillatory maps. Any prolate spheroidal wave
        # function quadrature would be an improvement.
        deg = (
            self._c["B(z)"].Y
            if isinstance(self._c["B(z)"], PiecewiseChebyshevSeries)
            else self._c["knots"].size // self._c["T(z)"].X
        ) // 2
        x, w = leggauss(deg) if quad is None else quad
        dz_dx = jnp.pi

        # Let m, n denote the poloidal and toroidal Fourier resolution. We need to
        # compute a set of 2D Fourier series each on non-uniform tensor product grids
        # of size |𝛉|×|𝛇| where |𝛉| = num alpha × num transit and |𝛇| is quadrature
        # resolution. Partial summation is more efficient than direct evaluation when
        # mn|𝛉||𝛇| > mn|𝛇| + m|𝛉||𝛇| or equivalently n|𝛉| > n + |𝛉|.

        zeta = bijection_from_disc(x, 0, 2 * jnp.pi)
        # Shape broadcasts with (num rho, num zeta, m)
        par_sum = ifft_non_uniform(
            zeta[:, jnp.newaxis],
            self._c["B^zeta"],  # Shape broadcasts with (num rho, 1, n, m).
            _modes=self._n_modes,
            domain=(0, 2 * jnp.pi / self._NFP),
            axis=-2,
        )
        # θ at roots of Legendre polynomial in ζ
        theta = idct_non_uniform(
            x, self._c["T(z)"].cheb[..., jnp.newaxis, :], self._c["T(z)"].Y
        )
        par_sum = irfft_non_uniform(
            theta, par_sum[..., jnp.newaxis, :, :], self._m, _modes=self._m_modes
        )
        # B⋅∇ζ never vanishes, and hence has the same sign on a flux surface,
        # so we may take absolute value after the reduction.
        return jnp.abs(jnp.reciprocal(par_sum).dot(w).sum(-1).mean(0)) * dz_dx
        # Simple mean over α because when the toroidal angle extends
        # beyond one transit we need to weight all field lines uniformly,
        # regardless of their area wrt α.

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
                B = PiecewiseChebyshevSeries(B.cheb[m, l], B.domain)
            elif B.cheb.ndim == 3:
                B = PiecewiseChebyshevSeries(B.cheb[m], B.domain)
            if pitch_inv is not None:
                z1, z2 = B.intersect1d(pitch_inv)
                kwargs["z1"] = z1
                kwargs["z2"] = z2
                kwargs["k"] = pitch_inv
            fig, ax = B.plot1d(B.cheb, **kwargs)
        else:
            if B.ndim == 4:
                B = B[m, l]
            elif B.ndim == 3:
                B = B[m]
            if pitch_inv is not None:
                z1, z2 = bounce_points(pitch_inv, self._c["knots"], B, polyder_vec(B))
                kwargs["z1"] = z1
                kwargs["z2"] = z2
                kwargs["k"] = pitch_inv
            fig, ax = plot_ppoly(PPoly(B.T, self._c["knots"]), **kwargs)
        return fig, ax

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
            T = PiecewiseChebyshevSeries(T.cheb[m, l], T.domain)
        elif T.cheb.ndim == 3:
            T = PiecewiseChebyshevSeries(T.cheb[m], T.domain)
        kwargs.setdefault(
            "title",
            rf"Poloidal angle $\theta$ on field line $\rho(l={l})$, $\alpha(m={m})$",
        )
        kwargs.setdefault("vlabel", r"$\theta$")
        fig, ax = T.plot1d(T.cheb, **_set_default_plot_kwargs(kwargs, l, m))
        return fig, ax


class Bounce1D(Bounce):
    """Computes bounce integrals using one-dimensional local spline methods.

    The bounce integral is defined as ∫ f(ρ,α,λ,ℓ) dℓ where

    * dℓ parameterizes the distance along the field line in meters.
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
        toroidal transit.
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
        respectively, reducing memory usage. To do so, set to true and provide
        only those chunks of the reshaped data. Default is false.
    check : bool
        Flag for debugging. Must be false for JAX transformations.

    """

    required_names = ["B^zeta", "B^zeta_z|r,a", "|B|", "|B|_z|r,a"]

    def __init__(
        self,
        grid,
        data,
        quad=default_quad,
        *,
        automorphism=None,
        Bref=1.0,
        Lref=1.0,
        is_reshaped=False,
        check=False,
    ):
        """Returns an object to compute bounce integrals."""
        assert grid.is_meshgrid
        self._data = {
            # Strictly increasing zeta knots enforces dζ > 0.
            # To retain dℓ = |B|/(B⋅∇ζ) dζ > 0 after fixing dζ > 0, we require
            # B⋅∇ζ > 0. This is equivalent to changing the sign of ∇ζ
            # or (∂ℓ/∂ζ)|ρ,a. Recall dζ = ∇ζ⋅dR ⇔ 1 = ∇ζ⋅(e_ζ|ρ,a).
            "|b^zeta|": jnp.abs(data["B^zeta"]) * Lref / data["|B|"],
            "|B|": data["|B|"] / Bref,
            "|B|_z|r,a": data["|B|_z|r,a"] / Bref,  # This is already the correct sign.
        }
        self._data["|b^zeta|_z|r,a"] = (
            data["B^zeta_z|r,a"] * jnp.sign(data["B^zeta"]) * Lref
            - self._data["|b^zeta|"] * data["|B|_z|r,a"]
        ) / data["|B|"]
        if not is_reshaped:
            for name in self._data:
                self._data[name] = Bounce1D.reshape(grid, self._data[name])
        self._x, self._w = get_quadrature(quad, automorphism)

        # Compute local splines.
        # Note it is simple to do FFT across field line axis, and spline
        # Fourier coefficients across ζ to obtain Fourier-CubicSpline of functions.
        # The point of Bounce2D is to do such a 2D interpolation without
        # rebuilding DESC transforms each time an objective is computed.
        self._zeta = grid.compress(grid.nodes[:, 2], surface_label="zeta")
        # Shape is (num rho, num alpha, N - 1, -1).
        self._B = jnp.moveaxis(
            CubicHermiteSpline(
                x=self._zeta,
                y=self._data["|B|"],
                dydx=self._data["|B|_z|r,a"],
                axis=-1,
                check=check,
            ).c,
            source=(0, 1),
            destination=(-1, -2),
        )
        self._dB_dz = polyder_vec(self._B)

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
        # if rho axis exists, then add alpha axis
        if jnp.ndim(pitch_inv) == 2:
            pitch_inv = pitch_inv[:, jnp.newaxis]
        return bounce_points(pitch_inv, self._zeta, self._B, self._dB_dz, num_well)

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
            z1=points[0],
            z2=points[1],
            pitch_inv=pitch_inv,
            knots=self._zeta,
            B=self._B,
            plot=plot,
            **kwargs,
        )

    # TODO (#1428): Add option for adaptive quadrature with quadax
    #  quadax.quadgk with the c.o.v. used for legendre works best.
    #  Some people want more accurate computation on W shaped wells.
    def integrate(
        self,
        integrand,
        pitch_inv,
        data=None,
        names=None,
        points=None,
        *,
        method="cubic",
        check=False,
        plot=False,
        quad=None,
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
            Do not include ``|B|``.
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
        check : bool
            Flag for debugging. Must be false for JAX transformations.
        plot : bool
            Whether to plot the quantities in the integrand interpolated to the
            quadrature points of each integral. Ignored if ``check`` is false.
        quad : tuple[jnp.ndarray]
            Optional quadrature points and weights. If given this overrides
            the quadrature chosen when this object was made.

        Returns
        -------
        result : jnp.ndarray or list[jnp.ndarray]
            Shape (num rho, num alpha, num pitch, num well).
            Last axis enumerates the bounce integrals for a given field line
            and pitch value.

        """
        if not isinstance(integrand, (list, tuple)):
            integrand = [integrand]
        data = setdefault(data, {})
        if names is None:
            names = data.keys()
        elif isinstance(names, str):
            names = [names]
        x, w = self._x, self._w if quad is None else quad

        # if rho axis exists, then add alpha axis
        if jnp.ndim(pitch_inv) == 2:
            pitch_inv = pitch_inv[:, jnp.newaxis]
        # add axis to broadcast against quadrature points
        pitch = jnp.atleast_1d(1 / pitch_inv)[..., jnp.newaxis]

        if points is None:
            points = bounce_points(pitch_inv, self._zeta, self._B, self._dB_dz)

        if kwargs.get("batch", True):
            result = self._integrate(
                x,
                w,
                integrand,
                pitch,
                data,
                names,
                points,
                method,
                check,
                plot,
                batch=True,
            )
        else:
            # Perform integrals for a particular field line and pitch one at a time.
            def loop(points):  # over num well axis
                return self._integrate(
                    x,
                    w,
                    integrand,
                    pitch,
                    data,
                    names,
                    points,
                    method,
                    check=False,
                    plot=False,
                    batch=False,
                )

            result = batch_map(loop, [jnp.moveaxis(z, -1, 0) for z in points], 1)
            result = [jnp.moveaxis(r, 0, -1) for r in result]

        return result[0] if len(result) == 1 else result

    # TODO (#1303).
    def _integrate(
        self, x, w, integrand, pitch, data, names, points, method, check, plot, batch
    ):
        z1, z2 = points
        # (..., num pitch, num well, num quad)
        shape = (*z1.shape, x.size)
        # ζ ∈ ℝ coordinates of quadrature points
        zeta = bijection_from_disc(x, z1[..., jnp.newaxis], z2[..., jnp.newaxis])
        if batch:
            zeta = flatten_matrix(zeta)

        b_sup_z = interp1d_Hermite_vec(
            zeta,
            self._zeta,
            self._data["|b^zeta|"][..., jnp.newaxis, :],
            self._data["|b^zeta|_z|r,a"][..., jnp.newaxis, :],
        )
        B = interp1d_Hermite_vec(
            zeta,
            self._zeta,
            self._data["|B|"][..., jnp.newaxis, :],
            self._data["|B|_z|r,a"][..., jnp.newaxis, :],
        )

        # Spline each function separately so that operations in the integrand
        # that do not preserve smoothness can be captured.
        data = {
            name: interp1d_vec(
                zeta, self._zeta, data[name][..., jnp.newaxis, :], method=method
            )
            for name in names
        }
        cov = grad_bijection_from_disc(z1, z2)
        result = [
            (f(data, B, pitch) / b_sup_z).reshape(shape).dot(w) * cov for f in integrand
        ]

        if check:
            _check_interp(
                shape,
                zeta,
                b_sup_z,
                B,
                [data[name] for name in names],
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
        return interp_to_argmin(f, points, self._zeta, self._B, self._dB_dz, method)

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
        B, dB_dz = self._B, self._dB_dz
        if B.ndim == 4:
            B = B[l]
            dB_dz = dB_dz[l]
        if B.ndim == 3:
            B = B[m]
            dB_dz = dB_dz[m]
        if pitch_inv is not None:
            errorif(
                jnp.ndim(pitch_inv) > 1,
                msg=f"Got pitch_inv.ndim={jnp.ndim(pitch_inv)}, but expected 1.",
            )
            z1, z2 = bounce_points(pitch_inv, self._zeta, B, dB_dz)
            kwargs["z1"] = z1
            kwargs["z2"] = z2
            kwargs["k"] = pitch_inv
        fig, ax = plot_ppoly(
            PPoly(B.T, self._zeta), **_set_default_plot_kwargs(kwargs, l, m)
        )
        return fig, ax
