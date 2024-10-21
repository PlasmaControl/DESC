"""Methods for computing bounce integrals (singular or otherwise)."""

from abc import ABC, abstractmethod

from interpax import CubicHermiteSpline, PPoly
from orthax.legendre import leggauss

from desc.backend import jnp
from desc.integrals.basis import FourierChebyshevSeries, PiecewiseChebyshevSeries
from desc.integrals.bounce_utils import (
    _bounce_quadrature,
    _check_bounce_points,
    _check_interp,
    _set_default_plot_kwargs,
    bounce_points,
    chebyshev,
    cubic_spline,
    fourier_chebyshev,
    get_pitch_inv_quad,
    interp_fft_to_argmin,
    interp_to_argmin,
    plot_ppoly,
)
from desc.integrals.interp_utils import (
    _fourier,
    idct_non_uniform,
    interp_rfft2,
    irfft2_non_uniform,
    polyder_vec,
)
from desc.integrals.quad_utils import (
    automorphism_sin,
    bijection_from_disc,
    get_quadrature,
    grad_automorphism_sin,
    grad_bijection_from_disc,
)
from desc.io import IOAble
from desc.utils import atleast_nd, errorif, flatten_matrix, setdefault


class Bounce(IOAble, ABC):
    """Abstract class for bounce integrals."""

    get_pitch_inv_quad = staticmethod(get_pitch_inv_quad)

    @abstractmethod
    def points(self, pitch_inv, *, num_well=None):
        """Compute bounce points."""

    @abstractmethod
    def check_points(self, points, pitch_inv, *, plot=True, **kwargs):
        """Check that bounce points are computed correctly."""

    @abstractmethod
    def integrate(
        self, integrand, pitch_inv, f=None, weight=None, points=None, *, quad=None
    ):
        """Bounce integrate ∫ f(λ, ℓ) dℓ."""


def _swap_pl(f):
    # Given shape (num rho, num pitch, -1) or (num pitch, num rho, -1)
    # or (num pitch, -1), swap num rho and num pitch axes.
    assert f.ndim <= 3
    return jnp.swapaxes(f, 0, -2)


class Bounce2D(Bounce):
    """Computes bounce integrals using two-dimensional pseudo-spectral methods.

    The bounce integral is defined as ∫ f(λ, ℓ) dℓ where

    * dℓ parameterizes the distance along the field line in meters.
    * f(λ, ℓ) is the quantity to integrate along the field line.
    * The boundaries of the integral are bounce points ℓ₁, ℓ₂ s.t. λ|B|(ℓᵢ) = 1.
    * λ is a constant defining the integral proportional to the magnetic moment
      over energy.
    * |B| is the norm of the magnetic field.

    For a particle with fixed λ, bounce points are defined to be the location on the
    field line such that the particle's velocity parallel to the magnetic field is zero.
    The bounce integral is defined up to a sign. We choose the sign that corresponds to
    the particle's guiding center trajectory traveling in the direction of increasing
    field-line-following coordinate ζ.


    Overview
    --------
    Magnetic field line with label α, defined by B = ∇ρ × ∇α, is determined from
      α : ρ, θ, ζ ↦ θ + λ(ρ,θ,ζ) − ι(ρ) [ζ + ω(ρ,θ,ζ)]
    Interpolate Fourier-Chebyshev series to DESC poloidal coordinate.
      θ : α, ζ ↦ tₘₙ exp(jmα) Tₙ(ζ)
    Compute |B| along field lines.
      |B| : α, ζ ↦  bₙ(θ(α, ζ)) Tₙ(ζ)
    Compute bounce points.
      r(ζₖ) = |B|(ζₖ) − 1/λ = 0
    Interpolate smooth components of integrand with FFTs.
      G : α, ζ ↦ gₘₙ exp(j [m θ(α,ζ) + n ζ] )
    Perform Gaussian quadrature after removing singularities.
      Fᵢ : λ, ζ₁, ζ₂ ↦  ∫ᵢ f(λ, ζ, {Gⱼ}) dζ

    If the map G is multivalued at a physical location, then it is still
    permissible if separable into single valued and multivalued parts.
    In that case, supply the single valued parts, which will be interpolated
    with FFTs, and use the provided coordinates θ,ζ ∈ ℝ to compose G.

    Notes
    -----
    For applications which reduce to computing a nonlinear function of distance
    along field lines between bounce points, it is required to identify these
    points with field-line-following coordinates. (In the special case of a linear
    function summing integrals between bounce points over a flux surface, arbitrary
    coordinate systems may be used as that task reduces to a surface integral,
    which is invariant to the order of summation).

    The DESC coordinate system is related to field-line-following coordinate
    systems by a relation whose solution is best found with Newton iteration
    since this solution is unique. Newton iteration is not a globally
    convergent algorithm to find the real roots of r : ζ ↦ |B|(ζ) − 1/λ where
    ζ is a field-line-following coordinate. For this, function approximation
    of |B| is necessary.

    Therefore, to compute bounce points {(ζ₁, ζ₂)}, we approximate |B| by a
    series expansion of basis functions parameterized by a single variable ζ,
    restricting the class of basis functions to low order (e.g. n = 2ᵏ where
    k is small) algebraic or trigonometric polynomial with integer frequencies.
    These are the two classes useful for function approximation and for which
    there exists globally convergent root-finding algorithms. We require low
    order because the computation expenses grow with the number of potential
    roots, and the theorem of algebra states that number is n (2n) for algebraic
    (trigonometric) polynomials of degree n.

    The frequency transform of a map under the chosen basis must be concentrated
    at low frequencies for the series to converge fast. For periodic
    (non-periodic) maps, the best basis is a Fourier (Chebyshev) series. Both
    converge exponentially, but the larger region of convergence in the complex
    plane of Fourier series make it preferable in practice to choose coordinate
    systems such that the function to approximate is periodic. The Chebyshev
    polynomials are preferred to other orthogonal polynomial series since
    fast discrete polynomial transforms (DPT) are implemented via fast transform
    to Chebyshev then DCT. Although nothing prohibits a direct DPT, we want to
    rely on existing libraries. Therefore, a Fourier-Chebyshev series is chosen
    to interpolate θ(α,ζ), and a piecewise Chebyshev series interpolates |B|(ζ).

    * An alternative to Chebyshev series is
      [filtered Fourier series](doi.org/10.1016/j.aml.2006.10.001).
      We did not implement or benchmark against that.
    * θ is not interpolated with a double Fourier series θ(ϑ, ζ) because
      it is impossible to approximate an unbounded function with a finite Fourier
      series. Due to Gibbs effects, this statement holds even when the goal is to
      approximate θ over one branch cut. The proof uses analytic continuation.
    * The advantage of Fourier series in DESC coordinates is that they may use the
      spectrally condensed variable ζ* = NFP ζ. This cannot be done in any other
      coordinate system, regardless of whether the basis functions are periodic.
      The strategy of parameterizing |B| along field lines with a single variable
      in Clebsch coordinates (as opposed to two variables in straight-field line
      coordinates) also serves to minimize this penalty since evaluation of |B|
      when computing bounce points will be less expensive (assuming the 2D
      Fourier resolution of |B|(ϑ, ϕ) is larger than the 1D Chebyshev resolution).

    Computing accurate series expansions in (α, ζ) coordinates demands
    particular interpolation points in that coordinate system. Newton iteration
    is used to compute θ at these points. Note that interpolation is necessary
    because there is no transformation that converts series coefficients in
    periodic coordinates, e.g. (ϑ, ϕ), to a low order polynomial basis in
    non-periodic coordinates. For example, one can obtain series coefficients in
    (α, ϕ) coordinates from those in (ϑ, ϕ) as follows
      g : ϑ, ϕ ↦ ∑ₘₙ aₘₙ exp(j [mϑ + nϕ])

      g : α, ϕ ↦ ∑ₘₙ aₘₙ exp(j [mα + (m ι + n)ϕ])
    However, the basis for the latter are trigonometric functions with
    irrational frequencies, courtesy of the irrational rotational transform.
    Globally convergent root-finding schemes for that basis (at fixed α) are
    not known. The denominator of a close rational could be absorbed into the
    coordinate ϕ, but this balloons the frequency, and hence the degree of the
    series.

    After computing the bounce points, the supplied quadrature is performed.
    By default, this is a Gauss quadrature after removing the singularity.
    Fast fourier transforms interpolate smooth functions in the integrand to the
    quadrature nodes. Quadrature is chosen over Runge-Kutta methods of the form
        ∂Fᵢ/∂ζ = f(λ,ζ,{Gⱼ}) subject to Fᵢ(ζ₁) = 0
    A fourth order Runge-Kutta method is equivalent to a quadrature
    with Simpson's rule. Our quadratures resolve these integrals more
    efficiently, and the fixed nature of quadrature performs better on GPUs.

    Fast transforms are used where possible. Fast multipoint methods are not
    implemented. For non-uniform interpolation, MMTs are used. It will be
    worthwhile to use the inverse non-uniform fast transforms.

    Additional notes on multivalued coordinates:
    The definition of α in B = ∇ρ × ∇α on an irrational magnetic surface
    implies the angle θ(α, ζ) is multivalued at a physical location.
    In particular, following an irrational field, the single-valued θ grows
    to ∞ as ζ → ∞. Therefore, it is impossible to approximate this map using
    single-valued basis functions defined on a bounded subset of ℝ²
    (recall continuous functions on compact sets attain their maximum).
    Still, it suffices to interpolate θ over one branch cut. We choose the
    branch cut defined by (α, ζ) ∈ [0, 2π]. Here the bound θ ∈ [0, 4π] holds.

    Likewise, α is multivalued. As the field line is followed, the label
    jumps to α ∉ [0, 2π] after completing some toroidal transit. Therefore,
    the map θ(α, ζ) must be periodic in α with period 2π. At every point
    ζₚ ∈ [2π k, 2π ℓ] where k, ℓ ∈ ℤ where the field line completes a
    poloidal transit there is guaranteed to exist a discrete jump
    discontinuity in θ at ζ = 2π ℓ(p), starting the toroidal transit.
    Recall a jump discontinuity appears as an infinitely sharp cut without
    Gibbs effects. To recover the single-valued θ(α, ζ) from the function
    approximation over one branch cut, at every ζ = 2π ℓ we can add either
    0 or 2π or 4π to the next cut of θ.

    Examples
    --------
    See ``tests/test_integrals.py::TestBounce2D::test_bounce2d_checks``.

    See Also
    --------
    Bounce1D : Uses one-dimensional local spline methods for the same task.


    Comparison to Bounce1D
    ----------------------
    ``Bounce2D`` solves the dominant cost of optimization objectives relying on
    ``Bounce1D``: interpolating DESC's 3D transforms to an optimization-step
    dependent grid that is dense enough for function approximation with local
    splines. This is sometimes referred to as off-grid interpolation in literature;
    it is often a bottleneck.

    * The function approximation done here requires DESC transforms on a fixed
      grid with typical resolution, using FFTs to compute the map α,ζ ↦ θ(α,ζ)
      between coordinate systems. This enables evaluating functions along
      field lines without root-finding.
    * The faster convergence of spectral interpolation requires a less dense
      grid to interpolate onto from DESC's 3D transforms.
    * Spectral approximation is more accurate than cubic splines.
    * 2D interpolation enables tracing the field line for many toroidal transits.
    * The drawback is that evaluating a Fourier series with resolution F at Q
      non-uniform quadrature points takes 𝒪([F+Q] log[F] log[1/ε]) time
      whereas cubic splines take 𝒪(C Q) time. However, as NFP increases,
      F decreases whereas C increases. Also, Q >> F and Q >> C.

    Attributes
    ----------
    required_names : list
        Names in ``data_index`` required to compute bounce integrals.

    """

    required_names = ["B^zeta", "|B|", "iota"]

    def __init__(
        self,
        grid,
        data,
        theta,
        Y_B=None,
        num_transit=32,
        # TODO: Allow multiple starting labels for near-rational surfaces.
        #  Can just add axis for piecewise chebyshev stuff cheb.
        alpha=0.0,
        quad=leggauss(32),
        automorphism=(automorphism_sin, grad_automorphism_sin),
        *,
        Bref=1.0,
        Lref=1.0,
        is_reshaped=False,
        check=False,
        spline=True,
    ):
        """Returns an object to compute bounce integrals.

        Notes
        -----
        Performance may improve significantly if the spectral
        resolutions ``M``, ``N``, ``X``, ``Y``, and ``Y_B`` are powers of two.

        Parameters
        ----------
        grid : Grid
            Tensor-product grid in (ρ, θ, ζ) with uniformly spaced nodes
            (θ, ζ) ∈ [0, 2π) × [0, 2π/NFP). The ζ coordinates (the unique values prior
            to taking the tensor-product) must be strictly increasing.
            Below shape notation defines ``M=grid.num_theta`` and ``N=grid.num_zeta``.
        data : dict[str, jnp.ndarray]
            Data evaluated on ``grid``.
            Must include names in ``Bounce2D.required_names``.
        theta : jnp.ndarray
            Shape (num rho, X, Y).
            DESC coordinates θ sourced from the Clebsch coordinates
            ``FourierChebyshevSeries.nodes(X,Y,rho,domain=(0,2*jnp.pi))``.
        Y_B : int
            Desired Chebyshev spectral resolution for |B|.
            Default is to double the resolution of ``theta``.
        alpha : float
            Starting field line poloidal label.
        num_transit : int
            Number of toroidal transits to follow field line.
        quad : tuple[jnp.ndarray]
            Quadrature points xₖ and weights wₖ for the approximate evaluation of an
            integral ∫₋₁¹ g(x) dx = ∑ₖ wₖ g(xₖ). Default is 32 points.
            For weak singular integrals, use ``chebgauss2`` from
            ``desc.integrals.quad_utils``.
            For strong singular integrals, use ``leggauss``.
        automorphism : tuple[Callable] or None
            The first callable should be an automorphism of the real interval [-1, 1].
            The second callable should be the derivative of the first. This map defines
            a change of variable for the bounce integral. The choice made for the
            automorphism will affect the performance of the quadrature method.
            For weak singular integrals, use ``None``.
            For strong singular integrals, use
            ``(automorphism_sin,grad_automorphism_sin)`` from
            ``desc.integrals.quad_utils``.
        Bref : float
            Optional. Reference magnetic field strength for normalization.
        Lref : float
            Optional. Reference length scale for normalization.
        is_reshaped : bool
            Whether the arrays in ``data`` are already reshaped to the expected form of
            shape (..., M, N) or (num rho, M, N). This option can be used to iteratively
            compute bounce integrals one flux surface at a time, reducing memory usage
            To do so, set to true and provide only those axes of the reshaped data.
            Default is false.
        check : bool
            Flag for debugging. Must be false for JAX transformations.
        spline : bool
            Whether to use cubic splines to compute bounce points.
            Default is true, because the algorithm for efficient root-finding on
            Chebyshev series algorithm is not yet implemented.
            When using splines, it is recommended to reduce the ``num_well``
            parameter in the ``points`` method from ``3*Y_B*num_transit`` to
            at most ``Y_B*num_transit``.

        """
        errorif(grid.sym, NotImplementedError, msg="Need grid that works with FFTs.")

        self._M = grid.num_theta
        self._N = grid.num_zeta
        self._NFP = grid.NFP
        self._alpha = alpha
        self._x, self._w = get_quadrature(quad, automorphism)

        # spectral coefficients
        self._c = {
            "|B|": _fourier(grid, data["|B|"] / Bref, is_reshaped),
            # Strictly increasing zeta knots enforces dζ > 0.
            # To retain dℓ = (|B|/B^ζ) dζ > 0 after fixing dζ > 0, we require
            # B^ζ = B⋅∇ζ > 0. This is equivalent to changing the sign of ∇ζ
            # or (∂ℓ/∂ζ)|ρ,a. Recall dζ = ∇ζ⋅dR ⇔ 1 = ∇ζ⋅(e_ζ|ρ,a).
            "B^zeta": _fourier(
                grid, jnp.abs(data["B^zeta"]) * Lref / Bref, is_reshaped
            ),
            "T(z)": fourier_chebyshev(
                theta, grid.compress(data["iota"]), alpha, num_transit
            ),
        }
        Y_B = setdefault(Y_B, theta.shape[-1] * 2)
        if spline:
            self._c["B(z)"], self._c["knots"] = cubic_spline(
                self._M,
                self._N,
                self._NFP,
                self._c["T(z)"],
                self._c["|B|"],
                Y_B,
                check=check,
            )
        else:
            self._c["B(z)"] = chebyshev(
                self._M,
                self._N,
                self._NFP,
                self._c["T(z)"],
                self._c["|B|"],
                Y_B,
            )

    @staticmethod
    def reshape_data(grid, *arys):
        """Reshape ``data`` arrays for acceptable input to ``integrate``.

        Parameters
        ----------
        grid : Grid
            Tensor-product grid in (ρ, θ, ζ).
        arys : jnp.ndarray
            Data evaluated on grid.

        Returns
        -------
        f : jnp.ndarray
            Shape (num rho, M, N).
            Reshaped data which may be given to ``integrate``.

        """
        f = [grid.meshgrid_reshape(d, "rtz") for d in arys]
        return f if len(f) > 1 else f[0]

    # TODO: After GitHub issue #1034 is resolved, we should pass in the previous
    #  θ(α, ζ) coordinates as an initial guess for the next coordinate mapping.
    #  Think more about whether possible to perturb the coefficients of the
    #  θ(α, ζ) since these are related to lambda.

    @staticmethod
    def compute_theta(eq, X=16, Y=32, rho=1.0, iota=None, clebsch=None, **kwargs):
        """Return DESC coordinates θ of (α,ζ) Fourier Chebyshev basis nodes.

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium to use defining the coordinate mapping.
        X : int
            Grid resolution in poloidal direction for Clebsch coordinate grid.
            Preferably power of 2.
        Y : int
            Grid resolution in toroidal direction for Clebsch coordinate grid.
            Preferably power of 2.
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
            kwargs["iota"] = jnp.broadcast_to(iota, shape=(Y, X, iota.size)).T.ravel()
        return eq.map_coordinates(
            coords=clebsch,
            inbasis=("rho", "alpha", "zeta"),
            period=(jnp.inf, jnp.inf, jnp.inf),
            tol=kwargs.pop("tol", 1e-7),
            maxiter=kwargs.pop("maxiter", 40),
            **kwargs,
        ).reshape(-1, X, Y, 3)[..., 1]

    def points(self, pitch_inv, *, num_well=None):
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
            pitch along each field line. This is useful if ``num_well`` tightly
            bounds the actual number. As a reference, there are typically 20 wells
            per toroidal transit for a given pitch. You can check this by plotting
            the field lines with the ``check_points`` method.

            If not specified, then all bounce points are returned. If there were fewer
            wells detected along a field line than the size of the last axis of the
            returned arrays, then that axis is padded with zero.

        Returns
        -------
        z1, z2 : tuple[jnp.ndarray]
            Shape (num rho, num pitch, num well).
            Tuple of length two (z1, z2) that stores ζ coordinates of bounce points.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of |B|.

            If there were less than ``num_well`` wells detected along a field line,
            then the last axis, which enumerates bounce points for a particular field
            line and pitch, is padded with zero.

        """
        if isinstance(self._c["B(z)"], PiecewiseChebyshevSeries):
            # We move num pitch axis to front so that the num rho axis broadcasts
            # with the spectral coefficients (whose first axis is also num rho),
            # assuming this axis exists.
            pitch_inv = atleast_nd(self._c["T(z)"].cheb.ndim - 1, pitch_inv).T
            z1, z2 = map(
                _swap_pl,
                self._c["B(z)"].intersect1d(pitch_inv, num_intersect=num_well),
            )
        else:
            z1, z2 = bounce_points(
                pitch_inv,
                self._c["knots"],
                self._c["B(z)"],
                polyder_vec(self._c["B(z)"]),
                num_well,
            )
        return z1, z2

    def _polish_points(self, points, pitch_inv):
        # TODO: One application of Newton on Fourier series |B| - pitch_inv.
        raise NotImplementedError

    def check_points(self, points, pitch_inv, *, plot=True, **kwargs):
        """Check that bounce points are computed correctly.

        Parameters
        ----------
        points : tuple[jnp.ndarray]
            Shape (num rho, num pitch, num well).
            Output of method ``self.points``.
            Tuple of length two (z1, z2) that stores ζ coordinates of bounce points.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of |B|.
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
            ``desc/integrals/bounce_utils.py::plot_ppoly``.

        Returns
        -------
        plots : list
            Matplotlib (fig, ax) tuples for the 1D plot of each field line.

        """
        kwargs.setdefault("hlabel", r"$\alpha = $" + str(self._alpha) + r", $\zeta$")
        kwargs = _set_default_plot_kwargs(kwargs)
        if isinstance(self._c["B(z)"], PiecewiseChebyshevSeries):
            return self._c["B(z)"].check_intersect1d(
                # We move num pitch axis to front so that the num rho axis broadcasts
                # with the spectral coefficients (whose first axis is also num rho),
                # assuming this axis exists.
                z1=_swap_pl(points[0]),
                z2=_swap_pl(points[1]),
                k=atleast_nd(self._c["B(z)"].cheb.ndim - 1, pitch_inv).T,
                plot=plot,
                **kwargs,
            )
        else:
            return _check_bounce_points(
                z1=points[0],
                z2=points[1],
                pitch_inv=pitch_inv,
                knots=self._c["knots"],
                B=self._c["B(z)"],
                plot=plot,
                **kwargs,
            )

    def integrate(
        self,
        integrand,
        pitch_inv,
        f=None,
        weight=None,
        points=None,
        *,
        check=False,
        plot=False,
        quad=None,
    ):
        """Bounce integrate ∫ f(λ, ℓ) dℓ.

        Computes the bounce integral ∫ f(λ, ℓ) dℓ for every field line and pitch.

        Notes
        -----
        Make sure to replace √(1−λ|B|) with √|1−λ|B|| in ``integrand`` to account
        for imperfect computation of bounce points.

        Parameters
        ----------
        integrand : callable
            The composition operator on the set of functions in ``f`` that maps the
            functions in ``f`` to the integrand f(λ, ℓ) in ∫ f(λ, ℓ) dℓ. It should
            accept the arrays in ``f`` as arguments as well as the additional keyword
            arguments: ``B``, ``pitch``, and ``zeta``. A quadrature will be performed
            to approximate the bounce integral of
            ``integrand(*f,B=B,pitch=pitch,zeta=zeta)``.
        pitch_inv : jnp.ndarray
            Shape (num rho, num pitch).
            1/λ values to compute the bounce integrals. 1/λ(ρ) is specified by
            ``pitch_inv[ρ]`` where in the latter the labels are interpreted
            as the indices that correspond to that field line.
        f : list[jnp.ndarray] or jnp.ndarray
            Shape (num rho, M, N).
            Real scalar-valued periodic functions in (θ, ζ) ∈ [0, 2π) × [0, 2π/NFP)
            evaluated on the ``grid`` supplied to construct this object. These functions
            should be arguments to the callable ``integrand``. Use the method
            ``Bounce2D.reshape_data`` to reshape the data into the expected shape.
        weight : jnp.ndarray
            Shape (num rho, M, N).
            Real scalar-valued periodic functions in (θ, ζ) ∈ [0, 2π) × [0, 2π/NFP)
            evaluated on the ``grid`` supplied to construct this object.
            If supplied, the bounce integral labeled by well j is weighted such that
            the returned value is w(j) ∫ f(λ, ℓ) dℓ, where w(j) is ``weight``
            interpolated to the deepest point in that magnetic well. Use the method
            ``Bounce2D.reshape_data`` to reshape the data into the expected shape.
        points : tuple[jnp.ndarray]
            Shape (num rho, num pitch, num well).
            Optional, output of method ``self.points``.
            Tuple of length two (z1, z2) that stores ζ coordinates of bounce points.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of |B|.
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
        result : jnp.ndarray
            Shape (num rho, num pitch, num well).
            Last axis enumerates the bounce integrals for a given field line,
            flux surface, and pitch value.

        """
        if points is None:
            points = self.points(pitch_inv)

        # We move num pitch axis to front so that the num rho axis broadcasts
        # with the spectral coefficients (whose first axis is also num rho),
        # assuming this axis exists.
        z1, z2 = map(_swap_pl, points)
        pitch_inv = atleast_nd(self._c["T(z)"].cheb.ndim - 1, pitch_inv).T

        result = self._integrate(
            self._x if quad is None else quad[0],
            self._w if quad is None else quad[1],
            integrand,
            pitch_inv,
            setdefault(f, []),
            z1,
            z2,
            check,
            plot,
        )
        if weight is not None:
            errorif(
                isinstance(self._c["B(z)"], PiecewiseChebyshevSeries),
                NotImplementedError,
                msg="Set spline to true until implemented.",
            )
            result *= interp_fft_to_argmin(
                self._NFP,
                self._c["T(z)"],
                weight,
                (z1, z2),
                self._c["knots"],
                self._c["B(z)"],
                polyder_vec(self._c["B(z)"]),
            )
        return _swap_pl(result)

    def _integrate(self, x, w, integrand, pitch_inv, f, z1, z2, check, plot):
        """Bounce integrate ∫ f(λ, ℓ) dℓ.

        Parameters
        ----------
        pitch_inv : jnp.ndarray
            Shape (num pitch, ) or (num pitch, num rho).
        f : list[jnp.ndarray]
            Shape (M, N) or (num rho, M, N).
        z1, z2 : jnp.ndarray
            Shape (num pitch, num well) or (num pitch, num rho, num well).

        Returns
        -------
        result : jnp.ndarray
            Shape (num pitch, num rho, num well).

        """
        if not isinstance(f, (list, tuple)):
            f = [f]
        shape = [*z1.shape, x.size]  # num pitch, num rho, num well, num quad
        # ζ ∈ ℝ and θ ∈ ℝ coordinates of quadrature points
        zeta = flatten_matrix(
            bijection_from_disc(x, z1[..., jnp.newaxis], z2[..., jnp.newaxis])
        )
        theta = self._c["T(z)"].eval1d(zeta)

        # Compute |B| from Fourier series instead of spline approximation
        # because integrals are sensitive to |B|. Using the ``polish_points``
        # method should resolve any issues. For now, we integrate with √|1−λB|
        # as justified in doi.org/10.1063/5.0160282.
        B = irfft2_non_uniform(
            theta,
            zeta,
            self._c["|B|"][..., jnp.newaxis, :, :],
            self._M,
            self._N,
            domain1=(0, 2 * jnp.pi / self._NFP),
            axes=(-1, -2),
        )
        B_sup_z = irfft2_non_uniform(
            theta,
            zeta,
            self._c["B^zeta"][..., jnp.newaxis, :, :],
            self._M,
            self._N,
            domain1=(0, 2 * jnp.pi / self._NFP),
            axes=(-1, -2),
        )
        f = [
            interp_rfft2(
                theta,
                zeta,
                f_i[..., jnp.newaxis, :, :],
                domain1=(0, 2 * jnp.pi / self._NFP),
                axes=(-1, -2),
            )
            for f_i in f
        ]
        result = (
            integrand(*f, B=B, pitch=1 / pitch_inv[..., jnp.newaxis], zeta=zeta)
            * B
            / B_sup_z
        ).reshape(shape).dot(w) * grad_bijection_from_disc(z1, z2)

        if check:
            shape[-3], shape[0] = shape[0], shape[-3]
            _check_interp(
                # shape is num alpha = 1, num rho, num pitch, num well, num quad
                (1, *shape),
                *map(_swap_pl, (zeta, B_sup_z, B, result)),
                list(map(_swap_pl, f)),
                plot,
            )

        return result

    def compute_fieldline_length(self, quad=None):
        """Compute the proper length of the field line ∫ dℓ / |B|.

        Parameters
        ----------
        quad : tuple[jnp.ndarray]
            Quadrature points xₖ and weights wₖ for the approximate evaluation
            of the integral ∫₋₁¹ f(x) dx ≈ ∑ₖ wₖ f(xₖ).
            Resolution equal to half the Chebyshev resolution of |B| works well.

        Returns
        -------
        length : jnp.ndarray
            Shape (num rho, ).

        """
        # Integrating an analytic oscillatory map so a high order quadrature is ideal.
        # Difficult to pick the right frequency for Filon quadrature in general, which
        # would work best, especially at high NFP. Gauss-Legendre is superior to
        # Clenshaw-Curtis for smooth oscillatory maps.
        deg = (
            self._c["B(z)"].Y // 2
            if isinstance(self._c["B(z)"], PiecewiseChebyshevSeries)
            else self._c["T(z)"].Y
        )
        x, w = leggauss(deg) if quad is None else quad

        shape = (
            *self._c["T(z)"].cheb.shape[:-2],  # num rho
            self._c["T(z)"].X * w.size,  # num transit * num quad points
        )
        # θ at quadrature points
        theta = idct_non_uniform(
            x, self._c["T(z)"].cheb[..., jnp.newaxis, :], self._c["T(z)"].Y
        ).reshape(shape)
        zeta = jnp.broadcast_to(
            bijection_from_disc(x, 0, 2 * jnp.pi), (self._c["T(z)"].X, w.size)
        ).ravel()

        B_sup_z = irfft2_non_uniform(
            theta,
            zeta,
            self._c["B^zeta"][..., jnp.newaxis, :, :],
            self._M,
            self._N,
            domain1=(0, 2 * jnp.pi / self._NFP),
            axes=(-1, -2),
        ).reshape(*shape[:-1], self._c["T(z)"].X, w.size)

        # Gradient of change of variable from [−1, 1] → [0, 2π] is π.
        return (1 / B_sup_z).dot(w).sum(axis=-1) * jnp.pi

    def plot(self, l, pitch_inv=None, **kwargs):
        """Plot the field line and bounce points of the given pitch angles.

        Parameters
        ----------
        l : int
            Index into the nodes of the grid supplied to make this object.
            ``rho=grid.compress(grid.nodes[:,0])[l]``.
        pitch_inv : jnp.ndarray
            Shape (num pitch, ).
            Optional, 1/λ values whose corresponding bounce points on the field line
            specified by Clebsch coordinate ρ(l) will be plotted.
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
        kwargs.setdefault("hlabel", r"$\alpha = $" + str(self._alpha) + r", $\zeta$")
        kwargs = _set_default_plot_kwargs(kwargs)

        B = self._c["B(z)"]
        if isinstance(B, PiecewiseChebyshevSeries):
            if B.cheb.ndim > 2:
                B = PiecewiseChebyshevSeries(B.cheb[l], B.domain)
            if pitch_inv is not None:
                z1, z2 = B.intersect1d(pitch_inv)
                kwargs["z1"] = z1
                kwargs["z2"] = z2
                kwargs["k"] = pitch_inv
            fig, ax = B.plot1d(B.cheb, **kwargs)
        else:
            if B.ndim == 3:
                B = B[l]
            if pitch_inv is not None:
                z1, z2 = bounce_points(pitch_inv, self._c["knots"], B, polyder_vec(B))
                kwargs["z1"] = z1
                kwargs["z2"] = z2
                kwargs["k"] = pitch_inv
            fig, ax = plot_ppoly(PPoly(B.T, self._c["knots"]), **kwargs)
        return fig, ax

    def plot_theta(self, l, **kwargs):
        """Plot θ(α, ζ) on the specified flux surface.

        Parameters
        ----------
        l : int
            Index into the nodes of the grid supplied to make this object.
            ``rho=grid.compress(grid.nodes[:,0])[l]``.
        kwargs
            Keyword arguments into
            ``desc/integrals/basis.py::PiecewiseChebyshevSeries.plot1d``.

        Returns
        -------
        fig, ax
            Matplotlib (fig, ax) tuple.

        """
        T = self._c["T(z)"]
        if T.cheb.ndim > 2:
            T = PiecewiseChebyshevSeries(T.cheb[l], T.domain)
        kwargs.setdefault(
            "title",
            r"DESC poloidal angle $\theta($"
            + r"$\alpha=$"
            + str(self._alpha)
            + r"$, \zeta)$",
        )
        kwargs.setdefault("hlabel", r"$\alpha = $" + str(self._alpha) + r", $\zeta$")
        kwargs.setdefault("vlabel", r"$\theta$")
        fig, ax = T.plot1d(T.cheb, **_set_default_plot_kwargs(kwargs))
        return fig, ax


class Bounce1D(Bounce):
    """Computes bounce integrals using one-dimensional local spline methods.

    The bounce integral is defined as ∫ f(λ, ℓ) dℓ where

    * dℓ parameterizes the distance along the field line in meters.
    * f(λ, ℓ) is the quantity to integrate along the field line.
    * The boundaries of the integral are bounce points ℓ₁, ℓ₂ s.t. λ|B|(ℓᵢ) = 1.
    * λ is a constant defining the integral proportional to the magnetic moment
      over energy.
    * |B| is the norm of the magnetic field.

    For a particle with fixed λ, bounce points are defined to be the location on the
    field line such that the particle's velocity parallel to the magnetic field is zero.
    The bounce integral is defined up to a sign. We choose the sign that corresponds to
    the particle's guiding center trajectory traveling in the direction of increasing
    field-line-following coordinate ζ.

    Notes
    -----
    For applications which reduce to computing a nonlinear function of distance
    along field lines between bounce points, it is required to identify these
    points with field-line-following coordinates. (In the special case of a linear
    function summing integrals between bounce points over a flux surface, arbitrary
    coordinate systems may be used as that task reduces to a surface integral,
    which is invariant to the order of summation).

    The DESC coordinate system is related to field-line-following coordinate
    systems by a relation whose solution is best found with Newton iteration
    since this solution is unique. Newton iteration is not a globally
    convergent algorithm to find the real roots of r : ζ ↦ |B|(ζ) − 1/λ where
    ζ is a field-line-following coordinate. For this, function approximation
    of |B| is necessary.

    The function approximation in ``Bounce1D`` is ignorant that the objects to
    approximate are defined on a bounded subset of ℝ². Instead, the domain is
    projected to ℝ, where information sampled about the function at infinity
    cannot support reconstruction of the function near the origin. As the
    functions of interest do not vanish at infinity, pseudo-spectral techniques
    are not used. Instead, function approximation is done with local splines.
    This is useful if one can efficiently obtain data along field lines and the
    number of toroidal transits to follow a field line is not large.

    After computing the bounce points, the supplied quadrature is performed.
    By default, this is a Gauss quadrature after removing the singularity.
    Local splines interpolate smooth functions in the integrand to the quadrature
    nodes. Quadrature is chosen over Runge-Kutta methods of the form
        ∂Fᵢ/∂ζ = f(λ,ζ,{Gⱼ}) subject to Fᵢ(ζ₁) = 0
    A fourth order Runge-Kutta method is equivalent to a quadrature
    with Simpson's rule. Our quadratures resolve these integrals more
    efficiently, and the fixed nature of quadrature performs better on GPUs.

    See Also
    --------
    Bounce2D
        Uses two-dimensional pseudo-spectral techniques for the same task.

    Examples
    --------
    See ``tests/test_integrals.py::TestBounce::test_bounce1d_checks``.

    Attributes
    ----------
    required_names : list
        Names in ``data_index`` required to compute bounce integrals.
    B : jnp.ndarray
        Shape (num alpha, num rho, N - 1, B.shape[-1]).
        Polynomial coefficients of the spline of |B| in local power basis.
        Last axis enumerates the coefficients of power series. For a polynomial
        given by ∑ᵢⁿ cᵢ xⁱ, coefficient cᵢ is stored at ``B[...,n-i]``.
        Third axis enumerates the polynomials that compose a particular spline.
        Second axis enumerates flux surfaces.
        First axis enumerates field lines of a particular flux surface.

    """

    required_names = ["B^zeta", "B^zeta_z|r,a", "|B|", "|B|_z|r,a"]

    def __init__(
        self,
        grid,
        data,
        quad=leggauss(32),
        automorphism=(automorphism_sin, grad_automorphism_sin),
        *,
        Bref=1.0,
        Lref=1.0,
        is_reshaped=False,
        check=False,
    ):
        """Returns an object to compute bounce integrals.

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
            For weak singular integrals, use ``chebgauss2`` from
            ``desc.integrals.quad_utils``.
            For strong singular integrals, use ``leggauss``.
        automorphism : tuple[Callable] or None
            The first callable should be an automorphism of the real interval [-1, 1].
            The second callable should be the derivative of the first. This map defines
            a change of variable for the bounce integral. The choice made for the
            automorphism will affect the performance of the quadrature method.
            For weak singular integrals, use ``None``.
            For strong singular integrals, use
            ``(automorphism_sin,grad_automorphism_sin)`` from
            ``desc.integrals.quad_utils``.
        Bref : float
            Optional. Reference magnetic field strength for normalization.
        Lref : float
            Optional. Reference length scale for normalization.
        is_reshaped : bool
            Whether the arrays in ``data`` are already reshaped to the expected form of
            shape (..., num zeta) or (..., num rho, num zeta) or
            (num alpha, num rho, num zeta). This option can be used to iteratively
            compute bounce integrals one field line or one flux surface at a time,
            respectively, reducing memory usage. To do so, set to true and provide
            only those axes of the reshaped data. Default is false.
        check : bool
            Flag for debugging. Must be false for JAX transformations.

        """
        data = {
            # Strictly increasing zeta knots enforces dζ > 0.
            # To retain dℓ = (|B|/B^ζ) dζ > 0 after fixing dζ > 0, we require
            # B^ζ = B⋅∇ζ > 0. This is equivalent to changing the sign of ∇ζ
            # or (∂ℓ/∂ζ)|ρ,a. Recall dζ = ∇ζ⋅dR ⇔ 1 = ∇ζ⋅(e_ζ|ρ,a).
            "B^zeta": jnp.abs(data["B^zeta"]) * Lref / Bref,
            "B^zeta_z|r,a": data["B^zeta_z|r,a"]
            * jnp.sign(data["B^zeta"])
            * Lref
            / Bref,
            "|B|": data["|B|"] / Bref,
            "|B|_z|r,a": data["|B|_z|r,a"] / Bref,  # This is already the correct sign.
        }
        self._data = (
            data
            if is_reshaped
            else dict(zip(data.keys(), Bounce1D.reshape_data(grid, *data.values())))
        )
        self._x, self._w = get_quadrature(quad, automorphism)

        # Compute local splines.
        self.zeta = grid.compress(grid.nodes[:, 2], surface_label="zeta")
        self.B = jnp.moveaxis(
            CubicHermiteSpline(
                x=self.zeta,
                y=self._data["|B|"],
                dydx=self._data["|B|_z|r,a"],
                axis=-1,
                check=check,
            ).c,
            source=(0, 1),
            destination=(-1, -2),
        )
        self.dB_dz = polyder_vec(self.B)
        # Note it is simple to do FFT across field line axis, and spline
        # Fourier coefficients across ζ to obtain Fourier-CubicSpline of functions.
        # The point of Bounce2D is to do such a 2D interpolation but also do so
        # without rebuilding DESC transforms each time an objective is computed.

        # Add axis here instead of in ``_bounce_quadrature``.
        for name in self._data:
            self._data[name] = self._data[name][..., jnp.newaxis, :]

    @staticmethod
    def reshape_data(grid, *arys):
        """Reshape arrays for acceptable input to ``integrate``.

        Parameters
        ----------
        grid : Grid
            Tensor-product grid in (ρ, α, ζ) Clebsch coordinates.
        arys : jnp.ndarray
            Data evaluated on grid.

        Returns
        -------
        f : jnp.ndarray
            Shape (num alpha, num rho, num zeta).
            Reshaped data which may be given to ``integrate``.

        """
        f = [grid.meshgrid_reshape(d, "arz") for d in arys]
        return f if len(f) > 1 else f[0]

    def points(self, pitch_inv, *, num_well=None):
        """Compute bounce points.

        Parameters
        ----------
        pitch_inv : jnp.ndarray
            Shape (num alpha, num rho, num pitch).
            1/λ values to compute the bounce points at each field line. 1/λ(α,ρ) is
            specified by ``pitch_inv[α,ρ]`` where in the latter the labels
            are interpreted as the indices that correspond to that field line.
        num_well : int or None
            Specify to return the first ``num_well`` pairs of bounce points for each
            pitch along each field line. This is useful if ``num_well`` tightly
            bounds the actual number. As a reference, there are typically 20 wells
            per toroidal transit for a given pitch. You can check this by plotting
            the field lines with the ``check_points`` method.

            If not specified, then all bounce points are returned. If there were fewer
            wells detected along a field line than the size of the last axis of the
            returned arrays, then that axis is padded with zero.

        Returns
        -------
        z1, z2 : tuple[jnp.ndarray]
            Shape (num alpha, num rho, num pitch, num well).
            Tuple of length two (z1, z2) that stores ζ coordinates of bounce points.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of |B|.

            If there were less than ``num_well`` wells detected along a field line,
            then the last axis, which enumerates bounce points for a particular field
            line and pitch, is padded with zero.

        """
        return bounce_points(pitch_inv, self.zeta, self.B, self.dB_dz, num_well)

    def check_points(self, points, pitch_inv, *, plot=True, **kwargs):
        """Check that bounce points are computed correctly.

        Parameters
        ----------
        points : tuple[jnp.ndarray]
            Shape (num alpha, num rho, num pitch, num well).
            Output of method ``self.points``.
            Tuple of length two (z1, z2) that stores ζ coordinates of bounce points.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of |B|.
        pitch_inv : jnp.ndarray
            Shape (num alpha, num rho, num pitch).
            1/λ values to compute the bounce points at each field line. 1/λ(α,ρ) is
            specified by ``pitch_inv[α,ρ]`` where in the latter the labels
            are interpreted as the indices that correspond to that field line.
        plot : bool
            Whether to plot the field lines and bounce points of the given pitch angles.
        kwargs
            Keyword arguments into ``desc/integrals/bounce_utils.py::plot_ppoly``.

        Returns
        -------
        plots : list
            Matplotlib (fig, ax) tuples for the 1D plot of each field line.

        """
        return _check_bounce_points(
            z1=points[0],
            z2=points[1],
            pitch_inv=pitch_inv,
            knots=self.zeta,
            B=self.B,
            plot=plot,
            **kwargs,
        )

    # TODO: Add option for adaptive quadrature with quadax.
    #  quadax.quadgk with the currently used c.o.v. seems to work nice.
    #  without change of variable one needs to use quadax.quadts.
    def integrate(
        self,
        integrand,
        pitch_inv,
        f=None,
        weight=None,
        points=None,
        *,
        method="cubic",
        batch=True,
        check=False,
        plot=False,
        quad=None,
    ):
        """Bounce integrate ∫ f(λ, ℓ) dℓ.

        Computes the bounce integral ∫ f(λ, ℓ) dℓ for every field line and pitch.

        Parameters
        ----------
        integrand : callable
            The composition operator on the set of functions in ``f`` that maps the
            functions in ``f`` to the integrand f(λ, ℓ) in ∫ f(λ, ℓ) dℓ. It should
            accept the arrays in ``f`` as arguments as well as the additional keyword
            arguments: ``B`` and ``pitch``. A quadrature will be performed to
            approximate the bounce integral of ``integrand(*f,B=B,pitch=pitch)``.
        pitch_inv : jnp.ndarray
            Shape (num alpha, num rho, num pitch).
            1/λ values to compute the bounce integrals. 1/λ(α,ρ) is specified by
            ``pitch_inv[α,ρ]`` where in the latter the labels are interpreted
            as the indices that correspond to that field line.
        f : list[jnp.ndarray] or jnp.ndarray
            Shape (num alpha, num rho, num zeta).
            Real scalar-valued functions evaluated on the ``grid`` supplied to
            construct this object. These functions should be arguments to the callable
            ``integrand``. Use the method ``Bounce1D.reshape_data`` to reshape the data
            into the expected shape.
        weight : jnp.ndarray
            Shape (num alpha, num rho, num zeta).
            If supplied, the bounce integral labeled by well j is weighted such that
            the returned value is w(j) ∫ f(λ, ℓ) dℓ, where w(j) is ``weight``
            interpolated to the deepest point in that magnetic well. Use the method
            ``Bounce1D.reshape_data`` to reshape the data into the expected shape.
        points : tuple[jnp.ndarray]
            Shape (num alpha, num rho, num pitch, num well).
            Optional, output of method ``self.points``.
            Tuple of length two (z1, z2) that stores ζ coordinates of bounce points.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of |B|.
        method : str
            Method of interpolation.
            See https://interpax.readthedocs.io/en/latest/_api/interpax.interp1d.html.
            Default is cubic C1 local spline.
        batch : bool
            Whether to perform computation in a batched manner. Default is true.
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
        result : jnp.ndarray
            Shape (num alpha, num rho, num pitch, num well).
            Last axis enumerates the bounce integrals for a given field line,
            flux surface, and pitch value.

        """
        if points is None:
            points = self.points(pitch_inv)
        result = _bounce_quadrature(
            x=self._x if quad is None else quad[0],
            w=self._w if quad is None else quad[1],
            integrand=integrand,
            points=points,
            pitch_inv=pitch_inv,
            f=setdefault(f, []),
            data=self._data,
            knots=self.zeta,
            method=method,
            batch=batch,
            check=check,
            plot=plot,
        )
        if weight is not None:
            result *= interp_to_argmin(
                weight,
                points,
                self.zeta,
                self.B,
                self.dB_dz,
                method,
            )
        return result

    def plot(self, m, l, pitch_inv=None, **kwargs):
        """Plot the field line and bounce points of the given pitch angles.

        Parameters
        ----------
        m, l : int, int
            Indices into the nodes of the grid supplied to make this object.
            ``alpha,rho=grid.meshgrid_reshape(grid.nodes[:,:2],"arz")[m,l,0]``.
        pitch_inv : jnp.ndarray
            Shape (num pitch, ).
            Optional, 1/λ values whose corresponding bounce points on the field line
            specified by Clebsch coordinate α(m), ρ(l) will be plotted.
        kwargs
            Keyword arguments into ``desc/integrals/bounce_utils.py::plot_ppoly``.

        Returns
        -------
        fig, ax
            Matplotlib (fig, ax) tuple.

        """
        B, dB_dz = self.B, self.dB_dz
        if B.ndim == 4:
            B = B[m]
            dB_dz = dB_dz[m]
        if B.ndim == 3:
            B = B[l]
            dB_dz = dB_dz[l]
        if pitch_inv is not None:
            errorif(
                jnp.ndim(pitch_inv) > 1,
                msg=f"Got pitch_inv.ndim={jnp.ndim(pitch_inv)}, but expected 1.",
            )
            z1, z2 = bounce_points(pitch_inv, self.zeta, B, dB_dz)
            kwargs["z1"] = z1
            kwargs["z2"] = z2
            kwargs["k"] = pitch_inv
        fig, ax = plot_ppoly(PPoly(B.T, self.zeta), **_set_default_plot_kwargs(kwargs))
        return fig, ax
