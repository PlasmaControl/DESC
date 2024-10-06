"""Methods for computing bounce integrals (singular or otherwise)."""

from interpax import CubicHermiteSpline, PPoly
from numpy.fft import rfft2
from orthax.legendre import leggauss

from desc.backend import dct, jnp
from desc.integrals.basis import FourierChebyshevSeries, PiecewiseChebyshevSeries
from desc.integrals.bounce_utils import (
    _bounce_quadrature,
    _check_bounce_points,
    _check_interp,
    _set_default_plot_kwargs,
    bounce_points,
    get_alpha,
    get_pitch_inv_quad,
    interp_to_argmin,
    plot_ppoly,
)
from desc.integrals.interp_utils import (
    cheb_from_dct,
    cheb_pts,
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
from desc.utils import (
    atleast_nd,
    check_posint,
    errorif,
    flatten_matrix,
    setdefault,
    warnif,
)


def _transform_to_desc(grid, f, is_reshaped=False):
    """Transform to DESC spectral domain.

    Parameters
    ----------
    grid : Grid
        Tensor-product grid in (θ, ζ) with uniformly spaced nodes [0, 2π) × [0, 2π/NFP).
        Preferably power of 2 for ``grid.num_theta`` and ``grid.num_zeta``.
    f : jnp.ndarray
        Function evaluated on ``grid``.

    Returns
    -------
    a : jnp.ndarray
        Shape (..., grid.num_theta // 2 + 1, grid.num_zeta)
        Complex coefficients of 2D real FFT of ``f``.

    """
    if not is_reshaped:
        f = grid.meshgrid_reshape(f, "rtz")
    # real fft over poloidal since usually m > n
    return rfft2(f, axes=(-1, -2), norm="forward")


def _transform_to_clebsch(grid, nodes, f, is_reshaped=False):
    """Transform to Clebsch spectral domain.

    Parameters
    ----------
    grid : Grid
        Tensor-product grid in (θ, ζ) with uniformly spaced nodes [0, 2π) × [0, 2π/NFP).
        Preferably power of 2 for ``grid.num_theta`` and ``grid.num_zeta``.
    nodes : jnp.ndarray
        Shape (L, M, N, 2) or (M, N, 2).
        DESC coordinates (θ, ζ) sourced from the Clebsch coordinates
        ``FourierChebyshevSeries.nodes(M,N,domain=(0,2*jnp.pi))``.
    f : jnp.ndarray
        Function evaluated on ``grid``.

    Returns
    -------
    a : FourierChebyshevSeries
        Spectral coefficients of f(α, ζ).

    """
    assert nodes.shape[-1] == 2
    if not is_reshaped:
        f = grid.meshgrid_reshape(f, "rtz")

    M, N = nodes.shape[-3], nodes.shape[-2]
    nodes = nodes.reshape(*nodes.shape[:-3], M * N, 2)
    return FourierChebyshevSeries(
        f=interp_rfft2(
            # Interpolate to nodes in Clebsch space,
            # which is not a tensor product node set in DESC space.
            xq0=nodes[..., 0],
            xq1=nodes[..., 1],
            f=f[..., jnp.newaxis, :, :],
            domain1=(0, 2 * jnp.pi / grid.NFP),
            axes=(-1, -2),
        ).reshape(*nodes.shape[:-2], M, N),
        domain=(0, 2 * jnp.pi),
    )


def _transform_to_clebsch_1d(grid, alpha, theta, B, N_B, is_reshaped=False):
    """Transform to single variable Clebsch spectral domain.

    Notes
    -----
    The field line label α changes discontinuously, so the approximation
    g defined with basis function in (α, ζ) coordinates to some continuous
    function f does not guarantee continuity between cuts of the field line
    until full convergence of g to f.

    Note if g were defined with basis functions in straight field line
    coordinates, then continuity between cuts of the field line, as
    determined by the straight field line coordinates (ϑ, ζ), is
    guaranteed even with incomplete convergence (because the
    parameters (ϑ, ζ) change continuously along the field line).

    Do not interpret this as superior function approximation.
    Indeed, if g is defined with basis functions in (α, ζ) coordinates, then
    g(α=α₀, ζ) will sample the approximation to f(α=α₀, ζ) for the full domain in ζ.
    This holds even with incomplete convergence of g to f.
    However, if g is defined with basis functions in (ϑ, ζ) coordinates, then
    g(ϑ(α=α₀,ζ), ζ) will sample the approximation to f(α=α₀ ± ε, ζ) with ε → 0 as
    g converges to f.

    (Visually, the small discontinuity apparent in g(α, ζ) at cuts of the field
    line will not be visible in g(ϑ, ζ) because when moving along the field line
    with g(ϑ, ζ) one is continuously flowing away from the starting field line,
    (whereas g(α, ζ) has to "decide" at the cut what the next field line is).
    (If full convergence is difficult to achieve, then in the context of surface
    averaging bounce integrals, function approximation in (α, ζ) coordinates
    might be preferable because most of the bounce integrals do not stretch
    across toroidal transits).)

    Now, it appears the Fourier transform of θ may have small oscillatory bumps
    outside reasonable bandwidths. This impedes full convergence of any
    approximation, and in particular the poloidal Fourier series for, θ(α, ζ=ζ₀).
    Maybe this is because θ is not a physical signal as it is determined by the
    optimizer. (Note the Fourier series converges fast for |B|, even in
    non-omnigenous configurations where (∂|B|/∂α)|ρ,ζ is not small, so this is
    indeed some feature with θ).

    Therefore, we explicitly enforce continuity of our approximation of θ between
    cuts to short-circuit the convergence. This works to remove the small
    discontinuity between cuts of the field line because the first cut is on α=0,
    which is a knot of the Fourier series, and the Chebyshev points include a knot
    near endpoints, so θ at the next cut of the field line is known with precision.

    Parameters
    ----------
    grid : Grid
        Tensor-product grid in (θ, ζ) with uniformly spaced nodes [0, 2π) × [0, 2π/NFP).
        Preferably power of 2 for ``grid.num_theta`` and ``grid.num_zeta``.
    alpha : jnp.ndarray
        Shape (L, num_transit) or (num_transit, ).
        Sequence of poloidal coordinates A = (α₀, α₁, …, αₘ₋₁) that specify field line.
    theta : jnp.ndarray
        Shape (L, M, N) or (M, N).
        DESC coordinates θ sourced from the Clebsch coordinates
        ``FourierChebyshevSeries.nodes(M,N,domain=(0,2*jnp.pi))``.
    B : jnp.ndarray
        |B| evaluated on ``grid``.
    N_B : int
        Desired Chebyshev spectral resolution for |B|. Preferably power of 2.

    Returns
    -------
    T, B : tuple[PiecewiseChebyshevSeries]
        Set of 1D Chebyshev spectral coefficients of θ along field line.
        {θ_α : ζ ↦ θ(α, ζ) | α ∈ A} where A = (α₀, α₁, …, αₘ₋₁) is ``alpha``.
        Likewise with |B|.

    """
    if not is_reshaped:
        B = grid.meshgrid_reshape(B, "rtz")

    # Evaluating set of single variable maps is more efficient than evaluating
    # multivariable map, so we project θ to a set of Chebyshev series.
    T = FourierChebyshevSeries(f=theta, domain=(0, 2 * jnp.pi)).compute_cheb(alpha)
    T.stitch()
    theta = T.evaluate(N_B)
    zeta = jnp.broadcast_to(cheb_pts(N_B, domain=T.domain), theta.shape)

    shape = (*alpha.shape[:-1], alpha.shape[-1] * N_B)
    B = interp_rfft2(
        theta.reshape(shape),
        zeta.reshape(shape),
        f=B[..., jnp.newaxis, :, :],
        domain1=(0, 2 * jnp.pi / grid.NFP),
        axes=(-1, -2),
    ).reshape(*alpha.shape, N_B)
    # Parameterize |B| by single variable to compute roots.
    B = PiecewiseChebyshevSeries(cheb_from_dct(dct(B, type=2, axis=-1)) / N_B, T.domain)
    # |B| guaranteed to be continuous because it was interpolated from B(θ(α, ζ),ζ).
    return T, B


def _swap_pl(f):
    # Given shape (L, num_pitch, -1) or (num_pitch, L, -1) or (num_pitch, -1)
    # swap L and num_pitch axes.
    assert f.ndim <= 3
    return jnp.swapaxes(f, 0, -2)


# TODO: After GitHub issue #1034 is resolved, we should pass in the previous
#  θ(α, ζ) coordinates as an initial guess for the next coordinate mapping.
#  Perhaps tell the optimizer to perturb the coefficients of the
#  θ(α, ζ) directly? think this is equivalent to perturbing lambda.


class Bounce2D(IOAble):
    """Computes bounce integrals using two-dimensional pseudo-spectral methods.

    The bounce integral is defined as ∫ f(λ, ℓ) dℓ, where
        dℓ parameterizes the distance along the field line in meters,
        f(λ, ℓ) is the quantity to integrate along the field line,
        and the boundaries of the integral are bounce points ℓ₁, ℓ₂ s.t. λ|B|(ℓᵢ) = 1,
        where λ is a constant defining the integral proportional to the magnetic moment
        over energy and |B| is the norm of the magnetic field.

    For a particle with fixed λ, bounce points are defined to be the location on the
    field line such that the particle's velocity parallel to the magnetic field is zero.
    The bounce integral is defined up to a sign. We choose the sign that corresponds to
    the particle's guiding center trajectory traveling in the direction of increasing
    field-line-following coordinate ζ.

    Notes
    -----
    Brief description of algorithm.

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
    permissible if separable into a single valued and multivalued parts.
    In that case, supply the single valued parts, which will be interpolated
    with FFTs, and use the provided coordinates θ,ζ ∈ ℝ to compose G.

    Longer description for developers.

    For applications which reduce to computing a nonlinear function of distance
    along field lines between bounce points, it is required to identify these
    points with field-line-following coordinates. (In the special case of a linear
    function summing integrals between bounce points over a flux surface, arbitrary
    coordinate systems may be used as that task reduces to a surface integral,
    which is invariant to the order of summation).

    The DESC coordinate system is related to field-line-following coordinate
    systems by a relation whose solution is best found with Newton iteration
    since this solution is unique.  Newton iteration is not a globally
    convergent algorithm to find the real roots of r : ζ ↦ |B|(ζ) − 1/λ where
    ζ is a field-line-following coordinate. For this, function approximation
    of |B| is necessary.

    Therefore, to compute bounce points {(ζ₁, ζ₂)}, we approximate |B| by a
    series expansion of basis functions parameterized by a single variable ζ,
    restricting the class of basis functions to low order (e.g. N = 2ᵏ where
    k is small) algebraic or trigonometric polynomial with integer frequencies.
    These are the two classes useful for function approximation and for which
    there exists globally convergent root-finding algorithms. We require low
    order because the computation expenses grow with the number of potential
    roots, and the theorem of algebra states that number is N (2N) for algebraic
    (trigonometric) polynomials of degree N.

    The frequency transform of a map under the chosen basis must be concentrated
    at low frequencies for the series to converge fast. For periodic
    (non-periodic) maps, the best basis is a Fourier (Chebyshev) series. Both
    converge exponentially, but the larger region of convergence in the complex
    plane of Fourier series make it preferable in practice to choose coordinate
    systems such that the function to approximate is periodic. The Chebyshev
    polynomials are preferred to other orthogonal polynomial series since
    fast discrete polynomial transforms (DPT) are implemented via fast transform
    to Chebyshev then DCT. Although nothing prohibits a direct DPT, we want to
    rely on existing libraries. There are other reasons to prefer Chebyshev series
    not discussed here. Therefore, a Fourier-Chebyshev series is chosen to
    interpolate θ(α,ζ), and a piecewise Chebyshev series interpolates |B|(ζ).

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

    Recall that periodicity enables faster convergence, motivating the desire
    to instead interpolate |B|(ϑ, ϕ) with a double Fourier series and applying
    bisection methods to find bounce points with mesh size inversely
    proportional to the max frequency along the field line: M ι + N. ``Bounce2D``
    does not use that approach as that root-finding scheme is inferior.
    The reason θ is not interpolated with a double Fourier series θ(ϑ, ζ) is
    because quadrature points along |B|(α=α₀, ζ) can be identified by a single
    variable; evaluating the multivariable map θ(ϑ(α, ζ), ζ) is expensive
    compared to evaluating the single variable map θ(α=α₀, ζ).
    Another option is to use a filtered Fourier series,
    doi.org/10.1016/j.aml.2006.10.001.

    After computing the bounce points, the supplied quadrature is performed.
    By default, this is a Gauss quadrature after removing the singularity.
    Fast fourier transforms interpolate smooth functions in the integrand to the
    quadrature nodes. Quadrature is chosen over Runge-Kutta methods of the form
        ∂Fᵢ/∂ζ = f(λ,ζ,{Gⱼ}) subject to Fᵢ(ζ₁) = 0
    because a fourth order Runge-Kutta method is equivalent to a quadrature
    with Simpson's rule. Our quadratures resolve these integrals more
    efficiently, and the fixed nature of quadrature performs better on GPUs.

    Fast transforms are used where possible. Fast multipoint methods are not
    implemented. For non-uniform interpolation, MMTs are used. It should be
    worthwhile to use the inverse non-uniform fast transforms, so long as the
    quadrature packs nodes at reasonable density. Fast multipoint methods are
    preferable because they are exact, but that requires more development work.

    Additional notes on multivalued coordinates.
    The definition of α in B = ∇ρ × ∇α on an irrational magnetic surface
    implies the angle θ(α, ζ) is multivalued at a physical location.
    In particular, following an irrational field, the single-valued θ grows
    to ∞ (always non-monotonically) as ζ → ∞. Therefore, it is impossible to
    approximate this map using single-valued basis functions defined on a
    bounded subset of ℝ² (recall continuous functions on compact sets attain
    their maximum).

    Still, it suffices to interpolate θ over one branch cut.
    DESC chooses the branch cut defined by (α, ζ) ∈ [0, 2π]² and we must
    maintain that convention with our basis functions. On such a branch
    cut, the bound θ ∈ [0, 4π] holds.

    Likewise, α is multivalued. As the field line is followed, the label
    jumps to α ∉ [0, 2π] after completing some toroidal transit. Therefore,
    the map θ(α, ζ) must be periodic in α with period 2π. At every point
    ζₚ ∈ [2π k, 2π ℓ] where k, ℓ ∈ ℤ where the field line completes a
    poloidal transit there is guaranteed to exist a discrete jump
    discontinuity in θ at ζ = 2π ℓ(p), starting the toroidal transit.
    Recall a jump discontinuity appears as an infinitely sharp cut;
    nearby the cut, the function must be blind to the cut.

    To recover the single-valued θ(α, ζ) from the function approximation
    over one branch cut, at every ζ = 2π ℓ we can add either 0 or 2π or
    4π to the next cut of θ.

    See Also
    --------
    Bounce1D
        Uses one-dimensional local spline methods for the same task.

        Below are some advantages of ``Bounce2D`` over ``Bounce1D``.
        The coordinates on which the root-finding must be done to map from DESC
        to Clebsch coords is fixed to ``L*M*N``, independent of the number of
        toroidal transits; generating the same data with DESC for input to
        ``Bounce1D`` requires ``L*M*N*num_transit``. Furthermore, pseudo-spectral
        interpolation of smooth functions such as |B| on each flux surface is more
        efficient. This reduces the number of bounce integrals to be done by an
        order of magnitude. Also, we have noticed C1 cubic spline interpolation
        is inefficient to reconstruct smooth local maxima of |B|, which might be
        important for (strongly) singular bounce integrals whose estimation
        depends on ∂|B|/∂ζ there.

    Attributes
    ----------
    required_names : list
        Names in ``data_index`` required to compute bounce integrals.

    """

    required_names = ["B^zeta", "|B|", "iota"]
    get_pitch_inv_quad = staticmethod(get_pitch_inv_quad)

    def __init__(
        self,
        grid,
        data,
        theta,
        N_B=None,
        num_transit=16,
        # TODO: Allow multiple starting labels for near-rational surfaces.
        #  think can just concatenate along second to last axis of cheb.
        #  Do this in different PR.
        alpha=0.0,
        quad=leggauss(32),
        automorphism=(automorphism_sin, grad_automorphism_sin),
        *,
        Bref=1.0,
        Lref=1.0,
        is_reshaped=False,
        check=False,
        **kwargs,
    ):
        """Returns an object to compute bounce integrals.

        Notes
        -----
        Performance may improve significantly if the spectral
        resolutions ``m``, ``n``, ``M``, ``N``, and ``N_B`` are powers of two.
        It is possible for the optimal ``M``, ``N``, and ``N_B`` to decrease
        when ``m`` and ``n`` increase, though ``M`` is usually still large.

        Parameters
        ----------
        grid : Grid
            Tensor-product grid in (ρ, θ, ζ) with uniformly spaced nodes
            [0, 2π) × [0, 2π/NFP). Note that below shape notation defines
            L = ``grid.num_rho``, m = ``grid.num_theta``, and n = ``grid.num_zeta``.
        data : dict[str, jnp.ndarray]
            Data evaluated on ``grid``.
            Must include names in ``Bounce2D.required_names``.
        theta : jnp.ndarray
            Shape (L, M, N).
            DESC coordinates θ sourced from the Clebsch coordinates
            ``FourierChebyshevSeries.nodes(M,N,L,domain=(0,2*jnp.pi))``.
        N_B : int
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
            shape (..., m, n) or (L, m, n). This option can be used to iteratively
            compute bounce integrals one flux surface at a time, reducing memory usage
            To do so, set to true and provide only those axes of the reshaped data.
            Default is false.
        check : bool
            Flag for debugging. Must be false for JAX transformations.

        """
        errorif(grid.sym, NotImplementedError, msg="Need grid that works with FFTs.")
        # Strictly increasing zeta knots enforces dζ > 0.
        # To retain dℓ = (|B|/B^ζ) dζ > 0 after fixing dζ > 0, we require
        # B^ζ = B⋅∇ζ > 0. This is equivalent to changing the sign of ∇ζ or [∂ℓ/∂ζ]|ρ,a.
        # Recall dζ = ∇ζ⋅dR, implying 1 = ∇ζ⋅(e_ζ|ρ,a). Hence, a sign change in ∇ζ
        # requires the same sign change in e_ζ|ρ,a to retain the metric identity.
        warnif(
            check and kwargs.pop("warn", True) and jnp.any(data["B^zeta"] <= 0),
            msg="(∂ℓ/∂ζ)|ρ,a > 0 is required. Enforcing positive B^ζ.",
        )
        N_B = setdefault(N_B, theta.shape[-1] * 2)
        self._alpha = alpha
        self._m = grid.num_theta
        self._n = grid.num_zeta
        self._NFP = grid.NFP
        self._x, self._w = get_quadrature(quad, automorphism)

        # peel off field lines
        iota = data["iota"].ravel()
        alpha = get_alpha(
            alpha,
            iota=(
                grid.compress(iota)
                if iota.size == grid.num_nodes
                # assume passed in reshaped data over flux surface
                else jnp.array(iota[0])
            ),
            num_transit=num_transit,
            period=2 * jnp.pi,
        )
        # Compute spectral coefficients.
        self._T, self._B = _transform_to_clebsch_1d(
            grid, alpha, theta, data["|B|"] / Bref, N_B, is_reshaped
        )
        self._B_sup_z = _transform_to_desc(
            grid, jnp.abs(data["B^zeta"]) * Lref / Bref, is_reshaped
        )
        assert self._T.M == self._B.M == num_transit
        assert self._T.N == theta.shape[-1]
        assert self._B.N == N_B

    @staticmethod
    def compute_theta(eq, M=16, N=32, rho=1.0, clebsch=None, **kwargs):
        """Return DESC coordinates θ of Fourier Chebyshev basis nodes.

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium to use defining the coordinate mapping.
        M : int
            Grid resolution in poloidal direction for Clebsch coordinate grid.
            Preferably power of 2.
        N : int
            Grid resolution in toroidal direction for Clebsch coordinate grid.
            Preferably power of 2.
        rho : float or jnp.ndarray
            Flux surfaces labels in [0, 1] on which to compute.
        clebsch : jnp.ndarray
            Optional, Clebsch coordinate tensor-product grid (ρ, α, ζ).
            ``FourierChebyshevSeries.nodes(M,N,L,domain=(0,2*jnp.pi))``.
            If given, ``rho`` is ignored.
        kwargs
            Additional parameters to supply to the coordinate mapping function.
            See ``desc.equilibrium.Equilibrium.map_coordinates``.

        Returns
        -------
        theta : jnp.ndarray
            Shape (L, M, N).
            DESC coordinates θ sourced from the Clebsch coordinates
            ``FourierChebyshevSeries.nodes(M,N,L,domain=(0,2*jnp.pi))``.

        """
        if clebsch is None:
            clebsch = FourierChebyshevSeries.nodes(
                check_posint(M),
                check_posint(N),
                rho,
                domain=(0, 2 * jnp.pi),
            )
        return eq.map_coordinates(
            coords=clebsch,
            inbasis=("rho", "alpha", "zeta"),
            period=(jnp.inf, jnp.inf, jnp.inf),
            **kwargs,
        ).reshape(-1, M, N, 3)[..., 1]

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
            Shape (L, M, N).
            Reshaped data which may be given to ``integrate``.

        """
        f = [grid.meshgrid_reshape(d, "rtz") for d in arys]
        return f if len(f) > 1 else f[0]

    def points(self, pitch_inv, *, num_well=None):
        """Compute bounce points.

        Parameters
        ----------
        pitch_inv : jnp.ndarray
            Shape (L, num_pitch).
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
            Shape (L, num_pitch, num_well).
            Tuple of length two (z1, z2) that stores ζ coordinates of bounce points.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of |B|.

            If there were less than ``num_well`` wells detected along a field line,
            then the last axis, which enumerates bounce points for a particular field
            line and pitch, is padded with zero.

        """
        pitch_inv = atleast_nd(self._B.cheb.ndim - 1, pitch_inv).T
        # Expects pitch_inv shape (num_pitch, L) if B.cheb.shape[0] is L.
        z1, z2 = map(_swap_pl, self._B.intersect1d(pitch_inv, num_intersect=num_well))
        return z1, z2

    def check_points(self, points, pitch_inv, *, plot=True, **kwargs):
        """Check that bounce points are computed correctly.

        Parameters
        ----------
        points : tuple[jnp.ndarray]
            Shape (L, num_pitch, num_well).
            Output of method ``self.points``.
            Tuple of length two (z1, z2) that stores ζ coordinates of bounce points.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of |B|.
        pitch_inv : jnp.ndarray
            Shape (L, num_pitch).
            1/λ values to compute the bounce integrals. 1/λ(ρ) is specified by
            ``pitch_inv[ρ]`` where in the latter the labels are interpreted
            as the indices that correspond to that field line.
        plot : bool
            Whether to plot the field lines and bounce points of the given pitch angles.
        kwargs : dict
            Keyword arguments into
            ``desc/integrals/basis.py::PiecewiseChebyshevSeries.plot1d``.

        Returns
        -------
        plots : list
            Matplotlib (fig, ax) tuples for the 1D plot of each field line.

        """
        kwargs.setdefault("hlabel", r"$\alpha = $" + str(self._alpha) + r", $\zeta$")
        return self._B.check_intersect1d(
            z1=_swap_pl(points[0]),
            z2=_swap_pl(points[1]),
            k=atleast_nd(self._B.cheb.ndim - 1, pitch_inv).T,
            plot=plot,
            **_set_default_plot_kwargs(kwargs),
        )

    def integrate(
        self,
        integrand,
        pitch_inv,
        f=None,
        f_vec=None,
        weight=None,
        points=None,
        *,
        check=False,
        plot=False,
    ):
        """Bounce integrate ∫ f(λ, ℓ) dℓ.

        Computes the bounce integral ∫ f(λ, ℓ) dℓ for every field line and pitch.

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
            Shape (L, num_pitch).
            1/λ values to compute the bounce integrals. 1/λ(ρ) is specified by
            ``pitch_inv[ρ]`` where in the latter the labels are interpreted
            as the indices that correspond to that field line.
        f : list[jnp.ndarray] or jnp.ndarray
            Shape (L, m, n).
            Real scalar-valued (2π × 2π/NFP) periodic in (θ, ζ) functions evaluated
            on the ``grid`` supplied to construct this object. These functions
            should be arguments to the callable ``integrand``. Use the method
            ``Bounce2D.reshape_data`` to reshape the data into the expected shape.
        f_vec : list[jnp.ndarray] or jnp.ndarray
            Shape (L, m, n, 3).
            Real vector-valued (2π × 2π/NFP) periodic in (θ, ζ) functions evaluated
            on the ``grid`` supplied to construct this object. These functions
            should be arguments to the callable ``integrand``. Use the method
            ``Bounce2D.reshape_data`` to reshape the data into the expected shape.
        weight : jnp.ndarray
            Shape (L, m, n).
            If supplied, the bounce integral labeled by well j is weighted such that
            the returned value is w(j) ∫ f(λ, ℓ) dℓ, where w(j) is ``weight``
            interpolated to the deepest point in that magnetic well. Use the method
            ``Bounce2D.reshape_data`` to reshape the data into the expected shape.
        points : tuple[jnp.ndarray]
            Shape (L, num_pitch, num_well).
            Optional, output of method ``self.points``.
            Tuple of length two (z1, z2) that stores ζ coordinates of bounce points.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of |B|.
        check : bool
            Flag for debugging. Must be false for JAX transformations.
        plot : bool
            Whether to plot the quantities in the integrand interpolated to the
            quadrature points of each integral. Ignored if ``check`` is false.

        Returns
        -------
        result : jnp.ndarray
            Shape (M, L, num_pitch, num_well).
            Last axis enumerates the bounce integrals for a given field line,
            flux surface, and pitch value.

        """
        errorif(weight is not None, NotImplementedError, msg="See Bounce1D")
        f = setdefault(f, [])
        f_vec = setdefault(f_vec, [])
        if not isinstance(f, (list, tuple)):
            f = [f]
        if not isinstance(f_vec, (list, tuple)):
            f_vec = [f_vec]

        points = map(_swap_pl, points)
        pitch_inv = atleast_nd(self._B.cheb.ndim - 1, pitch_inv).T
        result = self._integrate(integrand, points, pitch_inv, f, f_vec, check, plot)
        return result

    def _integrate(self, integrand, points, pitch_inv, f, f_vec, check, plot):
        """Bounce integrate ∫ f(λ, ℓ) dℓ.

        Parameters
        ----------
        points : jnp.ndarray
            Shape (num_pitch, num_well) or (num_pitch, L, num_well).
        pitch_inv : jnp.ndarray
            Shape (num_pitch, ) or (num_pitch, L).
        f : list[jnp.ndarray]
            Shape (m, n) or (L, m, n).
        f_vec : list[jnp.ndarray]
            Shape (m, n, 3) or (L, m, n, 3).

        """
        z1, z2 = points
        shape = [*z1.shape, self._x.size]

        # These are the ζ coordinates of the quadrature points.
        # Shape is (num_pitch, L, number of points to interpolate onto).
        zeta = flatten_matrix(
            bijection_from_disc(self._x, z1[..., jnp.newaxis], z2[..., jnp.newaxis])
        )
        # Note self._T expects shape (num_pitch, L) if T.cheb.shape[0] is L.
        # These are the θ coordinates of the quadrature points.
        theta = self._T.eval1d(zeta)

        B_sup_z = irfft2_non_uniform(
            theta,
            zeta,
            a=self._B_sup_z[..., jnp.newaxis, :, :],
            M=self._n,
            N=self._m,
            domain1=(0, 2 * jnp.pi / self._NFP),
            axes=(-1, -2),
        )
        B = self._B.eval1d(zeta)
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
        f_vec = [
            interp_rfft2(
                theta[..., jnp.newaxis],
                zeta[..., jnp.newaxis],
                f_i[..., jnp.newaxis, :, :, :],
                domain1=(0, 2 * jnp.pi / self._NFP),
                axes=(-2, -3),
            )
            for f_i in f_vec
        ]
        result = _swap_pl(
            (
                integrand(
                    *f,
                    *f_vec,
                    B=B,
                    pitch=1 / pitch_inv[..., jnp.newaxis],
                    zeta=zeta,
                )
                * B
                / B_sup_z
            )
            .reshape(shape)
            .dot(self._w)
            * grad_bijection_from_disc(z1, z2)
        )

        if check:
            shape[-3], shape[0] = shape[0], shape[-3]
            _check_interp(
                # num_alpha is 1, num_rho, num_pitch, num_well, num_quad
                (1, *shape),
                *map(_swap_pl, (zeta, B_sup_z, B)),
                result,
                list(map(_swap_pl, f)),
                plot,
            )
        return result

    def compute_length(self, quad=None):
        """Compute the proper length of the field line ∫ dℓ / |B|.

        Parameters
        ----------
        quad : tuple[jnp.ndarray]
            Quadrature points xₖ and weights wₖ for the approximate evaluation
            of the integral ∫₋₁¹ f(x) dx ≈ ∑ₖ wₖ f(xₖ).
            Should not use more points than half Chebyshev resolution of |B|.

        Returns
        -------
        length : jnp.ndarray
            Shape (L, ).

        """
        # Gauss quadrature captures double frequency of Chebyshev series.
        x, w = leggauss(self._B.N // 2) if quad is None else quad

        # TODO: There exits a fast transform from Chebyshev series to Legendre nodes.
        theta = idct_non_uniform(x, self._T.cheb[..., jnp.newaxis, :], self._T.N)
        zeta = jnp.broadcast_to(bijection_from_disc(x, 0, 2 * jnp.pi), theta.shape)

        shape = (-1, self._T.M * w.size)  # (num_rho, num transit * w.size)
        B_sup_z = irfft2_non_uniform(
            theta.reshape(shape),
            zeta.reshape(shape),
            a=self._B_sup_z[..., jnp.newaxis, :, :],
            M=self._n,
            N=self._m,
            domain1=(0, 2 * jnp.pi / self._NFP),
            axes=(-1, -2),
        ).reshape(-1, self._T.M, w.size)

        # Gradient of change of variable bijection from [−1, 1] → [0, 2π] is π.
        return (1 / B_sup_z).dot(w).sum(axis=-1) * jnp.pi

    def plot(self, l, pitch_inv=None, **kwargs):
        """Plot the field line and bounce points of the given pitch angles.

        Parameters
        ----------
        l : int
            Index into the nodes of the grid supplied to make this object.
            ``rho=grid.compress(grid.nodes[:,0])[l]``.
        pitch_inv : jnp.ndarray
            Shape (num_pitch, ).
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
        B = self._B
        if B.cheb.ndim > 2:
            B = PiecewiseChebyshevSeries(B.cheb[l], B.domain)
        if pitch_inv is not None:
            errorif(
                pitch_inv.ndim > 1,
                msg=f"Got pitch_inv.ndim={pitch_inv.ndim}, but expected 1.",
            )
            z1, z2 = B.intersect1d(pitch_inv)
            kwargs["z1"] = z1
            kwargs["z2"] = z2
            kwargs["k"] = pitch_inv
        kwargs.setdefault("hlabel", r"$\alpha = $" + str(self._alpha) + r", $\zeta$")
        fig, ax = B.plot1d(B.cheb, **_set_default_plot_kwargs(kwargs))
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
        T = self._T
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


class Bounce1D(IOAble):
    """Computes bounce integrals using one-dimensional local spline methods.

    The bounce integral is defined as ∫ f(λ, ℓ) dℓ, where
        dℓ parameterizes the distance along the field line in meters,
        f(λ, ℓ) is the quantity to integrate along the field line,
        and the boundaries of the integral are bounce points ℓ₁, ℓ₂ s.t. λ|B|(ℓᵢ) = 1,
        where λ is a constant defining the integral proportional to the magnetic moment
        over energy and |B| is the norm of the magnetic field.

    For a particle with fixed λ, bounce points are defined to be the location on the
    field line such that the particle's velocity parallel to the magnetic field is zero.
    The bounce integral is defined up to a sign. We choose the sign that corresponds to
    the particle's guiding center trajectory traveling in the direction of increasing
    field-line-following coordinate ζ.

    Notes
    -----
    Brief description of algorithm for developers.

    For applications which reduce to computing a nonlinear function of distance
    along field lines between bounce points, it is required to identify these
    points with field-line-following coordinates. (In the special case of a linear
    function summing integrals between bounce points over a flux surface, arbitrary
    coordinate systems may be used as that task reduces to a surface integral,
    which is invariant to the order of summation).

    The DESC coordinate system is related to field-line-following coordinate
    systems by a relation whose solution is best found with Newton iteration
    since this solution is unique.  Newton iteration is not a globally
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
    because a fourth order Runge-Kutta method is equivalent to a quadrature
    with Simpson's rule. Our quadratures resolve these integrals more
    efficiently, and the fixed nature of quadrature performs better on GPUs.

    See Also
    --------
    Bounce2D
        Uses two-dimensional pseudo-spectral techniques for the same task.

    Examples
    --------
    See ``tests/test_integrals.py::TestBounce1D::test_bounce1d_checks``.

    Attributes
    ----------
    required_names : list
        Names in ``data_index`` required to compute bounce integrals.
    B : jnp.ndarray
        Shape (M, L, N - 1, B.shape[-1]).
        Polynomial coefficients of the spline of |B| in local power basis.
        Last axis enumerates the coefficients of power series. For a polynomial
        given by ∑ᵢⁿ cᵢ xⁱ, coefficient cᵢ is stored at ``B[...,n-i]``.
        Third axis enumerates the polynomials that compose a particular spline.
        Second axis enumerates flux surfaces.
        First axis enumerates field lines of a particular flux surface.

    """

    required_names = ["B^zeta", "B^zeta_z|r,a", "|B|", "|B|_z|r,a"]
    get_pitch_inv_quad = staticmethod(get_pitch_inv_quad)

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
        **kwargs,
    ):
        """Returns an object to compute bounce integrals.

        Parameters
        ----------
        grid : Grid
            Tensor-product grid in (ρ, α, ζ) Clebsch coordinates.
            The ζ coordinates (the unique values prior to taking the tensor-product)
            must be strictly increasing and preferably uniformly spaced. These are used
            as knots to construct splines. A reference knot density is 100 knots per
            toroidal transit. Note that below shape notation defines
            L = ``grid.num_rho``, M = ``grid.num_alpha``, and N = ``grid.num_zeta``.
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
            shape (..., N) or (..., L, N) or (M, L, N). This option can be used to
            iteratively compute bounce integrals one field line or one flux surface
            at a time, respectively, reducing memory usage. To do so, set to true and
            provide only those axes of the reshaped data. Default is false.
        check : bool
            Flag for debugging. Must be false for JAX transformations.

        """
        # Strictly increasing zeta knots enforces dζ > 0.
        # To retain dℓ = (|B|/B^ζ) dζ > 0 after fixing dζ > 0, we require
        # B^ζ = B⋅∇ζ > 0. This is equivalent to changing the sign of ∇ζ or [∂ℓ/∂ζ]|ρ,a.
        # Recall dζ = ∇ζ⋅dR, implying 1 = ∇ζ⋅(e_ζ|ρ,a). Hence, a sign change in ∇ζ
        # requires the same sign change in e_ζ|ρ,a to retain the metric identity.
        warnif(
            check and kwargs.pop("warn", True) and jnp.any(data["B^zeta"] <= 0),
            msg="(∂ℓ/∂ζ)|ρ,a > 0 is required. Enforcing positive B^ζ.",
        )
        data = {
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
        self._zeta = grid.compress(grid.nodes[:, 2], surface_label="zeta")
        self.B = jnp.moveaxis(
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
        self._dB_dz = polyder_vec(self.B)

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
            Shape (M, L, N).
            Reshaped data which may be given to ``integrate``.

        """
        f = [grid.meshgrid_reshape(d, "arz") for d in arys]
        return f if len(f) > 1 else f[0]

    def points(self, pitch_inv, *, num_well=None):
        """Compute bounce points.

        Parameters
        ----------
        pitch_inv : jnp.ndarray
            Shape (M, L, num_pitch).
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
            Shape (M, L, num_pitch, num_well).
            Tuple of length two (z1, z2) that stores ζ coordinates of bounce points.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of |B|.

            If there were less than ``num_well`` wells detected along a field line,
            then the last axis, which enumerates bounce points for a particular field
            line and pitch, is padded with zero.

        """
        return bounce_points(pitch_inv, self._zeta, self.B, self._dB_dz, num_well)

    def check_points(self, points, pitch_inv, *, plot=True, **kwargs):
        """Check that bounce points are computed correctly.

        Parameters
        ----------
        points : tuple[jnp.ndarray]
            Shape (M, L, num_pitch, num_well).
            Output of method ``self.points``.
            Tuple of length two (z1, z2) that stores ζ coordinates of bounce points.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of |B|.
        pitch_inv : jnp.ndarray
            Shape (M, L, num_pitch).
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
            knots=self._zeta,
            B=self.B,
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
        method="cubic",
        batch=True,
        check=False,
        plot=False,
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
            Shape (M, L, num_pitch).
            1/λ values to compute the bounce integrals. 1/λ(α,ρ) is specified by
            ``pitch_inv[α,ρ]`` where in the latter the labels are interpreted
            as the indices that correspond to that field line.
        f : list[jnp.ndarray] or jnp.ndarray
            Shape (M, L, N).
            Real scalar-valued functions evaluated on the ``grid`` supplied to
            construct this object. These functions should be arguments to the callable
            ``integrand``. Use the method ``Bounce1D.reshape_data`` to reshape the data
            into the expected shape.
        weight : jnp.ndarray
            Shape (M, L, N).
            If supplied, the bounce integral labeled by well j is weighted such that
            the returned value is w(j) ∫ f(λ, ℓ) dℓ, where w(j) is ``weight``
            interpolated to the deepest point in that magnetic well. Use the method
            ``Bounce1D.reshape_data`` to reshape the data into the expected shape.
        points : tuple[jnp.ndarray]
            Shape (M, L, num_pitch, num_well).
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

        Returns
        -------
        result : jnp.ndarray
            Shape (M, L, num_pitch, num_well).
            Last axis enumerates the bounce integrals for a given field line,
            flux surface, and pitch value.

        """
        if points is None:
            points = self.points(pitch_inv)
        result = _bounce_quadrature(
            x=self._x,
            w=self._w,
            integrand=integrand,
            points=points,
            pitch_inv=pitch_inv,
            f=setdefault(f, []),
            data=self._data,
            knots=self._zeta,
            method=method,
            batch=batch,
            check=check,
            plot=plot,
        )
        if weight is not None:
            result *= interp_to_argmin(
                weight,
                points,
                self._zeta,
                self.B,
                self._dB_dz,
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
            Shape (num_pitch, ).
            Optional, 1/λ values whose corresponding bounce points on the field line
            specified by Clebsch coordinate α(m), ρ(l) will be plotted.
        kwargs
            Keyword arguments into ``desc/integrals/bounce_utils.py::plot_ppoly``.

        Returns
        -------
        fig, ax
            Matplotlib (fig, ax) tuple.

        """
        B, dB_dz = self.B, self._dB_dz
        if B.ndim == 4:
            B = B[m]
            dB_dz = dB_dz[m]
        if B.ndim == 3:
            B = B[l]
            dB_dz = dB_dz[l]
        if pitch_inv is not None:
            errorif(
                pitch_inv.ndim > 1,
                msg=f"Got pitch_inv.ndim={pitch_inv.ndim}, but expected 1.",
            )
            z1, z2 = bounce_points(pitch_inv, self._zeta, B, dB_dz)
            kwargs["z1"] = z1
            kwargs["z2"] = z2
            kwargs["k"] = pitch_inv
        fig, ax = plot_ppoly(PPoly(B.T, self._zeta), **_set_default_plot_kwargs(kwargs))
        return fig, ax
