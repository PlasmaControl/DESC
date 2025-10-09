"""Utilities for quadratures.

Notes
-----
Bounce integrals with bounce points where the derivative of B does not vanish
have 1/2 power law singularities. However, strongly singular integrals where the
domain of the integral ends at the local extrema of B are not integrable.

Everywhere except for the extrema, an implicit Chebyshev (``chebgauss1`` or
``chebgauss2`` or modified Legendre quadrature (with ``automorphism_sin``)
captures the integral because √(1−(ζ/ζᵢ)²) / √ (1−λB) ∼ k(λ, ζ) is smooth in ζ.
The clustering of the nodes near the singularities is sufficient to estimate
k(ζ, λ).
"""

import scipy
from orthax.chebyshev import chebgauss, chebweight
from orthax.legendre import legder, legval

from desc.backend import eigh_tridiagonal, fori_loop, jnp, put
from desc.utils import errorif


def bijection_to_disc(x, a, b):
    """[a, b] ∋ x ↦ y ∈ [−1, 1]."""
    return 2 * (x - a) / (b - a) - 1


def bijection_from_disc(x, a, b):
    """[−1, 1] ∋ x ↦ y ∈ [a, b]."""
    return 0.5 * (b - a) * (x + 1) + a


def grad_bijection_from_disc(a, b):
    """Gradient wrt ``x`` of ``bijection_from_disc``."""
    return 0.5 * (b - a)


def automorphism_arcsin(x, gamma=jnp.cos(0.5)):
    """[-1, 1] ∋ x ↦ y ∈ [−1, 1].

    This map decreases node density near the boundary by the asymptotic factor
    √(1−γ²x²) and adds a 1/√(1−γ²x²) factor to the integrand. When applied
    to any Gaussian quadrature, the default setting modifies the quadrature
    to be almost-equispaced without sacrificing spectral convergence.

    References
    ----------
    Kosloff and Tal-Ezer almost-equispaced grid where γ = 1−β.
    See Boyd, Chebyshev and Fourier Spectral Methods section 16.9.

    Parameters
    ----------
    x : jnp.ndarray
        Points to transform.
    gamma : float
        Transformation parameter γ = 1−β. Default is cos(0.5).

    Returns
    -------
    y : jnp.ndarray
        Transformed points.

    """
    return jnp.arcsin(gamma * x) / jnp.arcsin(gamma)


def grad_automorphism_arcsin(x, gamma=jnp.cos(0.5)):
    """Gradient of arcsin automorphism."""
    return gamma / jnp.arcsin(gamma) / jnp.sqrt(1 - (gamma * x) ** 2)


grad_automorphism_arcsin.__doc__ += "\n" + automorphism_arcsin.__doc__


def automorphism_sin(x, m=10):
    """[-1, 1] ∋ x ↦ y ∈ [−1, 1].

    This map increases node density near the boundary by the asymptotic factor
    1/√(1−x²) and adds a cosine factor to the integrand.

    Parameters
    ----------
    x : jnp.ndarray
        Points to transform.
    m : float
        Number of machine epsilons used for floating point error buffer.

    Returns
    -------
    y : jnp.ndarray
        Transformed points.

    """
    y = jnp.sin(0.5 * jnp.pi * x)
    # y is an expansion, so y(x) > x near x ∈ {−1, 1} and there is a tendency
    # for floating point error to overshoot the true value.
    eps = m * jnp.finfo(jnp.array(1.0).dtype).eps
    return jnp.clip(y, -1 + eps, 1 - eps)


def grad_automorphism_sin(x):
    """Gradient of sin automorphism."""
    return 0.5 * jnp.pi * jnp.cos(0.5 * jnp.pi * x)


grad_automorphism_sin.__doc__ += "\n" + automorphism_sin.__doc__


def tanh_sinh(deg, m=10):
    """Tanh-Sinh quadrature.

    Returns quadrature points xₖ and weights wₖ for the approximate evaluation
    of the integral ∫₋₁¹ f(x) dx ≈ ∑ₖ wₖ f(xₖ).

    Parameters
    ----------
    deg : int
        Number of quadrature points.
    m : float
        Number of machine epsilons used for floating point error buffer. Larger
        implies less floating point error, but increases the minimum achievable
        quadrature error.

    Returns
    -------
    x, w : tuple[jnp.ndarray]
        Shape (deg, ).
        Quadrature points and weights.

    """
    # buffer to avoid numerical instability
    x_max = jnp.array(1.0)
    x_max = x_max - m * jnp.finfo(x_max.dtype).eps
    t_max = jnp.arcsinh(2 * jnp.arctanh(x_max) / jnp.pi)
    # maximal-spacing scheme, doi.org/10.48550/arXiv.2007.15057
    t = jnp.linspace(-t_max, t_max, deg)
    dt = 2 * t_max / (deg - 1)
    arg = 0.5 * jnp.pi * jnp.sinh(t)
    x = jnp.tanh(arg)  # x = g(t)
    w = 0.5 * jnp.pi * jnp.cosh(t) / jnp.cosh(arg) ** 2 * dt  # w = (dg/dt) dt
    return x, w


def leggauss_lob(deg, interior_only=False):
    """Lobatto-Gauss-Legendre quadrature.

    Returns quadrature points xₖ and weights wₖ for the approximate evaluation
    of the integral ∫₋₁¹ f(x) dx ≈ ∑ₖ wₖ f(xₖ).

    Parameters
    ----------
    deg : int
        Number of quadrature points.
    interior_only : bool
        Whether to exclude the points and weights at -1 and 1;
        useful if f(-1) = f(1) = 0. If true, then ``deg`` points are still
        returned; these are the interior points for lobatto quadrature of ``deg+2``.

    Returns
    -------
    x, w : tuple[jnp.ndarray]
        Shape (deg, ).
        Quadrature points and weights.

    """
    N = deg + 2 * bool(interior_only)
    errorif(N < 2)

    # Golub-Welsh algorithm
    n = jnp.arange(2, N - 1)
    x = eigh_tridiagonal(
        jnp.zeros(N - 2),
        jnp.sqrt((n**2 - 1) / (4 * n**2 - 1)),
        eigvals_only=True,
    )
    c0 = put(jnp.zeros(N), -1, 1)

    # improve (single multiplicity) roots by one application of Newton
    c = legder(c0)
    dy = legval(x=x, c=c)
    df = legval(x=x, c=legder(c))
    x -= dy / df

    w = 2 / (N * (N - 1) * legval(x=x, c=c0) ** 2)

    if not interior_only:
        x = jnp.hstack([-1.0, x, 1.0])
        w_end = 2 / (deg * (deg - 1))
        w = jnp.hstack([w_end, w, w_end])

    assert x.size == w.size == deg
    return x, w


def uniform(deg):
    """Uniform open quadrature with nodes closer to boundary.

    Returns quadrature points xₖ and weights wₖ for the approximate evaluation
    of the integral ∫₋₁¹ f(x) dx ≈ ∑ₖ wₖ f(xₖ).

    Parameters
    ----------
    deg : int
        Number of quadrature points.

    Returns
    -------
    x, w : tuple[jnp.ndarray]
        Shape (deg, ).
        Quadrature points and weights.

    """
    # Define x = 2/π arcsin y and g : y ↦ f(x(y)).
    #   ∫₋₁¹ f(x) dx = 2/π ∫₋₁¹ (1−y²)⁻⁰ᐧ⁵ g(y) dy
    # ∑ₖ wₖ f(x(yₖ)) = 2/π ∑ₖ ωₖ g(yₖ)
    # Given roots yₖ of Chebyshev polynomial, x(yₖ) below is uniform in (-1, 1).
    x = jnp.arange(-deg + 1, deg + 1, 2) / deg
    w = 2 / deg * jnp.ones(deg)
    return x, w


def simpson2(deg):
    """Open Simpson rule completed by midpoint at boundary.

    Parameters
    ----------
    deg : int
        Number of quadrature points. Rounds up to odd integer.

    Returns
    -------
    x, w : tuple[jnp.ndarray]
        Shape (deg, ).
        Quadrature points and weights.

    """
    assert deg > 3
    deg -= 1 + (deg % 2)
    x = jnp.arange(-deg + 1, deg + 1, 2) / deg
    h_simp = (x[-1] - x[0]) / (deg - 1)
    h_midp = (x[0] + 1) / 2

    x = jnp.hstack([-1 + h_midp, x, 1 - h_midp], dtype=float)
    w = jnp.hstack(
        [
            2 * h_midp,
            h_simp
            / 3
            * jnp.hstack([1, jnp.tile(jnp.array([4, 2]), (deg - 3) // 2), 4, 1]),
            2 * h_midp,
        ],
        dtype=float,
    )
    return x, w


def chebgauss1(deg):
    """Gauss-Chebyshev quadrature of the first kind with implicit weighting.

    Returns quadrature points xₖ and weights wₖ for the approximate evaluation
    of the integral ∫₋₁¹ f(x) dx ≈ ∑ₖ wₖ f(xₖ) where f(x) = g(x) / √(1−x²).

    Parameters
    ----------
    deg : int
        Number of quadrature points.

    Returns
    -------
    x, w : tuple[jnp.ndarray]
        Shape (deg, ).
        Quadrature points and weights.

    """
    x, w = chebgauss(deg)
    return x, w / chebweight(x)


def chebgauss2(deg):
    """Gauss-Chebyshev quadrature of the second kind with implicit weighting.

    Returns quadrature points xₖ and weights wₖ for the approximate evaluation
    of the integral ∫₋₁¹ f(x) dx ≈ ∑ₖ wₖ f(xₖ) where f(x) = g(x) √(1−x²).

    Parameters
    ----------
    deg : int
        Number of quadrature points.

    Returns
    -------
    x, w : tuple[jnp.ndarray]
        Shape (deg, ).
        Quadrature points and weights.

    """
    # Adapted from
    # github.com/scipy/scipy/blob/v1.14.1/scipy/special/_orthogonal.py#L1803-L1851.
    t = jnp.arange(deg, 0, -1) * jnp.pi / (deg + 1)
    x = jnp.cos(t)
    w = jnp.pi * jnp.sin(t) / (deg + 1)
    return x, w


def get_quadrature(quad, automorphism):
    """Apply automorphism to given quadrature.

    Parameters
    ----------
    quad : tuple[jnp.ndarray]
        Quadrature points xₖ and weights wₖ for the approximate evaluation of
        the integral ∫₋₁¹ g(x) dx = ∑ₖ wₖ g(xₖ).
    automorphism : tuple[Callable] or None
        The first callable should be an automorphism of the real interval [-1, 1].
        The second callable should be the derivative of the first. This map defines
        a change of variable for the bounce integral. The choice made for the
        automorphism will affect the performance of the quadrature method.

    Returns
    -------
    x, w : tuple[jnp.ndarray]
        Quadrature points and weights.

    """
    x, w = quad
    assert x.ndim == w.ndim == 1
    if automorphism is not None:
        auto, grad_auto = automorphism
        w = w * grad_auto(x)
        x = auto(x)
    return x, w


def nfp_loop(source_grid, func, init_val):
    """Calculate effects from source points on a single field period.

    The integral is computed on the full domain because the kernels of interest
    have toroidal variation and are not NFP periodic. To that end, the integral
    is computed on every field period and summed. The ``source_grid`` is the
    first field period because DESC truncates the computational domain to
    ζ ∈ [0, 2π/grid.NFP) and changes variables to the spectrally condensed
    ζ* = basis.NFP ζ. The domain is shifted to the next field period by
    incrementing the toroidal coordinate of the grid by 2π/NFP.

    For an axisymmetric configuration, it is most efficient for ``source_grid`` to
    be a single toroidal cross-section. To capture toroidal effects of the kernels
    on those grids for axisymmetric configurations, we set a dummy value for NFP to
    an integer larger than 1 so that the toroidal increment can move to a new spot.

    Parameters
    ----------
    source_grid : _Grid
        Grid with points ζ ∈ [0, 2π/grid.NFP).
    func : callable
        Should accept argument ``zeta_j`` denoting toroidal coordinates of
        field period ``j``.
    init_val : jnp.ndarray
        Initial loop carry value.

    Returns
    -------
    result : jnp.ndarray
        Shape is ``init_val.shape``.

    """
    errorif(
        source_grid.num_zeta == 1 and source_grid.NFP == 1,
        msg="Source grid cannot compute toroidal effects.\n"
        "Increase NFP of source grid to e.g. 64.",
    )
    zeta = source_grid.nodes[:, 2]
    NFP = source_grid.NFP

    def body(j, f):
        return f + func(zeta + j * 2 * jnp.pi / NFP)

    return fori_loop(0, NFP, body, init_val)


def chi(r):
    """Partition of unity function in polar coordinates. Eq 39 in [2].

    Parameters
    ----------
    r : jnp.ndarray
        Absolute value of radial coordinate in polar domain.

    """
    return jnp.exp(-36 * jnp.abs(r) ** 8)


def eta(theta, zeta, theta0, zeta0, ht, hz, st, sz):
    """Partition of unity function in rectangular coordinates.

    Consider the mapping from
    (θ,ζ) ∈ [-π, π) × [-π/NFP, π/NFP) to (ρ,ω) ∈ [−1, 1] × [0, 2π)
    defined by
    θ − θ₀ = h₁ s₁/2 ρ sin ω
    ζ − ζ₀ = h₂ s₂/2 ρ cos ω
    with Jacobian determinant norm h₁h₂ s₁s₂/4 |ρ|.

    In general in dimensions higher than one, the mapping that determines a
    change of variable for integration must be bijective. This is satisfied
    only if s₁ = 2π/h₁ and s₂ = (2π/NFP)/h₂. In the particular case the
    integrand is nonzero in a subset of the domain, then the change of variable
    need only be a bijective map where the function does not vanish, more
    precisely, its set of compact support.

    The functions we integrate are proportional to η₀(θ,ζ) = χ₀(r) far from the
    singularity at (θ₀,ζ₀). Therefore, the support matches χ₀(r)'s, assuming
    this region is sufficiently large compared to the singular region.
    Here χ₀(r) has support where the argument r lies in [0, 1]. The map r
    defines a coordinate mapping between the toroidal domain and a polar domain
    such that the integration region in the polar domain (ρ,ω) ∈ [−1, 1] × [0, 2π)
    equals the compact support, and furthermore is a circular region around the
    singular point in (θ,ζ) geometry when s₁ × s₂ denote the number of grid points
    on a uniformly discretized toroidal domain (θ,ζ) ∈ [0, 2π)².
      χ₀ : r ↦ exp(−36r⁸)

      r : ρ, ω ↦ |ρ|

      r : θ, ζ ↦ 2 [ (θ−θ₀)²/(h₁s₁)² + (ζ−ζ₀)²/(h₂s₂)² ]⁰ᐧ⁵

    Hence, r ≥ 1 (r ≤ 1) outside (inside) the integration domain.

    The choice for the size of the support is determined by s₁ and s₂.
    The optimal choice is dependent on the nature of the singularity e.g. if the
    integrand decays quickly then the elliptical grid determined by s₁ and s₂
    can be made smaller and the integration will have higher resolution for the
    same number of quadrature points.

    With the above definitions the support lies on an s₁ × s₂ subset
    of a field period which has ``num_theta`` × ``num_zeta`` nodes total.
    Since kernels are 2π periodic, the choice for s₂ should be multiplied by NFP.
    Then the support lies on an s₁ × s₂ subset of the full domain. For large NFP
    devices such as Heliotron or tokamaks it is typical that s₁ ≪ s₂.

    Parameters
    ----------
    theta, zeta : jnp.ndarray
        Coordinates of points to evaluate partition function η₀(θ,ζ).
    theta0, zeta0 : jnp.ndarray
        Origin (θ₀,ζ₀) where the partition η₀ is unity.
    ht, hz : float
        Grid step size in θ and ζ.
    st, sz : int
        Extent of support is an ``st`` × ``sz`` subset
        of the full domain (θ,ζ) ∈ [0, 2π)² of ``source_grid``.
        Subset of ``source_grid.num_theta`` × ``source_grid.num_zeta*source_grid.NFP``.

    """
    dt = jnp.abs(theta - theta0)
    dz = jnp.abs(zeta - zeta0)
    # The distance spans (dθ,dζ) ∈ [0, π]², independent of NFP.
    dt = jnp.minimum(dt, 2 * jnp.pi - dt)
    dz = jnp.minimum(dz, 2 * jnp.pi - dz)
    r = 2 * jnp.hypot(dt / (ht * st), dz / (hz * sz))
    return chi(r)


def _get_polar_quadrature(q):
    """Polar nodes for quadrature around singular point.

    Parameters
    ----------
    q : int
        Order of quadrature in radial and azimuthal directions.

    Returns
    -------
    r, w : ndarray
        Radial and azimuthal coordinates.
    dr, dw : ndarray
        Radial and azimuthal spacing and quadrature weights.

    """
    Nr = Nw = q
    r, dr = scipy.special.roots_legendre(Nr)
    # integrate separately over [-1,0] and [0,1]
    r1 = 1 / 2 * r - 1 / 2
    r2 = 1 / 2 * r + 1 / 2
    r = jnp.concatenate([r1, r2])
    dr = jnp.concatenate([dr, dr]) / 2
    w = jnp.linspace(0, jnp.pi, Nw, endpoint=False)
    dw = jnp.ones_like(w) * jnp.pi / Nw
    r, w = jnp.meshgrid(r, w)
    r = r.ravel()
    w = w.ravel()
    dr, dw = jnp.meshgrid(dr, dw)
    dr = dr.ravel()
    dw = dw.ravel()
    return r, w, dr, dw


def _vanilla_params(grid):
    """Parameters for support size and quadrature resolution.

    These parameters do not account for grid anisotropy.

    Parameters
    ----------
    grid : LinearGrid
        Grid that can fft2.

    Returns
    -------
    st : int
        Extent of support is an ``st`` × ``sz`` subset
        of the full domain (θ,ζ) ∈ [0, 2π)² of ``grid``.
        Subset of ``grid.num_theta`` × ``grid.num_zeta*grid.NFP``.
    sz : int
        Extent of support is an ``st`` × ``sz`` subset
        of the full domain (θ,ζ) ∈ [0, 2π)² of ``grid``.
        Subset of ``grid.num_theta`` × ``grid.num_zeta*grid.NFP``.
    q : int
        Order of quadrature in radial and azimuthal directions.

    """
    Nt = grid.num_theta
    Nz = grid.num_zeta * grid.NFP
    q = int(jnp.sqrt(grid.num_nodes) // 2)
    s = min(q, Nt, Nz)
    return s, s, q


def _best_params(grid, ratio):
    """Parameters for heuristic support size and quadrature resolution.

    These parameters account for global grid anisotropy which ensures
    more robust convergence across a wider aspect ratio range.

    Parameters
    ----------
    grid : LinearGrid
        Grid that can fft2.
    ratio : float or jnp.ndarray
        Best ratio.

    Returns
    -------
    st : int
        Extent of support is an ``st`` × ``sz`` subset
        of the full domain (θ,ζ) ∈ [0, 2π)² of ``grid``.
        Subset of ``grid.num_theta`` × ``grid.num_zeta*grid.NFP``.
    sz : int
        Extent of support is an ``st`` × ``sz`` subset
        of the full domain (θ,ζ) ∈ [0, 2π)² of ``grid``.
        Subset of ``grid.num_theta`` × ``grid.num_zeta*grid.NFP``.
    q : int
        Order of quadrature in radial and azimuthal directions.

    """
    assert grid.can_fft2
    Nt = grid.num_theta
    Nz = grid.num_zeta * grid.NFP
    q = int(jnp.sqrt(grid.num_nodes if (grid.num_zeta > 1) else (Nt * Nz)) // 2)
    s = min(q, Nt, Nz)
    # Size of singular region in real space = s * h * |e_.|
    # For it to be a circle, choose radius ~ equal
    # s_t * h_t * |e_t| = s_z * h_z * |e_z|
    # s_z / s_t = h_t / h_z  |e_t| / |e_z| = Nz*NFP/Nt |e_t| / |e_z|
    # Denote ratio = < |e_z| / |e_t| > and
    #      s_ratio = s_z / s_t = Nz*NFP/Nt / ratio
    # Also want sqrt(s_z*s_t) ~ s = q.
    s_ratio = jnp.sqrt(Nz / Nt / ratio)
    st = jnp.clip(jnp.ceil(s / s_ratio).astype(int), None, Nt)
    sz = jnp.clip(jnp.ceil(s * s_ratio).astype(int), None, Nz)
    if s_ratio.size == 1:
        st = int(st)
        sz = int(sz)
    return st, sz, q


def _best_ratio(data):
    """Ratio to make singular integration partition ~circle in real space.

    Parameters
    ----------
    data : dict[str, jnp.ndarray]
        Dictionary of data evaluated on single flux surface grid that ``can_fft2``
        with keys ``|e_theta x e_zeta|``, ``e_theta``, and ``e_zeta``.

    """
    scale = jnp.linalg.norm(data["e_zeta"], axis=-1) / jnp.linalg.norm(
        data["e_theta"], axis=-1
    )
    return jnp.mean(scale * data["|e_theta x e_zeta|"]) / jnp.mean(
        data["|e_theta x e_zeta|"]
    )
