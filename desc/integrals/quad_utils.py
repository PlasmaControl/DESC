"""Utilities for quadratures."""

from jax.scipy.special import gammaln
from orthax.legendre import legder, legval

from desc.backend import eigh_tridiagonal, jnp, put
from desc.utils import check_posint, errorif


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


def automorphism_staircase1(x, x_0=0.5, m_1=2.0, m_2=2.0, eps=0.0):
    """[-1, 1] ∋ x ↦ y ∈ [0, 1].

    This map increases the node density near the point x_0 and the
    density on either side of x_0 is determined by m_1 and m_2.
    When plotting this function it looks like a staircase with
    a single step.

    Parameters
    ----------
    x : jnp.ndarray
        Points to transform.
    x_0 : float
        Point around which to concetrate node density.
    m_1 : float
        Variable to control node density to the left.
    m_2 : float
        Variable to control node density to the right.
    eps : float
        Lower bound of the transformed interval. The default, ``0``, preserves
        the original map to ``[0, 1]``. Set ``eps > 0`` to avoid a node at zero;
        the transformed interval is then ``[eps, 1]``.

    Returns
    -------
    y : jnp.ndarray
        Transformed points.

    """
    lower = x_0 * (1 - jnp.exp(-m_1 * (x + 1)) + 0.5 * (x + 1) * jnp.exp(-2 * m_1))
    upper = (1 - x_0) * (jnp.exp(m_2 * (x - 1)) + 0.5 * (x - 1) * jnp.exp(-2 * m_2))
    return eps + (1 - eps) * (lower + upper)


def automorphism_staircase2(x, x_0=0.0, x_1=0.5, m_1=1.0, m_2=1.0, m_3=10.0, m_4=10.0):
    """[-1, 1] ∋ x ↦ y ∈ [0, 1].

    This map increases the node density near the point x_0 and the
    density on either side of x_0 is determined by m_1 and m_2,
    whereas m_3 and m_4 make sure that the points are spaced more
    uniformly, especially further from the endpoints.
    When plotting this function it looks like a staircase with
    a three steps.

    Parameters
    ----------
    x : jnp.ndarray
        Points to transform.
    x_0 : float
        Point around which to concetrate node density.
    m_1 : float
        Variable to control node density to the left.
    m_2 : float
        Variable to control node density to the right.
    m_3 : float
        Variable to control node density near the left end.
    m_4 : float
        Variable to control node density near the right end.

    Returns
    -------
    y : jnp.ndarray
        Transformed points.
    """
    wL = 0.5 * (1.0 + x_0)  # left-side weight  (≥0, ≤1)
    wR = 0.5 * (1.0 - x_0)  # right-side weight (≥0, ≤1)

    lower = wL * (
        1.0 - jnp.exp(-m_1 * (x + 1.0)) + 0.5 * (x + 1.0) * jnp.exp(-2.0 * m_1)
    )
    upper = wR * (jnp.exp(m_2 * (x - 1.0)) + 0.5 * (x - 1.0) * jnp.exp(-2.0 * m_2))

    # Rescale to span [0,1] and shift to [-1,1]
    g_cluster = 2.0 * (lower + upper) - 1.0  # monotone, g_cluster(±1)=±1

    # Left logistic — increasing from 0 at x=-1 to 1 at x=+1
    s_axis = 1.0 / (1.0 + jnp.exp(-m_3 * (x + 1.0)))
    s_axis0 = 1.0 / (1.0 + jnp.exp(-m_3 * 0.0))  # value at x=-1
    s_axis1 = 1.0 / (1.0 + jnp.exp(-m_3 * 2.0))  # value at x=+1
    axis = wL * (s_axis - s_axis0) / (s_axis1 - s_axis0)  # maps 0→1

    # Right logistic — also increasing after the flip
    s_edge_raw = 1.0 / (1.0 + jnp.exp(m_4 * (x - 1.0)))  # decreasing
    s_edge = 1.0 - s_edge_raw  # now increasing
    s_edge0 = 1.0 - 1.0 / (1.0 + jnp.exp(m_4 * -2.0))
    s_edge1 = 1.0 - 1.0 / (1.0 + jnp.exp(0.0))
    edge = wR * (s_edge - s_edge0) / (s_edge1 - s_edge0)  # maps 0→1

    g_axisedge = 2.0 * (axis + edge) - 1.0  # monotone, ±1 at ends

    # Identity map contributes (1-x1) · x
    return (1.0 - x_1) * x + x_1 * (g_cluster + g_axisedge - x)  # still ↑, hits ±1


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
    """Simpson’s 1/3 in the interior completed by an open midpoint scheme.

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
    t = jnp.pi * jnp.arange(1, 2 * deg, 2) / (2 * deg)
    x = jnp.cos(t)
    w = jnp.pi * jnp.sin(t) / deg
    return x, w


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


def _jacobi_diag_offdiag(N, alpha, beta):
    """Return the Jacobi matrix diagonals for orthonormal Jacobi polynomials."""
    a0 = (beta - alpha) / (alpha + beta + 2.0)
    n = jnp.arange(1, N, dtype=jnp.float64)
    two_n_ab = 2.0 * n + alpha + beta
    a_rest = (beta**2 - alpha**2) / (two_n_ab * (two_n_ab + 2.0))
    a = jnp.concatenate([jnp.array([a0], dtype=jnp.float64), a_rest])

    # The usual expression has a removable 0/0 for n=1 when alpha + beta=-1.
    at_n1 = n == 1
    ratio = jnp.where(
        at_n1,
        1.0,
        (n + alpha + beta) / (two_n_ab - 1.0),
    )
    b_squared = (
        4.0 * n * (n + alpha) * (n + beta) * ratio / (two_n_ab**2 * (two_n_ab + 1.0))
    )
    return a, jnp.sqrt(b_squared)


def _gauss_jacobi(N, alpha, beta):
    """Return N-point Gauss-Jacobi nodes and weights on ``[-1, 1]``.

    The rule integrates polynomials through degree ``2 * N - 1`` against
    ``(1 - x)**alpha * (1 + x)**beta``.
    """
    N = check_posint(N, "N", False)
    errorif(alpha <= -1 or beta <= -1, ValueError, "alpha and beta must exceed -1.")

    diagonal, off_diagonal = _jacobi_diag_offdiag(N, alpha, beta)
    matrix = (
        jnp.diag(diagonal) + jnp.diag(off_diagonal, k=1) + jnp.diag(off_diagonal, k=-1)
    )
    nodes, eigenvectors = jnp.linalg.eigh(matrix)

    log_mu0 = (
        (alpha + beta + 1.0) * jnp.log(2.0)
        + gammaln(alpha + 1.0)
        + gammaln(beta + 1.0)
        - gammaln(alpha + beta + 2.0)
    )
    weights = jnp.exp(log_mu0) * eigenvectors[0] ** 2
    return nodes, weights


def gauss_radau_jacobi(N, alpha=0.0, beta=1.0):
    """Left-Gauss-Radau-Jacobi quadrature on ``[-1, 1]``.

    The left endpoint is fixed at ``x[0] = -1`` and the rule integrates
    polynomials through degree ``2 * N - 2`` against
    ``(1 - x)**alpha * (1 + x)**beta``.

    Parameters
    ----------
    N : int
        Number of quadrature nodes, at least 2.
    alpha, beta : float
        Jacobi weight exponents, both greater than -1. The default ``(0, 1)``
        corresponds to a cylindrical radial weight after shifting to ``[0, 1]``.

    Returns
    -------
    x, w : tuple[jax.Array]
        Ascending nodes and positive weights, both with shape ``(N,)``.
    """
    N = check_posint(N, "N", False)
    errorif(N < 2, ValueError, "N must be at least 2.")
    errorif(alpha <= -1 or beta <= -1, ValueError, "alpha and beta must exceed -1.")

    interior, _ = _gauss_jacobi(N - 1, alpha, beta + 1.0)
    nodes = jnp.concatenate([jnp.array([-1.0]), interior])

    # Integrate the nodal Lagrange basis with an auxiliary Gauss-Jacobi rule.
    auxiliary_nodes, auxiliary_weights = _gauss_jacobi(2 * N + 5, alpha, beta)
    auxiliary_difference = auxiliary_nodes[:, None] - nodes[None, :]
    full_product = jnp.prod(auxiliary_difference, axis=1)
    denominator_matrix = nodes[:, None] - nodes[None, :] + jnp.eye(N)
    denominator = jnp.prod(denominator_matrix, axis=1)
    lagrange = full_product[:, None] / auxiliary_difference / denominator[None, :]
    weights = jnp.sum(auxiliary_weights[:, None] * lagrange, axis=0)
    return nodes, weights


def zernike_nodes_weights(n_rho, n_theta):
    """Return radial and poloidal nodes and weights for a unit disc.

    Radial nodes are shifted Gauss-Jacobi ``(alpha=0, beta=1)`` nodes in
    ``(0, 1)``. Poloidal nodes are equally spaced on ``[0, 2*pi)``.

    Parameters
    ----------
    n_rho, n_theta : int
        Number of radial and poloidal nodes.

    Returns
    -------
    rho, w_rho, theta, w_theta : tuple[jax.Array]
        One-dimensional node and weight arrays for the two coordinates.
    """
    n_rho = check_posint(n_rho, "n_rho", False)
    n_theta = check_posint(n_theta, "n_theta", False)
    x, w = _gauss_jacobi(n_rho, 0.0, 1.0)
    rho = (1.0 + x) / 2.0
    w_rho = w / (4.0 * rho)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, n_theta, endpoint=False)
    w_theta = jnp.full(n_theta, 2.0 * jnp.pi / n_theta)
    return rho, w_rho, theta, w_theta


def _bspline_clamped_uniform_knots(N, degree):
    """Return a clamped-uniform B-spline knot vector on ``[-1, 1]``."""
    N = check_posint(N, "N", False)
    degree = check_posint(degree, "degree", False)
    errorif(
        N < degree + 1,
        ValueError,
        "N must be at least degree + 1.",
    )
    interior = jnp.linspace(-1.0, 1.0, N - degree + 1)[1:-1]
    return jnp.concatenate(
        [
            jnp.full(degree + 1, -1.0),
            interior,
            jnp.full(degree + 1, 1.0),
        ]
    )


def bspline_nodes_weights(N, degree=4):
    """Return Greville nodes and positive B-spline integration weights.

    Parameters
    ----------
    N : int
        Number of basis functions and collocation nodes.
    degree : int
        Polynomial degree. ``N`` must be at least ``degree + 1``.

    Returns
    -------
    x, w : tuple[jax.Array]
        Greville abscissae and exact basis-integral weights on ``[-1, 1]``.
    """
    knots = _bspline_clamped_uniform_knots(N, degree)
    i = jnp.arange(N)
    indices = i[:, None] + jnp.arange(1, degree + 1)[None, :]
    nodes = jnp.mean(knots[indices], axis=1)
    weights = (knots[i + degree + 1] - knots[i]) / (degree + 1)
    return nodes, weights
