"""Utilities for quadratures."""

from orthax.legendre import legder, legval

from desc.backend import eigh_tridiagonal, jnp, put
from desc.utils import errorif


def bijection_to_disc(x, a, b):
    """[a, b] ∋ x ↦ y ∈ [−1, 1]."""
    y = 2.0 * (x - a) / (b - a) - 1.0
    return y


def bijection_from_disc(x, a, b):
    """[−1, 1] ∋ x ↦ y ∈ [a, b]."""
    y = 0.5 * (b - a) * (x + 1.0) + a
    return y


def grad_bijection_from_disc(a, b):
    """Gradient of affine bijection from disc."""
    dy_dx = 0.5 * (b - a)
    return dy_dx


def automorphism_arcsin(x):
    """[-1, 1] ∋ x ↦ y ∈ [−1, 1].

    The arcsin transformation introduces a singularity that augments the singularity
    in the bounce integral, so the quadrature scheme used to evaluate the integral must
    work well on functions with large derivative near the boundary.

    Parameters
    ----------
    x : jnp.ndarray
        Points to transform.

    Returns
    -------
    y : jnp.ndarray
        Transformed points.

    """
    y = 2.0 * jnp.arcsin(x) / jnp.pi
    return y


def grad_automorphism_arcsin(x):
    """Gradient of arcsin automorphism."""
    dy_dx = 2.0 / (jnp.sqrt(1.0 - x**2) * jnp.pi)
    return dy_dx


grad_automorphism_arcsin.__doc__ += "\n" + automorphism_arcsin.__doc__


def automorphism_sin(x, s=0, m=10):
    """[-1, 1] ∋ x ↦ y ∈ [−1, 1].

    When used as the change of variable map for the bounce integral, the Lipschitzness
    of the sin transformation prevents generation of new singularities. Furthermore,
    its derivative vanishes to zero slowly near the boundary, which will suppress the
    large derivatives near the boundary of singular integrals.

    In effect, this map pulls the mass of the integral away from the singularities,
    which should improve convergence if the quadrature performs better on less singular
    integrands. Pairs well with Gauss-Legendre quadrature.

    Parameters
    ----------
    x : jnp.ndarray
        Points to transform.
    s : float
        Strength of derivative suppression, s ∈ [0, 1].
    m : float
        Number of machine epsilons used for floating point error buffer.

    Returns
    -------
    y : jnp.ndarray
        Transformed points.

    """
    errorif(not (0 <= s <= 1))
    # s = 0 -> derivative vanishes like cosine.
    # s = 1 -> derivative vanishes like cosine^k.
    y0 = jnp.sin(0.5 * jnp.pi * x)
    y1 = x + jnp.sin(jnp.pi * x) / jnp.pi  # k = 2
    y = (1 - s) * y0 + s * y1
    # y is an expansion, so y(x) > x near x ∈ {−1, 1} and there is a tendency
    # for floating point error to overshoot the true value.
    eps = m * jnp.finfo(jnp.array(1.0).dtype).eps
    return jnp.clip(y, -1 + eps, 1 - eps)


def grad_automorphism_sin(x, s=0):
    """Gradient of sin automorphism."""
    dy0_dx = 0.5 * jnp.pi * jnp.cos(0.5 * jnp.pi * x)
    dy1_dx = 1.0 + jnp.cos(jnp.pi * x)
    dy_dx = (1 - s) * dy0_dx + s * dy1_dx
    return dy_dx


grad_automorphism_sin.__doc__ += "\n" + automorphism_sin.__doc__


def tanh_sinh(deg, m=10):
    """Tanh-Sinh quadrature.

    Returns quadrature points xₖ and weights wₖ for the approximate evaluation of the
    integral ∫₋₁¹ f(x) dx ≈ ∑ₖ wₖ f(xₖ).

    Parameters
    ----------
    deg : int
        Number of quadrature points.
    m : float
        Number of machine epsilons used for floating point error buffer. Larger implies
        less floating point error, but increases the minimum achievable error.

    Returns
    -------
    x, w : (jnp.ndarray, jnp.ndarray)
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

    Returns quadrature points xₖ and weights wₖ for the approximate evaluation of the
    integral ∫₋₁¹ f(x) dx ≈ ∑ₖ wₖ f(xₖ).

    Parameters
    ----------
    deg : int
        Number of quadrature points.
    interior_only : bool
        Whether to exclude the points and weights at -1 and 1;
        useful if f(-1) = f(1) = 0. If ``True``, then ``deg`` points are still
        returned; these are the interior points for lobatto quadrature of ``deg+2``.

    Returns
    -------
    x, w : (jnp.ndarray, jnp.ndarray)
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


def get_quadrature(quad, automorphism):
    """Apply automorphism to given quadrature.

    Parameters
    ----------
    quad : (jnp.ndarray, jnp.ndarray)
        Quadrature points xₖ and weights wₖ for the approximate evaluation of an
        integral ∫₋₁¹ g(x) dx = ∑ₖ wₖ g(xₖ).
    automorphism : (Callable, Callable) or None
        The first callable should be an automorphism of the real interval [-1, 1].
        The second callable should be the derivative of the first. This map defines
        a change of variable for the bounce integral. The choice made for the
        automorphism will affect the performance of the quadrature method.

    Returns
    -------
    x, w : (jnp.ndarray, jnp.ndarray)
        Quadrature points and weights.

    """
    x, w = quad
    assert x.ndim == w.ndim == 1
    if automorphism is not None:
        # Apply automorphisms to supress singularities.
        auto, grad_auto = automorphism
        w = w * grad_auto(x)
        # Recall bijection_from_disc(auto(x), ζ_b₁, ζ_b₂) = ζ.
        x = auto(x)
    return x, w


def composite_linspace(x, num):
    """Returns linearly spaced values between every pair of values in ``x``.

    Parameters
    ----------
    x : jnp.ndarray
        First axis has values to return linearly spaced values between. The remaining
        axes are batch axes. Assumes input is sorted along first axis.
    num : int
        Number of values between every pair of values in ``x``.

    Returns
    -------
    vals : jnp.ndarray
        Shape ((x.shape[0] - 1) * num + x.shape[0], *x.shape[1:]).
        Linearly spaced values between ``x``.

    """
    x = jnp.atleast_1d(x)
    vals = jnp.linspace(x[:-1], x[1:], num + 1, endpoint=False)
    vals = jnp.swapaxes(vals, 0, 1).reshape(-1, *x.shape[1:])
    vals = jnp.append(vals, x[jnp.newaxis, -1], axis=0)
    assert vals.shape == ((x.shape[0] - 1) * num + x.shape[0], *x.shape[1:])
    return vals
