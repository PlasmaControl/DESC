"""Utilities for quadratures."""

from orthax.legendre import legder, legval

from desc.backend import eigh_tridiagonal, jnp, put
from desc.utils import errorif


def bijection_to_disc(x, a, b):
    """[a, b] ∋ x ↦ y ∈ [−1, 1]."""
    y = 2 * (x - a) / (b - a) - 1
    return y


def bijection_from_disc(x, a, b):
    """[−1, 1] ∋ x ↦ y ∈ [a, b]."""
    y = (x + 1) / 2 * (b - a) + a
    return y


def grad_bijection_from_disc(a, b):
    """Gradient of affine bijection."""
    dy_dx = (b - a) / 2
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
    y = 2 * jnp.arcsin(x) / jnp.pi
    return y


def grad_automorphism_arcsin(x):
    """Gradient of arcsin automorphism."""
    dy_dx = 2 / (jnp.sqrt(1 - x**2) * jnp.pi)
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
    y0 = jnp.sin(jnp.pi * x / 2)
    y1 = x + jnp.sin(jnp.pi * x) / jnp.pi  # k = 2
    y = (1 - s) * y0 + s * y1
    # y is an expansion, so y(x) > x near x ∈ {−1, 1} and there is a tendency
    # for floating point error to overshoot the true value.
    eps = m * jnp.finfo(jnp.array(1.0).dtype).eps
    return jnp.clip(y, -1 + eps, 1 - eps)


def grad_automorphism_sin(x, s=0):
    """Gradient of sin automorphism."""
    dy0_dx = jnp.pi * jnp.cos(jnp.pi * x / 2) / 2
    dy1_dx = 1 + jnp.cos(jnp.pi * x)
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


def leggausslob(deg):
    """Lobatto-Gauss-Legendre quadrature.

    Returns quadrature points xₖ and weights wₖ for the approximate evaluation of the
    integral ∫₋₁¹ f(x) dx ≈ ∑ₖ wₖ f(xₖ).

    Parameters
    ----------
    deg : int
        Number of (interior) quadrature points to return.

    Returns
    -------
    x, w : (jnp.ndarray, jnp.ndarray)
        Quadrature points in (-1, 1) and associated weights.
        Excludes points and weights at -1 and 1.

    """
    # Designate two degrees for endpoints.
    deg = int(deg) + 2

    # Golub-Welsh algorithm for eigenvalues of orthogonal polynomials
    n = jnp.arange(2, deg - 1)
    x = eigh_tridiagonal(
        jnp.zeros(deg - 2),
        jnp.sqrt((n**2 - 1) / (4 * n**2 - 1)),
        eigvals_only=True,
    )
    c0 = put(jnp.zeros(deg), -1, 1)

    # improve (single multiplicity) roots by one application of Newton
    c = legder(c0)
    dy = legval(x=x, c=c)
    df = legval(x=x, c=legder(c))
    x -= dy / df

    w = 2 / (deg * (deg - 1) * legval(x=x, c=c0) ** 2)
    return x, w
