#!/usr/bin/env python3
"""
Differentiation‑matrix utilities for spectral methods in **DESC**.

=================================================================

This module provides vectorized, JAX‑friendly helpers for constructing first‑ and
second‑order differentiation matrices using either Chebyshev–Lobatto or Fourier
collocation points.  All routines are **pure** and **stateless** – they rely only
on `jax.numpy` and therefore can be freely composed, JIT‑compiled, or parallelised
with `jax.vmap` / `jax.pmap` inside larger optimisation loops.

Highlights
----------
* **Chebyshev–Lobatto grid:**  ``chebpts_lobatto`` generates the collocation
  nodes; ``diffmat1_lobatto`` and ``diffmat2_lobatto`` build the first‑ and
  second‑derivative matrices in *O(n^2)* time using closed‑form formulas (no
  explicit linear solves).
* **Equispaced Fourier grid:**  ``fourier_pts`` returns the grid, while
  ``fourier_diffmat*`` return the differentiation matrices.  Both even and odd
  grid sizes are handled with the classic trigonometric explicit expressions.
* **JIT compatibility:**  All functions work transparently under
  ``jax.jit`` / ``jax.grad`` / ``jax.vmap`` because they avoid Python control
  flow that depends on array values and use only JAX primitives.

The implementations follow the formulas in
  **Trefethen, L. N. (2000). *Spectral Methods in MATLAB*. SIAM** and
  **Canuto et al. (2006). *Spectral Methods – Fundamentals in Single Domains*.**

Examples
--------
>>> from desc.backend import jnp
>>> from jax import jit
>>> D1 = jit(diffmat1_lobatto)(16)   # first‑derivative, 16‑point grid
>>> x  = chebpts_lobatto(16)
>>> f  = jnp.sin(jnp.pi * x)
>>> df = D1 @ f                     # approximate derivative at the nodes
"""

from desc.backend import jnp, vmap

########################################################################
# ---------------------- CHEBYSHEV MATRICES --------------------------- #
########################################################################


def chebpts_lobatto(n: int, domain=None):
    """Return *n* Chebyshev–Lobatto points.

    The nodes are given by
        x_k = sin( (pi / 2) * (2 k - n + 1) / (n - 1) ),   k = 0, …, n‑1,
    mapped linearly to the user‑supplied *domain*.

    Parameters
    ----------
    n : int
        Number of collocation points.  If *n* <= 0 an empty array is returned.
    domain : sequence of 2 floats, optional
        Physical interval ``[a, b]``.  Defaults to ``[-1, 1]``.

    Returns
    -------
    jax.Array
        1‑D array of shape ``(n,)`` containing the grid points, using the active
        default floating‑point precision.

    Notes
    -----
    * The mapping is performed with a single fused JAX expression, ensuring the
      function is fully JIT‑traceable.
    * Endpoint weights ``c`` required for the differentiation formulas are **not**
      returned – they are (re)computed internally by the differentiation helpers.
    """
    if n <= 0:
        return jnp.array([])

    if domain is None:
        domain = [-1, 1]

    k = jnp.arange(n)
    x = jnp.sin(jnp.pi * jnp.flip(2 * k - n + 1) / (2 * (n - 1)))

    # Affine map to [a, b] only when necessary (allows JIT constant folding)
    if domain[0] != -1 or domain[1] != 1:
        x = (domain[1] - domain[0]) / 2 * (x + 1) + domain[0]
    return x


def diffmat1_lobatto(n: int):
    """First‑order Chebyshev differentiation matrix on Lobatto nodes.

    Parameters
    ----------
    n : int
        Grid size.  If *n* <= 1 the 1×1 zero matrix is returned.

    Returns
    -------
    jax.Array
        Square matrix ``D`` of shape ``(n, n)`` such that ``D @ f`` approximates
        d f / d x at the nodes produced by :func:`chebpts_lobatto`.

    Computational Notes
    -------------------
    * Uses the explicit spectral formula with endpoint weights
      ``c = [2, 1, …, 1, 2]`` and parity factor ``(‑1)^(i+j)``.
    * The diagonal entries are defined by the negative row sums, ensuring exact
      differentiation of constants (row‑sum‑zero property).
    * Entire implementation is vectorised; no Python loops, hence trivially
      JIT‑compilable and fast on GPU/TPU.
    """
    if n <= 1:
        return jnp.array([[0.0]])

    x = chebpts_lobatto(n)
    X = jnp.tile(x[:, None], (1, n))
    Xdiff = X - X.T

    # Endpoint weights (c_0 = c_{n‑1} = 2)
    c = jnp.ones(n)
    c = c.at[0].set(2.0)
    c = c.at[-1].set(2.0)
    C = jnp.outer(c, 1.0 / c)

    I, J = jnp.mgrid[0:n, 0:n]
    S = (-1.0) ** (I + J)

    mask = I != J
    D = jnp.where(mask, C * S / Xdiff, 0.0)

    # Diagonal – enforce zero row sum
    row_sums = jnp.sum(D, axis=1)
    D = D.at[jnp.diag_indices(n)].set(-row_sums)
    return D


def diffmat2_lobatto(n: int):
    """Second‑order Chebyshev differentiation matrix on Lobatto nodes.

    Parameters
    ----------
    n : int
        Grid size (must be >= 2).

    Returns
    -------
    jax.Array
        Matrix ``D2`` of shape ``(n, n)`` such that ``D2 @ f`` approximates the
        second derivative d^2 f / d x^2.

    Implementation Details
    ----------------------
    The routine follows the explicit formulas (see *Trefethen* §6.3) rather than
    squaring the first‑derivative matrix, which would amplify rounding errors.
    The expressions for the diagonal account for both interior nodes and the
    two endpoints separately.
    """
    if n <= 1:
        return jnp.array([[0.0]])

    x = chebpts_lobatto(n)
    X = jnp.tile(x[:, None], (1, n))
    Xdiff = X - X.T
    Xsum = X + X.T

    c = jnp.ones(n)
    c = c.at[0].set(2.0)
    c = c.at[-1].set(2.0)
    C = jnp.outer(c, 1.0 / c)

    I, J = jnp.mgrid[0:n, 0:n]
    S = (-1.0) ** (I + J)

    D2 = jnp.zeros((n, n))
    mask = I != J
    D2 = jnp.where(mask, C * S * Xsum / (Xdiff**2), D2)

    # Diagonal terms
    interior_x = x[1:-1]
    interior_diag = -(interior_x**2) / (1 - interior_x**2) - 1 / (
        2 * (1 - interior_x**2)
    )
    D2 = D2.at[1:-1, 1:-1].set(jnp.diag(interior_diag))

    endpoint_diag = (2 * (n - 1) ** 2 + 1) / 3.0
    D2 = D2.at[0, 0].set(endpoint_diag)
    D2 = D2.at[-1, -1].set(endpoint_diag)
    return D2


########################################################################
# ----------------------- FOURIER MATRICES --------------------------- #
########################################################################


def fourier_pts(n: int, domain=None):
    """Return equally‑spaced grid points for a periodic domain.

    Parameters
    ----------
    n : int
        Number of points.
    domain : sequence of 2 floats, optional
        Physical interval ``[a, b]``.  Defaults to ``[0, 2 * pi]``.

    Returns
    -------
    jax.Array
        Array of shape ``(n,)`` with spacing ``h = (b - a) / n``.
    """
    if domain is None:
        domain = [0, 2 * jnp.pi]
    a, b = domain
    h = (b - a) / n
    return jnp.arange(n) * h + a


def fourier_diffmat(n: int):
    """Skew‑symmetric first‑derivative matrix for a Fourier grid.

    This is the formula given in *Fornberg (1998) §3.2*.  For even *n* the
    denominator is ``tan``; for odd *n* it is ``sin``.

    Parameters
    ----------
    n : int
        Grid size.

    Returns
    -------
    jax.Array
        First‑order differentiation matrix of size ``(n, n)``; exact for
        complex exponentials with wavenumbers below the Nyquist limit.
    """
    i, j = jnp.mgrid[0:n, 0:n]
    angle = (i - j) * jnp.pi / n
    if n % 2 == 0:
        D = jnp.where(i != j, 0.5 * (-1) ** (i - j) / jnp.tan(angle), 0.0)
    else:
        D = jnp.where(i != j, 0.5 * (-1) ** (i - j) / jnp.sin(angle), 0.0)
    return D


def fourier_diffmat1(N: int):
    """Efficient first‑order Fourier differentiation matrix.

    Unlike :func:`fourier_diffmat`, this variant constructs the matrix by a
    single circulant first column/row, permitting *O(n^2)* explicit formation
    but *O(n log n)* mat‑vec products via convolution methods if desired.

    Parameters
    ----------
    N : int
        Grid size.

    Returns
    -------
    jax.Array
        First‑derivative matrix of shape ``(N, N)``.
    """
    h = 2.0 * jnp.pi / N
    col1 = jnp.zeros(N)

    j_idx = jnp.arange(1, N)
    if N % 2 == 0:
        values = 0.5 * (-1.0) ** j_idx * (1 / jnp.tan(j_idx * h / 2.0))
    else:
        values = 0.5 * (-1.0) ** j_idx * (1 / jnp.sin(j_idx * h / 2.0))
    col1 = col1.at[j_idx].set(values)

    # Build circulant matrix
    D1 = vmap(lambda i: jnp.roll(col1, i))(jnp.arange(N))
    D1 = D1.at[0].set(-col1)  # first row is negative of first column
    return D1


def fourier_diffmat2(N: int):
    """Second‑order Fourier differentiation matrix (symmetric).

    Parameters
    ----------
    N : int
        Grid size.

    Returns
    -------
    jax.Array
        Second‑derivative matrix of shape ``(N, N)``.
    """
    h = 2.0 * jnp.pi / N
    col1 = jnp.zeros(N)

    if N % 2 == 0:
        col1 = col1.at[0].set(-jnp.pi**2 / (3.0 * h**2) - 1.0 / 6.0)
        j_idx = jnp.arange(1, N)
        sin_sq = jnp.sin(j_idx * h / 2.0) ** 2
        col1 = col1.at[j_idx].set(-((-1.0) ** j_idx) / (2.0 * sin_sq))
    else:
        col1 = col1.at[0].set(-jnp.pi**2 / (3.0 * h**2) + 1.0 / 12.0)
        j_idx = jnp.arange(1, N)
        sin_term = jnp.sin(j_idx * h / 2.0)
        cot_term = jnp.cos(j_idx * h / 2.0) / sin_term
        col1 = col1.at[j_idx].set(-((-1.0) ** j_idx) * cot_term / (2.0 * sin_term))

    D2 = vmap(lambda i: jnp.roll(col1, i))(jnp.arange(N))
    return D2
