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

from desc.backend import jnp

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


def cheb_D1(N):
    """First-order Chebyshev differentiation matrix."""
    # Indices for flipping trick
    n1 = N // 2
    n2 = (N + 1) // 2

    # Compute theta vector
    k = jnp.arange(N)
    th = k * jnp.pi / (N - 1)

    # Create matrices of theta values
    T1, T2 = jnp.meshgrid(th / 2, th / 2)

    # Compute differences using trigonometric identity
    DX = 2 * jnp.sin(T1 + T2) * jnp.sin(T1 - T2)

    # Apply the flipping trick
    DX_top = DX[:n1, :]
    DX_bottom = -jnp.flip(jnp.flip(DX[:n2, :], axis=0), axis=1)
    DX = jnp.vstack([DX_top, DX_bottom])

    # Put 1's on the main diagonal
    DX = DX.at[jnp.diag_indices(N)].set(1.0)

    # Create the C matrix (c_i/c_j)
    i, j = jnp.meshgrid(k, k)
    C = jnp.power(-1.0, i + j)

    # Adjust first and last rows
    C = C.at[0, :].multiply(2.0)
    C = C.at[N - 1, :].multiply(2.0)

    # Adjust first and last columns
    C = C.at[:, 0].multiply(0.5)
    C = C.at[:, N - 1].multiply(0.5)

    # Z contains entries 1/(x_i-x_j) with zeros on diagonal
    Z = 1.0 / DX
    Z = Z.at[jnp.diag_indices(N)].set(0.0)

    # Compute first differentiation matrix
    i, j = jnp.meshgrid(jnp.arange(N), jnp.arange(N))
    mask = i != j
    D = jnp.zeros((N, N))
    D = jnp.where(mask, C * Z, D)

    # Diagonal elements (negative row sum)
    diag = -jnp.sum(D, axis=1)
    D = D.at[jnp.diag_indices(N)].set(diag)

    return D


def cheb_D2(N):
    """Second-order Chebyshev differentiation matrix - direct implementation."""
    # Identity matrix
    I = jnp.eye(N)

    # Indices for flipping trick
    n1 = N // 2
    n2 = (N + 1) // 2

    # Compute theta vector
    k = jnp.arange(N)
    th = k * jnp.pi / (N - 1)

    # Create matrices of theta values
    T1, T2 = jnp.meshgrid(th / 2, th / 2)

    # Compute differences using trigonometric identity
    DX = 2 * jnp.sin(T1 + T2) * jnp.sin(T1 - T2)

    # Apply the flipping trick
    DX_top = DX[:n1, :]
    DX_bottom = -jnp.flip(jnp.flip(DX[:n2, :], axis=0), axis=1)
    DX = jnp.vstack([DX_top, DX_bottom])

    # Put 1's on the main diagonal
    DX = DX.at[jnp.diag_indices(N)].set(1.0)

    # Create the C matrix (c_i/c_j)
    i, j = jnp.meshgrid(k, k)
    C = jnp.power(-1.0, i + j)

    # Adjust first and last rows
    C = C.at[0, :].multiply(2.0)
    C = C.at[N - 1, :].multiply(2.0)

    # Adjust first and last columns
    C = C.at[:, 0].multiply(0.5)
    C = C.at[:, N - 1].multiply(0.5)

    # Z contains entries 1/(x_i-x_j) with zeros on diagonal
    Z = 1.0 / DX
    Z = Z.at[jnp.diag_indices(N)].set(0.0)

    # First initialize D as identity (ell = 0 case)
    D = I.copy()

    # For second derivative (ell = 2):
    # First do one iteration (ell = 1)
    D_diag_1 = jnp.diag(D)
    D1 = jnp.zeros((N, N))
    mask = i != j
    D1 = jnp.where(mask, 1 * Z * (C * jnp.tile(D_diag_1, (N, 1)).T - D), D1)
    D1_diag = -jnp.sum(D1, axis=1)
    D1 = D1.at[jnp.diag_indices(N)].set(D1_diag)

    # Now do second iteration (ell = 2)
    D_diag_2 = jnp.diag(D1)
    D2 = jnp.zeros((N, N))
    D2 = jnp.where(mask, 2 * Z * (C * jnp.tile(D_diag_2, (N, 1)).T - D1), D2)
    D2_diag = -jnp.sum(D2, axis=1)
    D2 = D2.at[jnp.diag_indices(N)].set(D2_diag)

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
