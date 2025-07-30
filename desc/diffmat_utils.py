#!/usr/bin/env python3
"""
Differentiation‑matrix utilities for spectral methods in **DESC**.

Adds various functions to create differentiation matrices.
"""

from functools import partial

from desc.backend import jax, jit, jnp

########################################################################
# ---------------------- CHEBYSHEV MATRICES -------------------------- #
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


########################################################################
# ---------------------- LEGENDRE MATRICES --------------------------- #
########################################################################


@partial(jit, static_argnums=0)
def legendre_lobatto_nodes(N: int) -> jnp.ndarray:
    """Return *N+1* Legendre–Gauss–Lobatto abscissae in ascending order.

    Interior points are the zeros of **P_{N-1}'(x)**.  These can be obtained
    as the eigenvalues of a symmetric tridiagonal matrix with off‑diagonal
    elements (Golub-Welsch)
        βₖ = √[ k (k+2) / ((2k+1)(2k+3)) ],   k = 1,…,N−3.
    """
    if N < 2:
        raise ValueError("N must be ≥ 2 for LGL nodes")

    k = jnp.arange(1, N - 2 + 1, dtype=jnp.float64)  # 1 … N−3  (empty if N≤3)
    beta = jnp.sqrt(k * (k + 2) / ((2 * k + 1) * (2 * k + 3)))  # new formula
    T = jnp.diag(beta, 1) + jnp.diag(beta, -1)  # (N−2)×(N−2)
    eigs = jnp.linalg.eigh(T)[0] if (N > 3) else jnp.array([])  # no eigs if <4

    return jnp.concatenate((-jnp.ones((1,)), jnp.sort(eigs), jnp.ones((1,))))


@partial(jit, static_argnums=0)
def legendre_lobatto_weights(N: int) -> jnp.ndarray:
    """Return the *N+1* Legendre–Gauss–Lobatto **quadrature** weights."""
    if N < 2:
        raise ValueError("N must be ≥ 2 for LGL weights")

    x = legendre_lobatto_nodes(N)
    n = x.size - 1  # degree of polynomial in formula
    pref = 2.0 / (n * (n + 1))

    # evaluate P_{n}(x) via a three‑term recurrence (JAX‑friendly)
    def P_n(x):
        Pkm1 = jnp.ones_like(x)  # P₀
        if n == 0:
            return Pkm1
        Pk = x  # P₁

        def body(k, state):
            Pkm1, Pk = state
            Pkp1 = ((2 * k + 1) * x * Pk - k * Pkm1) / (k + 1)
            return (Pk, Pkp1)

        _, Pk = jax.lax.fori_loop(1, n, body, (Pkm1, Pk))
        return Pk

    Pn = P_n(x)
    w = pref / (Pn**2)
    w = w.at[0].set(pref)
    w = w.at[-1].set(pref)
    return w


@jit
def _barycentric_weights(x: jnp.ndarray) -> jnp.ndarray:
    """λᵢ = 1 / ∏_{j≠i}(xᵢ − xⱼ) — used for differentiation only."""
    diff = x[:, None] - x[None, :]
    diff_eye = diff + jnp.eye(x.size)  # mask diagonal → 1 so prod unaffected
    return 1.0 / jnp.prod(diff_eye, axis=1)


@partial(jit, static_argnums=0)
def legendre_D1(N: int) -> jnp.ndarray:
    """Return the N+1×N+1 first‑order differentiation matrix on LGL nodes."""
    x = legendre_lobatto_nodes(N)
    n = x.size
    lam = _barycentric_weights(x)
    diff = x[:, None] - x[None, :]

    D = (lam[None, :] / lam[:, None]) / diff  # off‑diagonals
    D = D.at[jnp.diag_indices(n)].set(0.0)  # clear temporary diag
    D = D.at[jnp.diag_indices(n)].set(-jnp.sum(D, axis=1))  # enforce row‑sum 0

    return D


def create_lele_D1_6_matrix(n, h=1.0):
    """
    Create 6th-order Lele compact approximation matrix for first derivative.

    Parameters
    ----------
    n : int
        Size of the matrix (number of grid points)
    h : float
        Grid spacing

    Returns
    -------
    A : jax.numpy.ndarray
        LHS matrix (for coefficients of derivatives)
    B : jax.numpy.ndarray
        RHS matrix (for coefficients of function values)
    """
    # Create LHS matrix A (tridiagonal)
    A = jnp.zeros((n, n))

    # Diagonal elements
    A = A.at[jnp.arange(n), jnp.arange(n)].set(jnp.ones(n))

    # Boundary and near-boundary treatment
    A = A.at[0, 1].set(2.0)
    A = A.at[n - 1, n - 2].set(2.0)

    # Second point from boundary
    A = A.at[1, 0].set(0.25)
    A = A.at[1, 2].set(0.25)
    A = A.at[n - 2, n - 1].set(0.25)
    A = A.at[n - 2, n - 3].set(0.25)

    # Interior points
    interior = jnp.arange(2, n - 2)
    A = A.at[interior, interior - 1].set(jnp.ones(n - 4) * (1.0 / 3.0))
    A = A.at[interior, interior + 1].set(jnp.ones(n - 4) * (1.0 / 3.0))

    # Create RHS matrix B (maps function values to RHS vector)
    B = jnp.zeros((n, n))

    # Boundary points
    B = B.at[0, 0].set(-5.0 / (2.0 * h))
    B = B.at[0, 1].set(4.0 / (2.0 * h))
    B = B.at[0, 2].set(1.0 / (2.0 * h))

    B = B.at[n - 1, n - 1].set(5.0 / (2.0 * h))
    B = B.at[n - 1, n - 2].set(-4.0 / (2.0 * h))
    B = B.at[n - 1, n - 3].set(-1.0 / (2.0 * h))

    # Second point from boundary
    B = B.at[1, 0].set(-3.0 / (4.0 * h))
    B = B.at[1, 2].set(3.0 / (4.0 * h))

    B = B.at[n - 2, n - 3].set(-3.0 / (4.0 * h))
    B = B.at[n - 2, n - 1].set(3.0 / (4.0 * h))

    # Interior points
    interior = jnp.arange(2, n - 2)
    B = B.at[interior, interior + 1].set(14.0 / (18.0 * h))
    B = B.at[interior, interior - 1].set(-14.0 / (18.0 * h))
    B = B.at[interior, interior + 2].set(1.0 / (36.0 * h))
    B = B.at[interior, interior - 2].set(-1.0 / (36.0 * h))

    return A, B


def apply_compact_derivative(A, B, f):
    """
    Apply compact finite difference approximation to function values.

    Parameters
    ----------
    A : jax.numpy.ndarray
        LHS matrix (for coefficients of derivatives)
    B : jax.numpy.ndarray
        RHS matrix (for coefficients of function values)
    f : jax.numpy.ndarray
        Function values at grid points

    Returns
    -------
    df : jax.numpy.ndarray
        Derivative approximation
    """
    b = B @ f
    df = jnp.linalg.solve(A, b)
    return df


@partial(jit, static_argnums=(0,))
def D1_FD_4(n: int, *, domain: tuple[float, float] = (0.0, 1.0)):
    """
    CD‑4 interior + 4‑point 4th‑order one‑sided closures (Carpenter‑Gottlieb).
    Satisfies   Dᵀ W + W D = diag([-1,0,…,0,1])   with diagonal norm W.

    Parameters
    ----------
    n : int      (n ≥ 6)
    domain : (a,b), optional
        Physical interval.  Default = (-1,1).

    Returns
    -------
    D : (n,n) jax.Array     derivative matrix
    W : (n,)  jax.Array     diagonal SBP weights
    """
    if n < 6:
        raise ValueError("n must be at least 6 for 4‑point closures")

    a, b = domain
    h = (b - a) / (n - 1)

    # diagonal weights  (CD 4‑4)
    W = jnp.full((n,), h)
    W = W.at[jnp.array([0, 1, n - 2, n - 1])].set(jnp.array([31, 49, 49, 31]) * h / 72)

    D = jnp.zeros((n, n))

    # ---------- interior rows: centred 5‑point ----------------------
    rows_int = jnp.arange(2, n - 2)[:, None]  # shape (m,1)
    cols_int = rows_int + jnp.arange(-2, 3)  # shape (m,5)
    coeff_int = jnp.array([1, -8, 0, 8, -1]) / (12.0 * h)  # (5,)
    D = D.at[rows_int, cols_int].set(
        jnp.broadcast_to(coeff_int, rows_int.shape[:-1] + (5,))
    )

    # ---------- left boundary closures (rows 0 & 1) ----------------
    left_rows = jnp.array([0, 1])[:, None]  # (2,1)
    left_cols = jnp.arange(5)[None, :]  # (1,5)
    left_coef = jnp.array(
        [
            [-25, 48, -36, 16, -3],
            [-3, -10, 18, -6, 1],
        ]
    ) / (12.0 * h)
    D = D.at[left_rows, left_cols].set(left_coef)

    # ---------- right boundary closures  (rows n-2 & n-1) ----------
    right_rows = jnp.array([n - 2, n - 1])[:, None]  # (2,1)
    right_cols = jnp.arange(n - 5, n)[None, :]  # (1,5)
    right_coef = -left_coef[:, ::-1]  # antisymmetric
    D = D.at[right_rows, right_cols].set(right_coef)

    return D, W
