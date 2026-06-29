#!/usr/bin/env python3
"""
Differentiation‑matrix utilities for spectral methods in **DESC**.

=================================================================

This module provides vectorized, JAX‑friendly helpers for constructing first‑ and
second‑order differentiation matrices using either Chebyshev–Lobatto or Fourier
collocation points.  All routines are **pure** and **stateless** – they rely only
on `jax.numpy` and therefore can be freely composed, JIT‑compiled, or parallelised
with `jax.vmap` / `jax.pmap` inside larger optimisation loops.

The implementations follow the formulas in
  **Trefethen, L. N. (2000). *Spectral Methods in MATLAB*. SIAM** and
  **Canuto et al. (2006). *Spectral Methods – Fundamentals in Single Domains*.**

"""
from functools import partial

import numpy as np

from desc.backend import jit, jnp
from desc.integrals.quad_utils import leggauss_lob
from desc.io import IOAble


class DiffMat(IOAble):
    """Differentiation and quadrature matrices for a tensor-product grid.

    At least one differentiation/quadrature matrix pair must be supplied. The
    matrices must be built for the nodes on which they will be used; in particular,
    ``D_zeta`` and ``W_zeta`` must match the zeta nodes in the grid passed to
    :meth:`Equilibrium.compute <desc.equilibrium.Equilibrium.compute>`.
    Use :meth:`from_zeta_grid` to construct a compatible fourth-order
    finite-difference pair for a uniform zeta grid.

    Parameters
    ----------
    D_rho, D_theta, D_zeta : array-like, optional
        Differentiation matrices for each coordinate.
    W_rho, W_theta, W_zeta : array-like, optional
        Quadrature matrices corresponding to each differentiation matrix.
    """

    _io_attrs_ = ["D_rho", "D_theta", "D_zeta", "W_rho", "W_theta", "W_zeta"]
    _static_attrs = ["_token"]

    def __init__(
        self,
        *,
        D_rho=None,
        D_theta=None,
        D_zeta=None,
        W_rho=None,
        W_theta=None,
        W_zeta=None,
    ):
        self.D_rho = None if D_rho is None else jnp.asarray(D_rho)
        self.D_theta = None if D_theta is None else jnp.asarray(D_theta)
        self.D_zeta = None if D_zeta is None else jnp.asarray(D_zeta)
        self.W_rho = None if W_rho is None else jnp.asarray(W_rho)
        self.W_theta = None if W_theta is None else jnp.asarray(W_theta)
        self.W_zeta = None if W_zeta is None else jnp.asarray(W_zeta)
        self._set_up()

    def _set_up(self):
        """Validate the matrices and create JAX's static structure token."""
        matrix_pairs = (
            ("rho", self.D_rho, self.W_rho),
            ("theta", self.D_theta, self.W_theta),
            ("zeta", self.D_zeta, self.W_zeta),
        )
        if all(D is None and W is None for _, D, W in matrix_pairs):
            raise ValueError(
                "DiffMat requires at least one differentiation/quadrature matrix "
                "pair. Omit diffmat to use the default finite-difference solver."
            )
        for coordinate, D, W in matrix_pairs:
            if (D is None) != (W is None):
                raise ValueError(
                    f"D_{coordinate} and W_{coordinate} must be provided together."
                )
            if D is not None and (
                D.ndim != 2
                or W.ndim != 2
                or D.shape[0] != D.shape[1]
                or W.shape != D.shape
            ):
                raise ValueError(
                    f"D_{coordinate} and W_{coordinate} must be square matrices "
                    "with matching shapes."
                )

        # Matrix values are dynamic PyTree leaves. The token describes only their
        # static structure, so equal-shaped matrices share compiled code safely.
        self._token = (
            "DiffMat",
            (None if self.D_rho is None else getattr(self.D_rho, "shape", None)),
            (None if self.D_theta is None else getattr(self.D_theta, "shape", None)),
            (None if self.D_zeta is None else getattr(self.D_zeta, "shape", None)),
            (None if self.W_rho is None else getattr(self.W_rho, "shape", None)),
            (None if self.W_theta is None else getattr(self.W_theta, "shape", None)),
            (None if self.W_zeta is None else getattr(self.W_zeta, "shape", None)),
        )

    @classmethod
    def from_zeta_grid(cls, zeta):
        """Create a ``DiffMat`` for a uniform zeta grid.

        This convenience constructor uses the fourth-order summation-by-parts
        finite-difference stencil returned by :func:`finite_difference_diffmat`.
        Pass the resulting object with the same ``zeta`` nodes:

        .. code-block:: python

            zeta = jnp.linspace(-3 * jnp.pi, 3 * jnp.pi, 600)
            grid = Grid.create_meshgrid([rho, alpha, zeta], coordinates="raz")
            diffmat = DiffMat.from_zeta_grid(zeta)
            data = eq.compute("ideal ballooning lambda", grid=grid, diffmat=diffmat)

        Parameters
        ----------
        zeta : array-like
            One-dimensional, uniformly spaced zeta nodes. At least 8 nodes are
            required by the boundary stencil.
        """
        zeta = jnp.asarray(zeta)
        if zeta.ndim != 1:
            raise ValueError("zeta must be one-dimensional.")
        if zeta.size < 8:
            raise ValueError("At least 8 zeta nodes are required.")
        spacing = np.diff(np.asarray(zeta))
        if not np.allclose(spacing, spacing[0]):
            raise ValueError("zeta nodes must be uniformly spaced.")
        D_zeta, W_zeta = finite_difference_diffmat(
            zeta.size, spacing[0], dtype=zeta.dtype
        )
        return cls(D_zeta=D_zeta, W_zeta=W_zeta)

    def __hash__(self):
        """Hash the static matrix structure."""
        return hash(self._token)

    def __eq__(self, other):
        """Compare the static matrix structure."""
        return isinstance(other, DiffMat) and self._token == other._token


########################################################################
# ----------------------- LEGENDRE MATRICES -------------------------- #
########################################################################


@jit
def _barycentric_weights(x: jnp.ndarray) -> jnp.ndarray:
    """λᵢ = 1 / ∏_{j≠i}(xᵢ − xⱼ) — used for differentiation only."""
    diff = x[:, None] - x[None, :]
    diff_eye = diff + jnp.eye(x.size)
    return 1.0 / jnp.prod(diff_eye, axis=1)


@partial(jit, static_argnums=0)
def legendre_diffmat(N: int) -> jnp.ndarray:
    """Return the N+1×N+1 first‑order differentiation matrix on LGL nodes."""
    x, w = leggauss_lob(N)
    lam = _barycentric_weights(x)
    diff = x[:, None] - x[None, :]

    D = (lam[None, :] / lam[:, None]) / diff  # off‑diagonals
    D = D.at[jnp.diag_indices(N)].set(0.0)  # clear temporary diag
    D = D.at[jnp.diag_indices(N)].set(-jnp.sum(D, axis=1))  # enforce row‑sum 0

    W = jnp.zeros((N, N))
    W = W.at[jnp.diag_indices(N)].set(w)

    return D, W


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
    return jnp.linspace(domain[0], domain[1], n, endpoint=False)


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

    W = jnp.zeros((n, n))
    W = W.at[jnp.diag_indices(n)].set(2 * jnp.pi / n)

    return D, W


########################################################################
# ------------------- FINITE-DIFFERENCE MATRIX ----------------------- #
########################################################################


def finite_difference_diffmat(N, h, dtype=jnp.float64):
    """
    Diagonal‑norm SBP first‑derivative matrix.

    Fourth‑order / second‑order finite-difference matrix on a
    uniform grid of spacing h.

    Returns
    -------
    D : (N, N) jax.numpy.ndarray
    W : (N, N) jax.numpy.ndarray
    """
    D = jnp.zeros((N, N), dtype)
    H = jnp.ones((N,), dtype)
    W = jnp.zeros((N, N), dtype)

    # ---- interior rows (indices 4 … N‑5) 5‑point central stencil
    rows = jnp.arange(4, N - 4, dtype=jnp.int32)  # shape (Ni,)
    offsets = jnp.array([-2, -1, 0, 1, 2], dtype=jnp.int32)
    stencil_coeffs = jnp.array([1, -8, 0, 8, -1], dtype) / 12.0  # shape (5,)

    row_idx = jnp.repeat(rows, 5)  # (Ni*5,)
    col_idx = (rows[:, None] + offsets).reshape(-1)  # (Ni*5,)
    vals = jnp.tile(stencil_coeffs, rows.size)  # (Ni*5,)

    D = D.at[row_idx, col_idx].set(vals)

    # ---- forward boundary closures (Carpenter–Nordstrom)
    f0 = jnp.array([-24 / 17, 59 / 34, -4 / 17, -3 / 34], dtype)
    f1 = jnp.array([-1 / 2, 0.0, 1 / 2], dtype)
    f2 = jnp.array([4 / 43, -59 / 86, 0.0, 59 / 86, -4 / 43], dtype)
    f3 = jnp.array([3 / 98, 0.0, -59 / 98, 0.0, 32 / 49, -4 / 49], dtype)

    D = (
        D.at[0, :4]
        .set(f0)
        .at[1, :3]
        .set(f1)
        .at[2, :5]
        .set(f2)
        .at[3, :6]
        .set(f3)
        # lower boundary rows by SBP antisymmetry  D[N‑1‑i,N‑1‑j] = −D[i,j]
        .at[-1, -4:]
        .set(-f0[::-1])
        .at[-2, -3:]
        .set(-f1[::-1])
        .at[-3, -5:]
        .set(-f2[::-1])
        .at[-4, -6:]
        .set(-f3[::-1])
    )

    # specialised edge weights
    edge_vals = jnp.array([17 / 48, 59 / 48, 43 / 48, 49 / 48], dtype)
    H = (
        H.at[0]
        .set(edge_vals[0])
        .at[-1]
        .set(edge_vals[0])
        .at[1]
        .set(edge_vals[1])
        .at[-2]
        .set(edge_vals[1])
        .at[2]
        .set(edge_vals[2])
        .at[-3]
        .set(edge_vals[2])
        .at[3]
        .set(edge_vals[3])
        .at[-4]
        .set(edge_vals[3])
    )

    W = W.at[jnp.diag_indices(N)].set(H * h)

    return D / h, W
