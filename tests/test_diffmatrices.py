#!/usr/bin/env python3
"""
Test file for higher-order mixed derivatives in a single field period.

Using tensor product approach in 3D:
- Chebyshev methods in x dimension
- Fourier methods in y dimension
- Fourier methods in z dimension
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

from desc.backend import jax, jnp, vmap
from desc.diffmat_utils import (  # cheb_D1,; cheb_D2,; chebpts_lobatto,
    finite_difference_diffmat,
    fourier_diffmat,
    fourier_pts,
    legendre_diffmat,
)
from desc.integrals.quad_utils import automorphism_staircase1, leggauss_lob

NFP = 5


# === Helper Functions ===
def _eval_3D(f, x, y, z, NFP):
    """Evaluate function f at grid points (x[i], y[j], z[k])."""
    return vmap(
        lambda z_val: vmap(
            lambda y_val: vmap(lambda x_val: f(x_val, y_val, z_val, NFP))(x)
        )(y)
    )(z)


# --no-verify a = 0.015  # [0.0, 0.1]
# --no-verify #0.5 * (x + 1) + a * jnp.sin(jnp.pi * (x + 1))
# --no-verify 0.5 * (x + 1) - 0.3 * (1 - x**16) * jnp.cos(0.5 * jnp.pi * (x - 1.8))
# --no-verify map_term1 = 0.5 * (x + 1)
# --no-verify exp_term = jnp.exp(0.3 * (x - 1) ** 2)
# --no-verify map_term2 = 0.3 * (1 - x**16) *exp_term * jnp.cos(0.5*jnp.pi*(x-1.8))
# --no-verify return map_term1 - map_term2


def _tensor_product_derivative_3D(  # noqa: C901
    nx, ny, nz, dx_order, dy_order, dz_order, NFP
):
    """
    Create a tensor product differentiation matrix in 3D.

    Parameters
    ----------
    nx : int
        Number of Chebyshev points in x dimension
    ny : int
        Number of Fourier points in y dimension
    nz : int
        Number of Fourier points in z dimension
    dx_order : int
        Order of x derivative (Chebyshev), 0 <= dx_order <= 2
    dy_order : int
        Order of y derivative (Fourier), 0 <= dy_order <= 2
    dz_order : int
        Order of z derivative (Fourier), 0 <= dz_order <= 2

    Returns
    -------
    D : array
        Tensor product differentiation matrix
    x : array
        x collocation points
    y : array
        y collocation points
    z : array
        z collocation points
    """
    # Generate collocation points
    x_cheb, w = leggauss_lob(nx)
    y_four = fourier_pts(ny)
    z_four = fourier_pts(nz)

    x = automorphism_staircase1(x_cheb, x_0=0.8, m_1=2.0, m_2=3.0)
    dx_f = jax.vmap(
        lambda x_val: jax.grad(automorphism_staircase1, argnums=0)(
            x_val, x_0=0.8, m_1=2.0, m_2=3.0
        )
    )
    dxx_f = jax.vmap(
        lambda x_val: jax.grad(jax.grad(automorphism_staircase1, argnums=0), argnums=0)(
            x_val, x_0=0.8, m_1=2.0, m_2=3.0
        )
    )

    scale_x1 = 1 / dx_f(x_cheb)[:, None]
    scale_x2 = dxx_f(x_cheb)[:, None] / dx_f(x_cheb)[:, None]

    y = y_four  # Already in [0, 2π]
    z = z_four / NFP  # Already in [0, 2π/NFP]

    # Get x differentiation matrix (Chebyshev)
    if dx_order == 0:
        Dx = jnp.eye(nx)
    elif dx_order == 1:
        D, _ = legendre_diffmat(nx)
        Dx = D * scale_x1
    elif dx_order == 2:
        D, _ = legendre_diffmat(nx)
        Dx = (D @ D - D * scale_x2) * scale_x1**2

    # Get y differentiation matrix (Fourier)
    if dy_order == 0:
        Dy = jnp.eye(ny)
    elif dy_order == 1:
        Dy, _ = fourier_diffmat(ny)
    elif dy_order == 2:
        D, _ = fourier_diffmat(ny)
        Dy = D @ D

    # Get z differentiation matrix (Fourier)
    if dz_order == 0:
        Dz = jnp.eye(nz)
    elif dz_order == 1:
        D, _ = fourier_diffmat(nz)
        Dz = D * NFP
    elif dz_order == 2:
        D, _ = fourier_diffmat(nz)
        Dz = (D @ D) * NFP**2

    # Create identity matrices for tensor product
    Ix = jnp.eye(nx)
    Iy = jnp.eye(ny)
    Iz = jnp.eye(nz)

    # Tensor product approach using Kronecker products
    # Following the approach in the 2D code

    # Construct the appropriate matrix based on the derivative order
    if dx_order > 0 and dy_order > 0 and dz_order > 0:
        # Full 3D mixed derivative (x, y, z)
        D = jnp.kron(Dz, jnp.kron(Dy, Dx))

    elif dx_order > 0 and dy_order > 0:
        # Mixed derivative (x, y)
        D = jnp.kron(Dx, Dy)

    elif dx_order > 0 and dz_order > 0:
        # Mixed derivative (x, z)
        D = jnp.kron(Dx, Dz)

    elif dy_order > 0 and dz_order > 0:
        # Mixed derivative (y, z)
        D = jnp.kron(Dz, Dy)

    elif dx_order > 0:
        # Pure x derivative
        D = Dx

    elif dy_order > 0:
        # Pure y derivative
        D = Dy

    elif dz_order > 0:
        # Pure z derivative
        D = Dz
    else:
        # Identity (no derivative)
        D = jnp.kron(Iz, jnp.kron(Iy, Ix))

    # Clean up small values
    D = jnp.where(jnp.abs(D) < 1e-12, 0.0, D)

    return D, x, y, z


# === Test Function ===
def _test_function(x, y, z, NFP):
    """Test function that is smooth and effectively periodic in y and z."""
    return (
        jnp.exp(-100 * ((x - 0.8) ** 2))
        * jnp.sin(3 * x * 2 * jnp.pi)
        * (jnp.sin(4 * y) + jnp.cos(3 * y))
        * jnp.cos(NFP * z)
    )


# === Analytical Derivatives via Automatic Differentiation ===

# First derivatives
dx_f = jax.grad(_test_function, argnums=0)
dy_f = jax.grad(_test_function, argnums=1)
dz_f = jax.grad(_test_function, argnums=2)

# Second derivatives
dxx_f = jax.grad(dx_f, argnums=0)
dxy_f = jax.grad(dx_f, argnums=1)
dxz_f = jax.grad(dx_f, argnums=2)
dzx_f = dxz_f
dyy_f = jax.grad(dy_f, argnums=1)
dyz_f = jax.grad(dy_f, argnums=2)
dzz_f = jax.grad(dz_f, argnums=2)


# --- Test Cases Configuration ---
# List of (dx_order, dy_order, analytic_fn, tolerance)
test_cases = [
    # Pure x derivatives
    (1, 0, 0, dx_f, 4e-3, NFP),
    (2, 0, 0, dxx_f, 6e-3, NFP),
    # Pure y derivatives
    (0, 1, 0, dy_f, 1e-7, NFP),
    (0, 2, 0, dyy_f, 1e-5, NFP),
    # Pure z derivatives
    (0, 0, 1, dz_f, 1e-7, NFP),
    (0, 0, 2, dzz_f, 1e-5, NFP),
    # Mixed derivatives
    (1, 1, 0, dxy_f, 2e-3, NFP),
    (0, 1, 1, dyz_f, 2e-3, NFP),
    (1, 0, 1, dzx_f, 2e-3, NFP),
]

# a module‐level list to stash all of our (dx, dy, dz, n, error) tuples
collected_errors = []


@pytest.mark.regression
@pytest.mark.parametrize("n", [48])
@pytest.mark.parametrize("dx_order,dy_order,dz_order,analytic_fn,tol,NFP", test_cases)
def test_tensor_mixed_derivative(
    dx_order, dy_order, dz_order, analytic_fn, tol, NFP, n
):
    """Validate 3D tensor-product differentiation against analytical JAX derivatives.

    We calculate mixed and non-mixed derivatives w.r.t rho, theta, zeta.
    """
    # Grid resolution
    nx, ny, nz = n, n, n

    D, x, y, z = _tensor_product_derivative_3D(
        nx, ny, nz, dx_order, dy_order, dz_order, NFP=NFP
    )

    # Create meshgrid for evaluation
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    f_vals0 = _test_function(X, Y, Z, NFP)

    if dx_order > 0 and dy_order > 0 and dz_order > 0:
        f_vals = jnp.transpose(f_vals0, (2, 1, 0))
        f_flat = f_vals.flatten()

    elif dx_order > 0 and dy_order > 0:
        f_flat = jnp.reshape(f_vals0, (nx * ny, nz))
        df_flat = D @ f_flat
        df_grid = jnp.reshape(df_flat, (nx, ny, nz))

    elif dx_order > 0 and dz_order > 0:
        f_vals = jnp.transpose(f_vals0, (0, 2, 1))
        f_flat = jnp.reshape(f_vals, (nx * nz, ny))
        df_flat = D @ f_flat
        df_grid = jnp.reshape(df_flat, (nx, nz, ny))
        df_grid = jnp.transpose(df_grid, (0, 2, 1))

    elif dy_order > 0 and dz_order > 0:
        f_vals = jnp.transpose(f_vals0, (2, 1, 0))
        f_flat = jnp.reshape(f_vals, (nz * ny, nx))
        df_flat = D @ f_flat
        df_grid = jnp.reshape(df_flat, (nz, ny, nx))
        df_grid = jnp.transpose(df_grid, (2, 1, 0))

    elif dx_order > 0:
        f_flat = jnp.reshape(f_vals0, (nx, ny * nz))
        df_flat = D @ f_flat
        df_grid = jnp.reshape(df_flat, (nx, ny, nz))

    elif dy_order > 0:
        f_vals = jnp.transpose(f_vals0, (1, 0, 2))
        f_flat = jnp.reshape(f_vals, (ny, nx * nz))
        df_flat = D @ f_flat
        df_grid = jnp.reshape(df_flat, (ny, nx, nz))
        df_grid = jnp.transpose(df_grid, (1, 0, 2))

    elif dz_order > 0:
        f_vals = jnp.transpose(f_vals0, (2, 0, 1))
        f_flat = jnp.reshape(f_vals, (nz, nx * ny))
        df_flat = D @ f_flat
        df_grid = jnp.reshape(df_flat, (nz, nx, ny))
        df_grid = jnp.transpose(df_grid, (1, 2, 0))
    else:
        # Identity (no derivative)
        f_flat = f_vals0

    # Compute exact derivative
    df_exact = _eval_3D(analytic_fn, x, y, z, NFP).transpose(2, 1, 0)

    error = jnp.max(jnp.abs(df_grid - df_exact))

    # record it (pytest will still assert below)
    collected_errors.append((dx_order, dy_order, dz_order, n, error))

    assert (
        error < tol
    ), f"dx={dx_order}, dy={dy_order}, dz={dz_order}: \
        error {error:.2e} exceeds tol {tol}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "N, alpha, x0, tol",
    [
        (48, 100.0, 0.7, 8.0e-2),
        (96, 100.0, 0.7, 9.0e-3),
        (192, 100.0, 0.7, 9.0e-4),
    ],
)
def test_finite_difference_diffmat(N, alpha, x0, tol):
    """
    Test the accuracy of the finite differentation matrix.

    The test function is an oscillating gaussian.
    """
    a, b = 0.0, 1.0
    x = jnp.linspace(a, b, N)
    h = (b - a) / (N - 1)

    D, _ = finite_difference_diffmat(N, h)

    # oscillating Gaussian
    f = jnp.exp(-alpha * (x - x0) ** 2) * jnp.cos(4 * jnp.pi * (x - 0.5))

    df_true = jax.vmap(
        jax.grad(
            lambda x_val: jnp.exp(-alpha * (x_val - x0) ** 2)
            * jnp.cos(4 * jnp.pi * (x_val - 0.5))
        )
    )(x)

    df_num = D @ f

    # ignore lower-order boundary closures
    err = jnp.max(jnp.abs(df_num - df_true))

    assert float(err) < tol, f"max|err|={float(err):.2e} (N={N}, alpha={alpha})"


@pytest.mark.unit
def test_summation_by_parts():
    """
    Tests the summation by parts (SBP) property of differentiation matrices.

    SBP is a discretized version of integration by parts. This is a powerful
    property that is useful when a system needs to satisy conservation
    property.

    (W @ D) + (W @ D).T = B

    where B = diag(-1, 0, ...., 0, 1)
    """
    a, b = 0.0, 1.0
    N = 100
    h = (b - a) / (N - 1)

    D0, W0 = finite_difference_diffmat(N, h)
    D1, W1 = legendre_diffmat(N)
    D2, W2 = fourier_diffmat(N)

    B = jnp.zeros_like(D0)
    B = B.at[0, 0].set(-1)
    B = B.at[N - 1, N - 1].set(1)

    np.testing.assert_allclose(W0 @ D0 + (W0 @ D0).T, B, atol=1e-15)
    np.testing.assert_allclose(W1 @ D1 + (W1 @ D1).T, B, atol=5e-13)
    np.testing.assert_allclose(W2 @ D2 + (W2 @ D2).T, 0, atol=1e-16)


# To view the plots, run pytest -s
def teardown_module(module=None):
    """
    Optional convergence plotting routine.

    To create the convergence plot, set the envinronment variable
    `PLOT_CONVERGENCE=1` and rerun the test.
    """
    if not os.environ.get("PLOT_CONVERGENCE"):
        return
    # run extra resolutions (skip 48 to avoid duplicates from the base test)
    extra_ns = [16, 24, 32, 64, 96, 128]

    for dx_order, dy_order, dz_order, analytic_fn, tol, NFP in test_cases:
        for n in extra_ns:
            try:
                test_tensor_mixed_derivative(
                    dx_order=dx_order,
                    dy_order=dy_order,
                    dz_order=dz_order,
                    analytic_fn=analytic_fn,
                    tol=tol,
                    NFP=NFP,
                    n=n,
                )
            except AssertionError as e:
                print(
                    f"[teardown] FAIL n={n} (dx,dy,dz)"
                    + f"=({dx_order},{dy_order},{dz_order}):{e}"
                )

            # print the last appended entry (so you see errors as they happen)
            try:
                dx, dy, dz, nn, err = collected_errors[-1]
                if (dx, dy, dz, nn) == (dx_order, dy_order, dz_order, n):
                    print(f"[teardown] dx={dx}, dy={dy}, dz={dz}, n={nn} -> err={err}")
            except Exception:
                pass

    # make plot from collected_errors
    if "collected_errors" not in globals() or not collected_errors:
        print("[teardown] No collected_errors; nothing to plot.")
        return

    # group by derivative orders
    series = {}
    for dx, dy, dz, n, err in collected_errors:
        series.setdefault((dx, dy, dz), []).append((n, float(err)))

    plt.figure(figsize=(7, 5))
    for (dx, dy, dz), pts in series.items():
        pts = sorted(pts, key=lambda t: t[0])  # sort by n
        ns = np.array([t[0] for t in pts], float)
        errs = np.array([t[1] for t in pts], float)

        # slope from the last few valid points (stable, no NaNs)
        mask = np.isfinite(errs) & (errs > 0)
        if mask.sum() >= 2:
            k = min(3, mask.sum())
            slope = np.polyfit(np.log(ns[mask][-k:]), np.log(errs[mask][-k:]), 1)[0]
            label = f"dx={dx}, dy={dy}, dz={dz} (slope={slope:.2f})"
        else:
            label = f"dx={dx}, dy={dy}, dz={dz}"

        plt.loglog(ns, errs, "o-", label=label)

    xs = np.unique([n for _, _, _, n, _ in collected_errors]).astype(float)
    if xs.size:
        plt.loglog(xs, xs**-4, "k--", label="O(n⁻⁴)")
        plt.loglog(xs, xs**-8, "k-.", label="O(n⁻⁸)")

    plt.xlabel("n (grid points per dim)")
    plt.ylabel("max error")
    plt.title("Convergence from collected_errors")
    plt.grid(True, which="both", ls=":")
    plt.legend(fontsize=9)

    out_path = os.path.join(os.getcwd(), "diffmatrix_convergence_plots.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[teardown] Saved plot -> {out_path}")
