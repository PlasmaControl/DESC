"""AGNI finite-n stability tests.

Ground truth for all tests: ``finite-n lambda`` using the dense Cholesky-transformed
eigensolver.

test_lambda31       — finite-n lambda31 (psi_r-rescaled tangential variables) eigenvalue
                      and eigenfunction match finite-n lambda.  lambda31 can be removed
                      from _stability.py once this test passes.

test_lambda3        — finite-n lambda3 (upsilon = xi^theta - xi^zeta variables)
                      eigenvalue and eigenfunction match finite-n lambda.

test_lambda_matfree — finite-n lambda matfree operator satisfies A*v ≈ lambda*v when
                      v is the dense eigenvector.

test_lambda3_matfree — finite-n lambda3 matfree operator satisfies A*v ≈ lambda3*v.

Grid and equilibrium are built once at module level and shared across all tests.
Set AGNI_TEST_RES=N_RHO,N_THETA,N_ZETA (default 12,14,15) to control resolution.
Set AGNI_EQ_PATH to override the equilibrium file.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse.linalg import ArpackNoConvergence

from desc import set_device

if os.environ.get("AGNI_TEST_DEVICE", "cpu").strip().lower() == "gpu":
    set_device("gpu")

from desc.backend import jax, jnp
from desc.compute import _stability as stability
from desc.diffmat_utils import DiffMat, fourier_diffmat, legendre_diffmat
from desc.equilibrium import Equilibrium
from desc.grid import Grid, LinearGrid
from desc.integrals.quad_utils import automorphism_staircase1, leggauss_lob


def _load_old_equilibrium(path):
    """Load AGNI fixture equilibria saved before newer DESC attrs existed."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"\s*The object attribute .* was not loaded from the file\.",
            category=RuntimeWarning,
        )
        return Equilibrium.load(path)


# ---------------------------------------------------------------------------
# Module-level grid / equilibrium — built once, shared across all tests
# ---------------------------------------------------------------------------

_RES = os.environ.get("AGNI_TEST_RES", "12,14,15")
_N_RHO, _N_THETA, _N_ZETA = (int(v) for v in _RES.split(","))

_EQ_PATH = Path(
    os.environ.get(
        "AGNI_EQ_PATH",
        "/pscratch/sd/r/rgaur/AGNI_var/matrix-free/"
        "qh_beta1.5_imin1.02_modprof_221410.h5",
    )
)
_AGNI_SKIP_REASON = f"AGNI equilibrium fixture not found: {_EQ_PATH}"
pytestmark = pytest.mark.skipif(not _EQ_PATH.is_file(), reason=_AGNI_SKIP_REASON)

if _EQ_PATH.is_file():
    _EQ = _load_old_equilibrium(str(_EQ_PATH))

    # Radial quadrature: Gauss-Lobatto nodes mapped through staircase automorphism
    _x_lob, _ = leggauss_lob(_N_RHO)
    _rho = automorphism_staircase1(_x_lob, eps=1e-2, x_0=0.65, m_1=2.0, m_2=3.0)

    # Jacobian of the automorphism — needed for chain-rule correction to D and W
    _d_automorphism = jax.vmap(
        lambda x: jax.grad(automorphism_staircase1, argnums=0)(
            x, eps=1e-2, x_0=0.65, m_1=2.0, m_2=3.0
        )
    )(
        _x_lob
    )  # shape (N_RHO,)

    _d_rho_raw, _w_rho_raw = legendre_diffmat(_N_RHO)
    _d_rho = _d_rho_raw / _d_automorphism[:, None]
    _w_rho = _w_rho_raw * _d_automorphism[:, None]

    _theta = jnp.linspace(0.0, 2.0 * jnp.pi, _N_THETA, endpoint=False)
    _d_theta, _w_theta = fourier_diffmat(_N_THETA)

    _zeta = jnp.linspace(0.0, 2.0 * jnp.pi / _EQ.NFP, _N_ZETA, endpoint=False)
    _d_zeta, _w_zeta = fourier_diffmat(_N_ZETA)
    _d_zeta = _d_zeta * _EQ.NFP  # toroidal periodicity
    _w_zeta = _w_zeta / _EQ.NFP

    _DIFFMAT = DiffMat(
        D_rho=_d_rho,
        W_rho=jnp.diagonal(_w_rho),
        D_theta=_d_theta,
        W_theta=jnp.diagonal(_w_theta),
        D_zeta=_d_zeta,
        W_zeta=jnp.diagonal(_w_zeta),
    )

    # Map PEST coordinates to DESC straight-field-line coordinates
    _grid0 = LinearGrid(rho=_rho, theta=_theta, zeta=_zeta, NFP=1, sym=False)
    _rtz_nodes = _EQ.map_coordinates(
        jnp.reshape(_grid0.meshgrid_reshape(_grid0.nodes, order="rtz"), (-1, 3)),
        inbasis=("rho", "theta_PEST", "zeta"),
        outbasis=("rho", "theta", "zeta"),
        period=(jnp.inf, 2 * jnp.pi, jnp.inf),
        tol=1e-6,  # 1e-12 is overkill for tests and very slow on CPU
        maxiter=20,
    )
    _GRID = Grid(_rtz_nodes)

    _N = _N_RHO * _N_THETA * _N_ZETA  # total grid points
    _N_SHELL = _N_THETA * _N_ZETA  # one theta-zeta shell
    _N_KEEP = 3 * _N - 2 * _N_SHELL  # DOFs after xi^rho Dirichlet BCs

    # Common kwargs for every eq.compute call in this file
    _KW = dict(
        grid=_GRID,
        diffmat=_DIFFMAT,
        incompressible=False,
        gamma=5.0 / 3.0,
        v_guess=np.ones(_N_KEEP),
    )

    # Compute ground-truth results once and share across all tests
    _LAM = _EQ.compute("finite-n lambda", **_KW)
    _LAM3 = _EQ.compute("finite-n lambda3", **_KW)
else:
    _EQ = _GRID = _DIFFMAT = _LAM = _LAM3 = None
    _N = _N_SHELL = _N_KEEP = 0
    _KW = {}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.slow
def test_lambda31():
    """finite-n lambda31 eigenvalue and eigenfunction match finite-n lambda.

    finite-n lambda31 is a variant of lambda3 that rescales the tangential DOFs
    by 1/psi_r before solving, making the matrix assembly more compact.  Because
    the rescaling is pointwise-invertible the eigenvalue is unchanged and the
    eigenfunction (after inverting the rescaling) must match the lambda result.

    Inverse of the rescaling (see _AGNI31 docstring):
        xi^rho   = xr               (not rescaled)
        xi^zeta  = xz * psi_r_norm  (xz is the scaled xi^zeta_s)
        xi^theta = (xv + xz) * psi_r_norm
                   (xv is scaled upsilon_s = (xi^theta-xi^zeta)/psi_r)

    Also checks that xi^rho vanishes at the inner and outer radial boundaries.
    """
    lam = _LAM
    lam31 = _EQ.compute("finite-n lambda31", **_KW)

    lam_val = float(np.asarray(lam["finite-n lambda"])[0])
    lam31_val = float(np.asarray(lam31["finite-n lambda31"])[0])
    reldiff = abs(lam31_val - lam_val) / (abs(lam_val) + 1e-300)
    print(f"\n  lambda   = {lam_val:.6e}")
    print(f"  lambda31 = {lam31_val:.6e}  (reldiff={reldiff:.2e})")

    np.testing.assert_allclose(
        np.asarray(lam31["finite-n lambda31"]),
        np.asarray(lam["finite-n lambda"]),
        rtol=2e-2,
        atol=1e-6,
    )

    # Recover psi_r normalised the same way as inside _AGNI31
    eq_scalars = _EQ.compute(["psi_r", "a"], grid=_GRID)
    a_N = float(eq_scalars["a"])
    B_N = abs(float(_EQ.Psi) / (np.pi * a_N**2))
    psi_r_norm = np.asarray(eq_scalars["psi_r"]) / (a_N**2 * B_N)  # shape (_N,)

    xi31 = np.asarray(lam31["finite-n xi"])
    xi31_rho = xi31[:_N]
    xi31_ups_s = xi31[_N : 2 * _N]  # upsilon / psi_r
    xi31_zeta_s = xi31[2 * _N :]  # xi^zeta / psi_r

    # Invert psi_r rescaling to get physical (rho, theta, zeta) components
    xi31_zeta = xi31_zeta_s * psi_r_norm
    xi31_theta = (xi31_ups_s + xi31_zeta_s) * psi_r_norm
    xi31_rtz = np.concatenate([xi31_rho, xi31_theta, xi31_zeta])

    ref = np.asarray(lam["finite-n xi"]).reshape(-1)
    cand = xi31_rtz.reshape(-1)
    # Align global sign and amplitude before comparing shapes
    phase = np.vdot(cand, ref)
    if abs(phase) > 0:
        cand = cand * phase / abs(phase)
    cand = (np.vdot(cand, ref) / np.vdot(cand, cand)) * cand
    relerr = np.linalg.norm(cand - ref) / (np.linalg.norm(ref) + 1e-300)
    print(f"  eigenfunction relerr (lambda31 vs lambda) = {relerr:.3e}")
    assert relerr < 2e-1, f"eigenfunction relerr={relerr:.3e} exceeds tolerance"

    # xi^rho must be zero at both radial boundaries (Dirichlet BC)
    xi_rho = xi31_rho.reshape(_N_RHO, _N_THETA, _N_ZETA)
    np.testing.assert_allclose(xi_rho[0], 0.0, atol=1e-8)
    np.testing.assert_allclose(xi_rho[-1], 0.0, atol=1e-8)


@pytest.mark.unit
@pytest.mark.slow
def test_lambda3():
    """finite-n lambda3 eigenvalue and eigenfunction match finite-n lambda.

    finite-n lambda3 uses upsilon = xi^theta - xi^zeta as an independent DOF
    instead of xi^theta. The inverse of this substitution is
    xi^theta = upsilon + xi^zeta, after which the eigenvector must agree with the
    lambda result.

    Also checks that xi^rho vanishes at the inner and outer radial boundaries.
    """
    lam = _LAM
    lam3 = _LAM3

    lam_val = float(np.asarray(lam["finite-n lambda"])[0])
    lam3_val = float(np.asarray(lam3["finite-n lambda3"])[0])
    reldiff = abs(lam3_val - lam_val) / (abs(lam_val) + 1e-300)
    print(f"\n  lambda  = {lam_val:.6e}")
    print(f"  lambda3 = {lam3_val:.6e}  (reldiff={reldiff:.2e})")

    np.testing.assert_allclose(
        np.asarray(lam3["finite-n lambda3"]),
        np.asarray(lam["finite-n lambda"]),
        rtol=2e-2,
        atol=1e-6,
    )

    # xi3 is in (rho, upsilon, zeta) basis — recover physical (rho, theta, zeta)
    xi3 = np.asarray(lam3["finite-n xi"])
    xi3_rho = xi3[:_N]
    xi3_ups = xi3[_N : 2 * _N]
    xi3_zeta = xi3[2 * _N :]
    xi3_theta = xi3_ups + xi3_zeta  # invert: upsilon = xi^theta - xi^zeta
    xi3_rtz = np.concatenate([xi3_rho, xi3_theta, xi3_zeta])

    ref = np.asarray(lam["finite-n xi"]).reshape(-1)
    cand = xi3_rtz.reshape(-1)
    phase = np.vdot(cand, ref)
    if abs(phase) > 0:
        cand = cand * phase / abs(phase)
    cand = (np.vdot(cand, ref) / np.vdot(cand, cand)) * cand
    relerr = np.linalg.norm(cand - ref) / (np.linalg.norm(ref) + 1e-300)
    print(f"  eigenfunction relerr (lambda3 vs lambda) = {relerr:.3e}")
    assert relerr < 2e-1, f"eigenfunction relerr={relerr:.3e} exceeds tolerance"

    # xi^rho must be zero at both radial boundaries
    xi_rho = xi3_rho.reshape(_N_RHO, _N_THETA, _N_ZETA)
    np.testing.assert_allclose(xi_rho[0], 0.0, atol=1e-8)
    np.testing.assert_allclose(xi_rho[-1], 0.0, atol=1e-8)


@pytest.mark.unit
@pytest.mark.slow
def test_lambda_matfree():
    """finite-n lambda matfree operator satisfies A*v ≈ lambda*v.

    Given the eigenvector v from the dense finite-n lambda solve, the matfree
    operator A_op (built internally by the matfree solver) should satisfy
    A_op @ v ≈ lambda * v.  We verify this by checking that the element-wise
    ratio (A_op @ v) / v is approximately constant over entries where |v| is
    large, and that its mean value agrees with the dense eigenvalue.
    """
    lam = _LAM
    eigenvalue = float(np.asarray(lam["finite-n lambda"])[0])
    v_dense = np.asarray(lam["finite-n eigenfunction"]).reshape(-1)

    # Capture the LinearOperator built inside the matfree solver
    Aop = {}
    original_eigsh = stability.eigsh

    def capture_and_solve(op, *args, **kwargs):
        Aop["op"] = op
        return original_eigsh(op, *args, **kwargs)

    stability.eigsh = capture_and_solve
    try:
        try:
            _EQ.compute(
                "finite-n lambda matfree",
                **{
                    **_KW,
                    "v_guess": v_dense,
                    "matfree_solver": "eigsh_shiftinvert",
                    "eigsh_tol": 1e-6,
                    "eigsh_maxiter": 3000,
                },
            )
        except ArpackNoConvergence:
            pass  # We only need Aop, not the matfree eigenvalue
    finally:
        stability.eigsh = original_eigsh

    assert "op" in Aop, "Matfree operator was never built — eigsh was never called"

    av = np.asarray(Aop["op"] @ v_dense)
    dominant = np.abs(v_dense) > 0.2 * np.max(np.abs(v_dense))
    ratios = np.real(av[dominant] / v_dense[dominant])

    lo, hi = np.percentile(ratios, [20, 80])
    trimmed = ratios[(ratios >= lo) & (ratios <= hi)]
    ratio_mean = float(np.mean(trimmed))
    ratio_spread = float(
        np.max(np.abs(trimmed - ratio_mean)) / (abs(ratio_mean) + 1e-12)
    )

    print(f"\n  dense eigenvalue  = {eigenvalue:.6e}")
    reldiff = abs(ratio_mean - eigenvalue) / (abs(eigenvalue) + 1e-300)
    print(f"  Av/v ratio mean   = {ratio_mean:.6e}  (reldiff={reldiff:.2e})")
    print(f"  Av/v ratio spread = {ratio_spread:.3e}")
    assert ratio_spread < 3.5e-1, f"Av/v not constant: spread={ratio_spread:.3e}"
    np.testing.assert_allclose(ratio_mean, eigenvalue, rtol=3e-1, atol=2e-4)


@pytest.mark.unit
@pytest.mark.slow
def test_lambda3_matfree():
    """finite-n lambda3 matfree operator satisfies A*v ≈ lambda3*v.

    Same logic as test_lambda_matfree but using the lambda3 matfree operator
    and the eigenvector from the dense lambda3 solve.
    """
    lam3 = _LAM3
    eigenvalue3 = float(np.asarray(lam3["finite-n lambda3"])[0])
    v_dense3 = np.asarray(lam3["finite-n eigenfunction3"]).reshape(-1)

    Aop = {}
    original_eigsh = stability.eigsh

    def capture_and_solve(op, *args, **kwargs):
        Aop["op"] = op
        return original_eigsh(op, *args, **kwargs)

    stability.eigsh = capture_and_solve
    try:
        try:
            _EQ.compute(
                "finite-n lambda3 matfree",
                **{
                    **_KW,
                    "v_guess": v_dense3,
                    "matfree_solver": "eigsh_shiftinvert",
                    "eigsh_tol": 1e-6,
                    "eigsh_maxiter": 3000,
                },
            )
        except ArpackNoConvergence:
            pass  # We only need Aop, not the matfree eigenvalue
    finally:
        stability.eigsh = original_eigsh

    assert "op" in Aop, "Matfree operator was never built — eigsh was never called"

    av = np.asarray(Aop["op"] @ v_dense3)
    dominant = np.abs(v_dense3) > 0.2 * np.max(np.abs(v_dense3))
    ratios = np.real(av[dominant] / v_dense3[dominant])

    lo, hi = np.percentile(ratios, [20, 80])
    trimmed = ratios[(ratios >= lo) & (ratios <= hi)]
    ratio_mean = float(np.mean(trimmed))
    ratio_spread = float(
        np.max(np.abs(trimmed - ratio_mean)) / (abs(ratio_mean) + 1e-12)
    )

    print(f"\n  dense eigenvalue3  = {eigenvalue3:.6e}")
    reldiff = abs(ratio_mean - eigenvalue3) / (abs(eigenvalue3) + 1e-300)
    print(f"  Av/v ratio mean    = {ratio_mean:.6e}  (reldiff={reldiff:.2e})")
    print(f"  Av/v ratio spread  = {ratio_spread:.3e}")
    assert ratio_spread < 3.5e-1, f"Av/v not constant: spread={ratio_spread:.3e}"
    np.testing.assert_allclose(ratio_mean, eigenvalue3, rtol=3e-1, atol=2e-4)
