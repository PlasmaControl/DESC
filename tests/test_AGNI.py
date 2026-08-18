"""AGNI finite-n stability tests.

Ground truth for all tests: ``finite-n lambda3`` using the dense Cholesky-transformed
eigensolver.

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
    _LAM3 = _EQ.compute("finite-n lambda3", **_KW)
else:
    _EQ = _GRID = _DIFFMAT = _LAM3 = None
    _N = _N_SHELL = _N_KEEP = 0
    _KW = {}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.slow
def test_lambda3_matfree():
    """finite-n lambda3 matfree operator satisfies A*v ≈ lambda3*v.

    Uses the lambda3 matfree operator and the eigenvector from the dense
    lambda3 solve.
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
