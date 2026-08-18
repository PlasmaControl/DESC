"""AGNI finite-n stability tests.

Ground truth for the solver tests: ``finite-n lambda3``, the dense
Cholesky-transformed assembly solved by ARPACK ``eigsh``. Every other solver path
must reproduce it on the same equilibrium, grid and shift.

test_stability_kwargs_are_registered
    Every kwarg read by ``desc/compute/_stability.py`` is declared on some
    ``register_compute_fun``. Undeclared kwargs are REJECTED at runtime by
    ``desc/compute/utils.py`` (``bad_kwargs = kwargs.keys() - allowed_kwargs``),
    so a kwarg that is read but never declared is dead on arrival. Needs no
    equilibrium, so it runs everywhere and is the cheap guard against a
    registration being dropped while its reader stays.

test_jax_lanczos_matches_dense
    ``AGNI_EIGENSOLVER=jax_lanczos`` (JAX assembly + exact LU shift-invert
    Lanczos) reproduces the dense eigenvalue through
    ``finite-n lambda3 rayleigh``.

test_matfree_operator_matches_dense_matrix
    The matrix-free operator, materialized column by column, equals the dense
    ``_agni3_assemble`` matrix entry-for-entry on the kept DOFs.

The equilibrium ships with the repository as ``tests/inputs/AGNI_QH_lowres.h5``
(the low-resolution Patil QH case), so these tests need no external files.
Grid and equilibrium are built once at module level and shared across all tests.
Set AGNI_TEST_RES=N_RHO,N_THETA,N_ZETA (default 16,12,8) to change resolution.
The default is radial-heavy on purpose: these modes need rho resolution, and
starving it makes the eigensolve pick the wrong mode rather than merely a
less accurate one.
Set AGNI_EQ_PATH to override the equilibrium file.
"""

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

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

_RES = os.environ.get("AGNI_TEST_RES", "16,12,8")
_N_RHO, _N_THETA, _N_ZETA = (int(v) for v in _RES.split(","))

# Ships with the repo, so these tests need no machine-specific paths.
_DEFAULT_EQ = Path(__file__).parent / "inputs" / "AGNI_QH_lowres.h5"
_EQ_PATH = Path(os.environ.get("AGNI_EQ_PATH", _DEFAULT_EQ))
_AGNI_SKIP_REASON = f"AGNI equilibrium fixture not found: {_EQ_PATH}"


@pytest.fixture(scope="module")
def agni(request):
    """Dict of equilibrium, PEST grid, DiffMat and the dense lambda3 ground truth.

    Module-scoped and LAZY. Building this costs a coordinate map plus a dense
    eigensolve, so it must not run at import time: doing so charges every
    collection -- including `--collect-only` and tests that need no equilibrium
    at all, like the kwarg-registration guard -- for the full solve.
    """
    if not _EQ_PATH.is_file():
        pytest.skip(_AGNI_SKIP_REASON)
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

    # Dense ground truth, computed once and shared by every solver test.
    _LAM3 = _EQ.compute("finite-n lambda3", **_KW)

    return dict(
        eq=_EQ,
        grid=_GRID,
        diffmat=_DIFFMAT,
        kw=_KW,
        lam3=_LAM3,
        n_total=_N,
        n_keep=_N_KEEP,
        # Dirichlet keep-mask: xi^rho is dropped on the innermost and outermost
        # rho shells; the other two components are kept everywhere.
        keep=np.concatenate(
            [np.arange(_N_SHELL, _N - _N_SHELL), np.arange(_N, 3 * _N)]
        ),
        res=(_N_RHO, _N_THETA, _N_ZETA),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stability_kwargs_are_registered():
    """Every kwarg read in _stability.py is declared on a register_compute_fun.

    ``desc/compute/utils.py`` rejects any kwarg missing from ``allowed_kwargs``
    with ``ValueError: Unrecognized argument(s)``, and ``allowed_kwargs`` is
    populated ONLY from ``register_compute_fun`` declarations. So a kwarg that
    the code reads but no registration declares can never be supplied by a
    caller -- the read silently falls back to its default forever, or the call
    raises. Dropping a declaration while leaving its reader in place is exactly
    that bug, and it is invisible to every other test.
    """
    from desc.compute.data_index import allowed_kwargs

    src = (
        Path(stability.__file__).read_text()
        if Path(stability.__file__).is_file()
        else ""
    )
    assert src, "could not read desc/compute/_stability.py"

    read = set(re.findall(r"kwargs\.get\(\s*[\"'](\w+)[\"']", src))
    read |= set(re.findall(r"kwargs\[\s*[\"'](\w+)[\"']\s*\]", src))
    read |= set(re.findall(r"kwargs\.pop\(\s*[\"'](\w+)[\"']", src))
    assert read, "found no kwargs reads -- the regex needs updating"

    # Known pre-existing gaps, present before this test was written. Listed
    # explicitly so the guard cannot be weakened silently: to remove one, either
    # declare the kwarg on the compute function that reads it, or delete the read.
    known_unregistered = {
        "jq_lower_metric",  # read in _agni3_assemble
        "mirror",  # read in _agni3_assemble
        "phase_offset",  # read in _AGNI3
        # `ring_nodes` is INTERNAL: `_agni3_assemble` is called directly as a
        # Python function with ring_nodes=..., never through `eq.compute`, so it
        # never reaches the allowed_kwargs check. Declaring it on a registration
        # would advertise a user-facing knob that is not one.
        "ring_nodes",
    }

    missing = sorted(read - allowed_kwargs - known_unregistered)
    assert not missing, (
        "kwargs read by _stability.py but declared on no register_compute_fun, "
        f"so DESC will reject them at compute time: {missing}"
    )


@pytest.mark.unit
@pytest.mark.slow
def test_matfree_operator_matches_dense_matrix(agni):
    """The matrix-free operator equals the dense assembled matrix, exactly.

    Materializes the operator column by column as ``Ax(e_j)`` and compares
    against ``_agni3_assemble``'s dense ``A`` on the kept DOFs. The two are the
    same operator by construction -- one applied, one assembled -- so this is an
    equality test, not an approximation test.

    This is the check that keeps ``_agni3_matfree_operator`` honest, and it
    matters well beyond any single compute key: ``finite-n lambda3 rayleigh``
    builds its ``jax_lanczos`` and ``pcg_deflated`` operators from this same
    helper, and the ring preconditioner's blocks are sub-blocks of this matrix.
    """
    from desc.compute.data_index import data_index

    deps = data_index["desc.equilibrium.equilibrium.Equilibrium"]["finite-n lambda3"][
        "dependencies"
    ]["data"]
    data = agni["eq"].compute(deps, grid=agni["grid"], diffmat=agni["diffmat"])
    transforms = {"grid": agni["grid"], "diffmat": agni["diffmat"]}
    kw = {k: v for k, v in agni["kw"].items() if k not in ("grid", "diffmat")}

    op = stability._agni3_matfree_operator(
        agni["eq"].params_dict, transforms, {}, data, **kw
    )
    Ax, n_keep = op["Ax"], int(op["n_keep"])

    eye = jnp.eye(n_keep, dtype=Ax(jnp.ones(n_keep)).dtype)
    A_mf = np.asarray(jax.vmap(Ax)(eye)).T  # column j = A e_j

    dense = stability._agni3_assemble(
        agni["eq"].params_dict, transforms, {}, data, **kw
    )
    A_dense = np.asarray(dense["A"])

    assert A_mf.shape == A_dense.shape, f"{A_mf.shape} vs {A_dense.shape}"
    err = np.max(np.abs(A_mf - A_dense)) / np.max(np.abs(A_dense))
    print(f"\n  max|A_matfree - A_dense| / max|A_dense| = {err:.3e}")
    assert err < 1e-10, f"matrix-free operator disagrees with dense assembly: {err:.3e}"


@pytest.mark.unit
@pytest.mark.slow
def test_jax_lanczos_matches_dense(agni, monkeypatch):
    """AGNI_EIGENSOLVER=jax_lanczos reproduces the dense ARPACK eigenvalue.

    ``finite-n lambda3 rayleigh`` returns the Rayleigh quotient of a freshly
    eigensolved vector. With ``jax_lanczos`` that solve is a pure-JAX assembly
    plus an exact LU shift-invert Lanczos -- entirely different machinery from
    the dense ``finite-n lambda3`` + scipy ARPACK path -- so agreement between
    them is a real cross-check of the operator and the shift, not a tautology.

    ``sigma`` must sit BELOW the most-negative eigenvalue or shift-invert
    converges to the wrong mode, which is why it is derived from the dense
    answer rather than left at the default.
    """
    lam_dense = float(np.asarray(agni["lam3"]["finite-n lambda3"])[0])

    monkeypatch.setenv("AGNI_EIGENSOLVER", "jax_lanczos")
    monkeypatch.setenv("AGNI_NUM_MATVECS", "100")

    kw = {k: v for k, v in agni["kw"].items() if k != "v_guess"}
    data = agni["eq"].compute("finite-n lambda3 rayleigh", **kw, sigma=1.3 * lam_dense)
    lam_R = float(np.asarray(data["finite-n lambda3 rayleigh"]).reshape(-1)[0])
    resid = float(np.asarray(data["finite-n lambda3 rayleigh residual"]).reshape(-1)[0])

    reldiff = abs(lam_R - lam_dense) / (abs(lam_dense) + 1e-300)
    print(f"\n  lambda3 (dense ARPACK) = {lam_dense:.9e}")
    print(f"  lam_R   (jax_lanczos)  = {lam_R:.9e}  (reldiff={reldiff:.2e})")
    print(f"  Rayleigh residual      = {resid:.3e}")

    assert np.sign(lam_R) == np.sign(lam_dense), (
        f"jax_lanczos flipped the sign of the growth rate: {lam_R:.6e} vs "
        f"{lam_dense:.6e} -- a stable/unstable misclassification"
    )
    # The EIGENVALUE is the assertion that matters here, and it is checked
    # tightly below. The Rayleigh residual measures EIGENVECTOR convergence,
    # which lags badly and is strongly resolution-dependent -- MEASURED
    # 1.947e-03 at 12x14x15 and 1.861e-02 at 16x12x8, both with 100 matvecs,
    # while lam_R matched dense to 2.6e-11 and 6.0e-10 respectively. So this
    # bound is deliberately loose: it exists to catch a solve that diverged or
    # locked onto the wrong mode, not to police the Lanczos tail. Two earlier
    # values of this bound were guessed from a single resolution and both were
    # wrong; it is now set from the worst measurement, with margin.
    assert resid < 0.1, f"Rayleigh residual {resid:.3e} -- eigenvector not converged"
    np.testing.assert_allclose(lam_R, lam_dense, rtol=1e-4)
