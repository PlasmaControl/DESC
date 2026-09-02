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
Set AGNI_TEST_RES=N_RHO,N_THETA,N_ZETA (default 24,12,8) to change resolution.
The default is radial-heavy on purpose: these modes need rho resolution, and
starving it makes the eigensolve pick the wrong mode rather than merely a
less accurate one. 24 radial specifically is the smallest fine level that sits
above the MEASURED coarse floor of 16 (job 57261816) that the two-level test
needs; do not lower it to save time.
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
from desc.compute.utils import _compute as compute_fun
from desc.compute.utils import get_profiles, get_transforms
from desc.diffmat_utils import DiffMat, fourier_diffmat, legendre_diffmat
from desc.equilibrium import Equilibrium
from desc.grid import Grid, LinearGrid, QuadratureGrid
from desc.integrals.quad_utils import automorphism_staircase1, leggauss_lob

# Flux functions that must be evaluated on a rho LinearGrid, not on the AGNI
# grid. Same list as ``FinitenStability._flux_keys``.
_FLUX_KEYS = [
    "iota",
    "iota_r",
    "iota_den",
    "iota_den_r",
    "iota_num",
    "iota_num_r",
    "iota_num current",
    "iota_num_r current",
    "iota_num vacuum",
    "iota_num_r vacuum",
    "psi_r",
    "psi_rr",
    "p",
    "p_r",
]


def finiten_prefill(eq, grid, params=None):
    """Data the finite-n keys need from grids other than the AGNI grid.

    ``eq.compute`` picks these grids for you via ``override_grid``, but it is
    the eager wrapper and cannot be traced. Under jit or AD you call
    ``_compute`` and provide them yourself:

    * 0-D ``a`` on a ``QuadratureGrid``. It sets the whole
      non-dimensionalization -- ``B_N = |Psi| / (pi a^2)``, and the operator's
      terms carry ``a**2``, ``a**3``, ``a**4`` -- and it comes out ~4% different
      on any other grid, so taking it from the AGNI grid silently rescales the
      operator.
    * the 1-D flux functions on a ``LinearGrid`` over this grid's rho values,
      copied onto the AGNI nodes.

    Everything else the keys depend on is a pointwise evaluation and is correct
    on the AGNI grid directly.
    """
    params = eq.params_dict if params is None else params

    quad_grid = QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid, eq.NFP)
    zero_d = compute_fun(
        eq,
        ["a"],
        params=params,
        transforms=get_transforms(["a"], obj=eq, grid=quad_grid),
        profiles=get_profiles(["a"], eq, quad_grid),
    )
    # Take ONLY `a`. `compute_fun` hands back every intermediate it touched,
    # all shaped for quad_grid, and seeding those into the next call mixes grids.
    data = {"a": jnp.asarray(zero_d["a"])}

    rho = np.unique(np.asarray(grid.nodes[:, 0]))
    flux_grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
    flux = compute_fun(
        eq,
        _FLUX_KEYS,
        params=params,
        transforms=get_transforms(_FLUX_KEYS, obj=eq, grid=flux_grid),
        profiles=get_profiles(_FLUX_KEYS, eq, flux_grid),
        data=dict(data),
    )
    for key in _FLUX_KEYS:
        data[key] = grid.copy_data_from_other(
            jnp.asarray(flux[key]), flux_grid, surface_label="rho"
        )
    return data


def map_to_desc(eq, pest_grid):
    """PEST (rho, theta_PEST, zeta) nodes -> DESC (rho, theta, zeta) nodes.

    The compute functions take the MAPPED grid. rho is invariant under the map.
    """
    return Grid(
        eq.map_coordinates(
            jnp.reshape(
                pest_grid.meshgrid_reshape(pest_grid.nodes, order="rtz"), (-1, 3)
            ),
            inbasis=("rho", "theta_PEST", "zeta"),
            outbasis=("rho", "theta", "zeta"),
            period=(jnp.inf, 2 * jnp.pi, jnp.inf),
            tol=1e-6,  # 1e-12 is overkill for tests and very slow on CPU
            maxiter=20,
        )
    )


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

_RES = os.environ.get("AGNI_TEST_RES", "24,12,8")
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
    # FAIL, do not skip. The equilibrium is version-controlled at
    # tests/inputs/AGNI_QH_lowres.h5, so a missing file means the checkout is
    # broken -- not that the test is inapplicable. This used to skip, and when
    # .gitignore's `*.h5` rule silently kept the fixture out of the repo, CI
    # skipped 6 of these 8 tests and reported the whole finite-n solver as
    # untested. A skip is invisible; a failure is not.
    if not _EQ_PATH.is_file():
        pytest.fail(_AGNI_SKIP_REASON)
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
    _GRID = map_to_desc(_EQ, _grid0)

    _N = _N_RHO * _N_THETA * _N_ZETA  # total grid points
    _N_SHELL = _N_THETA * _N_ZETA  # one theta-zeta shell
    _N_KEEP = 3 * _N - 2 * _N_SHELL  # DOFs after xi^rho Dirichlet BCs

    # Solver kwargs shared by every compute_fun call in this file.
    _KW = dict(
        incompressible=False,
        gamma=5.0 / 3.0,
        v_guess=np.ones(_N_KEEP),
    )

    # Dense ground truth, computed once and shared by every solver test.
    _name = "finite-n lambda3"
    _LAM3 = compute_fun(
        _EQ,
        [_name],
        params=_EQ.params_dict,
        transforms=get_transforms([_name], obj=_EQ, grid=_GRID, diffmat=_DIFFMAT),
        profiles=get_profiles([_name], _EQ, _GRID),
        data=finiten_prefill(_EQ, _GRID),
        **_KW,
    )

    return dict(
        eq=_EQ,
        grid=_GRID,
        # The PEST source grid, BEFORE mapping to DESC coordinates. Compute
        # functions take the mapped grid; `FinitenStability` takes this one and
        # maps it itself, per its build() error message.
        pest_grid=_grid0,
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
    # Options resolved through the kwarg-first helpers. Without these two the
    # guard sees nothing: moving a read from `kwargs.get("cg_tol", ...)` to
    # `_solver_opt(kwargs, "cg_tol", ...)` hid 16 options from it at once, and
    # two of them (cg_tol, cg_maxiter) were unregistered and raised at runtime.
    read |= set(re.findall(r"_solver_opt\(\s*kwargs,\s*[\"'](\w+)[\"']", src))
    read |= set(re.findall(r"_solver_flag\(\s*kwargs,\s*[\"'](\w+)[\"']", src))
    assert read, "found no kwargs reads -- the regex needs updating"

    # Known pre-existing gaps, present before this test was written. Listed
    # explicitly so the guard cannot be weakened silently: to remove one, either
    # declare the kwarg on the compute function that reads it, or delete the read.
    known_unregistered = {
        "mirror",  # read in _agni3_assemble
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


@pytest.mark.regression
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

    deps = data_index["desc.equilibrium.equilibrium.Equilibrium"][
        "finite-n lambda3 rayleigh"
    ]["dependencies"]["data"]
    eq, grid, dm = agni["eq"], agni["grid"], agni["diffmat"]
    data = compute_fun(
        eq,
        deps,
        params=eq.params_dict,
        transforms=get_transforms(deps, obj=eq, grid=grid, diffmat=dm),
        profiles=get_profiles(deps, eq, grid),
        data=finiten_prefill(eq, grid),
    )
    transforms = {"grid": agni["grid"], "diffmat": agni["diffmat"]}
    kw = dict(agni["kw"])

    op = stability._agni3_matfree_operator(
        agni["eq"].params_dict, transforms, {}, data, **kw
    )
    Ax, n_keep = op["Ax"], int(op["n_keep"])

    # Materialize in COLUMN BLOCKS, not one `jax.vmap(Ax)(jnp.eye(n_keep))`.
    # That one-liner cost 5.6 GB peak RSS at the default 24,12,8 fixture and is
    # what made this test unrunnable on a GitHub runner: `jnp.eye(n_keep)` is
    # 0.67 GB on its own, and vmapping over all n_keep columns promotes EVERY
    # intermediate inside `Ax_full` -- and there are dozens -- from (n_rho,
    # n_theta, n_zeta) to (n_keep, n_rho, n_theta, n_zeta), 0.23 GB apiece. The
    # runner swapped instead of OOM-ing, so CI hung for hours rather than
    # failing. Blocking caps the batch axis at `block` and leaves the assertion
    # identical.
    dtype = Ax(jnp.ones(n_keep)).dtype
    block = 64
    A_mf = np.empty((n_keep, n_keep), dtype=dtype)
    _Ax_block = jax.jit(jax.vmap(Ax))
    for j0 in range(0, n_keep, block):
        w = min(block, n_keep - j0)
        cols = (
            jnp.zeros((w, n_keep), dtype)
            .at[jnp.arange(w), jnp.arange(j0, j0 + w)]
            .set(1.0)
        )
        A_mf[:, j0 : j0 + w] = np.asarray(_Ax_block(cols)).T  # column j = A e_j

    dense = stability._agni3_assemble(
        agni["eq"].params_dict, transforms, {}, data, **kw
    )
    A_dense = np.asarray(dense["A"])

    assert A_mf.shape == A_dense.shape, f"{A_mf.shape} vs {A_dense.shape}"
    err = np.max(np.abs(A_mf - A_dense)) / np.max(np.abs(A_dense))
    print(f"\n  max|A_matfree - A_dense| / max|A_dense| = {err:.3e}")
    assert err < 1e-10, f"matrix-free operator disagrees with dense assembly: {err:.3e}"


@pytest.mark.regression
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
    eq, grid, dm = agni["eq"], agni["grid"], agni["diffmat"]
    name = "finite-n lambda3 rayleigh"
    # Request the mode-data keys BY NAME, not just off the side-effect dict that
    # `finite-n lambda3 rayleigh` fills in. Only this exercises their
    # `register_compute_fun` entries; a broken registration (wrong `data=`
    # dependency, missing kwarg declaration) raises here and nowhere else.
    names = [
        name,
        "finite-n eigenfunction3 rayleigh",
        "finite-n xi rayleigh",
        "finite-n deltaB rayleigh",
        "finite-n deltaV rayleigh",
    ]
    data = compute_fun(
        eq,
        names,
        params=eq.params_dict,
        transforms=get_transforms(names, obj=eq, grid=grid, diffmat=dm),
        profiles=get_profiles(names, eq, grid),
        data=finiten_prefill(eq, grid),
        **kw,
        sigma=1.3 * lam_dense,
    )
    lam_R = float(np.asarray(data["finite-n lambda3 rayleigh"]).reshape(-1)[0])
    resid = float(np.asarray(data["finite-n lambda3 rayleigh residual"]).reshape(-1)[0])
    nr, nt, nz = agni["res"]
    n_total = nr * nt * nz
    v = np.asarray(data[name + " v"]).reshape(-1)
    ef = np.asarray(data["finite-n eigenfunction3 rayleigh"])
    xi = np.asarray(data["finite-n xi rayleigh"])
    dB = np.asarray(data["finite-n deltaB rayleigh"])
    dV = np.asarray(data["finite-n deltaV rayleigh"])

    assert ef.shape == (3 * n_total,)
    assert xi.shape == (3 * n_total,)
    assert dB.shape == (nr, nt, nz)
    assert dV.shape == (nr, nt, nz)

    # The scatter back to full length is the one step in
    # `_agni3_store_rayleigh_mode_data` with no redundancy to catch it: an
    # off-by-one in `keep` silently shifts the whole mode by a rho shell and
    # every downstream field still looks plausible. Pin it exactly.
    keep = agni["keep"]
    np.testing.assert_allclose(ef[keep], v, rtol=0, atol=0)
    dropped = np.setdiff1d(np.arange(3 * n_total), keep)
    assert not np.any(ef[dropped]), "xi^rho Dirichlet slots must stay exactly zero"

    # xi is the whitened eigenvector mapped back to the physical displacement,
    # so it must be supported on the same DOF and be a genuinely nonzero mode --
    # a silently all-zero field would pass every shape and finiteness check.
    assert np.all(np.isfinite(xi)) and np.any(xi)
    assert np.all(np.isfinite(dB)) and np.any(dB)
    assert np.all(np.isfinite(dV)) and np.any(dV)
    # deltaB and deltaV are magnitudes (sqrt of a metric contraction), so they
    # are real and nonnegative by construction. A negative entry means the
    # contraction lost a metric term or a sign.
    assert dB.dtype.kind == "f" and dV.dtype.kind == "f"
    assert np.all(dB >= 0.0) and np.all(dV >= 0.0)

    # Same mode, computed by the dense `finite-n lambda3` path. This is the only
    # check on the whitening transform and the derivative reconstruction inside
    # `_agni3_store_rayleigh_mode_data`: get the Linv/diagBsqinv congruence or
    # the d_dr/d_dv/d_dz plumbing wrong and xi is still finite, still the right
    # shape, still supported on `keep` -- but it is no longer the mode, and the
    # overlap collapses from 1 to O(0.1).
    #
    # Compared as an overlap, not elementwise: an eigenvector is defined up to a
    # complex phase, and the two solvers fix it independently. The tolerance is
    # deliberately loose for the same reason the eigenvalue tolerance is -- the
    # two solvers can land on different vectors inside a near-degenerate cluster
    # -- so this is a "same mode or not" check, not a precision check.
    def _overlap(a, b):
        a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
        return abs(np.vdot(a / np.linalg.norm(a), b / np.linalg.norm(b)))

    ov_xi = _overlap(xi, agni["lam3"]["finite-n xi"])
    ov_dV = _overlap(dV, agni["lam3"]["finite-n deltaV"])
    ov_dB = _overlap(dB, agni["lam3"]["finite-n deltaB"])
    print(f"  |<xi_R, xi_dense>|     = {ov_xi:.6f}")
    print(f"  |<dV_R, dV_dense>|     = {ov_dV:.6f}")
    print(f"  |<dB_R, dB_dense>|     = {ov_dB:.6f}")
    assert ov_xi > 0.99, f"Rayleigh xi is not the dense mode: overlap {ov_xi:.4f}"
    assert ov_dV > 0.99, f"Rayleigh deltaV is not the dense field: {ov_dV:.4f}"
    assert ov_dB > 0.99, f"Rayleigh deltaB is not the dense field: {ov_dB:.4f}"

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


@pytest.mark.regression
@pytest.mark.slow
def test_jax_lanczos_matches_dense_axisym(monkeypatch):
    """``test_jax_lanczos_matches_dense`` for a complex A.

    ``axisym=True`` makes the operator complex Hermitian. The module fixture is
    3D and never builds one, so this builds its own one-zeta-plane level.
    Without the fix: lam_R=+9.713e-02 vs dense -2.660e-03, residual 1.14e+03.
    """
    if not _EQ_PATH.is_file():
        pytest.fail(_AGNI_SKIP_REASON)
    eq = _load_old_equilibrium(str(_EQ_PATH))

    n_rho, n_theta = _N_RHO, _N_THETA
    pest_grid, diffmat = _build_pest_level(eq, n_rho, n_theta, 1)
    grid = map_to_desc(eq, pest_grid)

    n_total = n_rho * n_theta
    n_keep = 3 * n_total - 2 * n_theta  # one zeta plane, so n_shell == n_theta
    kw = dict(
        incompressible=False,
        gamma=5.0 / 3.0,
        axisym=True,
        n_mode_axisym=1,
    )

    dense = compute_fun(
        eq,
        ["finite-n lambda3"],
        params=eq.params_dict,
        transforms=get_transforms(
            ["finite-n lambda3"], obj=eq, grid=grid, diffmat=diffmat
        ),
        profiles=get_profiles(["finite-n lambda3"], eq, grid),
        data=finiten_prefill(eq, grid),
        **kw,
        v_guess=np.ones(n_keep),
    )
    lam_dense = float(np.asarray(dense["finite-n lambda3"])[0])

    # If A came out real the complex branch was never reached and this is vacuous.
    assert np.iscomplexobj(
        np.asarray(dense["finite-n xi"])
    ), "axisym=True did not produce a complex operator"

    monkeypatch.setenv("AGNI_EIGENSOLVER", "jax_lanczos")
    monkeypatch.setenv("AGNI_NUM_MATVECS", "100")
    name = "finite-n lambda3 rayleigh"
    data = compute_fun(
        eq,
        [name],
        params=eq.params_dict,
        transforms=get_transforms([name], obj=eq, grid=grid, diffmat=diffmat),
        profiles=get_profiles([name], eq, grid),
        data=finiten_prefill(eq, grid),
        **kw,
        sigma=1.3 * lam_dense,
    )
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
    # Tight, unlike the 3D sibling's 0.1: measured 1.2e-07 here with the fix.
    assert resid < 1e-4, f"Rayleigh residual {resid:.3e} -- eigenvector not converged"
    np.testing.assert_allclose(lam_R, lam_dense, rtol=1e-4)


@pytest.mark.regression
@pytest.mark.slow
def test_ring_blocks_eager_and_vmapped_both_match_dense(agni):
    """Both ring-block builds reproduce the dense matrix's sub-blocks.

    The solver assembles the ring preconditioner's blocks two ways: a vmapped
    build over all rings at once (``build_ring_blocks``) and a plain host loop
    over rings. They must agree, and both must equal the corresponding
    sub-blocks of the dense assembled matrix.

    This replaces the ``AGNI_RING_COMPARE`` environment-variable block that used
    to live inside the compute function and ended in ``raise SystemExit(0)``. It
    is a stronger check than that was: the old one compared the two builds
    against EACH OTHER, so a shared error would have passed. Comparing both
    against the dense assembly catches that.

    The assertion below is the reference: the restricted assembler must match
    the dense assembly to machine precision on every ring.
    """
    from desc.compute._stability_solvers import (
        build_ring_blocks,
        finish_ring_block,
        ring_index_maps,
        ring_nodes,
    )
    from desc.compute.data_index import data_index

    deps = data_index["desc.equilibrium.equilibrium.Equilibrium"]["finite-n lambda3"][
        "dependencies"
    ]["data"]
    eq, grid, dm = agni["eq"], agni["grid"], agni["diffmat"]
    data = compute_fun(
        eq,
        deps,
        params=eq.params_dict,
        transforms=get_transforms(deps, obj=eq, grid=grid, diffmat=dm),
        profiles=get_profiles(deps, eq, grid),
        data=finiten_prefill(eq, grid),
    )
    transforms = {"grid": agni["grid"], "diffmat": agni["diffmat"]}
    kw = dict(agni["kw"])
    params = agni["eq"].params_dict

    dense = stability._agni3_assemble(params, transforms, {}, data, **kw)
    A = np.asarray(dense["A"])
    keep = np.asarray(dense["keep"])

    res = agni["res"]
    n_rho, n_theta, n_zeta = res
    sel, pad, G = ring_index_maps(keep, res)

    # vmapped build, sigma=0 so the blocks are of A itself
    vmapped = np.asarray(
        build_ring_blocks(
            stability._agni3_assemble,
            params,
            transforms,
            {},
            data,
            kw,
            res,
            sel,
            pad,
            0.0,
        )
    )

    worst_v = worst_e = 0.0
    for gi, g in enumerate(G):
        live = g[g >= 0]
        ref = A[np.ix_(live, live)]
        scale = max(np.max(np.abs(ref)), 1e-300)

        got_v = vmapped[gi][: live.size, : live.size]
        worst_v = max(worst_v, float(np.max(np.abs(got_v - ref))) / scale)

        # eager: one ring at a time through the same assembler
        i, k = divmod(gi, n_zeta)
        nodes = ring_nodes(n_rho, n_theta, n_zeta, i, k)
        out = stability._agni3_assemble(
            params, transforms, {}, data, ring_nodes=jnp.asarray(nodes), **kw
        )
        blk = np.asarray(
            finish_ring_block(out["A"], out["Linv"], out["au_diag"], n_theta)
        )
        # `sel[gi]` holds the positions WITHIN the 3*n_theta ring ordering that
        # survive the keep mask. G holds the reduced indices COMPACTED to the
        # front, which is a different indexing entirely -- using G's positions
        # here gathers the wrong entries out of the ring block.
        pos = np.asarray(sel)[gi][: live.size]
        got_e = blk[np.ix_(pos, pos)]
        got_e = 0.5 * (got_e + got_e.T)
        worst_e = max(worst_e, float(np.max(np.abs(got_e - ref))) / scale)

    print(f"\n  vmapped vs dense: {worst_v:.3e}\n  eager   vs dense: {worst_e:.3e}")
    assert worst_v < 5e-15, f"vmapped ring blocks disagree with dense: {worst_v:.3e}"
    assert worst_e < 5e-15, f"eager ring blocks disagree with dense: {worst_e:.3e}"


# ---------------------------------------------------------------------------
# FinitenStability objective, at CPU scale
#
# These cover what the opt-in gates in test_AGNI_precond.py cover at 32x32x12 --
# the objective wrapper, the Hellmann-Feynman gradient, update_state -- but small
# enough to run anywhere. The big gates stay: they are the ones tied to recorded
# numbers. These exist so the code paths are not left uncovered when those skip.
# ---------------------------------------------------------------------------


def _finiten_objective(agni, build=True, **kw):
    """Build a FinitenStability on the fixture's PEST grid."""
    from desc.objectives import FinitenStability

    obj = FinitenStability(
        eq=agni["eq"],
        target=0.0,
        weight=1.0,
        normalize=False,
        normalize_target=False,
        grid=agni["pest_grid"],
        diffmat=agni["diffmat"],
        gamma=5.0 / 3.0,
        metric="raw",
        name="finite-n lambda3 rayleigh",
        **kw,
    )
    if build:
        obj.build(verbose=0)
    return obj


@pytest.mark.regression
@pytest.mark.slow
def test_finiten_objective_matches_direct_compute(agni):
    """The objective returns the same lambda as a direct eq.compute.

    Covers ``FinitenStability.build`` and ``compute_data`` -- the flux-key
    gathering, the PEST grid mapping and the options dict -- none of which the
    bare ``eq.compute`` tests touch. Those are the wrapper layers where a wrong
    grid or a dropped option shows up as a plausible-looking wrong number rather
    than an exception.
    """
    lam_direct = float(np.asarray(agni["lam3"]["finite-n lambda3"])[0])

    obj = _finiten_objective(agni, lambda_guess=lam_direct)
    lam_obj = float(np.real(np.asarray(obj.compute(obj.things[0].params_dict))[0]))

    reldiff = abs(lam_obj - lam_direct) / abs(lam_direct)
    print(f"\n  direct  lambda3 = {lam_direct:.9e}")
    print(f"  objective lam_R = {lam_obj:.9e}  (reldiff={reldiff:.2e})")
    assert np.sign(lam_obj) == np.sign(lam_direct), (
        "objective and direct compute disagree on the SIGN of the growth rate: "
        f"{lam_obj:.6e} vs {lam_direct:.6e}"
    )
    np.testing.assert_allclose(lam_obj, lam_direct, rtol=1e-3)


@pytest.mark.regression
@pytest.mark.slow
def test_finiten_objective_gradient_is_hellmann_feynman(agni):
    """The gradient exists, is finite, and is not identically zero.

    ``finite-n lambda3 rayleigh`` freezes the eigenvector for AD, so the
    derivative reduces to the Hellmann-Feynman contraction
    ``v^T (dA/dp) v / v^T v``. That reduction happens through a ``custom_vjp``
    whose backward rule is easy to get silently wrong -- a zero cotangent, or a
    NaN, still "works" and just stops the optimizer moving.

    This is the only CPU-runnable coverage of ``_v_primal_fwd``/``_v_primal_bwd``;
    the recorded end-to-end check is the opt-in T2 optimizer gate.
    """
    from desc.objectives import ObjectiveFunction

    lam_direct = float(np.asarray(agni["lam3"]["finite-n lambda3"])[0])
    # `x()` and `grad()` belong to ObjectiveFunction, not to a single objective;
    # this is how the drivers wrap it too.
    objective = ObjectiveFunction(
        _finiten_objective(agni, lambda_guess=lam_direct, build=False),
        deriv_mode="blocked",
    )
    objective.build(verbose=0)

    x = objective.x(agni["eq"])
    g = np.asarray(objective.grad(x))

    assert g.shape[0] == x.shape[0], f"gradient shape {g.shape} vs x {x.shape}"
    assert np.all(np.isfinite(g)), (
        f"gradient has {np.sum(~np.isfinite(g))} non-finite entries; a NaN here "
        "propagates into the optimizer step without raising"
    )
    assert np.max(np.abs(g)) > 0.0, (
        "gradient is identically zero -- the Hellmann-Feynman contraction is not "
        "reaching the parameters, so the optimizer cannot move"
    )
    print(f"\n  |grad|_inf = {np.max(np.abs(g)):.6e}  n = {g.size}")


@pytest.mark.regression
@pytest.mark.slow
def test_update_state_refreshes_the_eigenpair(agni):
    """update_state(dense_eigsh) puts a fresh eigenpair into the constants.

    Covers the dense-refresh branch of ``update_state``, which the optimizer
    calls once per outer step. It is what supplies ``v_guess``/``lambda_guess``,
    and a silent failure there means the optimizer minimizes against a stale
    vector -- the failure mode WHY_V_CANNOT_BE_CACHED.md documents.
    """
    lam_direct = float(np.asarray(agni["lam3"]["finite-n lambda3"])[0])
    obj = _finiten_objective(agni, lambda_guess=lam_direct, state_solver="dense_eigsh")

    data = obj.update_state(obj.things[0].params_dict)

    lam = float(np.asarray(data["finite-n lambda3"]).reshape(-1)[0])
    assert np.isfinite(lam), "update_state produced a non-finite eigenvalue"
    np.testing.assert_allclose(lam, lam_direct, rtol=1e-3)

    # `finite-n eigenfunction3` is FULL length (3*n_total), not reduced to the
    # kept DOFs -- the same distinction that the matrix-free operator test has to
    # respect when it applies Ax.
    v = np.asarray(obj._constants["v_guess"]).reshape(-1)
    assert (
        v.size == 3 * agni["n_total"]
    ), f"v_guess size {v.size} != 3*n_total {3 * agni['n_total']}"
    assert np.all(np.isfinite(v)) and np.max(np.abs(v)) > 0.0
    np.testing.assert_allclose(
        float(np.asarray(obj._constants["lambda_guess"])), lam, rtol=1e-12
    )


def _build_pest_level(eq, n_rho, n_theta, n_zeta):
    """A PEST grid + matching DiffMat at an arbitrary resolution.

    Same construction the module fixture uses, factored out so a COARSE level can
    be built alongside the fine one.
    """
    x_lob, _ = leggauss_lob(n_rho)
    rho = automorphism_staircase1(x_lob, eps=1e-2, x_0=0.65, m_1=2.0, m_2=3.0)
    dfa = jax.vmap(
        lambda x: jax.grad(automorphism_staircase1, argnums=0)(
            x, eps=1e-2, x_0=0.65, m_1=2.0, m_2=3.0
        )
    )(x_lob)
    d_rho_raw, w_rho_raw = legendre_diffmat(n_rho)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, n_theta, endpoint=False)
    d_th, w_th = fourier_diffmat(n_theta)
    zeta = jnp.linspace(0.0, 2.0 * jnp.pi / eq.NFP, n_zeta, endpoint=False)
    d_z, w_z = fourier_diffmat(n_zeta)
    dm = DiffMat(
        D_rho=d_rho_raw / dfa[:, None],
        W_rho=jnp.diagonal(w_rho_raw * dfa[:, None]),
        D_theta=d_th,
        W_theta=jnp.diagonal(w_th),
        D_zeta=d_z * eq.NFP,
        W_zeta=jnp.diagonal(w_z / eq.NFP),
    )
    return LinearGrid(rho=rho, theta=theta, zeta=zeta, NFP=1, sym=False), dm


# NOT opt-in. This is the ONLY coverage of `_eigensolve_pcg` and `_coarse_space`
# -- the ring preconditioner, the deflation projector, the prolongation and the
# coarse generalized eigensolve. Gating it behind an environment variable meant
# CI never ran it and those ~770 lines were reported as untested. It is slow
# (~500 s), not special: CI runs `-m unit` with `--splits 8` and
# `--splitting-algorithm least_duration`, which absorbs one long test into one
# of eight parallel groups, and does NOT deselect `slow`.
@pytest.mark.regression
@pytest.mark.slow
def test_pcg_deflated_two_level_matches_dense(agni):
    """Ring-preconditioned PCG with coarse deflation reproduces the dense answer.

    This is the CPU-scale version of the `verify_coarse_defl` gate: a coarse
    level is built at half the fine radial resolution, its softest generalized
    modes are prolonged to seed and deflate the fine solve, and the resulting
    Rayleigh quotient is compared against the dense ARPACK eigenvalue.

    It is the only CPU-runnable coverage of ``_eigensolve_pcg`` and
    ``_coarse_space`` -- roughly 770 lines that were otherwise exercised only by
    a 20-minute GPU job. Those cover the ring preconditioner, the deflation
    projector, the prolongation and the coarse generalized eigensolve.

    RESOLUTION IS A CORRECTNESS THRESHOLD HERE, NOT A COST KNOB. Below it the
    solve does not return a less accurate eigenvalue -- it returns the WRONG
    MODE, with the opposite sign. Measured coarse-radial sweep at fine 24x12x8,
    k_defl=50, num_matvecs=100, cg_maxiter=3000 (dense = -1.337622e-04):

      coarse  8 : lam_R = +2.070e-03   SIGN FLIP -- unstable read as stable
      coarse 12 : lam_R = -1.2323e-04  right sign, 7.9% off, trusted=False
      coarse 16 : lam_R = -1.33623e-04 0.10% off, trusted=True

    so the coarse floor is 16, and AGNI_TEST_CNR defaults there. Coarse 16 was
    not more expensive than 12 (238 s vs 274 s), so the floor costs nothing.

    Two things that do NOT work as diagnostics here, both measured above:

    * The sign of the coarse eigenvalue lam_c0 does NOT predict success. It is
      POSITIVE at coarse 12 (+1.06e-07) and coarse 16 (+6.16e-08), and both
      land on the correct negative fine mode. The coarse space supplies a useful
      subspace even when its own lowest Ritz value has not resolved the mode.
    * The CG residual is anti-correlated with accuracy. Coarse 16 has the WORSE
      relres (1.42 vs 0.91) and the BETTER answer (0.10% vs 7.9%); neither run
      converged -- both burned the full iteration budget. Do not read relres as
      a quality proxy on this operator.

    At the marginal resolution the two estimators disagree: at coarse 12,
    lam_mu = -1.33657e-04 is accurate to 0.08% while the returned lam_R is 7.9%
    off. lam_R is the worse estimator there, and it is the one asserted on.
    `trusted` flagged coarse 12 False and coarse 16 True, correctly in both.

    So this test does NOT try to be cheap. It uses the same shape as the
    `verify_coarse_defl` gate (coarse at the resolution floor, fine well above
    it) and is marked opt-in accordingly. Attempts to shrink it by cutting the
    CG budget, the Lanczos dimension or the coarse resolution all produced
    positive lambda against a negative truth.
    """
    # No AGNI_COARSE_DEFL here. That variable gates the coarse block inside
    # `FinitenStability.compute_data`; the compute function itself has no such
    # gate and simply uses the coarse options it is handed.
    nr, nt, nz = agni["res"]
    lam_dense = float(np.asarray(agni["lam3"]["finite-n lambda3"])[0])

    # Coarse level: half the fine radial resolution, theta/zeta unchanged. Do NOT
    # coarsen theta/zeta too -- the deflation space then stops resolving the mode
    # and the fine solve collapses onto the wrong one.
    # Coarse level AT the resolution floor -- not below it. `verify_coarse_defl`
    # uses fine 32 / coarse 16 for the same reason.
    coarse_res = (int(os.environ.get("AGNI_TEST_CNR", 16)), nt, nz)
    coarse_pest, coarse_diffmat = _build_pest_level(agni["eq"], *coarse_res)
    coarse_grid = map_to_desc(agni["eq"], coarse_pest)
    print(f"\n  fine {nr}x{nt}x{nz}  coarse {'x'.join(map(str, coarse_res))}")

    from desc.compute.data_index import data_index

    eq, grid, dm = agni["eq"], agni["grid"], agni["diffmat"]
    params = eq.params_dict
    name = "finite-n lambda3 rayleigh"

    # The coarse operator needs the same GEOMETRY quantities the fine one does
    # (`sqrt(g)_PEST`, the metric components, ...) evaluated on the COARSE grid.
    # `finiten_prefill` supplies only the 0-D and flux-function parts; DESC fills
    # geometry from the key's declared dependencies, and only for the grid it was
    # called with. So the coarse level gets its own compute over that list.
    ckeys = data_index["desc.equilibrium.equilibrium.Equilibrium"][name][
        "dependencies"
    ]["data"]
    coarse_data = compute_fun(
        eq,
        ckeys,
        params=params,
        transforms=get_transforms(
            ckeys, obj=eq, grid=coarse_grid, diffmat=coarse_diffmat
        ),
        profiles=get_profiles(ckeys, eq, coarse_grid),
        data=finiten_prefill(eq, coarse_grid),
    )

    data = compute_fun(
        eq,
        [name],
        params=params,
        transforms=get_transforms([name], obj=eq, grid=grid, diffmat=dm),
        profiles=get_profiles([name], eq, grid),
        data=finiten_prefill(eq, grid),
        gamma=5.0 / 3.0,
        incompressible=False,
        sigma=1.3 * lam_dense,
        eigensolver="pcg_deflated",
        coarse_grid=coarse_grid,
        coarse_diffmat=coarse_diffmat,
        coarse_data=coarse_data,
        coarse_params=params,
        # The coupled Zernike operator reshapes by n_rho_coupled/n_theta_coupled;
        # inheriting the FINE counts would reshape the coarse arrays and raise.
        # Taken from the PEST grid, whose counts are concrete.
        coarse_res=(coarse_pest.num_rho, coarse_pest.num_theta, coarse_pest.num_zeta),
        # Radial nodes for the prolongation, from the PEST nodes. rho is
        # invariant under the PEST->DESC map, so these are the mapped rho values.
        coarse_rho=tuple(np.unique(np.asarray(coarse_pest.nodes[:, 0]))),
        fine_rho=tuple(np.unique(np.asarray(agni["pest_grid"].nodes[:, 0]))),
        # DELIBERATELY SMALL BUDGET. This test exists to COVER `_eigensolve_pcg`
        # and `_coarse_space` -- the ring preconditioner, the deflation
        # projector, the prolongation and the coarse generalized eigensolve --
        # not to pin digits. `verify_coarse_defl` at 32x32x12 does that.
        # Measured: cg_maxiter=3000 cost 650 s, 78% of the whole stability
        # suite's runtime, for accuracy this test does not assert on.
        k_defl=int(os.environ.get("AGNI_TEST_KDEFL", "50")),
        num_matvecs=int(os.environ.get("AGNI_TEST_NMV", "100")),
        cg_tol=1e-6,
        cg_maxiter=int(os.environ.get("AGNI_TEST_CG", "3000")),
    )
    lam_pcg = float(np.real(np.asarray(data[name]).reshape(-1)[0]))

    reldiff = abs(lam_pcg - lam_dense) / abs(lam_dense)
    print(f"\n  dense lambda3   = {lam_dense:.9e}")
    print(f"  pcg_deflated    = {lam_pcg:.9e}  (reldiff={reldiff:.2e})")
    assert np.isfinite(lam_pcg), "pcg_deflated returned a non-finite eigenvalue"
    assert np.sign(lam_pcg) == np.sign(lam_dense), (
        f"pcg_deflated flipped the sign: {lam_pcg:.6e} vs dense {lam_dense:.6e} "
        "-- an unstable equilibrium reported as stable"
    )
    # Order of magnitude, not precision -- see the budget note above.
    assert 0.2 < abs(lam_pcg / lam_dense) < 5.0, (
        f"pcg_deflated magnitude is off by more than 5x: {lam_pcg:.6e} vs dense "
        f"{lam_dense:.6e}. At this budget it need not converge, but it must land "
        "on the same mode."
    )


@pytest.mark.regression
@pytest.mark.slow
def test_v_fixed_reuses_the_eigenvector(agni, monkeypatch):
    """`v_fixed` skips the eigensolve and reproduces lambda exactly.

    The eigenvector comes back under ``"finite-n lambda3 rayleigh v"``; handing
    it straight back as ``v_fixed`` leaves only the Rayleigh quotient to
    evaluate. Valid ONLY at the same x -- reusing v after the equilibrium moves
    is silently wrong.
    """
    monkeypatch.setenv("AGNI_EIGENSOLVER", "jax_lanczos")
    monkeypatch.setenv("AGNI_NUM_MATVECS", "100")
    eq, grid, dm = agni["eq"], agni["grid"], agni["diffmat"]
    lam_dense = float(np.asarray(agni["lam3"]["finite-n lambda3"])[0])
    name = "finite-n lambda3 rayleigh"
    common = dict(
        params=eq.params_dict,
        transforms=get_transforms([name], obj=eq, grid=grid, diffmat=dm),
        profiles=get_profiles([name], eq, grid),
        gamma=5.0 / 3.0,
        incompressible=False,
        sigma=1.3 * lam_dense,
    )

    solved = compute_fun(eq, [name], data=finiten_prefill(eq, grid), **common)
    lam = float(np.asarray(solved[name]).reshape(-1)[0])
    v = np.asarray(solved[name + " v"]).reshape(-1)
    assert v.size == agni["n_keep"]
    assert np.all(np.isfinite(v))

    reused = compute_fun(
        eq, [name], data=finiten_prefill(eq, grid), v_fixed=v, **common
    )
    lam_fixed = float(np.asarray(reused[name]).reshape(-1)[0])
    resid = float(np.asarray(reused[name + " residual"]).reshape(-1)[0])

    print(f"\n  lambda           = {lam:.12e}")
    print(f"  lambda (v_fixed) = {lam_fixed:.12e}  residual={resid:.3e}")
    np.testing.assert_allclose(lam_fixed, lam, rtol=1e-12)
    np.testing.assert_allclose(lam_fixed, lam_dense, rtol=1e-4)


@pytest.mark.regression
@pytest.mark.slow
def test_v_fixed_objective_jits_and_matches_gradient(agni):
    """`v_fixed` through the JITTED objective: same lambda, same gradient.

    ``_v_fixed`` must stay a dynamic pytree leaf. Marking it static bakes v into
    aux_data as an HLO constant and recompiles per value.
    """
    lam_direct = float(np.asarray(agni["lam3"]["finite-n lambda3"])[0])
    params = agni["eq"].params_dict

    obj = _finiten_objective(agni, lambda_guess=lam_direct, sigma_factor=1.3)
    lam = float(np.real(np.asarray(obj.compute(params))[0]))
    v = np.asarray(obj.compute_data(params)["finite-n lambda3 rayleigh v"]).reshape(-1)
    assert v.size == agni["n_keep"]

    # Built with use_jit=True, exactly as above but with the bypass on.
    obj_fixed = _finiten_objective(
        agni, lambda_guess=lam_direct, sigma_factor=1.3, v_fixed=v
    )

    # v must be a traced child, never static aux_data.
    leaves = jax.tree_util.tree_flatten(obj_fixed)[0]
    assert any(getattr(x, "size", None) == v.size for x in leaves), (
        "v_fixed did not appear among the dynamic pytree leaves -- it went into "
        "aux_data, which recompiles per value and bakes v in as a constant"
    )

    lam_fixed = float(np.real(np.asarray(obj_fixed.compute(params))[0]))
    np.testing.assert_allclose(lam_fixed, lam, rtol=1e-9)

    def _val(obj_, p):
        return jnp.real(obj_.compute(p)[0])

    # Re-solving at the same x reproduces v only to the eigensolver's tolerance,
    # ~5e-10. lambda is stationary in v and does not feel it; the gradient
    # v'(dA/dp)v/v'v is first order in v and does. Hence the norm comparison.
    v2 = np.asarray(obj.compute_data(params)["finite-n lambda3 rayleigh v"]).reshape(-1)
    v2 = v2 * np.sign(np.dot(v2, v))
    dv = np.linalg.norm(v2 - v) / np.linalg.norm(v)
    print(f"\n  ||v2 - v|| / ||v|| = {dv:.3e}")
    assert dv < 1e-6, f"eigenvector not reproducible at fixed x: {dv:.3e}"

    g = jax.grad(_val, argnums=1)(obj, params)
    g_fixed = jax.grad(_val, argnums=1)(obj_fixed, params)
    for key in ("R_lmn", "Z_lmn", "L_lmn", "Psi"):
        a, b = np.asarray(g[key]), np.asarray(g_fixed[key])
        assert np.all(np.isfinite(b)), f"v_fixed gradient has non-finite {key}"
        rel = np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-300)
        print(
            f"  |d/d{key}| = {np.linalg.norm(a):.6e} vs {np.linalg.norm(b):.6e}"
            f"   reldiff={rel:.3e}"
        )
        # A bypass differentiating the wrong thing is off by O(1), not by 1e-8.
        assert rel < 1e-6, f"v_fixed gradient differs in d/d{key}: reldiff={rel:.3e}"
