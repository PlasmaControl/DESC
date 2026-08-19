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

    deps = data_index["desc.equilibrium.equilibrium.Equilibrium"][
        "finite-n lambda3 rayleigh"
    ]["dependencies"]["data"]
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


@pytest.mark.unit
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

    Recorded reference: the restricted assembler matched dense to <= 2.8e-16 on
    all 192 rings at GJ 16x32x12 (precond_stage2/VERIFICATION.md).
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
    data = agni["eq"].compute(deps, grid=agni["grid"], diffmat=agni["diffmat"])
    transforms = {"grid": agni["grid"], "diffmat": agni["diffmat"]}
    kw = {k: v for k, v in agni["kw"].items() if k not in ("grid", "diffmat")}
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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

    The tolerance is loose on purpose. This is an inexact iterative solve at a
    small CG budget, so the point is that the deflated path lands on the SAME
    MODE with the same sign and the right magnitude -- not that it matches to
    machine precision. `verify_coarse_defl` at 32x32x12 is what pins the digits.
    """
    nr, nt, nz = agni["res"]
    lam_dense = float(np.asarray(agni["lam3"]["finite-n lambda3"])[0])

    coarse_grid, coarse_diffmat = _build_pest_level(agni["eq"], max(4, nr // 2), nt, nz)

    obj = _finiten_objective(
        agni,
        lambda_guess=lam_dense,
        sigma_factor=1.3,
        coarse_grid=coarse_grid,
        coarse_diffmat=coarse_diffmat,
        eigensolver="pcg_deflated",
        k_defl=8,
        num_matvecs=40,
        cg_tol=1e-8,
        cg_maxiter=3000,
    )
    lam_pcg = float(np.real(np.asarray(obj.compute(obj.things[0].params_dict))[0]))

    reldiff = abs(lam_pcg - lam_dense) / abs(lam_dense)
    print(f"\n  dense lambda3   = {lam_dense:.9e}")
    print(f"  pcg_deflated    = {lam_pcg:.9e}  (reldiff={reldiff:.2e})")
    assert np.isfinite(lam_pcg), "pcg_deflated returned a non-finite eigenvalue"
    assert np.sign(lam_pcg) == np.sign(lam_dense), (
        f"pcg_deflated flipped the sign: {lam_pcg:.6e} vs dense {lam_dense:.6e} "
        "-- an unstable equilibrium reported as stable"
    )
    np.testing.assert_allclose(lam_pcg, lam_dense, rtol=5e-2)
