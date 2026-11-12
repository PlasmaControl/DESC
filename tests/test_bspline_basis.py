"""Tests for the experimental B-spline x double Fourier basis."""

import numpy as np
import pytest

from desc.basis import BSplineDoubleFourierBasis
from desc.equilibrium import Equilibrium
from desc.examples import get
from desc.grid import LinearGrid, QuadratureGrid
from desc.transform import Transform


@pytest.mark.unit
def test_bspline_axis_regularity():
    """Radial functions of mode m scale as rho^|m| with the right parity."""
    p = 5
    basis = BSplineDoubleFourierBasis(L=6, M=8, N=0, degree=p)
    r = np.array([1e-3, 2e-3])
    for m in range(9):
        nl = int(basis.num_radial(m))
        y = np.asarray(basis.radial(r, np.arange(nl), np.full(nl, m)))
        nfree = (p - m) // 2 + 1 if m <= p else 0
        assert nl == 6 - 1 + nfree
        # functions not touching the axis vanish on the first interval
        np.testing.assert_allclose(y[:, nfree:], 0, atol=1e-14)
        if nfree:
            # leading power is rho^(m + 2k)
            power = np.log2(np.abs(y[1, :nfree] / y[0, :nfree]))
            assert np.all(power > m - 1e-3)
            np.testing.assert_allclose(power, np.round(power), atol=1e-2)
            assert np.all((np.round(power).astype(int) - m) % 2 == 0)
        # only the last function is nonzero at the boundary
        edge = np.asarray(basis.radial(1.0, np.arange(nl), np.full(nl, m)))[0]
        np.testing.assert_allclose(edge[:-1], 0, atol=1e-14)
        np.testing.assert_allclose(edge[-1], 1, atol=1e-14)


@pytest.mark.unit
def test_bspline_transform_methods_agree():
    """fft, direct1 and direct2 transforms give the same result."""
    basis = BSplineDoubleFourierBasis(L=4, M=4, N=2, NFP=3, sym="cos")
    c = np.random.default_rng(0).normal(size=basis.num_modes)
    grid = LinearGrid(rho=np.array([0.0, 0.3, 0.9]), M=5, N=3, NFP=3)
    t1 = Transform(grid, basis, derivs=3, method="direct1")
    for method in ["fft", "direct2"]:
        t2 = Transform(grid, basis, derivs=3, method=method)
        assert t2.method == method
        for d in [(0, 0, 0), (1, 0, 1), (2, 1, 0), (0, 1, 2)]:
            np.testing.assert_allclose(
                t2.transform(c, *d), t1.transform(c, *d), atol=1e-10
            )


@pytest.mark.unit
def test_bspline_equilibrium_solve(tmpdir_factory):
    """Fixed boundary solve with the spline basis reaches force balance."""
    ref = get("SOLOVEV")
    eq = Equilibrium(
        Psi=ref.Psi,
        surface=ref.surface,
        pressure=ref.pressure,
        iota=ref.iota,
        L=6,
        M=ref.M,
        L_grid=18,
        M_grid=ref.M_grid,
        radial_basis="bspline",
    )
    assert eq.radial_basis == "bspline"
    assert ref.radial_basis == "zernike"
    eq.solve(maxiter=30, verbose=0)
    grid = QuadratureGrid(30, 24, 0)
    names = ["<|F|>_vol", "<|grad(|B|^2)|/2mu0>_vol"]
    f = eq.compute(names, grid=grid)
    assert f["<|F|>_vol"] / f["<|grad(|B|^2)|/2mu0>_vol"] < 1e-5
    # axis and boundary agree with the Zernike solution
    g = LinearGrid(rho=np.array([0.0, 1.0]), M=8)
    np.testing.assert_allclose(
        eq.compute("R", grid=g)["R"], ref.compute("R", grid=g)["R"], atol=1e-3
    )
    # save / load round trip keeps the basis
    path = tmpdir_factory.mktemp("bspline").join("eq.h5")
    eq.save(path)
    import desc.io

    eq2 = desc.io.load(path)
    assert eq2.radial_basis == "bspline"
    np.testing.assert_allclose(eq2.R_lmn, eq.R_lmn)


@pytest.mark.unit
def test_bspline_axis_power():
    """With a rho^m prefactor every mode scales as rho^|m| near the axis."""
    for degree in [5, 7]:
        basis = BSplineDoubleFourierBasis(
            L=3, M=9, N=0, degree=degree, axis_power="full"
        )
        r = np.array([1e-3, 2e-3])
        rr = np.array([0.05, 0.4, 0.9])
        for m in range(10):
            nl = int(basis.num_radial(m))
            assert nl == int(basis.num_radial(0))  # no mode loses radial functions
            l, mm = np.arange(nl), np.full(nl, m)
            y = np.asarray(basis.radial(r, l, mm))
            lead = np.argmax(np.abs(y[0]))
            np.testing.assert_allclose(np.log2(y[1, lead] / y[0, lead]), m, atol=1e-2)
            # derivatives agree with finite differences
            h = 1e-5
            for d in range(1, 4):
                fd = (
                    np.asarray(basis.radial(rr + h, l, mm, d - 1))
                    - np.asarray(basis.radial(rr - h, l, mm, d - 1))
                ) / (2 * h)
                an = np.asarray(basis.radial(rr, l, mm, d))
                np.testing.assert_allclose(fd, an, atol=1e-6 * max(1, np.abs(an).max()))


def _blocks_to_dense(st, D, E):
    """Dense matrix from the block tridiagonal storage of BSplineStructure."""
    D, E = np.asarray(D), np.asarray(E)
    n = st.num_vars
    B = np.zeros((n, n))
    idx = [np.where(st.blk == k)[0] for k in range(st.num_blocks)]
    for k in range(st.num_blocks):
        lk = st.loc[idx[k]]
        B[np.ix_(idx[k], idx[k])] = D[k][np.ix_(lk, lk)]
        if k + 1 < st.num_blocks:
            Ek = E[k][np.ix_(st.loc[idx[k + 1]], lk)]
            B[np.ix_(idx[k + 1], idx[k])] = Ek
            B[np.ix_(idx[k], idx[k + 1])] = Ek.T
    return B


@pytest.mark.unit
def test_bspline_structured_jacobian_and_banded_solve():
    """Compressed Jacobian and block tridiagonal solve agree with dense versions."""
    from desc.backend import jnp
    from desc.objectives import (
        ForceBalance,
        ObjectiveFunction,
        get_fixed_boundary_constraints,
    )
    from desc.objectives.getters import maybe_add_self_consistency
    from desc.optimize._bspline_structure import (
        CompressedJacobian,
        _block_chol_failed,
        _block_cholesky,
        _block_solve,
        trust_region_step_block,
    )
    from desc.optimize._constraint_wrappers import LinearConstraintProjection
    from desc.optimize.tr_subproblems import trust_region_step_exact_qr

    ref = get("SOLOVEV")

    def make():
        return Equilibrium(
            Psi=ref.Psi,
            surface=ref.surface,
            pressure=ref.pressure,
            iota=ref.iota,
            L=16,
            M=6,
            L_grid=48,
            M_grid=9,
            radial_basis="bspline",
            spline_axis_power="full",
        )

    eq = make()
    obj = ObjectiveFunction(ForceBalance(eq))
    cons = ObjectiveFunction(
        maybe_add_self_consistency(eq, get_fixed_boundary_constraints(eq))
    )
    lcp = LinearConstraintProjection(obj, cons)
    lcp.build(verbose=0)
    st = lcp.structure
    assert st is not None and lcp._structure_jac
    assert st.num_colors < 0.5 * lcp._dim_x_reduced
    assert st.num_blocks > 2
    x = lcp.x(eq)
    J1 = np.asarray(lcp.jac_scaled_error(x))
    Jc = lcp.jac_scaled_error_compressed(x)
    lcp._structure_jac = False
    J0 = np.asarray(lcp.jac_scaled_error(x))
    scale = np.abs(J0).max()

    # compressed Jacobian and its operations equal the dense ones
    np.testing.assert_allclose(J1, J0, atol=1e-12 * scale)
    np.testing.assert_allclose(Jc.todense(), J0, atol=1e-12 * scale)
    rng = np.random.default_rng(0)
    v = rng.normal(size=J0.shape[1])
    u = rng.normal(size=J0.shape[0])
    np.testing.assert_allclose(Jc @ v, J0 @ v, atol=1e-12 * np.abs(J0 @ v).max())
    np.testing.assert_allclose(
        Jc.rmatvec(u), J0.T @ u, atol=1e-12 * np.abs(J0.T @ u).max()
    )
    np.testing.assert_allclose(
        Jc.column_norms(), np.linalg.norm(J0, axis=0), atol=1e-12 * scale
    )

    # work with the column scaled Jacobian, as lsqtr does
    d = 1 / np.where(np.linalg.norm(J0, axis=0) > 1e-12, np.linalg.norm(J0, axis=0), 1)
    Jc = Jc.scale_columns(jnp.asarray(d))
    J = Jc.todense()
    np.testing.assert_allclose(J, J0 * d, atol=1e-6 * np.abs(J).max())

    # block tridiagonal normal equations equal the dense ones
    D, E = Jc.normal_blocks()
    B = J.T @ J
    np.testing.assert_allclose(_blocks_to_dense(st, D, E), B, atol=1e-12)

    # block Cholesky solve equals the dense solve
    real = np.arange(st.block_size)[None, :] < st.block_sizes[:, None]
    alpha = 1e-3
    Ld, Lo = _block_cholesky(D, E, jnp.where(real, alpha, 0.0))
    assert not _block_chol_failed(Ld, jnp.asarray(real))
    b = rng.normal(size=st.num_vars)
    xb = st.from_blocks(_block_solve(Ld, Lo, st.to_blocks(jnp.asarray(b))))
    xd = np.linalg.solve(B + alpha * np.eye(st.num_vars), b)
    np.testing.assert_allclose(xb, xd, atol=1e-9 * np.abs(xd).max())

    # trust region step equals the QR based one
    f = np.asarray(lcp.compute_scaled_error(x))
    Q, R = np.linalg.qr(J)
    p_newton = np.linalg.lstsq(J, -f, rcond=None)[0]
    for radius in [1e-3, 1e-1]:
        p, hit, _ = trust_region_step_block(jnp.asarray(f), Jc, D, E, radius, 1e-6)
        pq, hitq, _ = trust_region_step_exact_qr(
            jnp.asarray(p_newton), jnp.asarray(Q.T @ f), jnp.asarray(R), radius
        )
        assert hit and hitq
        np.testing.assert_allclose(p, pq, atol=1e-8 * radius)
    # unconstrained step: J.T@J is singular to working precision here, so the
    # step can differ from the min-norm one in the near null space, but it has
    # to solve the least squares problem as well
    p, hit, alpha = trust_region_step_block(jnp.asarray(f), Jc, D, E, 1e10, 1e-6)
    assert not hit and alpha == 0
    r_block = np.linalg.norm(J @ np.asarray(p) + f)
    r_lstsq = np.linalg.norm(J @ p_newton + f)
    assert r_block < 1e-4 * np.linalg.norm(f)
    assert r_block < 10 * r_lstsq + 1e-10 * np.linalg.norm(f)

    # exactly singular J.T@J (a zero column): the factorization fails at alpha=0,
    # falls back to a small shift, and the step still solves the problem
    j0 = st.num_vars // 2
    data = np.array(Jc.data)
    data[:, st.color[j0]] = np.where(
        np.asarray(st.colmap)[st.row_elem, st.color[j0]] == j0,
        0.0,
        data[:, st.color[j0]],
    )
    Js = CompressedJacobian(st, jnp.asarray(data))
    Jd = Js.todense()
    assert np.all(Jd[:, j0] == 0)
    Ds, Es = Js.normal_blocks()
    Ld, _ = _block_cholesky(Ds, Es, jnp.where(real, 0.0, 0.0))
    assert _block_chol_failed(Ld, jnp.asarray(real))
    fs = f - Jd @ (0.1 * rng.normal(size=st.num_vars))  # consistent part plus f
    p, hit, alpha = trust_region_step_block(jnp.asarray(fs), Js, Ds, Es, 1e10, 1e-6)
    p = np.asarray(p)
    assert np.all(np.isfinite(p)) and not hit
    assert abs(p[j0]) < 1e-8 * np.linalg.norm(p)
    pl = np.linalg.lstsq(Jd, -fs, rcond=None)[0]
    r_block = np.linalg.norm(Jd @ p + fs)
    assert r_block < 10 * np.linalg.norm(Jd @ pl + fs) + 1e-10 * np.linalg.norm(fs)
    # and with a trust region, the same step as QR
    Q, R = np.linalg.qr(Jd)
    for radius in [1e-3, 1e-1]:
        p, hit, _ = trust_region_step_block(jnp.asarray(fs), Js, Ds, Es, radius, 1e-6)
        pq, hitq, _ = trust_region_step_exact_qr(
            jnp.asarray(pl), jnp.asarray(Q.T @ fs), jnp.asarray(R), radius
        )
        assert hit and hitq
        np.testing.assert_allclose(p, pq, atol=1e-8 * radius)

    # a short solve with the compressed Jacobian and block solve converges
    grid = QuadratureGrid(30, 18, 0)
    names = ["<|F|>_vol", "<|grad(|B|^2)|/2mu0>_vol"]
    for tr in ["qr", "banded"]:
        eq = make()
        eq.solve(maxiter=30, verbose=0, options={"tr_method": tr})
        f = eq.compute(names, grid=grid)
        assert f["<|F|>_vol"] / f["<|grad(|B|^2)|/2mu0>_vol"] < 1e-4
