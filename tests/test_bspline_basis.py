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
