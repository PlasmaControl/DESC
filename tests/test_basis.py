"""Tests for basis classes and evaluation functions."""

import mpmath
import numpy as np
import pytest

from desc.backend import jnp
from desc.basis import (
    ChebyshevDoubleFourierBasis,
    ChebyshevPolynomial,
    DoubleFourierSeries,
    FourierSeries,
    FourierZernikeBasis,
    PowerSeries,
    ZernikePolynomial,
    _jacobi,
    chebyshev,
    fourier,
    polyder_vec,
    polyval_vec,
    powers,
    zernike_radial,
    zernike_radial_coeffs,
    zernike_radial_poly,
)
from desc.derivatives import Derivative
from desc.grid import LinearGrid


class TestBasis:
    """Test Basis class."""

    @pytest.mark.unit
    def test_polyder(self):
        """Test polyder_vec function."""
        p0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
        p1 = polyder_vec(p0, 1)
        p2 = polyder_vec(p0, 2, exact=True)

        correct_p1 = np.array([[0, 2, 0], [0, 0, 1], [0, 0, 0], [0, 2, 1]])
        correct_p2 = np.array([[0, 0, 2], [0, 0, 0], [0, 0, 0], [0, 0, 2]])

        np.testing.assert_allclose(p1, correct_p1, atol=1e-8)
        np.testing.assert_allclose(p2, correct_p2, atol=1e-8)

    @pytest.mark.unit
    def test_polyval(self):
        """Test polyval_vec function."""
        p = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
        x = np.linspace(0, 1, 11)

        correct_vals = np.array([x**2, x, np.ones_like(x), x**2 + x + 1])
        values = polyval_vec(p, x)

        np.testing.assert_allclose(values, correct_vals, atol=1e-8)

    @pytest.mark.unit
    def test_zernike_coeffs(self):
        """Test calculation of zernike polynomial coefficients."""
        basis = FourierZernikeBasis(L=40, M=40, N=0, spectral_indexing="ansi")
        l, m = basis.modes[:, :2].T
        coeffs = zernike_radial_coeffs(l, m, exact=False)
        assert coeffs.dtype == np.int64
        basis = FourierZernikeBasis(L=60, M=30, N=0, spectral_indexing="fringe")
        l, m = basis.modes[:, :2].T
        coeffs = zernike_radial_coeffs(l, m, exact=False)
        assert coeffs.dtype == np.float64

    @pytest.mark.unit
    @pytest.mark.slow
    def test_polyval_exact(self):
        """Test "exact" polynomial evaluation using extended precision."""
        basis = FourierZernikeBasis(L=80, M=40, N=0)
        l, m = basis.modes[::40, 0], basis.modes[::40, 1]
        coeffs = zernike_radial_coeffs(l, m, exact=True)
        grid = LinearGrid(L=20)
        r = grid.nodes[:, 0]
        mpmath.mp.dps = 100
        exactf = np.array(
            [np.asarray(mpmath.polyval(list(ci), r), dtype=float) for ci in coeffs]
        ).T
        exactdf = np.array(
            [
                np.asarray(mpmath.polyval(list(ci), r), dtype=float)
                for ci in polyder_vec(coeffs, 1, exact=True)
            ]
        ).T
        exactddf = np.array(
            [
                np.asarray(mpmath.polyval(list(ci), r), dtype=float)
                for ci in polyder_vec(coeffs, 2, exact=True)
            ]
        ).T
        exactdddf = np.array(
            [
                np.asarray(mpmath.polyval(list(ci), r), dtype=float)
                for ci in polyder_vec(coeffs, 3, exact=True)
            ]
        ).T

        mpmath.mp.dps = 15
        approx1f = zernike_radial(r[:, np.newaxis], l, m)
        approx1df = zernike_radial(r[:, np.newaxis], l, m, dr=1)
        approx1ddf = zernike_radial(r[:, np.newaxis], l, m, dr=2)
        approx1dddf = zernike_radial(r[:, np.newaxis], l, m, dr=3)
        approx2f = zernike_radial_poly(r[:, np.newaxis], l, m)
        approx2df = zernike_radial_poly(r[:, np.newaxis], l, m, dr=1)
        approx2ddf = zernike_radial_poly(r[:, np.newaxis], l, m, dr=2)
        approx2dddf = zernike_radial_poly(r[:, np.newaxis], l, m, dr=3)

        np.testing.assert_allclose(approx1f, exactf, atol=1e-12)
        np.testing.assert_allclose(approx1df, exactdf, atol=1e-12)
        np.testing.assert_allclose(approx1ddf, exactddf, atol=1e-12)
        np.testing.assert_allclose(approx1dddf, exactdddf, atol=1e-12)
        np.testing.assert_allclose(approx2f, exactf, atol=1e-12)
        np.testing.assert_allclose(approx2df, exactdf, atol=1e-12)
        np.testing.assert_allclose(approx2ddf, exactddf, atol=1e-12)
        np.testing.assert_allclose(approx2dddf, exactdddf, atol=1e-12)

    @pytest.mark.unit
    def test_powers(self):
        """Test powers function for power series evaluation."""
        l = np.array([0, 1, 2])
        r = np.linspace(0, 1, 11)  # rho coordinates

        correct_vals = np.array([np.ones_like(r), r, r**2]).T
        correct_ders = np.array([np.zeros_like(r), np.ones_like(r), 2 * r]).T

        values = powers(r, l, dr=0)
        derivs = powers(r, l, dr=1)

        np.testing.assert_allclose(values, correct_vals, atol=1e-8)
        np.testing.assert_allclose(derivs, correct_ders, atol=1e-8)

    @pytest.mark.unit
    def test_chebyshev(self):
        """Test chebyshev function for Chebyshev polynomial evaluation."""
        l = np.array([0, 1, 2])
        r = np.linspace(0, 1, 11)  # rho coordinates

        correct_vals = np.array([np.ones_like(r), 2 * r - 1, 8 * r**2 - 8 * r + 1]).T
        values = chebyshev(r[:, np.newaxis], l, dr=0)
        np.testing.assert_allclose(values, correct_vals, atol=1e-8)

        with pytest.raises(NotImplementedError):
            chebyshev(r[:, np.newaxis], l, dr=1)

    @pytest.mark.unit
    def test_zernike_radial(self):  # noqa: C901
        """Test zernike_radial function, comparing to analytic formulas."""
        # https://en.wikipedia.org/wiki/Zernike_polynomials#Radial_polynomials

        def Z3_1(x, dx=0):
            if dx == 0:
                return 3 * x**3 - 2 * x
            if dx == 1:
                return 9 * x**2 - 2
            if dx == 2:
                return 18 * x
            if dx == 3:
                return np.full_like(x, 18)
            if dx >= 4:
                return np.zeros_like(x)

        def Z4_2(x, dx=0):
            if dx == 0:
                return 4 * x**4 - 3 * x**2
            if dx == 1:
                return 16 * x**3 - 6 * x
            if dx == 2:
                return 48 * x**2 - 6
            if dx == 3:
                return 96 * x
            if dx == 4:
                return np.full_like(x, 96)
            if dx >= 5:
                return np.zeros_like(x)

        def Z6_2(x, dx=0):
            if dx == 0:
                return 15 * x**6 - 20 * x**4 + 6 * x**2
            if dx == 1:
                return 90 * x**5 - 80 * x**3 + 12 * x
            if dx == 2:
                return 450 * x**4 - 240 * x**2 + 12
            if dx == 3:
                return 1800 * x**3 - 480 * x
            if dx == 4:
                return 5400 * x**2 - 480
            if dx == 5:
                return 10800 * x
            if dx == 6:
                return np.full_like(x, 10800)
            if dx >= 7:
                return np.zeros_like(x)

        l = np.array([3, 4, 6, 4])
        m = np.array([1, 2, 2, 2])
        r = np.linspace(0, 1, 11)  # rho coordinates
        max_dr = 4
        desired = {
            dr: np.array([Z3_1(r, dr), Z4_2(r, dr), Z6_2(r, dr), Z4_2(r, dr)]).T
            for dr in range(max_dr + 1)
        }
        radial = {
            dr: zernike_radial(r[:, np.newaxis], l, m, dr) for dr in range(max_dr + 1)
        }
        radial_poly = {
            dr: zernike_radial_poly(r[:, np.newaxis], l, m, dr)
            for dr in range(max_dr + 1)
        }
        for dr in range(max_dr + 1):
            np.testing.assert_allclose(radial[dr], desired[dr], err_msg=dr)
            np.testing.assert_allclose(radial_poly[dr], desired[dr], err_msg=dr)

    @pytest.mark.unit
    def test_fourier(self):
        """Test Fourier series evaluation."""
        m = np.array([-1, 0, 1])
        t = np.linspace(0, 2 * np.pi, 8, endpoint=False)  # theta coordinates

        correct_vals = np.array([np.sin(t), np.ones_like(t), np.cos(t)]).T
        correct_ders = np.array([np.cos(t), np.zeros_like(t), -np.sin(t)]).T

        values = fourier(t[:, np.newaxis], m, dt=0)
        derivs = fourier(t[:, np.newaxis], m, dt=1)

        np.testing.assert_allclose(values, correct_vals, atol=1e-8)
        np.testing.assert_allclose(derivs, correct_ders, atol=1e-8)

    @pytest.mark.unit
    def test_power_series(self):
        """Test PowerSeries evaluation."""
        grid = LinearGrid(rho=11)
        r = grid.nodes[:, 0]  # rho coordinates

        correct_vals = np.array([np.ones_like(r), r, r**2]).T
        correct_ders = np.array([np.zeros_like(r), np.ones_like(r), 2 * r]).T

        basis = PowerSeries(L=2, sym=False)
        values = basis.evaluate(grid.nodes, derivatives=np.array([0, 0, 0]))
        derivs = basis.evaluate(grid.nodes, derivatives=np.array([1, 0, 0]))

        np.testing.assert_allclose(values, correct_vals, atol=1e-8)
        np.testing.assert_allclose(derivs, correct_ders, atol=1e-8)

    @pytest.mark.unit
    def test_double_fourier(self):
        """Test DoubleFourierSeries evaluation."""
        grid = LinearGrid(M=2, N=2)
        t = grid.nodes[:, 1]  # theta coordinates
        z = grid.nodes[:, 2]  # zeta coordinates

        correct_vals = np.array(
            [
                np.sin(t) * np.sin(z),
                np.sin(z),
                np.cos(t) * np.sin(z),
                np.sin(t),
                np.ones_like(t),
                np.cos(t),
                np.sin(t) * np.cos(z),
                np.cos(z),
                np.cos(t) * np.cos(z),
            ]
        ).T

        basis = DoubleFourierSeries(M=1, N=1)
        values = basis.evaluate(grid.nodes, derivatives=np.array([0, 0, 0]))

        np.testing.assert_allclose(values, correct_vals, atol=1e-8)

    @pytest.mark.unit
    def test_change_resolution(self):
        """Test change_resolution function."""
        ps = PowerSeries(L=4, sym=False)
        ps.change_resolution(L=6)
        assert ps.num_modes == 7

        fs = FourierSeries(N=3)
        fs.change_resolution(N=2)
        assert fs.num_modes == 5

        dfs = DoubleFourierSeries(M=3, N=4)
        dfs.change_resolution(M=2, N=1)
        assert dfs.num_modes == 15

        zpa = ZernikePolynomial(L=0, M=3, spectral_indexing="ansi")
        zpa.change_resolution(L=3, M=3)
        assert zpa.num_modes == 10

        zpf = ZernikePolynomial(L=0, M=3, spectral_indexing="fringe")
        zpf.change_resolution(L=6, M=3)
        assert zpf.num_modes == 16

        cdf = ChebyshevDoubleFourierBasis(L=2, M=2, N=0)
        cdf.change_resolution(L=3, M=2, N=1)
        assert cdf.num_modes == 60

        fz = FourierZernikeBasis(L=3, M=3, N=0)
        fz.change_resolution(L=3, M=3, N=1)
        assert fz.num_modes == 30

    @pytest.mark.unit
    def test_repr(self):
        """Test string representation of basis classes."""
        fz = FourierZernikeBasis(L=6, M=3, N=0)
        s = str(fz)
        assert "FourierZernikeBasis" in s
        assert "ansi" in s
        assert "L=6" in s
        assert "M=3" in s
        assert "N=0" in s

    @pytest.mark.unit
    def test_zernike_indexing(self):
        """Test what modes are in the basis for given resolution and indexing."""
        basis = ZernikePolynomial(L=8, M=4, spectral_indexing="ansi")
        assert (basis.modes == [8, 4, 0]).all(axis=1).any()
        assert not (basis.modes == [8, 8, 0]).all(axis=1).any()

        basis = ZernikePolynomial(L=10, M=4, spectral_indexing="fringe")
        assert (basis.modes == [10, 0, 0]).all(axis=1).any()
        assert not (basis.modes == [10, 2, 0]).all(axis=1).any()

        basis = FourierZernikeBasis(L=8, M=4, N=0, spectral_indexing="ansi")
        assert (basis.modes == [8, 4, 0]).all(axis=1).any()
        assert not (basis.modes == [8, 8, 0]).all(axis=1).any()

        basis = FourierZernikeBasis(L=10, M=4, N=0, spectral_indexing="fringe")
        assert (basis.modes == [10, 0, 0]).all(axis=1).any()
        assert not (basis.modes == [10, 2, 0]).all(axis=1).any()

    @pytest.mark.unit
    def test_derivative_not_in_basis_zeros(self):
        """Test that d/dx = 0 when x is not in the basis."""
        nodes = np.random.random((10, 3))

        basis = PowerSeries(L=3)
        ft = basis.evaluate(nodes, derivatives=[0, 1, 0])
        fz = basis.evaluate(nodes, derivatives=[0, 0, 1])
        assert np.all(ft == 0)
        assert np.all(fz == 0)

        basis = FourierSeries(N=4)
        fr = basis.evaluate(nodes, derivatives=[1, 0, 0])
        ft = basis.evaluate(nodes, derivatives=[0, 1, 0])
        assert np.all(fr == 0)
        assert np.all(ft == 0)

        basis = DoubleFourierSeries(M=2, N=4)
        fr = basis.evaluate(nodes, derivatives=[1, 0, 0])
        assert np.all(fr == 0)

        basis = ZernikePolynomial(L=2, M=3)
        fz = basis.evaluate(nodes, derivatives=[0, 0, 1])
        assert np.all(fz == 0)

    @pytest.mark.unit
    def test_basis_resolutions_assert_integers(self):
        """Test that basis modes are asserted as integers."""
        with pytest.raises(ValueError):
            _ = PowerSeries(L=3.0)

        with pytest.raises(ValueError):
            _ = FourierSeries(N=3.0)
        with pytest.raises(ValueError):
            _ = FourierSeries(N=3, NFP=1.0)

        with pytest.raises(ValueError):
            _ = DoubleFourierSeries(M=3.0, N=2)
        with pytest.raises(ValueError):
            _ = DoubleFourierSeries(M=3, N=2.0)
        with pytest.raises(ValueError):
            _ = DoubleFourierSeries(M=3, N=2, NFP=1.0)

        with pytest.raises(ValueError):
            _ = ZernikePolynomial(L=3.0, M=1)
        with pytest.raises(ValueError):
            _ = ZernikePolynomial(L=3, M=1.0)

        with pytest.raises(ValueError):
            _ = ChebyshevPolynomial(L=3.0)

        with pytest.raises(ValueError):
            _ = FourierZernikeBasis(L=3.0, M=1, N=1)
        with pytest.raises(ValueError):
            _ = FourierZernikeBasis(L=3, M=1.0, N=1)
        with pytest.raises(ValueError):
            _ = FourierZernikeBasis(L=3, M=1, N=1.0)
        with pytest.raises(ValueError):
            _ = FourierZernikeBasis(L=3, M=1, N=1, NFP=1.0)

        with pytest.raises(ValueError):
            _ = ChebyshevDoubleFourierBasis(L=3.0, M=1, N=1)
        with pytest.raises(ValueError):
            _ = ChebyshevDoubleFourierBasis(L=3, M=1.0, N=1)
        with pytest.raises(ValueError):
            _ = ChebyshevDoubleFourierBasis(L=3, M=1, N=1.0)
        with pytest.raises(ValueError):
            _ = ChebyshevDoubleFourierBasis(L=3, M=1, N=1, NFP=1.0)

    @pytest.mark.unit
    def test_basis_hash(self):
        """Test that all basis classes can be hashable."""
        all_bases = []
        all_types = []
        all_bases.append([PowerSeries(L=3), PowerSeries(L=3)])
        all_bases.append([FourierSeries(N=3), FourierSeries(N=3)])
        all_bases.append([DoubleFourierSeries(M=3, N=3), DoubleFourierSeries(M=3, N=3)])
        all_bases.append([ZernikePolynomial(L=3, M=3), ZernikePolynomial(L=3, M=3)])
        all_bases.append(
            [FourierZernikeBasis(L=3, M=3, N=3), FourierZernikeBasis(L=3, M=3, N=3)]
        )
        all_bases.append(
            [
                ChebyshevDoubleFourierBasis(L=3, M=3, N=3),
                ChebyshevDoubleFourierBasis(L=3, M=3, N=3),
            ]
        )
        all_bases.append([ChebyshevPolynomial(L=3), ChebyshevPolynomial(L=3)])
        for basis1, basis2 in all_bases:
            # check that hash is consistent
            assert hash(basis1) == hash(basis2)
            # check that equality is consistent
            assert basis1 == basis2
            # check that they are not the same object (i.e., not the same instance)
            assert basis1 is not basis2

            all_types.append(str(basis1.__class__.__name__))

        bases1 = [hash(basis[0]) for basis in all_bases]
        bases2 = [hash(basis[1]) for basis in all_bases]

        # check that all bases have unique hashes
        assert len(bases1) == len(set(bases1))
        assert len(bases2) == len(set(bases2))

        import desc

        # chech that this test tests all basis types
        assert set(all_types) == set(
            desc.basis.__all__
        ), "Not all basis types were tested."


@pytest.mark.unit
def test_jacobi_jvp():
    """Test that custom derivative rule for jacobi polynomials works."""
    basis = ZernikePolynomial(25, 25)
    l, m = basis.modes[:, :2].T
    m = jnp.abs(m)
    alpha = m
    beta = 0
    n = (l - m) // 2
    r = np.linspace(0, 1, 1000)
    jacobi_arg = 1 - 2 * r**2
    for i in range(5):
        # custom jvp rule for derivative of jacobi should just call jacobi with dx+1
        f1 = jnp.vectorize(Derivative(_jacobi, 3, "grad"))(
            n, alpha, beta, jacobi_arg[:, None], i
        )
        f2 = _jacobi(n, alpha, beta, jacobi_arg[:, None], i + 1)
        np.testing.assert_allclose(f1, f2)
