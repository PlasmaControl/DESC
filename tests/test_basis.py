import unittest
import numpy as np
import mpmath
from desc.grid import LinearGrid
from desc.basis import polyder_vec, polyval_vec, powers, jacobi, fourier, jacobi_coeffs
from desc.basis import (
    PowerSeries,
    FourierSeries,
    DoubleFourierSeries,
    ZernikePolynomial,
    FourierZernikeBasis,
)


class TestBasis(unittest.TestCase):
    """Test Basis class."""

    def test_polyder(self):
        """Test polyder_vec function."""
        p0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
        p1 = polyder_vec(p0, 1)
        p2 = polyder_vec(p0, 2)

        correct_p1 = np.array([[0, 2, 0], [0, 0, 1], [0, 0, 0], [0, 2, 1]])
        correct_p2 = np.array([[0, 0, 2], [0, 0, 0], [0, 0, 0], [0, 0, 2]])

        np.testing.assert_allclose(p1, correct_p1, atol=1e-8)
        np.testing.assert_allclose(p2, correct_p2, atol=1e-8)

    def test_polyval(self):
        """Test polyval_vec function."""
        p = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
        x = np.linspace(0, 1, 11)

        correct_vals = np.array([x ** 2, x, np.ones_like(x), x ** 2 + x + 1])
        values = polyval_vec(p, x)

        np.testing.assert_allclose(values, correct_vals, atol=1e-8)

    def test_polyval_exact(self):
        basis = FourierZernikeBasis(L=80, M=2, N=0)
        l, m = basis.modes[:, 0], basis.modes[:, 1]
        coeffs = jacobi_coeffs(l, m, exact=True)
        grid = LinearGrid(L=20)
        r = grid.nodes[:, 0]
        mpmath.mp.dps = 100
        exact = np.array(
            [
                np.asarray(mpmath.polyval(list(ci), r, derivative=True), dtype=float)
                for ci in coeffs
            ]
        )
        mpmath.mp.dps = 15
        exactf = exact[:, 0, :].T
        exactdf = exact[:, 1, :].T
        approxf = jacobi(r, l, m)
        approxdf = jacobi(r, l, m, dr=1)

        np.testing.assert_allclose(approxf, exactf, atol=1e-12)
        np.testing.assert_allclose(approxdf, exactdf, atol=1e-12)

    def test_powers(self):
        """Test powers function."""
        l = np.array([0, 1, 2])
        r = np.linspace(0, 1, 11)  # rho coordinates

        correct_vals = np.array([np.ones_like(r), r, r ** 2]).T
        correct_ders = np.array([np.zeros_like(r), np.ones_like(r), 2 * r]).T

        values = powers(r, l, dr=0)
        derivs = powers(r, l, dr=1)

        np.testing.assert_allclose(values, correct_vals, atol=1e-8)
        np.testing.assert_allclose(derivs, correct_ders, atol=1e-8)

    def test_jacobi(self):
        """Test Jacobi evaluation."""
        l = np.array([3, 4, 6])
        m = np.array([1, 2, 2])
        r = np.linspace(0, 1, 11)  # rho coordinates

        # correct value functions
        def Z3_1(x):
            return 3 * x ** 3 - 2 * x

        def Z4_2(x):
            return 4 * x ** 4 - 3 * x ** 2

        def Z6_2(x):
            return 15 * x ** 6 - 20 * x ** 4 + 6 * x ** 2

        # correct derivative functions
        def dZ3_1(x):
            return 9 * x ** 2 - 2

        def dZ4_2(x):
            return 16 * x ** 3 - 6 * x

        def dZ6_2(x):
            return 90 * x ** 5 - 80 * x ** 3 + 12 * x

        correct_vals = np.array([Z3_1(r), Z4_2(r), Z6_2(r)]).T
        correct_ders = np.array([dZ3_1(r), dZ4_2(r), dZ6_2(r)]).T

        values = jacobi(r, l, m, 0)
        derivs = jacobi(r, l, m, 1)

        np.testing.assert_allclose(values, correct_vals, atol=1e-8)
        np.testing.assert_allclose(derivs, correct_ders, atol=1e-8)

    def test_fourier(self):
        """Test Fourier evaluation."""
        m = np.array([-1, 0, 1])
        t = np.linspace(0, 2 * np.pi, 8, endpoint=False)  # theta coordinates

        correct_vals = np.array([np.sin(t), np.ones_like(t), np.cos(t)]).T
        correct_ders = np.array([np.cos(t), np.zeros_like(t), -np.sin(t)]).T

        values = fourier(t, m, dt=0)
        derivs = fourier(t, m, dt=1)

        np.testing.assert_allclose(values, correct_vals, atol=1e-8)
        np.testing.assert_allclose(derivs, correct_ders, atol=1e-8)

    def test_power_series(self):
        """Test PowerSeries evaluation."""
        grid = LinearGrid(L=11, endpoint=True)
        r = grid.nodes[:, 0]  # rho coordinates

        correct_vals = np.array([np.ones_like(r), r, r ** 2]).T
        correct_ders = np.array([np.zeros_like(r), np.ones_like(r), 2 * r]).T

        basis = PowerSeries(L=2)
        values = basis.evaluate(grid.nodes, derivatives=np.array([0, 0, 0]))
        derivs = basis.evaluate(grid.nodes, derivatives=np.array([1, 0, 0]))

        np.testing.assert_allclose(values, correct_vals, atol=1e-8)
        np.testing.assert_allclose(derivs, correct_ders, atol=1e-8)

    def test_double_fourier(self):
        """Test DoubleFourierSeries evaluation."""
        grid = LinearGrid(M=5, N=5)
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

    def test_change_resolution(self):
        """Test change_resolution function."""
        ps = PowerSeries(L=4)
        ps.change_resolution(L=6)
        assert len(ps.modes) == 7

        fs = FourierSeries(N=3)
        fs.change_resolution(N=2)
        assert len(fs.modes) == 5

        dfs = DoubleFourierSeries(M=3, N=4)
        dfs.change_resolution(M=2, N=1)
        assert len(dfs.modes) == 15

        zp = ZernikePolynomial(L=0, M=3, spectral_indexing="ansi")
        zp.change_resolution(L=3, M=3)
        assert len(zp.modes) == 10

        zp2 = ZernikePolynomial(L=0, M=3, spectral_indexing="fringe")
        zp2.change_resolution(L=6, M=3)
        assert len(zp2.modes) == 16

        fz = FourierZernikeBasis(L=6, M=3, N=0)
        fz.change_resolution(L=6, M=3, N=1)
        assert len(fz.modes) == 48

    def test_repr(self):

        fz = FourierZernikeBasis(L=6, M=3, N=0)
        s = str(fz)
        assert "FourierZernikeBasis" in s
        assert "fringe" in s
        assert "L=6" in s
        assert "M=3" in s
        assert "N=0" in s
