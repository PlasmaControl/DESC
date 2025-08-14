"""Tests for jax autodiff wrappers and finite differences."""

import numpy as np
import pytest
from numpy.random import default_rng

from desc.backend import jnp
from desc.derivatives import AutoDiffDerivative, FiniteDiffDerivative


class TestDerivative:
    """Tests Derivative classes."""

    @pytest.mark.unit
    def test_finite_diff_vec(self):
        """Test finite differences of vector function."""

        def test_fun(x, y, a):
            return x * y + a

        x = np.array([1, 5, 0.01, 200])
        y = np.array([60, 1, 100, 0.02])
        a = -2

        jac = FiniteDiffDerivative(test_fun, argnum=0)
        J = jac.compute(x, y, a)
        correct_J = np.diag(y)

        np.testing.assert_allclose(J, correct_J, atol=1e-8)

    @pytest.mark.unit
    def test_finite_diff_scalar(self):
        """Test finite differences of scalar function."""

        def test_fun(x, y, a):
            return np.dot(x, y) + a

        x = np.array([1, 5, 0.01, 200])
        y = np.array([60, 1, 100, 0.02])
        a = -2

        jac = FiniteDiffDerivative(test_fun, argnum=0)
        J = jac.compute(x, y, a)
        correct_J = y

        np.testing.assert_allclose(J, correct_J, atol=1e-8)

        jac.argnum = 1
        J = jac.compute(x, y, a)
        np.testing.assert_allclose(J, x, atol=1e-8)

    @pytest.mark.unit
    def test_auto_diff(self):
        """Test automatic differentiation."""

        def test_fun(x, y, a):
            return jnp.cos(x) + x * y + a

        x = np.array([1, 5, 0.01, 200])
        y = np.array([60, 1, 100, 0.02])
        a = -2

        jac = AutoDiffDerivative(test_fun, argnum=0)
        J = jac.compute(x, y, a)
        correct_J = np.diag(-np.sin(x) + y)

        np.testing.assert_allclose(J, correct_J, atol=1e-8)

    @pytest.mark.unit
    def test_compare_AD_FD(self):
        """Compare finite differences to automatic differentiation."""

        def test_fun(x, y, a):
            return jnp.cos(x) + x * y + a

        x = np.array([1, 5, 0.01, 200])
        y = np.array([60, 1, 100, 0.02])
        a = -2

        jacf_AD = AutoDiffDerivative(test_fun, argnum=0, mode="fwd")
        Jf_AD = jacf_AD.compute(x, y, a)
        jacr_AD = AutoDiffDerivative(test_fun, argnum=0, mode="rev")
        Jr_AD = jacr_AD.compute(x, y, a)

        jac_FD = FiniteDiffDerivative(test_fun, argnum=0)
        J_FD = jac_FD.compute(x, y, a)

        np.testing.assert_allclose(Jf_AD, Jr_AD, atol=1e-8)
        np.testing.assert_allclose(J_FD, Jf_AD, rtol=1e-2)
        np.testing.assert_allclose(J_FD, Jr_AD, rtol=1e-2)

    @pytest.mark.unit
    def test_fd_hessian(self):
        """Test finite difference calculation of hessian."""
        rando = default_rng(seed=0)

        n = 5
        A = rando.random((n, n))
        A = A + A.T
        g = rando.random(n)

        def f(x):
            return 5 + g.dot(x) + x.dot(1 / 2 * A.dot(x))

        hess = FiniteDiffDerivative(f, argnum=0, mode="hess")

        y = rando.random(n)
        A1 = hess(y)

        np.testing.assert_allclose(A1, A)

    @pytest.mark.unit
    def test_block_jacobian(self):
        """Test calculation of jacobian using blocked method."""
        rando = default_rng(seed=0)
        A = rando.random((19, 17))

        def fun(x):
            return jnp.dot(A, x)

        x = rando.random(17)

        jac = AutoDiffDerivative(fun, block_size=4, shape=A.shape)
        np.testing.assert_allclose(jac(x), A)
        jac = AutoDiffDerivative(fun, num_blocks=3, shape=A.shape)
        np.testing.assert_allclose(jac(x), A)


class TestJVP:
    """Test calculation of jacobian vector products."""

    @staticmethod
    def fun(x, c1, c2):
        """Function for testing."""
        Amat = np.arange(12).reshape((4, 3))
        return jnp.dot(Amat, (x + c1 * c2) ** 3)

    x = np.ones(3).astype(float)
    c1 = np.arange(3).astype(float)
    c2 = np.arange(3).astype(float) + 2

    dx = np.array([1, 2, 3]).astype(float)
    dc1 = np.array([3, 4, 5]).astype(float)
    dc2 = np.array([-3, 1, -2]).astype(float)
    y = np.array([1, 2, 3, 4])

    @pytest.mark.unit
    def test_autodiff_jvp(self):
        """Tests using AD for JVP calculation."""
        df = AutoDiffDerivative.compute_jvp(
            self.fun, 0, self.dx, self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([1554.0, 4038.0, 6522.0, 9006.0]))
        df = AutoDiffDerivative.compute_jvp(
            self.fun, 1, self.dc1, self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([10296.0, 26658.0, 43020.0, 59382.0]))
        df = AutoDiffDerivative.compute_jvp(
            self.fun, (0, 2), (self.dx, self.dc2), self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([-342.0, -630.0, -918.0, -1206.0]))

    @pytest.mark.unit
    def test_finitediff_jvp(self):
        """Tests using FD for JVP calculation."""
        df = FiniteDiffDerivative.compute_jvp(
            self.fun, 0, self.dx, self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([1554.0, 4038.0, 6522.0, 9006.0]))
        df = FiniteDiffDerivative.compute_jvp(
            self.fun, 1, self.dc1, self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([10296.0, 26658.0, 43020.0, 59382.0]))
        df = FiniteDiffDerivative.compute_jvp(
            self.fun, (0, 2), (self.dx, self.dc2), self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([-342.0, -630.0, -918.0, -1206.0]))

    @pytest.mark.unit
    def test_autodiff_jvp2(self):
        """Tests using AD for 2nd order JVP calculation."""
        df = AutoDiffDerivative.compute_jvp2(
            self.fun, 0, 0, self.dx + 1, self.dx, self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([1440.0, 3852.0, 6264.0, 8676.0]))
        df = AutoDiffDerivative.compute_jvp2(
            self.fun, 1, 1, self.dc1 + 1, self.dc1, self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(
            df, np.array([56160.0, 147744.0, 239328.0, 330912.0])
        )
        df = AutoDiffDerivative.compute_jvp2(
            self.fun, 0, 2, self.dx, self.dc2, self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([-1248.0, -3048.0, -4848.0, -6648.0]))
        df = AutoDiffDerivative.compute_jvp2(
            self.fun, 0, (1, 2), self.dx, (self.dc1, self.dc2), self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([5808.0, 15564.0, 25320.0, 35076.0]))
        df = AutoDiffDerivative.compute_jvp2(
            self.fun,
            (1, 2),
            (1, 2),
            (self.dc1, self.dc2),
            (self.dc1, self.dc2),
            self.x,
            self.c1,
            self.c2,
        )
        np.testing.assert_allclose(df, np.array([22368.0, 63066.0, 103764.0, 144462.0]))
        df = AutoDiffDerivative.compute_jvp2(
            self.fun, 0, (1, 2), self.dx, (self.dc1, self.dc2), self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([5808.0, 15564.0, 25320.0, 35076.0]))

    @pytest.mark.unit
    def test_finitediff_jvp2(self):
        """Tests using FD for 2nd order JVP calculation."""
        df = FiniteDiffDerivative.compute_jvp2(
            self.fun, 0, 0, self.dx + 1, self.dx, self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([1440.0, 3852.0, 6264.0, 8676.0]))
        df = FiniteDiffDerivative.compute_jvp2(
            self.fun, 1, 1, self.dc1 + 1, self.dc1, self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(
            df, np.array([56160.0, 147744.0, 239328.0, 330912.0])
        )
        df = FiniteDiffDerivative.compute_jvp2(
            self.fun, 0, 2, self.dx, self.dc2, self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([-1248.0, -3048.0, -4848.0, -6648.0]))
        df = FiniteDiffDerivative.compute_jvp2(
            self.fun, 0, (1, 2), self.dx, (self.dc1, self.dc2), self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([5808.0, 15564.0, 25320.0, 35076.0]))
        df = FiniteDiffDerivative.compute_jvp2(
            self.fun,
            (1, 2),
            (1, 2),
            (self.dc1, self.dc2),
            (self.dc1, self.dc2),
            self.x,
            self.c1,
            self.c2,
        )
        np.testing.assert_allclose(df, np.array([22368.0, 63066.0, 103764.0, 144462.0]))
        df = FiniteDiffDerivative.compute_jvp2(
            self.fun, 0, (1, 2), self.dx, (self.dc1, self.dc2), self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([5808.0, 15564.0, 25320.0, 35076.0]))

    @pytest.mark.unit
    def test_autodiff_jvp3(self):
        """Tests using AD for 3rd order JVP calculation."""
        df = AutoDiffDerivative.compute_jvp3(
            self.fun, 0, 0, 0, self.dx + 1, self.dx, self.dx, self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([504.0, 1404.0, 2304.0, 3204.0]))
        df = AutoDiffDerivative.compute_jvp3(
            self.fun, 0, 1, 1, self.dx, self.dc1 + 1, self.dc1, self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([19440.0, 52704.0, 85968.0, 119232.0]))
        df = AutoDiffDerivative.compute_jvp3(
            self.fun, 0, 1, 2, self.dx, self.dc1, self.dc2, self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(
            df, np.array([-5784.0, -14118.0, -22452.0, -30786.0])
        )
        df = AutoDiffDerivative.compute_jvp3(
            self.fun,
            0,
            0,
            (1, 2),
            self.dx,
            self.dx,
            (self.dc1, self.dc2),
            self.x,
            self.c1,
            self.c2,
        )
        np.testing.assert_allclose(df, np.array([2040.0, 5676.0, 9312.0, 12948.0]))
        df = AutoDiffDerivative.compute_jvp3(
            self.fun,
            (1, 2),
            (1, 2),
            (1, 2),
            (self.dc1, self.dc2),
            (self.dc1, self.dc2),
            (self.dc1, self.dc2),
            self.x,
            self.c1,
            self.c2,
        )
        np.testing.assert_allclose(
            df, np.array([-33858.0, -55584.0, -77310.0, -99036.0])
        )

    @pytest.mark.unit
    def test_finitediff_jvp3(self):
        """Tests using FD for 3rd order JVP calculation."""
        df = FiniteDiffDerivative.compute_jvp3(
            self.fun, 0, 0, 0, self.dx + 1, self.dx, self.dx, self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(
            df, np.array([504.0, 1404.0, 2304.0, 3204.0]), rtol=1e-4
        )
        df = FiniteDiffDerivative.compute_jvp3(
            self.fun, 0, 1, 1, self.dx, self.dc1 + 1, self.dc1, self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(
            df, np.array([19440.0, 52704.0, 85968.0, 119232.0]), rtol=1e-4
        )
        df = FiniteDiffDerivative.compute_jvp3(
            self.fun, 0, 1, 2, self.dx, self.dc1, self.dc2, self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(
            df, np.array([-5784.0, -14118.0, -22452.0, -30786.0]), rtol=1e-4
        )
        df = FiniteDiffDerivative.compute_jvp3(
            self.fun,
            0,
            0,
            (1, 2),
            self.dx,
            self.dx,
            (self.dc1, self.dc2),
            self.x,
            self.c1,
            self.c2,
        )
        np.testing.assert_allclose(
            df, np.array([2040.0, 5676.0, 9312.0, 12948.0]), rtol=1e-4
        )
        df = FiniteDiffDerivative.compute_jvp3(
            self.fun,
            (1, 2),
            (1, 2),
            (1, 2),
            (self.dc1, self.dc2),
            (self.dc1, self.dc2),
            (self.dc1, self.dc2),
            self.x,
            self.c1,
            self.c2,
        )
        np.testing.assert_allclose(
            df, np.array([-33858.0, -55584.0, -77310.0, -99036.0]), rtol=1e-4
        )

    @pytest.mark.unit
    def test_vjp(self):
        """Tests using AD and FD for VJP calculation."""
        df = AutoDiffDerivative.compute_vjp(
            self.fun, 0, self.y, self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([180.0, 3360.0, 19440.0]))
        df = FiniteDiffDerivative.compute_vjp(
            self.fun, 0, self.y, self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([180.0, 3360.0, 19440.0]), rtol=1e-4)
