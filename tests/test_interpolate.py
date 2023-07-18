"""Tests for interpolation functions."""

import numpy as np
import pytest

from desc.interpolate import interp1d, interp2d, interp3d


class TestInterp1D:
    """Tests for interp1d function."""

    @pytest.mark.unit
    def test_interp1d(self):
        """Test accuracy of different 1d interpolation methods."""
        xp = np.linspace(0, 2 * np.pi, 100)
        x = np.linspace(0, 2 * np.pi, 10000)
        f = lambda x: np.sin(x)
        fp = f(xp)

        fq = interp1d(x, xp, fp, method="nearest")
        np.testing.assert_allclose(fq, f(x), rtol=1e-2, atol=1e-1)

        fq = interp1d(x, xp, fp, method="linear")
        np.testing.assert_allclose(fq, f(x), rtol=1e-4, atol=1e-3)

        fq = interp1d(x, xp, fp, method="cubic")
        np.testing.assert_allclose(fq, f(x), rtol=1e-6, atol=1e-5)

        fq = interp1d(x, xp, fp, method="cubic2")
        np.testing.assert_allclose(fq, f(x), rtol=1e-6, atol=1e-5)

        fq = interp1d(x, xp, fp, method="cardinal")
        np.testing.assert_allclose(fq, f(x), rtol=1e-6, atol=1e-5)

        fq = interp1d(x, xp, fp, method="catmull-rom")
        np.testing.assert_allclose(fq, f(x), rtol=1e-6, atol=1e-5)

        fq = interp1d(x, xp, fp, method="monotonic")
        np.testing.assert_allclose(fq, f(x), rtol=1e-4, atol=1e-3)

        fq = interp1d(x, xp, fp, method="monotonic-0")
        np.testing.assert_allclose(fq, f(x), rtol=1e-4, atol=1e-2)

    @pytest.mark.unit
    def test_interp1d_extrap_periodic(self):
        """Test extrapolation and periodic BC of 1d interpolation."""
        xp = np.linspace(0, 2 * np.pi, 200)
        x = np.linspace(-1, 2 * np.pi + 1, 10000)
        f = lambda x: np.sin(x)
        fp = f(xp)

        fq = interp1d(x, xp, fp, method="cubic", extrap=False)
        assert np.isnan(fq[0])
        assert np.isnan(fq[-1])

        fq = interp1d(x, xp, fp, method="cubic", extrap=True)
        assert not np.isnan(fq[0])
        assert not np.isnan(fq[-1])

        fq = interp1d(x, xp, fp, method="cubic", period=2 * np.pi)
        np.testing.assert_allclose(fq, f(x), rtol=1e-6, atol=1e-2)

    @pytest.mark.unit
    def test_interp1d_monotonic(self):
        """Ensure monotonic interpolation is actually monotonic."""
        # true function is just linear with a jump discontinuity at x=1.5
        x = np.linspace(-4, 5, 10)
        f = np.heaviside(x - 1.5, 0) + 0.1 * x
        xq = np.linspace(-4, 5, 1000)
        dfc = interp1d(xq, x, f, derivative=1, method="cubic")
        dfm = interp1d(xq, x, f, derivative=1, method="monotonic")
        dfm0 = interp1d(xq, x, f, derivative=1, method="monotonic-0")
        assert dfc.min() < 0  # cubic interpolation undershoots, giving negative slope
        assert dfm.min() > 0  # monotonic interpolation doesn't
        assert dfm0.min() >= 0  # monotonic-0 doesn't overshoot either
        # ensure monotonic-0 has 0 slope at end points
        np.testing.assert_allclose(dfm0[np.array([0, -1])], 0, atol=1e-12)


class TestInterp2D:
    """Tests for interp2d function."""

    @pytest.mark.unit
    def test_interp2d(self):
        """Test accuracy of different 2d interpolation methods."""
        xp = np.linspace(0, 4 * np.pi, 40)
        yp = np.linspace(0, 2 * np.pi, 40)
        y = np.linspace(0, 2 * np.pi, 10000)
        x = np.linspace(0, 2 * np.pi, 10000)
        xxp, yyp = np.meshgrid(xp, yp, indexing="ij")

        f = lambda x, y: np.sin(x) * np.cos(y)
        fp = f(xxp, yyp)

        fq = interp2d(x, y, xp, yp, fp)
        np.testing.assert_allclose(fq, f(x, y), rtol=1e-6, atol=1e-3)

        fq = interp2d(x, y, xp, yp, fp, method="nearest")
        np.testing.assert_allclose(fq, f(x, y), rtol=1e-2, atol=1)

        fq = interp2d(x, y, xp, yp, fp, method="linear")
        np.testing.assert_allclose(fq, f(x, y), rtol=1e-4, atol=1e-2)
        atol = 2e-3
        rtol = 1e-5
        fq = interp2d(x, y, xp, yp, fp, method="cubic")
        np.testing.assert_allclose(fq, f(x, y), rtol=rtol, atol=atol)

        fq = interp2d(x, y, xp, yp, fp, method="cubic2")
        np.testing.assert_allclose(fq, f(x, y), rtol=rtol, atol=atol)

        fq = interp2d(x, y, xp, yp, fp, method="catmull-rom")
        np.testing.assert_allclose(fq, f(x, y), rtol=rtol, atol=atol)

        fq = interp2d(x, y, xp, yp, fp, method="cardinal")
        np.testing.assert_allclose(fq, f(x, y), rtol=rtol, atol=atol)


class TestInterp3D:
    """Tests for interp3d function."""

    @pytest.mark.unit
    def test_interp3d(self):
        """Test accuracy of different 3d interpolation methods."""
        xp = np.linspace(0, np.pi, 20)
        yp = np.linspace(0, 2 * np.pi, 20)
        zp = np.linspace(0, np.pi, 20)
        x = np.linspace(0, np.pi, 1000)
        y = np.linspace(0, 2 * np.pi, 1000)
        z = np.linspace(0, np.pi, 1000)
        xxp, yyp, zzp = np.meshgrid(xp, yp, zp, indexing="ij")

        f = lambda x, y, z: np.sin(x) * np.cos(y) * z**2
        fp = f(xxp, yyp, zzp)

        fq = interp3d(x, y, z, xp, yp, zp, fp)
        np.testing.assert_allclose(fq, f(x, y, z), rtol=1e-5, atol=1e-2)

        fq = interp3d(x, y, z, xp, yp, zp, fp, method="nearest")
        np.testing.assert_allclose(fq, f(x, y, z), rtol=1e-2, atol=1)

        fq = interp3d(x, y, z, xp, yp, zp, fp, method="linear")
        np.testing.assert_allclose(fq, f(x, y, z), rtol=1e-3, atol=1e-1)
        atol = 5.5e-3
        rtol = 1e-5
        fq = interp3d(x, y, z, xp, yp, zp, fp, method="cubic")
        np.testing.assert_allclose(fq, f(x, y, z), rtol=rtol, atol=atol)

        fq = interp3d(x, y, z, xp, yp, zp, fp, method="cubic2")
        np.testing.assert_allclose(fq, f(x, y, z), rtol=rtol, atol=atol)

        fq = interp3d(x, y, z, xp, yp, zp, fp, method="catmull-rom")
        np.testing.assert_allclose(fq, f(x, y, z), rtol=rtol, atol=atol)

        fq = interp3d(x, y, z, xp, yp, zp, fp, method="cardinal")
        np.testing.assert_allclose(fq, f(x, y, z), rtol=rtol, atol=atol)
