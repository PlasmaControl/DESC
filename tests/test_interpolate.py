import numpy as np
import unittest

from desc.interpolate import interp1d, interp2d, interp3d


class TestInterp1D(unittest.TestCase):
    def test_interp1d(self):
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

    def test_interp1d_extrap_periodic(self):
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


class TestInterp2D(unittest.TestCase):
    def test_interp2d(self):

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


class TestInterp3D(unittest.TestCase):
    def test_interp3d(self):

        xp = np.linspace(0, np.pi, 20)
        yp = np.linspace(0, 2 * np.pi, 20)
        zp = np.linspace(0, np.pi, 20)
        x = np.linspace(0, np.pi, 1000)
        y = np.linspace(0, 2 * np.pi, 1000)
        z = np.linspace(0, np.pi, 1000)
        xxp, yyp, zzp = np.meshgrid(xp, yp, zp, indexing="ij")

        f = lambda x, y, z: np.sin(x) * np.cos(y) * z ** 2
        fp = f(xxp, yyp, zzp)

        fq = interp3d(x, y, z, xp, yp, zp, fp)
        np.testing.assert_allclose(fq, f(x, y, z), rtol=1e-5, atol=1e-2)

        fq = interp3d(x, y, z, xp, yp, zp, fp, method="nearest")
        np.testing.assert_allclose(fq, f(x, y, z), rtol=1e-2, atol=1)

        fq = interp3d(x, y, z, xp, yp, zp, fp, method="linear")
        np.testing.assert_allclose(fq, f(x, y, z), rtol=1e-3, atol=1e-1)
