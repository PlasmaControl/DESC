import pytest
import numpy as np
from desc.basis import sinbasis


@pytest.fixture
def ms():
    return np.array([-1, 0, 1, 2])


@pytest.fixture
def zetas():
    return np.linspace(0, 2 * np.pi, 20)


class TestSinBasis:
    @pytest.mark.mirror_unit
    def test_sin_basis_dz0(self, ms, zetas):
        correct = np.zeros_like(ms.reshape((-1, 1)) * zetas)
        for i, m in enumerate(ms):
            for j, z in enumerate(zetas):
                match m:
                    case -1:
                        val = np.cos(z / 2)
                    case 0:
                        val = 1
                    case _:
                        val = np.sin(m * z / 2)
                correct[i, j] = val
        out = sinbasis(zetas, ms.reshape((-1, 1)))
        np.testing.assert_allclose(out, correct)

    @pytest.mark.mirror_unit
    def test_sin_basis_dz1(self, ms, zetas):
        correct = np.zeros_like(ms.reshape((-1, 1)) * zetas)
        for i, m in enumerate(ms):
            for j, z in enumerate(zetas):
                match m:
                    case -1:
                        val = -np.sin(z / 2) / 2
                    case 0:
                        val = 0
                    case _:
                        val = np.cos(m * z / 2) * m / 2
                correct[i, j] = val
        out = sinbasis(zetas, ms.reshape((-1, 1)), dz=1)
        np.testing.assert_allclose(out, correct, atol=1e-15)

    @pytest.mark.mirror_unit
    def test_sin_basis_dz2(self, ms, zetas):
        correct = np.zeros_like(ms.reshape((-1, 1)) * zetas)
        for i, m in enumerate(ms):
            for j, z in enumerate(zetas):
                match m:
                    case -1:
                        val = -np.cos(z / 2) / 4
                    case 0:
                        val = 0
                    case _:
                        val = -np.sin(m * z / 2) * m**2 / 4
                correct[i, j] = val
        out = sinbasis(zetas, ms.reshape((-1, 1)), dz=2)
        np.testing.assert_allclose(out, correct, atol=1e-15)

    @pytest.mark.mirror_unit
    def test_sin_basis_dz3(self, ms, zetas):
        correct = np.zeros_like(ms.reshape((-1, 1)) * zetas)
        for i, m in enumerate(ms):
            for j, z in enumerate(zetas):
                match m:
                    case -1:
                        val = np.sin(z / 2) / 8
                    case 0:
                        val = 0
                    case _:
                        val = -np.cos(m * z / 2) * m**3 / 8
                correct[i, j] = val
        out = sinbasis(zetas, ms.reshape((-1, 1)), dz=3)
        np.testing.assert_allclose(out, correct, atol=1e-15)
