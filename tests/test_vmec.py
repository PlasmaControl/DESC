import unittest
import numpy as np
from netCDF4 import Dataset

from desc.backend import put
from desc.vmec import VMECIO
from desc.basis import FourierZernikeBasis


class TestVMECIO(unittest.TestCase):
    """Tests VMECIO class"""

    def test_ptolemy_identity_fwd(self):
        """Tests forward implementation of Ptolemy's identity."""
        a0 = 3
        a1 = -1
        a2 = 1
        a3 = 2

        m_0 = np.array([0, 0, 1, 1, 1])
        n_0 = np.array([0, 1, -1, 0, 1])
        s = np.array([0, a0, a1, a3, 0])
        c = np.array([a0, 0, a2, 0, a3])

        # a0*sin(-z) + a1*sin(t+z) + a3*sin(t) = -a0*sin(z) + a1*sin(t)*cos(z) + a1*cos(t)*sin(z) + a3*sin(t)
        # a0 + a2*cos(t+z) + a3*cos(t-z) = a0 + (a2+a3)*cos(t)*cos(z) + (a3-a2)*sin(t)*sin(z)

        m_1_correct = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
        n_1_correct = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
        x_correct = np.array([[a3 - a2, a3, a1, -a0, a0, 0, a1, 0, a2 + a3]])

        m_1, n_1, x = VMECIO._ptolemy_identity_fwd(m_0, n_0, s, c)

        np.testing.assert_allclose(m_1, m_1_correct, atol=1e-8)
        np.testing.assert_allclose(n_1, n_1_correct, atol=1e-8)
        np.testing.assert_allclose(x, x_correct, atol=1e-8)

    def test_ptolemy_identity_rev(self):
        """Tests reverse implementation of Ptolemy's identity."""
        a0 = 3
        a1 = -1
        a2 = 1
        a3 = 2

        m_1 = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
        n_1 = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
        x = np.array([[a3 - a2, a3, a1, -a0, a0, 0, a1, 0, a2 + a3]])

        # -a0*sin(z) + a1*sin(t)*cos(z) + a1*cos(t)*sin(z) + a3*sin(t) = a0*sin(-z) + a1*sin(t+z) + a3*sin(t)
        # a0 + (a2+a3)*cos(t)*cos(z) + (a3-a2)*sin(t)*sin(z) = a0 + a2*cos(t+z) + a3*cos(t-z)

        m_0_correct = np.array([0, 0, 1, 1, 1])
        n_0_correct = np.array([0, 1, -1, 0, 1])
        s_correct = np.array([[0, a0, a1, a3, 0]])
        c_correct = np.array([[a0, 0, a2, 0, a3]])

        m_0, n_0, s, c = VMECIO._ptolemy_identity_rev(m_1, n_1, x)

        np.testing.assert_allclose(m_0, m_0_correct, atol=1e-8)
        np.testing.assert_allclose(n_0, n_0_correct, atol=1e-8)
        np.testing.assert_allclose(s, s_correct, atol=1e-8)
        np.testing.assert_allclose(c, c_correct, atol=1e-8)

    def test_fourier_to_zernike(self):
        """Tests conversion from radial-Fourier series to Fourier-Zernike polynomials."""
        M = 1
        N = 1

        a0 = 3
        a1 = -1
        a2 = 1
        a3 = 2

        surfs = 16
        rho = np.sqrt(np.linspace(0, 1, surfs))

        m = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
        n = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
        x = np.array([a3 - a2, a3, a1, -a0, a0, 0, a1, 0, a2 + a3])

        # x * rho^|m|
        x_mn = np.power(np.atleast_2d(rho).T, np.atleast_2d(np.abs(m))) * np.atleast_2d(
            x
        )
        basis = FourierZernikeBasis(M=M, N=N)
        x_lmn = VMECIO._fourier_to_zernike(m, n, x_mn, basis)

        x_lmn_correct = np.zeros((basis.num_modes,))
        for k in range(basis.num_modes):
            idx = np.where(
                np.logical_and.reduce(
                    (
                        basis.modes[:, 0] == np.abs(m[k]),
                        basis.modes[:, 1] == m[k],
                        basis.modes[:, 2] == n[k],
                    )
                )
            )[0]
            x_lmn_correct = put(x_lmn_correct, idx, x[k])

        np.testing.assert_allclose(x_lmn, x_lmn_correct, atol=1e-8)

    def test_zernike_to_fourier(self):
        """Tests conversion from Fourier-Zernike polynomials to radial-Fourier series."""
        M = 1
        N = 1

        a0 = 3
        a1 = -1
        a2 = 1
        a3 = 2

        surfs = 16
        rho = np.sqrt(np.linspace(0, 1, surfs))

        m_correct = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
        n_correct = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
        x = np.array([a3 - a2, a3, a1, -a0, a0, 0, a1, 0, a2 + a3])

        # x * rho^|m|
        x_mn_correct = np.power(
            np.atleast_2d(rho).T, np.atleast_2d(np.abs(m_correct))
        ) * np.atleast_2d(x)
        basis = FourierZernikeBasis(M=M, N=N)

        x_lmn = np.zeros((basis.num_modes,))
        for k in range(basis.num_modes):
            idx = np.where(
                np.logical_and.reduce(
                    (
                        basis.modes[:, 0] == np.abs(m_correct[k]),
                        basis.modes[:, 1] == m_correct[k],
                        basis.modes[:, 2] == n_correct[k],
                    )
                )
            )[0]
            x_lmn = put(x_lmn, idx, x[k])

        m, n, x_mn = VMECIO._zernike_to_fourier(x_lmn, basis, rho)

        np.testing.assert_allclose(m, m_correct, atol=1e-8)
        np.testing.assert_allclose(n, n_correct, atol=1e-8)
        np.testing.assert_allclose(x_mn, x_mn_correct, atol=1e-8)


def test_load_then_save(TmpDir):
    """Tests if loading and then saving gives the original result."""

    input_path = "examples//VMEC//wout_SOLOVEV.nc"
    output_path = str(TmpDir.join("DESC_SOLOVEV.nc"))

    eq = VMECIO.load(input_path)
    VMECIO.save(eq, output_path)

    file1 = Dataset(input_path, mode="r")
    file2 = Dataset(output_path, mode="r")

    rmnc1 = file1.variables["rmnc"][:]
    rmnc2 = file2.variables["rmnc"][:]
    zmns1 = file1.variables["zmns"][:]
    zmns2 = file2.variables["zmns"][:]
    lmns1 = file1.variables["lmns"][:]
    lmns2 = file2.variables["lmns"][:]

    np.testing.assert_allclose(rmnc2, rmnc1, atol=1e-1)
    np.testing.assert_allclose(zmns2, zmns1, atol=1e-1)
    np.testing.assert_allclose(lmns2, lmns1, atol=1e-1)

    file1.close
    file2.close
