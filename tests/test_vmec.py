"""Tests for reading/writing/converting VMEC data."""

import numpy as np
import pytest
from netCDF4 import Dataset

import desc.examples
from desc.basis import DoubleFourierSeries, FourierZernikeBasis
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.examples import get
from desc.grid import Grid, LinearGrid
from desc.input_reader import InputReader
from desc.io import load
from desc.transform import Transform
from desc.vmec import VMECIO
from desc.vmec_utils import (
    fourier_to_zernike,
    make_boozmn_output,
    ptolemy_identity_fwd,
    ptolemy_identity_rev,
    ptolemy_linear_transform,
    vmec_boundary_subspace,
    zernike_to_fourier,
)

from .utils import area_difference_desc


class TestVMECIO:
    """Tests VMECIO class."""

    @pytest.mark.unit
    def test_ptolemy_identity_fwd(self):
        """Tests forward implementation of Ptolemy's identity."""
        a0 = 3
        a1 = -1
        a2 = 1
        a3 = 2

        m_0 = np.array([0, 1, 1, 1, 0])
        n_0 = np.array([1, -1, 0, 1, 0])
        s = np.array([a0, a1, a3, 0, 0])
        c = np.array([0, a2, 0, a3, a0])

        # a0*sin(-z) + a1*sin(t+z) + a3*sin(t)                          # noqa: E800
        #    = -a0*sin(z) + a1*sin(t)*cos(z) + a1*cos(t)*sin(z) + a3*sin(t)
        # a0 + a2*cos(t+z) + a3*cos(t-z)                                # noqa: E800
        #    = a0 + (a2+a3)*cos(t)*cos(z) + (a3-a2)*sin(t)*sin(z)

        m_1_correct = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
        n_1_correct = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
        x_correct = np.array([[a3 - a2, -a0, a1, a3, a0, 0, a1, 0, a2 + a3]])

        m_1, n_1, x = ptolemy_identity_fwd(m_0, n_0, s, c)

        np.testing.assert_allclose(m_1, m_1_correct, atol=1e-8)
        np.testing.assert_allclose(n_1, n_1_correct, atol=1e-8)
        np.testing.assert_allclose(x, x_correct, atol=1e-8)

    @pytest.mark.unit
    def test_ptolemy_identity_fwd_sin_series(self):
        """Tests forward implementation of Ptolemy's identity for sin only."""
        a1 = -1

        m_0 = np.array([1, 1])
        n_0 = np.array([-1, 1])
        s = np.array([[a1, 0]])
        c = np.array([[0, 0]])

        m_1, n_1, x = ptolemy_identity_fwd(m_0, n_0, s, c)

        print(m_1)
        print(n_1)
        print(x)

        #  a1*sin(t+z)  # noqa: E800
        # = a1*sin(t)*cos(z) + a1*cos(t)*sin(z) # noqa: E800

        m_1_correct = np.array([-1, 1, -1, 1])
        n_1_correct = np.array([-1, -1, 1, 1])
        x_correct = np.array([[0, a1, a1, 0]])

        np.testing.assert_allclose(m_1, m_1_correct, atol=1e-8)
        np.testing.assert_allclose(n_1, n_1_correct, atol=1e-8)
        np.testing.assert_allclose(x, x_correct, atol=1e-8)

    @pytest.mark.unit
    def test_ptolemy_identity_rev(self):
        """Tests reverse implementation of Ptolemy's identity."""
        a0 = 3
        a1 = -1
        a2 = 1
        a3 = 2

        m_1 = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
        n_1 = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
        x = np.array([[a3 - a2, a3, a1, -a0, a0, 0, a1, 0, a2 + a3]])

        # -a0*sin(z) + a1*sin(t)*cos(z) + a1*cos(t)*sin(z) + a3*sin(t)   # noqa: E800
        #   = a0*sin(-z) + a1*sin(t+z) + a3*sin(t)
        # a0 + (a2+a3)*cos(t)*cos(z) + (a3-a2)*sin(t)*sin(z)             # noqa: E800
        #    = a0 + a2*cos(t+z) + a3*cos(t-z)

        m_0_correct = np.array([0, 0, 1, 1, 1])
        n_0_correct = np.array([0, 1, -1, 0, 1])
        s_correct = np.array([[0, a0, a1, a3, 0]])
        c_correct = np.array([[a0, 0, a2, 0, a3]])

        m_0, n_0, s, c = ptolemy_identity_rev(m_1, n_1, x)

        np.testing.assert_allclose(m_0, m_0_correct, atol=1e-8)
        np.testing.assert_allclose(n_0, n_0_correct, atol=1e-8)
        np.testing.assert_allclose(s, s_correct, atol=1e-8)
        np.testing.assert_allclose(c, c_correct, atol=1e-8)

    @pytest.mark.unit
    def test_ptolemy_identity_rev_sin_sym(self):
        """Tests reverse implementation of Ptolemy's identity for sin series."""
        a1 = -1

        m_1 = np.array([-1, 1])
        n_1 = np.array([1, -1])
        x = np.array([[a1, a1]])

        # a1*sin(t)*cos(z) + a1*cos(t)*sin(z)    # noqa: E800
        #   = a1*sin(t+z)

        m_0_correct = np.array([0, 1, 1])
        n_0_correct = np.array([0, -1, 1])
        s_correct = np.array([[0, a1, 0]])
        c_correct = np.array([[0, 0, 0]])

        m_0, n_0, s, c = ptolemy_identity_rev(m_1, n_1, x)

        np.testing.assert_allclose(m_0, m_0_correct, atol=1e-8)
        np.testing.assert_allclose(n_0, n_0_correct, atol=1e-8)
        np.testing.assert_allclose(s, s_correct, atol=1e-8)
        np.testing.assert_allclose(c, c_correct, atol=1e-8)

    @pytest.mark.unit
    def test_ptolemy_identities_inverse(self):
        """Tests that forward and reverse Ptolemy's identities are inverses."""
        basis = DoubleFourierSeries(4, 3, sym=False)
        modes = basis.modes
        x_correct = np.random.rand(basis.num_modes)

        m1, n1, s, c = ptolemy_identity_rev(modes[:, 1], modes[:, 2], x_correct)
        m0, n0, x = ptolemy_identity_fwd(m1, n1, s, c)

        np.testing.assert_allclose(m0, modes[:, 1])
        np.testing.assert_allclose(n0, modes[:, 2])
        np.testing.assert_allclose(x, np.atleast_2d(x_correct))

    @pytest.mark.unit
    def test_ptolemy_linear_transform(self):
        """Tests Ptolemy basis linear transformation utility function."""
        basis = DoubleFourierSeries(M=4, N=3, sym=False)
        matrix, modes, idx = ptolemy_linear_transform(
            basis.modes, helicity=(1, 1), NFP=1
        )

        x_correct = np.random.rand(basis.num_modes)
        x_transformed = matrix @ x_correct

        x_sin = np.insert(x_transformed[1::2], 0, 0)
        x_cos = x_transformed[::2]

        m, n, s, c = ptolemy_identity_rev(
            basis.modes[:, 1], basis.modes[:, 2], x_correct
        )
        np.testing.assert_allclose(modes[::2, 1], m)
        np.testing.assert_allclose(modes[::2, 2], n)
        np.testing.assert_allclose(np.atleast_2d(x_sin), s)
        np.testing.assert_allclose(np.atleast_2d(x_cos), c)

        _, _, x_original = ptolemy_identity_fwd(
            modes[::2, 1], modes[::2, 2], x_sin, x_cos
        )
        np.testing.assert_allclose(x_original, np.atleast_2d(x_correct))

        sym_modes = np.array(
            [
                [1, 0, 0],
                [-1, 1, 1],
                [1, 1, 1],
                [-1, 2, 2],
                [1, 2, 2],
                [-1, 3, 3],
                [1, 3, 3],
            ]
        )
        np.testing.assert_allclose(np.delete(modes, idx, axis=0), sym_modes)

    @pytest.mark.unit
    def test_fourier_to_zernike(self):
        """Test conversion from radial-Fourier series to Fourier-Zernike polynomials."""
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
        basis = FourierZernikeBasis(L=M, M=M, N=N, spectral_indexing="ansi")
        x_lmn = fourier_to_zernike(m, n, x_mn, basis)

        x_lmn_correct = np.zeros((basis.num_modes,))
        for k in range(basis.num_modes):
            idx = np.where((basis.modes == [np.abs(m[k]), m[k], n[k]]).all(axis=1))[0]
            x_lmn_correct[idx] = x[k]

        np.testing.assert_allclose(x_lmn, x_lmn_correct, atol=1e-8)

    @pytest.mark.unit
    def test_zernike_to_fourier(self):
        """Test conversion from Fourier-Zernike polynomials to radial-Fourier series."""
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
        basis = FourierZernikeBasis(L=M, M=M, N=N, spectral_indexing="ansi")

        x_lmn = np.zeros((basis.num_modes,))
        for k in range(basis.num_modes):
            idx = np.where(
                (basis.modes == [np.abs(m_correct[k]), m_correct[k], n_correct[k]]).all(
                    axis=1
                )
            )[0]
            x_lmn[idx] = x[k]

        m, n, x_mn = zernike_to_fourier(x_lmn, basis, rho)

        np.testing.assert_allclose(m, m_correct, atol=1e-8)
        np.testing.assert_allclose(n, n_correct, atol=1e-8)
        np.testing.assert_allclose(x_mn, x_mn_correct, atol=1e-8)


@pytest.mark.unit
def test_vmec_load_profiles(TmpDir):
    """Tests that loading with iota or current profiles give same result."""
    input_path = "./tests/inputs/wout_SOLOVEV.nc"

    eq_iota = VMECIO.load(input_path, profile="iota")
    eq_current = VMECIO.load(input_path, profile="current")

    assert eq_iota.current is None
    assert eq_current.iota is None

    grid = LinearGrid(
        M=eq_iota.M_grid,
        N=eq_iota.N_grid,
        NFP=eq_iota.NFP,
        rho=np.linspace(0.5, 1.0, 21),
    )
    data_iota = eq_iota.compute(["iota", "current"], grid=grid)
    data_current = eq_current.compute(["iota", "current"], grid=grid)

    iota_iota = grid.compress(data_iota["iota"])
    iota_current = grid.compress(data_current["iota"])
    current_iota = grid.compress(data_iota["current"])
    current_current = grid.compress(data_current["current"])

    np.testing.assert_allclose(iota_iota, iota_current, rtol=2e-2)
    np.testing.assert_allclose(current_current, current_iota, rtol=2e-2)


@pytest.mark.slow
@pytest.mark.unit
def test_load_then_save(TmpDir):
    """Tests if loading and then saving gives the original result."""
    input_path = "./tests/inputs/wout_SOLOVEV.nc"
    output_path = str(TmpDir.join("DESC_SOLOVEV.nc"))

    eq = VMECIO.load(input_path, profile="iota")
    VMECIO.save(eq, output_path)

    file1 = Dataset(input_path, mode="r")
    file2 = Dataset(output_path, mode="r")

    rmnc1 = file1.variables["rmnc"][:]
    rmnc2 = file2.variables["rmnc"][:]
    zmns1 = file1.variables["zmns"][:]
    zmns2 = file2.variables["zmns"][:]
    lmns1 = file1.variables["lmns"][:]
    lmns2 = file2.variables["lmns"][:]

    np.testing.assert_allclose(rmnc2, rmnc1, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(zmns2, zmns1, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(lmns2, lmns1, rtol=1e-3, atol=5e-2)

    file1.close()
    file2.close()


@pytest.mark.slow
@pytest.mark.unit
def test_axis_surf_after_load():
    """Tests if loading VMEC solution preserves axis and surface."""
    input_path = "./tests/inputs/wout_HELIOTRON.nc"

    eq = VMECIO.load(input_path, profile="iota")
    assert eq.is_nested()
    f = Dataset(input_path)

    surf1 = eq.surface
    surf2 = eq.get_surface_at(rho=1)
    axis1 = eq.axis
    axis2 = eq.get_axis()

    np.testing.assert_allclose(surf1.R_lmn, surf2.R_lmn, atol=1e-14)
    np.testing.assert_allclose(surf1.Z_lmn, surf2.Z_lmn, atol=1e-14)

    np.testing.assert_allclose(axis1.R_n, axis2.R_n, atol=1e-14)
    np.testing.assert_allclose(axis1.Z_n, axis2.Z_n, atol=1e-14)

    # surface
    rm, rn, Rb_cs, Rb_cc = ptolemy_identity_rev(
        surf1.R_basis.modes[:, 1],
        surf1.R_basis.modes[:, 2],
        np.where(surf1.R_basis.modes[:, 1] < 0, -surf1.R_lmn, surf1.R_lmn)[None, :],
    )

    zm, zn, Zb_cs, Zb_cc = ptolemy_identity_rev(
        surf1.Z_basis.modes[:, 1],
        surf1.Z_basis.modes[:, 2],
        np.where(surf1.Z_basis.modes[:, 1] < 0, -surf1.Z_lmn, surf1.Z_lmn)[None, :],
    )

    rmnc = f.variables["rmnc"][:].filled()
    zmns = f.variables["zmns"][:].filled()

    # axis
    rax_cc = f.variables["raxis_cc"][:].filled()
    zax_cs = -f.variables["zaxis_cs"][:].filled()

    np.testing.assert_allclose(rmnc[-1], Rb_cc.squeeze(), atol=1e-14)
    np.testing.assert_allclose(zmns[-1], Zb_cs.squeeze(), atol=1e-14)

    np.testing.assert_allclose(rax_cc, axis1.R_n, atol=1e-14)
    np.testing.assert_allclose(zax_cs[1:][::-1], axis1.Z_n, atol=1e-14)

    f.close()


@pytest.mark.unit
def test_vmec_save_kinetic(TmpDir):
    """Tests that saving an equilibrium with kinetic profiles runs without errors."""
    output_path = str(TmpDir.join("output.nc"))
    eq = Equilibrium(
        L=2,
        M=2,
        N=2,
        NFP=3,
        electron_density=np.array([[0, 1], [2, -1]]),
        electron_temperature=np.array([[0, 1], [2, -1]]),
        sym=True,
    )
    VMECIO.save(eq, output_path, M_grid=8, N_grid=8)


@pytest.mark.regression
@pytest.mark.slow
def test_vmec_save_1(VMEC_save):
    """Tests that saving in NetCDF format agrees with VMEC."""
    vmec, desc = VMEC_save
    # parameters
    assert vmec.variables["version_"][:] == desc.variables["version_"][:]
    assert vmec.variables["mgrid_mode"][:] == desc.variables["mgrid_mode"][:]
    assert np.all(
        np.char.compare_chararrays(
            vmec.variables["mgrid_file"][:],
            desc.variables["mgrid_file"][:],
            "==",
            False,
        )
    )
    assert vmec.variables["ier_flag"][:] == desc.variables["ier_flag"][:]
    assert (
        vmec.variables["lfreeb__logical__"][:] == desc.variables["lfreeb__logical__"][:]
    )
    assert (
        vmec.variables["lrecon__logical__"][:] == desc.variables["lrecon__logical__"][:]
    )
    assert vmec.variables["lrfp__logical__"][:] == desc.variables["lrfp__logical__"][:]
    assert (
        vmec.variables["lasym__logical__"][:] == desc.variables["lasym__logical__"][:]
    )
    assert vmec.variables["nfp"][:] == desc.variables["nfp"][:]
    assert vmec.variables["ns"][:] == desc.variables["ns"][:]
    assert vmec.variables["mpol"][:] == desc.variables["mpol"][:]
    assert vmec.variables["ntor"][:] == desc.variables["ntor"][:]
    assert vmec.variables["mnmax"][:] == desc.variables["mnmax"][:]
    np.testing.assert_allclose(vmec.variables["xm"][:], desc.variables["xm"][:])
    np.testing.assert_allclose(vmec.variables["xn"][:], desc.variables["xn"][:])
    assert vmec.variables["mnmax_nyq"][:] == desc.variables["mnmax_nyq"][:]
    np.testing.assert_allclose(vmec.variables["xm_nyq"][:], desc.variables["xm_nyq"][:])
    np.testing.assert_allclose(vmec.variables["xn_nyq"][:], desc.variables["xn_nyq"][:])
    assert vmec.variables["signgs"][:] == desc.variables["signgs"][:]
    assert vmec.variables["gamma"][:] == desc.variables["gamma"][:]
    assert vmec.variables["nextcur"][:] == desc.variables["nextcur"][:]
    assert np.all(
        np.char.compare_chararrays(
            vmec.variables["pmass_type"][:],
            desc.variables["pmass_type"][:],
            "==",
            False,
        )
    )
    assert np.all(
        np.char.compare_chararrays(
            vmec.variables["piota_type"][:],
            desc.variables["piota_type"][:],
            "==",
            False,
        )
    )
    assert np.all(
        np.char.compare_chararrays(
            vmec.variables["pcurr_type"][:],
            desc.variables["pcurr_type"][:],
            "==",
            False,
        )
    )
    np.testing.assert_allclose(
        vmec.variables["am"][:], desc.variables["am"][:], atol=1e-5
    )
    np.testing.assert_allclose(
        vmec.variables["ai"][:], desc.variables["ai"][:], atol=1e-8
    )
    np.testing.assert_allclose(
        vmec.variables["ac"][:], desc.variables["ac"][:], atol=1e-8
    )
    np.testing.assert_allclose(
        vmec.variables["presf"][:], desc.variables["presf"][:], atol=2e-2
    )
    np.testing.assert_allclose(vmec.variables["pres"][:], desc.variables["pres"][:])
    np.testing.assert_allclose(vmec.variables["mass"][:], desc.variables["mass"][:])
    np.testing.assert_allclose(vmec.variables["iotaf"][:], desc.variables["iotaf"][:])
    np.testing.assert_allclose(
        vmec.variables["q_factor"][:], desc.variables["q_factor"][:]
    )
    np.testing.assert_allclose(vmec.variables["iotas"][:], desc.variables["iotas"][:])
    np.testing.assert_allclose(vmec.variables["phi"][:], desc.variables["phi"][:])
    np.testing.assert_allclose(vmec.variables["phipf"][:], desc.variables["phipf"][:])
    np.testing.assert_allclose(vmec.variables["phips"][:], desc.variables["phips"][:])
    np.testing.assert_allclose(
        vmec.variables["chi"][:], desc.variables["chi"][:], atol=2e-5
    )
    np.testing.assert_allclose(vmec.variables["chipf"][:], desc.variables["chipf"][:])
    np.testing.assert_allclose(
        vmec.variables["Rmajor_p"][:], desc.variables["Rmajor_p"][:], rtol=3e-4
    )
    np.testing.assert_allclose(
        vmec.variables["Aminor_p"][:],
        desc.variables["Aminor_p"][:],
        rtol=2e-4,
    )
    np.testing.assert_allclose(
        vmec.variables["aspect"][:], desc.variables["aspect"][:], rtol=4e-4
    )
    np.testing.assert_allclose(
        vmec.variables["volume_p"][:], desc.variables["volume_p"][:], rtol=1e-5
    )
    np.testing.assert_allclose(
        vmec.variables["volavgB"][:], desc.variables["volavgB"][:], rtol=1e-5
    )
    np.testing.assert_allclose(
        vmec.variables["betatotal"][:], desc.variables["betatotal"][:], rtol=1e-5
    )
    np.testing.assert_allclose(
        vmec.variables["betapol"][:], desc.variables["betapol"][:], rtol=1e-5
    )
    np.testing.assert_allclose(
        vmec.variables["betator"][:], desc.variables["betator"][:], rtol=1e-5
    )
    np.testing.assert_allclose(vmec.variables["wb"][:], desc.variables["wb"][:])
    np.testing.assert_allclose(
        vmec.variables["wp"][:], desc.variables["wp"][:], rtol=1e-6
    )
    np.testing.assert_allclose(
        vmec.variables["ctor"][:], desc.variables["ctor"][:], rtol=1e-5
    )
    np.testing.assert_allclose(
        vmec.variables["rbtor"][:], desc.variables["rbtor"][:], rtol=1e-5
    )
    np.testing.assert_allclose(
        vmec.variables["rbtor0"][:], desc.variables["rbtor0"][:], rtol=1e-5
    )
    np.testing.assert_allclose(
        vmec.variables["b0"][:], desc.variables["b0"][:], rtol=5e-5
    )
    np.testing.assert_allclose(
        vmec.variables["buco"][20:100], desc.variables["buco"][20:100], rtol=1e-5
    )
    np.testing.assert_allclose(
        vmec.variables["bvco"][20:100], desc.variables["bvco"][20:100], rtol=1e-5
    )
    np.testing.assert_allclose(
        vmec.variables["vp"][20:100], desc.variables["vp"][20:100], rtol=1e-6
    )
    np.testing.assert_allclose(
        vmec.variables["bdotb"][20:100], desc.variables["bdotb"][20:100], rtol=1e-6
    )
    np.testing.assert_allclose(
        vmec.variables["jdotb"][20:100], desc.variables["jdotb"][20:100], rtol=1e-5
    )
    np.testing.assert_allclose(
        vmec.variables["jcuru"][20:100], desc.variables["jcuru"][20:100], rtol=1e-2
    )
    np.testing.assert_allclose(
        vmec.variables["jcurv"][20:100], desc.variables["jcurv"][20:100], rtol=3e-2
    )
    np.testing.assert_allclose(
        vmec.variables["DShear"][20:100], desc.variables["DShear"][20:100], rtol=1e-2
    )
    np.testing.assert_allclose(
        vmec.variables["DCurr"][20:100], desc.variables["DCurr"][20:100], rtol=1e-2
    )
    np.testing.assert_allclose(
        vmec.variables["DWell"][20:100], desc.variables["DWell"][20:100], rtol=1e-2
    )
    np.testing.assert_allclose(
        vmec.variables["DGeod"][20:100], desc.variables["DGeod"][20:100], atol=1e-9
    )
    np.testing.assert_allclose(
        vmec.variables["DMerc"][20:100], desc.variables["DMerc"][20:100], rtol=5e-2
    )
    np.testing.assert_allclose(
        vmec.variables["raxis_cc"][:], desc.variables["raxis_cc"][:], rtol=5e-5
    )
    np.testing.assert_allclose(
        vmec.variables["zaxis_cs"][:], desc.variables["zaxis_cs"][:], rtol=5e-5
    )
    np.testing.assert_allclose(
        vmec.variables["rmin_surf"][:], desc.variables["rmin_surf"][:], rtol=5e-3
    )
    np.testing.assert_allclose(
        vmec.variables["rmax_surf"][:], desc.variables["rmax_surf"][:], rtol=5e-3
    )
    np.testing.assert_allclose(
        vmec.variables["zmax_surf"][:], desc.variables["zmax_surf"][:], rtol=5e-3
    )
    np.testing.assert_allclose(
        vmec.variables["beta_vol"][:], desc.variables["beta_vol"][:], rtol=5e-5
    )
    np.testing.assert_allclose(
        vmec.variables["betaxis"][:], desc.variables["betaxis"][:], rtol=5e-5
    )


@pytest.mark.regression
@pytest.mark.slow
def test_vmec_save_2(VMEC_save):
    """Tests that saving in NetCDF format agrees with VMEC."""
    vmec, desc = VMEC_save

    # straight field-line grid to compare quantities in full volume
    grid = LinearGrid(L=15, M=6, N=0, NFP=desc.variables["nfp"][:])
    theta_vmec = VMECIO.compute_theta_coords(
        vmec.variables["lmns"][:],
        vmec.variables["xm"][:],
        vmec.variables["xn"][:],
        grid.nodes[:, 0],
        grid.nodes[:, 1],
        grid.nodes[:, 2],
    )
    theta_desc = VMECIO.compute_theta_coords(
        desc.variables["lmns"][:],
        desc.variables["xm"][:],
        desc.variables["xn"][:],
        grid.nodes[:, 0],
        grid.nodes[:, 1],
        grid.nodes[:, 2],
    )

    # R & Z
    R_vmec, Z_vmec = VMECIO.vmec_interpolate(
        vmec.variables["rmnc"][:],
        vmec.variables["zmns"][:],
        vmec.variables["xm"][:],
        vmec.variables["xn"][:],
        theta=theta_vmec,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=True,
    )
    R_desc, Z_desc = VMECIO.vmec_interpolate(
        desc.variables["rmnc"][:],
        desc.variables["zmns"][:],
        desc.variables["xm"][:],
        desc.variables["xn"][:],
        theta=theta_desc,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=True,
    )
    np.testing.assert_allclose(R_vmec, R_desc, rtol=1e-3)
    np.testing.assert_allclose(Z_vmec, Z_desc, rtol=1e-3)

    # |B|
    b_vmec = VMECIO.vmec_interpolate(
        vmec.variables["bmnc"][:],
        np.zeros_like(vmec.variables["bmnc"][:]),
        vmec.variables["xm_nyq"][:],
        vmec.variables["xn_nyq"][:],
        theta=theta_vmec,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=False,
    )
    b_desc = VMECIO.vmec_interpolate(
        desc.variables["bmnc"][:],
        np.zeros_like(desc.variables["bmnc"][:]),
        desc.variables["xm_nyq"][:],
        desc.variables["xn_nyq"][:],
        theta=theta_desc,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=False,
    )
    np.testing.assert_allclose(b_vmec, b_desc, rtol=1e-3)

    # B^zeta
    bsupv_vmec = VMECIO.vmec_interpolate(
        vmec.variables["bsupvmnc"][:],
        np.zeros_like(vmec.variables["bsupvmnc"][:]),
        vmec.variables["xm_nyq"][:],
        vmec.variables["xn_nyq"][:],
        theta=theta_vmec,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=False,
    )
    bsupv_desc = VMECIO.vmec_interpolate(
        desc.variables["bsupvmnc"][:],
        np.zeros_like(desc.variables["bsupvmnc"][:]),
        desc.variables["xm_nyq"][:],
        desc.variables["xn_nyq"][:],
        theta=theta_desc,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=False,
    )
    np.testing.assert_allclose(bsupv_vmec, bsupv_desc, rtol=1e-3)

    # B_zeta
    bsubv_vmec = VMECIO.vmec_interpolate(
        vmec.variables["bsubvmnc"][:],
        np.zeros_like(vmec.variables["bsubvmnc"][:]),
        vmec.variables["xm_nyq"][:],
        vmec.variables["xn_nyq"][:],
        theta=theta_vmec,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=False,
    )
    bsubv_desc = VMECIO.vmec_interpolate(
        desc.variables["bsubvmnc"][:],
        np.zeros_like(desc.variables["bsubvmnc"][:]),
        desc.variables["xm_nyq"][:],
        desc.variables["xn_nyq"][:],
        theta=theta_desc,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=False,
    )
    np.testing.assert_allclose(bsubv_vmec, bsubv_desc, rtol=1e-3)

    # straight field-line grid to compare quantities on boundary
    grid = LinearGrid(M=6, N=0, NFP=desc.variables["nfp"][:], rho=np.array([1.0]))
    theta_vmec = VMECIO.compute_theta_coords(
        vmec.variables["lmns"][:],
        vmec.variables["xm"][:],
        vmec.variables["xn"][:],
        grid.nodes[:, 0],
        grid.nodes[:, 1],
        grid.nodes[:, 2],
    )
    theta_desc = VMECIO.compute_theta_coords(
        desc.variables["lmns"][:],
        desc.variables["xm"][:],
        desc.variables["xn"][:],
        grid.nodes[:, 0],
        grid.nodes[:, 1],
        grid.nodes[:, 2],
    )

    # lambda
    L_vmec = VMECIO.vmec_interpolate(
        np.zeros_like(vmec.variables["lmns"][:]),
        vmec.variables["lmns"][:],
        vmec.variables["xm"][:],
        vmec.variables["xn"][:],
        theta=theta_vmec,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=False,
    )
    L_desc = VMECIO.vmec_interpolate(
        np.zeros_like(desc.variables["lmns"][:]),
        desc.variables["lmns"][:],
        desc.variables["xm"][:],
        desc.variables["xn"][:],
        theta=theta_desc,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=False,
    )
    np.testing.assert_allclose(L_vmec, L_desc, rtol=1e-2)

    # Jacobian
    g_vmec = VMECIO.vmec_interpolate(
        vmec.variables["gmnc"][:],
        np.zeros_like(vmec.variables["gmnc"][:]),
        vmec.variables["xm_nyq"][:],
        vmec.variables["xn_nyq"][:],
        theta=theta_vmec,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=False,
    )
    g_desc = VMECIO.vmec_interpolate(
        desc.variables["gmnc"][:],
        np.zeros_like(desc.variables["gmnc"][:]),
        desc.variables["xm_nyq"][:],
        desc.variables["xn_nyq"][:],
        theta=theta_desc,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=False,
    )
    np.testing.assert_allclose(g_vmec, g_desc, rtol=1e-2)

    # B^theta
    bsupu_vmec = VMECIO.vmec_interpolate(
        vmec.variables["bsupumnc"][:],
        np.zeros_like(vmec.variables["bsupumnc"][:]),
        vmec.variables["xm_nyq"][:],
        vmec.variables["xn_nyq"][:],
        theta=theta_vmec,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=False,
    )
    bsupu_desc = VMECIO.vmec_interpolate(
        desc.variables["bsupumnc"][:],
        np.zeros_like(desc.variables["bsupumnc"][:]),
        desc.variables["xm_nyq"][:],
        desc.variables["xn_nyq"][:],
        theta=theta_desc,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=False,
    )
    np.testing.assert_allclose(bsupu_vmec, bsupu_desc, rtol=1e-2)

    # B_theta
    bsubu_vmec = VMECIO.vmec_interpolate(
        vmec.variables["bsubumnc"][:],
        np.zeros_like(vmec.variables["bsubumnc"][:]),
        vmec.variables["xm_nyq"][:],
        vmec.variables["xn_nyq"][:],
        theta=theta_vmec,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=False,
    )
    bsubu_desc = VMECIO.vmec_interpolate(
        desc.variables["bsubumnc"][:],
        np.zeros_like(desc.variables["bsubumnc"][:]),
        desc.variables["xm_nyq"][:],
        desc.variables["xn_nyq"][:],
        theta=theta_desc,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=False,
    )
    np.testing.assert_allclose(bsubu_vmec, bsubu_desc, rtol=1e-2)

    # B_psi
    bsubs_vmec = VMECIO.vmec_interpolate(
        np.zeros_like(vmec.variables["bsubsmns"][:]),
        vmec.variables["bsubsmns"][:],
        vmec.variables["xm_nyq"][:],
        vmec.variables["xn_nyq"][:],
        theta=theta_vmec,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=False,
    )
    bsubs_desc = VMECIO.vmec_interpolate(
        np.zeros_like(desc.variables["bsubsmns"][:]),
        desc.variables["bsubsmns"][:],
        desc.variables["xm_nyq"][:],
        desc.variables["xn_nyq"][:],
        theta=theta_desc,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=False,
    )
    np.testing.assert_allclose(bsubs_vmec, bsubs_desc, rtol=1e-2, atol=2e-3)

    # J^theta
    curru_vmec = VMECIO.vmec_interpolate(
        vmec.variables["currumnc"][:],
        np.zeros_like(vmec.variables["currumnc"][:]),
        vmec.variables["xm_nyq"][:],
        vmec.variables["xn_nyq"][:],
        theta=theta_vmec,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=False,
    )
    curru_desc = VMECIO.vmec_interpolate(
        desc.variables["currumnc"][:],
        np.zeros_like(desc.variables["currumnc"][:]),
        desc.variables["xm_nyq"][:],
        desc.variables["xn_nyq"][:],
        theta=theta_desc,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=False,
    )
    np.testing.assert_allclose(curru_vmec, curru_desc, rtol=1e-2)

    # J^zeta
    currv_vmec = VMECIO.vmec_interpolate(
        vmec.variables["currvmnc"][:],
        np.zeros_like(vmec.variables["currvmnc"][:]),
        vmec.variables["xm_nyq"][:],
        vmec.variables["xn_nyq"][:],
        theta=theta_vmec,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=False,
    )
    currv_desc = VMECIO.vmec_interpolate(
        desc.variables["currvmnc"][:],
        np.zeros_like(desc.variables["currvmnc"][:]),
        desc.variables["xm_nyq"][:],
        desc.variables["xn_nyq"][:],
        theta=theta_desc,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=False,
    )
    np.testing.assert_allclose(currv_vmec, currv_desc, rtol=1e-2)


@pytest.mark.regression
@pytest.mark.slow
def test_vmec_save_asym(VMEC_save_asym):
    """Tests that saving in NetCDF format agrees with VMEC."""
    vmec, desc, eq = VMEC_save_asym
    # first, compare some quantities which don't require calculation
    assert vmec.variables["version_"][:] == desc.variables["version_"][:]
    assert vmec.variables["mgrid_mode"][:] == desc.variables["mgrid_mode"][:]
    assert np.all(
        np.char.compare_chararrays(
            vmec.variables["mgrid_file"][:],
            desc.variables["mgrid_file"][:],
            "==",
            False,
        )
    )
    assert vmec.variables["ier_flag"][:] == desc.variables["ier_flag"][:]
    assert (
        vmec.variables["lfreeb__logical__"][:] == desc.variables["lfreeb__logical__"][:]
    )
    assert (
        vmec.variables["lrecon__logical__"][:] == desc.variables["lrecon__logical__"][:]
    )
    assert vmec.variables["lrfp__logical__"][:] == desc.variables["lrfp__logical__"][:]
    assert (
        vmec.variables["lasym__logical__"][:] == desc.variables["lasym__logical__"][:]
    )
    assert vmec.variables["nfp"][:] == desc.variables["nfp"][:]
    assert vmec.variables["ns"][:] == desc.variables["ns"][:]
    assert vmec.variables["mpol"][:] == desc.variables["mpol"][:]
    assert vmec.variables["ntor"][:] == desc.variables["ntor"][:]
    assert vmec.variables["mnmax"][:] == desc.variables["mnmax"][:]
    np.testing.assert_allclose(vmec.variables["xm"][:], desc.variables["xm"][:])
    np.testing.assert_allclose(vmec.variables["xn"][:], desc.variables["xn"][:])
    assert vmec.variables["mnmax_nyq"][:] == desc.variables["mnmax_nyq"][:]
    np.testing.assert_allclose(vmec.variables["xm_nyq"][:], desc.variables["xm_nyq"][:])
    np.testing.assert_allclose(vmec.variables["xn_nyq"][:], desc.variables["xn_nyq"][:])
    assert vmec.variables["signgs"][:] == desc.variables["signgs"][:]
    assert vmec.variables["gamma"][:] == desc.variables["gamma"][:]
    assert vmec.variables["nextcur"][:] == desc.variables["nextcur"][:]
    assert np.all(
        np.char.compare_chararrays(
            vmec.variables["pmass_type"][:],
            desc.variables["pmass_type"][:],
            "==",
            False,
        )
    )
    assert np.all(
        np.char.compare_chararrays(
            vmec.variables["piota_type"][:],
            desc.variables["piota_type"][:],
            "==",
            False,
        )
    )
    assert np.all(
        np.char.compare_chararrays(
            vmec.variables["pcurr_type"][:],
            desc.variables["pcurr_type"][:],
            "==",
            False,
        )
    )
    np.testing.assert_allclose(
        vmec.variables["am"][:], desc.variables["am"][:], atol=1e-5
    )
    np.testing.assert_allclose(
        vmec.variables["ai"][:], desc.variables["ai"][:], atol=1e-8
    )
    np.testing.assert_allclose(
        vmec.variables["ac"][:], desc.variables["ac"][:], atol=3e-5
    )
    np.testing.assert_allclose(
        vmec.variables["presf"][:], desc.variables["presf"][:], atol=2e-5
    )
    np.testing.assert_allclose(vmec.variables["pres"][:], desc.variables["pres"][:])
    np.testing.assert_allclose(vmec.variables["mass"][:], desc.variables["mass"][:])
    np.testing.assert_allclose(
        vmec.variables["iotaf"][:], desc.variables["iotaf"][:], rtol=5e-4
    )
    np.testing.assert_allclose(
        vmec.variables["q_factor"][:], desc.variables["q_factor"][:], rtol=5e-4
    )
    np.testing.assert_allclose(
        vmec.variables["iotas"][:], desc.variables["iotas"][:], rtol=5e-4
    )
    np.testing.assert_allclose(vmec.variables["phi"][:], desc.variables["phi"][:])
    np.testing.assert_allclose(vmec.variables["phipf"][:], desc.variables["phipf"][:])
    np.testing.assert_allclose(vmec.variables["phips"][:], desc.variables["phips"][:])
    np.testing.assert_allclose(
        vmec.variables["chi"][:], desc.variables["chi"][:], atol=3e-5, rtol=1e-3
    )
    np.testing.assert_allclose(
        vmec.variables["chipf"][:], desc.variables["chipf"][:], atol=3e-5, rtol=1e-3
    )
    np.testing.assert_allclose(
        vmec.variables["Rmajor_p"][:], desc.variables["Rmajor_p"][:]
    )
    np.testing.assert_allclose(
        vmec.variables["Aminor_p"][:], desc.variables["Aminor_p"][:]
    )
    np.testing.assert_allclose(vmec.variables["aspect"][:], desc.variables["aspect"][:])
    np.testing.assert_allclose(
        vmec.variables["volume_p"][:], desc.variables["volume_p"][:], rtol=1e-5
    )
    np.testing.assert_allclose(
        vmec.variables["volavgB"][:], desc.variables["volavgB"][:], rtol=1e-5
    )
    np.testing.assert_allclose(
        vmec.variables["betatotal"][:], desc.variables["betatotal"][:], rtol=1e-5
    )
    np.testing.assert_allclose(
        vmec.variables["betapol"][:], desc.variables["betapol"][:], rtol=1e-5
    )
    np.testing.assert_allclose(
        vmec.variables["betator"][:], desc.variables["betator"][:], rtol=1e-5
    )
    np.testing.assert_allclose(
        vmec.variables["ctor"][:],
        desc.variables["ctor"][:],
        atol=1e-9,  # it is a zero current solve
    )
    np.testing.assert_allclose(
        vmec.variables["rbtor"][:], desc.variables["rbtor"][:], rtol=1e-5
    )
    np.testing.assert_allclose(
        vmec.variables["rbtor0"][:], desc.variables["rbtor0"][:], rtol=1e-5
    )
    np.testing.assert_allclose(
        vmec.variables["b0"][:], desc.variables["b0"][:], rtol=4e-3
    )
    np.testing.assert_allclose(
        vmec.variables["buco"][20:100], desc.variables["buco"][20:100], atol=1e-15
    )
    np.testing.assert_allclose(
        vmec.variables["bvco"][20:100], desc.variables["bvco"][20:100], rtol=1e-5
    )
    np.testing.assert_allclose(
        vmec.variables["vp"][20:100], desc.variables["vp"][20:100], rtol=3e-4
    )
    np.testing.assert_allclose(
        vmec.variables["bdotb"][20:100], desc.variables["bdotb"][20:100], rtol=3e-4
    )
    np.testing.assert_allclose(
        vmec.variables["jdotb"][20:100],
        desc.variables["jdotb"][20:100],
        atol=4e-3,  # nearly zero bc is vacuum
    )
    np.testing.assert_allclose(
        vmec.variables["jcuru"][20:100], desc.variables["jcuru"][20:100], atol=2
    )
    np.testing.assert_allclose(
        vmec.variables["jcurv"][20:100], desc.variables["jcurv"][20:100], rtol=2
    )
    np.testing.assert_allclose(
        vmec.variables["DShear"][20:100], desc.variables["DShear"][20:100], rtol=6e-2
    )
    np.testing.assert_allclose(
        vmec.variables["DCurr"][20:100],
        desc.variables["DCurr"][20:100],
        atol=1e-4,  # nearly zero bc vacuum
    )
    np.testing.assert_allclose(
        vmec.variables["DWell"][20:100], desc.variables["DWell"][20:100], rtol=1e-2
    )
    np.testing.assert_allclose(
        vmec.variables["DGeod"][20:100],
        desc.variables["DGeod"][20:100],
        atol=4e-3,
        rtol=1e-2,
    )

    # the Mercier stability is pretty off,
    # but these are not exactly similar solutions to eachother
    np.testing.assert_allclose(
        vmec.variables["DMerc"][20:100], desc.variables["DMerc"][20:100], atol=4e-3
    )
    np.testing.assert_allclose(
        vmec.variables["raxis_cc"][:],
        desc.variables["raxis_cc"][:],
        rtol=5e-5,
        atol=4e-3,
    )
    np.testing.assert_allclose(
        vmec.variables["zaxis_cs"][:],
        desc.variables["zaxis_cs"][:],
        rtol=5e-5,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        vmec.variables["rmin_surf"][:], desc.variables["rmin_surf"][:], rtol=5e-3
    )
    np.testing.assert_allclose(
        vmec.variables["rmax_surf"][:], desc.variables["rmax_surf"][:], rtol=5e-3
    )
    np.testing.assert_allclose(
        vmec.variables["zmax_surf"][:], desc.variables["zmax_surf"][:], rtol=5e-3
    )
    np.testing.assert_allclose(
        vmec.variables["beta_vol"][:], desc.variables["beta_vol"][:], rtol=5e-5
    )
    np.testing.assert_allclose(
        vmec.variables["betaxis"][:], desc.variables["betaxis"][:], rtol=5e-5
    )
    # Next, calculate some quantities and compare
    # the DESC wout -> DESC (should be very close)
    # and the DESC wout -> VMEC wout (should be approximately close)
    surfs = desc.variables["ns"][:]
    s_full = np.linspace(0, 1, surfs)
    s_half = s_full[0:-1] + 0.5 / (surfs - 1)
    r_full = np.sqrt(s_full)
    r_half = np.sqrt(s_half)

    vol_grid = LinearGrid(
        rho=r_full[10::10],
        M=15,
        N=15,
        NFP=eq.NFP,
        axis=False,
        sym=False,
    )
    vol_half_grid = LinearGrid(
        rho=r_half[10::10],
        M=15,
        N=15,
        NFP=eq.NFP,
        axis=False,
        sym=False,
    )
    bdry_grid = LinearGrid(rho=1.0, M=15, N=15, NFP=eq.NFP, axis=False, sym=False)

    def test(
        nc_str,
        desc_str,
        negate_DESC_quant=False,
        use_nyq=True,
        convert_sqrt_g_or_B_rho=False,
        atol_desc_desc_wout=5e-5,
        rtol_desc_desc_wout=1e-5,
        atol_vmec_desc_wout=1e-5,
        rtol_vmec_desc_wout=1e-2,
        grid=vol_grid,
        is_half_grid=False,
    ):
        """Helper fxn to evaluate Fourier series from wout and compare to DESC."""
        xm = desc.variables["xm_nyq"][:] if use_nyq else desc.variables["xm"][:]
        xn = desc.variables["xn_nyq"][:] if use_nyq else desc.variables["xn"][:]

        si = np.insert(s_half, 0, 0) if is_half_grid else s_full
        rho = grid.nodes[:, 0]
        s = rho**2
        # some quantities must be negated before comparison bc
        # they are negative in the wout i.e. B^theta
        negate = -1 if negate_DESC_quant else 1

        quant_from_desc_wout = VMECIO.vmec_interpolate(
            desc.variables[nc_str + "c"][:],
            desc.variables[nc_str + "s"][:],
            xm,
            xn,
            theta=-grid.nodes[:, 1],  # -theta bc when we save wout we reverse theta
            phi=grid.nodes[:, 2],
            s=s,
            sym=False,
            si=si,
        )

        quant_from_vmec_wout = VMECIO.vmec_interpolate(
            vmec.variables[nc_str + "c"][:],
            vmec.variables[nc_str + "s"][:],
            xm,
            xn,
            # pi - theta bc VMEC, when it gets a CW angle bdry,
            # changes poloidal angle to  theta -> pi-theta
            theta=np.pi - grid.nodes[:, 1],
            phi=grid.nodes[:, 2],
            s=s,
            sym=False,
            si=si,
        )

        data = eq.compute(["rho", "sqrt(g)", desc_str], grid=grid)
        # convert sqrt(g) or B_rho->B_psi if needed
        quant_desc = (
            data[desc_str] / 2 / data["rho"]
            if convert_sqrt_g_or_B_rho
            else data[desc_str]
        )

        # add sqrt(g) factor if currents being compared
        quant_desc = (
            quant_desc * abs(data["sqrt(g)"]) / 2 / data["rho"]
            if "J" in desc_str
            else quant_desc
        )

        np.testing.assert_allclose(
            negate * quant_desc,
            quant_from_desc_wout,
            atol=atol_desc_desc_wout,
            rtol=rtol_desc_desc_wout,
        )
        np.testing.assert_allclose(
            quant_from_desc_wout,
            quant_from_vmec_wout,
            atol=atol_vmec_desc_wout,
            rtol=rtol_vmec_desc_wout,
        )

    # R & Z
    test("rmn", "R", use_nyq=False)
    test("zmn", "Z", use_nyq=False, atol_vmec_desc_wout=1e-2)
    test(
        "lmn",
        "lambda",
        use_nyq=False,
        atol_vmec_desc_wout=1e-2,
        negate_DESC_quant=True,
        grid=vol_half_grid,
        is_half_grid=True,
    )
    # |B|
    test("bmn", "|B|", rtol_desc_desc_wout=7e-4)

    # B^zeta
    test("bsupvmn", "B^zeta")

    # B_zeta
    test("bsubvmn", "B_zeta", grid=vol_half_grid, is_half_grid=True)

    # hard to compare to VMEC for the currents, since
    # VMEC F error is worse and equilibria are not exactly similar
    # just compare back to DESC
    test("currumn", "J^theta", atol_vmec_desc_wout=1e4)
    test("currvmn", "J^zeta", negate_DESC_quant=True, atol_vmec_desc_wout=1e5)

    test(
        "gmn",
        "sqrt(g)",
        convert_sqrt_g_or_B_rho=True,
        negate_DESC_quant=True,
        grid=vol_half_grid,
        is_half_grid=True,
    )

    # Compare B_psi B^theta and B_theta at bdry only
    test(
        "bsupumn",
        "B^theta",
        negate_DESC_quant=True,
        grid=bdry_grid,
        is_half_grid=True,
        atol_vmec_desc_wout=6e-4,
    )
    test(
        "bsubumn",
        "B_theta",
        negate_DESC_quant=True,
        grid=bdry_grid,
        is_half_grid=True,
    )
    test(
        "bsubsmn",
        "B_rho",
        convert_sqrt_g_or_B_rho=True,
        grid=bdry_grid,
        atol_vmec_desc_wout=2e-3,
    )


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=1)
def test_plot_vmec_comparison():
    """Test that DESC and VMEC flux surface plots match."""
    eq = desc.examples.get("SOLOVEV")
    fig, ax = VMECIO.plot_vmec_comparison(eq, "tests/inputs/wout_SOLOVEV.nc")
    return fig


@pytest.mark.unit
def test_vmec_boundary_subspace(DummyStellarator):
    """Test VMEC boundary subspace is enforced properly."""
    eq = load(load_from=str(DummyStellarator["output_path"]), file_format="hdf5")

    RBC = np.array([[1, 2], [-1, 2], [1, 0], [2, 2]])
    ZBS = np.array([[2, 1], [-2, 1], [0, 2], [-1, 1]])
    opt_subspace = vmec_boundary_subspace(eq, RBC, ZBS)

    y_opt = np.arange(8)
    y = np.dot(y_opt, opt_subspace.T)

    m_desc = np.concatenate(
        (eq.surface.R_basis.modes[:, 1], eq.surface.Z_basis.modes[:, 1])
    )
    n_desc = np.concatenate(
        (eq.surface.R_basis.modes[:, 2], eq.surface.Z_basis.modes[:, 2])
    )

    m_vmec, n_vmec, zbs, rbc = ptolemy_identity_rev(m_desc, n_desc, y)

    tol = 1e-15
    rbc_ref = np.atleast_2d(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1]))
    zbs_ref = np.atleast_2d(np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0]))
    np.testing.assert_allclose(rbc_ref, np.abs(rbc) > tol)
    np.testing.assert_allclose(zbs_ref, np.abs(zbs) > tol)


@pytest.mark.unit
def test_make_boozmn_output_DESC(TmpDir):
    """Test that booz_xform-style outputs accurately reconstruct quantities."""
    # load in precise QA equilibrium
    eq = get("precise_QA")
    output_path = str(TmpDir.join("boozmn_out.nc"))

    boozer_res = 30
    surfs = 4
    # Use DESC to calculate the boozer harmonics and create a booz_xform style .nc file
    # on 3 surfaces (surfs-1 by booz_xform convention) using boozer resolution of 50
    make_boozmn_output(
        eq, output_path, surfs=surfs, verbose=2, M_booz=boozer_res, N_booz=boozer_res
    )
    # load in the .nc file
    file = Dataset(output_path, mode="r")

    # make half grid in s
    s_full = np.linspace(0, 1, surfs)
    hs = 1 / (surfs - 1)
    s_half = s_full[0:-1] + hs / 2

    R_mnc = file.variables["rmnc_b"][:].filled()
    Z_mns = file.variables["zmns_b"][:].filled()
    B_mnc = file.variables["bmnc_b"][:].filled()
    nu_B_mns = file.variables["pmns_b"][:].filled()
    g_mnc = file.variables["gmn_b"][:].filled()

    quantities = [R_mnc, Z_mns, B_mnc, g_mnc, nu_B_mns]
    quant_names = ["R", "Z", "|B|", "sqrt(g)_Boozer", "nu"]
    quant_parity = ["cos", "sin", "cos", "cos", "sin"]

    xm = file.variables["ixm_b"][:].filled()
    xn = file.variables["ixn_b"][:].filled()

    # create grid on which to check values btwn DESC eq.compute and the
    # evaluated Boozer harmonics for each quantity
    M = 20
    N = 20

    for surf_index in range(surfs - 1):
        rho = np.sqrt(s_half[surf_index])
        grid = LinearGrid(rho=rho, M=M, N=N, NFP=eq.NFP)
        data = eq.compute(
            ["theta_B", "zeta_B", "psi_r"] + quant_names,
            grid=grid,
            M_booz=boozer_res,
            N_booz=boozer_res,
        )
        # make the grid in Boozer angles corresponding
        # to the DESC theta,zeta angles that our quantities
        # are computed on
        grid_boozer = Grid(
            np.vstack(
                (np.ones_like(data["theta_B"]) * rho, data["theta_B"], data["zeta_B"])
            ).T,
            sort=False,
        )

        for quant_index in range(len(quantities)):
            name = quant_names[quant_index]
            quant = quantities[quant_index]
            parity = quant_parity[quant_index]

            # B_mn
            m, n, quant_mn = (
                ptolemy_identity_fwd(xm, xn, np.zeros_like(quant), quant)
                if parity == "cos"
                else ptolemy_identity_fwd(xm, xn, quant, np.zeros_like(quant))
            )
            # negate nu since we save it as negative of what our convention is
            # to comply with booz_xform notation
            quant_mn = -quant_mn if name == "nu" else quant_mn
            # mutliply by psi_r because the saved jacobian is
            # from (psi,theta_B,zeta_B) to lab frame, and we in DESC
            # calculate the jacobian for (rho, theta_B, zeta_B) to lab frame
            quant_mn = (
                quant_mn * data["psi_r"][0] if name == "sqrt(g)_Boozer" else quant_mn
            )

            # Must use sym=False even if eq.sym=True
            # because otherwise it claims that certain modes are not in the basis...
            basis = DoubleFourierSeries(
                round(np.max(m)), round(np.max(n) / eq.NFP), sym=False, NFP=eq.NFP
            )
            quant_mn = np.where(basis.modes[:, 1] < 0, -quant_mn, quant_mn)
            quant_mn_desc_basis = np.zeros((basis.num_modes,))
            for i, (mm, nn) in enumerate(zip(m, n / eq.NFP)):
                idx = basis.get_idx(L=0, M=mm, N=nn)
                quant_mn_desc_basis[idx] = quant_mn[surf_index, i]

            quant_trans = Transform(grid=grid_boozer, basis=basis)
            quant_from_booz = quant_trans.transform(quant_mn_desc_basis)

            np.testing.assert_allclose(
                quant_from_booz,
                data[name],
                rtol=8e-4,
                err_msg=f"{name} at surf index {surf_index}",
            )


@pytest.mark.slow
@pytest.mark.unit
def test_make_boozmn_output_DESC_asym(TmpDir):
    """Test that asym booz_xform-style outputs accurately reconstruct quantities."""
    # load in asymmetric equilibrium
    eq = load("./tests/inputs/NAE_QA_asym_eq_output.h5")
    print(eq)
    output_path = str(TmpDir.join("boozmn_asym_out.nc"))

    boozer_res = 30
    surfs = 3
    # Use DESC to calculate the boozer harmonics and create a booz_xform style .nc file
    # on 2 surfaces (surfs-1 by convention) using boozer resolution of 40
    with pytest.warns(UserWarning, match="numnc"):
        make_boozmn_output(
            eq,
            output_path,
            surfs=surfs,
            verbose=2,
            M_booz=boozer_res,
            N_booz=boozer_res,
        )
    # load in the .nc file
    file = Dataset(output_path, mode="r")

    # make half grid in s
    s_full = np.linspace(0, 1, surfs)
    hs = 1 / (surfs - 1)
    s_half = s_full[0:-1] + hs / 2

    R_mnc = file.variables["rmnc_b"][:].filled()
    R_mns = file.variables["rmns_b"][:].filled()

    Z_mns = file.variables["zmns_b"][:].filled()
    Z_mnc = file.variables["zmnc_b"][:].filled()

    B_mnc = file.variables["bmnc_b"][:].filled()
    B_mns = file.variables["bmns_b"][:].filled()

    nu_B_mns = file.variables["pmns_b"][:].filled()
    nu_B_mnc = file.variables["pmnc_b"][:].filled()

    g_mnc = file.variables["gmn_b"][:].filled()
    g_mns = file.variables["gmns_b"][:].filled()

    quantities = [
        [R_mnc, R_mns],
        [Z_mnc, Z_mns],
        [B_mnc, B_mns],
        [g_mnc, g_mns],
        [nu_B_mnc, nu_B_mns],
    ]
    quant_names = ["R", "Z", "|B|", "sqrt(g)_Boozer", "nu"]
    quant_parity = [
        ["cos", "sin"],
        ["cos", "sin"],
        ["cos", "sin"],
        ["cos", "sin"],
        ["cos", "sin"],
    ]

    xm = file.variables["ixm_b"][:].filled()
    xn = file.variables["ixn_b"][:].filled()

    # create grid on which to check values btwn DESC eq.compute and the
    # evaluated Boozer harmonics for each quantity
    M = 20
    N = 20

    for surf_index in range(surfs - 1):
        print("surface index", surf_index)
        rho = np.sqrt(s_half[surf_index])
        grid = LinearGrid(rho=rho, M=M, N=N, NFP=eq.NFP)
        data = eq.compute(
            ["theta_B", "zeta_B", "psi_r"] + quant_names,
            grid=grid,
            M_booz=boozer_res,
            N_booz=boozer_res,
        )
        # make the grid in Boozer angles corresponding
        # to the DESC theta,zeta angles that our quantities
        # are computed on
        grid_boozer = Grid(
            np.vstack(
                (np.ones_like(data["theta_B"]) * rho, data["theta_B"], data["zeta_B"])
            ).T,
            sort=False,
        )

        for quant_index in range(len(quantities)):
            name = quant_names[quant_index]
            quant_from_booz = None
            # must add up both the sin and cos terms
            for quant_index2 in [0, 1]:
                quant = quantities[quant_index][quant_index2]
                parity = quant_parity[quant_index][quant_index2]

                # B_mn
                m, n, quant_mn = (
                    ptolemy_identity_fwd(xm, xn, np.zeros_like(quant), quant)
                    if parity == "cos"
                    else ptolemy_identity_fwd(xm, xn, quant, np.zeros_like(quant))
                )
                # negate nu since we save it as negative of what our convention is
                # to comply with booz_xform notation
                quant_mn = -quant_mn if name == "nu" else quant_mn
                quant_mn = (
                    quant_mn * data["psi_r"][0]
                    if name == "sqrt(g)_Boozer"
                    else quant_mn
                )

                basis = DoubleFourierSeries(
                    round(np.max(m)), round(np.max(n) / eq.NFP), sym=False, NFP=eq.NFP
                )
                quant_mn = np.where(basis.modes[:, 1] < 0, -quant_mn, quant_mn)
                quant_mn_desc_basis = np.zeros((basis.num_modes,))
                for i, (mm, nn) in enumerate(zip(m, n / eq.NFP)):
                    idx = basis.get_idx(L=0, M=mm, N=nn)
                    quant_mn_desc_basis[idx] = quant_mn[surf_index, i]

                quant_trans = Transform(grid=grid_boozer, basis=basis)
                if np.any(quant_from_booz):  # add to it if it exists
                    quant_from_booz += quant_trans.transform(quant_mn_desc_basis)
                else:
                    quant_from_booz = quant_trans.transform(quant_mn_desc_basis)

            np.testing.assert_allclose(
                quant_from_booz,
                data[name],
                rtol=3e-3,
                atol=1e-7,
                err_msg=f"{name} at surf index {surf_index}",
            )


@pytest.mark.slow
@pytest.mark.unit
def test_make_boozmn_output_against_hidden_symmetries_booz_xform(TmpDir):
    """Test that booz_xform-style outputs compare well against C++ implementation."""
    # testing against https://github.com/hiddenSymmetries/booz_xform/tree/main
    # commit 881907058ece03
    # load in precise_QA equilibrium
    eq = get("precise_QA")
    output_path = str(TmpDir.join("boozmn_out.nc"))

    boozer_res_M = 25
    boozer_res_N = 25

    # compare against a 128 surface Mboz=Nboz=25 run of precise QA
    # with the hidden symmetries C++ booz_xform implementation
    # (ran on a wout created with VMECIO.save of precise QA example with 100 surfs
    # and Mnyq = Nnyq = 30)
    surfs = 128
    Cpp_booz_output_path = (
        f"./tests/inputs/boozmn_{surfs}_surfs_precise_QA"
        + f"_sims_booz_{boozer_res_M}.nc"
    )

    # Use DESC to calculate the boozer harmonics and create a booz_xform style .nc file
    make_boozmn_output(
        eq,
        output_path,
        surfs=4,
        verbose=0,
        M_booz=boozer_res_M,
        N_booz=boozer_res_N,
    )

    # load in the .nc file
    file = Dataset(output_path, mode="r")

    R_mnc = file.variables["rmnc_b"][:].filled()
    Z_mns = file.variables["zmns_b"][:].filled()
    B_mnc = file.variables["bmnc_b"][:].filled()
    nu_B_mns = file.variables["pmns_b"][:].filled()
    g_mnc = file.variables["gmn_b"][:].filled()
    I = file.variables["buco_b"][:].filled()
    G = file.variables["bvco_b"][:].filled()
    iota = file.variables["iota_b"][:].filled()

    quantities = [R_mnc, Z_mns, B_mnc, g_mnc, nu_B_mns, I, G, iota]
    quant_names = ["R", "Z", "|B|", "sqrt(g)_Boozer", "nu", "I", "G", "iota"]

    # load in the .nc from the cpp version

    file_cpp = Dataset(Cpp_booz_output_path, mode="r")

    R_mnc_cpp = file_cpp.variables["rmnc_b"][:].filled()
    Z_mns_cpp = file_cpp.variables["zmns_b"][:].filled()
    B_mnc_cpp = file_cpp.variables["bmnc_b"][:].filled()
    nu_B_mns_cpp = file_cpp.variables["pmns_b"][:].filled()
    g_mnc_cpp = file_cpp.variables["gmn_b"][:].filled()
    I_cpp = file_cpp.variables["buco_b"][:].filled()
    G_cpp = file_cpp.variables["bvco_b"][:].filled()
    iota_cpp = file_cpp.variables["iota_b"][:].filled()

    quantities_cpp = [
        R_mnc_cpp,
        Z_mns_cpp,
        B_mnc_cpp,
        g_mnc_cpp,
        nu_B_mns_cpp,
        I_cpp,
        G_cpp,
        iota_cpp,
    ]

    # the half-grid for ns=128 contains the 3 half-grid pts of ns=4
    # s = 1/6, 1/2, 5/6 so we will compare the quantities there
    s_full_cpp = np.linspace(0, 1, surfs)
    hs = 1 / (surfs - 1)
    s_half_cpp = s_full_cpp[0:-1] + hs / 2
    surf_ind_1_6 = np.argmin(abs(s_half_cpp - 1 / 6))
    surf_ind_1_2 = np.argmin(abs(s_half_cpp - 1 / 2))
    surf_ind_5_6 = np.argmin(abs(s_half_cpp - 5 / 6))

    for quant, quant_cpp, name in zip(quantities, quantities_cpp, quant_names):
        if quant.ndim == 1:  # test the 1D arrays I, G and iota
            np.testing.assert_allclose(
                quant[1],
                quant_cpp[surf_ind_1_6],
                rtol=1e-3,
                atol=7e-4,
                err_msg=f"{name} at s=1/6",
            )
            np.testing.assert_allclose(
                quant[2],
                quant_cpp[surf_ind_1_2],
                rtol=1e-3,
                atol=7e-4,
                err_msg=f"{name} at s=1/2",
            )
            np.testing.assert_allclose(
                quant[3],
                quant_cpp[surf_ind_5_6],
                rtol=1e-3,
                atol=7e-4,
                err_msg=f"{name} at s=5/6",
            )
        else:
            np.testing.assert_allclose(
                quant[0, :],
                quant_cpp[surf_ind_1_6, :],
                rtol=1e-3,
                atol=7e-4,
                err_msg=f"{name} at s=1/6",
            )
            np.testing.assert_allclose(
                quant[1, :],
                quant_cpp[surf_ind_1_2, :],
                rtol=1e-3,
                atol=7e-4,
                err_msg=f"{name} at s=1/2",
            )
            np.testing.assert_allclose(
                quant[2, :],
                quant_cpp[surf_ind_5_6, :],
                rtol=1e-3,
                atol=7e-4,
                err_msg=f"{name} at s=5/6",
            )
    # test some misc quantities
    misc = ["mboz_b", "nboz_b", "ixm_b", "ixn_b"]
    for name in misc:
        np.testing.assert_allclose(
            file.variables[name][:].filled(), file_cpp.variables[name][:].filled()
        )


@pytest.mark.regression
@pytest.mark.solve
def test_write_vmec_input(TmpDir):
    """Test generated VMEC input file gives the original equilibrium when solved."""
    # write VMEC input file
    fname = str(TmpDir.join("input.SOLOVEV"))
    eq0 = get("SOLOVEV")
    W0 = eq0.compute("W")["W"]
    VMECIO.write_vmec_input(eq0, fname)

    # read VMEC input file and override defaults to match desc/examples/SOLOVEV
    ir = InputReader()
    inputs = ir.parse_inputs(fname)
    inputs[0]["spectral_indexing"] = "fringe"
    inputs[0]["L"] = 24

    # solve from VMEC input file
    fam = EquilibriaFamily(inputs)
    fam.solve_continuation(
        objective=inputs[0]["objective"],
        optimizer=inputs[0]["optimizer"],
        pert_order=[inp["pert_order"] for inp in inputs],
        ftol=[inp["ftol"] for inp in inputs],
        xtol=[inp["xtol"] for inp in inputs],
        gtol=[inp["gtol"] for inp in inputs],
        maxiter=[inp["maxiter"] for inp in inputs],
        verbose=2,
    )
    eq1 = fam[-1]
    W1 = eq1.compute("W")["W"]

    # check that solution matches original equilibrium
    rho_err, theta_err = area_difference_desc(eq0, eq1)
    np.testing.assert_allclose(eq0.L, eq1.L)
    np.testing.assert_allclose(eq0.M, eq1.M)
    np.testing.assert_allclose(eq0.N, eq1.N)
    np.testing.assert_allclose(eq0.NFP, eq1.NFP)
    np.testing.assert_allclose(eq0.Psi, eq1.Psi)
    np.testing.assert_allclose(eq0.p_l, eq1.p_l)
    np.testing.assert_allclose(eq0.i_l, eq1.i_l)
    np.testing.assert_allclose(eq0.Rb_lmn, eq1.Rb_lmn)
    np.testing.assert_allclose(eq0.Zb_lmn, eq1.Zb_lmn)
    np.testing.assert_allclose(W0, W1)
    np.testing.assert_allclose(rho_err, 0, atol=1e-4)
    np.testing.assert_allclose(theta_err, 0, atol=1e-4)
