import unittest
import pytest
import numpy as np
from netCDF4 import Dataset

from desc.grid import LinearGrid
from desc.basis import FourierZernikeBasis
from desc.equilibrium import Equilibrium, EquilibriaFamily
from desc.vmec import VMECIO
from desc.vmec_utils import (
    ptolemy_identity_fwd,
    ptolemy_identity_rev,
    fourier_to_zernike,
    zernike_to_fourier,
    vmec_boundary_subspace,
)


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

        m_1, n_1, x = ptolemy_identity_fwd(m_0, n_0, s, c)

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

        m_0, n_0, s, c = ptolemy_identity_rev(m_1, n_1, x)

        np.testing.assert_allclose(m_0, m_0_correct, atol=1e-8)
        np.testing.assert_allclose(n_0, n_0_correct, atol=1e-8)
        np.testing.assert_allclose(s, s_correct, atol=1e-8)
        np.testing.assert_allclose(c, c_correct, atol=1e-8)

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
        basis = FourierZernikeBasis(L=-1, M=M, N=N, spectral_indexing="ansi")
        x_lmn = fourier_to_zernike(m, n, x_mn, basis)

        x_lmn_correct = np.zeros((basis.num_modes,))
        for k in range(basis.num_modes):
            idx = np.where((basis.modes == [np.abs(m[k]), m[k], n[k]]).all(axis=1))[0]
            x_lmn_correct[idx] = x[k]

        np.testing.assert_allclose(x_lmn, x_lmn_correct, atol=1e-8)

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
        basis = FourierZernikeBasis(L=-1, M=M, N=N, spectral_indexing="ansi")

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


@pytest.mark.slow
def test_load_then_save(TmpDir):
    """Tests if loading and then saving gives the original result."""

    input_path = "./tests/inputs/wout_SOLOVEV.nc"
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

    np.testing.assert_allclose(rmnc2, rmnc1, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(zmns2, zmns1, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(lmns2, lmns1, rtol=1e-3, atol=5e-2)

    file1.close
    file2.close


@pytest.mark.slow
def test_vmec_save(DSHAPE, TmpDir):
    """Tests that saving in NetCDF format agrees with VMEC."""

    vmec = Dataset(str(DSHAPE["vmec_nc_path"]), mode="r")
    eq = EquilibriaFamily.load(load_from=str(DSHAPE["desc_h5_path"]))[-1]
    eq.change_resolution(M=vmec.variables["mpol"][:] - 1, N=vmec.variables["ntor"][:])
    eq._solved = True
    VMECIO.save(
        eq, str(DSHAPE["desc_nc_path"]), surfs=vmec.variables["ns"][:], verbose=0
    )
    desc = Dataset(str(DSHAPE["desc_nc_path"]), mode="r")

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
        vmec.variables["am"][:], desc.variables["am"][:], atol=1e-1
    )
    np.testing.assert_allclose(
        vmec.variables["ai"][:], desc.variables["ai"][:], atol=1e-4
    )
    np.testing.assert_allclose(vmec.variables["ac"][:], desc.variables["ac"][:])
    np.testing.assert_allclose(
        vmec.variables["presf"][:], desc.variables["presf"][:], rtol=5e-2, atol=5e-2
    )
    np.testing.assert_allclose(vmec.variables["pres"][:], desc.variables["pres"][:])
    np.testing.assert_allclose(vmec.variables["mass"][:], desc.variables["mass"][:])
    np.testing.assert_allclose(vmec.variables["iotaf"][:], desc.variables["iotaf"][:])
    np.testing.assert_allclose(vmec.variables["iotas"][:], desc.variables["iotas"][:])
    np.testing.assert_allclose(vmec.variables["phi"][:], desc.variables["phi"][:])
    np.testing.assert_allclose(vmec.variables["phipf"][:], desc.variables["phipf"][:])
    np.testing.assert_allclose(vmec.variables["phips"][:], desc.variables["phips"][:])
    np.testing.assert_allclose(
        vmec.variables["chi"][:], desc.variables["chi"][:], rtol=1e-3, atol=1e-5
    )
    np.testing.assert_allclose(vmec.variables["chipf"][:], desc.variables["chipf"][:])
    np.testing.assert_allclose(
        vmec.variables["Rmajor_p"][:], desc.variables["Rmajor_p"][:]
    )
    np.testing.assert_allclose(
        vmec.variables["Aminor_p"][:], desc.variables["Aminor_p"][:]
    )
    np.testing.assert_allclose(vmec.variables["aspect"][:], desc.variables["aspect"][:])
    np.testing.assert_allclose(
        vmec.variables["volume_p"][:], desc.variables["volume_p"][:]
    )
    # raxis_cc & zaxis_cs excluded b/c VMEC saves initial guess, not final solution
    np.testing.assert_allclose(
        vmec.variables["rmin_surf"][:], desc.variables["rmin_surf"][:], rtol=5e-3
    )
    np.testing.assert_allclose(
        vmec.variables["rmax_surf"][:], desc.variables["rmax_surf"][:], rtol=5e-3
    )
    np.testing.assert_allclose(
        vmec.variables["zmax_surf"][:], desc.variables["zmax_surf"][:], rtol=5e-3
    )

    # straight field-line grid to compare quantities
    grid = LinearGrid(L=15, M=2, N=2, NFP=eq.NFP)
    vartheta_vmec = VMECIO.compute_theta_coords(
        vmec.variables["lmns"][:],
        vmec.variables["xm"][:],
        vmec.variables["xn"][:],
        grid.nodes[:, 0],
        grid.nodes[:, 1],
        grid.nodes[:, 2],
    )
    vartheta_desc = VMECIO.compute_theta_coords(
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
        theta=vartheta_vmec,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=True,
    )
    R_desc, Z_desc = VMECIO.vmec_interpolate(
        desc.variables["rmnc"][:],
        desc.variables["zmns"][:],
        desc.variables["xm"][:],
        desc.variables["xn"][:],
        theta=vartheta_desc,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=True,
    )
    np.testing.assert_allclose(R_vmec, R_desc, rtol=2e-3)
    np.testing.assert_allclose(Z_vmec, Z_desc, rtol=2e-3)

    # TODO: not testing Jacobian because VMEC & DESC coordinate systems are different

    # |B|
    b_vmec = VMECIO.vmec_interpolate(
        vmec.variables["bmnc"][:],
        np.zeros_like(vmec.variables["bmnc"][:]),
        vmec.variables["xm_nyq"][:],
        vmec.variables["xn_nyq"][:],
        theta=vartheta_vmec,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=False,
    )
    b_desc = VMECIO.vmec_interpolate(
        desc.variables["bmnc"][:],
        np.zeros_like(desc.variables["bmnc"][:]),
        desc.variables["xm_nyq"][:],
        desc.variables["xn_nyq"][:],
        theta=vartheta_desc,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=False,
    )
    np.testing.assert_allclose(b_vmec, b_desc, rtol=1e-3)

    # B^theta
    bsupu_vmec = VMECIO.vmec_interpolate(
        vmec.variables["bsupumnc"][:],
        np.zeros_like(vmec.variables["bsupumnc"][:]),
        vmec.variables["xm_nyq"][:],
        vmec.variables["xn_nyq"][:],
        theta=vartheta_vmec,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=False,
    )
    bsupu_desc = VMECIO.vmec_interpolate(
        desc.variables["bsupumnc"][:],
        np.zeros_like(desc.variables["bsupumnc"][:]),
        desc.variables["xm_nyq"][:],
        desc.variables["xn_nyq"][:],
        theta=vartheta_desc,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=False,
    )
    # FIXME: this is a bad test because VMEC and DESC use different poloidal angles
    np.testing.assert_allclose(bsupu_vmec, bsupu_desc, rtol=5e-2, atol=2e-2)

    # B^zeta
    bsupv_vmec = VMECIO.vmec_interpolate(
        vmec.variables["bsupvmnc"][:],
        np.zeros_like(vmec.variables["bsupvmnc"][:]),
        vmec.variables["xm_nyq"][:],
        vmec.variables["xn_nyq"][:],
        theta=vartheta_vmec,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=False,
    )
    bsupv_desc = VMECIO.vmec_interpolate(
        desc.variables["bsupvmnc"][:],
        np.zeros_like(desc.variables["bsupvmnc"][:]),
        desc.variables["xm_nyq"][:],
        desc.variables["xn_nyq"][:],
        theta=vartheta_desc,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=False,
    )
    np.testing.assert_allclose(bsupv_vmec, bsupv_desc, rtol=1e-3, atol=1e-3)

    # TODO: not testing B_psi because VMEC radial derivatives are inaccurate

    # B_theta
    bsubu_vmec = VMECIO.vmec_interpolate(
        vmec.variables["bsubumnc"][:],
        np.zeros_like(vmec.variables["bsubumnc"][:]),
        vmec.variables["xm_nyq"][:],
        vmec.variables["xn_nyq"][:],
        theta=vartheta_vmec,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=False,
    )
    bsubu_desc = VMECIO.vmec_interpolate(
        desc.variables["bsubumnc"][:],
        np.zeros_like(desc.variables["bsubumnc"][:]),
        desc.variables["xm_nyq"][:],
        desc.variables["xn_nyq"][:],
        theta=vartheta_desc,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=False,
    )
    # FIXME: this is a bad test because VMEC and DESC use different poloidal angles
    np.testing.assert_allclose(bsubu_vmec, bsubu_desc, rtol=5e-2, atol=6e-3)

    # B_zeta
    bsubv_vmec = VMECIO.vmec_interpolate(
        vmec.variables["bsubvmnc"][:],
        np.zeros_like(vmec.variables["bsubvmnc"][:]),
        vmec.variables["xm_nyq"][:],
        vmec.variables["xn_nyq"][:],
        theta=vartheta_vmec,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=False,
    )
    bsubv_desc = VMECIO.vmec_interpolate(
        desc.variables["bsubvmnc"][:],
        np.zeros_like(desc.variables["bsubvmnc"][:]),
        desc.variables["xm_nyq"][:],
        desc.variables["xn_nyq"][:],
        theta=vartheta_desc,
        phi=grid.nodes[:, 2],
        s=grid.nodes[:, 0],
        sym=False,
    )
    np.testing.assert_allclose(bsubv_vmec, bsubv_desc, rtol=1e-3)

    # TODO: not testing J^theta & J^zeta because VMEC radial derivatives are inaccurate

    vmec.close
    desc.close


@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_vmec_comparison(SOLOVEV):
    """Test that DESC and VMEC flux surface plots match."""

    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = VMECIO.plot_vmec_comparison(eq, str(SOLOVEV["vmec_nc_path"]))
    return fig


def test_vmec_boundary_subspace(DummyStellarator):
    """Test VMEC boundary subspace is enforced properly."""

    eq = Equilibrium.load(
        load_from=str(DummyStellarator["output_path"]), file_format="hdf5"
    )

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
