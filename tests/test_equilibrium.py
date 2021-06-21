import numpy as np
from netCDF4 import Dataset

from desc.equilibrium import Equilibrium, EquilibriaFamily


def test_compute_volume(DSHAPE):
    """Test plasma volume computation."""

    # VMEC value
    file = Dataset(str(DSHAPE["vmec_nc_path"]), mode="r")
    V_vmec = float(file.variables["volume_p"][-1])
    file.close

    # DESC value
    eq = EquilibriaFamily.load(
        load_from=str(DSHAPE["output_path"]), file_format="hdf5"
    )[-1]
    V_desc = eq.compute_volume()

    assert abs(V_vmec - V_desc) < 5e-3


def test_major_radius(DSHAPE):
    """Test major radius computation."""

    # VMEC value
    file = Dataset(str(DSHAPE["vmec_nc_path"]), mode="r")
    R_vmec = float(file.variables["Rmajor_p"][-1])
    file.close

    # DESC value
    eq = EquilibriaFamily.load(
        load_from=str(DSHAPE["output_path"]), file_format="hdf5"
    )[-1]
    R_desc = eq.major_radius

    assert abs(R_vmec - R_desc) < 5e-3


def test_minor_radius(DSHAPE):
    """Test minor radius computation."""

    # VMEC value
    file = Dataset(str(DSHAPE["vmec_nc_path"]), mode="r")
    A_vmec = float(file.variables["Aminor_p"][-1])
    file.close

    # DESC value
    eq = EquilibriaFamily.load(
        load_from=str(DSHAPE["output_path"]), file_format="hdf5"
    )[-1]
    A_desc = eq.minor_radius

    assert abs(A_vmec - A_desc) < 5e-3


def test_aspect_ratio(DSHAPE):
    """Test aspect ratio computation."""

    # VMEC value
    file = Dataset(str(DSHAPE["vmec_nc_path"]), mode="r")
    AR_vmec = float(file.variables["aspect_p"][-1])
    file.close

    # DESC value
    eq = EquilibriaFamily.load(
        load_from=str(DSHAPE["output_path"]), file_format="hdf5"
    )[-1]
    AR_desc = eq.aspect_ratio

    assert abs(AR_vmec - AR_desc) < 5e-3


def test_magnetic_axis_guess(DummyStellarator):
    """Test that the magnetic axis initial guess is used correctly."""

    eq = Equilibrium.load(
        load_from=str(DummyStellarator["output_path"]), file_format="hdf5"
    )
    zeta = np.linspace(0, 2 * np.pi, num=33, endpoint=False)

    # axis guess for Dummy Stellarator:
    R0 = 3.4 + 0.2 * np.cos(eq.NFP * zeta)
    Z0 = -0.2 * np.sin(eq.NFP * zeta)

    # axis location as input
    R0_eq, Z0_eq = eq.compute_axis_location(zeta)

    np.testing.assert_allclose(R0_eq, R0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(Z0_eq, Z0, rtol=0, atol=1e-6)
