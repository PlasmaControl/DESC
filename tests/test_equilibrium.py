import numpy as np
from netCDF4 import Dataset

from desc.equilibrium import Equilibrium, EquilibriaFamily
from desc.grid import Grid


def test_compute_volume(DSHAPE):
    """Tests plasma volume computation."""

    # VMEC value
    file = Dataset(str(DSHAPE["vmec_nc_path"]), mode="r")
    V_vmec = float(file.variables["volume_p"][-1])
    file.close

    # DESC value
    eq = EquilibriaFamily.load(load_from=str(DSHAPE["desc_h5_path"]))[-1]
    V_desc = eq.compute_volume()

    assert abs(V_vmec - V_desc) < 5e-3


def test_major_radius(DSHAPE):
    """Tests major radius computation."""

    # VMEC value
    file = Dataset(str(DSHAPE["vmec_nc_path"]), mode="r")
    R_vmec = float(file.variables["Rmajor_p"][-1])
    file.close

    # DESC value
    eq = EquilibriaFamily.load(load_from=str(DSHAPE["desc_h5_path"]))[-1]
    R_desc = eq.major_radius

    assert abs(R_vmec - R_desc) < 5e-3


def test_minor_radius(DSHAPE):
    """Tests minor radius computation."""

    # VMEC value
    file = Dataset(str(DSHAPE["vmec_nc_path"]), mode="r")
    A_vmec = float(file.variables["Aminor_p"][-1])
    file.close

    # DESC value
    eq = EquilibriaFamily.load(load_from=str(DSHAPE["desc_h5_path"]))[-1]
    A_desc = eq.minor_radius

    assert abs(A_vmec - A_desc) < 5e-3


def test_aspect_ratio(DSHAPE):
    """Tests aspect ratio computation."""

    # VMEC value
    file = Dataset(str(DSHAPE["vmec_nc_path"]), mode="r")
    AR_vmec = float(file.variables["aspect"][-1])
    file.close

    # DESC value
    eq = EquilibriaFamily.load(load_from=str(DSHAPE["desc_h5_path"]))[-1]
    AR_desc = eq.aspect_ratio

    assert abs(AR_vmec - AR_desc) < 5e-3


def test_magnetic_axis_guess(DummyStellarator):
    """Tests that the magnetic axis initial guess is used correctly."""

    eq = Equilibrium.load(
        load_from=str(DummyStellarator["output_path"]), file_format="hdf5"
    )
    zeta = np.linspace(0, 2 * np.pi, num=33, endpoint=False) / eq.NFP

    # axis guess for Dummy Stellarator:
    R0 = 3.4 + 0.2 * np.cos(eq.NFP * zeta)
    Z0 = -0.2 * np.sin(eq.NFP * zeta)

    # axis location as input
    R0_eq, phi0, Z0_eq = eq.axis.compute_coordinates(grid=zeta).T

    np.testing.assert_allclose(R0_eq, R0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(Z0_eq, Z0, rtol=0, atol=1e-6)


def test_compute_theta_coords(SOLOVEV):
    """Test root finding for theta(theta*, lambda(theta))"""

    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]

    rho = np.linspace(0.01, 0.99, 200)
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi, 200, endpoint=False)

    nodes = np.vstack([rho, theta, zeta]).T
    coords = eq.compute_toroidal_coords(Grid(nodes, sort=False))
    flux_coords = nodes.copy()
    flux_coords[:, 1] += coords["lambda"]

    geom_coords = np.array(eq.compute_theta_coords(flux_coords))

    # catch difference between 0 and 2*pi
    if geom_coords[0, 1] > np.pi:  # theta[0] = 0
        geom_coords[0, 1] = geom_coords[0, 1] - 2 * np.pi

    np.testing.assert_allclose(nodes, geom_coords, rtol=1e-5, atol=1e-5)


def test_compute_flux_coords(SOLOVEV):
    """Test root finding for (rho,theta,zeta) from (R,phi,Z)"""

    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]

    rho = np.linspace(0.01, 0.99, 20)
    theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi, 20, endpoint=False)

    nodes = np.vstack([rho, theta, zeta]).T
    coords = eq.compute_toroidal_coords(Grid(nodes, sort=False))
    real_coords = np.vstack([coords["R"].flatten(), zeta, coords["Z"].flatten()]).T

    flux_coords = np.array(eq.compute_flux_coords(real_coords))

    # catch difference between 0 and 2*pi
    if flux_coords[0, 1] > np.pi:  # theta[0] = 0
        flux_coords[0, 1] = flux_coords[0, 1] - 2 * np.pi

    np.testing.assert_allclose(nodes, flux_coords, rtol=1e-5, atol=1e-5)


# can't test booz_xform because it can't be installed by Travis
"""
def test_booz_xform(DSHAPE):
    "Tests booz_xform run."

    import booz_xform as bx

    eq = EquilibriaFamily.load(load_from=str(DSHAPE["desc_h5_path"]))[-1]
    b1 = eq.run_booz_xform(filename=str(DSHAPE["booz_nc_path"]), verbose=False)
    b2 = bx.read_boozmn(str(DSHAPE["booz_nc_path"]))

    np.testing.assert_allclose(b1.bmnc_b, b2.bmnc_b, rtol=1e-6, atol=1e-6)
"""
