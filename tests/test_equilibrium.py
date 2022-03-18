import os
import numpy as np
from netCDF4 import Dataset
import pytest
from desc.equilibrium import Equilibrium, EquilibriaFamily
from desc.grid import Grid, LinearGrid
from desc.utils import area_difference
from desc.__main__ import main


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


@pytest.mark.slow
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

    geom_coords = eq.compute_theta_coords(flux_coords)
    geom_coords = np.array(geom_coords)

    # catch difference between 0 and 2*pi
    if geom_coords[0, 1] > np.pi:  # theta[0] = 0
        geom_coords[0, 1] = geom_coords[0, 1] - 2 * np.pi

    np.testing.assert_allclose(nodes, geom_coords, rtol=1e-5, atol=1e-5)


@pytest.mark.slow
def test_compute_flux_coords(SOLOVEV):
    """Test root finding for (rho,theta,zeta) from (R,phi,Z)"""

    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]

    rho = np.linspace(0.01, 0.99, 200)
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi, 200, endpoint=False)

    nodes = np.vstack([rho, theta, zeta]).T
    coords = eq.compute_toroidal_coords(Grid(nodes, sort=False))
    real_coords = np.vstack([coords["R"].flatten(), zeta, coords["Z"].flatten()]).T

    flux_coords = eq.compute_flux_coords(real_coords)
    flux_coords = np.array(flux_coords)

    # catch difference between 0 and 2*pi
    if flux_coords[0, 1] > np.pi:  # theta[0] = 0
        flux_coords[0, 1] = flux_coords[0, 1] - 2 * np.pi

    np.testing.assert_allclose(nodes, flux_coords, rtol=1e-5, atol=1e-5)


def _compute_coords(equil):

    if equil.N == 0:
        Nz = 1
    else:
        Nz = 6

    Nr = 10
    Nt = 8
    num_theta = 1000
    num_rho = 1000

    # flux surfaces to plot
    rr = np.linspace(0, 1, Nr)
    rt = np.linspace(0, 2 * np.pi, num_theta)
    rz = np.linspace(0, 2 * np.pi / equil.NFP, Nz, endpoint=False)
    r_grid = LinearGrid(rho=rr, theta=rt, zeta=rz)

    # straight field-line angles to plot
    tr = np.linspace(0, 1, num_rho)
    tt = np.linspace(0, 2 * np.pi, Nt, endpoint=False)
    tz = np.linspace(0, 2 * np.pi / equil.NFP, Nz, endpoint=False)
    t_grid = LinearGrid(rho=tr, theta=tt, zeta=tz)

    # Note: theta* (also known as vartheta) is the poloidal straight field-line
    # angle in PEST-like flux coordinates

    # find theta angles corresponding to desired theta* angles
    v_grid = Grid(equil.compute_theta_coords(t_grid.nodes))
    r_coords = equil.compute_toroidal_coords(r_grid)
    v_coords = equil.compute_toroidal_coords(v_grid)

    # rho contours
    Rr1 = r_coords["R"].reshape((r_grid.M, r_grid.L, r_grid.N), order="F")
    Rr1 = np.swapaxes(Rr1, 0, 1)
    Zr1 = r_coords["Z"].reshape((r_grid.M, r_grid.L, r_grid.N), order="F")
    Zr1 = np.swapaxes(Zr1, 0, 1)

    # vartheta contours
    Rv1 = v_coords["R"].reshape((t_grid.M, t_grid.L, t_grid.N), order="F")
    Rv1 = np.swapaxes(Rv1, 0, 1)
    Zv1 = v_coords["Z"].reshape((t_grid.M, t_grid.L, t_grid.N), order="F")
    Zv1 = np.swapaxes(Zv1, 0, 1)
    return Rr1, Zr1, Rv1, Zv1


@pytest.mark.slow
def test_to_sfl(plot_eq):

    Rr1, Zr1, Rv1, Zv1 = _compute_coords(plot_eq)
    Rr2, Zr2, Rv2, Zv2 = _compute_coords(plot_eq.to_sfl())
    rho_err, theta_err = area_difference(Rr1, Rr2, Zr1, Zr2, Rv1, Rv2, Zv1, Zv2)

    np.testing.assert_allclose(rho_err, 0, atol=1e-5)
    np.testing.assert_allclose(theta_err, 0, atol=5e-10)


@pytest.mark.slow
def test_continuation_resolution(tmpdir_factory):
    input_path = ".//tests//inputs//res_test"
    output_dir = tmpdir_factory.mktemp("result")
    desc_h5_path = output_dir.join("res_test_out.h5")

    cwd = os.path.dirname(__file__)
    exec_dir = os.path.join(cwd, "..")
    input_filename = os.path.join(exec_dir, input_path)

    args = ["-o", str(desc_h5_path), input_filename, "-vv"]
    main(args)
