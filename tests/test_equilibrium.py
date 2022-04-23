import os
import numpy as np
from netCDF4 import Dataset
import pytest

from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.grid import Grid, LinearGrid
from desc.utils import area_difference
from desc.__main__ import main
from desc.geometry import ZernikeRZToroidalSection
from desc.basis import FourierZernike_to_PoincareZernikePolynomial
from desc.objectives import (
    PoincareBoundaryR,
    PoincareBoundaryZ,
    PoincareLambda,
    LambdaGauge,
    FixedPressure,
    FixedIota,
    FixedPsi,
    ForceBalance,
    ObjectiveFunction,
)


def test_compute_geometry(DSHAPE):
    """Test computation of plasma geometric values."""

    # VMEC values
    file = Dataset(str(DSHAPE["vmec_nc_path"]), mode="r")
    V_vmec = float(file.variables["volume_p"][-1])
    R0_vmec = float(file.variables["Rmajor_p"][-1])
    a_vmec = float(file.variables["Aminor_p"][-1])
    ar_vmec = float(file.variables["aspect"][-1])
    file.close

    # DESC values
    eq = EquilibriaFamily.load(load_from=str(DSHAPE["desc_h5_path"]))[-1]
    data = eq.compute("R0/a")
    V_desc = data["V"]
    R0_desc = data["R0"]
    a_desc = data["a"]
    ar_desc = data["R0/a"]

    assert abs(V_vmec - V_desc) < 5e-3
    assert abs(R0_vmec - R0_desc) < 5e-3
    assert abs(a_vmec - a_desc) < 5e-3
    assert abs(ar_vmec - ar_desc) < 5e-3


@pytest.mark.slow
def test_compute_theta_coords(SOLOVEV):
    """Test root finding for theta(theta*,lambda(theta))."""

    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]

    rho = np.linspace(0.01, 0.99, 200)
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi, 200, endpoint=False)

    nodes = np.vstack([rho, theta, zeta]).T
    coords = eq.compute("lambda", Grid(nodes, sort=False))
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
    """Test root finding for (rho,theta,zeta) from (R,phi,Z)."""

    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]

    rho = np.linspace(0.01, 0.99, 200)
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi, 200, endpoint=False)

    nodes = np.vstack([rho, theta, zeta]).T
    coords = eq.compute("R", Grid(nodes, sort=False))
    real_coords = np.vstack([coords["R"].flatten(), zeta, coords["Z"].flatten()]).T

    flux_coords = eq.compute_flux_coords(real_coords)
    flux_coords = np.array(flux_coords)

    # catch difference between 0 and 2*pi
    if flux_coords[0, 1] > np.pi:  # theta[0] = 0
        flux_coords[0, 1] = flux_coords[0, 1] - 2 * np.pi

    np.testing.assert_allclose(nodes, flux_coords, rtol=1e-5, atol=1e-5)


def _compute_coords(equil, check_all_zeta=False):

    if equil.N == 0 and not check_all_zeta:
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
    r_coords = equil.compute("R", r_grid)
    v_coords = equil.compute("Z", v_grid)

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
def test_to_sfl(SOLOVEV):

    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]

    Rr1, Zr1, Rv1, Zv1 = _compute_coords(eq)
    Rr2, Zr2, Rv2, Zv2 = _compute_coords(eq.to_sfl())
    rho_err, theta_err = area_difference(Rr1, Rr2, Zr1, Zr2, Rv1, Rv2, Zv1, Zv2)

    np.testing.assert_allclose(rho_err, 0, atol=2.5e-5)
    np.testing.assert_allclose(theta_err, 0, atol=1e-7)


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


@pytest.mark.slow
def test_poincare_bc(SOLOVEV, SOLOVEV_Poincare):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    eq_poin = EquilibriaFamily.load(load_from=str(SOLOVEV_Poincare["desc_h5_path"]))[-1]
    Rr1, Zr1, Rv1, Zv1 = _compute_coords(eq, check_all_zeta=True)
    Rr2, Zr2, Rv2, Zv2 = _compute_coords(eq_poin, check_all_zeta=True)
    rho_err, theta_err = area_difference(Rr1, Rr2, Zr1, Zr2, Rv1, Rv2, Zv1, Zv2)
    np.testing.assert_allclose(rho_err, 0, atol=1e-2)
    np.testing.assert_allclose(theta_err, 0, atol=1e-2)


@pytest.mark.slow
def test_poincare_sfl_bc(
    SOLOVEV,
):  # solve an equilibrium with R,Z and lambda specified on zeta=0 surface
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]

    Rb_lmn, Rb_basis = FourierZernike_to_PoincareZernikePolynomial(eq.R_lmn, eq.R_basis)
    Zb_lmn, Zb_basis = FourierZernike_to_PoincareZernikePolynomial(eq.Z_lmn, eq.Z_basis)

    surf = ZernikeRZToroidalSection(
        R_lmn=Rb_lmn,
        modes_R=Rb_basis.modes[:, :2].astype(int),
        Z_lmn=Zb_lmn,
        modes_Z=Zb_basis.modes[:, :2].astype(int),
        spectral_indexing=eq._spectral_indexing,
    )
    eq_poin = Equilibrium(
        surface=surf,
        pressure=eq.pressure,
        iota=eq.iota,
        Psi=eq.Psi,  # flux (in Webers) within the last closed flux surface
        NFP=eq.NFP,  # number of field periods
        L=eq.L,  # radial spectral resolution
        M=eq.M,  # poloidal spectral resolution
        N=eq.N,  # toroidal spectral resolution
        L_grid=eq.L_grid,  # real space radial resolution, slightly oversampled
        M_grid=eq.M_grid,  # real space poloidal resolution, slightly oversampled
        N_grid=eq.N_grid,  # real space toroidal resolution
        sym=True,  # explicitly enforce stellarator symmetry
        bdry_mode="poincare",
    )
    eq_poin.L_lmn = (
        eq.L_lmn
    )  # initialize the poincare eq with the lambda of the original eq
    eq_poin.change_resolution(
        eq_poin.L, eq_poin.M, 1
    )  # add toroidal modes to the equilibrium
    eq_poin.N_grid = 2  # set resolution of toroidal grid
    eq_poin.R_lmn[1:4] = (
        eq_poin.R_lmn[1:4] + 0.02
    )  # perturb slightly from the axisymmetric equilibrium

    constraints = (
        PoincareBoundaryR(),
        PoincareBoundaryZ(),
        PoincareLambda(),  # this constrains lambda at the zeta=0 surface, using the eq's current value of lambda
        LambdaGauge(),
        FixedPressure(),
        FixedIota(),
        FixedPsi(),
    )
    objectives = ForceBalance()
    obj = ObjectiveFunction(objectives, constraints)
    eq_poin.solve(verbose=1, ftol=1e-6, objective=obj, maxiter=100, xtol=1e-6)

    Rr1, Zr1, Rv1, Zv1 = _compute_coords(eq, check_all_zeta=True)
    Rr2, Zr2, Rv2, Zv2 = _compute_coords(eq_poin, check_all_zeta=True)
    rho_err, theta_err = area_difference(Rr1, Rr2, Zr1, Zr2, Rv1, Rv2, Zv1, Zv2)
    np.testing.assert_allclose(rho_err, 0, atol=1e-2)
    np.testing.assert_allclose(theta_err, 0, atol=1e-2)
