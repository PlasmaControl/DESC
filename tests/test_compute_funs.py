"""Tests for compute functions."""

import numpy as np
import pytest
from scipy.signal import convolve2d

from desc.compute.utils import compress
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.grid import LinearGrid

# convolve kernel is reverse of FD coeffs
FD_COEF_1_2 = np.array([-1 / 2, 0, 1 / 2])[::-1]
FD_COEF_1_4 = np.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12])[::-1]
FD_COEF_2_2 = np.array([1, -2, 1])[::-1]
FD_COEF_2_4 = np.array([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12])[::-1]


# TODO: add more tests for compute_geometry
@pytest.mark.unit
def test_total_volume(DummyStellarator):
    """Test that the volume enclosed by the LCFS is equal to the total volume."""
    eq = Equilibrium.load(
        load_from=str(DummyStellarator["output_path"]), file_format="hdf5"
    )

    grid = LinearGrid(M=12, N=12, NFP=eq.NFP, sym=eq.sym)  # rho = 1
    lcfs_volume = eq.compute("V(r)", grid)["V(r)"].mean()
    total_volume = eq.compute("V")["V"]  # default quadrature grid
    np.testing.assert_allclose(lcfs_volume, total_volume)


@pytest.mark.unit
def test_enclosed_volumes():
    """Test that the volume enclosed by flux surfaces matches analytic formulas."""
    eq = Equilibrium()  # torus
    rho = np.linspace(1 / 128, 1, 128)
    grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=rho)
    data = eq.compute("V_rr(r)", grid=grid)
    np.testing.assert_allclose(
        2 * data["R0"] * (np.pi * rho) ** 2,
        compress(grid, data["V(r)"]),
    )
    np.testing.assert_allclose(
        4 * data["R0"] * np.pi**2 * rho,
        compress(grid, data["V_r(r)"]),
    )
    np.testing.assert_allclose(
        4 * data["R0"] * np.pi**2,
        compress(grid, data["V_rr(r)"]),
    )


@pytest.mark.unit
def test_surface_areas():
    """Test that the flux surface areas match known analytic formulas."""
    eq = Equilibrium()  # torus
    rho = np.linspace(1 / 128, 1, 128)
    grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=rho)
    data = eq.compute("S(r)", grid=grid)
    S = 4 * data["R0"] * np.pi**2 * rho
    np.testing.assert_allclose(S, compress(grid, data["S(r)"]))


# TODO: remove or combine with above
@pytest.mark.unit
def test_surface_areas_2():
    """Alternate test that the flux surface areas match known analytic formulas."""
    eq = Equilibrium()

    grid_r = LinearGrid(rho=1, theta=10, zeta=10)
    grid_t = LinearGrid(rho=10, theta=1, zeta=10)
    grid_z = LinearGrid(rho=10, theta=10, zeta=1)

    data_r = eq.compute("|e_theta x e_zeta|", grid_r)
    data_t = eq.compute("|e_zeta x e_rho|", grid_t)
    data_z = eq.compute("|e_rho x e_theta|", grid_z)

    Ar = np.sum(
        data_r["|e_theta x e_zeta|"] * grid_r.spacing[:, 1] * grid_r.spacing[:, 2]
    )
    At = np.sum(
        data_t["|e_zeta x e_rho|"] * grid_t.spacing[:, 2] * grid_t.spacing[:, 0]
    )
    Az = np.sum(
        data_z["|e_rho x e_theta|"] * grid_z.spacing[:, 0] * grid_z.spacing[:, 1]
    )

    np.testing.assert_allclose(Ar, 4 * 10 * np.pi**2)
    np.testing.assert_allclose(At, np.pi * (11**2 - 10**2))
    np.testing.assert_allclose(Az, np.pi)


@pytest.mark.slow
@pytest.mark.unit
def test_magnetic_field_derivatives(DummyStellarator):
    """Test that the derivatives of B and |B| are close to numerical derivatives."""
    eq = Equilibrium.load(
        load_from=str(DummyStellarator["output_path"]), file_format="hdf5"
    )

    # partial derivatives wrt rho
    num_rho = 75
    grid = LinearGrid(rho=num_rho, NFP=eq.NFP)
    drho = grid.nodes[1, 0]
    data = eq.compute("J", grid)

    B_sup_theta_r = np.convolve(data["B^theta"], FD_COEF_1_4, "same") / drho
    B_sup_zeta_r = np.convolve(data["B^zeta"], FD_COEF_1_4, "same") / drho
    B_sub_rho_r = np.convolve(data["B_rho"], FD_COEF_1_4, "same") / drho
    B_sub_theta_r = np.convolve(data["B_theta"], FD_COEF_1_4, "same") / drho
    B_sub_zeta_r = np.convolve(data["B_zeta"], FD_COEF_1_4, "same") / drho

    np.testing.assert_allclose(
        data["B^theta_r"][3:-2],
        B_sup_theta_r[3:-2],
        rtol=1e-3,
        atol=1e-3 * np.nanmean(np.abs(data["B^theta_r"])),
    )
    np.testing.assert_allclose(
        data["B^zeta_r"][3:-2],
        B_sup_zeta_r[3:-2],
        rtol=1e-3,
        atol=1e-3 * np.nanmean(np.abs(data["B^zeta_r"])),
    )
    np.testing.assert_allclose(
        data["B_rho_r"][3:-2],
        B_sub_rho_r[3:-2],
        rtol=1e-3,
        atol=1e-3 * np.nanmean(np.abs(data["B_rho_r"])),
    )
    np.testing.assert_allclose(
        data["B_theta_r"][3:-2],
        B_sub_theta_r[3:-2],
        rtol=1e-3,
        atol=1e-3 * np.nanmean(np.abs(data["B_theta_r"])),
    )
    np.testing.assert_allclose(
        data["B_zeta_r"][3:-2],
        B_sub_zeta_r[3:-2],
        rtol=1e-3,
        atol=1e-3 * np.nanmean(np.abs(data["B_zeta_r"])),
    )

    # partial derivatives wrt theta
    num_theta = 120
    grid = LinearGrid(NFP=eq.NFP, theta=num_theta)
    dtheta = grid.nodes[1, 1]
    data = eq.compute("J", grid)
    data = eq.compute("|B|_tt", grid, data=data)

    B_sup_theta_t = np.convolve(data["B^theta"], FD_COEF_1_4, "same") / dtheta
    B_sup_theta_tt = np.convolve(data["B^theta"], FD_COEF_2_4, "same") / dtheta**2
    B_sup_zeta_t = np.convolve(data["B^zeta"], FD_COEF_1_4, "same") / dtheta
    B_sup_zeta_tt = np.convolve(data["B^zeta"], FD_COEF_2_4, "same") / dtheta**2
    B_sub_rho_t = np.convolve(data["B_rho"], FD_COEF_1_4, "same") / dtheta
    B_sub_zeta_t = np.convolve(data["B_zeta"], FD_COEF_1_4, "same") / dtheta
    B_t = np.convolve(data["|B|"], FD_COEF_1_4, "same") / dtheta
    B_tt = np.convolve(data["|B|"], FD_COEF_2_4, "same") / dtheta**2

    np.testing.assert_allclose(
        data["B^theta_t"][2:-2],
        B_sup_theta_t[2:-2],
        rtol=2e-3,
        atol=3e-3 * np.mean(np.abs(data["B^theta_t"])),
    )
    np.testing.assert_allclose(
        data["B^theta_tt"][2:-2],
        B_sup_theta_tt[2:-2],
        rtol=2e-4,
        atol=2e-2 * np.mean(np.abs(data["B^theta_tt"])),
    )
    np.testing.assert_allclose(
        data["B^zeta_t"][2:-2],
        B_sup_zeta_t[2:-2],
        rtol=3e-3,
        atol=3e-3 * np.mean(np.abs(data["B^zeta_t"])),
    )
    np.testing.assert_allclose(
        data["B^zeta_tt"][2:-2],
        B_sup_zeta_tt[2:-2],
        rtol=6e-3,
        atol=6e-3 * np.mean(np.abs(data["B^zeta_tt"])),
    )
    np.testing.assert_allclose(
        data["B_rho_t"][2:-2],
        B_sub_rho_t[2:-2],
        rtol=1e-2,
        atol=1e-3 * np.mean(np.abs(data["B_rho_t"])),
    )
    np.testing.assert_allclose(
        data["B_zeta_t"][2:-2],
        B_sub_zeta_t[2:-2],
        rtol=2e-2,
        atol=2e-2 * np.mean(np.abs(data["B_zeta_t"])),
    )
    np.testing.assert_allclose(
        data["|B|_t"][2:-2],
        B_t[2:-2],
        rtol=1e-2,
        atol=1e-3 * np.mean(np.abs(data["|B|_t"])),
    )
    np.testing.assert_allclose(
        data["|B|_tt"][2:-2],
        B_tt[2:-2],
        rtol=2e-2,
        atol=2e-2 * np.mean(np.abs(data["|B|_tt"])),
    )

    # partial derivatives wrt zeta
    num_zeta = 120
    grid = LinearGrid(NFP=eq.NFP, zeta=num_zeta)
    dzeta = grid.nodes[1, 2]
    data = eq.compute("J", grid)
    data = eq.compute("|B|_zz", grid, data=data)

    B_sup_theta_z = np.convolve(data["B^theta"], FD_COEF_1_4, "same") / dzeta
    B_sup_theta_zz = np.convolve(data["B^theta"], FD_COEF_2_4, "same") / dzeta**2
    B_sup_zeta_z = np.convolve(data["B^zeta"], FD_COEF_1_4, "same") / dzeta
    B_sup_zeta_zz = np.convolve(data["B^zeta"], FD_COEF_2_4, "same") / dzeta**2
    B_sub_rho_z = np.convolve(data["B_rho"], FD_COEF_1_4, "same") / dzeta
    B_sub_theta_z = np.convolve(data["B_theta"], FD_COEF_1_4, "same") / dzeta
    B_z = np.convolve(data["|B|"], FD_COEF_1_4, "same") / dzeta
    B_zz = np.convolve(data["|B|"], FD_COEF_2_4, "same") / dzeta**2

    np.testing.assert_allclose(
        data["B^theta_z"][2:-2],
        B_sup_theta_z[2:-2],
        rtol=1e-3,
        atol=1e-3 * np.mean(np.abs(data["B^theta_z"])),
    )
    np.testing.assert_allclose(
        data["B^theta_zz"][2:-2],
        B_sup_theta_zz[2:-2],
        rtol=1e-4,
        atol=1e-4 * np.mean(np.abs(data["B^theta_zz"])),
    )
    np.testing.assert_allclose(
        data["B^zeta_z"][2:-2],
        B_sup_zeta_z[2:-2],
        rtol=1e-3,
        atol=1e-3 * np.mean(np.abs(data["B^zeta_z"])),
    )
    np.testing.assert_allclose(
        data["B^zeta_zz"][2:-2],
        B_sup_zeta_zz[2:-2],
        rtol=1e-4,
        atol=1e-4 * np.mean(np.abs(data["B^zeta_zz"])),
    )
    np.testing.assert_allclose(
        data["B_rho_z"][2:-2],
        B_sub_rho_z[2:-2],
        rtol=1e-3,
        atol=1e-3 * np.mean(np.abs(data["B_rho_z"])),
    )
    np.testing.assert_allclose(
        data["B_theta_z"][2:-2],
        B_sub_theta_z[2:-2],
        rtol=1e-3,
        atol=1e-3 * np.mean(np.abs(data["B_theta_z"])),
    )
    np.testing.assert_allclose(
        data["|B|_z"][2:-2],
        B_z[2:-2],
        rtol=1e-3,
        atol=1e-3 * np.mean(np.abs(data["|B|_z"])),
    )
    np.testing.assert_allclose(
        data["|B|_zz"][2:-2],
        B_zz[2:-2],
        rtol=1e-3,
        atol=1e-3 * np.mean(np.abs(data["|B|_zz"])),
    )

    # mixed derivatives wrt theta & zeta
    num_theta = 180
    num_zeta = 180
    grid = LinearGrid(NFP=eq.NFP, theta=num_theta, zeta=num_zeta)
    dtheta = grid.nodes[:, 1].reshape((num_zeta, num_theta))[0, 1]
    dzeta = grid.nodes[:, 2].reshape((num_zeta, num_theta))[1, 0]
    data = eq.compute("|B|_tz", grid)

    B_sup_theta = data["B^theta"].reshape((num_zeta, num_theta))
    B_sup_zeta = data["B^zeta"].reshape((num_zeta, num_theta))
    B = data["|B|"].reshape((num_zeta, num_theta))

    B_sup_theta_tz = convolve2d(
        B_sup_theta,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="wrap",
    ) / (dtheta * dzeta)
    B_sup_zeta_tz = convolve2d(
        B_sup_zeta,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="wrap",
    ) / (dtheta * dzeta)
    B_tz = convolve2d(
        B,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="wrap",
    ) / (dtheta * dzeta)

    np.testing.assert_allclose(
        data["B^theta_tz"].reshape((num_zeta, num_theta))[2:-2, 2:-2],
        B_sup_theta_tz[2:-2, 2:-2],
        rtol=2e-2,
        atol=1e-2 * np.mean(np.abs(data["B^theta_tz"])),
    )
    np.testing.assert_allclose(
        data["B^zeta_tz"].reshape((num_zeta, num_theta))[2:-2, 2:-2],
        B_sup_zeta_tz[2:-2, 2:-2],
        rtol=2e-2,
        atol=1e-2 * np.mean(np.abs(data["B^zeta_tz"])),
    )
    np.testing.assert_allclose(
        data["|B|_tz"].reshape((num_zeta, num_theta))[2:-2, 2:-2],
        B_tz[2:-2, 2:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(data["|B|_tz"])),
    )


@pytest.mark.slow
@pytest.mark.unit
def test_magnetic_pressure_gradient(DummyStellarator):
    """Test that the components of grad(|B|^2)) match with numerical gradients."""
    eq = Equilibrium.load(
        load_from=str(DummyStellarator["output_path"]), file_format="hdf5"
    )

    # partial derivatives wrt rho
    num_rho = 110
    grid = LinearGrid(NFP=eq.NFP, rho=num_rho)
    drho = grid.nodes[1, 0]
    data = eq.compute("|B|", grid)
    data = eq.compute("grad(|B|^2)_rho", grid, data=data)
    B2_r = np.convolve(data["|B|"] ** 2, FD_COEF_1_4, "same") / drho
    np.testing.assert_allclose(
        data["grad(|B|^2)_rho"][3:-2],
        B2_r[3:-2],
        rtol=1e-3,
        atol=1e-3 * np.nanmean(np.abs(data["grad(|B|^2)_rho"])),
    )

    # partial derivative wrt theta
    num_theta = 90
    grid = LinearGrid(NFP=eq.NFP, theta=num_theta)
    dtheta = grid.nodes[1, 1]
    data = eq.compute("|B|", grid)
    data = eq.compute("grad(|B|^2)_theta", grid, data=data)
    B2_t = np.convolve(data["|B|"] ** 2, FD_COEF_1_4, "same") / dtheta
    np.testing.assert_allclose(
        data["grad(|B|^2)_theta"][2:-2],
        B2_t[2:-2],
        rtol=1e-2,
        atol=1e-2 * np.nanmean(np.abs(data["grad(|B|^2)_theta"])),
    )

    # partial derivative wrt zeta
    num_zeta = 90
    grid = LinearGrid(NFP=eq.NFP, zeta=num_zeta)
    dzeta = grid.nodes[1, 2]
    data = eq.compute("|B|", grid)
    data = eq.compute("grad(|B|^2)_zeta", grid, data=data)
    B2_z = np.convolve(data["|B|"] ** 2, FD_COEF_1_4, "same") / dzeta
    np.testing.assert_allclose(
        data["grad(|B|^2)_zeta"][2:-2],
        B2_z[2:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(data["grad(|B|^2)_zeta"])),
    )


@pytest.mark.unit
@pytest.mark.solve
def test_currents(DSHAPE_current):
    """Test that different methods for computing I and G agree."""
    eq = EquilibriaFamily.load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]

    grid_full = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    grid_sym = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=True)

    data_booz = eq.compute("|B|_mn", grid_full, M_booz=eq.M, N_booz=eq.N)
    data_full = eq.compute("I", grid_full)
    data_sym = eq.compute("I", grid_sym)

    np.testing.assert_allclose(data_full["I"].mean(), data_booz["I"], atol=1e-16)
    np.testing.assert_allclose(data_sym["I"].mean(), data_booz["I"], atol=1e-16)
    np.testing.assert_allclose(data_full["G"].mean(), data_booz["G"], atol=1e-16)
    np.testing.assert_allclose(data_sym["G"].mean(), data_booz["G"], atol=1e-16)


@pytest.mark.slow
@pytest.mark.unit
def test_BdotgradB(DummyStellarator):
    """Test that the components of grad(B*grad(|B|)) match with numerical gradients."""
    eq = Equilibrium.load(
        load_from=str(DummyStellarator["output_path"]), file_format="hdf5"
    )

    # partial derivative wrt theta
    num_theta = 120
    grid = LinearGrid(NFP=eq.NFP, theta=num_theta)
    dtheta = grid.nodes[1, 1]
    data = eq.compute("(B*grad(|B|))_t", grid)
    Btilde_t = np.convolve(data["B*grad(|B|)"], FD_COEF_1_4, "same") / dtheta
    np.testing.assert_allclose(
        data["(B*grad(|B|))_t"][2:-2],
        Btilde_t[2:-2],
        rtol=2e-2,
        atol=2e-2 * np.mean(np.abs(data["(B*grad(|B|))_t"])),
    )

    # partial derivative wrt zeta
    num_zeta = 120
    grid = LinearGrid(NFP=eq.NFP, zeta=num_zeta)
    dzeta = grid.nodes[1, 2]
    data = eq.compute("(B*grad(|B|))_z", grid)
    Btilde_z = np.convolve(data["B*grad(|B|)"], FD_COEF_1_4, "same") / dzeta
    np.testing.assert_allclose(
        data["(B*grad(|B|))_z"][2:-2],
        Btilde_z[2:-2],
        rtol=2e-2,
        atol=2e-2 * np.mean(np.abs(data["(B*grad(|B|))_z"])),
    )


# TODO: add test with stellarator example
@pytest.mark.unit
@pytest.mark.solve
def test_boozer_transform(DSHAPE_current):
    """Test that Boozer coordinate transform agrees with BOOZ_XFORM."""
    eq = EquilibriaFamily.load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    data = eq.compute("|B|_mn", grid, M_booz=eq.M, N_booz=eq.N)
    booz_xform = np.array(
        [
            2.49792355e-01,
            5.16668333e-02,
            1.11374584e-02,
            7.31614588e-03,
            3.36187451e-03,
            2.08897051e-03,
            1.20694516e-03,
            7.84513291e-04,
            5.19293744e-04,
            3.61983430e-04,
            2.57745929e-04,
            1.86013067e-04,
            1.34610049e-04,
            9.68119345e-05,
        ]
    )
    np.testing.assert_allclose(
        np.flipud(np.sort(np.abs(data["|B|_mn"]))),
        booz_xform,
        rtol=1e-3,
        atol=1e-4,
    )


@pytest.mark.unit
def test_compute_grad_p_volume_avg():
    """Test calculation of volume averaged pressure gradient."""
    eq = Equilibrium()  # default pressure profile is 0 pressure
    pres_grad_vol_avg = eq.compute("<|grad(p)|>_vol")["<|grad(p)|>_vol"]
    np.testing.assert_allclose(pres_grad_vol_avg, 0)
