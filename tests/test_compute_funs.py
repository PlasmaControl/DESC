"""Tests for compute functions."""

import numpy as np
import pytest
from scipy.io import netcdf_file
from scipy.signal import convolve2d

from desc.compute import data_index
from desc.compute.utils import compress
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.grid import LinearGrid, QuadratureGrid

# convolve kernel is reverse of FD coeffs
FD_COEF_1_2 = np.array([-1 / 2, 0, 1 / 2])[::-1]
FD_COEF_1_4 = np.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12])[::-1]
FD_COEF_2_2 = np.array([1, -2, 1])[::-1]
FD_COEF_2_4 = np.array([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12])[::-1]


def my_convolve(arr, stencil):
    """Wrapper to convolve 1D arrs."""
    return np.convolve(arr, stencil, "same")


def myconvolve_2d(arr_1d, stencil, shape):
    """Wrapper to convolve 2D arrs."""
    arr = arr_1d.reshape((shape[0], shape[1]))
    conv = convolve2d(
        arr,
        stencil[:, np.newaxis] * stencil[np.newaxis, :],
        mode="same",
        boundary="wrap",
    )
    return conv


# TODO: add more tests for compute_geometry
@pytest.mark.unit
def test_total_volume(DummyStellarator):
    """Test that the volume enclosed by the LCFS is equal to the total volume."""
    eq = Equilibrium.load(
        load_from=str(DummyStellarator["output_path"]), file_format="hdf5"
    )

    grid = LinearGrid(M=12, N=12, NFP=eq.NFP, sym=eq.sym)  # rho = 1
    lcfs_volume = eq.compute("V(r)", grid=grid)["V(r)"].mean()
    total_volume = eq.compute("V")["V"]  # default quadrature grid
    np.testing.assert_allclose(lcfs_volume, total_volume)


@pytest.mark.unit
def test_enclosed_volumes():
    """Test that the volume enclosed by flux surfaces matches analytic formulas."""
    eq = Equilibrium()  # torus
    rho = np.linspace(1 / 128, 1, 128)
    grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=rho)
    data = eq.compute(["V_rr(r)", "R0", "V(r)", "V_r(r)"], grid=grid)
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
    data = eq.compute(["S(r)", "R0"], grid=grid)
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

    data_r = eq.compute("|e_theta x e_zeta|", grid=grid_r)
    data_t = eq.compute("|e_zeta x e_rho|", grid=grid_t)
    data_z = eq.compute("|e_rho x e_theta|", grid=grid_z)

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
    rtol = 1e-3
    atol = 1e-3
    num_rho = 100
    grid = LinearGrid(rho=num_rho, NFP=eq.NFP)
    drho = grid.nodes[1, 0]
    data = eq.compute(
        [
            "B^theta",
            "B^theta_r",
            "B^theta_rr",
            "B^zeta",
            "B^zeta_r",
            "B^zeta_rr",
            "B_rho",
            "B_rho_r",
            "B_rho_rr",
            "B_theta",
            "B_theta_r",
            "B_theta_rr",
            "B_zeta",
            "B_zeta_r",
            "B_zeta_rr",
            "|B|",
            "|B|_r",
            "|B|_rr",
            "B",
            "B_r",
            "B_rr",
        ],
        grid=grid,
    )

    B_sup_theta_r = np.convolve(data["B^theta"], FD_COEF_1_4, "same") / drho
    B_sup_theta_rr = np.convolve(data["B^theta"], FD_COEF_2_4, "same") / drho**2
    B_sup_zeta_r = np.convolve(data["B^zeta"], FD_COEF_1_4, "same") / drho
    B_sup_zeta_rr = np.convolve(data["B^zeta"], FD_COEF_2_4, "same") / drho**2
    B_sub_rho_r = np.convolve(data["B_rho"], FD_COEF_1_4, "same") / drho
    B_sub_rho_rr = np.convolve(data["B_rho"], FD_COEF_2_4, "same") / drho**2
    B_sub_theta_r = np.convolve(data["B_theta"], FD_COEF_1_4, "same") / drho
    B_sub_theta_rr = np.convolve(data["B_theta"], FD_COEF_2_4, "same") / drho**2
    B_sub_zeta_r = np.convolve(data["B_zeta"], FD_COEF_1_4, "same") / drho
    B_sub_zeta_rr = np.convolve(data["B_zeta"], FD_COEF_2_4, "same") / drho**2
    Bmag_r = np.convolve(data["|B|"], FD_COEF_1_4, "same") / drho
    Bmag_rr = np.convolve(data["|B|"], FD_COEF_2_4, "same") / drho**2
    B_r = np.apply_along_axis(my_convolve, 0, data["B"], FD_COEF_1_4) / drho
    B_rr = np.apply_along_axis(my_convolve, 0, data["B"], FD_COEF_2_4) / drho**2

    np.testing.assert_allclose(
        data["B^theta_r"][3:-2],
        B_sup_theta_r[3:-2],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B^theta_r"])),
    )
    np.testing.assert_allclose(
        data["B^theta_rr"][3:-2],
        B_sup_theta_rr[3:-2],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B^theta_rr"])),
    )
    np.testing.assert_allclose(
        data["B^zeta_r"][3:-2],
        B_sup_zeta_r[3:-2],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B^zeta_r"])),
    )
    np.testing.assert_allclose(
        data["B^zeta_rr"][3:-2],
        B_sup_zeta_rr[3:-2],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B^zeta_rr"])),
    )
    np.testing.assert_allclose(
        data["B_rho_r"][3:-2],
        B_sub_rho_r[3:-2],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_rho_r"])),
    )
    np.testing.assert_allclose(
        data["B_rho_rr"][3:-2],
        B_sub_rho_rr[3:-2],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_rho_rr"])),
    )
    np.testing.assert_allclose(
        data["B_theta_r"][3:-2],
        B_sub_theta_r[3:-2],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_theta_r"])),
    )
    np.testing.assert_allclose(
        data["B_theta_rr"][3:-2],
        B_sub_theta_rr[3:-2],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_theta_rr"])),
    )
    np.testing.assert_allclose(
        data["B_zeta_r"][3:-2],
        B_sub_zeta_r[3:-2],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_zeta_r"])),
    )
    np.testing.assert_allclose(
        data["B_zeta_rr"][3:-2],
        B_sub_zeta_rr[3:-2],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_zeta_rr"])),
    )
    np.testing.assert_allclose(
        data["|B|_r"][3:-2],
        Bmag_r[3:-2],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["|B|_r"])),
    )
    np.testing.assert_allclose(
        data["|B|_rr"][3:-2],
        Bmag_rr[3:-2],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["|B|_rr"])),
    )
    np.testing.assert_allclose(
        data["B_r"][3:-2, :],
        B_r[3:-2, :],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_r"])),
    )
    np.testing.assert_allclose(
        data["B_rr"][3:-2, :],
        B_rr[3:-2, :],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_rr"])),
    )

    # partial derivatives wrt theta
    rtol = 3e-3
    atol = 3e-3
    num_theta = 150
    grid = LinearGrid(NFP=eq.NFP, theta=num_theta)
    dtheta = grid.nodes[1, 1]
    data = eq.compute(
        [
            "B^theta",
            "B^theta_t",
            "B^theta_tt",
            "B^zeta",
            "B^zeta_t",
            "B^zeta_tt",
            "B_rho",
            "B_rho_t",
            "B_rho_tt",
            "B_theta",
            "B_theta_t",
            "B_theta_tt",
            "B_zeta",
            "B_zeta_t",
            "B_zeta_tt",
            "|B|",
            "|B|_t",
            "|B|_tt",
            "B",
            "B_t",
            "B_tt",
        ],
        grid=grid,
    )

    B_sup_theta_t = np.convolve(data["B^theta"], FD_COEF_1_4, "same") / dtheta
    B_sup_theta_tt = np.convolve(data["B^theta"], FD_COEF_2_4, "same") / dtheta**2
    B_sup_zeta_t = np.convolve(data["B^zeta"], FD_COEF_1_4, "same") / dtheta
    B_sup_zeta_tt = np.convolve(data["B^zeta"], FD_COEF_2_4, "same") / dtheta**2
    B_sub_rho_t = np.convolve(data["B_rho"], FD_COEF_1_4, "same") / dtheta
    B_sub_rho_tt = np.convolve(data["B_rho"], FD_COEF_2_4, "same") / dtheta**2
    B_sub_theta_t = np.convolve(data["B_theta"], FD_COEF_1_4, "same") / dtheta
    B_sub_theta_tt = np.convolve(data["B_theta"], FD_COEF_2_4, "same") / dtheta**2
    B_sub_zeta_t = np.convolve(data["B_zeta"], FD_COEF_1_4, "same") / dtheta
    B_sub_zeta_tt = np.convolve(data["B_zeta"], FD_COEF_2_4, "same") / dtheta**2
    Bmag_t = np.convolve(data["|B|"], FD_COEF_1_4, "same") / dtheta
    Bmag_tt = np.convolve(data["|B|"], FD_COEF_2_4, "same") / dtheta**2
    B_t = np.apply_along_axis(my_convolve, 0, data["B"], FD_COEF_1_4) / dtheta
    B_tt = np.apply_along_axis(my_convolve, 0, data["B"], FD_COEF_2_4) / dtheta**2

    np.testing.assert_allclose(
        data["B^theta_t"][2:-2],
        B_sup_theta_t[2:-2],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B^theta_t"])),
    )
    np.testing.assert_allclose(
        data["B^theta_tt"][2:-2],
        B_sup_theta_tt[2:-2],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B^theta_tt"])),
    )
    np.testing.assert_allclose(
        data["B^zeta_t"][2:-2],
        B_sup_zeta_t[2:-2],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B^zeta_t"])),
    )
    np.testing.assert_allclose(
        data["B^zeta_tt"][2:-2],
        B_sup_zeta_tt[2:-2],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B^zeta_tt"])),
    )
    np.testing.assert_allclose(
        data["B_rho_t"][2:-2],
        B_sub_rho_t[2:-2],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B_rho_t"])),
    )
    np.testing.assert_allclose(
        data["B_rho_tt"][2:-2],
        B_sub_rho_tt[2:-2],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B_rho_tt"])),
    )
    np.testing.assert_allclose(
        data["B_theta_t"][2:-2],
        B_sub_theta_t[2:-2],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B_theta_t"])),
    )
    np.testing.assert_allclose(
        data["B_theta_tt"][2:-2],
        B_sub_theta_tt[2:-2],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B_theta_tt"])),
    )
    np.testing.assert_allclose(
        data["B_zeta_t"][2:-2],
        B_sub_zeta_t[2:-2],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B_zeta_t"])),
    )
    np.testing.assert_allclose(
        data["B_zeta_tt"][2:-2],
        B_sub_zeta_tt[2:-2],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B_zeta_tt"])),
    )
    np.testing.assert_allclose(
        data["|B|_t"][2:-2],
        Bmag_t[2:-2],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["|B|_t"])),
    )
    np.testing.assert_allclose(
        data["|B|_tt"][2:-2],
        Bmag_tt[2:-2],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["|B|_tt"])),
    )
    np.testing.assert_allclose(
        data["B_t"][2:-2, :],
        B_t[2:-2, :],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_t"])),
    )
    np.testing.assert_allclose(
        data["B_tt"][2:-2, :],
        B_tt[2:-2, :],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_tt"])),
    )

    # partial derivatives wrt zeta
    rtol = 1e-4
    atol = 1e-4
    num_zeta = 130
    grid = LinearGrid(NFP=eq.NFP, zeta=num_zeta)
    dzeta = grid.nodes[1, 2]
    data = eq.compute(
        [
            "B^theta",
            "B^theta_z",
            "B^theta_zz",
            "B^zeta",
            "B^zeta_z",
            "B^zeta_zz",
            "B_rho",
            "B_rho_z",
            "B_rho_zz",
            "B_theta",
            "B_theta_z",
            "B_theta_zz",
            "B_zeta",
            "B_zeta_z",
            "B_zeta_zz",
            "|B|",
            "|B|_z",
            "|B|_zz",
            "B",
            "B_z",
            "B_zz",
        ],
        grid=grid,
    )

    B_sup_theta_z = np.convolve(data["B^theta"], FD_COEF_1_4, "same") / dzeta
    B_sup_theta_zz = np.convolve(data["B^theta"], FD_COEF_2_4, "same") / dzeta**2
    B_sup_zeta_z = np.convolve(data["B^zeta"], FD_COEF_1_4, "same") / dzeta
    B_sup_zeta_zz = np.convolve(data["B^zeta"], FD_COEF_2_4, "same") / dzeta**2
    B_sub_rho_z = np.convolve(data["B_rho"], FD_COEF_1_4, "same") / dzeta
    B_sub_rho_zz = np.convolve(data["B_rho"], FD_COEF_2_4, "same") / dzeta**2
    B_sub_theta_z = np.convolve(data["B_theta"], FD_COEF_1_4, "same") / dzeta
    B_sub_theta_zz = np.convolve(data["B_theta"], FD_COEF_2_4, "same") / dzeta**2
    B_sub_zeta_z = np.convolve(data["B_zeta"], FD_COEF_1_4, "same") / dzeta
    B_sub_zeta_zz = np.convolve(data["B_zeta"], FD_COEF_2_4, "same") / dzeta**2
    Bmag_z = np.convolve(data["|B|"], FD_COEF_1_4, "same") / dzeta
    Bmag_zz = np.convolve(data["|B|"], FD_COEF_2_4, "same") / dzeta**2
    B_z = np.apply_along_axis(my_convolve, 0, data["B"], FD_COEF_1_4) / dzeta
    B_zz = np.apply_along_axis(my_convolve, 0, data["B"], FD_COEF_2_4) / dzeta**2

    np.testing.assert_allclose(
        data["B^theta_z"][2:-2],
        B_sup_theta_z[2:-2],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B^theta_z"])),
    )
    np.testing.assert_allclose(
        data["B^theta_zz"][2:-2],
        B_sup_theta_zz[2:-2],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B^theta_zz"])),
    )
    np.testing.assert_allclose(
        data["B^zeta_z"][2:-2],
        B_sup_zeta_z[2:-2],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B^zeta_z"])),
    )
    np.testing.assert_allclose(
        data["B^zeta_zz"][2:-2],
        B_sup_zeta_zz[2:-2],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B^zeta_zz"])),
    )
    np.testing.assert_allclose(
        data["B_rho_z"][2:-2],
        B_sub_rho_z[2:-2],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B_rho_z"])),
    )
    np.testing.assert_allclose(
        data["B_rho_zz"][2:-2],
        B_sub_rho_zz[2:-2],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B_rho_zz"])),
    )
    np.testing.assert_allclose(
        data["B_theta_z"][2:-2],
        B_sub_theta_z[2:-2],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B_theta_z"])),
    )
    np.testing.assert_allclose(
        data["B_theta_zz"][2:-2],
        B_sub_theta_zz[2:-2],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B_theta_zz"])),
    )
    np.testing.assert_allclose(
        data["B_zeta_z"][2:-2],
        B_sub_zeta_z[2:-2],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B_zeta_z"])),
    )
    np.testing.assert_allclose(
        data["B_zeta_zz"][2:-2],
        B_sub_zeta_zz[2:-2],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B_zeta_zz"])),
    )
    np.testing.assert_allclose(
        data["|B|_z"][2:-2],
        Bmag_z[2:-2],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["|B|_z"])),
    )
    np.testing.assert_allclose(
        data["|B|_zz"][2:-2],
        Bmag_zz[2:-2],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["|B|_zz"])),
    )
    np.testing.assert_allclose(
        data["B_z"][2:-2, :],
        B_z[2:-2, :],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_z"])),
    )
    np.testing.assert_allclose(
        data["B_zz"][2:-2, :],
        B_zz[2:-2, :],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_zz"])),
    )

    # mixed derivatives wrt rho & theta
    rtol = 1e-2
    atol = 1e-2
    num_rho = 150
    num_theta = 180
    grid = LinearGrid(NFP=eq.NFP, rho=num_rho, theta=num_theta)
    drho = grid.nodes[:, 0].reshape((num_rho, num_theta))[1, 0]
    dtheta = grid.nodes[:, 1].reshape((num_rho, num_theta))[0, 1]
    data = eq.compute(
        [
            "B^theta",
            "B^theta_rt",
            "B^zeta",
            "B^zeta_rt",
            "B_rho",
            "B_rho_rt",
            "B_theta",
            "B_theta_rt",
            "B_zeta",
            "B_zeta_rt",
            "|B|",
            "|B|_rt",
            "B",
            "B_rt",
        ],
        grid=grid,
    )

    B_sup_theta = data["B^theta"].reshape((num_rho, num_theta))
    B_sup_zeta = data["B^zeta"].reshape((num_rho, num_theta))
    B_sub_rho = data["B_rho"].reshape((num_rho, num_theta))
    B_sub_theta = data["B_theta"].reshape((num_rho, num_theta))
    B_sub_zeta = data["B_zeta"].reshape((num_rho, num_theta))
    Bmag = data["|B|"].reshape((num_rho, num_theta))

    B_sup_theta_rt = convolve2d(
        B_sup_theta,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="wrap",
    ) / (drho * dtheta)
    B_sup_zeta_rt = convolve2d(
        B_sup_zeta,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="wrap",
    ) / (drho * dtheta)
    B_sub_rho_rt = convolve2d(
        B_sub_rho,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="wrap",
    ) / (drho * dtheta)
    B_sub_theta_rt = convolve2d(
        B_sub_theta,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="wrap",
    ) / (drho * dtheta)
    B_sub_zeta_rt = convolve2d(
        B_sub_zeta,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="wrap",
    ) / (drho * dtheta)
    Bmag_rt = convolve2d(
        Bmag,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="wrap",
    ) / (drho * dtheta)
    B_rt = np.apply_along_axis(
        myconvolve_2d, 0, data["B"], FD_COEF_1_4, (num_rho, num_theta)
    ) / (drho * dtheta)

    np.testing.assert_allclose(
        data["B^theta_rt"].reshape((num_rho, num_theta))[3:-2, 2:-2],
        B_sup_theta_rt[3:-2, 2:-2],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B^theta_rt"])),
    )
    np.testing.assert_allclose(
        data["B^zeta_rt"].reshape((num_rho, num_theta))[3:-2, 2:-2],
        B_sup_zeta_rt[3:-2, 2:-2],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B^zeta_rt"])),
    )
    np.testing.assert_allclose(
        data["B_rho_rt"].reshape((num_rho, num_theta))[3:-2, 2:-2],
        B_sub_rho_rt[3:-2, 2:-2],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_rho_rt"])),
    )
    np.testing.assert_allclose(
        data["B_theta_rt"].reshape((num_rho, num_theta))[3:-2, 2:-2],
        B_sub_theta_rt[3:-2, 2:-2],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_theta_rt"])),
    )
    np.testing.assert_allclose(
        data["B_zeta_rt"].reshape((num_rho, num_theta))[3:-2, 2:-2],
        B_sub_zeta_rt[3:-2, 2:-2],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_zeta_rt"])),
    )
    np.testing.assert_allclose(
        data["|B|_rt"].reshape((num_rho, num_theta))[3:-2, 2:-2],
        Bmag_rt[3:-2, 2:-2],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["|B|_rt"])),
    )
    np.testing.assert_allclose(
        data["B_rt"].reshape((num_rho, num_theta, 3))[3:-2, 2:-2, :],
        B_rt[3:-2, 2:-2, :],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_rt"])),
    )

    # mixed derivatives wrt theta & zeta
    rtol = 1e-2
    atol = 1e-2
    num_theta = 180
    num_zeta = 180
    grid = LinearGrid(NFP=eq.NFP, theta=num_theta, zeta=num_zeta)
    dtheta = grid.nodes[:, 1].reshape((num_zeta, num_theta))[0, 1]
    dzeta = grid.nodes[:, 2].reshape((num_zeta, num_theta))[1, 0]
    data = eq.compute(
        [
            "B^theta",
            "B^theta_tz",
            "B^zeta",
            "B^zeta_tz",
            "B_rho",
            "B_rho_tz",
            "B_theta",
            "B_theta_tz",
            "B_zeta",
            "B_zeta_tz",
            "|B|",
            "|B|_tz",
            "B",
            "B_tz",
        ],
        grid=grid,
    )

    B_sup_theta = data["B^theta"].reshape((num_zeta, num_theta))
    B_sup_zeta = data["B^zeta"].reshape((num_zeta, num_theta))
    B_sub_rho = data["B_rho"].reshape((num_zeta, num_theta))
    B_sub_theta = data["B_theta"].reshape((num_zeta, num_theta))
    B_sub_zeta = data["B_zeta"].reshape((num_zeta, num_theta))
    Bmag = data["|B|"].reshape((num_zeta, num_theta))

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
    B_sub_rho_tz = convolve2d(
        B_sub_rho,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="wrap",
    ) / (dtheta * dzeta)
    B_sub_theta_tz = convolve2d(
        B_sub_theta,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="wrap",
    ) / (dtheta * dzeta)
    B_sub_zeta_tz = convolve2d(
        B_sub_zeta,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="wrap",
    ) / (dtheta * dzeta)
    Bmag_tz = convolve2d(
        Bmag,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="wrap",
    ) / (dtheta * dzeta)
    B_tz = np.apply_along_axis(
        myconvolve_2d, 0, data["B"], FD_COEF_1_4, (num_zeta, num_theta)
    ) / (dzeta * dtheta)

    np.testing.assert_allclose(
        data["B^theta_tz"].reshape((num_zeta, num_theta))[2:-2, 2:-2],
        B_sup_theta_tz[2:-2, 2:-2],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B^theta_tz"])),
    )
    np.testing.assert_allclose(
        data["B^zeta_tz"].reshape((num_zeta, num_theta))[2:-2, 2:-2],
        B_sup_zeta_tz[2:-2, 2:-2],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B^zeta_tz"])),
    )
    np.testing.assert_allclose(
        data["B_rho_tz"].reshape((num_zeta, num_theta))[2:-2, 2:-2],
        B_sub_rho_tz[2:-2, 2:-2],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B_rho_tz"])),
    )
    np.testing.assert_allclose(
        data["B_theta_tz"].reshape((num_zeta, num_theta))[2:-2, 2:-2],
        B_sub_theta_tz[2:-2, 2:-2],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B_theta_tz"])),
    )
    np.testing.assert_allclose(
        data["B_zeta_tz"].reshape((num_zeta, num_theta))[2:-2, 2:-2],
        B_sub_zeta_tz[2:-2, 2:-2],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B_zeta_tz"])),
    )
    np.testing.assert_allclose(
        data["|B|_tz"].reshape((num_zeta, num_theta))[2:-2, 2:-2],
        Bmag_tz[2:-2, 2:-2],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["|B|_tz"])),
    )
    np.testing.assert_allclose(
        data["B_tz"].reshape((num_zeta, num_theta, 3))[2:-2, 2:-2, :],
        B_tz[2:-2, 2:-2, :],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_tz"])),
    )

    # mixed derivatives wrt rho & zeta
    rtol = 1e-2
    atol = 1e-2
    num_rho = 150
    num_zeta = 180
    grid = LinearGrid(NFP=eq.NFP, rho=num_rho, zeta=num_zeta)
    drho = grid.nodes[:, 0].reshape((num_zeta, num_rho))[0, 1]
    dzeta = grid.nodes[:, 2].reshape((num_zeta, num_rho))[1, 0]
    data = eq.compute(
        [
            "B^theta",
            "B^theta_rz",
            "B^zeta",
            "B^zeta_rz",
            "B_rho",
            "B_rho_rz",
            "B_theta",
            "B_theta_rz",
            "B_zeta",
            "B_zeta_rz",
            "|B|",
            "|B|_rz",
            "B",
            "B_rz",
        ],
        grid=grid,
    )

    B_sup_theta = data["B^theta"].reshape((num_zeta, num_rho))
    B_sup_zeta = data["B^zeta"].reshape((num_zeta, num_rho))
    B_sub_rho = data["B_rho"].reshape((num_zeta, num_rho))
    B_sub_theta = data["B_theta"].reshape((num_zeta, num_rho))
    B_sub_zeta = data["B_zeta"].reshape((num_zeta, num_rho))
    Bmag = data["|B|"].reshape((num_zeta, num_rho))

    B_sup_theta_rz = convolve2d(
        B_sup_theta,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="wrap",
    ) / (drho * dzeta)
    B_sup_zeta_rz = convolve2d(
        B_sup_zeta,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="wrap",
    ) / (drho * dzeta)
    B_sub_rho_rz = convolve2d(
        B_sub_rho,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="wrap",
    ) / (drho * dzeta)
    B_sub_theta_rz = convolve2d(
        B_sub_theta,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="wrap",
    ) / (drho * dzeta)
    B_sub_zeta_rz = convolve2d(
        B_sub_zeta,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="wrap",
    ) / (drho * dzeta)
    Bmag_rz = convolve2d(
        Bmag,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="wrap",
    ) / (drho * dzeta)
    B_rz = np.apply_along_axis(
        myconvolve_2d, 0, data["B"], FD_COEF_1_4, (num_zeta, num_rho)
    ) / (drho * dzeta)

    np.testing.assert_allclose(
        data["B^theta_rz"].reshape((num_zeta, num_rho))[2:-2, 3:-2],
        B_sup_theta_rz[2:-2, 3:-2],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B^theta_rz"])),
    )
    np.testing.assert_allclose(
        data["B^zeta_rz"].reshape((num_zeta, num_rho))[2:-2, 3:-2],
        B_sup_zeta_rz[2:-2, 3:-2],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B^zeta_rz"])),
    )
    np.testing.assert_allclose(
        data["B_rho_rz"].reshape((num_zeta, num_rho))[2:-2, 3:-2],
        B_sub_rho_rz[2:-2, 3:-2],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_rho_rz"])),
    )
    np.testing.assert_allclose(
        data["B_theta_rz"].reshape((num_zeta, num_rho))[2:-2, 3:-2],
        B_sub_theta_rz[2:-2, 3:-2],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_theta_rz"])),
    )
    np.testing.assert_allclose(
        data["B_zeta_rz"].reshape((num_zeta, num_rho))[2:-2, 3:-2],
        B_sub_zeta_rz[2:-2, 3:-2],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_zeta_rz"])),
    )
    np.testing.assert_allclose(
        data["|B|_rz"].reshape((num_zeta, num_rho))[2:-2, 3:-2],
        Bmag_rz[2:-2, 3:-2],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["|B|_rz"])),
    )
    np.testing.assert_allclose(
        data["B_rz"].reshape((num_zeta, num_rho, 3))[2:-2, 3:-2, :],
        B_rz[2:-2, 3:-2, :],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_rz"])),
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
    data = eq.compute(["|B|", "grad(|B|^2)_rho"], grid=grid)
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
    data = eq.compute(["|B|", "grad(|B|^2)_theta"], grid=grid)
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
    data = eq.compute(["|B|", "grad(|B|^2)_zeta"], grid=grid)
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

    data_booz = eq.compute("|B|_mn", grid=grid_full, M_booz=eq.M, N_booz=eq.N)
    data_full = eq.compute(["I", "G"], grid=grid_full)
    data_sym = eq.compute(["I", "G"], grid=grid_sym)

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
    data = eq.compute(["B*grad(|B|)", "(B*grad(|B|))_t"], grid=grid)
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
    data = eq.compute(["B*grad(|B|)", "(B*grad(|B|))_z"], grid=grid)
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
    data = eq.compute("|B|_mn", grid=grid, M_booz=eq.M, N_booz=eq.N)
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


@pytest.mark.unit
def test_compare_quantities_to_vmec():
    """Compare several computed quantities to vmec."""
    wout_file = ".//tests//inputs//wout_DSHAPE.nc"
    desc_file = ".//tests//inputs//DSHAPE_output_saved_without_current.h5"

    fid = netcdf_file(wout_file, mmap=False)
    ns = fid.variables["ns"][()]
    J_dot_B_vmec = fid.variables["jdotb"][()]
    volavgB = fid.variables["volavgB"][()]
    betatotal = fid.variables["betatotal"][()]
    fid.close()

    eq = EquilibriaFamily.load(desc_file)[-1]

    # Compare 0D quantities:
    grid = QuadratureGrid(eq.L, M=eq.M, N=eq.N, NFP=eq.NFP)
    data = eq.compute("<beta>_vol", grid=grid)
    data = eq.compute("<|B|>_rms", grid=grid, data=data)

    np.testing.assert_allclose(volavgB, data["<|B|>_rms"], rtol=1e-7)
    np.testing.assert_allclose(betatotal, data["<beta>_vol"], rtol=1e-5)

    # Compare radial profile quantities:
    s = np.linspace(0, 1, ns)
    rho = np.sqrt(s)
    grid = LinearGrid(rho=rho, M=eq.M, N=eq.N, NFP=eq.NFP)
    data = eq.compute("<J*B>", grid=grid)
    J_dot_B_desc = compress(grid, data["<J*B>"])

    # Drop first point since desc gives NaN:
    np.testing.assert_allclose(J_dot_B_desc[1:], J_dot_B_vmec[1:], rtol=0.005)

@pytest.mark.unit
def test_Bmn_symmetrized1():
    """Test calculation of |B|_mn symmetrized."""
    
    # For an axisymmetric configuration, symmetrizing should have no effect:
    filename = ".//tests//inputs//circular_model_tokamak_output.h5"
    #print(filename)
    eq = EquilibriaFamily.load(filename)[-1]
    grid = LinearGrid(rho=[1], M=eq.M*4, N=eq.N*4, NFP=eq.NFP)
    data = eq.compute("|B|_mn symmetrized", grid=grid)
    #print("|B|_mn:            ", data["|B|_mn"])
    #print("|B|_mn symmetrized:", data["|B|_mn symmetrized"])
    #print("Max diff 1:        ", np.max(np.abs(data["|B|_mn"] - data["|B|_mn symmetrized"])))
    np.testing.assert_allclose(data["|B|_mn"], data["|B|_mn symmetrized"], rtol=1e-14, atol=1e-14)
    
    filename = ".//tests//inputs//LandremanPaul2022_QA_reactorScale_lowRes.h5"
    #print(filename)
    eq = Equilibrium.load(filename)
    grid = LinearGrid(rho=[1], M=eq.M*4, N=eq.N*4, NFP=eq.NFP)

    # For a QA configuration, symmetrizing with respect to QA should have a tiny but
    # nonzero effect:
    data = eq.compute("|B|_mn symmetrized", grid=grid)
    #print("|B|_mn:            ", data["|B|_mn"])
    #print("|B|_mn symmetrized:", data["|B|_mn symmetrized"])
    #print("Max diff 2:        ", np.max(np.abs(data["|B|_mn"] - data["|B|_mn symmetrized"])))
    assert np.max(np.abs(data["|B|_mn"] - data["|B|_mn symmetrized"])) > 0
    np.testing.assert_allclose(data["|B|_mn"], data["|B|_mn symmetrized"], rtol=1e-14, atol=0.003)

    # Symmetrizing a QA with respect to QH modes should cause a big difference:
    data = eq.compute("|B|_mn symmetrized", grid=grid, helicity=(1, 2))
    #print("|B|_mn:            ", data["|B|_mn"])
    #print("|B|_mn symmetrized:", data["|B|_mn symmetrized"])
    #print("Max diff 3:", np.max(np.abs(data["|B|_mn"] - data["|B|_mn symmetrized"])))
    assert np.max(np.abs(data["|B|_mn"] - data["|B|_mn symmetrized"])) > 0.6

    # For a QH configuration, symmetrizing with respect to QH should have a tiny but
    # nonzero effect:
    filename = ".//tests//inputs//LandremanPaul2022_QH_reactorScale_lowRes.h5"
    #print(filename)
    eq = Equilibrium.load(filename)
    grid = LinearGrid(rho=[1], M=eq.M*4, N=eq.N*4, NFP=eq.NFP)
    data = eq.compute("|B|_mn symmetrized", grid=grid, helicity=(1, 4))
    #print("|B|_mn:            ", data["|B|_mn"])
    #print("|B|_mn symmetrized:", data["|B|_mn symmetrized"])
    #print("Max diff 4:", np.max(np.abs(data["|B|_mn"] - data["|B|_mn symmetrized"])))
    assert np.max(np.abs(data["|B|_mn"] - data["|B|_mn symmetrized"])) > 0
    np.testing.assert_allclose(data["|B|_mn"], data["|B|_mn symmetrized"], rtol=1e-14, atol=0.003)

    # Symmetrizing a QH with respect to QA modes should cause a big difference:
    data = eq.compute("|B|_mn symmetrized", grid=grid, helicity=(1, 0))
    #print("|B|_mn:            ", data["|B|_mn"])
    #print("|B|_mn symmetrized:", data["|B|_mn symmetrized"])
    #print("Max diff 5:", np.max(np.abs(data["|B|_mn"] - data["|B|_mn symmetrized"])))
    assert np.max(np.abs(data["|B|_mn"] - data["|B|_mn symmetrized"])) > 0.9

@pytest.mark.unit
def test_Bmn_symmetrized2():
    """Test calculation of |B|_mn symmetrized."""

    NFP = 4
    eq = Equilibrium(L=3, M=3, N=4, NFP=NFP)
    grid = LinearGrid(rho=[1], M=eq.M*4, N=eq.N*4, NFP=eq.NFP)
    data = eq.compute(["|B|_mn", "B modes"], grid=grid)
    #p = np.zeros_like(data["|B|_mn"])
    #q = np.zeros_like(data["|B|_mn"])
    Bmn_initial = np.zeros_like(data["|B|_mn"])
    Bmn_QA = np.zeros_like(data["|B|_mn"])
    Bmn_QH_plus = np.zeros_like(data["|B|_mn"])
    Bmn_QH_minus = np.zeros_like(data["|B|_mn"])
    modes = data["B modes"]
    print("B modes:\n", modes)
    m_desc = modes[:, 1]
    n_desc = modes[:, 2]
    """
    sym_mask = (m * n >= 0)
    p = (0.1 + 1.2 * np.abs(m)) * sym_mask
    q = (0.14 + 0.13 * np.abs(n)) * sym_mask
    """

    mmax = modes[-1, 1]
    nmax = modes[-1, 2]
    # Loop over modes of the cos(m theta - n zeta) basis:
    for m in range(mmax + 1):
        nmin = -nmax
        if m == 0:
            nmin = 0
        for n in range(nmin, nmax + 1):
            amplitude = 0.14 + 0.4 * m + 0.13 * n
            amplitude_signed = amplitude
            if n < 0:
                amplitude_signed = -amplitude
            # Find corresponding indices of the desc basis:
            index1 = np.nonzero(np.logical_and(m_desc == m, n_desc == abs(n)))[0][0]
            index2 = np.nonzero(np.logical_and(m_desc == -m, n_desc == -abs(n)))[0][0]
            assert len(np.nonzero(np.logical_and(m_desc == m, n_desc == abs(n)))[0]) == 1
            assert len(np.nonzero(np.logical_and(m_desc == -m, n_desc == -abs(n)))[0]) == 1
            print(f"m {m}  n {n}  amp {amplitude} amp2 {amplitude_signed}  idx1 {index1}  idx2 {index2}")
            Bmn_initial[index1] += amplitude
            Bmn_initial[index2] += amplitude_signed
            if n == 0:
                Bmn_QA[index1] += amplitude
                Bmn_QA[index2] += amplitude_signed
            if n == m:
                Bmn_QH_plus[index1] += amplitude
                Bmn_QH_plus[index2] += amplitude_signed
            if n == -m:
                Bmn_QH_minus[index1] += amplitude
                Bmn_QH_minus[index2] += amplitude_signed


    """
    for j in range(len(p)):
        m = modes[j, 1]
        n = modes[j, 2]
        if m * n < 0:
            continue
        p[j] = 0.1 + 1.2 * m
        q[j] = 0.14 + 0.13 * n
    """
    data["|B|_mn"] = Bmn_initial
    data2 = eq.compute("|B|_mn symmetrized", data=data.copy(), helicity=(1, 0), grid=grid)
    np.testing.assert_allclose(data2["|B|_mn"], Bmn_initial)
    assert np.max(np.abs(data2["|B|_mn symmetrized"] - Bmn_initial)) > 0.1
    assert np.max(np.abs(data2["|B|_mn symmetrized"] - Bmn_QH_plus)) > 0.1
    assert np.max(np.abs(data2["|B|_mn symmetrized"] - Bmn_QH_minus)) > 0.1
    np.testing.assert_allclose(data2["|B|_mn symmetrized"], Bmn_QA)

    data2 = eq.compute("|B|_mn symmetrized", data=data.copy(), helicity=(1, NFP), grid=grid)
    np.testing.assert_allclose(data2["|B|_mn"], Bmn_initial)
    assert np.max(np.abs(data2["|B|_mn symmetrized"] - Bmn_initial)) > 0.1
    assert np.max(np.abs(data2["|B|_mn symmetrized"] - Bmn_QA)) > 0.1
    assert np.max(np.abs(data2["|B|_mn symmetrized"] - Bmn_QH_minus)) > 0.1
    np.testing.assert_allclose(data2["|B|_mn symmetrized"], Bmn_QH_plus)

    data2 = eq.compute("|B|_mn symmetrized", data=data.copy(), helicity=(1, -NFP), grid=grid)
    np.testing.assert_allclose(data2["|B|_mn"], Bmn_initial)
    assert np.max(np.abs(data2["|B|_mn symmetrized"] - Bmn_initial)) > 0.1
    assert np.max(np.abs(data2["|B|_mn symmetrized"] - Bmn_QH_plus)) > 0.1
    assert np.max(np.abs(data2["|B|_mn symmetrized"] - Bmn_QA)) > 0.1
    np.testing.assert_allclose(data2["|B|_mn symmetrized"], Bmn_QH_minus)

@pytest.mark.unit
def test_B_symmetrized():
    filename = ".//tests//inputs//precise_QH_step0.h5"
    #print(filename)
    eq = Equilibrium.load(filename)[-1]
    grid = LinearGrid(rho=[1], M=eq.M*4, N=eq.N*4, NFP=eq.NFP)
    #helicity = (1, 0)
    #helicity = (1, eq.NFP)
    helicity = (1, -eq.NFP)
    data = eq.compute(["|B| Boozer", "|B| Boozer symmetrized"], grid=grid, helicity=helicity)

    modB = data["|B| Boozer"].reshape((grid.num_theta, grid.num_zeta), order="F")
    modB_sym = data["|B| Boozer symmetrized"].reshape((grid.num_theta, grid.num_zeta), order="F")

    import matplotlib.pyplot as plt
    nrows = 1
    ncols = 2
    plt.figure(figsize=(14.5, 6))
    plt.subplot(nrows, ncols, 1)
    plt.contourf(modB)
    plt.title("|B| vs Boozer angles, no filtering")
    plt.colorbar()
    plt.subplot(nrows, ncols, 2)
    plt.contourf(modB_sym)
    plt.title("|B| vs Boozer angles, filtered")
    plt.colorbar()
    plt.show()

@pytest.mark.unit
def test_compute_everything():
    """Make sure we can compute everything without errors."""
    eq = Equilibrium(1, 1, 1)
    grid = LinearGrid(1, 1, 1)
    for key in data_index.keys():
        data = eq.compute(key, grid=grid)
        assert key in data
