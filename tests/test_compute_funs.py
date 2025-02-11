"""Tests for compute functions."""

import numpy as np
import pytest
from scipy.signal import convolve2d

from desc.compute import rpz2xyz_vec
from desc.equilibrium import Equilibrium
from desc.equilibrium.coords import get_rtz_grid
from desc.examples import get
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid
from desc.io import load
from desc.utils import cross, dot

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
        boundary="fill",  # not periodic in rho, easier to pad and truncate in all dims
    )
    return conv


@pytest.mark.unit
def test_aliases():
    """Tests that data_index aliases are equal."""
    surface = FourierRZToroidalSurface(
        R_lmn=[10, 1, 0.2],
        Z_lmn=[-2, -0.2],
        modes_R=[[0, 0], [1, 0], [0, 1]],
        modes_Z=[[-1, 0], [0, -1]],
    )

    eq = Equilibrium(surface=surface)

    # automatic case
    primary_data = eq.compute("R_tz")
    alias_data = eq.compute("R_zt")
    np.testing.assert_allclose(primary_data["R_tz"], alias_data["R_zt"])

    # manual case
    primary_data = eq.compute("e_rho_rt")
    alias_data = eq.compute(["x_rrt", "e_theta_rr"])
    np.testing.assert_allclose(primary_data["e_rho_rt"], alias_data["x_rrt"])
    np.testing.assert_allclose(primary_data["e_rho_rt"], alias_data["e_theta_rr"])


@pytest.mark.unit
def test_total_volume(DummyStellarator):
    """Test that the volume enclosed by the LCFS is equal to the total volume."""
    eq = load(load_from=str(DummyStellarator["output_path"]), file_format="hdf5")

    grid = LinearGrid(M=12, N=12, NFP=eq.NFP, sym=eq.sym)  # rho = 1
    lcfs_volume = eq.compute("V(r)", grid=grid)["V(r)"]
    total_volume = eq.compute("V")["V"]  # default quadrature grid
    np.testing.assert_allclose(lcfs_volume, total_volume)


@pytest.mark.unit
def test_enclosed_volumes():
    """Test that the volume enclosed by flux surfaces matches analytic formulas."""
    R0 = 10
    surf = FourierRZToroidalSurface(
        R_lmn=[R0, 1, 0.2],
        Z_lmn=[-2, -0.2],
        modes_R=[[0, 0], [1, 0], [0, 1]],
        modes_Z=[[-1, 0], [0, -1]],
    )
    # 𝐞(ρ, θ, ζ) = R(ρ, θ, ζ) 𝐫 + Z(ρ, θ, ζ) 𝐳
    # V(ρ) = ∯ dθ dζ (∂_θ 𝐞 × ∂_ζ 𝐞) ⋅ (0, 0, Z)
    #      = ∯ dθ dζ (R₀ + ρ cos θ + 0.2 cos ζ) (2 ρ² sin²θ − 0.2 ρ sin θ sin ζ)
    np.testing.assert_allclose(4 * R0 * np.pi**2, surf.compute(["V"])["V"])
    eq = Equilibrium(surface=surf)  # elliptical cross-section with torsion
    rho = np.linspace(0, 1, 64)
    grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=rho)
    data = eq.compute(["R0", "V(r)", "V_r(r)", "V_rr(r)", "V_rrr(r)"], grid=grid)
    np.testing.assert_allclose(
        4 * data["R0"] * (np.pi * rho) ** 2, grid.compress(data["V(r)"])
    )
    np.testing.assert_allclose(
        8 * data["R0"] * np.pi**2 * rho, grid.compress(data["V_r(r)"])
    )
    np.testing.assert_allclose(8 * data["R0"] * np.pi**2, data["V_rr(r)"])
    np.testing.assert_allclose(0, data["V_rrr(r)"], atol=2e-14)


@pytest.mark.unit
def test_enclosed_areas():
    """Test that the area enclosed by flux surfaces matches analytic formulas."""
    surf = FourierRZToroidalSurface(
        R_lmn=[10, 1, 0.2],
        Z_lmn=[-2, -0.2],
        modes_R=[[0, 0], [1, 0], [0, 1]],
        modes_Z=[[-1, 0], [0, -1]],
    )
    eq = Equilibrium(surface=surf)  # elliptical cross-section with torsion
    rho = np.linspace(0, 1, 64)
    grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=rho)
    data = eq.compute(["A(r)"], grid=grid)
    # area = π a b = 2 π ρ²
    np.testing.assert_allclose(2 * np.pi * rho**2, grid.compress(data["A(r)"]))


@pytest.mark.unit
def test_surface_areas():
    """Test that the flux surface areas match known analytic formulas."""
    eq = Equilibrium()  # torus
    rho = np.linspace(0, 1, 64)
    grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=rho)
    data = eq.compute(["R0", "S(r)", "S_r(r)", "S_rr(r)"], grid=grid)
    np.testing.assert_allclose(
        4 * data["R0"] * np.pi**2 * rho, grid.compress(data["S(r)"])
    )
    np.testing.assert_allclose(4 * data["R0"] * np.pi**2, data["S_r(r)"])
    np.testing.assert_allclose(0, data["S_rr(r)"], atol=3e-12)


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


@pytest.mark.unit
def test_elongation():
    """Test that elongation approximation is correct."""
    surf1 = FourierRZToroidalSurface(
        R_lmn=[10, 1, 0.2],
        Z_lmn=[-1, -0.2],
        modes_R=[[0, 0], [1, 0], [0, 1]],
        modes_Z=[[-1, 0], [0, -1]],
    )
    surf2 = FourierRZToroidalSurface(
        R_lmn=[10, 1, 0.2],
        Z_lmn=[-2, -0.2],
        modes_R=[[0, 0], [1, 0], [0, 1]],
        modes_Z=[[-1, 0], [0, -1]],
    )
    surf3 = FourierRZToroidalSurface(
        R_lmn=[10, 1, 0.2],
        Z_lmn=[-3, -0.2],
        modes_R=[[0, 0], [1, 0], [0, 1]],
        modes_Z=[[-1, 0], [0, -1]],
    )
    assert surf3.sym
    grid = LinearGrid(rho=1, M=3 * surf3.M, N=surf3.N, NFP=surf3.NFP, sym=False)
    data1 = surf1.compute(["a_major/a_minor LCFS"], grid=grid)
    data2 = surf2.compute(["a_major/a_minor LCFS"], grid=grid)
    data3 = surf3.compute(["a_major/a_minor LCFS"], grid=grid)
    # elongation approximation is less accurate as elongation increases
    np.testing.assert_allclose(1.0, data1["a_major/a_minor LCFS"])
    np.testing.assert_allclose(2.0, data2["a_major/a_minor LCFS"], rtol=1e-4)
    np.testing.assert_allclose(3.0, data3["a_major/a_minor LCFS"], rtol=1e-3)


@pytest.mark.slow
@pytest.mark.unit
def test_magnetic_field_derivatives(DummyStellarator):
    """Test that the derivatives of B and |B| are close to numerical derivatives."""
    eq = load(load_from=str(DummyStellarator["output_path"]), file_format="hdf5")

    # partial derivatives wrt rho
    rtol = 1e-3
    atol = 1e-3
    num_rho = 180
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
            "phi",
        ],
        grid=grid,
    )
    data["B"] = rpz2xyz_vec(data["B"], phi=data["phi"])
    data["B_r"] = rpz2xyz_vec(data["B_r"], phi=data["phi"])
    data["B_rr"] = rpz2xyz_vec(data["B_rr"], phi=data["phi"])

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
        data["B^theta_r"][4:-4],
        B_sup_theta_r[4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B^theta_r"])),
    )
    np.testing.assert_allclose(
        data["B^theta_rr"][4:-4],
        B_sup_theta_rr[4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B^theta_rr"])),
    )
    np.testing.assert_allclose(
        data["B^zeta_r"][4:-4],
        B_sup_zeta_r[4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B^zeta_r"])),
    )
    np.testing.assert_allclose(
        data["B^zeta_rr"][4:-4],
        B_sup_zeta_rr[4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B^zeta_rr"])),
    )
    np.testing.assert_allclose(
        data["B_rho_r"][4:-4],
        B_sub_rho_r[4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_rho_r"])),
    )
    np.testing.assert_allclose(
        data["B_rho_rr"][4:-4],
        B_sub_rho_rr[4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_rho_rr"])),
    )
    np.testing.assert_allclose(
        data["B_theta_r"][4:-4],
        B_sub_theta_r[4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_theta_r"])),
    )
    np.testing.assert_allclose(
        data["B_theta_rr"][4:-4],
        B_sub_theta_rr[4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_theta_rr"])),
    )
    np.testing.assert_allclose(
        data["B_zeta_r"][4:-4],
        B_sub_zeta_r[4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_zeta_r"])),
    )
    np.testing.assert_allclose(
        data["B_zeta_rr"][4:-4],
        B_sub_zeta_rr[4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_zeta_rr"])),
    )
    np.testing.assert_allclose(
        data["|B|_r"][4:-4],
        Bmag_r[4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["|B|_r"])),
    )
    np.testing.assert_allclose(
        data["|B|_rr"][4:-4],
        Bmag_rr[4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["|B|_rr"])),
    )
    np.testing.assert_allclose(
        data["B_r"][4:-4],
        B_r[4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_r"])),
    )
    np.testing.assert_allclose(
        data["B_rr"][4:-4],
        B_rr[4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_rr"])),
    )

    # partial derivatives wrt theta
    rtol = 1e-3
    atol = 1e-3
    num_theta = 180
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
            "phi",
        ],
        grid=grid,
    )
    data["B"] = rpz2xyz_vec(data["B"], phi=data["phi"])
    data["B_t"] = rpz2xyz_vec(data["B_t"], phi=data["phi"])
    data["B_tt"] = rpz2xyz_vec(data["B_tt"], phi=data["phi"])

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
        data["B^theta_t"][4:-4],
        B_sup_theta_t[4:-4],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B^theta_t"])),
    )
    np.testing.assert_allclose(
        data["B^theta_tt"][4:-4],
        B_sup_theta_tt[4:-4],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B^theta_tt"])),
    )
    np.testing.assert_allclose(
        data["B^zeta_t"][4:-4],
        B_sup_zeta_t[4:-4],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B^zeta_t"])),
    )
    np.testing.assert_allclose(
        data["B^zeta_tt"][4:-4],
        B_sup_zeta_tt[4:-4],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B^zeta_tt"])),
    )
    np.testing.assert_allclose(
        data["B_rho_t"][4:-4],
        B_sub_rho_t[4:-4],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B_rho_t"])),
    )
    np.testing.assert_allclose(
        data["B_rho_tt"][4:-4],
        B_sub_rho_tt[4:-4],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B_rho_tt"])),
    )
    np.testing.assert_allclose(
        data["B_theta_t"][4:-4],
        B_sub_theta_t[4:-4],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B_theta_t"])),
    )
    np.testing.assert_allclose(
        data["B_theta_tt"][4:-4],
        B_sub_theta_tt[4:-4],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B_theta_tt"])),
    )
    np.testing.assert_allclose(
        data["B_zeta_t"][4:-4],
        B_sub_zeta_t[4:-4],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B_zeta_t"])),
    )
    np.testing.assert_allclose(
        data["B_zeta_tt"][4:-4],
        B_sub_zeta_tt[4:-4],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B_zeta_tt"])),
    )
    np.testing.assert_allclose(
        data["|B|_t"][4:-4],
        Bmag_t[4:-4],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["|B|_t"])),
    )
    np.testing.assert_allclose(
        data["|B|_tt"][4:-4],
        Bmag_tt[4:-4],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["|B|_tt"])),
    )
    np.testing.assert_allclose(
        data["B_t"][4:-4],
        B_t[4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_t"])),
    )
    np.testing.assert_allclose(
        data["B_tt"][4:-4],
        B_tt[4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_tt"])),
    )

    # partial derivatives wrt zeta
    rtol = 1e-3
    atol = 1e-3
    num_zeta = 180
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
            "phi",
        ],
        grid=grid,
    )
    data["B"] = rpz2xyz_vec(data["B"], phi=data["phi"])
    data["B_z"] = rpz2xyz_vec(data["B_z"], phi=data["phi"])
    data["B_zz"] = rpz2xyz_vec(data["B_zz"], phi=data["phi"])

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
        data["B^theta_z"][4:-4],
        B_sup_theta_z[4:-4],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B^theta_z"])),
    )
    np.testing.assert_allclose(
        data["B^theta_zz"][4:-4],
        B_sup_theta_zz[4:-4],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B^theta_zz"])),
    )
    np.testing.assert_allclose(
        data["B^zeta_z"][4:-4],
        B_sup_zeta_z[4:-4],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B^zeta_z"])),
    )
    np.testing.assert_allclose(
        data["B^zeta_zz"][4:-4],
        B_sup_zeta_zz[4:-4],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B^zeta_zz"])),
    )
    np.testing.assert_allclose(
        data["B_rho_z"][4:-4],
        B_sub_rho_z[4:-4],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B_rho_z"])),
    )
    np.testing.assert_allclose(
        data["B_rho_zz"][4:-4],
        B_sub_rho_zz[4:-4],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B_rho_zz"])),
    )
    np.testing.assert_allclose(
        data["B_theta_z"][4:-4],
        B_sub_theta_z[4:-4],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B_theta_z"])),
    )
    np.testing.assert_allclose(
        data["B_theta_zz"][4:-4],
        B_sub_theta_zz[4:-4],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B_theta_zz"])),
    )
    np.testing.assert_allclose(
        data["B_zeta_z"][4:-4],
        B_sub_zeta_z[4:-4],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B_zeta_z"])),
    )
    np.testing.assert_allclose(
        data["B_zeta_zz"][4:-4],
        B_sub_zeta_zz[4:-4],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B_zeta_zz"])),
    )
    np.testing.assert_allclose(
        data["|B|_z"][4:-4],
        Bmag_z[4:-4],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["|B|_z"])),
    )
    np.testing.assert_allclose(
        data["|B|_zz"][4:-4],
        Bmag_zz[4:-4],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["|B|_zz"])),
    )
    np.testing.assert_allclose(
        data["B_z"][4:-4],
        B_z[4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_z"])),
    )
    np.testing.assert_allclose(
        data["B_zz"][4:-4],
        B_zz[4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_zz"])),
    )

    # mixed derivatives wrt rho & theta
    rtol = 1e-2
    atol = 1e-2
    num_rho = 180
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
            "phi",
        ],
        grid=grid,
    )
    data["B"] = rpz2xyz_vec(data["B"], phi=data["phi"])
    data["B_rt"] = rpz2xyz_vec(data["B_rt"], phi=data["phi"])

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
        boundary="fill",
    ) / (drho * dtheta)
    B_sup_zeta_rt = convolve2d(
        B_sup_zeta,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="fill",
    ) / (drho * dtheta)
    B_sub_rho_rt = convolve2d(
        B_sub_rho,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="fill",
    ) / (drho * dtheta)
    B_sub_theta_rt = convolve2d(
        B_sub_theta,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="fill",
    ) / (drho * dtheta)
    B_sub_zeta_rt = convolve2d(
        B_sub_zeta,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="fill",
    ) / (drho * dtheta)
    Bmag_rt = convolve2d(
        Bmag,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="fill",
    ) / (drho * dtheta)
    B_rt = np.apply_along_axis(
        myconvolve_2d, 0, data["B"], FD_COEF_1_4, (num_rho, num_theta)
    ) / (drho * dtheta)

    np.testing.assert_allclose(
        data["B^theta_rt"].reshape((num_rho, num_theta))[4:-4, 4:-4],
        B_sup_theta_rt[4:-4, 4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B^theta_rt"])),
    )
    np.testing.assert_allclose(
        data["B^zeta_rt"].reshape((num_rho, num_theta))[4:-4, 4:-4],
        B_sup_zeta_rt[4:-4, 4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B^zeta_rt"])),
    )
    np.testing.assert_allclose(
        data["B_rho_rt"].reshape((num_rho, num_theta))[4:-4, 4:-4],
        B_sub_rho_rt[4:-4, 4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_rho_rt"])),
    )
    np.testing.assert_allclose(
        data["B_theta_rt"].reshape((num_rho, num_theta))[4:-4, 4:-4],
        B_sub_theta_rt[4:-4, 4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_theta_rt"])),
    )
    np.testing.assert_allclose(
        data["B_zeta_rt"].reshape((num_rho, num_theta))[4:-4, 4:-4],
        B_sub_zeta_rt[4:-4, 4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_zeta_rt"])),
    )
    np.testing.assert_allclose(
        data["|B|_rt"].reshape((num_rho, num_theta))[4:-4, 4:-4],
        Bmag_rt[4:-4, 4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["|B|_rt"])),
    )
    np.testing.assert_allclose(
        data["B_rt"].reshape((num_rho, num_theta, 3))[4:-4, 4:-4],
        B_rt[4:-4, 4:-4],
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
            "phi",
        ],
        grid=grid,
    )
    data["B"] = rpz2xyz_vec(data["B"], phi=data["phi"])
    data["B_tz"] = rpz2xyz_vec(data["B_tz"], phi=data["phi"])

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
        boundary="fill",
    ) / (dtheta * dzeta)
    B_sup_zeta_tz = convolve2d(
        B_sup_zeta,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="fill",
    ) / (dtheta * dzeta)
    B_sub_rho_tz = convolve2d(
        B_sub_rho,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="fill",
    ) / (dtheta * dzeta)
    B_sub_theta_tz = convolve2d(
        B_sub_theta,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="fill",
    ) / (dtheta * dzeta)
    B_sub_zeta_tz = convolve2d(
        B_sub_zeta,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="fill",
    ) / (dtheta * dzeta)
    Bmag_tz = convolve2d(
        Bmag,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="fill",
    ) / (dtheta * dzeta)
    B_tz = np.apply_along_axis(
        myconvolve_2d, 0, data["B"], FD_COEF_1_4, (num_zeta, num_theta)
    ) / (dzeta * dtheta)

    np.testing.assert_allclose(
        data["B^theta_tz"].reshape((num_zeta, num_theta))[4:-4, 4:-4],
        B_sup_theta_tz[4:-4, 4:-4],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B^theta_tz"])),
    )
    np.testing.assert_allclose(
        data["B^zeta_tz"].reshape((num_zeta, num_theta))[4:-4, 4:-4],
        B_sup_zeta_tz[4:-4, 4:-4],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B^zeta_tz"])),
    )
    np.testing.assert_allclose(
        data["B_rho_tz"].reshape((num_zeta, num_theta))[4:-4, 4:-4],
        B_sub_rho_tz[4:-4, 4:-4],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B_rho_tz"])),
    )
    np.testing.assert_allclose(
        data["B_theta_tz"].reshape((num_zeta, num_theta))[4:-4, 4:-4],
        B_sub_theta_tz[4:-4, 4:-4],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B_theta_tz"])),
    )
    np.testing.assert_allclose(
        data["B_zeta_tz"].reshape((num_zeta, num_theta))[4:-4, 4:-4],
        B_sub_zeta_tz[4:-4, 4:-4],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["B_zeta_tz"])),
    )
    np.testing.assert_allclose(
        data["|B|_tz"].reshape((num_zeta, num_theta))[4:-4, 4:-4],
        Bmag_tz[4:-4, 4:-4],
        rtol=rtol,
        atol=atol * np.mean(np.abs(data["|B|_tz"])),
    )
    np.testing.assert_allclose(
        data["B_tz"].reshape((num_zeta, num_theta, 3))[4:-4, 4:-4],
        B_tz[4:-4, 4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_tz"])),
    )

    # mixed derivatives wrt rho & zeta
    rtol = 1e-2
    atol = 1e-2
    num_rho = 180
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
            "phi",
        ],
        grid=grid,
    )
    data["B"] = rpz2xyz_vec(data["B"], phi=data["phi"])
    data["B_rz"] = rpz2xyz_vec(data["B_rz"], phi=data["phi"])

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
        boundary="fill",
    ) / (drho * dzeta)
    B_sup_zeta_rz = convolve2d(
        B_sup_zeta,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="fill",
    ) / (drho * dzeta)
    B_sub_rho_rz = convolve2d(
        B_sub_rho,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="fill",
    ) / (drho * dzeta)
    B_sub_theta_rz = convolve2d(
        B_sub_theta,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="fill",
    ) / (drho * dzeta)
    B_sub_zeta_rz = convolve2d(
        B_sub_zeta,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="fill",
    ) / (drho * dzeta)
    Bmag_rz = convolve2d(
        Bmag,
        FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
        mode="same",
        boundary="fill",
    ) / (drho * dzeta)
    B_rz = np.apply_along_axis(
        myconvolve_2d, 0, data["B"], FD_COEF_1_4, (num_zeta, num_rho)
    ) / (drho * dzeta)

    np.testing.assert_allclose(
        data["B^theta_rz"].reshape((num_zeta, num_rho))[4:-4, 4:-4],
        B_sup_theta_rz[4:-4, 4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B^theta_rz"])),
    )
    np.testing.assert_allclose(
        data["B^zeta_rz"].reshape((num_zeta, num_rho))[4:-4, 4:-4],
        B_sup_zeta_rz[4:-4, 4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B^zeta_rz"])),
    )
    np.testing.assert_allclose(
        data["B_rho_rz"].reshape((num_zeta, num_rho))[4:-4, 4:-4],
        B_sub_rho_rz[4:-4, 4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_rho_rz"])),
    )
    np.testing.assert_allclose(
        data["B_theta_rz"].reshape((num_zeta, num_rho))[4:-4, 4:-4],
        B_sub_theta_rz[4:-4, 4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_theta_rz"])),
    )
    np.testing.assert_allclose(
        data["B_zeta_rz"].reshape((num_zeta, num_rho))[4:-4, 4:-4],
        B_sub_zeta_rz[4:-4, 4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_zeta_rz"])),
    )
    np.testing.assert_allclose(
        data["|B|_rz"].reshape((num_zeta, num_rho))[4:-4, 4:-4],
        Bmag_rz[4:-4, 4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["|B|_rz"])),
    )
    np.testing.assert_allclose(
        data["B_rz"].reshape((num_zeta, num_rho, 3))[4:-4, 4:-4],
        B_rz[4:-4, 4:-4],
        rtol=rtol,
        atol=atol * np.nanmean(np.abs(data["B_rz"])),
    )


@pytest.mark.unit
def test_metric_derivatives(DummyStellarator):
    """Compare analytic formula for metric derivatives with finite differences."""
    eq = load(load_from=str(DummyStellarator["output_path"]), file_format="hdf5")

    metric_components = ["g^rr", "g^rt", "g^rz", "g^tt", "g^tz", "g^zz"]

    # rho derivatives
    grid = LinearGrid(rho=np.linspace(0.5, 0.7, 100))
    drho = np.diff(grid.nodes[:, 0]).mean()
    data = eq.compute(
        metric_components + [foo + "_r" for foo in metric_components], grid=grid
    )
    for thing in metric_components:
        # some of these are so close to zero FD doesn't really work...
        scale = np.linalg.norm(data[thing]) / data[thing].size
        if scale < 1e-16:
            continue
        dthing_fd = np.convolve(data[thing], FD_COEF_1_4, "same") / drho
        dthing_ex = data[thing + "_r"]
        np.testing.assert_allclose(
            dthing_fd[3:-3], dthing_ex[3:-3], err_msg=thing, rtol=1e-3, atol=1e-3
        )

    # theta derivatives
    grid = LinearGrid(theta=np.linspace(0, np.pi / 4, 100))
    dtheta = np.diff(grid.nodes[:, 1]).mean()
    data = eq.compute(
        metric_components + [foo + "_t" for foo in metric_components], grid=grid
    )
    for thing in metric_components:
        # some of these are so close to zero FD doesn't really work...
        scale = np.linalg.norm(data[thing]) / data[thing].size
        if scale < 1e-16:
            continue
        dthing_fd = np.convolve(data[thing], FD_COEF_1_4, "same") / dtheta
        dthing_ex = data[thing + "_t"]
        np.testing.assert_allclose(
            dthing_fd[3:-3], dthing_ex[3:-3], err_msg=thing, rtol=1e-3, atol=1e-3
        )

    # zeta derivatives
    grid = LinearGrid(zeta=np.linspace(0, np.pi / 4, 100), NFP=3)
    dzeta = np.diff(grid.nodes[:, 2]).mean()
    data = eq.compute(
        metric_components + [foo + "_z" for foo in metric_components], grid=grid
    )
    for thing in metric_components:
        # some of these are so close to zero FD doesn't really work...
        scale = np.linalg.norm(data[thing]) / data[thing].size
        if scale < 1e-16:
            continue
        dthing_fd = np.convolve(data[thing], FD_COEF_1_4, "same") / dzeta
        dthing_ex = data[thing + "_z"]
        np.testing.assert_allclose(
            dthing_fd[3:-3], dthing_ex[3:-3], err_msg=thing, rtol=1e-3, atol=1e-3
        )


@pytest.mark.slow
@pytest.mark.unit
def test_magnetic_pressure_gradient(DummyStellarator):
    """Test that the components of grad(|B|^2)) match with numerical gradients."""
    eq = load(load_from=str(DummyStellarator["output_path"]), file_format="hdf5")

    # partial derivatives wrt rho
    num_rho = 110
    grid = LinearGrid(NFP=eq.NFP, rho=num_rho)
    drho = grid.nodes[1, 0]
    data = eq.compute(["|B|^2", "grad(|B|^2)_rho"], grid=grid)
    B2_r = np.convolve(data["|B|^2"], FD_COEF_1_4, "same") / drho
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
    data = eq.compute(["|B|^2", "grad(|B|^2)_theta"], grid=grid)
    B2_t = np.convolve(data["|B|^2"], FD_COEF_1_4, "same") / dtheta
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
    data = eq.compute(["|B|^2", "grad(|B|^2)_zeta"], grid=grid)
    B2_z = np.convolve(data["|B|^2"], FD_COEF_1_4, "same") / dzeta
    np.testing.assert_allclose(
        data["grad(|B|^2)_zeta"][2:-2],
        B2_z[2:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(data["grad(|B|^2)_zeta"])),
    )


@pytest.mark.slow
@pytest.mark.unit
def test_BdotgradB(DummyStellarator):
    """Test that the components of grad(B*grad(|B|)) match with numerical gradients."""
    eq = load(load_from=str(DummyStellarator["output_path"]), file_format="hdf5")

    def test_partial_derivative(name):
        cases = {
            "r": {"label": "rho", "column_id": 0},
            "t": {"label": "theta", "column_id": 1},
            "z": {"label": "zeta", "column_id": 2},
        }[name[-1]]
        grid = LinearGrid(NFP=eq.NFP, **{cases["label"]: 120})
        dx = grid.nodes[1, cases["column_id"]]
        data = eq.compute(["B*grad(|B|)", name], grid=grid)
        Btilde_x = np.convolve(data["B*grad(|B|)"], FD_COEF_1_4, "same") / dx
        np.testing.assert_allclose(
            actual=data[name][2:-2],
            desired=Btilde_x[2:-2],
            rtol=2e-2,
            atol=2e-2 * np.mean(np.abs(data[name])),
        )

    test_partial_derivative("(B*grad(|B|))_r")
    test_partial_derivative("(B*grad(|B|))_t")
    test_partial_derivative("(B*grad(|B|))_z")


@pytest.mark.unit
@pytest.mark.solve
def test_boozer_transform():
    """Test that Boozer coordinate transform agrees with BOOZ_XFORM."""
    # TODO (#680): add test with stellarator example
    eq = get("DSHAPE_CURRENT")
    grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    data = eq.compute("|B|_mn_B", grid=grid, M_booz=eq.M, N_booz=eq.N)
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
        np.flipud(np.sort(np.abs(data["|B|_mn_B"]))),
        booz_xform,
        rtol=1e-3,
        atol=1e-4,
    )


@pytest.mark.unit
def test_boozer_transform_multiple_surfaces():
    """Test that computing over multiple surfaces is the same as over 1 at a time."""
    eq = get("HELIOTRON")
    grid1 = LinearGrid(rho=0.6, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    grid2 = LinearGrid(rho=0.8, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    grid3 = LinearGrid(rho=np.array([0.6, 0.8]), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    data1 = eq.compute("|B|_mn_B", grid=grid1, M_booz=eq.M, N_booz=eq.N)
    data2 = eq.compute("|B|_mn_B", grid=grid2, M_booz=eq.M, N_booz=eq.N)
    data3 = eq.compute("|B|_mn_B", grid=grid3, M_booz=eq.M, N_booz=eq.N)
    np.testing.assert_allclose(
        data1["|B|_mn_B"], data3["|B|_mn_B"].reshape((grid3.num_rho, -1))[0]
    )
    np.testing.assert_allclose(
        data2["|B|_mn_B"], data3["|B|_mn_B"].reshape((grid3.num_rho, -1))[1]
    )


@pytest.mark.unit
def test_compute_averages():
    """Test that computing averages uses the correct grid."""
    eq = get("HELIOTRON")
    V_r = eq.get_profile("V_r(r)")
    rho = np.linspace(0.01, 1, 20)
    grid = LinearGrid(rho=rho, NFP=eq.NFP)
    out = eq.compute("V_r(r)", grid=grid)
    np.testing.assert_allclose(V_r(rho), out["V_r(r)"], rtol=1e-4)

    eq = Equilibrium(1, 1, 1)
    grid = LinearGrid(rho=[0.3], theta=[np.pi / 3], zeta=[0])
    out = eq.compute("A", grid=grid)
    np.testing.assert_allclose(out["A"], np.pi)


@pytest.mark.unit
def test_covariant_basis_vectors(DummyStellarator):
    """Test calculation of covariant basis vectors by comparing to finite diff of x."""
    eq = load(load_from=str(DummyStellarator["output_path"]), file_format="hdf5")
    keys = [
        "e_rho",
        "e_rho_r",
        "e_rho_rr",
        "e_rho_rrr",
        "e_rho_rrt",
        "e_rho_rrz",
        "e_rho_rt",
        "e_rho_rtt",
        "e_rho_rtz",
        "e_rho_rz",
        "e_rho_rzz",
        "e_rho_t",
        "e_rho_tt",
        "e_rho_tz",
        "e_rho_z",
        "e_rho_zz",
        "e_theta",
        "e_theta_r",
        "e_theta_rr",
        "e_theta_rrr",
        "e_theta_rrt",
        "e_theta_rrz",
        "e_theta_rt",
        "e_theta_rtt",
        "e_theta_rtz",
        "e_theta_rz",
        "e_theta_rzz",
        "e_theta_t",
        "e_theta_tt",
        "e_theta_tz",
        "e_theta_z",
        "e_theta_zz",
        "e_zeta",
        "e_zeta_r",
        "e_zeta_rr",
        "e_zeta_rrr",
        "e_zeta_rrt",
        "e_zeta_rrz",
        "e_zeta_rt",
        "e_zeta_rtt",
        "e_zeta_rtz",
        "e_zeta_rz",
        "e_zeta_rzz",
        "e_zeta_t",
        "e_zeta_tt",
        "e_zeta_tz",
        "e_zeta_z",
        "e_zeta_zz",
    ]
    grids = {
        "r": LinearGrid(1000, 0, 0, NFP=eq.NFP),
        "t": LinearGrid(0, 1000, 0, NFP=eq.NFP),
        "z": LinearGrid(0, 0, 1000, NFP=eq.NFP),
    }

    for key in keys:
        print(key)
        split = key.split("_")
        # higher order finite differences are unstable, so we only ever do 1 order
        # eg compare e_rho vs fd of x, e_rho_t vs fd of e_rho etc.
        if len(split) == 2:  # stuff like e_rho, e_theta
            base = ["X", "Y", "Z"]
            deriv = split[-1][0]
        else:
            deriv = split[-1]
            if len(deriv) == 1:  # first derivative of basis vector
                base = [split[0] + "_" + split[1]]
            else:
                base = [split[0] + "_" + split[1] + "_" + deriv[:-1]]
                deriv = deriv[-1]

        grid = grids[deriv]
        data = eq.compute([key] + base + ["phi"], grid=grid)
        data[key] = rpz2xyz_vec(data[key], phi=data["phi"]).reshape(
            (grid.num_theta, grid.num_rho, grid.num_zeta, -1), order="F"
        )
        if base == ["X", "Y", "Z"]:
            base = np.array([data["X"], data["Y"], data["Z"]]).T.reshape(
                (grid.num_theta, grid.num_rho, grid.num_zeta, -1), order="F"
            )

        else:
            base = rpz2xyz_vec(data[base[0]], phi=data["phi"]).reshape(
                (grid.num_theta, grid.num_rho, grid.num_zeta, -1), order="F"
            )

        spacing = {
            "r": grid.spacing[0, 0],
            "t": grid.spacing[0, 1],
            "z": grid.spacing[0, 2] / grid.NFP,
        }

        dx = np.apply_along_axis(my_convolve, 0, base, FD_COEF_1_4) / spacing[deriv]
        np.testing.assert_allclose(
            data[key][4:-4],
            dx[4:-4],
            rtol=1e-6,
            atol=1e-6,
            err_msg=key,
        )


@pytest.mark.unit
def test_contravariant_basis_vectors():
    """Test calculation of contravariant basis vectors by comparing to finite diff."""
    eq = get("HELIOTRON")
    keys = [
        "e^rho",
        "e^theta",
        "e^zeta",
        "e^rho_r",
        "e^rho_t",
        "e^rho_z",
        "e^theta_r",
        "e^theta_t",
        "e^theta_z",
        "e^zeta_r",
        "e^zeta_t",
        "e^zeta_z",
        "e^rho_rr",
        "e^theta_rr",
        "e^zeta_rr",
        "e^rho_rt",
        "e^rho_tt",
        "e^theta_rt",
        "e^theta_tt",
        "e^zeta_rt",
        "e^zeta_tt",
        "e^rho_rz",
        "e^rho_tz",
        "e^rho_zz",
        "e^theta_rz",
        "e^theta_tz",
        "e^theta_zz",
        "e^zeta_rz",
        "e^zeta_tz",
        "e^zeta_zz",
    ]
    gridsize = 300
    grids = {
        "r": LinearGrid(gridsize, 0, 0, NFP=eq.NFP, axis=False),
        "t": LinearGrid(0, gridsize, 0, NFP=eq.NFP, axis=False),
        "z": LinearGrid(0, 0, gridsize, NFP=eq.NFP, axis=False),
        "rt": LinearGrid(gridsize, gridsize, 0, NFP=eq.NFP, axis=False),
        "tz": LinearGrid(0, gridsize, gridsize, NFP=eq.NFP, axis=False),
        "rz": LinearGrid(gridsize, 0, gridsize, NFP=eq.NFP, axis=False),
    }

    atol = 2e-3

    for key in keys[3:]:  # don't test base vectors for now
        split = key.split("_")
        base_quant = split[0]
        second_deriv = False
        # higher order finite differences are unstable, so we only ever do 1 order
        # eg compare e_rho vs fd of x, e_rho_t vs fd of e_rho etc.
        if len(split) == 1:  # stuff like e^rho, e^theta
            # testing the e^rho etc against finite difference would involve
            # getting a equal spaced grid in R,phi,Z, which I will
            # punt on for now, could be in a separate test
            second_deriv = False
        else:
            deriv = split[-1]
            if len(deriv) > 1:
                if deriv[0] != deriv[1]:  # don't do this loop
                    continue
                else:
                    second_deriv = True
                    deriv = deriv[0]
        print(key)
        grid = grids[deriv]
        data = eq.compute([key, base_quant, "phi"], grid=grid)
        data[key] = (
            rpz2xyz_vec(data[key], phi=data["phi"])
            .reshape((grid.num_theta, grid.num_rho, grid.num_zeta, -1), order="F")
            .squeeze()
        )
        data[base_quant] = (
            rpz2xyz_vec(data[base_quant], phi=data["phi"])
            .reshape((grid.num_theta, grid.num_rho, grid.num_zeta, -1), order="F")
            .squeeze()
        )

        spacing = {
            "r": grid.spacing[0, 0],
            "t": grid.spacing[0, 1],
            "z": grid.spacing[0, 2] / grid.NFP,
        }
        # do one for loop for 1st derivs, one for 2nd derivs...

        dx = (
            np.apply_along_axis(my_convolve, 0, data[base_quant], FD_COEF_1_2)
            / spacing[deriv]
        ).squeeze()
        if second_deriv:  # is a 2nd deriv like rr,tt,zz, apply again
            dx = (
                np.apply_along_axis(my_convolve, 0, data[base_quant], FD_COEF_2_4)
                / spacing[deriv] ** 2
            ).squeeze()

        if "^theta" in key and "r" in deriv:
            # this vector goes to infinity at the magnetic axis
            # so dont compare to finite differences too close to the axis
            compare_data_compute = data[key][75:4]
            compare_data_FD = dx[75:4]
        else:
            compare_data_compute = data[key][4:-4]
            compare_data_FD = dx[4:-4]

        np.testing.assert_allclose(
            compare_data_compute,
            compare_data_FD,
            rtol=1e-5,
            atol=atol,
            err_msg=key,
        )

    # second derivatives, mixed
    for key in keys[3:]:
        split = key.split("_")
        base_quant = split[0]

        if len(split) == 1:  # stuff like e^rho, e^theta
            pass
        else:
            deriv = split[-1]
            if len(deriv) == 1:
                continue  # don't do this loop for single derivs
            elif deriv[0] == deriv[1]:
                continue  # dont do this for rr,tt,zz
        print(key)
        grid = grids[deriv]
        data = eq.compute([key, base_quant, "phi"], grid=grid)
        data[key] = rpz2xyz_vec(data[key].squeeze(), phi=data["phi"]).squeeze()
        data[base_quant] = rpz2xyz_vec(
            data[base_quant].squeeze(), phi=data["phi"]
        ).squeeze()

        spacing = {
            "r": grid.spacing[0, 0],
            "t": grid.spacing[0, 1],
            "z": grid.spacing[0, 2] / grid.NFP,
        }

        shapes = {
            "tz": (grid.num_zeta, grid.num_theta),
            "rz": (grid.num_zeta, grid.num_rho),
            "rt": (grid.num_rho, grid.num_theta),
        }
        dx = (
            np.apply_along_axis(
                myconvolve_2d, 0, data[base_quant], FD_COEF_1_4, shapes[deriv]
            )
            / spacing[deriv[0]]
            / spacing[deriv[1]]
        )

        compare_data_compute = data[key].reshape(shapes[deriv] + (3,))

        if "^theta" in key and "r" in deriv:
            # this vector goes to infinity at the magnetic axis
            # so dont compare to finite differences too close to the axis
            compare_data_compute = compare_data_compute[75:4, 4:-4]
            compare_data_FD = dx[75:4, 4:-4]
        else:
            compare_data_compute = compare_data_compute[4:-4, 4:-4]
            compare_data_FD = dx[4:-4, 4:-4]

        np.testing.assert_allclose(
            compare_data_compute,
            compare_data_FD,
            rtol=1e-5,
            atol=atol,
            err_msg=key,
        )


@pytest.mark.unit
def test_iota_components():
    """Test that iota components are computed correctly."""
    # axisymmetric, so all rotational transform should be from the current
    eq_i = get("DSHAPE")  # iota profile assigned
    eq_c = get("DSHAPE_CURRENT")  # current profile assigned
    grid = LinearGrid(L=100, M=max(eq_i.M_grid, eq_c.M_grid), N=0, NFP=1, axis=True)
    data_i = eq_i.compute(["iota", "iota current", "iota vacuum"], grid)
    data_c = eq_c.compute(["iota", "iota current", "iota vacuum"], grid)
    np.testing.assert_allclose(data_i["iota"], data_i["iota current"])
    np.testing.assert_allclose(data_c["iota"], data_c["iota current"])
    np.testing.assert_allclose(data_i["iota vacuum"], 0)
    np.testing.assert_allclose(data_c["iota vacuum"], 0)

    # vacuum stellarator, so all rotational transform should be from the external field
    eq = get("ESTELL")
    grid = LinearGrid(L=100, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, axis=True)
    data = eq.compute(["iota", "iota current", "iota vacuum"], grid)
    np.testing.assert_allclose(data["iota"], data["iota vacuum"])
    np.testing.assert_allclose(data["iota current"], 0)


@pytest.mark.unit
def test_surface_equilibrium_geometry():
    """Test that computing stuff from surface gives same result as equilibrium."""
    names = ["HELIOTRON"]
    # TODO (#1397): expand this to include all angular derivatives
    #  once they are implemented for surfaces
    data_basis_vecs_fourierRZ = [
        "e_theta",
        "e_zeta",
        "e_theta_t",
        "e_theta_z",
        "e_zeta_t",
        "e_zeta_z",
    ]
    data_basis_vecs_ZernikeRZ = [
        "e_theta",
        "e_rho",
        "e_rho_r",
        "e_rho_rr",
        "e_rho_t",
        "e_theta_r",
        "e_theta_rr",
        "e_theta_t",
    ]
    for name in names:
        eq = get(name)
        # compare at rho=1, where we expect the eq.compute and the
        # surface.compute to agree for these surface basis vectors
        grid = LinearGrid(rho=np.array(1.0), M=10, N=10, NFP=eq.NFP)
        data_eq = eq.compute(data_basis_vecs_fourierRZ, grid=grid)
        data_surf = eq.surface.compute(
            data_basis_vecs_fourierRZ, grid=grid, basis="rpz"
        )
        for thing in data_basis_vecs_fourierRZ:
            np.testing.assert_allclose(
                data_eq[thing],
                data_surf[thing],
                err_msg=thing,
                rtol=1e-14,
                atol=6e-12,
            )
        # compare at zeta=0, where we expect the eq.compute and the
        # poincare surface.compute to agree for these surface basis vectors
        grid = LinearGrid(zeta=np.array(0.0), M=10, L=10, NFP=eq.NFP)
        data_eq = eq.compute(data_basis_vecs_ZernikeRZ, grid=grid)
        data_surf = eq.get_surface_at(zeta=0.0).compute(
            data_basis_vecs_ZernikeRZ, grid=grid, basis="rpz"
        )
        for thing in data_basis_vecs_ZernikeRZ:
            np.testing.assert_allclose(
                data_eq[thing],
                data_surf[thing],
                err_msg=thing,
                rtol=3e-13,
                atol=1e-13,
            )


@pytest.mark.unit
def test_clebsch_sfl_funs():
    """Test geometric and physical methods of computing B agree."""

    def test(eq):
        with pytest.warns(UserWarning, match="Reducing radial"):
            eq.change_resolution(2, 2, 2, 4, 4, 4)
        data = eq.compute(
            [
                "e_zeta|r,a",
                "B",
                "B^zeta",
                "B^phi",
                "|B|_z|r,a",
                "grad(|B|)",
                "|e_zeta|r,a|_z|r,a",
                "B^zeta_z|r,a",
                "|B|",
                "sqrt(g)_Clebsch",
                "sqrt(g)_PEST",
                "psi_r",
                "grad(psi)",
                "grad(alpha)",
                "grad(phi)",
                "B_phi",
                "gbdrift (secular)",
                "gbdrift (secular)/phi",
                "phi",
            ],
        )
        np.testing.assert_allclose(data["e_zeta|r,a"], (data["B"].T / data["B^zeta"]).T)
        np.testing.assert_allclose(
            data["|B|_z|r,a"], dot(data["grad(|B|)"], data["e_zeta|r,a"])
        )
        np.testing.assert_allclose(
            data["|e_zeta|r,a|_z|r,a"],
            data["|B|_z|r,a"] / np.abs(data["B^zeta"])
            - data["|B|"]
            * data["B^zeta_z|r,a"]
            * np.sign(data["B^zeta"])
            / data["B^zeta"] ** 2,
        )
        np.testing.assert_allclose(
            data["B"], cross(data["grad(psi)"], data["grad(alpha)"])
        )
        np.testing.assert_allclose(
            data["B^zeta"], data["psi_r"] / data["sqrt(g)_Clebsch"]
        )
        np.testing.assert_allclose(data["B^phi"], data["psi_r"] / data["sqrt(g)_PEST"])
        np.testing.assert_allclose(data["B^phi"], dot(data["B"], data["grad(phi)"]))
        np.testing.assert_allclose(data["B_phi"], data["B"][:, 1])
        np.testing.assert_allclose(
            data["gbdrift (secular)"], data["gbdrift (secular)/phi"] * data["phi"]
        )

    test(get("W7-X"))
    test(get("NCSX"))


@pytest.mark.unit
def test_parallel_grad_fd(DummyStellarator):
    """Test that the parallel gradients match with numerical gradients."""
    eq = load(load_from=str(DummyStellarator["output_path"]), file_format="hdf5")
    grid = get_rtz_grid(
        eq,
        0.5,
        0,
        np.linspace(0, 2 * np.pi, 50),
        coordinates="raz",
        period=(np.inf, 2 * np.pi, np.inf),
    )
    data = eq.compute(
        [
            "|B|",
            "|B|_z|r,a",
            "|e_zeta|r,a|",
            "|e_zeta|r,a|_z|r,a",
            "B^zeta",
            "B^zeta_z|r,a",
        ],
        grid=grid,
    )
    dz = grid.source_grid.spacing[:, 2]
    fd = np.convolve(data["|B|"], FD_COEF_1_4, "same") / dz
    np.testing.assert_allclose(
        data["|B|_z|r,a"][2:-2],
        fd[2:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(data["|B|_z|r,a"])),
    )
    fd = np.convolve(data["|e_zeta|r,a|"], FD_COEF_1_4, "same") / dz
    np.testing.assert_allclose(
        data["|e_zeta|r,a|_z|r,a"][2:-2],
        fd[2:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(data["|e_zeta|r,a|_z|r,a"])),
    )
    fd = np.convolve(data["B^zeta"], FD_COEF_1_4, "same") / dz
    np.testing.assert_allclose(
        data["B^zeta_z|r,a"][2:-2],
        fd[2:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(data["B^zeta_z|r,a"])),
    )


@pytest.mark.unit
def test_compute_deprecation_warning():
    """Test DeprecationWarning for deprecated compute names."""
    eq = Equilibrium()
    grid = LinearGrid(L=1, M=2, N=2, NFP=eq.NFP)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        eq.compute("sqrt(g)_B", grid=grid)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        eq.compute("|B|_mn", grid=grid, M_booz=eq.M, N_booz=eq.N)
