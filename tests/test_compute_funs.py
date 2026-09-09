"""Tests for compute functions."""

import numpy as np
import pytest
from scipy.signal import convolve2d
from tests.test_magnetic_fields import make_constructed_qi_samples
from tests.utils import FiniteDiffDerivative

from desc.backend import jax, jit, jnp
from desc.basis import DoubleFourierSeries
from desc.compute import compute, data_index, get_params, get_transforms
from desc.compute._omnigenity import (
    _construct_field,
    _construct_single_well,
    _evaluate_bounce,
    _evaluate_field,
    _evaluate_field_native,
    _project_centers,
    _sample_boozer_data,
    _single_well_bounce_points,
)
from desc.compute.utils import _compute
from desc.equilibrium import Equilibrium
from desc.equilibrium.coords import get_rtz_grid, map_coordinates
from desc.examples import get
from desc.geometry import FourierRZToroidalSurface
from desc.grid import Grid, LinearGrid
from desc.io import load
from desc.magnetic_fields import OmnigenousFieldConstructed
from desc.transform import Transform
from desc.utils import cross, dot, rpz2xyz_vec

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
    np.testing.assert_allclose(0, data["V_rrr(r)"], atol=3e-14)


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
    data1 = surf1.compute(["a_major/a_minor"], grid=grid)
    data2 = surf2.compute(["a_major/a_minor"], grid=grid)
    data3 = surf3.compute(["a_major/a_minor"], grid=grid)
    # elongation approximation is less accurate as elongation increases
    np.testing.assert_allclose(1.0, data1["a_major/a_minor"])
    np.testing.assert_allclose(2.0, data2["a_major/a_minor"], rtol=1e-4)
    np.testing.assert_allclose(3.0, data3["a_major/a_minor"], rtol=1e-3)


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


def _asymmetric_samples():
    """Avoid extrema ties and root/clip switches for derivative comparisons."""
    return jnp.array(
        [
            [
                [2.2, 2.6, 1.9, 1.45, 1.0, 1.3, 1.8, 2.4, 3.2],
                [3.0, 2.5, 1.8, 1.1, 1.4, 1.9, 2.1, 2.7, 3.6],
                [2.9, 2.3, 1.9, 1.7, 1.4, 1.15, 1.8, 2.5, 3.3],
            ]
        ]
    )


def _tied_symmetric_samples():
    """Separated minima and off-grid midpoints, including reflection partners."""
    period = 2 * np.pi / 3
    first = [3.0, 2.4, 1.6, 1.0, 1.7, 1.7, 1.0, 1.6, 2.4, 3.0]
    second = [3.2, 2.5, 1.8, 1.1, 1.6, 1.1, 1.9, 2.3, 2.8, 3.6]
    third = [3.4, 2.6, 1.0, 1.5, 1.8, 1.8, 1.5, 1.0, 2.6, 3.4]
    B = np.asarray([[first, second, third, second[::-1]]])
    # Roundoff-scale splitting must not select one of a reflected pair of minima.
    B[0, 0, 3] = np.nextafter(B[0, 0, 3], np.inf)
    return B, dict(
        rho=[0.7],
        fieldline_labels=np.arange(4) * np.pi / 2,
        zeta=-2 * period + period * np.linspace(0, 1, B.shape[-1]),
        iota=[0.73],
        NFP=3,
        sym=True,
        num_B_levels=7,
    )


class TestConstructedQI:
    """Test the array construction and continuous representation of QI fields."""

    @pytest.mark.unit
    def test_constructed_registry(self):
        """Dispatch both strengths with surface ordering and dynamic iota intact."""
        parameterization = "desc.magnetic_fields._core.OmnigenousFieldConstructed"
        entries = data_index[parameterization]
        assert set(entries) == {"|B| constructed", "Bc normalized"}
        B, options = make_constructed_qi_samples()
        samples = np.concatenate((B, 1.3 * B + 0.4))
        field = OmnigenousFieldConstructed.from_samples(
            samples, **dict(options, rho=[0.4, 0.7], iota=[0.6, 0.73])
        )
        # Interleaved stored surfaces and shifted angles preserve input node order.
        grid = Grid(
            [[0.7, 0.1, -0.2], [0.4, 2.3, 0.4], [0.7, 3.4, 2.5]],
            NFP=field.NFP,
            sort=False,
        )
        names = list(entries)
        params = get_params(names, field)
        transforms = get_transforms(names, field, grid=grid)
        actual = compute(field, names, params, transforms, {})
        expected = field.compute(names, grid=grid)
        assert set(actual) == set(expected) == set(names)
        for name in names:
            np.testing.assert_allclose(actual[name], expected[name], atol=2e-14)
        indices = np.array([1, 0, 1])
        np.testing.assert_allclose(
            actual["|B| constructed"],
            field.B_min[indices]
            + (field.B_max - field.B_min)[indices] * actual["Bc normalized"],
            atol=2e-14,
        )

        # Literal query angles stay fixed while iota changes their stored label.
        def evaluate(iota):
            return _compute(
                field, "Bc normalized", dict(params, iota=iota), transforms, {}
            )["Bc normalized"]

        direction = jnp.array([0.17, -0.21])
        _, tangent = jit(lambda iota, diota: jax.jvp(evaluate, (iota,), (diota,)))(
            field.iota, direction
        )
        step = 1e-6
        expected_tangent = (
            evaluate(field.iota + step * direction)
            - evaluate(field.iota - step * direction)
        ) / (2 * step)
        assert np.linalg.norm(tangent) > 0
        np.testing.assert_allclose(tangent, expected_tangent, rtol=2e-7, atol=2e-9)

    @pytest.mark.unit
    def test_constructed_registry_invalid_surface_derivatives(self):
        """Invalid normalized values and ranges cannot poison valid-surface AD."""
        parameterization = "desc.magnetic_fields._core.OmnigenousFieldConstructed"
        grid = Grid([[0.7, 0.2, 0.3], [0.4, 0.4, 0.6]], sort=False)
        b = jnp.array([jnp.nan, 0.4])
        bmin, bmax = jnp.array([2.0, jnp.nan]), jnp.array([3.0, jnp.nan])

        def evaluate(normalized, minimum, maximum):
            params = dict(
                rho=jnp.array([0.4, 0.7]),
                zeta=jnp.array([0.0, 2 * jnp.pi]),
                B_min=minimum,
                B_max=maximum,
                valid_surface=jnp.array([True, False]),
            )
            return _compute(
                parameterization,
                "|B| constructed",
                params,
                {"grid": grid},
                {},
                data={"Bc normalized": normalized},
            )["|B| constructed"]

        value, tangent = jit(
            lambda x, lo, hi: jax.jvp(
                evaluate, (x, lo, hi), (jnp.ones(2), jnp.ones(2), jnp.ones(2))
            )
        )(b, bmin, bmax)
        assert np.isnan(value[0])
        np.testing.assert_allclose(value[1], 2.4)
        np.testing.assert_allclose(tangent[1], 2.0)
        gradients = jit(
            lambda x, lo, hi: jax.vjp(evaluate, x, lo, hi)[1](jnp.array([0.0, 1.0]))
        )(b, bmin, bmax)
        for actual, expected in zip(gradients, ([0, 1], [0.6, 0], [0.4, 0])):
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, atol=2e-14)

    @pytest.mark.unit
    def test_native_field_triangular_oracle(self):
        """Stored inverse branches recover triangular wells and their derivatives."""
        zeta = jnp.linspace(-0.37, -0.37 + 2 * np.pi / 3, 13)
        period = zeta[-1] - zeta[0]
        levels = jnp.array([0.0, 0.2, 0.65, 1.0])
        minima = zeta[0] + period * jnp.array([[0.31, 0.47, 0.61], [0.39, 0.57, 0.73]])
        direction = jnp.array([[0.13, -0.07, 0.03], [-0.11, 0.09, 0.05]])

        def evaluate(bottom):
            data = dict(
                zeta=zeta,
                B_levels=levels,
                bounce_centers=bottom[..., None] * (1 - levels)
                + (zeta[0] + zeta[-1]) / 2 * levels,
                bounce_distances=jnp.broadcast_to(period * levels, (2, levels.size)),
                valid_knots=jnp.ones(bottom.shape, dtype=bool),
            )
            return _evaluate_field_native(data)

        actual, tangent = jax.jvp(evaluate, (minima,), (direction,))
        bottom = minima[..., None]
        expected = jnp.where(
            zeta < bottom,
            (bottom - zeta) / (bottom - zeta[0]),
            (zeta - bottom) / (zeta[-1] - bottom),
        )
        derivative = (
            jnp.where(
                zeta < bottom,
                (zeta - zeta[0]) / (bottom - zeta[0]) ** 2,
                (zeta - zeta[-1]) / (zeta[-1] - bottom) ** 2,
            )
            * direction[..., None]
        )
        np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=3e-14)
        np.testing.assert_allclose(tangent, derivative, atol=3e-13, rtol=3e-13)
        np.testing.assert_array_equal(actual[..., [0, -1]], 1.0)
        # Interpolation's quotient JVP can leave roundoff when endpoint terms cancel.
        np.testing.assert_allclose(tangent[..., [0, -1]], 0.0, atol=1e-15, rtol=0)

    @pytest.mark.unit
    def test_shared_knot_derivatives(self):
        """A smooth change of a QI well has zero residual derivative at shared knots."""
        zeta = jnp.linspace(0, 1, 9)
        u = jnp.abs(2 * zeta - 1)
        B = (2 + u)[None, None, :]
        direction = (u * (1 - u))[None, None, :]

        def residual(t):
            data = _construct_field(B + t * direction, zeta, jnp.linspace(0, 1, 5))
            return (data["B_normalized"] - data["B_target"]).ravel()

        value, tangent = jax.jvp(residual, (0.0,), (1.0,))
        np.testing.assert_allclose(value, 0, atol=1e-14)
        np.testing.assert_allclose(tangent, 0, atol=1e-13)
        # The inverse interpolant agrees to first order; its remaining error is O(t²).
        for step in (2e-6, 1e-6):
            finite_difference = (residual(step) - residual(-step)) / (2 * step)
            np.testing.assert_allclose(tangent, finite_difference, atol=3e-7)

    @pytest.mark.unit
    def test_single_well(self):
        """The base operation aligns minima without imposing QI top constraints."""
        B = np.asarray(_asymmetric_samples())
        result = _construct_single_well(B)
        expected = np.empty_like(B)
        indices = np.argmin(B, axis=-1)
        for a, index in enumerate(indices[0]):
            left = np.minimum.accumulate(B[0, a, : index + 1])
            right = np.minimum.accumulate(B[0, a, index:][::-1])[::-1]
            expected[0, a] = np.r_[left[:-1], right] - B[0, a, index] + B.min()
        np.testing.assert_allclose(result["B_single"], expected, atol=2e-15)
        np.testing.assert_allclose(np.min(result["B_single"], axis=-1), B.min())
        np.testing.assert_allclose(result["endpoint_B"], expected[..., [0, -1]])
        assert np.all(result["valid_single_well"])
        assert np.unique(np.asarray(result["endpoint_B"])).size > 2
        # A later interior maximum must not replace the left endpoint.
        assert result["B_single"][0, 0, 0] == B[0, 0, 0]
        for a, index in enumerate(indices[0]):
            assert np.all(np.diff(expected[0, a, : index + 1]) <= 0)
            assert np.all(np.diff(expected[0, a, index:]) >= 0)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "values,interval",
        [
            ([3, 2.2, 1.2, 1, 1, 1.4, 2.4, 3.4], (3, 4)),
            ([3, 2.4, 1, 1.8, 1.9, 1.6, 1, 2.7, 3.4], (2, 6)),
            ([3, 2.4, 1, 1, 1.8, 1.7, 1, 2.2, 2.8, 3.4], (2, 6)),
            ([3, 2.4, np.nextafter(1.0, np.inf), 1.7, 1.8, 1, 2.2, 2.8, 3.4], (2, 5)),
        ],
    )
    def test_minimum_midpoint(self, values, interval):
        """Adjacent or separated minima define a continuous, unrounded anchor."""
        B = np.asarray(values)[None, None]
        zeta = np.linspace(-2 * np.pi, 0, B.shape[-1])
        data = _construct_field(B, zeta, jnp.linspace(0, 1, 7))
        lo, hi = interval
        np.testing.assert_array_equal(data["minimum_indices"], [[interval]])
        np.testing.assert_array_equal(data["B_single"][0, 0, lo : hi + 1], B.min())
        np.testing.assert_array_equal(data["endpoint_B"], B[..., [0, -1]])
        midpoint = (zeta[lo] + zeta[hi]) / 2
        np.testing.assert_allclose(data["minimum_zeta"], midpoint, atol=1e-15)
        np.testing.assert_allclose(data["bounce_centers"][..., 0], midpoint, atol=1e-15)
        np.testing.assert_array_equal(data["bounce_distances"][..., 0], 0)
        assert np.all(data["valid_surface"])
        # The base layer retains the plateau; the QI enhancement contracts level zero.
        assert hi > lo
        at_minimum = _evaluate_field(data, jnp.array([0.0]), 0.0, midpoint, 0)
        np.testing.assert_allclose(at_minimum, 0, atol=2e-14)
        left, right = _evaluate_bounce(data, jnp.array([[0.0]]))
        np.testing.assert_allclose(left, midpoint, atol=1e-15)
        np.testing.assert_array_equal(right, left)

    @pytest.mark.unit
    def test_tied_minimum_symmetry(self):
        """Keep reflection symmetry and common widths between construction levels."""
        B, options = _tied_symmetric_samples()
        levels = jnp.linspace(0, 1, options.pop("num_B_levels"))
        data = _construct_field(B, levels=levels, **options)
        assert np.all(data["valid_surface"])
        actual = data["B_target"]
        reflected = actual[:, (-np.arange(B.shape[1])) % B.shape[1], ::-1]
        np.testing.assert_allclose(actual, reflected, atol=3e-14)
        beta = jnp.array([[0, 0.0003, 0.045, 0.327, 0.913, 1]])
        left, right = _evaluate_bounce(data, beta)
        np.testing.assert_allclose(
            right - left,
            np.broadcast_to((right - left)[:, :1], left.shape),
            atol=3e-14,
        )
        labels = jnp.asarray(options["fieldline_labels"])
        for roots in (left[0], right[0]):
            at_roots = _evaluate_field(data, labels, labels[:, None], roots, 0)
            np.testing.assert_allclose(
                at_roots, np.broadcast_to(beta, roots.shape), atol=3e-14
            )
        anchors = data["bounce_centers"][0, :, 0]
        np.testing.assert_allclose(
            _evaluate_field(data, labels, labels, anchors, 0), 0, atol=3e-14
        )

    @pytest.mark.unit
    def test_affine_stretch(self):
        """Minimum translation cancels from direct per-branch affine stretch."""
        B = _asymmetric_samples()
        indices = np.argmin(np.asarray(B[0]), axis=-1)
        zeta, levels = jnp.linspace(-2.0, -1.0, 9), jnp.linspace(0, 1, 7)

        def direct_stretch(B):
            rows = []
            for a, index in enumerate(indices):
                row = B[0, a]
                squashed = jnp.stack(
                    [
                        jnp.min(row[: k + 1]) if k <= index else jnp.min(row[k:])
                        for k in range(row.size)
                    ]
                )
                minimum = squashed[index]
                endpoint = jnp.where(jnp.arange(row.size) <= index, row[0], row[-1])
                rows.append((squashed - minimum) / (endpoint - minimum))
            return jnp.stack(rows)[None]

        def separated(B):
            return _construct_field(B, zeta, levels)["B_stretched"]

        direction = jnp.sin(jnp.arange(B.size).reshape(B.shape) + 0.2)
        expected, dexp = jax.jvp(direct_stretch, (B,), (direction,))
        actual, dactual = jax.jvp(separated, (B,), (direction,))
        np.testing.assert_allclose(actual, expected, atol=3e-15)
        np.testing.assert_allclose(dactual, dexp, atol=3e-14)
        data = _construct_field(B, zeta, levels)
        error = np.mean((np.asarray(data["B_normalized"]) - expected) ** 2, axis=-1)
        inverse = 1 / (error + 1e-12)
        np.testing.assert_allclose(
            data["weights"], inverse / inverse.sum(axis=-1)[:, None]
        )

    @pytest.mark.unit
    def test_center_projection_preserves_width(self):
        """Keep a feasible shared width of 0.31 for asymmetric inverse branches."""
        left = np.array([[0.5, 0.49, 0.1, 0.05, 0], [0.5, 0.45, 0.445, 0.44, 0]])
        right = np.array([[0.5, 0.6, 0.61, 0.65, 1], [0.5, 0.55, 0.555, 0.56, 1]])
        raw = (left + right)[None] / 2
        width = np.mean(right - left, axis=0)[None]
        centers, valid = _project_centers(raw, width, jnp.array([0.0, 1.0]), 1e-12)
        assert np.all(valid)
        lc, rc = centers - width[:, None] / 2, centers + width[:, None] / 2
        np.testing.assert_allclose(rc - lc, np.broadcast_to(width[:, None], raw.shape))
        np.testing.assert_allclose((rc - lc)[0, :, 2], 0.31)
        np.testing.assert_allclose(lc[..., -1], 0, atol=1e-15)
        np.testing.assert_allclose(rc[..., -1], 1)
        np.testing.assert_array_equal(centers[..., 0], raw[..., 0])
        assert np.all(np.diff(lc, axis=-1) < 0)
        assert np.all(np.diff(rc, axis=-1) > 0)
        reflected, reflection_valid = _project_centers(
            1 - raw, width, jnp.array([0.0, 1.0]), 1e-12
        )
        assert np.all(reflection_valid)
        np.testing.assert_allclose(centers + reflected, 1, atol=1e-15)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "raw,width",
        [([[1e-12, 0.5, 0.5]], [0, 0.5, 1]), ([[0.5, 0.5, 0.5]], [0, 1e-12, 1])],
    )
    def test_infeasible_center_projection(self, raw, width):
        """Neither a boundary minimum nor insufficient width growth is repaired."""
        _, valid = _project_centers(
            jnp.asarray(raw)[None],
            jnp.asarray(width)[None],
            jnp.array([0.0, 1.0]),
            1e-12,
        )
        assert not np.all(valid)

    @pytest.mark.unit
    def test_triangular_fixed_point(self):
        """Preserve triangular wells with analytic roots at new labels and levels."""
        B, options = make_constructed_qi_samples()
        levels = jnp.linspace(0, 1, options["num_B_levels"])
        data = _construct_field(B, options["zeta"], levels)
        assert np.all(data["valid_surface"])
        span = data["B_max"] - data["B_min"]
        np.testing.assert_allclose(data["B_min"], 2)
        np.testing.assert_allclose(data["B_max"], 3)
        np.testing.assert_allclose(
            data["B_min"][:, None, None] + span[:, None, None] * data["B_target"],
            B,
            atol=3e-14,
        )
        beta = jnp.array([[0, 0.037, 0.29, 0.523, 0.918, 1]])
        left, right = _evaluate_bounce(data, beta)
        period = 2 * np.pi / options["NFP"]
        centers = 0.5 + 0.125 * np.sin(options["fieldline_labels"])
        np.testing.assert_allclose(
            left[0], period * centers[:, None] * (1 - beta), atol=3e-14
        )
        np.testing.assert_allclose(
            right[0],
            period * (centers[:, None] + (1 - centers[:, None]) * beta),
            atol=3e-14,
        )
        np.testing.assert_allclose(
            right - left, np.broadcast_to(period * beta, left.shape), atol=3e-14
        )

        # A new label interpolates the analytic centers between two sampled lines.
        chi = np.pi / 4
        center0 = 0.5 + 0.125 / 2
        center = period * (center0 + (0.5 - center0) * beta[0])
        physical = 2 + beta[0]
        for roots in (center - period * beta[0] / 2, center + period * beta[0] / 2):
            actual = _evaluate_field(
                data, jnp.asarray(options["fieldline_labels"]), chi, roots, 0
            )
            np.testing.assert_allclose(actual, beta[0], atol=3e-14)
            np.testing.assert_allclose(data["B_min"][0] + span[0] * actual, physical)

    @pytest.mark.unit
    @pytest.mark.parametrize("scale", [1e-8, 1e8])
    def test_field_scale(self, scale):
        """Keep tied-minimum classification and normalized fields across unit scales."""
        B, options = _tied_symmetric_samples()
        levels = jnp.linspace(0, 1, options.pop("num_B_levels"))
        data = _construct_field(B, levels=levels, **options)
        scaled = _construct_field(scale * B, levels=levels, **options)
        assert np.all(data["valid_surface"])
        assert np.all(scaled["valid_surface"])
        np.testing.assert_array_equal(
            scaled["minimum_indices"], data["minimum_indices"]
        )
        for name in ("B_target", "bounce_centers", "bounce_distances"):
            np.testing.assert_allclose(scaled[name], data[name], atol=1e-13)
        for name in ("B_min", "B_max"):
            np.testing.assert_allclose(scaled[name], scale * data[name])

    @staticmethod
    def _check_derivatives(B, direction):
        """Compare AD and finite differences while retaining the active segments."""
        zeta, levels = jnp.linspace(-2.0, -1.0, B.shape[-1]), jnp.linspace(0, 1, 7)

        def residual(values):
            data = _construct_field(values, zeta, levels)
            return (data["B_normalized"] - data["B_target"]).ravel()

        def active_segments(data):
            # Interior level signs identify the source piecewise-linear root segments.
            source = data["B_stretched"][..., None] < levels[1:-1]
            left = data["bounce_centers"] - data["bounce_distances"][:, None] / 2
            right = data["bounce_centers"] + data["bounce_distances"][:, None] / 2
            knots = jnp.concatenate((left[..., ::-1], right[..., 1:]), axis=-1)
            target = jnp.sum(zeta[:, None] >= knots[..., None, :], axis=-1)
            return source, target

        baseline = _construct_field(B, zeta, levels)
        assert np.all(baseline["valid_surface"])
        segments = active_segments(baseline)
        actual, tangent = jax.jvp(jit(residual), (B,), (direction,))
        assert np.linalg.norm(actual) > 0
        assert np.linalg.norm(tangent) > 0
        # Differentiate a scalar path so rel_step means exactly B +/- step * direction.
        # Both steps are small relative to the O(1) physical field samples.
        for step in (2e-6, 1e-6):
            for sign in (-1, 1):
                trial = _construct_field(B + sign * step * direction, zeta, levels)
                assert np.all(trial["valid_surface"])
                np.testing.assert_array_equal(
                    trial["minimum_indices"], baseline["minimum_indices"]
                )
                for trial_segments, expected in zip(active_segments(trial), segments):
                    np.testing.assert_array_equal(trial_segments, expected)
            finite_difference = FiniteDiffDerivative.compute_jvp(
                lambda t: residual(B + t * direction), 0, 1.0, 0.0, rel_step=step
            )
            np.testing.assert_allclose(tangent, finite_difference, rtol=2e-5, atol=2e-7)
        cotangent = jnp.sin(jnp.arange(actual.size) + 0.19)
        _, pullback = jax.vjp(residual, B)
        gradient = pullback(cotangent)[0]
        assert np.isfinite(gradient).all()
        np.testing.assert_allclose(
            jnp.vdot(cotangent, tangent),
            jnp.vdot(gradient, direction),
            rtol=2e-12,
            atol=2e-12,
        )
        np.testing.assert_allclose(actual, residual(B), atol=3e-14)

    @pytest.mark.unit
    def test_derivatives(self):
        """Differentiate extrema, stretch, roots, weights and reconstructed values."""
        B = _asymmetric_samples()
        direction = jnp.cos(jnp.arange(B.size).reshape(B.shape) + 0.37)
        self._check_derivatives(B, direction)

    @pytest.mark.unit
    def test_tied_minimum_derivatives(self):
        """Differentiate along tied minimum sets without splitting or merging them."""
        B = _asymmetric_samples()
        tied_indices = ((2, 4), (3, 5), (3, 5))
        for line, indices in enumerate(tied_indices):
            B = B.at[0, line, jnp.asarray(indices)].set(jnp.min(B[0, line]))
        direction = jnp.cos(jnp.arange(B.size).reshape(B.shape) + 0.37)
        for line, indices in enumerate(tied_indices):
            direction = direction.at[0, line, jnp.asarray(indices)].set(
                0.17 + 0.23 * line
            )
        data = _construct_field(
            B, jnp.linspace(-2, -1, B.shape[-1]), jnp.linspace(0, 1, 7)
        )
        np.testing.assert_array_equal(data["minimum_indices"], [tied_indices])
        # The direction changes the physical minimum while staying tangent to each tie.
        _, minimum_derivative = jax.jvp(jnp.min, (B,), (direction,))
        np.testing.assert_allclose(minimum_derivative, 0.17)
        self._check_derivatives(B, direction)

    @pytest.mark.unit
    def test_batching(self):
        """Independent surfaces and DESC batch chunk sizes define the same field."""
        B = _asymmetric_samples()
        zeta, levels = jnp.linspace(-2.0, -1.0, 9), jnp.linspace(0, 1, 7)
        together = jnp.concatenate([B, 2.3 * B + 0.4], axis=0)
        unchunked = _construct_field(together, zeta, levels, surf_batch_size=None)
        chunked = _construct_field(
            together, zeta, levels, fieldline_batch_size=2, surf_batch_size=1
        )
        surfaces = [_construct_field(row[None], zeta, levels) for row in together]
        for name in (
            "B_target",
            "bounce_centers",
            "bounce_distances",
            "weights",
            "valid_surface",
        ):
            separate = jnp.concatenate([surface[name] for surface in surfaces])
            np.testing.assert_allclose(unchunked[name], separate, atol=3e-14)
            np.testing.assert_allclose(chunked[name], separate, atol=3e-14)

    @pytest.mark.unit
    @pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
    def test_linear_shared_knots(self, scale):
        """Analytic single-well roots survive field normalization at shared knots."""
        knots = jnp.linspace(0, 1, 9)
        centers = jnp.array([0.5, 0.625, 0.375])
        normalized = jnp.where(
            knots < centers[:, None],
            1 - knots / centers[:, None],
            (knots - centers[:, None]) / (1 - centers[:, None]),
        )
        physical = scale * (2 + normalized)
        normalized = (physical - physical.min()) / (physical.max() - physical.min())
        levels = jnp.array([0.25, 0.5, 0.75 - 1e-12, 0.75, 0.75 + 1e-12])
        left, right, valid = _single_well_bounce_points(levels, knots, normalized)
        assert np.all(valid)
        np.testing.assert_allclose(
            left, centers[:, None] * (1 - levels), rtol=2e-14, atol=2e-15
        )
        np.testing.assert_allclose(
            right,
            centers[:, None] + (1 - centers[:, None]) * levels,
            rtol=2e-14,
            atol=2e-15,
        )

    @pytest.mark.unit
    def test_linear_plateau(self):
        """Plateaus use the inner edges; the minimum has no positive-width sublevel."""
        shift = -3.0
        knots = jnp.arange(8.0) + shift
        B = jnp.array([3.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0])
        left, right, valid = _single_well_bounce_points(
            jnp.array([1.0, 2.0, 3.0]), knots, B
        )
        np.testing.assert_array_equal(valid, [False, True, True])
        np.testing.assert_array_equal(left[1:], jnp.array([2.0, 0.0]) + shift)
        np.testing.assert_array_equal(right[1:], jnp.array([5.0, 7.0]) + shift)
        np.testing.assert_array_equal(left[~valid], 0.0)
        np.testing.assert_array_equal(right[~valid], 0.0)

    @pytest.mark.unit
    def test_linear_derivatives(self):
        """Simple roots retain analytic derivatives beside inactive flat segments."""
        knots = jnp.arange(5.0)
        B = jnp.array([3.0, 1.0, 1.0, 1.0, 4.0])
        direction = jnp.array([0.4, -0.2, 0.1, 0.3, -0.5])
        level = jnp.array([2.1])

        def roots(values):
            left, right, _ = _single_well_bounce_points(level, knots, values)
            return jnp.concatenate((left, right))

        expected = np.array([(3 - 2.1) / 2, 3 + (2.1 - 1) / 3])
        jacobian = np.zeros((2, 5))
        jacobian[0, :2] = [(2.1 - 1) / 2**2, (3 - 2.1) / 2**2]
        jacobian[1, 3:] = [(2.1 - 4) / 3**2, -(2.1 - 1) / 3**2]
        value, tangent = jax.jvp(roots, (B,), (direction,))
        np.testing.assert_allclose(value, expected, atol=1e-14)
        np.testing.assert_allclose(tangent, jacobian @ direction, atol=1e-14)
        cotangent = jnp.array([0.7, -0.2])
        _, pullback = jax.vjp(roots, B)
        gradient = pullback(cotangent)[0]
        assert np.isfinite(gradient).all()
        np.testing.assert_allclose(gradient, cotangent @ jacobian, atol=1e-14)
        for step in (1e-5, 5e-6):
            for sign in (-1, 1):
                np.testing.assert_array_equal(
                    jnp.floor(roots(B + sign * step * direction)), [0, 3]
                )
            finite_difference = (
                roots(B + step * direction) - roots(B - step * direction)
            ) / (2 * step)
            np.testing.assert_allclose(
                tangent, finite_difference, rtol=1e-8, atol=1e-10
            )

    @pytest.mark.unit
    def test_linear_shared_knot_parameter_derivatives(self):
        """Root AD includes samples, levels and moving knots at smooth shared nodes."""
        knots = jnp.linspace(0, 1, 5)
        values = jnp.array([3.0, 2.0, 1.0, 2.0, 3.0])
        levels = jnp.array([2.0])

        @jit
        def roots(B, level, zeta):
            left, right, _ = _single_well_bounce_points(level, zeta, B)
            return jnp.concatenate((left, right))

        directions = (
            (jnp.array([0.1, -0.3, 0.2, 0.4, -0.2]), jnp.zeros(1), jnp.zeros(5)),
            (jnp.zeros(5), jnp.array([0.2]), jnp.zeros(5)),
            (jnp.zeros(5), jnp.zeros(1), jnp.array([0.02, -0.03, 0.01, 0.04, -0.02])),
        )
        for dB, dlevel, dzeta in directions:
            expected = jnp.array(
                [
                    dzeta[1] + (dB[1] - dlevel[0]) / 4,
                    dzeta[3] + (dlevel[0] - dB[3]) / 4,
                ]
            )
            _, tangent = jax.jvp(roots, (values, levels, knots), (dB, dlevel, dzeta))
            np.testing.assert_allclose(tangent, expected, atol=1e-13)
            for step in (5e-6, 2e-6):
                plus = roots(
                    values + step * dB, levels + step * dlevel, knots + step * dzeta
                )
                minus = roots(
                    values - step * dB, levels - step * dlevel, knots - step * dzeta
                )
                np.testing.assert_allclose(
                    tangent, (plus - minus) / (2 * step), atol=3e-7
                )
        cotangent = jnp.array([0.7, -0.2])
        _, pullback = jax.vjp(roots, values, levels, knots)
        dB, dlevel, dzeta = pullback(cotangent)
        np.testing.assert_allclose(dB, [0, 0.7 / 4, 0, 0.2 / 4, 0], atol=1e-13)
        np.testing.assert_allclose(dlevel, [-0.9 / 4], atol=1e-13)
        np.testing.assert_allclose(dzeta, [0, 0.7, 0, -0.2, 0], atol=1e-13)

    @pytest.mark.unit
    def test_linear_invalid_and_broadcast(self):
        """Invalid wells and levels retain finite empty pairs beside valid surfaces."""
        knots = jnp.linspace(-2, 1, 5)
        values = jnp.array([[3.0, 2.0, 1.0, 2.0, 3.0], [3.0, 1.0, 2.0, 1.0, 3.0]])[
            :, None, :
        ]
        values = jnp.broadcast_to(values, (2, 3, 5))
        levels = jnp.array([[2.0, 1.0, 4.0], [2.0, 2.5, 3.0]])[:, None, :]
        left, right, valid = jit(_single_well_bounce_points)(levels, knots, values)
        np.testing.assert_array_equal(
            valid[0], jnp.broadcast_to(jnp.array([True, False, False]), (3, 3))
        )
        assert not np.any(valid[1])
        np.testing.assert_allclose(left[0, :, 0], -1.25)
        np.testing.assert_allclose(right[0, :, 0], 0.25)
        np.testing.assert_array_equal(left[~valid], 0)
        np.testing.assert_array_equal(right[~valid], 0)

    @pytest.mark.unit
    def test_smooth_qi_field_level_convergence(self):
        """A smooth scalar QI well converges as its inverse field levels are refined."""
        zeta = jnp.linspace(0, 2 * jnp.pi, 65)
        B = (2.5 + 0.5 * jnp.cos(zeta))[None, None, :]
        errors = []
        for number in (9, 17, 33):
            data = _construct_field(B, zeta, jnp.linspace(0, 1, number))
            assert np.all(data["valid_surface"])
            errors.append(
                float(
                    jnp.sqrt(jnp.mean((data["B_normalized"] - data["B_target"]) ** 2))
                )
            )
        # This nonlinear well is not the exact triangular fixed point. Uniform
        # field levels resolve the inverse branches increasingly accurately.
        assert 0 < errors[2] < errors[1] < errors[0]
        assert errors[2] < errors[0] / 3

    @pytest.mark.unit
    def test_boozer_sampler_harmonic_oracle(self):
        """Sample known harmonics at fixed chi with the full iota derivative."""
        nfp = 3
        grid = LinearGrid(rho=[0.4, 0.8], M=3, N=2, NFP=nfp, sym=False)
        basis = DoubleFourierSeries(M=2, N=1, NFP=nfp, sym=False)
        transforms = {"grid": grid, "B": Transform(grid, basis, build=False)}
        labels = jnp.array([0.21, 1.0, 3.7])
        zeta = jnp.linspace(-0.37, -0.37 + 2 * jnp.pi / nfp, 11)
        iota = jnp.array([0.31, -0.42])
        direction = jnp.array([0.17, -0.23])
        a, b = jnp.array([0.4, -0.2]), jnp.array([0.15, 0.3])
        coefficients = jnp.zeros((2, basis.num_modes))
        # cos(2 theta - NFP zeta) and sin(theta + NFP zeta), expanded in
        # DESC's tensor-product sine/cosine basis. The oracle below is analytic.
        for m, n, amplitude in (
            (0, 0, jnp.array([2.0, 2.0])),
            (2, 1, a),
            (-2, -1, a),
            (-1, 1, b),
            (1, -1, b),
        ):
            index = np.flatnonzero(np.all(basis.modes == [0, m, n], axis=1)).item()
            coefficients = coefficients.at[:, index].set(amplitude)

        def samples(current_iota):
            data = {"|B|_mn_B": coefficients.ravel(), "iota": grid.expand(current_iota)}
            return _sample_boozer_data(transforms, data, labels, zeta)[0]

        actual, tangent = jax.jvp(jit(samples), (iota,), (direction,))
        offset = zeta - (zeta[0] + zeta[-1]) / 2
        theta = labels[None, :, None] + iota[:, None, None] * offset
        phase_cos, phase_sin = 2 * theta - nfp * zeta, theta + nfp * zeta
        expected = (
            2
            + a[:, None, None] * jnp.cos(phase_cos)
            + b[:, None, None] * jnp.sin(phase_sin)
        )
        derivative = -2 * a[:, None, None] * jnp.sin(phase_cos) + b[
            :, None, None
        ] * jnp.cos(phase_sin)
        derivative *= offset * direction[:, None, None]
        np.testing.assert_allclose(actual, expected, atol=2e-14)
        np.testing.assert_allclose(tangent, derivative, atol=2e-14)


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
@pytest.mark.parametrize("eq", [get("W7-X"), get("NCSX")])
def test_covariant_basis_vectors_PEST(eq):
    """
    Test calculation of covariant basis vectors in PEST.

    We compare the basis vectors by comparing with finite diff of the position vector
    x and lower-order covariant basis vectors.
    """
    keys_PEST = [
        "e_rho|v,p",
        "e_vartheta|r,p",
        "e_phi|r,v",
        "e_vartheta_v|PEST",
        "e_vartheta_p|PEST",
        "e_vartheta_r|PEST",
        "e_phi_r|PEST",
        "e_phi_p|PEST",
        "e_rho_r|PEST",
    ]

    N = 4000

    # spacing grids in each native direction
    grids_PEST = {
        "r": LinearGrid(L=N, M=0, N=0, NFP=eq.NFP),
        "v": LinearGrid(rho=1, M=N, N=0, NFP=eq.NFP, sym=True),
        "z": LinearGrid(rho=1, M=0, N=N, NFP=eq.NFP, sym=True),
    }

    # find native (ρ,θ,ζ) nodes that correspond to the uniform θ_PEST grid
    rtz_nodes = map_coordinates(
        eq,
        grids_PEST["v"].nodes,  # (ρ,θ_PEST,ζ)
        inbasis=("rho", "theta_PEST", "zeta"),
        outbasis=("rho", "theta", "zeta"),
        period=(np.inf, 2 * np.pi, np.inf),
    )

    # find native (ρ,θ,ζ) nodes that correspond to the uniform ζ grid
    rtz_nodes1 = map_coordinates(
        eq,
        grids_PEST["z"].nodes,  # (ρ,θ_PEST,ζ)
        inbasis=("rho", "theta_PEST", "zeta"),
        outbasis=("rho", "theta", "zeta"),
        period=(np.inf, 2 * np.pi, np.inf),
    )

    def _get_deriv(deriv_tokn):
        d0 = deriv_tokn.lower()
        if d0.startswith(("r", "rho")):
            deriv = "r"
        elif d0.startswith(("v", "vartheta", "theta")):
            deriv = "v"
        elif d0.startswith(("z", "phi", "p")):  # ζ or φ share spacing
            deriv = "z"
        else:
            raise ValueError(f"Cannot parse derivative direction from '{key}'")
        return deriv

    for key in keys_PEST:
        lhs, rhs = key.split("|")
        # rhs can be PEST or r,v or p,v or r,p etc.
        # So it's not really needed for the FD calculation
        parts = lhs.split("_")
        if len(parts) == 2:  # like "e_rho"
            deriv_tokn = parts[-1]
            base_bits = [parts[-1]]
            base = ["X", "Y", "Z"]
        else:  # like "e_rho_v"

            deriv_tokn = parts[-1]
            base_bits = parts[:-1]
            base = []

        deriv = _get_deriv(deriv_tokn)

        # Grid will have to be custom for vartheta
        grid_used = (
            Grid(rtz_nodes)
            if deriv == "v"
            else (grids_PEST[deriv] if deriv == "r" else Grid(rtz_nodes1))
        )

        # Decide base vector
        if len(base_bits) == 1:  # only happens for X,Y,Z
            base_key = None  # triggers Cartesian path
        else:
            base_key = "_".join(base_bits)  # e_rho, e_vartheta,
            deriv0 = _get_deriv(
                base_bits[-1]
            )  # get the first derivative, like rho from e_rho_vartheta
            base_key = base_key + (
                "|r,p" if deriv0 == "v" else ("|p,v" if deriv0 == "r" else "|r,v")
            )
            # adding parenthesis to the higher-order vectors
            list0 = key.split("|")
            key = "(" + list0[0] + ")" + "|" + list0[1]

        req_keys = [key, "phi"] + base + ([] if base_key is None else [base_key])
        data = eq.compute(req_keys, grid=grid_used)

        # Determine reshape dimensions based on deriv
        reshape_dims = (
            (1, 1, grid_used.num_zeta, -1)
            if deriv == "z"
            else (grid_used.num_rho, grid_used.num_theta, grid_used.num_zeta, -1)
        )

        # reshape everything to (θ,ρ,ζ,3) and convert to xyz
        data[key] = rpz2xyz_vec(data[key], phi=data["phi"]).reshape(reshape_dims)

        if base_key is None:  # take derivatives of X,Y,Z
            base = np.array([data["X"], data["Y"], data["Z"]]).T
            base = base.reshape(reshape_dims)
        else:  # derivatives of e_rho, etc.
            base = rpz2xyz_vec(data[base_key], phi=data["phi"]).reshape(reshape_dims)

        # First-order, 4-point stencil finite-difference
        spacing = {
            "r": grids_PEST[deriv].spacing[0, 0],
            "v": grids_PEST[deriv].spacing[0, 1] / 2,
            "z": grids_PEST[deriv].spacing[0, 2] / eq.NFP,
        }

        if deriv == "r":
            fd = np.apply_along_axis(my_convolve, 0, base, FD_COEF_1_4) / spacing[deriv]
        elif deriv == "v":
            fd = np.apply_along_axis(my_convolve, 1, base, FD_COEF_1_4) / spacing[deriv]
            fd = fd[0]
            data[key] = data[key][0]
        else:
            fd = np.apply_along_axis(my_convolve, 2, base, FD_COEF_1_4) / spacing[deriv]
            fd = fd[0][0]
            data[key] = data[key][0][0]

        np.testing.assert_allclose(
            data[key][4:-4],
            fd[4:-4],
            rtol=3e-3,
            atol=3e-3,
            err_msg=key,
        )


@pytest.mark.unit
@pytest.mark.parametrize("eq", [get("W7-X"), get("NCSX")])
def test_contravariant_basis_vectors_PEST(eq):
    """
    Test only the derivatives of contravariant basis vectors in PEST.

    We compare higher order derivatives by taking the finite-difference derivative
    of the lower-order contravariant basis vectors.
    """
    keys_PEST = [
        "e^vartheta_v|PEST",
        "e^vartheta_p|PEST",
        "e^zeta_v|PEST",
        "e^zeta_p|PEST",
        "e^rho_v|PEST",
        "e^rho_p|PEST",
    ]

    N = 4000

    # spacing grids in each native direction
    grids_PEST = {
        "r": LinearGrid(N, 0, 0, NFP=eq.NFP),
        "v": LinearGrid(0, N, 0, NFP=eq.NFP, sym=True),
        "p": LinearGrid(0, 0, N, NFP=eq.NFP, sym=True),
    }

    # find native (ρ,θ,ζ) nodes that correspond to the uniform θ_PEST grid
    rtz_nodes = map_coordinates(
        eq,
        grids_PEST["v"].nodes,  # (ρ,θ_PEST,ζ)
        inbasis=("rho", "theta_PEST", "zeta"),
        outbasis=("rho", "theta", "zeta"),
        period=(np.inf, 2 * np.pi, np.inf),
    )

    # find native (ρ,θ,ζ) nodes that correspond to the uniform ζ grid
    rtz_nodes1 = map_coordinates(
        eq,
        grids_PEST["p"].nodes,  # (ρ,θ_PEST,ζ)
        inbasis=("rho", "theta_PEST", "zeta"),
        outbasis=("rho", "theta", "zeta"),
        period=(np.inf, 2 * np.pi, np.inf),
    )

    for key in keys_PEST:
        lhs, rhs = key.split("|")
        # rhs can be PEST or r,v or p,v or r,p etc.
        # So it's not really needed for the FD calculation
        parts = lhs.split("^")
        parts2 = parts[-1].split("_")

        deriv_tokn = parts[-1]
        base_bits = parts[:-1] + ["^"] + [parts2[0]]
        base = []

        deriv = deriv_tokn.split("_")[-1]

        # Grid will have to be custom for vartheta
        grid_used = (
            Grid(rtz_nodes)
            if deriv == "v"
            else (grids_PEST[deriv] if deriv == "r" else Grid(rtz_nodes1))
        )

        # Decide base vector
        if len(base_bits) == 1:  # only happens for X,Y,Z which is not applicable here
            base_key = None  # triggers Cartesian path
        else:
            base_key = "".join(base_bits)  # e_rho, e_vartheta,

        # adding parenthesis to the higher-order vectors
        list0 = key.split("|")
        key = "(" + list0[0] + ")" + "|" + list0[1]

        req_keys = [key, "phi"] + base + ([] if base_key is None else [base_key])
        data = eq.compute(req_keys, grid=grid_used)

        # Determine reshape dimensions based on deriv
        reshape_dims = (
            (1, 1, grid_used.num_zeta, -1)
            if deriv == "p"
            else (grid_used.num_rho, grid_used.num_theta, grid_used.num_zeta, -1)
        )

        # reshape everything to (θ,ρ,ζ,3) and convert to xyz
        data[key] = rpz2xyz_vec(data[key], phi=data["phi"]).reshape(reshape_dims)

        if base_key is None:
            pass
        else:  # derivatives of e^rho, etc.
            base = rpz2xyz_vec(data[base_key], phi=data["phi"]).reshape(reshape_dims)

        # First-order, 4-point stencil finite-difference
        spacing = {
            "r": grids_PEST[deriv].spacing[0, 0],
            "v": grids_PEST[deriv].spacing[0, 1] / 2,
            "p": grids_PEST[deriv].spacing[0, 2] / eq.NFP,
        }

        if deriv == "r":
            fd = np.apply_along_axis(my_convolve, 0, base, FD_COEF_1_4) / spacing[deriv]
        elif deriv == "v":
            fd = np.apply_along_axis(my_convolve, 1, base, FD_COEF_1_4) / spacing[deriv]
            fd = fd[0]
            data[key] = data[key][0]
        else:
            fd = np.apply_along_axis(my_convolve, 2, base, FD_COEF_1_4) / spacing[deriv]
            fd = fd[0][0]
            data[key] = data[key][0][0]

        np.testing.assert_allclose(
            data[key][4:-4],
            fd[4:-4],
            rtol=7e-3,
            atol=6e-3,
            err_msg=key,
        )


@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.parametrize("eq, mapping_tol", [(get("precise_QA"), 1e-10)])
def test_PEST_derivative_math(eq, mapping_tol):
    """Verify math to write PEST derivative quantities by redefining θ to θ_PEST."""
    from desc.compute import data_index

    # TODO: can reduce rtol of test if resolution is increased. See DESC git #1919
    eq_PEST = eq.to_sfl(3 * eq.L, 3 * eq.M, 3 * eq.N, copy=True, tol=mapping_tol)
    eq.change_resolution(
        L_grid=eq_PEST.L_grid, M_grid=eq_PEST.M_grid, N_grid=eq_PEST.N_grid
    )
    grid_PEST = LinearGrid(
        rho=np.linspace(0.2, 1, 10), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym
    )

    keys_DESC = [
        "e_theta",
        "e_theta_r",
        "e_theta_t",
        "e_theta_z",
        "e_zeta_r",
        "e_zeta_t",
        "e_zeta_z",
        "e_rho_r",
        "e^rho_t",
        "e^rho_z",
        "e^theta",
        "e^theta_t",
        "e^theta_z",
        "e^zeta_t",
        "e^zeta_z",
        "g_rr",
        "g_rt",
        "g_rz",
        "g_tt",
        "g_tz",
        "g_zz",
        "g_rr_t",
        "g_rr_z",
        "g_tt_r",
        "g_tt_z",
        "g_zz_t",
        "g_rt_z",
        "g^rt",
        "g^rr_t",
        "g^rr_z",
        "g^rt_t",
        "g^rt_z",
        "g^rz_t",
        "g^rz_z",
        "sqrt(g)_r",
        "sqrt(g)_t",
        "sqrt(g)_z",
        "J^theta",
        "J^theta_t",
        "J^theta_z",
        "J^zeta_t",
        "J^zeta_z",
    ]
    keys_PEST = [
        "e_vartheta",
        "(e_theta_PEST_r)|PEST",
        "(e_theta_PEST_v)|PEST",
        "(e_theta_PEST_p)|PEST",
        "(e_phi_r)|PEST",
        "(e_phi_v)|PEST",
        "(e_phi_p)|PEST",
        "(e_rho_r)|PEST",
        "(e^rho_v)|PEST",
        "(e^rho_p)|PEST",
        "e^vartheta",
        "(e^vartheta_v)|PEST",
        "(e^vartheta_p)|PEST",
        "(e^zeta_v)|PEST",
        "(e^zeta_p)|PEST",
        "g_rr|PEST",
        "g_rv|PEST",
        "g_rp|PEST",
        "g_vv|PEST",
        "g_vp|PEST",
        "g_pp|PEST",
        "(g_rr_v)|PEST",
        "(g_rr_p)|PEST",
        "(g_vv_r)|PEST",
        "(g_vv_z)|PEST",
        "(g_pp_v)|PEST",
        "(g_rv_p)|PEST",
        "g^rv",
        "(g^rr_v)|PEST",
        "(g^rr_p)|PEST",
        "(g^rv_v)|PEST",
        "(g^rv_p)|PEST",
        "(g^rz_v)|PEST",
        "(g^rz_p)|PEST",
        "(sqrt(g)_PEST_r)|PEST",
        "(sqrt(g)_PEST_v)|PEST",
        "(sqrt(g)_PEST_p)|PEST",
        "J^theta_PEST",
        "(J^theta_PEST_v)|PEST",
        "(J^theta_PEST_p)|PEST",
        "(J^zeta_v)|PEST",
        "(J^zeta_p)|PEST",
    ]
    index = data_index["desc.equilibrium.equilibrium.Equilibrium"].keys()
    keys_DESC, keys_PEST = zip(
        *[(d, p) for d, p in zip(keys_DESC, keys_PEST) if (d in index) and (p in index)]
    )
    keys_DESC = list(keys_DESC)
    keys_PEST = list(keys_PEST)

    data = eq_PEST.compute(keys_DESC + keys_PEST, grid_PEST)
    data_to_verify = eq.compute(
        keys_PEST,
        Grid(
            eq.map_coordinates(
                grid_PEST.nodes, ("rho", "theta_PEST", "zeta"), tol=mapping_tol
            )
        ),
    )

    for key_DESC, key_PEST in zip(keys_DESC, keys_PEST):
        np.testing.assert_allclose(
            data[key_PEST], data[key_DESC], err_msg=f"{key_PEST} vs {key_DESC}"
        )
        # This should have spectrally accurate error tolerance, but it doesn't.
        # Checks correctness of PEST quantities beyond ensuring there are no
        # missing or additional factors of lambda.
        near_zero_atol = 1e-4
        near_zero = np.isclose(data[key_DESC], 0, rtol=0, atol=near_zero_atol)
        np.testing.assert_allclose(
            data_to_verify[key_PEST][near_zero],
            data[key_DESC][near_zero],
            atol=2 * near_zero_atol,
            err_msg=key_PEST,
        )
        try:
            np.testing.assert_allclose(
                data_to_verify[key_PEST][~near_zero],
                data[key_DESC][~near_zero],
                rtol=7e-3,
                err_msg=key_PEST,
            )
        except AssertionError as e:
            print(e)


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
        data[key] = grid.meshgrid_reshape(
            rpz2xyz_vec(data[key], phi=data["phi"]), "trz"
        ).squeeze()
        data[base_quant] = grid.meshgrid_reshape(
            rpz2xyz_vec(data[base_quant], phi=data["phi"]), "trz"
        ).squeeze()

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
        for key in ["A", "V", "a", "R0", "R0/a", "a_major/a_minor"]:
            x = eq.compute(key)[key].max()  # max needed for elongation broadcasting
            y = eq.surface.compute(key)[key].max()
            if key == "a_major/a_minor":
                rtol, atol = 1e-2, 0  # need looser tol here bc of different grids
            else:
                rtol, atol = 1e-8, 0
            np.testing.assert_allclose(x, y, rtol=rtol, atol=atol, err_msg=name + key)
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
    grid = get_rtz_grid(eq, 0.5, 0, np.linspace(0, 2 * np.pi, 50), coordinates="raz")
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
@pytest.mark.slow
def test_fieldline_average():
    """Test that fieldline average converges to surface average."""
    rho = np.array([1])
    alpha = np.array([0])
    eq = get("DSHAPE")
    iota_grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
    iota = iota_grid.compress(eq.compute("iota", grid=iota_grid)["iota"]).item()
    # For axisymmetric devices, one poloidal transit must be exact.
    zeta = np.linspace(0, 2 * np.pi / iota, 25)
    grid = Grid.create_meshgrid([rho, alpha, zeta], coordinates="raz")
    data = eq.compute(
        ["fieldline length", "fieldline length/volume", "V_r(r)"], grid=grid
    )
    np.testing.assert_allclose(
        data["fieldline length"] / data["fieldline length/volume"],
        data["V_r(r)"] / (4 * np.pi**2),
        rtol=1e-3,
    )
    assert np.all(data["fieldline length"] > 0)
    assert np.all(data["fieldline length/volume"] > 0)

    # Otherwise, many toroidal transits are necessary to sample surface.
    eq = get("W7-X")
    zeta = np.linspace(0, 40 * np.pi, 300)
    grid = Grid.create_meshgrid([rho, alpha, zeta], coordinates="raz")
    data = eq.compute(
        ["fieldline length", "fieldline length/volume", "V_r(r)"], grid=grid
    )
    np.testing.assert_allclose(
        data["fieldline length"] / data["fieldline length/volume"],
        data["V_r(r)"] / (4 * np.pi**2),
        rtol=2e-3,
    )
    assert np.all(data["fieldline length"] > 0)
    assert np.all(data["fieldline length/volume"] > 0)


@pytest.mark.unit
def test_compute_deprecation_warning():
    """Test DeprecationWarning for deprecated compute names."""
    eq = Equilibrium()
    grid = LinearGrid(L=1, M=2, N=2, NFP=eq.NFP)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        eq.compute("sqrt(g)_B", grid=grid)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        eq.compute("|B|_mn", grid=grid, M_booz=eq.M, N_booz=eq.N)
