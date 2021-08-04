import numpy as np
from scipy.signal import convolve2d

from desc.grid import LinearGrid
from desc.equilibrium import Equilibrium
from desc.transform import Transform
from desc.compute_funs import (
    compute_covariant_magnetic_field,
    compute_magnetic_field_magnitude,
    compute_magnetic_pressure_gradient,
    compute_B_dot_gradB,
)

# convolve kernel is reverse of FD coeffs
FD_COEF_1_2 = np.array([-1 / 2, 0, 1 / 2])[::-1]
FD_COEF_1_4 = np.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12])[::-1]
FD_COEF_2_2 = np.array([1, -2, 1])[::-1]
FD_COEF_2_4 = np.array([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12])[::-1]


def test_magnetic_field_derivatives(DummyStellarator):
    """Test that the partial derivatives of B and |B| match with numerical derivatives
    for a dummy stellarator example."""

    eq = Equilibrium.load(
        load_from=str(DummyStellarator["output_path"]), file_format="hdf5"
    )

    # partial derivatives wrt rho
    L = 50
    grid = LinearGrid(L=L)
    drho = grid.nodes[1, 0]

    R_transform = Transform(grid, eq.R_basis, derivs=3)
    Z_transform = Transform(grid, eq.Z_basis, derivs=3)
    L_transform = Transform(grid, eq.L_basis, derivs=3)
    iota = eq.iota.copy()
    iota.grid = grid

    data = compute_covariant_magnetic_field(
        eq.R_lmn,
        eq.Z_lmn,
        eq.L_lmn,
        eq.i_l,
        eq.Psi,
        R_transform,
        Z_transform,
        L_transform,
        iota,
        dr=1,
        dt=1,
        dz=1,
    )

    B_sup_theta_r = np.convolve(data["B^theta"], FD_COEF_1_4, "same") / drho
    B_sup_zeta_r = np.convolve(data["B^zeta"], FD_COEF_1_4, "same") / drho
    B_sub_rho_r = np.convolve(data["B_rho"], FD_COEF_1_4, "same") / drho
    B_sub_theta_r = np.convolve(data["B_theta"], FD_COEF_1_4, "same") / drho
    B_sub_zeta_r = np.convolve(data["B_zeta"], FD_COEF_1_4, "same") / drho

    np.testing.assert_allclose(
        data["B^theta_r"][3:-2],
        B_sup_theta_r[3:-2],
        rtol=1e-2,
        atol=1e-2 * np.nanmean(np.abs(data["B^theta_r"])),
    )
    np.testing.assert_allclose(
        data["B^zeta_r"][3:-2],
        B_sup_zeta_r[3:-2],
        rtol=1e-2,
        atol=1e-2 * np.nanmean(np.abs(data["B^zeta_r"])),
    )
    np.testing.assert_allclose(
        data["B_rho_r"][3:-2],
        B_sub_rho_r[3:-2],
        rtol=1e-2,
        atol=1e-2 * np.nanmean(np.abs(data["B_rho_r"])),
    )
    np.testing.assert_allclose(
        data["B_theta_r"][3:-2],
        B_sub_theta_r[3:-2],
        rtol=1e-2,
        atol=1e-2 * np.nanmean(np.abs(data["B_theta_r"])),
    )
    np.testing.assert_allclose(
        data["B_zeta_r"][3:-2],
        B_sub_zeta_r[3:-2],
        rtol=1e-2,
        atol=1e-2 * np.nanmean(np.abs(data["B_zeta_r"])),
    )

    # partial derivatives wrt theta
    M = 90
    grid = LinearGrid(M=M, NFP=eq.NFP)
    dtheta = grid.nodes[1, 1]

    R_transform = Transform(grid, eq.R_basis, derivs=3)
    Z_transform = Transform(grid, eq.Z_basis, derivs=3)
    L_transform = Transform(grid, eq.L_basis, derivs=3)
    iota = eq.iota.copy()
    iota.grid = grid

    data = compute_covariant_magnetic_field(
        eq.R_lmn,
        eq.Z_lmn,
        eq.L_lmn,
        eq.i_l,
        eq.Psi,
        R_transform,
        Z_transform,
        L_transform,
        iota,
        dr=1,
        dt=1,
        dz=1,
    )
    data = compute_magnetic_field_magnitude(
        eq.R_lmn,
        eq.Z_lmn,
        eq.L_lmn,
        eq.i_l,
        eq.Psi,
        R_transform,
        Z_transform,
        L_transform,
        iota,
        dr=0,
        dt=2,
        dz=0,
        data=data,
    )

    B_sup_theta_t = np.convolve(data["B^theta"], FD_COEF_1_4, "same") / dtheta
    B_sup_theta_tt = np.convolve(data["B^theta"], FD_COEF_2_4, "same") / dtheta ** 2
    B_sup_zeta_t = np.convolve(data["B^zeta"], FD_COEF_1_4, "same") / dtheta
    B_sup_zeta_tt = np.convolve(data["B^zeta"], FD_COEF_2_4, "same") / dtheta ** 2
    B_sub_rho_t = np.convolve(data["B_rho"], FD_COEF_1_4, "same") / dtheta
    B_sub_zeta_t = np.convolve(data["B_zeta"], FD_COEF_1_4, "same") / dtheta
    B_t = np.convolve(data["|B|"], FD_COEF_1_4, "same") / dtheta
    B_tt = np.convolve(data["|B|"], FD_COEF_2_4, "same") / dtheta ** 2

    np.testing.assert_allclose(
        data["B^theta_t"][23:-2],
        B_sup_theta_t[2:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(data["B^theta_t"])),
    )
    np.testing.assert_allclose(
        data["B^theta_tt"][2:-2],
        B_sup_theta_tt[2:-2],
        rtol=2e-2,
        atol=2e-2 * np.mean(np.abs(data["B^theta_tt"])),
    )
    np.testing.assert_allclose(
        data["B^zeta_t"][2:-2],
        B_sup_zeta_t[2:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(data["B^zeta_t"])),
    )
    np.testing.assert_allclose(
        data["B^zeta_tt"][2:-2],
        B_sup_zeta_tt[2:-2],
        rtol=2e-2,
        atol=2e-2 * np.mean(np.abs(data["B^zeta_tt"])),
    )
    np.testing.assert_allclose(
        data["B_rho_t"][2:-2],
        B_sub_rho_t[2:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(data["B_rho_t"])),
    )
    np.testing.assert_allclose(
        data["B_zeta_t"][2:-2],
        B_sub_zeta_t[2:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(data["B_zeta_t"])),
    )
    np.testing.assert_allclose(
        data["|B|_t"][2:-2],
        B_t[2:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(data["|B|_t"])),
    )
    np.testing.assert_allclose(
        data["|B|_tt"][2:-2],
        B_tt[2:-2],
        rtol=2e-2,
        atol=2e-2 * np.mean(np.abs(data["|B|_tt"])),
    )

    # partial derivatives wrt zeta
    N = 90
    grid = LinearGrid(N=N, NFP=eq.NFP)
    dzeta = grid.nodes[1, 2]

    R_transform = Transform(grid, eq.R_basis, derivs=3)
    Z_transform = Transform(grid, eq.Z_basis, derivs=3)
    L_transform = Transform(grid, eq.L_basis, derivs=3)
    iota = eq.iota.copy()
    iota.grid = grid

    data = compute_covariant_magnetic_field(
        eq.R_lmn,
        eq.Z_lmn,
        eq.L_lmn,
        eq.i_l,
        eq.Psi,
        R_transform,
        Z_transform,
        L_transform,
        iota,
        dr=1,
        dt=1,
        dz=1,
    )
    data = compute_magnetic_field_magnitude(
        eq.R_lmn,
        eq.Z_lmn,
        eq.L_lmn,
        eq.i_l,
        eq.Psi,
        R_transform,
        Z_transform,
        L_transform,
        iota,
        dr=0,
        dt=0,
        dz=2,
        data=data,
    )

    B_sup_theta_z = np.convolve(data["B^theta"], FD_COEF_1_4, "same") / dzeta
    B_sup_theta_zz = np.convolve(data["B^theta"], FD_COEF_2_4, "same") / dzeta ** 2
    B_sup_zeta_z = np.convolve(data["B^zeta"], FD_COEF_1_4, "same") / dzeta
    B_sup_zeta_zz = np.convolve(data["B^zeta"], FD_COEF_2_4, "same") / dzeta ** 2
    B_sub_rho_z = np.convolve(data["B_rho"], FD_COEF_1_4, "same") / dzeta
    B_sub_theta_z = np.convolve(data["B_theta"], FD_COEF_1_4, "same") / dzeta
    B_z = np.convolve(data["|B|"], FD_COEF_1_4, "same") / dzeta
    B_zz = np.convolve(data["|B|"], FD_COEF_2_4, "same") / dzeta ** 2

    np.testing.assert_allclose(
        data["B^theta_z"][2:-2],
        B_sup_theta_z[2:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(data["B^theta_z"])),
    )
    np.testing.assert_allclose(
        data["B^theta_zz"][2:-2],
        B_sup_theta_zz[2:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(data["B^theta_zz"])),
    )
    np.testing.assert_allclose(
        data["B^zeta_z"][2:-2],
        B_sup_zeta_z[2:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(data["B^zeta_z"])),
    )
    np.testing.assert_allclose(
        data["B^zeta_zz"][2:-2],
        B_sup_zeta_zz[2:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(data["B^zeta_zz"])),
    )
    np.testing.assert_allclose(
        data["B_rho_z"][2:-2],
        B_sub_rho_z[2:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(data["B_rho_z"])),
    )
    np.testing.assert_allclose(
        data["B_theta_z"][2:-2],
        B_sub_theta_z[2:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(data["B_theta_z"])),
    )
    np.testing.assert_allclose(
        data["|B|_z"][2:-2],
        B_z[2:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(data["|B|_z"])),
    )
    np.testing.assert_allclose(
        data["|B|_zz"][2:-2],
        B_zz[2:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(data["|B|_zz"])),
    )

    # mixed derivatives wrt theta & zeta
    M = 125
    N = 125
    grid = LinearGrid(M=M, N=N, NFP=eq.NFP)
    dtheta = grid.nodes[:, 1].reshape((N, M))[0, 1]
    dzeta = grid.nodes[:, 2].reshape((N, M))[1, 0]

    R_transform = Transform(grid, eq.R_basis, derivs=3)
    Z_transform = Transform(grid, eq.Z_basis, derivs=3)
    L_transform = Transform(grid, eq.L_basis, derivs=3)
    iota = eq.iota.copy()
    iota.grid = grid

    data = compute_magnetic_field_magnitude(
        eq.R_lmn,
        eq.Z_lmn,
        eq.L_lmn,
        eq.i_l,
        eq.Psi,
        R_transform,
        Z_transform,
        L_transform,
        iota,
        dr=0,
        dt=2,
        dz=2,
    )

    B_sup_theta = data["B^theta"].reshape((N, M))
    B_sup_zeta = data["B^zeta"].reshape((N, M))
    B = data["|B|"].reshape((N, M))

    B_sup_theta_tz = (
        convolve2d(
            B_sup_theta,
            FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
            mode="same",
            boundary="wrap",
        )
        / (dtheta * dzeta)
    )
    B_sup_zeta_tz = (
        convolve2d(
            B_sup_zeta,
            FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
            mode="same",
            boundary="wrap",
        )
        / (dtheta * dzeta)
    )
    B_tz = (
        convolve2d(
            B,
            FD_COEF_1_4[:, np.newaxis] * FD_COEF_1_4[np.newaxis, :],
            mode="same",
            boundary="wrap",
        )
        / (dtheta * dzeta)
    )

    np.testing.assert_allclose(
        data["B^theta_tz"].reshape((N, M))[2:-2, 2:-2],
        B_sup_theta_tz[2:-2, 2:-2],
        rtol=2e-2,
        atol=2e-2 * np.mean(np.abs(data["B^theta_tz"])),
    )
    np.testing.assert_allclose(
        data["B^zeta_tz"].reshape((N, M))[2:-2, 2:-2],
        B_sup_zeta_tz[2:-2, 2:-2],
        rtol=2e-2,
        atol=2e-2 * np.mean(np.abs(data["B^zeta_tz"])),
    )
    np.testing.assert_allclose(
        data["|B|_tz"].reshape((N, M))[2:-2, 2:-2],
        B_tz[2:-2, 2:-2],
        rtol=2e-2,
        atol=2e-2 * np.mean(np.abs(data["|B|_tz"])),
    )


def test_magnetic_pressure_gradient(DummyStellarator):
    """Test that the components of grad(|B|^2)) match with numerical gradients
    for a dummy stellarator example."""

    eq = Equilibrium.load(
        load_from=str(DummyStellarator["output_path"]), file_format="hdf5"
    )

    # partial derivatives wrt rho
    L = 50
    grid = LinearGrid(L=L, NFP=eq.NFP)
    drho = grid.nodes[1, 0]

    R_transform = Transform(grid, eq.R_basis, derivs=2)
    Z_transform = Transform(grid, eq.Z_basis, derivs=2)
    L_transform = Transform(grid, eq.L_basis, derivs=2)
    iota = eq.iota.copy()
    iota.grid = grid

    data = compute_magnetic_field_magnitude(
        eq.R_lmn,
        eq.Z_lmn,
        eq.L_lmn,
        eq.i_l,
        eq.Psi,
        R_transform,
        Z_transform,
        L_transform,
        iota,
    )
    data = compute_magnetic_pressure_gradient(
        eq.R_lmn,
        eq.Z_lmn,
        eq.L_lmn,
        eq.i_l,
        eq.Psi,
        R_transform,
        Z_transform,
        L_transform,
        iota,
        data=data,
    )
    B2_r = np.convolve(data["|B|"] ** 2, FD_COEF_1_4, "same") / drho

    np.testing.assert_allclose(
        data["grad(|B|^2)_rho"][3:-2],
        B2_r[3:-2],
        rtol=1e-2,
        atol=1e-2 * np.nanmean(np.abs(data["grad(|B|^2)_rho"])),
    )

    # partial derivative wrt theta
    M = 90
    grid = LinearGrid(M=M, NFP=eq.NFP)
    dtheta = grid.nodes[1, 1]

    R_transform = Transform(grid, eq.R_basis, derivs=2)
    Z_transform = Transform(grid, eq.Z_basis, derivs=2)
    L_transform = Transform(grid, eq.L_basis, derivs=2)
    iota = eq.iota.copy()
    iota.grid = grid

    data = compute_magnetic_pressure_gradient(
        eq.R_lmn,
        eq.Z_lmn,
        eq.L_lmn,
        eq.i_l,
        eq.Psi,
        R_transform,
        Z_transform,
        L_transform,
        iota,
    )
    B2_t = np.convolve(data["|B|"], FD_COEF_1_4, "same") / dtheta

    np.testing.assert_allclose(
        data["grad(|B|^2)_theta"][2:-2],
        B2_t[2:-2],
        rtol=1e-2,
        atol=1e-2 * np.nanmean(np.abs(data["grad(|B|^2)_theta"])),
    )

    # partial derivative wrt zeta
    N = 90
    grid = LinearGrid(N=N, NFP=eq.NFP)
    dzeta = grid.nodes[1, 2]

    R_transform = Transform(grid, eq.R_basis, derivs=2)
    Z_transform = Transform(grid, eq.Z_basis, derivs=2)
    L_transform = Transform(grid, eq.L_basis, derivs=2)
    iota = eq.iota.copy()
    iota.grid = grid

    data = compute_magnetic_pressure_gradient(
        eq.R_lmn,
        eq.Z_lmn,
        eq.L_lmn,
        eq.i_l,
        eq.Psi,
        R_transform,
        Z_transform,
        L_transform,
        iota,
    )
    B2_z = np.convolve(data["|B|"], FD_COEF_1_4, "same") / dzeta

    np.testing.assert_allclose(
        data["grad(|B|^2)_zeta"][2:-2],
        B2_z[2:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(data["grad(|B|^2)_zeta"])),
    )


def test_quasisymmetry(DummyStellarator):
    """Test that the components of grad(B*grad(|B|)) match with numerical gradients
    for a dummy stellarator example."""

    eq = Equilibrium.load(
        load_from=str(DummyStellarator["output_path"]), file_format="hdf5"
    )

    # partial derivative wrt theta
    M = 120
    grid = LinearGrid(M=M, NFP=eq.NFP)
    dtheta = grid.nodes[1, 1]

    R_transform = Transform(grid, eq.R_basis, derivs=3)
    Z_transform = Transform(grid, eq.Z_basis, derivs=3)
    L_transform = Transform(grid, eq.L_basis, derivs=3)
    iota = eq.iota.copy()
    iota.grid = grid

    data = compute_B_dot_gradB(
        eq.R_lmn,
        eq.Z_lmn,
        eq.L_lmn,
        eq.i_l,
        eq.Psi,
        R_transform,
        Z_transform,
        L_transform,
        iota,
        dr=0,
        dt=1,
        dz=1,
    )
    Btilde_t = np.convolve(data["B*grad(|B|)"], FD_COEF_1_4, "same") / dtheta

    np.testing.assert_allclose(
        data["B*grad(|B|)_t"][2:-2],
        Btilde_t[2:-2],
        rtol=2e-2,
        atol=2e-2 * np.mean(np.abs(data["B*grad(|B|)_t"])),
    )

    # partial derivative wrt zeta
    N = 120
    grid = LinearGrid(N=N, NFP=eq.NFP)
    dzeta = grid.nodes[1, 2]

    R_transform = Transform(grid, eq.R_basis, derivs=3)
    Z_transform = Transform(grid, eq.Z_basis, derivs=3)
    L_transform = Transform(grid, eq.L_basis, derivs=3)
    iota = eq.iota.copy()
    iota.grid = grid

    data = compute_B_dot_gradB(
        eq.R_lmn,
        eq.Z_lmn,
        eq.L_lmn,
        eq.i_l,
        eq.Psi,
        R_transform,
        Z_transform,
        L_transform,
        iota,
        dr=0,
        dt=1,
        dz=1,
    )
    Btilde_z = np.convolve(data["B*grad(|B|)"], FD_COEF_1_4, "same") / dzeta

    np.testing.assert_allclose(
        data["B*grad(|B|)_z"][2:-2],
        Btilde_z[2:-2],
        rtol=2e-2,
        atol=2e-2 * np.mean(np.abs(data["B*grad(|B|)_t"])),
    )
