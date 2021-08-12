import numpy as np
from scipy.signal import convolve2d

from desc.grid import LinearGrid
from desc.equilibrium import Equilibrium
from desc.transform import Transform
from desc.compute_funs import compute_quasisymmetry

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
    pres = eq.pressure.copy()
    pres.grid = grid
    iota = eq.iota.copy()
    iota.grid = grid

    (
        quasisymmetry,
        current_density,
        magnetic_field,
        con_basis,
        jacobian,
        cov_basis,
        toroidal_coords,
        profiles,
    ) = compute_quasisymmetry(
        eq.Psi,
        eq.R_lmn,
        eq.Z_lmn,
        eq.L_lmn,
        eq.p_l,
        eq.i_l,
        R_transform,
        Z_transform,
        L_transform,
        pres,
        iota,
    )

    B_sup_theta_r = np.convolve(magnetic_field["B^theta"], FD_COEF_1_4, "same") / drho
    B_sub_theta_r = np.convolve(magnetic_field["B_theta"], FD_COEF_1_4, "same") / drho
    B_sup_zeta_r = np.convolve(magnetic_field["B^zeta"], FD_COEF_1_4, "same") / drho
    B_sub_zeta_r = np.convolve(magnetic_field["B_zeta"], FD_COEF_1_4, "same") / drho

    np.testing.assert_allclose(
        magnetic_field["B^theta_r"][3:-2],
        B_sup_theta_r[3:-2],
        rtol=1e-2,
        atol=1e-2 * np.nanmean(np.abs(magnetic_field["B^theta_r"])),
    )
    np.testing.assert_allclose(
        magnetic_field["B^zeta_r"][3:-2],
        B_sup_zeta_r[3:-2],
        rtol=1e-2,
        atol=1e-2 * np.nanmean(np.abs(magnetic_field["B^zeta_r"])),
    )
    np.testing.assert_allclose(
        magnetic_field["B_theta_r"][3:-2],
        B_sub_theta_r[3:-2],
        rtol=1e-2,
        atol=1e-2 * np.nanmean(np.abs(magnetic_field["B_theta_r"])),
    )
    np.testing.assert_allclose(
        magnetic_field["B_zeta_r"][3:-2],
        B_sub_zeta_r[3:-2],
        rtol=1e-2,
        atol=1e-2 * np.nanmean(np.abs(magnetic_field["B_zeta_r"])),
    )

    # partial derivatives wrt theta
    M = 90
    grid = LinearGrid(M=M, NFP=eq.NFP)
    dtheta = grid.nodes[1, 1]

    R_transform = Transform(grid, eq.R_basis, derivs=3)
    Z_transform = Transform(grid, eq.Z_basis, derivs=3)
    L_transform = Transform(grid, eq.L_basis, derivs=3)
    pres = eq.pressure.copy()
    pres.grid = grid
    iota = eq.iota.copy()
    iota.grid = grid

    (
        quasisymmetry,
        current_density,
        magnetic_field,
        con_basis,
        jacobian,
        cov_basis,
        toroidal_coords,
        profiles,
    ) = compute_quasisymmetry(
        eq.Psi,
        eq.R_lmn,
        eq.Z_lmn,
        eq.L_lmn,
        eq.p_l,
        eq.i_l,
        R_transform,
        Z_transform,
        L_transform,
        pres,
        iota,
    )

    B_sup_theta_t = np.convolve(magnetic_field["B^theta"], FD_COEF_1_4, "same") / dtheta
    B_sup_theta_tt = (
        np.convolve(magnetic_field["B^theta"], FD_COEF_2_4, "same") / dtheta ** 2
    )
    B_sup_zeta_t = np.convolve(magnetic_field["B^zeta"], FD_COEF_1_4, "same") / dtheta
    B_sup_zeta_tt = (
        np.convolve(magnetic_field["B^zeta"], FD_COEF_2_4, "same") / dtheta ** 2
    )
    B_sub_rho_t = np.convolve(magnetic_field["B_rho"], FD_COEF_1_4, "same") / dtheta
    B_sub_zeta_t = np.convolve(magnetic_field["B_zeta"], FD_COEF_1_4, "same") / dtheta
    B_t = np.convolve(magnetic_field["|B|"], FD_COEF_1_4, "same") / dtheta
    B_tt = np.convolve(magnetic_field["|B|"], FD_COEF_2_4, "same") / dtheta ** 2

    np.testing.assert_allclose(
        magnetic_field["B^theta_t"][3:-2],
        B_sup_theta_t[3:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_field["B^theta_t"])),
    )
    np.testing.assert_allclose(
        magnetic_field["B^theta_tt"][3:-2],
        B_sup_theta_tt[3:-2],
        rtol=2e-2,
        atol=2e-2 * np.mean(np.abs(magnetic_field["B^theta_tt"])),
    )
    np.testing.assert_allclose(
        magnetic_field["B^zeta_t"][3:-2],
        B_sup_zeta_t[3:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_field["B^zeta_t"])),
    )
    np.testing.assert_allclose(
        magnetic_field["B^zeta_tt"][3:-2],
        B_sup_zeta_tt[3:-2],
        rtol=2e-2,
        atol=2e-2 * np.mean(np.abs(magnetic_field["B^zeta_tt"])),
    )
    np.testing.assert_allclose(
        magnetic_field["B_rho_t"][3:-2],
        B_sub_rho_t[3:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_field["B_rho_t"])),
    )
    np.testing.assert_allclose(
        magnetic_field["B_zeta_t"][3:-2],
        B_sub_zeta_t[3:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_field["B_zeta_t"])),
    )
    np.testing.assert_allclose(
        magnetic_field["|B|_t"][3:-2],
        B_t[3:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_field["|B|_t"])),
    )
    np.testing.assert_allclose(
        magnetic_field["|B|_tt"][3:-2],
        B_tt[3:-2],
        rtol=2e-2,
        atol=2e-2 * np.mean(np.abs(magnetic_field["|B|_tt"])),
    )

    # partial derivatives wrt zeta
    N = 90
    grid = LinearGrid(N=N, NFP=eq.NFP)
    dzeta = grid.nodes[1, 2]

    R_transform = Transform(grid, eq.R_basis, derivs=3)
    Z_transform = Transform(grid, eq.Z_basis, derivs=3)
    L_transform = Transform(grid, eq.L_basis, derivs=3)
    pres = eq.pressure.copy()
    pres.grid = grid
    iota = eq.iota.copy()
    iota.grid = grid

    (
        quasisymmetry,
        current_density,
        magnetic_field,
        con_basis,
        jacobian,
        cov_basis,
        toroidal_coords,
        profiles,
    ) = compute_quasisymmetry(
        eq.Psi,
        eq.R_lmn,
        eq.Z_lmn,
        eq.L_lmn,
        eq.p_l,
        eq.i_l,
        R_transform,
        Z_transform,
        L_transform,
        pres,
        iota,
    )

    B_sup_theta_z = np.convolve(magnetic_field["B^theta"], FD_COEF_1_4, "same") / dzeta
    B_sup_theta_zz = (
        np.convolve(magnetic_field["B^theta"], FD_COEF_2_4, "same") / dzeta ** 2
    )
    B_sup_zeta_z = np.convolve(magnetic_field["B^zeta"], FD_COEF_1_4, "same") / dzeta
    B_sup_zeta_zz = (
        np.convolve(magnetic_field["B^zeta"], FD_COEF_2_4, "same") / dzeta ** 2
    )
    B_sub_rho_z = np.convolve(magnetic_field["B_rho"], FD_COEF_1_4, "same") / dzeta
    B_sub_theta_z = np.convolve(magnetic_field["B_theta"], FD_COEF_1_4, "same") / dzeta
    B_z = np.convolve(magnetic_field["|B|"], FD_COEF_1_4, "same") / dzeta
    B_zz = np.convolve(magnetic_field["|B|"], FD_COEF_2_4, "same") / dzeta ** 2

    np.testing.assert_allclose(
        magnetic_field["B^theta_z"][3:-2],
        B_sup_theta_z[3:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_field["B^theta_z"])),
    )
    np.testing.assert_allclose(
        magnetic_field["B^theta_zz"][3:-2],
        B_sup_theta_zz[3:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_field["B^theta_zz"])),
    )
    np.testing.assert_allclose(
        magnetic_field["B^zeta_z"][3:-2],
        B_sup_zeta_z[3:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_field["B^zeta_z"])),
    )
    np.testing.assert_allclose(
        magnetic_field["B^zeta_zz"][3:-2],
        B_sup_zeta_zz[3:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_field["B^zeta_zz"])),
    )
    np.testing.assert_allclose(
        magnetic_field["B_rho_z"][3:-2],
        B_sub_rho_z[3:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_field["B_rho_z"])),
    )
    np.testing.assert_allclose(
        magnetic_field["B_theta_z"][3:-2],
        B_sub_theta_z[3:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_field["B_theta_z"])),
    )
    np.testing.assert_allclose(
        magnetic_field["|B|_z"][3:-2],
        B_z[3:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_field["|B|_z"])),
    )
    np.testing.assert_allclose(
        magnetic_field["|B|_zz"][3:-2],
        B_zz[3:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_field["|B|_zz"])),
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
    pres = eq.pressure.copy()
    pres.grid = grid
    iota = eq.iota.copy()
    iota.grid = grid

    (
        quasisymmetry,
        current_density,
        magnetic_field,
        con_basis,
        jacobian,
        cov_basis,
        toroidal_coords,
        profiles,
    ) = compute_quasisymmetry(
        eq.Psi,
        eq.R_lmn,
        eq.Z_lmn,
        eq.L_lmn,
        eq.p_l,
        eq.i_l,
        R_transform,
        Z_transform,
        L_transform,
        pres,
        iota,
    )

    B_sup_theta = magnetic_field["B^theta"].reshape((N, M))
    B_sup_zeta = magnetic_field["B^zeta"].reshape((N, M))
    B = magnetic_field["|B|"].reshape((N, M))

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
        magnetic_field["B^theta_tz"].reshape((N, M))[2:-2, 2:-2],
        B_sup_theta_tz[2:-2, 2:-2],
        rtol=2e-2,
        atol=2e-2 * np.mean(np.abs(magnetic_field["B^theta_tz"])),
    )
    np.testing.assert_allclose(
        magnetic_field["B^zeta_tz"].reshape((N, M))[2:-2, 2:-2],
        B_sup_zeta_tz[2:-2, 2:-2],
        rtol=2e-2,
        atol=2e-2 * np.mean(np.abs(magnetic_field["B^zeta_tz"])),
    )
    np.testing.assert_allclose(
        magnetic_field["|B|_tz"].reshape((N, M))[2:-2, 2:-2],
        B_tz[2:-2, 2:-2],
        rtol=2e-2,
        atol=2e-2 * np.mean(np.abs(magnetic_field["|B|_tz"])),
    )


def test_magnetic_pressure_gradient(DummyStellarator):
    """Test that the components of grad(|B|^2)) match with numerical gradients
    for a dummy stellarator example."""

    eq = Equilibrium.load(
        load_from=str(DummyStellarator["output_path"]), file_format="hdf5"
    )

    # partial derivative wrt rho
    L = 90
    grid = LinearGrid(L=L, NFP=eq.NFP)
    magnetic_pressure = eq.compute_magnetic_pressure_gradient(grid)
    magnetic_field = eq.compute_magnetic_field(grid)
    B2 = magnetic_field["|B|"] ** 2

    drho = grid.nodes[1, 0]
    B2_rho = np.convolve(B2, FD_COEF_1_4, "same") / drho

    np.testing.assert_allclose(
        magnetic_pressure["grad(|B|^2)_rho"][2:-2],
        B2_rho[2:-2],
        rtol=1e-2,
        atol=1e-2 * np.nanmean(np.abs(magnetic_pressure["grad(|B|^2)_rho"])),
    )

    # partial derivative wrt theta
    M = 90
    grid = LinearGrid(M=M, NFP=eq.NFP)
    magnetic_pressure = eq.compute_magnetic_pressure_gradient(grid)
    magnetic_field = eq.compute_magnetic_field(grid)
    B2 = magnetic_field["|B|"] ** 2

    dtheta = grid.nodes[1, 1]
    B2_theta = np.convolve(B2, FD_COEF_1_4, "same") / dtheta

    np.testing.assert_allclose(
        magnetic_pressure["grad(|B|^2)_theta"][2:-2],
        B2_theta[2:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_pressure["grad(|B|^2)_theta"])),
    )

    # partial derivative wrt zeta
    N = 90
    grid = LinearGrid(N=N, NFP=eq.NFP)
    magnetic_pressure = eq.compute_magnetic_pressure_gradient(grid)
    magnetic_field = eq.compute_magnetic_field(grid)
    B2 = magnetic_field["|B|"] ** 2

    dzeta = grid.nodes[1, 2]
    B2_zeta = np.convolve(B2, FD_COEF_1_4, "same") / dzeta

    np.testing.assert_allclose(
        magnetic_pressure["grad(|B|^2)_zeta"][2:-2],
        B2_zeta[2:-2],
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_pressure["grad(|B|^2)_zeta"])),
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
    quasisymmetry = eq.compute_quasisymmetry(grid)
    Btilde = quasisymmetry["B*grad(|B|)"]

    dtheta = grid.nodes[1, 1]
    Btilde_theta = np.convolve(Btilde, FD_COEF_1_4, "same") / dtheta

    np.testing.assert_allclose(
        quasisymmetry["B*grad(|B|)_t"][2:-2],
        Btilde_theta[2:-2],
        rtol=2e-2,
        atol=2e-2 * np.mean(np.abs(quasisymmetry["B*grad(|B|)_t"])),
    )

    # partial derivative wrt zeta
    N = 120
    grid = LinearGrid(N=N, NFP=eq.NFP)
    quasisymmetry = eq.compute_quasisymmetry(grid)
    Btilde = quasisymmetry["B*grad(|B|)"]

    dzeta = grid.nodes[1, 2]
    Btilde_zeta = np.convolve(Btilde, FD_COEF_1_4, "same") / dzeta

    np.testing.assert_allclose(
        quasisymmetry["B*grad(|B|)_z"][2:-2],
        Btilde_zeta[2:-2],
        rtol=2e-2,
        atol=2e-2 * np.mean(np.abs(quasisymmetry["B*grad(|B|)_z"])),
    )
