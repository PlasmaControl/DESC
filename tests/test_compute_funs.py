import numpy as np

from desc.grid import LinearGrid, Grid
from desc.equilibrium import Equilibrium, EquilibriaFamily
from desc.transform import Transform
from desc.compute_funs import compute_quasisymmetry


def test_magnetic_field_derivatives(DummyStellarator):
    """Test that the partial derivatives of B and |B| match with numerical derivatives
    for a dummy stellarator example."""

    eq = Equilibrium.load(
        load_from=str(DummyStellarator["output_path"]), file_format="hdf5"
    )

    # partial derivatives wrt rho
    L = 201
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

    B_sup_theta_r = np.zeros_like(magnetic_field["B^theta"])
    B_sup_zeta_r = np.zeros_like(magnetic_field["B^zeta"])
    B_sub_theta_r = np.zeros_like(magnetic_field["B_theta"])
    B_sub_zeta_r = np.zeros_like(magnetic_field["B_zeta"])

    for i in range(1, L - 1):
        B_sup_theta_r[i] = (
            magnetic_field["B^theta"][i + 1] - magnetic_field["B^theta"][i - 1]
        ) / (2 * drho)
        B_sup_zeta_r[i] = (
            magnetic_field["B^zeta"][i + 1] - magnetic_field["B^zeta"][i - 1]
        ) / (2 * drho)
        B_sub_theta_r[i] = (
            magnetic_field["B_theta"][i + 1] - magnetic_field["B_theta"][i - 1]
        ) / (2 * drho)
        B_sub_zeta_r[i] = (
            magnetic_field["B_zeta"][i + 1] - magnetic_field["B_zeta"][i - 1]
        ) / (2 * drho)

    np.testing.assert_allclose(
        magnetic_field["B^theta_r"][2:-1],
        B_sup_theta_r[2:-1],
        rtol=1e-2,
        atol=1e-2 * np.nanmean(np.abs(magnetic_field["B^theta_r"])),
    )
    np.testing.assert_allclose(
        magnetic_field["B^zeta_r"][2:-1],
        B_sup_zeta_r[2:-1],
        rtol=1e-2,
        atol=1e-2 * np.nanmean(np.abs(magnetic_field["B^zeta_r"])),
    )
    np.testing.assert_allclose(
        magnetic_field["B_theta_r"][2:-1],
        B_sub_theta_r[2:-1],
        rtol=1e-2,
        atol=1e-2 * np.nanmean(np.abs(magnetic_field["B_theta_r"])),
    )
    np.testing.assert_allclose(
        magnetic_field["B_zeta_r"][2:-1],
        B_sub_zeta_r[2:-1],
        rtol=1e-2,
        atol=1e-2 * np.nanmean(np.abs(magnetic_field["B_zeta_r"])),
    )

    # partial derivatives wrt theta
    M = 360
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

    B_sup_theta_t = np.zeros_like(magnetic_field["B^theta"])
    B_sup_theta_tt = np.zeros_like(magnetic_field["B^theta"])
    B_sup_zeta_t = np.zeros_like(magnetic_field["B^zeta"])
    B_sup_zeta_tt = np.zeros_like(magnetic_field["B^zeta"])
    B_sub_rho_t = np.zeros_like(magnetic_field["B_rho"])
    B_sub_zeta_t = np.zeros_like(magnetic_field["B_zeta"])
    B_t = np.zeros_like(magnetic_field["|B|"])
    B_tt = np.zeros_like(magnetic_field["|B|"])

    # theta=0
    B_sup_theta_t[0] = (
        magnetic_field["B^theta"][1] - magnetic_field["B^theta"][-1]
    ) / (2 * dtheta)
    B_sup_theta_tt[0] = (
        magnetic_field["B^theta"][1]
        - 2 * magnetic_field["B^theta"][0]
        + magnetic_field["B^theta"][-1]
    ) / (dtheta ** 2)
    B_sup_zeta_t[0] = (magnetic_field["B^zeta"][1] - magnetic_field["B^zeta"][-1]) / (
        2 * dtheta
    )
    B_sup_zeta_tt[0] = (
        magnetic_field["B^zeta"][1]
        - 2 * magnetic_field["B^zeta"][0]
        + magnetic_field["B^zeta"][-1]
    ) / (dtheta ** 2)
    B_sub_rho_t[0] = (magnetic_field["B_rho"][1] - magnetic_field["B_rho"][-1]) / (
        2 * dtheta
    )
    B_sub_zeta_t[0] = (magnetic_field["B_zeta"][1] - magnetic_field["B_zeta"][-1]) / (
        2 * dtheta
    )
    B_t[0] = (magnetic_field["|B|"][1] - magnetic_field["|B|"][-1]) / (2 * dtheta)
    B_tt[0] = (
        magnetic_field["|B|"][1]
        - 2 * magnetic_field["|B|"][0]
        + magnetic_field["|B|"][-1]
    ) / (dtheta ** 2)
    # theta = (0,2pi)
    for i in range(1, M - 1):
        B_sup_theta_t[i] = (
            magnetic_field["B^theta"][i + 1] - magnetic_field["B^theta"][i - 1]
        ) / (2 * dtheta)
        B_sup_theta_tt[i] = (
            magnetic_field["B^theta"][i + 1]
            - 2 * magnetic_field["B^theta"][i]
            + magnetic_field["B^theta"][i - 1]
        ) / (dtheta ** 2)
        B_sup_zeta_t[i] = (
            magnetic_field["B^zeta"][i + 1] - magnetic_field["B^zeta"][i - 1]
        ) / (2 * dtheta)
        B_sup_zeta_tt[i] = (
            magnetic_field["B^zeta"][i + 1]
            - 2 * magnetic_field["B^zeta"][i]
            + magnetic_field["B^zeta"][i - 1]
        ) / (dtheta ** 2)
        B_sub_rho_t[i] = (
            magnetic_field["B_rho"][i + 1] - magnetic_field["B_rho"][i - 1]
        ) / (2 * dtheta)
        B_sub_zeta_t[i] = (
            magnetic_field["B_zeta"][i + 1] - magnetic_field["B_zeta"][i - 1]
        ) / (2 * dtheta)
        B_t[i] = (magnetic_field["|B|"][i + 1] - magnetic_field["|B|"][i - 1]) / (
            2 * dtheta
        )
        B_tt[i] = (
            magnetic_field["|B|"][i + 1]
            - 2 * magnetic_field["|B|"][i]
            + magnetic_field["|B|"][i - 1]
        ) / (dtheta ** 2)
    # theta = 2pi
    B_sup_theta_t[-1] = (
        magnetic_field["B^theta"][0] - magnetic_field["B^theta"][-2]
    ) / (2 * dtheta)
    B_sup_theta_tt[-1] = (
        magnetic_field["B^theta"][0]
        - 2 * magnetic_field["B^theta"][-1]
        + magnetic_field["B^theta"][-2]
    ) / (dtheta ** 2)
    B_sup_zeta_t[-1] = (magnetic_field["B^zeta"][0] - magnetic_field["B^zeta"][-2]) / (
        2 * dtheta
    )
    B_sup_zeta_tt[-1] = (
        magnetic_field["B^zeta"][0]
        - 2 * magnetic_field["B^zeta"][-1]
        + magnetic_field["B^zeta"][-2]
    ) / (dtheta ** 2)
    B_sub_rho_t[-1] = (magnetic_field["B_rho"][0] - magnetic_field["B_rho"][-2]) / (
        2 * dtheta
    )
    B_sub_zeta_t[-1] = (magnetic_field["B_zeta"][0] - magnetic_field["B_zeta"][-2]) / (
        2 * dtheta
    )
    B_t[-1] = (magnetic_field["|B|"][0] - magnetic_field["|B|"][-2]) / (2 * dtheta)
    B_tt[-1] = B_tt[-1] = (
        magnetic_field["|B|"][0]
        - 2 * magnetic_field["|B|"][-1]
        + magnetic_field["|B|"][-2]
    ) / (dtheta ** 2)

    np.testing.assert_allclose(
        magnetic_field["B^theta_t"],
        B_sup_theta_t,
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_field["B^theta_t"])),
    )
    np.testing.assert_allclose(
        magnetic_field["B^theta_tt"],
        B_sup_theta_tt,
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_field["B^theta_tt"])),
    )
    np.testing.assert_allclose(
        magnetic_field["B^zeta_t"],
        B_sup_zeta_t,
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_field["B^zeta_t"])),
    )
    np.testing.assert_allclose(
        magnetic_field["B^zeta_tt"],
        B_sup_zeta_tt,
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_field["B^zeta_tt"])),
    )
    np.testing.assert_allclose(
        magnetic_field["B_rho_t"],
        B_sub_rho_t,
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_field["B_rho_t"])),
    )
    np.testing.assert_allclose(
        magnetic_field["B_zeta_t"],
        B_sub_zeta_t,
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_field["B_zeta_t"])),
    )
    np.testing.assert_allclose(
        magnetic_field["|B|_t"],
        B_t,
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_field["|B|_t"])),
    )
    np.testing.assert_allclose(
        magnetic_field["|B|_tt"],
        B_tt,
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_field["|B|_tt"])),
    )

    # partial derivatives wrt zeta
    N = 360
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

    B_sup_theta_z = np.zeros_like(magnetic_field["B^theta"])
    B_sup_theta_zz = np.zeros_like(magnetic_field["B^theta"])
    B_sup_zeta_z = np.zeros_like(magnetic_field["B^zeta"])
    B_sup_zeta_zz = np.zeros_like(magnetic_field["B^zeta"])
    B_sub_rho_z = np.zeros_like(magnetic_field["B_rho"])
    B_sub_theta_z = np.zeros_like(magnetic_field["B_theta"])
    B_z = np.zeros_like(magnetic_field["|B|"])
    B_zz = np.zeros_like(magnetic_field["|B|"])

    # theta = 0
    B_sup_theta_z[0] = (
        magnetic_field["B^theta"][1] - magnetic_field["B^theta"][-1]
    ) / (2 * dzeta)
    B_sup_theta_zz[0] = (
        magnetic_field["B^theta"][1]
        - 2 * magnetic_field["B^theta"][0]
        + magnetic_field["B^theta"][-1]
    ) / (dzeta ** 2)
    B_sup_zeta_z[0] = (magnetic_field["B^zeta"][1] - magnetic_field["B^zeta"][-1]) / (
        2 * dzeta
    )
    B_sup_zeta_zz[0] = (
        magnetic_field["B^zeta"][1]
        - 2 * magnetic_field["B^zeta"][0]
        + magnetic_field["B^zeta"][-1]
    ) / (dzeta ** 2)
    B_sub_rho_z[0] = (magnetic_field["B_rho"][1] - magnetic_field["B_rho"][-1]) / (
        2 * dzeta
    )
    B_sub_theta_z[0] = (
        magnetic_field["B_theta"][1] - magnetic_field["B_theta"][-1]
    ) / (2 * dzeta)
    B_z[0] = (magnetic_field["|B|"][1] - magnetic_field["|B|"][-1]) / (2 * dzeta)
    B_zz[0] = (
        magnetic_field["|B|"][1]
        - 2 * magnetic_field["|B|"][0]
        + magnetic_field["|B|"][-1]
    ) / (dzeta ** 2)
    # theta = (0,2pi)
    for i in range(1, N - 1):
        B_sup_theta_z[i] = (
            magnetic_field["B^theta"][i + 1] - magnetic_field["B^theta"][i - 1]
        ) / (2 * dzeta)
        B_sup_theta_zz[i] = (
            magnetic_field["B^theta"][i + 1]
            - 2 * magnetic_field["B^theta"][i]
            + magnetic_field["B^theta"][i - 1]
        ) / (dzeta ** 2)
        B_sup_zeta_z[i] = (
            magnetic_field["B^zeta"][i + 1] - magnetic_field["B^zeta"][i - 1]
        ) / (2 * dzeta)
        B_sup_zeta_zz[i] = (
            magnetic_field["B^zeta"][i + 1]
            - 2 * magnetic_field["B^zeta"][i]
            + magnetic_field["B^zeta"][i - 1]
        ) / (dzeta ** 2)
        B_sub_rho_z[i] = (
            magnetic_field["B_rho"][i + 1] - magnetic_field["B_rho"][i - 1]
        ) / (2 * dzeta)
        B_sub_theta_z[i] = (
            magnetic_field["B_theta"][i + 1] - magnetic_field["B_theta"][i - 1]
        ) / (2 * dzeta)
        B_z[i] = (magnetic_field["|B|"][i + 1] - magnetic_field["|B|"][i - 1]) / (
            2 * dzeta
        )
        B_zz[i] = (
            magnetic_field["|B|"][i + 1]
            - 2 * magnetic_field["|B|"][i]
            + magnetic_field["|B|"][i - 1]
        ) / (dzeta ** 2)
    # theta = 2pi
    B_sup_theta_z[-1] = (
        magnetic_field["B^theta"][0] - magnetic_field["B^theta"][-2]
    ) / (2 * dzeta)
    B_sup_theta_zz[-1] = (
        magnetic_field["B^theta"][0]
        - 2 * magnetic_field["B^theta"][-1]
        + magnetic_field["B^theta"][-2]
    ) / (dzeta ** 2)
    B_sup_zeta_z[-1] = (magnetic_field["B^zeta"][0] - magnetic_field["B^zeta"][-2]) / (
        2 * dzeta
    )
    B_sup_zeta_zz[-1] = (
        magnetic_field["B^zeta"][0]
        - 2 * magnetic_field["B^zeta"][-1]
        + magnetic_field["B^zeta"][-2]
    ) / (dzeta ** 2)
    B_sub_rho_z[-1] = (magnetic_field["B_rho"][0] - magnetic_field["B_rho"][-2]) / (
        2 * dzeta
    )
    B_sub_theta_z[-1] = (
        magnetic_field["B_theta"][0] - magnetic_field["B_theta"][-2]
    ) / (2 * dzeta)
    B_z[-1] = (magnetic_field["|B|"][0] - magnetic_field["|B|"][-2]) / (2 * dzeta)
    B_zz[-1] = B_tt[-1] = (
        magnetic_field["|B|"][0]
        - 2 * magnetic_field["|B|"][-1]
        + magnetic_field["|B|"][-2]
    ) / (dzeta ** 2)

    np.testing.assert_allclose(
        magnetic_field["B^theta_z"],
        B_sup_theta_z,
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_field["B^theta_z"])),
    )
    np.testing.assert_allclose(
        magnetic_field["B^theta_zz"],
        B_sup_theta_zz,
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_field["B^theta_zz"])),
    )
    np.testing.assert_allclose(
        magnetic_field["B^zeta_z"],
        B_sup_zeta_z,
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_field["B^zeta_z"])),
    )
    np.testing.assert_allclose(
        magnetic_field["B^zeta_zz"],
        B_sup_zeta_zz,
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_field["B^zeta_zz"])),
    )
    np.testing.assert_allclose(
        magnetic_field["B_rho_z"],
        B_sub_rho_z,
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_field["B_rho_z"])),
    )
    np.testing.assert_allclose(
        magnetic_field["B_theta_z"],
        B_sub_theta_z,
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_field["B_theta_z"])),
    )
    np.testing.assert_allclose(
        magnetic_field["|B|_z"],
        B_z,
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_field["|B|_z"])),
    )
    np.testing.assert_allclose(
        magnetic_field["|B|_zz"],
        B_zz,
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_field["|B|_zz"])),
    )

    # mixed derivatives wrt theta & zeta
    M = 540
    N = 540
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

    B_sup_theta_t = np.zeros_like(B_sup_theta)
    B_sup_zeta_t = np.zeros_like(B_sup_zeta)
    B_t = np.zeros_like(B)

    B_sup_theta_tz = np.zeros_like(B_sup_theta)
    B_sup_zeta_tz = np.zeros_like(B_sup_zeta)
    B_tz = np.zeros_like(B)

    # theta = 0
    B_sup_theta_t[:, 0] = (B_sup_theta[:, 1] - B_sup_theta[:, -1]) / (2 * dtheta)
    B_sup_zeta_t[:, 0] = (B_sup_zeta[:, 1] - B_sup_zeta[:, -1]) / (2 * dtheta)
    B_t[:, 0] = (B[:, 1] - B[:, -1]) / (2 * dtheta)
    # theta = (0,2pi)
    for i in range(1, M - 1):
        B_sup_theta_t[:, i] = (B_sup_theta[:, i + 1] - B_sup_theta[:, i - 1]) / (
            2 * dtheta
        )
        B_sup_zeta_t[:, i] = (B_sup_zeta[:, i + 1] - B_sup_zeta[:, i - 1]) / (
            2 * dtheta
        )
        B_t[:, i] = (B[:, i + 1] - B[:, i - 1]) / (2 * dtheta)
    # theta = 2pi
    B_sup_theta_t[:, -1] = (B_sup_theta[:, 0] - B_sup_theta[:, -2]) / (2 * dtheta)
    B_sup_zeta_t[:, -1] = (B_sup_zeta[:, 0] - B_sup_zeta[:, -2]) / (2 * dtheta)
    B_t[:, -1] = (B[:, 0] - B[:, -2]) / (2 * dtheta)
    # zeta = 0
    B_sup_theta_tz[0, :] = (B_sup_theta_t[1, :] - B_sup_theta_t[-1, :]) / (2 * dzeta)
    B_sup_zeta_tz[0, :] = (B_sup_zeta_t[1, :] - B_sup_zeta_t[-1, :]) / (2 * dzeta)
    B_tz[0, :] = (B_t[1, :] - B_t[-1, :]) / (2 * dzeta)
    # zeta = (0,2pi)
    for i in range(1, N - 1):
        B_sup_theta_tz[i, :] = (B_sup_theta_t[i + 1, :] - B_sup_theta_t[i - 1, :]) / (
            2 * dzeta
        )
        B_sup_zeta_tz[i, :] = (B_sup_zeta_t[i + 1, :] - B_sup_zeta_t[i - 1, :]) / (
            2 * dzeta
        )
        B_tz[i, :] = (B_t[i + 1, :] - B_t[i - 1, :]) / (2 * dzeta)
    # zeta = 2pi
    B_sup_theta_tz[-1, :] = (B_sup_theta_t[0, :] - B_sup_theta_t[-2, :]) / (2 * dzeta)
    B_sup_zeta_tz[-1, :] = (B_sup_zeta_t[0, :] - B_sup_zeta_t[-2, :]) / (2 * dzeta)
    B_tz[-1, :] = (B_t[0, :] - B_t[-2, :]) / (2 * dzeta)

    np.testing.assert_allclose(
        magnetic_field["B^theta_tz"],
        B_sup_theta_tz.flatten(),
        rtol=2e-2,
        atol=2e-2 * np.mean(np.abs(magnetic_field["B^theta_tz"])),
    )
    np.testing.assert_allclose(
        magnetic_field["B^zeta_tz"],
        B_sup_zeta_tz.flatten(),
        rtol=2e-2,
        atol=2e-2 * np.mean(np.abs(magnetic_field["B^zeta_tz"])),
    )
    np.testing.assert_allclose(
        magnetic_field["|B|_tz"],
        B_tz.flatten(),
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
    L = 201
    grid = LinearGrid(L=L, NFP=eq.NFP)
    magnetic_pressure = eq.compute_magnetic_pressure_gradient(grid)
    magnetic_field = eq.compute_magnetic_field(grid)
    B2 = magnetic_field["|B|"] ** 2

    B2_rho = np.zeros_like(B2)
    drho = grid.nodes[1, 0]
    for i in range(1, L - 1):
        B2_rho[i] = (B2[i + 1] - B2[i - 1]) / (2 * drho)

    np.testing.assert_allclose(
        magnetic_pressure["grad(|B|^2)_rho"][1:-1],
        B2_rho[1:-1],
        rtol=1e-2,
        atol=1e-2 * np.nanmean(np.abs(magnetic_pressure["grad(|B|^2)_rho"])),
    )

    # partial derivative wrt theta
    M = 360
    grid = LinearGrid(M=M, NFP=eq.NFP)
    magnetic_pressure = eq.compute_magnetic_pressure_gradient(grid)
    magnetic_field = eq.compute_magnetic_field(grid)
    B2 = magnetic_field["|B|"] ** 2

    B2_theta = np.zeros_like(B2)
    dtheta = grid.nodes[1, 1]
    B2_theta[0] = (B2[1] - B2[-1]) / (2 * dtheta)
    for i in range(1, M - 1):
        B2_theta[i] = (B2[i + 1] - B2[i - 1]) / (2 * dtheta)
    B2_theta[-1] = (B2[0] - B2[-2]) / (2 * dtheta)

    np.testing.assert_allclose(
        magnetic_pressure["grad(|B|^2)_theta"],
        B2_theta,
        rtol=1e-2,
        atol=1e-2 * np.mean(np.abs(magnetic_pressure["grad(|B|^2)_theta"])),
    )

    # partial derivative wrt zeta
    N = 360
    grid = LinearGrid(N=N, NFP=eq.NFP)
    magnetic_pressure = eq.compute_magnetic_pressure_gradient(grid)
    magnetic_field = eq.compute_magnetic_field(grid)
    B2 = magnetic_field["|B|"] ** 2

    B2_zeta = np.zeros_like(B2)
    dzeta = grid.nodes[1, 2]
    B2_zeta[0] = (B2[1] - B2[-1]) / (2 * dzeta)
    for i in range(1, N - 1):
        B2_zeta[i] = (B2[i + 1] - B2[i - 1]) / (2 * dzeta)
    B2_zeta[-1] = (B2[0] - B2[-2]) / (2 * dzeta)

    np.testing.assert_allclose(
        magnetic_pressure["grad(|B|^2)_zeta"],
        B2_zeta,
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
    M = 540
    grid = LinearGrid(M=M, NFP=eq.NFP)
    quasisymmetry = eq.compute_quasisymmetry(grid)
    Btilde = quasisymmetry["B*grad(|B|)"]

    Btilde_theta = np.zeros_like(Btilde)
    dtheta = grid.nodes[1, 1]
    Btilde_theta[0] = (Btilde[1] - Btilde[-1]) / (2 * dtheta)
    for i in range(1, M - 1):
        Btilde_theta[i] = (Btilde[i + 1] - Btilde[i - 1]) / (2 * dtheta)
    Btilde_theta[-1] = (Btilde[0] - Btilde[-2]) / (2 * dtheta)

    np.testing.assert_allclose(
        quasisymmetry["B*grad(|B|)_t"],
        Btilde_theta,
        rtol=2e-2,
        atol=2e-2 * np.mean(np.abs(quasisymmetry["B*grad(|B|)_t"])),
    )

    # partial derivative wrt zeta
    N = 540
    grid = LinearGrid(N=N, NFP=eq.NFP)
    quasisymmetry = eq.compute_quasisymmetry(grid)
    Btilde = quasisymmetry["B*grad(|B|)"]

    Btilde_zeta = np.zeros_like(Btilde)
    dzeta = grid.nodes[1, 2]
    Btilde_zeta[0] = (Btilde[1] - Btilde[-1]) / (2 * dzeta)
    for i in range(1, N - 1):
        Btilde_zeta[i] = (Btilde[i + 1] - Btilde[i - 1]) / (2 * dzeta)
    Btilde_zeta[-1] = (Btilde[0] - Btilde[-2]) / (2 * dzeta)

    np.testing.assert_allclose(
        quasisymmetry["B*grad(|B|)_z"],
        Btilde_zeta,
        rtol=2e-2,
        atol=2e-2 * np.mean(np.abs(quasisymmetry["B*grad(|B|)_z"])),
    )


def test_compute_flux_coords(SOLOVEV):

    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]

    rho = np.linspace(0.01, 0.99, 20)
    theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi, 20, endpoint=False)

    nodes = np.vstack([rho, theta, zeta]).T
    coords = eq.compute_toroidal_coords(Grid(nodes, sort=False))
    real_coords = np.vstack([coords["R"].flatten(), zeta, coords["Z"].flatten()]).T

    flux_coords = eq.compute_flux_coords(real_coords)

    np.testing.assert_allclose(nodes, flux_coords)
