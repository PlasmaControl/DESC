import numpy as np
from desc.grid import LinearGrid
from desc.equilibrium import EquilibriaFamily


def test_magnetic_pressure_gradient(DSHAPE):
    """Test that the components of grad(|B|^2)) match with numerical gradients
    for the DSHAPE example."""

    eq = EquilibriaFamily.load(
        load_from=str(DSHAPE["output_path"]), file_format="hdf5"
    )[-1]

    # partial derivative wrt rho
    L = 100
    grid = LinearGrid(L=L)
    magnetic_pressure = eq.compute_magnetic_pressure_gradient(grid)
    magnetic_field = eq.compute_magnetic_field(grid)
    B2 = magnetic_field["|B|"] ** 2

    B2_rho = np.ones_like(B2)
    drho = grid.nodes[1, 0]
    B2_rho[0] = (B2[1] - B2[0]) / drho
    for i in range(1, L - 1):
        B2_rho[i] = (B2[i + 1] - B2[i - 1]) / (2 * drho)
    B2_rho[-1] = (B2[-1] - B2[-2]) / drho

    np.testing.assert_allclose(
        magnetic_pressure["grad(|B|^2)_rho"][2:-2], B2_rho[2:-2], rtol=1e-2
    )

    # partial derivative wrt theta
    M = 240
    grid = LinearGrid(M=M)
    magnetic_pressure = eq.compute_magnetic_pressure_gradient(grid)
    magnetic_field = eq.compute_magnetic_field(grid)
    B2 = magnetic_field["|B|"] ** 2

    B2_theta = np.ones_like(B2)
    dtheta = grid.nodes[1, 1]
    B2_theta[0] = (B2[1] - B2[-1]) / (2 * dtheta)
    for i in range(1, M - 1):
        B2_theta[i] = (B2[i + 1] - B2[i - 1]) / (2 * dtheta)
    B2_theta[-1] = (B2[0] - B2[-2]) / (2 * dtheta)

    np.testing.assert_allclose(
        magnetic_pressure["grad(|B|^2)_theta"][1:-1], B2_theta[1:-1], rtol=1e-2
    )
