import unittest
import numpy as np
from desc.equilibrium import Equilibrium
from desc.basis import FourierZernikeBasis
from desc.objectives import FixLambdaGauge
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile


def test_LambdaGauge_axis_sym(DummyStellarator):
    # symmetric cases only have the axis constraint
    eq = Equilibrium.load(
        load_from=str(DummyStellarator["output_path"]), file_format="hdf5"
    )
    eq.change_resolution(L=2, M=1, N=1)
    correct_constraint_matrix = np.zeros((1, 5))
    correct_constraint_matrix[0, 0] = 1
    correct_constraint_matrix[0, 2] = -1

    lam_con = FixLambdaGauge(eq)

    np.testing.assert_array_equal(lam_con._A, correct_constraint_matrix)


def test_LambdaGauge_asym():
    # just testing the gauge condition
    inputs = {
        "sym": False,
        "NFP": 3,
        "Psi": 1.0,
        "L": 2,
        "M": 0,
        "N": 1,
        "pressure": np.array([[0, 1e4], [2, -2e4], [4, 1e4]]),
        "iota": np.array([[0, 0.5], [2, 0.5]]),
        "surface": np.array(
            [
                [0, 0, 0, 3, 0],
                [0, 1, 0, 1, 0],
                [0, -1, 0, 0, 1],
                [0, 1, 1, 0.3, 0],
                [0, -1, -1, -0.3, 0],
                [0, 1, -1, 0, -0.3],
                [0, -1, 1, 0, -0.3],
            ],
        ),
        "axis": np.array([[-1, 0, -0.2], [0, 3.4, 0], [1, 0.2, 0]]),
        "objective": "force",
        "optimizer": "lsq-exact",
    }
    eq = Equilibrium(**inputs)
    correct_constraint_matrix = np.zeros((3, eq.L_basis.num_modes))
    correct_constraint_matrix[0, 0] = 1.0
    correct_constraint_matrix[0, 1] = -1.0
    correct_constraint_matrix[2, 4] = 1.0
    correct_constraint_matrix[2, 5] = -1.0
    correct_constraint_matrix[1, 2] = 1.0
    correct_constraint_matrix[1, 3] = -1.0

    lam_con = FixLambdaGauge(eq)

    np.testing.assert_array_equal(
        lam_con._A[0:3, 0 : eq.L_basis.num_modes], correct_constraint_matrix
    )


def test_bc_on_interior_surfaces():
    surf = FourierRZToroidalSurface(rho=0.5)
    iota = PowerSeriesProfile([1, 0, 0.5])
    eq = Equilibrium(L=4, M=4, N=0, surface=surf, iota=iota)
    eq.solve(verbose=0)
    surf5 = eq.get_surface_at(0.5)

    np.testing.assert_allclose(surf.R_lmn, surf5.R_lmn, atol=1e-12)
    np.testing.assert_allclose(surf.Z_lmn, surf5.Z_lmn, atol=1e-12)
