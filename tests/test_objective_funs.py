import unittest
import numpy as np
import pytest
from desc.grid import LinearGrid, ConcentricGrid
from desc.equilibrium import Equilibrium
from desc.basis import (
    PowerSeries,
    DoubleFourierSeries,
    FourierZernikeBasis,
    FourierSeries,
)
from desc.transform import Transform
from desc.objective_funs import (
    get_objective_function,
    ForceErrorNodes,
    EnergyVolIntegral,
)


class TestObjectiveFunctionFactory(unittest.TestCase):
    """Test basic functionality of objective function getter"""

    def test_obj_fxn_types(self):
        """test the correct objective function is returned for 'force', 'energy', and unimplemented"""
        RZ_grid = ConcentricGrid(M=2, N=0)
        R_basis = FourierZernikeBasis(L=-1, M=2, N=0)
        Z_basis = FourierZernikeBasis(L=-1, M=2, N=0)
        L_basis = FourierZernikeBasis(L=-1, M=2, N=0)
        RZb_basis = DoubleFourierSeries(M=3, N=0)
        PI_basis = PowerSeries(L=3)

        R_transform = Transform(RZ_grid, R_basis)
        Z_transform = Transform(RZ_grid, Z_basis)
        L_transform = Transform(RZ_grid, L_basis)
        RZb_transform = Transform(RZ_grid, RZb_basis)
        PI_transform = Transform(RZ_grid, PI_basis)

        errr_mode = "force"
        obj_fun = get_objective_function(
            errr_mode,
            R_transform=R_transform,
            Z_transform=Z_transform,
            L_transform=L_transform,
            Rb_transform=RZb_transform,
            Zb_transform=RZb_transform,
            p_transform=PI_transform,
            i_transform=PI_transform,
        )
        self.assertIsInstance(obj_fun, ForceErrorNodes)
        with pytest.warns(UserWarning):
            errr_mode = "energy"
            obj_fun = get_objective_function(
                errr_mode,
                R_transform=R_transform,
                Z_transform=Z_transform,
                L_transform=L_transform,
                Rb_transform=RZb_transform,
                Zb_transform=RZb_transform,
                p_transform=PI_transform,
                i_transform=PI_transform,
            )
            self.assertIsInstance(obj_fun, EnergyVolIntegral)

        # test unimplemented errr_mode
        with self.assertRaises(ValueError):
            errr_mode = "not implemented"
            obj_fun = get_objective_function(
                errr_mode,
                R_transform=R_transform,
                Z_transform=Z_transform,
                L_transform=L_transform,
                Rb_transform=RZb_transform,
                Zb_transform=RZb_transform,
                p_transform=PI_transform,
                i_transform=PI_transform,
            )


class TestIsNested(unittest.TestCase):
    """tests for functions"""

    def test_is_nested(self):

        inputs = {
            "L": 4,
            "M": 2,
            "N": 0,
            "NFP": 1,
            "Psi": 1.0,
            "profiles": np.array([[0, 0, 0.23]]),
            "boundary": np.array([[0, 0, 0, 10, 0], [0, 1, 0, 1, 0]]),
            "index": "fringe",
        }

        eq1 = Equilibrium(inputs)
        eq1.R_lmn = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0])
        eq1.Z_lmn = np.array([0, 0, -1, 0, 0, 0, 0, 0, 0])
        eq2 = Equilibrium(inputs)
        eq2.R_lmn = np.array([0, 1, 0, 0, 0, 0, 5, 0, 0])
        eq2.Z_lmn = np.array([0, 0, -1, 0, 0, 4, 0, 0, 0])
        self.assertTrue(eq1.is_nested())
        self.assertFalse(eq2.is_nested())
