import unittest
import numpy as np

from desc.grid import LinearGrid, ConcentricGrid
from desc.basis import (
    PowerSeries,
    DoubleFourierSeries,
    FourierZernikeBasis,
    FourierSeries,
)
from desc.transform import Transform
from desc.objective_funs import ObjectiveFunctionFactory, ForceErrorNodes, EnergyVolIntegral


class TestObjectiveFunctionFactory(unittest.TestCase):
    """Test basic functionality of ObjectiveFunctionFactory"""

    def test_obj_fxn_types(self):
        """test the correct objective function is returned for 'force', 'energy', and unimplemented"""
        RZ_grid = ConcentricGrid(M=2, N=0)
        R_basis = FourierZernikeBasis(M=2, N=0)
        Z_basis = FourierZernikeBasis(M=2, N=0)
        L_basis = FourierZernikeBasis(M=2, N=0)
        RZb_basis = DoubleFourierSeries(M=3, N=1)
        PI_basis = PowerSeries(L=3)

        R_transform = Transform(RZ_grid, R_basis)
        Z_transform = Transform(RZ_grid, Z_basis)
        L_transform = Transform(RZ_grid, L_basis)
        RZb_transform = Transform(RZ_grid, RZb_basis)
        PI_transform = Transform(RZ_grid, PI_basis)

        errr_mode = "force"
        obj_fun = ObjectiveFunctionFactory.get_equil_obj_fun(
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

        errr_mode = "energy"
        obj_fun = ObjectiveFunctionFactory.get_equil_obj_fun(
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
            obj_fun = ObjectiveFunctionFactory.get_equil_obj_fun(
                errr_mode,
                R_transform=R_transform,
                Z_transform=Z_transform,
                L_transform=L_transform,
                Rb_transform=RZb_transform,
                Zb_transform=RZb_transform,
                p_transform=PI_transform,
                i_transform=PI_transform,
            )
