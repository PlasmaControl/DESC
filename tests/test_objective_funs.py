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
from desc.objective_funs import ObjectiveFunctionFactory, ForceErrorNodes


class TestObjectiveFunctionFactory(unittest.TestCase):
    """Test basic functionality of ObjectiveFunctionFactory"""

    def test_obj_fxn_types(self):
        """test the correct objective function is returned for 'force', 'accel', and unimplemented"""
        RZ_grid = ConcentricGrid(M=2, N=0)
        R0_basis = FourierSeries(N=1)
        Z0_basis = FourierSeries(N=1)
        r_basis = FourierZernikeBasis(M=2, N=0)
        l_basis = FourierZernikeBasis(M=3, N=1)
        RZ1_basis = DoubleFourierSeries(M=3, N=1)
        PI_basis = PowerSeries(L=3)

        R0_transform = Transform(RZ_grid, R0_basis)
        Z0_transform = Transform(RZ_grid, Z0_basis)
        RZ1_transform = Transform(RZ_grid, RZ1_basis)
        r_transform = Transform(RZ_grid, r_basis)
        l_transform = Transform(RZ_grid, l_basis)
        PI_transform = Transform(RZ_grid, PI_basis)

        errr_mode = "force"
        obj_fun = ObjectiveFunctionFactory.get_equil_obj_fun(
            errr_mode,
            R0_transform=R0_transform,
            Z0_transform=R0_transform,
            r_transform=r_transform,
            l_transform=l_transform,
            R1_transform=RZ1_transform,
            Z1_transform=RZ1_transform,
            p_transform=PI_transform,
            i_transform=PI_transform,
        )
        self.assertIsInstance(obj_fun, ForceErrorNodes)

        # test unimplemented errr_mode
        with self.assertRaises(ValueError):
            errr_mode = "not implemented"
            obj_fun = ObjectiveFunctionFactory.get_equil_obj_fun(
                errr_mode,
                R0_transform=R0_transform,
                Z0_transform=R0_transform,
                r_transform=r_transform,
                l_transform=l_transform,
                R1_transform=RZ1_transform,
                Z1_transform=RZ1_transform,
                p_transform=PI_transform,
                i_transform=PI_transform,
            )
