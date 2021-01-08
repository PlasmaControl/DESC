import unittest
import numpy as np

from desc.grid import LinearGrid, ConcentricGrid
from desc.basis import PowerSeries, DoubleFourierSeries, FourierZernikeBasis
from desc.transform import Transform
from desc.objective_funs import ObjectiveFunctionFactory, ForceErrorNodes


class TestObjectiveFunctionFactory(unittest.TestCase):
    """Test basic functionality of ObjectiveFunctionFactory"""

    def test_obj_fxn_types(self):
        """test the correct objective function is returned for 'force', 'accel', and unimplemented"""
        RZ_grid = ConcentricGrid(M=2, N=0)
        L_grid = LinearGrid(M=2, N=1)
        RZ_basis = FourierZernikeBasis(M=2, N=0)
        L_basis = DoubleFourierSeries(M=2, N=0)
        PI_basis = PowerSeries(L=3)
        RZ_transform = Transform(RZ_grid, RZ_basis)
        RZ1_transform = Transform(L_grid, RZ_basis)
        L_transform = Transform(L_grid, L_basis)
        PI_transform = Transform(RZ_grid, PI_basis)

        errr_mode = 'force'
        obj_fun = ObjectiveFunctionFactory.get_equil_obj_fun(errr_mode,
                                                             R_transform=RZ_transform, Z_transform=RZ_transform,
                                                             R1_transform=RZ1_transform, Z1_transform=RZ1_transform,
                                                             L_transform=L_transform, P_transform=PI_transform,
                                                             I_transform=PI_transform)
        self.assertIsInstance(obj_fun, ForceErrorNodes)

        # test unimplemented errr_mode
        with self.assertRaises(ValueError):
            errr_mode = 'not implemented'
            obj_fun = ObjectiveFunctionFactory.get_equil_obj_fun(errr_mode,
                                                                 R_transform=RZ_transform, Z_transform=RZ_transform,
                                                                 R1_transform=RZ1_transform, Z1_transform=RZ1_transform,
                                                                 L_transform=L_transform, P_transform=PI_transform,
                                                                 I_transform=PI_transform)
