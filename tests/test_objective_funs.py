import unittest
import numpy as np

from desc.objective_funs import is_nested, curve_self_intersects
from desc.objective_funs import ObjectiveFunctionFactory, ForceErrorNodes, AccelErrorSpectral


"""
class TestIsNested(unittest.TestCase):
    ""tests for  functions""

    def test_is_nested(self):
        zidx = get_zern_basis_idx_dense(2, 0)
        cR1 = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0])
        cZ1 = np.array([0, 0, -1, 0, 0, 0, 0, 0, 0])
        cR2 = np.array([0, 1, 0, 0, 0, 0, 5, 0, 0])
        cZ2 = np.array([0, 0, -1, 0, 0, 4, 0, 0, 0])
        self.assertTrue(is_nested(cR1, cZ1, zidx, 1))
        self.assertFalse(is_nested(cR2, cZ2, zidx, 1))

    def test_self_intersection(self):

        # elipse: not self intersected

        a = 1
        b = 1
        d = np.pi/2
        t = np.linspace(0, 2*np.pi, 361)
        x = np.sin(a*t+d)
        y = np.sin(b*t)
        self.assertFalse(curve_self_intersects(x, y))

        # lissajois: is self intersected
        a = 1
        b = 2
        d = np.pi/2
        t = np.linspace(0, 2*np.pi, 361)
        x = np.sin(a*t+d)
        y = np.sin(b*t)
        self.assertTrue(curve_self_intersects(x, y))
"""


class TestObjectiveFunctionFactory(unittest.TestCase):
    """Test basic functionality of ObjectiveFunctionFactory"""
    def test_obj_fxn_types(self):
        """test the correct objective function is returned for 'force', 'accel', and unimplemented"""
        obj_fun_factory = ObjectiveFunctionFactory()
        stell_sym = True
        M = 1
        N = 1
        NFP = 1
        zernike_transform = None
        bdry_zernike_transform = None
        zern_idx = np.eye(3)
        lambda_idx = np.eye(3)
        bdry_pol = np.array([1,1,1])
        bdry_tor = bdry_pol
        bdry_mode = 'spectral'
        errr_mode = 'force'
        obj_fun = obj_fun_factory.get_equil_obj_fun(stell_sym, errr_mode, bdry_mode, M, N,
                                                    NFP, zernike_transform, bdry_zernike_transform, zern_idx, lambda_idx,
                                                    bdry_pol, bdry_tor)
        self.assertIsInstance(obj_fun, ForceErrorNodes)
        errr_mode = 'accel'
        obj_fun = obj_fun_factory.get_equil_obj_fun(stell_sym, errr_mode, bdry_mode, M, N,
                                                    NFP, zernike_transform, bdry_zernike_transform, zern_idx, lambda_idx,
                                                    bdry_pol, bdry_tor)
        self.assertIsInstance(obj_fun, AccelErrorSpectral)     
        
        #test that 
        with self.assertRaises(ValueError):
            errr_mode = 'not implemented' #an unimplemented errr_mode
            obj_fun = obj_fun_factory.get_equil_obj_fun(stell_sym, errr_mode, bdry_mode, M, N,
                                                    NFP, zernike_transform, bdry_zernike_transform, zern_idx, lambda_idx,
                                                    bdry_pol, bdry_tor)
        with self.assertRaises(ValueError):
            bdry_mode = 'not implemented' #an unimplemented bdry_mode
            errr_mode = 'force'
            obj_fun = obj_fun_factory.get_equil_obj_fun(stell_sym, errr_mode, bdry_mode, M, N,
                                                    NFP, zernike_transform, bdry_zernike_transform, zern_idx, lambda_idx,
                                                    bdry_pol, bdry_tor)
