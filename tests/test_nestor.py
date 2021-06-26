#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 21:14:09 2021

@author: jonathan
"""

import os
import numpy as np
from netCDF4 import Dataset
import unittest
import ooNESTOR

#test_folder = "/data2/jonathan/work/code/educational_VMEC/test"
ref_in_folder = "/home/fouriest/SCHOOL/Princeton/PPPL/NESTOR/tests/ref_in/"
ref_out_folder = "/home/fouriest/SCHOOL/Princeton/PPPL/NESTOR/tests/ref_out/"

runId = "test.vmec"
maxIter = 3



class TestNestor(unittest.TestCase):

    def test_same_outputs(self):

        for iteration in range(maxIter):

            ref_in = os.path.join(ref_in_folder, "vacin_%s_%06d.nc"%(runId, iteration))
            ref_out = os.path.join(ref_out_folder, "vacout_ref_%s_%06d.nc"%(runId, iteration))
            if not os.path.isfile(ref_in):
                raise RuntimeError("reference %s not found"%(ref_in,))
            if not os.path.isfile(ref_out):
                raise RuntimeError("reference %s not found"%(ref_out,))
                
            tst_fname = os.path.join("/home/fouriest/SCHOOL/Princeton/PPPL/NESTOR/tests/tmp/", "vacout_%s_%06d.nc"%(runId, iteration))

            ooNESTOR.main(ref_in, tst_fname)
            if not os.path.isfile(tst_fname):
                raise RuntimeError("test %s not found"%(tst_fname,))
            
 
            ref_data = {}
            d = Dataset(ref_out, "r")
            for key in d.variables:
                ref_data[key] = d[key][()]
            d.close()
                
            tst_data = {}
            d = Dataset(tst_fname, "r")
            for key in d.variables:
                tst_data[key] = d[key][()]
            d.close()

            # compare data
            for key in tst_data:
                r = ref_data[key]
                t = tst_data[key]

                np.testing.assert_allclose(r,t, rtol=1e-10, atol=1e-10, err_msg="iter={}, key={}".format(iteration, key))
