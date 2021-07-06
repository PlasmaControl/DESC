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
from desc.nestor import Nestor, firstIterationPrintout, produceOutputFile, eval_surface_geometry, eval_surface_geometry_vmec, eval_axis_geometry, evaluate_axis_vmec
from desc.magnetic_fields import SplineMagneticField
from desc.grid import LinearGrid
from desc.basis import DoubleFourierSeries
from desc.vmec_utils import ptolemy_identity_fwd
from desc.utils import copy_coeffs
from desc.transform import Transform


here = os.path.abspath(os.path.dirname(__file__))

runId = "test.vmec"
maxIter = 11
mu0 = 4.0e-7*np.pi


ref_in_folder = here + "/inputs/nestor/ref_in/"
ref_out_folder = here +"/inputs/nestor/ref_out/"


def main(vacin_filename, vacout_filename=None, mgrid=None, method="vmec"):
    vacin = Dataset(vacin_filename, "r")

    ntor    = int(vacin['ntor'][()])
    mpol    = int(vacin['mpol'][()])
    nzeta   = int(vacin['nzeta'][()])
    ntheta  = int(vacin['ntheta'][()])
    NFP     = int(vacin['nfp'][()])
    sym     = bool(vacin['lasym__logical__'][()] == 0)
    
    rbtor   = vacin['rbtor'][()]
    ctor    = vacin['ctor'][()]
    signgs  = vacin['signgs'][()]

    raxis   = vacin['raxis_nestor'][()]
    zaxis   = vacin['zaxis_nestor'][()]
    wint    = np.array(vacin['wint'][()])

    xm      = vacin['xm'][()]
    xn      = vacin['xn'][()]
    rmnc    = vacin['rmnc'][()]
    zmns    = vacin['zmns'][()]
    
    extcur = vacin['extcur'][()]        
    folder = os.getcwd()
    mgridFilename = os.path.join(folder, mgrid)
    ext_field = SplineMagneticField.from_mgrid(mgridFilename, extcur)

    mf = mpol+1
    nf = ntor

    nestor = Nestor(ext_field, signgs, mf, nf, ntheta, nzeta, NFP)                    

    if method == "vmec":
        surface_coords = eval_surface_geometry_vmec(xm, xn, ntheta, nzeta, NFP, rmnc, zmns, sym=sym)
    elif method == "desc":
        grid = LinearGrid(rho=1, M=ntheta, N=nzeta, NFP=NFP)
        mr, nr, R_mn = ptolemy_identity_fwd(xm, xn//NFP, np.zeros_like(rmnc), rmnc)
        mz, nz, Z_mn = ptolemy_identity_fwd(xm, xn//NFP, zmns, np.zeros_like(zmns))
        R_mn = R_mn[0]
        Z_mn = Z_mn[0]
        modes_R = np.array([np.zeros_like(mr), mr, nr]).T
        modes_Z = np.array([np.zeros_like(mz), mz, nz]).T
        R_basis = DoubleFourierSeries(M=np.max(abs(mr)), N=np.max(abs(nr)), NFP=NFP, sym="cos" if sym else False)
        Z_basis = DoubleFourierSeries(M=np.max(abs(mz)), N=np.max(abs(nz)), NFP=NFP, sym="sin" if sym else False)
        R_mn = copy_coeffs(R_mn, modes_R, R_basis.modes)
        Z_mn = copy_coeffs(Z_mn, modes_Z, Z_basis.modes)
        R_transform = Transform(grid, R_basis, derivs=2)
        Z_transform = Transform(grid, Z_basis, derivs=2)
        surface_coords = eval_surface_geometry(R_mn, Z_mn, R_transform, Z_transform, ntheta, nzeta, NFP, sym)
    axis_coords = evaluate_axis_vmec(raxis, zaxis, nzeta, NFP)
    phi_mn, Btot = nestor.compute(surface_coords, axis_coords, ctor/mu0)
    firstIterationPrintout(Btot, ctor, rbtor, nestor.signgs, nestor.mf, nestor.nf, nestor.ntheta, nestor.nzeta, nestor.NFP, nestor.weights)
    print(np.linalg.norm(Btot["Bn"]))

    if vacout_filename is None:
        vacout_filename = vacin_filename.replace("vacin_", "vacout_")
    produceOutputFile(vacout_filename, phi_mn, Btot, nestor.mf, nestor.nf, nestor.ntheta, nestor.nzeta, nestor.NFP)
    



def test_same_outputs(tmpdir_factory):

    output_dir = tmpdir_factory.mktemp("nestor_result")
    for iteration in range(maxIter):

        ref_in = os.path.join(ref_in_folder, "vacin_%s_%06d.nc"%(runId, iteration))
        ref_out = os.path.join(ref_out_folder, "vacout_%s_%06d.nc"%(runId, iteration))
        if not os.path.isfile(ref_in):
            raise RuntimeError("reference %s not found"%(ref_in,))
        if not os.path.isfile(ref_out):
            raise RuntimeError("reference %s not found"%(ref_out,))
        
        vmec_tst_fname = output_dir.join("vacout_%s_%06d_vmec.nc"%(runId, iteration))
        desc_tst_fname = output_dir.join("vacout_%s_%06d_desc.nc"%(runId, iteration))
        mgrid = here + "/inputs/nestor/mgrid_test.nc"
        main(ref_in, vmec_tst_fname, mgrid, "vmec")
        main(ref_in, desc_tst_fname, mgrid, "desc")        
        if not os.path.isfile(vmec_tst_fname):
            raise RuntimeError("test %s not found"%(vmec_tst_fname,))
        if not os.path.isfile(desc_tst_fname):
            raise RuntimeError("test %s not found"%(desc_tst_fname,))
        

        ref_data = {}
        d = Dataset(ref_out, "r")
        for key in d.variables:
            ref_data[key] = d[key][()]
        d.close()
            
        vmec_tst_data = {}
        d = Dataset(vmec_tst_fname, "r")
        for key in d.variables:
            vmec_tst_data[key] = d[key][()]
        d.close()

        desc_tst_data = {}
        d = Dataset(vmec_tst_fname, "r")
        for key in d.variables:
            desc_tst_data[key] = d[key][()]
        d.close()
        
        # compare data
        for key in desc_tst_data:
            r = ref_data[key]
            v = vmec_tst_data[key]
            d = desc_tst_data[key]

            np.testing.assert_allclose(r,v, rtol=1e-10, atol=1e-4, err_msg="vmec iter={}, key={}".format(iteration, key))
            np.testing.assert_allclose(r,d, rtol=1e-10, atol=1e-4, err_msg="desc iter={}, key={}".format(iteration, key))
