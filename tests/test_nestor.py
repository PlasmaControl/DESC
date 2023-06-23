#!/usr/bin/env python3
"""
Created on Mon Apr 12 21:14:09 2021

@author: jonathan
"""

import os
import unittest

import numpy as np
from netCDF4 import Dataset
from scipy.constants import mu_0

from desc.basis import DoubleFourierSeries, FourierSeries
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid
from desc.magnetic_fields import SplineMagneticField
from desc.nestor import Nestor, firstIterationPrintout
from desc.transform import Transform
from desc.utils import copy_coeffs
from desc.vmec_utils import ptolemy_identity_fwd

here = os.path.abspath(os.path.dirname(__file__))

runId = "test.vmec"
maxIter = 11


ref_in_folder = here + "/inputs/nestor/ref_in/"
ref_out_folder = here + "/inputs/nestor/ref_out/"


def def_ncdim(ncfile, size):
    dimname = "dim_%05d" % (size,)
    ncfile.createDimension(dimname, size)
    return dimname


def produceOutputFile(vacoutFilename, potvac, Btot, mf, nf, ntheta, nzeta, NFP):
    # mode numbers for potvac
    xmpot = np.zeros([(mf + 1) * (2 * nf + 1)])
    xnpot = np.zeros([(mf + 1) * (2 * nf + 1)])
    mn = 0
    for n in range(-nf, nf + 1):
        for m in range(mf + 1):
            xmpot[mn] = m
            xnpot[mn] = n * NFP
            mn += 1

    vacout = Dataset(vacoutFilename, "w")

    dim_nuv2 = def_ncdim(vacout, (ntheta // 2 + 1) * nzeta)
    dim_mnpd2 = def_ncdim(vacout, (mf + 1) * (2 * nf + 1))

    var_bsqvac = vacout.createVariable("bsqvac", "f8", (dim_nuv2,))
    var_mnpd = vacout.createVariable("mnpd", "i4")
    var_mnpd2 = vacout.createVariable("mnpd2", "i4")
    var_xmpot = vacout.createVariable("xmpot", "f8", (dim_mnpd2,))
    var_xnpot = vacout.createVariable("xnpot", "f8", (dim_mnpd2,))
    var_potvac = vacout.createVariable("potvac", "f8", (dim_mnpd2,))
    var_brv = vacout.createVariable("brv", "f8", (dim_nuv2,))
    var_bphiv = vacout.createVariable("bphiv", "f8", (dim_nuv2,))
    var_bzv = vacout.createVariable("bzv", "f8", (dim_nuv2,))

    var_bsqvac[:] = Btot["|B|^2"]
    var_mnpd.assignValue((mf + 1) * (2 * nf + 1))
    var_mnpd2.assignValue((mf + 1) * (2 * nf + 1))
    var_xmpot[:] = xmpot
    var_xnpot[:] = xnpot
    var_potvac[:] = np.fft.fftshift(
        potvac.reshape([mf + 1, 2 * nf + 1]), axes=1
    ).T.flatten()
    var_brv[:] = Btot["BR"]
    var_bphiv[:] = Btot["Bphi"]
    var_bzv[:] = Btot["BZ"]

    vacout.close()


def nestor_to_eq(vacin):

    ntor = int(vacin["ntor"][()])
    mpol = int(vacin["mpol"][()])
    nzeta = int(vacin["nzeta"][()])
    ntheta = int(vacin["ntheta"][()])
    NFP = int(vacin["nfp"][()])
    sym = bool(vacin["lasym__logical__"][()] == 0)

    raxis = vacin["raxis_nestor"][()]
    zaxis = vacin["zaxis_nestor"][()]
    wint = np.array(vacin["wint"][()])

    xm = vacin["xm"][()]
    xn = vacin["xn"][()]
    rmnc = vacin["rmnc"][()]
    zmns = vacin["zmns"][()]

    bdry_grid = LinearGrid(rho=1, theta=ntheta, zeta=nzeta, NFP=NFP)
    mr, nr, Rb_mn = ptolemy_identity_fwd(xm, xn // NFP, np.zeros_like(rmnc), rmnc)
    mz, nz, Zb_mn = ptolemy_identity_fwd(xm, xn // NFP, zmns, np.zeros_like(zmns))
    M = max(np.max(abs(mr)), np.max(abs(mz)))
    N = max(np.max(abs(nr)), np.max(abs(nz)))
    Rb_mn = Rb_mn[0]
    Zb_mn = Zb_mn[0]
    modes_Rb = np.array([np.zeros_like(mr), mr, nr]).T
    modes_Zb = np.array([np.zeros_like(mz), mz, nz]).T

    a_basis = FourierSeries(N=N, NFP=NFP, sym=False)
    axis_grid = LinearGrid(rho=0, theta=0, zeta=nzeta, NFP=NFP)
    a_transform = Transform(axis_grid, a_basis)
    a_transform.build_pinv()
    Ra_n = a_transform.fit(raxis)
    Za_n = a_transform.fit(zaxis)

    temp_basis = DoubleFourierSeries(M=M, N=N, NFP=NFP, sym=False)
    Rb_lmn = copy_coeffs(Rb_mn, modes_Rb, temp_basis.modes).reshape((-1, 1))
    Zb_lmn = copy_coeffs(Zb_mn, modes_Zb, temp_basis.modes).reshape((-1, 1))
    boundary = np.hstack([temp_basis.modes, Rb_lmn, Zb_lmn])
    axis = np.hstack([a_basis.modes[:, 2:], Ra_n[:, np.newaxis], Za_n[:, np.newaxis]])
    profiles = np.array([0, 0, 0]).reshape((1, 3))
    eq = {
        "L": 2 * M,
        "M": M,
        "N": N,
        "profiles": profiles,
        "surface": boundary,
        "NFP": NFP,
        "Psi": 1.0,
        "axis": axis,
        "sym": sym,
    }
    eq = Equilibrium(**eq)
    return eq


def main(vacin_filename, vacout_filename=None, mgrid=None):
    vacin = Dataset(vacin_filename, "r")

    ntor = int(vacin["ntor"][()])
    mpol = int(vacin["mpol"][()])
    nzeta = int(vacin["nzeta"][()])
    ntheta = int(vacin["ntheta"][()])
    NFP = int(vacin["nfp"][()])
    sym = bool(vacin["lasym__logical__"][()] == 0)

    rbtor = vacin["rbtor"][()]
    ctor = vacin["ctor"][()]
    signgs = vacin["signgs"][()]

    extcur = vacin["extcur"][()]
    folder = os.getcwd()
    mgridFilename = os.path.join(folder, mgrid)
    ext_field = SplineMagneticField.from_mgrid(mgridFilename, extcur)

    mf = mpol + 1
    nf = ntor

    eq = nestor_to_eq(vacin)
    nestor = Nestor(eq, ext_field, mf, nf, ntheta, nzeta)
    phi_mn, Btot = nestor.compute(eq.R_lmn, eq.Z_lmn, ctor / mu_0)
    firstIterationPrintout(
        Btot,
        ctor,
        rbtor,
        nestor.signgs,
        nestor.M,
        nestor.N,
        nestor.ntheta,
        nestor.nzeta,
        nestor.NFP,
        nestor.weights,
    )
    print(np.linalg.norm(Btot["Bn"]))

    if vacout_filename is None:
        vacout_filename = vacin_filename.replace("vacin_", "vacout_")
    produceOutputFile(
        vacout_filename,
        phi_mn,
        Btot,
        nestor.M,
        nestor.N,
        nestor.ntheta,
        nestor.nzeta,
        nestor.NFP,
    )


def test_same_outputs(tmpdir_factory):

    output_dir = tmpdir_factory.mktemp("nestor_result")
    for iteration in range(maxIter):

        ref_in = os.path.join(ref_in_folder, "vacin_%s_%06d.nc" % (runId, iteration))
        ref_out = os.path.join(ref_out_folder, "vacout_%s_%06d.nc" % (runId, iteration))
        if not os.path.isfile(ref_in):
            raise RuntimeError("reference {} not found".format(ref_in))
        if not os.path.isfile(ref_out):
            raise RuntimeError("reference {} not found".format(ref_out))

        desc_tst_fname = output_dir.join("vacout_%s_%06d_desc.nc" % (runId, iteration))
        mgrid = here + "/inputs/nestor/mgrid_test.nc"
        main(ref_in, desc_tst_fname, mgrid)
        if not os.path.isfile(desc_tst_fname):
            raise RuntimeError("test {} not found".format(desc_tst_fname))

        ref_data = {}
        d = Dataset(ref_out, "r")
        for key in d.variables:
            ref_data[key] = d[key][()]
        d.close()

        desc_tst_data = {}
        d = Dataset(desc_tst_fname, "r")
        for key in d.variables:
            desc_tst_data[key] = d[key][()]
        d.close()

        # compare data
        for key in desc_tst_data:
            r = ref_data[key]
            d = desc_tst_data[key]

            np.testing.assert_allclose(
                r,
                d,
                rtol=1e-10,
                atol=1e-4,
                err_msg="desc iter={}, key={}".format(iteration, key),
            )
