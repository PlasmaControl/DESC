#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 13:15:05 2021

Neumann Solver for Toroidal Systems

@author: Jonathan Schilling (jonathan.schilling@ipp.mpg.de)
"""

import os
import sys

import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

from netCDF4 import Dataset

mu0 = 4.0e-7*np.pi

def def_ncdim(ncfile, size):
    dimname = "dim_%05d"%(size,)
    ncfile.createDimension(dimname, size)
    return dimname

# fast evaluation of external (coils-produced) magnetic field by bi-linear interpolation
class MGridFile:

    # filename of the mgrid file
    mgridFilename = None

    # number of radial grid knots
    ir = None

    # number of vertical grid knots
    jz = None

    # number of toroidal grid knots
    kp = None

    # number of field periods
    nfp = None

    # number of external currents/coil sets
    nextcur = None

    # coil currents defined in mgrid file (?)
    cur = None

    # normalization mode for magnetic field: "raw" or "scaled"
    mode = None

    # minimum grid position in radial direction
    rMin = None

    # maximum grid position in radial direction
    rMax = None

    # minimum grid position in vertical direction
    zMin = None

    # maximum grid position in vertical direction
    zMax = None

    # cylindrical magnetic field component B^R on grid
    br = None

    # cylindrical magnetic field component B^\phi on grid
    bp = None

    # cylindrical magnetic field component B^Z on grid
    bz = None

    # coil currents used to scale contributions from different coils
    extcur = None

    # grid knots in radial direction
    rGrid = None

    # grid knots in vertical direction
    zGrid = None

    # interpolator objects (one for each poloidal cut) for cylindrical magnetic field component B^R on grid
    brInterp = None

    # interpolator objects (one for each poloidal cut) for cylindrical magnetic field component B^\phi on grid
    bpInterp = None

    # interpolator objects (one for each poloidal cut) for cylindrical magnetic field component B^Z on grid
    bzInterp = None

    # Create a new MGridFile object and load the data from disk.
    # Also scale the magnetic field contributions due to different coils
    # according to the currents given in extcur.
    def __init__(self, mgridFilename, extcur):
        self.mgridFilename = mgridFilename
        self.extcur = extcur
        self.load()

    # Load the mgrid data from disk.
    # Also scale the magnetic field contributions due to different coils
    # according to the currents given in extcur.
    def load(self):
        mgrid = Dataset(self.mgridFilename, "r")
        self.ir = int(mgrid['ir'][()])
        self.jz = int(mgrid['jz'][()])
        self.kp = int(mgrid['kp'][()])
        self.nfp = mgrid['nfp'][()].data
        self.nextcur = int(mgrid['nextcur'][()])
        self.cur = mgrid['raw_coil_cur'][()]
        self.rMin = mgrid['rmin'][()]
        self.rMax = mgrid['rmax'][()]
        self.zMin = mgrid['zmin'][()]
        self.zMax = mgrid['zmax'][()]

        mgrid_mode = mgrid['mgrid_mode'][()]
        self.mode = bytearray(mgrid_mode).decode('utf-8')

        self.br = np.zeros([self.kp, self.jz, self.ir])
        self.bp = np.zeros([self.kp, self.jz, self.ir])
        self.bz = np.zeros([self.kp, self.jz, self.ir])
        for i in range(self.nextcur):

            # apply scaling by currents given in VMEC input file
            scale = self.extcur[i]
            if mgrid_mode == 'S':
                scale *= 1.0/self.mgrid_cur[i]
            # elif mgrid_mode == 'R':
            else: # 'R' or nothing seems to be equal
                pass
            # else:
            #     raise RuntimeError("mgrid mode '"+mgrid_mode+"' not recognized; expecting 'R' or 'S'")

            # sum up contributions from different coils
            coil_id = "%03d"%(i+1,)
            self.br[:,:,:] += np.multiply(mgrid['br_'+coil_id][()], scale)
            self.bp[:,:,:] += np.multiply(mgrid['bp_'+coil_id][()], scale)
            self.bz[:,:,:] += np.multiply(mgrid['bz_'+coil_id][()], scale)
        mgrid.close()

        # re-compute grid knots in radial and vertical direction
        self.rGrid = np.linspace(self.rMin, self.rMax, self.ir)
        self.zGrid = np.linspace(self.zMin, self.zMax, self.jz)
        self.pGrid = 2.0*np.pi/(self.nfp*self.kp) * np.arange(self.kp)
        # setup interpolations for mgrid file
        self.brInterp = []
        self.bpInterp = []
        self.bzInterp = []
        for k in range(self.kp):
            # I think kx=1, ky=1 makes this a bi-linear interpolation just as is used in NESTOR/VMEC
            self.brInterp.append(RectBivariateSpline(self.rGrid, self.zGrid, self.br[k,:,:].T, kx=1, ky=1))
            self.bpInterp.append(RectBivariateSpline(self.rGrid, self.zGrid, self.bp[k,:,:].T, kx=1, ky=1))
            self.bzInterp.append(RectBivariateSpline(self.rGrid, self.zGrid, self.bz[k,:,:].T, kx=1, ky=1))

    # debugging plot: show log10(|radial magnetic field component|)
    def plotBr(self):
        for k in range(self.kp):
            plt.figure()
            plt.pcolormesh(np.log10(np.abs(self.br[k,:,:])))
            cb = plt.colorbar()
            cb.set_label("B^R / T")
            plt.title("toroidal index %d / %d"%(k+1, self.kp))

    # evaluate MGRID on grid over flux surface
    def interpB(self, R, Z, phi):
        
        brad = np.zeros_like(R)
        bphi = np.zeros_like(phi)
        bz   = np.zeros_like(Z)
        for k, pp in enumerate(self.pGrid):
            
            brad = np.where(phi == pp, self.brInterp[k].ev(R, Z), brad)
            bphi = np.where(phi == pp, self.bpInterp[k].ev(R, Z), bphi)
            bz = np.where(phi == pp, self.bzInterp[k].ev(R, Z), bz)

        return brad, bphi, bz

def copy_vector_periods(vec, zetas):
    """Copies a vector into each field period by rotation

    Parameters
    ----------
    vec : ndarray, shape(3,...)
        vector(s) to rotate
    zetas : ndarray
        angles to rotate by (eg start of each field period)

    Returns
    -------
    vec : ndarray, shape(3,...,nzeta)
        vector(s) repeated and rotated by angle zeta
    """
    if vec.shape[0] == 3:
        x,y,z = vec
    else:
        x,y = vec
    shp = x.shape
    xx = x.reshape((*shp, 1)) * np.cos(zetas) - y.reshape((*shp, 1))*np.sin(zetas)
    yy = y.reshape((*shp, 1)) * np.cos(zetas) + x.reshape((*shp, 1))*np.sin(zetas)
    if vec.shape[0] == 3:
        zz = np.broadcast_to(z.reshape((*shp,1)),(*shp, zetas.size))
        return np.array((xx,yy,zz))
    return np.array((xx,yy))
    
# Neumann Solver for Toroidal Systems
class Nestor:

    # number of calls to vacuum() subroutine from VMEC
    # == NESTOR iteration counter
    vacuum_calls = None

    # error flag from/to VMEC
    ier_flag = None

    # skip counter --> only do full NESTOR calculation every nvacskip iterations
    ivacskip = None

    # 0,1,2; depending on initialization status of NESTOR in VMEC
    ivac = None

    # number of field periods
    nfp = None

    # number of toroidal Fourier harmonics in geometry input
    ntor = None

    # number of poloidal Fourier harmonics in geometry input
    mpol = None

    # number of toroidal grid points; has to match mgrid file!
    nzeta = None

    # number of poloidal grid points
    ntheta = None

    # total number of Fourier coefficients in geometry input
    mnmax = None

    # poloidal mode numbers m of geometry input
    xm = None

    # toroidal mode numbers n*nfp of geometry input
    xn = None

    # Fourier coefficients for R*cos(m theta - n zeta) of geometry input
    rmnc = None

    # Fourier coefficients for Z*sin(m theta - n zeta) of geometry input
    zmns = None

    # net poloidal current; only used for comparison
    rbtor = None

    # net toroial current in A*mu0; used for filament model along magnetic axis
    ctor = None

    # flag to indicate non-stellarator-symmetry mode
    lasym = None

    # sign of Jacobian; needed for surface normal vector sign
    signgs = None

    # coil currents for scaling mgrid file
    extcur = None

    # toroidal Fourier coefficients for magnetic axis: R*cos(n zeta)
    raxis_nestor = None

    # toroidal Fourier coefficients for magnetic axis: Z*sin(n zeta)
    zaxis_nestor = None

    # normalization factor for surface integrals;
    # essentially 1/(ntheta*nzeta) with 1/2 at the ends in the poloidal direction
    wint = None

    # bvec from previous iteration to be used when skipping full NESTOR calculation
    bvecsav = None

    # amat from previous iteration to be used when skipping full NESTOR calculation
    amatsav = None

    # size of bvec (1d) and amat (2d)
    mnpd2 = None

    # poloidal current (?) from previous iteration; has to be carried over for use in VMEC
    bsubvvac = None

    # MGridFile object holding the external magnetic field to interpolate
    mgrid = None

    def __init__(self, vacinFilename):
        self.load(vacinFilename)

    # read input file for NESTOR
    def load(self, vacinFilename):
        vacin = Dataset(vacinFilename, "r")
        self.vacuum_calls    = int(vacin['vacuum_calls'][()])
        self.ier_flag        = int(vacin['ier_flag'][()])
        mgrid_file           = vacin['mgrid_file'][()]
        input_extension      = vacin['input_extension'][()]
        self.ivacskip        = int(vacin['ivacskip'][()])
        self.ivac            = int(vacin['ivac'][()])
        self.nfp             = int(vacin['nfp'][()])
        self.ntor            = int(vacin['ntor'][()])
        self.mpol            = int(vacin['mpol'][()])
        self.nzeta           = int(vacin['nzeta'][()])
        self.ntheta          = int(vacin['ntheta'][()])
        self.mnmax           = int(vacin['mnmax'][()])
        self.xm              = vacin['xm'][()]
        self.xn              = vacin['xn'][()]
        self.rmnc            = vacin['rmnc'][()]
        self.zmns            = vacin['zmns'][()]
        self.rbtor           = vacin['rbtor'][()]
        self.ctor            = vacin['ctor'][()]
        self.lasym           = (vacin['lasym__logical__'][()] != 0)
        self.signgs          = vacin['signgs'][()]
        self.extcur          = vacin['extcur'][()]
        self.raxis_nestor    = vacin['raxis_nestor'][()]
        self.zaxis_nestor    = vacin['zaxis_nestor'][()]
        self.wint            = np.array(vacin['wint'][()])
        self.bvecsav         = vacin['bvecsav'][()]
        self.amatsav         = vacin['amatsav'][()]
        self.mnpd2           = int(vacin['mnpd2'][()])
        self.bsubvvac        = vacin['bsubvvac'][()]
        vacin.close()

        self.mgrid_file = bytearray(mgrid_file).decode('utf-8')
        self.input_extension = bytearray(input_extension).decode('utf-8')

    # pre-computable quantities and arrays
    def precompute(self):
        self.mf = self.mpol+1
        self.nf = self.ntor

        if self.nzeta == 1:
            self.nvper = 64
        else:
            self.nvper = self.nfp

        self.nvp = self.nzeta * self.nvper

        # toroidal extent of one module
        dPhi_per = 2.0*np.pi/self.nfp

        # cos, sin for starting points of toroidal modules
        self.arg_per = 2.0*np.pi/self.nvper * np.arange(self.nvper)

        # cos, sin args for going from cylindrical to cartesian coordinates
        self.cosv = np.cos(2.0*np.pi/(self.nfp*self.nzeta) * np.arange(self.nzeta))
        self.sinv = np.sin(2.0*np.pi/(self.nfp*self.nzeta) * np.arange(self.nzeta))
        self.phiaxis = 2.0*np.pi/(self.nfp*self.nzeta) * np.arange(self.nzeta)

        
        # tanu, tanv
        epstan = 2.22e-16
        bigno = 1.0e50 # allows proper comparison against implementation used in VMEC
        #bigno = np.inf # allows proper plotting

        self.tanu = 2.0*np.tan( np.pi*np.arange(2*self.ntheta)/self.ntheta )
        # mask explicit singularities at tan(pi/2), tan(3/2 pi)
        self.tanu[np.where( (np.arange(2*self.ntheta)/self.ntheta-0.5)%1 < epstan )] = bigno

        if self.nzeta == 1:
            # Tokamak: need nvper toroidal grid points
            argv = np.arange(self.nvper)/self.nvper
        else:
            # Stellarator: need nzeta toroidal grdi points
            argv = np.arange(self.nzeta)/self.nzeta
        self.tanv = 2.0*np.tan( np.pi*argv )
        # mask explicit singularities at tan(pi/2)
        self.tanv[np.where( (argv-0.5)%1 < epstan )] = bigno

        self.cmn = np.zeros([self.mf+self.nf+1, self.mf+1, self.nf+1])
        for m in range(self.mf+1):
            for n in range(self.nf+1):
                jmn = m+n
                imn = m-n
                kmn = abs(imn)
                smn = (jmn+kmn)/2
                f1 = 1
                f2 = 1
                f3 = 1
                for i in range(1, kmn+1):
                    f1 *= (smn-(i-1))
                    f2 *= i
                for l in range(kmn, jmn+1, 2):
                    self.cmn[l,m,n] = f1/(f2*f3)*((-1)**((l-imn)/2))
                    f1 *= (jmn+l+2)*(jmn-l)/4
                    f2 *= (l+2+kmn)/2
                    f3 *= (l+2-kmn)/2

        # cmns from cmn
        self.cmns = np.zeros([self.mf+self.nf+1, self.mf+1, self.nf+1])
        for m in range(1, self.mf+1):
            for n in range(1, self.nf+1):
                self.cmns[:,m,n] = 0.5*dPhi_per*(self.cmn[:,m,n] + self.cmn[:, m-1, n] + self.cmn[:, m, n-1] + self.cmn[:, m-1, n-1])
        self.cmns[:,1:self.mf+1,0] = 0.5*dPhi_per*(self.cmn[:,1:self.mf+1,0]+self.cmn[:,0:self.mf,0])
        self.cmns[:,0,1:self.nf+1] = 0.5*dPhi_per*(self.cmn[:,0,1:self.nf+1]+self.cmn[:,0,0:self.nf])
        self.cmns[:,0,0] = 0.5*dPhi_per*(self.cmn[:,0,0]+self.cmn[:,0,0])

        # integer mode number arrays
        ixm=np.array(np.round(self.xm), dtype=int)
        ixn=np.array(np.round(np.divide(self.xn, self.nfp)), dtype=int)

        # Fourier mode sorting array
        # for a given mn in 0, 1, ..., (mnmax-1), it tells you at which position in the FFT input array this coefficient goes
        self.mIdx = np.zeros([self.mnmax], dtype=int)
        self.nIdx = np.zeros([self.mnmax], dtype=int)
        for mn in range(self.mnmax):
            m = ixm[mn]
            n = ixn[mn]

            # m from VMEC is always positive
            self.mIdx[mn] = m

            # reverse toroidal mode numbers, since VMEC kernel (mu-nv) is reversed in n
            if n<=0:
                idx_n = -n
            else:
                idx_n = self.nzeta - n
            self.nIdx[mn] = idx_n

    #eval surface geometry
    def evalSurfaceGeometry(self):

        self.ntheta_even = 2*int(self.ntheta/2)
        self.ntheta_stellsym = int(self.ntheta_even/2)+1

        self.nzeta_even = 2*int(self.nzeta/2)
        self.nzeta_stellsym = int(self.nzeta_even/2)+1

        # input arrays for FFTs
        Rmn   = np.zeros([self.ntheta, self.nzeta], dtype=np.complex128) # for R
        mRmn  = np.zeros([self.ntheta, self.nzeta], dtype=np.complex128) # for dRdTheta
        nRmn  = np.zeros([self.ntheta, self.nzeta], dtype=np.complex128) # for dRdZeta
        mmRmn = np.zeros([self.ntheta, self.nzeta], dtype=np.complex128) # for d2RdTheta2
        mnRmn = np.zeros([self.ntheta, self.nzeta], dtype=np.complex128) # for d2RdThetaZeta
        nnRmn = np.zeros([self.ntheta, self.nzeta], dtype=np.complex128) # for d2RdZeta2
        Zmn   = np.zeros([self.ntheta, self.nzeta], dtype=np.complex128) # for Z
        mZmn  = np.zeros([self.ntheta, self.nzeta], dtype=np.complex128) # for dZdTheta
        nZmn  = np.zeros([self.ntheta, self.nzeta], dtype=np.complex128) # for dZdZeta
        mmZmn = np.zeros([self.ntheta, self.nzeta], dtype=np.complex128) # for d2ZdTheta2
        mnZmn = np.zeros([self.ntheta, self.nzeta], dtype=np.complex128) # for d2ZdThetaZeta
        nnZmn = np.zeros([self.ntheta, self.nzeta], dtype=np.complex128) # for d2ZdZeta2

        # multiply with mode numbers to get tangential derivatives
        # for mn in range(self.mnmax):
        Rmn  [self.mIdx, self.nIdx] =                   self.rmnc
        mRmn [self.mIdx, self.nIdx] =  -        self.xm*self.rmnc
        nRmn [self.mIdx, self.nIdx] =           self.xn*self.rmnc
        mmRmn[self.mIdx, self.nIdx] =  -self.xm*self.xm*self.rmnc
        mnRmn[self.mIdx, self.nIdx] =   self.xm*self.xn*self.rmnc
        nnRmn[self.mIdx, self.nIdx] =  -self.xn*self.xn*self.rmnc
        Zmn  [self.mIdx, self.nIdx] =  -                self.zmns*1j
        mZmn [self.mIdx, self.nIdx] =           self.xm*self.zmns*1j
        nZmn [self.mIdx, self.nIdx] =  -        self.xn*self.zmns*1j
        mmZmn[self.mIdx, self.nIdx] =   self.xm*self.xm*self.zmns*1j
        mnZmn[self.mIdx, self.nIdx] =  -self.xm*self.xn*self.zmns*1j
        nnZmn[self.mIdx, self.nIdx] =   self.xn*self.xn*self.zmns*1j
        # TODO: if lasym, must also include corresponding terms above!

        R_2d             = (np.fft.ifft2(  Rmn)*self.ntheta*self.nzeta).real
        dRdTheta_2d      = (np.fft.ifft2( mRmn)*self.ntheta*self.nzeta).imag
        dRdZeta_2d       = (np.fft.ifft2( nRmn)*self.ntheta*self.nzeta).imag
        d2RdTheta2_2d    = (np.fft.ifft2(mmRmn)*self.ntheta*self.nzeta).real
        d2RdThetaZeta_2d = (np.fft.ifft2(mnRmn)*self.ntheta*self.nzeta).real
        d2RdZeta2_2d     = (np.fft.ifft2(nnRmn)*self.ntheta*self.nzeta).real
        Z_2d             = (np.fft.ifft2(  Zmn)*self.ntheta*self.nzeta).real
        dZdTheta_2d      = (np.fft.ifft2( mZmn)*self.ntheta*self.nzeta).imag
        dZdZeta_2d       = (np.fft.ifft2( nZmn)*self.ntheta*self.nzeta).imag
        d2ZdTheta2_2d    = (np.fft.ifft2(mmZmn)*self.ntheta*self.nzeta).real
        d2ZdThetaZeta_2d = (np.fft.ifft2(mnZmn)*self.ntheta*self.nzeta).real
        d2ZdZeta2_2d     = (np.fft.ifft2(nnZmn)*self.ntheta*self.nzeta).real

        coords = {}
        # vectorize arrays, since most operations to follow act on all grid points anyway
        coords["R"] = R_2d.flatten()
        coords["Z"] = Z_2d.flatten()
        if self.lasym:
            coords["R_sym"]         = coords["R"]
            coords["Z_sym"]         = coords["Z"]
            coords["dRdTheta"]      = dRdTheta_2d.flatten()
            coords["dRdZeta"]       = dRdZeta_2d.flatten()
            coords["d2RdTheta2"]    = d2RdTheta2_2d.flatten()
            coords["d2RdThetaZeta"] = d2RdThetaZeta_2d.flatten()
            coords["d2RdZeta2"]     = d2RdZeta2_2d.flatten()
            coords["dZdTheta"]      = dZdTheta_2d.flatten()
            coords["dZdZeta"]       = dZdZeta_2d.flatten()
            coords["d2ZdTheta2"]    = d2ZdTheta2_2d.flatten()
            coords["d2ZdThetaZeta"] = d2ZdThetaZeta_2d.flatten()
            coords["d2ZdZeta2"]     = d2ZdZeta2_2d.flatten()
        else:
            coords["R_sym"]         = R_2d            [:self.ntheta_stellsym,:].flatten()
            coords["Z_sym"]         = Z_2d            [:self.ntheta_stellsym,:].flatten()
            coords["dRdTheta"]      = dRdTheta_2d     [:self.ntheta_stellsym,:].flatten()
            coords["dRdZeta"]       = dRdZeta_2d      [:self.ntheta_stellsym,:].flatten()
            coords["d2RdTheta2"]    = d2RdTheta2_2d   [:self.ntheta_stellsym,:].flatten()
            coords["d2RdThetaZeta"] = d2RdThetaZeta_2d[:self.ntheta_stellsym,:].flatten()
            coords["d2RdZeta2"]     = d2RdZeta2_2d    [:self.ntheta_stellsym,:].flatten()
            coords["dZdTheta"]      = dZdTheta_2d     [:self.ntheta_stellsym,:].flatten()
            coords["dZdZeta"]       = dZdZeta_2d      [:self.ntheta_stellsym,:].flatten()
            coords["d2ZdTheta2"]    = d2ZdTheta2_2d   [:self.ntheta_stellsym,:].flatten()
            coords["d2ZdThetaZeta"] = d2ZdThetaZeta_2d[:self.ntheta_stellsym,:].flatten()
            coords["d2ZdZeta2"]     = d2ZdZeta2_2d    [:self.ntheta_stellsym,:].flatten()

        coords["phi_sym"] = np.broadcast_to(self.phiaxis, (self.ntheta_stellsym, self.nzeta)).flatten()

        coords["X"] = (R_2d * self.cosv).flatten()
        coords["Y"] = (R_2d * self.sinv).flatten()     
        return coords
    
    def compute_jacobian(self, coords):

        jacobian = {}        
        # compute metric coefficients and surface normal components
        jacobian["surfNormR"]   =  self.signgs * (coords["R_sym"] * coords["dZdTheta"])
        jacobian["surfNormPhi"] =  self.signgs * (coords["dRdTheta"] * coords["dZdZeta"] - coords["dRdZeta"] * coords["dZdTheta"])
        jacobian["surfNormZ"]   = -self.signgs * (coords["R_sym"] * coords["dRdTheta"])

        # a, b, c in NESTOR article: dot-products of first-order derivatives of surface
        jacobian["g_uu"] = (coords["dRdTheta"] * coords["dRdTheta"] + coords["dZdTheta"] * coords["dZdTheta"])
        jacobian["g_uv"] = (coords["dRdTheta"] * coords["dRdZeta"]  + coords["dZdTheta"] * coords["dZdZeta"])/self.nfp
        jacobian["g_vv"] = (coords["dRdZeta"]  * coords["dRdZeta"]  + coords["dZdZeta"]  * coords["dZdZeta"] + coords["R_sym"] * coords["R_sym"])/self.nfp**2
        
        # A, B and C in NESTOR article: surface normal dotted with second-order derivative of surface (?)
        jacobian["a_uu"]   = 0.5 * (jacobian["surfNormR"] * coords["d2RdTheta2"] + jacobian["surfNormZ"] * coords["d2ZdTheta2"])
        jacobian["a_uv"] = (jacobian["surfNormR"] * coords["d2RdThetaZeta"] + jacobian["surfNormPhi"] * coords["dRdTheta"] + jacobian["surfNormZ"] * coords["d2ZdThetaZeta"])/self.nfp
        jacobian["a_vv"]   = (jacobian["surfNormPhi"] * coords["dRdZeta"] + 0.5*(jacobian["surfNormR"] * (coords["d2RdZeta2"] - coords["R_sym"]) + jacobian["surfNormZ"] * coords["d2ZdZeta2"]) )/self.nfp**2

        return jacobian
    
    # read mgrid file; only need to do this once!
    def loadMGridFile(self, mgrid_file=None):
        folder = os.getcwd()
        if mgrid_file is None:
            mgrid_file = self.mgrid_file
        mgridFilename = os.path.join(folder, mgrid_file)
        self.mgrid = MGridFile(mgridFilename, self.extcur)

    # evaluate MGRID on grid over flux surface
    def interpolateMGridFile(self, R, Z, phi):
        brad, bphi, bz = self.mgrid.interpB(R, Z, phi)

        return np.array([brad,
                         bphi,
                         bz])

    # model net toroidal plasma current as filament along the magnetic axis
    # and add its magnetic field on the LCFS to the MGRID magnetic field
    def modelNetToroidalCurrent(self, raxis, phiaxis, zaxis, current, R_sym, phi_sym, Z_sym):

        # TODO: we can simplify this by evaluating the field directly in cylindrical coordinates
        # copy 1 field period around to make full torus
        xyz = np.array([raxis*np.cos(phiaxis),
                        raxis*np.sin(phiaxis),
                        zaxis])
        xpts = np.moveaxis(copy_vector_periods(xyz, self.arg_per), -1,1).reshape((3,-1))
        # first point == last point        
        xpts = np.hstack([xpts[:,-1:], xpts])

        eval_pts = np.array([R_sym*np.cos(phi_sym),
                             R_sym*np.sin(phi_sym),
                             Z_sym])

        # TODO: vectorize this over multiple coils
        def biot_savart(eval_pts, coil_pts, current):
            """Biot-Savart law following [1]

            Parameters
            ----------
            eval_pts : array-like shape(3,n)
                evaluation points in cartesian coordinates
            coil_pts : array-like shape(3,m)
                points in cartesian space defining coil
            current : float
                current through the coil

            Returns
            -------
            B : ndarray, shape(3,k)
                magnetic field in cartesian components at specified points

            [1] Hanson & Hirshman, "Compact expressions for the Biot-Savart fields of a filamentary segment" (2002)
            """
            dvec = np.diff(coil_pts, axis=1)
            L = np.linalg.norm(dvec, axis=0)

            Ri_vec = eval_pts[:, :,np.newaxis] - coil_pts[:,np.newaxis,:-1]
            Ri = np.linalg.norm(Ri_vec, axis=0)
            Rf = np.linalg.norm(eval_pts[:, :, np.newaxis] - coil_pts[:,np.newaxis,1:], axis=0)
            Ri_p_Rf = Ri + Rf

            # 1.0e-7 == mu0/(4 pi)
            Bmag = 1.0e-7 * current * 2.0 * Ri_p_Rf / ( Ri * Rf * (Ri_p_Rf*Ri_p_Rf - L*L) )

            # cross product of L*hat(eps)==dvec with Ri_vec, scaled by Bmag
            vec = np.cross(dvec, Ri_vec, axis=0)
            return np.sum(Bmag * vec, axis=-1)

        B = biot_savart(eval_pts, xpts, current)

        # add B^X and B^Y to MGRID magnetic field; need to convert to cylindrical components first
        return np.array([B[0]*np.cos(phi_sym) + B[1]*np.sin(phi_sym),
                         B[1]*np.cos(phi_sym) - B[0]*np.sin(phi_sym),
                         B[2]])

    def compute_T_S(self, jacobian):
        a = jacobian["g_uu"]
        b = jacobian["g_uv"]
        c = jacobian["g_vv"]
        ap = a + 2*b + c
        am = a - 2*b + c
        cma = c - a

        sqrt_a = np.sqrt(a)
        sqrt_c = np.sqrt(c)
        sqrt_ap = np.sqrt(ap)
        sqrt_am = np.sqrt(am)

        delt1u  = ap*am  - cma*cma
        azp1u  = jacobian["a_uu"]  + jacobian["a_uv"]  + jacobian["a_vv"]
        azm1u  = jacobian["a_uu"]  - jacobian["a_uv"]  + jacobian["a_vv"]
        cma11u  = jacobian["a_vv"]  - jacobian["a_uu"]
        r1p  = (azp1u*(delt1u - cma*cma)/ap - azm1u*ap + 2.0*cma11u*cma)/delt1u
        r1m  = (azm1u*(delt1u - cma*cma)/am - azp1u*am + 2.0*cma11u*cma)/delt1u
        r0p  = (-azp1u*am*cma/ap - azm1u*cma + 2.0*cma11u*am)/delt1u
        r0m  = (-azm1u*ap*cma/am - azp1u*cma + 2.0*cma11u*ap)/delt1u
        ra1p = azp1u/ap
        ra1m = azm1u/am

        # compute T^{\pm}_l, S^{\pm}_l

        num_four = self.mf + self.nf + 1
            
        # storage for all T^{\pm}_l, S^{\pm}_l
        T_p_l = np.zeros([num_four, self.ntheta_stellsym*self.nzeta]) # T^{+}_l
        T_m_l = np.zeros([num_four, self.ntheta_stellsym*self.nzeta]) # T^{-}_l
        S_p_l = np.zeros([num_four, self.ntheta_stellsym*self.nzeta]) # S^{+}_l
        S_m_l = np.zeros([num_four, self.ntheta_stellsym*self.nzeta]) # S^{-}_l

        # T^{\pm}_0
        T_p_l[0, :] = 1.0/sqrt_ap*np.log((sqrt_ap*2*sqrt_c + ap + cma)/(sqrt_ap*2*sqrt_a - ap + cma))
        T_m_l[0, :] = 1.0/sqrt_am*np.log((sqrt_am*2*sqrt_c + am + cma)/(sqrt_am*2*sqrt_a - am + cma))

        # S^{\pm}_0
        S_p_l[0, :] = ra1p * T_p_l[0, :] - (r1p + r0p)/(2*sqrt_c) + (r0p - r1p)/(2*sqrt_a)
        S_m_l[0, :] = ra1m * T_m_l[0, :] - (r1m + r0m)/(2*sqrt_c) + (r0m - r1m)/(2*sqrt_a)

        # now use recurrence relation for l > 0
        for l in range(1, self.mf+self.nf+1):

            # compute T^{\pm}_l
            if l > 1:
                T_p_l[l, :] = ((2*sqrt_c + (-1)**l * 2*sqrt_a) - (2.0*l - 1.0)*cma*T_p_l[l-1, :] - (l-1)*am*T_p_l[l-2, :])/(ap*l)
                T_m_l[l, :] = ((2*sqrt_c + (-1)**l * 2*sqrt_a) - (2.0*l - 1.0)*cma*T_m_l[l-1, :] - (l-1)*ap*T_m_l[l-2, :])/(am*l)
            else:
                T_p_l[l, :] = ((2*sqrt_c + (-1)**l * 2*sqrt_a) - (2.0*l - 1.0)*cma*T_p_l[l-1, :])/(ap*l)
                T_m_l[l, :] = ((2*sqrt_c + (-1)**l * 2*sqrt_a) - (2.0*l - 1.0)*cma*T_m_l[l-1, :])/(am*l)

            # compute S^{\pm}_l based on T^{\pm}_l and T^{\pm}_{l-1}
            S_p_l[l, :] = (r1p*l + ra1p)*T_p_l[l, :] + r0p*l*T_p_l[l-1, :] - (r1p + r0p)/(2*sqrt_c) + (-1)**l * (r0p - r1p)/(2*sqrt_a)
            S_m_l[l, :] = (r1m*l + ra1m)*T_m_l[l, :] + r0m*l*T_m_l[l-1, :] - (r1m + r0m)/(2*sqrt_c) + (-1)**l * (r0m - r1m)/(2*sqrt_a)

        return T_p_l, T_m_l, S_p_l, S_m_l
        
    # scalpot; analyt
    def analyticalIntegrals(self, jacobian, T_p_l, T_m_l, S_p_l, S_m_l, B_field):

        # analysum, analysum2 using FFTs
        brad, bphi, bz = B_field        
        bexni = -self.wint * (jacobian["surfNormR"] * brad + jacobian["surfNormPhi"] * bphi + jacobian["surfNormZ"] * bz) * 4.0*np.pi*np.pi
        T_p = (T_p_l*bexni).reshape(-1, self.ntheta_stellsym, self.nzeta)
        T_m = (T_m_l*bexni).reshape(-1, self.ntheta_stellsym, self.nzeta)

        ft_T_p = np.fft.ifft(T_p, axis=1, n=self.ntheta)*self.ntheta
        ft_T_p = np.fft.fft(ft_T_p, axis=2)

        ft_T_m = np.fft.ifft(T_m, axis=1, n=self.ntheta)*self.ntheta
        ft_T_m = np.fft.fft(ft_T_m, axis=2)

        ku, kv = np.meshgrid(np.arange(self.ntheta_stellsym), np.arange(self.nzeta))
        i = self.nzeta*ku + kv
        
        num_four = self.mf+self.nf+1
        S_p_4d = np.zeros([num_four, self.ntheta_stellsym, self.nzeta, self.ntheta_stellsym*self.nzeta])
        S_m_4d = np.zeros([num_four, self.ntheta_stellsym, self.nzeta, self.ntheta_stellsym*self.nzeta])

        S_p_4d[:,ku, kv, i] = S_p_l.reshape(num_four, self.ntheta_stellsym, self.nzeta)[:,ku,kv]
        S_m_4d[:,ku, kv, i] = S_m_l.reshape(num_four, self.ntheta_stellsym, self.nzeta)[:,ku,kv]
        
        # TODO: figure out a faster way to do this, its very sparse
        ft_S_p = np.fft.ifft(S_p_4d, axis=1, n=self.ntheta)*self.ntheta
        ft_S_p = np.fft.fft(ft_S_p, axis=2)
        
        ft_S_m = np.fft.ifft(S_m_4d, axis=1, n=self.ntheta)*self.ntheta
        ft_S_m = np.fft.fft(ft_S_m, axis=2)

        m,n = np.meshgrid(np.arange(self.mf+1), np.concatenate([np.arange(self.nf+1), np.arange(-self.nf,0)]), indexing="ij")

        I_mn = np.zeros([self.mf+1, 2*self.nf+1])
        I_mn = np.where(np.logical_or(m==0,n==0), (n>=0)*np.sum(self.cmns[:,m,n] * (ft_T_p[:, m, n].imag + ft_T_m[:, m, n].imag), axis=0), I_mn)
        I_mn = np.where(np.logical_and(m != 0, n>0), np.sum(self.cmns[:,m,n] * ft_T_p[:, m, n].imag, axis=0), I_mn)
        I_mn = np.where(np.logical_and(m != 0, n<0), np.sum(self.cmns[:,m,-n] * ft_T_m[:, m, n].imag, axis=0), I_mn)                

        K_mn_grid = np.zeros([self.mf+1, 2*self.nf+1, self.ntheta_stellsym*self.nzeta])
        K_mn_grid = np.where(np.logical_or(m==0,n==0)[:,:,np.newaxis], np.sum(self.cmns[:,m,n, np.newaxis] * (ft_S_p[:, m, n, :].imag + ft_S_m[:, m, n, :].imag), axis=0), K_mn_grid)
        K_mn_grid = np.where(np.logical_and(m!=0,n>0)[:,:,np.newaxis], np.sum(self.cmns[:,m,n, np.newaxis] * ft_S_p[:, m, n, :].imag,  axis=0), K_mn_grid)
        K_mn_grid = np.where(np.logical_and(m!=0,n<0)[:,:,np.newaxis], np.sum(self.cmns[:,m,-n, np.newaxis] * ft_S_m[:, m, n, :].imag, axis=0), K_mn_grid)

        return I_mn, K_mn_grid

    def computeScalarMagneticPotential(self, I_mn, K_mn_grid, B_field, jacobian, coords):
        # this makes bvec to be in Fortran order (-nf, ..., nf)
        bvec = np.fft.fftshift(I_mn, axes=1).T.flatten()
        if self.ivacskip != 0:

            # Here, bvecsav contains the previous non-singular contribution to the "final" bvec from fouri.
            # For ivacskip != 0, this contribution is used from the cache in bvecsav.
            bvec += self.bvecsav

        # greenf
        else:
            # Here, bvecsav stores the singular part of bvec from analyt
            # to be subtracted from the "final" bvec computed below by fouri
            self.bvecsav = bvec

            ft_gsource, grpmn_4d = self.regularizedFourierTransforms(K_mn_grid, B_field, jacobian, coords)

            # combine with contribution from analyt(); available here in I_mn
            full_bvec = ft_gsource + I_mn

            # final fixup from fouri: zero out (m=0, n<0) components (#TODO: why ?)
            full_bvec[0, self.nf+1:] = 0.0

            # compute Fourier transform of grpmn to arrive at amatrix
            # (to be compared against ref_amatrix)
            grpmn_4d = grpmn_4d * self.wint.reshape([1,1,self.ntheta_stellsym, self.nzeta])
            grpmn_4d = np.fft.ifft(grpmn_4d, axis=2, n=self.ntheta)*self.ntheta
            grpmn_4d   = np.fft.fft(grpmn_4d, axis=3)

            amatrix_4d = np.concatenate([grpmn_4d[:, :, :self.mf+1, :self.nf+1].imag, grpmn_4d[:, :, :self.mf+1, -self.nf:].imag], axis=-1)

            # scale amatrix by (2 pi)^2 (#TODO: why ?)
            amatrix_4d *= (2.0*np.pi)**2

            m, n = np.meshgrid(np.arange(self.mf+1), np.arange(2*self.nf+1), indexing="ij")            
            # zero out (m=0, n<0, m', n') modes for all m', n' (#TODO: why ?)
            amatrix_4d = np.where(np.logical_and(m==0, n>self.nf)[:,:,np.newaxis, np.newaxis], 0, amatrix_4d)

            # add diagnonal terms (#TODO: why 4*pi^3 instead of 1 ?)         
            amatrix_4d[m, n, m, n] += 4.0*np.pi**3

            bvec = np.fft.fftshift(full_bvec, axes=1).T.flatten()

            self.amatsav = np.fft.fftshift(np.transpose(amatrix_4d, (1,0,3,2)), axes=(0,2)).reshape([(self.mf+1)*(2*self.nf+1), (self.mf+1)*(2*self.nf+1)]).T.flatten()

            # remove singular contribution from bvec to save the non-singular part
            self.bvecsav = bvec - self.bvecsav

        amatrix = self.amatsav

        self.potvac_2d = self.solveForScalarMagneticPotential(amatrix, bvec)

    def regularizedFourierTransforms(self, K_mn_grid, B_field, jacobian, coords):

        brad, bphi, bz = B_field        
        bexni = -self.wint * (jacobian["surfNormR"] * brad + jacobian["surfNormPhi"] * bphi + jacobian["surfNormZ"] * bz) * 4.0*np.pi*np.pi

        green  = np.zeros([self.ntheta_stellsym, self.nzeta, self.ntheta, self.nzeta, self.nvper])
        greenp = np.zeros([self.ntheta_stellsym, self.nzeta, self.ntheta, self.nzeta, self.nvper])

        #max_ip = self.ntheta_stellsym*self.nzeta
        ku_ip, kv_ip, ku_i, kv_i = np.meshgrid(np.arange(self.ntheta_stellsym), np.arange(self.nzeta), np.arange(self.ntheta), np.arange(self.nzeta), indexing="ij")

        # linear index over primed grid
        ip = (ku_ip*self.nzeta+kv_ip)

        # cartesial coordinates in first field period
        xip = coords["X"].reshape((-1,self.nzeta))[ku_ip, kv_ip]
        yip = coords["Y"].reshape((-1,self.nzeta))[ku_ip, kv_ip]

        # field-period invariant vectors
        r_squared = (coords["R"]**2 + coords["Z"]**2).reshape((-1,self.nzeta))
        gsave = r_squared[ku_ip, kv_ip] + r_squared - 2.0*coords["Z_sym"][ip].reshape(ku_ip.shape)*coords["Z"].reshape((-1, self.nzeta))
        drv  = -(coords["R_sym"]*jacobian["surfNormR"] + coords["Z_sym"]*jacobian["surfNormZ"])      
        dsave = drv[ip] + coords["Z"].reshape((-1, self.nzeta))*jacobian["surfNormZ"].reshape((self.ntheta_stellsym, self.nzeta))[ku_ip, kv_ip]
        # x,y in period j
        xper, yper = copy_vector_periods(np.array([xip, yip]), self.arg_per)

        # cartesian components of surface normal ... ???
        sxsave = (jacobian["surfNormR"][ip][:,:,:,:,np.newaxis]*xper - jacobian["surfNormPhi"][ip][:,:,:,:,np.newaxis]*yper)/coords["R_sym"][ip][:,:,:,:,np.newaxis]
        sysave = (jacobian["surfNormR"][ip][:,:,:,:,np.newaxis]*yper + jacobian["surfNormPhi"][ip][:,:,:,:,np.newaxis]*xper)/coords["R_sym"][ip][:,:,:,:,np.newaxis]



        ftemp = (gsave[:,:,:,:,np.newaxis]
                 - 2*xper*coords["X"].reshape((-1, self.nzeta))[np.newaxis,np.newaxis,:,:,np.newaxis]
                 - 2*yper*coords["Y"].reshape((-1, self.nzeta))[np.newaxis,np.newaxis,:,:,np.newaxis])
        ftemp = 1/np.where(ftemp<=0, 1, ftemp)
        htemp = np.sqrt(ftemp)
        gtemp = (  coords["X"].reshape((-1, self.nzeta))[np.newaxis,np.newaxis:,:,np.newaxis]*sxsave
                 + coords["Y"].reshape((-1, self.nzeta))[np.newaxis,np.newaxis:,:,np.newaxis]*sysave
                 + dsave[:,:,:,:,np.newaxis])
        greenp_update = ftemp*htemp*gtemp
        green_update = htemp
        mask = ~((self.arg_per == 0) | (self.nzeta == 1)).reshape((1,1,1,1,-1,))                
        greenp = np.where(mask, greenp + greenp_update, greenp)
        green  = np.where(mask, green + green_update, green)
               

        if self.nzeta == 1:
            # Tokamak: nvper toroidal "modules"
            delta_kv = (kv_i - kv_ip)%self.nvper
        else:
            # Stellarator: nv toroidal grid points
            delta_kv = (kv_i - kv_ip)%self.nzeta

        # TODO: why is there an additional offset of ntheta?
        delta_ku = ku_i - ku_ip + self.ntheta
        ga1 = self.tanu[delta_ku]*(jacobian["g_uu"][ip]*self.tanu[delta_ku] + 2*jacobian["g_uv"][ip]*self.tanv[delta_kv]) + jacobian["g_vv"][ip]*self.tanv[delta_kv]*self.tanv[delta_kv]
        ga2 = self.tanu[delta_ku]*(jacobian["a_uu"][ip]*self.tanu[delta_ku] +   jacobian["a_uv"][ip]*self.tanv[delta_kv]) + jacobian["a_vv"][ip]*self.tanv[delta_kv]*self.tanv[delta_kv]

        greenp_sing = - (ga2/ga1*1/np.sqrt(ga1))[:,:,:,:,np.newaxis]
        green_sing = - 1/np.sqrt(ga1)[:,:,:,:,np.newaxis]
        mask = ((ku_ip != ku_i) | (kv_ip != kv_i) | (self.nzeta == 1 and kp > 0))[:,:,:,:,np.newaxis] & ((self.arg_per == 0) |  (self.nzeta == 1))
        greenp = np.where(mask, greenp + greenp_update + greenp_sing, greenp)
        green = np.where(mask, green + green_update + green_sing, green)                               

        if self.nzeta == 1:
            # Tokamak: need to do toroidal average / integral:
            # normalize by number of toroidal "modules"
            greenp /= self.nvper
            green  /= self.nvper

        greenp = np.sum(greenp, -1)
        green = np.sum(green, -1)
        gstore = np.sum(bexni.reshape((self.ntheta_stellsym, self.nzeta,1,1)) * green[:self.ntheta_stellsym, :, :, :], axis=(0,1))

        # Here, grpmn should contain already the contributions from S^{\pm}_l as computed in analyt/analysum(2).
        # Thus Fourier-transform greenp and add to grpmn.

        # step 1: "fold over" contribution from (pi ... 2pi) in greenp

        # stellarator-symmetric first half-module is copied directly
        # the other half of the first module is "folded over" according to odd symmetry under the stellarator-symmetry operation
        ku, kv = np.meshgrid(np.arange(self.ntheta_stellsym), np.arange(self.nzeta), indexing="ij")
        # anti-symmetric part from stellarator-symmetric half in second half of first toroidal module
        kernel_4d = greenp[:,:,ku, kv] - greenp[:,:,-ku, -kv]

        # accumulated magic from fourp and (sin/cos)mui
        kernel_4d = kernel_4d * 1/self.nfp * (2*np.pi)/self.ntheta * (2.0*np.pi)/self.nzeta
        kernel_4d[:,:,0,:] *= 0.5
        kernel_4d[:,:,-1,:] *= 0.5

        ft_kernel = np.fft.ifft(kernel_4d, axis=2, n=self.ntheta)*self.ntheta
        ft_kernel = np.fft.fft(ft_kernel, axis=3)

        ft_kernel = np.concatenate([ft_kernel[:self.ntheta_stellsym, :self.nzeta, :self.mf+1, :self.nf+1].imag,
                                    ft_kernel[:self.ntheta_stellsym, :self.nzeta, :self.mf+1, -self.nf:].imag],
                                   axis=-1).transpose((2,3,0,1))
        # now assemble final grpmn by adding K_mn_grid and ft_kernel_4d
        K_mn_grid_4d = K_mn_grid.reshape(self.mf+1, 2*self.nf+1, self.ntheta_stellsym, self.nzeta)
        # whoooo... :-)
        grpmn_4d = K_mn_grid_4d + ft_kernel


        # first step: "fold over" upper half of gsource to make use of stellarator symmetry
        # anti-symmetric part from stellarator-symmetric half in second half of first toroidal module
        gsource_sym = gstore[ku, kv] - gstore[-ku, -kv]

        # compute Fourier-transform of gsource
        # when only computing the contribution to bvec from gstore, the reference value can be found in potvac,

        # accumulated magic from fouri and (sin/cos)mui
        gsource_sym = gsource_sym * 1/self.nfp * (2*np.pi)/self.ntheta * (2.0*np.pi)/self.nzeta
        gsource_sym[0,:] *= 0.5
        gsource_sym[-1,:] *= 0.5

        ft_gsource = np.fft.ifft(gsource_sym, axis=0, n=self.ntheta)*self.ntheta
        ft_gsource = np.fft.fft(ft_gsource, axis=1)
        ft_gsource = np.concatenate([ft_gsource[:self.mf+1,:self.nf+1].imag, ft_gsource[:self.mf+1,-self.nf:].imag], axis=1)

        return ft_gsource, grpmn_4d
    # call the solver for the linear system of equations that defines the scalar magnetic potential
    def solveForScalarMagneticPotential(self, Amat, bvec):
        # matrix          is ref_amatrix_4d_fft
        # right-hand-side is full_bvec

        # solution is potvac

        # Note: have to re-create proper shapes here from amatrix, bvec
        # since these are the only ones available for ivacskip != 0 .

        bvec_2d = bvec.reshape([2*self.nf+1, self.mf+1]).T
        bvec_1d = np.fft.ifftshift(bvec_2d, axes=1).flatten()

        amatrix = np.fft.ifftshift(Amat.reshape([2*self.nf+1, self.mf+1, 2*self.nf+1, self.mf+1]).T, axes=(1,3))

        amatrix = amatrix.reshape([(self.mf+1)*(2*self.nf+1), (self.mf+1)*(2*self.nf+1)])

        potvac = np.linalg.solve(amatrix, bvec_1d)

        return potvac.reshape([self.mf+1, 2*self.nf+1])



    # compute co- and contravariant magnetic field components
    def analyzeScalarMagneticPotential(self, B_field, potvac_2d, jacobian, coords):

        brad, bphi, bz = B_field
        m_potvac = np.zeros([self.ntheta_stellsym, self.nzeta]) # m*potvac --> for poloidal derivative
        n_potvac = np.zeros([self.ntheta_stellsym, self.nzeta]) # n*potvac --> for toroidal derivative

        m,n = np.meshgrid(np.arange(self.mf+1), np.arange(self.nf+1), indexing="ij")

        m_potvac[m, n] = m * potvac_2d[m, n]
        n_potvac[m, n] = n * potvac_2d[m, n]
        m_potvac[m, -n] =  m * potvac_2d[m, -n]
        n_potvac[m, -n] = -n * potvac_2d[m, -n]

        potu = np.fft.ifft(m_potvac, axis=0, n=self.ntheta) * self.ntheta
        potu = np.fft.fft(potu, axis=1).real[:self.ntheta_stellsym, :]
        potu = potu.flatten()

        potv = np.fft.ifft(n_potvac, axis=0, n=self.ntheta)*self.ntheta
        potv = -np.fft.fft(potv, axis=1).real[:self.ntheta_stellsym, :] * self.nfp
        potv = potv.flatten()

        # compute covariant magnetic field components: B_u, B_v
        bexu = coords["dRdTheta"] * brad  + coords["dZdTheta"] * bz
        bexv = coords["dRdZeta"] * brad + coords["R_sym"] * bphi + coords["dZdZeta"] * bz

        # B_u = potu + bexu
        self.bsubu = potu + bexu

        # B_v = potv + bexv
        self.bsubv = potv + bexv

        # compute B^u, B^v and (with B_u, B_v) then also |B|^2/2

        # TODO: for now, simply copied over from NESTOR code; have to understand what is actually done here!
        huv = self.nfp*jacobian["g_uv"]
        hvv = jacobian["g_vv"]*self.nfp*self.nfp
        det = 1.0/(jacobian["g_uu"]*hvv-huv*huv)

        # contravariant components of magnetic field: B^u, B^v

        # B^u
        bsupu = (hvv*self.bsubu - huv*self.bsubv)*det

        # B^v
        bsupv = (-huv*self.bsubu+jacobian["g_uu"]*self.bsubv)*det

        # |B|^2/2 = (B^u*B_u + B^v*B_v)/2
        self.bsqvac = (self.bsubu*bsupu + self.bsubv*bsupv)/2.0

        # compute cylindrical components B^R, B^\phi, B^Z
        self.brv   = coords["dRdTheta"] * bsupu + coords["dRdZeta"] * bsupv
        self.bphiv = coords["R_sym"] * bsupv
        self.bzv   = coords["dZdTheta"] * bsupu + coords["dZdZeta"] * bsupv

    def firstIterationPrintout(self):
        print("  In VACUUM, np = %2d mf = %2d nf = %2d nu = %2d nv = %2d"%(self.nfp, self.mf, self.nf, self.ntheta, self.nzeta))

        # -plasma current/pi2
        self.bsubuvac = np.sum(self.bsubu*self.wint)*self.signgs*2.0*np.pi
        self.bsubvvac = np.sum(self.bsubv*self.wint)

        # currents in MA
        fac = 1.0e-6/mu0

        print(("  2*pi * a * -BPOL(vac) = %10.2E TOROIDAL CURRENT = %10.2E\n"
              +"  R * BTOR(vac) = %10.2E R * BTOR(plasma) = %10.2E")%(self.bsubuvac*fac, self.ctor*fac, self.bsubvvac, self.rbtor))

        if self.rbtor*self.bsubvvac < 0:
            # rbtor and bsubvvac must have the same sign --> phiedge_error_flag
            self.ier_flag = 7

        if np.abs((self.ctor - self.bsubuvac)/self.rbtor) > 1.0e-2:
            # 'VAC-VMEC I_TOR MISMATCH : BOUNDARY MAY ENCLOSE EXT. COIL'
            self.ier_flag = 10

    def produceOutputFile(self, vacoutFilename):
        # mode numbers for potvac
        self.xmpot = np.zeros([(self.mf+1)*(2*self.nf+1)])
        self.xnpot = np.zeros([(self.mf+1)*(2*self.nf+1)])
        mn = 0
        for n in range(-self.nf, self.nf+1):
            for m in range(self.mf+1):
                self.xmpot[mn] = m
                self.xnpot[mn] = n*self.nfp
                mn += 1

        vacout = Dataset(vacoutFilename, "w")

        dim_nuv2 = def_ncdim(vacout, self.ntheta_stellsym*self.nzeta)
        dim_mnpd2 = def_ncdim(vacout, (self.mf+1)*(2*self.nf+1))
        dim_mnpd2_sq = def_ncdim(vacout, (self.mf+1)*(2*self.nf+1)*(self.mf+1)*(2*self.nf+1))

        var_ivac     = vacout.createVariable("ivac", "i4")
        var_ier_flag = vacout.createVariable("ier_flag", "i4")
        var_bsqvac   = vacout.createVariable("bsqvac", "f8", (dim_nuv2,))
        var_mnpd     = vacout.createVariable("mnpd", "i4")
        var_mnpd2    = vacout.createVariable("mnpd2", "i4")
        var_xmpot    = vacout.createVariable("xmpot", "f8", (dim_mnpd2,))
        var_xnpot    = vacout.createVariable("xnpot", "f8", (dim_mnpd2,))
        var_potvac   = vacout.createVariable("potvac", "f8", (dim_mnpd2,))
        var_brv      = vacout.createVariable("brv", "f8", (dim_nuv2,))
        var_bphiv    = vacout.createVariable("bphiv", "f8", (dim_nuv2,))
        var_bzv      = vacout.createVariable("bzv", "f8", (dim_nuv2,))
        var_bsubvvac = vacout.createVariable("bsubvvac", "f8")
        var_amatsav  = vacout.createVariable("amatsav", "f8", (dim_mnpd2_sq,))
        var_bvecsav  = vacout.createVariable("bvecsav", "f8", (dim_mnpd2,))

        var_ivac.assignValue(self.ivac)
        var_ier_flag.assignValue(self.ier_flag)
        var_bsqvac[:] = self.bsqvac
        var_mnpd.assignValue((self.mf+1)*(2*self.nf+1))
        var_mnpd2.assignValue(self.mnpd2)
        var_xmpot[:] = self.xmpot
        var_xnpot[:] = self.xnpot
        var_potvac[:] = np.fft.fftshift(self.potvac_2d, axes=1).T.flatten()
        var_brv[:] = self.brv
        var_bphiv[:] = self.bphiv
        var_bzv[:] = self.bzv
        var_bsubvvac.assignValue(self.bsubvvac)
        var_amatsav[:] = self.amatsav
        var_bvecsav[:] = self.bvecsav

        vacout.close()


def main(vacin_filename, vacout_filename=None, mgrid=None):
    nestor = Nestor(vacin_filename)

    # in principle, this needs to be done only once
    nestor.precompute()
    nestor.loadMGridFile(mgrid)

    # the following calls need to be done on every iteration
    coords = nestor.evalSurfaceGeometry()
    jacobian = nestor.compute_jacobian(coords)
    B_extern = nestor.interpolateMGridFile(coords["R_sym"], coords["Z_sym"], coords["phi_sym"])
    B_plasma = nestor.modelNetToroidalCurrent(nestor.raxis_nestor,
                                   nestor.phiaxis,
                                   nestor.zaxis_nestor,
                                   nestor.ctor/mu0,
                                   coords["R_sym"],
                                   coords["phi_sym"],
                                   coords["Z_sym"])
    B_field = B_extern + B_plasma
    T_p_l, T_m_l, S_p_l, S_m_l = nestor.compute_T_S(jacobian)
    I_mn, K_mn = nestor.analyticalIntegrals(jacobian, T_p_l, T_m_l, S_p_l, S_m_l, B_field)
    nestor.computeScalarMagneticPotential(I_mn, K_mn, B_field, jacobian, coords)
    nestor.analyzeScalarMagneticPotential(B_field,
                                          nestor.potvac_2d,
                                          jacobian,
                                          coords)

    if nestor.ivac == 0:
        nestor.ivac += 1
        nestor.firstIterationPrintout()

    if vacout_filename is None:
        vacout_filename = vacin_filename.replace("vacin_", "vacout_")
    nestor.produceOutputFile(vacout_filename)
    
if __name__ == '__main__':
    if len(sys.argv) > 1:
        vacin_filename = sys.argv[1]
        folder = os.getcwd()
        main(vacin_filename)
    else:
        print("usage: NESTOR.py vacin.nc")

