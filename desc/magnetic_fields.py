import numpy as np
from abc import ABC, abstractmethod
from netCDF4 import Dataset

from desc.backend import jnp
from desc.io import IOAble
from desc.grid import Grid
from desc.interpolate import interp3d

class MagneticField(IOAble, ABC):

    _io_attrs_ = []

    @abstractmethod
    def compute_magnetic_field(self, grid, params, dR, dp, dZ):
        """compute magnetic field on a grid in real (R, phi, Z) space"""


class SplineMagneticField(MagneticField):
    """Magnetic field from precomputed values on a grid

    Parameters
    ----------
    R : array-like, size(NR)
        R coordinates where field is specified
    phi : array-like, size(Nphi)
        phi coordinates where field is specified
    Z : array-like, size(NZ)
        Z coordinates where field is specified
    BR : array-like, shape(NR,Nphi,NZ)
        radial magnetic field on grid
    Bphi : array-like, shape(NR,Nphi,NZ)
        toroidal magnetic field on grid
    BZ : array-like, shape(NR,Nphi,NZ)
        vertical magnetic field on grid
    
    """
    
    def __init__(self, R,phi,Z,BR,Bphi,BZ, method="cubic", extrap=False, period=0):

        R, phi, Z = np.atleast_1d(R), np.atleast_1d(phi), np.atleast_1d(Z)
        assert R.ndim == 1
        assert phi.ndim == 1
        assert Z.ndim == 1
        BR, Bphi, BZ = np.atleast_3d(BR), np.atleast_3d(Bphi), np.atleast_3d(BZ)
        assert BR.shape == Bphi.shape == BZ.shape == (R.size, phi.size, Z.size)

        self._R = R
        self._phi = phi
        self._Z = Z
        self._BR = BR
        self._Bphi = Bphi
        self._BZ = BZ

        self._method = method
        self._extrap = extrap
        self._period = period


    def compute_magnetic_field(self, grid, params=None, dR=0, dp=0, dZ=0):

        if isinstance(grid, Grid):
            Rq, phiq, Zq = grid.nodes.T
        else:
            Rq, phiq, Zq = grid.T

        BRq = interp3d(Rq, phiq, Zq, self._R, self._phi, self._Z, self._BR,
                       self._method, (dR, dp, dZ), self._extrap, self._period)
        Bphiq = interp3d(Rq, phiq, Zq, self._R, self._phi, self._Z, self._Bphi,
                       self._method, (dR, dp, dZ), self._extrap, self._period)
        BZq = interp3d(Rq, phiq, Zq, self._R, self._phi, self._Z, self._BZ,
                       self._method, (dR, dp, dZ), self._extrap, self._period)

        return jnp.array([BRq, Bphiq, BZq]).T

    @classmethod
    def from_mgrid(cls, mgrid_file, extcur=1, method="cubic", extrap=False, period=0):

        mgrid = Dataset(mgrid_file, "r")
        ir = int(mgrid['ir'][()])
        jz = int(mgrid['jz'][()])
        kp = int(mgrid['kp'][()])
        nfp = mgrid['nfp'][()].data
        nextcur = int(mgrid['nextcur'][()])
        cur = mgrid['raw_coil_cur'][()]
        rMin = mgrid['rmin'][()]
        rMax = mgrid['rmax'][()]
        zMin = mgrid['zmin'][()]
        zMax = mgrid['zmax'][()]

        mgrid_mode = mgrid['mgrid_mode'][()]
        mode = bytearray(mgrid_mode).decode('utf-8')

        br = np.zeros([kp, jz, ir])
        bp = np.zeros([kp, jz, ir])
        bz = np.zeros([kp, jz, ir])
        extcur = np.broadcast_to(extcur, nextcur)
        for i in range(nextcur):

            # apply scaling by currents given in VMEC input file
            scale = extcur[i]

            # sum up contributions from different coils
            coil_id = "%03d"%(i+1,)
            br[:,:,:] += scale * mgrid['br_'+coil_id][()]
            bp[:,:,:] += scale * mgrid['bp_'+coil_id][()]
            bz[:,:,:] += scale * mgrid['bz_'+coil_id][()]
        mgrid.close()

        # re-compute grid knots in radial and vertical direction
        Rgrid = np.linspace(rMin, rMax, ir)
        Zgrid = np.linspace(zMin, zMax, jz)
        pgrid = 2.0*np.pi/(nfp*kp) * np.arange(kp)

        return cls(Rgrid, pgrid, Zgrid, br, bp, bz, method, extrap, period)
