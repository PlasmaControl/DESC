import numpy as np
from abc import ABC, abstractmethod
from netCDF4 import Dataset

from desc.backend import jnp
from desc.io import IOAble
from desc.grid import Grid
from desc.interpolate import interp3d
from desc.derivatives import Derivative


class MagneticField(IOAble, ABC):
    """Base class for all magnetic fields

    Subclasses must implement the "compute_magnetic_field" method

    """

    _io_attrs_ = []

    def __mul__(self, x):
        if np.isscalar(x):
            return ScaledMagneticField(x, self)
        else:
            return NotImplemented

    def __rmul__(self, x):
        return self.__mul__(x)

    def __add__(self, x):
        if isinstance(x, MagneticField):
            return SumMagneticField(self, x)
        else:
            return NotImplemented

    def __neg__(self):
        return ScaledMagneticField(-1, self)

    def __sub__(self, x):
        return self.__add__(-x)

    @abstractmethod
    def compute_magnetic_field(self, coords, params=None, dR=0, dp=0, dZ=0):
        """Compute magnetic field at a set of points

        Parameters
        ----------
        coords : array-like shape(N,3) or Grid
            cylindrical coordinates to evaluate field at [R,phi,Z]
        params : tuple, optional
            parameters to pass to scalar potential function
        dR, dp, dZ : int, optional
            order of derivative to take in R,phi,Z directions

        Returns
        -------
        field : ndarray, shape(N,3)
            magnetic field at specified points, in cylindrical form [BR, Bphi,BZ]

        """


class ScaledMagneticField(MagneticField):
    """Magnetic field scaled by a scalar value

    ie B_new = scalar * B_old

    Parameters
    ----------
    scalar : float, int
        scaling factor for magnetic field
    field : MagneticField
        base field to be scaled
        
    """

    _io_attrs = MagneticField._io_attrs_ + ["_field", "_scalar"]

    def __init__(self, scalar, field):
        assert np.isscalar(scalar), "scalar must actually be a scalar value"
        assert isinstance(
            field, MagneticField
        ), "field should be a subclass of MagneticField, got type {}".format(
            type(field)
        )
        self._scalar = scalar
        self._field = field

    def compute_magnetic_field(self, coords, params=None, dR=0, dp=0, dZ=0):
        """Compute magnetic field at a set of points

        Parameters
        ----------
        coords : array-like shape(N,3) or Grid
            cylindrical coordinates to evaluate field at [R,phi,Z]
        params : tuple, optional
            parameters to pass to scalar potential function
        dR, dp, dZ : int, optional
            order of derivative to take in R,phi,Z directions

        Returns
        -------
        field : ndarray, shape(N,3)
            scaled magnetic field at specified points, in cylindrical 
            form [BR, Bphi,BZ]

        """
        return self._scalar * self._field.compute_magnetic_field(
            coords, params=None, dR=0, dp=0, dZ=0
        )


class SumMagneticField(MagneticField):
    """Sum of two or more magnetic field sources

    Parameters
    ----------
    fields : MagneticField
        two or more MagneticFields to add together
    """

    _io_attrs = MagneticField._io_attrs_ + ["_fields"]

    def __init__(self, *fields):
        assert all(
            [isinstance(field, MagneticField) for field in fields]
        ), "fields should each be a subclass of MagneticField, got {}".format(
            [type(field) for field in fields]
        )
        self._fields = fields

    def compute_magnetic_field(self, coords, params=None, dR=0, dp=0, dZ=0):
        """Compute magnetic field at a set of points

        Parameters
        ----------
        coords : array-like shape(N,3) or Grid
            cylindrical coordinates to evaluate field at [R,phi,Z]
        params : tuple, optional
            parameters to pass to scalar potential function. If None,
            uses the default parameters for each field. If a tuple, should have
            one entry for each component field.
        dR, dp, dZ : int, optional
            order of derivative to take in R,phi,Z directions

        Returns
        -------
        field : ndarray, shape(N,3)
            scaled magnetic field at specified points, in cylindrical 
            form [BR, Bphi,BZ]

        """
        if params is None:
            params = [None] * len(self._fields)
        B = 0
        for i, field in enumerate(self._fields):
            B += field.compute_magnetic_field(coords, params[i], dR=dR, dp=dp, dZ=dZ)
        return B


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
    method : str
        interpolation method
    extrap : bool
        whether to extrapolate beyond the domain of known field values or return nan
    period : float
        period in the toroidal direction (usually 2pi/NFP)

    """

    _io_attrs_ = [
        "_R",
        "_phi",
        "_Z",
        "_BR",
        "_Bphi",
        "_BZ",
        "_method",
        "_extrap",
        "_period",
    ]

    def __init__(self, R, phi, Z, BR, Bphi, BZ, method="cubic", extrap=False, period=0):

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

        # TODO: precompute derivative matrices

    def compute_magnetic_field(self, coords, params=None, dR=0, dp=0, dZ=0):
        """Compute magnetic field at a set of points

        Parameters
        ----------
        coords : array-like shape(N,3) or Grid
            cylindrical coordinates to evaluate field at [R,phi,Z]
        params : tuple, optional
            parameters to pass to scalar potential function
        dR, dp, dZ : int, optional
            order of derivative to take in R,phi,Z directions

        Returns
        -------
        field : ndarray, shape(N,3)
            magnetic field at specified points, in cylindrical form [BR, Bphi,BZ]

        """

        if isinstance(coords, Grid):
            Rq, phiq, Zq = coords.nodes.T
        else:
            Rq, phiq, Zq = coords.T

        BRq = interp3d(
            Rq,
            phiq,
            Zq,
            self._R,
            self._phi,
            self._Z,
            self._BR,
            self._method,
            (dR, dp, dZ),
            self._extrap,
            self._period,
        )
        Bphiq = interp3d(
            Rq,
            phiq,
            Zq,
            self._R,
            self._phi,
            self._Z,
            self._Bphi,
            self._method,
            (dR, dp, dZ),
            self._extrap,
            self._period,
        )
        BZq = interp3d(
            Rq,
            phiq,
            Zq,
            self._R,
            self._phi,
            self._Z,
            self._BZ,
            self._method,
            (dR, dp, dZ),
            self._extrap,
            self._period,
        )

        return jnp.array([BRq, Bphiq, BZq]).T

    @classmethod
    def from_mgrid(
        cls, mgrid_file, extcur=1, method="cubic", extrap=False, period=None
    ):
        """Create a SplineMagneticField from an "mgrid" file from MAKEGRID

        Parameters
        ----------
        mgrid_file : str or path-like
            path to mgrid file in netCDF format
        extcur : array-like
            currents for each subset of the field
        method : str
            interpolation method
        extrap : bool
            whether to extrapolate beyond the domain of known field values or return nan
        period : float
            period in the toroidal direction (usually 2pi/NFP)

        """
        mgrid = Dataset(mgrid_file, "r")
        ir = int(mgrid["ir"][()])
        jz = int(mgrid["jz"][()])
        kp = int(mgrid["kp"][()])
        nfp = mgrid["nfp"][()].data
        nextcur = int(mgrid["nextcur"][()])
        rMin = mgrid["rmin"][()]
        rMax = mgrid["rmax"][()]
        zMin = mgrid["zmin"][()]
        zMax = mgrid["zmax"][()]

        br = np.zeros([kp, jz, ir])
        bp = np.zeros([kp, jz, ir])
        bz = np.zeros([kp, jz, ir])
        extcur = np.broadcast_to(extcur, nextcur)
        for i in range(nextcur):

            # apply scaling by currents given in VMEC input file
            scale = extcur[i]

            # sum up contributions from different coils
            coil_id = "%03d" % (i + 1,)
            br[:, :, :] += scale * mgrid["br_" + coil_id][()]
            bp[:, :, :] += scale * mgrid["bp_" + coil_id][()]
            bz[:, :, :] += scale * mgrid["bz_" + coil_id][()]
        mgrid.close()

        # shift axes to correct order
        br = np.moveaxis(br, (0, 1, 2), (1, 2, 0))
        bp = np.moveaxis(bp, (0, 1, 2), (1, 2, 0))
        bz = np.moveaxis(bz, (0, 1, 2), (1, 2, 0))

        # re-compute grid knots in radial and vertical direction
        Rgrid = np.linspace(rMin, rMax, ir)
        Zgrid = np.linspace(zMin, zMax, jz)
        pgrid = 2.0 * np.pi / (nfp * kp) * np.arange(kp)
        if period is None:
            period = 2 * np.pi / (nfp)

        return cls(Rgrid, pgrid, Zgrid, br, bp, bz, method, extrap, period)


class ScalarPotentialField(MagneticField):
    """Magnetic field due to a scalar magnetic potential in cylindrical coordinates

    Parameters
    ----------
    potential : callable
        function to compute the scalar potential. Should have a signature of
        the form potential(R,phi,Z,*params) -> ndarray.
        R,phi,Z are arrays of cylindrical coordinates.
    params : tuple, optional
        default parameters to pass to potential function

    """

    def __init__(self, potential, params=()):
        self._potential = potential
        self._params = params

    def compute_magnetic_field(self, coords, params=None, dR=0, dp=0, dZ=0):
        """Compute magnetic field at a set of points

        Parameters
        ----------
        coords : array-like shape(N,3) or Grid
            cylindrical coordinates to evaluate field at [R,phi,Z]
        params : tuple, optional
            parameters to pass to scalar potential function
        dR, dp, dZ : int, optional
            order of derivative to take in R,phi,Z directions

        Returns
        -------
        field : ndarray, shape(N,3)
            magnetic field at specified points, in cylindrical form [BR, Bphi,BZ]

        """
        if any([(dR != 0), (dp != 0), (dZ != 0)]):
            raise NotImplementedError(
                "Derivatives of scalar potential fields have not been implemented"
            )
        if isinstance(coords, Grid):
            coords = coords.nodes
        if params is None:
            params = self._params
        r, p, z = coords.T
        funR = lambda x: self._potential(x, p, z, *params)
        funP = lambda x: self._potential(r, x, z, *params)
        funZ = lambda x: self._potential(r, p, x, *params)
        br = Derivative.compute_jvp(funR, 0, (jnp.ones_like(r),), r)
        bp = Derivative.compute_jvp(funP, 0, (jnp.ones_like(p),), p)
        bz = Derivative.compute_jvp(funZ, 0, (jnp.ones_like(z),), z)
        return jnp.array([br, bp / r, bz]).T
