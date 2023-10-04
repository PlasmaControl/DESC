"""Classes for magnetic fields."""

from abc import ABC, abstractmethod

import numpy as np
from netCDF4 import Dataset

from desc.backend import jit, jnp, odeint
from desc.basis import DoubleFourierSeries
from desc.compute import rpz2xyz_vec, xyz2rpz
from desc.derivatives import Derivative
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.grid import LinearGrid
from desc.interpolate import _approx_df, interp2d, interp3d
from desc.io import IOAble
from desc.transform import Transform
from desc.utils import copy_coeffs
from desc.vmec_utils import ptolemy_identity_fwd, ptolemy_identity_rev


def biot_savart(eval_pts, coil_pts, current):
    """Biot-Savart law following [1].

    Parameters
    ----------
    eval_pts : array-like shape(n,3)
        evaluation points in cartesian coordinates
    coil_pts : array-like shape(m,3)
        points in cartesian space defining coil, should be closed curve
    current : float
        current through the coil

    Returns
    -------
    B : ndarray, shape(n,3)
        magnetic field in cartesian components at specified points

    [1] Hanson & Hirshman, "Compact expressions for the Biot-Savart
    fields of a filamentary segment" (2002)
    """
    # TODO: vectorize this over multiple coils
    d_vec = jnp.diff(coil_pts, axis=0)
    L = jnp.linalg.norm(d_vec, axis=-1)

    Ri_vec = eval_pts[jnp.newaxis, :] - coil_pts[:-1, jnp.newaxis, :]
    Ri = jnp.linalg.norm(Ri_vec, axis=-1)
    Rf = jnp.linalg.norm(
        eval_pts[jnp.newaxis, :] - coil_pts[1:, jnp.newaxis, :], axis=-1
    )
    Ri_p_Rf = Ri + Rf

    # 1.0e-7 == mu_0/(4 pi)
    B_mag = (
        1.0e-7
        * current
        * 2.0
        * Ri_p_Rf
        / (Ri * Rf * (Ri_p_Rf * Ri_p_Rf - (L * L)[:, jnp.newaxis]))
    )

    # cross product of L*hat(eps)==d_vec with Ri_vec, scaled by B_mag
    vec = jnp.cross(d_vec[:, jnp.newaxis, :], Ri_vec, axis=-1)
    B = jnp.sum(B_mag[:, :, jnp.newaxis] * vec, axis=0)
    return B


def read_BNORM_file(fname, surface, eval_grid=None, scale_by_curpol=True):
    """Read BNORM-style .txt file containing Bnormal Fourier coefficients.

    Parameters
    ----------
    fname : str
        name of BNORM file to read and use to calculate Bnormal from.
    surface : Surface or Equilibrium
        Surface to calculate the magnetic field's Bnormal on.
        If an Equilibrium is supplied, will use its boundary surface.
    eval_grid : Grid, optional
        Grid of points on the plasma surface to evaluate the Bnormal at,
        if None defaults to a LinearGrid with twice
        the surface grid's poloidal and toroidal resolutions
    scale_by_curpol : bool, optional
        Whether or not to un-scale the Bnormal coefficients by curpol
        before calculating Bnormal, by default True
        (set to False if it is known that the BNORM file was saved without scaling
        by curpol)
        requires an Equilibrium to be passed in

    Returns
    -------
    Bnormal: ndarray,
        Bnormal distribution from the BNORM Fourier coefficients,
        evaluated on the given eval_grid
    """
    if isinstance(surface, EquilibriaFamily):
        surface = surface[-1]
    if isinstance(surface, Equilibrium):
        eq = surface
        surface = eq.surface
    else:
        eq = None

    assert surface.sym, (
        "BNORM assumes stellarator symmetry, but" "a non-symmetric surface was given!"
    )

    if scale_by_curpol and eq is None:
        raise RuntimeError(
            "an Equilibrium must be supplied when scale_by_curpol is True!"
        )

    curpol = (
        (2 * jnp.pi / eq.NFP * eq.compute("G", grid=LinearGrid(rho=jnp.array(1)))["G"])
        if scale_by_curpol
        else 1
    )

    data = np.genfromtxt(fname)

    xm = data[:, 0]
    xn = -data[:, 1]  # negate since BNORM uses sin(mu+nv) convention
    Bnorm_mn = data[:, 2] / curpol  # these will only be sin terms

    # convert to DESC Fourier representation i.e. like cos(mt)*cos(nz)
    m, n, Bnorm_mn = ptolemy_identity_fwd(xm, xn, Bnorm_mn, np.zeros_like(Bnorm_mn))
    basis = DoubleFourierSeries(
        int(np.max(m)), int(np.max(n)), sym="sin", NFP=surface.NFP
    )

    Bnorm_mn_desc_basis = copy_coeffs(
        Bnorm_mn.squeeze(), np.vstack((np.zeros_like(m), m, n)).T, basis.modes
    )

    if eval_grid is None:
        eval_grid = LinearGrid(
            rho=jnp.array(1.0), M=surface.M_grid, N=surface.N_grid, NFP=surface.NFP
        )
    trans = Transform(basis=basis, grid=eval_grid, build_pinv=True)

    # Evaluate Fourier Series
    Bnorm = trans.transform(Bnorm_mn_desc_basis)

    return Bnorm


class _MagneticField(IOAble, ABC):
    """Base class for all magnetic fields.

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
        if isinstance(x, _MagneticField):
            return SumMagneticField(self, x)
        else:
            return NotImplemented

    def __neg__(self):
        return ScaledMagneticField(-1, self)

    def __sub__(self, x):
        return self.__add__(-x)

    @abstractmethod
    def compute_magnetic_field(self, coords, params=None, basis="rpz", grid=None):
        """Compute magnetic field at a set of points.

        Parameters
        ----------
        coords : array-like shape(N,3) or Grid
            cylindrical or cartesian coordinates
        params : dict, optional
            parameters to pass to scalar potential function
        basis : {"rpz", "xyz"}
            basis for input coordinates and returned magnetic field
        grid : Grid, int or None
            Grid used to discretize MagneticField object if calculating
            B from biot savart. If an integer, uses that many equally spaced
            points.

        Returns
        -------
        field : ndarray, shape(N,3)
            magnetic field at specified points

        """

    def __call__(self, coords, params=None, basis="rpz"):
        """Compute magnetic field at a set of points."""
        return self.compute_magnetic_field(coords, params, basis)

    def compute_Bnormal(
        self, surface, eval_grid=None, source_grid=None, params=None, basis="rpz"
    ):
        """Compute Bnormal from self on the given surface.

        Parameters
        ----------
        surface : Surface or Equilibrium
            Surface to calculate the magnetic field's Bnormal on.
            If an Equilibrium is supplied, will use its boundary surface.
        eval_grid : Grid, optional
            Grid of points on the surface to calculate the Bnormal at,
            if None defaults to a LinearGrid with twice
            the surface poloidal and toroidal resolutions
            points are in surface angular coordinates i.e theta and zeta
        source_grid : Grid, int or None
            Grid used to discretize MagneticField object if calculating
            B from biot savart. If an integer, uses that many equally spaced
            points.
        params : list or tuple of dict, optional
            parameters to pass to underlying field's compute_magnetic_field function.
            If None, uses the default parameters for each field.
            If a list or tuple, should have one entry for each component field.
        basis : {"rpz", "xyz"}
            basis for returned coordinates on the surface
            cylindrical "rpz" by default

        Returns
        -------
        Bnorm : ndarray
            The normal magnetic field to the surface given, of size grid.num_nodes.
        coords: ndarray
            the locations (in specified basis) at which the Bnormal was calculated

        """
        if isinstance(surface, EquilibriaFamily):
            surface = surface[-1]
        if isinstance(surface, Equilibrium):
            surface = surface.surface
        if eval_grid is None:
            eval_grid = LinearGrid(
                rho=jnp.array(1.0), M=2 * surface.M, N=2 * surface.N, NFP=surface.NFP
            )
        data = surface.compute(["x", "n_rho"], grid=eval_grid, basis="xyz")
        coords = data["x"]
        surf_normal = data["n_rho"]
        B = self.compute_magnetic_field(
            coords, basis="xyz", grid=source_grid, params=params
        )

        Bnorm = jnp.sum(B * surf_normal, axis=-1)

        if basis.lower() == "rpz":
            coords = xyz2rpz(coords)

        return Bnorm, coords

    def save_BNORM_file(
        self,
        surface,
        fname,
        basis_M=24,
        basis_N=24,
        eval_grid=None,
        source_grid=None,
        params=None,
        sym="sin",
        scale_by_curpol=True,
    ):
        """Create BNORM-style .txt file containing Bnormal Fourier coefficients.

        Parameters
        ----------
        surface : Surface or Equilibrium
            Surface to calculate the magnetic field's Bnormal on.
            If an Equilibrium is supplied, will use its boundary surface.
        fname : str
            name of file to save the BNORM Bnormal Fourier coefficients to.
        basis_M : int, optional
            Poloidal resolution of the DoubleFourierSeries used to fit the Bnormal
            on the plasma surface, by default 24
        basis_N : int, optional
            Toroidal resolution of the DoubleFourierSeries used to fit the Bnormal
            on the plasma surface, by default 24
        eval_grid : Grid, optional
            Grid of points on the surface to calculate the Bnormal at,
            if None defaults to a LinearGrid with twice
            the surface poloidal and toroidal resolutions
            points are in surface angular coordinates i.e theta and zeta
        source_grid : Grid, int or None
            Grid used to discretize MagneticField object if calculating
            B from biot savart. If an integer, uses that many equally spaced
            points.
        params : list or tuple of dict, optional
            parameters to pass to underlying field's compute_magnetic_field function.
            If None, uses the default parameters for each field.
            If a list or tuple, should have one entry for each component field.
        sym : str, optional
            if Bnormal is symmetric, by default "sin"
            NOTE: BNORM code only ever deals with sin-symmetric modes, so results
            may not be as expected if attempt to create a BNORM file with a
            non-symmetric Bnormal distribution, as only the sin-symmetric modes
            will be saved.
        scale_by_curpol : bool, optional
            Whether or not to scale the Bnormal coefficients by curpol
            which is expected by most other codes that accept BNORM files,
            by default True

        Returns
        -------
        None
        """
        if sym != "sin":
            raise UserWarning(
                "BNORM code assumes that |B| has sin symmetry,"
                + " and so BNORM file only saves the sin coefficients!"
                + " Resulting BNORM file will not contain the cos modes"
            )

        if isinstance(surface, EquilibriaFamily):
            surface = surface[-1]
        if isinstance(surface, Equilibrium):
            eq = surface
            surface = eq.surface
        else:
            eq = None
        if scale_by_curpol and eq is None:
            raise RuntimeError(
                "an Equilibrium must be supplied when scale_by_curpol is True!"
            )
        if eval_grid is None:
            eval_grid = LinearGrid(
                rho=jnp.array(1.0), M=2 * basis_M, N=2 * basis_N, NFP=surface.NFP
            )

        basis = DoubleFourierSeries(M=basis_M, N=basis_N, NFP=surface.NFP, sym=sym)
        trans = Transform(basis=basis, grid=eval_grid, build_pinv=True)

        # compute Bnormal on the grid
        Bnorm, _ = self.compute_Bnormal(
            surface, eval_grid=eval_grid, source_grid=source_grid, params=params
        )

        # fit Bnorm with Fourier Series
        Bnorm_mn = trans.fit(Bnorm)
        # convert to VMEC-style mode numbers to conform with BNORM format
        xm, xn, s, c = ptolemy_identity_rev(
            basis.modes[:, 1], basis.modes[:, 2], Bnorm_mn.reshape((1, Bnorm_mn.size))
        )

        Bnorm_xn = -xn  # need to negate Xn for BNORM code format of cos(mu+nv)

        # BNORM also scales values by curpol, a VMEC output which is calculated by
        # (source:
        #  https://princetonuniversity.github.io/FOCUS/
        #   notes/Coil_design_codes_benchmark.html )
        # "BNORM scales B_n by curpol=(2*pi/nfp)*bsubv(m=0,n=0)
        # where bsubv is the extrapolation to the last full mesh point of
        # bsubvmnc."
        # this corresponds to 2pi/NFP*G(rho=1) in DESC
        curpol = (
            (
                2
                * jnp.pi
                / surface.NFP
                * eq.compute("G", grid=LinearGrid(rho=jnp.array(1)))["G"]
            )
            if scale_by_curpol
            else 1
        )

        # BNORM assumes |B| has sin sym so c=0, so we only need s
        data = np.vstack((xm, Bnorm_xn, s * curpol)).T

        np.savetxt(f"{fname}", data, fmt="%d %d %1.12e")
        return None


class ScaledMagneticField(_MagneticField):
    """Magnetic field scaled by a scalar value.

    ie B_new = scalar * B_old

    Parameters
    ----------
    scalar : float, int
        scaling factor for magnetic field
    field : MagneticField
        base field to be scaled

    """

    _io_attrs = _MagneticField._io_attrs_ + ["_field", "_scalar"]

    def __init__(self, scalar, field):
        assert np.isscalar(scalar), "scalar must actually be a scalar value"
        assert isinstance(
            field, _MagneticField
        ), "field should be a subclass of MagneticField, got type {}".format(
            type(field)
        )
        self._scalar = scalar
        self._field = field

    def compute_magnetic_field(self, coords, params=None, basis="rpz", grid=None):
        """Compute magnetic field at a set of points.

        Parameters
        ----------
        coords : array-like shape(N,3) or Grid
            cylindrical or cartesian coordinates
        params : tuple, optional
            parameters to pass to underlying field
        basis : {"rpz", "xyz"}
            basis for input coordinates and returned magnetic field
        grid : Grid, int or None
            Grid used to discretize MagneticField object if calculating
            B from biot savart. If an integer, uses that many equally spaced
            points.

        Returns
        -------
        field : ndarray, shape(N,3)
            scaled magnetic field at specified points
        """
        return self._scalar * self._field.compute_magnetic_field(
            coords, params, basis, grid
        )


class SumMagneticField(_MagneticField):
    """Sum of two or more magnetic field sources.

    Parameters
    ----------
    fields : MagneticField
        two or more MagneticFields to add together
    """

    _io_attrs = _MagneticField._io_attrs_ + ["_fields"]

    def __init__(self, *fields):
        assert all(
            [isinstance(field, _MagneticField) for field in fields]
        ), "fields should each be a subclass of MagneticField, got {}".format(
            [type(field) for field in fields]
        )
        self._fields = fields

    def compute_magnetic_field(self, coords, params=None, basis="rpz", grid=None):
        """Compute magnetic field at a set of points.

        Parameters
        ----------
        coords : array-like shape(N,3) or Grid
            cylindrical or cartesian coordinates
        params : list or tuple of dict, optional
            parameters to pass to underlying fields. If None,
            uses the default parameters for each field. If a list or tuple, should have
            one entry for each component field.
        basis : {"rpz", "xyz"}
            basis for input coordinates and returned magnetic field
        grid : Grid, int or None
            Grid used to discretize MagneticField object if calculating
            B from biot savart. If an integer, uses that many equally spaced
            points.

        Returns
        -------
        field : ndarray, shape(N,3)
            scaled magnetic field at specified points
        """
        if params is None:
            params = [None] * len(self._fields)
        if isinstance(params, dict):
            params = [params]
        B = 0
        for i, field in enumerate(self._fields):
            B += field.compute_magnetic_field(
                coords, params[i % len(params)], basis, grid=grid
            )

        return B


class ToroidalMagneticField(_MagneticField):
    """Magnetic field purely in the toroidal (phi) direction.

    Magnitude is B0*R0/R where R0 is the major radius of the axis and B0
    is the field strength on axis

    Parameters
    ----------
    B0 : float
        field strength on axis
    R0 : major radius of axis

    """

    _io_attrs_ = _MagneticField._io_attrs_ + ["_B0", "_R0"]

    def __init__(self, B0, R0):
        assert float(B0) == B0, "B0 must be a scalar"
        assert float(R0) == R0, "R0 must be a scalar"
        self._B0 = float(B0)
        self._R0 = float(R0)

    def compute_magnetic_field(self, coords, params=None, basis="rpz", grid=None):
        """Compute magnetic field at a set of points.

        Parameters
        ----------
        coords : array-like shape(N,3) or Grid
            cylindrical or cartesian coordinates
        params : tuple, optional
            unused by this method
        basis : {"rpz", "xyz"}
            basis for input coordinates and returned magnetic field
        grid : Grid, int or None
            Grid used to discretize MagneticField object if calculating
            B from biot savart. If an integer, uses that many equally spaced
            points.
            Unused by this MagneticField class
        Returns
        -------
        field : ndarray, shape(N,3)
            magnetic field at specified points

        """
        assert basis.lower() in ["rpz", "xyz"]
        if hasattr(coords, "nodes"):
            coords = coords.nodes
        coords = jnp.atleast_2d(coords)
        if basis == "xyz":
            coords = xyz2rpz(coords)
        bp = self._B0 * self._R0 / coords[:, 0]
        brz = jnp.zeros_like(bp)
        B = jnp.array([brz, bp, brz]).T
        if basis == "xyz":
            B = rpz2xyz_vec(B, phi=coords[:, 1])

        return B


class VerticalMagneticField(_MagneticField):
    """Uniform magnetic field purely in the vertical (Z) direction.

    Parameters
    ----------
    B0 : float
        field strength

    """

    _io_attrs_ = _MagneticField._io_attrs_ + ["_B0"]

    def __init__(self, B0):
        assert np.isscalar(B0), "B0 must be a scalar"
        self._B0 = B0

    def compute_magnetic_field(self, coords, params=None, basis="rpz", grid=None):
        """Compute magnetic field at a set of points.

        Parameters
        ----------
        coords : array-like shape(N,3) or Grid
            cylindrical or cartesian coordinates
        params : tuple, optional
            unused by this method
        basis : {"rpz", "xyz"}
            basis for input coordinates and returned magnetic field
        grid : Grid, int or None
            Grid used to discretize MagneticField object if calculating
            B from biot savart. If an integer, uses that many equally spaced
            points.
            Unused by this MagneticField class

        Returns
        -------
        field : ndarray, shape(N,3)
            magnetic field at specified points

        """
        assert basis.lower() in ["rpz", "xyz"]
        if hasattr(coords, "nodes"):
            coords = coords.nodes
        coords = jnp.atleast_2d(coords)
        if basis == "xyz":
            coords = xyz2rpz(coords)
        bz = self._B0 * jnp.ones_like(coords[:, 2])
        brp = jnp.zeros_like(bz)
        B = jnp.array([brp, brp, bz]).T
        if basis == "xyz":
            B = rpz2xyz_vec(B, phi=coords[:, 1])

        return B


class PoloidalMagneticField(_MagneticField):
    """Pure poloidal magnetic field (ie in theta direction).

    Field strength is B0*iota*r/R0 where B0 is the toroidal field on axis,
    R0 is the major radius of the axis, iota is the desired rotational transform,
    and r is the minor radius centered on the magnetic axis.

    Combined with a toroidal field with the same B0 and R0, creates an
    axisymmetric field with rotational transform iota

    Note that the divergence of such a field is proportional to Z/R so is generally
    nonzero except on the midplane, but still serves as a useful test case

    Parameters
    ----------
    B0 : float
        field strength on axis
    R0 : float
        major radius of magnetic axis
    iota : float
        desired rotational transform

    """

    _io_attrs_ = _MagneticField._io_attrs_ + ["_B0", "_R0", "_iota"]

    def __init__(self, B0, R0, iota):
        assert np.isscalar(B0), "B0 must be a scalar"
        assert np.isscalar(R0), "R0 must be a scalar"
        assert np.isscalar(iota), "iota must be a scalar"
        self._B0 = B0
        self._R0 = R0
        self._iota = iota

    def compute_magnetic_field(self, coords, params=None, basis="rpz", grid=None):
        """Compute magnetic field at a set of points.

        Parameters
        ----------
        coords : array-like shape(N,3) or Grid
            cylindrical or cartesian coordinates
        params : tuple, optional
            unused by this method
        basis : {"rpz", "xyz"}
            basis for input coordinates and returned magnetic field
        grid : Grid, int or None
            Grid used to discretize MagneticField object if calculating
            B from biot savart. If an integer, uses that many equally spaced
            points.
            Unused by this MagneticField class

        Returns
        -------
        field : ndarray, shape(N,3)
            magnetic field at specified points, in cylindrical form [BR, Bphi,BZ]

        """
        assert basis.lower() in ["rpz", "xyz"]
        if hasattr(coords, "nodes"):
            coords = coords.nodes
        coords = jnp.atleast_2d(coords)
        if basis == "xyz":
            coords = xyz2rpz(coords)

        R, phi, Z = coords.T
        r = jnp.sqrt((R - self._R0) ** 2 + Z**2)
        theta = jnp.arctan2(Z, R - self._R0)
        br = -r * jnp.sin(theta)
        bp = jnp.zeros_like(br)
        bz = r * jnp.cos(theta)
        bmag = self._B0 * self._iota / self._R0
        B = bmag * jnp.array([br, bp, bz]).T
        if basis == "xyz":
            B = rpz2xyz_vec(B, phi=coords[:, 1])

        return B


class SplineMagneticField(_MagneticField):
    """Magnetic field from precomputed values on a grid.

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
        "_derivs",
        "_axisym",
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
        if len(phi) == 1:
            self._axisym = True
        else:
            self._axisym = False
        self._Z = Z
        self._BR = BR
        self._Bphi = Bphi
        self._BZ = BZ

        self._method = method
        self._extrap = extrap
        self._period = period

        self._derivs = {}
        self._derivs["BR"] = self._approx_derivs(self._BR)
        self._derivs["Bphi"] = self._approx_derivs(self._Bphi)
        self._derivs["BZ"] = self._approx_derivs(self._BZ)

    def _approx_derivs(self, Bi):
        tempdict = {}
        tempdict["fx"] = _approx_df(self._R, Bi, self._method, 0)
        tempdict["fz"] = _approx_df(self._Z, Bi, self._method, 2)
        tempdict["fxz"] = _approx_df(self._Z, tempdict["fx"], self._method, 2)
        if self._axisym:
            tempdict["fy"] = jnp.zeros_like(tempdict["fx"])
            tempdict["fxy"] = jnp.zeros_like(tempdict["fx"])
            tempdict["fyz"] = jnp.zeros_like(tempdict["fx"])
            tempdict["fxyz"] = jnp.zeros_like(tempdict["fx"])
        else:
            tempdict["fy"] = _approx_df(self._phi, Bi, self._method, 1)
            tempdict["fxy"] = _approx_df(self._phi, tempdict["fx"], self._method, 1)
            tempdict["fyz"] = _approx_df(self._Z, tempdict["fy"], self._method, 2)
            tempdict["fxyz"] = _approx_df(self._Z, tempdict["fxy"], self._method, 2)
        if self._axisym:
            for key, val in tempdict.items():
                tempdict[key] = val[:, 0, :]
        return tempdict

    def compute_magnetic_field(self, coords, params=None, basis="rpz", grid=None):
        """Compute magnetic field at a set of points.

        Parameters
        ----------
        coords : array-like shape(N,3) or Grid
            cylindrical or cartesian coordinates
        params : tuple, optional
            unused by this method
        basis : {"rpz", "xyz"}
            basis for input coordinates and returned magnetic field
        grid : Grid, int or None
            Grid used to discretize MagneticField object if calculating
            B from biot savart. If an integer, uses that many equally spaced
            points.
            Unused by this MagneticField class

        Returns
        -------
        field : ndarray, shape(N,3)
            magnetic field at specified points, in cylindrical form [BR, Bphi,BZ]

        """
        assert basis.lower() in ["rpz", "xyz"]
        if hasattr(coords, "nodes"):
            coords = coords.nodes
        coords = jnp.atleast_2d(coords)
        if basis == "xyz":
            coords = xyz2rpz(coords)
        Rq, phiq, Zq = coords.T
        if self._axisym:
            BRq = interp2d(
                Rq,
                Zq,
                self._R,
                self._Z,
                self._BR[:, 0, :],
                self._method,
                (0, 0),
                self._extrap,
                (None, None),
                **self._derivs["BR"],
            )
            Bphiq = interp2d(
                Rq,
                Zq,
                self._R,
                self._Z,
                self._Bphi[:, 0, :],
                self._method,
                (0, 0),
                self._extrap,
                (None, None),
                **self._derivs["Bphi"],
            )
            BZq = interp2d(
                Rq,
                Zq,
                self._R,
                self._Z,
                self._BZ[:, 0, :],
                self._method,
                (0, 0),
                self._extrap,
                (None, None),
                **self._derivs["BZ"],
            )

        else:
            BRq = interp3d(
                Rq,
                phiq,
                Zq,
                self._R,
                self._phi,
                self._Z,
                self._BR,
                self._method,
                (0, 0, 0),
                self._extrap,
                (None, self._period, None),
                **self._derivs["BR"],
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
                (0, 0, 0),
                self._extrap,
                (None, self._period, None),
                **self._derivs["Bphi"],
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
                (0, 0, 0),
                self._extrap,
                (None, self._period, None),
                **self._derivs["BZ"],
            )
        B = jnp.array([BRq, Bphiq, BZq]).T
        if basis == "xyz":
            B = rpz2xyz_vec(B, phi=coords[:, 1])
        return B

    @classmethod
    def from_mgrid(
        cls, mgrid_file, extcur=1, method="cubic", extrap=False, period=None
    ):
        """Create a SplineMagneticField from an "mgrid" file from MAKEGRID.

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

    @classmethod
    def from_field(
        cls, field, R, phi, Z, params=None, method="cubic", extrap=False, period=None
    ):
        """Create a splined magnetic field from another field for faster evaluation.

        Parameters
        ----------
        field : MagneticField or callable
            field to interpolate. If a callable, should take a vector of
            cylindrical coordinates and return the field in cylindrical components
        R, phi, Z : ndarray
            1d arrays of interpolation nodes in cylindrical coordinates
        params : dict, optional
            parameters passed to field
        method : str
            spline method for SplineMagneticField
        extrap : bool
            whether to extrapolate splines beyond specified R,phi,Z
        period : float
            period for phi coordinate. Usually 2pi/NFP

        """
        R, phi, Z = map(np.asarray, (R, phi, Z))
        rr, pp, zz = np.meshgrid(R, phi, Z, indexing="ij")
        shp = rr.shape
        coords = np.array([rr.flatten(), pp.flatten(), zz.flatten()]).T
        BR, BP, BZ = field.compute_magnetic_field(coords, params, basis="rpz").T
        return cls(
            R,
            phi,
            Z,
            BR.reshape(shp),
            BP.reshape(shp),
            BZ.reshape(shp),
            method,
            extrap,
            period,
        )


class ScalarPotentialField(_MagneticField):
    """Magnetic field due to a scalar magnetic potential in cylindrical coordinates.

    Parameters
    ----------
    potential : callable
        function to compute the scalar potential. Should have a signature of
        the form potential(R,phi,Z,*params) -> ndarray.
        R,phi,Z are arrays of cylindrical coordinates.
    params : dict, optional
        default parameters to pass to potential function

    """

    def __init__(self, potential, params=None):
        self._potential = potential
        self._params = params

    def compute_magnetic_field(self, coords, params=None, basis="rpz", grid=None):
        """Compute magnetic field at a set of points.

        Parameters
        ----------
        coords : array-like shape(N,3) or Grid
            cylindrical or cartesian coordinates
        params : dict, optional
            parameters to pass to scalar potential function
        basis : {"rpz", "xyz"}
            basis for input coordinates and returned magnetic field
        grid : Grid, int or None
            Grid used to discretize MagneticField object if calculating
            B from biot savart. If an integer, uses that many equally spaced
            points.
            Unused by this MagneticField class

        Returns
        -------
        field : ndarray, shape(N,3)
            magnetic field at specified points

        """
        assert basis.lower() in ["rpz", "xyz"]
        if hasattr(coords, "nodes"):
            coords = coords.nodes
        coords = jnp.atleast_2d(coords)
        if basis == "xyz":
            coords = xyz2rpz(coords)
        Rq, phiq, Zq = coords.T

        if params is None:
            params = self._params
        r, p, z = coords.T
        funR = lambda x: self._potential(x, p, z, **params)
        funP = lambda x: self._potential(r, x, z, **params)
        funZ = lambda x: self._potential(r, p, x, **params)
        br = Derivative.compute_jvp(funR, 0, (jnp.ones_like(r),), r)
        bp = Derivative.compute_jvp(funP, 0, (jnp.ones_like(p),), p)
        bz = Derivative.compute_jvp(funZ, 0, (jnp.ones_like(z),), z)
        B = jnp.array([br, bp / r, bz]).T
        if basis == "xyz":
            B = rpz2xyz_vec(B, phi=coords[:, 1])
        return B


def field_line_integrate(
    r0, z0, phis, field, params=None, rtol=1e-8, atol=1e-8, maxstep=1000
):
    """Trace field lines by integration.

    Parameters
    ----------
    r0, z0 : array-like
        initial starting coordinates for r,z on phi=phis[0] plane
    phis : array-like
        strictly increasing array of toroidal angles to output r,z at
        Note that phis is the geometric toroidal angle for positive Bphi,
        and the negative toroidal angle for negative Bphi
    field : MagneticField
        source of magnetic field to integrate
    params: dict
        parameters passed to field
    rtol, atol : float
        relative and absolute tolerances for ode integration
    maxstep : int
        maximum number of steps between different phis

    Returns
    -------
    r, z : ndarray
        arrays of r, z coordinates at specified phi angles

    """
    r0, z0, phis = map(jnp.asarray, (r0, z0, phis))
    assert r0.shape == z0.shape, "r0 and z0 must have the same shape"
    rshape = r0.shape
    r0 = r0.flatten()
    z0 = z0.flatten()
    x0 = jnp.array([r0, phis[0] * jnp.ones_like(r0), z0]).T

    @jit
    def odefun(rpz, s):
        rpz = rpz.reshape((3, -1)).T
        r = rpz[:, 0]
        br, bp, bz = field.compute_magnetic_field(rpz, params, basis="rpz").T
        return jnp.array(
            [r * br / bp * jnp.sign(bp), jnp.sign(bp), r * bz / bp * jnp.sign(bp)]
        ).squeeze()

    intfun = lambda x: odeint(odefun, x, phis, rtol=rtol, atol=atol, mxstep=maxstep)
    x = jnp.vectorize(intfun, signature="(k)->(n,k)")(x0)
    r = x[:, :, 0].T.reshape((len(phis), *rshape))
    z = x[:, :, 2].T.reshape((len(phis), *rshape))
    return r, z
