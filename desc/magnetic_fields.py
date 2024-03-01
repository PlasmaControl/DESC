"""Classes for magnetic fields."""

from abc import ABC, abstractmethod
from collections.abc import MutableSequence

import numpy as np
from interpax import approx_df, interp2d, interp3d
from jax import jacfwd
from netCDF4 import Dataset, chartostring, stringtochar

from desc.backend import cond, fori_loop, gammaln, jit, jnp, odeint
from desc.basis import DoubleFourierSeries
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec
from desc.derivatives import Derivative
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid
from desc.io import IOAble
from desc.optimizable import Optimizable, OptimizableCollection, optimizable_parameter
from desc.transform import Transform
from desc.utils import copy_coeffs, errorif, flatten_list, setdefault, warnif
from desc.vmec_utils import ptolemy_identity_fwd, ptolemy_identity_rev


def biot_savart_general(re, rs, J, dV):
    """Biot-Savart law for arbitrary sources.

    Parameters
    ----------
    re : ndarray, shape(n_eval_pts, 3)
        evaluation points to evaluate B at, in cartesian.
    rs : ndarray, shape(n_src_pts, 3)
        source points for current density J, in cartesian.
    J : ndarray, shape(n_src_pts, 3)
        current density vector at source points, in cartesian.
    dV : ndarray, shape(n_src_pts)
        volume element at source points

    Returns
    -------
    B : ndarray, shape(n,3)
        magnetic field in cartesian components at specified points
    """
    re, rs, J, dV = map(jnp.asarray, (re, rs, J, dV))
    assert J.shape == rs.shape
    JdV = J * dV[:, None]
    B = jnp.zeros_like(re)

    def body(i, B):
        r = re - rs[i, :]
        num = jnp.cross(JdV[i, :], r, axis=-1)
        den = jnp.linalg.norm(r, axis=-1) ** 3
        B = B + jnp.where(den[:, None] == 0, 0, num / den[:, None])
        return B

    return 1e-7 * fori_loop(0, J.shape[0], body, B)


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
        if np.isscalar(x) or len(x) == 1:
            return ScaledMagneticField(x, self)
        else:
            return NotImplemented

    def __rmul__(self, x):
        return self.__mul__(x)

    def __add__(self, x):
        if isinstance(x, _MagneticField):
            if isinstance(self, SumMagneticField):
                return SumMagneticField(*self, x)
            else:
                return SumMagneticField(self, x)
        else:
            return NotImplemented

    def __radd__(self, x):
        return self + x

    def __neg__(self):
        return ScaledMagneticField(-1, self)

    def __sub__(self, x):
        return self.__add__(-x)

    @abstractmethod
    def compute_magnetic_field(
        self, coords, params=None, basis="rpz", source_grid=None
    ):
        """Compute magnetic field at a set of points.

        Parameters
        ----------
        coords : array-like shape(n,3)
            Nodes to evaluate field at in [R,phi,Z] or [X,Y,Z] coordinates.
        params : dict or array-like of dict, optional
            Dictionary of optimizable parameters, eg field.params_dict.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic field.
        source_grid : Grid, int or None or array-like, optional
            Grid used to discretize MagneticField object if calculating B from
            Biot-Savart. Should NOT include endpoint at 2pi.

        Returns
        -------
        field : ndarray, shape(N,3)
            magnetic field at specified points

        """

    def __call__(self, grid, params=None, basis="rpz"):
        """Compute magnetic field at a set of points."""
        return self.compute_magnetic_field(grid, params, basis)

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
            Grid used to discretize MagneticField object if calculating B from
            Biot-Savart. Should NOT include endpoint at 2pi.
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
            coords, basis="xyz", source_grid=source_grid, params=params
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
            Grid used to discretize MagneticField object if calculating B from
            Biot-Savart. Should NOT include endpoint at 2pi.
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

    def save_mgrid(
        self,
        path,
        Rmin,
        Rmax,
        Zmin,
        Zmax,
        nR=101,
        nZ=101,
        nphi=90,
    ):
        """Save the magnetic field to an mgrid NetCDF file in "raw" format.

        Parameters
        ----------
        path : str
            File path of mgrid file to write.
        Rmin : float
            Minimum R coordinate (meters).
        Rmax : float
            Maximum R coordinate (meters).
        Zmin : float
            Minimum Z coordinate (meters).
        Zmax : float
            Maximum Z coordinate (meters).
        nR : int, optional
            Number of grid points in the R coordinate (default = 101).
        nZ : int, optional
            Number of grid points in the Z coordinate (default = 101).
        nphi : int, optional
            Number of grid points in the toroidal angle (default = 90).

        Returns
        -------
        None

        """
        # cylindrical coordinates grid
        NFP = self.NFP if hasattr(self, "_NFP") else 1
        R = np.linspace(Rmin, Rmax, nR)
        Z = np.linspace(Zmin, Zmax, nZ)
        phi = np.linspace(0, 2 * np.pi / NFP, nphi, endpoint=False)
        [PHI, ZZ, RR] = np.meshgrid(phi, Z, R, indexing="ij")
        grid = np.array([RR.flatten(), PHI.flatten(), ZZ.flatten()]).T

        # evaluate magnetic field on grid
        field = self.compute_magnetic_field(grid, basis="rpz")
        B_R = field[:, 0].reshape(nphi, nZ, nR)
        B_phi = field[:, 1].reshape(nphi, nZ, nR)
        B_Z = field[:, 2].reshape(nphi, nZ, nR)

        # write mgrid file
        file = Dataset(path, mode="w", format="NETCDF3_64BIT_OFFSET")

        # dimensions
        file.createDimension("dim_00001", 1)
        file.createDimension("stringsize", 30)
        file.createDimension("external_coil_groups", 1)
        file.createDimension("external_coils", 1)
        file.createDimension("rad", nR)
        file.createDimension("zee", nZ)
        file.createDimension("phi", nphi)

        # variables
        mgrid_mode = file.createVariable("mgrid_mode", "S1", ("dim_00001",))
        mgrid_mode[:] = stringtochar(
            np.array(["R"], "S" + str(file.dimensions["dim_00001"].size))
        )

        coil_group = file.createVariable(
            "coil_group", "S1", ("external_coil_groups", "stringsize")
        )
        coil_group[:] = stringtochar(
            np.array(
                ["single coil representing field"],
                "S" + str(file.dimensions["stringsize"].size),
            )
        )

        ir = file.createVariable("ir", np.int32)
        ir.long_name = "Number of grid points in the R coordinate."
        ir[:] = nR

        jz = file.createVariable("jz", np.int32)
        jz.long_name = "Number of grid points in the Z coordinate."
        jz[:] = nZ

        kp = file.createVariable("kp", np.int32)
        kp.long_name = "Number of grid points in the phi coordinate."
        kp[:] = nphi

        nfp = file.createVariable("nfp", np.int32)
        nfp.long_name = "Number of field periods."
        nfp[:] = NFP

        nextcur = file.createVariable("nextcur", np.int32)
        nextcur.long_name = "Number of coils."
        nextcur[:] = 1

        rmin = file.createVariable("rmin", np.float64)
        rmin.long_name = "Minimum R coordinate (m)."
        rmin[:] = Rmin

        rmax = file.createVariable("rmax", np.float64)
        rmax.long_name = "Maximum R coordinate (m)."
        rmax[:] = Rmax

        zmin = file.createVariable("zmin", np.float64)
        zmin.long_name = "Minimum Z coordinate (m)."
        zmin[:] = Zmin

        zmax = file.createVariable("zmax", np.float64)
        zmax.long_name = "Maximum Z coordinate (m)."
        zmax[:] = Zmax

        raw_coil_cur = file.createVariable(
            "raw_coil_cur", np.float64, ("external_coils",)
        )
        raw_coil_cur.long_name = "Raw coil currents (A)."
        raw_coil_cur[:] = np.array([1])  # this is 1 because mgrid_mode = "raw"

        br_001 = file.createVariable("br_001", np.float64, ("phi", "zee", "rad"))
        br_001.long_name = "B_R = radial component of magnetic field in lab frame (T)."
        br_001[:] = B_R

        bp_001 = file.createVariable("bp_001", np.float64, ("phi", "zee", "rad"))
        bp_001.long_name = (
            "B_phi = toroidal component of magnetic field in lab frame (T)."
        )
        bp_001[:] = B_phi

        bz_001 = file.createVariable("bz_001", np.float64, ("phi", "zee", "rad"))
        bz_001.long_name = (
            "B_Z = vertical component of magnetic field in lab frame (T)."
        )
        bz_001[:] = B_Z

        file.close()


class ScaledMagneticField(_MagneticField, Optimizable):
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

    def __init__(self, scale, field):
        assert (
            np.isscalar(scale) or np.asarray(scale).size == 1
        ), "scale must be a scalar value"
        scale = float(scale)
        assert isinstance(
            field, _MagneticField
        ), "field should be a subclass of MagneticField, got type {}".format(
            type(field)
        )
        object.__setattr__(self, "_scale", scale)
        object.__setattr__(self, "_field", field)
        object.__setattr__(
            self, "_optimizable_params", field.optimizable_params + ["scale"]
        )

    @optimizable_parameter
    @property
    def scale(self):
        """float: scaling factor for magnetic field."""
        return self._scale

    @scale.setter
    def scale(self, new):
        assert float(new) == new, "scale must be a scalar"
        self._scale = new

    # want this class to pretend like its the underlying field
    def __getattr__(self, attr):
        if attr in ["_scale", "_optimizable_params"]:
            return getattr(self, attr)
        return getattr(self._field, attr)

    def __setattr__(self, name, value):
        if name in ["scale", "_scale", "_optimizable_params"]:
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_field"), name, value)

    def __hasattr__(self, attr):
        return hasattr(self, attr) or hasattr(self._field, attr)

    def compute_magnetic_field(
        self, coords, params=None, basis="rpz", source_grid=None
    ):
        """Compute magnetic field at a set of points.

        Parameters
        ----------
        coords : array-like shape(n,3)
            Nodes to evaluate field at in [R,phi,Z] or [X,Y,Z] coordinates.
        params : dict or array-like of dict, optional
            Dictionary of optimizable parameters, eg field.params_dict.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic field.
        source_grid : Grid, int or None or array-like, optional
            Grid used to discretize MagneticField object if calculating B from
            Biot-Savart. Should NOT include endpoint at 2pi.

        Returns
        -------
        field : ndarray, shape(N,3)
            scaled magnetic field at specified points

        """
        return self._scale * self._field.compute_magnetic_field(
            coords, params, basis, source_grid
        )


class SumMagneticField(_MagneticField, MutableSequence, OptimizableCollection):
    """Sum of two or more magnetic field sources.

    Parameters
    ----------
    fields : MagneticField
        two or more MagneticFields to add together
    """

    _io_attrs = _MagneticField._io_attrs_ + ["_fields"]

    def __init__(self, *fields):
        fields = flatten_list(fields, flatten_tuple=True)
        assert all(
            [isinstance(field, _MagneticField) for field in fields]
        ), "fields should each be a subclass of MagneticField, got {}".format(
            [type(field) for field in fields]
        )
        self._fields = fields

    def compute_magnetic_field(
        self, coords, params=None, basis="rpz", source_grid=None
    ):
        """Compute magnetic field at a set of points.

        Parameters
        ----------
        coords : array-like shape(n,3)
            Nodes to evaluate field at in [R,phi,Z] or [X,Y,Z] coordinates.
        params : dict or array-like of dict, optional
            Dictionary of optimizable parameters, eg field.params_dict.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic field.
        source_grid : Grid, int or None or array-like, optional
            Grid used to discretize MagneticField object if calculating B from
            Biot-Savart. Should NOT include endpoint at 2pi.

        Returns
        -------
        field : ndarray, shape(N,3)
            scaled magnetic field at specified points

        """
        if params is None:
            params = [None] * len(self._fields)
        if isinstance(params, dict):
            params = [params]
        if source_grid is None:
            source_grid = [None] * len(self._fields)
        if not isinstance(source_grid, (list, tuple)):
            source_grid = [source_grid]
        if len(source_grid) != len(self._fields):
            # ensure that if source_grid is shorter, that it is simply repeated so that
            # zip does not terminate early
            source_grid = source_grid * len(self._fields)

        B = 0
        for i, (field, g) in enumerate(zip(self._fields, source_grid)):
            B += field.compute_magnetic_field(
                coords, params[i % len(params)], basis, source_grid=g
            )

        return B

    # dunder methods required by MutableSequence
    def __getitem__(self, i):
        return self._fields[i]

    def __setitem__(self, i, new_item):
        if not isinstance(new_item, _MagneticField):
            raise TypeError(
                "Members of SumMagneticField must be of type MagneticField."
            )
        self._fields[i] = new_item

    def __delitem__(self, i):
        del self._fields[i]

    def __len__(self):
        return len(self._fields)

    def insert(self, i, new_item):
        """Insert a new field into the sum at position i."""
        if not isinstance(new_item, _MagneticField):
            raise TypeError(
                "Members of SumMagneticField must be of type MagneticField."
            )
        self._fields.insert(i, new_item)


class ToroidalMagneticField(_MagneticField, Optimizable):
    """Magnetic field purely in the toroidal (phi) direction.

    Magnitude is B0*R0/R where R0 is the major radius of the axis and B0
    is the field strength on axis

    Parameters
    ----------
    B0 : float
        field strength on axis
    R0 : float
        major radius of axis

    """

    _io_attrs_ = _MagneticField._io_attrs_ + ["_B0", "_R0"]

    def __init__(self, B0, R0):
        assert float(R0) == R0, "R0 must be a scalar"
        self.B0 = float(B0)
        self.R0 = float(R0)

    @optimizable_parameter
    @property
    def R0(self):
        """float: major radius of axis."""
        return self._R0

    @R0.setter
    def R0(self, new):
        assert float(new) == new, "R0 must be a scalar"
        self._R0 = new

    @optimizable_parameter
    @property
    def B0(self):
        """float: field strength on axis."""
        return self._B0

    @B0.setter
    def B0(self, new):
        assert float(new) == new, "B0 must be a scalar"
        self._B0 = new

    def compute_magnetic_field(
        self, coords, params=None, basis="rpz", source_grid=None
    ):
        """Compute magnetic field at a set of points.

        Parameters
        ----------
        coords : array-like shape(n,3)
            Nodes to evaluate field at in [R,phi,Z] or [X,Y,Z] coordinates.
        params : dict or array-like of dict, optional
            Dict of values for R0 and B0.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic field.
        source_grid : Grid, int or None or array-like, optional
            Unused by this MagneticField class.

        Returns
        -------
        field : ndarray, shape(N,3)
            magnetic field at specified points

        """
        params = setdefault(params, {})
        B0 = params.get("B0", self.B0)
        R0 = params.get("R0", self.R0)

        assert basis.lower() in ["rpz", "xyz"]
        coords = jnp.atleast_2d(coords)
        if basis == "xyz":
            coords = xyz2rpz(coords)
        bp = B0 * R0 / coords[:, 0]
        brz = jnp.zeros_like(bp)
        B = jnp.array([brz, bp, brz]).T
        if basis == "xyz":
            B = rpz2xyz_vec(B, phi=coords[:, 1])

        return B


class VerticalMagneticField(_MagneticField, Optimizable):
    """Uniform magnetic field purely in the vertical (Z) direction.

    Parameters
    ----------
    B0 : float
        field strength

    """

    _io_attrs_ = _MagneticField._io_attrs_ + ["_B0"]

    def __init__(self, B0):
        self.B0 = B0

    @optimizable_parameter
    @property
    def B0(self):
        """float: field strength."""
        return self._B0

    @B0.setter
    def B0(self, new):
        assert float(new) == new, "B0 must be a scalar"
        self._B0 = new

    def compute_magnetic_field(
        self, coords, params=None, basis="rpz", source_grid=None
    ):
        """Compute magnetic field at a set of points.

        Parameters
        ----------
        coords : array-like shape(n,3)
            Nodes to evaluate field at in [R,phi,Z] or [X,Y,Z] coordinates.
        params : dict or array-like of dict, optional
            Dict of values for B0.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic field.
        source_grid : Grid, int or None or array-like, optional
            Unused by this MagneticField class.

        Returns
        -------
        field : ndarray, shape(N,3)
            magnetic field at specified points

        """
        params = setdefault(params, {})
        B0 = params.get("B0", self.B0)

        assert basis.lower() in ["rpz", "xyz"]
        coords = jnp.atleast_2d(coords)
        if basis == "xyz":
            coords = xyz2rpz(coords)
        bz = B0 * jnp.ones_like(coords[:, 2])
        brp = jnp.zeros_like(bz)
        B = jnp.array([brp, brp, bz]).T
        if basis == "xyz":
            B = rpz2xyz_vec(B, phi=coords[:, 1])

        return B


class PoloidalMagneticField(_MagneticField, Optimizable):
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
        self.B0 = B0
        self.R0 = R0
        self.iota = iota

    @optimizable_parameter
    @property
    def R0(self):
        """float: major radius of axis."""
        return self._R0

    @R0.setter
    def R0(self, new):
        assert float(new) == new, "R0 must be a scalar"
        self._R0 = new

    @optimizable_parameter
    @property
    def B0(self):
        """float: field strength on axis."""
        return self._B0

    @B0.setter
    def B0(self, new):
        assert float(new) == new, "B0 must be a scalar"
        self._B0 = new

    @optimizable_parameter
    @property
    def iota(self):
        """float: desired rotational transform."""
        return self._iota

    @iota.setter
    def iota(self, new):
        assert float(new) == new, "iota must be a scalar"
        self._iota = new

    def compute_magnetic_field(
        self, coords, params=None, basis="rpz", source_grid=None
    ):
        """Compute magnetic field at a set of points.

        Parameters
        ----------
        coords : array-like shape(n,3)
            Nodes to evaluate field at in [R,phi,Z] or [X,Y,Z] coordinates.
        params : dict or array-like of dict, optional
            Dict of values for R0, B0, and iota.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic field.
        source_grid : Grid, int or None or array-like, optional
            Unused by this MagneticField class.

        Returns
        -------
        field : ndarray, shape(N,3)
            magnetic field at specified points, in cylindrical form [BR, Bphi,BZ]

        """
        params = setdefault(params, {})
        B0 = params.get("B0", self.B0)
        R0 = params.get("R0", self.R0)
        iota = params.get("iota", self.iota)

        assert basis.lower() in ["rpz", "xyz"]
        coords = jnp.atleast_2d(coords)
        if basis == "xyz":
            coords = xyz2rpz(coords)

        R, phi, Z = coords.T
        r = jnp.sqrt((R - R0) ** 2 + Z**2)
        theta = jnp.arctan2(Z, R - R0)
        br = -r * jnp.sin(theta)
        bp = jnp.zeros_like(br)
        bz = r * jnp.cos(theta)
        bmag = B0 * iota / R0
        B = bmag * jnp.array([br, bp, bz]).T
        if basis == "xyz":
            B = rpz2xyz_vec(B, phi=coords[:, 1])

        return B


class SplineMagneticField(_MagneticField, Optimizable):
    """Magnetic field from precomputed values on a grid.

    Parameters
    ----------
    R : array-like, size(NR)
        R coordinates where field is specified
    phi : array-like, size(Nphi)
        phi coordinates where field is specified
    Z : array-like, size(NZ)
        Z coordinates where field is specified
    BR : array-like, shape(NR,Nphi,NZ,Ngroups)
        radial magnetic field on grid
    Bphi : array-like, shape(NR,Nphi,NZ,Ngroups)
        toroidal magnetic field on grid
    BZ : array-like, shape(NR,Nphi,NZ,Ngroups)
        vertical magnetic field on grid
    currents : array-like, shape(Ngroups)
        Currents or scaling factors for each field group.
    NFP : int, optional
        Number of toroidal field periods.
    method : str
        interpolation method.
    extrap : bool, optional
        whether to extrapolate beyond the domain of known field values or return nan.

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
        "_derivs",
        "_axisym",
        "_currents",
        "_NFP",
    ]

    def __init__(
        self, R, phi, Z, BR, Bphi, BZ, currents=1.0, NFP=1, method="cubic", extrap=False
    ):
        R, phi, Z, currents = map(jnp.atleast_1d, (R, phi, Z, currents))
        assert R.ndim == 1
        assert phi.ndim == 1
        assert Z.ndim == 1
        assert currents.ndim == 1
        shape = (R.size, phi.size, Z.size, currents.size)

        def _atleast_4d(x):
            x = jnp.atleast_3d(x)
            if x.ndim < 4:
                x = x.reshape(x.shape + (1,))
            return x

        BR, Bphi, BZ = map(_atleast_4d, (BR, Bphi, BZ))
        assert BR.shape == Bphi.shape == BZ.shape == shape

        self._R = R
        self._phi = phi
        self._Z = Z
        if len(phi) == 1:
            self._axisym = True
        else:
            self._axisym = False

        self._BR = BR
        self._Bphi = Bphi
        self._BZ = BZ

        self._currents = currents

        self._NFP = NFP
        self._method = method
        self._extrap = extrap

        self._derivs = {}
        self._derivs["BR"] = self._approx_derivs(self._BR)
        self._derivs["Bphi"] = self._approx_derivs(self._Bphi)
        self._derivs["BZ"] = self._approx_derivs(self._BZ)

    @property
    def NFP(self):
        """int: Number of toroidal field periods."""
        return self._NFP

    @optimizable_parameter
    @property
    def currents(self):
        """ndarray: currents or scaling factors for each field group."""
        return self._currents

    @currents.setter
    def currents(self, new):
        new = jnp.atleast_1d(new)
        assert len(new) == len(self.currents)
        self._currents = new

    def _approx_derivs(self, Bi):
        tempdict = {}
        tempdict["fx"] = approx_df(self._R, Bi, self._method, 0)
        tempdict["fz"] = approx_df(self._Z, Bi, self._method, 2)
        tempdict["fxz"] = approx_df(self._Z, tempdict["fx"], self._method, 2)
        if self._axisym:
            tempdict["fy"] = jnp.zeros_like(tempdict["fx"])
            tempdict["fxy"] = jnp.zeros_like(tempdict["fx"])
            tempdict["fyz"] = jnp.zeros_like(tempdict["fx"])
            tempdict["fxyz"] = jnp.zeros_like(tempdict["fx"])
        else:
            tempdict["fy"] = approx_df(self._phi, Bi, self._method, 1)
            tempdict["fxy"] = approx_df(self._phi, tempdict["fx"], self._method, 1)
            tempdict["fyz"] = approx_df(self._Z, tempdict["fy"], self._method, 2)
            tempdict["fxyz"] = approx_df(self._Z, tempdict["fxy"], self._method, 2)
        if self._axisym:
            for key, val in tempdict.items():
                tempdict[key] = val[:, 0, :]
        return tempdict

    def compute_magnetic_field(
        self, coords, params=None, basis="rpz", source_grid=None
    ):
        """Compute magnetic field at a set of points.

        Parameters
        ----------
        coords : array-like shape(n,3)
            Nodes to evaluate field at in [R,phi,Z] or [X,Y,Z] coordinates.
        params : dict or array-like of dict, optional
            Dictionary of optimizable parameters, eg field.params_dict.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic field.
        source_grid : Grid, int or None
            Unused by this MagneticField class.

        Returns
        -------
        field : ndarray, shape(N,3)
            magnetic field at specified points, in cylindrical form [BR, Bphi,BZ]

        """
        assert basis.lower() in ["rpz", "xyz"]
        currents = self.currents if params is None else params["currents"]
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
                (None, 2 * np.pi / self.NFP, None),
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
                (None, 2 * np.pi / self.NFP, None),
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
                (None, 2 * np.pi / self.NFP, None),
                **self._derivs["BZ"],
            )
        # BRq etc shape(nq, ngroups)
        B = jnp.stack([BRq, Bphiq, BZq], axis=1)
        # B shape(nq, 3, ngroups)
        B = jnp.sum(B * currents, axis=-1)
        if basis == "xyz":
            B = rpz2xyz_vec(B, phi=coords[:, 1])
        return B

    @classmethod
    def from_mgrid(cls, mgrid_file, extcur=None, method="cubic", extrap=False):
        """Create a SplineMagneticField from an "mgrid" file from MAKEGRID.

        Parameters
        ----------
        mgrid_file : str or path-like
            File path to mgrid netCDF file to load from.
        extcur : array-like, optional
            Currents for each coil group. They default to the coil currents from the
            mgrid file for "scaled" mode, or to 1 for "raw" mode.
        method : str
            Interpolation method.
        extrap : bool
            Whether to extrapolate beyond the domain of known field values (True)
            or return NaN (False).

        """
        mgrid = Dataset(mgrid_file, "r")
        mode = chartostring(mgrid["mgrid_mode"][()])
        if extcur is None:
            if mode == "S":  # "scaled"
                extcur = np.array(mgrid["raw_coil_cur"])  # raw coil currents (A)
            else:  # "raw"
                extcur = 1  # coil current scaling factor
        nextcur = int(mgrid["nextcur"][()])  # number of coils
        extcur = np.broadcast_to(extcur, nextcur)

        # compute grid knots in cylindrical coordinates
        ir = int(mgrid["ir"][()])  # number of grid points in the R coordinate
        jz = int(mgrid["jz"][()])  # number of grid points in the Z coordinate
        kp = int(mgrid["kp"][()])  # number of grid points in the phi coordinate
        Rmin = mgrid["rmin"][()]  # Minimum R coordinate (m)
        Rmax = mgrid["rmax"][()]  # Maximum R coordinate (m)
        Zmin = mgrid["zmin"][()]  # Minimum Z coordinate (m)
        Zmax = mgrid["zmax"][()]  # Maximum Z coordinate (m)
        nfp = int(mgrid["nfp"][()])  # Number of field periods
        Rgrid = np.linspace(Rmin, Rmax, ir)
        Zgrid = np.linspace(Zmin, Zmax, jz)
        pgrid = 2.0 * np.pi / (nfp * kp) * np.arange(kp)

        # sum magnetic fields from each coil
        br = np.zeros([kp, jz, ir, nextcur])
        bp = np.zeros([kp, jz, ir, nextcur])
        bz = np.zeros([kp, jz, ir, nextcur])
        for i in range(nextcur):
            coil_id = "%03d" % (i + 1,)
            br[:, :, :, i] += mgrid["br_" + coil_id][()]  # B_R radial magnetic field
            bp[:, :, :, i] += mgrid["bp_" + coil_id][()]  # B_phi toroidal field (T)
            bz[:, :, :, i] += mgrid["bz_" + coil_id][()]  # B_Z vertical magnetic field

        # shift axes to correct order
        br = np.moveaxis(br, (0, 1, 2), (1, 2, 0))
        bp = np.moveaxis(bp, (0, 1, 2), (1, 2, 0))
        bz = np.moveaxis(bz, (0, 1, 2), (1, 2, 0))

        mgrid.close()
        return cls(Rgrid, pgrid, Zgrid, br, bp, bz, extcur, nfp, method, extrap)

    @classmethod
    def from_field(
        cls, field, R, phi, Z, params=None, method="cubic", extrap=False, NFP=1
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
        NFP : int, optional
            Number of toroidal field periods.

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
            currents=1.0,
            NFP=NFP,
            method=method,
            extrap=extrap,
        )

    def tree_flatten(self):
        """Convert DESC objects to JAX pytrees."""
        # the default flattening method in the IOAble base class assumes all floats
        # are non-static, but for the periodic BC to work we need the period to be
        # a static value, so we override the default tree flatten/unflatten method
        # so that we can pass a SplineMagneticField into a jitted function such as
        # an objective.
        static = ["_method", "_extrap", "_period", "_axisym"]
        children = {key: val for key, val in self.__dict__.items() if key not in static}
        aux_data = tuple(
            [(key, val) for key, val in self.__dict__.items() if key in static]
        )
        return ((children,), aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreate a DESC object from JAX pytree."""
        obj = cls.__new__(cls)
        obj.__dict__.update(children[0])
        for kv in aux_data:
            setattr(obj, kv[0], kv[1])
        return obj


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

    def compute_magnetic_field(
        self, coords, params=None, basis="rpz", source_grid=None
    ):
        """Compute magnetic field at a set of points.

        Parameters
        ----------
        coords : array-like shape(n,3)
            Nodes to evaluate field at in [R,phi,Z] or [X,Y,Z] coordinates.
        params : dict or array-like of dict, optional
            Dictionary of optimizable parameters, eg field.params_dict.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic field.
        source_grid : Grid, int or None
            Unused by this MagneticField class.

        Returns
        -------
        field : ndarray, shape(N,3)
            magnetic field at specified points

        """
        assert basis.lower() in ["rpz", "xyz"]
        coords = jnp.atleast_2d(coords)
        coords = coords.astype(float)  # ensure coords are float
        if basis == "xyz":
            coords = xyz2rpz(coords)

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


class DommaschkPotentialField(ScalarPotentialField):
    """Magnetic field due to a Dommaschk scalar magnetic potential in rpz coordinates.

        From Dommaschk 1986 paper https://doi.org/10.1016/0010-4655(86)90109-8

        this is the field due to the dommaschk potential (eq. 1) for
        a given set of m,l indices and their corresponding
        coefficients a_ml, b_ml, c_ml d_ml.

    Parameters
    ----------
    ms : 1D array-like of int
        first indices of V_m_l terms (eq. 12 of reference)
    ls : 1D array-like of int
        second indices of V_m_l terms (eq. 12 of reference)
    a_arr : 1D array-like of float
        a_m_l coefficients of V_m_l terms, which multiply the cos(m*phi)*D_m_l terms
    b_arr : 1D array-like of float
        b_m_l coefficients of V_m_l terms, which multiply the sin(m*phi)*D_m_l terms
    c_arr : 1D array-like of float
        c_m_l coefficients of V_m_l terms, which multiply the cos(m*phi)*N_m_l-1 term
    d_arr : 1D array-like of float
        d_m_l coefficients of V_m_l terms, which multiply the sin(m*phi)*N_m_l-1 terms
    B0: float
        scale strength of the magnetic field's 1/R portion

    """

    def __init__(
        self,
        ms=jnp.array([0]),
        ls=jnp.array([0]),
        a_arr=jnp.array([0.0]),
        b_arr=jnp.array([0.0]),
        c_arr=jnp.array([0.0]),
        d_arr=jnp.array([0.0]),
        B0=1.0,
    ):
        ms = jnp.atleast_1d(ms)
        ls = jnp.atleast_1d(ls)
        a_arr = jnp.atleast_1d(a_arr)
        b_arr = jnp.atleast_1d(b_arr)
        c_arr = jnp.atleast_1d(c_arr)
        d_arr = jnp.atleast_1d(d_arr)

        assert (
            ms.size == ls.size == a_arr.size == b_arr.size == c_arr.size == d_arr.size
        ), "Passed in arrays must all be of the same size!"
        assert not jnp.any(
            jnp.logical_or(ms < 0, ls < 0)
        ), "m and l mode numbers must be >= 0!"
        assert (
            jnp.isscalar(B0) or jnp.atleast_1d(B0).size == 1
        ), "B0 should be a scalar value!"

        params = {}
        params["ms"] = ms
        params["ls"] = ls
        params["a_arr"] = a_arr
        params["b_arr"] = b_arr
        params["c_arr"] = c_arr
        params["d_arr"] = d_arr
        params["B0"] = B0

        super().__init__(dommaschk_potential, params)

    @classmethod
    def fit_magnetic_field(  # noqa: C901 - FIXME - simplify
        cls, field, coords, max_m, max_l, sym=False, verbose=1
    ):
        """Fit a vacuum magnetic field with a Dommaschk Potential field.

        Parameters
        ----------
            field (MagneticField or callable or ndarray): magnetic field to fit
                if callable, must accept (num_nodes,3) array of rpz coords as argument
                    and output (num_nodes,3) as the B field in cylindrical rpz basis.
                if ndarray, must be an ndarray of the magnetic field in rpz,
                    of shape (num_nodes,3) with the columns being (B_R, B_phi, B_Z)
            coords (ndarray): shape (num_nodes,3) of R,phi,Z points to fit field at
            max_m (int): maximum m to use for Dommaschk Potentials
            max_l (int): maximum l to use for Dommaschk Potentials
            sym (bool): if field is stellarator symmetric or not.
                if True, only stellarator-symmetric modes will
                be included in the fitting
            verbose (int): verbosity level of fitting routine, > 0 prints residuals
        """
        # We seek c in  Ac = b
        # A will be the BR, Bphi and BZ from each individual
        # dommaschk potential basis function evaluated at each node
        # c is the dommaschk potential coefficients
        # c will be [B0, a_00, a_10, a_01, a_11... etc]
        # b is the magnetic field at each node which we are fitting
        if isinstance(field, _MagneticField):
            B = field.compute_magnetic_field(coords)
        elif callable(field):
            B = field(coords)
        else:  # it must be the field evaluated at the passed-in coords
            B = field
        # TODO: add basis argument for if passed-in field or callable
        # evaluates rpz or xyz basis magnetic field vector,
        # and what basis coords is

        #########
        # make b
        #########
        # we will have the rhs be 3*num_nodes in length (bc of vector B)

        rhs = jnp.vstack((B[:, 0], B[:, 1], B[:, 2])).T.flatten(order="F")

        #####################
        # b is made, now do A
        #####################
        num_modes = 1 + (max_l + 1) * (max_m + 1) * 4
        # TODO: if symmetric, technically only need half the modes
        # however, the field and functions are setup to accept equal
        # length arrays for a,b,c,d, so we will just zero out the
        # modes that don't fit symmetry, but in future
        # should refactor code to have a 3rd index so that
        # we have a = V_ml0, b = V_ml1, c = V_ml2, d = V_ml3
        # and the modes array can then be [m,l,x] where x is 0,1,2,3
        # and we dont need to keep track of a,b,c,d separately

        # TODO: technically we can drop some modes
        # since if max_l=0, there are only ever nonzero terms for a and b
        # and if max_m=0, there are only ever nonzero terms for a and c
        # but since we are only fitting in a least squares sense,
        # and max_l and max_m should probably be both nonzero anyways,
        # this is not an issue right now

        # mode numbers
        ms = []
        ls = []

        # order of coeffs in the vector c are B0, a_ml, b_ml, c_ml, d_ml
        a_s = []
        b_s = []
        c_s = []
        d_s = []
        zero_due_to_sym_inds = []
        abcd_zero_due_to_sym_inds = [
            [],
            [],
            [],
            [],
        ]  # indices that should be 0 due to symmetry
        for l in range(max_l + 1):
            for m in range(max_m + 1):
                if not sym:
                    pass  # no sym, use all coefs
                elif l // 2 == 0:
                    zero_due_to_sym_inds = [0, 3]  # a=d=0 for even l with sym
                elif l // 2 == 1:
                    zero_due_to_sym_inds = [1, 2]  # b=c=0 for odd l with sym
                for which_coef in range(4):
                    if which_coef == 0:
                        a_s.append(1)
                    elif which_coef == 1:
                        b_s.append(1)
                    elif which_coef == 2:
                        c_s.append(1)
                    elif which_coef == 3:
                        d_s.append(1)
                    if which_coef in zero_due_to_sym_inds:
                        abcd_zero_due_to_sym_inds[which_coef].append(0)
                    else:
                        abcd_zero_due_to_sym_inds[which_coef].append(1)

                ms.append(m)
                ls.append(l)
        for i in range(4):
            abcd_zero_due_to_sym_inds[i] = jnp.asarray(abcd_zero_due_to_sym_inds[i])
        assert (len(a_s) + len(b_s) + len(c_s) + len(d_s)) == num_modes - 1
        params = {
            "ms": ms,
            "ls": ls,
            "a_arr": a_s,
            "b_arr": b_s,
            "c_arr": c_s,
            "d_arr": d_s,
            "B0": 0.0,
        }
        n = (
            round(num_modes - 1) / 4
        )  # how many l-m mode pairs there are, also is len(a_s)
        n = int(n)
        domm_field = DommaschkPotentialField(0, 0, 0, 0, 0, 0, 1)

        def get_B_dom(coords, X, ms, ls):
            """Fxn wrapper to find jacobian of dommaschk B wrt coefs a,b,c,d."""
            # zero out any terms that should be zero due to symmetry, which
            # we cataloged earlier for each a_arr,b_arr,c_arr,d_arr
            # that way the resulting modes after pinv don't contain them either
            return domm_field.compute_magnetic_field(
                coords,
                params={
                    "ms": jnp.asarray(ms),
                    "ls": jnp.asarray(ls),
                    "a_arr": jnp.asarray(X[1 : n + 1]) * abcd_zero_due_to_sym_inds[0],
                    "b_arr": jnp.asarray(X[n + 1 : 2 * n + 1])
                    * abcd_zero_due_to_sym_inds[1],
                    "c_arr": jnp.asarray(X[2 * n + 1 : 3 * n + 1])
                    * abcd_zero_due_to_sym_inds[2],
                    "d_arr": jnp.asarray(X[3 * n + 1 : 4 * n + 1])
                    * abcd_zero_due_to_sym_inds[3],
                    "B0": X[0],
                },
            )

        X = []
        for key in ["B0", "a_arr", "b_arr", "c_arr", "d_arr"]:
            obj = params[key]
            if isinstance(obj, list):
                X += obj
            else:
                X += [obj]
        X = jnp.asarray(X)

        jac = jit(jacfwd(get_B_dom, argnums=1))(coords, X, params["ms"], params["ls"])

        A = jac.reshape((rhs.size, len(X)), order="F")

        # now solve Ac=b for the coefficients c

        # TODO: use min singular value to give sense of cond number?
        c, res, _, _ = jnp.linalg.lstsq(A, rhs)

        if verbose > 0:
            # res is a list of len(1) so index into it
            print(f"Sum of Squares Residual of fit: {res[0]:1.4e} T")

        # recover the params from the c coefficient vector
        B0 = c[0]

        # we zero out the terms that should be zero due to symmetry here
        a_arr = c[1 : n + 1] * abcd_zero_due_to_sym_inds[0]
        b_arr = c[n + 1 : 2 * n + 1] * abcd_zero_due_to_sym_inds[1]
        c_arr = c[2 * n + 1 : 3 * n + 1] * abcd_zero_due_to_sym_inds[2]
        d_arr = c[3 * n + 1 : 4 * n + 1] * abcd_zero_due_to_sym_inds[3]

        return cls(ms, ls, a_arr, b_arr, c_arr, d_arr, B0)


def field_line_integrate(
    r0,
    z0,
    phis,
    field,
    params=None,
    source_grid=None,
    rtol=1e-8,
    atol=1e-8,
    maxstep=1000,
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
    source_grid : Grid, optional
        Collocation points used to discretize source field.
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
        br, bp, bz = field.compute_magnetic_field(
            rpz, params, basis="rpz", source_grid=source_grid
        ).T
        return jnp.array(
            [r * br / bp * jnp.sign(bp), jnp.sign(bp), r * bz / bp * jnp.sign(bp)]
        ).squeeze()

    intfun = lambda x: odeint(odefun, x, phis, rtol=rtol, atol=atol, mxstep=maxstep)
    x = jnp.vectorize(intfun, signature="(k)->(n,k)")(x0)
    r = x[:, :, 0].T.reshape((len(phis), *rshape))
    z = x[:, :, 2].T.reshape((len(phis), *rshape))
    return r, z


### Dommaschk potential utility functions ###

# based off Representations for vacuum potentials in stellarators
# https://doi.org/10.1016/0010-4655(86)90109-8

# written with naive for loops initially and can jax-ify later

true_fun = lambda m_n: 0.0  # used for returning 0 when conditionals evaluate to True


@jit
def gamma(n):
    """Gamma function, only implemented for integers (equiv to factorial of (n-1))."""
    return jnp.exp(gammaln(n))


@jit
def alpha(m, n):
    """Alpha of eq 27, 1st ind comes from C_m_k, 2nd is the subscript of alpha."""
    # modified for eqns 31 and 32

    def false_fun(m_n):
        m, n = m_n
        return (-1) ** n / (gamma(m + n + 1) * gamma(n + 1) * 2.0 ** (2 * n + m))

    def bool_fun(n):
        return n < 0

    return cond(
        bool_fun(n),
        true_fun,
        false_fun,
        (
            m,
            n,
        ),
    )


@jit
def alphastar(m, n):
    """Alphastar of eq 27, 1st ind comes from C_m_k, 2nd is the subscript of alpha."""

    def false_fun(m_n):  # modified for eqns 31 and 32
        m, n = m_n
        return (2 * n + m) * alpha(m, n)

    return cond(n < 0, true_fun, false_fun, (m, n))


@jit
def beta(m, n):
    """Beta of eq 28, modified for eqns 31 and 32."""

    def false_fun(m_n):
        m, n = m_n
        return gamma(m - n) / (gamma(n + 1) * 2.0 ** (2 * n - m + 1))

    return cond(jnp.logical_or(n < 0, n >= m), true_fun, false_fun, (m, n))


@jit
def betastar(m, n):
    """Beta* of eq 28, modified for eqns 31 and 32."""

    def false_fun(m_n):
        m, n = m_n
        return (2 * n - m) * beta(m, n)

    return cond(jnp.logical_or(n < 0, n >= m), true_fun, false_fun, (m, n))


@jit
def gamma_n(m, n):
    """gamma_n of eq 33."""

    def body_fun(i, val):
        return val + 1 / i + 1 / (m + i)

    def false_fun(m_n):
        m, n = m_n
        return alpha(m, n) / 2 * fori_loop(1, n, body_fun, 0)

    return cond(n <= 0, true_fun, false_fun, (m, n))


@jit
def gamma_nstar(m, n):
    """gamma_n star of eq 33."""

    def false_fun(m_n):
        m, n = m_n
        return (2 * n + m) * gamma_n(m, n)

    return cond(n <= 0, true_fun, false_fun, (m, n))


@jit
def CD_m_k(R, m, k):
    """Eq 31 of Dommaschk paper."""

    def body_fun(j, val):
        result = (
            val
            + (
                -(
                    alpha(m, j)
                    * (
                        alphastar(m, k - m - j) * jnp.log(R)
                        + gamma_nstar(m, k - m - j)
                        - alpha(m, k - m - j)
                    )
                    - gamma_n(m, j) * alphastar(m, k - m - j)
                    + alpha(m, j) * betastar(m, k - j)
                )
                * R ** (2 * j + m)
            )
            + beta(m, j) * alphastar(m, k - j) * R ** (2 * j - m)
        )
        return result

    return fori_loop(0, k + 1, body_fun, jnp.zeros_like(R))


@jit
def CN_m_k(R, m, k):
    """Eq 32 of Dommaschk paper."""

    def body_fun(j, val):
        result = (
            val
            + (
                (
                    alpha(m, j)
                    * (alpha(m, k - m - j) * jnp.log(R) + gamma_n(m, k - m - j))
                    - gamma_n(m, j) * alpha(m, k - m - j)
                    + alpha(m, j) * beta(m, k - j)
                )
                * R ** (2 * j + m)
            )
            - beta(m, j) * alpha(m, k - j) * R ** (2 * j - m)
        )
        return result

    return fori_loop(0, k + 1, body_fun, jnp.zeros_like(R))


@jit
def D_m_n(R, Z, m, n):
    """D_m_n term in eqn 8 of Dommaschk paper."""
    # the sum comes from fact that D_mn = I_mn and the def of I_mn in eq 2 of the paper

    def body_fun(k, val):
        coef = CD_m_k(R, m, k) / gamma(n - 2 * k + 1)
        exp = n - 2 * k
        # derivative of 0**0 is ill defined, so we do this to enforce it being 0
        exp = jnp.where((Z == 0) & (exp == 0), 1, exp)
        return val + coef * Z**exp

    return fori_loop(0, n // 2 + 1, body_fun, jnp.zeros_like(R))


@jit
def N_m_n(R, Z, m, n):
    """N_m_n term in eqn 9 of Dommaschk paper."""
    # the sum comes from fact that N_mn = I_mn and the def of I_mn in eq 2 of the paper

    def body_fun(k, val):
        coef = CN_m_k(R, m, k) / gamma(n - 2 * k + 1)
        exp = n - 2 * k
        # derivative of 0**0 is ill defined, so we do this to enforce it being 0
        exp = jnp.where((Z == 0) & (exp == 0), 1, exp)
        return val + coef * Z**exp

    return fori_loop(0, n // 2 + 1, body_fun, jnp.zeros_like(R))


@jit
def V_m_l(R, phi, Z, m, l, a, b, c, d):
    """Eq 12 of Dommaschk paper.

    Parameters
    ----------
    R,phi,Z : array-like
        Cylindrical coordinates (1-D arrays of each of size num_eval_pts)
            to evaluate the Dommaschk potential term at.
    m : int
        first index of V_m_l term
    l : int
        second index of V_m_l term
    a : float
        a_m_l coefficient of V_m_l term, which multiplies cos(m*phi)*D_m_l
    b : float
        b_m_l coefficient of V_m_l term, which multiplies sin(m*phi)*D_m_l
    c : float
        c_m_l coefficient of V_m_l term, which multiplies cos(m*phi)*N_m_l-1
    d : float
        d_m_l coefficient of V_m_l term, which multiplies sin(m*phi)*N_m_l-1

    Returns
    -------
    value : array-like
        Value of this V_m_l term evaluated at the given R,phi,Z points
        (same size as the size of the given R,phi, or Z arrays).

    """
    return (a * jnp.cos(m * phi) + b * jnp.sin(m * phi)) * D_m_n(R, Z, m, l) + (
        c * jnp.cos(m * phi) + d * jnp.sin(m * phi)
    ) * N_m_n(R, Z, m, l - 1)


@jit
def dommaschk_potential(R, phi, Z, ms, ls, a_arr, b_arr, c_arr, d_arr, B0=1):
    """Eq 1 of Dommaschk paper.

        this is the total dommaschk potential for
        a given set of m,l indices and their corresponding
        coefficients a_ml, b_ml, c_ml d_ml.

    Parameters
    ----------
    R,phi,Z : array-like
        Cylindrical coordinates (1-D arrays of each of size num_eval_pts)
        to evaluate the Dommaschk potential term at.
    ms : 1D array-like of int
        first indices of V_m_l terms
    ls : 1D array-like of int
        second indices of V_m_l terms
    a_arr : 1D array-like of float
        a_m_l coefficients of V_m_l terms, which multiplies cos(m*phi)*D_m_l
    b_arr : 1D array-like of float
        b_m_l coefficients of V_m_l terms, which multiplies sin(m*phi)*D_m_l
    c_arr : 1D array-like of float
        c_m_l coefficients of V_m_l terms, which multiplies cos(m*phi)*N_m_l-1
    d_arr : 1D array-like of float
        d_m_l coefficients of V_m_l terms, which multiplies sin(m*phi)*N_m_l-1
    B0: float, toroidal magnetic field strength scale, this is the strength of the
        1/R part of the magnetic field and is the Bphi at R=1.

    Returns
    -------
    value : array-like
        Value of the total dommaschk potential evaluated
        at the given R,phi,Z points
        (same size as the size of the given R,phi, Z arrays).
    """
    ms, ls, a_arr, b_arr, c_arr, d_arr = map(
        jnp.atleast_1d, (ms, ls, a_arr, b_arr, c_arr, d_arr)
    )
    R, phi, Z = map(jnp.atleast_1d, (R, phi, Z))
    R, phi, Z = jnp.broadcast_arrays(R, phi, Z)
    ms, ls, a_arr, b_arr, c_arr, d_arr = jnp.broadcast_arrays(
        ms, ls, a_arr, b_arr, c_arr, d_arr
    )
    value = B0 * phi  # phi term

    def body(i, val):
        val += V_m_l(R, phi, Z, ms[i], ls[i], a_arr[i], b_arr[i], c_arr[i], d_arr[i])
        return val

    return fori_loop(0, len(ms), body, value)


class CurrentPotentialField(_MagneticField, FourierRZToroidalSurface):
    """Magnetic field due to a surface current potential on a toroidal surface.

        surface current K is assumed given by
         K = n x ∇ Φ
        where:
               n is the winding surface unit normal.
               Phi is the current potential function,
                which is a function of theta and zeta.
        This function then uses biot-savart to find the
        B field from this current density K on the surface.

    Parameters
    ----------
    potential : callable
        function to compute the current potential. Should have a signature of
        the form potential(theta,zeta,**params) -> ndarray.
        theta,zeta are poloidal and toroidal angles on the surface
    potential_dtheta: callable
        function to compute the theta derivative of the current potential
    potential_dzeta: callable
        function to compute the zeta derivative of the current potential
    params : dict, optional
        default parameters to pass to potential function (and its derivatives)
    R_lmn, Z_lmn : array-like, shape(k,)
        Fourier coefficients for winding surface R and Z in cylindrical coordinates
    modes_R : array-like, shape(k,2)
        poloidal and toroidal mode numbers [m,n] for R_lmn.
    modes_Z : array-like, shape(k,2)
        mode numbers associated with Z_lmn, defaults to modes_R
    NFP : int
        number of field periods
    sym : bool
        whether to enforce stellarator symmetry for the surface geometry.
        Default is "auto" which enforces if modes are symmetric. If True,
        non-symmetric modes will be truncated.
    M, N: int or None
        Maximum poloidal and toroidal mode numbers. Defaults to maximum from modes_R
        and modes_Z.
    name : str
        name for this field
    check_orientation : bool
        ensure that this surface has a right handed orientation. Do not set to False
        unless you are sure the parameterization you have given is right handed
        (ie, e_theta x e_zeta points outward from the surface).

    """

    _io_attrs_ = (
        _MagneticField._io_attrs_
        + FourierRZToroidalSurface._io_attrs_
        + [
            "_params",
        ]
    )

    def __init__(
        self,
        potential,
        potential_dtheta,
        potential_dzeta,
        params=None,
        R_lmn=None,
        Z_lmn=None,
        modes_R=None,
        modes_Z=None,
        NFP=1,
        sym="auto",
        M=None,
        N=None,
        name="",
        check_orientation=True,
    ):
        assert callable(potential), "Potential must be callable!"
        assert callable(potential_dtheta), "Potential derivative must be callable!"
        assert callable(potential_dzeta), "Potential derivative must be callable!"

        self._potential = potential
        self._potential_dtheta = potential_dtheta
        self._potential_dzeta = potential_dzeta
        self._params = params

        super().__init__(
            R_lmn=R_lmn,
            Z_lmn=Z_lmn,
            modes_R=modes_R,
            modes_Z=modes_Z,
            NFP=NFP,
            sym=sym,
            M=M,
            N=N,
            name=name,
            check_orientation=check_orientation,
        )

    @property
    def params(self):
        """Dict of parameters to pass to potential function and its derivatives."""
        return self._params

    @params.setter
    def params(self, new):
        warnif(
            len(new) != len(self._params),
            UserWarning,
            "Length of new params is different from length of current params! "
            "May cause errors unless potential function is also changed.",
        )
        self._params = new

    @property
    def potential(self):
        """Potential function, signature (theta,zeta,**params) -> potential value."""
        return self._potential

    @potential.setter
    def potential(self, new):
        if new != self._potential:
            assert callable(new), "Potential must be callable!"
            self._potential = new

    @property
    def potential_dtheta(self):
        """Phi poloidal deriv. function, signature (theta,zeta,**params) -> value."""
        return self._potential_dtheta

    @potential_dtheta.setter
    def potential_dtheta(self, new):
        if new != self._potential_dtheta:
            assert callable(new), "Potential derivative must be callable!"
            self._potential_dtheta = new

    @property
    def potential_dzeta(self):
        """Phi toroidal deriv. function, signature (theta,zeta,**params) -> value."""
        return self._potential_dzeta

    @potential_dzeta.setter
    def potential_dzeta(self, new):
        if new != self._potential_dzeta:
            assert callable(new), "Potential derivative must be callable!"
            self._potential_dzeta = new

    def save(self, file_name, file_format=None, file_mode="w"):
        """Save the object.

        **Not supported for this object!

        Parameters
        ----------
        file_name : str file path OR file instance
            location to save object
        file_format : str (Default hdf5)
            format of save file. Only used if file_name is a file path
        file_mode : str (Default w - overwrite)
            mode for save file. Only used if file_name is a file path

        """
        raise OSError(
            "Saving CurrentPotentialField is not supported,"
            " as the potential function cannot be serialized."
        )

    def compute_magnetic_field(
        self, coords, params=None, basis="rpz", source_grid=None
    ):
        """Compute magnetic field at a set of points.

        Parameters
        ----------
        coords : array-like shape(n,3)
            Nodes to evaluate field at in [R,phi,Z] or [X,Y,Z] coordinates.
        params : dict or array-like of dict, optional
            Dictionary of optimizable parameters, eg field.params_dict.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic field.
        source_grid : Grid, int or None or array-like, optional
            Source grid upon which to evaluate the surface current density K.

        Returns
        -------
        field : ndarray, shape(N,3)
            magnetic field at specified points

        """
        source_grid = source_grid or LinearGrid(
            M=30 + 2 * self.M,
            N=30 + 2 * self.N,
            NFP=self.NFP,
        )
        return _compute_magnetic_field_from_CurrentPotentialField(
            field=self,
            coords=coords,
            params=params,
            basis=basis,
            source_grid=source_grid,
        )

    @classmethod
    def from_surface(
        cls,
        surface,
        potential,
        potential_dtheta,
        potential_dzeta,
        params=None,
    ):
        """Create CurrentPotentialField using geometry provided by given surface.

        Parameters
        ----------
        surface: FourierRZToroidalSurface, optional, default None
            Existing FourierRZToroidalSurface object to create a
            CurrentPotentialField with.
        potential : callable
            function to compute the current potential. Should have a signature of
            the form potential(theta,zeta,**params) -> ndarray.
            theta,zeta are poloidal and toroidal angles on the surface
        potential_dtheta: callable
            function to compute the theta derivative of the current potential
        potential_dzeta: callable
            function to compute the zeta derivative of the current potential
        params : dict, optional
            default parameters to pass to potential function (and its derivatives)

        """
        errorif(
            not isinstance(surface, FourierRZToroidalSurface),
            TypeError,
            "Expected type FourierRZToroidalSurface for argument surface, "
            f"instead got type {type(surface)}",
        )

        R_lmn = surface.R_lmn
        Z_lmn = surface.Z_lmn
        modes_R = surface._R_basis.modes[:, 1:]
        modes_Z = surface._Z_basis.modes[:, 1:]
        NFP = surface.NFP
        sym = surface.sym
        name = surface.name

        return cls(
            potential,
            potential_dtheta,
            potential_dzeta,
            params,
            R_lmn,
            Z_lmn,
            modes_R,
            modes_Z,
            NFP,
            sym,
            name=name,
            check_orientation=False,
        )


class FourierCurrentPotentialField(
    _MagneticField, FourierRZToroidalSurface, Optimizable
):
    """Magnetic field due to a surface current potential on a toroidal surface.

        surface current K is assumed given by

        K = n x ∇ Φ
        Φ(θ,ζ) = Φₛᵥ(θ,ζ) + Gζ/2π + Iθ/2π

        where:
              n is the winding surface unit normal.
              Phi is the current potential function,
                which is a function of theta and zeta,
                and is given as a secular linear term in theta/zeta
                and a double Fourier series in theta/zeta.
        This function then uses biot-savart to find the
        B field from this current density K on the surface.

    Parameters
    ----------
    Phi_mn : ndarray
        Fourier coefficients of the double FourierSeries part of the current potential.
    modes_Phi : array-like, shape(k,2)
        Poloidal and Toroidal mode numbers corresponding to passed-in Phi_mn
        coefficients.
    I : float
        Net current linking the plasma and the surface toroidally
        Denoted I in the algorithm
    G : float
        Net current linking the plasma and the surface poloidally
        Denoted G in the algorithm
        NOTE: a negative G will tend to produce a positive toroidal magnetic field
        B in DESC, as in DESC the poloidal angle is taken to be positive
        and increasing when going in the clockwise direction, which with the
        convention n x grad(phi) will result in a toroidal field in the negative
        toroidal direction.
    sym_Phi :  {False,"cos","sin"}
        whether to enforce a given symmetry for the DoubleFourierSeries part of the
        current potential.
    M_Phi, N_Phi: int or None
        Maximum poloidal and toroidal mode numbers for the single valued part of the
        current potential.
    R_lmn, Z_lmn : array-like, shape(k,)
        Fourier coefficients for winding surface R and Z in cylindrical coordinates
    modes_R : array-like, shape(k,2)
        poloidal and toroidal mode numbers [m,n] for R_lmn.
    modes_Z : array-like, shape(k,2)
        mode numbers associated with Z_lmn, defaults to modes_R
    NFP : int
        number of field periods
    sym : bool
        whether to enforce stellarator symmetry for the surface geometry.
        Default is "auto" which enforces if modes are symmetric. If True,
        non-symmetric modes will be truncated.
    M, N: int or None
        Maximum poloidal and toroidal mode numbers. Defaults to maximum from modes_R
        and modes_Z.
    name : str
        name for this field
    check_orientation : bool
        ensure that this surface has a right handed orientation. Do not set to False
        unless you are sure the parameterization you have given is right handed
        (ie, e_theta x e_zeta points outward from the surface).

    """

    _io_attrs_ = (
        _MagneticField._io_attrs_
        + FourierRZToroidalSurface._io_attrs_
        + ["_Phi_mn", "_I", "_G", "_Phi_basis", "_M_Phi", "_N_Phi", "_sym_Phi"]
    )

    def __init__(
        self,
        Phi_mn=np.array([0.0]),
        modes_Phi=np.array([[0, 0]]),
        I=0,
        G=0,
        sym_Phi=False,
        M_Phi=None,
        N_Phi=None,
        R_lmn=None,
        Z_lmn=None,
        modes_R=None,
        modes_Z=None,
        NFP=1,
        sym="auto",
        M=None,
        N=None,
        name="",
        check_orientation=True,
    ):
        Phi_mn, modes_Phi = map(np.asarray, (Phi_mn, modes_Phi))
        assert (
            Phi_mn.size == modes_Phi.shape[0]
        ), "Phi_mn size and modes_Phi.shape[0] must be the same size!"

        assert np.issubdtype(modes_Phi.dtype, np.integer)

        M_Phi = setdefault(M_Phi, np.max(abs(modes_Phi[:, 0])))
        N_Phi = setdefault(N_Phi, np.max(abs(modes_Phi[:, 1])))

        self._M_Phi = M_Phi
        self._N_Phi = N_Phi

        self._sym_Phi = sym_Phi
        self._Phi_basis = DoubleFourierSeries(M=M_Phi, N=N_Phi, NFP=NFP, sym=sym_Phi)
        self._Phi_mn = copy_coeffs(Phi_mn, modes_Phi, self._Phi_basis.modes[:, 1:])

        assert np.isscalar(I) or np.asarray(I).size == 1, "I must be a scalar"
        assert np.isscalar(G) or np.asarray(G).size == 1, "G must be a scalar"
        self._I = float(I)
        self._G = float(G)

        super().__init__(
            R_lmn=R_lmn,
            Z_lmn=Z_lmn,
            modes_R=modes_R,
            modes_Z=modes_Z,
            NFP=NFP,
            sym=sym,
            M=M,
            N=N,
            name=name,
            check_orientation=check_orientation,
        )

    @optimizable_parameter
    @property
    def I(self):  # noqa: E743
        """Net current linking the plasma and the surface toroidally."""
        return self._I

    @I.setter
    def I(self, new):  # noqa: E743
        assert np.isscalar(new) or np.asarray(new).size == 1, "I must be a scalar"
        self._I = float(new)

    @optimizable_parameter
    @property
    def G(self):
        """Net current linking the plasma and the surface poloidally."""
        return self._G

    @G.setter
    def G(self, new):
        assert np.isscalar(new) or np.asarray(new).size == 1, "G must be a scalar"
        self._G = float(new)

    @optimizable_parameter
    @property
    def Phi_mn(self):
        """Fourier coefficients describing single-valued part of potential."""
        return self._Phi_mn

    @Phi_mn.setter
    def Phi_mn(self, new):
        if len(new) == self.Phi_basis.num_modes:
            self._Phi_mn = jnp.asarray(new)
        else:
            raise ValueError(
                f"Phi_mn should have the same size as the basis, got {len(new)} for "
                + f"basis with {self.Phi_basis.num_modes} modes."
            )

    @property
    def Phi_basis(self):
        """DoubleFourierSeries: Spectral basis for Phi."""
        return self._Phi_basis

    @property
    def sym_Phi(self):
        """str: Type of symmetry of periodic part of Phi (no symmetry if False)."""
        return self._sym_Phi

    @property
    def M_Phi(self):
        """int: Poloidal resolution of periodic part of Phi."""
        return self._M_Phi

    @property
    def N_Phi(self):
        """int: Toroidal resolution of periodic part of Phi."""
        return self._N_Phi

    def change_Phi_resolution(self, M=None, N=None, NFP=None, sym_Phi=None):
        """Change the maximum poloidal and toroidal resolution for Phi.

        Parameters
        ----------
        M : int
            Poloidal resolution to change Phi basis to.
            If None, defaults to current self.Phi_basis poloidal resolution
        N : int
            Toroidal resolution to change Phi basis to.
            If None, defaults to current self.Phi_basis toroidal resolution
        NFP : int
            Number of field periods for surface and Phi basis.
            If None, defaults to current NFP.
            Note: will change the NFP of the surface geometry as well as the
            Phi basis.
        sym_Phi :  {"auto","cos","sin",False}
            whether to enforce a given symmetry for the DoubleFourierSeries part of the
            current potential. Default is "auto" which enforces if modes are symmetric.
            If True, non-symmetric modes will be truncated.

        """
        M = M or self._M_Phi
        N = N or self._M_Phi
        NFP = NFP or self.NFP
        sym_Phi = sym_Phi or self.sym_Phi

        Phi_modes_old = self.Phi_basis.modes
        self.Phi_basis.change_resolution(M=M, N=N, NFP=self.NFP, sym=sym_Phi)

        self._Phi_mn = copy_coeffs(self.Phi_mn, Phi_modes_old, self.Phi_basis.modes)
        self._M_Phi = M
        self._N_Phi = N
        self._sym_Phi = sym_Phi
        self.change_resolution(
            NFP=NFP
        )  # make sure surface and Phi basis NFP are the same

    def compute_magnetic_field(
        self, coords, params=None, basis="rpz", source_grid=None
    ):
        """Compute magnetic field at a set of points.

        Parameters
        ----------
        coords : array-like shape(n,3)
            Nodes to evaluate field at in [R,phi,Z] or [X,Y,Z] coordinates.
        params : dict or array-like of dict, optional
            Dictionary of optimizable parameters, eg field.params_dict.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic field.
        source_grid : Grid, int or None or array-like, optional
            Source grid upon which to evaluate the surface current density K.

        Returns
        -------
        field : ndarray, shape(N,3)
            magnetic field at specified points

        """
        source_grid = source_grid or LinearGrid(
            M=30 + 2 * max(self.M, self.M_Phi),
            N=30 + 2 * max(self.N, self.N_Phi),
            NFP=self.NFP,
        )
        return _compute_magnetic_field_from_CurrentPotentialField(
            field=self,
            coords=coords,
            params=params,
            basis=basis,
            source_grid=source_grid,
        )

    @classmethod
    def from_surface(
        cls,
        surface,
        Phi_mn=np.array([0.0]),
        modes_Phi=np.array([[0, 0]]),
        I=0,
        G=0,
        sym_Phi=False,
        M_Phi=None,
        N_Phi=None,
    ):
        """Create FourierCurrentPotentialField using geometry of given surface.

        Parameters
        ----------
        surface: FourierRZToroidalSurface, optional, default None
            Existing FourierRZToroidalSurface object to create a
            CurrentPotentialField with.
        Phi_mn : ndarray
            Fourier coefficients of the double FourierSeries of the current potential.
            Should correspond to the given DoubleFourierSeries basis object passed in.
        modes_Phi : array-like, shape(k,2)
            Poloidal and Toroidal mode numbers corresponding to passed-in Phi_mn
            coefficients
        I : float
            Net current linking the plasma and the surface toroidally
            Denoted I in the algorithm
        G : float
            Net current linking the plasma and the surface poloidally
            Denoted G in the algorithm
            NOTE: a negative G will tend to produce a positive toroidal magnetic field
            B in DESC, as in DESC the poloidal angle is taken to be positive
            and increasing when going in the clockwise direction, which with the
            convention n x grad(phi) will result in a toroidal field in the negative
            toroidal direction.
        sym_Phi :  {False,"cos","sin"}
            whether to enforce a given symmetry for the DoubleFourierSeries part of the
            current potential.
        M_Phi, N_Phi: int or None
            Maximum poloidal and toroidal mode numbers for the single valued part of the
            current potential.

        """
        if not isinstance(surface, FourierRZToroidalSurface):
            raise TypeError(
                "Expected type FourierRZToroidalSurface for argument surface, "
                f"instead got type {type(surface)}"
            )
        R_lmn = surface.R_lmn
        Z_lmn = surface.Z_lmn
        modes_R = surface._R_basis.modes[:, 1:]
        modes_Z = surface._Z_basis.modes[:, 1:]
        NFP = surface.NFP
        sym = surface.sym
        name = surface.name

        return cls(
            Phi_mn=Phi_mn,
            modes_Phi=modes_Phi,
            I=I,
            G=G,
            sym_Phi=sym_Phi,
            M_Phi=M_Phi,
            N_Phi=N_Phi,
            R_lmn=R_lmn,
            Z_lmn=Z_lmn,
            modes_R=modes_R,
            modes_Z=modes_Z,
            NFP=NFP,
            sym=sym,
            name=name,
            check_orientation=False,
        )


def _compute_magnetic_field_from_CurrentPotentialField(
    field,
    coords,
    source_grid,
    params=None,
    basis="rpz",
):
    """Compute magnetic field at a set of points.

    Parameters
    ----------
    field : CurrentPotentialField or FourierCurrentPotentialField
        current potential field object from which to compute magnetic field.
    coords : array-like shape(N,3)
        cylindrical or cartesian coordinates
    source_grid : Grid,
        source grid upon which to evaluate the surface current density K
    params : dict, optional
        parameters to pass to compute function
        should include the potential
    basis : {"rpz", "xyz"}
        basis for input coordinates and returned magnetic field


    Returns
    -------
    field : ndarray, shape(N,3)
        magnetic field at specified points

    """
    assert basis.lower() in ["rpz", "xyz"]
    coords = jnp.atleast_2d(coords)
    if basis == "rpz":
        coords = rpz2xyz(coords)

    # compute surface current, and store grid quantities
    # needed for integration in class
    # TODO: does this have to be xyz, or can it be computed in rpz as well?
    data = field.compute(["K", "x"], grid=source_grid, basis="xyz", params=params)

    _rs = xyz2rpz(data["x"])
    _K = xyz2rpz_vec(data["K"], phi=source_grid.nodes[:, 2])

    # surface element, must divide by NFP to remove the NFP multiple on the
    # surface grid weights, as we account for that when doing the for loop
    # over NFP
    _dV = source_grid.weights * data["|e_theta x e_zeta|"] / source_grid.NFP

    def nfp_loop(j, f):
        # calculate (by rotating) rs, rs_t, rz_t
        phi = (source_grid.nodes[:, 2] + j * 2 * jnp.pi / source_grid.NFP) % (
            2 * jnp.pi
        )
        # new coords are just old R,Z at a new phi (bc of discrete NFP symmetry)
        rs = jnp.vstack((_rs[:, 0], phi, _rs[:, 2])).T
        rs = rpz2xyz(rs)
        K = rpz2xyz_vec(_K, phi=phi)
        fj = biot_savart_general(
            coords,
            rs,
            K,
            _dV,
        )
        f += fj
        return f

    B = fori_loop(0, source_grid.NFP, nfp_loop, jnp.zeros_like(coords))
    if basis == "rpz":
        B = xyz2rpz_vec(B, x=coords[:, 0], y=coords[:, 1])
    return B
