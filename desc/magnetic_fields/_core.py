"""Classes for magnetic fields."""

from abc import ABC, abstractmethod
from collections.abc import MutableSequence

import numpy as np
from interpax import approx_df, interp2d, interp3d
from netCDF4 import Dataset, chartostring, stringtochar

from desc.backend import fori_loop, jit, jnp, odeint
from desc.basis import DoubleFourierSeries
from desc.compute import rpz2xyz_vec, xyz2rpz
from desc.derivatives import Derivative
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.grid import LinearGrid
from desc.io import IOAble
from desc.optimizable import Optimizable, OptimizableCollection, optimizable_parameter
from desc.transform import Transform
from desc.utils import copy_coeffs, flatten_list, setdefault
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
        scale = float(np.squeeze(scale))
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
        self._scale = float(np.squeeze(new))

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
        self.B0 = float(np.squeeze(B0))
        self.R0 = float(np.squeeze(R0))

    @optimizable_parameter
    @property
    def R0(self):
        """float: major radius of axis."""
        return self._R0

    @R0.setter
    def R0(self, new):
        self._R0 = float(np.squeeze(new))

    @optimizable_parameter
    @property
    def B0(self):
        """float: field strength on axis."""
        return self._B0

    @B0.setter
    def B0(self, new):
        self._B0 = float(np.squeeze(new))

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
        coords = jnp.atleast_2d(jnp.asarray(coords))
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
        self._B0 = float(np.squeeze(new))

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
        coords = jnp.atleast_2d(jnp.asarray(coords))
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
        self._R0 = float(np.squeeze(new))

    @optimizable_parameter
    @property
    def B0(self):
        """float: field strength on axis."""
        return self._B0

    @B0.setter
    def B0(self, new):
        self._B0 = float(np.squeeze(new))

    @optimizable_parameter
    @property
    def iota(self):
        """float: desired rotational transform."""
        return self._iota

    @iota.setter
    def iota(self, new):
        self._iota = float(np.squeeze(new))

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
        coords = jnp.atleast_2d(jnp.asarray(coords))
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
        R, phi, Z, currents = map(
            lambda x: jnp.atleast_1d(jnp.asarray(x)), (R, phi, Z, currents)
        )
        assert R.ndim == 1
        assert phi.ndim == 1
        assert Z.ndim == 1
        assert currents.ndim == 1
        shape = (R.size, phi.size, Z.size, currents.size)

        def _atleast_4d(x):
            x = jnp.atleast_3d(jnp.asarray(x))
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
        new = jnp.atleast_1d(jnp.asarray(new))
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
        coords = jnp.atleast_2d(jnp.asarray(coords))
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
        coords = jnp.atleast_2d(jnp.asarray(coords))
        coords = coords.astype(float)
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
    bounds_R=(0, np.inf),
    bounds_Z=(-np.inf, np.inf),
    decay_accel=1e6,
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
    bounds_R : tuple of (float,float), optional
        R bounds for field line integration bounding box.
        If supplied, the RHS of the field line equations will be
        multipled by exp(-r) where r is the distance to the bounding box,
        this is meant to prevent the field lines which escape to infinity from
        slowing the integration down by being traced to infinity.
        defaults to (0,np.inf)
    bounds_Z : tuple of (float,float), optional
        Z bounds for field line integration bounding box.
        If supplied, the RHS of the field line equations will be
        multipled by exp(-r) where r is the distance to the bounding box,
        this is meant to prevent the field lines which escape to infinity from
        slowing the integration down by being traced to infinity.
        Defaults to (-np.inf,np.inf)
    decay_accel : float, optional
        An extra factor to the exponential that decays the RHS, i.e.
        the RHS is multiplied by exp(-r * decay_accel), this is to
        accelerate the decay of the RHS and stop the integration sooner
        after exiting the bounds. Defaults to 1e6


    Returns
    -------
    r, z : ndarray
        arrays of r, z coordinates at specified phi angles

    """
    r0, z0, phis = map(jnp.asarray, (r0, z0, phis))
    assert r0.shape == z0.shape, "r0 and z0 must have the same shape"
    assert decay_accel > 0, "decay_accel must be positive"
    rshape = r0.shape
    r0 = r0.flatten()
    z0 = z0.flatten()
    x0 = jnp.array([r0, phis[0] * jnp.ones_like(r0), z0]).T

    @jit
    def odefun(rpz, s):
        rpz = rpz.reshape((3, -1)).T
        r = rpz[:, 0]
        z = rpz[:, 2]
        # if bounds are given, will decay the magnetic field line eqn
        # RHS if the trajectory is outside of bounds to avoid
        # integrating the field line to infinity, which is costly
        # and not useful in most cases
        decay_factor = jnp.where(
            jnp.array(
                [
                    jnp.less(r, bounds_R[0]),
                    jnp.greater(r, bounds_R[1]),
                    jnp.less(z, bounds_Z[0]),
                    jnp.greater(z, bounds_Z[1]),
                ]
            ),
            jnp.array(
                [
                    # we mult by decay_accel to accelerate the decay so that the
                    # integration is stopped soon after the bounds are exited.
                    jnp.exp(-(decay_accel * (r - bounds_R[0]) ** 2)),
                    jnp.exp(-(decay_accel * (r - bounds_R[1]) ** 2)),
                    jnp.exp(-(decay_accel * (z - bounds_Z[0]) ** 2)),
                    jnp.exp(-(decay_accel * (z - bounds_Z[1]) ** 2)),
                ]
            ),
            1.0,
        )
        # multiply all together, the conditions that are not violated
        # are just one while the violated ones are continous decaying exponentials
        decay_factor = jnp.prod(decay_factor, axis=0)

        br, bp, bz = field.compute_magnetic_field(
            rpz, params, basis="rpz", source_grid=source_grid
        ).T
        return (
            decay_factor
            * jnp.array(
                [r * br / bp * jnp.sign(bp), jnp.sign(bp), r * bz / bp * jnp.sign(bp)]
            ).squeeze()
        )

    intfun = lambda x: odeint(odefun, x, phis, rtol=rtol, atol=atol, mxstep=maxstep)
    x = jnp.vectorize(intfun, signature="(k)->(n,k)")(x0)
    r = x[:, :, 0].T.reshape((len(phis), *rshape))
    z = x[:, :, 2].T.reshape((len(phis), *rshape))
    return r, z
