"""Classes for dipoles."""

from abc import ABC
from collections.abc import MutableSequence
from functools import partial

import numpy as np
from desc.integrals import compute_B_plasma

from desc.backend import (
    fori_loop,
    jit,
    jnp,
    scan,
    tree_flatten,
    tree_leaves,
    tree_stack,
    tree_unflatten,
    tree_unstack,
    vmap,
)
from desc.compute import get_params
from desc.compute.utils import _compute as compute_fun
from desc.grid import Grid, LinearGrid
from desc.magnetic_fields import _MagneticField
from desc.magnetic_fields._core import (
    dipole_field,
    dipole_vector_potential
)
from desc.optimizable import Optimizable, OptimizableCollection, optimizable_parameter
from desc.utils import (
    cross,
    dot,
    equals,
    errorif,
    flatten_list,
    reflection_matrix,
    rpz2xyz,
    rpz2xyz_vec,
    safenorm,
    warnif,
    xyz2rpz,
    xyz2rpz_vec,
)
from desc.utils import errorif, reflection_matrix, rotation_matrix

@partial(jit, static_argnames=["chunk_size"])
def magnetic_dipole_field(
    eval_pts, mag_points, phi, theta, m0, *, chunk_size=None
):
    """External magnetic field produced by a magnetic dipole, following [1].

    The magnetic dipole is approximated by a single, dimensionless point.

    References
    ----------
    [1] Chow, "Introduction to electromagnetic theory: a modern perspective" (2006)

    Parameters
    ----------
    eval_pts : array-like shape(n,3)
        Evaluation points in cartesian coordinates
    mag_points : array-like shape(m,3)
        Points in cartesian space defining the location of each point dipole.
    phi : float
        Azimuthal orientation of the dipole (in radians).
    theta : float
        Polar orientation of the dipole (in radians).   
    m0 : float
        Effective dipole moment strength, with radial direction.
    chunk_size : int or None
        Unused by this function, only kept for API compatibility.
        Size to split computation into chunks of evaluation points.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``None``.

    Returns
    -------
    B : ndarray, shape(n,3)
        magnetic field in cartesian components at specified points

    """
    m_hat = jnp.array(
        [
            jnp.sin(theta) * jnp.cos(phi),
            jnp.sin(theta) * jnp.sin(phi),
            jnp.cos(theta),
        ]
    )
    m_vector = m0 * m_hat
    return dipole_field(
        eval_pts, mag_points, m_vector, chunk_size=chunk_size
    )


@partial(jit, static_argnames=["chunk_size"])
def magnetic_dipole_vector_field(
    eval_pts, mag_points, phi, theta, m0, *, chunk_size=None
):
    """Vector potential of a magnetic dipole, following [1].

    The magnetic dipole is approximated by a single, dimensionless point.

    References
    ----------
    [1] Hanson & Hirshman, "Compact expressions for the Biot-Savart
        fields of a filamentary segment" (2002)

    Parameters
    ----------
    eval_pts : array-like shape(n,3)
        Evaluation points in cartesian coordinates
    mag_points : array-like shape(m,3)
        Points in cartesian space defining the location of each point dipole.
    phi : float
        Azimuthal orientation of the dipole (in radians).
    theta : float
        Polar orientation of the dipole (in radians).   
    m0 : float
        Effective dipole moment strength, with radial direction.
    chunk_size : int or None
        Unused by this function, only kept for API compatibility.
        Size to split computation into chunks of evaluation points.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``None``.

    Returns
    -------
    A : ndarray, shape(n,3)
        Magnetic vector potential in cartesian components at specified points

    """
    m_hat = jnp.array(
        [
            jnp.sin(theta) * jnp.cos(phi),
            jnp.sin(theta) * jnp.sin(phi),
            jnp.cos(theta),
        ]
    )
    m_vector = m0 * m_hat
    return dipole_vector_potential(
        eval_pts, mag_points, m_vector, chunk_size=chunk_size
    )
    

class _Dipole(_MagneticField, Optimizable, ABC):
    """Base class representing an ideal magnetic dipole.

    Represents dipoles as single point with orientation

    Parameters
    ----------
    phi : float
        Azimuthal orientation of the dipole (in radians).
    theta : float
        Polar orientation of the dipole (in radians).   
    m0 : float
        Effective dipole moment strength, with radial direction.
    rho : float
        Dimensionless optimization parameter in range (-1, 1) that defines radial
        direction and magntiude of the dipole; positive is radially outward, negative is 
        radially inward.
    """
    
    _io_attrs_ = _MagneticField._io_attrs_ + ["_x"] + ["_y"] + ["_z"] + ["_phi"] + ["_theta"] + ["_m0"] + ["_rho"] + ["_name", "_shift", "_rotmat"] + ["_name"]
    _static_attrs = _MagneticField._static_attrs + Optimizable._static_attrs + ["_name"]

    def __init__(self, x=0.0, y=0.0, z=0.0, phi=0.0, theta=0.0, m0=1.0, rho=1.0, name=""):
        self._x = jnp.float64(float(np.squeeze(x)))
        self._y = jnp.float64(float(np.squeeze(y)))
        self._z = jnp.float64(float(np.squeeze(z)))
        self._phi = jnp.float64(float(np.squeeze(phi)))
        self._theta = jnp.float64(float(np.squeeze(theta)))
        self._m0 = jnp.float64(float(np.squeeze(m0)))
        self._rho = jnp.float64(float(np.squeeze(rho)))
        self._name = str(name)
        super().__init__()

    def _set_up(self):
        for attribute in self._io_attrs_:
            if not hasattr(self, attribute):
                setattr(self, attribute, None)

    @optimizable_parameter
    @property
    def shift(self):
        """Displacement of curve in X, Y, Z."""
        return self.__dict__.setdefault("_shift", jnp.array([0, 0, 0], dtype=float))

    @shift.setter
    def shift(self, new):
        if len(new) == 3:
            self._shift = jnp.asarray(new)
        else:
            raise ValueError("shift should be a 3 element vector, got {}".format(new))

    @optimizable_parameter
    @property
    def rotmat(self):
        """Rotation matrix of curve in X, Y, Z."""
        return self.__dict__.setdefault("_rotmat", jnp.eye(3, dtype=float).flatten())

    @rotmat.setter
    def rotmat(self, new):
        if len(new) == 9:
            self._rotmat = jnp.asarray(new)
        else:
            self._rotmat = jnp.asarray(new.flatten())


    @optimizable_parameter
    @property
    def x(self):
        """float: X-coordinate of dipole position."""
        return self._x

    @x.setter
    def x(self, new):
        assert jnp.isscalar(new) or new.size == 1
        self._x = jnp.float64(float(np.squeeze(new)))

    @optimizable_parameter
    @property
    def y(self):
        """float: Y-coordinate of dipole position."""
        return self._y

    @y.setter
    def y(self, new):
        assert jnp.isscalar(new) or new.size == 1
        self._y = jnp.float64(float(np.squeeze(new)))

    @optimizable_parameter
    @property
    def z(self):
        """float: Z-coordinate of dipole position."""
        return self._z

    @z.setter
    def z(self, new):
        assert jnp.isscalar(new) or new.size == 1
        self._z = jnp.float64(float(np.squeeze(new)))

    @optimizable_parameter
    @property
    def phi(self):
        """float: Azimuthal angle of dipole orientation."""
        return self._phi

    @phi.setter
    def phi(self, new):
        assert jnp.isscalar(new) or new.size == 1
        self._phi = jnp.float64(float(np.squeeze(new)))

    @optimizable_parameter
    @property
    def theta(self):
        """float: Polar angle of dipole orientation."""
        return self._theta

    @theta.setter
    def theta(self, new):
        assert jnp.isscalar(new) or new.size == 1
        self._theta = jnp.float64(float(np.squeeze(new)))

    @optimizable_parameter
    @property
    def m0(self):
        """float: Magnitude of magnetic dipole moment in (Amp * meters ^2)."""
        return self._m0

    @m0.setter
    def m0(self, new):
        assert jnp.isscalar(new) or new.size == 1
        self._m0 = jnp.float64(float(np.squeeze(new)))

    @optimizable_parameter
    @property
    def rho(self):
        """float: Dimensionless optimization parameter in range (-1, 1) that defines radial
        direction and magntiude of the dipole; positive is radially outward, negative is 
        radially inward."""
        return self._rho

    @rho.setter
    def rho(self, new):
        assert jnp.isscalar(new) or new.size == 1
        errorif(
            new < -1 or new > 1,
            ValueError,
            f"rho must be in range (-1, 1), got {new}"
        )
        self._rho = jnp.float64(float(np.squeeze(new)))

    @property
    def name(self):
        """str: Name of the dipole."""
        return self._name

    @name.setter
    def name(self, new):
        self._name = str(new)

    @property
    def position(self):
        """ndarray: Position of dipole as [x, y, z]."""
        return jnp.array([self._x, self._y, self._z])
    
    @property
    def M0(self):
        """float: Effective dipole moment strength, with radial direction."""
        return self._m0 * self._rho
    
    @property
    def m_xyz(self):
        """float: Effective dipole moment strength, with radial direction expressed
        in an array of its x, y, and z components."""
        M0 = self.m0 * self.rho
        theta = self._theta
        phi = self._phi
        m_hat = jnp.array(
            [
                jnp.sin(theta) * jnp.cos(phi),
                jnp.sin(theta) * jnp.sin(phi),
                jnp.cos(theta),
            ]
        )
        return M0 * m_hat
    
    def _compute_A_or_B(
        self,
        coords,
        params=None,
        basis="rpz",
        source_grid=None,
        transforms=None,
        compute_A_or_B="B",
        chunk_size=None,
    ):
        """Compute magnetic field or vector potential at a set of points.

        Parameters
        ----------
        coords : array-like shape(n,3)
            Nodes to evaluate field at in [R,phi,Z] or [X,Y,Z] coordinates.
        params : dict, optional
            Parameters to pass to Dipole.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic field.
        source_grid : Grid, int or None, optional
            Grid used to discretize coil. If an integer, uses that many equally spaced
            points. Should NOT include endpoint at 2pi.
        transforms : dict of Transform or array-like
            Transforms for R, Z, lambda, etc. Default is to build from grid.
        compute_A_or_B: {"A", "B"}, optional
            whether to compute the magnetic vector potential "A" or the magnetic field
            "B". Defaults to "B"
        chunk_size : int or None
            Size to split computation into chunks of evaluation points.
            If no chunking should be done or the chunk size is the full input
            then supply ``None``. Default is ``None``.

        Returns
        -------
        field : ndarray, shape(n,3)
            magnetic field at specified points, in either rpz or xyz coordinates

        """
        errorif(
            compute_A_or_B not in ["A", "B"],
            ValueError,
            f'Expected "A" or "B" for compute_A_or_B, instead got {compute_A_or_B}',
        )
        op = {"B": magnetic_dipole_field, "A": magnetic_dipole_vector_field}[
            compute_A_or_B
        ]
        assert basis.lower() in ["rpz", "xyz"]
        coords = jnp.atleast_2d(jnp.asarray(coords))
        if basis.lower() == "rpz":
            phi_coords = coords[:, 1]
            coords = rpz2xyz(coords)

        if params is None:
            params = {
                #get_params(["x", "y", "z", "phi", "theta", "m0", "rho"], dipole, basis=basis) for dipole in self
                "x": self.x,
                "y": self.y,  
                "z": self.z,
                "phi": self.phi,
                "theta": self.theta,
                "m0": self.m0,
                "rho": self.rho,
                "M0": self.M0,
            }

        NFP = getattr(self, "NFP", 1)
        if source_grid is None:
            # NFP=1 to ensure points span the entire length of the coil
            # multiply resolution by NFP to ensure Biot-Savart integration is accurate
            #source_grid = LinearGrid(N=2 * self.N * NFP + 5)
            source_grid = LinearGrid(N=2 * 20 * NFP + 5)
        else:
            # coil grids should have NFP=1. The only possible exception is FourierRZCoil
            # which in theory can be different as long as it matches the coils NFP.
            errorif(
                getattr(source_grid, "NFP", 1) not in [1, NFP],
                ValueError,
                f"source_grid for coils must have NFP=1 or NFP={NFP}",
            )

        x = params.get("x", self.x)
        y = params.get("y", self.y)
        z = params.get("z", self.z)
        phi = params.get("phi", self.phi)
        theta = params.get("theta", self.theta)
        if "M0" in params:
            m0 = params["M0"]
        else:
            m0 = params.get("m0", self.m0) * params.get("rho", self.rho)

        dipole_pos = jnp.array([[x, y, z]])

        AB = op(
            coords,
            dipole_pos,
            phi,
            theta, 
            m0,
            chunk_size=chunk_size, 
        )

        if basis.lower() == "rpz":
            AB = xyz2rpz_vec(AB, phi=phi_coords)
        return AB

    def compute_magnetic_field(
        self,
        coords,
        params=None,
        basis="rpz",
        source_grid=None,
        transforms=None,
        chunk_size=None,
    ):
        """Compute magnetic field at a set of points.

        Parameters
        ----------
        coords : array-like shape(n,3)
            Nodes to evaluate field at in [R,phi,Z] or [X,Y,Z] coordinates.
        params : dict, optional
            Parameters to pass to Dipole.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic field.
        source_grid : Grid, int or None, optional
            Grid used to discretize coil. If an integer, uses that many equally spaced
            points. Should NOT include endpoint at 2pi.
        transforms : dict of Transform or array-like
            Transforms for R, Z, lambda, etc. Default is to build from grid.
        chunk_size : int or None
            Size to split computation into chunks of evaluation points.
            If no chunking should be done or the chunk size is the full input
            then supply ``None``. Default is ``None``.


        Returns
        -------
        field : ndarray, shape(n,3)
            magnetic field at specified points, in either rpz or xyz coordinates

        """
        return self._compute_A_or_B(
            coords, params, basis, source_grid, transforms, "B", chunk_size=chunk_size
        )

    def compute_magnetic_vector_potential(
        self,
        coords,
        params=None,
        basis="rpz",
        source_grid=None,
        transforms=None,
        chunk_size=None,
    ):
        """Compute magnetic vector potential at a set of points.

        Parameters
        ----------
        coords : array-like shape(n,3)
            Nodes to evaluate field at in [R,phi,Z] or [X,Y,Z] coordinates.
        params : dict, optional
            Parameters to pass to Curve.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic field.
        source_grid : Grid, int or None, optional
            Grid used to discretize coil. If an integer, uses that many equally spaced
            points. Should NOT include endpoint at 2pi.
        transforms : dict of Transform or array-like
            Transforms for R, Z, lambda, etc. Default is to build from grid.
        chunk_size : int or None
            Size to split computation into chunks of evaluation points.
            If no chunking should be done or the chunk size is the full input
            then supply ``None``. Default is ``None``.

        Returns
        -------
        vector_potential : ndarray, shape(n,3)
            Magnetic vector potential at specified points, in either rpz or
             xyz coordinates.

        """
        return self._compute_A_or_B(
            coords, params, basis, source_grid, transforms, "A", chunk_size=chunk_size
        )
    
    def translate(self, displacement=[0, 0, 0]):
        """Translate the Dipole by a rigid displacement in X,Y,Z coordinates."""
        self._x += jnp.asarray(displacement)[0]
        self._y += jnp.asarray(displacement)[1]
        self._z += jnp.asarray(displacement)[2]
        self.shift = self.shift + jnp.asarray(displacement)

    def rotate(self, axis=[0, 0, 1], angle=0):
        """Rotate the Dipole by a fixed angle about axis in X,Y,Z coordinates."""
        R = rotation_matrix(axis=axis, angle=angle)
        
        pos = jnp.array([self._x, self._y, self._z])
        new_pos = R @ pos
        self._x = new_pos[0]
        self._y = new_pos[1]
        self._z = new_pos[2]
        
        m_vec = self.m_xyz
        new_m = R @ m_vec
        
        m_magnitude = jnp.linalg.norm(new_m)
        self._theta = jnp.arccos(new_m[2] / m_magnitude)
        self._phi = jnp.arctan2(new_m[1], new_m[0])
        
        self.rotmat = (R @ self.rotmat.reshape(3, 3)).flatten()
        self.shift = self.shift @ R.T

    def flip(self, normal=[0, 0, 1]):
        """Flip the Dipole about the plane with specified normal in X,Y,Z coordinates."""
        F = reflection_matrix(normal)
        
        pos = jnp.array([self._x, self._y, self._z])
        new_pos = F @ pos
        self._x = new_pos[0]
        self._y = new_pos[1]
        self._z = new_pos[2]
        
        m_vec = self.m_xyz
        new_m = F @ m_vec
        
        m_magnitude = jnp.linalg.norm(new_m)
        self._theta = jnp.arccos(new_m[2] / m_magnitude)
        self._phi = jnp.arctan2(new_m[1], new_m[0])
        
        self.rotmat = (F @ self.rotmat.reshape(3, 3)).flatten()
        self.shift = self.shift @ F.T

    def __repr__(self):
        """Get the string form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (name={}, x={}, y={}, z={}, phi={}, theta={}, m0={}, rho={})".format(self.name, self.x, self.y, self.z, self.phi, self.theta, self.m0, self.rho)
        )
    
class DipoleSet(OptimizableCollection, _Dipole, MutableSequence):
    """Set of dipoles with shared parameterization.

    Parameters
    ----------
    dipoles : Dipole or array-like of Dipoles
        Collection of dipoles.
    NFP : int (optional)
        Number of field periods for enforcing field period symmetry.
        If NFP > 1, only include the unique coils in the first field period,
        and the magnetic field will be computed assuming 'virtual' dipoles from the other
        field periods. Default = 1.
    sym : bool (optional)
        Whether to enforce stellarator symmetry. If sym = True, only include the
        unique dipoles in a half field period, and the magnetic field will be computed
        assuming 'virtual' dipoles from the other half field period. Default = False.
    name : str
        Name of this DipoleSet.
    check_intersection: bool
        Whether or not to check the dipoles in the dipoleset for intersections.

    """
    _io_attrs_ = _Dipole._io_attrs_ + ["_dipoles", "_NFP", "_sym"]
    _io_attrs_.remove("_rho")
    _static_attrs = (
        OptimizableCollection._static_attrs
        + _Dipole._static_attrs
        + ["_NFP", "_sym", "_name"]
    )

    def __init__(self, *dipoles, NFP=1, sym=False, name="", check_intersection=False):
        dipoles = flatten_list(dipoles, flatten_tuple=True)
        assert all([isinstance(dipole, (_Dipole)) for dipole in dipoles])
        self._dipoles = list(dipoles)
        self._NFP = int(NFP)
        self._sym = bool(sym)
        self._name = str(name)

        #if check_intersection:
            #self.is_self_intersecting()

    @property
    def name(self):
        """str: Name of the dipole."""
        return self.__dict__.setdefault("_name", "")
    
    @name.setter
    def name(self, new):
        self._name = str(new)

    @property
    def dipoles(self):
        """list: dipoles in the dipoleset."""
        return self._dipoles

    @property
    def num_dipoles(self):
        """int: Number of dipoles."""
        return len(self) * (int(self.sym) + 1) * self.NFP

    @property
    def NFP(self):
        """int: Number of (toroidal) field periods."""
        return self._NFP

    @property
    def sym(self):
        """bool: Whether this dipole set is stellarator symmetric."""
        return self._sym
    
    @property
    def rho(self):
        """list: Optimization parameters for each dipole from (-1, 1)."""
        return [dipole.rho for dipole in self.dipoles]

    @rho.setter
    def rho(self, new):
        # new must be a 1D iterable regardless of the tree structure of the dipoleSet
        old, tree = tree_flatten(self.rho)
        new = jnp.atleast_1d(new).flatten()
        new = jnp.broadcast_to(new, (len(old),))
        new = tree_unflatten(tree, new)
        for dipole, cur in zip(self.dipoles, new):
            dipole.rho = cur

    def _all_rhos(self, rhos=None):
        """Return an array of all the rhos."""
        if rhos is None:
            rhos = self.rho
        rhos = jnp.asarray(rhos)
        if self.sym:
            rhos = jnp.concatenate([rhos, -1 * rhos[::-1]])
        return jnp.tile(rhos, self.NFP)

    def _make_arraylike(self, x):
        if isinstance(x, dict):
            x = [x] * len(self)
        try:
            len(x)
        except TypeError:
            x = [x] * len(self)
        assert len(x) == len(self)
        return x

    def compute(
        self, names, grid=None, params=None, transforms=None, data=None, **kwargs
    ):
        """Compute the quantity given by name on grid, for each dipole in the dipoleset.

        Parameters
        ----------
        names : str or array-like of str
            Name(s) of the quantity(s) to compute.
        grid : Grid or int, optional
            Grid of coordinates to evaluate at. Defaults to a Linear grid.
            If an integer, uses that many equally spaced points.
        params : dict of ndarray or array-like
            Parameters from the equilibrium. Defaults to attributes of self.
            If array-like, should be 1 value per coil.
        transforms : dict of Transform or array-like
            Transforms for R, Z, lambda, etc. Default is to build from grid.
        data : dict of ndarray or array-like
            Data computed so far, generally output from other compute functions
            If array-like, should be 1 value per dipole.

        Returns
        -------
        data : list of dict of ndarray
            Computed quantity and intermediate variables, for each dipoles in the set.
            List entries map to dipoles in dipoleset, each dict contains data for an
            individual dipole.

        """
        if params is None:
            params = [
                get_params(names, dipole, basis=kwargs.get("basis", "rpz"))
                for dipole in self
            ]
        if data is None:
            data = [{}] * len(self)

        # if user supplied initial data for each dipole we also need to vmap over that.
        data = vmap(
            lambda d, x: self[0].compute(
                names, grid=grid, transforms=transforms, data=d, params=x, **kwargs
            )
        )(tree_stack(data), tree_stack(params))
        return tree_unstack(data)

    def translate(self, *args, **kwargs):
        """Translate the dipoles along an axis."""
        [dipole.translate(*args, **kwargs) for dipole in self.dipoles]

    def rotate(self, *args, **kwargs):
        """Rotate the dipoles about an axis."""
        [dipole.rotate(*args, **kwargs) for dipole in self.dipoles]

    def flip(self, *args, **kwargs):
        """Flip the dipoles across a plane."""
        [dipole.flip(*args, **kwargs) for dipole in self.dipoles]

    def _compute_position(self, params=None, grid=None, dx1=False, **kwargs):
        basis = kwargs.pop("basis", "xyz")
        keys = ["x", "x_s"] if dx1 else ["x"]
        if params is None:
            params = [get_params(keys, dipole, basis=basis) for dipole in self]
        data = self.compute(keys, grid=grid, params=params, basis=basis, **kwargs)
        data = tree_leaves(data, is_leaf=lambda x: isinstance(x, dict))
        x = jnp.dstack([d["x"].T for d in data]).T  # shape=(ndipoles,num_nodes,3)
        if dx1:
            x_s = jnp.dstack([d["x_s"].T for d in data]).T  # shape=(ndipoles,num_nodes,3)
        # stellarator symmetry is easiest in [X,Y,Z] coordinates
        xyz = rpz2xyz(x) if basis.lower() == "rpz" else x
        if dx1:
            xyz_s = (
                rpz2xyz_vec(x_s, xyz[:, :, 0], xyz[:, :, 1])
                if basis.lower() == "rpz"
                else x_s
            )

        # if stellarator symmetric, add reflected dipoles from the other half field period
        if self.sym:
            normal = jnp.array(
                [-jnp.sin(jnp.pi / self.NFP), jnp.cos(jnp.pi / self.NFP), 0]
            )
            xyz_sym = xyz @ reflection_matrix(normal).T @ reflection_matrix([0, 0, 1]).T
            xyz = jnp.vstack((xyz, jnp.flipud(xyz_sym)))
            if dx1:
                xyz_s_sym = (
                    xyz_s @ reflection_matrix(normal).T @ reflection_matrix([0, 0, 1]).T
                )
                xyz_s = jnp.vstack((xyz_s, jnp.flipud(xyz_s_sym)))

        # field period rotation is easiest in [R,phi,Z] coordinates
        rpz = xyz2rpz(xyz)
        if dx1:
            rpz_s = xyz2rpz_vec(xyz_s, xyz[:, :, 0], xyz[:, :, 1])

        # if field period symmetry, add rotated dipoles from other field periods
        rpz0 = rpz
        for k in range(1, self.NFP):
            rpz = jnp.vstack((rpz, rpz0 + jnp.array([0, 2 * jnp.pi * k / self.NFP, 0])))
        if dx1:
            rpz_s = jnp.tile(rpz_s, (self.NFP, 1, 1))

        # ensure phi in [0, 2pi)
        rpz = rpz.at[:, :, 1].set(jnp.mod(rpz[:, :, 1], 2 * jnp.pi))

        x = rpz2xyz(rpz) if basis.lower() == "xyz" else rpz
        if dx1:
            x_s = (
                rpz2xyz_vec(rpz_s, phi=rpz[:, :, 1])
                if basis.lower() == "xyz"
                else rpz_s
            )
            return x, x_s
        return x
    
    def _compute_A_or_B(
        self,
        coords,
        params=None,
        basis="rpz",
        source_grid=None,
        transforms=None,
        compute_A_or_B="B",
        chunk_size=None,
    ):
        """Compute magnetic field at a set of points.

        Parameters
        ----------
        coords : array-like shape(n,3)
            Nodes to evaluate field at in [R,phi,Z] or [X,Y,Z] coordinates.
        params : dict or array-like of dict, optional
            Parameters to pass to dipoles, either the same for all dipoles or one for each.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic field.
        source_grid : Grid, int or None, optional
            Grid used to discretize dipoles. If an integer, uses that many equally spaced
            points. Should NOT include endpoint at 2pi.
        transforms : dict of Transform or array-like
            Transforms for R, Z, lambda, etc. Default is to build from grid.
        compute_A_or_B: {"A", "B"}, optional
            whether to compute the magnetic vector potential "A" or the magnetic field
            "B". Defaults to "B"
        chunk_size : int or None
            Size to split computation into chunks of evaluation points.
            If no chunking should be done or the chunk size is the full input
            then supply ``None``. Default is ``None``.

        Returns
        -------
        field : ndarray, shape(n,3)
            Magnetic field or vector potential at specified nodes, in [R,phi,Z]
            or [X,Y,Z] coordinates.

        """
        errorif(
            compute_A_or_B not in ["A", "B"],
            ValueError,
            f'Expected "A" or "B" for compute_A_or_B, instead got {compute_A_or_B}',
        )
        # NFP symmetry applies to dipoleset as a whole, not individual dipoles, so the grid
        # should have NFP=1.
        errorif(
            getattr(source_grid, "NFP", 1) != 1,
            ValueError,
            "source_grid for DipoleSet must have NFP=1",
        )
        assert basis.lower() in ["rpz", "xyz"]
        coords = jnp.atleast_2d(jnp.asarray(coords))
        if params is None:
            params = [{
                #get_params(["x", "y", "z", "phi", "theta", "m0", "rho"], dipole, basis=basis) for dipole in self
                "x": dipole.x,
                "y": dipole.y,  
                "z": dipole.z,
                "phi": dipole.phi,
                "theta": dipole.theta,
                "m0": dipole.m0,
                "rho": dipole.rho,
                "M0": dipole.M0,
            }
            for dipole in self
            ]
            for par, dipole in zip(params, self):
                par["rho"] = dipole.rho

        # stellarator symmetry is easiest in [X,Y,Z] coordinates
        if basis.lower() == "rpz":
            coords_xyz = rpz2xyz(coords)
        else:
            coords_xyz = coords

        # if stellarator symmetric, add reflected nodes from the other half field period
        if self.sym:
            normal = jnp.array(
                [-jnp.sin(jnp.pi / self.NFP), jnp.cos(jnp.pi / self.NFP), 0]
            )
            coords_sym = (
                coords_xyz
                @ reflection_matrix(normal).T
                @ reflection_matrix([0, 0, 1]).T
            )
            coords_xyz = jnp.vstack((coords_xyz, coords_sym))

        # field period rotation is easiest in [R,phi,Z] coordinates
        coords_rpz = xyz2rpz(coords_xyz)
        kernel = {"B": dipole_field, "A": dipole_vector_potential}[compute_A_or_B]

        # Stack every dipole's position and moment so the kernel can sum over all
        # sources in a single vectorized call. This replaces a sequential scan
        # over each dipole (which also re-transformed the eval points once per
        # dipole) and is mathematically identical, just summed in one shot.
        #
        # We stack on the host with NumPy rather than the jitted ``tree_stack``:
        # routing ~10^5 scalar leaves through jit compiles an enormous XLA graph
        # (minutes), while NumPy stacking is effectively instant. Fall back to
        # ``tree_stack`` only if params are traced (e.g. inside an optimizer).
        try:
            sources = {
                k: jnp.asarray(np.array([np.asarray(p[k]) for p in params]))
                for k in params[0]
            }
        except Exception:
            sources = tree_stack(params)
        rs = jnp.stack([sources["x"], sources["y"], sources["z"]], axis=-1)
        if "M0" in sources:
            M0 = sources["M0"]
        else:
            M0 = sources["m0"] * sources["rho"]
        sin_theta = jnp.sin(sources["theta"])
        m_hat = jnp.stack(
            [
                sin_theta * jnp.cos(sources["phi"]),
                sin_theta * jnp.sin(sources["phi"]),
                jnp.cos(sources["theta"]),
            ],
            axis=-1,
        )
        m = M0[:, jnp.newaxis] * m_hat

        # sum the magnetic fields from each field period
        def nfp_loop(k, AB):
            coords_nfp = coords_rpz + jnp.array([0, 2 * jnp.pi * k / self.NFP, 0])
            AB_xyz = kernel(rpz2xyz(coords_nfp), rs, m, chunk_size=chunk_size)
            AB += xyz2rpz_vec(AB_xyz, phi=coords_nfp[:, 1])
            return AB

        AB = fori_loop(0, self.NFP, nfp_loop, jnp.zeros_like(coords_rpz))

        # sum the magnetic field/potential from both halves of
        # the symmetric field period
        if self.sym:
            AB = AB[: coords.shape[0], :] + AB[coords.shape[0] :, :] * jnp.array(
                [-1, 1, 1]
            )

        if basis.lower() == "xyz":
            AB = rpz2xyz_vec(AB, x=coords[:, 0], y=coords[:, 1])
        return AB

    def compute_magnetic_field(
        self,
        coords,
        params=None,
        basis="rpz",
        source_grid=None,
        transforms=None,
        chunk_size=None,
    ):
        """Compute magnetic field at a set of points.

        Parameters
        ----------
        coords : array-like shape(n,3)
            Nodes to evaluate field at in [R,phi,Z] or [X,Y,Z] coordinates.
        params : dict or array-like of dict, optional
            Parameters to pass to coils, either the same for all dipoles or one for each.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic field.
        source_grid : Grid, int or None, optional
            Grid used to discretize dipoles. If an integer, uses that many equally spaced
            points. Should NOT include endpoint at 2pi.
        transforms : dict of Transform or array-like
            Transforms for R, Z, lambda, etc. Default is to build from grid.
        chunk_size : int or None
            Size to split computation into chunks of evaluation points.
            If no chunking should be done or the chunk size is the full input
            then supply ``None``. Default is ``None``.

        Returns
        -------
        field : ndarray, shape(n,3)
            Magnetic field at specified nodes, in [R,phi,Z] or [X,Y,Z] coordinates.

        """
        return self._compute_A_or_B(
            coords, params, basis, source_grid, transforms, "B", chunk_size=chunk_size
        )

    def compute_magnetic_vector_potential(
        self,
        coords,
        params=None,
        basis="rpz",
        source_grid=None,
        transforms=None,
        chunk_size=None,
    ):
        """Compute magnetic vector potential at a set of points.

        Parameters
        ----------
        coords : array-like shape(n,3)
            Nodes to evaluate potential at in [R,phi,Z] or [X,Y,Z] coordinates.
        params : dict or array-like of dict, optional
            Parameters to pass to dipoles, either the same for all dipoles or one for each.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic field.
        source_grid : Grid, int or None, optional
            Grid used to discretize dipoles. If an integer, uses that many equally spaced
            points. Should NOT include endpoint at 2pi.
        transforms : dict of Transform or array-like
            Transforms for R, Z, lambda, etc. Default is to build from grid.
        chunk_size : int or None
            Size to split computation into chunks of evaluation points.
            If no chunking should be done or the chunk size is the full input
            then supply ``None``. Default is ``None``.

        Returns
        -------
        vector_potential : ndarray, shape(n,3)
            magnetic vector potential at specified points, in either rpz
            or xyz coordinates

        """
        return self._compute_A_or_B(
            coords, params, basis, source_grid, transforms, "A", chunk_size=chunk_size
        )

    @classmethod
    def from_symmetry(cls, dipoles, NFP=1, sym=False):
        """Create a dipole group by reflection and symmetry.

        Given dipoles over one field period, repeat dipoles NFP times between
        0 and 2pi to form full dipole set.

        Or, given dipoles over 1/2 of a field period, repeat dipoles 2*NFP times
        between 0 and 2pi to form full stellarator symmetric dipole set.

        Parameters
        ----------
        dipoles : Dipole, DipoleSet
            Dipole or collection of dipoles in one field period or half field period.
        NFP : int (optional)
            Number of field periods for enforcing field period symmetry.
            The dipoles will be duplicated NFP times. Default = 1.
        sym : bool (optional)
            Whether to enforce stellarator symmetry.
            If True, the dipoles will be duplicated 2*NFP times. Default = False.
        check_intersection : bool
            whether to check the resulting dipoles for intersecting coils.

        Returns
        -------
        dipoleset : DipoleSet
            A new dipole set with NFP=1 and sym=False that is equivalent to the unique
            dipoles with field period symmetry and stellarator symmetry.
            The total number of dipoles in the new dipole set is:
            len(dipoleset) = len(dipoles) * NFP * (int(sym) + 1)

        """
        if not isinstance(dipoles, DipoleSet):
            dipoles = DipoleSet(dipoles)
        dipoleset = []
        if sym:
            # first reflect/flip original dipoleset
            # ie, given dipoles [1, 2, 3] at angles [pi/6, pi/2, 5pi/6]
            # we want a new set like [1, 2, 3, flip(3), flip(2), flip(1)]
            # at [pi/6, pi/2, 5pi/6, 7pi/6, 3pi/2, 11pi/6]
            flipped_dipoles = []
            normal = jnp.array([-jnp.sin(jnp.pi / NFP), jnp.cos(jnp.pi / NFP), 0])
            for dipole in dipoles[::-1]:
                fdipole = dipole.copy()
                fdipole.flip(normal)
                fdipole.flip([0, 0, 1])
                fdipole.rho = -1 * dipole.rho
                flipped_dipoles.append(fdipole)
            dipoles = dipoles + flipped_dipoles
        # next rotate the dipoleset for each field period
        for k in range(0, NFP):
            rotated_dipoles = dipoles.copy()
            rotated_dipoles.rotate(axis=[0, 0, 1], angle=2 * jnp.pi * k / NFP)
            dipoleset += rotated_dipoles

        return cls(*dipoleset)
    
    def calc_g(dipoles,eq, M_surf=8, N_surf=12):
        '''
        Calculate the inductance matrix

        takes a dipole list
        and a desc eq object, for computing surface normals
        '''

        mu_0 = 4 * np.pi * 1e-7
        grid = LinearGrid(M=M_surf, N=N_surf, NFP=eq.NFP, sym=False, endpoint=True)

        # n surface vectors
        n_surf = eq.surface.compute(['n_rho'], grid=grid)['n_rho']
        data = eq.compute(["X", "Y", "Z"], grid=grid)

        # n surface positions
        xyz = np.column_stack([data["X"], data["Y"], data["Z"]])

        # m-vector of the dipole, in xyz coordinates
        m_vec = np.array([d.m_xyz for d in dipoles]) 

        # m dipole positions
        m_pos = np.array([[d.x, d.y, d.z] for d in dipoles]) 

        # compute (n x m) pairwise distances
        nax = np.newaxis
        r_ij = xyz[:,nax,:] - m_pos[nax,:,:]

        # take (n x m) scalar magnitude
        r_mag = np.linalg.norm(r_ij, axis=-1)

        # get unit vector
        r_unit = r_ij / r_mag[:,:,nax]

        # these dot products will be used to compute the inductance matrix
        r_dot_n = np.sum(r_unit * n_surf[:,nax,:], axis=-1)
        r_dot_m = np.sum(r_unit * m_vec[nax,:,:], axis=-1)
        n_dot_m = np.sum( n_surf[:,nax,:] * m_vec[nax,:,:], axis=-1)

        # compute: mu0/4pi (3 r.n r.m - n.m) / r^3
        g_ij = mu_0 / (4*np.pi) * (3 * r_dot_n * r_dot_m - n_dot_m) / r_mag**3

        return g_ij, xyz

    # add is_self_intersecting and save_in_makegrid_format later on


    def __add__(self, other):
        if isinstance(other, (DipoleSet)):
            return DipoleSet(*self.dipoles, *other.dipoles)
        if isinstance(other, (list, tuple)):
            return DipoleSet(*self.dipoles, *other)
        else:
            return NotImplemented

    # dunder methods required by MutableSequence
    def __getitem__(self, i):
        return self.dipoles[i]

    def __setitem__(self, i, new_item):
        if not isinstance(new_item, _Dipole):
            raise TypeError("Members of DipoleSet must be of type Dipole.")
        self._dipoles[i] = new_item

    def __delitem__(self, i):
        del self._dipoles[i]

    def __len__(self):
        return len(self._dipoles)

    def insert(self, i, new_item):
        """Insert a new dipole into the dipoleset at position i."""
        if not isinstance(new_item, _Dipole):
            raise TypeError("Members of DipoleSet must be of type Dipole.")
        self._dipoles.insert(i, new_item)
    
    def __repr__(self):
        """Get the string form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " ((name={}, with {} submembers), x={}, y={}, z={}, phi={}, theta={}, m0={}, rho={})".format(self.name, len(self), self.x, self.y, self.z, self.phi, self.theta, self.m0, self.rho)
        )
    
def export_dipoles(dipole_set, f):
    '''
    Exports dipoles to a CSV file.

    Writes the data of each dipole to a comma-separated file with a header row.

    Parameters
    ----------
    dipole_set : DipoleSet
        Object containing a list of dipoles.
    f : str or path-like
        Path to the output file.

    Format
    -------------
    x (m), y (m), z (m), rho (unitless), phi (rad), theta (rad)
    '''
    d = dipole_set.dipoles

    outfile = open(f, 'w')

    print("x (m), y (m), z (m), rho (unitless), phi (rad), theta (rad)", file=outfile)
    
    for i in range(len(d)):
        print(f"{d[i].x}, {d[i].y}, {d[i].z}, {d[i].rho}, {d[i].phi}, {d[i].theta}", file=outfile)

    outfile.close()

import csv

def create_dipole(x, y, z, phi, theta, m0, rho):
    '''
    Creates a Dipole object using given data
    '''
    return _Dipole(x=x, y=y,z=z, phi=phi, theta=theta, m0=m0, rho=rho)

def import_dipoles(NFP, sym, filename):
    '''
    Creates a DipoleSet object using data from a given CSV file containing
    each dipole's attributes, including x, y, z, phi, theta, m0, and rho.
    '''
    with open(filename, newline="") as f:
        reader = csv.DictReader(f)

        csv_data = [
            (float(line["x (m)"]), float(line["y (m)"]), float(line["z (m)"]), float(line["phi (rad)"]), float(line["theta (rad)"]), float(line["m0"]),float(line["rho (unitless)"]))
            for line in reader
        ]
    dipole_set = DipoleSet(NFP=NFP, sym=sym)
    for line in csv_data:
        if (line[-1] != 0):
            dipole_set.append( create_dipole(*line))

    return dipole_set
