"""Classes for dipoles."""

import numbers
import os
from abc import ABC
from collections.abc import MutableSequence
from functools import partial

import numpy as np
from scipy.constants import mu_0

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
    eval_pts, mag_points, phi, theta, m, *, chunk_size=None
):
    m_x = m * jnp.sin(theta) * jnp.cos(phi)
    m_y = m * jnp.sin(theta) * jnp.sin(phi)
    m_z = m * jnp.cos(theta)
    m_vector = jnp.array([m_x, m_y, m_z])
    return dipole_field(
        eval_pts, mag_points, m_vector, chunk_size=chunk_size
    )


@partial(jit, static_argnames=["chunk_size"])
def magnetic_dipole_vector_field(
    eval_pts, mag_points, phi, theta, m, *, chunk_size=None
):
    m_x = m * jnp.sin(theta) * jnp.cos(phi)
    m_y = m * jnp.sin(theta) * jnp.sin(phi)
    m_z = m * jnp.cos(theta)
    m_vector = jnp.array([m_x, m_y, m_z])
    return dipole_vector_potential(
        eval_pts, mag_points, m_vector, chunk_size=chunk_size
    )

def test_magnetic_dipole():
    mu_0 = 4 * np.pi * 1e-7
    
    # single dipole at origin, pointing in z-direction and measured from the z axis
    mag_points = jnp.array([[0.0, 0.0, 0.0]])
    m_magnitude = 1.0  
    theta = 0.0 # +z
    phi = 0.0
    
    eval_pts = jnp.array([[0.0, 0.0, 1.0]]) 
    
    B = magnetic_dipole_field(eval_pts, mag_points, phi, theta, m_magnitude)
    A = magnetic_dipole_vector_field(eval_pts, mag_points, phi, theta, m_magnitude)
    
    r = 1.0
    B_expected_mag = (mu_0 / (4 * np.pi)) * (2 * m_magnitude / r**3)
    B_expected = jnp.array([[0.0, 0.0, B_expected_mag]])
    A_expected = jnp.array([[0.0, 0.0, 0.0]])
    
    print(f"calculated magnetic field: {B[0]}")
    print(f"expected magnetic field: {B_expected[0]}")
    print(f"calculated vector potential field: {A[0]}")
    print(f"expected vector potential field: {A_expected[0]}")

    # single dipole at origin, pointing in x-direction and measured from the z axis
    mag_points = jnp.array([[0.0, 0.0, 0.0]])
    m_magnitude = 1.0  
    theta = jnp.pi/2 # +x
    phi = 0.0
    
    eval_pts = jnp.array([[0.0, 0.0, 1.0]]) 
    
    B = magnetic_dipole_field(eval_pts, mag_points, phi, theta, m_magnitude)
    A = magnetic_dipole_vector_field(eval_pts, mag_points, phi, theta, m_magnitude)
    
    r = 1.0
    B_expected_mag = (mu_0 / (4 * np.pi)) * (-m_magnitude / r**3)
    B_expected = jnp.array([[B_expected_mag, 0.0, 0.0]])
    A_expected_mag = (mu_0 / (4 * np.pi)) * (m_magnitude)/(r**2)
    A_expected = jnp.array([[0.0, A_expected_mag, 0.0]])
    
    print(f"calculated magnetic field: {B[0]}")
    print(f"expected magnetic field: {B_expected[0]}")
    print(f"calculated vector potential field: {A[0]}")
    print(f"expected vector potential field: {A_expected[0]}")

if __name__ == "__main__":
    test_magnetic_dipole()
    

class _Dipole(_MagneticField, Optimizable, ABC):
    """Implements ideal dipole that can be used in place of _dipole in a MixeddipoleSet"""
    
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
        """float: Dimensionless optimization parameter in range (-1, 1)."""
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
    def M(self):
        """float: ."""
        return self._m0 * self._rho
    
    @property
    def m_xyz(self):
        m = self.M
        theta = self._theta
        phi = self._phi
        m_x = m * jnp.sin(theta) * jnp.cos(phi)
        m_y = m * jnp.sin(theta) * jnp.sin(phi)
        m_z = m * jnp.cos(theta)
        return jnp.array([m_x, m_y, m_z])
    
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

        #if params is None:
            #params = {}

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
        M = params.get("M", self.M)

        dipole_pos = jnp.array([[x, y, z]])

        m_vec = self.m_xyz

        AB = op(
            coords,
            dipole_pos,
            phi,
            theta, 
            M,
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
        return self._compute_A_or_B(
            coords, params, basis, source_grid, transforms, "A", chunk_size=chunk_size
        )
    
    def translate(self, displacement=[0, 0, 0]):
        """Translate the curve by a rigid displacement in X,Y,Z coordinates."""
        self._x += jnp.asarray(displacement)[0]
        self._y += jnp.asarray(displacement)[1]
        self._z += jnp.asarray(displacement)[2]
        self.shift = self.shift + jnp.asarray(displacement)

    def rotate(self, axis=[0, 0, 1], angle=0):
        """Rotate the curve by a fixed angle about axis in X,Y,Z coordinates."""
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
        """Flip the curve about the plane with specified normal in X,Y,Z coordinates."""
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
        """str: Name of the curve."""
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




        # note to find out what cur is and maybe make an edit here




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
        op = {
            "B": self[0].compute_magnetic_field,
            "A": self[0].compute_magnetic_vector_potential,
        }[compute_A_or_B]

        # sum the magnetic fields from each field period
        def nfp_loop(k, AB):
            coords_nfp = coords_rpz + jnp.array([0, 2 * jnp.pi * k / self.NFP, 0])

            def body(AB, x):
                AB += op(
                    coords_nfp,
                    params=x,
                    basis="rpz",
                    source_grid=source_grid,
                    chunk_size=chunk_size,
                )
                return AB, None

            AB += scan(body, jnp.zeros(coords_nfp.shape), tree_stack(params))[0]
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
        return self._compute_A_or_B(
            coords, params, basis, source_grid, transforms, "A", chunk_size=chunk_size
        )

    @classmethod
    def linspaced_angular(
        cls,
        dipole,
        rho=None,
        axis=[0, 0, 1],
        angle=2 * np.pi,
        n=10,
        endpoint=False,
        check_intersection=True,
    ):
        assert isinstance(dipole, _Dipole) and not isinstance(dipole, DipoleSet)
        if rho is None:
            rho = dipole.rho
        rhos = jnp.broadcast_to(rho, (n,))
        phi = jnp.linspace(0, angle, n, endpoint=endpoint)
        dipoles = []
        for i in range(n):
            dipolei = dipole.copy()
            dipolei.rotate(axis=axis, angle=phi[i])
            dipolei.rho = rhos[i]
            dipoles.append(dipolei)
        return cls(*dipoles, check_intersection=check_intersection)

    @classmethod
    def linspaced_linear(
        cls,
        dipole,
        rho=None,
        displacement=[2, 0, 0],
        n=4,
        endpoint=False,
        check_intersection=True,
    ):
        assert isinstance(dipole, _Dipole) and not isinstance(dipole, DipoleSet)
        if rho is None:
            rho = dipole.rho
        rhos = jnp.broadcast_to(rho, (n,))
        displacement = jnp.asarray(displacement)
        a = jnp.linspace(0, 1, n, endpoint=endpoint)
        dipoles = []
        for i in range(n):
            dipolei = dipole.copy()
            dipolei.translate(a[i] * displacement)
            dipolei.rho = rhos[i]
            dipoles.append(dipolei)
        return cls(*dipoles, check_intersection=check_intersection)

    @classmethod
    def from_symmetry(cls, dipoles, NFP=1, sym=False):
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
    d = dipole_set.dipoles

    outfile = open(f, 'w')

    print("x (m), y (m), z (m), rho (unitless), phi (rad), theta (rad)", file=outfile)
    
    for i in range(len(d)):
        print(f"{d[i].x}, {d[i].y}, {d[i].z}, {d[i].rho}, {d[i].phi}, {d[i].theta}", file=outfile)

    outfile.close()