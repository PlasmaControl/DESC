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
from desc.geometry import (
    FourierPlanarCurve,
    FourierRZCurve,
    FourierXYCurve,
    FourierXYZCurve,
    SplineXYZCurve,
)
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
    """Implements ideal dipole that can be used in place of _Coil in a MixedCoilSet"""
    
    _io_attrs_ = _MagneticField._io_attrs_ + ["_x"] + ["_y"] + ["_z"] + ["_phi"] + ["_theta"] + ["_m0"] + ["_rho"] + ["_name"]
    _static_attrs = _MagneticField._static_attrs + Optimizable._static_attrs + ["_name"]

    def __init__(self, x=0.0, y=0.0, z=0.0, phi=0.0, theta=0.0, m0=1.0, rho=0.0, name=""):
        self._x = jnp.float64(float(np.squeeze(x)))
        self._y = jnp.float64(float(np.squeeze(x)))
        self._z = jnp.float64(float(np.squeeze(x)))
        self._phi = jnp.float64(float(np.squeeze(x)))
        self._theta = jnp.float64(float(np.squeeze(x)))
        self._m0 = jnp.float64(float(np.squeeze(x)))
        self._rho = jnp.float64(float(np.squeeze(x)))
        self._name = jnp.float64(float(np.squeeze(x)))
        super().__init__()

    def _set_up(self):
        for attribute in self._io_attrs_:
            if not hasattr(self, attribute):
                setattr(self, attribute, None)

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
        """float: Magnitude of magnetic dipole moment."""
        return self._m0

    @m0.setter
    def m0(self, new):
        assert jnp.isscalar(new) or new.size == 1
        self._m0 = jnp.float64(float(np.squeeze(new)))

    @optimizable_parameter
    @property
    def rho(self):
        """float: Optimization parameter in range (-1, 1)."""
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


    def _compute_A_or_B(
        self,
        coords,
        params=None,
        basis="rpz",
        compute_A_or_B="B",
        chunk_size=None,
    ):
        errorif(
            compute_A_or_B not in ["A", "B"],
            ValueError,
            f'Expected "A" or "B" for compute_A_or_B, instead got {compute_A_or_B}',
        )
        op = {"B": dipole_field, "A": dipole_vector_potential}[
            compute_A_or_B
        ]
        assert basis.lower() in ["rpz", "xyz"]
        coords = jnp.atleast_2d(jnp.asarray(coords))
        if basis.lower() == "rpz":
            phi_coords = coords[:, 1]
            coords = rpz2xyz(coords)

        if params is None:
            params = {}

        x = params.get("x", self.x)
        y = params.get("y", self.y)
        z = params.get("z", self.z)
        phi = params.get("phi", self.phi)
        theta = params.get("theta", self.theta)
        m0 = params.get("m0", self.m0)
        rho = params.get("rho", self.rho)

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
        chunk_size=None,
    ):
        return self._compute_A_or_B(
            coords, params, basis, "B", chunk_size=chunk_size
        )

    def compute_magnetic_vector_potential(
        self,
        coords,
        params=None,
        basis="rpz",
        chunk_size=None,
    ):
        return self._compute_A_or_B(
            coords, params, basis, "A", chunk_size=chunk_size
        )

    def __repr__(self):
        """Get the string form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (name={}, x={}, y={}, z={}, phi={}, theta={}, m0={}, rho={})".format(self.name, self.x, self.y, self.z, self.phi, self.theta, self.m0, self.rho)
        )