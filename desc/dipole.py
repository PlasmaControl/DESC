"""Classes for dipoles."""

import os
import csv
import jax

from abc import ABC
from collections.abc import MutableSequence
from functools import partial

import numpy as np

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
from desc.grid import Grid, LinearGrid, QuadratureGrid, _Grid
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
    rotation_matrix,
    rpz2xyz,
    rpz2xyz_vec,
    safenorm,
    warnif,
    xyz2rpz,
    xyz2rpz_vec,
)
import numbers
from desc.compute import data_index
from desc.compute.utils import (
    _parse_parameterization,
    get_data_deps,
    get_params,
    get_transforms,
)

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
    
    _io_attrs_ = _MagneticField._io_attrs_ + [
        "_X",
        "_Y",
        "_Z",
        "_phi",
        "_theta",
        "_m0",
        "_rho",
        "_name",
        "_shift",
        "_rotmat",
    ]
    _static_attrs = _MagneticField._static_attrs + Optimizable._static_attrs + ["_name"]

    def __init__(self, X=0.0, Y=0.0, Z=0.0, phi=0.0, theta=0.0, m0=1.0, rho=1.0, name=""):
        self._X = jnp.float64(float(np.squeeze(X)))
        self._Y = jnp.float64(float(np.squeeze(Y)))
        self._Z = jnp.float64(float(np.squeeze(Z)))
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


    @property
    def X(self):
        """float: X-coordinate of dipole position."""
        return self._X

    @X.setter
    def X(self, new):
        assert jnp.isscalar(new) or new.size == 1
        self._X = jnp.float64(float(np.squeeze(new)))

    @property
    def Y(self):
        """float: Y-coordinate of dipole position."""
        return self._Y

    @Y.setter
    def Y(self, new):
        assert jnp.isscalar(new) or new.size == 1
        self._Y = jnp.float64(float(np.squeeze(new)))

    @property
    def Z(self):
        """float: Z-coordinate of dipole position."""
        return self._Z

    @Z.setter
    def Z(self, new):
        assert jnp.isscalar(new) or new.size == 1
        self._Z = jnp.float64(float(np.squeeze(new)))

    @property
    def phi(self):
        """float: Azimuthal angle of dipole orientation."""
        return self._phi

    @phi.setter
    def phi(self, new):
        assert jnp.isscalar(new) or new.size == 1
        self._phi = jnp.float64(float(np.squeeze(new)))

    @property
    def theta(self):
        """float: Polar angle of dipole orientation."""
        return self._theta

    @theta.setter
    def theta(self, new):
        assert jnp.isscalar(new) or new.size == 1
        self._theta = jnp.float64(float(np.squeeze(new)))

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
        """float: Dimensionless ofree unconstrained parameter seen by the optimizer"""
        return self._rho

    @rho.setter
    def rho(self, new):
        assert jnp.isscalar(new) or new.size == 1
        self._rho = jnp.float64(float(np.squeeze(new)))

    @property
    def rho_tilde(self):
        """float: Dimensionless parameter in range (-1, 1) that defines radial
        direction and magntiude of the dipole; positive is radially outward, negative is 
        radially inward.
        """
        return jnp.tanh(self._rho)

    @rho_tilde.setter
    def rho_tilde(self, new):
        new = jnp.clip(jnp.asarray(float(np.squeeze(new))), -1 + 1e-7, 1 - 1e-7)
        self._rho = jnp.arctanh(new)

    @property
    def name(self):
        """str: Name of the dipole."""
        return self._name

    @name.setter
    def name(self, new):
        self._name = str(new)

    @property
    def position(self):
        """ndarray: Position of dipole as [X,Y,Z]."""
        return jnp.array([self._X, self._Y, self._Z])
    
    @property
    def M0(self):
        """float: Effective dipole moment strength, with radial direction."""
        return self._m0 * self.rho_tilde
    
    @property
    def m_xyz(self):
        """float: Effective dipole moment strength, with radial direction expressed
        in an array of its x, y, and z components."""
        M0 = self.m0 * self.rho_tilde
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
            Unused, for the same reason as ``source_grid``. Kept for API
            compatibility.
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

        # rho is the only registered dof, so it's the only quantity that
        # needs to support being overridden via `params`. Everything else below is fixed
        # geometry, read straight off `self`.
        if params is None:
            rho = jnp.tanh(self.rho)
        else:
            rho = jnp.tanh(params.get("rho", self.rho))

        dipole_pos = jnp.array([[self.X, self.Y, self.Z]])

        AB = op(
            coords,
            dipole_pos,
            self.phi,
            self.theta,
            self.m0 * rho,
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
        self._X += jnp.asarray(displacement)[0]
        self._Y += jnp.asarray(displacement)[1]
        self._Z += jnp.asarray(displacement)[2]
        self.shift = self.shift + jnp.asarray(displacement)

    def rotate(self, axis=[0, 0, 1], angle=0):
        """Rotate the Dipole by a fixed angle about axis in X,Y,Z coordinates."""
        R = rotation_matrix(axis_or_vector=axis, angle_or_vector=angle)
        
        pos = jnp.array([self._X, self._Y, self._Z])
        new_pos = R @ pos
        self._X = new_pos[0]
        self._Y = new_pos[1]
        self._Z = new_pos[2]
        
        m_vec = jnp.array([jnp.sin(self.theta)*jnp.cos(self.phi),
                           jnp.sin(self.theta)*jnp.sin(self.phi), jnp.cos(self.theta)])
        new_m = R @ m_vec

        m_magnitude = jnp.linalg.norm(new_m)
        self._theta = jnp.arccos(new_m[2] / m_magnitude)
        self._phi = jnp.arctan2(new_m[1], new_m[0])
        
        self.rotmat = (R @ self.rotmat.reshape(3, 3)).flatten()
        self.shift = self.shift @ R.T

    def flip(self, normal=[0, 0, 1]):
        """Flip the Dipole about the plane with specified normal in X,Y,Z coordinates."""
        F = reflection_matrix(normal)
        
        pos = jnp.array([self._X, self._Y, self._Z])
        new_pos = F @ pos
        self._X = new_pos[0]
        self._Y = new_pos[1]
        self._Z = new_pos[2]
        
        m_vec = jnp.array([jnp.sin(self.theta)*jnp.cos(self.phi),
                           jnp.sin(self.theta)*jnp.sin(self.phi), jnp.cos(self.theta)])
        new_m = F @ m_vec

        m_magnitude = jnp.linalg.norm(new_m)
        self._theta = jnp.arccos(new_m[2] / m_magnitude)
        self._phi = jnp.arctan2(new_m[1], new_m[0])
        
        self.rotmat = (F @ self.rotmat.reshape(3, 3)).flatten()
        self.shift = self.shift @ F.T

    def compute(
            self,
            names,
            grid=None,
            params=None,
            transforms=None,
            data=None,
            override_grid=True,
            **kwargs,
        ):
            """Compute the quantity given by name on grid.

            Parameters
            ----------
            names : str or array-like of str
                Name(s) of the quantity(s) to compute.
            grid : Grid or int, optional
                Grid of coordinates to evaluate at. Defaults to a Linear grid.
                If an integer, uses that many equally spaced points.
            params : dict of ndarray
                Parameters from the equilibrium. Defaults to attributes of self.
            transforms : dict of Transform
                Transforms for R, Z, lambda, etc. Default is to build from grid
            data : dict of ndarray
                Data computed so far, generally output from other compute functions.
                Any vector v = v¹ R̂ + v² ϕ̂ + v³ Ẑ should be given in components
                v = [v¹, v², v³] where R̂, ϕ̂, Ẑ are the normalized basis vectors
                of the cylindrical coordinates R, ϕ, Z.
            override_grid : bool
                If True, override the user supplied grid if necessary and use a full
                resolution grid to compute quantities and then downsample to user requested
                grid. If False, uses only the user specified grid, which may lead to
                inaccurate values for surface or volume averages.

            Returns
            -------
            data : dict of ndarray
                Computed quantity and intermediate variables.

            """
            if isinstance(names, str):
                names = [names]
            if grid is None:
                grid = LinearGrid(N=2 * 15 * getattr(self, "NFP", 1) + 5)
            elif isinstance(grid, numbers.Integral):
                grid = LinearGrid(N=grid)
            errorif(
                not isinstance(grid, _Grid),
                TypeError,
                f"grid argument must be a Grid object or an integer, got type {type(grid)}",
            )

            if params is None:
                params = get_params(names, obj=self, basis=kwargs.get("basis", "rpz"))
            if transforms is None:
                transforms = get_transforms(
                    names,
                    obj=self,
                    grid=grid,
                    jitable=True,
                )
            if data is None:
                data = {}
            profiles = {}

            p = _parse_parameterization(self)
            deps = list(set(get_data_deps(names, obj=p) + names))
            dep0d = [
                dep
                for dep in deps
                if (data_index[p][dep]["coordinates"] == "") and (dep not in data)
            ]
            calc0d = bool(len(dep0d))
            # see if the grid we're already using will work for desired qtys
            if calc0d and (grid.N >= 2 * 15 + 5) and isinstance(grid, LinearGrid):
                calc0d = False

            if calc0d and override_grid:
                grid0d = LinearGrid(N=2 * 15 * getattr(self, "NFP", 1) + 5)
                data0d = compute_fun(
                    self,
                    dep0d,
                    params=params,
                    transforms=get_transforms(
                        dep0d,
                        obj=self,
                        grid=grid0d,
                        jitable=True,
                        **kwargs,
                    ),
                    profiles={},
                    data=None,
                    **kwargs,
                )
                # these should all be 0d quantities so don't need to compress/expand
                data0d = {key: val for key, val in data0d.items() if key in dep0d}
                data.update(data0d)

            data = compute_fun(
                self,
                names,
                params=params,
                transforms=transforms,
                profiles=profiles,
                data=data,
                **kwargs,
            )
            return data

    def __repr__(self):
        """Get the string form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (name={}, x={}, y={}, z={}, phi={}, theta={}, m0={}, rho={})".format(self.name, self.X, self.Y, self.Z, self.phi, self.theta, self.m0, self.rho)
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

    def tree_flatten(self):
        """Custom JAX pytree registration, overriding _AutoRegisterPytree's
        generic behavior (desc/io/core.py).

        Without this override: `_dipoles` isn't in `_static_attrs`, so the
        generic flattener treats it as dynamic state and recurses into it --
        and since every individual `_Dipole` is ALSO independently pytree-
        registered, JAX walks all the way down into each dipole's own
        X/Y/Z/phi/theta/m0/rho attributes. At n_dipoles=75,460 this produced
        528,220 leaves from a single flatten (confirmed via
        jax.tree_util.tree_flatten timing), and since `self` is an implicit
        positional argument to every @jit-decorated _Objective method
        (compute_scalar, grad, jac_scaled, ...), this recursive flatten ran
        on every trace/compile -- not the actual rho array itself.

        The evolving optimization state (rho) is never read from `self`
        inside a traced compute() call -- it arrives through the separate
        `x`/`field_params` argument via unpack_params, which is the normal
        DESC Optimizable.params_dict/pack_params machinery (a completely
        separate, non-jit, Python-level system from this pytree
        registration). So DipoleSet can be treated as fully static from
        JIT's perspective: zero dynamic children, everything as hashable
        aux_data. jit correctly still recompiles if a genuinely different
        DipoleSet object (different Python identity) is ever passed in.
        """
        children = ()
        aux_data = (tuple(self._dipoles), self._NFP, self._sym, self._name)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Inverse of tree_flatten."""
        dipoles, NFP, sym, name = aux_data
        return cls(*dipoles, NFP=NFP, sym=sym, name=name)

    def _invalidate_geometry_cache(self):
        """Clear cached aggregate arrays after the dipole list itself changes. 
        Position etc are fixed, non-optimizable geometry, so caching them avoids 
        rebuilding a big array via a Python loop on every access -- these properties
        were previously rebuilt from scratch every single call, including
        calls made from inside JIT-traced objective compute functions."""
        for key in ("_X_cache", "_Y_cache", "_Z_cache", "_phi_cache", "_theta_cache", "_m0_cache"):
            self.__dict__.pop(key, None)

    def _cached_aggregate(self, cache_key, compute_fn):
        """Return a cached aggregate array, but only ever persist a CONCRETE
        (non-traced) result. If the first computation happens to occur while
        inside an active JAX trace, the returned value would be a tracer tied
        to that specific trace -- caching it would leak that tracer past the
        trace's lifetime and crash on any later use (UnexpectedTracerError).
        So: always recompute while tracing (cheap, and JAX handles it fine
        inside the trace itself); only cache once we get a real array back."""
        cached = self.__dict__.get(cache_key)
        if cached is not None and not isinstance(cached, jax.core.Tracer):
            return cached
        value = compute_fn()
        if not isinstance(value, jax.core.Tracer):
            self.__dict__[cache_key] = value
        return value

    @property
    def X(self):
        """X coordinates of all dipoles."""
        return self._cached_aggregate(
            "_X_cache", lambda: jnp.asarray([dipole.X for dipole in self])
        )

    @property
    def Y(self):
        """Y coordinates of all dipoles."""
        return self._cached_aggregate(
            "_Y_cache", lambda: jnp.asarray([dipole.Y for dipole in self])
        )

    @property
    def Z(self):
        """Z coordinates of all dipoles."""
        return self._cached_aggregate(
            "_Z_cache", lambda: jnp.asarray([dipole.Z for dipole in self])
        )

    @property
    def phi(self):
        """Azimuthal angles of all dipoles."""
        return self._cached_aggregate(
            "_phi_cache", lambda: jnp.asarray([dipole.phi for dipole in self])
        )

    @property
    def theta(self):
        """Polar angles of all dipoles."""
        return self._cached_aggregate(
            "_theta_cache", lambda: jnp.asarray([dipole.theta for dipole in self])
        )

    @property
    def m0(self):
        """Dipole moment magnitudes of all dipoles."""
        return self._cached_aggregate(
            "_m0_cache", lambda: jnp.asarray([dipole.m0 for dipole in self])
        )

    @property
    def shift(self):
        """Shift vectors of all dipoles."""
        return jnp.asarray([dipole.shift for dipole in self])


    @property
    def rotmat(self):
        """Rotation matrices of all dipoles."""
        return jnp.asarray([dipole.rotmat for dipole in self])
    

    @property
    def rho_tilde(self):
        """list: """
        return [dipole.rho_tilde for dipole in self.dipoles]

    @optimizable_parameter
    @property
    def rho(self):
        """ndarray, shape(n,): raw (unconstrained) optimizer variable for
        every dipole in this set, as ONE array leaf rather than one leaf
        per dipole (previously a python list -- 75,460 separate leaves at
        scale, which is what blew up autodiff/JIT compilation)."""
        return jnp.asarray([dipole._rho for dipole in self.dipoles])

    @rho.setter
    def rho(self, new):
        new = jnp.broadcast_to(jnp.atleast_1d(new), (len(self),))
        for dipole, cur in zip(self.dipoles, new):
            dipole._rho = cur

    optimizable_params = Optimizable.optimizable_params
    params_dict = Optimizable.params_dict
    dimensions = Optimizable.dimensions
    x_idx = Optimizable.x_idx
    dim_x = Optimizable.dim_x
    pack_params = Optimizable.pack_params
    unpack_params = Optimizable.unpack_params
    _get_ess_scale = Optimizable._get_ess_scale

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
        errorif(
            getattr(source_grid, "NFP", 1) != 1,
            ValueError,
            "source_grid for DipoleSet must have NFP=1",
        )
        assert basis.lower() in ["rpz", "xyz"]
        coords = jnp.atleast_2d(jnp.asarray(coords))
        if params is None:
            rho_arr = jnp.tanh(jnp.asarray(self.rho))
        else:
            params_list = self._make_arraylike(params)
            rho_arr = jnp.stack(
                [
                    jnp.tanh(jnp.asarray(par.get("rho", dipole.rho)).reshape(-1)[0])
                    for par, dipole in zip(params_list, self)
                ]
            )

        rs = jnp.stack([self.X, self.Y, self.Z], axis=-1)
        M0 = self.m0 * rho_arr
        sin_theta = jnp.sin(self.theta)
        m_hat = jnp.stack(
            [
                sin_theta * jnp.cos(self.phi),
                sin_theta * jnp.sin(self.phi),
                jnp.cos(self.theta),
            ],
            axis=-1,
        )
        m = M0[:, jnp.newaxis] * m_hat

        if basis.lower() == "rpz":
            coords_xyz = rpz2xyz(coords)
        else:
            coords_xyz = coords

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

        coords_rpz = xyz2rpz(coords_xyz)
        kernel = {"B": dipole_field, "A": dipole_vector_potential}[compute_A_or_B]

        def nfp_loop(k, AB):
            coords_nfp = coords_rpz + jnp.array([0, 2 * jnp.pi * k / self.NFP, 0])
            AB_xyz = kernel(rpz2xyz(coords_nfp), rs, m, chunk_size=chunk_size)
            AB += xyz2rpz_vec(AB_xyz, phi=coords_nfp[:, 1])
            return AB

        AB = fori_loop(0, self.NFP, nfp_loop, jnp.zeros_like(coords_rpz))

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
    def linspaced_angular(
        cls, dipole, rho=None, axis=(0, 0, 1), angle=2 * np.pi,
        n=10, endpoint=False, *, NFP=1, sym=False,
    ):
        """Copy a dipole ``n`` times at equally spaced rotation angles.

        Parameters
        ----------
        dipole : _Dipole
            Single template dipole (not a DipoleSet).
        rho : float or array-like, optional
            Raw optimization parameter(s), not ``rho_tilde``. By default
            preserve the template's value. A scalar broadcasts to all copies.
        axis : array-like (3,)
            Cartesian rotation axis through the origin.
        angle : float
            Angular extent in radians.
        n : int
            Number of dipoles to generate.
        endpoint : bool
            Whether to include the final angular position.
        NFP, sym : int, bool
            Virtual field-period and stellarator symmetry of the result.
        """
        if not isinstance(dipole, _Dipole) or isinstance(dipole, DipoleSet):
            raise TypeError("dipole must be a single _Dipole")
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("n must be a positive integer")
        axis = np.asarray(axis, dtype=float)
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) == 0:
            raise ValueError("axis must be a nonzero finite Cartesian 3-vector")
        values = np.broadcast_to(
            np.asarray(dipole.rho if rho is None else rho, dtype=float), (n,)
        )
        result = []
        for phi, strength in zip(np.linspace(0, angle, n, endpoint=endpoint), values):
            item = dipole.copy()
            item.rotate(axis=axis, angle=float(phi))
            item.rho = float(strength)
            result.append(item)
        return cls(*result, NFP=NFP, sym=sym)

    @classmethod
    def linspaced_linear(
        cls, dipole, rho=None, displacement=(2, 0, 0), n=4,
        endpoint=False, *, NFP=1, sym=False,
    ):
        """Copy a dipole ``n`` times at equally spaced Cartesian offsets.

        ``displacement`` is the total offset at fraction 1, so with
        ``endpoint=False`` the last point falls before that endpoint.
        ``rho`` is the raw (unconstrained) optimization parameter.
        """
        if not isinstance(dipole, _Dipole) or isinstance(dipole, DipoleSet):
            raise TypeError("dipole must be a single _Dipole")
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("n must be a positive integer")
        displacement = np.asarray(displacement, dtype=float)
        if displacement.shape != (3,) or not np.isfinite(displacement).all():
            raise ValueError("displacement must be a finite Cartesian 3-vector")
        values = np.broadcast_to(
            np.asarray(dipole.rho if rho is None else rho, dtype=float), (n,)
        )
        result = []
        for frac, strength in zip(np.linspace(0, 1, n, endpoint=endpoint), values):
            item = dipole.copy()
            item.translate(displacement * float(frac))
            item.rho = float(strength)
            result.append(item)
        return cls(*result, NFP=NFP, sym=sym)

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

    
    
    def calc_g(self, eq, grid=None, M_surf=8, N_surf=12):
        '''
        Calculate the inductance matrix

        takes a dipole list
        and a desc eq object, for computing surface normals
        '''

        mu_0 = 4 * jnp.pi * 1e-7
        if grid is None:
            grid = LinearGrid(M=M_surf, N=N_surf, NFP=eq.NFP, sym=False, endpoint=True)

        data = eq.compute(["x", "n_rho"], grid=grid, basis="xyz")
        xyz = jnp.asarray(data["x"])
        n_surf = jnp.asarray(data["n_rho"])
 
        theta = jnp.asarray([d.theta for d in self])
        phi = jnp.asarray([d.phi for d in self])
        m_hat_unique = jnp.stack(
            [jnp.sin(theta) * jnp.cos(phi),
             jnp.sin(theta) * jnp.sin(phi),
             jnp.cos(theta)],
            axis=-1,
        )
        m_pos_unique = jnp.stack([self.X, self.Y, self.Z], axis=-1)
        n_unique = len(self)
 
        m_pos_full, m_hat_full = m_pos_unique, m_hat_unique
        owner_full = jnp.arange(n_unique)
 
        if self.sym:
            normal = jnp.array(
                [-jnp.sin(jnp.pi / self.NFP), jnp.cos(jnp.pi / self.NFP), 0]
            )
            F = reflection_matrix(normal) @ reflection_matrix([0, 0, 1])
            m_pos_sym = m_pos_unique @ F.T
            m_hat_sym = -(m_hat_unique @ F.T)
            m_pos_full = jnp.concatenate([m_pos_full, m_pos_sym], axis=0)
            m_hat_full = jnp.concatenate([m_hat_full, m_hat_sym], axis=0)
            owner_full = jnp.concatenate([owner_full, jnp.arange(n_unique)])
 
        m_pos_all = [m_pos_full]
        m_hat_all = [m_hat_full]
        owner_all = [owner_full]
        for k in range(1, self.NFP):
            Rz = rotation_matrix(axis_or_vector=[0, 0, 1], angle_or_vector=2 * jnp.pi * k / self.NFP)
            m_pos_all.append(m_pos_full @ Rz.T)
            m_hat_all.append(m_hat_full @ Rz.T)
            owner_all.append(owner_full)
        m_pos_all = jnp.concatenate(m_pos_all, axis=0)
        m_hat_all = jnp.concatenate(m_hat_all, axis=0)
        owner_all = jnp.concatenate(owner_all, axis=0)
 
        r_ij = xyz[:, None, :] - m_pos_all[None, :, :]
        r_mag = jnp.linalg.norm(r_ij, axis=-1)
        r_unit = r_ij / r_mag[:, :, None]
 
        r_dot_n = jnp.sum(r_unit * n_surf[:, None, :], axis=-1)
        r_dot_m = jnp.sum(r_unit * m_hat_all[None, :, :], axis=-1)
        n_dot_m = jnp.sum(n_surf[:, None, :] * m_hat_all[None, :, :], axis=-1)
 
        g_image = mu_0 / (4 * jnp.pi) * (3 * r_dot_n * r_dot_m - n_dot_m) / r_mag ** 3
 
        g_ij = jnp.zeros((xyz.shape[0], n_unique))
        g_ij = g_ij.at[:, owner_all].add(g_image)
 
        return g_ij

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
        self._invalidate_geometry_cache()

    def __delitem__(self, i):
        del self._dipoles[i]
        self._invalidate_geometry_cache()

    def insert(self, i, new_item):
        """Insert a new dipole into the dipoleset at position i."""
        if not isinstance(new_item, _Dipole):
            raise TypeError("Members of DipoleSet must be of type Dipole.")
        self._dipoles.insert(i, new_item)
        self._invalidate_geometry_cache()

    def __len__(self):
        return len(self._dipoles)
    
    def __repr__(self):
        """Get the string form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (name={}, with {} submembers)".format(self.name, len(self))
        )

    
def export_dipoles(dipole_set, f):
    """
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
    """
    d = dipole_set.dipoles

    outfile = open(f, 'w')

    print("x (m), y (m), z (m), phi (rad), theta (rad), m0, rho (unitless)", file=outfile)
    
    for i in range(len(d)):
        print(f"{d[i].X}, {d[i].Y}, {d[i].Z}, {d[i].phi}, {d[i].theta}, {d[i].m0}, {d[i].rho_tilde}", file=outfile)

    outfile.close()


def import_dipoles(NFP, sym, filename):
    """
    Create a DipoleSet from a .focus or .csv file.

    FOCUS files are automatically converted to CSV before importing.

    Parameters
    ----------
    NFP : int
        Number of field periods for enforcing field-period symmetry.
    sym : bool
        Whether to enforce stellarator symmetry.
    filename : str or path-like
        Path to a .focus or .csv file.

    Returns
    -------
    dipole_set : DipoleSet
        Imported collection of dipoles.
    """
    from pathlib import Path

    filename = Path(filename)

    if filename.suffix.lower() == ".focus":
        output_filename = filename.with_suffix(".csv")

        data = np.genfromtxt(
            filename,
            delimiter=",",
            skip_header=3,
            usecols=(3, 4, 5, 10, 11, 7, 8, 6),
            ndmin=2,
        )

        if data.size == 0 or not np.all(np.isfinite(data)):
            raise ValueError(f"Invalid FOCUS data in {filename}")

        header = [
            "x (m)",
            "y (m)",
            "z (m)",
            "phi (rad)",
            "theta (rad)",
            "m0",
            "rho (unitless)",
            "Ic",
        ]

        with open(output_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)

        filename = output_filename

    elif filename.suffix.lower() != ".csv":
        raise ValueError(
            f"Unsupported file format '{filename.suffix}'. "
            "Expected .focus or .csv."
        )

    dipole_set = DipoleSet(NFP=NFP, sym=sym)

    with open(filename, newline="") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        has_Ic = "Ic" in reader.fieldnames

        for line in reader:
            Ic = float(line["Ic"]) if has_Ic else 1.0

            # Skips dipoles marked as inactive, assuming they are blocked by ports
            if Ic == 0:
                continue

            dip = _Dipole(
                X=float(line["x (m)"]),
                Y=float(line["y (m)"]),
                Z=float(line["z (m)"]),
                phi=float(line["phi (rad)"]),
                theta=float(line["theta (rad)"]),
                m0=float(line["m0"]),
            )

            dip.rho_tilde = float(line["rho (unitless)"])

            dipole_set.append(dip)

    return dipole_set

def initialize_dipole_grid(
    axis_xyz=None, a=None, target_size=None, L=1, *, layer_thick=None,
    wall_thick=0.0, gap=0.0, magnetization=1.0e6, m0=None,
    rho=0.0, ports=(), NFP=1, sym=False, n_fine=512,
    eq=None, M_surface=48, N_surface=96,
    d_offset=0.02, seed_theta=0.0, L_straight=0.0,
    theta_line_samples=720, z_step=0.002, verbose=False,
):
    """Generate a plasma-following or axis-tube permanent-magnet grid.

    With ``eq`` supplied (or as the first positional argument), the magnet
    towers follow the actual sampled LCFS, including noncircular and
    nonaxisymmetric shaping. Without ``eq``, the original circular
    axis-normal tube is retained for backwards compatibility.

    The legacy path is based on ``make_circular_normal_PM_grid_global.py``
    from Djin; The input ``axis_xyz`` samples a *single* toroidal domain. 
    Towers use arc-length spacing along each displaced toroidal curve, and magnets 
    point inward toward the axis.

    Parameters
    ----------
    axis_xyz : array-like, shape (k, 3), or Equilibrium, optional
        Legacy centerline samples; alternatively supply an equilibrium here.
    a : float, optional
        Circular vessel radius [m] for the legacy centerline mode. Not used
        when ``eq`` is supplied.
    eq : Equilibrium, optional
        DESC equilibrium defining the LCFS to follow. Its surface normal is
        used to set magnet positions and inward dipole directions.
    M_surface, N_surface : int
        Poloidal and toroidal resolutions for sampling the plasma boundary.
    target_size : float
        Square magnet footprint width [m].
    L : int
        Maximum number of radial magnet layers per tower.
    layer_thick : float, optional
        Magnet radial thickness [m]; defaults to ``target_size``.
    wall_thick : float
        Extra footprint pitch [m] between towers.
    gap : float
        Surface-to-first-magnet-face clearance [m].
    magnetization : float
        Magnetization [A/m], giving m0 = magnetization * volume.
    m0 : float, optional
        Explicit moment [A m^2], overrides magnetization-derived moment.
    rho : float
        Initial *raw* strength, e.g. 0 for a cold start.
    ports : sequence of mappings
        Port definitions ``theta_port``, ``s_port`` in [0, 1], and
        ``port_radius`` [m]. Blocked towers are OMITTED since DipoleSet
        has no Ic parameter (unlike FAMUS/FOCUS).
    NFP, sym : int, bool
        Virtual symmetries; only generate unique-domain dipoles.
    d_offset : float
        Vessel-to-plasma normal offset [m].
    seed_theta, L_straight : float
        Line seed angle and optional end-cut setback, respectively.
    theta_line_samples, z_step : int, float
        Line packing integration resolutions.
    n_fine : int
        Axis interpolation samples for arc-length measurements.

    Returns
    -------
    DipoleSet
        Stored dipoles in tower-major order. Radial orientation is inward.

    Notes
    -----
    The plasma-following mode positions the *inner magnet faces* at ``gap``
    from sampled LCFS points along the local outward unit normal. Its
    distance test does not certify clearance everywhere on the continuous
    LCFS or detect oriented-box intersections.

    Inter-tower clearance is checked using a KD-tree of magnet centers,
    excluding pairs in the same tower. This is a center-distance proxy,
    not a full oriented box-intersection test. Also verify that the chosen
    surface lies outside the actual plasma before manufacturing magnets.
    """
    from scipy.interpolate import CubicSpline
    from scipy.spatial import cKDTree

    if eq is None and hasattr(axis_xyz, "compute") and hasattr(axis_xyz, "NFP"):
        eq, axis_xyz = axis_xyz, None
    if eq is not None:
        if a is not None:
            raise ValueError("a describes a circular tube; omit a when using eq")
        if target_size is None:
            raise ValueError("target_size is required")
        return _initialize_dipole_grid_from_equilibrium_lines(
                eq, target_size, L=L, layer_thick=layer_thick,
                wall_thick=wall_thick, gap=gap, magnetization=magnetization,
                m0=m0, rho=rho, ports=ports, M_surface=M_surface,
                N_surface=N_surface, d_offset=d_offset,
                seed_theta=seed_theta, L_straight=L_straight,
                theta_line_samples=theta_line_samples,
                z_step=z_step, verbose=verbose,
        )
    if a is None or target_size is None:
        raise ValueError("axis_xyz, a, and target_size required without eq")
    points = np.asarray(axis_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("axis_xyz must have shape (k,3) with k >= 4")
    if not np.all(np.isfinite(points)):
        raise ValueError("axis_xyz contains non-finite coordinates")
    if not isinstance(L, (int, np.integer)) or L < 1:
        raise ValueError("L must be a positive integer")
    if not isinstance(n_fine, (int, np.integer)) or n_fine < 16:
        raise ValueError("n_fine must be an integer >= 16")
    layer_thick = target_size if layer_thick is None else layer_thick
    if min(a, target_size, layer_thick) <= 0 or min(wall_thick, gap) < 0:
        raise ValueError("radii, footprint, and thickness must be positive; gaps nonnegative")
    if NFP < 1:
        raise ValueError("NFP must be positive")
    pitch = target_size + wall_thick
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(segments < 1e-12):
        raise ValueError("axis_xyz has duplicate consecutive samples")
    s_nodes = np.r_[0., np.cumsum(segments)]
    axis_length = s_nodes[-1]
    xyz_spline = [CubicSpline(s_nodes, points[:, i]) for i in range(3)]

    def center_at(s):
        return np.column_stack([f(s) for f in xyz_spline])

    def frame_at(s):
        pos = center_at(s)
        tangent = np.column_stack([f(s, 1) for f in xyz_spline])
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        radial = np.column_stack((
            np.cos(np.arctan2(pos[:, 1], pos[:, 0])),
            np.sin(np.arctan2(pos[:, 1], pos[:, 0])),
            np.zeros(len(pos)),
        ))
        e1 = radial - np.sum(radial * tangent, axis=1)[:, None] * tangent
        norm = np.linalg.norm(e1, axis=1)
        bad = norm < 1e-9
        if np.any(bad):
            z = np.tile([0., 0., 1.], (int(bad.sum()), 1))
            e1[bad] = z - np.sum(z * tangent[bad], axis=1)[:, None] * tangent[bad]
            norm = np.linalg.norm(e1, axis=1)
        if np.any(norm < 1e-9):
            raise ValueError("Could not construct local normal plane")
        e1 /= norm[:, None]
        return e1, np.cross(tangent, e1)

    fine_s = np.linspace(0, axis_length, n_fine)
    fine_center = center_at(fine_s)
    fine_e1, fine_e2 = frame_at(fine_s)
    # Warn/raise if the normal tube locally folds around the axis curvature.
    d1 = np.column_stack([f(fine_s, 1) for f in xyz_spline])
    d2 = np.column_stack([f(fine_s, 2) for f in xyz_spline])
    curvature = np.linalg.norm(np.cross(d1, d2), axis=1) / np.linalg.norm(d1, axis=1)**3
    if a >= 1.0 / max(float(curvature.max()), 1e-12):
        raise ValueError("a exceeds the sampled axis minimum curvature radius")

    n_poloidal = max(1, int(np.floor(2 * np.pi * a / pitch)))
    bases, directions, theta_values, s_values = [], [], [], []
    for theta in np.linspace(0, 2 * np.pi, n_poloidal, endpoint=False):
        out_f = np.cos(theta) * fine_e1 + np.sin(theta) * fine_e2
        curve = fine_center + a * out_f
        lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
        n_toroidal = max(1, int(np.floor(lengths[-1] / pitch)))
        arc_targets = (np.arange(n_toroidal) + .5) * lengths[-1] / n_toroidal
        s_samples = np.interp(arc_targets, lengths, fine_s)
        e1, e2 = frame_at(s_samples)
        outward = np.cos(theta) * e1 + np.sin(theta) * e2
        bases.extend(center_at(s_samples) + a * outward)
        directions.extend(outward)
        theta_values.extend([theta] * n_toroidal)
        s_values.extend(s_samples)

    bases = np.asarray(bases)
    directions = np.asarray(directions)
    theta_values, s_values = np.asarray(theta_values), np.asarray(s_values)
    active = np.ones(len(bases), dtype=bool)
    for port in ports:
        theta_delta = np.abs((theta_values - float(port["theta_port"]) + np.pi) % (2 * np.pi) - np.pi)
        s_delta = np.abs(s_values - float(port["s_port"]) * axis_length)
        active &= np.hypot(a * theta_delta, s_delta) > float(port["port_radius"])

    # Grow towers simultaneously; reject candidate points too close to
    # a *different* tower (not the previous layers of their own stack).
    placed, owners = [], []
    growing = active.copy()
    for layer in range(L):
        indices = np.flatnonzero(growing)
        if not len(indices):
            break
        candidates = bases[indices] + directions[indices] * (gap + (layer + .5)*layer_thick)
        acceptable = np.ones(len(indices), dtype=bool)
        if placed:
            old_points = np.asarray(placed)
            # Exact exclusion of same-tower points: search all nearby
            # candidates, not just k nearest, to avoid missing cross-tower hits.
            tree = cKDTree(old_points)
            for i, neighbors in enumerate(tree.query_ball_point(candidates, r=target_size + .5*wall_thick - 1e-10)):
                if any(owners[j] != indices[i] for j in neighbors):
                    acceptable[i] = False
        if len(indices) > 1:
            pairs = cKDTree(candidates).query_pairs(
                r=target_size + .5 * wall_thick - 1e-10, output_type="ndarray"
            )
            if len(pairs):
                acceptable[np.unique(pairs.ravel())] = False
        for i in np.flatnonzero(acceptable):
            placed.append(candidates[i])
            owners.append(int(indices[i]))
        growing[indices[~acceptable]] = False
    if not placed:
        raise ValueError("No dipoles fit; decrease target_size/wall_thick or adjust a")

    positions = np.asarray(placed)
    owners = np.asarray(owners)
    moments = magnetization * target_size**2 * layer_thick if m0 is None else float(m0)
    result = []
    for k in np.lexsort((np.arange(len(owners)), owners)):
        outward = directions[owners[k]]
        inward = -outward
        result.append(_Dipole(
            X=positions[k, 0], Y=positions[k, 1], Z=positions[k, 2],
            phi=np.arctan2(inward[1], inward[0]),
            theta=np.arccos(np.clip(inward[2], -1, 1)),
            m0=moments, rho=rho,
        ))
    return DipoleSet(*result, NFP=NFP, sym=sym)


def _initialize_dipole_grid_from_equilibrium_lines(
    eq, target_size, *, L=1, layer_thick=None, wall_thick=0.0,
    gap=0.0, magnetization=1e6, m0=None, rho=0.0,
    ports=(), M_surface=96, N_surface=128,
    d_offset=0.02, seed_theta=0.0, L_straight=0.0,
    theta_line_samples=720, z_step=0.002, verbose=False,
):
    """Pack square magnets on an outward normal-offset LCFS using continuous lines.

    Adapted from ``constant_offset_QI.py``: surface-spline normals, opposing
    geodesic-distance fronts, arc-length tower placement, half-period axis
    end-cuts, and cross-tower / symmetry-image proximity and oriented-box tests.
    Unlike sampled-node thinning, the number of candidate sites is determined
    by ``target_size + wall_thick`` and the physical vessel area, not by the
    number of equilibrium sample nodes.

    ``d_offset`` positions the vessel outside the plasma; ``gap`` separates the
    vessel from the first magnet inner face. The inward direction is ``-normal``.
    ``rho`` is the RAW optimization parameter, not normalized tanh(rho).
    Ports are omitted from the returned DipoleSet (no FOCUS Ic field).
    """
    from scipy.interpolate import CubicSpline, RectBivariateSpline
    from scipy.spatial import cKDTree
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    from desc.grid import LinearGrid

    if not eq.sym:
        raise ValueError("line packing currently requires eq.sym=True")
    if not isinstance(L, (int, np.integer)) or L < 1:
        raise ValueError("L must be a positive integer")
    if not isinstance(M_surface, (int, np.integer)) or M_surface < 8:
        raise ValueError("M_surface must be >= 8")
    if not isinstance(N_surface, (int, np.integer)) or N_surface < 8:
        raise ValueError("N_surface must be >= 8")
    layer_thick = target_size if layer_thick is None else layer_thick
    if target_size <= 0 or layer_thick <= 0 or wall_thick < 0 or gap < 0 or d_offset <= 0:
        raise ValueError("Invalid magnet footprint, thickness, or clearance")
    if L_straight < 0 or theta_line_samples < 64 or z_step <= 0:
        raise ValueError("Invalid end cut or packing resolution")
    if m0 is None and magnetization <= 0:
        raise ValueError("magnetization must be positive")
    NFP = eq.NFP
    footprint = float(target_size)
    L_requested = int(L)
    # Sample one half-period, with periodic theta but inclusive zeta end planes.
    th_nodes = np.linspace(0, 2*np.pi, int(M_surface), endpoint=False)
    ze_nodes = np.linspace(0, np.pi/NFP, int(N_surface), endpoint=True)
    surf_grid = LinearGrid(rho=[1.0], theta=th_nodes, zeta=ze_nodes,
                           NFP=NFP, sym=False)
    surf_data = eq.compute(["x"], grid=surf_grid, basis="xyz")
    surf_pts = np.asarray(surf_grid.meshgrid_reshape(surf_data["x"], "rtz"))[0]
    if surf_pts.shape != (len(th_nodes), len(ze_nodes), 3):
        raise ValueError(f"Unexpected LCFS shape {surf_pts.shape}")

    # Axis serves ONLY to define the two transverse cutting planes.
    axis_grid = LinearGrid(rho=[0.0], theta=[0.0],
                           zeta=np.linspace(0, np.pi/NFP, 100),
                           NFP=NFP, sym=False)
    axis_xyz = np.asarray(eq.compute(["x"], grid=axis_grid, basis="xyz")["x"])
    if axis_xyz.shape != (100, 3):
        raise ValueError(f"Unexpected magnetic-axis shape {axis_xyz.shape}")
    PORTS = []
    for p in ports:
        if "theta_port" not in p or "zeta_port" not in p or "port_radius" not in p:
            raise ValueError("Each port needs theta_port, zeta_port, port_radius (angles in radians)")
        if p["port_radius"] < 0:
            raise ValueError("Port radius must be nonnegative")
        PORTS.append(p)
    # The vessel is the normal offset of the actual plasma surface.
    cell_spacing   = footprint + wall_thick   # center-to-center tower spacing (pitch)
    collision_dist = footprint + wall_thick / 2.0

    # --- Spline the axis (only used for the two end cuts) ---
    seg    = np.linalg.norm(np.diff(axis_xyz, axis=0), axis=1)
    s_raw  = np.concatenate([[0.0], np.cumsum(seg)])
    L_axis = s_raw[-1]

    csx = CubicSpline(s_raw, axis_xyz[:, 0])
    csy = CubicSpline(s_raw, axis_xyz[:, 1])
    csz = CubicSpline(s_raw, axis_xyz[:, 2])
    d1x, d1y, d1z = csx.derivative(1), csy.derivative(1), csz.derivative(1)

    # --- Stop the magnets L_straight (in x) back from the end of the axis ---
    x_end = axis_xyz[-1, 0]
    s_chk = np.linspace(0.0, L_axis, 20000)
    dx    = np.abs(csx(s_chk) - x_end)
    if L_straight > 0:
        far = np.where(dx >= L_straight)[0]
        if len(far) == 0:
            raise ValueError(f"ERROR: L_straight ({L_straight:.4f} m) is longer than the axis's x extent.")
        i = far[-1]
        s_cut = np.interp(L_straight, [dx[i + 1], dx[i]], [s_chk[i + 1], s_chk[i]])
    else:
        s_cut = L_axis


    def axis_plane(s):
        """Point and unit tangent of the axis at s: the plane through them,
        perpendicular to the axis, is a domain end."""
        c = np.array([csx(s), csy(s), csz(s)])
        T = np.array([d1x(s), d1y(s), d1z(s)])
        return c, T / np.linalg.norm(T)


    c0, T0 = axis_plane(0.0)       # nose: symmetry plane
    c1, T1 = axis_plane(s_cut)     # L_straight cut (a symmetry plane only if L_straight = 0)


    def in_domain(P):
        """f0 >= 0 and f1 <= 0 inside the domain (between the two end planes)."""
        return (P - c0) @ T0, (P - c1) @ T1


    # =============================================================================
    # PLASMA SURFACE AND THE OFFSET (VACUUM VESSEL) SURFACE
    # =============================================================================

    surf_pts = np.asarray(surf_pts, dtype=float)
    if surf_pts.ndim != 3 or surf_pts.shape[2] != 3:
        raise ValueError(f"ERROR: surf_pts must have shape (n_theta, n_zeta, 3), got {surf_pts.shape}.")
    n_th, n_ze = surf_pts.shape[:2]


    def stell_flip(p):
        """Stellarator-symmetry map (x, y, z) -> (x, -y, -z)."""
        return p * np.array([1.0, -1.0, -1.0])


    def rot_z(p, ang):
        c, s = np.cos(ang), np.sin(ang)
        return np.stack([c * p[..., 0] - s * p[..., 1],
                         s * p[..., 0] + c * p[..., 1],
                         p[..., 2]], axis=-1)


    # --- Check the (theta, zeta) layout with stellarator symmetry ---
    jm      = (-np.arange(n_th)) % n_th                     # index of -theta
    size    = np.ptp(surf_pts[:, 0, :], axis=0).max()
    err_0   = np.abs(surf_pts[jm, 0]  - stell_flip(surf_pts[:, 0])).max()
    err_end = np.abs(surf_pts[jm, -1] - rot_z(stell_flip(surf_pts[:, -1]), 2 * np.pi / NFP)).max()
    if max(err_0, err_end) > 1e-3 * size:
        raise ValueError(f"ERROR: surf_pts does not look like (n_theta, n_zeta, 3) over zeta = 0 .. pi/NFP "
                 f"(symmetry error {max(err_0, err_end)*1000:.3f} mm). Check the reshape.")

    # --- Rebuild the surface over zeta = -pi/NFP .. 2*pi/NFP from the half period ---
    zeta_half   = np.linspace(0.0, np.pi / NFP, n_ze)
    period      = np.concatenate([stell_flip(surf_pts[jm, n_ze - 2:0:-1]), surf_pts], axis=1)
    zeta_period = np.concatenate([-zeta_half[n_ze - 2:0:-1], zeta_half])
    surf_win    = np.concatenate([period, rot_z(period, 2 * np.pi / NFP)], axis=1)
    zeta_win    = np.concatenate([zeta_period, zeta_period + 2 * np.pi / NFP])

    # Periodic in theta: pad a few columns on each side, then fit cubic splines.
    th_grid = 2 * np.pi * np.arange(n_th) / n_th
    PAD     = 4
    th_pad  = np.concatenate([th_grid[-PAD:] - 2 * np.pi, th_grid, th_grid[:PAD] + 2 * np.pi])
    sp      = [RectBivariateSpline(th_pad, zeta_win,
                                   np.concatenate([surf_win[-PAD:, :, i], surf_win[:, :, i],
                                                   surf_win[:PAD, :, i]], axis=0))
               for i in range(3)]


    def _ev(th, ze, dth, dze):
        th = np.mod(th, 2 * np.pi)
        return np.stack([f.ev(th, ze, dx=dth, dy=dze) for f in sp], axis=-1)


    # Outward orientation of X_theta x X_zeta (decided once, from the given surface).
    _X  = _ev(np.repeat(th_grid, 5), np.tile(np.linspace(0.1, 1.4, 5) / NFP, n_th), 0, 0)
    _n  = np.cross(_ev(np.repeat(th_grid, 5), np.tile(np.linspace(0.1, 1.4, 5) / NFP, n_th), 1, 0),
                   _ev(np.repeat(th_grid, 5), np.tile(np.linspace(0.1, 1.4, 5) / NFP, n_th), 0, 1))
    _cen = _X.reshape(n_th, 5, 3).mean(axis=0)
    N_SIGN = 1.0 if np.sum(_n * (_X - np.tile(_cen, (n_th, 1)))) > 0 else -1.0


    def vessel(th, ze):
        """Vacuum-vessel (offset) surface at plasma coordinates (theta, zeta).

        Returns P (point), P_th, P_ze (tangent vectors) and n (outward unit normal,
        the same for the plasma surface and the offset surface)."""
        X,   Xt  = _ev(th, ze, 0, 0), _ev(th, ze, 1, 0)
        Xz,  Xtt = _ev(th, ze, 0, 1), _ev(th, ze, 2, 0)
        Xtz, Xzz = _ev(th, ze, 1, 1), _ev(th, ze, 0, 2)
        nv  = N_SIGN * np.cross(Xt, Xz)
        nm  = np.linalg.norm(nv, axis=-1, keepdims=True)
        n   = nv / nm
        nt  = N_SIGN * (np.cross(Xtt, Xz) + np.cross(Xt, Xtz))
        nz  = N_SIGN * (np.cross(Xtz, Xz) + np.cross(Xt, Xzz))
        n_t = (nt - n * np.sum(n * nt, axis=-1, keepdims=True)) / nm
        n_z = (nz - n * np.sum(n * nz, axis=-1, keepdims=True)) / nm
        return X + d_offset * n, Xt + d_offset * n_t, Xz + d_offset * n_z, n


    # =============================================================================
    # WHERE THE DOMAIN SITS IN ZETA, AND THE OFFSET-FOLD CHECK
    # =============================================================================

    N_T = int(theta_line_samples)                                            # theta samples per cross-line
    th_f = seed_theta + np.linspace(0.0, 2 * np.pi, N_T + 1)
    ze_c = np.linspace(-np.pi / NFP, 2 * np.pi / NFP, 600)
    Pc, Ptc, Pzc, nc = vessel(np.repeat(th_f[:-1:4], len(ze_c)), np.tile(ze_c, N_T // 4))
    f0, f1 = in_domain(Pc)
    inside = ((f0 >= 0) & (f1 <= 0)).reshape(N_T // 4, len(ze_c))
    # keep only the patch connected to the given half period (planes also cut far-away parts)
    cols = np.where(inside.any(axis=0) & (ze_c > -0.5 * np.pi / NFP) & (ze_c < 1.5 * np.pi / NFP))[0]
    if len(cols) == 0:
        raise ValueError("ERROR: the end planes do not enclose any of the vessel surface.")
    ze_lo = ze_c[max(cols.min() - 2, 0)]
    ze_hi = ze_c[min(cols.max() + 2, len(ze_c) - 1)]
    if ze_lo <= ze_c[0] or ze_hi >= ze_c[-1]:
        raise ValueError("ERROR: the domain reaches the edge of the rebuilt surface window.")

    jac = N_SIGN * np.sum(np.cross(Ptc, Pzc) * nc, axis=-1)
    patch = ((f0 >= 0) & (f1 <= 0)).reshape(N_T // 4, len(ze_c)) & \
            ((ze_c >= ze_lo) & (ze_c <= ze_hi))[None, :]
    patch = patch.ravel()
    if np.any(jac[patch] <= 0):
        raise ValueError(f"ERROR: the vessel surface folds over itself at offset d_offset = {d_offset} m "
                 f"(the plasma is more concave than d_offset somewhere). Reduce d_offset.")


    # =============================================================================
    # LINES OF SQUARES
    # =============================================================================
    # Every line follows the vessel surface the long way around. The SEED line is
    # the constant-theta line theta = seed_theta. Lines are added outward from it in
    # both directions, each at least one pitch (perpendicular, measured on the
    # surface) from the previous one. The two fronts meet on the far side; there a
    # line only exists where it still fits, so lines start and stop individually
    # exactly where room appears or disappears.
    #
    # At each zeta the cross-line coordinate w (distance across constant-theta
    # lines) is integrated around the vessel from the seed. Lines are placed at
    # chosen w; because a line can tilt relative to the constant-theta lines, the
    # true perpendicular spacing is then measured in 3D and the w-gaps are widened
    # wherever it falls below one pitch (repeated until it does not).

    ZE_STEP = float(z_step)                                      # m along the vessel, approx.
    R_typ   = np.linalg.norm(Pc[:, :2], axis=-1).max()
    N_Z     = int(np.ceil((ze_hi - ze_lo) * R_typ / ZE_STEP)) + 1
    ze_l    = np.linspace(ze_lo, ze_hi, N_Z)

    # w(theta) at every zeta of the line grid
    TH = np.tile(th_f, (N_Z, 1))
    ZE = np.repeat(ze_l[:, None], N_T + 1, axis=1)
    _, Pt, Pz, _ = vessel(TH.ravel(), ZE.ravel())
    Pt, Pz = Pt.reshape(N_Z, N_T + 1, 3), Pz.reshape(N_Z, N_T + 1, 3)
    h  = np.linalg.norm(np.cross(Pt, Pz), axis=-1) / np.linalg.norm(Pz, axis=-1)
    w  = np.concatenate([np.zeros((N_Z, 1)),
                         np.cumsum(0.5 * (h[:, 1:] + h[:, :-1]) * np.diff(th_f)[None, :], axis=1)], axis=1)
    W  = w[:, -1]                                        # once around the vessel

    K = int(np.ceil(W.max() / (2 * cell_spacing))) + 10  # candidate lines per side
    SMOOTH_N = 2 * int(np.ceil(2 * cell_spacing / ZE_STEP)) + 1   # ~3 cm along the line


    def line_points(wpos):
        """(theta, P) of lines at cross-line positions wpos (K, N_Z)."""
        th = np.empty_like(wpos)
        for j in range(N_Z):
            th[:, j] = np.interp(np.clip(wpos[:, j], 0.0, W[j]), w[j], th_f)
        P, _, _, _ = vessel(th.ravel(), np.tile(ze_l, wpos.shape[0]))
        return th, P.reshape(wpos.shape[0], N_Z, 3)


    def dist_to_line(Q, R, m=8):
        """Distance from each point Q[k, j] to the polyline R[k, :] (only the
        segments around index j are searched)."""
        best = np.full(Q.shape[:2], np.inf)
        J = Q.shape[1]
        for o in range(-m, m):
            ia = np.clip(np.arange(J) + o, 0, J - 1)
            ib = np.clip(np.arange(J) + o + 1, 0, J - 1)
            a, b = R[:, ia], R[:, ib]
            ab = b - a
            t  = np.clip(np.sum((Q - a) * ab, axis=-1) / np.maximum(np.sum(ab * ab, axis=-1), 1e-30), 0, 1)
            best = np.minimum(best, np.linalg.norm(Q - a - t[..., None] * ab, axis=-1))
        return best


    _, seed_P = line_points(np.zeros((1, N_Z)))
    gaps = {+1: np.full((K, N_Z), cell_spacing), -1: np.full((K, N_Z), cell_spacing)}
    for it in range(30):
        worst = 0.0
        lines = {}
        for side in (+1, -1):
            pos = np.cumsum(gaps[side], axis=0)
            wpos = pos if side > 0 else W[None, :] - pos
            th_k, P_k = line_points(wpos)
            inner = np.concatenate([seed_P, P_k[:-1]], axis=0)
            D = dist_to_line(P_k, inner)
            ok = pos < 0.6 * W[None, :]                   # past ~halfway the other front is there first
            fac = np.where(ok, np.maximum(1.0, cell_spacing * (1 + 1e-4) / np.maximum(D, 1e-9)), 1.0)
            # widen smoothly along each line (never less than needed): running max, then running mean
            fac = uniform_filter1d(maximum_filter1d(fac, SMOOTH_N, axis=1, mode='nearest'),
                                   SMOOTH_N // 2 + 1, axis=1, mode='nearest')
            worst = max(worst, fac.max())
            gaps[side] *= fac
            lines[side] = (pos, th_k, P_k)
        if worst <= 1 + 2e-4:
            break
    else:
        raise ValueError("ERROR: the line spacing did not converge.")

    # --- Where each line exists: alternate sides until the two fronts meet ---
    exists = {+1: np.zeros((K, N_Z), bool), -1: np.zeros((K, N_Z), bool)}
    for j in range(N_Z):
        pos_p, pos_m = lines[+1][0][:, j], lines[-1][0][:, j]
        npl = nmi = 0
        while True:
            added = False
            if npl < K:
                b_out = W[j] - (pos_m[nmi - 1] if nmi else 0.0)
                if b_out - pos_p[npl] >= gaps[+1][npl, j]:
                    npl += 1; added = True
            if nmi < K:
                a_out = pos_p[npl - 1] if npl else 0.0
                if (W[j] - pos_m[nmi]) - a_out >= gaps[-1][nmi, j]:
                    nmi += 1; added = True
            if not added:
                break
        exists[+1][:npl, j] = True
        exists[-1][:nmi, j] = True

    # Check the meeting gap in 3D and drop lines that come too close.
    jj = np.clip(np.arange(N_Z)[:, None] + np.arange(-8, 9)[None, :], 0, N_Z - 1)
    for _ in range(10):
        npl = exists[+1].sum(axis=0); nmi = exists[-1].sum(axis=0)
        kp, km = np.maximum(npl - 1, 0), np.maximum(nmi - 1, 0)
        Pp = np.where((npl > 0)[:, None], lines[+1][2][kp, np.arange(N_Z)], seed_P[0])
        Rm = np.where((nmi > 0)[:, None, None], lines[-1][2][km[:, None], jj], seed_P[0][jj])
        a, ab = Rm[:, :-1], Rm[:, 1:] - Rm[:, :-1]
        t = np.clip(np.sum((Pp[:, None] - a) * ab, axis=-1) / np.maximum(np.sum(ab * ab, axis=-1), 1e-30), 0, 1)
        D = np.linalg.norm(Pp[:, None] - a - t[..., None] * ab, axis=-1).min(axis=1)
        bad = (D < cell_spacing) & ((npl > 0) | (nmi > 0))
        if not bad.any():
            break
        for j in np.where(bad)[0]:
            if npl[j] >= nmi[j]:
                exists[+1][npl[j] - 1, j] = False
            else:
                exists[-1][nmi[j] - 1, j] = False

    # Drop line pieces too short to hold a tower without flicker.
    MIN_RUN = int(np.ceil(3 * cell_spacing / ZE_STEP))
    for side in (+1, -1):
        for k in range(K):
            e = exists[side][k]
            edges = np.flatnonzero(np.diff(np.concatenate([[0], e.astype(int), [0]])))
            for a, b in zip(edges[::2], edges[1::2]):
                if b - a < MIN_RUN:
                    e[a:b] = False

    lines_all = [(seed_P[0], np.full(N_Z, seed_theta), np.ones(N_Z, bool))]
    for side in (+1, -1):
        for k in range(K):
            if exists[side][k].any():
                lines_all.append((lines[side][2][k], lines[side][1][k], exists[side][k]))


    # =============================================================================
    # TOWERS ALONG THE LINES
    # =============================================================================

    bases, outs, tans, th_all, ze_all, line_of = [], [], [], [], [], []
    for li, (Pl, thl, ex) in enumerate(lines_all):
        g0, g1 = in_domain(Pl)
        keep = ex & (g0 >= 0) & (g1 <= 0)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], keep.astype(int), [0]])))
        for a, b in zip(edges[::2], edges[1::2]):
            # polyline in index space; extend to the exact plane crossings at the domain ends
            idx = np.arange(a, b, dtype=float)
            for end, nb in ((a, a - 1), (b - 1, b)):
                if 0 <= nb < N_Z and ex[nb]:
                    for g in (g0, -g1):
                        if g[end] >= 0 > g[nb]:
                            f = g[end] / (g[end] - g[nb])
                            idx = np.append(idx, end + f * (nb - end))
            idx = np.sort(idx)
            P_run  = np.stack([np.interp(idx, np.arange(N_Z), Pl[:, i]) for i in range(3)], axis=1)
            th_run = np.interp(idx, np.arange(N_Z), np.unwrap(thl))
            ze_run = np.interp(idx, np.arange(N_Z), ze_l)
            cum    = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P_run, axis=0), axis=1))])
            N = int(np.floor(cum[-1] / cell_spacing))
            if N < 1:
                continue
            tgt = (np.arange(N) + 0.5) * (cum[-1] / N)       # half-step from both ends
            th_t, ze_t = np.interp(tgt, cum, th_run), np.interp(tgt, cum, ze_run)
            P_t, _, _, n_t = vessel(th_t, ze_t)
            seg_i = np.clip(np.searchsorted(cum, tgt) - 1, 0, len(cum) - 2)
            t_dir = P_run[seg_i + 1] - P_run[seg_i]
            t_dir = t_dir - n_t * np.sum(t_dir * n_t, axis=1, keepdims=True)
            t_dir /= np.linalg.norm(t_dir, axis=1, keepdims=True)
            bases += list(P_t); outs += list(n_t); tans += list(t_dir)
            th_all += list(th_t); ze_all += list(ze_t); line_of += [li] * N

    bases, outs, tans = np.asarray(bases), np.asarray(outs), np.asarray(tans)
    th_all, ze_all, line_of = np.asarray(th_all), np.asarray(ze_all), np.asarray(line_of)
    T = len(bases)
    if T == 0:
        raise ValueError("ERROR: no towers fit in the domain.")


    # =============================================================================
    # PORTS
    # =============================================================================

    port_centers = (vessel(np.array([p["theta_port"] for p in PORTS]),
                           np.array([p["zeta_port"] for p in PORTS]))[0] if PORTS else np.zeros((0, 3)))
    port_radius  = np.array([p["port_radius"] for p in PORTS])
    if PORTS:
        port_blocked = (np.linalg.norm(bases[:, None, :] - port_centers[None], axis=-1)
                        <= port_radius[None, :]).any(axis=1)
    else:
        port_blocked = np.zeros(T, dtype=bool)


    # =============================================================================
    # SYMMETRY IMAGES
    # =============================================================================
    # The nose plane is a stellarator-symmetry plane: the next half period is the
    # flipped copy. The far end is one too only when L_straight = 0.

    IMAGE_OPS = [lambda p: stell_flip(p)]
    if L_straight == 0:
        IMAGE_OPS.append(lambda p: rot_z(stell_flip(p), 2 * np.pi / NFP))


    def images_of(pts):
        if len(pts) == 0:
            return np.zeros((0, 3))
        return np.vstack([op(pts) for op in IMAGE_OPS])


    # =============================================================================
    # SQUARE-MAGNET BOX TEST
    # =============================================================================
    # Each magnet is a footprint x footprint x layer_thick box: one side along its
    # line, the other across it, the thickness along the surface normal. Boxes are
    # grown by wall/4 on each side across the surface, so two boxes "overlap" when
    # the wall between them would be thinner than wall/2 -- the same allowance as
    # collision_dist. Separating-axis test for pairs of boxes.

    BOX_HALF = np.array([footprint / 2 + wall_thick / 4, footprint / 2 + wall_thick / 4, layer_thick / 2])
    BOX_R    = 2 * np.linalg.norm(BOX_HALF)          # boxes farther apart than this cannot touch


    def box_frames(t):
        n = outs[t]; a = tans[t]
        return np.stack([a, np.cross(n, a), n], axis=1)          # (.., 3 axes, 3)


    def boxes_overlap(cA, FA, cB, FB):
        """True where box A (centre cA, axes FA) and box B overlap."""
        D = cB - cA
        sep = np.zeros(len(D), dtype=bool)
        axes = [FA[:, i] for i in range(3)] + [FB[:, i] for i in range(3)] + \
               [np.cross(FA[:, i], FB[:, j]) for i in range(3) for j in range(3)]
        for L in axes:
            ra = sum(BOX_HALF[i] * np.abs(np.sum(FA[:, i] * L, axis=1)) for i in range(3))
            rb = sum(BOX_HALF[i] * np.abs(np.sum(FB[:, i] * L, axis=1)) for i in range(3))
            sep |= np.abs(np.sum(D * L, axis=1)) > ra + rb + 1e-12
        return ~sep


    def image_frames(F, k):
        """Box axes of the k-th symmetry image (the symmetry maps are linear)."""
        return np.stack([IMAGE_OPS[k](F[:, i]) for i in range(3)], axis=1)


    # =============================================================================
    # GRID GENERATION  (global layer-by-layer growth)
    # =============================================================================
    # Every tower is grown together, one layer at a time. Before a layer is placed,
    # we check each still-growing tower's candidate magnet against ALL magnets already
    # placed (any tower), the other candidates being placed on this same layer, and
    # the symmetry images of both. If a candidate would come within
    # (footprint + wall/2) of any of them, or its square box would overlap any of
    # their boxes, that tower is frozen -- it stops at its current height and
    # places no more layers.

    vol_per_dipole = footprint * footprint * layer_thick
    M_0            = (magnetization if m0 is None else float(m0)/vol_per_dipole) * vol_per_dipole      # max dipole moment (A*m^2)
    K_NEIGH        = int(np.ceil(collision_dist / layer_thick)) + 3

    placed_xyz, placed_tower, placed_layer = [], [], []
    L_tower = np.zeros(T, dtype=int)
    growing = np.ones(T, dtype=bool)

    for l in range(L_requested):
        if not growing.any():
            break

        idx  = np.where(growing)[0]
        cand = bases[idx] + outs[idx] * (gap + (l + 0.5) * layer_thick)
        ok   = np.ones(len(idx), dtype=bool)

        # placed magnets of OTHER towers
        if placed_xyz:
            placed_arr = np.asarray(placed_xyz)
            placed_tw  = np.asarray(placed_tower)
            kq = min(len(placed_arr), K_NEIGH)
            dq, iq = cKDTree(placed_arr).query(cand, k=kq)
            if kq == 1:
                dq = dq[:, None]; iq = iq[:, None]
            dq = np.where(placed_tw[iq] == idx[:, None], np.inf, dq)
            ok &= dq.min(axis=1) >= collision_dist

        # other towers' candidates on this layer (both members of a clash are rejected)
        if len(idx) > 1:
            pairs = cKDTree(cand).query_pairs(r=collision_dist, output_type='ndarray')
            if len(pairs) > 0:
                ok[np.unique(pairs.ravel())] = False

        # symmetry images of placed magnets and of this layer's candidates
        src = np.vstack([np.asarray(placed_xyz).reshape(-1, 3), cand])
        di, _ = cKDTree(images_of(src)).query(cand, k=1)
        ok &= di >= collision_dist

        # --- the same three checks with the actual square boxes ---
        Fc = box_frames(idx)
        if placed_xyz:
            nb = cKDTree(placed_arr).query_ball_point(cand, r=BOX_R)
            ci = np.repeat(np.arange(len(idx)), [len(v) for v in nb])
            pj = np.concatenate([np.asarray(v, dtype=int) for v in nb]) if len(ci) else np.zeros(0, int)
            m  = placed_tw[pj] != idx[ci]
            ci, pj = ci[m], pj[m]
            if len(ci):
                hit = boxes_overlap(cand[ci], Fc[ci], placed_arr[pj], box_frames(placed_tw[pj]))
                ok[np.unique(ci[hit])] = False
        if len(idx) > 1:
            pairs = cKDTree(cand).query_pairs(r=BOX_R, output_type='ndarray')
            if len(pairs) > 0:
                hit = boxes_overlap(cand[pairs[:, 0]], Fc[pairs[:, 0]], cand[pairs[:, 1]], Fc[pairs[:, 1]])
                ok[np.unique(pairs[hit].ravel())] = False
        src_tw = np.concatenate([np.asarray(placed_tower, dtype=int), idx])
        src_F  = box_frames(src_tw)
        for kop in range(len(IMAGE_OPS)):
            img = IMAGE_OPS[kop](src)
            nb  = cKDTree(img).query_ball_point(cand, r=BOX_R)
            ci  = np.repeat(np.arange(len(idx)), [len(v) for v in nb])
            if len(ci):
                pj  = np.concatenate([np.asarray(v, dtype=int) for v in nb])
                hit = boxes_overlap(cand[ci], Fc[ci], img[pj], image_frames(src_F[pj], kop))
                ok[np.unique(ci[hit])] = False

        for t in idx[ok]:
            placed_xyz.append(bases[t] + outs[t] * (gap + (l + 0.5) * layer_thick))
            placed_tower.append(int(t)); placed_layer.append(l)
            L_tower[t] = l + 1
        growing[idx[~ok]] = False

    placed_xyz   = np.asarray(placed_xyz).reshape(-1, 3)
    placed_tower = np.asarray(placed_tower, dtype=int)
    placed_layer = np.asarray(placed_layer, dtype=int)
    if len(placed_xyz) == 0:
        raise ValueError("No magnets fit; reduce footprint/wall or change offset")
    # Output only magnets from unblocked towers. Keep the same orientation across
    # the entire radial stack, just as in the FOCUS generator.
    order = np.lexsort((placed_layer, placed_tower))
    moment = (magnetization * target_size**2 * layer_thick
              if m0 is None else float(m0))
    dipoles = []
    for k in order:
        t = int(placed_tower[k])
        if port_blocked[t]:
            continue
        p = placed_xyz[k]
        inward = -outs[t]
        dipoles.append(_Dipole(
            X=float(p[0]), Y=float(p[1]), Z=float(p[2]),
            phi=float(np.arctan2(inward[1], inward[0])),
            theta=float(np.arccos(np.clip(inward[2], -1.0, 1.0))),
            m0=float(moment), rho=float(rho),
        ))
    if verbose:
        print(f"Surface-following grid: {T:,} towers, {len(dipoles):,} stored magnets")
        print(f"Layers per tower: min={L_tower.min()}, max={L_tower.max()}, mean={L_tower.mean():.2f}")
        print(f"Pitch: {cell_spacing*1000:.2f} mm; offset: {d_offset*1000:.2f} mm")
        print(f"Estimated full symmetry-expanded magnets: {len(dipoles)*2*NFP:,}")
    return DipoleSet(*dipoles, NFP=NFP, sym=True)


def initialize_axisymmetric_dipole_grid(
    eq, target_size, L=1, *, a=None, clearance=0.0,
    layer_thick=None, wall_thick=0.0, gap=0.0,
    magnetization=1.0e6, m0=None, rho=0.0,
    n_axis=64, M_surface=32, N_surface=32,
):
    """Build a circular-axis (zero-torsion) dipole grid around a DESC equilibrium.

    Estimate the circular centerline using the magnetic-axis mean major radius
    and mean Z. Estimate a conservative required minor radius as the largest
    distance of LCFS R,Z samples to that circle's (R0,Z0) center. Set the
    tube radius to that bound plus ``clearance``, or the user-provided ``a``
    if it is *at least* as large. The resulting centerline is a planar circle
    with torsion zero, not a nonaxisymmetric approximation to the actual axis.

    This guarantees separation at the sampled LCFS nodes in R,Z, subject to
    positive gap/layer thickness. It is not a continuous-surface clearance
    certificate between sample points, nor a box collision certificate.
    """
    from desc.grid import LinearGrid

    if not isinstance(n_axis, (int, np.integer)) or n_axis < 4:
        raise ValueError("n_axis must be an integer >= 4")
    if clearance < 0:
        raise ValueError("clearance must be nonnegative")
    # Use axis major-radius and vertical centers, averaged over a full turn.
    axis_grid = LinearGrid(rho=[0.0], M=0, N=n_axis, NFP=eq.NFP, sym=False)
    axis_data = eq.compute(["R", "Z"], grid=axis_grid)
    R0 = float(np.mean(np.asarray(axis_data["R"])))
    Z0 = float(np.mean(np.asarray(axis_data["Z"])))
    surf_grid = LinearGrid(rho=[1.0], M=M_surface, N=N_surface, NFP=eq.NFP, sym=False)
    surf_data = eq.compute(["R", "Z"], grid=surf_grid)
    Rp = np.asarray(surf_data["R"])
    Zp = np.asarray(surf_data["Z"])
    a_required = float(np.max(np.hypot(Rp - R0, Zp - Z0))) + clearance
    a = a_required if a is None else float(a)
    if a < a_required:
        raise ValueError(f"Requested radius a={a:g} is inside sampled plasma bound {a_required:g}")
    extent = np.pi / eq.NFP if eq.sym else 2 * np.pi / eq.NFP
    angles = np.linspace(0, extent, n_axis)
    axis_xyz = np.column_stack((R0 * np.cos(angles), R0 * np.sin(angles),
                                np.full(n_axis, Z0)))
    return initialize_dipole_grid(
        axis_xyz, a, target_size, L=L, layer_thick=layer_thick,
        wall_thick=wall_thick, gap=gap, magnetization=magnetization,
        m0=m0, rho=rho, NFP=eq.NFP, sym=eq.sym,
    )
