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
        R = rotation_matrix(axis=axis, angle=angle)
        
        pos = jnp.array([self._X, self._Y, self._Z])
        new_pos = R @ pos
        self._X = new_pos[0]
        self._Y = new_pos[1]
        self._Z = new_pos[2]
        
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
        
        pos = jnp.array([self._X, self._Y, self._Z])
        new_pos = F @ pos
        self._X = new_pos[0]
        self._Y = new_pos[1]
        self._Z = new_pos[2]
        
        m_vec = self.m_xyz
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
            Rz = rotation_matrix(axis=[0, 0, 1], angle=2 * jnp.pi * k / self.NFP)
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