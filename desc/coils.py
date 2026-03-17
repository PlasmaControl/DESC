"""Classes for magnetic field coils."""

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
    biot_savart_general,
    biot_savart_general_vector_potential,
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
def biot_savart_hh(eval_pts, coil_pts_start, coil_pts_end, current, *, chunk_size=None):
    """Biot-Savart law for filamentary coils following [1].

    The coil is approximated by a series of straight line segments
    and an analytic expression is used to evaluate the field from each
    segment.

    References
    ----------
    [1] Hanson & Hirshman, "Compact expressions for the Biot-Savart
        fields of a filamentary segment" (2002)

    Parameters
    ----------
    eval_pts : array-like shape(n,3)
        Evaluation points in cartesian coordinates
    coil_pts_start, coil_pts_end : array-like shape(m,3)
        Points in cartesian space defining the start and end of each segment.
        Should be a closed curve, such that coil_pts_start[0] == coil_pts_end[-1]
        though this is not checked.
    current : float
        Current through the coil (in Amps).
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
    d_vec = coil_pts_end - coil_pts_start
    L = jnp.linalg.norm(d_vec, axis=-1)

    Ri_vec = eval_pts[jnp.newaxis, :] - coil_pts_start[:, jnp.newaxis, :]
    Ri = jnp.linalg.norm(Ri_vec, axis=-1)
    Rf = jnp.linalg.norm(
        eval_pts[jnp.newaxis, :] - coil_pts_end[:, jnp.newaxis, :], axis=-1
    )
    Ri_p_Rf = Ri + Rf

    B_mag = (
        mu_0
        / (2 * jnp.pi)
        * current
        * Ri_p_Rf
        / (Ri * Rf * (Ri_p_Rf * Ri_p_Rf - (L * L)[:, jnp.newaxis]))
    )

    # cross product of L*hat(eps)==d_vec with Ri_vec, scaled by B_mag
    vec = jnp.cross(d_vec[:, jnp.newaxis, :], Ri_vec, axis=-1)
    B = jnp.sum(B_mag[:, :, jnp.newaxis] * vec, axis=0)
    return B


@partial(jit, static_argnames=["chunk_size"])
def biot_savart_vector_potential_hh(
    eval_pts, coil_pts_start, coil_pts_end, current, *, chunk_size=None
):
    """Biot-Savart law for vector potential for filamentary coils following [1].

    The coil is approximated by a series of straight line segments
    and an analytic expression is used to evaluate the vector potential from each
    segment. This expression assumes the Coulomb gauge.

    References
    ----------
    [1] Hanson & Hirshman, "Compact expressions for the Biot-Savart
        fields of a filamentary segment" (2002)

    Parameters
    ----------
    eval_pts : array-like shape(n,3)
        Evaluation points in cartesian coordinates
    coil_pts_start, coil_pts_end : array-like shape(m,3)
        Points in cartesian space defining the start and end of each segment.
        Should be a closed curve, such that coil_pts_start[0] == coil_pts_end[-1]
        though this is not checked.
    current : float
        Current through the coil (in Amps).
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
    d_vec = coil_pts_end - coil_pts_start
    L = jnp.linalg.norm(d_vec, axis=-1)
    d_vec_over_L = ((1 / L) * d_vec.T).T

    Ri_vec = eval_pts[jnp.newaxis, :] - coil_pts_start[:, jnp.newaxis, :]
    Ri = jnp.linalg.norm(Ri_vec, axis=-1)
    Rf = jnp.linalg.norm(
        eval_pts[jnp.newaxis, :] - coil_pts_end[:, jnp.newaxis, :], axis=-1
    )
    Ri_p_Rf = Ri + Rf

    eps = L[:, jnp.newaxis] / (Ri_p_Rf)

    A_mag = mu_0 / (4 * jnp.pi) * current * jnp.log((1 + eps) / (1 - eps))

    # Now just need  to multiply by e^ = d_vec/L = (x_f - x_i)/L
    A = jnp.sum(A_mag[:, :, jnp.newaxis] * d_vec_over_L[:, jnp.newaxis, :], axis=0)
    return A


@partial(jit, static_argnames=["chunk_size"])
def biot_savart_quad(eval_pts, coil_pts, tangents, current, *, chunk_size=None):
    """Biot-Savart law for filamentary coil using numerical quadrature.

    Notes
    -----
    This method does not give curl(B) == 0 exactly. The error in the curl
    scales the same as the error in B itself, so will only be zero when fully
    converged. However in practice, for smooth curves described by Fourier series,
    this method converges exponentially in the number of coil points.

    Parameters
    ----------
    eval_pts : array-like
        Shape (n, 3).
        Evaluation points in cartesian coordinates.
    coil_pts : array-like
        Shape (m, 3).
        Points in cartesian space defining coil.
    tangents : array-like
        Shape (m, 3).
        Tangent vectors to the coil at coil_pts. If the curve is given
        by x(s) with curve parameter s, coil_pts = x, tangents = dx/ds*ds where
        ds is the spacing between points.
    current : float
        Current through the coil (in Amps).
    chunk_size : int or None
        Size to split computation into chunks of evaluation points.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``None``.

    Returns
    -------
    B : ndarray
        Shape(n, 3).
        Magnetic field in cartesian components at specified points.

    """
    return biot_savart_general(
        eval_pts, coil_pts, current * tangents, chunk_size=chunk_size
    )


@partial(jit, static_argnames=["chunk_size"])
def biot_savart_vector_potential_quad(
    eval_pts, coil_pts, tangents, current, *, chunk_size=None
):
    """Biot-Savart law (for A) for filamentary coil using numerical quadrature.

    This expression assumes the Coulomb gauge.

    Parameters
    ----------
    eval_pts : array-like
        Shape (n, 3).
        Evaluation points in cartesian coordinates.
    coil_pts : array-like
        Shape (m, 3).
        Points in cartesian space defining coil.
    tangents : array-like
        Shape (m, 3).
        Tangent vectors to the coil at coil_pts. If the curve is given
        by x(s) with curve parameter s, coil_pts = x, tangents = dx/ds*ds where
        ds is the spacing between points.
    current : float
        Current through the coil (in Amps).
    chunk_size : int or None
        Size to split computation into chunks of evaluation points.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``None``.

    Returns
    -------
    A : ndarray
        Shape (n, 3).
        Magnetic vector potential in cartesian components at specified points.

    """
    return biot_savart_general_vector_potential(
        eval_pts, coil_pts, current * tangents, chunk_size=chunk_size
    )


class _Coil(_MagneticField, Optimizable, ABC):
    """Base class representing a magnetic field coil.

    Represents coils as a combination of a Curve and current

    Subclasses for a particular parameterization of a coil should inherit
    from Coil and the appropriate Curve type, eg MyCoil(Coil, MyCurve)
    - note that Coil must be the first parent for correct inheritance.

    Subclasses based on curves that follow the Curve API should only have
    to implement a new __init__ method, all others will be handled by default

    Parameters
    ----------
    current : float
        Current through the coil, in Amperes.
    """

    _io_attrs_ = _MagneticField._io_attrs_ + ["_current"]
    _static_attrs = _MagneticField._static_attrs + Optimizable._static_attrs

    def __init__(self, current, *args, **kwargs):
        self._current = jnp.float64(float(np.squeeze(current)))
        super().__init__(*args, **kwargs)

    def _set_up(self):
        for attribute in self._io_attrs_:
            if not hasattr(self, attribute):
                setattr(self, attribute, None)

    @optimizable_parameter
    @property
    def current(self):
        """float: Current passing through the coil, in Amperes."""
        return self._current

    @current.setter
    def current(self, new):
        assert jnp.isscalar(new) or new.size == 1
        self._current = jnp.float64(float(np.squeeze(new)))

    @property
    def num_coils(self):
        """int: Number of coils."""
        return 1

    def _compute_position(self, params=None, grid=None, dx1=False, **kwargs):
        """Compute coil positions accounting for stellarator symmetry.

        Parameters
        ----------
        params : dict or array-like of dict, optional
            Parameters to pass to coils, either the same for all coils or one for each.
        grid : Grid or int, optional
            Grid of coordinates to evaluate at. Defaults to a Linear grid.
            If an integer, uses that many equally spaced points.
        dx1 : bool
            If True, also return dx/ds for the curve.

        Returns
        -------
        x : ndarray, shape(len(self),source_grid.num_nodes,3)
            Coil positions, in [R,phi,Z] or [X,Y,Z] coordinates.
        x_s : ndarray, shape(len(self),source_grid.num_nodes,3)
            Coil position derivatives, in [R,phi,Z] or [X,Y,Z] coordinates.
            Only returned if dx1=True.

        """
        kwargs.setdefault("basis", "xyz")
        keys = ["x", "x_s"] if dx1 else ["x"]
        data = self.compute(keys, grid=grid, params=params, **kwargs)
        x = jnp.transpose(jnp.atleast_3d(data["x"]), [2, 0, 1])  # shape=(1,num_nodes,3)
        if dx1:
            x_s = jnp.transpose(
                jnp.atleast_3d(data["x_s"]), [2, 0, 1]
            )  # shape=(1,num_nodes,3)
        basis = kwargs.get("basis", "xyz")
        if basis.lower() == "rpz":
            x = x.at[:, :, 1].set(jnp.mod(x[:, :, 1], 2 * jnp.pi))
        if dx1:
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
        """Compute magnetic field or vector potential at a set of points.

        The coil current may be overridden by including `current`
        in the `params` dictionary.

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

        Notes
        -----
        Uses direct quadrature of the Biot-Savart integral for filamentary coils with
        tangents provided by the underlying curve class. Convergence should be
        exponential in the number of points used to discretize the curve, though curl(B)
        may not be zero if not fully converged.

        """
        errorif(
            compute_A_or_B not in ["A", "B"],
            ValueError,
            f'Expected "A" or "B" for compute_A_or_B, instead got {compute_A_or_B}',
        )
        op = {"B": biot_savart_quad, "A": biot_savart_vector_potential_quad}[
            compute_A_or_B
        ]
        assert basis.lower() in ["rpz", "xyz"]
        coords = jnp.atleast_2d(jnp.asarray(coords))
        if basis.lower() == "rpz":
            phi = coords[:, 1]
            coords = rpz2xyz(coords)
        if params is None:
            current = self.current
        else:
            current = params.pop("current", self.current)
        NFP = getattr(self, "NFP", 1)
        if source_grid is None:
            # NFP=1 to ensure points span the entire length of the coil
            # multiply resolution by NFP to ensure Biot-Savart integration is accurate
            source_grid = LinearGrid(N=2 * self.N * NFP + 5)
        else:
            # coil grids should have NFP=1. The only possible exception is FourierRZCoil
            # which in theory can be different as long as it matches the coils NFP.
            errorif(
                getattr(source_grid, "NFP", 1) not in [1, NFP],
                ValueError,
                f"source_grid for coils must have NFP=1 or NFP={NFP}",
            )

        if not params or not transforms:
            data = self.compute(
                ["x", "x_s", "ds"],
                grid=source_grid,
                params=params,
                transforms=transforms,
                basis="xyz",
            )
        else:
            data = compute_fun(
                self,
                names=["x", "x_s", "ds"],
                params=params,
                transforms=transforms,
                profiles={},
            )
            data["x_s"] = rpz2xyz_vec(data["x_s"], phi=data["x"][:, 1])
            data["x"] = rpz2xyz(data["x"])

        AB = op(
            coords,
            data["x"],
            data["x_s"] * data["ds"][:, None],
            current,
            chunk_size=chunk_size,
        )

        if basis.lower() == "rpz":
            AB = xyz2rpz_vec(AB, phi=phi)
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

        The coil current may be overridden by including `current`
        in the `params` dictionary.

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
        field : ndarray, shape(n,3)
            magnetic field at specified points, in either rpz or xyz coordinates

        Notes
        -----
        Uses direct quadrature of the Biot-Savart integral for filamentary coils with
        tangents provided by the underlying curve class. Convergence should be
        exponential in the number of points used to discretize the curve, though curl(B)
        may not be zero if not fully converged.

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

        The coil current may be overridden by including `current`
        in the `params` dictionary.

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

        Notes
        -----
        Uses direct quadrature of the Biot-Savart integral for filamentary coils with
        tangents provided by the underlying curve class. Convergence should be
        exponential in the number of points used to discretize the curve, though curl(B)
        may not be zero if not fully converged.

        """
        return self._compute_A_or_B(
            coords, params, basis, source_grid, transforms, "A", chunk_size=chunk_size
        )

    def __repr__(self):
        """Get the string form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (name={}, current={})".format(self.name, self.current)
        )

    def to_FourierXYZ(self, N=10, grid=None, s=None, name="", **kwargs):
        """Convert coil to FourierXYZCoil representation.

        Parameters
        ----------
        N : int
            Fourier resolution of the new X,Y,Z representation.
        grid : Grid, int or None
            Grid used to evaluate curve coordinates on to fit with FourierXYZCoil.
            If an integer, uses that many equally spaced points.
        s : ndarray or "arclength"
            arbitrary curve parameter to use for the fitting.
            Should be monotonic, 1D array of same length as
            coords. if None, defaults linearly spaced in [0,2pi)
            Alternative, can pass "arclength" to use normalized distance between points.
        name : str
            Name for this coil

        Returns
        -------
        coil : FourierXYZCoil
            New representation of the coil parameterized by Fourier series for X,Y,Z.

        """
        if (grid is None) and (s is not None) and (not isinstance(s, str)):
            grid = LinearGrid(zeta=s)
        if grid is None:
            grid = LinearGrid(N=2 * N + 1)
        coords = self.compute("x", grid=grid, basis="xyz")["x"]
        return FourierXYZCoil.from_values(
            self.current, coords, N=N, s=s, basis="xyz", name=name
        )

    def to_SplineXYZ(self, knots=None, grid=None, method="cubic", name="", **kwargs):
        """Convert coil to SplineXYZCoil.

        Parameters
        ----------
        knots : ndarray or "arclength"
            arbitrary curve parameter values to use for spline knots,
            should be an 1D ndarray of same length as the input.
            (input length in this case is determined by grid argument, since
            the input coordinates come from Curve.compute("x",grid=grid))
            If None, defaults to using an linearly spaced points in [0, 2pi) as the
            knots. If supplied, should lie in [0,2pi].
            Alternatively, the string "arclength" can be supplied to use the normalized
            distance between points.
        grid : Grid, int or None
            Grid used to evaluate curve coordinates on to fit with SplineXYZCoil.
            If an integer, uses that many equally spaced points.
        method : str
            method of interpolation
            - `'nearest'`: nearest neighbor interpolation
            - `'linear'`: linear interpolation
            - `'cubic'`: C1 cubic splines (aka local splines)
            - `'cubic2'`: C2 cubic splines (aka natural splines)
            - `'catmull-rom'`: C1 cubic centripetal "tension" splines
        name : str
            Name for this coil

        Returns
        -------
        coil: SplineXYZCoil
            New representation of the coil parameterized by a spline for X,Y,Z.

        """
        if (grid is None) and (knots is not None) and (not isinstance(knots, str)):
            grid = LinearGrid(zeta=knots)
        coords = self.compute("x", grid=grid, basis="xyz")["x"]
        return SplineXYZCoil.from_values(
            self.current, coords, knots=knots, method=method, name=name, basis="xyz"
        )

    def to_FourierRZ(self, N=10, grid=None, NFP=None, sym=False, name="", **kwargs):
        """Convert Coil to FourierRZCoil representation.

        Note that some types of coils may not be representable in this basis.

        Parameters
        ----------
        N : int
            Fourier resolution of the new R,Z representation.
        grid : Grid, int or None
            Grid used to evaluate curve coordinates on to fit with FourierRZCoil.
            If an integer, uses that many equally spaced points.
        NFP : int
            Number of field periods, the coil will have a discrete toroidal symmetry
            according to NFP.
        sym : bool, optional
            Whether the curve is stellarator-symmetric or not. Default is False.
        name : str
            Name for this coil.

        Returns
        -------
        curve : FourierRZCoil
            New representation of the coil parameterized by Fourier series for R,Z.

        """
        NFP = 1 or NFP
        if grid is None:
            grid = LinearGrid(N=2 * N + 1)
        coords = self.compute("x", grid=grid, basis="xyz")["x"]
        return FourierRZCoil.from_values(
            self.current, coords, N=N, NFP=NFP, basis="xyz", sym=sym, name=name
        )

    def to_FourierPlanar(self, N=10, grid=None, basis="xyz", name="", **kwargs):
        """Convert Coil to FourierPlanarCoil representation.

        Note that some types of coils may not be representable in this basis.
        In this case, a least-squares fit will be done to find the
        planar coil that best represents the coil.

        Parameters
        ----------
        N : int
            Fourier resolution of the new FourierPlanarCoil representation.
        grid : Grid, int or None
            Grid used to evaluate curve coordinates on to fit with FourierPlanarCoil.
            If an integer, uses that many equally spaced points.
        basis : {'xyz', 'rpz'}
            Coordinate system for center and normal vectors. Default = 'xyz'.
        name : str
            Name for this coil.

        Returns
        -------
        coil : FourierPlanarCoil
            New representation of the coil parameterized by Fourier series for minor
            radius r in a plane specified by a center position and normal vector.

        """
        if grid is None:
            grid = LinearGrid(N=2 * N + 1)
        coords = self.compute("x", grid=grid, basis=basis)["x"]
        return FourierPlanarCoil.from_values(
            self.current, coords, N=N, basis=basis, name=name
        )

    def to_FourierXY(self, N=10, grid=None, s=None, basis="xyz", name="", **kwargs):
        """Convert Coil to FourierXYCoil representation.

        Note that some types of coils may not be representable in this basis.
        In this case, a least-squares fit will be done to find the
        planar coil that best represents the coil.

        Parameters
        ----------
        N : int
            Fourier resolution of the new FourierXYCoil representation.
        grid : Grid, int or None
            Grid used to evaluate curve coordinates on to fit with FourierXYCoil.
            If an integer, uses that many equally spaced points.
        s : ndarray or "arclength"
            Arbitrary curve parameter to use for the fitting.
            Should be monotonic, 1D array of same length as
            coords. if None, defaults linearly spaced in [0,2pi)
            Alternative, can pass "arclength" to use normalized distance between points.
        basis : {'xyz', 'rpz'}
            Coordinate system for center and normal vectors. Default = 'xyz'.
        name : str
            Name for this coil.

        Returns
        -------
        coil : FourierXYCoil
            New representation of the coil parameterized by Fourier series for the X and
            Y coordinates in a plane specified by a center position and normal vector.

        """
        if (grid is None) and (s is not None) and (not isinstance(s, str)):
            grid = LinearGrid(zeta=s)
        if grid is None:
            grid = LinearGrid(N=2 * N + 1)
        coords = self.compute("x", grid=grid, basis=basis)["x"]
        return FourierXYCoil.from_values(
            self.current, coords, N=N, s=s, basis=basis, name=name
        )


class FourierRZCoil(_Coil, FourierRZCurve):
    """Coil parameterized by fourier series for R,Z in terms of toroidal angle phi.

    Parameters
    ----------
    current : float
        Current through the coil, in Amperes.
    R_n, Z_n: array-like
        fourier coefficients for R, Z
    modes_R : array-like
        mode numbers associated with R_n. If not given defaults to [-n:n]
    modes_Z : array-like
        mode numbers associated with Z_n, defaults to modes_R
    NFP : int
        number of field periods
    sym : bool
        whether to enforce stellarator symmetry
    name : str
        Name for this coil

    Examples
    --------
    .. code-block:: python

        from desc.coils import FourierRZCoil
        from desc.grid import LinearGrid
        import numpy as np

        I = 10
        mu0 = 4 * np.pi * 1e-7
        R_coil = 10
        # circular coil given by R(phi) = 10
        coil = FourierRZCoil(
            current=I, R_n=R_coil, Z_n=0, modes_R=[0]
        )
        z0 = 10
        field_evaluated = coil.compute_magnetic_field(
            np.array([[0, 0, 0], [0, 0, z0]]), basis="rpz"
        )
        np.testing.assert_allclose(
            field_evaluated[0, :], np.array([0, 0, mu0 * I / 2 / R_coil]), atol=1e-8
        )
        np.testing.assert_allclose(
            field_evaluated[1, :],
            np.array(
                [0, 0, mu0 * I / 2 * R_coil**2 / (R_coil**2 + z0**2) ** (3 / 2)]
            ),
            atol=1e-8,
        )

    """

    _io_attrs_ = _Coil._io_attrs_ + FourierRZCurve._io_attrs_
    _static_attrs = _Coil._static_attrs + FourierRZCurve._static_attrs

    def __init__(
        self,
        current=1,
        R_n=10,
        Z_n=0,
        modes_R=None,
        modes_Z=None,
        NFP=1,
        sym="auto",
        name="",
    ):
        super().__init__(current, R_n, Z_n, modes_R, modes_Z, NFP, sym, name)

    @classmethod
    def from_values(cls, current, coords, N=10, NFP=1, basis="rpz", sym=False, name=""):
        """Fit coordinates to FourierRZCoil representation.

        Parameters
        ----------
        current : float
            Current through the coil, in Amperes.
        coords: ndarray, shape (num_coords,3)
            coordinates to fit a FourierRZCurve object with each column
            corresponding to xyz or rpz depending on the basis argument.
        N : int
            Fourier resolution of the new R,Z representation.
        NFP : int
            Number of field periods, the curve will have a discrete toroidal symmetry
            according to NFP.
        basis : {"rpz", "xyz"}
            basis for input coordinates. Defaults to "rpz"
        sym : bool
            Whether to enforce stellarator symmetry.
        name : str
            Name for this coil.

        Returns
        -------
        coil : FourierRZCoil
            New representation of the coil parameterized by Fourier series for R,Z.

        """
        curve = super().from_values(
            coords=coords, N=N, NFP=NFP, basis=basis, sym=sym, name=name
        )
        return FourierRZCoil(
            current=current,
            R_n=curve.R_n,
            Z_n=curve.Z_n,
            modes_R=curve.R_basis.modes[:, 2],
            modes_Z=curve.Z_basis.modes[:, 2],
            NFP=NFP,
            sym=curve.sym,
            name=name,
        )


class FourierXYZCoil(_Coil, FourierXYZCurve):
    """Coil parameterized by Fourier series for X,Y,Z in terms of an arbitrary angle s.

    Parameters
    ----------
    current : float
        Current through the coil, in Amperes.
    X_n, Y_n, Z_n: array-like
        fourier coefficients for X, Y, Z
    modes : array-like
        mode numbers associated with X_n etc.
    name : str
        Name for this coil

    Examples
    --------
    .. code-block:: python

        from desc.coils import FourierXYZCoil
        from desc.grid import LinearGrid
        import numpy as np

        I = 10
        mu0 = 4 * np.pi * 1e-7
        R_coil = 10
        # circular coil given by X(s) = 10*cos(s), Y(s) = 10*sin(s)
        coil = FourierXYZCoil(
            current=I,
            X_n=[0, R_coil, 0],
            Y_n=[0, 0, R_coil],
            Z_n=[0, 0, 0],
            modes=[0, 1, -1],
        )
        z0 = 10
        field_evaluated = coil.compute_magnetic_field(
            np.array([[0, 0, 0], [0, 0, z0]]), basis="rpz"
        )
        np.testing.assert_allclose(
            field_evaluated[0, :], np.array([0, 0, mu0 * I / 2 / R_coil]), atol=1e-8
        )
        np.testing.assert_allclose(
            field_evaluated[1, :],
            np.array([0, 0, mu0 * I / 2 * R_coil**2 / (R_coil**2 + z0**2) ** (3 / 2)]),
            atol=1e-8,
        )

    """

    _io_attrs_ = _Coil._io_attrs_ + FourierXYZCurve._io_attrs_
    _static_attrs = _Coil._static_attrs + FourierXYZCurve._static_attrs

    def __init__(
        self,
        current=1,
        X_n=[0, 10, 2],
        Y_n=[0, 0, 0],
        Z_n=[-2, 0, 0],
        modes=None,
        name="",
    ):
        super().__init__(current, X_n, Y_n, Z_n, modes, name)

    @classmethod
    def from_values(cls, current, coords, N=10, s=None, basis="xyz", name=""):
        """Fit coordinates to FourierXYZCoil representation.

        Parameters
        ----------
        current : float
            Current through the coil, in Amperes.
        coords: ndarray
            Coordinates to fit a FourierXYZCoil object with.
        N : int
            Fourier resolution of the new X,Y,Z representation.
            default is 10
        s : ndarray
            arbitrary curve parameter to use for the fitting.
            Should be monotonic, 1D array of same length as
            coords
            if None, defaults to normalized arclength
        basis : {"rpz", "xyz"}
            basis for input coordinates. Defaults to "xyz"
        name : str
            Name for this coil.

        Returns
        -------
        coil : FourierXYZCoil
            New representation of the coil parameterized by Fourier series for X,Y,Z.

        """
        curve = super().from_values(coords=coords, N=N, s=s, basis=basis, name=name)
        return FourierXYZCoil(
            current=current,
            X_n=curve.X_n,
            Y_n=curve.Y_n,
            Z_n=curve.Z_n,
            modes=curve.X_basis.modes[:, 2],
            name=name,
        )


class FourierPlanarCoil(_Coil, FourierPlanarCurve):
    """Coil that lies in a plane.

    Parameterized by a point (the center of the coil), a vector (normal to the plane),
    and a Fourier series defining the radius from the center as a function of the polar
    angle theta.

    Parameters
    ----------
    current : float
        Current through the coil, in Amperes.
    center : array-like, shape(3,)
        Coordinates of center of curve, in system determined by basis.
    normal : array-like, shape(3,)
        Components of normal vector to planar surface, in system determined by basis.
    r_n : array-like
        Fourier coefficients for radius from center as function of polar angle
    modes : array-like
        mode numbers associated with r_n
    basis : {'xyz', 'rpz'}
        Coordinate system for center and normal vectors. Default = 'xyz'.
    name : str
        Name for this coil

    Examples
    --------
    .. code-block:: python

        from desc.coils import FourierPlanarCoil
        from desc.grid import LinearGrid
        import numpy as np

        I = 10
        mu0 = 4 * np.pi * 1e-7
        R_coil = 10
        # circular coil given by center at (0,0,0)
        # and normal vector in Z direction (0,0,1) and radius 10
        coil = FourierPlanarCoil(
            current=I,
            center=[0, 0, 0],
            normal=[0, 0, 1],
            r_n=R_coil,
            modes=[0],
        )
        z0 = 10
        field_evaluated = coil.compute_magnetic_field(
            np.array([[0, 0, 0], [0, 0, z0]]), basis="rpz"
        )
        np.testing.assert_allclose(
            field_evaluated[0, :], np.array([0, 0, mu0 * I / 2 / R_coil]), atol=1e-8
        )
        np.testing.assert_allclose(
            field_evaluated[1, :],
            np.array([0, 0, mu0 * I / 2 * R_coil**2 / (R_coil**2 + z0**2) ** (3 / 2)]),
            atol=1e-8,
        )

    """

    _io_attrs_ = _Coil._io_attrs_ + FourierPlanarCurve._io_attrs_
    _static_attrs = _Coil._static_attrs + FourierPlanarCurve._static_attrs

    def __init__(
        self,
        current=1,
        center=[10, 0, 0],
        normal=[0, 1, 0],
        r_n=2,
        modes=None,
        basis="xyz",
        name="",
    ):
        super().__init__(current, center, normal, r_n, modes, basis, name)

    @classmethod
    def from_values(cls, current, coords, N=10, basis="xyz", name=""):
        """Fit coordinates to FourierPlanarCoil representation.

        Parameters
        ----------
        current : float
            Current through the coil, in Amperes.
        coords: ndarray, shape (num_coords,3)
            Coordinates to fit a FourierPlanarCurve object with each column
            corresponding to xyz or rpz depending on the basis argument.
        N : int
            Fourier resolution of the new r representation.
        basis : {"rpz", "xyz"}
            Basis for input coordinates. Defaults to "xyz".
        name : str
            Name for this curve.

        Returns
        -------
        curve : FourierPlanarCoil
            New representation of the coil parameterized by a Fourier series for r.

        """
        curve = super().from_values(coords=coords, N=N, basis=basis, name=name)
        return FourierPlanarCoil(
            current=current,
            center=curve.center,
            normal=curve.normal,
            r_n=curve.r_n,
            modes=curve.r_basis.modes[:, 2],
            basis="xyz",
            name=name,
        )


class FourierXYCoil(_Coil, FourierXYCurve):
    """Coil that lies in a plane.

    Parameterized by a point (the center of the coil), a vector (normal to the plane),
    and Fourier series defining the X and Y coordinates in the plane as a function of
    an arbitrary angle s.

    Parameters
    ----------
    current : float
        Current through the coil, in Amperes.
    center : array-like, shape(3,)
        Coordinates of center of curve, in system determined by basis.
    normal : array-like, shape(3,)
        Components of normal vector to planar surface, in system determined by basis.
    X_n : array-like
        Fourier coefficients of the X coordinate in the plane.
    Y_n : array-like
        Fourier coefficients of the Y coordinate in the plane.
    modes : array-like
        Mode numbers associated with X_n and Y_n. The n=0 mode will be ignored.
    basis : {'xyz', 'rpz'}
        Coordinate system for center and normal vectors. Default = 'xyz'.
    name : str
        Name for this coil.

    """

    _io_attrs_ = _Coil._io_attrs_ + FourierXYCurve._io_attrs_
    _static_attrs = _Coil._static_attrs + FourierXYCurve._static_attrs

    def __init__(
        self,
        current=1,
        center=[10, 0, 0],
        normal=[0, 1, 0],
        X_n=[0, 2],
        Y_n=[2, 0],
        modes=None,
        basis="xyz",
        name="",
    ):
        super().__init__(current, center, normal, X_n, Y_n, modes, basis, name)

    @classmethod
    def from_values(cls, current, coords, N=10, s=None, basis="xyz", name=""):
        """Fit coordinates to FourierXYCoil representation.

        Parameters
        ----------
        current : float
            Current through the coil, in Amperes.
        coords: ndarray, shape (num_coords,3)
            Coordinates to fit a FourierXYCurve object with each column
            corresponding to xyz or rpz depending on the basis argument.
        N : int
            Fourier resolution of the new X & Y representation.
        s : ndarray or "arclength"
            Arbitrary curve parameter to use for the fitting.
            Should be monotonic, 1D array of same length as
            coords. if None, defaults linearly spaced in [0,2pi)
            Alternative, can pass "arclength" to use normalized distance between points.
        basis : {"rpz", "xyz"}
            Basis for input coordinates. Defaults to "xyz".
        name : str
            Name for this curve.

        Returns
        -------
        curve : FourierXYCoil
            New representation of the coil parameterized by a Fourier series for X & Y.

        """
        curve = super().from_values(coords=coords, N=N, s=s, basis=basis, name=name)
        return FourierXYCoil(
            current=current,
            center=curve.center,
            normal=curve.normal,
            X_n=curve.X_n,
            Y_n=curve.Y_n,
            modes=curve.X_basis.modes[:, 2],
            basis="xyz",
            name=name,
        )


class SplineXYZCoil(_Coil, SplineXYZCurve):
    """Coil parameterized by spline points in X,Y,Z.

    Parameters
    ----------
    current : float
        Current through the coil, in Amperes.
    X, Y, Z: array-like
        Points for X, Y, Z describing the curve. If the endpoint is included
        (ie, X[0] == X[-1]), then the final point will be dropped.
    knots : ndarray
        arbitrary curve parameter values to use for spline knots,
        should be a monotonic, 1D ndarray of same length as the input X,Y,Z.
        If None, defaults to using an equal-arclength angle as the knots
        If supplied, will be rescaled to lie in [0,2pi]
    method : str
        method of interpolation

        - ``'nearest'``: nearest neighbor interpolation
        - ``'linear'``: linear interpolation
        - ``'cubic'``: C1 cubic splines (aka local splines)
        - ``'cubic2'``: C2 cubic splines (aka natural splines)
        - ``'catmull-rom'``: C1 cubic centripetal "tension" splines
        - ``'cardinal'``: C1 cubic general tension splines. If used, default tension of
          c = 0 will be used
        - ``'monotonic'``: C1 cubic splines that attempt to preserve monotonicity in the
          data, and will not introduce new extrema in the interpolated points
        - ``'monotonic-0'``: same as `'monotonic'` but with 0 first derivatives at both
          endpoints
    name : str
        Name for this coil

    """

    _io_attrs_ = _Coil._io_attrs_ + SplineXYZCurve._io_attrs_
    _static_attrs = _Coil._static_attrs + SplineXYZCurve._static_attrs

    def __init__(self, current, X, Y, Z, knots=None, method="cubic", name=""):
        super().__init__(current, X, Y, Z, knots, method, name)

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

        The coil current may be overridden by including `current`
        in the `params` dictionary.

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

        Notes
        -----
        Discretizes the coil into straight segments between grid points, and uses the
        Hanson-Hirshman expression for exact field from a straight segment. Convergence
        is approximately quadratic in the number of coil points.

        """
        errorif(
            compute_A_or_B not in ["A", "B"],
            ValueError,
            f'Expected "A" or "B" for compute_A_or_B, instead got {compute_A_or_B}',
        )
        op = {"B": biot_savart_hh, "A": biot_savart_vector_potential_hh}[compute_A_or_B]
        assert basis.lower() in ["rpz", "xyz"]
        coords = jnp.atleast_2d(jnp.asarray(coords))
        if basis == "rpz":
            coords = rpz2xyz(coords)
        if params is None:
            current = self.current
        else:
            current = params.pop("current", self.current)

        if source_grid is None:
            # NFP=1 to ensure points span the entire length of the coil
            # using more points than knots.size (self.N) to better sample coil
            source_grid = LinearGrid(N=self.N * 2 + 5)
        else:
            # coil grids should have NFP=1. The only possible exception is FourierRZCoil
            # which in theory can be different as long as it matches the coils NFP.
            errorif(
                getattr(source_grid, "NFP", 1) != 1,
                ValueError,
                "source_grid for coils must have NFP=1",
            )

        data = self.compute(
            ["x"], grid=source_grid, params=params, basis="xyz", transforms=transforms
        )
        # need to make sure the curve is closed. If it's already closed, this doesn't
        # do anything (effectively just adds a segment of zero length which has no
        # effect on the overall result)
        coil_pts_start = data["x"]
        coil_pts_end = jnp.concatenate([data["x"][1:], data["x"][:1]])
        # could get up to 4th order accuracy by shifting points outward as in
        # (McGreivy, Zhu, Gunderson, Hudson 2021), however that requires knowing the
        # coils curvature which is a 2nd derivative of the position, and doing that
        # with only possibly c1 cubic splines is inaccurate, so we don't do it
        # (for now, maybe in the future?)
        AB = op(coords, coil_pts_start, coil_pts_end, current, chunk_size=chunk_size)

        if basis == "rpz":
            AB = xyz2rpz_vec(AB, x=coords[:, 0], y=coords[:, 1])
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

        The coil current may be overridden by including `current`
        in the `params` dictionary.

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
        field : ndarray, shape(n,3)
            magnetic field at specified points, in either rpz or xyz coordinates

        Notes
        -----
        Discretizes the coil into straight segments between grid points, and uses the
        Hanson-Hirshman expression for exact field from a straight segment. Convergence
        is approximately quadratic in the number of coil points.

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

        The coil current may be overridden by including `current`
        in the `params` dictionary.

        Parameters
        ----------
        coords : array-like shape(n,3)
            Nodes to evaluate magnetic vector potential at in [R,phi,Z]
            or [X,Y,Z] coordinates.
        params : dict, optional
            Parameters to pass to Curve.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic vector potential.
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
        A : ndarray, shape(n,3)
            Magnetic vector potential at specified points, in either
            rpz or xyz coordinates

        Notes
        -----
        Discretizes the coil into straight segments between grid points, and uses the
        Hanson-Hirshman expression for exact vector potential from a straight segment.
        Convergence is approximately quadratic in the number of coil points.

        """
        return self._compute_A_or_B(
            coords, params, basis, source_grid, transforms, "A", chunk_size=chunk_size
        )

    @classmethod
    def from_values(
        cls, current, coords, knots=None, method="cubic", name="", basis="xyz"
    ):
        """Create SplineXYZCoil from coordinate values.

        Parameters
        ----------
        current : float
            Current through the coil, in Amperes.
        coords: ndarray
            Points for X, Y, Z describing the curve. If the endpoint is included
            (ie, X[0] == X[-1]), then the final point will be dropped.
        knots : ndarray
            arbitrary curve parameter values to use for spline knots,
            should be an 1D ndarray of same length as the input.
            (input length in this case is determined by grid argument, since
            the input coordinates come from
            Curve.compute("x",grid=grid))
            If None, defaults to using an equal-arclength angle as the knots
            If supplied, will be rescaled to lie in [0,2pi]
        method : str
            method of interpolation

            - `'nearest'`: nearest neighbor interpolation
            - `'linear'`: linear interpolation
            - `'cubic'`: C1 cubic splines (aka local splines)
            - `'cubic2'`: C2 cubic splines (aka natural splines)
            - `'catmull-rom'`: C1 cubic centripetal "tension" splines

        name : str
            Name for this curve

        basis : {"rpz", "xyz"}
            basis for input coordinates. Defaults to "xyz"

        Returns
        -------
        coil: SplineXYZCoil
            New representation of the coil parameterized by splines in X,Y,Z.

        """
        curve = super().from_values(
            coords=coords, knots=knots, method=method, basis=basis, name=name
        )
        return SplineXYZCoil(
            current=current,
            X=curve.X,
            Y=curve.Y,
            Z=curve.Z,
            knots=curve.knots,
            method=curve.method,
            name=name,
        )


def _check_type(coil0, coil):
    errorif(
        not isinstance(coil, coil0.__class__),
        TypeError,
        (
            "coils in a CoilSet must all be the same type, got types "
            + f"{type(coil0)}, {type(coil)}. Consider using a MixedCoilSet"
        ),
    )
    errorif(
        isinstance(coil0, CoilSet),
        TypeError,
        (
            "coils in a CoilSet must all be base Coil types, not CoilSet. "
            + "Consider using a MixedCoilSet"
        ),
    )
    attrs = {
        FourierPlanarCoil: ["r_basis"],
        FourierRZCoil: ["R_basis", "Z_basis", "NFP", "sym"],
        FourierXYCoil: ["X_basis", "Y_basis"],
        FourierXYZCoil: ["X_basis", "Y_basis", "Z_basis"],
        SplineXYZCoil: ["method", "N", "knots"],
    }

    for attr in attrs[coil0.__class__]:
        a0 = getattr(coil0, attr)
        a1 = getattr(coil, attr)
        errorif(
            not equals(a0, a1),
            ValueError,
            (
                "coils in a CoilSet must have the same parameterization, got a "
                + f"mismatch between attr {attr}, with values {a0} and {a1}."
                + " Consider using a MixedCoilSet"
            ),
        )


class CoilSet(OptimizableCollection, _Coil, MutableSequence):
    """Set of coils of different geometry but shared parameterization and resolution.

    Parameters
    ----------
    coils : Coil or array-like of Coils
        Collection of coils. Must all be the same type and resolution.
    NFP : int (optional)
        Number of field periods for enforcing field period symmetry.
        If NFP > 1, only include the unique coils in the first field period,
        and the magnetic field will be computed assuming 'virtual' coils from the other
        field periods. Default = 1.
    sym : bool (optional)
        Whether to enforce stellarator symmetry. If sym = True, only include the
        unique coils in a half field period, and the magnetic field will be computed
        assuming 'virtual' coils from the other half field period. Default = False.
    name : str
        Name of this CoilSet.
    check_intersection: bool
        Whether or not to check the coils in the coilset for intersections.

    """

    _io_attrs_ = _Coil._io_attrs_ + ["_coils", "_NFP", "_sym"]
    _io_attrs_.remove("_current")
    _static_attrs = (
        OptimizableCollection._static_attrs
        + _Coil._static_attrs
        + ["_NFP", "_sym", "_name"]
    )

    def __init__(self, *coils, NFP=1, sym=False, name="", check_intersection=True):
        coils = flatten_list(coils, flatten_tuple=True)
        assert all([isinstance(coil, (_Coil)) for coil in coils])
        [_check_type(coil, coils[0]) for coil in coils]
        self._coils = list(coils)
        self._NFP = int(NFP)
        self._sym = bool(sym)
        self._name = str(name)

        if check_intersection:
            self.is_self_intersecting()

    @property
    def name(self):
        """str: Name of the curve."""
        return self.__dict__.setdefault("_name", "")

    @name.setter
    def name(self, new):
        self._name = str(new)

    @property
    def coils(self):
        """list: coils in the coilset."""
        return self._coils

    @property
    def num_coils(self):
        """int: Number of coils."""
        return len(self) * (int(self.sym) + 1) * self.NFP

    @property
    def NFP(self):
        """int: Number of (toroidal) field periods."""
        return self._NFP

    @property
    def sym(self):
        """bool: Whether this coil set is stellarator symmetric."""
        return self._sym

    @property
    def current(self):
        """list: currents in each coil."""
        return [coil.current for coil in self.coils]

    @current.setter
    def current(self, new):
        # new must be a 1D iterable regardless of the tree structure of the CoilSet
        old, tree = tree_flatten(self.current)
        new = jnp.atleast_1d(new).flatten()
        new = jnp.broadcast_to(new, (len(old),))
        new = tree_unflatten(tree, new)
        for coil, cur in zip(self.coils, new):
            coil.current = cur

    def _all_currents(self, currents=None):
        """Return an array of all the currents (including those in virtual coils)."""
        if currents is None:
            currents = self.current
        currents = jnp.asarray(currents)
        if self.sym:
            currents = jnp.concatenate([currents, -1 * currents[::-1]])
        return jnp.tile(currents, self.NFP)

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
        """Compute the quantity given by name on grid, for each coil in the coilset.

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
            If array-like, should be 1 value per coil.

        Returns
        -------
        data : list of dict of ndarray
            Computed quantity and intermediate variables, for each coil in the set.
            List entries map to coils in coilset, each dict contains data for an
            individual coil.

        """
        if params is None:
            params = [
                get_params(names, coil, basis=kwargs.get("basis", "rpz"))
                for coil in self
            ]
        if data is None:
            data = [{}] * len(self)

        # if user supplied initial data for each coil we also need to vmap over that.
        data = vmap(
            lambda d, x: self[0].compute(
                names, grid=grid, transforms=transforms, data=d, params=x, **kwargs
            )
        )(tree_stack(data), tree_stack(params))
        return tree_unstack(data)

    def translate(self, *args, **kwargs):
        """Translate the coils along an axis."""
        [coil.translate(*args, **kwargs) for coil in self.coils]

    def rotate(self, *args, **kwargs):
        """Rotate the coils about an axis."""
        [coil.rotate(*args, **kwargs) for coil in self.coils]

    def flip(self, *args, **kwargs):
        """Flip the coils across a plane."""
        [coil.flip(*args, **kwargs) for coil in self.coils]

    def _compute_position(self, params=None, grid=None, dx1=False, **kwargs):
        """Compute coil positions accounting for stellarator symmetry.

        Parameters
        ----------
        params : dict or array-like of dict, optional
            Parameters to pass to coils, either the same for all coils or one for each.
        grid : Grid or int, optional
            Grid of coordinates to evaluate at. Defaults to a Linear grid.
            If an integer, uses that many equally spaced points.
        dx1 : bool
            If True, also return dx/ds for each curve.

        Returns
        -------
        x : ndarray, shape(len(self),source_grid.num_nodes,3)
            Coil positions, in [R,phi,Z] or [X,Y,Z] coordinates.
        x_s : ndarray, shape(len(self),source_grid.num_nodes,3)
            Coil position derivatives, in [R,phi,Z] or [X,Y,Z] coordinates.
            Only returned if dx1=True.

        """
        basis = kwargs.pop("basis", "xyz")
        keys = ["x", "x_s"] if dx1 else ["x"]
        if params is None:
            params = [get_params(keys, coil, basis=basis) for coil in self]
        data = self.compute(keys, grid=grid, params=params, basis=basis, **kwargs)
        data = tree_leaves(data, is_leaf=lambda x: isinstance(x, dict))
        x = jnp.dstack([d["x"].T for d in data]).T  # shape=(ncoils,num_nodes,3)
        if dx1:
            x_s = jnp.dstack([d["x_s"].T for d in data]).T  # shape=(ncoils,num_nodes,3)
        # stellarator symmetry is easiest in [X,Y,Z] coordinates
        xyz = rpz2xyz(x) if basis.lower() == "rpz" else x
        if dx1:
            xyz_s = (
                rpz2xyz_vec(x_s, xyz[:, :, 0], xyz[:, :, 1])
                if basis.lower() == "rpz"
                else x_s
            )

        # if stellarator symmetric, add reflected coils from the other half field period
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

        # if field period symmetry, add rotated coils from other field periods
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

    def _compute_linking_number(self, params=None, grid=None):
        """Calculate linking numbers for coils in the coilset.

        Parameters
        ----------
        params : dict or array-like of dict, optional
            Parameters to pass to coils, either the same for all coils or one for each.
        grid : Grid or int, optional
            Grid of coordinates to evaluate at. Defaults to a Linear grid.
            If an integer, uses that many equally spaced points.

        Returns
        -------
        link : ndarray, shape(num_coils, num_coils)
            Linking number of each coil with each other coil. link=0 means they are not
            linked, +/- 1 means the coils link each other in one direction or another.

        """
        if grid is None:
            grid = LinearGrid(N=50)
        dx = grid.spacing[:, 2]
        x, x_s = self._compute_position(params, grid, dx1=True, basis="xyz")
        link = _linking_number(
            x[:, None], x[None, :], x_s[:, None], x_s[None, :], dx, dx
        )
        return link / (4 * jnp.pi)

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
            Parameters to pass to coils, either the same for all coils or one for each.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic field.
        source_grid : Grid, int or None, optional
            Grid used to discretize coils. If an integer, uses that many equally spaced
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
        # NFP symmetry applies to coilset as a whole, not individual coils, so the grid
        # should have NFP=1.
        errorif(
            getattr(source_grid, "NFP", 1) != 1,
            ValueError,
            "source_grid for CoilSet must have NFP=1",
        )
        assert basis.lower() in ["rpz", "xyz"]
        coords = jnp.atleast_2d(jnp.asarray(coords))
        if params is None:
            params = [
                get_params(["x_s", "x", "s", "ds"], coil, basis=basis) for coil in self
            ]
            for par, coil in zip(params, self):
                par["current"] = coil.current

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
        """Compute magnetic field at a set of points.

        Parameters
        ----------
        coords : array-like shape(n,3)
            Nodes to evaluate field at in [R,phi,Z] or [X,Y,Z] coordinates.
        params : dict or array-like of dict, optional
            Parameters to pass to coils, either the same for all coils or one for each.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic field.
        source_grid : Grid, int or None, optional
            Grid used to discretize coils. If an integer, uses that many equally spaced
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
            Parameters to pass to coils, either the same for all coils or one for each.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic field.
        source_grid : Grid, int or None, optional
            Grid used to discretize coils. If an integer, uses that many equally spaced
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
        cls,
        coil,
        current=None,
        axis=[0, 0, 1],
        angle=2 * np.pi,
        n=10,
        endpoint=False,
        check_intersection=True,
    ):
        """Create a CoilSet by repeating a coil at equal spacing around the torus.

        Parameters
        ----------
        coil : Coil
            Base coil to repeat.
        current : float or array-like, shape(n,)
            Current through (each) coil, in Amperes. Overrides coil.current.
        axis : array-like, shape(3,)
            Axis to rotate about, in X,Y,Z coordinates.
        angle : float
            Total rotational extent of the final coil, in radians.
        n : int
            Number of copies of original coil.
        endpoint : bool
            Whether to include a coil at final rotation angle. Default = False.
        check_intersection : bool
            whether to check the resulting coilsets for intersecting coils.

        """
        assert isinstance(coil, _Coil) and not isinstance(coil, CoilSet)
        if current is None:
            current = coil.current
        currents = jnp.broadcast_to(current, (n,))
        phi = jnp.linspace(0, angle, n, endpoint=endpoint)
        coils = []
        for i in range(n):
            coili = coil.copy()
            coili.rotate(axis=axis, angle=phi[i])
            coili.current = currents[i]
            coils.append(coili)
        return cls(*coils, check_intersection=check_intersection)

    @classmethod
    def linspaced_linear(
        cls,
        coil,
        current=None,
        displacement=[2, 0, 0],
        n=4,
        endpoint=False,
        check_intersection=True,
    ):
        """Create a CoilSet by repeating a coil at equal spacing in a straight line.

        Parameters
        ----------
        coil : Coil
            Base coil to repeat.
        current : float or array-like, shape(n,)
            Current through (each) coil, in Amperes. Overrides coil.current.
        displacement : array-like, shape(3,)
            Total displacement of the final coil, relative to the initial coil position,
            in X,Y,Z coordinates.
        n : int
            Number of copies of original coil.
        endpoint : bool
            Whether to include a coil at final displacement location. Default = False.
        check_intersection : bool
            whether to check the resulting coilsets for intersecting coils.

        """
        assert isinstance(coil, _Coil) and not isinstance(coil, CoilSet)
        if current is None:
            current = coil.current
        currents = jnp.broadcast_to(current, (n,))
        displacement = jnp.asarray(displacement)
        a = jnp.linspace(0, 1, n, endpoint=endpoint)
        coils = []
        for i in range(n):
            coili = coil.copy()
            coili.translate(a[i] * displacement)
            coili.current = currents[i]
            coils.append(coili)
        return cls(*coils, check_intersection=check_intersection)

    @classmethod
    def from_symmetry(cls, coils, NFP=1, sym=False, check_intersection=True):
        """Create a coil group by reflection and symmetry.

        Given coils over one field period, repeat coils NFP times between
        0 and 2pi to form full coil set.

        Or, given coils over 1/2 of a field period, repeat coils 2*NFP times
        between 0 and 2pi to form full stellarator symmetric coil set.

        Parameters
        ----------
        coils : Coil, CoilSet
            Coil or collection of coils in one field period or half field period.
        NFP : int (optional)
            Number of field periods for enforcing field period symmetry.
            The coils will be duplicated NFP times. Default = 1.
        sym : bool (optional)
            Whether to enforce stellarator symmetry.
            If True, the coils will be duplicated 2*NFP times. Default = False.
        check_intersection : bool
            whether to check the resulting coilsets for intersecting coils.

        Returns
        -------
        coilset : CoilSet
            A new coil set with NFP=1 and sym=False that is equivalent to the unique
            coils with field period symmetry and stellarator symmetry.
            The total number of coils in the new coil set is:
            len(coilset) = len(coils) * NFP * (int(sym) + 1)

        """
        if not isinstance(coils, CoilSet):
            try:
                coils = CoilSet(coils)
            except (TypeError, ValueError):
                # likely there are multiple coil types,
                # so make a MixedCoilSet
                coils = MixedCoilSet(coils)
        if not isinstance(coils, MixedCoilSet):
            # only need to check this for a CoilSet, not MixedCoilSet
            [_check_type(coil, coils[0]) for coil in coils]

        coilset = []
        if sym:
            # first reflect/flip original coilset
            # ie, given coils [1, 2, 3] at angles [pi/6, pi/2, 5pi/6]
            # we want a new set like [1, 2, 3, flip(3), flip(2), flip(1)]
            # at [pi/6, pi/2, 5pi/6, 7pi/6, 3pi/2, 11pi/6]
            flipped_coils = []
            normal = jnp.array([-jnp.sin(jnp.pi / NFP), jnp.cos(jnp.pi / NFP), 0])
            for coil in coils[::-1]:
                fcoil = coil.copy()
                fcoil.flip(normal)
                fcoil.flip([0, 0, 1])
                fcoil.current = -1 * coil.current
                flipped_coils.append(fcoil)
            coils = coils + flipped_coils
        # next rotate the coilset for each field period
        for k in range(0, NFP):
            rotated_coils = coils.copy()
            rotated_coils.rotate(axis=[0, 0, 1], angle=2 * jnp.pi * k / NFP)
            coilset += rotated_coils

        return cls(*coilset, check_intersection=check_intersection)

    @classmethod
    def from_makegrid_coilfile(cls, coil_file, method="cubic", check_intersection=True):
        """Create a CoilSet of SplineXYZCoils from a MAKEGRID-formatted coil txtfile.

        If the MAKEGRID contains more than one coil group (denoted by the number listed
        after the current on the last line defining a given coil), this function will
        attempt to return only a single CoilSet of all of the coils in the file.

        Parameters
        ----------
        coil_file : str or path-like
            path to coil file in txt format
        method : str
            method of interpolation

            - ``'nearest'``: nearest neighbor interpolation
            - ``'linear'``: linear interpolation
            - ``'cubic'``: C1 cubic splines (aka local splines)
            - ``'cubic2'``: C2 cubic splines (aka natural splines)
            - ``'catmull-rom'``: C1 cubic centripetal "tension" splines
            - ``'cardinal'``: C1 cubic general tension splines. If used, default tension
              of c = 0 will be used
            - ``'monotonic'``: C1 cubic splines that attempt to preserve monotonicity in
              the data, and will not introduce new extrema in the interpolated points
            - ``'monotonic-0'``: same as `'monotonic'` but with 0 first derivatives at
              both endpoints
        check_intersection : bool
            whether to check the resulting coilsets for intersecting coils.

        """
        coils = []  # list of SplineXYZCoils, ignoring coil groups
        coilinds = [2]  # List of line indices where coils are at in the file.
        # always start at the 3rd line after periods
        coilnames = []  # the coilgroup each coil belongs to
        # corresponds to each coil in the coilinds list
        coil_file = os.path.expanduser(coil_file)
        # read in the coils file
        headind = -1
        with open(coil_file) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.find("periods") != -1:
                    headind = i  # skip anything that is above the periods line
                    coilinds[0] += headind
                    if len(lines[3 + headind].split()) != 4:
                        raise OSError(
                            "4th line in file must be the start of the first coil! "
                            + "Expected a line of length 4 (after .split()), "
                            + f"instead got length {lines[3+headind].split()}"
                        )
                    header_lines_not_as_expected = np.array(
                        [
                            len(lines[0 + headind].split()) != 2,
                            len(lines[1 + headind].split()) != 2,
                            len(lines[2 + headind].split()) != 2,
                        ]
                    )
                    if np.any(header_lines_not_as_expected):
                        wronglines = lines[
                            np.where(header_lines_not_as_expected)[0] + headind
                        ]
                        raise OSError(
                            "First 3 lines in file starting with the periods line "
                            + "must be the header lines,"
                            + " each of length 2 (after .split())! "
                            + f"Line(s) {wronglines}"
                            + " are not length 2"
                        )

                    continue
                if (
                    line.find("begin filament") != -1
                    or line.find("end") != -1
                    or line.find("mirror") != -1
                ):
                    continue  # skip headers and last line
                if (
                    len(line.split()) != 4  # find the line immediately before a coil,
                    # where the line length is greater than 4
                    and line.strip()  # ensure not counting blank lines
                    # if we have not found the header yet, skip the line
                    and headind != -1
                ):
                    coilinds.append(i)
                    groupname = " ".join(line.split()[4:])
                    coilnames.append(groupname)

        for i, (start, end, coilname) in enumerate(
            zip(coilinds[0:-1], coilinds[1:], coilnames)
        ):
            coords = np.genfromtxt(lines[start + 1 : end])
            coils.append(
                SplineXYZCoil(
                    coords[:, -1][0],
                    coords[:, 0],
                    coords[:, 1],
                    coords[:, 2],
                    method=method,
                    name=coilname,
                )
            )

        try:
            return cls(*coils, check_intersection=check_intersection)
        except ValueError as e:
            errorif(
                True,
                ValueError,
                f"Unable to create CoilSet with the coils in the file, got error {e}."
                + "The issue is likely differing numbers of knots for the coils, "
                "try using a MixedCoilSet instead of a CoilSet.",
            )

    def save_in_makegrid_format(self, coilsFilename, NFP=None, grid=None):
        """Save CoilSet as a MAKEGRID-formatted coil txtfile.

        By default, each coil is assigned to the same Coilgroup in MAKEGRID
        with the name "Modular". For more details see the MAKEGRID documentation
        https://princetonuniversity.github.io/STELLOPT/MAKEGRID.html

        Note: if a nested CoilSet, will flatten it first before saving

        Parameters
        ----------
        filename : str or path-like
            path save CoilSet as a file in MAKEGRID txt format
        NFP : int, default None
            If > 1, assumes that the CoilSet is the coils for a coilset
            with a nominal discrete toroidal symmetry of NFP, and will
            put that NFP in the periods line of the coils file generated.
            defaults to 1
        grid: Grid, ndarray, int,
            Grid of sample points along each coil to save.
            if None, will default to the coil compute functions's
            default grid
        """
        # TODO(#1376): name each group based off of CoilSet name?
        # TODO(#1376): have CoilGroup be automatically assigned based off of
        # CoilSet if current coilset is a collection of coilsets?

        coilsFilename = os.path.expanduser(coilsFilename)

        NFP = 1 if NFP is None else NFP

        def flatten_coils(coilset):
            if hasattr(coilset, "__len__"):
                if isinstance(coilset, CoilSet):
                    if coilset.NFP > 1 or coilset.sym:
                        # hit a CoilSet with symmetries, this coilset only contains
                        # its unique coils. However, for this function we
                        # need to get the entire coilset, not just the unique coils,
                        # so make a MixedCoilSet using this CoilSet's coils and NFP/sym
                        coilset_full = MixedCoilSet.from_symmetry(
                            coilset,
                            NFP=coilset.NFP,
                            sym=coilset.sym,
                            check_intersection=False,
                        )
                        return [c for c in coilset_full]
                return [a for i in coilset for a in flatten_coils(i)]
            else:
                return [coilset]

        coils = flatten_coils(self)
        # after flatten, should have as many elements in list as self.num_coils, if
        # flatten worked correctly.
        assert len(coils) == self.num_coils

        assert (
            int(len(coils) / NFP) == len(coils) / NFP
        ), "Number of coils in coilset must be evenly divisible by NFP!"

        header = (
            # number of field period
            "periods "
            + str(NFP)
            + "\n"
            + "begin filament\n"
            # not 100% sure of what this line is, neither is MAKEGRID,
            # but it is needed and expected by other codes
            # "The third line is read by MAKEGRID but ignored"
            # https://princetonuniversity.github.io/STELLOPT/MAKEGRID.html
            + "mirror NIL"
        )
        footer = "end\n"

        x_arr = []
        y_arr = []
        z_arr = []
        currents_arr = []
        coil_end_inds = []  # indices where the coils end, need to track these
        # to place the coilgroup number and name later, which MAKEGRID expects
        # at the end of each individual coil
        if hasattr(grid, "endpoint"):
            endpoint = grid.endpoint
        elif isinstance(grid, numbers.Integral) or grid is None:
            # if int or None, will create a grid w/ endpoint=False in compute
            endpoint = False
        for i in range(int(len(coils))):
            coil = coils[i]
            coords = coil.compute("x", basis="xyz", grid=grid)["x"]

            contour_X = np.asarray(coords[0:, 0])
            contour_Y = np.asarray(coords[0:, 1])
            contour_Z = np.asarray(coords[0:, 2])

            currents = np.ones_like(contour_X) * float(coil.current)
            if endpoint:
                currents[-1] = 0  # this last point must have 0 current
            else:  # close the curves if needed
                contour_X = np.append(contour_X, contour_X[0])
                contour_Y = np.append(contour_Y, contour_Y[0])
                contour_Z = np.append(contour_Z, contour_Z[0])
                currents = np.append(currents, 0)  # this last point must have 0 current

            coil_end_inds.append(contour_X.size)

            x_arr.append(contour_X)
            y_arr.append(contour_Y)
            z_arr.append(contour_Z)
            currents_arr.append(currents)
        # form full array to save
        x_arr = np.concatenate(x_arr)
        y_arr = np.concatenate(y_arr)
        z_arr = np.concatenate(z_arr)
        currents_arr = np.concatenate(currents_arr)

        save_arr = np.vstack((x_arr, y_arr, z_arr, currents_arr)).T
        # save initial file
        np.savetxt(
            coilsFilename,
            save_arr,
            delimiter=" ",
            header=header,
            footer=footer,
            fmt="%14.12e",
            comments="",  # to avoid the # appended to the start of the header/footer
        )
        # now need to re-load the file and place coilgroup markers at end of each coil
        with open(coilsFilename) as f:
            lines = f.readlines()
        for i in range(len(coil_end_inds)):
            name = coils[i].name if coils[i].name != "" else "1 Modular"
            real_end_ind = int(
                np.sum(coil_end_inds[0 : i + 1]) + 2
            )  # to account for the 3 header lines
            lines[real_end_ind] = lines[real_end_ind].strip("\n") + f" {name}\n"
        with open(coilsFilename, "w") as f:
            f.writelines(lines)

    def to_FourierPlanar(
        self, N=10, grid=None, basis="xyz", name="", check_intersection=True
    ):
        """Convert all coils to FourierPlanarCoil.

        Note that some types of coils may not be representable in this basis.
        In this case, a least-squares fit will be done to find the
        planar coil that best represents the coil.

        Parameters
        ----------
        N : int
            Fourier resolution of the new FourierPlanarCoil representation.
        grid : Grid, int or None
            Grid used to evaluate curve coordinates on to fit with FourierPlanarCoil.
            If an integer, uses that many equally spaced points.
        basis : {'xyz', 'rpz'}
            Coordinate system for center and normal vectors. Default = 'xyz'.
        name : str
            Name for this coilset.
        check_intersection: bool
            Whether or not to check the coils in the new coilset for intersections.

        Returns
        -------
        coilset : CoilSet
            New representation of the coilset parameterized by a Fourier series for
            minor radius r in a plane specified by a center position and normal vector.

        """
        coils = [coil.to_FourierPlanar(N=N, grid=grid, basis=basis) for coil in self]
        return self.__class__(
            *coils,
            NFP=self.NFP,
            sym=self.sym,
            name=name,
            check_intersection=check_intersection,
        )

    def to_FourierXY(
        self, N=10, grid=None, s=None, basis="xyz", name="", check_intersection=True
    ):
        """Convert all coils to FourierXYCoil.

        Note that some types of coils may not be representable in this basis.
        In this case, a least-squares fit will be done to find the
        planar coil that best represents the coil.

        Parameters
        ----------
        N : int
            Fourier resolution of the new FourierXYCoil representation.
        grid : Grid, int or None
            Grid used to evaluate curve coordinates on to fit with FourierXYCoil.
            If an integer, uses that many equally spaced points.
        s : ndarray or "arclength"
            Arbitrary curve parameter to use for the fitting.
            Should be monotonic, 1D array of same length as
            coords. if None, defaults linearly spaced in [0,2pi)
            Alternative, can pass "arclength" to use normalized distance between points.
        basis : {'xyz', 'rpz'}
            Coordinate system for center and normal vectors. Default = 'xyz'.
        name : str
            Name for this coilset.
        check_intersection: bool
            Whether or not to check the coils in the new coilset for intersections.

        Returns
        -------
        coilset : CoilSet
            New representation of the coilset parameterized by Fourier series for the X
            & Y coordinates in a plane specified by a center position and normal vector.

        """
        coils = [coil.to_FourierXY(N=N, grid=grid, s=s, basis=basis) for coil in self]
        return self.__class__(
            *coils,
            NFP=self.NFP,
            sym=self.sym,
            name=name,
            check_intersection=check_intersection,
        )

    def to_FourierRZ(
        self, N=10, grid=None, NFP=None, sym=False, name="", check_intersection=True
    ):
        """Convert all coils to FourierRZCoil representation.

        Note that some types of coils may not be representable in this basis.

        Parameters
        ----------
        N : int
            Fourier resolution of the new R,Z representation.
        grid : Grid, int or None
            Grid used to evaluate curve coordinates on to fit with FourierRZCoil.
            If an integer, uses that many equally spaced points.
        NFP : int
            Number of field periods, the coil will have a discrete toroidal symmetry
            according to NFP.
        sym : bool, optional
            Whether the curve is stellarator-symmetric or not. Default is False.
        name : str
            Name for this coilset.
        check_intersection: bool
            Whether or not to check the coils in the new coilset for intersections.

        Returns
        -------
        coilset : CoilSet
            New representation of the coilset parameterized by a Fourier series for R,Z.

        """
        coils = [coil.to_FourierRZ(N=N, grid=grid, NFP=NFP, sym=sym) for coil in self]
        return self.__class__(
            *coils,
            NFP=self.NFP,
            sym=self.sym,
            name=name,
            check_intersection=check_intersection,
        )

    def to_FourierXYZ(self, N=10, grid=None, s=None, name="", check_intersection=True):
        """Convert all coils to FourierXYZCoil representation.

        Parameters
        ----------
        N : int
            Fourier resolution of the new X,Y,Z representation.
        grid : Grid, int or None
            Grid used to evaluate curve coordinates on to fit with FourierXYZCoil.
            If an integer, uses that many equally spaced points.
        s : ndarray
            Arbitrary curve parameter to use for the fitting. If None, defaults to
            normalized arclength.
        name : str
            Name for the new CoilSet.
        check_intersection: bool
            Whether or not to check the coils in the new coilset for intersections.

        Returns
        -------
        coilset : CoilSet
            New representation of the coil set parameterized by a Fourier series for
            X,Y,Z.

        """
        coils = [coil.to_FourierXYZ(N, grid, s) for coil in self]
        return self.__class__(
            *coils,
            NFP=self.NFP,
            sym=self.sym,
            name=name,
            check_intersection=check_intersection,
        )

    def to_SplineXYZ(
        self, knots=None, grid=None, method="cubic", name="", check_intersection=True
    ):
        """Convert all coils to SplineXYZCoil representation.

        Parameters
        ----------
        knots : ndarray
            Arbitrary curve parameter values to use for spline knots,
            should be an 1D ndarray of same length as the input.
            (Input length in this case is determined by grid argument, since
            the input coordinates come from Coil.compute("x",grid=grid))
            If None, defaults to using an equal-arclength angle as the knots.
            If supplied, will be rescaled to the range [0,2pi].
        grid : Grid, int or None
            Grid used to evaluate curve coordinates on to fit with SplineXYZCoil.
            If an integer, uses that many equally spaced points.
        method : str
            method of interpolation
            - `'nearest'`: nearest neighbor interpolation
            - `'linear'`: linear interpolation
            - `'cubic'`: C1 cubic splines (aka local splines)
            - `'cubic2'`: C2 cubic splines (aka natural splines)
            - `'catmull-rom'`: C1 cubic centripetal "tension" splines
        name : str
            Name for the new CoilSet.
        check_intersection: bool
            Whether or not to check the coils in the new coilset for intersections.

        Returns
        -------
        coilset : CoilSet
            New representation of the coilset parameterized by a spline for X,Y,Z.

        """
        coils = [coil.to_SplineXYZ(knots, grid, method) for coil in self]
        return self.__class__(
            *coils,
            NFP=self.NFP,
            sym=self.sym,
            name=name,
            check_intersection=check_intersection,
        )

    def is_self_intersecting(self, grid=None, tol=None):
        """Check if any coils in the CoilSet intersect.

        By default, checks intersection by checking that for each point on a given coil
        the closest point in the coilset is on that same coil. If the closest point is
        on another coil, that indicates that the coils may be close to intersecting.

        If instead the ``tol`` argument is provided, then the function will
        check the minimum distance from each coil to each other coil against
        that tol and if it finds the minimum distance is less than the ``tol``,
        it will take it as intersecting coils, returning True and raising a warning.

        NOTE: If grid resolution used is too low, this function may fail to return
        the correct answer.

        Parameters
        ----------
        grid : Grid, optional
            Collocation grid containing the nodes to evaluate the coil positions at.
            If a list, must have the same structure as the coilset. Defaults to a
            LinearGrid(N=100)
        tol : float, optional
            the tolerance (in meters) to check the intersections to, if points on any
            two coils are closer than this tolerance, then the function will return
            True and a warning will be raised. If not passed, then the method used
            to determine coilset intersection will be based off of checking that
            each point on a coil is closest to a point on the same coil, which does
            not rely on a ``tol`` parameter.

        Returns
        -------
        is_self_intersecting : bool
            Whether or not any coils in the CoilSet come close enough to each other to
            possibly be intersecting.

        """
        from desc.objectives._coils import CoilSetMinDistance

        grid = grid if grid else LinearGrid(N=100)
        obj = CoilSetMinDistance(self, grid=grid)
        obj.build(verbose=0)
        if tol:
            min_dists = obj.compute(self.params_dict)
            is_nearly_intersecting = np.any(min_dists < tol)
            warnif(
                is_nearly_intersecting,
                UserWarning,
                "Found coils which are nearly intersecting according to the given tol "
                + "(min coil-coil distance = "
                + f"{np.min(min_dists):1.3e} m < {tol:1.3e} m)"
                + " in the coilset, it is recommended to check coils closely.",
            )
            return is_nearly_intersecting
        else:

            pts = obj._constants["coilset"]._compute_position(
                params=self.params_dict, grid=obj._constants["grid"], basis="xyz"
            )
            pts = np.array(pts)
            num_nodes = pts.shape[1]
            bad_coil_inds = []
            # We will raise the warning if the jth point on the
            # kth coil is closer to a point on a different coil than
            # it is to the neighboring points on itself
            for k in range(self.num_coils):
                # dist[i,j,n] is the distance from the jth point on the kth coil
                # to the nth point on the ith coil
                dist = np.asarray(
                    safenorm(pts[k][None, :, None] - pts[:, None, :], axis=-1)
                )
                for j in range(num_nodes):
                    dists_for_this_pt = dist[:, j, :].copy()
                    dists_for_this_pt[k][
                        j
                    ] = np.inf  # Set the dist from the pt to itself to inf to ignore
                    ind_min = np.argmin(dists_for_this_pt)
                    # check if the index returned corresponds to a point on the same
                    # coil. if it does not, then this jth pt on the kth coil is closer
                    #  to a point on another coil than it is to pts on its own coil,
                    # which means it may be intersecting it.
                    if ind_min not in np.arange((num_nodes) * k, (num_nodes) * (k + 1)):
                        bad_coil_inds.append(k)
            bad_coil_inds = set(bad_coil_inds)
            is_nearly_intersecting = True if bad_coil_inds else False
            warnif(
                is_nearly_intersecting,
                UserWarning,
                "Found coils which are nearly intersecting according to the given grid"
                + " it is recommended to check coils closely or run function "
                + "again with a higher resolution grid."
                + f" Offending coil indices are {bad_coil_inds}.",
            )
            return is_nearly_intersecting

    def __add__(self, other):
        if isinstance(other, (CoilSet)):
            return CoilSet(*self.coils, *other.coils)
        if isinstance(other, (list, tuple)):
            return CoilSet(*self.coils, *other)
        else:
            return NotImplemented

    # dunder methods required by MutableSequence
    def __getitem__(self, i):
        return self.coils[i]

    def __setitem__(self, i, new_item):
        if not isinstance(new_item, _Coil):
            raise TypeError("Members of CoilSet must be of type Coil.")
        _check_type(new_item, self[0])
        self._coils[i] = new_item

    def __delitem__(self, i):
        del self._coils[i]

    def __len__(self):
        return len(self._coils)

    def insert(self, i, new_item):
        """Insert a new coil into the coilset at position i."""
        if not isinstance(new_item, _Coil):
            raise TypeError("Members of CoilSet must be of type Coil.")
        _check_type(new_item, self[0])
        self._coils.insert(i, new_item)

    def __repr__(self):
        """Get the string form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (name={}, with {} submembers)".format(self.name, len(self))
        )


class MixedCoilSet(CoilSet):
    """Set of coils or coilsets of different geometry.

    Parameters
    ----------
    coils : Coil or array-like of Coils
        Collection of coils.
    name : str
        Name of this CoilSet.
    check_intersection: bool
        Whether or not to check the coils in the coilset for intersections.

    """

    _io_attrs_ = CoilSet._io_attrs_

    def __init__(self, *coils, name="", check_intersection=True):
        coils = flatten_list(coils, flatten_tuple=True)
        assert all([isinstance(coil, (_Coil)) for coil in coils])
        self._coils = list(coils)
        self._NFP = 1
        self._sym = False
        self._name = str(name)
        if check_intersection:
            self.is_self_intersecting()

    @property
    def num_coils(self):
        """int: Number of coils."""
        return sum([c.num_coils for c in self])

    def _all_currents(self, currents=None):
        """Return an array of all the currents (including those in virtual coils)."""
        if currents is None:
            currents = jnp.array(flatten_list(self.current))
        all_currents = []
        i = 0
        for coil in self.coils:
            if isinstance(coil, CoilSet):
                curr = currents[i : i + len(coil)]
                all_currents += [coil._all_currents(curr)]
                i += len(coil)
            else:
                all_currents += [jnp.atleast_1d(currents[i])]
                i += 1
        return jnp.concatenate(all_currents)

    def compute(
        self,
        names,
        grid=None,
        params=None,
        transforms=None,
        data=None,
        **kwargs,
    ):
        """Compute the quantity given by name on grid, for each coil in the coilset.

        Parameters
        ----------
        names : str or array-like of str
            Name(s) of the quantity(s) to compute.
        grid : Grid or int or array-like, optional
            Grid of coordinates to evaluate at. Defaults to a Linear grid.
            If an integer, uses that many equally spaced points.
            If array-like, should be 1 value per coil.
        params : dict of ndarray or array-like
            Parameters from the equilibrium. Defaults to attributes of self.
            If array-like, should be 1 value per coil.
        transforms : dict of Transform or array-like
            Transforms for R, Z, lambda, etc. Default is to build from grid.
            If array-like, should be 1 value per coil.
        data : dict of ndarray or array-like
            Data computed so far, generally output from other compute functions
            If array-like, should be 1 value per coil.

        Returns
        -------
        data : list of dict of ndarray
            Computed quantity and intermediate variables, for each coil in the set.
            List entries map to coils in coilset, each dict contains data for an
            individual coil.

        """
        grid = self._make_arraylike(grid)
        params = self._make_arraylike(params)
        transforms = self._make_arraylike(transforms)
        data = self._make_arraylike(data)
        return [
            coil.compute(
                names, grid=grd, params=par, transforms=tran, data=dat, **kwargs
            )
            for (coil, grd, par, tran, dat) in zip(
                self.coils, grid, params, transforms, data
            )
        ]

    def _compute_position(self, params=None, grid=None, dx1=False, **kwargs):
        """Compute coil positions accounting for stellarator symmetry.

        Parameters
        ----------
        params : dict or array-like of dict, optional
            Parameters to pass to coils, either the same for all coils or one for each.
        grid : Grid or int or array-like, optional
            Grid of coordinates to evaluate at. Defaults to a Linear grid.
            If an integer, uses that many equally spaced points.
            If array-like, should be 1 value per coil.
        dx1 : bool
            If True, also return dx/ds for each curve.

        Returns
        -------
        x : ndarray, shape(len(self),source_grid.num_nodes,3)
            Coil positions, in [R,phi,Z] or [X,Y,Z] coordinates.
        x_s : ndarray, shape(len(self),source_grid.num_nodes,3)
            Coil position derivatives, in [R,phi,Z] or [X,Y,Z] coordinates.
            Only returned if dx1=True.

        """
        errorif(
            grid is None,
            ValueError,
            "grid must be supplied to MixedCoilSet._compute_position, since the "
            + "default grid for each coil could have a different number of nodes.",
        )
        kwargs.setdefault("basis", "xyz")
        params = self._make_arraylike(params)
        grid = self._make_arraylike(grid)
        out = []
        for coil, par, grd in zip(self.coils, params, grid):
            out.append(coil._compute_position(par, grd, dx1, **kwargs))
        if dx1:
            x = jnp.vstack([foo[0] for foo in out])
            x_s = jnp.vstack([foo[1] for foo in out])
            return x, x_s
        return jnp.vstack(out)

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
        params : dict or array-like of dict, optional
            Parameters to pass to coils, either the same for all coils or one for each.
            If array-like, should be 1 value per coil.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic field.
        source_grid : Grid, int or None or array-like, optional
            Grid used to discretize coils. If an integer, uses that many equally spaced
            points. Should NOT include endpoint at 2pi.
            If array-like, should be 1 value per coil.
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
            magnetic field or vector potential at specified points, in either rpz
            or xyz coordinates

        """
        errorif(
            compute_A_or_B not in ["A", "B"],
            ValueError,
            f'Expected "A" or "B" for compute_A_or_B, instead got {compute_A_or_B}',
        )
        params = self._make_arraylike(params)
        source_grid = self._make_arraylike(source_grid)
        transforms = self._make_arraylike(transforms)

        AB = 0
        if compute_A_or_B == "B":
            for coil, par, grd, tr in zip(self.coils, params, source_grid, transforms):
                AB += coil.compute_magnetic_field(
                    coords, par, basis, grd, transforms=tr, chunk_size=chunk_size
                )
        elif compute_A_or_B == "A":
            for coil, par, grd, tr in zip(self.coils, params, source_grid, transforms):
                AB += coil.compute_magnetic_vector_potential(
                    coords, par, basis, grd, transforms=tr, chunk_size=chunk_size
                )
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
            Parameters to pass to coils, either the same for all coils or one for each.
            If array-like, should be 1 value per coil.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic field.
        source_grid : Grid, int or None or array-like, optional
            Grid used to discretize coils. If an integer, uses that many equally spaced
            points. Should NOT include endpoint at 2pi.
            If array-like, should be 1 value per coil.
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
            Nodes to evaluate potential at in [R,phi,Z] or [X,Y,Z] coordinates.
        params : dict or array-like of dict, optional
            Parameters to pass to coils, either the same for all coils or one for each.
            If array-like, should be 1 value per coil.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic field.
        source_grid : Grid, int or None or array-like, optional
            Grid used to discretize coils. If an integer, uses that many equally spaced
            points. Should NOT include endpoint at 2pi.
            If array-like, should be 1 value per coil.
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

    def to_FourierPlanar(
        self, N=10, grid=None, basis="xyz", name="", check_intersection=True
    ):
        """Convert all coils to FourierPlanarCoil representation.

        Note that some types of coils may not be representable in this basis.
        In this case, a least-squares fit will be done to find the
        planar coil that best represents the coil.

        Parameters
        ----------
        N : int
            Fourier resolution of the new FourierPlanarCoil representation.
        grid : Grid, int or None
            Grid used to evaluate curve coordinates on to fit with FourierPlanarCoil.
            If an integer, uses that many equally spaced points.
        basis : {'xyz', 'rpz'}
            Coordinate system for center and normal vectors. Default = 'xyz'.
        name : str
            Name for the new MixedCoilSet.
        check_intersection: bool
            Whether or not to check the coils in the new coilset for intersections.

        Returns
        -------
        coilset : MixedCoilSet
            New representation of the coilset parameterized by a Fourier series for
            minor radius r in a plane specified by a center position and normal vector.

        """
        coils = [
            coil.to_FourierPlanar(
                N=N, grid=grid, basis=basis, check_intersection=check_intersection
            )
            for coil in self
        ]
        return self.__class__(*coils, name=name, check_intersection=check_intersection)

    def to_FourierXY(
        self, N=10, grid=None, s=None, basis="xyz", name="", check_intersection=True
    ):
        """Convert all coils to FourierXYCoil.

        Note that some types of coils may not be representable in this basis.
        In this case, a least-squares fit will be done to find the
        planar coil that best represents the coil.

        Parameters
        ----------
        N : int
            Fourier resolution of the new FourierXYCoil representation.
        grid : Grid, int or None
            Grid used to evaluate curve coordinates on to fit with FourierXYCoil.
            If an integer, uses that many equally spaced points.
        s : ndarray or "arclength"
            Arbitrary curve parameter to use for the fitting.
            Should be monotonic, 1D array of same length as
            coords. if None, defaults linearly spaced in [0,2pi)
            Alternative, can pass "arclength" to use normalized distance between points.
        basis : {'xyz', 'rpz'}
            Coordinate system for center and normal vectors. Default = 'xyz'.
        name : str
            Name for this coilset.
        check_intersection: bool
            Whether or not to check the coils in the new coilset for intersections.

        Returns
        -------
        coilset : CoilSet
            New representation of the coilset parameterized by Fourier series for the X
            & Y coordinates in a plane specified by a center position and normal vector.

        """
        coils = [
            coil.to_FourierXY(
                N=N, grid=grid, s=s, basis=basis, check_intersection=check_intersection
            )
            for coil in self
        ]
        return self.__class__(*coils, name=name, check_intersection=check_intersection)

    def to_FourierRZ(
        self, N=10, grid=None, NFP=None, sym=False, name="", check_intersection=True
    ):
        """Convert all coils to FourierRZCoil representation.

        Note that some types of coils may not be representable in this basis.

        Parameters
        ----------
        N : int
            Fourier resolution of the new R,Z representation.
        grid : Grid, int or None
            Grid used to evaluate curve coordinates on to fit with FourierRZCoil.
            If an integer, uses that many equally spaced points.
        NFP : int
            Number of field periods, the coil will have a discrete toroidal symmetry
            according to NFP.
        sym : bool, optional
            Whether the curve is stellarator-symmetric or not. Default is False.
        name : str
            Name for the new MixedCoilSet.
        check_intersection: bool
            Whether or not to check the coils in the new coilset for intersections.

        Returns
        -------
        coilset : MixedCoilSet
            New representation of the coilset parameterized by a Fourier series for R,Z.

        """
        coils = [
            coil.to_FourierRZ(
                N=N, grid=grid, NFP=NFP, sym=sym, check_intersection=check_intersection
            )
            for coil in self
        ]
        return self.__class__(*coils, name=name, check_intersection=check_intersection)

    def to_FourierXYZ(self, N=10, grid=None, s=None, name="", check_intersection=True):
        """Convert all coils to FourierXYZCoil representation.

        Parameters
        ----------
        N : int
            Fourier resolution of the new X,Y,Z representation.
        grid : Grid, int or None
            Grid used to evaluate curve coordinates on to fit with FourierXYZCoil.
            If an integer, uses that many equally spaced points.
        s : ndarray
            Arbitrary curve parameter to use for the fitting. If None, defaults to
            normalized arclength.
        name : str
            Name for the new MixedCoilSet.
        check_intersection: bool
            Whether or not to check the coils in the new coilset for intersections.

        Returns
        -------
        coilset : MixedCoilSet
            New representation of the coil set parameterized by a Fourier series for
            X,Y,Z.

        """
        coils = [
            coil.to_FourierXYZ(N, grid, s, check_intersection=check_intersection)
            for coil in self
        ]
        return self.__class__(*coils, name=name, check_intersection=check_intersection)

    def to_SplineXYZ(
        self, knots=None, grid=None, method="cubic", name="", check_intersection=True
    ):
        """Convert all coils to SplineXYZCoil representation.

        Parameters
        ----------
        knots : ndarray
            Arbitrary curve parameter values to use for spline knots,
            should be an 1D ndarray of same length as the input.
            (Input length in this case is determined by grid argument, since
            the input coordinates come from Coil.compute("x",grid=grid))
            If None, defaults to using an equal-arclength angle as the knots.
            If supplied, will be rescaled to the range [0,2pi].
        grid : Grid, int or None
            Grid used to evaluate curve coordinates on to fit with SplineXYZCoil.
            If an integer, uses that many equally spaced points.
        method : str
            method of interpolation
            - `'nearest'`: nearest neighbor interpolation
            - `'linear'`: linear interpolation
            - `'cubic'`: C1 cubic splines (aka local splines)
            - `'cubic2'`: C2 cubic splines (aka natural splines)
            - `'catmull-rom'`: C1 cubic centripetal "tension" splines
        name : str
            Name for the new MixedCoilSet.
        check_intersection: bool
            Whether or not to check the coils in the new coilset for intersections.

        Returns
        -------
        coilset : MixedCoilSet
            New representation of the coilset parameterized by a spline for X,Y,Z.

        """
        coils = [
            coil.to_SplineXYZ(
                knots, grid, method, check_intersection=check_intersection
            )
            for coil in self
        ]
        return self.__class__(*coils, name=name, check_intersection=check_intersection)

    def __add__(self, other):
        if isinstance(other, (CoilSet, MixedCoilSet)):
            return MixedCoilSet(*self.coils, *other.coils)
        if isinstance(other, (list, tuple)):
            return MixedCoilSet(*self.coils, *other)
        else:
            return NotImplemented

    def __setitem__(self, i, new_item):
        if not isinstance(new_item, _Coil):
            raise TypeError("Members of CoilSet must be of type Coil.")
        self._coils[i] = new_item

    def insert(self, i, new_item):
        """Insert a new coil into the coilset at position i."""
        if not isinstance(new_item, _Coil):
            raise TypeError("Members of CoilSet must be of type Coil.")
        self._coils.insert(i, new_item)

    @classmethod
    def from_makegrid_coilfile(  # noqa: C901
        cls, coil_file, method="cubic", ignore_groups=False, check_intersection=True
    ):
        """Create a MixedCoilSet of SplineXYZCoils from a MAKEGRID coil txtfile.

        If ignore_groups=False and the MAKEGRID contains more than one coil group
        (denoted by the number listed after the current on the last line defining a
        given coil), this function will try to return a MixedCoilSet of CoilSets, with
        each sub CoilSet pertaining to the different coil groups. If the coils in a
        group have differing numbers of knots, then it will return MixedCoilSets
        instead. The name of the sub (Mixed)CoilSet will be the number and the name
        of the group listed in the MAKEGRID file.

        Parameters
        ----------
        coil_file : str or path-like
            path to coil file in txt format
        method : str
            method of interpolation

            - ``'nearest'``: nearest neighbor interpolation
            - ``'linear'``: linear interpolation
            - ``'cubic'``: C1 cubic splines (aka local splines)
            - ``'cubic2'``: C2 cubic splines (aka natural splines)
            - ``'catmull-rom'``: C1 cubic centripetal "tension" splines
            - ``'cardinal'``: C1 cubic general tension splines. If used, default tension
              of c = 0 will be used
            - ``'monotonic'``: C1 cubic splines that attempt to preserve monotonicity in
              the data, and will not introduce new extrema in the interpolated points
            - ``'monotonic-0'``: same as `'monotonic'` but with 0 first derivatives at
              both endpoints
        ignore_groups : bool
            If False, return the coils in a nested MixedCoilSet, with a sub coilset per
            single coilgroup. If there is only a single group, however, this will not
            return a nested coilset, but just a single coilset for that group. if True,
            return the coils as just a single MixedCoilSet.
        check_intersection : bool
            whether to check the resulting coilsets for intersecting coils.


        """
        coils = {}  # dict of list of SplineXYZCoils, one list per coilgroup
        coilinds = [2]  # List of line indices where coils are at in the file.
        # always start at the 3rd line after periods
        coilnames = []  # the coilgroup each coil belongs to
        # corresponds to each coil in the coilinds list
        groupnames = []  # this is the groupind + the name of the first coil in
        # the group
        groupinds = []  # the coilgroup ind each coil belongs to
        # corresponds to each coil in the coilinds list
        # (sometimes, coils in the same group could have different names,
        # so this separately tracks just the number of the group)

        # read in the coils file
        headind = -1
        with open(coil_file) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.find("periods") != -1:
                    headind = i  # skip anything that is above the periods line
                    coilinds[0] += headind
                    if len(lines[3 + headind].split()) != 4:
                        raise OSError(
                            "4th line in file must be the start of the first coil! "
                            + "Expected a line of length 4 (after .split()), "
                            + f"instead got length {lines[3+headind].split()}"
                        )
                    header_lines_not_as_expected = np.array(
                        [
                            len(lines[0 + headind].split()) != 2,
                            len(lines[1 + headind].split()) != 2,
                            len(lines[2 + headind].split()) != 2,
                        ]
                    )
                    if np.any(header_lines_not_as_expected):
                        wronglines = lines[
                            np.where(header_lines_not_as_expected)[0] + headind
                        ]
                        raise OSError(
                            "First 3 lines in file starting with the periods line "
                            + "must be the header lines,"
                            + " each of length 2 (after .split())! "
                            + f"Line(s) {wronglines}"
                            + " are not length 2"
                        )

                    continue
                if (
                    line.find("begin filament") != -1
                    or line.find("end") != -1
                    or line.find("mirror") != -1
                ):
                    continue  # skip headers and last line
                if (
                    len(line.split()) != 4  # find the line immediately before a coil,
                    # where the line length is greater than 4
                    and line.strip()  # ensure not counting blank lines
                    # if we have not found the header yet, skip the line
                    and headind != -1
                ):
                    coilinds.append(i)
                    groupname = " ".join(line.split()[4:])
                    groupind = int(groupname.split()[0].strip())
                    if groupind not in coils.keys():
                        coils[groupind] = []
                        groupnames.append(groupname)
                    coilnames.append(groupname)
                    groupinds.append(groupind)

        for i, (start, end, groupind, coilname) in enumerate(
            zip(coilinds[0:-1], coilinds[1:], groupinds, coilnames)
        ):
            coords = np.genfromtxt(lines[start + 1 : end])
            coils[groupind].append(
                SplineXYZCoil(
                    coords[:, -1][0],
                    coords[:, 0],
                    coords[:, 1],
                    coords[:, 2],
                    method=method,
                    name=coilname,
                )
            )

        def flatten_coils(coilset):
            # helper function for flattening coilset
            if hasattr(coilset, "__len__"):
                return [a for i in coilset for a in flatten_coils(i)]
            else:
                return [coilset]

        # if it is a single group, then only return one coilset, not a
        # nested coilset
        groupinds = list(coils.keys())
        if len(groupinds) == 1:
            return cls(
                *coils[groupinds[0]],
                name=groupnames[0],
                check_intersection=check_intersection,
            )

        # if not, possibly return a nested coilset, containing one coilset per coilgroup
        coilsets = []  # list of coilsets, so we can attempt to use CoilSet for each one
        for groupname, groupind in zip(groupnames, groupinds):
            try:
                # try making the coilgroup use a CoilSet
                coilsets.append(
                    CoilSet(
                        *coils[groupind],
                        name=groupname,
                        check_intersection=check_intersection,
                    )
                )
            except ValueError:  # can't load as a CoilSet if any of the coils have
                # different length of knots, so load as MixedCoilSet instead
                coilsets.append(
                    cls(
                        *coils[groupind],
                        name=groupname,
                        check_intersection=check_intersection,
                    )
                )
        cset = cls(*coilsets, check_intersection=check_intersection)
        if ignore_groups:
            cset = cls(*flatten_coils(cset), check_intersection=check_intersection)
        return cset


@partial(jnp.vectorize, signature="(m,3),(n,3),(m,3),(n,3),(m),(n)->()")
def _linking_number(x1, x2, x1_s, x2_s, dx1, dx2):
    """Linking number between curves x1 and x2 with tangents x1_s, x2_s."""
    x1_s *= dx1[:, None]
    x2_s *= dx2[:, None]
    dx = x1[:, None, :] - x2[None, :, :]  # shape(m,n,3)
    dx_norm = safenorm(dx, axis=-1)  # shape(m,n)
    den = dx_norm**3
    dr1xdr2 = cross(x1_s[:, None, :], x2_s[None, :, :], axis=-1)  # shape(m,n,3)
    num = dot(dx, dr1xdr2, axis=-1)  # shape(m,n)
    small = dx_norm < jnp.finfo(x1.dtype).eps
    ratio = jnp.where(small, 0.0, num / jnp.where(small, 1.0, den))
    return ratio.sum()


def initialize_modular_coils(eq, num_coils, r_over_a=2.0):
    """Initialize a CoilSet of modular coils for stage 2 optimization.

    The coils will be planar, circular coils centered on the equilibrium magnetic axis,
    and aligned such that the normal to the coil points along the axis. The currents
    will be set to match the equilibrium required poloidal linking current.

    The coils will be ``FourierPlanarCoil`` with N=0, if another type is desired use
    ``coilset.to_FourierXYZ(N=10)``, ``coilset.to_SplineXYZ()`` etc.

    Parameters
    ----------
    eq : Equilibrium
        Stage 1 equilibrium the coils are being optimized for.
    num_coils : int
        Number of coils to create per field period. For stellarator symmetric
        equilibria, this will be the number of coils per half-period.
    r_over_a : float
        Minor radius of the coils, in units of equilibrium minor radius. Note that for
        strongly shaped equilibria this may need to be large to avoid having the coils
        intersect the plasma.

    Returns
    -------
    coilset : CoilSet of FourierPlanarCoil
        Planar coils centered on magnetic axis, with appropriate symmetry.
    """
    extent = 2 * np.pi / (eq.NFP * (eq.sym + 1))
    zeta = np.linspace(0, extent, num_coils, endpoint=False) + extent / (2 * num_coils)
    grid = LinearGrid(rho=[0.0], M=0, zeta=zeta, NFP=eq.NFP)

    minor_radius = eq.compute("a")["a"]
    G = eq.compute("G", grid=LinearGrid(rho=1.0))["G"]
    data = eq.axis.compute(["x", "x_s"], grid=grid, basis="rpz")

    centers = data["x"]  # center coils on axis position
    normals = data["x_s"]  # make normal to coil align with tangent along axis

    unique_coils = []
    for k in range(num_coils):
        coil = FourierPlanarCoil(
            current=2 * np.pi * G / (mu_0 * eq.NFP * num_coils * (eq.sym + 1)),
            center=centers[k, :],
            normal=normals[k, :],
            r_n=minor_radius * r_over_a,
            basis="rpz",
        )
        unique_coils.append(coil)
    coilset = CoilSet(unique_coils, NFP=eq.NFP, sym=eq.sym)
    return coilset


def initialize_saddle_coils(eq, num_coils, r_over_a=0.5, offset=2.0, position="outer"):
    """Initialize a CoilSet of saddle coils for stage 2 optimization.

    The coils will be planar, circular coils positioned around the plasma without
    linking it, and aligned such that the normal to the coil points towards the
    magnetic axis. The currents will be initialized to zero.

    The coils will be ``FourierPlanarCoil`` with N=0, if another type is desired use
    ``coilset.to_FourierXYZ(N=10)``, ``coilset.to_SplineXYZ()`` etc.

    Parameters
    ----------
    eq : Equilibrium
        Stage 1 equilibrium the coils are being optimized for.
    num_coils : int
        Number of coils to create per field period. For stellarator symmetric
        equilibria, this will be the number of coils per half-period.
    r_over_a : float
        Minor radius of the coils, in units of equilibrium minor radius.
    offset : float
        Distance from coil to magnetic axis, in units of equilibrium minor radius.
        Note that for strongly shaped equilibria this may need to be large to avoid
        having the coils intersect the plasma.
    position : {"outer", "inner", "top", "bottom"}
        Placement of coils relative to plasma. "outer" will place coils on the outboard
        side, "inner" on the inboard side, "top" will place coils above the plasma,
        "bottom" will place them below.

    Returns
    -------
    coilset : CoilSet of FourierPlanarCoil
        Planar coils centered on magnetic axis, with appropriate symmetry.
    """
    errorif(
        position not in {"outer", "inner", "top", "bottom"},
        ValueError,
        f"position must be one of 'outer', 'inner'', 'top', 'bottom', got {position}",
    )
    extent = 2 * np.pi / (eq.NFP * (eq.sym + 1))
    zeta = np.linspace(0, extent, num_coils, endpoint=False) + extent / (2 * num_coils)
    grid = LinearGrid(rho=[0.0], M=0, zeta=zeta, NFP=eq.NFP)

    minor_radius = eq.compute("a")["a"]
    data = eq.axis.compute(["x", "x_s"], grid=grid, basis="rpz")

    centers = data["x"]  # center coils on axis position
    normals = data["x_s"]  # make normal to coil align with tangent along axis

    offset_vecs = {
        "outer": np.array([1, 0, 0]),
        "inner": np.array([-1, 0, 0]),
        "top": np.array([0, 0, 1]),
        "bottom": np.array([0, 0, -1]),
    }
    normal_vecs = {
        "outer": np.array([0, 0, -1]),
        "inner": np.array([0, 0, 1]),
        "top": np.array([1, 0, 0]),
        "bottom": np.array([-1, 0, 0]),
    }

    windowpane_coils = []
    for k in range(num_coils):
        coil = FourierPlanarCoil(
            current=0.0,
            center=centers[k, :] + offset_vecs[position] * offset * minor_radius,
            normal=np.cross(normals[k, :], normal_vecs[position]),
            r_n=minor_radius * r_over_a,
            basis="rpz",
        )
        windowpane_coils.append(coil)

    windowpane_coilset = CoilSet(windowpane_coils, NFP=int(eq.NFP), sym=eq.sym)
    return windowpane_coilset


def initialize_helical_coils(eq, num_coils, r_over_a=2.0, helicity=(1, 1), npts=100):
    """Initialize a CoilSet of helical coils for stage 2 optimization.

    The coils will be roughly a constant distance from the plasma surface as they wind
    around. The currents will be set to match the equilibrium required poloidal
    linking current.

    The coils will be ``SplineXYZCoil``, if another type is desired use
    ``coilset.to_FourierXYZ(N=...)``, etc.

    Parameters
    ----------
    eq : Equilibrium
        Stage 1 equilibrium the coils are being optimized for.
    num_coils : int
        Number of coils to create.
    r_over_a : float
        Approximate minor radius of the coils, in units of equilibrium minor radius.
        The approximate minor radius will be r_over_a * equilibrium minor radius. The
        actual coils will be contoured to keep a roughly constant offset from the plasma
        surface.
    helicity : tuple of int
        (M,N) - How many times each coil should link the plasma poloidally and
        toroidally. Note that M is the poloidal linking number per field period, so the
        total linking number will be M*NFP.
    npts : int
        How many points to use when creating the coils. Equilibria with very high NFP
        may need more points.

    Returns
    -------
    coilset : CoilSet of FourierPlanarCoil
        Planar coils centered on magnetic axis, with appropriate symmetry.
    """
    M = helicity[0] * eq.NFP
    N = helicity[1]

    errorif(
        int(M) != M or int(N) != N,
        TypeError,
        f"helicity should be a tuple of two integers, got {helicity}",
    )
    warnif(
        N == 0 and M != 0,
        UserWarning,
        "Toroidal helicity is zero, meaning these are modular coils. Consider using "
        "desc.coils.initialize_modular_coils",
    )
    warnif(
        N == 0 and M == 0,
        UserWarning,
        "Toroidal and poloidal helicity are zero, meaning these are windowpane/saddle "
        "coils. Consider using desc.coils.initialize_saddle_coils",
    )
    # for M=0 these are technically PF coils not helical, but we don't have a specific
    # function for PF coils so don't bother warning.

    a = eq.compute("a")["a"]
    G = eq.compute("G", grid=LinearGrid(rho=1.0))["G"]
    s = np.linspace(0, 2 * np.pi, npts, endpoint=False)

    theta = M * s
    zeta = N * s

    theta %= 2 * np.pi
    zeta %= 2 * np.pi

    theta_offset = np.linspace(0, 2 * np.pi, num_coils, endpoint=False)

    coils = []
    for t in theta_offset:
        grid = Grid(np.array([np.ones_like(s), (theta + t) % (2 * np.pi), zeta]).T)
        data = eq.surface.compute(["x", "n_rho"], grid=grid, basis="xyz")
        offset = r_over_a * a - a
        x = data["x"] + offset * data["n_rho"]
        coil = SplineXYZCoil(
            2 * np.pi * G / mu_0 / num_coils / M, x[:, 0], x[:, 1], x[:, 2]
        )
        coils.append(coil)
    return CoilSet(*coils)
