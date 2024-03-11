"""Classes for magnetic field coils."""

import numbers
from abc import ABC
from collections.abc import MutableSequence

import numpy as np

from desc.backend import fori_loop, jit, jnp, scan, tree_stack, tree_unstack, vmap
from desc.compute import get_params, rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec
from desc.compute.geom_utils import reflection_matrix
from desc.geometry import (
    FourierPlanarCurve,
    FourierRZCurve,
    FourierXYZCurve,
    SplineXYZCurve,
)
from desc.grid import LinearGrid
from desc.magnetic_fields import _MagneticField
from desc.optimizable import Optimizable, OptimizableCollection, optimizable_parameter
from desc.utils import equals, errorif, flatten_list, warnif


@jit
def biot_savart_hh(eval_pts, coil_pts_start, coil_pts_end, current):
    """Biot-Savart law for filamentary coils following [1].

    The coil is approximated by a series of straight line segments
    and an analytic expression is used to evaluate the field from each
    segment.

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

    Returns
    -------
    B : ndarray, shape(n,3)
        magnetic field in cartesian components at specified points

    [1] Hanson & Hirshman, "Compact expressions for the Biot-Savart
    fields of a filamentary segment" (2002)
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
        2.0e-7  #  == 2 * mu_0/(4 pi)
        * current
        * Ri_p_Rf
        / (Ri * Rf * (Ri_p_Rf * Ri_p_Rf - (L * L)[:, jnp.newaxis]))
    )

    # cross product of L*hat(eps)==d_vec with Ri_vec, scaled by B_mag
    vec = jnp.cross(d_vec[:, jnp.newaxis, :], Ri_vec, axis=-1)
    B = jnp.sum(B_mag[:, :, jnp.newaxis] * vec, axis=0)
    return B


@jit
def biot_savart_quad(eval_pts, coil_pts, tangents, current):
    """Biot-Savart law for filamentary coil using numerical quadrature.

    Parameters
    ----------
    eval_pts : array-like shape(n,3)
        Evaluation points in cartesian coordinates
    coil_pts : array-like shape(m,3)
        Points in cartesian space defining coil
    tangents : array-like, shape(m,3)
        Tangent vectors to the coil at coil_pts. If the curve is given
        by x(s) with curve parameter s, coil_pts = x, tangents = dx/ds*ds where
        ds is the spacing between points.
    current : float
        Current through the coil (in Amps).

    Returns
    -------
    B : ndarray, shape(n,3)
        magnetic field in cartesian components at specified points

    Notes
    -----
    This method does not give curl(B) == 0 exactly. The error in the curl
    scales the same as the error in B itself, so will only be zero when fully
    converged. However in practice, for smooth curves described by Fourier series,
    this method converges exponentially in the number of coil points.
    """
    dl = tangents
    R_vec = eval_pts[jnp.newaxis, :] - coil_pts[:, jnp.newaxis, :]
    R_mag = jnp.linalg.norm(R_vec, axis=-1)

    vec = jnp.cross(dl[:, jnp.newaxis, :], R_vec, axis=-1)
    denom = R_mag**3

    # 1e-7 == mu_0/(4 pi)
    B = jnp.sum(1.0e-7 * current * vec / denom[:, :, None], axis=0)
    return B


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
        current passing through the coil, in Amperes
    """

    _io_attrs_ = _MagneticField._io_attrs_ + ["_current"]

    def __init__(self, current, *args, **kwargs):
        self._current = float(current)
        super().__init__(*args, **kwargs)

    @optimizable_parameter
    @property
    def current(self):
        """float: Current passing through the coil, in Amperes."""
        return self._current

    @current.setter
    def current(self, new):
        assert jnp.isscalar(new) or new.size == 1
        self._current = float(new)

    def compute_magnetic_field(
        self, coords, params=None, basis="rpz", source_grid=None
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
        assert basis.lower() in ["rpz", "xyz"]
        coords = jnp.atleast_2d(jnp.asarray(coords))
        if basis.lower() == "rpz":
            phi = coords[:, 1]
            coords = rpz2xyz(coords)
        if params is None:
            current = self.current
        else:
            current = params.pop("current", self.current)

        data = self.compute(
            ["x", "x_s", "ds"], grid=source_grid, params=params, basis="xyz"
        )
        B = biot_savart_quad(
            coords, data["x"], data["x_s"] * data["ds"][:, None], current
        )

        if basis.lower() == "rpz":
            B = xyz2rpz_vec(B, phi=phi)
        return B

    def __repr__(self):
        """Get the string form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (name={}, current={})".format(self.name, self.current)
        )

    def to_FourierXYZ(self, N=10, grid=None, s=None, name=""):
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
            name for this coil

        Returns
        -------
        coil : FourierXYZCoil
            New representation of the coil parameterized by Fourier series for X,Y,Z.

        """
        if (grid is None) and (s is not None) and (not isinstance(s, str)):
            grid = LinearGrid(zeta=s)
        coords = self.compute("x", grid=grid, basis="xyz")["x"]
        return FourierXYZCoil.from_values(
            self.current, coords, N=N, s=s, basis="xyz", name=name
        )

    def to_SplineXYZ(self, knots=None, grid=None, method="cubic", name=""):
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
            name for this coil

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


class FourierRZCoil(_Coil, FourierRZCurve):
    """Coil parameterized by fourier series for R,Z in terms of toroidal angle phi.

    Parameters
    ----------
    current : float
        current through coil, in Amperes
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
        name for this coil

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


class FourierXYZCoil(_Coil, FourierXYZCurve):
    """Coil parameterized by fourier series for X,Y,Z in terms of arbitrary angle s.

    Parameters
    ----------
    current : float
        current through coil, in Amperes
    X_n, Y_n, Z_n: array-like
        fourier coefficients for X, Y, Z
    modes : array-like
        mode numbers associated with X_n etc.
    name : str
        name for this coil

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
            Current through the coil, in Amps.
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
        Returns
        -------
        coil : FourierXYZCoil
            New representation of the coil parameterized by Fourier series for X,Y,Z.

        """
        curve = super().from_values(coords, N, s, basis)
        return cls(
            current,
            X_n=curve.X_n,
            Y_n=curve.Y_n,
            Z_n=curve.Z_n,
            name=name,
        )


class FourierPlanarCoil(_Coil, FourierPlanarCurve):
    """Coil that lines in a plane.

    Parameterized by a point (the center of the coil), a vector (normal to the plane),
    and a fourier series defining the radius from the center as a function of a polar
    angle theta.

    Parameters
    ----------
    current : float
        current through the coil, in Amperes
    center : array-like, shape(3,)
        x,y,z coordinates of center of coil
    normal : array-like, shape(3,)
        x,y,z components of normal vector to planar surface
    r_n : array-like
        fourier coefficients for radius from center as function of polar angle
    modes : array-like
        mode numbers associated with r_n
    name : str
        name for this coil

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

    def __init__(
        self,
        current=1,
        center=[10, 0, 0],
        normal=[0, 1, 0],
        r_n=2,
        modes=None,
        name="",
    ):
        super().__init__(current, center, normal, r_n, modes, name)


class SplineXYZCoil(_Coil, SplineXYZCurve):
    """Coil parameterized by spline points in X,Y,Z.

    Parameters
    ----------
    current : float
        current through coil, in Amperes
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
        name for this curve

    """

    _io_attrs_ = _Coil._io_attrs_ + SplineXYZCurve._io_attrs_

    def __init__(
        self,
        current,
        X,
        Y,
        Z,
        knots=None,
        method="cubic",
        name="",
    ):
        super().__init__(current, X, Y, Z, knots, method, name)

    def compute_magnetic_field(
        self, coords, params=None, basis="rpz", source_grid=None
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
        assert basis.lower() in ["rpz", "xyz"]
        coords = jnp.atleast_2d(jnp.asarray(coords))
        if basis == "rpz":
            coords = rpz2xyz(coords)
        if params is None:
            current = self.current
        else:
            current = params.pop("current", self.current)

        data = self.compute(["x"], grid=source_grid, params=params, basis="xyz")
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
        B = biot_savart_hh(coords, coil_pts_start, coil_pts_end, current)

        if basis == "rpz":
            B = xyz2rpz_vec(B, x=coords[:, 0], y=coords[:, 1])
        return B

    @classmethod
    def from_values(
        cls, current, coords, knots=None, method="cubic", name="", basis="xyz"
    ):
        """Create SplineXYZCoil from coordinate values.

        Parameters
        ----------
        current : float
            Current through the coil, in Amps.
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
            name for this curve
        basis : {"rpz", "xyz"}
            basis for input coordinates. Defaults to "xyz"

        Returns
        -------
        coil: SplineXYZCoil
            New representation of the coil parameterized by splines in X,Y,Z.

        """
        curve = super().from_values(coords, knots, method, basis=basis)
        return cls(
            current,
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
        FourierRZCoil: ["R_basis", "Z_basis", "NFP", "sym"],
        FourierXYZCoil: ["X_basis", "Y_basis", "Z_basis"],
        FourierPlanarCoil: ["r_basis"],
        SplineXYZCoil: ["method", "N"],
    }

    for attr in attrs[coil0.__class__]:
        a0 = getattr(coil0, attr)
        a1 = getattr(coil, attr)
        errorif(
            not equals(a0, a1),
            ValueError,
            (
                "coils in a CoilSet must have the same parameterization, got a "
                + f"mismatch between attr {attr}, with values {a0} and {a1}"
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

    """

    _io_attrs_ = _Coil._io_attrs_ + ["_coils", "_NFP", "_sym"]

    def __init__(self, *coils, NFP=1, sym=False, name=""):
        coils = flatten_list(coils, flatten_tuple=True)
        assert all([isinstance(coil, (_Coil)) for coil in coils])
        [_check_type(coil, coils[0]) for coil in coils]
        self._coils = list(coils)
        self._NFP = NFP
        self._sym = sym
        self._name = str(name)

    @property
    def name(self):
        """str: Name of the curve."""
        return self._name

    @name.setter
    def name(self, new):
        self._name = str(new)

    @property
    def coils(self):
        """list: coils in the coilset."""
        return self._coils

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
        if jnp.isscalar(new):
            new = [new] * len(self)
        for coil, cur in zip(self.coils, new):
            coil.current = cur

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
            params = [get_params(names, coil) for coil in self]
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

    def compute_magnetic_field(
        self, coords, params=None, basis="rpz", source_grid=None
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

        Returns
        -------
        field : ndarray, shape(n,3)
            Magnetic field at specified nodes, in [R,phi,Z] or [X,Y,Z] coordinates.

        """
        assert basis.lower() in ["rpz", "xyz"]
        coords = jnp.atleast_2d(jnp.asarray(coords))
        if params is None:
            params = [get_params(["x_s", "x", "s", "ds"], coil) for coil in self]
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

        # sum the magnetic fields from each field period
        def nfp_loop(k, B):
            coords_nfp = coords_rpz + jnp.array([0, 2 * jnp.pi * k / self.NFP, 0])

            def body(B, x):
                B += self[0].compute_magnetic_field(
                    coords_nfp, params=x, basis="rpz", source_grid=source_grid
                )
                return B, None

            B += scan(body, jnp.zeros(coords_nfp.shape), tree_stack(params))[0]
            return B

        B = fori_loop(0, self.NFP, nfp_loop, jnp.zeros_like(coords_rpz))

        # sum the magnetic fields from both halves of the symmetric field period
        if self.sym:
            B = B[: coords.shape[0], :] + B[coords.shape[0] :, :] * jnp.array(
                [-1, 1, 1]
            )

        if basis.lower() == "xyz":
            B = rpz2xyz_vec(B, x=coords[:, 0], y=coords[:, 1])
        return B

    @classmethod
    def linspaced_angular(
        cls, coil, current=None, axis=[0, 0, 1], angle=2 * np.pi, n=10, endpoint=False
    ):
        """Create a coil set by repeating a coil n times rotationally.

        Parameters
        ----------
        coil : Coil
            base coil to repeat
        current : float or array-like, shape(n,)
            current in (each) coil, overrides coil.current
        axis : array-like, shape(3,)
            axis to rotate about
        angle : float
            total rotational extent of coil set
        n : int
            number of copies of original coil
        endpoint : bool
            whether to include a coil at final angle

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
        return cls(*coils)

    @classmethod
    def linspaced_linear(
        cls, coil, current=None, displacement=[2, 0, 0], n=4, endpoint=False
    ):
        """Create a coil group by repeating a coil n times in a straight line.

        Parameters
        ----------
        coil : Coil
            base coil to repeat
        current : float or array-like, shape(n,)
            current in (each) coil
        displacement : array-like, shape(3,)
            total displacement of the final coil
        n : int
            number of copies of original coil
        endpoint : bool
            whether to include a coil at final point

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
        return cls(*coils)

    @classmethod
    def from_symmetry(cls, coils, NFP=1, sym=False):
        """Create a coil group by reflection and symmetry.

        Given coils over one field period, repeat coils NFP times between
        0 and 2pi to form full coil set.

        Or, given coils over 1/2 of a field period, repeat coils 2*NFP times
        between 0 and 2pi to form full stellarator symmetric coil set.

        Parameters
        ----------
        coils : Coil, CoilGroup, Coilset
            Coil or collection of coils in one field period or half field period.
        NFP : int (optional)
            Number of field periods for enforcing field period symmetry.
            The coils will be duplicated NFP times. Default = 1.
        sym : bool (optional)
            Whether to enforce stellarator symmetry.
            If True, the coils will be duplicated 2*NFP times. Default = False.

        Returns
        -------
        coilset : CoilSet
            A new coil set with NFP=1 and sym=False that is equivalent to the unique
            coils with field period symmetry and stellarator symmetry.
            The total number of coils in the new coil set is:
            len(coilset) = len(coils) * NFP * (int(sym) + 1)

        """
        if not isinstance(coils, CoilSet):
            coils = CoilSet(coils)

        [_check_type(coil, coils[0]) for coil in coils]

        # check toroidal extent of coils to be repeated
        maxphi = 2 * np.pi / NFP / (sym + 1)
        data = coils.compute("phi")
        for i, cdata in enumerate(data):
            errorif(
                np.any(cdata["phi"] > maxphi),
                ValueError,
                f"coil {i} exceeds the toroidal extent for NFP={NFP} and sym={sym}",
            )
            warnif(
                sym and np.any(cdata["phi"] < np.finfo(cdata["phi"].dtype).eps),
                UserWarning,
                f"coil {i} is on the symmetry plane phi=0",
            )

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

        return cls(*coilset)

    @classmethod
    def from_makegrid_coilfile(cls, coil_file, method="cubic"):
        """Create a CoilSet of SplineXYZCoils from a MAKEGRID-formatted coil txtfile.

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

        """
        coils = []  # list of SplineXYZCoils
        coilinds = [2]  # always start at the 3rd line after periods
        names = []

        # read in the coils file
        headind = -1
        with open(coil_file) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.find("periods") != -1:
                    headind = i  # skip anything that is above the periods line
                    coilinds[0] += headind
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
                    names.append(" ".join(line.split()[4:]))
        if len(lines[3 + headind].split()) != 4:
            raise OSError(
                "4th line in file must be the start of the first coil! "
                + "Expected a line of length 4 (after .split()), "
                + f"instead got length {lines[3].split()}"
            )
        header_lines_not_as_expected = np.array(
            [
                len(lines[0 + headind].split()) != 2,
                len(lines[1 + headind].split()) != 2,
                len(lines[2 + headind].split()) != 2,
            ]
        )
        if np.any(header_lines_not_as_expected):
            raise OSError(
                "First 3 lines in file starting with the periods line "
                + "must be the header lines,"
                + " each of length 2 (after .split())! "
                + f"Line(s) {lines[np.where(header_lines_not_as_expected)[0]+headind]}"
                + " are not length 2"
            )

        for i, (start, end) in enumerate(zip(coilinds[0:-1], coilinds[1:])):
            coords = np.genfromtxt(lines[start + 1 : end])
            coils.append(
                SplineXYZCoil(
                    coords[:, -1][0],
                    coords[:, 0],
                    coords[:, 1],
                    coords[:, 2],
                    method=method,
                    name=names[i],
                )
            )

        return cls(*coils)

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
        # TODO: name each group based off of CoilSet name?
        # TODO: have CoilGroup be automatically assigned based off of
        # CoilSet if current coilset is a collection of coilsets?

        NFP = 1 if NFP is None else NFP

        def flatten_coils(coilset):
            if hasattr(coilset, "__len__"):
                return [a for i in coilset for a in flatten_coils(i)]
            else:
                return [coilset]

        coils = flatten_coils(self.coils)
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
        elif isinstance(grid, numbers.Integral):
            endpoint = False  # if int, will create a grid w/ endpoint=False in compute
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

    def to_FourierXYZ(self, N=10, grid=None, s=None, name=""):
        """Convert all coils to FourierXYZCoil representation.

        Parameters
        ----------
        N : int
            Fourier resolution of the new X,Y,Z representation.
        grid : Grid, int or None
            Grid used to evaluate curve coordinates on to fit with FourierXYZCoil.
            If an integer, uses that many equally spaced points.
        s : ndarray
            arbitrary curve parameter to use for the fitting. if None, defaults to
            normalized arclength
        name : str
            name for the new CoilSet

        Returns
        -------
        coilset : CoilSet
            New representation of the coilset parameterized by Fourier series for X,Y,Z.

        """
        coils = [coil.to_FourierXYZ(N, grid, s) for coil in self]
        return self.__class__(*coils, name=name)

    def to_SplineXYZ(self, knots=None, grid=None, method="cubic", name=""):
        """Convert all coils to SplineXYZCoil.

        Parameters
        ----------
        knots : ndarray
            arbitrary curve parameter values to use for spline knots,
            should be an 1D ndarray of same length as the input.
            (input length in this case is determined by grid argument, since
            the input coordinates come from
            Coil.compute("x",grid=grid))
            If None, defaults to using an equal-arclength angle as the knots
            If supplied, will be rescaled to lie in [0,2pi]
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
            name for the new CoilSet

        Returns
        -------
        coilset : CoilSet
            New representation of the coilset parameterized by a spline for X,Y,Z.

        """
        coils = [coil.to_SplineXYZ(knots, grid, method) for coil in self]
        return self.__class__(*coils, name=name)

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

    """

    def __init__(self, *coils, name=""):
        coils = flatten_list(coils, flatten_tuple=True)
        assert all([isinstance(coil, (_Coil)) for coil in coils])
        self._coils = list(coils)
        self._NFP = 1
        self._sym = False
        self._name = str(name)

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

    def compute_magnetic_field(
        self, coords, params=None, basis="rpz", source_grid=None
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

        Returns
        -------
        field : ndarray, shape(n,3)
            magnetic field at specified points, in either rpz or xyz coordinates

        """
        params = self._make_arraylike(params)
        source_grid = self._make_arraylike(source_grid)

        B = 0
        for coil, par, grd in zip(self.coils, params, source_grid):
            B += coil.compute_magnetic_field(coords, par, basis, grd)

        return B

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
