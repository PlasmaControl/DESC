"""Classes for magnetic field coils."""

import numbers
from abc import ABC
from collections.abc import MutableSequence

import numpy as np

from desc.backend import jnp
from desc.compute import rpz2xyz, xyz2rpz_vec
from desc.geometry import (
    FourierPlanarCurve,
    FourierRZCurve,
    FourierXYZCurve,
    SplineXYZCurve,
)
from desc.magnetic_fields import _MagneticField, biot_savart
from desc.utils import flatten_list


class _Coil(_MagneticField, ABC):
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
        self._current = current
        super().__init__(*args, **kwargs)

    @property
    def current(self):
        """float: Current passing through the coil, in Amperes."""
        return self._current

    @current.setter
    def current(self, new):
        assert jnp.isscalar(new) or new.size == 1
        self._current = new

    def compute_magnetic_field(self, coords, params=None, basis="rpz", grid=None):
        """Compute magnetic field at a set of points.

        The coil current may be overridden by including `current`
        in the `params` dictionary.

        Parameters
        ----------
        coords : array-like shape(n,3) or Grid
            coordinates to evaluate field at [R,phi,Z] or [x,y,z]
        params : dict, optional
            parameters to pass to curve
        basis : {"rpz", "xyz"}
            basis for input coordinates and returned magnetic field
        grid : Grid, int or None
            Grid used to discretize coil. If an integer, uses that many equally spaced
            points.

        Returns
        -------
        field : ndarray, shape(n,3)
            magnetic field at specified points, in either rpz or xyz coordinates
        """
        assert basis.lower() in ["rpz", "xyz"]
        if hasattr(coords, "nodes"):
            coords = coords.nodes
        coords = jnp.atleast_2d(coords)
        if basis == "rpz":
            coords = rpz2xyz(coords)
        if params is None:
            current = self.current
        else:
            current = params.pop("current", self.current)
        coil_coords = self.compute("x", params=params, grid=grid, basis="xyz")["x"]
        B = biot_savart(coords, coil_coords, current)
        if basis == "rpz":
            B = xyz2rpz_vec(B, x=coords[:, 0], y=coords[:, 1])
        return B

    def __repr__(self):
        """Get the string form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (name={}, current={})".format(self.name, self.current)
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
    """Coil parameterized by fourier series for X,Y,Z in terms of arbitrary angle phi.

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
        # circular coil given by X(phi) = 10*cos(phi), Y(phi) = 10*sin(phi)
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
        points for X, Y, Z describing a closed curve
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


class CoilSet(_Coil, MutableSequence):
    """Set of coils of different geometry.

    Parameters
    ----------
    coils : Coil or array-like of Coils
        collection of coils
    currents : float or array-like of float
        currents in each coil, or a single current shared by all coils in the set
    name : str
        name of this CoilSet

    """

    _io_attrs_ = _Coil._io_attrs_ + ["_coils"]

    def __init__(self, *coils, name=""):
        coils = flatten_list(coils, flatten_tuple=True)
        assert all([isinstance(coil, (_Coil)) for coil in coils])
        self._coils = list(coils)
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

    def translate(self, *args, **kwargs):
        """Translate the coils along an axis."""
        [coil.translate(*args, **kwargs) for coil in self.coils]

    def rotate(self, *args, **kwargs):
        """Rotate the coils about an axis."""
        [coil.rotate(*args, **kwargs) for coil in self.coils]

    def flip(self, *args, **kwargs):
        """Flip the coils across a plane."""
        [coil.flip(*args, **kwargs) for coil in self.coils]

    def compute_magnetic_field(self, coords, params=None, basis="rpz", grid=None):
        """Compute magnetic field at a set of points.

        Parameters
        ----------
        coords : array-like shape(n,3) or Grid
            coordinates to evaluate field at [R,phi,Z] or [x,y,z]
        params : dict or array-like of dict, optional
            parameters to pass to curves, either the same for all curves,
            or one for each member
        basis : {"rpz", "xyz"}
            basis for input coordinates and returned magnetic field
        grid : Grid, int or None or array-like, optional
            Grid used to discretize coil, either the same for all coils or one for each
            member of the coilset. If an integer, uses that many equally spaced
            points.

        Returns
        -------
        field : ndarray, shape(n,3)
            magnetic field at specified points, in either rpz or xyz coordinates
        """
        params = self._make_arraylike(params)
        grid = self._make_arraylike(grid)

        B = 0
        for coil, par, grd in zip(self.coils, params, grid):
            B += coil.compute_magnetic_field(coords, par, basis, grd)

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
            total rotational extend of coil set.
        n : int
            number of copies of original coil
        endpoint : bool
            whether to include a coil at final angle
        """
        assert isinstance(coil, _Coil)
        if current is None:
            current = coil.current
        currents = jnp.broadcast_to(current, (n,))
        coils = []
        phis = jnp.linspace(0, angle, n, endpoint=endpoint)
        for i in range(n):
            coili = coil.copy()
            coili.rotate(axis, angle=phis[i])
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
        assert isinstance(coil, _Coil)
        if current is None:
            current = coil.current
        currents = jnp.broadcast_to(current, (n,))
        displacement = jnp.asarray(displacement)
        coils = []
        a = jnp.linspace(0, 1, n, endpoint=endpoint)
        for i in range(n):
            coili = coil.copy()
            coili.translate(a[i] * displacement)
            coili.current = currents[i]
            coils.append(coili)
        return cls(*coils)

    @classmethod
    def from_symmetry(cls, coils, NFP, sym=False):
        """Create a coil group by reflection and symmetry.

        Given coils over one field period, repeat coils NFP times between
        0 and 2pi to form full coil set.

        Or, give coils over 1/2 of a field period, repeat coils 2*NFP times
        between 0 and 2pi to form full stellarator symmetric coil set.

        Parameters
        ----------
        coils : Coil, CoilGroup, Coilset
            base coil or collection of coils to repeat
        NFP : int
            number of field periods
        sym : bool
            whether coils should be stellarator symmetric
        """
        if not isinstance(coils, CoilSet):
            coils = CoilSet(coils)
        coilset = []
        if sym:
            # first reflect/flip original coilset
            # ie, given coils [1,2,3] at angles [0, pi/6, 2pi/6]
            # we want a new set like [1,2,3,flip(3),flip(2),flip(1)]
            # at [0, pi/6, 2pi/6, 3pi/6, 4pi/6, 5pi/6]
            flipped_coils = []
            normal = jnp.array([-jnp.sin(jnp.pi / NFP), jnp.cos(jnp.pi / NFP), 0])
            for coil in coils[::-1]:
                fcoil = coil.copy()
                fcoil.flip(normal)
                fcoil.flip([0, 0, 1])
                fcoil.current = -1 * coil.current
                flipped_coils.append(fcoil)
            coils = coils + flipped_coils
        for k in range(0, NFP):
            coil = coils.copy()
            coil.rotate(axis=[0, 0, 1], angle=2 * jnp.pi * k / NFP)
            coilset.append(coil)

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
        coilinds = [2]  # always start at the 3rd line
        names = []

        # read in the coils file
        with open(coil_file) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.find("periods") != -1:
                    continue
                if (
                    line.find("begin filament") != -1
                    or line.find("end") != -1
                    or line.find("mirror") != -1
                ):
                    continue  # skip headers and last line
                if (
                    len(line.split()) != 4
                    and line.strip()  # ensure not counting blank lines
                ):
                    coilinds.append(i)
                    names.append(" ".join(line.split()[4:]))
        if len(lines[3].split()) != 4:
            raise OSError(
                "4th line in file must be the start of the first coil! "
                + "Expected a line of length 4 (after .split()), "
                + f"instead got length {lines[3].split()}"
            )
        header_lines_not_as_expected = np.array(
            [
                len(lines[0].split()) != 2,
                len(lines[1].split()) != 2,
                len(lines[2].split()) != 2,
            ]
        )
        if np.any(header_lines_not_as_expected):
            raise OSError(
                "First 3 lines in file must be the header lines,"
                + " each of length 2 (after .split())! "
                + f"Line(s) {lines[np.where(header_lines_not_as_expected)[0]]}"
                + " are not length 2"
            )

        for i, (start, end) in enumerate(zip(coilinds[0:-1], coilinds[1:])):
            coords = np.genfromtxt(lines[start + 1 : end])

            tempx = np.append(coords[:, 0], np.array([coords[0, 0]]))
            tempy = np.append(coords[:, 1], np.array([coords[0, 1]]))
            tempz = np.append(coords[:, 2], np.array([coords[0, 2]]))

            coils.append(
                SplineXYZCoil(
                    coords[:, -1][0],
                    tempx,
                    tempy,
                    tempz,
                    method=method,
                    name=names[i],
                )
            )

        return CoilSet(*coils)

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
            endpoint = True  # if int, will create a grid w/ endpoint=True in compute
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

    def __add__(self, other):
        if isinstance(other, (CoilSet)):
            return CoilSet(*self.coils, *other.coils)
        if isinstance(other, (list, tuple)):
            return CoilSet(*self.coils, *other)
        raise TypeError

    # dunder methods required by MutableSequence
    def __getitem__(self, i):
        return self.coils[i]

    def __setitem__(self, i, new_item):
        if not isinstance(new_item, _Coil):
            raise TypeError("Members of CoilSet must be of type Coil.")
        self._coils[i] = new_item

    def __delitem__(self, i):
        del self._coils[i]

    def __len__(self):
        return len(self._coils)

    def insert(self, i, new_item):
        """Insert a new coil into the coilset at position i."""
        if not isinstance(new_item, _Coil):
            raise TypeError("Members of CoilSet must be of type Coil.")
        self._coils.insert(i, new_item)

    def __repr__(self):
        """Get the string form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (name={}, with {} submembers)".format(self.name, len(self))
        )
