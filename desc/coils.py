"""Classes for magnetic field coils."""

from abc import ABC
from collections.abc import MutableSequence

import numpy as np

from desc.backend import jnp
from desc.geometry import FourierPlanarCurve, FourierRZCurve, FourierXYZCurve, XYZCurve
from desc.geometry.utils import rpz2xyz, xyz2rpz_vec
from desc.grid import Grid
from desc.magnetic_fields import MagneticField, biot_savart


class Coil(MagneticField, ABC):
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

    _io_attrs_ = MagneticField._io_attrs_ + ["_current"]

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

    def compute_magnetic_field(self, coords, params={}, basis="rpz"):
        """Compute magnetic field at a set of points.

        The coil is discretized into a series of straight line segments, using
        the coil ``grid`` attribute. To override this, include 'grid' as a key
        in the `params` dictionary with the desired grid resolution.

        Similarly, the coil current may be overridden by including `current`
        in the `params` dictionary.

        Parameters
        ----------
        coords : array-like shape(n,3) or Grid
            coordinates to evaluate field at [R,phi,Z] or [x,y,z]
        params : dict, optional
            parameters to pass to curve
        basis : {"rpz", "xyz"}
            basis for input coordinates and returned magnetic field

        Returns
        -------
        field : ndarray, shape(n,3)
            magnetic field at specified points, in either rpz or xyz coordinates
        """
        if params is None:
            params = {}
        assert basis.lower() in ["rpz", "xyz"]
        if isinstance(coords, Grid):
            coords = coords.nodes
        coords = jnp.atleast_2d(coords)
        if basis == "rpz":
            coords = rpz2xyz(coords)
        current = params.pop("current", self.current)
        coil_coords = self.compute_coordinates(**params, basis="xyz")
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


class FourierRZCoil(Coil, FourierRZCurve):
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
    grid : Grid
        default grid for computation
    name : str
        name for this coil
    """

    _io_attrs_ = Coil._io_attrs_ + FourierRZCurve._io_attrs_

    def __init__(
        self,
        current=1,
        R_n=10,
        Z_n=0,
        modes_R=None,
        modes_Z=None,
        NFP=1,
        sym="auto",
        grid=None,
        name="",
    ):
        super().__init__(current, R_n, Z_n, modes_R, modes_Z, NFP, sym, grid, name)


class FourierXYZCoil(Coil, FourierXYZCurve):
    """Coil parameterized by fourier series for X,Y,Z in terms of arbitrary angle phi.

    Parameters
    ----------
    current : float
        current through coil, in Amperes
    X_n, Y_n, Z_n: array-like
        fourier coefficients for X, Y, Z
    modes : array-like
        mode numbers associated with X_n etc.
    grid : Grid
        default grid or computation
    name : str
        name for this coil

    """

    _io_attrs_ = Coil._io_attrs_ + FourierXYZCurve._io_attrs_

    def __init__(
        self,
        current=1,
        X_n=[0, 10, 2],
        Y_n=[0, 0, 0],
        Z_n=[-2, 0, 0],
        modes=None,
        grid=None,
        name="",
    ):
        super().__init__(current, X_n, Y_n, Z_n, modes, grid, name)


# TODO: add a from_XYZ?


class FourierPlanarCoil(Coil, FourierPlanarCurve):
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
    grid : Grid
        default grid for computation
    name : str
        name for this coil

    """

    _io_attrs_ = Coil._io_attrs_ + FourierPlanarCurve._io_attrs_

    def __init__(
        self,
        current=1,
        center=[10, 0, 0],
        normal=[0, 1, 0],
        r_n=2,
        modes=None,
        grid=None,
        name="",
    ):
        super().__init__(current, center, normal, r_n, modes, grid, name)


class XYZCoil(Coil, XYZCurve):
    """Coil parameterized by spline points in X,Y,Z.

    Parameters
    ----------
    current : float
        current through coil, in Amperes
    X, Y, Z: array-like
        points for X, Y, Z describing a closed curve
    knots : ndarray
        arbitrary theta values to use for spline knots,
        should be an 1D ndarray of same length as the input.
        If None, defaults to using an equal-arclength angle as the knot
    period: float
        period of the theta variable used for the spline knots.
        if knots is None, this defaults to 2pi. If knots is not None, this must be
        supplied by the user
    grid : Grid
        default grid for computation
    method : str
        method of interpolation
        - `'nearest'`: nearest neighbor interpolation
        - `'linear'`: linear interpolation
        - `'cubic'`: C1 cubic splines (aka local splines)
        - `'cubic2'`: C2 cubic splines (aka natural splines)
        - `'catmull-rom'`: C1 cubic centripetal "tension" splines
    name : str
        name for this coil

    """

    _io_attrs_ = Coil._io_attrs_

    def __init__(
        self,
        current,
        X,
        Y,
        Z,
        knots=None,
        period=None,
        grid=None,
        method="cubic2",
        name="",
    ):
        super().__init__(current, X, Y, Z, knots, period, grid, method, name)


class CoilSet(Coil, MutableSequence):
    """Set of coils of different geometry.

    Parameters
    ----------
    coils : Coil or array-like of Coils
        collection of coils
    currents : float or array-like of float
        currents in each coil, or a single current shared by all coils in the set
    """

    _io_attrs_ = Coil._io_attrs_ + ["_coils"]

    def __init__(
        self, *coils, name=""
    ):  # FIXME: if a list of of Coils is passed, this fails...
        assert all([isinstance(coil, (Coil)) for coil in coils])
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

    @property
    def grid(self):
        """Grid: nodes for computation."""
        return self.coils[0].grid

    @grid.setter
    def grid(self, new):
        for coil in self.coils:
            coil.grid = new

    def compute_coordinates(self, *args, **kwargs):
        """Compute real space coordinates using underlying curve method."""
        return [coil.compute_coordinates(*args, **kwargs) for coil in self.coils]

    def compute_frenet_frame(self, *args, **kwargs):
        """Compute Frenet frame using underlying curve method."""
        return [coil.compute_frenet_frame(*args, **kwargs) for coil in self.coils]

    def compute_curvature(self, *args, **kwargs):
        """Compute curvature using underlying curve method."""
        return [coil.compute_curvature(*args, **kwargs) for coil in self.coils]

    def compute_torsion(self, *args, **kwargs):
        """Compute torsion using underlying curve method."""
        return [coil.compute_torsion(*args, **kwargs) for coil in self.coils]

    def compute_length(self, *args, **kwargs):
        """Compute the length of the curve using underlying curve method."""
        return [coil.compute_length(*args, **kwargs) for coil in self.coils]

    def translate(self, *args, **kwargs):
        """Translate the coils along an axis."""
        [coil.translate(*args, **kwargs) for coil in self.coils]

    def rotate(self, *args, **kwargs):
        """Rotate the coils about an axis."""
        [coil.rotate(*args, **kwargs) for coil in self.coils]

    def flip(self, *args, **kwargs):
        """Flip the coils across a plane."""
        [coil.flip(*args, **kwargs) for coil in self.coils]

    def compute_magnetic_field(self, coords, params={}, basis="rpz"):
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

        Returns
        -------
        field : ndarray, shape(n,3)
            magnetic field at specified points, in either rpz or xyz coordinates
        """
        if isinstance(params, dict) or params is None:
            params = [params] * len(self)
        assert len(params) == len(self)
        B = 0
        for coil, par in zip(self.coils, params):
            B += coil.compute_magnetic_field(coords, par, basis)

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
        assert isinstance(coil, Coil)
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
        assert isinstance(coil, Coil)
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
    def from_makegrid_coilfile(cls, coil_file, method="cubic", grid=None):
        """Create a CoilSet of XYZCoils from a MAKEGRID-formatted coil txtfile.

        Parameters
        ----------
        coil_file : str or path-like
            path to coil file in txt format
        method : str
            method of interpolation
            - `'nearest'`: nearest neighbor interpolation
            - `'linear'`: linear interpolation
            - `'cubic'`: C1 cubic splines (aka local splines)
            - `'cubic2'`: C2 cubic splines (aka natural splines)
            - `'catmull-rom'`: C1 cubic centripetal "tension" splines
        grid : Grid
            default grid for computation
        """
        coils = []  # list of XYZCoils
        coilinds = []

        # read in the coils file
        with open(coil_file) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.find("Modular") != -1:
                    coilinds.append(i)
                if line.find("mirror") != -1:
                    coilinds.append(i)
        for i, (start, end) in enumerate(zip(coilinds[0:-1], coilinds[1:])):
            coords = np.genfromtxt(lines[start + 1 : end])
            if i % 20 == 0:
                print("reading coil " + f"{i}")

            tempx = np.append(coords[:, 0], np.array([coords[0, 0]]))
            tempy = np.append(coords[:, 1], np.array([coords[0, 1]]))
            tempz = np.append(coords[:, 2], np.array([coords[0, 2]]))

            coils.append(
                XYZCoil(coords[:, -1][0], tempx, tempy, tempz, grid=grid, method=method)
            )

        return CoilSet(*coils)

    def save_in_MAKEGRID_format(self, coilsFilename, NFP=1, grid=None):
        """Save CoilSet of as a MAKEGRID-formatted coil txtfile.

        By default, each coil is assigned to the same Coilgroup in MAKEGRID
        with the name "Modular". For more details see the MAKEGRID documentation
        https://princetonuniversity.github.io/STELLOPT/MAKEGRID.html

        Parameters
        ----------
        filename : str or path-like
            path save CoilSet as a file in MAKEGRID txt format
        NFP : int, default 1
            If > 1, assumes that the CoilSet is the coils for a coilset
            with a discrete toroidal symmetry of NFP, and so will only
            save the first len(coils)/NFP coils in the MAKEGRID file.
        grid: Grid, ndarray, int,
            Grid of sample points along each coil to save.
            if None, will default to each coils self._grid
        """
        assert (
            int(len(self.coils) / NFP) == len(self.coils) / NFP
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
        for i in range(int(len(self.coils) / NFP)):
            coil = self.coils[i]
            coords = coil.compute_coordinates(basis="xyz", grid=grid)

            contour_X = np.asarray(coords[0:-1, 0])
            contour_Y = np.asarray(coords[0:-1, 1])
            contour_Z = np.asarray(coords[0:-1, 2])

            currents = np.ones_like(contour_X) * float(coil.current)
            # close the curves
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
            fmt="%14.22e",
            comments="",  # to avoid the # appended to the start of the header/footer
        )
        # now need to re-load the file and place coilgroup markers at end of each coil
        with open(coilsFilename) as f:
            lines = f.readlines()
        for i in range(len(coil_end_inds)):
            real_end_ind = int(
                np.sum(coil_end_inds[0 : i + 1]) + 2
            )  # to account for the 3 header lines
            lines[real_end_ind] = lines[real_end_ind].strip("\n") + " 1 Modular\n"
        with open(coilsFilename, "w") as f:
            f.writelines(lines)

        print(f"Saved coils file at : {coilsFilename}")

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
        if not isinstance(new_item, Coil):
            raise TypeError("Members of CoilSet must be of type Coil.")
        self._coils[i] = new_item

    def __delitem__(self, i):
        del self._coils[i]

    def __len__(self):
        return len(self._coils)

    def insert(self, i, new_item):
        """Insert a new coil into the coilset at position i."""
        if not isinstance(new_item, Coil):
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
