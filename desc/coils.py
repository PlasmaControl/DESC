import numpy as np
from collections.abc import MutableSequence
from desc.backend import jnp
from desc.geometry.core import Curve, xyz2rpz, xyz2rpz_vec, rpz2xyz, rpz2xyz_vec
from desc.magnetic_fields import MagneticField, biot_savart
from desc.grid import Grid


class Coil(MagneticField, Curve):
    """Class representing a magnetic field coil, as a combination of a curve and current

    Parameters
    ----------
    curve : Curve
        underlying geometric curve definining path of coil
    current : float
        current passing through the coil, in Amperes
    """

    _io_attrs_ = MagneticField._io_attrs_ + ["_curve", "_current"]

    def __init__(self, curve, current, name=""):
        super(Coil, self).__init__(name)
        assert isinstance(curve, Curve)
        self._curve = curve
        self._current = current

    @property
    def curve(self):
        return self._curve

    @property
    def current(self):
        return self._current

    @current.setter
    def current(self, new):
        assert jnp.isscalar(new) or new.size == 1
        self._current = new

    @property
    def grid(self):
        """Default grid for computation."""
        return self.curve.grid

    @grid.setter
    def grid(self, new):
        if isinstance(new, Grid):
            self.curve.grid = new
        elif isinstance(new, (np.ndarray, jnp.ndarray)):
            self.curve.grid = Grid(new, sort=False)
        else:
            raise TypeError(
                f"grid should be a Grid or subclass, or ndarray, got {type(new)}"
            )

    def compute_coordinates(self, *args, **kwargs):
        """Compute real space coordinates using underlying curve method."""
        return self.curve.compute_coordinates(*args, **kwargs)

    def compute_frenet_frame(self, *args, **kwargs):
        """Compute Frenet frame using underlying curve method."""
        return self.curve.compute_frenet_frame(*args, **kwargs)

    def compute_curvature(self, *args, **kwargs):
        """Compute curvature using underlying curve method."""
        return self.curve.compute_curvature(*args, **kwargs)

    def compute_torsion(self, *args, **kwargs):
        """Compute torsion using underlying curve method."""
        return self.curve.compute_torsion(*args, **kwargs)

    def compute_length(self, *args, **kwargs):
        """Compute the length of the curve using underlying curve method."""
        return self.curve.compute_length(*args, **kwargs)

    def translate(self, *args, **kwargs):
        """translate the coil along an axis"""
        self.curve.translate(*args, **kwargs)

    def rotate(self, *args, **kwargs):
        """rotate the coil about an axis"""
        self.curve.rotate(*args, **kwargs)

    def flip(self, *args, **kwargs):
        """flip the coil across a plane"""
        self.curve.flip(*args, **kwargs)

    def compute_magnetic_field(self, coords, params={}, basis="rpz"):
        """Compute magnetic field at a set of points

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
        assert basis.lower() in ["rpz", "xyz"]
        if isinstance(coords, Grid):
            coords = coords.nodes
        coords = jnp.atleast_2d(coords)
        if basis == "rpz":
            coords = rpz2xyz(coords)
        current = params.pop("current", self.current)
        coil_coords = self.curve.compute_coordinates(**params, basis="xyz")
        B = biot_savart(coords, coil_coords, current)
        if basis == "rpz":
            B = xyz2rpz_vec(B, x=coords[:, 0], y=coords[:, 1])
        return B


class CoilGroup(Coil, MutableSequence, Curve):
    """Group of coils that share the same geometry

    Parameters
    ----------
    coils : array-like of Coil
        collection of coils, should be of the same type and share geometry
    """

    _io_attrs_ = Coil._io_attrs_ + ["_coils"]

    def __init__(self, *coils):
        assert all(
            [isinstance(coil, (Coil)) for coil in coils]
        ), "each coil should be an instance of Coil"
        assert (
            len(set([coil.__class__ for coil in coils])) == 1
        ), "Coils in a CoilGroup must be of the same type"
        self._coils = coils
        self._grid = coils[0].grid

    @property
    def coils(self):
        return self._coils

    @property
    def curve(self):
        return [coil.curve for coil in self.coils]

    @property
    def current(self):
        return [coil.current for coil in self.coils]

    @current.setter
    def current(self, new):
        assert jnp.isscalar(new) or (len(new) == len(self.curve))
        new = jnp.broadcast_to(
            jnp.asarray(new),
            len(
                self.coils,
            ),
        )
        for coil, cur in zip(self.coils, new):
            coil.current = cur

    @property
    def grid(self):
        """Default grid for computation."""
        return self.curve[0].grid

    @grid.setter
    def grid(self, new):
        if isinstance(new, Grid):
            grid = new
        elif isinstance(new, (np.ndarray, jnp.ndarray)):
            grid = Grid(new, sort=False)
        else:
            raise TypeError(
                f"grid should be a Grid or subclass, or ndarray, got {type(new)}"
            )
        for curve in self.curve:
            curve.grid = grid

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
        """translate the coils along an axis"""
        [coil.translate(*args, **kwargs) for coil in self.coils]

    def rotate(self, *args, **kwargs):
        """rotate the coils about an axis"""
        [coil.rotate(*args, **kwargs) for coil in self.coils]

    def flip(self, *args, **kwargs):
        """flip the coils across a plane"""
        [coil.flip(*args, **kwargs) for coil in self.coils]

    def compute_magnetic_field(self, coords, params={}, basis="rpz"):
        """Compute magnetic field at a set of points

        Parameters
        ----------
        coords : array-like shape(n,3) or Grid
            coordinates to evaluate field at [R,phi,Z] or [x,y,z]
        params : dict, optional
            parameters to pass to curves. The same parameters will be passed to each
            curve.
        basis : {"rpz", "xyz"}
            basis for input coordinates and returned magnetic field

        Returns
        -------
        field : ndarray, shape(n,3)
            magnetic field at specified points, in either rpz or xyz coordinates
        """
        current = jnp.asarray(params.pop("current", self.current))
        currents = jnp.broadcast_to(current, (len(self.curve),))
        B = 0
        for coil, cur in zip(self.coils, currents):
            params["current"] = cur
            B += coil.compute_magnetic_field(coords, params, basis)
        return B

    @classmethod
    def linspaced_angular(
        cls, curve, current, axis=[0, 0, 1], angle=2 * np.pi, n=10, endpoint=False
    ):
        """Create a coil set by repeating a curve n times rotationally.

        Parameters
        ----------
        curve : Curve
            base curve to repeat
        current : float or array-like, shape(n,)
            current in (each) coil
        axis : array-like, shape(3,)
            axis to rotate about
        angle : float
            total rotational extend of coil set.
        n : int
            number of copies of original coil
        endpoint : bool
            whether to include a coil at final angle
        """
        assert isinstance(curve, Curve) and not isinstance(curve, Coil)
        currents = jnp.broadcast_to(current, (n,))
        coils = []
        phis = jnp.linspace(0, angle, n, endpoint=endpoint)
        for i in range(n):
            coil = curve.copy()
            coil.rotate(axis, angle=phis[i])
            coils.append(Coil(coil, currents[i]))
        return cls(*coils)

    @classmethod
    def linspaced_linear(
        cls, curve, current, displacement=[2, 0, 0], n=4, endpoint=False
    ):
        """Create a coil group by repeating a curve n times in a straight line.

        Parameters
        ----------
        curve : Curve
            base curve to repeat
        current : float or array-like, shape(n,)
            current in (each) coil
        displacement : array-like, shape(3,)
            total displacement of the final coil
        n : int
            number of copies of original coil
        endpoint : bool
            whether to include a coil at final point
        """
        assert isinstance(curve, Curve) and not isinstance(curve, Coil)
        currents = jnp.broadcast_to(current, (n,))
        displacement = jnp.asarray(displacement)
        coils = []
        a = jnp.linspace(0, 1, n, endpoint=endpoint)
        for i in range(n):
            temp_curve = curve.copy()
            temp_curve.translate(a[i] * displacement)
            coils.append(Coil(temp_curve, currents[i]))
        return cls(*coils)

    @classmethod
    def from_symmetry(cls, coils, NFP, sym=False):
        """Create a coil group by reflection and symmetry

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
        if not isinstance(coils, (CoilGroup, CoilSet)):
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

    def __add__(self, other):
        if isinstance(other, (CoilGroup, CoilSet)):
            return CoilGroup(*self.coils, *other.coils)
        if isinstance(other, (list, tuple)):
            return CoilGroup(*self.coils, *other)
        raise TypeError

    # dunder methods required by MutableSequence
    def __getitem__(self, i):
        return self.coils[i]

    def __setitem__(self, i, new_item):
        if not (new_item.__class__ is Coil):
            raise TypeError("Members of CoilGroup must be of type Coil.")
        if not (new_item.curve.__class__ is self.coils[0].curve.__class__):
            raise TypeError(
                "Members of CoilGroup must share the same curve parameterization."
            )
        self._coils[i] = new_item

    def __delitem__(self, i):
        del self._coils[i]

    def __len__(self):
        return len(self._coils)

    def insert(self, i, new_item):
        if not (new_item.__class__ is Coil):
            raise TypeError("Members of CoilGroup must be of type Coil.")
        if not (new_item.curve.__class__ is self.coils[0].curve.__class__):
            raise TypeError(
                "Members of CoilGroup must share the same curve parameterization."
            )
        self._coils.insert(i, new_item)

    def __slice__(self, idx):
        if idx is None:
            theslice = slice(None, None)
        elif isinstance(idx, int):
            theslice = idx
        elif isinstance(idx, list):
            try:
                theslice = slice(idx[0], idx[1], idx[2])
            except IndexError:
                theslice = slice(idx[0], idx[1])
        else:
            raise TypeError("index is not a valid type.")
        return theslice


class CoilSet(Coil, MutableSequence, Curve):
    """Set of coils of different geometry

    Parameters
    ----------
    curves : array-like of Curve
        collection of curves
    currents : float or array-like of float
        currents in each coil, or a single current shared by all coils in the set
    """

    _io_attrs_ = Coil._io_attrs_ + ["_coils"]

    def __init__(self, *coils):
        assert all([isinstance(coil, (Coil)) for coil in coils])
        self._coils = coils

    @property
    def coils(self):
        return self._coils

    @property
    def curve(self):
        return [coil.curve for coil in self.coils]

    @property
    def current(self):
        return [coil.current for coil in self.coils]

    @current.setter
    def current(self, new):
        assert jnp.isscalar(new) or (len(jnp.asarray(new)) == len(self.curve))
        new = jnp.broadcast_to(
            jnp.asarray(new),
            len(
                self.coils,
            ),
        )
        for coil, cur in zip(self.coils, new):
            coil.current = cur

    @property
    def grid(self):
        """Default grid for computation."""
        return self.curve[0].grid

    @grid.setter
    def grid(self, new):
        if isinstance(new, Grid):
            grid = new
        elif isinstance(new, (np.ndarray, jnp.ndarray)):
            grid = Grid(new, sort=False)
        else:
            raise TypeError(
                f"grid should be a Grid or subclass, or ndarray, got {type(new)}"
            )
        for curve in self.curve:
            curve.grid = grid

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
        """translate the coils along an axis"""
        [coil.translate(*args, **kwargs) for coil in self.coils]

    def rotate(self, *args, **kwargs):
        """rotate the coils about an axis"""
        [coil.rotate(*args, **kwargs) for coil in self.coils]

    def flip(self, *args, **kwargs):
        """flip the coils across a plane"""
        [coil.flip(*args, **kwargs) for coil in self.coils]

    def compute_magnetic_field(self, coords, params=[{}], basis="rpz"):
        """Compute magnetic field at a set of points

        Parameters
        ----------
        coords : array-like shape(n,3) or Grid
            coordinates to evaluate field at [R,phi,Z] or [x,y,z]
        params : array-like of dict, optional
            parameters to pass to curves.
        basis : {"rpz", "xyz"}
            basis for input coordinates and returned magnetic field

        Returns
        -------
        field : ndarray, shape(n,3)
            magnetic field at specified points, in either rpz or xyz coordinates
        """
        B = 0
        for i, coil in enumerate(self):
            par = params[i % len(params)]
            B += coil.compute_magnetic_field(coords, par, basis)

        return B

    @classmethod
    def linspaced_angular(
        cls, curve, current, axis=[0, 0, 1], angle=2 * np.pi, n=10, endpoint=False
    ):
        """Create a coil set by repeating a curve n times rotationally.

        Parameters
        ----------
        curve : Curve
            base curve to repeat
        current : float or array-like, shape(n,)
            current in (each) coil
        axis : array-like, shape(3,)
            axis to rotate about
        angle : float
            total rotational extend of coil set.
        n : int
            number of copies of original coil
        endpoint : bool
            whether to include a coil at final angle
        """
        assert isinstance(curve, Curve) and not isinstance(curve, Coil)
        currents = jnp.broadcast_to(current, (n,))
        coils = []
        phis = jnp.linspace(0, angle, n, endpoint=endpoint)
        for i in range(n):
            coil = curve.copy()
            coil.rotate(axis, angle=phis[i])
            coils.append(Coil(coil, currents[i]))
        return cls(*coils)

    @classmethod
    def linspaced_linear(
        cls, curve, current, displacement=[2, 0, 0], n=4, endpoint=False
    ):
        """Create a coil group by repeating a curve n times in a straight line.

        Parameters
        ----------
        curve : Curve
            base curve to repeat
        current : float or array-like, shape(n,)
            current in (each) coil
        displacement : array-like, shape(3,)
            total displacement of the final coil
        n : int
            number of copies of original coil
        endpoint : bool
            whether to include a coil at final point
        """
        assert isinstance(curve, Curve) and not isinstance(curve, Coil)
        currents = jnp.broadcast_to(current, (n,))
        displacement = jnp.asarray(displacement)
        coils = []
        a = jnp.linspace(0, 1, n, endpoint=endpoint)
        for i in range(n):
            temp_curve = curve.copy()
            temp_curve.translate(a[i] * displacement)
            coils.append(Coil(temp_curve, currents[i]))
        return cls(*coils)

    @classmethod
    def from_symmetry(cls, coils, NFP, sym=False):
        """Create a coil group by reflection and symmetry

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
        if not isinstance(coils, (CoilGroup, CoilSet)):
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

    def __add__(self, other):
        if isinstance(other, (CoilSet)):
            return CoilSet(*self.coils, *other.coils)
        if isinstance(other, (CoilGroup)):
            return CoilGroup(*self.coils, *other.coils)
        if isinstance(other, (list, tuple)):
            return CoilSet(*self.coils, *other)
        raise TypeError

    # dunder methods required by MutableSequence
    def __getitem__(self, i):
        return self.coils[i]

    def __setitem__(self, i, new_item):
        if not (new_item.__class__ is Coil):
            raise TypeError("Members of CoilSet must be of type Coil.")
        self._coils[i] = new_item

    def __delitem__(self, i):
        del self._coils[i]

    def __len__(self):
        return len(self._coils)

    def insert(self, i, new_item):
        if not (new_item.__class__ is Coil):
            raise TypeError("Members of CoilSet must be of type Coil.")
        self._coils.insert(i, new_item)

    def __slice__(self, idx):
        if idx is None:
            theslice = slice(None, None)
        elif isinstance(idx, int):
            theslice = idx
        elif isinstance(idx, list):
            try:
                theslice = slice(idx[0], idx[1], idx[2])
            except IndexError:
                theslice = slice(idx[0], idx[1])
        else:
            raise TypeError("index is not a valid type.")
        return theslice
