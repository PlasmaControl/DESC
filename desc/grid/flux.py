"""Classes for representing flux coordinates."""

import numpy as np
from scipy import special

from desc.backend import jnp, put, repeat
from desc.utils import check_nonnegint, check_posint, errorif, setdefault

from .core import AbstractGrid
from .utils import midpoint_spacing, periodic_spacing


class AbstractGridFlux(AbstractGrid):
    """Base class for collocation grids in flux coordinates."""

    _io_attrs_ = AbstractGrid._io_attrs_ + ["_NFP", "_sym", "_axis"]

    _static_attrs = AbstractGrid._static_attrs + ["_NFP", "_sym"]

    def __repr__(self):
        """str: String form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + f" (coordinates={self.coordinates}, L={self.L}, M={self.M}, N={self.N}, "
            + f"NFP={self.NFP}, sym={self.sym}, is_meshgrid={self.is_meshgrid})"
        )

    def _set_up(self):
        """Do things after loading."""
        super()._set_up()

        # ensure things that should be ints are ints
        self._NFP = int(self._NFP)

        # backwards compatability
        if hasattr(self, "_unique_rho_idx"):
            self._unique_x0_idx = self._unique_rho_idx
            del self._unique_rho_idx
        if hasattr(self, "_unique_theta_idx"):
            self._unique_x1_idx = self._unique_theta_idx
            del self._unique_theta_idx
        if hasattr(self, "_unique_poloidal_idx"):
            self._unique_x1_idx = self._unique_poloidal_idx
            del self._unique_poloidal_idx
        if hasattr(self, "_unique_zeta_idx"):
            self._unique_x2_idx = self._unique_zeta_idx
            del self._unique_zeta_idx
        if hasattr(self, "_inverse_rho_idx"):
            self._inverse_x0_idx = self._inverse_rho_idx
            del self._inverse_rho_idx
        if hasattr(self, "_inverse_theta_idx"):
            self._inverse_x1_idx = self._inverse_theta_idx
            del self._inverse_theta_idx
        if hasattr(self, "_inverse_poloidal_idx"):
            self._inverse_x1_idx = self._inverse_poloidal_idx
            del self._inverse_theta_idx
        if hasattr(self, "_inverse_zeta_idx"):
            self._inverse_x2_idx = self._inverse_zeta_idx
            del self._inverse_zeta_idx

    def _enforce_symmetry(self):
        """Remove unnecessary nodes assuming poloidal symmetry.

        1. Remove nodes with θ > π.
        2. Rescale θ spacing to preserve dθ weight.
           Need to rescale on each θ coordinate curve by a different factor.
           dθ = 2π / number of nodes remaining on that θ curve.
           Nodes on the symmetry line should not be rescaled.

        """
        if not self.sym:
            return
        # indices where poloidal coordinate is off the symmetry line of
        # poloidal coord=0 or π
        off_sym_line_idx = self.nodes[:, 1] % np.pi != 0
        _, inverse, off_sym_line_per_rho_surf_count = np.unique(
            self.nodes[off_sym_line_idx, 0], return_inverse=True, return_counts=True
        )
        # indices of nodes to be deleted
        to_delete_idx = self.nodes[:, 1] > np.pi
        _, to_delete_per_rho_surf_count = np.unique(
            self.nodes[to_delete_idx, 0], return_counts=True
        )
        assert (
            2 * np.pi not in self.nodes[:, 1]
            and off_sym_line_per_rho_surf_count.size
            >= to_delete_per_rho_surf_count.size
        )
        if off_sym_line_per_rho_surf_count.size > to_delete_per_rho_surf_count.size:
            # edge case where surfaces closest to axis lack theta > π nodes
            # The number of nodes to delete on those surfaces is zero.
            pad_count = (
                off_sym_line_per_rho_surf_count.size - to_delete_per_rho_surf_count.size
            )
            to_delete_per_rho_surf_count = np.pad(
                to_delete_per_rho_surf_count, (pad_count, 0)
            )
        # The computation of this scale factor assumes
        # 1. number of nodes to delete is constant over zeta
        # 2. number of nodes off symmetry line is constant over zeta
        # 3. uniform poloidal spacing between nodes
        # The first two assumptions let _per_poloidal_curve = _per_rho_surf.
        # The third assumption lets the scale factor be constant over a
        # particular theta curve, so that each node in the open interval
        # (0, π) has its spacing scaled up by the same factor.
        # Nodes at endpoints 0, π should not be scaled.
        scale = off_sym_line_per_rho_surf_count / (
            off_sym_line_per_rho_surf_count - to_delete_per_rho_surf_count
        )
        # Arrange scale factors to match spacing's arbitrary ordering.
        scale = scale[inverse]

        # Scale up all nodes so that their spacing accounts for the node
        # that is their reflection across the symmetry line.
        self._spacing[off_sym_line_idx, 1] *= scale
        self._nodes = self.nodes[~to_delete_idx]
        self._spacing = self.spacing[~to_delete_idx]

    def _find_axis(self):
        """Find indices of axis nodes."""
        return np.nonzero(self.nodes[:, 0] == 0)[0]

    def compress(self, x, surface_label="rho"):
        """Return elements of ``x`` at indices of unique surface label values.

        Parameters
        ----------
        x : ndarray
            The array to compress. Should usually represent a surface function (constant
            over a surface) in an array that matches the grid's pattern.
        surface_label : str, optional
            The surface label. Must be one of the elements in self.coordinates.
            Default = "rho".

        Returns
        -------
        compress_x : ndarray
            This array will be sorted such that the first element corresponds to
            the value associated with the smallest surface, and the last element
            corresponds to the value associated with the largest surface.

        """
        return super().compress(x, surface_label)

    def expand(self, x, surface_label="rho"):
        """Expand ``x`` by duplicating elements to match the grid's pattern.

        Parameters
        ----------
        x : ndarray
            Stores the values of a surface function (constant over a surface) for all
            unique surfaces of the specified label on the grid. The length of ``x``
            should match the number of unique surfaces of the corresponding label in
            this grid. ``x`` should be sorted such that the first element corresponds to
            the value associated with the smallest surface, and the last element
            corresponds to the value associated with the largest surface.
        surface_label : str, optional
            The surface label. Must be one of the elements in self.coordinates.
            Default = "rho".

        Returns
        -------
        expand_x : ndarray
            ``x`` expanded to match the grid's pattern.

        """
        return super().expand(x, surface_label)

    def copy_data_from_other(self, x, other_grid, surface_label="rho", tol=1e-14):
        """Copy data x from other_grid to this grid at matching surface label.

        Given data x corresponding to nodes of other_grid, copy data to a new array that
        corresponds to this grid.

        Parameters
        ----------
        x : ndarray, shape(other_grid.num_nodes,...)
            Data to copy. Assumed to be constant over the specified surface.
        other_grid: AbstractGridFlux
            Grid to copy from.
        surface_label : str, optional
            The surface label. Must be one of the elements in self.coordinates.
            Default = "rho".
        tol : float, optional
            Tolerance for considering nodes the same. Default = 1e-14.

        Returns
        -------
        y : ndarray, shape(grid2.num_nodes, ...)
            Data copied to grid2

        """
        return super().copy_data_from_other(x, other_grid, surface_label, tol)

    def get_label(self, label):
        """Get general label that specifies the direction of given coordinate label."""
        if label in {"rho", "poloidal", "zeta"}:
            return label
        rad = {"r": "rho"}[self.coordinates[0]]
        pol = {"a": "alpha", "t": "theta", "v": "theta_PEST"}[self.coordinates[1]]
        tor = {"z": "zeta"}[self.coordinates[2]]
        return {rad: "rho", pol: "poloidal", tor: "zeta"}[label]

    def get_label_axis(self, label):
        """Get node axis index associated with given coordinate label."""
        label = self.get_label(label)
        return {"rho": 0, "poloidal": 1, "zeta": 2}[label]

    def replace_at_axis(self, x, y, copy=False, **kwargs):
        """Replace elements of ``x`` with elements of ``y`` at the axis of grid.

        Parameters
        ----------
        x : array-like
            Values to selectively replace. Should have length ``grid.num_nodes``.
        y : array-like
            Replacement values. Should broadcast with arrays of size
            ``grid.num_nodes``. Can also be a function that returns such an
            array. Additional keyword arguments are then input to ``y``.
        copy : bool
            If some value of ``x`` is to be replaced by some value in ``y``,
            then setting ``copy`` to true ensures that ``x`` will not be
            modified in-place.

        Returns
        -------
        out : ndarray
            An array of size ``grid.num_nodes`` where elements at the indices
            corresponding to the axis of this grid match those of ``y`` and all
            others match ``x``.

        """
        if self.axis.size:
            if callable(y):
                y = y(**kwargs)
            x = put(
                x.copy() if copy else x,
                self.axis,
                y[self.axis] if jnp.ndim(y) else y,
            )
        return x

    @property
    def coordinates(self):
        """Coordinates specified by the nodes.

        Examples
        --------
        rtz : rho, theta, zeta
        rvp : rho, theta_PEST, phi
        raz : rho, alpha, zeta
        """
        return self.__dict__.setdefault("_coordinates", "rtz")

    @property
    def bounds(self):
        """Bounds of coordinates."""
        return ((0, 1), (0, 2 * np.pi), (0, 2 * np.pi))

    @property
    def period(self):
        """Periodicity of coordinates."""
        if self.coordinates in ["rtz", "rvp"]:
            return (np.inf, 2 * np.pi, 2 * np.pi / self.NFP)
        else:
            return (np.inf, np.inf, np.inf)

    @property
    def num_rho(self):
        """int: Number of unique rho coordinates."""
        return self.num_x0

    @property
    def num_poloidal(self):
        """int: Number of unique poloidal angle coordinates."""
        return self.num_x1

    @property
    def num_alpha(self):
        """ndarray: Number of unique field line poloidal angles."""
        errorif(self.coordinates[1] != "a", AttributeError)
        return self.num_x1

    @property
    def num_theta(self):
        """ndarray: Number of unique theta coordinates."""
        errorif(self.coordinates[1] != "t", AttributeError)
        return self.num_x1

    @property
    def num_theta_PEST(self):
        """ndarray: Number of unique straight field line poloidal angles."""
        errorif(self.coordinates[1] != "v", AttributeError)
        return self.num_x1

    @property
    def num_zeta(self):
        """int: Number of unique zeta coordinates."""
        return self.num_x2

    @property
    def unique_rho_idx(self):
        """ndarray: Indices of unique rho coordinates."""
        return self.unique_x0_idx

    @property
    def unique_poloidal_idx(self):
        """ndarray: Indices of unique poloidal angle coordinates."""
        return self.unique_x1_idx

    @property
    def unique_alpha_idx(self):
        """ndarray: Indices of unique field line poloidal angles."""
        errorif(self.coordinates[1] != "a", AttributeError)
        return self.unique_x1_idx

    @property
    def unique_theta_idx(self):
        """ndarray: Indices of unique theta coordinates."""
        errorif(self.coordinates[1] != "t", AttributeError)
        return self.unique_x1_idx

    @property
    def unique_theta_PEST_idx(self):
        """ndarray: Indices of unique straight field line poloidal angles."""
        errorif(self.coordinates[1] != "v", AttributeError)
        return self.unique_x1_idx

    @property
    def unique_zeta_idx(self):
        """ndarray: Indices of unique zeta coordinates."""
        return self.unique_x2_idx

    @property
    def inverse_rho_idx(self):
        """ndarray: Indices that recover the rho coordinates."""
        return self.inverse_x0_idx

    @property
    def inverse_poloidal_idx(self):
        """ndarray: Indices the recover the unique poloidal angle coordinates."""
        return self.inverse_x1_idx

    @property
    def inverse_alpha_idx(self):
        """ndarray: Indices that recover the field line poloidal angles."""
        errorif(self.coordinates[1] != "a", AttributeError)
        return self.inverse_x1_idx

    @property
    def inverse_theta_idx(self):
        """ndarray: Indices that recover the theta coordinates."""
        errorif(self.coordinates[1] != "t", AttributeError)
        return self.inverse_x1_idx

    @property
    def inverse_theta_PEST_idx(self):
        """ndarray: Indices that recover the straight field line poloidal angles."""
        errorif(self.coordinates[1] != "v", AttributeError)
        return self.inverse_x1_idx

    @property
    def inverse_zeta_idx(self):
        """ndarray: Indices that recover the zeta coordinates."""
        return self.inverse_x2_idx

    @property
    def NFP(self):
        """int: Number of (toroidal) field periods."""
        return self.__dict__.setdefault("_NFP", 1)

    @property
    def sym(self):
        """bool: ``True`` for poloidal up/down symmetry, ``False`` otherwise.

        Whether the poloidal domain of this grid is truncated to [0, π] ⊂ [0, 2π)
        to take advantage of poloidal up/down symmetry,
        which is a stronger condition than stellarator symmetry.
        Still, when stellarator symmetry exists, flux surface integrals and
        volume integrals are invariant to this truncation.
        """
        return self.__dict__.setdefault("_sym", False)

    @property
    def axis(self):
        """ndarray: Indices of nodes at magnetic axis."""
        return self.__dict__.setdefault("_axis", np.array([]))


class Grid(AbstractGridFlux):
    """Collocation grid with custom node placement.

    Unlike subclasses LinearGrid and ConcentricGrid, the base Grid allows the user
    to pass in a custom set of collocation nodes.

    Parameters
    ----------
    nodes : ndarray of float, size(num_nodes,3)
        Node coordinates, in (rho,theta,zeta)
    spacing : ndarray of float, size(num_nodes, 3)
        Spacing between nodes in each direction.
    weights : ndarray of float, size(num_nodes, )
        Quadrature weights for each node.
    coordinates : str
        Coordinates that are specified by the nodes.
        raz : rho, alpha, zeta
        rvp : rho, theta_PEST, phi
        rtz : rho, theta, zeta
    period : tuple of float
        Assumed periodicity for each coordinate.
        Use np.inf to denote no periodicity.
    NFP : int
        Number of field periods (Default = 1).
        Change this only if your nodes are placed within one field period.
    source_grid : AbstractGridFlux
        Grid from which coordinates were mapped from.
    sort : bool
        Whether to sort the nodes for use with FFT method.
    is_meshgrid : bool
        Whether this grid is a tensor-product grid.
        Let the tuple (r, p, t) ∈ R³ denote a radial, poloidal, and toroidal
        coordinate value. The is_meshgrid flag denotes whether any coordinate
        can be iterated over along the relevant axis of the reshaped grid:
        nodes.reshape((num_poloidal, num_radial, num_toroidal, 3), order="F").
    jitable : bool
        Whether to skip certain checks and conditionals that don't work under jit.
        Allows grid to be created on the fly with custom nodes, but weights,
        symmetry etc. may be wrong if grid contains duplicate nodes.
    """

    _io_attrs_ = AbstractGridFlux._io_attrs_ + ["_source_grid"]

    def __init__(
        self,
        nodes,
        spacing=None,
        weights=None,
        coordinates="rtz",
        period=None,
        NFP=1,
        source_grid=None,
        sort=False,
        is_meshgrid=False,
        jitable=False,
        **kwargs,
    ):
        # Python 3.3 (PEP 412) introduced key-sharing dictionaries.
        # This change measurably reduces memory usage of objects that
        # define all attributes in their __init__ method.
        self._NFP = check_posint(NFP, "NFP", False)
        self._sym = False
        self._coordinates = coordinates
        self._period = setdefault(
            period,
            (
                (np.inf, 2 * np.pi, 2 * np.pi / NFP)
                if coordinates == "rtz"
                else (np.inf, np.inf, np.inf)
            ),
        )
        self._source_grid = source_grid
        self._is_meshgrid = bool(is_meshgrid)
        # If you are using a custom grid it almost always is not uniform, or is under
        # jit where we cannot properly check this anyways, so just set to False.
        self._fft_x1 = False
        self._fft_x2 = False
        self._nodes = self._create_nodes(nodes)
        self._spacing = (
            jnp.atleast_2d(jnp.asarray(spacing)).reshape(self.nodes.shape).astype(float)
            if spacing is not None
            else None
        )
        self._weights = (
            jnp.atleast_1d(jnp.asarray(weights))
            .reshape(self.nodes.shape[0])
            .astype(float)
            if weights is not None
            else None
        )
        if sort:
            self._sort_nodes()
        setable_attr = [
            "_unique_x0_idx",
            "_unique_x1_idx",
            "_unique_x2_idx",
            "_inverse_x0_idx",
            "_inverse_x1_idx",
            "_inverse_x2_idx",
        ]
        if jitable:
            # Don't do anything with symmetry since that changes number of nodes
            # avoid point at the axis, for now.
            r, t, z = self._nodes.T
            r = jnp.where(r == 0, kwargs.pop("axis_shift", 1e-12), r)
            self._nodes = jnp.column_stack([r, t, z])
            self._axis = np.array([], dtype=int)
            # allow for user supplied indices/inverse indices for special cases
            for attr in setable_attr:
                if attr in kwargs:
                    setattr(self, attr, jnp.asarray(kwargs.pop(attr)))
        else:
            for attr in setable_attr:
                kwargs.pop(attr, None)
            self._axis = self._find_axis()
            (
                self._unique_x0_idx,
                self._inverse_x0_idx,
                self._unique_x1_idx,
                self._inverse_x1_idx,
                self._unique_x2_idx,
                self._inverse_x2_idx,
            ) = self._find_unique_inverse_nodes()
        # Assign with logic in setter method if possible else 0.
        self._L = self.num_x0 - 1 if hasattr(self, "num_x0") else 0
        self._M = (
            (self.num_x1 - 1 if self.sym else self.num_x1 // 2)
            if hasattr(self, "num_x1")
            else 0
        )
        self._N = self.num_x2 // 2 if hasattr(self, "num_x2") else 0
        errorif(len(kwargs), ValueError, f"Got unexpected kwargs {kwargs.keys()}.")

    def _create_nodes(self, nodes):
        """Allow for custom node creation.

        Parameters
        ----------
        nodes : ndarray of float, size(num_nodes,3)
            Node coordinates, in (rho,theta,zeta).

        Returns
        -------
        nodes : ndarray of float, size(num_nodes,3)
            Node coordinates, in (rho,theta,zeta).

        """
        nodes = jnp.atleast_2d(jnp.asarray(nodes)).reshape((-1, 3)).astype(float)
        # Do not alter nodes given by the user for custom grids.
        # In particular, do not modulo nodes by 2π or 2π/NFP.
        return nodes

    def _sort_nodes(self):
        """Sort nodes for use with FFT."""
        sort_idx = np.lexsort((self.nodes[:, 1], self.nodes[:, 0], self.nodes[:, 2]))
        self._nodes = self.nodes[sort_idx]
        try:
            self._spacing = self.spacing[sort_idx]
        except AttributeError:
            pass
        try:
            self._weights = self.weights[sort_idx]
        except AttributeError:
            pass

    @staticmethod
    def create_meshgrid(
        nodes,
        spacing=None,
        coordinates="rtz",
        period=None,
        NFP=1,
        jitable=True,
        **kwargs,
    ):
        """Create a tensor-product grid from the given coordinates in a jitable manner.

        Parameters
        ----------
        nodes : list of ndarray
            Three arrays, one for each coordinate.
            Unique values of each coordinate sorted in increasing order.
        spacing : list of ndarray
            Three arrays, one for each coordinate.
            Weights for integration. Defaults to a midpoint rule.
        coordinates : str
            Coordinates that are specified by the ``nodes[0]``, ``nodes[1]``,
            and ``nodes[2]``, respectively.
            raz : rho, alpha, zeta
            rvp : rho, theta_PEST, phi
            rtz : rho, theta, zeta
        period : tuple of float
            Assumed periodicity for each coordinate.
            Use ``np.inf`` to denote no periodicity.
        NFP : int
            Number of field periods (Default = 1).
            Only makes sense to change from 1 if last coordinate is periodic
            with some constant divided by ``NFP`` and the nodes are placed
            within one field period.
        jitable : bool
            Whether to skip certain checks and conditionals that don't work under jit.
            Allows grid to be created on the fly with custom nodes, but weights,
            symmetry etc. may be wrong if grid contains duplicate nodes.

        Returns
        -------
        grid : Grid
            Meshgrid.

        """
        NFP = check_posint(NFP, "NFP", False)
        period = setdefault(
            period,
            (
                (np.inf, 2 * np.pi, 2 * np.pi / NFP)
                if coordinates == "rtz"
                else (np.inf, np.inf, np.inf)
            ),
        )
        a, b, c = jnp.atleast_1d(*nodes)
        if spacing is None:
            errorif(coordinates[0] != "r", NotImplementedError)
            da = midpoint_spacing(a)
            db = periodic_spacing(b, period[1])[1]
            dc = periodic_spacing(c, period[2])[1] * NFP
        else:
            da, db, dc = spacing

        bb, aa, cc = jnp.meshgrid(b, a, c, indexing="ij")

        nodes = jnp.column_stack(
            [aa.flatten(order="F"), bb.flatten(order="F"), cc.flatten(order="F")]
        )
        bb, aa, cc = jnp.meshgrid(db, da, dc, indexing="ij")

        spacing = jnp.column_stack(
            [aa.flatten(order="F"), bb.flatten(order="F"), cc.flatten(order="F")]
        )
        weights = (
            spacing.prod(axis=1)
            if period[1] * period[2] == 4 * np.pi**2 / NFP
            # Doesn't make sense to assign weights if the coordinates aren't periodic
            # since it's not clear how to form a surface and hence its enclosed volume.
            else None
        )

        unique_a_idx = jnp.arange(a.size) * b.size
        unique_b_idx = jnp.arange(b.size)
        unique_c_idx = jnp.arange(c.size) * a.size * b.size
        inverse_a_idx = jnp.tile(
            repeat(unique_a_idx // b.size, b.size, total_repeat_length=a.size * b.size),
            c.size,
        )
        inverse_b_idx = jnp.tile(unique_b_idx, a.size * c.size)
        inverse_c_idx = repeat(unique_c_idx // (a.size * b.size), (a.size * b.size))
        return Grid(
            nodes=nodes,
            spacing=spacing,
            weights=weights,
            coordinates=coordinates,
            period=period,
            NFP=NFP,
            sort=False,
            is_meshgrid=True,
            jitable=jitable,
            _unique_x0_idx=unique_a_idx,
            _unique_x1_idx=unique_b_idx,
            _unique_x2_idx=unique_c_idx,
            _inverse_x0_idx=inverse_a_idx,
            _inverse_x1_idx=inverse_b_idx,
            _inverse_x2_idx=inverse_c_idx,
            **kwargs,
        )

    @property
    def source_grid(self):
        """Coordinates from which this grid was mapped from."""
        errorif(self._source_grid is None, AttributeError)
        return self._source_grid


class LinearGrid(AbstractGridFlux):
    """Grid in which the nodes are linearly spaced in each coordinate.

    Useful for plotting and other analysis, though not very efficient for using as the
    solution grid.

    Parameters
    ----------
    L : int, optional
        Radial grid resolution.
    M : int, optional
        Poloidal grid resolution.
    N : int, optional
        Toroidal grid resolution.
    NFP : int
        Number of field periods (Default = 1).
        Change this only if your nodes are placed within one field period
        or should be interpreted as spanning one field period.
    sym : bool
        ``True`` for poloidal up/down symmetry, ``False`` otherwise.
        Default is ``False``.
        Whether to truncate the poloidal domain to [0, π] ⊂ [0, 2π)
        to take advantage of poloidal up/down symmetry,
        which is a stronger condition than stellarator symmetry.
        Still, when stellarator symmetry exists, flux surface integrals and
        volume integrals are invariant to this truncation.
    axis : bool
        True to include a point at rho=0 (default), False for rho[0] = rho[1]/2.
    endpoint : bool
        If True, theta=0 and zeta=0 are duplicated after a full period.
        Should be False for use with FFT. (Default = False).
        This boolean is ignored if an array is given for theta or zeta.
    rho : int or ndarray of float, optional
        Radial coordinates (Default = 1.0).
        Alternatively, the number of radial coordinates (if an integer).
        Note that if supplied the values may be reordered in the resulting grid.
    theta : int or ndarray of float, optional
        Poloidal coordinates (Default = 0.0).
        Alternatively, the number of poloidal coordinates (if an integer).
        Note that if supplied the values may be reordered in the resulting grid.
    zeta : int or ndarray of float, optional
        Toroidal coordinates (Default = 0.0).
        Alternatively, the number of toroidal coordinates (if an integer).
        Note that if supplied the values may be reordered in the resulting grid.
    """

    _io_attrs_ = AbstractGridFlux._io_attrs_ + [
        "_poloidal_endpoint",
        "_toroidal_endpoint",
    ]

    _static_attrs = AbstractGridFlux._static_attrs + ["_endpoint"]

    def __init__(
        self,
        L=None,
        M=None,
        N=None,
        NFP=1,
        sym=False,
        axis=True,
        endpoint=False,
        rho=None,
        theta=None,
        zeta=None,
    ):
        assert (L is None) or (rho is None), "cannot specify both L and rho"
        assert (M is None) or (theta is None), "cannot specify both M and theta"
        assert (N is None) or (zeta is None), "cannot specify both N and zeta"
        self._L = check_nonnegint(L, "L")
        self._M = check_nonnegint(M, "M")
        self._N = check_nonnegint(N, "N")
        self._NFP = check_posint(NFP, "NFP", False)
        self._sym = sym
        self._endpoint = bool(endpoint)
        self._is_meshgrid = True
        self._fft_x1 = False
        self._fft_x2 = False
        self._can_fft2 = not sym and not endpoint
        # these are just default values that may get overwritten in _create_nodes
        self._poloidal_endpoint = False
        self._toroidal_endpoint = False

        self._nodes, self._spacing = self._create_nodes(
            L=L,
            M=M,
            N=N,
            NFP=NFP,
            axis=axis,
            endpoint=endpoint,
            rho=rho,
            theta=theta,
            zeta=zeta,
        )
        # symmetry handled in create_nodes()
        self._sort_nodes()
        self._axis = self._find_axis()
        (
            self._unique_x0_idx,
            self._inverse_x0_idx,
            self._unique_x1_idx,
            self._inverse_x1_idx,
            self._unique_x2_idx,
            self._inverse_x2_idx,
        ) = self._find_unique_inverse_nodes()
        self._weights = self._scale_weights()

    def _create_nodes(  # noqa: C901
        self,
        L=None,
        M=None,
        N=None,
        NFP=1,
        axis=True,
        endpoint=False,
        rho=1.0,
        theta=0.0,
        zeta=0.0,
    ):
        """Create grid nodes and weights.

        Parameters
        ----------
        L : int, optional
            Radial grid resolution.
        M : int, optional
            Poloidal grid resolution.
        N : int, optional
            Toroidal grid resolution.
        NFP : int
            Number of field periods (Default = 1).
            Only change this if your nodes are placed within one field period
            or should be interpreted as spanning one field period.
        axis : bool
            True to include a point at rho=0 (default), False for rho[0] = rho[1]/2.
        endpoint : bool
            If True, theta=0 and zeta=0 are duplicated after a full period.
            Should be False for use with FFT. (Default = False).
            This boolean is ignored if an array is given for theta or zeta.
        rho : int or ndarray of float, optional
            Radial coordinates (Default = 1.0).
            Alternatively, the number of radial coordinates (if an integer).
        theta : int or ndarray of float, optional
            Poloidal coordinates (Default = 0.0).
            Alternatively, the number of poloidal coordinates (if an integer).
        zeta : int or ndarray of float, optional
            Toroidal coordinates (Default = 0.0).
            Alternatively, the number of toroidal coordinates (if an integer).

        Returns
        -------
        nodes : ndarray of float, size(num_nodes,3)
            node coordinates, in (rho,theta,zeta)
        spacing : ndarray of float, size(num_nodes,3)
            node spacing, based on local volume around the node

        """
        # TODO:
        #  https://github.com/PlasmaControl/DESC/pull/1204#pullrequestreview-2246771337
        self._NFP = check_posint(NFP, "NFP", False)
        axis = bool(axis)
        endpoint = bool(endpoint)
        theta_period = self.period[1]
        zeta_period = self.period[2]

        # rho
        if L is not None:
            self._L = check_nonnegint(L, "L")
            rho = L + 1
        if np.isscalar(rho) and (int(rho) == rho) and rho > 0:
            r = np.flipud(np.linspace(1, 0, int(rho), endpoint=axis))
            # choose dr such that each node has the same weight
            dr = np.ones_like(r) / r.size
        elif rho is not None:
            r = np.sort(np.atleast_1d(rho))
            dr = midpoint_spacing(r, jnp=np)
        else:
            r = np.array(1.0, ndmin=1)
            dr = np.ones_like(r)

        # theta
        if M is not None:
            self._M = check_nonnegint(M, "M")
            theta = 2 * (M + 1) if self.sym else 2 * M + 1
        if np.isscalar(theta) and (int(theta) == theta) and theta > 0:
            theta = int(theta)
            if self.sym and theta > 1:
                # Enforce that no node lies on theta=0 or theta=2π, so that
                # each node has a symmetric counterpart, and that, for all i,
                # t[i]-t[i-1] = 2 t[0] = 2 (π - t[last node before π]).
                # Both conditions necessary to evenly space nodes with constant dt.
                # This can be done by making (theta + endpoint) an even integer.
                if (theta + endpoint) % 2 != 0:
                    theta += 1
                t = np.linspace(0, theta_period, theta, endpoint=endpoint)
                t += t[1] / 2
                # delete theta > π nodes
                t = t[: np.searchsorted(t, np.pi, side="right")]
            else:
                t = np.linspace(0, theta_period, theta, endpoint=endpoint)
            dt = theta_period / t.size * np.ones_like(t)
            if (endpoint and not self.sym) and t.size > 1:
                # increase node weight to account for duplicate node
                dt *= t.size / (t.size - 1)
                # scale_weights() will reduce endpoint (dt[0] and dt[-1])
                # duplicate node weight
            # if custom theta used usually safe to assume its non-uniform so no fft
            self._fft_x1 = (not endpoint) and (not self.sym)
        elif theta is not None:
            t = np.atleast_1d(theta).astype(float)
            # enforce periodicity
            t[t != theta_period] %= theta_period
            # need to sort to compute correct spacing
            t = np.sort(t)
            if self.sym:
                # cut domain to relevant subdomain: delete theta > π nodes
                t = t[: np.searchsorted(t, np.pi, side="right")]
            if t.size > 1:
                if not self.sym:
                    dt = periodic_spacing(t, theta_period, jnp=np)[1]
                    if t[0] == 0 and t[-1] == theta_period:
                        # periodic_spacing above correctly weights
                        # the duplicate endpoint node spacing at theta = 0 and 2π
                        # to be half the weight of the other nodes.
                        # However, scale_weights() is not aware of this, so we
                        # counteract the reduction that will be done there.
                        dt[0] += dt[-1]
                        dt[-1] = dt[0]
                else:
                    dt = np.zeros(t.shape)
                    dt[1:-1] = t[2:] - t[:-2]
                    first_positive_idx = np.searchsorted(t, 0, side="right")
                    # total spacing of nodes at theta=0 should be half the
                    # distance between first positive node and its
                    # reflection across the theta=0 line.
                    dt[0] = t[first_positive_idx]
                    if first_positive_idx == 0:
                        # then there are no nodes at theta=0
                        dt[0] += t[1]
                    else:
                        assert dt[0] == dt[first_positive_idx - 1]
                        # If first_positive_idx != 1,
                        # then both of those dt should be halved.
                        # The scale_weights() function will handle this.
                    first_pi_idx = np.searchsorted(t, np.pi, side="left")
                    # total spacing of nodes at theta=π should be half the
                    # distance between first node < π and its
                    # reflection across the theta=π line.
                    if first_pi_idx == t.size:
                        # then there are no nodes at theta=π
                        dt[-1] = (theta_period - t[-1]) - t[-2]
                    else:
                        dt[-1] = (theta_period - t[-1]) - t[first_pi_idx - 1]
                        assert dt[first_pi_idx] == dt[-1]
                        # If first_pi_idx != t.size - 1,
                        # then both of those dt should be halved.
                        # The scale_weights() function will handle this.
            else:
                dt = np.array([theta_period])
        else:
            t = np.array(0.0, ndmin=1)
            dt = theta_period * np.ones_like(t)
            self._fft_x1 = not self.sym

        # zeta
        # note: dz spacing should not depend on NFP
        # spacing corresponds to a node's weight in an integral --
        # such as integral = sum(dt * dz * data["B"]) -- not the node's coordinates
        if N is not None:
            self._N = check_nonnegint(N, "N")
            zeta = 2 * N + 1
        if np.isscalar(zeta) and (int(zeta) == zeta) and zeta > 0:
            zeta = int(zeta)
            z = np.linspace(0, zeta_period, zeta, endpoint=endpoint)
            dz = 2 * np.pi / z.size * np.ones_like(z)
            if endpoint and z.size > 1:
                # increase node weight to account for duplicate node
                dz *= z.size / (z.size - 1)
                # scale_weights() will reduce endpoint (dz[0] and dz[-1])
                # duplicate node weight
            # if custom zeta used usually safe to assume its non-uniform so no fft
            self._fft_x2 = not endpoint
        elif zeta is not None:
            errorif(
                np.any(np.asarray(zeta) > zeta_period),
                msg="LinearGrid should be defined on 1 field period.",
            )
            z, dz = periodic_spacing(zeta, zeta_period, sort=True, jnp=np)
            dz = dz * NFP
            if z[0] == 0 and z[-1] == zeta_period:
                # periodic_spacing above correctly weights
                # the duplicate node spacing at zeta = 0 and 2π/NFP.
                # However, scale_weights() is not aware of this, so we
                # counteract the reduction that will be done there.
                dz[0] += dz[-1]
                dz[-1] = dz[0]
        else:
            z = np.array(0.0, ndmin=1)
            dz = zeta_period * np.ones_like(z) * NFP
            self._fft_x2 = True  # trivially true

        self._poloidal_endpoint = (
            t.size > 0
            and np.isclose(t[0], 0, atol=1e-12)
            and np.isclose(t[-1], theta_period, atol=1e-12)
        )
        self._toroidal_endpoint = (
            z.size > 0
            and np.isclose(z[0], 0, atol=1e-12)
            and np.isclose(z[-1], zeta_period, atol=1e-12)
        )
        # if only one theta or one zeta point, can have endpoint=True
        # if the other one is a full array
        self._endpoint = (self._poloidal_endpoint or (t.size == 1 and z.size > 1)) and (
            self._toroidal_endpoint or (z.size == 1 and t.size > 1)
        )
        self._can_fft2 = (
            self._can_fft2
            and not self._poloidal_endpoint
            and not self._toroidal_endpoint
        )

        r, t, z = map(np.ravel, np.meshgrid(r, t, z, indexing="ij"))
        dr, dt, dz = map(np.ravel, np.meshgrid(dr, dt, dz, indexing="ij"))
        nodes = np.column_stack([r, t, z])
        spacing = np.column_stack([dr, dt, dz])

        return nodes, spacing

    def change_resolution(self, L, M, N, NFP=None):
        """Change the resolution of the grid.

        Parameters
        ----------
        L : int
            new radial grid resolution (L radial nodes)
        M : int
            new poloidal grid resolution (M poloidal nodes)
        N : int
            new toroidal grid resolution (N toroidal nodes)
        NFP : int
            Number of field periods.

        """
        if NFP is None:
            NFP = self.NFP
        if L != self.L or M != self.M or N != self.N or NFP != self.NFP:
            self._nodes, self._spacing = self._create_nodes(
                L=L, M=M, N=N, NFP=NFP, axis=self.axis.size > 0, endpoint=self.endpoint
            )
            # symmetry handled in create_nodes()
            self._sort_nodes()
            self._axis = self._find_axis()
            (
                self._unique_x0_idx,
                self._inverse_x0_idx,
                self._unique_x1_idx,
                self._inverse_x1_idx,
                self._unique_x2_idx,
                self._inverse_x2_idx,
            ) = self._find_unique_inverse_nodes()
            self._weights = self._scale_weights()

    @property
    def L(self):
        """int: Radial coordinate resolution."""
        if self._L is None:
            self._L = self.num_rho - 1
        return self._L

    @property
    def M(self):
        """int: Poloidal coordinate resolution."""
        if self._M is None:
            self._M = self.num_poloidal - 1 if self.sym else self.num_poloidal // 2
        return self._M

    @property
    def N(self):
        """int: Toroidal coordinate resolution."""
        if self._N is None:
            self._N = self.num_zeta // 2
        return self._N

    @property
    def endpoint(self):
        """bool: Whether the grid is made of open or closed intervals."""
        return self.__dict__.setdefault("_endpoint", False)


class QuadratureGrid(AbstractGridFlux):
    """Grid used for numerical quadrature.

    Exactly integrates a Fourier-Zernike basis of resolution (L,M,N)
    This grid is never symmetric.

    Parameters
    ----------
    L : int
        radial grid resolution (exactly integrates radial modes up to order L)
    M : int
        poloidal grid resolution (exactly integrates poloidal modes up to order M)
    N : int
        toroidal grid resolution (exactly integrates toroidal modes up to order N)
    NFP : int
        number of field periods (Default = 1)

    """

    def __init__(self, L, M, N, NFP=1):
        self._L = check_nonnegint(L, "L", False)
        self._M = check_nonnegint(M, "N", False)
        self._N = check_nonnegint(N, "N", False)
        self._NFP = check_posint(NFP, "NFP", False)
        self._sym = False
        self._is_meshgrid = True
        self._fft_x1 = True
        self._fft_x2 = True
        self._nodes, self._spacing = self._create_nodes(L=L, M=M, N=N, NFP=NFP)
        # symmetry is never enforced for Quadrature Grid
        self._sort_nodes()
        self._axis = self._find_axis()
        (
            self._unique_x0_idx,
            self._inverse_x0_idx,
            self._unique_x1_idx,
            self._inverse_x1_idx,
            self._unique_x2_idx,
            self._inverse_x2_idx,
        ) = self._find_unique_inverse_nodes()
        # quadrature weights do not need scaling
        self._weights = self.spacing.prod(axis=1)

    def _create_nodes(self, L=1, M=1, N=1, NFP=1):
        """Create grid nodes and weights.

        Parameters
        ----------
        L : int
            radial grid resolution (L radial nodes, Default = 1)
        M : int
            poloidal grid resolution (M poloidal nodes, Default = 1)
        N : int
            toroidal grid resolution (N toroidal nodes, Default = 1)
        NFP : int
            number of field periods (Default = 1)

        Returns
        -------
        nodes : ndarray of float, size(num_nodes,3)
            node coordinates, in (rho,theta,zeta)
        spacing : ndarray of float, size(num_nodes,3)
            node spacing, based on local volume around the node

        """
        self._L = check_nonnegint(L, "L", False)
        self._M = check_nonnegint(M, "M", False)
        self._N = check_nonnegint(N, "N", False)
        self._NFP = check_posint(NFP, "NFP", False)
        # floor divide (L+2) by 2 bc only need (L+1)/2  points to
        # integrate L-th order jacobi polynomial exactly, so this
        # ensures we have enough pts for both odd and even L
        L = (L + 2) // 2
        M = 2 * M + 1
        N = 2 * N + 1

        # rho
        r, dr = special.js_roots(L, 2, 2)
        dr /= r  # remove r weight function associated with the shifted Jacobi weights

        # theta/vartheta
        t = np.linspace(0, 2 * np.pi, M, endpoint=False)
        dt = 2 * np.pi / M * np.ones_like(t)

        # zeta/phi
        z = np.linspace(0, 2 * np.pi / NFP, N, endpoint=False)
        dz = 2 * np.pi / N * np.ones_like(z)

        r, t, z = map(np.ravel, np.meshgrid(r, t, z, indexing="ij"))
        dr, dt, dz = map(np.ravel, np.meshgrid(dr, dt, dz, indexing="ij"))

        nodes = np.column_stack([r, t, z])
        spacing = np.column_stack([dr, dt, dz])

        return nodes, spacing

    def change_resolution(self, L, M, N, NFP=None):
        """Change the resolution of the grid.

        Parameters
        ----------
        L : int
            new radial grid resolution (L radial nodes)
        M : int
            new poloidal grid resolution (M poloidal nodes)
        N : int
            new toroidal grid resolution (N toroidal nodes)
        NFP : int
            Number of field periods.

        """
        if NFP is None:
            NFP = self.NFP
        if L != self.L or M != self.M or N != self.N or NFP != self.NFP:
            self._nodes, self._spacing = self._create_nodes(L=L, M=M, N=N, NFP=NFP)
            self._sort_nodes()
            self._axis = self._find_axis()
            (
                self._unique_x0_idx,
                self._inverse_x0_idx,
                self._unique_x1_idx,
                self._inverse_x1_idx,
                self._unique_x2_idx,
                self._inverse_x2_idx,
            ) = self._find_unique_inverse_nodes()
            self._weights = self.spacing.prod(axis=1)  # instead of _scale_weights


class ConcentricGrid(AbstractGridFlux):
    """Grid in which the nodes are arranged in concentric circles.

    Nodes are arranged concentrically within each toroidal cross-section, with more
    nodes per flux surface at larger radius. Typically used as the solution grid,
    cannot be easily used for plotting due to non-uniform spacing.

    Parameters
    ----------
    L : int
        radial grid resolution
    M : int
        poloidal grid resolution
    N : int
        toroidal grid resolution
    NFP : int
        number of field periods (Default = 1)
    sym : bool
        ``True`` for poloidal up/down symmetry, ``False`` otherwise.
        Default is ``False``.
        Whether to truncate the poloidal domain to [0, π] ⊂ [0, 2π)
        to take advantage of poloidal up/down symmetry,
        which is a stronger condition than stellarator symmetry.
        Still, when stellarator symmetry exists, flux surface integrals and
        volume integrals are invariant to this truncation.
    axis : bool
        True to include the magnetic axis, False otherwise (Default = False)
    node_pattern : {``'cheb1'``, ``'cheb2'``, ``'jacobi'``, ``linear``}
        pattern for radial coordinates

            * ``'cheb1'``: Chebyshev-Gauss-Lobatto nodes scaled to r=[0,1]
            * ``'cheb2'``: Chebyshev-Gauss-Lobatto nodes scaled to r=[-1,1]
            * ``'jacobi'``: Radial nodes are roots of Shifted Jacobi polynomial of
              degree M+1 r=(0,1), and angular nodes are equispaced 2(M+1) per surface
            * ``'ocs'``: optimal concentric sampling to minimize the condition number
              of the resulting transform matrix, for doing inverse transform.
            * ``linear`` : linear spacing in r=[0,1]

    """

    _io_attrs_ = AbstractGridFlux._io_attrs_ + ["_node_pattern"]

    _static_attrs = AbstractGridFlux._static_attrs + ["_node_pattern"]

    def __repr__(self):
        """str: String form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + f" (coordinates={self.coordinates}, L={self.L}, M={self.M}, N={self.N}, "
            + f"NFP={self.NFP}, sym={self.sym}, is_meshgrid={self.is_meshgrid}, "
            + f"node_pattern={self.node_pattern})"
        )

    def __init__(self, L, M, N, NFP=1, sym=False, axis=False, node_pattern="jacobi"):
        self._L = check_nonnegint(L, "L", False)
        self._M = check_nonnegint(M, "M", False)
        self._N = check_nonnegint(N, "N", False)
        self._NFP = check_posint(NFP, "NFP", False)
        self._sym = sym
        self._is_meshgrid = False
        self._fft_x1 = False
        self._fft_x2 = True
        self._nodes, self._spacing = self._create_nodes(
            L=L, M=M, N=N, NFP=NFP, axis=axis, node_pattern=node_pattern
        )
        self._enforce_symmetry()
        self._sort_nodes()
        self._axis = self._find_axis()
        (
            self._unique_x0_idx,
            self._inverse_x0_idx,
            self._unique_x1_idx,
            self._inverse_x1_idx,
            self._unique_x2_idx,
            self._inverse_x2_idx,
        ) = self._find_unique_inverse_nodes()
        self._weights = self._scale_weights()

    def _create_nodes(self, L, M, N, NFP=1, axis=False, node_pattern="jacobi"):
        """Create grid nodes and weights.

        Parameters
        ----------
        L : int
            radial grid resolution
        M : int
            poloidal grid resolution
        N : int
            toroidal grid resolution
        NFP : int
            number of field periods (Default = 1)
        axis : bool
            True to include the magnetic axis, False otherwise (Default = False)
        node_pattern : {``'linear'``, ``'cheb1'``, ``'cheb2'``, ``'jacobi'``, ``None``}
            pattern for radial coordinates
                * ``linear`` : linear spacing in r=[0,1]
                * ``'cheb1'``: Chebyshev-Gauss-Lobatto nodes scaled to r=[0,1]
                * ``'cheb2'``: Chebyshev-Gauss-Lobatto nodes scaled to r=[-1,1]
                * ``'jacobi'``: Radial nodes are roots of Shifted Jacobi polynomial of
                  degree M+1 r=(0,1), and angular nodes are equispaced 2(M+1) per
                  surface.
                * ``'ocs'``: optimal concentric sampling to minimize the condition
                  number of the resulting transform matrix, for doing inverse transform.

        Returns
        -------
        nodes : ndarray of float, size(num_nodes, 3)
            node coordinates, in (rho,theta,zeta)
        spacing : ndarray of float, size(num_nodes,3)
            node spacing, based on local volume around the node

        """
        self._L = check_nonnegint(L, "L", False)
        self._M = check_nonnegint(M, "M", False)
        self._N = check_nonnegint(N, "N", False)
        self._NFP = check_posint(NFP, "NFP", False)
        self._node_pattern = node_pattern

        def ocs(L):
            # Ramos-Lopez, et al. “Optimal Sampling Patterns for Zernike Polynomials.”
            # Applied Mathematics and Computation 274 (February 2016): 247–57.
            # https://doi.org/10.1016/j.amc.2015.11.006.
            j = np.arange(1, L // 2 + 2)
            z = np.cos((2 * j - 1) * np.pi / (2 * L + 2))
            rj = 1.1565 * z - 0.76535 * z**2 + 0.60517 * z**3
            return np.sort(rj)

        pattern = {
            "linear": np.linspace(0, 1, num=L // 2 + 1),
            "cheb1": (np.cos(np.arange(L // 2, -1, -1) * np.pi / (L // 2)) + 1) / 2,
            "cheb2": -np.cos(np.arange(L // 2, L + 1, 1) * np.pi / L),
            "jacobi": special.js_roots(L // 2 + 1, 2, 2)[0],
            "ocs": ocs(L),
        }
        rho = pattern.get(node_pattern)
        if rho is None:
            raise ValueError("node_pattern '{}' is not supported".format(node_pattern))
        rho = np.sort(rho, axis=None)
        if axis:
            rho[0] = 0
        elif rho[0] == 0:
            rho[0] = rho[1] / 10

        drho = midpoint_spacing(rho, jnp=np)
        r = []
        t = []
        dr = []
        dt = []

        for iring in range(L // 2 + 1, 0, -1):
            ntheta = 2 * M + np.ceil((M / L) * (5 - 4 * iring)).astype(int)
            if ntheta % 2 == 0:
                # ensure an odd number of nodes on each surface
                ntheta += 1
            if self.sym:
                # for symmetry, we want M+1 nodes on outer surface, so (2M+1+1)
                # for now, cut in half in _enforce_symmetry
                ntheta += 1
            dtheta = 2 * np.pi / ntheta
            theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
            if self.sym:
                theta = (theta + dtheta / 2) % (2 * np.pi)
            for tk in theta:
                r.append(rho[-iring])
                t.append(tk)
                dt.append(dtheta)
                dr.append(drho[-iring])

        r = np.asarray(r)
        t = np.asarray(t)
        dr = np.asarray(dr)
        dt = np.asarray(dt)
        dimzern = r.size

        z = np.linspace(0, 2 * np.pi / NFP, 2 * N + 1, endpoint=False)
        dz = 2 * np.pi / z.size

        r = np.tile(r, 2 * N + 1)
        t = np.tile(t, 2 * N + 1)
        z = np.tile(z[np.newaxis], (dimzern, 1)).flatten(order="F")
        dr = np.tile(dr, 2 * N + 1)
        dt = np.tile(dt, 2 * N + 1)
        dz = np.ones_like(z) * dz
        nodes = np.column_stack([r, t, z])
        spacing = np.column_stack([dr, dt, dz])

        return nodes, spacing

    def change_resolution(self, L, M, N, NFP=None):
        """Change the resolution of the grid.

        Parameters
        ----------
        L : int
            new radial grid resolution
        M : int
            new poloidal grid resolution
        N : int
            new toroidal grid resolution
        NFP : int
            Number of field periods.

        """
        if NFP is None:
            NFP = self.NFP
        if L != self.L or M != self.M or N != self.N or NFP != self.NFP:
            self._nodes, self._spacing = self._create_nodes(
                L=L,
                M=M,
                N=N,
                NFP=NFP,
                axis=self.axis.size > 0,
                node_pattern=self.node_pattern,
            )
            self._enforce_symmetry()
            self._sort_nodes()
            self._axis = self._find_axis()
            (
                self._unique_x0_idx,
                self._inverse_x0_idx,
                self._unique_x1_idx,
                self._inverse_x1_idx,
                self._unique_x2_idx,
                self._inverse_x2_idx,
            ) = self._find_unique_inverse_nodes()
            self._weights = self._scale_weights()

    @property
    def node_pattern(self):
        """str: Pattern for placement of nodes in (rho,theta,zeta)."""
        return self._node_pattern
