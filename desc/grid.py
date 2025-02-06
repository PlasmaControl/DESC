"""Classes for representing flux coordinates."""

from abc import ABC, abstractmethod

import numpy as np
from scipy import optimize, special

from desc.backend import fori_loop, jnp, put, repeat, take
from desc.io import IOAble
from desc.utils import Index, check_nonnegint, check_posint, errorif

__all__ = [
    "Grid",
    "LinearGrid",
    "QuadratureGrid",
    "ConcentricGrid",
    "find_least_rational_surfaces",
    "find_most_rational_surfaces",
]


class _Grid(IOAble, ABC):
    """Base class for collocation grids."""

    _io_attrs_ = [
        "_L",
        "_M",
        "_N",
        "_NFP",
        "_NFP_umbilic_factor",
        "_sym",
        "_nodes",
        "_spacing",
        "_weights",
        "_axis",
        "_node_pattern",
        "_coordinates",
        "_period",
        "_source_grid",
        "_unique_rho_idx",
        "_unique_poloidal_idx",
        "_unique_zeta_idx",
        "_inverse_rho_idx",
        "_inverse_poloidal_idx",
        "_inverse_zeta_idx",
        "_is_meshgrid",
        "_can_fft2",
    ]

    @abstractmethod
    def _create_nodes(self, *args, **kwargs):
        """Allow for custom node creation."""
        pass

    def _set_up(self):
        """Do things after loading."""
        # ensure things that should be ints are ints
        self._L = int(self._L)
        self._M = int(self._M)
        self._N = int(self._N)
        self._NFP = int(self._NFP)
        if hasattr(self, "_inverse_theta_idx"):
            self._inverse_poloidal_idx = self._inverse_theta_idx
            del self._inverse_theta_idx
        if hasattr(self, "_unique_theta_idx"):
            self._unique_poloidal_idx = self._unique_theta_idx
            del self._unique_theta_idx

    def _enforce_symmetry(self):
        """Enforce stellarator symmetry.

        1. Remove nodes with theta > pi.
        2. Rescale theta spacing to preserve dtheta weight.
            Need to rescale on each theta coordinate curve by a different factor.
            dtheta should = 2π / number of nodes remaining on that theta curve.
            Nodes on the symmetry line should not be rescaled.

        """
        if not self.sym:
            return
        # indices where poloidal coordinate is off the symmetry line of
        # poloidal coord=0 or pi
        off_sym_line_idx = self.nodes[:, 1] % np.pi != 0
        __, inverse, off_sym_line_per_rho_surf_count = np.unique(
            self.nodes[off_sym_line_idx, 0], return_inverse=True, return_counts=True
        )
        # indices of nodes to be deleted
        to_delete_idx = self.nodes[:, 1] > np.pi
        __, to_delete_per_rho_surf_count = np.unique(
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
        # (0, pi) has its spacing scaled up by the same factor.
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

    def _sort_nodes(self):
        """Sort nodes for use with FFT."""
        sort_idx = np.lexsort((self.nodes[:, 1], self.nodes[:, 0], self.nodes[:, 2]))
        self._nodes = self.nodes[sort_idx]
        self._spacing = self.spacing[sort_idx]

    def _find_axis(self):
        """Find indices of axis nodes."""
        return np.nonzero(self.nodes[:, 0] == 0)[0]

    def _find_unique_inverse_nodes(self):
        """Find unique values of coordinates and their indices."""
        __, unique_rho_idx, inverse_rho_idx = np.unique(
            self.nodes[:, 0], return_index=True, return_inverse=True
        )
        __, unique_poloidal_idx, inverse_poloidal_idx = np.unique(
            self.nodes[:, 1], return_index=True, return_inverse=True
        )
        __, unique_zeta_idx, inverse_zeta_idx = np.unique(
            self.nodes[:, 2], return_index=True, return_inverse=True
        )
        return (
            unique_rho_idx,
            inverse_rho_idx,
            unique_poloidal_idx,
            inverse_poloidal_idx,
            unique_zeta_idx,
            inverse_zeta_idx,
        )

    def _scale_weights(self):
        """Scale weights sum to full volume and reduce duplicate node weights."""
        nodes = self.nodes.copy().astype(float)
        nodes = put(nodes, Index[:, 1], nodes[:, 1] % (2 * np.pi))
        nodes = put(
            nodes,
            Index[:, 2],
            nodes[:, 2] % (2 * np.pi / self.NFP * self.NFP_umbilic_factor),
        )
        # reduce weights for duplicated nodes
        _, inverse, counts = np.unique(
            nodes, axis=0, return_inverse=True, return_counts=True
        )
        duplicates = counts[inverse]
        temp_spacing = self.spacing.copy()
        temp_spacing = (temp_spacing.T / duplicates ** (1 / 3)).T
        # scale weights sum to full volume
        if temp_spacing.prod(axis=1).sum():
            temp_spacing *= (4 * np.pi**2 / temp_spacing.prod(axis=1).sum()) ** (1 / 3)
        weights = temp_spacing.prod(axis=1)

        # Spacing is the differential element used for integration over surfaces.
        # For this, 2 columns of the matrix are used.
        # Spacing is rescaled below to get the correct double product for each pair
        # of columns in grid.spacing.
        # The reduction of weight on duplicate nodes should be accounted for
        # by the 2 columns of spacing which span the surface.
        self._spacing = (self.spacing.T / duplicates ** (1 / 2)).T
        # Note we rescale 3 columns by the factor that 'should' rescale 2 columns,
        # so grid.spacing is valid for integrals over all surface labels.
        # Because a surface integral always ignores 1 column, with this approach,
        # duplicates nodes are scaled down properly regardless of which two columns
        # span the surface.
        return weights

    @property
    def L(self):
        """int: Radial grid resolution."""
        if getattr(self, "_L", None) is None:
            # Setting default values for LinearGrid.
            # This code will never run for Quadrature and Concentric grid.
            self._L = self.num_rho - 1
        return self._L

    @property
    def M(self):
        """int: Poloidal grid resolution."""
        if getattr(self, "_M", None) is None:
            # Setting default values for LinearGrid.
            # This code will never run for Quadrature and Concentric grid.
            self._M = self.num_poloidal - 1 if self.sym else self.num_poloidal // 2
        return self._M

    @property
    def N(self):
        """int: Toroidal grid resolution."""
        if getattr(self, "_N", None) is None:
            # Setting default values for LinearGrid.
            # This code will never run for Quadrature and Concentric grid.
            self._N = self.num_zeta // 2
        return self._N

    @property
    def NFP(self):
        """int: Number of (toroidal) field periods."""
        return self.__dict__.setdefault("_NFP", 1)

    @property
    def NFP_umbilic_factor(self):
        """float: Umbilic factor for (toroidal) field periods."""
        return self.__dict__.setdefault("_NFP_umbilic_factor", 1)

    @property
    def sym(self):
        """bool: True for stellarator symmetry, False otherwise."""
        return self.__dict__.setdefault("_sym", False)

    @property
    def is_meshgrid(self):
        """bool: Whether this grid is a tensor-product grid.

        Let the tuple (r, p, t) ∈ R³ denote a radial, poloidal, and toroidal
        coordinate value. The is_meshgrid flag denotes whether any coordinate
        can be iterated over along the relevant axis of the reshaped grid:
        nodes.reshape((num_poloidal, num_radial, num_toroidal, 3), order="F").
        """
        return self.__dict__.setdefault("_is_meshgrid", False)

    @property
    def can_fft2(self):
        """bool: Whether this grid is compatible with 2D FFT.

        Tensor product grid with uniformly spaced points on
        (θ, ζ) ∈ [0, 2π) × [0, 2π/NFP).
        """
        # TODO: GitHub issue 1243?
        return self.__dict__.setdefault("_can_fft2", self.is_meshgrid and not self.sym)

    @property
    def coordinates(self):
        """Coordinates specified by the nodes.

        Examples
        --------
        raz : rho, alpha, zeta
        rvp : rho, theta_PEST, phi
        rtz : rho, theta, zeta
        """
        return self.__dict__.setdefault("_coordinates", "rtz")

    @property
    def period(self):
        """Periodicity of coordinates."""
        return self.__dict__.setdefault(
            "_period", (np.inf, 2 * np.pi, 2 * np.pi / self.NFP)
        )

    @property
    def num_nodes(self):
        """int: Total number of nodes."""
        return self.nodes.shape[0]

    @property
    def num_rho(self):
        """int: Number of unique rho coordinates."""
        return self.unique_rho_idx.size

    @property
    def num_poloidal(self):
        """int: Number of unique poloidal angle coordinates."""
        return self.unique_poloidal_idx.size

    @property
    def num_alpha(self):
        """ndarray: Number of unique field line poloidal angles."""
        errorif(self.coordinates[1] != "a", AttributeError)
        return self.num_poloidal

    @property
    def num_theta(self):
        """ndarray: Number of unique theta coordinates."""
        errorif(self.coordinates[1] != "t", AttributeError)
        return self.num_poloidal

    @property
    def num_theta_PEST(self):
        """ndarray: Number of unique straight field line poloidal angles."""
        errorif(self.coordinates[1] != "v", AttributeError)
        return self.num_poloidal

    @property
    def num_zeta(self):
        """int: Number of unique zeta coordinates."""
        return self.unique_zeta_idx.size

    @property
    def unique_rho_idx(self):
        """ndarray: Indices of unique rho coordinates."""
        errorif(
            not hasattr(self, "_unique_rho_idx"),
            AttributeError,
            "Grid does not have unique indices assigned. "
            "It is not possible to do this automatically on grids made under JIT.",
        )
        return self._unique_rho_idx

    @property
    def unique_poloidal_idx(self):
        """ndarray: Indices of unique poloidal angle coordinates."""
        errorif(
            not hasattr(self, "_unique_poloidal_idx"),
            AttributeError,
            "Grid does not have unique indices assigned. "
            "It is not possible to do this automatically on grids made under JIT.",
        )
        return self._unique_poloidal_idx

    @property
    def unique_alpha_idx(self):
        """ndarray: Indices of unique field line poloidal angles."""
        errorif(self.coordinates[1] != "a", AttributeError)
        return self.unique_poloidal_idx

    @property
    def unique_theta_idx(self):
        """ndarray: Indices of unique theta coordinates."""
        errorif(self.coordinates[1] != "t", AttributeError)
        return self.unique_poloidal_idx

    @property
    def unique_theta_PEST_idx(self):
        """ndarray: Indices of unique straight field line poloidal angles."""
        errorif(self.coordinates[1] != "v", AttributeError)
        return self.unique_poloidal_idx

    @property
    def unique_zeta_idx(self):
        """ndarray: Indices of unique zeta coordinates."""
        errorif(
            not hasattr(self, "_unique_zeta_idx"),
            AttributeError,
            "Grid does not have unique indices assigned. "
            "It is not possible to do this automatically on grids made under JIT.",
        )
        return self._unique_zeta_idx

    @property
    def inverse_rho_idx(self):
        """ndarray: Indices of unique_rho_idx that recover the rho coordinates."""
        errorif(
            not hasattr(self, "_inverse_rho_idx"),
            AttributeError,
            "Grid does not have inverse indices assigned. "
            "It is not possible to do this automatically on grids made under JIT.",
        )
        return self._inverse_rho_idx

    @property
    def inverse_poloidal_idx(self):
        """ndarray: Indices that recover the unique poloidal coordinates."""
        errorif(
            not hasattr(self, "_inverse_poloidal_idx"),
            AttributeError,
            "Grid does not have inverse indices assigned. "
            "It is not possible to do this automatically on grids made under JIT.",
        )
        return self._inverse_poloidal_idx

    @property
    def inverse_alpha_idx(self):
        """ndarray: Indices that recover field line poloidal angles."""
        errorif(self.coordinates[1] != "a", AttributeError)
        return self.inverse_poloidal_idx

    @property
    def inverse_theta_idx(self):
        """ndarray: Indices that recover unique theta coordinates."""
        errorif(self.coordinates[1] != "t", AttributeError)
        return self.inverse_poloidal_idx

    @property
    def inverse_theta_PEST_idx(self):
        """ndarray: Indices that recover unique straight field line poloidal angles."""
        errorif(self.coordinates[1] != "v", AttributeError)
        return self.inverse_poloidal_idx

    @property
    def inverse_zeta_idx(self):
        """ndarray: Indices of unique_zeta_idx that recover the zeta coordinates."""
        errorif(
            not hasattr(self, "_inverse_zeta_idx"),
            AttributeError,
            "Grid does not have inverse indices assigned. "
            "It is not possible to do this automatically on grids made under JIT.",
        )
        return self._inverse_zeta_idx

    @property
    def axis(self):
        """ndarray: Indices of nodes at magnetic axis."""
        return self.__dict__.setdefault("_axis", np.array([]))

    @property
    def node_pattern(self):
        """str: Pattern for placement of nodes in (rho,theta,zeta)."""
        return self.__dict__.setdefault("_node_pattern", "custom")

    @property
    def nodes(self):
        """ndarray: Node coordinates, in (rho,theta,zeta)."""
        return self.__dict__.setdefault("_nodes", np.array([]).reshape((0, 3)))

    @property
    def spacing(self):
        """Quadrature weights for integration over surfaces.

        This is typically the distance between nodes when ``NFP=1``, as the quadrature
        weight is by default a midpoint rule. The returned matrix has three columns,
        corresponding to the radial, poloidal, and toroidal coordinate, respectively.
        Each element of the matrix specifies the quadrature area associated with a
        particular node for each coordinate. I.e. on a grid with coordinates
        of "rtz", the columns specify dρ, dθ, dζ, respectively. An integration
        over a ρ flux surface will assign quadrature weight dθ*dζ to each node.
        Note that dζ is the distance between toroidal surfaces multiplied by ``NFP``.

        On a LinearGrid with duplicate nodes, the columns of spacing no longer
        specify dρ, dθ, dζ. Rather, the product of each adjacent column specifies
        dρ*dθ, dθ*dζ, dζ*dρ, respectively.

        Returns
        -------
        spacing : ndarray
            Quadrature weights for integration over surface.

        """
        errorif(
            self._spacing is None,
            AttributeError,
            "Custom grids must have spacing specified by user.\n"
            "Recall that the accurate computation of surface integral quantities "
            "requires a specific set of quadrature nodes.\n"
            "In particular, flux surface integrals are best performed on grids with "
            "uniform spacing in (θ,ζ).\n"
            "It is recommended to compute such quantities on the proper grid and use "
            "the ``copy_data_from_other`` method to transfer values to custom grids.",
        )
        return self._spacing

    @property
    def weights(self):
        """ndarray: Weight for each node, either exact quadrature or volume based."""
        errorif(
            self._weights is None,
            AttributeError,
            "Custom grids must have weights specified by user.\n"
            "Recall that the accurate computation of volume integral quantities "
            "requires a specific set of quadrature nodes.\n"
            "It is recommended to compute such quantities on a QuadratureGrid and use "
            "the ``copy_data_from_other`` method to transfer values to custom grids.",
        )
        return self._weights

    def __repr__(self):
        """str: string form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (L={}, M={}, N={}, NFP={}, NFP_umbilic_factor = {},\
               sym={}, node_pattern={}, coordinates={})"
        ).format(
            self.L,
            self.M,
            self.N,
            self.NFP,
            self.NFP_umbilic_factor,
            self.sym,
            self.node_pattern,
            self.coordinates,
        )

    def get_label(self, label):
        """Get general label that specifies direction given label."""
        if label in {"rho", "poloidal", "zeta"}:
            return label
        rad = {"r": "rho"}[self.coordinates[0]]
        pol = {"a": "alpha", "t": "theta", "v": "theta_PEST"}[self.coordinates[1]]
        tor = {"z": "zeta"}[self.coordinates[2]]
        return {rad: "rho", pol: "poloidal", tor: "zeta"}[label]

    def compress(self, x, surface_label="rho"):
        """Return elements of ``x`` at indices of unique surface label values.

        Parameters
        ----------
        x : ndarray
            The array to compress.
            Should usually represent a surface function (constant over a surface)
            in an array that matches the grid's pattern.
        surface_label : {"rho", "poloidal", "zeta"}
            The surface label of rho, poloidal, or zeta.

        Returns
        -------
        compress_x : ndarray
            This array will be sorted such that the first element corresponds to
            the value associated with the smallest surface, and the last element
            corresponds to the value associated with the largest surface.

        """
        surface_label = self.get_label(surface_label)
        errorif(len(x) != self.num_nodes)
        return take(
            x, getattr(self, f"unique_{surface_label}_idx"), axis=0, unique_indices=True
        )

    def expand(self, x, surface_label="rho"):
        """Expand ``x`` by duplicating elements to match the grid's pattern.

        Parameters
        ----------
        x : ndarray
            Stores the values of a surface function (constant over a surface)
            for all unique surfaces of the specified label on the grid.
            The length of ``x`` should match the number of unique surfaces of
            the corresponding label in this grid. ``x`` should be sorted such
            that the first element corresponds to the value associated with the
            smallest surface, and the last element corresponds to the value
            associated with the largest surface.
        surface_label : {"rho", "poloidal", "zeta"}
            The surface label of rho, poloidal, or zeta.

        Returns
        -------
        expand_x : ndarray
            ``x`` expanded to match the grid's pattern.

        """
        surface_label = self.get_label(surface_label)
        errorif(len(x) != getattr(self, f"num_{surface_label}"))
        return x[getattr(self, f"inverse_{surface_label}_idx")]

    def copy_data_from_other(self, x, other_grid, surface_label="rho", tol=1e-14):
        """Copy data x from other_grid to this grid at matching surface label.

        Given data x corresponding to nodes of other_grid, copy data
        to a new array that corresponds to this grid.

        Parameters
        ----------
        x : ndarray, shape(other_grid.num_nodes,...)
            Data to copy. Assumed to be constant over the specified surface.
        other_grid: Grid
            Grid to copy from.
        surface_label : {"rho", "poloidal", "zeta"}
            The surface label of rho, poloidal, or zeta.
        tol : float
            tolerance for considering nodes the same.

        Returns
        -------
        y : ndarray, shape(grid2.num_nodes, ...)
            Data copied to grid2
        """
        sl1 = self.get_label(surface_label)
        sl2 = other_grid.get_label(surface_label)
        axis = {"rho": 0, "poloidal": 1, "zeta": 2}
        errorif(self.coordinates[axis[sl1]] != other_grid.coordinates[axis[sl2]])
        axis = axis[sl1]

        x = jnp.asarray(x)
        try:
            xc = other_grid.compress(x, surface_label)
            y = self.expand(xc, surface_label)
        except AttributeError:
            self_nodes = jnp.asarray(self.nodes[:, axis])
            other_nodes = jnp.asarray(other_grid.nodes[:, axis])
            y = jnp.zeros((self.num_nodes, *x.shape[1:]))

            def body(i, y):
                y = jnp.where(
                    jnp.abs(self_nodes - other_nodes[i]) <= tol,
                    x[i],
                    y,
                )
                return y

            y = fori_loop(0, other_grid.num_nodes, body, y)
        return y

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

    def meshgrid_reshape(self, x, order):
        """Reshape data to match grid coordinates.

        Given flattened data on a tensor product grid, reshape the data such that
        the axes of the array correspond to coordinate values on the grid.

        Parameters
        ----------
        x : ndarray, shape(N,) or shape(N,3)
            Data to reshape.
        order : str
            Desired order of axes for returned data. Should be a permutation of
            ``grid.coordinates``, eg ``order="rtz"`` has the first axis of the returned
            data correspond to different rho coordinates, the second axis to different
            theta, etc.  ``order="trz"`` would have the first axis correspond to theta,
            and so on.

        Returns
        -------
        x : ndarray
            Data reshaped to align with grid nodes.

        """
        errorif(
            not self.is_meshgrid,
            ValueError,
            "grid is not a tensor product grid, so meshgrid_reshape doesn't "
            "make any sense",
        )
        errorif(
            sorted(order) != sorted(self.coordinates),
            ValueError,
            f"order should be a permutation of {self.coordinates}, got {order}",
        )
        shape = (self.num_poloidal, self.num_rho, self.num_zeta)
        vec = False
        if x.ndim > 1:
            vec = True
            shape += (-1,)
        x = x.reshape(shape, order="F")
        # swap to change shape from trz/arz to rtz/raz etc.
        x = jnp.swapaxes(x, 1, 0)
        newax = tuple(self.coordinates.index(c) for c in order)
        if vec:
            newax += (3,)
        x = jnp.transpose(x, newax)
        return x


class Grid(_Grid):
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
    source_grid : Grid
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

    def __init__(
        self,
        nodes,
        spacing=None,
        weights=None,
        coordinates="rtz",
        period=(np.inf, 2 * np.pi, 2 * np.pi),
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
        self._NFP_umbilic_factor = 1
        self._NFP = check_posint(NFP, "NFP", False)
        self._sym = False
        self._node_pattern = "custom"
        self._coordinates = coordinates
        self._period = period
        self._source_grid = source_grid
        self._is_meshgrid = bool(is_meshgrid)
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
            "_unique_rho_idx",
            "_unique_poloidal_idx",
            "_unique_zeta_idx",
            "_inverse_rho_idx",
            "_inverse_poloidal_idx",
            "_inverse_zeta_idx",
        ]
        if jitable:
            # Don't do anything with symmetry since that changes # of nodes
            # avoid point at the axis, for now.
            r, t, z = self._nodes.T
            r = jnp.where(r == 0, 1e-12, r)
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
                self._unique_rho_idx,
                self._inverse_rho_idx,
                self._unique_poloidal_idx,
                self._inverse_poloidal_idx,
                self._unique_zeta_idx,
                self._inverse_zeta_idx,
            ) = self._find_unique_inverse_nodes()
        # Assign with logic in setter method if possible else 0.
        self._L = None if hasattr(self, "num_rho") else 0
        self._M = None if hasattr(self, "num_poloidal") else 0
        self._N = None if hasattr(self, "num_zeta") else 0
        errorif(len(kwargs), ValueError, f"Got unexpected kwargs {kwargs.keys()}")

    @staticmethod
    def create_meshgrid(
        nodes,
        spacing=None,
        coordinates="rtz",
        period=(np.inf, 2 * np.pi, 2 * np.pi),
        NFP=1,
        jitable=True,
        **kwargs,
    ):
        """Create a tensor-product grid from the given coordinates in a jitable manner.

        Parameters
        ----------
        nodes : list of ndarray
            Three arrays, one for each coordinate.
            Sorted unique values of each coordinate.
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
        a, b, c = jnp.atleast_1d(*nodes)
        if spacing is None:
            errorif(coordinates[0] != "r", NotImplementedError)
            da = _midpoint_spacing(a)
            db = _periodic_spacing(b, period[1])[1]
            dc = _periodic_spacing(c, period[2])[1] * NFP
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
            _unique_rho_idx=unique_a_idx,
            _unique_poloidal_idx=unique_b_idx,
            _unique_zeta_idx=unique_c_idx,
            _inverse_rho_idx=inverse_a_idx,
            _inverse_poloidal_idx=inverse_b_idx,
            _inverse_zeta_idx=inverse_c_idx,
            **kwargs,
        )

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
        return nodes

    @property
    def source_grid(self):
        """Coordinates from which this grid was mapped from."""
        errorif(self._source_grid is None, AttributeError)
        return self._source_grid


class LinearGrid(_Grid):
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
    NFP_umbilic_factor : int
        Integer>=1.
        This is needed for the umbilic torus design.
        Change this only if your nodes are placed within one field period
        or should be interpreted as spanning one field period.
    sym : bool
        True for stellarator symmetry, False otherwise (Default = False).
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

    def __init__(
        self,
        L=None,
        M=None,
        N=None,
        NFP=1,
        NFP_umbilic_factor=int(1),
        sym=False,
        axis=True,
        endpoint=False,
        rho=np.array(1.0),
        theta=np.array(0.0),
        zeta=np.array(0.0),
    ):
        self._L = check_nonnegint(L, "L")
        self._M = check_nonnegint(M, "M")
        self._N = check_nonnegint(N, "N")
        self._NFP = check_posint(NFP, "NFP", False)
        self._NFP_umbilic_factor = check_posint(
            NFP_umbilic_factor, "NFP_umbilic_factor", False
        )
        self._sym = sym
        self._endpoint = bool(endpoint)
        self._poloidal_endpoint = False
        self._toroidal_endpoint = False
        self._node_pattern = "linear"
        self._coordinates = "rtz"

        self._is_meshgrid = True
        self._can_fft2 = not sym and not endpoint
        self._period = (
            np.inf,
            2 * np.pi,
            2 * np.pi / self._NFP * self._NFP_umbilic_factor,
        )

        self._nodes, self._spacing = self._create_nodes(
            L=L,
            M=M,
            N=N,
            NFP=NFP,
            NFP_umbilic_factor=NFP_umbilic_factor,
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
            self._unique_rho_idx,
            self._inverse_rho_idx,
            self._unique_poloidal_idx,
            self._inverse_poloidal_idx,
            self._unique_zeta_idx,
            self._inverse_zeta_idx,
        ) = self._find_unique_inverse_nodes()
        self._weights = self._scale_weights()

    def _create_nodes(  # noqa: C901
        self,
        L=None,
        M=None,
        N=None,
        NFP=1,
        NFP_umbilic_factor=int(1),
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
        NFP_umbilic_factor : float
            Rational number of the form 1/integer with integer>=1.
            This is needed for the umbilic torus design.
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
        self._NFP = check_posint(NFP, "NFP", False)
        self._NFP_umbilic_factor = check_posint(
            NFP_umbilic_factor, "NFP_umbilic_factor", False
        )
        self._period = (
            np.inf,
            2 * np.pi,
            2 * np.pi / self._NFP * self._NFP_umbilic_factor,
        )
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
        else:
            r = np.sort(np.atleast_1d(rho))
            dr = _midpoint_spacing(r, jnp=np)

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
        else:
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
                    dt = _periodic_spacing(t, theta_period, jnp=np)[1]
                    if t[0] == 0 and t[-1] == theta_period:
                        # _periodic_spacing above correctly weights
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

        # zeta
        # note: dz spacing should not depend on NFP
        # spacing corresponds to a node's weight in an integral --
        # such as integral = sum(dt * dz * data["B"]) -- not the node's coordinates
        if N is not None:
            self._N = check_nonnegint(N, "N")
            zeta = 2 * N + 1
        if np.isscalar(zeta) and (int(zeta) == zeta) and zeta > 0:
            z = np.linspace(0, zeta_period, int(zeta), endpoint=endpoint)
            dz = 2 * np.pi / z.size * np.ones_like(z)
            if endpoint and z.size > 1:
                # increase node weight to account for duplicate node
                dz *= z.size / (z.size - 1)
                # scale_weights() will reduce endpoint (dz[0] and dz[-1])
                # duplicate node weight
        else:
            z, dz = _periodic_spacing(zeta, zeta_period, sort=True, jnp=np)
            dz = dz * NFP
            if z[0] == 0 and z[-1] == zeta_period:
                # _periodic_spacing above correctly weights
                # the duplicate node spacing at zeta = 0 and 2π/NFP.
                # However, scale_weights() is not aware of this, so we
                # counteract the reduction that will be done there.
                dz[0] += dz[-1]
                dz[-1] = dz[0]

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

    def change_resolution(self, L, M, N, NFP=None, NFP_umbilic_factor=None):
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
        NFP_umbilic_factor : float
            Rational number of the form 1/integer.
            This is needed for the umbilic torus design.

        """
        if NFP is None:
            NFP = self.NFP
        if NFP_umbilic_factor is None:
            NFP_umbilic_factor = self.NFP_umbilic_factor
        if (
            L != self.L
            or M != self.M
            or N != self.N
            or NFP != self.NFP
            or NFP_umbilic_factor != self.NFP_umbilic_factor
        ):
            self._nodes, self._spacing = self._create_nodes(
                L=L,
                M=M,
                N=N,
                NFP=NFP,
                NFP_umbilic_factor=NFP_umbilic_factor,
                axis=self.axis.size > 0,
                endpoint=self.endpoint,
            )
            # symmetry handled in create_nodes()
            self._sort_nodes()
            self._axis = self._find_axis()
            (
                self._unique_rho_idx,
                self._inverse_rho_idx,
                self._unique_poloidal_idx,
                self._inverse_poloidal_idx,
                self._unique_zeta_idx,
                self._inverse_zeta_idx,
            ) = self._find_unique_inverse_nodes()
            self._weights = self._scale_weights()

    @property
    def endpoint(self):
        """bool: Whether the grid is made of open or closed intervals."""
        return self.__dict__.setdefault("_endpoint", False)


class QuadratureGrid(_Grid):
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
        self._node_pattern = "quad"
        self._coordinates = "rtz"
        self._is_meshgrid = True
        self._period = (
            np.inf,
            2 * np.pi,
            2 * np.pi / self._NFP,
        )
        self._nodes, self._spacing = self._create_nodes(L=L, M=M, N=N, NFP=NFP)
        # symmetry is never enforced for Quadrature Grid
        self._sort_nodes()
        self._axis = self._find_axis()
        (
            self._unique_rho_idx,
            self._inverse_rho_idx,
            self._unique_poloidal_idx,
            self._inverse_poloidal_idx,
            self._unique_zeta_idx,
            self._inverse_zeta_idx,
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
        self._period = (np.inf, 2 * np.pi, 2 * np.pi / self._NFP)
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
        z = np.linspace(0, 2 * np.pi / (NFP), N, endpoint=False)
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
                self._unique_rho_idx,
                self._inverse_rho_idx,
                self._unique_poloidal_idx,
                self._inverse_poloidal_idx,
                self._unique_zeta_idx,
                self._inverse_zeta_idx,
            ) = self._find_unique_inverse_nodes()
            self._weights = self.spacing.prod(axis=1)  # instead of _scale_weights


class ConcentricGrid(_Grid):
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
        True for stellarator symmetry, False otherwise (Default = False)
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

    def __init__(
        self,
        L,
        M,
        N,
        NFP=1,
        sym=False,
        axis=False,
        node_pattern="jacobi",
    ):
        self._L = check_nonnegint(L, "L", False)
        self._M = check_nonnegint(M, "M", False)
        self._N = check_nonnegint(N, "N", False)
        self._NFP = check_posint(NFP, "NFP", False)
        self._sym = sym
        self._node_pattern = node_pattern
        self._coordinates = "rtz"
        self._is_meshgrid = False
        self._period = (np.inf, 2 * np.pi, 2 * np.pi / self._NFP)
        self._nodes, self._spacing = self._create_nodes(
            L=L,
            M=M,
            N=N,
            NFP=NFP,
            axis=axis,
            node_pattern=node_pattern,
        )
        self._enforce_symmetry()
        self._sort_nodes()
        self._axis = self._find_axis()
        (
            self._unique_rho_idx,
            self._inverse_rho_idx,
            self._unique_poloidal_idx,
            self._inverse_poloidal_idx,
            self._unique_zeta_idx,
            self._inverse_zeta_idx,
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
        self._period = (
            np.inf,
            2 * np.pi,
            2 * np.pi / self._NFP,
        )

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

        drho = _midpoint_spacing(rho, jnp=np)
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
                self._unique_rho_idx,
                self._inverse_rho_idx,
                self._unique_poloidal_idx,
                self._inverse_poloidal_idx,
                self._unique_zeta_idx,
                self._inverse_zeta_idx,
            ) = self._find_unique_inverse_nodes()
            self._weights = self._scale_weights()


def _round(x, tol):
    # we do this to avoid some floating point issues with things
    # that are basically low order rationals to near machine precision
    if abs(x % 1) < tol or abs((x % 1) - 1) < tol:
        return round(x)
    return x


def dec_to_cf(x, dmax=20, itol=1e-14):
    """Compute continued fraction form of a number.

    Parameters
    ----------
    x : float
        floating point form of number
    dmax : int
        maximum iterations (ie, number of coefficients of continued fraction).
        (Default value = 20)
    itol : float, optional
        tolerance for rounding float to nearest int

    Returns
    -------
    cf : ndarray of int
        coefficients of continued fraction form of x.

    """
    x = float(_round(x, itol))
    cf = []
    q = np.floor(x).astype(int)
    cf.append(q)
    x = _round(x - q, itol)
    i = 0
    while x != 0 and i < dmax:
        q = np.floor(1 / x).astype(int)
        cf.append(q)
        x = _round(1 / x - q, itol)
        i = i + 1
    return np.array(cf).astype(int)


def cf_to_dec(cf):
    """Compute decimal form of a continued fraction.

    Parameters
    ----------
    cf : array-like
        coefficients of continued fraction.

    Returns
    -------
    x : float
        floating point representation of cf

    """
    if len(cf) == 1:
        return cf[0]
    else:
        return cf[0] + 1 / cf_to_dec(cf[1:])


def most_rational(a, b, itol=1e-14):
    """Compute the most rational number in the range [a,b].

    Parameters
    ----------
    a,b : float
        lower and upper bounds
    itol : float, optional
        tolerance for rounding float to nearest int

    Returns
    -------
    x : float
        most rational number between [a,b]

    """
    a = float(_round(a, itol))
    b = float(_round(b, itol))

    # Handle empty range
    if a == b:
        return a

    # Return 0 if in range
    if np.sign(a * b) <= 0:
        return 0

    # Handle negative ranges
    if np.sign(a) < 0:
        s = -1
        a *= -1
        b *= -1
    else:
        s = 1

    # Ensure a < b
    if a > b:
        a, b = b, a

    a_cf = dec_to_cf(a)
    b_cf = dec_to_cf(b)
    idx = 0  # first index of dissimilar digits
    for i in range(min(a_cf.size, b_cf.size)):
        if a_cf[i] != b_cf[i]:
            idx = i
            break
    f = 1
    while True:
        dec = cf_to_dec(np.append(a_cf[0:idx], f))
        if a <= dec <= b:
            return dec * s
        f += 1


def n_most_rational(a, b, n, eps=1e-12, itol=1e-14):
    """Find the n most rational numbers in a given interval.

    Parameters
    ----------
    a, b : float
        start and end points of the interval
    n : integer
        number of rationals to find
    eps : float, optional
        amount to displace points to avoid duplicates
    itol : float, optional
        tolerance for rounding float to nearest int

    Returns
    -------
    c : ndarray
        most rational points in (a,b), in approximate
        order of "rationality"
    """
    assert eps > itol
    a = float(_round(a, itol))
    b = float(_round(b, itol))
    # start with the full interval, find first most rational
    # then subdivide at that point and look for next most
    # rational in the largest sub-interval
    out = []
    intervals = np.array(sorted([a, b]))
    for i in range(n):
        i = np.argmax(np.diff(intervals))
        ai, bi = intervals[i : i + 2]
        if ai in out:
            ai += eps
        if bi in out:
            bi -= eps
        c = most_rational(ai, bi)
        out.append(c)
        j = np.searchsorted(intervals, c)
        intervals = np.insert(intervals, j, c)
    return np.array(out)


def _find_rho(iota, iota_vals, tol=1e-14):
    """Find rho values for iota_vals in iota profile."""
    r = np.linspace(0, 1, 1000)
    io = iota(r)
    rho = []
    for ior in iota_vals:
        f = lambda r: iota(np.atleast_1d(r))[0] - ior
        df = lambda r: iota(np.atleast_1d(r), dr=1)[0]
        # nearest neighbor search for initial guess
        x0 = r[np.argmin(np.abs(io - ior))]
        rho_i = optimize.root_scalar(f, x0=x0, fprime=df, xtol=tol).root
        rho.append(rho_i)
    return np.array(rho)


def find_most_rational_surfaces(iota, n, atol=1e-14, itol=1e-14, eps=1e-12, **kwargs):
    """Find "most rational" surfaces for a give iota profile.

    By most rational, we generally mean lowest order ie smallest continued fraction.

    Note: May not work as expected for non-monotonic profiles with duplicate rational
    surfaces. (generally only 1 of each rational is found)

    Parameters
    ----------
    iota : Profile
        iota profile to search
    n : integer
        number of rational surfaces to find
    atol : float, optional
        stopping tolerance for root finding
    itol : float, optional
        tolerance for rounding float to nearest int
    eps : float, optional
        amount to displace points to avoid duplicates

    Returns
    -------
    rho : ndarray
        sorted radial locations of rational surfaces
    rationals: ndarray
        values of iota at rational surfaces
    """
    # find approx min/max
    r = np.linspace(0, 1, kwargs.get("nsamples", 1000))
    io = iota(r)
    iomin, iomax = np.min(io), np.max(io)
    # find rational values of iota and corresponding rho
    io_rational = n_most_rational(iomin, iomax, n, itol=itol, eps=eps)
    rho = _find_rho(iota, io_rational, tol=atol)
    idx = np.argsort(rho)
    return rho[idx], io_rational[idx]


def find_most_distant(pts, n, a=None, b=None, atol=1e-14, **kwargs):
    """Find n points in interval that are maximally distant from pts and each other.

    Parameters
    ----------
    pts : ndarray
        Points to avoid
    n : int
        Number of points to find.
    a, b : float, optional
        Start and end points for interval. Default is min/max of pts
    atol : float, optional
        Stopping tolerance for minimization
    """

    def foo(x, xs):
        xs = np.atleast_1d(xs)
        d = x - xs[:, None]
        return -np.prod(np.abs(d), axis=0).squeeze()

    if a is None:
        a = np.min(pts)
    if b is None:
        b = np.max(pts)

    pts = list(pts)
    nsamples = kwargs.get("nsamples", 1000)
    x = np.linspace(a, b, nsamples)
    out = []
    for i in range(n):
        y = foo(x, pts)
        x0 = x[np.argmin(y)]
        bracket = (max(a, x0 - 5 / nsamples), min(x0 + 5 / nsamples, b))
        c = optimize.minimize_scalar(
            foo, bracket=bracket, bounds=(a, b), args=(pts,), options={"xatol": atol}
        ).x
        pts.append(c)
        out.append(c)
    return np.array(out)


def find_least_rational_surfaces(
    iota, n, nrational=100, atol=1e-14, itol=1e-14, eps=1e-12, **kwargs
):
    """Find "least rational" surfaces for given iota profile.

    By least rational we mean points farthest in iota from the nrational lowest
    order rational surfaces and each other.

    Note: May not work as expected for non-monotonic profiles with duplicate rational
    surfaces. (generally only 1 of each rational is found)

    Parameters
    ----------
    iota : Profile
        iota profile to search
    n : integer
        number of approximately irrational surfaces to find
    nrational : int, optional
        number of lowest order rational surfaces to avoid.
    atol : float, optional
        Stopping tolerance for minimization
    itol : float, optional
        tolerance for rounding float to nearest int
    eps : float, optional
        amount to displace points to avoid duplicates

    Returns
    -------
    rho : ndarray
        locations of least rational surfaces
    io : ndarray
        values of iota at least rational surfaces
    rho_rat : ndarray
        rho values of lowest order rational surfaces
    io_rat : ndarray
        iota values at lowest order rational surfaces
    """
    rho_rat, io_rat = find_most_rational_surfaces(
        iota, nrational, atol, itol, eps, **kwargs
    )
    a, b = iota([0.0, 1.0])
    io = find_most_distant(io_rat, n, a, b, tol=atol, **kwargs)
    rho = _find_rho(iota, io, tol=atol)
    return rho, io


def _periodic_spacing(x, period=2 * jnp.pi, sort=False, jnp=jnp):
    """Compute dx between points in x assuming periodicity.

    Parameters
    ----------
    x : Array
        Points, assumed sorted in the cyclic domain [0, period], unless
        specified otherwise.
    period : float
        Number such that f(x + period) = f(x) for any function f on this domain.
    sort : bool
        Set to true if x is not sorted in the cyclic domain [0, period].

    Returns
    -------
    x, dx : Array
        Points in [0, period] and assigned spacing.

    """
    x = jnp.atleast_1d(x)
    x = jnp.where(x == period, x, x % period)
    if sort:
        x = jnp.sort(x, axis=0)
    # choose dx to be half the distance between its neighbors
    if x.size > 1:
        if np.isfinite(period):
            dx_0 = x[1] + (period - x[-1]) % period
            dx_1 = x[0] + (period - x[-2]) % period
        else:
            # just set to 0 to stop nan gradient, even though above gives expected value
            dx_0 = 0
            dx_1 = 0
        if x.size == 2:
            # then dx[0] == period and dx[-1] == 0, so fix this
            dx_1 = dx_0
        dx = jnp.hstack([dx_0, x[2:] - x[:-2], dx_1]) / 2
    else:
        dx = jnp.array([period])
    return x, dx


def _midpoint_spacing(x, jnp=jnp):
    """Compute dx between points in x in [0, 1].

    Parameters
    ----------
    x : Array
        Points in [0, 1], assumed sorted.

    Returns
    -------
    dx : Array
        Spacing assigned to points in x.

    """
    x = jnp.atleast_1d(x)
    if x.size > 1:
        # choose dx such that cumulative sums of dx[] are node midpoints
        # and the total sum is 1
        dx_0 = (x[0] + x[1]) / 2
        dx_1 = 1 - (x[-2] + x[-1]) / 2
        dx = jnp.hstack([dx_0, (x[2:] - x[:-2]) / 2, dx_1])
    else:
        dx = jnp.array([1.0])
    return dx
