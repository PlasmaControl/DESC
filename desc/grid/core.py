"""Base classes for representing collocation grids of coordinates."""

from abc import ABC, abstractmethod

import numpy as np

from desc.backend import fori_loop, jnp, put, take
from desc.io import IOAble
from desc.utils import Index, errorif


class AbstractGrid(IOAble, ABC):
    """Base class for collocation grids."""

    _io_attrs_ = [
        "_coordinates",
        "_bounds",
        "_period",
        "_L",
        "_M",
        "_N",
        "_nodes",
        "_spacing",
        "_weights",
        "_unique_x0_idx",
        "_unique_x1_idx",
        "_unique_x2_idx",
        "_inverse_x0_idx",
        "_inverse_x1_idx",
        "_inverse_x2_idx",
        "_is_meshgrid",
        "_fft_x1",
        "_fft_x2",
        "_can_fft2",
    ]

    _static_attrs = [
        "_coordinates",
        "_bounds",
        "_L",
        "_M",
        "_N",
        "_is_meshgrid",
        "_fft_x1",
        "_fft_x2",
        "_can_fft2",
    ]

    def __repr__(self):
        """str: String form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + (" (coordinates={}, L={}, M={}, N={}, is_meshgrid={})").format(
                self.coordinates, self.L, self.M, self.N, self.is_meshgrid
            )
        )

    def _set_up(self):
        """Do things after loading."""
        # ensure things that should be ints are ints
        self._L = int(self._L)
        self._M = int(self._M)
        self._N = int(self._N)

    def _find_unique_inverse_nodes(self):
        """Find unique values of coordinates and their indices."""
        __, unique_x0_idx, inverse_x0_idx = np.unique(
            self.nodes[:, 0], return_index=True, return_inverse=True
        )
        __, unique_x1_idx, inverse_x1_idx = np.unique(
            self.nodes[:, 1], return_index=True, return_inverse=True
        )
        __, unique_x2_idx, inverse_x2_idx = np.unique(
            self.nodes[:, 2], return_index=True, return_inverse=True
        )
        return (
            unique_x0_idx,
            inverse_x0_idx,
            unique_x1_idx,
            inverse_x1_idx,
            unique_x2_idx,
            inverse_x2_idx,
        )

    def _scale_weights(self):
        """Scale weights sum to full volume and reduce duplicate node weights."""
        nodes = self.nodes.copy().astype(float)
        # mod nodes by their periodicity
        for k, period in enumerate(self.period):
            if period < np.inf:
                nodes = put(nodes, Index[:, k], nodes[:, k] % period)
        # reduce weights for duplicated nodes
        _, inverse, counts = np.unique(
            nodes, axis=0, return_inverse=True, return_counts=True
        )
        duplicates = counts[inverse]
        temp_spacing = self.spacing.copy()
        temp_spacing = (temp_spacing.T / duplicates ** (1 / 3)).T
        # scale weights sum to full volume
        dx0 = self.bounds[0][1] - self.bounds[0][0]
        dx1 = self.bounds[1][1] - self.bounds[1][0]
        dx2 = self.bounds[2][1] - self.bounds[2][0]
        volume = dx0 * dx1 * dx2
        if temp_spacing.prod(axis=1).sum() and np.abs(volume) < np.inf:
            temp_spacing *= (volume / temp_spacing.prod(axis=1).sum()) ** (1 / 3)
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

    def compress(self, x, surface_label):
        """Return elements of ``x`` at indices of unique surface label values.

        Parameters
        ----------
        x : ndarray
            The array to compress. Should usually represent a surface function (constant
            over a surface) in an array that matches the grid's pattern.
        surface_label : str
            The surface label. Must be one of the elements in self.coordinates.

        Returns
        -------
        compress_x : ndarray
            This array will be sorted such that the first element corresponds to
            the value associated with the smallest surface, and the last element
            corresponds to the value associated with the largest surface.

        """
        surface_label_axis = self.get_label_axis(surface_label)
        errorif(len(x) != self.num_nodes)
        return take(
            x,
            getattr(self, f"unique_x{surface_label_axis}_idx"),
            axis=0,
            unique_indices=True,
        )

    def expand(self, x, surface_label):
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
        surface_label : str
            The surface label. Must be one of the elements in self.coordinates.

        Returns
        -------
        expand_x : ndarray
            ``x`` expanded to match the grid's pattern.

        """
        surface_label_axis = self.get_label_axis(surface_label)
        errorif(len(x) != getattr(self, f"num_x{surface_label_axis}"))
        return x[getattr(self, f"inverse_x{surface_label_axis}_idx")]

    def meshgrid_reshape(self, x, order):
        """Reshape data to match grid coordinates. Inverse of grid.meshgrid_flatten.

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
            theta, etc. ``order="trz"`` would have the first axis correspond to theta,
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
        shape = (self.num_x1, self.num_x0, self.num_x2)
        vec = False
        if x.ndim > 1:
            vec = True
            shape += (-1,)
        x = x.reshape(shape, order="F")
        # swap to change shape from trz/arz to rtz/raz etc.
        x = x.swapaxes(1, 0)
        newax = tuple(self.coordinates.index(c) for c in order)
        if vec:
            newax += (3,)
        x = x.transpose(newax)
        return x

    def meshgrid_flatten(self, x, order):
        """Flatten data to match standard ordering. Inverse of grid.meshgrid_reshape.

        Given data on a tensor product grid, flatten the data in the standard DESC
        ordering.

        Parameters
        ----------
        x : ndarray, shape(n1, n2, n3,...)
            Data to reshape.
        order : str
            Order of axes for input data. Should be a permutation of
            ``grid.coordinates``, eg ``order="rtz"`` has the first axis of the data
            correspond to different rho coordinates, the second axis to different
            theta, etc. ``order="trz"`` would have the first axis correspond to theta,
            and so on.

        Returns
        -------
        x : ndarray, shape(n1*n2*n3,...)
            Data flattened in standard DESC ordering.

        """
        errorif(
            not self.is_meshgrid,
            ValueError,
            "grid is not a tensor product grid, so meshgrid_flatten doesn't "
            "make any sense",
        )
        errorif(
            sorted(order) != sorted(self.coordinates),
            ValueError,
            f"order should be a permutation of {self.coordinates}, got {order}",
        )
        errorif(
            not ((x.ndim == 3) or (x.ndim == 4)),
            ValueError,
            f"x should be 3D or 4D, got shape {x.shape}",
        )
        vec = x.ndim == 4
        # reshape to radial/poloidal/toroidal
        newax = tuple(order.index(c) for c in self.coordinates)
        if vec:
            newax += (3,)
        x = x.transpose(newax)
        # swap to change shape from rtz/raz to trz/arz etc.
        x = x.swapaxes(1, 0)

        shape = (self.num_x1 * self.num_x0 * self.num_x2,)
        if vec:
            shape += (-1,)
        x = x.reshape(shape, order="F")
        return x

    def copy_data_from_other(self, x, other_grid, surface_label, tol=1e-14):
        """Copy data x from other_grid to this grid at matching surface label.

        Given data x corresponding to nodes of other_grid, copy data to a new array that
        corresponds to this grid.

        Parameters
        ----------
        x : ndarray, shape(other_grid.num_nodes,...)
            Data to copy. Assumed to be constant over the specified surface.
        other_grid: AbstractGrid
            Grid to copy from.
        surface_label : str
            The surface label. Must be one of the elements in self.coordinates.
        tol : float
            Tolerance for considering nodes the same.

        Returns
        -------
        y : ndarray, shape(grid2.num_nodes, ...)
            Data copied to grid2

        """
        axis = self.get_label_axis(surface_label)
        other_axis = other_grid.get_label_axis(surface_label)
        errorif(self.coordinates[axis] != other_grid.coordinates[other_axis])

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

    @abstractmethod
    def _create_nodes(self, *args, **kwargs):
        """Create grid nodes."""
        pass

    @abstractmethod
    def get_label(self, label) -> str:
        """Get general label that specifies the direction of given coordinate label."""
        pass

    @abstractmethod
    def get_label_axis(self, label) -> int:
        """Get node axis index associated with given coordinate label."""
        pass

    @property
    @abstractmethod
    def coordinates(self):
        """Coordinates specified by the nodes."""
        pass

    @property
    def bounds(self):
        """Bounds of coordinates."""
        return self.__dict__.setdefault(
            "_bounds", ((-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf))
        )

    @property
    def period(self):
        """Periodicity of coordinates."""
        return self.__dict__.setdefault("_period", (np.inf, np.inf, np.inf))

    @property
    def L(self):
        """int: x0 coordinate resolution."""
        return self._L

    @property
    def M(self):
        """int: x1 coordinate resolution."""
        return self._M

    @property
    def N(self):
        """int: x2 coordinate resolution."""
        return self._N

    @property
    def nodes(self):
        """ndarray: Node coordinates, in (x0,x1,x2)."""
        return self.__dict__.setdefault("_nodes", np.array([]).reshape((0, 3)))

    @property
    def spacing(self):
        """Quadrature weights for integration over surfaces.

        This is typically the distance between nodes, as the quadrature weight is by
        default a midpoint rule. The returned matrix has three columns, corresponding to
        the x0, x1, x2 coordinates, respectively. Each element of the matrix specifies
        the quadrature area associated with a particular node for each coordinate.

        Returns
        -------
        spacing : ndarray
            Quadrature weights for integration over a surface.

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
            "It is recommended to compute such quantities on a QuadratureGrid.",
        )
        return self._weights

    @property
    def num_nodes(self):
        """int: Total number of nodes."""
        return self.nodes.shape[0]

    @property
    def num_x0(self):
        """int: Number of unique x0 coordinates."""
        return self.unique_x0_idx.size

    @property
    def num_x1(self):
        """int: Number of unique x1 coordinates."""
        return self.unique_x1_idx.size

    @property
    def num_x2(self):
        """int: Number of unique x2 coordinates."""
        return self.unique_x2_idx.size

    @property
    def unique_x0_idx(self):
        """ndarray: Indices of unique x0 coordinates."""
        errorif(
            not hasattr(self, "_unique_x0_idx"),
            AttributeError,
            "Grid does not have unique indices assigned. "
            "It is not possible to do this automatically on grids made under JIT.",
        )
        return self._unique_x0_idx

    @property
    def unique_x1_idx(self):
        """ndarray: Indices of unique x1 coordinates."""
        errorif(
            not hasattr(self, "_unique_x1_idx"),
            AttributeError,
            "Grid does not have unique indices assigned. "
            "It is not possible to do this automatically on grids made under JIT.",
        )
        return self._unique_x1_idx

    @property
    def unique_x2_idx(self):
        """ndarray: Indices of unique x2 coordinates."""
        errorif(
            not hasattr(self, "_unique_x2_idx"),
            AttributeError,
            "Grid does not have unique indices assigned. "
            "It is not possible to do this automatically on grids made under JIT.",
        )
        return self._unique_x2_idx

    @property
    def inverse_x0_idx(self):
        """ndarray: Indices of unique_x0_idx that recover the x0 coordinates."""
        errorif(
            not hasattr(self, "_inverse_x0_idx"),
            AttributeError,
            "Grid does not have inverse indices assigned. "
            "It is not possible to do this automatically on grids made under JIT.",
        )
        return self._inverse_x0_idx

    @property
    def inverse_x1_idx(self):
        """ndarray: Indices that recover the unique x1 coordinates."""
        errorif(
            not hasattr(self, "_inverse_x1_idx"),
            AttributeError,
            "Grid does not have inverse indices assigned. "
            "It is not possible to do this automatically on grids made under JIT.",
        )
        return self._inverse_x1_idx

    @property
    def inverse_x2_idx(self):
        """ndarray: Indices of unique_x2_idx that recover the x2 coordinates."""
        errorif(
            not hasattr(self, "_inverse_x2_idx"),
            AttributeError,
            "Grid does not have inverse indices assigned. "
            "It is not possible to do this automatically on grids made under JIT.",
        )
        return self._inverse_x2_idx

    @property
    def is_meshgrid(self):
        """bool: Whether this grid is a tensor-product grid.

        Let the tuple (x0, x1, x2) ∈ R³ denote a coordinate value. The is_meshgrid flag
        denotes whether any coordinate can be iterated over along the relevant axis of
        the reshaped grid:
        nodes.reshape((num_x1, num_x0, num_x2, 3), order="F").
        """
        return self.__dict__.setdefault("_is_meshgrid", False)

    @property
    def fft_x1(self):
        """bool: whether this grid is compatible with FFT in the x1 direction."""
        if not hasattr(self, "_fft_x1"):
            self._fft_x1 = False
        return self._fft_x1

    @property
    def fft_x2(self):
        """bool: whether this grid is compatible with FFT in the x2 direction."""
        if not hasattr(self, "_fft_x2"):
            self._fft_x2 = False
        return self._fft_x2

    @property
    def can_fft2(self):
        """bool: Whether this grid is compatible with 2D FFT.

        Tensor product grid with uniformly spaced points in the x1 and x2 coordinates.
        """
        # TODO: GitHub issue 1243?
        return self.__dict__.setdefault(
            "_can_fft2", self.is_meshgrid and self.fft_x1 and self.fft_x2
        )
