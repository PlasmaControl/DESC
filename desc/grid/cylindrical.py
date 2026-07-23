"""Classes for representing cylindrical coordinates."""

import warnings

import numpy as np
from scipy import special

from desc.backend import jnp, put, repeat
from desc.utils import check_nonnegint, check_posint, errorif, warnif

from .core import AbstractGrid
from .utils import _create_linear_nodes, midpoint_spacing, periodic_spacing

# CHECK TO MAKE SURE SCALE WEIGHTS ARE CORRECT FOR CYLINDRICAL GRID
# ADD SYMMETRY
# ADD COMMENT ASKING ABOUT rpz or RpZ
# add real space bounds to easily rescale to physical space


class AbstractGridCylindrical(AbstractGrid):
    """Base class for collocation grids in cylindrical coordinates.

    Specifically, these grids represent annular cylinders in real space,
    with the R and Z coordinates linearly rescaled to [0,1] and the
    phi coordinate periodic with period 2*pi/NFP."""

    _io_attrs_ = AbstractGrid._io_attrs_ + ["_NFP"]

    _static_attrs = AbstractGrid._static_attrs + ["_NFP"]

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

    def get_label(self, label):
        """Get general label that specifies the direction of given coordinate label."""
        if label in {"x0", "x1", "x2"}:
            return label
        x0 = {"r": "r"}[self.coordinates[0]]
        x1 = {"phi": "phi"}[self.coordinates[1]]
        x2 = {"z": "z"}[self.coordinates[2]]
        return {x0: "x0", x1: "x1", x2: "x2"}[label]

    @property
    def coordinates(self):
        """Coordinates specified by the nodes.

        Options for x0 coordinate:
        - r = R

        Options for x1 coordinate:
        - p = phi

        Options for x2 coordinate:
        - z = Z
        """
        coordinates = self.__dict__.setdefault("_coordinates", "rpz")
        errorif(coordinates != "rpz", NotImplementedError)
        return coordinates

    @property
    def bounds(self):
        """Bounds of coordinates."""
        return ((0, 1), (0, 2 * np.pi), (0, 1))

    @property
    def period(self):
        """Periodicity of coordinates."""
        return (np.inf, 2 * np.pi / self.NFP, np.inf)

    @property
    def num_r(self):
        """int: Number of unique R coordinates."""
        return self.num_x0

    @property
    def num_phi(self):
        """int: Number of unique phi coordinates."""
        return self.num_x1

    @property
    def num_z(self):
        """int: Number of unique Z coordinates."""
        return self.num_x2

    @property
    def unique_r_idx(self):
        """ndarray: Indices of unique R coordinates."""
        return self.unique_x0_idx

    @property
    def unique_phi_idx(self):
        """ndarray: Indices of unique phi coordinates."""
        return self.unique_x1_idx

    @property
    def unique_z_idx(self):
        """ndarray: Indices of unique Z coordinates."""
        return self.unique_x2_idx

    @property
    def inverse_r_idx(self):
        """ndarray: Indices that recover the R coordinates."""
        return self.inverse_x0_idx

    @property
    def inverse_phi_idx(self):
        """ndarray: Indices that recover the cylindrical angle coordinates."""
        return self.inverse_x1_idx

    @property
    def inverse_z_idx(self):
        """ndarray: Indices that recover the Z coordinates."""
        return self.inverse_x2_idx

    @property
    def NFP(self):
        """int: Number of (toroidal) field periods."""
        return self.__dict__.setdefault("_NFP", 1)


class CustomGridCylindrical(AbstractGridCylindrical):
    """Collocation grid with custom node placement.

    Unlike subclass QuadratureGridCylindrical, the base
    CustomGridCylindrical allows the user to pass in a
    custom set of collocation nodes.

    Parameters
    ----------
    nodes : ndarray of float, size(num_nodes,3)
        Node coordinates, in (R,phi,Z), with R and Z linearly rescaled to
        [0,1] and phi with period 2*pi/NFP.
    spacing : ndarray of float, size(num_nodes, 3)
        Spacing between nodes in each direction.
    weights : ndarray of float, size(num_nodes, )
        Quadrature weights for each node.
    NFP : int
        Number of field periods (Default = 1).
        Change this only if your nodes are placed within one field period.
    source_grid : AbstractGridCylindrical
        Grid from which coordinates were mapped from.
    sort : bool
        Whether to sort the nodes for use with FFT method.
    is_meshgrid : bool
        Whether this grid is a tensor-product grid.
        Let the tuple (r, p, z) ∈ R³ denote a radial, angular, and vertical
        coordinate value. The is_meshgrid flag denotes whether any coordinate
        can be iterated over along the relevant axis of the reshaped grid:
        nodes.reshape((num_phi, num_r, num_z, 3), order="F").
    jitable : bool
        Whether to skip certain checks and conditionals that don't work under jit.
        Allows grid to be created on the fly with custom nodes, but weights,
        symmetry etc. may be wrong if grid contains duplicate nodes.
    """

    _io_attrs_ = AbstractGridCylindrical._io_attrs_ + ["_source_grid"]

    def __init__(
        self,
        nodes,
        spacing=None,
        weights=None,
        NFP=1,
        source_grid=None,
        sort=False,
        is_meshgrid=False,
        jitable=False,
        **kwargs,
    ):
        warnif(
            kwargs.pop("period", False),
            FutureWarning,
            msg="Argument `period` is deprecated and is now set by `coordinates`.",
        )
        nodes = jnp.atleast_2d(jnp.asarray(nodes))
        assert len(nodes.shape) == 2
        assert nodes.shape[1] == 3
        self._nodes = self._create_nodes(nodes)

        if spacing is not None:
            spacing = jnp.atleast_2d(jnp.asarray(spacing))
            assert len(spacing.shape) == 2
            assert spacing.shape[1] == 3
            self._spacing = spacing.reshape(self.nodes.shape).astype(float)
        else:
            self._spacing = None

        self._weights = (
            jnp.atleast_1d(jnp.asarray(weights))
            .reshape(self.nodes.shape[0])
            .astype(float)
            if weights is not None
            else None
        )

        self._NFP = check_posint(NFP, "NFP", False)
        self._source_grid = source_grid
        self._is_meshgrid = bool(is_meshgrid)
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
            # allow for user supplied indices/inverse indices for special cases
            for attr in setable_attr:
                if attr in kwargs:
                    setattr(self, attr, jnp.asarray(kwargs.pop(attr)))
        else:
            for attr in setable_attr:
                kwargs.pop(attr, None)
            (
                self._unique_x0_idx,
                self._inverse_x0_idx,
                self._unique_x1_idx,
                self._inverse_x1_idx,
                self._unique_x2_idx,
                self._inverse_x2_idx,
            ) = self._find_unique_inverse_nodes()

        # assign with logic in setter method if possible else 0
        self._L = self.num_x0 - 1 if hasattr(self, "num_x0") else 0
        self._M = self.num_x1 // 2 if hasattr(self, "num_x1") else 0
        self._N = self.num_x2 - 1 if hasattr(self, "num_x2") else 0
        errorif(len(kwargs), ValueError, f"Got unexpected kwargs {kwargs.keys()}.")

    def _create_nodes(self, nodes):
        """Allow for custom node creation.

        Parameters
        ----------
        nodes : ndarray of float, size(num_nodes,3)
            Node coordinates, in (R,phi,Z).

        Returns
        -------
        nodes : ndarray of float, size(num_nodes,3)
            Node coordinates, in (R,phi,Z).

        """
        # do not alter nodes given by the user for custom grids
        return nodes.reshape((-1, 3)).astype(float)

    @staticmethod
    def create_meshgrid(
        nodes,
        spacing=None,
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
        grid : CustomGridFlux
            Meshgrid.

        """
        NFP = check_posint(NFP, "NFP", False)

        period = (np.inf, 2 * np.pi / NFP, np.inf)

        x0, x1, x2 = jnp.atleast_1d(*nodes)
        if spacing is None:
            dx0 = midpoint_spacing(x0)
            dx1 = periodic_spacing(x1, period[1])[1] * NFP
            dx2 = midpoint_spacing(x2)
        else:
            dx0, dx1, dx2 = spacing

        xx1, xx0, xx2 = jnp.meshgrid(x1, x0, x2, indexing="ij")

        nodes = jnp.column_stack(
            [xx0.flatten(order="F"), xx1.flatten(order="F"), xx2.flatten(order="F")]
        )
        xx1, xx0, xx2 = jnp.meshgrid(dx1, dx0, dx2, indexing="ij")

        spacing = jnp.column_stack(
            [xx0.flatten(order="F"), xx1.flatten(order="F"), xx2.flatten(order="F")]
        )
        weights = spacing.prod(axis=1)

        unique_x0_idx = jnp.arange(x0.size) * x1.size
        unique_x1_idx = jnp.arange(x1.size)
        unique_x2_idx = jnp.arange(x2.size) * x0.size * x1.size
        inverse_x0_idx = jnp.tile(
            repeat(
                unique_x0_idx // x1.size, x1.size, total_repeat_length=x0.size * x1.size
            ),
            x2.size,
        )
        inverse_x1_idx = jnp.tile(unique_x1_idx, x0.size * x2.size)
        inverse_x2_idx = repeat(
            unique_x2_idx // (x0.size * x1.size), (x0.size * x1.size)
        )
        return CustomGridCylindrical(
            nodes=nodes,
            spacing=spacing,
            weights=weights,
            NFP=NFP,
            sort=False,
            is_meshgrid=True,
            jitable=jitable,
            _unique_x0_idx=unique_x0_idx,
            _unique_x1_idx=unique_x1_idx,
            _unique_x2_idx=unique_x2_idx,
            _inverse_x0_idx=inverse_x0_idx,
            _inverse_x1_idx=inverse_x1_idx,
            _inverse_x2_idx=inverse_x2_idx,
            **kwargs,
        )

    @property
    def source_grid(self):
        """Coordinates from which this grid was mapped from."""
        errorif(self._source_grid is None, AttributeError)
        return self._source_grid


class QuadratureGridCylindrical(AbstractGridCylindrical):
    """Exactly integrates a DoubleChebyshevFourierBasis of resolution (L,M,N).

    Nodes are arranged linearly in the second (assumed phi) coordinate, and
    at the Chebyshev-Gauss-Lobatto nodes in the first and third (respectively,
    normalized R and Z) coordinates.

    For now, this grid is assumed to never be symmetric.

    Parameters
    ----------
    L : int
        radial grid resolution
    M : int
        toroidal grid resolution
    N : int
        vertical grid resolution
    NFP : int
        number of field periods (Default = 1)
    R : np.ndarray
        radial coordinates (Default None, in which case L
        must be specified).
    phi : np.ndarray
        toroidal coordinates (Default None, in which case
        M must be specified)
    Z: np.ndarray
        vertical coordinates (Default None, in which case
        N must be specified)
    """

    def __init__(
        self,
        L,
        M,
        N,
        NFP=1,
    ):
        self._L = check_nonnegint(L, "L", False)
        self._M = check_nonnegint(M, "M", False)
        self._N = check_nonnegint(N, "N", False)
        self._NFP = check_posint(NFP, "NFP", False)
        self._is_meshgrid = True
        self._fft_x1 = True
        self._fft_x2 = False
        self._nodes, self._spacing = self._create_nodes(L=L, M=M, N=N, NFP=NFP)
        self._sort_nodes()
        (
            self._unique_x0_idx,
            self._inverse_x0_idx,
            self._unique_x1_idx,
            self._inverse_x1_idx,
            self._unique_x2_idx,
            self._inverse_x2_idx,
        ) = self._find_unique_inverse_nodes()
        self._weights = self._scale_weights()

    def _create_nodes(self, L=1, M=1, N=1, NFP=1):
        """Create grid nodes and weights.

        Parameters
        ----------
        L : int
            radial grid resolution
        M : int
            toroidal grid resolution
        N : int
            vertical grid resolution
        NFP : int
            number of field periods (Default = 1)

        Returns
        -------
        nodes : ndarray of float, size(num_nodes, 3)
            node coordinates, in (R,phi,Z) with
            R and Z normalized to [0,1] and phi with period 2*pi/NFP.
        spacing : ndarray of float, size(num_nodes,3)
            node spacing, based on local volume around the node

        """
        self._L = check_nonnegint(L, "L", False)
        self._M = check_nonnegint(M, "M", False)
        self._N = check_nonnegint(N, "N", False)
        self._NFP = check_posint(NFP, "NFP", False)
        self._period = (np.inf, 2 * np.pi / self._NFP, np.inf)

        R = lobatto(L)
        dR = midpoint_spacing(R, jnp=np)

        phi = np.linspace(0, 2 * np.pi / NFP, 2 * M + 1, endpoint=False)
        dphi = 2 * np.pi / (2 * M + 1) * np.ones_like(phi)

        Z = lobatto(N)
        dZ = midpoint_spacing(Z, jnp=np)

        R, phi, Z = map(np.ravel, np.meshgrid(R, phi, Z, indexing="ij"))
        dR, dphi, dZ = map(np.ravel, np.meshgrid(dR, dphi, dZ, indexing="ij"))

        nodes = np.column_stack([R, phi, Z])
        spacing = np.column_stack([dR, dphi, dZ])

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
            (
                self._unique_x0_idx,
                self._inverse_x0_idx,
                self._unique_x1_idx,
                self._inverse_x1_idx,
                self._unique_x2_idx,
                self._inverse_x2_idx,
            ) = self._find_unique_inverse_nodes()
            self._weights = self._scale_weights()


def lobatto(res):
    if res == 0:
        return np.array([1])
    x = (np.cos(np.arange(res, -1, -1) * np.pi / res) + 1) / 2
    x = np.sort(x, axis=None)
    return x
