"""Classes for representing 1D curve coordinates."""

import numpy as np

from desc.backend import jnp
from desc.utils import check_nonnegint, check_posint, errorif

from .core import AbstractGrid
from .utils import periodic_spacing


class AbstractGridCurve(AbstractGrid):
    """Base class for collocation grids along 1D filamentary curves."""

    _io_attrs_ = AbstractGrid._io_attrs_ + ["_NFP"]

    _static_attrs = AbstractGrid._static_attrs + ["_NFP"]

    def __repr__(self):
        """str: String form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + f" (coordinates={self.coordinates}, N={self.N}, NFP={self.NFP}, "
            + f"is_meshgrid={self.is_meshgrid}"
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
        x2 = {"s": "s"}[self.coordinates[2]]
        return {x2: "x2"}[label]

    @property
    def coordinates(self):
        """Coordinates specified by the nodes.

        Options for x0 coordinate: None

        Options for x1 coordinate: None

        Options for x2 coordinate:
        - s = s
        """
        coordinates = self.__dict__.setdefault("_coordinates", "__s")
        errorif(coordinates != "__s", NotImplementedError)
        return coordinates

    @property
    def bounds(self):
        """Bounds of coordinates."""
        return ((0, 0), (0, 0), (0, 2 * np.pi))

    @property
    def period(self):
        """Periodicity of coordinates."""
        return (np.inf, np.inf, 2 * np.pi / self.NFP)

    @property
    def num_s(self):
        """int: Number of unique s coordinates."""
        return self.num_x2

    @property
    def unique_s_idx(self):
        """ndarray: Indices of unique s coordinates."""
        return self.unique_x2_idx

    @property
    def inverse_s_idx(self):
        """ndarray: Indices that recover the s coordinates."""
        return self.inverse_x2_idx

    @property
    def NFP(self):
        """int: Number of (toroidal) field periods."""
        return self.__dict__.setdefault("_NFP", 1)


class LinearGridCurve(AbstractGridCurve):
    """Grid in which the nodes are linearly spaced in the curve coordinate.

    Parameters
    ----------
    N : int, optional
        Grid resolution.
    NFP : int
        Number of field periods (Default = 1).
    endpoint : bool
        If True, s=0 is duplicated after a full period.
        Should be False for use with FFT. (Default = False).
        This boolean is ignored if an array is given for s.
    s : int or ndarray of float, optional
        Curve coordinates (Default = 0.0).
        Alternatively, the number of coordinates (if an integer).
        Note that if supplied the values may be reordered in the resulting grid.
    """

    _io_attrs_ = AbstractGridCurve._io_attrs_ + ["_endpoint"]

    _static_attrs = AbstractGridCurve._static_attrs + ["_endpoint"]

    def __init__(
        self,
        N=None,
        NFP=1,
        endpoint=False,
        s=None,
    ):
        assert (N is None) or (s is None), "cannot specify both N and s"
        self._N = check_nonnegint(N, "N")
        self._NFP = check_posint(NFP, "NFP", False)
        self._endpoint = bool(endpoint)
        self._is_meshgrid = True
        self._fft_x1 = False
        self._fft_x2 = False
        self._can_fft2 = not endpoint

        self._nodes, self._spacing = self._create_nodes(
            N=N, NFP=NFP, endpoint=endpoint, s=s
        )
        # symmetry handled in create_nodes()
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

    def _create_nodes(self, N=None, NFP=1, endpoint=False, s=0.0):  # noqa: C901
        """Create grid nodes and weights.

        Parameters
        ----------
        N : int, optional
            Grid resolution.
        NFP : int
            Number of field periods (Default = 1).
        endpoint : bool
            If True, s=0 is duplicated after a full period.
            Should be False for use with FFT. (Default = False).
            This boolean is ignored if an array is given for s.
        s : int or ndarray of float, optional
            Curve coordinates (Default = 0.0).
            Alternatively, the number of coordinates (if an integer).
            Note that if supplied the values may be reordered in the resulting grid.

        Returns
        -------
        nodes : ndarray of float, size(num_nodes,3)
            node coordinates, in (_,_,s)
        spacing : ndarray of float, size(num_nodes,3)
            node spacing, based on local volume around the node

        """
        self._NFP = check_posint(NFP, "NFP", False)
        endpoint = bool(endpoint)
        s_period = self.period[2]

        # curve coordinate s
        # note: ds spacing should not depend on NFP
        # spacing corresponds to a node's weight in an integral, not the coordinates
        if N is not None:
            self._N = check_nonnegint(N, "N")
            s = 2 * N + 1
        if np.isscalar(s) and (int(s) == s) and s > 0:
            s = int(s)
            ss = np.linspace(0, s_period, s, endpoint=endpoint)
            ds = 2 * np.pi / ss.size * np.ones_like(ss)
            if endpoint and ss.size > 1:
                # increase node weight to account for duplicate node
                ds *= ss.size / (ss.size - 1)
                # scale_weights() will reduce endpoint (ds[0] and ds[-1])
                # duplicate node weight
            # if custom s used usually safe to assume its non-uniform so no fft
            self._fft_x2 = not endpoint
        elif s is not None:
            errorif(
                np.any(np.asarray(s) > s_period),
                msg="LinearGridCurve should be defined on 1 field period.",
            )
            ss, ds = periodic_spacing(s, s_period, sort=True, jnp=np)
            ds = ds * NFP
            if ss[0] == 0 and ss[-1] == s_period:
                # periodic_spacing above correctly weights
                # the duplicate node spacing at s = 0 and 2π/NFP.
                # However, scale_weights() is not aware of this, so we
                # counteract the reduction that will be done there.
                ds[0] += ds[-1]
                ds[-1] = ds[0]
        else:
            ss = np.array(0.0, ndmin=1)
            ds = s_period * np.ones_like(ss) * NFP
            self._fft_x2 = True  # trivially true

        _ = np.zeros(1)
        d_ = np.zeros_like(_)

        self._endpoint = (
            (ss.size > 1)
            and np.isclose(ss[0], 0, atol=1e-12)
            and np.isclose(ss[-1], s_period, atol=1e-12)
        )
        self._can_fft2 = not self._endpoint

        _, _, ss = map(np.ravel, np.meshgrid(_, _, ss, indexing="ij"))
        d_, d_, ds = map(np.ravel, np.meshgrid(d_, d_, ds, indexing="ij"))
        nodes = np.column_stack([_, _, ss])
        spacing = np.column_stack([d_, d_, ds])

        return nodes, spacing

    def change_resolution(self, N, NFP=None):
        """Change the resolution of the grid.

        Parameters
        ----------
        N : int
            New grid resolution.
        NFP : int
            Number of field periods.

        """
        if NFP is None:
            NFP = self.NFP
        if N != self.N or NFP != self.NFP:
            self._nodes, self._spacing = self._create_nodes(
                N=N, NFP=NFP, endpoint=self.endpoint
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
    def N(self):
        """int: Coordinate resolution."""
        if self._N is None:
            self._N = self.num_s // 2
        return self._N

    @property
    def endpoint(self):
        """bool: Whether the grid is made of open or closed intervals."""
        return self.__dict__.setdefault("_endpoint", False)


class CustomGridCurve(AbstractGridCurve):
    """Collocation grid with custom node placement.

    Parameters
    ----------
    nodes : ndarray of float, size(num_nodes,3)
        Node coordinates, in (_,_,s).
    spacing : ndarray of float, size(num_nodes,)
        Spacing between each node.
    weights : ndarray of float, size(num_nodes,)
        Quadrature weights for each node.
    NFP : int
        Number of field periods (Default = 1).
    sort : bool
        Whether to sort the nodes for use with FFT method.
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
        NFP=1,
        sort=False,
        jitable=False,
        **kwargs,
    ):
        self._NFP = check_posint(NFP, "NFP", False)
        # If you are using a custom grid it almost always is not uniform, or is under
        # jit where we cannot properly check this anyways, so just set to False.
        self._fft_x1 = False
        self._fft_x2 = False
        self._can_fft2 = False
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
        self._N = self.num_x2 // 2 if hasattr(self, "num_x2") else 0
        errorif(len(kwargs), ValueError, f"Got unexpected kwargs {kwargs.keys()}.")

    def _create_nodes(self, nodes):
        """Allow for custom node creation.

        Parameters
        ----------
        nodes : ndarray of float, size(num_nodes,3)
            Node coordinates, in (_,_,s).

        Returns
        -------
        nodes : ndarray of float, size(num_nodes,3)
            Node coordinates, in (_,_,s).

        """
        # do not alter nodes given by the user for custom grids
        nodes = jnp.atleast_2d(jnp.asarray(nodes)).reshape((-1, 3)).astype(float)
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
