"""Classes for representing 1D curve coordinates."""

import numpy as np

from desc.utils import check_nonnegint, check_posint, errorif

from .core import AbstractGrid
from .utils import periodic_spacing


class AbstractGridCurve(AbstractGrid):
    """Base class for collocation grids along 1D curves."""

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
        # ensure things that should be ints are ints
        self._NFP = int(self._NFP)

    def _sort_nodes(self):
        """Sort nodes for use with FFT."""
        sort_idx = np.lexsort((self.nodes[:, 1], self.nodes[:, 0], self.nodes[:, 2]))
        self._nodes = self.nodes[sort_idx]
        self._spacing = self.spacing[sort_idx]

    def get_label(self, label):
        """Get general label that specifies the direction of given coordinate label."""
        assert label == "s"
        return label

    def get_label_axis(self, label):
        """Get node axis index associated with given coordinate label."""
        label = self.get_label(label)
        return {"s": 2}[label]

    @property
    def coordinates(self):
        """Coordinates specified by the nodes."""
        return self.__dict__.setdefault("_coordinates", "__s")

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
    """Grid in which the nodes are linearly spaced in the x2 coordinate.

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
        Coordinates values (Default = 0.0).
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
        # these are just default values that may get overwritten in _create_nodes
        self._poloidal_endpoint = False
        self._toroidal_endpoint = False

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
            Coordinates values (Default = 0.0).
            Alternatively, the number of coordinates (if an integer).
            Note that if supplied the values may be reordered in the resulting grid.

        Returns
        -------
        nodes : ndarray of float, size(num_nodes,3)
            node coordinates, in (rho,theta,s)
        spacing : ndarray of float, size(num_nodes,3)
            node spacing, based on local volume around the node

        """
        self._NFP = check_posint(NFP, "NFP", False)
        endpoint = bool(endpoint)
        s_period = self.period[2]

        # s
        # note: dz spacing should not depend on NFP
        # spacing corresponds to a node's weight in an integral --
        # such as integral = sum(dt * dz * data["B"]) -- not the node's coordinates
        if N is not None:
            self._N = check_nonnegint(N, "N")
            s = 2 * N + 1
        if np.isscalar(s) and (int(s) == s) and s > 0:
            s = int(s)
            z = np.linspace(0, s_period, s, endpoint=endpoint)
            dz = 2 * np.pi / z.size * np.ones_like(z)
            if endpoint and z.size > 1:
                # increase node weight to account for duplicate node
                dz *= z.size / (z.size - 1)
                # scale_weights() will reduce endpoint (dz[0] and dz[-1])
                # duplicate node weight
            # if custom s used usually safe to assume its non-uniform so no fft
            self._fft_x2 = not endpoint
        elif s is not None:
            errorif(
                np.any(np.asarray(s) > s_period),
                msg="LinearGrid should be defined on 1 field period.",
            )
            z, dz = periodic_spacing(s, s_period, sort=True, jnp=np)
            dz = dz * NFP
            if z[0] == 0 and z[-1] == s_period:
                # periodic_spacing above correctly weights
                # the duplicate node spacing at s = 0 and 2π/NFP.
                # However, scale_weights() is not aware of this, so we
                # counteract the reduction that will be done there.
                dz[0] += dz[-1]
                dz[-1] = dz[0]
        else:
            z = np.array(0.0, ndmin=1)
            dz = s_period * np.ones_like(z) * NFP
            self._fft_x2 = True  # trivially true

        r = np.zeros_like(z)
        t = np.zeros_like(z)
        dr = np.zeros_like(dz)
        dt = np.zeros_like(dz)

        self._toroidal_endpoint = (
            z.size > 0
            and np.isclose(z[0], 0, atol=1e-12)
            and np.isclose(z[-1], s_period, atol=1e-12)
        )
        # if only one theta or one s point, can have endpoint=True
        # if the other one is a full array
        self._endpoint = (t.size == 1 and z.size > 1) and (
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

    def change_resolution(self, N, NFP=None):
        """Change the resolution of the grid.

        Parameters
        ----------
        N : int
            New toroidal grid resolution.
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
