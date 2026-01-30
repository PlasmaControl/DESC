"""Classes for representing 2D surface coordinates."""

import numpy as np

from desc.utils import check_nonnegint, check_posint, errorif

from .core import AbstractGrid
from .utils import periodic_spacing


class AbstractGridSurface(AbstractGrid):
    """Base class for collocation grids along 2D geometric surfaces."""

    _io_attrs_ = AbstractGrid._io_attrs_ + ["_NFP", "_sym"]

    _static_attrs = AbstractGrid._static_attrs + ["_NFP", "_sym"]

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
        # indices where θ is off the symmetry line of θ=0 or π
        off_sym_line_idx = self.nodes[:, 1] % np.pi != 0
        off_sym_line_count = np.sum(off_sym_line_idx)
        # indices of nodes to be deleted
        to_delete_idx = self.nodes[:, 1] > np.pi
        to_delete_count = np.sum(to_delete_idx)
        assert 2 * np.pi not in self.nodes[:, 1]
        # The computation of this scale factor assumes
        # 1. number of nodes to delete is constant over zeta
        # 2. number of nodes off symmetry line is constant over zeta
        # 3. uniform poloidal spacing between nodes
        # The first two assumptions let _per_poloidal_curve = _per_rho_surf.
        # The third assumption lets the scale factor be constant over a
        # particular theta curve, so that each node in the open interval
        # (0, π) has its spacing scaled up by the same factor.
        # Nodes at endpoints 0, π should not be scaled.
        scale = off_sym_line_count / (off_sym_line_count - to_delete_count)

        # Scale up all nodes so that their spacing accounts for the node
        # that is their reflection across the symmetry line.
        self._spacing[off_sym_line_idx, 1] *= scale
        self._nodes = self.nodes[~to_delete_idx]
        self._spacing = self.spacing[~to_delete_idx]

    def get_label(self, label):
        """Get general label that specifies the direction of given coordinate label."""
        if label in {"x0", "x1", "x2"}:
            return label
        x1 = {"t": "theta"}[self.coordinates[1]]
        x2 = {"z": "zeta"}[self.coordinates[2]]
        return {x1: "x1", x2: "x2"}[label]

    @property
    def coordinates(self):
        """Coordinates specified by the nodes.

        Options for x0 coordinate: None

        Options for x1 coordinate:
        - t = theta

        Options for x2 coordinate:
        - z = zeta
        """
        coordinates = self.__dict__.setdefault("_coordinates", "_tz")
        errorif(coordinates != "_tz", NotImplementedError)
        return coordinates

    @property
    def bounds(self):
        """Bounds of coordinates."""
        return ((0, 0), (0, 2 * np.pi), (0, 2 * np.pi))

    @property
    def period(self):
        """Periodicity of coordinates."""
        return (np.inf, 2 * np.pi, 2 * np.pi / self.NFP)

    @property
    def num_theta(self):
        """ndarray: Number of unique theta coordinates."""
        return self.num_x1

    @property
    def num_zeta(self):
        """int: Number of unique zeta coordinates."""
        return self.num_x2

    @property
    def unique_theta_idx(self):
        """ndarray: Indices of unique theta coordinates."""
        return self.unique_x1_idx

    @property
    def unique_zeta_idx(self):
        """ndarray: Indices of unique zeta coordinates."""
        return self.unique_x2_idx

    @property
    def inverse_theta_idx(self):
        """ndarray: Indices that recover the theta coordinates."""
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


class LinearGridSurface(AbstractGridSurface):
    """Grid in which the nodes are linearly spaced in the surface coordinate.

    Parameters
    ----------
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
    endpoint : bool
        If True, theta=0 and zeta=0 are duplicated after a full period.
        Should be False for use with FFT. (Default = False).
        This boolean is ignored if an array is given for theta or zeta.
    theta : int or ndarray of float, optional
        Poloidal coordinates (Default = 0.0).
        Alternatively, the number of poloidal coordinates (if an integer).
        Note that if supplied the values may be reordered in the resulting grid.
    zeta : int or ndarray of float, optional
        Toroidal coordinates (Default = 0.0).
        Alternatively, the number of toroidal coordinates (if an integer).
        Note that if supplied the values may be reordered in the resulting grid.
    """

    _io_attrs_ = AbstractGridSurface._io_attrs_ + [
        "_poloidal_endpoint",
        "_toroidal_endpoint",
    ]

    _static_attrs = AbstractGridSurface._static_attrs + ["_endpoint"]

    def __init__(
        self,
        M=None,
        N=None,
        NFP=1,
        sym=False,
        endpoint=False,
        theta=None,
        zeta=None,
    ):
        assert (M is None) or (theta is None), "cannot specify both M and theta"
        assert (N is None) or (zeta is None), "cannot specify both N and zeta"
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
            M=M,
            N=N,
            NFP=NFP,
            endpoint=endpoint,
            theta=theta,
            zeta=zeta,
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
        self._NFP = check_posint(NFP, "NFP", False)
        endpoint = bool(endpoint)
        theta_period = self.period[1]
        zeta_period = self.period[2]

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
                msg="LinearGridFlux should be defined on 1 field period.",
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

        _ = np.zeros(1)
        d_ = np.zeros_like(_)

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

        _, t, z = map(np.ravel, np.meshgrid(_, t, z, indexing="ij"))
        d_, dt, dz = map(np.ravel, np.meshgrid(d_, dt, dz, indexing="ij"))
        nodes = np.column_stack([_, t, z])
        spacing = np.column_stack([d_, dt, dz])

        return nodes, spacing

    def change_resolution(self, M, N, NFP=None):
        """Change the resolution of the grid.

        Parameters
        ----------
        M : int
            new poloidal grid resolution (M poloidal nodes)
        N : int
            new toroidal grid resolution (N toroidal nodes)
        NFP : int
            Number of field periods.

        """
        if NFP is None:
            NFP = self.NFP
        if M != self.M or N != self.N or NFP != self.NFP:
            self._nodes, self._spacing = self._create_nodes(
                M=M, N=N, NFP=NFP, endpoint=self.endpoint
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

    @property
    def M(self):
        """int: Poloidal coordinate resolution."""
        if self._M is None:
            self._M = self.num_theta - 1 if self.sym else self.num_theta // 2
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
