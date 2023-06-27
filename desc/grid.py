"""Classes for representing flux coordinates."""

import numpy as np
from scipy import optimize, special

from desc.io import IOAble

__all__ = [
    "Grid",
    "LinearGrid",
    "QuadratureGrid",
    "ConcentricGrid",
    "find_least_rational_surfaces",
    "find_most_rational_surfaces",
]


class Grid(IOAble):
    """Base class for collocation grids.

    Unlike subclasses LinearGrid and ConcentricGrid, the base Grid allows the user
    to pass in a custom set of collocation nodes.

    Parameters
    ----------
    nodes : ndarray of float, size(num_nodes,3)
        node coordinates, in (rho,theta,zeta)
    sort : bool
        whether to sort the nodes for use with FFT method.

    """

    # TODO: calculate weights automatically using voronoi / delaunay triangulation
    _io_attrs_ = [
        "_L",
        "_M",
        "_N",
        "_NFP",
        "_sym",
        "_nodes",
        "_spacing",
        "_weights",
        "_axis",
        "_node_pattern",
        "_unique_rho_idx",
        "_unique_theta_idx",
        "_unique_zeta_idx",
        "_inverse_rho_idx",
        "_inverse_theta_idx",
        "_inverse_zeta_idx",
        "_num_rho",
        "_num_theta",
        "_num_zeta",
    ]

    def __init__(self, nodes, sort=True):
        self._NFP = 1
        self._sym = False
        self._node_pattern = "custom"

        self._nodes, self._spacing = self._create_nodes(nodes)

        self._enforce_symmetry()
        if sort:
            self._sort_nodes()
        self._find_axis()
        self._count_nodes()
        self._scale_weights()

    def _enforce_symmetry(self):
        """Enforce stellarator symmetry.

        1. Remove nodes with theta > pi.
        2. Rescale theta spacing to preserve dtheta weight.
            Need to rescale on each theta coordinate curve by a different factor.
            dtheta should = 2pi / number of nodes remaining on that theta curve.
            Nodes on the symmetry line should not be rescaled.

        """
        if self.sym:
            # indices where theta coordinate is off the symmetry line of theta=0 or pi
            off_sym_line_idx = np.where(self.nodes[:, 1] % np.pi != 0)[0]
            __, inverse, off_sym_line_per_rho_surf_count = np.unique(
                self.nodes[off_sym_line_idx, 0], return_inverse=True, return_counts=True
            )
            # indices of nodes to be deleted
            to_delete_idx = np.where(self.nodes[:, 1] > np.pi)[0]
            __, to_delete_per_rho_surf_count = np.unique(
                self.nodes[to_delete_idx, 0], return_counts=True
            )
            assert (
                2 * np.pi not in self.nodes[:, 1]
                and off_sym_line_per_rho_surf_count.size
                >= to_delete_per_rho_surf_count.size
            )
            if off_sym_line_per_rho_surf_count.size > to_delete_per_rho_surf_count.size:
                # edge case where surfaces closest to axis lack theta > pi nodes
                # The number of nodes to delete on those surfaces is zero.
                pad_count = (
                    off_sym_line_per_rho_surf_count.size
                    - to_delete_per_rho_surf_count.size
                )
                to_delete_per_rho_surf_count = np.pad(
                    to_delete_per_rho_surf_count, (pad_count, 0)
                )
            # The computation of this scale factor assumes
            # 1. number of nodes to delete is constant over zeta
            # 2. number of nodes off symmetry line is constant over zeta
            # 3. uniform theta spacing between nodes
            # The first two assumptions let _per_theta_curve = _per_rho_surf.
            # The third assumption lets the scale factor be constant over a
            # particular theta curve, so that each node in the open interval
            # (0, pi) has its spacing scaled up by the same factor.
            # Nodes at endpoints 0, pi should not be scaled.
            scale = off_sym_line_per_rho_surf_count / (
                off_sym_line_per_rho_surf_count - to_delete_per_rho_surf_count
            )
            # Arrange scale factors to match spacing's arbitrary ordering.
            scale = scale[inverse]

            # Scale up all nodes so that their spacing accounts for the node
            # that is their reflection across the symmetry line.
            self._spacing[off_sym_line_idx, 1] *= scale
            self._nodes = np.delete(self.nodes, to_delete_idx, axis=0)
            self._spacing = np.delete(self.spacing, to_delete_idx, axis=0)

    def _sort_nodes(self):
        """Sort nodes for use with FFT."""
        sort_idx = np.lexsort((self.nodes[:, 1], self.nodes[:, 0], self.nodes[:, 2]))
        self._nodes = self.nodes[sort_idx]
        self._spacing = self.spacing[sort_idx]

    def _find_axis(self):
        """Find indices of axis nodes."""
        self._axis = np.where(self.nodes[:, 0] == 0)[0]

    def _count_nodes(self):
        """Count unique values of coordinates."""
        __, self._unique_rho_idx, self._inverse_rho_idx = np.unique(
            self.nodes[:, 0], return_index=True, return_inverse=True
        )
        __, self._unique_theta_idx, self._inverse_theta_idx = np.unique(
            self.nodes[:, 1], return_index=True, return_inverse=True
        )
        __, self._unique_zeta_idx, self._inverse_zeta_idx = np.unique(
            self.nodes[:, 2], return_index=True, return_inverse=True
        )
        self._num_rho = self._unique_rho_idx.size
        self._num_theta = self._unique_theta_idx.size
        self._num_zeta = self._unique_zeta_idx.size

    def _scale_weights(self):
        """Scale weights sum to full volume and reduce duplicate node weights."""
        nodes = self.nodes.copy().astype(float)
        nodes[:, 1] %= 2 * np.pi
        nodes[:, 2] %= 2 * np.pi / self.NFP
        # reduce weights for duplicated nodes
        _, inverse, counts = np.unique(
            nodes, axis=0, return_inverse=True, return_counts=True
        )
        duplicates = counts[inverse]
        temp_spacing = self.spacing.copy()
        temp_spacing = (temp_spacing.T / duplicates ** (1 / 3)).T
        # scale weights sum to full volume
        temp_spacing *= (4 * np.pi**2 / temp_spacing.prod(axis=1).sum()) ** (1 / 3)
        self._weights = temp_spacing.prod(axis=1)

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

        # scale areas sum to full area
        # The following operation is not a general solution to return the weight
        # removed from the duplicate nodes back to the unique nodes.
        # (For the 3 predefined grid types this line of code has no effect).
        # For this reason, duplicates should typically be deleted rather than rescaled.
        # Note we multiply each column by duplicates^(1/6) to account for the extra
        # division by duplicates^(1/2) in one of the columns above.
        self._spacing *= (
            4 * np.pi**2 / (self.spacing.T * duplicates ** (1 / 6)).prod(axis=0).sum()
        ) ** (1 / 3)

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
        spacing : ndarray of float, size(num_nodes,3)
            Node spacing, in (rho,theta,zeta).

        """
        nodes = np.atleast_2d(nodes).reshape((-1, 3)).astype(float)
        # Do not alter nodes given by the user for custom grids.
        # In particular, do not modulo nodes by 2pi or 2pi/NFP.
        # This may cause the surface_integrals() function to fail recognizing
        # surfaces outside the interval [0, 2pi] as duplicates. However, most
        # surface integral computations are done with LinearGrid anyway.
        spacing = (  # make weights sum to 4pi^2
            np.ones_like(nodes) * np.array([1, 2 * np.pi, 2 * np.pi]) / nodes.shape[0]
        )
        self._L = len(np.unique(nodes[:, 0]))
        self._M = len(np.unique(nodes[:, 1]))
        self._N = len(np.unique(nodes[:, 2]))
        return nodes, spacing

    @property
    def L(self):
        """int: Radial grid resolution."""
        return self.__dict__.setdefault("_L", 0)

    @property
    def M(self):
        """int: Poloidal grid resolution."""
        return self.__dict__.setdefault("_M", 0)

    @property
    def N(self):
        """int: Toroidal grid resolution."""
        return self.__dict__.setdefault("_N", 0)

    @property
    def NFP(self):
        """int: Number of (toroidal) field periods."""
        return self.__dict__.setdefault("_NFP", 1)

    @property
    def sym(self):
        """bool: True for stellarator symmetry, False otherwise."""
        return self.__dict__.setdefault("_sym", False)

    @property
    def nodes(self):
        """ndarray: Node coordinates, in (rho,theta,zeta)."""
        return self.__dict__.setdefault("_nodes", np.array([]).reshape((0, 3)))

    @nodes.setter
    def nodes(self, nodes):
        self._nodes = nodes

    @property
    def spacing(self):
        """ndarray: Node spacing, in (rho,theta,zeta)."""
        return self.__dict__.setdefault("_spacing", np.array([]).reshape((0, 3)))

    @spacing.setter
    def spacing(self, spacing):
        self._spacing = spacing

    @property
    def weights(self):
        """ndarray: Weight for each node, either exact quadrature or volume based."""
        return self.__dict__.setdefault("_weights", np.array([]).reshape((0, 3)))

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def num_nodes(self):
        """int: Total number of nodes."""
        return self.nodes.shape[0]

    @property
    def num_rho(self):
        """int: Number of unique rho coordinates."""
        return self._num_rho

    @property
    def num_theta(self):
        """int: Number of unique theta coordinates."""
        return self._num_theta

    @property
    def num_zeta(self):
        """int: Number of unique zeta coordinates."""
        return self._num_zeta

    @property
    def unique_rho_idx(self):
        """ndarray: Indices of unique rho coordinates."""
        return self._unique_rho_idx

    @property
    def unique_theta_idx(self):
        """ndarray: Indices of unique theta coordinates."""
        return self._unique_theta_idx

    @property
    def unique_zeta_idx(self):
        """ndarray: Indices of unique zeta coordinates."""
        return self._unique_zeta_idx

    @property
    def inverse_rho_idx(self):
        """ndarray: Indices of unique_rho_idx that recover the rho coordinates."""
        return self._inverse_rho_idx

    @property
    def inverse_theta_idx(self):
        """ndarray: Indices of unique_theta_idx that recover the theta coordinates."""
        return self._inverse_theta_idx

    @property
    def inverse_zeta_idx(self):
        """ndarray: Indices of unique_zeta_idx that recover the zeta coordinates."""
        return self._inverse_zeta_idx

    @property
    def axis(self):
        """ndarray: Indices of nodes at magnetic axis."""
        return self.__dict__.setdefault("_axis", np.array([]))

    @property
    def node_pattern(self):
        """str: Pattern for placement of nodes in (rho,theta,zeta)."""
        return self.__dict__.setdefault("_node_pattern", "custom")

    def __repr__(self):
        """str: string form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (L={}, M={}, N={}, NFP={}, sym={}, node_pattern={})".format(
                self.L, self.M, self.N, self.NFP, self.sym, self.node_pattern
            )
        )


class LinearGrid(Grid):
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
    sym : bool
        True for stellarator symmetry, False otherwise (Default = False).
    axis : bool
        True to include a point at rho=0 (default), False for rho[0] = rho[1]/2.
    endpoint : bool
        If True, theta=0 and zeta=0 are duplicated after a full period.
        Should be False for use with FFT. (Default = False).
        This boolean is ignored if an array is given for theta or zeta.
    rho : ndarray of float, optional
        Radial coordinates (Default = 1.0).
    theta : ndarray of float, optional
        Poloidal coordinates (Default = 0.0).
    zeta : ndarray of float, optional
        Toroidal coordinates (Default = 0.0).

    """

    def __init__(
        self,
        L=None,
        M=None,
        N=None,
        NFP=1,
        sym=False,
        axis=True,
        endpoint=False,
        rho=np.array(1.0),
        theta=np.array(0.0),
        zeta=np.array(0.0),
    ):

        self._L = L
        self._M = M
        self._N = N
        self._NFP = NFP
        self._axis = axis
        self._sym = sym
        self._endpoint = bool(endpoint)
        self._node_pattern = "linear"

        self._nodes, self._spacing = self._create_nodes(
            L=self.L,
            M=self.M,
            N=self.N,
            NFP=self.NFP,
            axis=self.axis,
            endpoint=self.endpoint,
            rho=rho,
            theta=theta,
            zeta=zeta,
        )

        # symmetry handled in create_nodes()
        self._sort_nodes()
        self._find_axis()
        self._count_nodes()
        self._scale_weights()

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
        # rho
        if self.L is not None:
            rho = self.L + 1
        else:
            self._L = len(np.atleast_1d(rho))
        if np.isscalar(rho) and (int(rho) == rho) and rho > 0:
            r = np.flipud(np.linspace(1, 0, int(rho), endpoint=axis))
            # choose dr such that each node has the same weight
            dr = np.ones_like(r) / r.size
        else:
            # need to sort to compute correct spacing
            r = np.sort(np.atleast_1d(rho))
            dr = np.zeros_like(r)
            if r.size > 1:
                # choose dr such that cumulative sums of dr[] are node midpoints
                # and the total sum is 1
                dr[0] = (r[0] + r[1]) / 2
                dr[1:-1] = (r[2:] - r[:-2]) / 2
                dr[-1] = 1 - (r[-2] + r[-1]) / 2
            else:
                dr = np.array([1.0])

        # theta
        if self.M is not None:
            if self.sym:
                theta = 2 * (self.M + 1)
            else:
                theta = 2 * self.M + 1
        else:
            self._M = len(np.atleast_1d(theta))
        if np.isscalar(theta) and (int(theta) == theta) and theta > 0:
            theta = int(theta)
            if self.sym and theta > 1:
                # Enforce that no node lies on theta=0 or theta=2pi, so that
                # each node has a symmetric counterpart, and that, for all i,
                # t[i]-t[i-1] = 2 t[0] = 2 (pi - t[last node before pi]).
                # Both conditions necessary to evenly space nodes with constant dt.
                # This can be done by making (theta + endpoint) an even integer.
                if (theta + endpoint) % 2 != 0:
                    theta += 1
                t = np.linspace(0, 2 * np.pi, theta, endpoint=endpoint)
                t += t[1] / 2
                # delete theta > pi nodes
                t = t[: np.searchsorted(t, np.pi, side="right")]
            else:
                t = np.linspace(0, 2 * np.pi, theta, endpoint=endpoint)
            dt = 2 * np.pi / t.size * np.ones_like(t)
            if (endpoint and not self.sym) and t.size > 1:
                # increase node weight to account for duplicate node
                dt *= t.size / (t.size - 1)
                # scale_weights() will reduce endpoint (dt[0] and dt[-1])
                # duplicate node weight
        else:
            t = np.atleast_1d(theta).astype(float)
            SUP = 2 * np.pi  # supremum
            # enforce periodicity
            t[t != SUP] %= SUP
            # need to sort to compute correct spacing
            t = np.sort(t)
            if self.sym:
                # cut domain to relevant subdomain: delete theta > pi nodes
                t = t[: np.searchsorted(t, np.pi, side="right")]
            dt = np.zeros_like(t)
            if t.size > 1:
                # choose dt to be the cyclic distance of the surrounding two nodes
                dt[1:-1] = t[2:] - t[:-2]
                if not self.sym:
                    dt[0] = t[1] + (SUP - t[-1]) % SUP
                    dt[-1] = t[0] + (SUP - t[-2]) % SUP
                    dt /= 2  # choose dt to be half the cyclic distance
                    if t.size == 2:
                        assert dt[0] == np.pi and dt[-1] == 0
                        dt[-1] = dt[0]
                    if t[0] == 0 and t[-1] == SUP:
                        # The cyclic distance algorithm above correctly weights
                        # the duplicate endpoint node spacing at theta = 0 and 2pi
                        # to be half the weight of the other nodes.
                        # However, scale_weights() is not aware of this, so we
                        # counteract the reduction that will be done there.
                        dt[0] += dt[-1]
                        dt[-1] = dt[0]
                else:
                    first_positive_idx = np.searchsorted(t, 0, side="right")
                    if first_positive_idx == 0:
                        # then there are no nodes at theta=0
                        dt[0] = t[0] + t[1]
                    else:
                        # total spacing of nodes at theta=0 should be half the
                        # distance between first positive node and its
                        # reflection across the theta=0 line.
                        dt[0] = t[first_positive_idx]
                        assert (first_positive_idx == 1) or (
                            dt[0] == dt[first_positive_idx - 1]
                        )
                        # If the first condition is false and the latter true,
                        # then both of those dt should be halved.
                        # The scale_weights() function will handle this.
                    first_pi_idx = np.searchsorted(t, np.pi, side="left")
                    if first_pi_idx == t.size:
                        # then there are no nodes at theta=pi
                        dt[-1] = (SUP - t[-1]) - t[-2]
                    else:
                        # total spacing of nodes at theta=pi should be half the
                        # distance between first node < pi and its
                        # reflection across the theta=pi line.
                        dt[-1] = (SUP - t[-1]) - t[first_pi_idx - 1]
                        assert (first_pi_idx == t.size - 1) or (
                            dt[first_pi_idx] == dt[-1]
                        )
                        # If the first condition is false and the latter true,
                        # then both of those dt should be halved.
                        # The scale_weights() function will handle this.
            else:
                dt = np.array([2 * np.pi])

        # zeta
        # note: dz spacing should not depend on NFP
        # spacing corresponds to a node's weight in an integral --
        # such as integral = sum(dt * dz * data["B"]) -- not the node's coordinates
        if self.N is not None:
            zeta = 2 * self.N + 1
        else:
            self._N = len(np.atleast_1d(zeta))
        if np.isscalar(zeta) and (int(zeta) == zeta) and zeta > 0:
            z = np.linspace(0, 2 * np.pi / self.NFP, int(zeta), endpoint=endpoint)
            dz = 2 * np.pi / z.size * np.ones_like(z)
            if endpoint and z.size > 1:
                # increase node weight to account for duplicate node
                dz *= z.size / (z.size - 1)
                # scale_weights() will reduce endpoint (dz[0] and dz[-1])
                # duplicate node weight
        else:
            z = np.atleast_1d(zeta).astype(float)
            # enforce periodicity
            SUP = 2 * np.pi / self.NFP  # supremum
            z[z != SUP] %= SUP
            # need to sort to compute correct spacing
            z = np.sort(z)
            dz = np.zeros_like(z)
            if z.size > 1:
                # choose dz to be half the cyclic distance of the surrounding two nodes
                dz[0] = z[1] + (SUP - z[-1]) % SUP
                dz[1:-1] = z[2:] - z[:-2]
                dz[-1] = z[0] + (SUP - z[-2]) % SUP
                dz /= 2
                dz *= self.NFP
                if z.size == 2:
                    dz[-1] = dz[0]
                if z[0] == 0 and z[-1] == SUP:
                    # The cyclic distance algorithm above correctly weights
                    # the duplicate node spacing at zeta = 0 and 2pi / NFP.
                    # However, scale_weights() is not aware of this, so we
                    # counteract the reduction that will be done there.
                    dz[0] += dz[-1]
                    dz[-1] = dz[0]
            else:
                dz = np.array([2 * np.pi])

        self._endpoint = (
            t.size > 0
            and z.size > 0
            and (t[0] == 0 and t[-1] == 2 * np.pi)
            and (z[0] == 0 and z[-1] == 2 * np.pi / self.NFP)
        )

        r, t, z = np.meshgrid(r, t, z, indexing="ij")
        r = r.flatten()
        t = t.flatten()
        z = z.flatten()

        dr, dt, dz = np.meshgrid(dr, dt, dz, indexing="ij")
        dr = dr.flatten()
        dt = dt.flatten()
        dz = dz.flatten()

        nodes = np.stack([r, t, z]).T
        spacing = np.stack([dr, dt, dz]).T

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
        self._NFP = NFP if NFP is not None else self.NFP
        if L != self.L or M != self.M or N != self.N:
            self._L = L
            self._M = M
            self._N = N
            self._nodes, self._spacing = self._create_nodes(
                L=L,
                M=M,
                N=N,
                NFP=self.NFP,
                axis=len(self.axis) > 0,
                endpoint=self.endpoint,
            )
            # symmetry handled in create_nodes()
            self._sort_nodes()
            self._find_axis()
            self._scale_weights()

    @property
    def endpoint(self):
        """bool: Whether the grid is made of open or closed intervals."""
        return self.__dict__.setdefault("_endpoint", False)


class QuadratureGrid(Grid):
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
        self._L = L
        self._M = M
        self._N = N
        self._NFP = NFP
        self._sym = False
        self._node_pattern = "quad"

        self._nodes, self._spacing = self._create_nodes(
            L=self.L, M=self.M, N=self.N, NFP=self.NFP
        )

        # symmetry is never enforced for Quadrature Grid
        self._sort_nodes()
        self._find_axis()
        self._count_nodes()
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
        L = self.L + 1
        M = 2 * self.M + 1
        N = 2 * self.N + 1

        # rho
        r, dr = special.js_roots(L, 2, 2)
        dr /= r  # remove r weight function associated with the shifted Jacobi weights

        # theta/vartheta
        t = np.linspace(0, 2 * np.pi, M, endpoint=False)
        dt = 2 * np.pi / M * np.ones_like(t)

        # zeta/phi
        z = np.linspace(0, 2 * np.pi / self.NFP, N, endpoint=False)
        dz = 2 * np.pi / N * np.ones_like(z)

        r, t, z = np.meshgrid(r, t, z, indexing="ij")
        r = r.flatten()
        t = t.flatten()
        z = z.flatten()

        dr, dt, dz = np.meshgrid(dr, dt, dz, indexing="ij")
        dr = dr.flatten()
        dt = dt.flatten()
        dz = dz.flatten()

        nodes = np.stack([r, t, z]).T
        spacing = np.stack([dr, dt, dz]).T

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
        self._NFP = NFP if NFP is not None else self.NFP
        if L != self.L or M != self.M or N != self.N:
            self._L = L
            self._M = M
            self._N = N
            self._nodes, self._spacing = self._create_nodes(L=L, M=M, N=N, NFP=self.NFP)
            self._sort_nodes()
            self._find_axis()
            self._weights = self.spacing.prod(axis=1)  # instead of _scale_weights


class ConcentricGrid(Grid):
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

    def __init__(self, L, M, N, NFP=1, sym=False, axis=False, node_pattern="jacobi"):

        self._L = L
        self._M = M
        self._N = N
        self._NFP = NFP
        self._sym = sym
        self._axis = axis
        self._node_pattern = node_pattern

        self._nodes, self._spacing = self._create_nodes(
            L=self.L,
            M=self.M,
            N=self.N,
            NFP=self.NFP,
            axis=self.axis,
            node_pattern=self.node_pattern,
        )

        self._enforce_symmetry()
        self._sort_nodes()
        self._find_axis()
        self._count_nodes()
        self._scale_weights()

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

        drho = np.zeros_like(rho)
        if rho.size > 1:
            drho[0] = (rho[0] + rho[1]) / 2
            drho[1:-1] = (rho[2:] - rho[:-2]) / 2
            drho[-1] = 1 - (rho[-2] + rho[-1]) / 2
        else:
            drho = np.array([1.0])
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
        nodes = np.stack([r, t, z]).T
        spacing = np.stack([dr, dt, dz]).T

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
        self._NFP = NFP if NFP is not None else self.NFP
        if L != self.L or M != self.M or N != self.N:
            self._L = L
            self._M = M
            self._N = N
            self._nodes, self._spacing = self._create_nodes(
                L=L,
                M=M,
                N=N,
                NFP=self.NFP,
                axis=len(self.axis) > 0,
                node_pattern=self.node_pattern,
            )
            self._enforce_symmetry()
            self._sort_nodes()
            self._find_axis()
            self._scale_weights()


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
    # handle empty range
    if a == b:
        return a
    # ensure a < b
    elif a > b:
        c = a
        a = b
        b = c
    # return 0 if in range
    if np.sign(a * b) <= 0:
        return 0
    # handle negative ranges
    elif np.sign(a) < 0:
        s = -1
        a *= -1
        b *= -1
    else:
        s = 1

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
        amount to dislace points to avoid duplicates
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
        amount to dislace points to avoid duplicates

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
        Stopping tolerance for mimimization
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
        Stopping tolerance for mimimization
    itol : float, optional
        tolerance for rounding float to nearest int
    eps : float, optional
        amount to dislace points to avoid duplicates

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
