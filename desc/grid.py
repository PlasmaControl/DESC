import numpy as np

from desc.io import IOAble
from scipy import special

__all__ = ["Grid", "LinearGrid", "QuadratureGrid", "ConcentricGrid"]


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
        "_num_rho",
        "_num_theta",
        "_num_zeta",
    ]

    def __init__(self, nodes, sort=True):

        nodes = np.atleast_2d(nodes)
        self._L = np.unique(nodes[:, 0]).size
        self._M = np.unique(nodes[:, 1]).size
        self._N = np.unique(nodes[:, 2]).size
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
        """Enforce stellarator symmetry."""
        if self.sym:  # remove nodes with theta > pi
            non_sym_idx = np.where(self.nodes[:, 1] > np.pi)
            self._nodes = np.delete(self.nodes, non_sym_idx, axis=0)
            self._spacing = np.delete(self.spacing, non_sym_idx, axis=0)

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
        self._num_rho = np.unique(self.nodes[:, 0]).size
        self._num_theta = np.unique(self.nodes[:, 1]).size
        self._num_zeta = np.unique(self.nodes[:, 2]).size

    def _scale_weights(self):
        """Scale weights sum to full volume and reduce weights for duplicated nodes."""
        nodes = self.nodes.copy().astype(float)
        nodes[:, 1] %= 2 * np.pi
        nodes[:, 2] %= 2 * np.pi / self.NFP
        # reduce weights for duplicated nodes
        _, inverse, counts = np.unique(
            nodes, axis=0, return_inverse=True, return_counts=True
        )
        self._spacing /= np.tile(np.atleast_2d(counts[inverse]).T, 3) ** (1 / 3)
        self._spacing *= (4 * np.pi ** 2 / self.spacing.prod(axis=1).sum()) ** (1 / 3)
        self._weights = self.spacing.prod(axis=1)

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
        nodes = np.atleast_2d(nodes).reshape((-1, 3))
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
        """ndarray: node spacing, in (rho,theta,zeta)"""
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
        """int: total number of nodes."""
        return self.nodes.shape[0]

    @property
    def num_rho(self):
        """int: number of unique rho coordinates"""
        return self._num_rho

    @property
    def num_theta(self):
        """int: number of unique theta coordinates"""
        return self._num_theta

    @property
    def num_zeta(self):
        """int: number of unique zeta coordinates"""
        return self._num_zeta

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
    L : int
        radial grid resolution (L radial nodes, Defualt = 1)
    M : int
        poloidal grid resolution (M poloidal nodes, Default = 1)
    N : int
        toroidal grid resolution (N toroidal nodes, Default = 1)
    NFP : int
        number of field periods (Default = 1)
    sym : bool
        True for stellarator symmetry, False otherwise (Default = False)
    axis : bool
        True to include a point at rh0==0, False for rho[0] = rho[1]/4. (Default = True)
    endpoint : bool
        if True, theta=0 and zeta=0 are duplicated after a full period.
        Should be False for use with FFT (Default = False)
    rho : ndarray of float, optional
        radial coordinates
    theta : ndarray of float, optional
        poloidal coordinates
    zeta : ndarray of float, optional
        toroidal coordinates

    """

    def __init__(
        self,
        L=1,
        M=1,
        N=1,
        NFP=1,
        sym=False,
        axis=True,
        endpoint=False,
        rho=None,
        theta=None,
        zeta=None,
    ):

        self._L = L
        self._M = M
        self._N = N
        self._NFP = NFP
        self._axis = axis
        self._sym = sym
        self._endpoint = endpoint
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

        self._enforce_symmetry()
        self._sort_nodes()
        self._find_axis()
        self._count_nodes()
        self._scale_weights()

    def _create_nodes(
        self,
        L=1,
        M=1,
        N=1,
        NFP=1,
        axis=True,
        endpoint=False,
        rho=None,
        theta=None,
        zeta=None,
    ):
        """Create grid nodes and weights.

        Parameters
        ----------
        L : int
            radial grid resolution (L radial nodes, Defualt = 1)
        M : int
            poloidal grid resolution (M poloidal nodes, Default = 1)
        N : int
            toroidal grid resolution (N toroidal nodes, Default = 1)
        NFP : int
            number of field periods (Default = 1)
        axis : bool
            True to include a point at rh0==0, False to include points at rho==1e-4.
        endpoint : bool
            if True, theta=0 and zeta=0 are duplicated after a full period.
            Should be False for use with FFT (Default = False)
        rho : ndarray of float, optional
            radial coordinates
        theta : ndarray of float, optional
            poloidal coordinates
        zeta : ndarray of float, optional
            toroidal coordinates

        Returns
        -------
        nodes : ndarray of float, size(num_nodes,3)
            node coordinates, in (rho,theta,zeta)
        spacing : ndarray of float, size(num_nodes,3)
            node spacing, based on local volume around the node

        """
        self._L = L
        self._M = M
        self._N = N
        self._NFP = NFP

        # rho
        if rho is not None:
            r = np.atleast_1d(rho)
            self._L = r.size
        elif self.L == 0:
            r = np.array([], dtype=float)
        elif self.L == 1:
            r = np.array([1.0])
        else:
            if axis:
                r0 = 0
            else:
                r0 = 1.0 / self.L
            r = np.linspace(r0, 1, self.L)
        if self.L > 0:
            dr = (r[-1] - r[0]) / self.L
            if dr == 0:
                dr = 1
        else:
            dr = 1

        # theta/vartheta
        if theta is not None:
            t = np.asarray(theta)
            self._M = t.size
        else:
            t = np.linspace(0, 2 * np.pi, self.M, endpoint=endpoint)
        if self.M > 0:
            dt = 2 * np.pi / self.M
        else:
            dt = 2 * np.pi

        # zeta/phi
        if zeta is not None:
            z = np.asarray(zeta)
            self._N = z.size
        else:
            z = np.linspace(0, 2 * np.pi / self.NFP, self.N, endpoint=endpoint)
        if self.N > 0:
            dz = 2 * np.pi / self.NFP / self.N
        else:
            dz = 2 * np.pi / self.NFP

        r, t, z = np.meshgrid(r, t, z, indexing="ij")
        r = r.flatten()
        t = t.flatten()
        z = z.flatten()

        dr = dr * np.ones_like(r)
        dt = dt * np.ones_like(t)
        dz = dz * np.ones_like(z)

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
            self._enforce_symmetry()
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

        self._enforce_symmetry()  # symmetry is never enforced for Quadrature Grid
        self._sort_nodes()
        self._find_axis()
        self._count_nodes()
        self._weights = self.spacing.prod(axis=1)  # Quad weights don't need scaling

    def _create_nodes(self, L=1, M=1, N=1, NFP=1):
        """Create grid nodes and weights.

        Parameters
        ----------
        L : int
            radial grid resolution (L radial nodes, Defualt = 1)
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
        self._L = L
        self._M = M
        self._N = N
        self._NFP = NFP

        L = self.L + 1
        M = 2 * self.M + 1
        N = 2 * self.N + 1

        # rho
        r, wr = special.js_roots(L, 2, 2)

        # theta/vartheta
        t = np.linspace(0, 2 * np.pi, M, endpoint=False)
        wt = 2 * np.pi / M * np.ones_like(t)

        # zeta/phi
        z = np.linspace(0, 2 * np.pi / self.NFP, N, endpoint=False)
        wz = 2 * np.pi / N * np.ones_like(z)

        r, t, z = np.meshgrid(r, t, z, indexing="ij")
        r = r.flatten()
        t = t.flatten()
        z = z.flatten()

        wr, wt, wz = np.meshgrid(wr, wt, wz, indexing="ij")
        wr = wr.flatten()
        wt = wt.flatten()
        wz = wz.flatten()
        wr /= r  # remove r weight function associated with the shifted Jacobi weights

        nodes = np.stack([r, t, z]).T
        spacing = np.stack([wr, wt, wz]).T

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
            self._enforce_symmetry()
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
    rotation : {``'cos'``, ``'sin'``, False}
        * ``'cos'`` for cos(m*t-n*z) symmetry, gives nodes at theta=0
        * ``'sin'`` for sin(m*t-n*z) symmetry, gives nodes at theta=pi/2
        * ``None`` for no symmetry (Default), rotates halfway between other options
    node_pattern : {``'cheb1'``, ``'cheb2'``, ``'jacobi'``, ``None``}
        pattern for radial coordinates

            * ``'cheb1'``: Chebyshev-Gauss-Lobatto nodes scaled to r=[0,1]
            * ``'cheb2'``: Chebyshev-Gauss-Lobatto nodes scaled to r=[-1,1]
            * ``'jacobi'``: Radial nodes are roots of Shifted Jacobi polynomial of
              degree M+1 r=(0,1), and angular nodes are equispaced 2(M+1) per surface
            * ``'ocs'``: optimal concentric sampling to minimize the condition number
              of the resulting transform matrix, for doing inverse transform.
            * ``None`` : linear spacing in r=[0,1]

    """

    def __init__(
        self,
        L,
        M,
        N,
        NFP=1,
        sym=False,
        axis=False,
        rotation=None,
        node_pattern="jacobi",
    ):

        self._L = L
        self._M = M
        self._N = N
        self._NFP = NFP
        self._sym = sym
        self._axis = axis
        self._rotation = rotation
        self._node_pattern = node_pattern

        self._nodes, self._spacing = self._create_nodes(
            L=self.L,
            M=self.M,
            N=self.N,
            NFP=self.NFP,
            axis=self.axis,
            rotation=self._rotation,
            node_pattern=self.node_pattern,
        )

        self._enforce_symmetry()
        self._sort_nodes()
        self._find_axis()
        self._count_nodes()
        self._scale_weights()

    def _create_nodes(
        self, L, M, N, NFP=1, axis=False, rotation=None, node_pattern="jacobi"
    ):
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
        rotation : {``'cos'``, ``'sin'``, False}
            * ``'cos'`` for cos(m*t-n*z) symmetry, gives nodes at theta=0
            * ``'sin'`` for sin(m*t-n*z) symmetry, gives nodes at theta=pi/2
            * ``None`` for no symmetry (Default), rotates halfway between other options
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
            # Ramos-Lopez, et al “Optimal Sampling Patterns for Zernike Polynomials.”
            # Applied Mathematics and Computation 274 (February 2016): 247–57.
            # https://doi.org/10.1016/j.amc.2015.11.006.
            j = np.arange(1, L // 2 + 2)
            z = np.cos((2 * j - 1) * np.pi / (2 * L + 2))
            rj = 1.1565 * z - 0.76535 * z ** 2 + 0.60517 * z ** 3
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
            for i in range(rho.size):
                if i == 0:
                    drho[i] = (rho[0] + rho[1]) / 2
                elif i == rho.size - 1:
                    drho[i] = 1 - (rho[-2] + rho[-1]) / 2
                else:
                    drho[i] = (rho[i + 1] - rho[i - 1]) / 2
        else:
            drho = np.array([1.0])
        r = []
        t = []
        dr = []
        dt = []

        for iring in range(L // 2 + 1, 0, -1):
            ntheta = 2 * M + np.ceil((M / L) * (5 - 4 * iring)).astype(int)
            dtheta = 2 * np.pi / ntheta
            theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
            if rotation in {None, False}:
                if self.sym:
                    # this is emperically chosen, could be something different, just
                    # need to avoid symmetry at theta=0, pi
                    offset = dtheta / 3
                else:
                    offset = 0
            elif rotation in ["cos", "cosine"]:  # cos(m*t-n*z) symmetry
                offset = 0
            elif rotation in ["sin", "sine"]:  # sin(m*t-n*z) symmetry
                offset = dtheta / 4
            else:
                raise ValueError(f"Unknown rotation type {rotation}")

            theta = (theta + offset) % (2 * np.pi)
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

        dz = 2 * np.pi / (NFP * (2 * N + 1))
        z = np.linspace(0, 2 * np.pi / NFP, 2 * N + 1, endpoint=False)

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


# these functions are currently unused ---------------------------------------

# TODO: finish option for placing nodes at irrational surfaces


def dec_to_cf(x, dmax=6):  # pragma: no cover
    """Compute continued fraction form of a number.

    Parameters
    ----------
    x : float
        floating point form of number
    dmax : int
        maximum iterations (ie, number of coefficients of continued fraction).
        (Default value = 6)

    Returns
    -------
    cf : ndarray of int
        coefficients of continued fraction form of x.

    """
    cf = []
    q = np.floor(x)
    cf.append(q)
    x = x - q
    i = 0
    while x != 0 and i < dmax:
        q = np.floor(1 / x)
        cf.append(q)
        x = 1 / x - q
        i = i + 1
    return np.array(cf)


def cf_to_dec(cf):  # pragma: no cover
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


def most_rational(a, b):  # pragma: no cover
    """Compute the most rational number in the range [a,b].

    Parameters
    ----------
    a,b : float
        lower and upper bounds

    Returns
    -------
    x : float
        most rational number between [a,b]

    """
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
    idx = 0  # first idex of dissimilar digits
    for i in range(min(a_cf.size, b_cf.size)):
        if a_cf[i] != b_cf[i]:
            idx = i
            break
    f = 1
    while True:
        dec = cf_to_dec(np.append(a_cf[0:idx], f))
        if dec >= a and dec <= b:
            return dec * s
        f += 1
