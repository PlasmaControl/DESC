import numpy as np
from termcolor import colored

from desc.io import IOAble
from scipy import special

__all__ = ["Grid", "LinearGrid", "QuadratureGrid", "ConcentricGrid"]


class Grid(IOAble):
    """Base class for collocation grids

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
        "_weights",
        "_axis",
        "_node_pattern",
    ]

    def __init__(self, nodes, sort=True):

        self._L = np.unique(nodes[:, 0]).size
        self._M = np.unique(nodes[:, 1]).size
        self._N = np.unique(nodes[:, 2]).size
        self._NFP = 1
        self._sym = False
        self._node_pattern = "custom"

        self._nodes, self._weights = self._create_nodes(nodes)

        self._enforce_symmetry()
        if sort:
            self._sort_nodes()
        self._find_axis()
        self._scale_weights()

    def _enforce_symmetry(self):
        """Enforces stellarator symmetry"""
        if self.sym:  # remove nodes with theta > pi
            non_sym_idx = np.where(self.nodes[:, 1] > np.pi)
            self._nodes = np.delete(self.nodes, non_sym_idx, axis=0)
            self._weights = np.delete(self.weights, non_sym_idx, axis=0)

    def _sort_nodes(self):
        """Sorts nodes for use with FFT"""

        sort_idx = np.lexsort((self.nodes[:, 1], self.nodes[:, 0], self.nodes[:, 2]))
        self._nodes = self.nodes[sort_idx]
        self._weights = self.weights[sort_idx]

    def _find_axis(self):
        """Finds indices of axis nodes"""
        self._axis = np.where(self.nodes[:, 0] == 0)[0]

    def _scale_weights(self):
        """Scales weights to sum to full volume and reduces weights for duplicated nodes"""

        nodes = self.nodes.copy().astype(float)
        nodes[:, 1] %= 2 * np.pi
        nodes[:, 2] %= 2 * np.pi / self.NFP
        _, inverse, counts = np.unique(
            nodes, axis=0, return_inverse=True, return_counts=True
        )
        self._weights /= counts[inverse]
        self._weights *= 4 * np.pi ** 2 / self._weights.sum()

    def _create_nodes(self, nodes):
        """Allows for custom node creation

        Parameters
        ----------
        nodes : ndarray of float, size(num_nodes,3)
            node coordinates, in (rho,theta,zeta)

        Returns
        -------
        nodes : ndarray of float, size(num_nodes,3)
            node coordinates, in (rho,theta,zeta)

        """
        nodes = np.atleast_2d(nodes).reshape((-1, 3))
        # make weights sum to 4pi^2
        weights = np.ones(nodes.shape[0]) / nodes.shape[0] * 4 * np.pi ** 2
        self._L = len(np.unique(nodes[:, 0]))
        self._M = len(np.unique(nodes[:, 1]))
        self._N = len(np.unique(nodes[:, 2]))
        return nodes, weights

    @property
    def L(self):
        """int: radial grid resolution"""
        return self.__dict__.setdefault("_L", 0)

    @property
    def M(self):
        """ int: poloidal grid resolution"""
        return self.__dict__.setdefault("_M", 0)

    @property
    def N(self):
        """ int: toroidal grid resolution"""
        return self.__dict__.setdefault("_N", 0)

    @property
    def NFP(self):
        """ int: number of field periods"""
        return self.__dict__.setdefault("_NFP", 1)

    @property
    def sym(self):
        """ bool: True for stellarator symmetry, False otherwise"""
        return self.__dict__.setdefault("_sym", False)

    @property
    def nodes(self):
        """ndarray: node coordinates, in (rho,theta,zeta)"""
        return self.__dict__.setdefault("_nodes", np.array([]).reshape((0, 3)))

    @nodes.setter
    def nodes(self, nodes):
        self._nodes = nodes

    @property
    def weights(self):
        """ndarray: weight for each node, either exact quadrature or volume based"""
        return self.__dict__.setdefault("_weights", np.array([]).reshape((0, 3)))

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def num_nodes(self):
        """int: total number of nodes"""
        return self.nodes.shape[0]

    @property
    def axis(self):
        """ndarray: indices of nodes at magnetic axis"""
        return self.__dict__.setdefault("_axis", np.array([]))

    @property
    def node_pattern(self):
        """str: pattern for placement of nodes in rho,theta,zeta"""
        return self.__dict__.setdefault("_node_pattern", "custom")

    def __repr__(self):
        """string form of the object"""
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

        self._nodes, self._weights = self._create_nodes(
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
        """

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
        weights : ndarray of float, size(num_nodes,)
            weight for each node, based on local volume around the node

        """
        self._L = L
        self._M = M
        self._N = N
        self._NFP = NFP

        # rho
        if rho is not None:
            r = np.atleast_1d(rho)
            r0 = r[0]
            self._L = r.size
        elif self.L == 1:
            r = np.array([1.0])
            r0 = 0
        else:
            if axis:
                r0 = 0
            else:
                r0 = 1.0 / self.L
            r = np.linspace(r0, 1, self.L)
        dr = (1 - r0) / self.L
        if dr == 0:
            dr = 1

        # theta/vartheta
        if theta is not None:
            t = np.asarray(theta)
            self._M = t.size
        else:
            t = np.linspace(0, 2 * np.pi, self.M, endpoint=endpoint)
        dt = 2 * np.pi / self.M

        # zeta/phi
        if zeta is not None:
            z = np.asarray(zeta)
            self._N = z.size
        else:
            z = np.linspace(0, 2 * np.pi / self.NFP, self.N, endpoint=endpoint)
        dz = 2 * np.pi / self.NFP / self.N

        r, t, z = np.meshgrid(r, t, z, indexing="ij")
        r = r.flatten()
        t = t.flatten()
        z = z.flatten()

        dr = dr * np.ones_like(r)
        dt = dt * np.ones_like(t)
        dz = dz * np.ones_like(z)

        nodes = np.stack([r, t, z]).T
        weights = dr * dt * dz

        return nodes, weights

    def change_resolution(self, L, M, N):
        """Change the resolution of the grid

        Parameters
        ----------
        L : int
            new radial grid resolution (L radial nodes)
        M : int
            new poloidal grid resolution (M poloidal nodes)
        N : int
            new toroidal grid resolution (N toroidal nodes)

        """
        if L != self.L or M != self.M or N != self.N:
            self._L = L
            self._M = M
            self._N = N
            self._nodes, self._weights = self._create_nodes(
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

    @property
    def endpoint(self):
        """bool: whether the grid is made of open or closed intervals"""
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

        self._nodes, self._weights = self._create_nodes(
            L=self.L, M=self.M, N=self.N, NFP=self.NFP
        )

        self._enforce_symmetry()  # symmetry is never enforced for Quadrature Grid
        self._sort_nodes()
        self._find_axis()
        # quad grid should already be exact, so we don't scale weights

    def _create_nodes(self, L=1, M=1, N=1, NFP=1):
        """

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
        weights : ndarray of float, size(num_nodes,)
            weight for each node, based on local volume around the node

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
        weights = wr * wt * wz

        return nodes, weights

    def change_resolution(self, L, M, N):
        """Change the resolution of the grid

        Parameters
        ----------
        L : int
            new radial grid resolution (L radial nodes)
        M : int
            new poloidal grid resolution (M poloidal nodes)
        N : int
            new toroidal grid resolution (N toroidal nodes)

        """
        if L != self.L or M != self.M or N != self.N:
            self._L = L
            self._M = M
            self._N = N
            self._nodes, self._weights = self._create_nodes(L=L, M=M, N=N, NFP=self.NFP)
            self._enforce_symmetry()
            self._sort_nodes()
            self._find_axis()


class ConcentricGrid(Grid):
    """Grid in which the nodes are arranged in concentric circles

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
    node_pattern : {``'cheb1'``, ``'cheb2'``, ``'jacobi'``, ``None``}
        pattern for radial coordinates

            * ``'cheb1'``: Chebyshev-Gauss-Lobatto nodes scaled to r=[0,1]
            * ``'cheb2'``: Chebyshev-Gauss-Lobatto nodes scaled to r=[-1,1]
            * ``'jacobi'``: Radial nodes are roots of Shifted Jacobi polynomial of degree
              M+1 r=(0,1), and angular nodes are equispaced 2(M+1) per surface
            * ``'ocs'``: optimal concentric sampling to minimize the condition number
              of the resulting transform matrix, for doing inverse transform.
            * ``None`` : linear spacing in r=[0,1]


    """

    def __init__(self, L, M, N, NFP=1, sym=False, axis=False, node_pattern="jacobi"):

        self._L = L
        self._M = M
        self._N = N
        self._NFP = NFP
        self._sym = sym
        self._axis = axis
        self._node_pattern = node_pattern

        self._nodes, self._weights = self._create_nodes(
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
        self._scale_weights()

    def _create_nodes(self, L, M, N, NFP=1, axis=False, node_pattern="jacobi"):
        """

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
        node_pattern : {``'cheb1'``, ``'cheb2'``, ``'jacobi'``, ``None``}
            pattern for radial coordinates

                * ``'cheb1'``: Chebyshev-Gauss-Lobatto nodes scaled to r=[0,1]
                * ``'cheb2'``: Chebyshev-Gauss-Lobatto nodes scaled to r=[-1,1]
                * ``'jacobi'``: Radial nodes are roots of Shifted Jacobi polynomial of degree
                M+1 r=(0,1), and angular nodes are equispaced 2(M+1) per surface
                * ``'ocs'``: optimal concentric sampling to minimize the condition number
                  of the resulting transform matrix, for doing inverse transform.
                * ``None`` : linear spacing in r=[0,1]

        Returns
        -------
        nodes : ndarray of float, size(num_nodes, 3)
            node coordinates, in (rho,theta,zeta)
        weights : ndarray of float, size(num_nodes,)
            weight for each node, either exact quadrature or volume based

        """

        def ocs(L):
            # from Ramos-Lopez, et al “Optimal Sampling Patterns for Zernike Polynomials.”
            # Applied Mathematics and Computation 274 (February 2016): 247–57.
            # https://doi.org/10.1016/j.amc.2015.11.006.
            j = np.arange(1, L // 2 + 2)
            z = np.cos((2 * j - 1) * np.pi / (2 * L + 2))
            rj = 1.1565 * z - 0.76535 * z ** 2 + 0.60517 * z ** 3
            return np.sort(rj)

        pattern = {
            "cheb1": (np.cos(np.arange(L // 2, -1, -1) * np.pi / (L // 2)) + 1) / 2,
            "cheb2": -np.cos(np.arange(L // 2, L + 1, 1) * np.pi / L),
            "jacobi": special.js_roots(L // 2 + 1, 2, 2)[0],
            "ocs": ocs(L),
        }
        rho = pattern.get(node_pattern, np.linspace(0, 1, num=L // 2 + 1))
        rho = np.sort(rho, axis=None)
        if axis:
            rho[0] = 0
        elif rho[0] == 0:
            rho[0] = rho[1] / 10

        drho = np.zeros_like(rho)
        for i in range(rho.size):
            if i == 0:
                drho[i] = (rho[0] + rho[1]) / 2
            elif i == rho.size - 1:
                drho[i] = 1 - (rho[-2] + rho[-1]) / 2
            else:
                drho[i] = (rho[i + 1] - rho[i - 1]) / 2

        r = []
        t = []
        dr = []
        dt = []

        for iring in range(L // 2 + 1, 0, -1):
            dtheta = (
                2 * np.pi / (2 * M + np.ceil((M / L) * (5 - 4 * iring)).astype(int))
            )
            theta = np.arange(0, 2 * np.pi, dtheta)
            if self.sym:
                theta = (theta + dtheta / 3) % (2 * np.pi)
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
        z = np.arange(0, 2 * np.pi / NFP, dz)

        r = np.tile(r, 2 * N + 1)
        t = np.tile(t, 2 * N + 1)
        z = np.tile(z[np.newaxis], (dimzern, 1)).flatten(order="F")
        dr = np.tile(dr, 2 * N + 1)
        dt = np.tile(dt, 2 * N + 1)
        dz = np.ones_like(z) * dz
        nodes = np.stack([r, t, z]).T
        weights = dr * dt * dz

        return nodes, weights

    def change_resolution(self, L, M, N):
        """Change the resolution of the grid

        Parameters
        ----------
        L : int
            new radial grid resolution
        M : int
            new poloidal grid resolution
        N : int
            new toroidal grid resolution

        """
        if L != self.L or M != self.M or N != self.N:
            self._L = L
            self._M = M
            self._N = N
            self._nodes, self._weights = self._create_nodes(
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


# these functions are currently unused ---------------------------------------

# TODO: finish option for placing nodes at irrational surfaces


def dec_to_cf(x, dmax=6):  # pragma: no cover
    """Compute continued fraction form of a number.

    Parameters
    ----------
    x : float
        floating point form of number
    dmax : int
        maximum iterations (ie, number of coefficients of continued fraction). (Default value = 6)

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
    """Compute the most rational number in the range [a,b]

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
