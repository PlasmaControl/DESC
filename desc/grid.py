import numpy as np
from termcolor import colored
from desc.utils import equals
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

    """

    # TODO: calculate weights automatically using voronoi / delaunay triangulation
    _io_attrs_ = ["_L", "_M", "_N", "_NFP", "_sym", "_nodes", "_weights"]

    def __init__(self, nodes, load_from=None, file_format=None, obj_lib=None):

        self._file_format_ = file_format

        if load_from is None:
            self._L = None
            self._M = None
            self._N = None
            self._NFP = None
            self._sym = False

            self._nodes, self._weights = self._create_nodes(nodes)

            self._enforce_symmetry()
            self._sort_nodes()
            self._find_axis()

        else:
            self._init_from_file_(
                load_from=load_from, file_format=file_format, obj_lib=obj_lib
            )

    def __eq__(self, other):
        """Overloads the == operator

        Parameters
        ----------
        other : Grid
            another Grid object to compare to

        Returns
        -------
        bool
            True if other is a Grid with the same attributes as self
            False otherwise

        """
        if self.__class__ != other.__class__:
            return False
        return equals(self.__dict__, other.__dict__)

    def _enforce_symmetry(self):
        """Enforces stellarator symmetry"""
        if self._sym:  # remove nodes with theta > pi
            non_sym_idx = np.where(self._nodes[:, 1] > np.pi)
            self._nodes = np.delete(self._nodes, non_sym_idx, axis=0)
            self._weights = np.delete(self._weights, non_sym_idx, axis=0)

    def _sort_nodes(self):
        """Sorts nodes for use with FFT"""

        sort_idx = np.lexsort((self._nodes[:, 1], self._nodes[:, 0], self._nodes[:, 2]))
        self._nodes = self._nodes[sort_idx]
        self._weights = self._weights[sort_idx]

    def _find_axis(self):
        """Finds indices of axis nodes"""
        self._axis = np.where(self._nodes[:, 0] == 0)[0]

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
        weights = np.ones(nodes.shape[0])
        self._L = len(np.unique(nodes[:, 0]))
        self._M = len(np.unique(nodes[:, 1]))
        self._N = len(np.unique(nodes[:, 2]))
        return nodes, weights

    @property
    def L(self):
        """int: radial grid resolution"""
        return self._L

    @property
    def M(self):
        """ int: poloidal grid resolution"""
        return self._M

    @property
    def N(self):
        """ int: toroidal grid resolution"""
        return self._N

    @property
    def NFP(self):
        """ int: number of field periods"""
        return self._NFP

    @property
    def sym(self):
        """ bool: True for stellarator symmetry, False otherwise"""
        return self._sym

    @property
    def nodes(self):
        """ndarray: node coordinates, in (rho,theta,zeta)"""
        return self._nodes

    @nodes.setter
    def nodes(self, nodes):
        self._nodes = nodes

    @property
    def weights(self):
        """ndarray: weight for each node, either exact quadrature or volume based"""
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def num_nodes(self):
        """int: total number of nodes"""
        return self._nodes.shape[0]

    @property
    def axis(self):
        """ndarray: indices of nodes at magnetic axis"""
        return self._axis


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
        load_from=None,
        file_format=None,
        obj_lib=None,
    ):

        self._file_format_ = file_format

        if load_from is None:
            self._L = L
            self._M = M
            self._N = N
            self._NFP = NFP
            self._axis = axis
            self._sym = sym
            self._endpoint = endpoint
            self._rho = rho
            self._theta = theta
            self._zeta = zeta

            self._nodes, self._weights = self._create_nodes(
                L=self._L,
                M=self._M,
                N=self._N,
                NFP=self._NFP,
                axis=self._axis,
                endpoint=self._endpoint,
                rho=self._rho,
                theta=self._theta,
                zeta=self._zeta,
            )

            self._enforce_symmetry()
            self._sort_nodes()
            self._find_axis()

        else:
            self._init_from_file_(
                load_from=load_from, file_format=file_format, obj_lib=obj_lib
            )

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
        elif self._L == 1:
            r = np.array([1.0])
            r0 = 0
        else:
            if axis:
                r0 = 0
            else:
                r0 = 1.0 / self._L
            r = np.linspace(r0, 1, self._L)
        dr = (1 - r0) / self._L

        # theta/vartheta
        if theta is not None:
            t = np.asarray(theta)
            self._M = t.size
        else:
            t = np.linspace(0, 2 * np.pi, self._M, endpoint=endpoint)
        dt = 2 * np.pi / self._M

        # zeta/phi
        if zeta is not None:
            z = np.asarray(zeta)
            self._N = z.size
        else:
            z = np.linspace(0, 2 * np.pi / self._NFP, self._N, endpoint=endpoint)
        dz = 2 * np.pi / self._NFP / self._N

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
        if L != self._L or M != self._M or N != self._N:
            self._L = L
            self._M = M
            self._N = N
            self._nodes, self._weights = self._create_nodes(
                L=L,
                M=M,
                N=N,
                NFP=self._NFP,
                axis=self._axis,
                endpoint=self._endpoint,
            )
            self._sort_nodes()


class QuadratureGrid(Grid):
    """Grid used for numerical quadrature.

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

    """

    def __init__(
        self,
        L=1,
        M=1,
        N=1,
        NFP=1,
        sym=False,
        load_from=None,
        file_format=None,
        obj_lib=None,
    ):

        self._file_format_ = file_format

        if load_from is None:
            self._L = L
            self._M = M
            self._N = N
            self._NFP = NFP
            self._sym = sym

            self._nodes, self._weights = self._create_nodes(
                L=self._L,
                M=self._M,
                N=self._N,
                NFP=self._NFP,
            )

            self._enforce_symmetry()
            self._sort_nodes()
            self._find_axis()

        else:
            self._init_from_file_(
                load_from=load_from, file_format=file_format, obj_lib=obj_lib
            )

    def _create_nodes(
        self,
        L=1,
        M=1,
        N=1,
        NFP=1,
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
        r, wr = special.js_roots(self._L, 2, 2)

        # theta/vartheta
        t = np.linspace(0, 2 * np.pi, self._M, endpoint=False)
        wt = 2 * np.pi / self._M * np.ones_like(t)

        # zeta/phi
        z = np.linspace(0, 2 * np.pi / self._NFP, self._N, endpoint=False)
        wz = 2 * np.pi / self._N * np.ones_like(z)

        r, t, z = np.meshgrid(r, t, z, indexing="ij")
        r = r.flatten()
        t = t.flatten()
        z = z.flatten()

        wr, wt, wz = np.meshgrid(wr, wt, wz, indexing="ij")
        wr = wr.flatten()
        wt = wt.flatten()
        wz = wz.flatten()

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
        if L != self._L or M != self._M or N != self._N:
            self._L = L
            self._M = M
            self._N = N
            self._nodes, self._weights = self._create_nodes(
                L=L, M=M, N=N, NFP=self._NFP
            )
            self._sort_nodes()


class ConcentricGrid(Grid):
    """Grid in which the nodes are arranged in concentric circles

    Nodes are arranged concentrically within each toroidal cross-section, with more
    nodes per flux surface at larger radius. Typically used as the solution grid,
    cannot be easily used for plotting due to non-uniform spacing.

    Parameters
    ----------
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
    index : {'ansi', 'chevron', 'fringe', 'house'}
        Zernike indexing scheme
    surfs : {'cheb1', 'cheb2', 'quad', None}
        pattern for radial coordinates

            * 'cheb1': Chebyshev-Gauss-Lobatto nodes scaled to r=[0,1]
            * 'cheb2': Chebyshev-Gauss-Lobatto nodes scaled to r=[-1,1]
            * 'quad': Radial nodes are roots of Shifted Jacobi polynomial of degree
              M+1 r=(0,1), and angular nodes are equispaced 2(M+1) per surface
            * None : linear spacing in r=[0,1]


    """

    def __init__(
        self,
        M,
        N,
        NFP=1,
        sym=False,
        axis=False,
        index="ansi",
        surfs="cheb1",
        load_from=None,
        file_format=None,
        obj_lib=None,
    ):

        self._file_format_ = file_format

        if load_from is None:
            self._L = M + 1
            self._M = M
            self._N = N
            self._NFP = NFP
            self._sym = sym
            self._axis = axis
            self._index = index
            self._surfs = surfs

            self._nodes, self._weights = self._create_nodes(
                M=self._M,
                N=self._N,
                NFP=self._NFP,
                axis=self._axis,
                index=self._index,
                surfs=self._surfs,
            )

            self._enforce_symmetry()
            self._sort_nodes()
            self._find_axis()

        else:
            self._init_from_file_(
                load_from=load_from, file_format=file_format, obj_lib=obj_lib
            )

    def _create_nodes(
        self,
        M,
        N,
        NFP=1,
        axis=False,
        index="ansi",
        surfs="cheb1",
    ):
        """

        Parameters
        ----------
        M : int
            poloidal grid resolution
        N : int
            toroidal grid resolution
        NFP : int
            number of field periods (Default = 1)
        axis : bool
            True to include the magnetic axis, False otherwise (Default = False)
        index : {'ansi', 'chevron', 'fringe', 'house'}
            Zernike indexing scheme
        surfs : {'cheb1', 'cheb2', 'jacobi', None}
            pattern for radial coordinates

                * 'cheb1': Chebyshev-Gauss-Lobatto nodes scaled to r=[0,1]
                * 'cheb2': Chebyshev-Gauss-Lobatto nodes scaled to r=[-1,1]
                * 'jacobi': Radial nodes are roots of Shifted Jacobi polynomial of degree
                M+1 r=(0,1), and angular nodes are equispaced 2(M+1) per surface
                * None : linear spacing in r=[0,1]

        Returns
        -------
        nodes : ndarray of float, size(num_nodes, 3)
            node coordinates, in (rho,theta,zeta)
        weights : ndarray of float, size(num_nodes,)
            weight for each node, either exact quadrature or volume based

        """
        dim_fourier = 2 * N + 1
        if index in ["ansi", "chevron"]:
            dim_zernike = int((M + 1) * (M + 2) / 2)
            a = 1
        elif index in ["fringe", "house"]:
            dim_zernike = int((M + 1) ** 2)
            a = 2
        else:
            raise ValueError(
                colored(
                    "Zernike indexing must be one of 'ansi', 'fringe', 'chevron', 'house'",
                    "red",
                )
            )

        pattern = {
            "cheb1": (np.cos(np.arange(M, -1, -1) * np.pi / M) + 1) / 2,
            "cheb2": -np.cos(np.arange(M, 2 * M + 1, 1) * np.pi / (2 * M)),
            "jacobi": special.js_roots(M + 1, 2, 2)[0],
        }
        rho = pattern.get(surfs, np.linspace(0, 1, num=M + 1))
        rho = np.sort(rho, axis=None)
        if axis:
            rho[0] = 0
        elif rho[0] == 0:
            rho[0] = rho[1] / 10

        if surfs == "jacobi":
            drho = special.js_roots(M + 1, 2, 2)[1]
        else:
            drho = np.zeros_like(rho)
            for i in range(rho.size):
                if i == 0:
                    drho[i] = (rho[0] + rho[1]) / 2
                elif i == rho.size - 1:
                    drho[i] = 1 - (rho[-2] + rho[-1]) / 2
                else:
                    drho[i] = (rho[i + 1] - rho[i - 1]) / 2

        r = np.zeros(dim_zernike)
        t = np.zeros(dim_zernike)
        dr = np.zeros(dim_zernike)
        dt = np.zeros(dim_zernike)

        i = 0
        for m in range(M + 1):
            dtheta = 2 * np.pi / (a * m + 1)
            theta = np.arange(0, 2 * np.pi, dtheta)
            for j in range(a * m + 1):
                r[i] = rho[m]
                t[i] = theta[j]
                dr[i] = drho[m]
                dt[i] = dtheta
                i += 1

        dz = 2 * np.pi / (NFP * dim_fourier)
        z = np.arange(0, 2 * np.pi / NFP, dz)

        r = np.tile(r, dim_fourier)
        t = np.tile(t, dim_fourier)
        z = np.tile(z[np.newaxis], (dim_zernike, 1)).flatten(order="F")
        dr = np.tile(dr, dim_fourier)
        dt = np.tile(dt, dim_fourier)
        dz = np.ones_like(z) * dz

        nodes = np.stack([r, t, z]).T
        weights = dr * dt * dz

        return nodes, weights

    def change_resolution(self, M, N):
        """Change the resolution of the grid

        Parameters
        ----------
        M : int
            new poloidal grid resolution
        N : int
            new toroidal grid resolution

        """
        if M != self._M or N != self._N:
            self._L = M + 1
            self._M = M
            self._N = N
            self._nodes, self._weights = self._create_nodes(
                M=M, N=N, NFP=self._NFP, surfs=self._surfs
            )
            self._sort_nodes()


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
