import numpy as np
from abc import abstractmethod

from desc.backend import TextColors, equals
from desc.equilibrium_io import IOAble


class Grid(IOAble):
    """Grid is a base class for collocation grids

    Attributes
    ----------
    L : int
        radial grid resolution
    M : int
        poloidal grid resolution
    N : int
        toroidal grid resolution
    NFP : int
        number of field periods
    sym : bool
        True for stellarator symmetry, False otherwise (Default = False)
    nodes : ndarray of float, size(3,Nnodes)
        node coordinates, in (rho,theta,zeta)
    volumes : ndarray of float, size(3,Nnodes)
        node spacing (drho,dtheta,dzeta) at each node coordinate

    """
    _save_attrs_ = ['_Grid__L', '_Grid__M', '_Grid__N', '_Grid__NFP',
                             '_Grid__sym', '_Grid__nodes', '_Grid__volumes']

    def __init__(self, nodes, load_from=None, file_format=None, obj_lib=None) -> None:
        """Initializes a custom grid without a pre-defined pattern

        Parameters
        ----------
        nodes : ndarray of float, size(3,Nnodes)
            node coordinates, in (rho,theta,zeta)

        Returns
        -------
        None

        """
        if load_from is None:
            self.__L = None
            self.__M = None
            self.__N = None
            self.__NFP = None
            self.__sym = False

            self.__nodes, self.__volumes = self.create_nodes(nodes)

            self._enforce_symmetry_()
            self._sort_nodes_()
            self._find_axis_()
            #self._def_save_attrs()
        else:
            self._init_from_file(load_from, file_format=file_format, obj_lib=obj_lib)

    def __eq__(self, other) -> bool:
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

    def _enforce_symmetry_(self) -> None:
        """Enforces stellarator symmetry

        Returns
        -------
        None

        """
        if self.__sym:  # remove nodes with theta > pi
            non_sym_idx = np.where(self.__nodes[:, 1] > np.pi)
            self.__nodes = np.delete(self.__nodes, non_sym_idx, axis=0)
            self.__volumes = np.delete(self.__volumes, non_sym_idx, axis=0)

    def _sort_nodes_(self) -> None:
        """Sorts nodes for use with FFT

            Returns
            -------
            None

        """
        sort_idx = np.lexsort((self.__nodes[:, 0], self.__nodes[:, 1],
                               self.__nodes[:, 2]))
        self.__nodes = self.__nodes[sort_idx]
        self.__volumes = self.__volumes[sort_idx]

    def _find_axis_(self) -> None:
        """Finds indices of axis nodes

        Returns
        -------
        None

        """
        self.__axis = np.where(self.__nodes[:, 0] == 0)[0]

    def _def_save_attrs_(self) -> None:
        """Defines attributes to save

        Returns
        -------
        None

        """
        self._save_attrs_ = ['_Grid__L', '_Grid__M', '_Grid__N', '_Grid__NFP',
                             '_Grid__sym', '_Grid__nodes', '_Grid__volumes']

    def create_nodes(self, nodes):
        """Allows for custom node creation

        Parameters
        ----------
        nodes : ndarray of float, size(3,Nnodes)
            node coordinates, in (rho,theta,zeta)

        Returns
        -------
        nodes : ndarray of float, size(3,Nnodes)
            node coordinates, in (rho,theta,zeta)

        """
        nodes = np.atleast_2d(nodes).reshape((-1, 3))
        volumes = np.zeros_like(nodes)
        return nodes, volumes

    @abstractmethod
    def change_resolution(self) -> None:
        pass

    @property
    def L(self) -> int:
        return self.__L

    @property
    def M(self) -> int:
        return self.__M

    @property
    def N(self) -> int:
        return self.__N

    @property
    def NFP(self) -> int:
        return self.__NFP

    @property
    def sym(self) -> bool:
        return self.__sym

    @property
    def nodes(self):
        return self.__nodes

    @nodes.setter
    def nodes(self, nodes) -> None:
        self.__nodes = nodes

    @property
    def volumes(self):
        return self.__volumes

    @volumes.setter
    def volumes(self, volumes) -> None:
        self.__volumes = volumes

    @property
    def num_nodes(self):
        return self.__nodes.shape[0]

    @property
    def axis(self):
        return self.__axis


class LinearGrid(Grid):
    """LinearGrid is a collocation grid in which the nodes are linearly
       spaced in each coordinate.
    """

    def __init__(self, L:int=1, M:int=1, N:int=1, NFP:int=1, sym:bool=False,
                 endpoint:bool=False, rho=np.array([1.0]),
                 theta=np.array([1.0]), zeta=np.array([1.0]),
                 load_from=None, file_format=None, obj_lib=None) -> None:
        """Initializes a LinearGrid

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
        endpoint : bool
            if True, theta=0 and zeta=0 are duplicated after a full period.
            Should be False for use with FFT (Default = False)
        rho : ndarray of float
            radial coordinates (if L == rho.size)
        theta : ndarray of float
            poloidal coordinates (if M == theta.size)
        zeta : ndarray of float
            toroidal coordinates (if N == zeta.size)

        Returns
        -------
        None

        """
        if load_from is None:
            self._Grid__L = L
            self._Grid__M = M
            self._Grid__N = N
            self._Grid__NFP = NFP
            self._Grid__sym = sym
            self.__endpoint = endpoint
            self.__rho = rho
            self.__theta = theta
            self.__zeta = zeta

            self._Grid__nodes, self._Grid__volumes = self.create_nodes(
                            L=self._Grid__L, M=self._Grid__M, N=self._Grid__N,
                            NFP=self._Grid__NFP, endpoint=self.__endpoint,
                            rho=self.__rho, theta=self.__theta, zeta=self.__zeta)

            self._enforce_symmetry_()
            self._sort_nodes_()
            self._find_axis_()
            #self._def_save_attrs_()
        else:
            self._init_from_file_(load_from=load_from, file_format=file_format, obj_lib=obj_lib)

    def create_nodes(self, L:int=1, M:int=1, N:int=1, NFP:int=1,
                     endpoint:bool=False, rho=np.array([1.0]),
                     theta=np.array([1.0]), zeta=np.array([1.0])):
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
        endpoint : bool
            if True, theta=0 and zeta=0 are duplicated after a full period.
            Should be False for use with FFT (Default = False)
        rho : ndarray of float
            radial coordinates (if L == rho.size)
        theta : ndarray of float
            poloidal coordinates (if M == theta.size)
        zeta : ndarray of float
            toroidal coordinates (if N == zeta.size)

        Returns
        -------
        nodes : ndarray of float, size(3,Nnodes)
            node coordinates, in (rho,theta,zeta)
        volumes : ndarray of float, size(3,Nnodes)
            node spacing (drho,dtheta,dzeta) at each node coordinate

        """
        # rho
        if rho.size == L:
            r = rho
        else:
            r = np.linspace(0, 1, L)
        dr = 1/L

        # theta/vartheta
        if theta.size == M:
            t = theta
        else:
            t = np.linspace(0, 2*np.pi, M, endpoint=endpoint)
        dt = 2*np.pi/M

        # zeta/phi
        if zeta.size == N:
            z = zeta
        else:
            z = np.linspace(0, 2*np.pi/NFP, N, endpoint=endpoint)
        dz = 2*np.pi/NFP/N

        r, t, z = np.meshgrid(r, t, z, indexing='ij')
        r = r.flatten()
        t = t.flatten()
        z = z.flatten()

        dr = dr*np.ones_like(r)
        dt = dt*np.ones_like(t)
        dz = dz*np.ones_like(z)

        nodes = np.stack([r, t, z]).T
        volumes = np.stack([dr, dt, dz]).T

        return nodes, volumes

    def change_resolution(self, L:int, M:int, N:int) -> None:
        """

        Parameters
        ----------
        L : int
            new radial grid resolution (L radial nodes)
        M : int
            new poloidal grid resolution (2*M+1 poloidal nodes)
        N : int
            new toroidal grid resolution (2*N+1 toroidal nodes)

        Returns
        -------
        None

        """
        if L != self._Grid__L or M != self._Grid__M or N != self._Grid__N:
            self._Grid__L = L
            self._Grid__M = M
            self._Grid__N = N
            self._Grid__nodes, self._Grid__volumes = self.create_nodes(L=L, M=M, N=N,
                                 NFP=self._Grid__NFP, sym=self._Grid__sym,
                                 endpoint=self.__endpoint, surfs=self.__surfs)
            self.sort_nodes()


class ConcentricGrid(Grid):
    """ConcentricGrid is a collocation grid in which the nodes are arranged
       in concentric circles within each toroidal cross-section.
    """

    def __init__(self, M:int, N:int, NFP:int=1, sym:bool=False, axis:bool=True,
                 index='ansi', surfs='cheb1', load_from=None, file_format=None,
                 obj_lib=None) -> None:
        """Initializes a ConcentricGrid

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
            True to include the magnetic axis, False otherwise (Default = True)
        index : string
            Zernike indexing scheme
                ansi (Default), chevron, fringe, house
        surfs : string
            pattern for radial coordinates
                cheb1 = Chebyshev-Gauss-Lobatto nodes scaled to r=[0,1]
                cheb2 = Chebyshev-Gauss-Lobatto nodes scaled to r=[-1,1]
                anything else defaults to linear spacing in r=[0,1]

        Returns
        -------
        None

        """
        if load_from is None:
            self._Grid__L = M+1
            self._Grid__M = M
            self._Grid__N = N
            self._Grid__NFP = NFP
            self._Grid__sym = sym
            self.__axis = axis
            self.__index = index
            self.__surfs = surfs

            self._Grid__nodes, self._Grid__volumes = self.create_nodes(
                        M=self._Grid__M, N=self._Grid__N, NFP=self._Grid__NFP,
                        axis=self.__axis, index=self.__index, surfs=self.__surfs)

            self._enforce_symmetry_()
            self._sort_nodes_()
            self._find_axis_()
            #self._def_save_attrs_()
        else:
            self._init_from_file(load_from=load_from, file_format=file_format, obj_lib=obj_lib)

    def create_nodes(self, M:int, N:int, NFP:int=1, axis:bool=True,
                     index='ansi', surfs='cheb1'):
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
            True to include the magnetic axis, False otherwise (Default = True)
        index : string
            Zernike indexing scheme
                ansi (Default), chevron, fringe, house
        surfs : string
            pattern for radial coordinates
                cheb1 = Chebyshev-Gauss-Lobatto nodes scaled to r=[0,1]
                cheb2 = Chebyshev-Gauss-Lobatto nodes scaled to r=[-1,1]
                anything else defaults to linear spacing in r=[0,1]

        Returns
        -------
        nodes : ndarray of float, size(3,Nnodes)
            node coordinates, in (rho,theta,zeta)
        volumes : ndarray of float, size(3,Nnodes)
            node spacing (drho,dtheta,dzeta) at each node coordinate

        """
        dim_fourier = 2*N+1
        if index in ['ansi', 'chevron']:
            dim_zernike = int((M+1)*(M+2)/2)
            a = 1
        elif index in ['fringe', 'house']:
            dim_zernike = int((M+1)**2)
            a = 2
        else:
            raise ValueError(TextColors.FAIL +
                             "Invalid index input." + TextColors.ENDC)

        pattern = {
            'cheb1': (np.cos(np.arange(M, -1, -1)*np.pi/M)+1)/2,
            'cheb2': -np.cos(np.arange(M, 2*M+1, 1)*np.pi/(2*M))
        }
        rho = pattern.get(surfs, np.linspace(0, 1, num=M+1))
        rho = np.sort(rho, axis=None)
        if axis:
            rho[0] = 0
        else:
            rho[0] = rho[1]/4

        drho = np.zeros_like(rho)
        for i in range(rho.size):
            if i == 0:
                drho[i] = (rho[0]+rho[1])/2
            elif i == rho.size-1:
                drho[i] = 1-(rho[-2]+rho[-1])/2
            else:
                drho[i] = (rho[i+1]-rho[i-1])/2

        r = np.zeros(dim_zernike)
        t = np.zeros(dim_zernike)
        dr = np.zeros(dim_zernike)
        dt = np.zeros(dim_zernike)

        i = 0
        for m in range(M+1):
            dtheta = 2*np.pi/(a*m+1)
            theta = np.arange(0, 2*np.pi, dtheta)
            for j in range(a*m+1):
                r[i] = rho[m]
                t[i] = theta[j]
                dr[i] = drho[m]
                dt[i] = dtheta
                i += 1

        dz = 2*np.pi/(NFP*dim_fourier)
        z = np.arange(0, 2*np.pi/NFP, dz)

        r = np.tile(r, dim_fourier)
        t = np.tile(t, dim_fourier)
        z = np.tile(z[np.newaxis], (dim_zernike, 1)).flatten(order='F')
        dr = np.tile(dr, dim_fourier)
        dt = np.tile(dt, dim_fourier)
        dz = np.ones_like(z)*dz

        nodes = np.stack([r, t, z]).T
        volumes = np.stack([dr, dt, dz]).T

        return nodes, volumes

    def change_resolution(self, M:int, N:int) -> None:
        """

        Parameters
        ----------
        M : int
            new poloidal grid resolution
        N : int
            new toroidal grid resolution

        Returns
        -------
        None

        """
        if M != self._Grid__M or N != self._Grid__N:
            self._Grid__L = M+1
            self._Grid__M = M
            self._Grid__N = N
            self._Grid__nodes, self._Grid__volumes = self.create_nodes(M=M, N=N,
                         NFP=self._Grid__NFP, sym=self._Grid__sym, surfs=self.__surfs)
            self.sort_nodes()


# these functions are currently unused ---------------------------------------

# TODO: finish option for placing nodes at irrational surfaces

def dec_to_cf(x, dmax=6):
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
        return cf[0] + 1/cf_to_dec(cf[1:])


def most_rational(a, b):
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
    if np.sign(a*b) <= 0:
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
            return dec*s
        f += 1
