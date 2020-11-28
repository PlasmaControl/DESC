import numpy as np
import functools
import warnings
import numba
from abc import ABC, abstractmethod

from desc.backend import jnp, conditional_decorator, jit, use_jax, fori_loop
from desc.backend import polyder_vec, polyval_vec, flatten_list


class Basis(ABC):
    """Basis is an abstract base class for spectral basis sets

    Attributes
    ----------
    L : int
        maximum radial resolution
    M : int
        maximum poloidal resolution
    N : int
        maximum toroidal resolution
    NFP : int
        number of field periods
    modes : ndarray of int, shape(Nmodes,3)
        array of mode numbers [l,m,n]
        each row is one basis function with modes (l,m,n)

    """

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_modes(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def change_resolution(self) -> None:
        pass

    def sort_modes(self) -> None:
        """Sorts modes for use with FFT

        Returns
        -------
        None

        """
        sort_idx = np.lexsort((self.__modes[:, 1], self.__modes[:, 0],
                               self.__modes[:, 2]))
        self.__modes = self.__modes[sort_idx]

    @property
    def L(self):
        return self.__L

    @property
    def M(self):
        return self.__M

    @property
    def N(self):
        return self.__N

    @property
    def NFP(self):
        return self.__NFP

    @property
    def modes(self):
        return self.__modes

    @modes.setter
    def modes(self, modes):
        self.__modes = modes


class PowerSeries(Basis):
    """1D basis set for flux surface quantities.
       Power series in the radial coordinate.
    """

    def __init__(self, L:int=0) -> None:
        """Initializes a PowerSeries

        Returns
        -------
        None

        """
        self._Basis__L = L
        self._Basis__M = 0
        self._Basis__N = 0
        self._Basis__NFP = 1

        self._Basis__modes = self.get_modes(L=self._Basis__L)
        self.sort_modes()

    def get_modes(self, L:int=0):
        """Gets mode numbers for power series

        Parameters
        ----------
        L : int
            maximum radial resolution

        Returns
        -------
        modes : ndarray of int, shape(Nmodes,3)
            array of mode numbers [l,m,n]
            each row is one basis function with modes (l,m,n)

        """
        return np.array([[l, 0, 0] for l in range(L+1)])

    def evaluate(self, nodes, derivatives=np.array([0, 0, 0])):
        """Evaluates basis functions at specified nodes

        Parameters
        ----------
        nodes : ndarray of float, size(3,N)
            node coordinates, in (rho,theta,zeta)
        derivatives : ndarray of int, shape(3,)
            order of derivatives to compute in (rho,theta,zeta)

        Returns
        -------
        y : ndarray, shape(N,K)
            basis functions evaluated at nodes

        """
        return powers(nodes[0, :], self._Basis__modes[:, 0], dr=derivatives[0])

    def change_resolution(self, L:int) -> None:
        """

        Parameters
        ----------
        L : int
            maximum radial resolution

        Returns
        -------
        None

        """
        if L != self._Basis__L:
            self._Basis__L = L
            self._Basis__modes = self.get_modes(self._Basis__L)
            self.sort_nodes()


class DoubleFourierSeries(Basis):
    """2D basis set for use on a single flux surface.
       Fourier series in both the poloidal and toroidal coordinates.
    """

    def __init__(self, M:int=0, N:int=0, NFP:int=1) -> None:
        """Initializes a DoubleFourierSeries

        Returns
        -------
        None

        """
        self._Basis__L = 0
        self._Basis__M = M
        self._Basis__N = N
        self._Basis__NFP = NFP

        self._Basis__modes = self.get_modes(M=self._Basis__M, N=self._Basis__N)
        self.sort_modes()

    def get_modes(self, M:int=0, N:int=0) -> None:
        """Gets mode numbers for double fourier series

        Parameters
        ----------
        M : int
            maximum poloidal resolution
        N : int
            maximum toroidal resolution

        Returns
        -------
        modes : ndarray of int, shape(Nmodes,3)
            array of mode numbers [l,m,n]
            each row is one basis function with modes (l,m,n)

        """
        dim_pol = 2*M+1
        dim_tor = 2*N+1
        return np.array([[0, m-M, n-N] for m in range(dim_pol) for n in range(dim_tor)])

    def evaluate(self, nodes, derivatives=np.array([0, 0, 0])):
        """Evaluates basis functions at specified nodes

        Parameters
        ----------
        nodes : ndarray of float, size(3,N)
            node coordinates, in (rho,theta,zeta)
        derivatives : ndarray of int, shape(3,)
            order of derivatives to compute in (rho,theta,zeta)

        Returns
        -------
        y : ndarray, shape(N,K)
            basis functions evaluated at nodes

        """
        poloidal = fourier(nodes[1, :], self._Basis__modes[:, 1], dt=derivatives[1])
        toroidal = fourier(nodes[2, :], self._Basis__modes[:, 2], NFP=self._Basis__NFP, dt=derivatives[2])
        return poloidal*toroidal

    def change_resolution(self, M:int, N:int) -> None:
        """

        Parameters
        ----------
        M : int
            maximum poloidal resolution
        N : int
            maximum toroidal resolution

        Returns
        -------
        None

        """
        if M != self._Basis__M or N != self._Basis__N:
            self._Basis__M = M
            self._Basis__N = N
            self._Basis__modes = self.get_modes(self._Basis__M, self._Basis__N)
            self.sort_nodes()


class FourierZernikeBasis(Basis):
    """3D basis set for analytic functions in a toroidal volume.
       Zernike polynomials in the radial & poloidal coordinates, and a Fourier
       series in the toroidal coordinate.
    """

    def __init__(self, L:int=-1, M:int=0, N:int=0, NFP:int=1,
                 indexing:str='ansi') -> None:
        """Initializes a FourierZernikeBasis

        Parameters
        ----------
        L : int
            maximum radial resolution
        M : int
            maximum poloidal resolution
        N : int
            maximum toroidal resolution
        indexing : str
            Indexing method, one of the following options: 
            ('ansi','frige','chevron','house').
            For L=0, all methods are equivalent and give a "chevron" shaped
            basis (only the outer edge of the zernike pyramid of width M).
            For L>0, the indexing scheme defines order of the basis functions:
            ``'ansi'``: ANSI indexing fills in the pyramid with triangles of
            decreasing size, ending in a triagle shape. The maximum L is M,
            at which point the traditional ANSI indexing is recovered.
            Gives a single mode at m=M, and multiple modes at l=L, from m=0 to m=l.
            Total number of modes = (M-(L//2)+1)*((L//2)+1)
            ``'fringe'``: Fringe indexing fills in the pyramid with chevrons of
            decreasing size, ending in a diamond shape. The maximum L is 2*M,
            for which the traditional fringe/U of Arizona indexing is recovered.
            Gives a single mode at m=M and a single mode at l=L and m=0.
            Total number of modes = (M+1)*(M+2)/2 - (M-L//2+1)*(M-L//2)/2
            ``'chevron'``: Beginning from the initial chevron of width M,
            increasing L adds additional chevrons of the same width.
            Similar to "house" but with fewer modes with high l and low m.
            Total number of modes = (M+1)*(2*(L//2)+1)
            ``'house'``: Fills in the pyramid row by row, with a maximum
            horizontal width of M and a maximum radial resolution of L.
            For L=M, it is equivalent to ANSI, while for L>M it takes on a
            "house" like shape. Gives multiple modes at m=M and l=L.
            (Default value = 'ansi')

        Returns
        -------
        modes : ndarray of int, shape(Nmodes,3)
            array of mode numbers [l,m,n]
            each row is one basis function with modes (l,m,n)

        """
        self._Basis__L = L
        self._Basis__M = M
        self._Basis__N = N
        self._Basis__NFP = NFP
        self.__indexing = indexing

        self._Basis__modes = self.get_modes(L=self._Basis__L, M=self._Basis__M, N=self._Basis__N,
                                      indexing=self.__indexing)
        self.sort_modes()

    def get_modes(self, L:int=-1, M:int=0, N:int=0, indexing:str='ansi'):
        """Gets mode numbers for Fourier-Zernike basis functions

        Parameters
        ----------
        L : int
            maximum radial resolution
        M : int
            maximum poloidal resolution
        N : int
            maximum toroidal resolution
        indexing : str
            Indexing method, one of the following options: 
            ('ansi','frige','chevron','house').
            For L=0, all methods are equivalent and give a "chevron" shaped
            basis (only the outer edge of the zernike pyramid of width M).
            For L>0, the indexing scheme defines order of the basis functions:
            ``'ansi'``: ANSI indexing fills in the pyramid with triangles of
            decreasing size, ending in a triagle shape. The maximum L is M,
            at which point the traditional ANSI indexing is recovered.
            Gives a single mode at m=M, and multiple modes at l=L, from m=0 to m=l.
            Total number of modes = (M-(L//2)+1)*((L//2)+1)
            ``'fringe'``: Fringe indexing fills in the pyramid with chevrons of
            decreasing size, ending in a diamond shape. The maximum L is 2*M,
            for which the traditional fringe/U of Arizona indexing is recovered.
            Gives a single mode at m=M and a single mode at l=L and m=0.
            Total number of modes = (M+1)*(M+2)/2 - (M-L//2+1)*(M-L//2)/2
            ``'chevron'``: Beginning from the initial chevron of width M,
            increasing L adds additional chevrons of the same width.
            Similar to "house" but with fewer modes with high l and low m.
            Total number of modes = (M+1)*(2*(L//2)+1)
            ``'house'``: Fills in the pyramid row by row, with a maximum
            horizontal width of M and a maximum radial resolution of L.
            For L=M, it is equivalent to ANSI, while for L>M it takes on a
            "house" like shape. Gives multiple modes at m=M and l=L.
            (Default value = 'ansi')

        Returns
        -------
        modes : ndarray of int, shape(Nmodes,3)
            array of mode numbers [l,m,n]
            each row is one basis function with modes (l,m,n)

        """
        default_L = {'ansi': M,
                     'fringe': 2*M,
                     'chevron': M,
                     'house': 2*M}
        L = L if L >= 0 else default_L[indexing]

        if indexing == 'ansi':
            pol_posm = [[(m+d, m) for m in range(0, M+1) if m+d < M+1]
                        for d in range(0, L+1, 2)]

        elif indexing == 'fringe':
            pol_posm = [[(m+d//2, m-d//2) for m in range(0, M+1) if m-d//2 >= 0]
                        for d in range(0, L+1, 2)]

        elif indexing == 'chevron':
            pol_posm = [(m+d, m) for m in range(0, M+1)
                        for d in range(0, L+1, 2)]

        elif indexing == 'house':
            pol_posm = [[(l, m) for m in range(0, M+1) if l >= m and (l-m) % 2 == 0]
                        for l in range(0, L+1)] + [(m, m) for m in range(M+1)]
            pol_posm = list(dict.fromkeys(flatten_list(pol_posm)))

        pol = [[(l, m), (l, -m)] if m != 0 else [(l, m)]
               for l, m in flatten_list(pol_posm)]
        pol = np.array(flatten_list(pol))
        num_pol = len(pol)

        pol = np.tile(pol, (2*N+1, 1))
        tor = np.atleast_2d(
            np.tile(np.arange(-N, N+1), (num_pol, 1)).flatten(order='f')).T
        return np.hstack([pol, tor])

    def evaluate(self, nodes, derivatives=np.array([0, 0, 0])):
        """Evaluates basis functions at specified nodes

        Parameters
        ----------
        nodes : ndarray of float, size(3,N)
            node coordinates, in (rho,theta,zeta)
        derivatives : ndarray of int, shape(3,)
            order of derivatives to compute in (rho,theta,zeta)

        Returns
        -------
        y : ndarray, shape(N,K)
            basis functions evaluated at nodes

        """
        radial = jacobi(nodes[0, :], self._Basis__modes[:, 0], self._Basis__modes[:, 1], dr=derivatives[0])
        poloidal = fourier(nodes[1, :], self._Basis__modes[:, 1], dt=derivatives[1])
        toroidal = fourier(nodes[2, :], self._Basis__modes[:, 2], NFP=self._Basis__NFP, dt=derivatives[2])
        return radial*poloidal*toroidal

    def change_resolution(self, M:int, N:int, delta_lm:int) -> None:
        """

        Parameters
        ----------
        M : int
            maximum poloidal resolution
        N : int
            maximum toroidal resolution
        delta_lm : int
            maximum difference between poloidal and radial resolution (l-m).
            If < 0, defaults to ``M`` for 'ansi' or 'chevron' indexing, and
            ``2*M`` for 'fringe' or 'house'. Unused for 'fourier' indexing.

        Returns
        -------
        None

        """
        if M != self._Basis__M or N != self._Basis__N or delta_lm != self.__delta_lm:
            self._Basis__M = M
            self._Basis__N = N
            self.__delta_lm = delta_lm
            self._Basis__modes = self.get_modes(self._Basis__M, self._Basis__N,
                          delta_lm=self.__delta_lm, indexing=self.__indexing)
            self.sort_nodes()


def powers(rho, l, dr=0):
    """Power series

    Parameters
    ----------
    rho : ndarray, shape(N,)
        radial coordiantes to evaluate basis
    l : ndarray of int, shape(K,)
        radial mode number(s)
    dr : int
        order of derivative (Default = 0)

    Returns
    -------
    y : ndarray, shape(N,K)
        basis function evaluated at specified points

    """
    l = np.atleast_1d(l).astype(int)
    npoly = len(l)
    lmax = np.max(l)
    coeffs = polyder_vec(np.ones((npoly, lmax+1)), dr)
    y = polyval_vec(coeffs, rho).T
    return y


# use numba because array size depends on inputs, which jax cannot handle
@numba.jit(forceobj=True)
@conditional_decorator(functools.partial(jit, static_argnums=(1, 2)), use_jax)
def jacobi(rho, l, m, dr=0):
    """Jacobi polynomials

    Parameters
    ----------
    rho : ndarray, shape(N,)
        radial coordiantes to evaluate basis
    l : ndarray of int, shape(K,)
        radial mode number(s)
    m : ndarray of int, shape(K,)
        azimuthal mode number(s)
    dr : int
        order of derivative (Default = 0)

    Returns
    -------
    y : ndarray, shape(N,K)
        basis function evaluated at specified points

    """
    factorial = np.math.factorial
    l = np.atleast_1d(l).astype(int)
    m = np.atleast_1d(np.abs(m)).astype(int)
    npoly = len(l)
    lmax = np.max(l)
    coeffs = np.zeros((npoly, lmax+1))
    lm_even = ((l-m) % 2 == 0)[:, np.newaxis]
    for ii in range(npoly):
        ll = l[ii]
        mm = m[ii]
        for s in range(mm, ll+1, 2):
            coeffs[ii, s] = (-1)**((ll-s)/2)*factorial((ll+s)/2)/(
                factorial((ll-s)/2)*factorial((s+mm)/2)*factorial((s-mm)/2))
    coeffs = np.fliplr(np.where(lm_even, coeffs, 0))
    coeffs = polyder_vec(coeffs, dr)
    y = polyval_vec(coeffs, rho).T
    return y


@conditional_decorator(functools.partial(jit, static_argnums=(1, 2)), use_jax)
def fourier(theta, m, NFP=1, dt=0):
    """Fourier series

    Parameters
    ----------
    theta : ndarray, shape(N,)
        poloidal/toroidal coordinates to evaluate basis
    m : ndarray of int, shape(K,)
        poloidal/toroidal mode number(s)
    NFP : int
        number of field periods (Default = 1)
    dt : int
        order of derivative (Default = 0)

    Returns
    -------
    y : ndarray, shape(N,K)
        basis functions evaluated at specified points

    """
    theta = jnp.atleast_1d(theta)[:, jnp.newaxis]
    m = jnp.atleast_1d(m)[jnp.newaxis]
    m_pos = (m >= 0)
    m_neg = (m < 0)
    m_abs = jnp.abs(m)
    der = (1j*m_abs*NFP)**dt
    exp = der*jnp.exp(1j*m_abs*NFP*theta)
    y = m_pos*jnp.real(exp) + m_neg*jnp.imag(exp)
    return y
