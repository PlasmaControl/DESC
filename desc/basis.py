import numpy as np
import functools
import numba
from abc import ABC, abstractmethod

from desc.backend import jnp, jit, sign, fori_loop, flatten_list, factorial, equals, Tristate


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
    sym : Tristate
        True for cos(m*t-n*z) symmetry, False for sin(m*t-n*z) symmetry,
        None for no symmetry (Default)
    modes : ndarray of int, shape(Nmodes,3)
        array of mode numbers [l,m,n]
        each row is one basis function with modes (l,m,n)

    """

    @abstractmethod
    def __init__(self) -> None:
        pass

    def __eq__(self, other) -> bool:
        """Overloads the == operator

        Parameters
        ----------
        other : Basis
            another Basis object to compare to

        Returns
        -------
        bool
            True if other is a Basis with the same attributes as self
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
        if self.__sym == True:     # cos(m*t-n*z) symmetry
            non_sym_idx = np.where(sign(self.__modes[:, 1]) !=
                                   sign(self.__modes[:, 2]))
            self.__modes = np.delete(self.__modes, non_sym_idx, axis=0)
        elif self.__sym == False:  # sin(m*t-n*z) symmetry
            non_sym_idx = np.where(sign(self.__modes[:, 1]) ==
                                   sign(self.__modes[:, 2]))
            self.__modes = np.delete(self.__modes, non_sym_idx, axis=0)

    def _sort_modes_(self) -> None:
        """Sorts modes for use with FFT

        Returns
        -------
        None

        """
        sort_idx = np.lexsort((self.__modes[:, 0], self.__modes[:, 1],
                               self.__modes[:, 2]))
        self.__modes = self.__modes[sort_idx]

    def _def_save_attrs_(self) -> None:
        """Defines attributes to save

        Returns
        -------
        None

        """
        self._save_attrs_ = ['_Basis__L', '_Basis__M', '_Basis__N', '_Basis__NFP',
                             '_Basis__modes']

    @abstractmethod
    def get_modes(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

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
    def sym(self) -> Tristate:
        return self.__sym

    @property
    def modes(self):
        return self.__modes

    @modes.setter
    def modes(self, modes) -> None:
        self.__modes = modes

    @property
    def num_modes(self) -> int:
        return self.__modes.shape[0]


class PowerSeries(Basis):
    """1D basis set for flux surface quantities.
       Power series in the radial coordinate.
    """

    def __init__(self, L:int=0) -> None:
        """Initializes a PowerSeries

        Parameters
        ----------
        L : int
            maximum radial resolution

        Returns
        -------
        None

        """
        self._Basis__L = L
        self._Basis__M = 0
        self._Basis__N = 0
        self._Basis__NFP = 1
        self._Basis__sym = None

        self._Basis__modes = self.get_modes(L=self._Basis__L)

        self._enforce_symmetry_()
        self._sort_modes_()
        self._def_save_attrs_()

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
        return powers(nodes[:, 0], self._Basis__modes[:, 0], dr=derivatives[0])

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

    def __init__(self, M:int=0, N:int=0, NFP:int=1, sym:Tristate=None) -> None:
        """Initializes a DoubleFourierSeries

        Parameters
        ----------
        M : int
            maximum poloidal resolution
        N : int
            maximum toroidal resolution
        NFP : int
            number of field periods
        sym : Tristate
            True for cos(m*t-n*z) symmetry, False for sin(m*t-n*z) symmetry,
            None for no symmetry (Default)

        Returns
        -------
        None

        """
        self._Basis__L = 0
        self._Basis__M = M
        self._Basis__N = N
        self._Basis__NFP = NFP
        self._Basis__sym = sym

        self._Basis__modes = self.get_modes(M=self._Basis__M, N=self._Basis__N)

        self._enforce_symmetry_()
        self._sort_modes_()
        self._def_save_attrs_()

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
        poloidal = fourier(nodes[:, 1], self._Basis__modes[:, 1], dt=derivatives[1])
        toroidal = fourier(nodes[:, 2], self._Basis__modes[:, 2], NFP=self._Basis__NFP, dt=derivatives[2])
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
                 sym:Tristate=None, index:str='ansi') -> None:
        """Initializes a FourierZernikeBasis

        Parameters
        ----------
        L : int
            maximum radial resolution
        M : int
            maximum poloidal resolution
        N : int
            maximum toroidal resolution
        NFP : int
            number of field periods
        sym : Tristate
            True for cos(m*t-n*z) symmetry, False for sin(m*t-n*z) symmetry,
            None for no symmetry (Default)
        index : str
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
        self._Basis__sym = sym
        self.__index = index

        self._Basis__modes = self.get_modes(L=self._Basis__L, M=self._Basis__M,
                                        N=self._Basis__N, index=self.__index)

        self._enforce_symmetry_()
        self._sort_modes_()
        self._def_save_attrs_()

    def get_modes(self, L:int=-1, M:int=0, N:int=0, index:str='ansi'):
        """Gets mode numbers for Fourier-Zernike basis functions

        Parameters
        ----------
        L : int
            maximum radial resolution
        M : int
            maximum poloidal resolution
        N : int
            maximum toroidal resolution
        index : str
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
        L = L if L >= 0 else default_L[index]

        if index == 'ansi':
            pol_posm = [[(m+d, m) for m in range(0, M+1) if m+d < M+1]
                        for d in range(0, L+1, 2)]

        elif index == 'fringe':
            pol_posm = [[(m+d//2, m-d//2) for m in range(0, M+1) if m-d//2 >= 0]
                        for d in range(0, L+1, 2)]

        elif index == 'chevron':
            pol_posm = [(m+d, m) for m in range(0, M+1)
                        for d in range(0, L+1, 2)]

        elif index == 'house':
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
        radial = jacobi(nodes[:, 0], self._Basis__modes[:, 0], self._Basis__modes[:, 1], dr=derivatives[0])
        poloidal = fourier(nodes[:, 1], self._Basis__modes[:, 1], dt=derivatives[1])
        toroidal = fourier(nodes[:, 2], self._Basis__modes[:, 2], NFP=self._Basis__NFP, dt=derivatives[2])
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


@functools.partial(jit)
def polyder_vec(p, m):
    """Vectorized version of polyder for differentiating multiple polynomials of the same degree

    Parameters
    ----------
    p : ndarray, shape(N,M)
        polynomial coefficients. Each row is 1 polynomial, in descending powers of x,
        each column is a power of x
    m : int >=0
        order of derivative

    Returns
    -------
    der : ndarray, shape(N,M)
        polynomial coefficients for derivative in descending order

    """
    m = jnp.asarray(m, dtype=int)  # order of derivative
    p = jnp.atleast_2d(p)
    n = p.shape[1] - 1             # order of polynomials

    D = jnp.arange(n, -1, -1)
    D = factorial(D) / factorial(D-m)

    p = jnp.roll(D*p, m, axis=1)
    idx = jnp.arange(p.shape[1])
    p = jnp.where(idx < m, 0, p)

    return p


@functools.partial(jit)
def polyval_vec(p, x):
    """Evaluate a polynomial at specific values,
    vectorized for evaluating multiple polynomials of the same degree.

    Parameters
    ----------
    p : ndarray, shape(N,M)
        Array of coefficient for N polynomials of order M. 
        Each row is one polynomial, given in descending powers of x. 
    x : ndarray, shape(K,)
        A number, or 1d array of numbers at
        which to evaluate p. If greater than 1d it is flattened.

    Returns
    -------
    y : ndarray, shape(N,K)
        polynomials evaluated at x.
        Each row corresponds to a polynomial, each column to a value of x

    Notes:
        Horner's scheme is used to evaluate the polynomial. Even so,
        for polynomials of high degree the values may be inaccurate due to
        rounding errors. Use carefully.

    """
    p = jnp.atleast_2d(p)
    x = jnp.atleast_1d(x).flatten()
    npoly = p.shape[0]  # number of polynomials
    order = p.shape[1]  # order of polynomials
    nx = len(x)         # number of coordinates
    y = jnp.zeros((npoly, nx))

    def body_fun(k, y):
        return y*x + np.atleast_2d(p[:, k]).T

    return fori_loop(0, order, body_fun, y)


@numba.jit(forceobj=True)
def power_coeffs(l):
    """Power series

    Parameters
    ----------
    l : ndarray of int, shape(K,)
        radial mode number(s)

    Returns
    -------
    coeffsy : ndarray, shape(l+1,)
        

    """
    l = np.atleast_1d(l).astype(int)
    npoly = len(l)      # number of polynomials
    order = np.max(l)   # order of polynomials
    coeffs = np.zeros((npoly, order+1))
    coeffs[range(npoly), l] = 1
    return coeffs


@functools.partial(jit, static_argnums=(1))
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
        basis function(s) evaluated at specified points

    """
    coeffs = power_coeffs(l)
    coeffs = polyder_vec(jnp.fliplr(coeffs), dr)
    return polyval_vec(coeffs, rho).T


@numba.jit(forceobj=True)
def jacobi_coeffs(l, m):
    """Jacobi polynomials

    Parameters
    ----------
    l : ndarray of int, shape(K,)
        radial mode number(s)
    m : ndarray of int, shape(K,)
        azimuthal mode number(s)

    Returns
    -------
    coeffs : ndarray
        

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
    return np.fliplr(np.where(lm_even, coeffs, 0))


@functools.partial(jit, static_argnums=(1, 2))
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
        basis function(s) evaluated at specified points

    """
    coeffs = jacobi_coeffs(l, m)
    coeffs = polyder_vec(coeffs, dr)
    return polyval_vec(coeffs, rho).T


@functools.partial(jit, static_argnums=(1))
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
        basis function(s) evaluated at specified points

    """
    theta_2d = jnp.atleast_2d(theta).T
    m_2d = jnp.atleast_2d(m)
    m_pos = (m_2d >= 0).astype(int)
    m_neg = (m_2d < 0).astype(int)
    m_abs = jnp.abs(m_2d)*NFP
    if dt == 0:
        return m_pos*np.cos(m_abs*theta_2d) + m_neg*np.sin(m_abs*theta_2d)
    else:
        return m_abs*(m_neg-m_pos)*fourier(theta, -m, NFP=NFP, dt=dt-1)

    """
    theta = jnp.atleast_1d(theta)[:, jnp.newaxis]
    m = jnp.atleast_1d(m)[jnp.newaxis]
    m_pos = (m >= 0)
    m_neg = (m < 0)
    m_abs = jnp.abs(m)
    der = (1j*m_abs*NFP)**dt
    exp = der*jnp.exp(1j*m_abs*NFP*theta)
    return m_pos*jnp.real(exp) + m_neg*jnp.imag(exp)
    """
