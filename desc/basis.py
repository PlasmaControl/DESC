import numpy as np
import mpmath
from abc import ABC, abstractmethod
from math import factorial
from desc.utils import sign, flatten_list
from desc.io import IOAble

__all__ = [
    "PowerSeries",
    "FourierSeries",
    "DoubleFourierSeries",
    "ZernikePolynomial",
    "FourierZernikeBasis",
]


class Basis(IOAble, ABC):
    """Basis is an abstract base class for spectral basis sets"""

    _io_attrs_ = ["_L", "_M", "_N", "_NFP", "_modes", "_sym", "_spectral_indexing"]

    def _enforce_symmetry(self):
        """Enforces stellarator symmetry"""

        if self.sym in ["cos", "cosine"]:  # cos(m*t-n*z) symmetry
            non_sym_idx = np.where(sign(self.modes[:, 1]) != sign(self.modes[:, 2]))
            self._modes = np.delete(self.modes, non_sym_idx, axis=0)
        elif self.sym in ["sin", "sine"]:  # sin(m*t-n*z) symmetry
            non_sym_idx = np.where(sign(self.modes[:, 1]) == sign(self.modes[:, 2]))
            self._modes = np.delete(self.modes, non_sym_idx, axis=0)
        elif self.sym is not False:
            raise ValueError(f"Unknown symmetry type {self.sym}")

    def _sort_modes(self):
        """Sorts modes for use with FFT"""

        sort_idx = np.lexsort((self.modes[:, 1], self.modes[:, 0], self.modes[:, 2]))
        self._modes = self.modes[sort_idx]

    def get_idx(self, L=0, M=0, N=0):
        """Get the index into the ``'modes'`` array corresponding to a given mode number

        Parameters
        ----------
        L : int or ndarray of int
            radial mode number
        M : int or ndarray of int
            poliodal mode number
        N : int or ndarray of int
            toroidal mode number

        Returns
        -------
        idx : ndarray of int
            indices of given mode numbers
        """
        L = np.atleast_1d(L)
        M = np.atleast_1d(M)
        N = np.atleast_1d(N)

        num = max(len(L), len(M), len(N))
        L = np.broadcast_to(L, num)
        M = np.broadcast_to(M, num)
        N = np.broadcast_to(N, num)

        idx = np.array(
            [
                np.where(
                    np.logical_and(
                        np.logical_and(l == self.modes[:, 0], m == self.modes[:, 1]),
                        n == self.modes[:, 2],
                    )
                )[0]
                for l, m, n in zip(L, M, N)
            ]
        )
        return idx

    @abstractmethod
    def _get_modes(self):
        """ndarray: the modes numbers for the basis"""

    @abstractmethod
    def evaluate(self, nodes, derivatives=np.array([0, 0, 0]), modes=None):
        """Evaluates basis functions at specified nodes

        Parameters
        ----------
        nodes : ndarray of float, size(num_nodes,3)
            node coordinates, in (rho,theta,zeta)
        derivatives : ndarray of int, shape(3,)
            order of derivatives to compute in (rho,theta,zeta)
        modes : ndarray of in, shape(num_modes,3), optional
            basis modes to evaluate (if None, full basis is used)

        Returns
        -------
        y : ndarray, shape(num_nodes,num_modes)
            basis functions evaluated at nodes

        """

    @abstractmethod
    def change_resolution(self):
        """Change resolution of the basis to the given resolutions"""

    @property
    def L(self):
        """int: maximum radial resolution"""
        if not hasattr(self, "_L"):
            self._L = 0
        return self._L

    @property
    def M(self):
        """int:  maximum poloidal resolution"""
        if not hasattr(self, "_M"):
            self._M = 0
        return self._M

    @property
    def N(self):
        """int: maximum toroidal resolution"""
        if not hasattr(self, "_N"):
            self._N = 0
        return self._N

    @property
    def NFP(self):
        """int: number of field periods"""
        if not hasattr(self, "_NFP"):
            self._NFP = 1
        return self._NFP

    @property
    def sym(self):
        """str: {``'cos'``, ``'sin'``, ``False``} type of symmetry"""
        if not hasattr(self, "_sym"):
            self._sym = False
        return self._sym

    @property
    def modes(self):
        """ndarray: mode numbers [l,m,n]"""
        if not hasattr(self, "_modes"):
            self._modes = np.array([]).reshape((0, 3))
        return self._modes

    @modes.setter
    def modes(self, modes):
        self._modes = modes

    @property
    def num_modes(self):
        """int: number of modes in the spectral basis"""
        return self.modes.shape[0]

    @property
    def spectral_indexing(self):
        """str: type of indexing used for the spectral basis"""
        if not hasattr(self, "_spectral_indexing"):
            self._spectral_indexing = "linear"
        return self._spectral_indexing

    def __repr__(self):
        """string form of the object"""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (L={}, M={}, N={}, NFP={}, sym={}, spectral_indexing={})".format(
                self.L, self.M, self.N, self.NFP, self.sym, self.spectral_indexing
            )
        )


class PowerSeries(Basis):
    """1D basis set for flux surface quantities.

    Power series in the radial coordinate.


    Parameters
    ---------
    L : int
        maximum radial resolution

    """

    def __init__(self, L):

        self._L = L
        self._M = 0
        self._N = 0
        self._NFP = 1
        self._sym = False
        self._spectral_indexing = "linear"

        self._modes = self._get_modes(L=self.L)

        self._enforce_symmetry()
        self._sort_modes()

    def _get_modes(self, L=0):
        """Gets mode numbers for power series

        Parameters
        ----------
        L : int
            maximum radial resolution

        Returns
        -------
        modes : ndarray of int, shape(num_modes,3)
            array of mode numbers [l,m,n]
            each row is one basis function with modes (l,m,n)

        """
        l = np.arange(L + 1).reshape((-1, 1))
        z = np.zeros((L + 1, 2))
        return np.hstack([l, z])

    def evaluate(self, nodes, derivatives=np.array([0, 0, 0]), modes=None):
        """Evaluates basis functions at specified nodes

        Parameters
        ----------
        nodes : ndarray of float, size(num_nodes,3)
            node coordinates, in (rho,theta,zeta)
        derivatives : ndarray of int, shape(num_derivatives,3)
            order of derivatives to compute in (rho,theta,zeta)
        modes : ndarray of in, shape(num_modes,3), optional
            basis modes to evaluate (if None, full basis is used)

        Returns
        -------
        y : ndarray, shape(num_nodes,num_modes)
            basis functions evaluated at nodes

        """
        if modes is None:
            modes = self.modes
        if not len(modes):
            return np.array([]).reshape((len(nodes), 0))
        return powers(nodes[:, 0], modes[:, 0], dr=derivatives[0])

    def change_resolution(self, L):
        """Change resolution of the basis to the given resolution.

        Parameters
        ----------
        L : int
            maximum radial resolution

        """
        if L != self.L:
            self._L = L
            self._modes = self._get_modes(self.L)
            self._sort_modes()


class FourierSeries(Basis):
    """1D basis set for use with the magnetic axis.
    Fourier series in the toroidal coordinate.

    Parameters
    ----------
    N : int
        maximum toroidal resolution
    NFP : int
        number of field periods
    sym : {``'cos'``, ``'sin'``, False}
        * ``'cos'`` for cos(m*t-n*z) symmetry
        * ``'sin'`` for sin(m*t-n*z) symmetry
        * ``False`` for no symmetry (Default)

    """

    def __init__(self, N, NFP=1, sym=False):

        self._L = 0
        self._M = 0
        self._N = N
        self._NFP = NFP
        self._sym = sym
        self._spectral_indexing = "linear"

        self._modes = self._get_modes(N=self.N)

        self._enforce_symmetry()
        self._sort_modes()

    def _get_modes(self, N=0):
        """Gets mode numbers for fourier series

        Parameters
        ----------
        N : int
            maximum toroidal resolution

        Returns
        -------
        modes : ndarray of int, shape(num_modes,3)
            array of mode numbers [l,m,n]
            each row is one basis function with modes (l,m,n)

        """
        dim_tor = 2 * N + 1
        n = np.arange(dim_tor).reshape((-1, 1)) - N
        z = np.zeros((dim_tor, 2))
        return np.hstack([z, n])

    def evaluate(self, nodes, derivatives=np.array([0, 0, 0]), modes=None):
        """Evaluates basis functions at specified nodes

        Parameters
        ----------
        nodes : ndarray of float, size(num_nodes,3)
            node coordinates, in (rho,theta,zeta)
        derivatives : ndarray of int, shape(num_derivatives,3)
            order of derivatives to compute in (rho,theta,zeta)
        modes : ndarray of in, shape(num_modes,3), optional
            basis modes to evaluate (if None, full basis is used)

        Returns
        -------
        y : ndarray, shape(num_nodes,num_modes)
            basis functions evaluated at nodes

        """
        if modes is None:
            modes = self.modes
        if not len(modes):
            return np.array([]).reshape((len(nodes), 0))

        return fourier(nodes[:, 2], modes[:, 2], NFP=self._NFP, dt=derivatives[2])

    def change_resolution(self, N):
        """Change resolution of the basis to the given resolutions.

        Parameters
        ----------
        N : int
            maximum toroidal resolution

        """
        if N != self.N:
            self._N = N
            self._modes = self._get_modes(self.N)
            self._enforce_symmetry()
            self._sort_modes()


class DoubleFourierSeries(Basis):
    """2D basis set for use on a single flux surface.
    Fourier series in both the poloidal and toroidal coordinates.

    Parameters
    ----------
    M : int
        maximum poloidal resolution
    N : int
        maximum toroidal resolution
    NFP : int
        number of field periods
    sym : {``'cos'``, ``'sin'``, ``False``}
        * ``'cos'`` for cos(m*t-n*z) symmetry
        * ``'sin'`` for sin(m*t-n*z) symmetry
        * ``False`` for no symmetry (Default)

    """

    def __init__(self, M, N, NFP=1, sym=False):

        self._L = 0
        self._M = M
        self._N = N
        self._NFP = NFP
        self._sym = sym
        self._spectral_indexing = "linear"

        self._modes = self._get_modes(M=self.M, N=self.N)

        self._enforce_symmetry()
        self._sort_modes()

    def _get_modes(self, M=0, N=0):
        """Gets mode numbers for double fourier series

        Parameters
        ----------
        M : int
            maximum poloidal resolution
        N : int
            maximum toroidal resolution

        Returns
        -------
        modes : ndarray of int, shape(num_modes,3)
            array of mode numbers [l,m,n]
            each row is one basis function with modes (l,m,n)

        """
        dim_pol = 2 * M + 1
        dim_tor = 2 * N + 1
        m = np.arange(dim_pol) - M
        n = np.arange(dim_tor) - N
        mm, nn = np.meshgrid(m, n)
        mm = mm.reshape((-1, 1), order="F")
        nn = nn.reshape((-1, 1), order="F")
        z = np.zeros_like(mm)
        y = np.hstack([z, mm, nn])
        return y

    def evaluate(self, nodes, derivatives=np.array([0, 0, 0]), modes=None):
        """Evaluates basis functions at specified nodes

        Parameters
        ----------
        nodes : ndarray of float, size(num_nodes,3)
            node coordinates, in (rho,theta,zeta)
        derivatives : ndarray of int, shape(num_derivatives,3)
            order of derivatives to compute in (rho,theta,zeta)
        modes : ndarray of in, shape(num_modes,3), optional
            basis modes to evaluate (if None, full basis is used)

        Returns
        -------
        y : ndarray, shape(num_nodes,num_modes)
            basis functions evaluated at nodes

        """
        if modes is None:
            modes = self.modes
        if not len(modes):
            return np.array([]).reshape((len(nodes), 0))

        poloidal = fourier(nodes[:, 1], modes[:, 1], dt=derivatives[1])
        toroidal = fourier(nodes[:, 2], modes[:, 2], NFP=self.NFP, dt=derivatives[2])
        return poloidal * toroidal

    def change_resolution(self, M, N):
        """Change resolution of the basis to the given resolutions.

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
        if M != self.M or N != self.N:
            self._M = M
            self._N = N
            self._modes = self._get_modes(self.M, self.N)
            self._enforce_symmetry()
            self._sort_modes()


class ZernikePolynomial(Basis):
    """2D basis set for analytic functions in a unit disc.

    Initializes a ZernikePolynomial

    Parameters
    ----------
    L : int
        maximum radial resolution. Use L=-1 for default based on M
    M : int
        maximum poloidal resolution
    sym : {``'cos'``, ``'sin'``, ``False``}
        * ``'cos'`` for cos(m*t-n*z) symmetry
        * ``'sin'`` for sin(m*t-n*z) symmetry
        * ``False`` for no symmetry (Default)
    spectral_indexing : {``'ansi'``, ``'fringe'``}
        Indexing method, default value = ``'fringe'``

        For L=0, all methods are equivalent and give a "chevron" shaped
        basis (only the outer edge of the zernike pyramid of width M).
        For L>0, the indexing scheme defines order of the basis functions:

        ``'ansi'``: ANSI indexing fills in the pyramid with triangles of
        decreasing size, ending in a triagle shape. For L == M,
        the traditional ANSI pyramid indexing is recovered. For L>M, adds rows
        to the bottom of the pyramid, increasing L while keeping M constant,
        giving a "house" shape

        ``'fringe'``: Fringe indexing fills in the pyramid with chevrons of
        decreasing size, ending in a diamond shape for L=2*M where
        the traditional fringe/U of Arizona indexing is recovered.
        For L > 2*M, adds chevrons to the bottom, making a hexagonal diamond

    """

    def __init__(self, L, M, sym=False, spectral_indexing="fringe"):

        self._L = L
        self._M = M
        self._N = 0
        self._NFP = 1
        self._sym = sym
        self._spectral_indexing = spectral_indexing

        self._modes = self._get_modes(
            L=self.L, M=self.M, spectral_indexing=self.spectral_indexing
        )

        self._enforce_symmetry()
        self._sort_modes()

    def _get_modes(self, L=-1, M=0, spectral_indexing="fringe"):
        """Gets mode numbers for Fourier-Zernike basis functions

        Parameters
        ----------
        L : int
            maximum radial resolution
        M : int
            maximum poloidal resolution
        spectral_indexing : {``'ansi'``, ``'fringe'``}
            Indexing method, default value = ``'fringe'``

            For L=0, all methods are equivalent and give a "chevron" shaped
            basis (only the outer edge of the zernike pyramid of width M).
            For L>0, the indexing scheme defines order of the basis functions:

            ``'ansi'``: ANSI indexing fills in the pyramid with triangles of
            decreasing size, ending in a triagle shape. For L == M,
            the traditional ANSI pyramid indexing is recovered. For L>M, adds rows
            to the bottom of the pyramid, increasing L while keeping M constant,
            giving a "house" shape

            ``'fringe'``: Fringe indexing fills in the pyramid with chevrons of
            decreasing size, ending in a diamond shape for L=2*M where
            the traditional fringe/U of Arizona indexing is recovered.
            For L > 2*M, adds chevrons to the bottom, making a hexagonal diamond

        Returns
        -------
        modes : ndarray of int, shape(num_modes,3)
            array of mode numbers [l,m,n]
            each row is one basis function with modes (l,m,n)

        """
        default_L = {"ansi": M, "fringe": 2 * M}
        L = L if L >= 0 else default_L.get(spectral_indexing, M)
        self._L = L

        if spectral_indexing == "ansi":
            pol_posm = [
                [(m + d, m) for m in range(0, M + 1) if m + d < M + 1]
                for d in range(0, L + 1, 2)
            ]
            if L > M:
                pol_posm += [
                    (l, m)
                    for l in range(M + 1, L + 1)
                    for m in range(0, M + 1)
                    if (l - m) % 2 == 0
                ]

        elif spectral_indexing == "fringe":
            pol_posm = [
                [(m + d // 2, m - d // 2) for m in range(0, M + 1) if m - d // 2 >= 0]
                for d in range(0, L + 1, 2)
            ]
            if L > 2 * M:
                Ladd = L - 2 * M
                pol_posm += [
                    [(l - m, m) for m in range(0, M + 1)]
                    for l in range(2 * M, L + 1, 2)
                ]

        else:
            raise ValueError("Unknown spectral_indexing: {}".format(spectral_indexing))

        pol = [
            [(l, m), (l, -m)] if m != 0 else [(l, m)] for l, m in flatten_list(pol_posm)
        ]
        pol = np.array(flatten_list(pol))
        num_pol = len(pol)
        tor = np.zeros((num_pol, 1))

        return np.hstack([pol, tor])

    def evaluate(self, nodes, derivatives=np.array([0, 0, 0]), modes=None):
        """Evaluates basis functions at specified nodes

        Parameters
        ----------
        nodes : ndarray of float, size(num_nodes,3)
            node coordinates, in (rho,theta,zeta)
        derivatives : ndarray of int, shape(num_derivatives,3)
            order of derivatives to compute in (rho,theta,zeta)
        modes : ndarray of int, shape(num_modes,3), optional
            basis modes to evaluate (if None, full basis is used)

        Returns
        -------
        y : ndarray, shape(num_nodes,num_modes)
            basis functions evaluated at nodes

        """
        if modes is None:
            modes = self.modes
        if not len(modes):
            return np.array([]).reshape((len(nodes), 0))

        radial = jacobi(nodes[:, 0], modes[:, 0], modes[:, 1], dr=derivatives[0])
        poloidal = fourier(nodes[:, 1], modes[:, 1], dt=derivatives[1])
        return radial * poloidal

    def change_resolution(self, L, M):
        """Change resolution of the basis to the given resolutions.

        Parameters
        ----------
        L : int
            maximum radial resolution
        M : int
            maximum poloidal resolution

        """
        if L != self.L or M != self.M:
            self._L = L
            self._M = M
            self._modes = self._get_modes(
                self.L, self.M, spectral_indexing=self.spectral_indexing
            )
            self._enforce_symmetry()
            self._sort_modes()


class FourierZernikeBasis(Basis):
    """3D basis set for analytic functions in a toroidal volume.
    Zernike polynomials in the radial & poloidal coordinates, and a Fourier
    series in the toroidal coordinate.

    Initializes a FourierZernikeBasis

    Parameters
    ----------
    L : int
        maximum radial resolution. Use L=-1 for default based on M
    M : int
        maximum poloidal resolution
    N : int
        maximum toroidal resolution
    NFP : int
        number of field periods
    sym : {``'cos'``, ``'sin'``, ``False``}
        * ``'cos'`` for cos(m*t-n*z) symmetry
        * ``'sin'`` for sin(m*t-n*z) symmetry
        * ``False`` for no symmetry (Default)
    spectral_indexing : {``'ansi'``, ``'fringe'``}
        Indexing method, default value = ``'fringe'``

        For L=0, all methods are equivalent and give a "chevron" shaped
        basis (only the outer edge of the zernike pyramid of width M).
        For L>0, the indexing scheme defines order of the basis functions:

        ``'ansi'``: ANSI indexing fills in the pyramid with triangles of
        decreasing size, ending in a triagle shape. For L == M,
        the traditional ANSI pyramid indexing is recovered. For L>M, adds rows
        to the bottom of the pyramid, increasing L while keeping M constant,
        giving a "house" shape

        ``'fringe'``: Fringe indexing fills in the pyramid with chevrons of
        decreasing size, ending in a diamond shape for L=2*M where
        the traditional fringe/U of Arizona indexing is recovered.
        For L > 2*M, adds chevrons to the bottom, making a hexagonal diamond

    """

    def __init__(self, L, M, N, NFP=1, sym=False, spectral_indexing="fringe"):

        self._L = L
        self._M = M
        self._N = N
        self._NFP = NFP
        self._sym = sym
        self._spectral_indexing = spectral_indexing

        self._modes = self._get_modes(
            L=self.L, M=self.M, N=self.N, spectral_indexing=self.spectral_indexing
        )

        self._enforce_symmetry()
        self._sort_modes()

    def _get_modes(self, L=-1, M=0, N=0, spectral_indexing="fringe"):
        """Gets mode numbers for Fourier-Zernike basis functions

        Parameters
        ----------
        L : int
            maximum radial resolution
        M : int
            maximum poloidal resolution
        N : int
            maximum toroidal resolution
        spectral_indexing : {``'ansi'``, ``'fringe'``}
            Indexing method, default value = ``'fringe'``

            For L=0, all methods are equivalent and give a "chevron" shaped
            basis (only the outer edge of the zernike pyramid of width M).
            For L>0, the indexing scheme defines order of the basis functions:

            ``'ansi'``: ANSI indexing fills in the pyramid with triangles of
            decreasing size, ending in a triagle shape. For L == M,
            the traditional ANSI pyramid indexing is recovered. For L>M, adds rows
            to the bottom of the pyramid, increasing L while keeping M constant,
            giving a "house" shape

            ``'fringe'``: Fringe indexing fills in the pyramid with chevrons of
            decreasing size, ending in a diamond shape for L=2*M where
            the traditional fringe/U of Arizona indexing is recovered.
            For L > 2*M, adds chevrons to the bottom, making a hexagonal diamond

        Returns
        -------
        modes : ndarray of int, shape(num_modes,3)
            array of mode numbers [l,m,n]
            each row is one basis function with modes (l,m,n)

        """
        default_L = {"ansi": M, "fringe": 2 * M}
        L = L if L >= 0 else default_L.get(spectral_indexing, M)
        self._L = L

        if spectral_indexing == "ansi":
            pol_posm = [
                [(m + d, m) for m in range(0, M + 1) if m + d < M + 1]
                for d in range(0, L + 1, 2)
            ]
            if L > M:
                pol_posm += [
                    (l, m)
                    for l in range(M + 1, L + 1)
                    for m in range(0, M + 1)
                    if (l - m) % 2 == 0
                ]

        elif spectral_indexing == "fringe":
            pol_posm = [
                [(m + d // 2, m - d // 2) for m in range(0, M + 1) if m - d // 2 >= 0]
                for d in range(0, L + 1, 2)
            ]
            if L > 2 * M:
                Ladd = L - 2 * M
                pol_posm += [
                    [(l - m, m) for m in range(0, M + 1)]
                    for l in range(2 * M, L + 1, 2)
                ]

        else:
            raise ValueError("Unknown spectral_indexing: {}".format(spectral_indexing))

        pol = [
            [(l, m), (l, -m)] if m != 0 else [(l, m)] for l, m in flatten_list(pol_posm)
        ]
        pol = np.array(flatten_list(pol))
        num_pol = len(pol)

        pol = np.tile(pol, (2 * N + 1, 1))
        tor = np.atleast_2d(
            np.tile(np.arange(-N, N + 1), (num_pol, 1)).flatten(order="f")
        ).T
        return np.unique(np.hstack([pol, tor]), axis=0)

    def evaluate(self, nodes, derivatives=np.array([0, 0, 0]), modes=None):
        """Evaluates basis functions at specified nodes

        Parameters
        ----------
        nodes : ndarray of float, size(num_nodes,3)
            node coordinates, in (rho,theta,zeta)
        derivatives : ndarray of int, shape(num_derivatives,3)
            order of derivatives to compute in (rho,theta,zeta)
        modes : ndarray of int, shape(num_modes,3), optional
            basis modes to evaluate (if None, full basis is used)

        Returns
        -------
        y : ndarray, shape(num_nodes,num_modes)
            basis functions evaluated at nodes

        """
        if modes is None:
            modes = self.modes
        if not len(modes):
            return np.array([]).reshape((len(nodes), 0))
        # TODO: avoid duplicate calculations when mixing derivatives

        radial = jacobi(nodes[:, 0], modes[:, 0], modes[:, 1], dr=derivatives[0])
        poloidal = fourier(nodes[:, 1], modes[:, 1], dt=derivatives[1])
        toroidal = fourier(nodes[:, 2], modes[:, 2], NFP=self.NFP, dt=derivatives[2])
        return radial * poloidal * toroidal

    def change_resolution(self, L, M, N):
        """Change resolution of the basis to the given resolutions.

        Parameters
        ----------
        L : int
            maximum radial resolution
        M : int
            maximum poloidal resolution
        N : int
            maximum toroidal resolution

        """
        if L != self.L or M != self.M or N != self.N:
            self._L = L
            self._M = M
            self._N = N
            self._modes = self._get_modes(
                self.L, self.M, self.N, spectral_indexing=self.spectral_indexing
            )
            self._enforce_symmetry()
            self._sort_modes()


def polyder_vec(p, m):
    """Vectorized version of polyder.

    For differentiating multiple polynomials of the same degree

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
    factorial = np.math.factorial
    m = np.asarray(m, dtype=int)  # order of derivative
    p = np.atleast_2d(p)
    # for modest to large arrays, faster to find unique values and
    # only evaluate those. Have to cast to float because np.unique
    # can't handle object types like python native int
    _, pidx, outidx = np.unique(
        p.astype(float), return_index=True, return_inverse=True, axis=0
    )
    p = p[pidx]
    order = p.shape[1] - 1

    D = np.arange(order, -1, -1)
    num = np.array([factorial(i) for i in D], dtype=object)
    den = np.array([factorial(max(i - m, 0)) for i in D], dtype=object)
    D = (num // den).astype(p.dtype)

    p = np.roll(D * p, m, axis=1)
    idx = np.arange(p.shape[1])
    p = np.where(idx < m, 0, p)
    p = p[outidx]

    return p


def polyval_vec(p, x, prec=None):
    """Evaluate a polynomial at specific values.

    Vectorized for evaluating multiple polynomials of the same degree.

    Parameters
    ----------
    p : ndarray, shape(N,M)
        Array of coefficient for N polynomials of order M.
        Each row is one polynomial, given in descending powers of x.
    x : ndarray, shape(K,)
        A number, or 1d array of numbers at
        which to evaluate p. If greater than 1d it is flattened.
    prec : int, optional
        precision to use, in number of decimal places. Default is
        double precision (~16 decimals) which should be enough for
        most cases with L <= 24

    Returns
    -------
    y : ndarray, shape(N,K)
        polynomials evaluated at x.
        Each row corresponds to a polynomial, each column to a value of x

    """
    p = np.atleast_2d(p)
    x = np.atleast_1d(x).flatten()
    # for modest to large arrays, faster to find unique values and
    # only evaluate those. Have to cast to float because np.unique
    # can't handle object types like python native int
    unq_x, xidx = np.unique(x, return_inverse=True)
    _, pidx, outidx = np.unique(
        p.astype(float), return_index=True, return_inverse=True, axis=0
    )
    unq_p = p[pidx]

    if prec is not None and prec > 16:
        # TODO: possibly multithread this bit
        mpmath.mp.dps = prec
        y = np.array([np.asarray(mpmath.polyval(list(pi), unq_x)) for pi in unq_p])
    else:
        npoly = unq_p.shape[0]  # number of polynomials
        order = unq_p.shape[1]  # order of polynomials
        nx = len(unq_x)  # number of coordinates
        y = np.zeros((npoly, nx))

        for k in range(order):
            y = y * unq_x + np.atleast_2d(unq_p[:, k]).T

    return y[outidx][:, xidx].astype(float)


def jacobi_coeffs(l, m, exact=True):
    """Jacobi polynomial coefficients.

    Parameters
    ----------
    l : ndarray of int, shape(K,)
        radial mode number(s)
    m : ndarray of int, shape(K,)
        azimuthal mode number(s)
    exact : bool
        whether to return exact coefficients with `object` dtype
        or return integer or floating point approximation

    Returns
    -------
    coeffs : ndarray


    Notes:
        integer representation is exact up to l~54, so
        leaving `exact` arg as False can speed up
        evaluation with no loss in accuracy

    """
    l = np.atleast_1d(l).astype(int)
    m = np.atleast_1d(np.abs(m)).astype(int)
    lm = np.vstack([l, m]).T
    # for modest to large arrays, faster to find unique values and
    # only evaluate those
    lms, idx = np.unique(lm, return_inverse=True, axis=0)

    npoly = len(lms)
    lmax = np.max(lms[:, 0])
    coeffs = np.zeros((npoly, lmax + 1), dtype=object)
    lm_even = ((lms[:, 0] - lms[:, 1]) % 2 == 0)[:, np.newaxis]
    for ii in range(npoly):
        ll = lms[ii, 0]
        mm = lms[ii, 1]
        for s in range(mm, ll + 1, 2):
            coeffs[ii, s] = (
                (-1) ** ((ll - s) // 2)
                * factorial((ll + s) // 2)
                // (
                    factorial((ll - s) // 2)
                    * factorial((s + mm) // 2)
                    * factorial((s - mm) // 2)
                )
            )
    c = np.fliplr(np.where(lm_even, coeffs, 0))
    if not exact:
        try:
            c = c.astype(int)
        except OverflowError:
            c = c.astype(float)
    c = c[idx]
    return c


def jacobi(rho, l, m, dr=0):
    """Jacobi polynomials.

    Parameters
    ----------
    rho : ndarray, shape(N,)
        radial coordinates to evaluate basis
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
    lmax = np.max(l)
    coeffs = polyder_vec(coeffs, dr)
    # this should give accuracy of ~1e-6 in the eval'd polynomials
    prec = int(0.4 * lmax + 5.4)
    return polyval_vec(coeffs, rho, prec=prec).T


def power_coeffs(l):
    """Power series coefficients.

    Parameters
    ----------
    l : ndarray of int, shape(K,)
        radial mode number(s)

    Returns
    -------
    coeffs : ndarray, shape(l+1,)

    """
    l = np.atleast_1d(l).astype(int)
    npoly = len(l)  # number of polynomials
    order = np.max(l)  # order of polynomials
    coeffs = np.zeros((npoly, order + 1))
    coeffs[range(npoly), l] = 1
    return coeffs


def powers(rho, l, dr=0):
    """Power series.

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
    coeffs = polyder_vec(np.fliplr(coeffs), dr)
    return polyval_vec(coeffs, rho).T


def fourier(theta, m, NFP=1, dt=0):
    """Fourier series.

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
    # for modest to large arrays, faster to find unique values and
    # only evaluate those
    unq_theta, tidx = np.unique(theta, return_inverse=True)
    unq_m, midx = np.unique(m, return_inverse=True)

    theta_2d = np.atleast_2d(unq_theta).T
    m_2d = np.atleast_2d(unq_m)
    m_pos = (m_2d >= 0).astype(int)
    m_neg = (m_2d < 0).astype(int)
    m_abs = np.abs(m_2d) * NFP
    if dt == 0:
        out = np.where(m_pos, np.cos(m_abs * theta_2d), np.sin(m_abs * theta_2d))
    else:
        out = m_abs * (m_neg - m_pos) * fourier(unq_theta, -unq_m, NFP=NFP, dt=dt - 1)
    # put duplicate values back in
    out = out[tidx][:, midx]
    return out


def zernike_norm(l, m):
    """Norm of a Zernike polynomial with l, m indexing.

    Parameters
    ----------
    l,m : int
        radial and azimuthal mode numbers.

    Returns
    -------
    norm : float
        the integral (Z^m_l)^2 r dr dt, r=[0,1], t=[0,2*pi]

    """
    return np.sqrt((2 * (l + 1)) / (np.pi * (1 + int(m == 0))))
