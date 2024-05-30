"""Classes for spectral bases and functions for evaluation."""

import functools
import warnings
from abc import ABC, abstractmethod
from math import factorial

import mpmath
import numpy as np
import skfem as fem
from matplotlib import pyplot as plt

from desc.backend import custom_jvp, fori_loop, gammaln, jit, jnp, sign
from desc.io import IOAble
from desc.utils import check_nonnegint, check_posint, flatten_list

__all__ = [
    "PowerSeries",
    "FourierSeries",
    "FiniteElementBasis",
    "DoubleFourierSeries",
    "ZernikePolynomial",
    "ChebyshevDoubleFourierBasis",
    "FourierZernikeBasis",
    "FiniteElementMesh1D",
    "FiniteElementMesh2D",
    "FiniteElementMesh3D",
    "ChebyshevPolynomial",
]


class _Basis(IOAble, ABC):
    """Basis is an abstract base class for spectral basis sets."""

    _io_attrs_ = [
        "_L",
        "_M",
        "_N",
        "_NFP",
        "_modes",
        "_sym",
        "_spectral_indexing",
    ]

    def __init__(self):
        self._enforce_symmetry()
        self._sort_modes()
        self._create_idx()
        # ensure things that should be ints are ints
        self._L = int(self._L)
        self._M = int(self._M)
        self._N = int(self._N)
        self._NFP = int(self._NFP)
        self._modes = self._modes.astype(int)

    def _set_up(self):
        """Do things after loading or changing resolution."""
        # Also recreates any attributes not in _io_attrs on load from input file.
        # See IOAble class docstring for more info.
        self._enforce_symmetry()
        self._sort_modes()
        self._create_idx()
        # ensure things that should be ints are ints
        self._L = int(self._L)
        self._M = int(self._M)
        self._N = int(self._N)
        self._NFP = int(self._NFP)
        self._modes = self._modes.astype(int)

    def _enforce_symmetry(self):
        """Enforce stellarator symmetry."""
        assert self.sym in [
            "sin",
            "sine",
            "cos",
            "cosine",
            "even",
            "cos(t)",
            False,
            None,
        ], f"Unknown symmetry type {self.sym}"
        if self.sym in ["cos", "cosine"]:  # cos(m*t-n*z) symmetry
            self._modes = self.modes[
                np.asarray(sign(self.modes[:, 1]) == sign(self.modes[:, 2]))
            ]
        elif self.sym in ["sin", "sine"]:  # sin(m*t-n*z) symmetry
            self._modes = self.modes[
                np.asarray(sign(self.modes[:, 1]) != sign(self.modes[:, 2]))
            ]
        elif self.sym == "even":  # even powers of rho
            self._modes = self.modes[np.asarray(self.modes[:, 0] % 2 == 0)]
        elif self.sym == "cos(t)":  # cos(m*t) terms only
            self._modes = self.modes[np.asarray(sign(self.modes[:, 1]) >= 0)]
        elif self.sym is None:
            self._sym = False

    def _sort_modes(self):
        """Sorts modes for use with FFT."""
        sort_idx = np.lexsort((self.modes[:, 1], self.modes[:, 0], self.modes[:, 2]))
        self._modes = self.modes[sort_idx]

    def _create_idx(self):
        """Create index for use with self.get_idx()."""
        self._idx = {}
        for idx, (L, M, N) in enumerate(self.modes):
            if L not in self._idx:
                self._idx[L] = {}
            if M not in self._idx[L]:
                self._idx[L][M] = {}
            self._idx[L][M][N] = idx

    def get_idx(self, L=0, M=0, N=0, error=True):
        """Get the index of the ``'modes'`` array corresponding to given mode numbers.

        Parameters
        ----------
        L : int
            Radial mode number.
        M : int
            Poloidal mode number.
        N : int
            Toroidal mode number.
        error : bool
            Whether to raise exception if the mode is not in the basis (default),
            or to return an empty array.

        Returns
        -------
        idx : int
            Index of given mode numbers.

        """
        try:
            return self._idx[L][M][N]
        except KeyError as e:
            if error:
                raise ValueError(
                    "mode ({}, {}, {}) is not in basis {}".format(L, M, N, str(self))
                ) from e
            else:
                return np.array([], dtype=int)

    @abstractmethod
    def _get_modes(self):
        """ndarray: Mode numbers for the basis."""

    @abstractmethod
    def evaluate(
        self, nodes, derivatives=np.array([0, 0, 0]), modes=None, unique=False
    ):
        """Evaluate basis functions at specified nodes.

        Parameters
        ----------
        nodes : ndarray of float, size(num_nodes,3)
            node coordinates, in (rho,theta,zeta)
        derivatives : ndarray of int, shape(3,)
            order of derivatives to compute in (rho,theta,zeta)
        modes : ndarray of in, shape(num_modes,3), optional
            basis modes to evaluate (if None, full basis is used)
        unique : bool, optional
            whether to reduce workload by only calculating for unique values of nodes,
            modes can be faster, but doesn't work with jit or autodiff

        Returns
        -------
        y : ndarray, shape(num_nodes,num_modes)
            basis functions evaluated at nodes

        """

    @abstractmethod
    def change_resolution(self):
        """Change resolution of the basis to the given resolutions."""

    @property
    def L(self):
        """int: Maximum radial resolution."""
        return self.__dict__.setdefault("_L", 0)

    @property
    def M(self):
        """int:  Maximum poloidal resolution."""
        return self.__dict__.setdefault("_M", 0)

    @property
    def N(self):
        """int: Maximum toroidal resolution."""
        return self.__dict__.setdefault("_N", 0)

    @property
    def NFP(self):
        """int: Number of field periods."""
        return self.__dict__.setdefault("_NFP", 1)

    @property
    def sym(self):
        """str: {``'cos'``, ``'sin'``, ``False``} Type of symmetry."""
        return self.__dict__.setdefault("_sym", False)

    @property
    def modes(self):
        """ndarray: Mode numbers [l,m,n]."""
        return self.__dict__.setdefault("_modes", np.array([]).reshape((0, 3)))

    @property
    def num_modes(self):
        """int: Total number of modes in the spectral basis."""
        return self.modes.shape[0]

    @property
    def spectral_indexing(self):
        """str: Type of indexing used for the spectral basis."""
        return self.__dict__.setdefault("_spectral_indexing", "linear")

    def __repr__(self):
        """Get the string form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (L={}, M={}, N={}, NFP={}, sym={}, spectral_indexing={})".format(
                self.L, self.M, self.N, self.NFP, self.sym, self.spectral_indexing
            )
        )


class _FE_Basis(IOAble, ABC):
    """Basis is an abstract base class for finite-element basis sets."""

    _io_attrs_ = [
        "_I_2ML",
        "_Q",
        "_NFP",
        "_modes",
        "_sym",
    ]

    def __init__(self):
        self._enforce_symmetry()
        self._create_idx()

    def _set_up(self):
        """Do things after loading or changing resolution."""
        # Also recreates any attributes not in _io_attrs on load from input file.
        # See IOAble class docstring for more info.
        self._enforce_symmetry()
        self._create_idx()

    def _create_idx(self):
        """Create index for use with self.get_idx()."""
        self._idx = {}
        for idx, (I_2ML, Q) in enumerate(self.modes):
            self._idx[I_2ML] = {}
            self._idx[I_2ML][Q] = idx

    def get_idx(self, I_2ML=0, Q=0, error=True):
        """Get the index of the ``'modes'`` array corresponding to given mode numbers.

        Parameters
        ----------
        I_2ML : int
            Maximum number of triangles in a 2D tesselation in the (theta, zeta)
            plane. If have (M x N) points in the grid, I_2ML = 2NM.
        Q : int
            Number of basis functions. For order K triangle FE, should be
            q = (K + 1)(K + 2) / 2.
        error : bool
            whether to raise exception if mode is not in basis, or return empty array

        Returns
        -------
        idx : ndarray of int
            Index of given mode numbers.

        """
        try:
            return self._idx[I_2ML][Q]
        except KeyError as e:
            if error:
                raise ValueError(
                    "mode ({}, {}) is not in basis {}".format(I_2ML, Q, str(self))
                ) from e
            else:
                return np.array([]).astype(int)

    @abstractmethod
    def _get_modes(self):
        """ndarray: Mode numbers for the basis."""

    @abstractmethod
    def evaluate(
        self, nodes, derivatives=np.array([0, 0, 0]), modes=None, unique=False
    ):
        """Evaluate basis functions at specified nodes.

        Parameters
        ----------
        nodes : ndarray of float, size(num_nodes,3)
            node coordinates, in (rho,theta,zeta)
        derivatives : ndarray of int, shape(3,)
            order of derivatives to compute in (rho,theta,zeta)
        modes : ndarray of in, shape(num_modes,2), optional
            basis modes to evaluate (if None, full basis is used)
        unique : bool, optional
            whether to workload by only calculating for unique values of nodes, modes
            can be faster, but doesn't work with jit or autodiff

        Returns
        -------
        y : ndarray, shape(num_nodes,num_modes)
            basis functions evaluated at nodes

        """

    def _enforce_symmetry(self):
        """Do nothing for now."""

    @abstractmethod
    def change_resolution(self):
        """Change resolution of the basis to the given resolutions."""

    @property
    def L(self):
        """int: Maximum radial resolution."""
        return self.__dict__.setdefault("_L", 0)

    @L.setter
    def L(self, L):
        assert int(L) == L, "Radial Resolution must be an integer!"
        self._L = int(L)

    @property
    def I_2ML(self):
        """int:  Maximum triangle index."""
        return self.__dict__.setdefault("_I_2ML", 0)

    @I_2ML.setter
    def I_2ML(self, I_2ML):
        assert int(I_2ML) == I_2ML, "Number of triangles must be an integer!"
        self._I_2ML = int(I_2ML)

    @property
    def Q(self):
        """int: Maximum basis function index = (K + 1)(K + 2) / 2."""
        return self.__dict__.setdefault("_Q", 0)

    @Q.setter
    def Q(self, Q):
        assert int(Q) == Q, "Number of basis functions must be an integer!"
        self._Q = int(Q)

    @property
    def NFP(self):
        """int: Number of field periods."""
        return self.__dict__.setdefault("_NFP", 1)

    @property
    def sym(self):
        """str: {``'cos'``, ``'sin'``, ``False``} Type of symmetry."""
        return self.__dict__.setdefault("_sym", False)

    @property
    def modes(self):
        """ndarray: Mode numbers [l,i,j]."""
        return self.__dict__.setdefault("_modes", np.array([]).reshape((0, 3)))

    @modes.setter
    def modes(self, modes):
        self._modes = modes

    @property
    def num_modes(self):
        """int: Total number of modes in the finite element basis."""
        return self.modes.shape[0]

    def __repr__(self):
        """Get the string form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (L={}, M={}, N={}, NFP={}, sym={})".format(
                self.L,
                self.I_2ML,
                self.Q,
                self.NFP,
                self.sym,
            )
        )


class PowerSeries(_Basis):
    """1D basis set for flux surface quantities.

    Power series in the radial coordinate.

    Parameters
    ----------
    L : int
        Maximum radial resolution.
    sym : {"even", False}
        Type of symmetry. "even" has only even powers of rho, for an analytic profile
        on the disc. False uses the full (odd + even) powers.

    """

    def __init__(self, L, sym="even"):
        self._L = check_nonnegint(L, "L", False)
        self._M = 0
        self._N = 0
        self._NFP = 1
        self._sym = sym
        self._spectral_indexing = "linear"

        self._modes = self._get_modes(L=self.L)

        super().__init__()

    def _get_modes(self, L):
        """Get mode numbers for power series.

        Parameters
        ----------
        L : int
            Maximum radial resolution.

        Returns
        -------
        modes : ndarray of int, shape(num_modes,3)
            Array of mode numbers [l,m,n].
            Each row is one basis function with modes (l,m,n).

        """
        l = np.arange(L + 1).reshape((-1, 1))
        z = np.zeros((L + 1, 2))
        return np.hstack([l, z])

    def evaluate(
        self, nodes, derivatives=np.array([0, 0, 0]), modes=None, unique=False
    ):
        """Evaluate basis functions at specified nodes.

        Parameters
        ----------
        nodes : ndarray of float, size(num_nodes,3)
            Node coordinates, in (rho,theta,zeta).
        derivatives : ndarray of int, shape(num_derivatives,3)
            Order of derivatives to compute in (rho,theta,zeta).
        modes : ndarray of in, shape(num_modes,3), optional
            Basis modes to evaluate (if None, full basis is used)
        unique : bool, optional
            whether to reduce workload by only calculating for unique values of nodes,
            modes can be faster, but doesn't work with jit or autodiff

        Returns
        -------
        y : ndarray, shape(num_nodes,num_modes)
            basis functions evaluated at nodes

        """
        if modes is None:
            modes = self.modes
        if (derivatives[1] != 0) or (derivatives[2] != 0):
            return jnp.zeros((nodes.shape[0], modes.shape[0]))
        if not len(modes):
            return np.array([]).reshape((len(nodes), 0))

        r, t, z = nodes.T
        l, m, n = modes.T

        if unique:
            _, ridx, routidx = np.unique(
                r, return_index=True, return_inverse=True, axis=0
            )
            _, lidx, loutidx = np.unique(
                l, return_index=True, return_inverse=True, axis=0
            )
            r = r[ridx]
            l = l[lidx]

        radial = powers(r, l, dr=derivatives[0])
        if unique:
            radial = radial[routidx][:, loutidx]

        return radial

    def change_resolution(self, L):
        """Change resolution of the basis to the given resolution.

        Parameters
        ----------
        L : int
            Maximum radial resolution.

        """
        if L != self.L:
            self._L = check_nonnegint(L, "L", False)
            self._modes = self._get_modes(self.L)
            self._set_up()


class FourierSeries(_Basis):
    """1D basis set for use with the magnetic axis.

    Fourier series in the toroidal coordinate.

    Parameters
    ----------
    N : int
        Maximum toroidal resolution.
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
        self._N = check_nonnegint(N, "N", False)
        self._NFP = check_posint(NFP, "NFP", False)
        self._sym = sym
        self._spectral_indexing = "linear"

        self._modes = self._get_modes(N=self.N)

        super().__init__()

    def _get_modes(self, N):
        """Get mode numbers for Fourier series.

        Parameters
        ----------
        N : int
            Maximum toroidal resolution.

        Returns
        -------
        modes : ndarray of int, shape(num_modes,3)
            Array of mode numbers [l,m,n].
            Each row is one basis function with modes (l,m,n).

        """
        dim_tor = 2 * N + 1
        n = np.arange(dim_tor).reshape((-1, 1)) - N
        z = np.zeros((dim_tor, 2))
        return np.hstack([z, n])

    def evaluate(
        self, nodes, derivatives=np.array([0, 0, 0]), modes=None, unique=False
    ):
        """Evaluate basis functions at specified nodes.

        Parameters
        ----------
        nodes : ndarray of float, size(num_nodes,3)
            Node coordinates, in (rho,theta,zeta).
        derivatives : ndarray of int, shape(num_derivatives,3)
            Order of derivatives to compute in (rho,theta,zeta).
        modes : ndarray of in, shape(num_modes,3), optional
            Basis modes to evaluate (if None, full basis is used).
        unique : bool, optional
            Whether to workload by only calculating for unique values of nodes, modes
            can be faster, but doesn't work with jit or autodiff.

        Returns
        -------
        y : ndarray, shape(num_nodes,num_modes)
            Basis functions evaluated at nodes.

        """
        if modes is None:
            modes = self.modes
        if (derivatives[0] != 0) or (derivatives[1] != 0):
            return jnp.zeros((nodes.shape[0], modes.shape[0]))
        if not len(modes):
            return np.array([]).reshape((len(nodes), 0))

        r, t, z = nodes.T
        l, m, n = modes.T

        if unique:
            _, zidx, zoutidx = np.unique(
                z, return_index=True, return_inverse=True, axis=0
            )
            _, nidx, noutidx = np.unique(
                n, return_index=True, return_inverse=True, axis=0
            )
            z = z[zidx]
            n = n[nidx]

        toroidal = fourier(z[:, np.newaxis], n, self.NFP, derivatives[2])
        if unique:
            toroidal = toroidal[zoutidx][:, noutidx]

        return toroidal

    def change_resolution(self, N, NFP=None, sym=None):
        """Change resolution of the basis to the given resolutions.

        Parameters
        ----------
        N : int
            Maximum toroidal resolution.
        NFP : int
            Number of field periods.
        sym : bool
            Whether to enforce stellarator symmetry.

        """
        NFP = check_posint(NFP, "NFP")
        self._NFP = NFP if NFP is not None else self.NFP
        if N != self.N:
            self._N = check_nonnegint(N, "N", False)
            self._sym = sym if sym is not None else self.sym
            self._modes = self._get_modes(self.N)
            self._set_up()


class DoubleFourierSeries(_Basis):
    """2D basis set for use on a single flux surface.

    Fourier series in both the poloidal and toroidal coordinates.

    Parameters
    ----------
    M : int
        Maximum poloidal resolution.
    N : int
        Maximum toroidal resolution.
    NFP : int
        Number of field periods.
    sym : {``'cos'``, ``'sin'``, ``False``}
        * ``'cos'`` for cos(m*t-n*z) symmetry
        * ``'sin'`` for sin(m*t-n*z) symmetry
        * ``False`` for no symmetry (Default)

    """

    def __init__(self, M, N, NFP=1, sym=False):
        self._L = 0
        self._M = check_nonnegint(M, "M", False)
        self._N = check_nonnegint(N, "N", False)
        self._NFP = check_posint(NFP, "NFP", False)
        self._sym = sym
        self._spectral_indexing = "linear"

        self._modes = self._get_modes(M=self.M, N=self.N)

        super().__init__()

    def _get_modes(self, M, N):
        """Get mode numbers for double Fourier series.

        Parameters
        ----------
        M : int
            Maximum poloidal resolution.
        N : int
            Maximum toroidal resolution.

        Returns
        -------
        modes : ndarray of int, shape(num_modes,3)
            Array of mode numbers [l,m,n].
            Each row is one basis function with modes (l,m,n).

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

    def evaluate(
        self, nodes, derivatives=np.array([0, 0, 0]), modes=None, unique=False
    ):
        """Evaluate basis functions at specified nodes.

        Parameters
        ----------
        nodes : ndarray of float, size(num_nodes,3)
            Node coordinates, in (rho,theta,zeta).
        derivatives : ndarray of int, shape(num_derivatives,3)
            Order of derivatives to compute in (rho,theta,zeta).
        modes : ndarray of in, shape(num_modes,3), optional
            Basis modes to evaluate (if None, full basis is used).
        unique : bool, optional
            Whether to workload by only calculating for unique values of nodes, modes
            can be faster, but doesn't work with jit or autodiff.

        Returns
        -------
        y : ndarray, shape(num_nodes,num_modes)
            Basis functions evaluated at nodes.

        """
        if modes is None:
            modes = self.modes
        if derivatives[0] != 0:
            return jnp.zeros((nodes.shape[0], modes.shape[0]))
        if not len(modes):
            return np.array([]).reshape((len(nodes), 0))

        r, t, z = nodes.T
        l, m, n = modes.T

        if unique:
            _, tidx, toutidx = np.unique(
                t, return_index=True, return_inverse=True, axis=0
            )
            _, zidx, zoutidx = np.unique(
                z, return_index=True, return_inverse=True, axis=0
            )
            _, midx, moutidx = np.unique(
                m, return_index=True, return_inverse=True, axis=0
            )
            _, nidx, noutidx = np.unique(
                n, return_index=True, return_inverse=True, axis=0
            )
            t = t[tidx]
            z = z[zidx]
            m = m[midx]
            n = n[nidx]

        poloidal = fourier(t[:, np.newaxis], m, 1, derivatives[1])
        toroidal = fourier(z[:, np.newaxis], n, self.NFP, derivatives[2])
        if unique:
            poloidal = poloidal[toutidx][:, moutidx]
            toroidal = toroidal[zoutidx][:, noutidx]

        return poloidal * toroidal

    def change_resolution(self, M, N, NFP=None, sym=None):
        """Change resolution of the basis to the given resolutions.

        Parameters
        ----------
        M : int
            Maximum poloidal resolution.
        N : int
            Maximum toroidal resolution.
        NFP : int
            Number of field periods.
        sym : bool
            Whether to enforce stellarator symmetry.

        Returns
        -------
        None

        """
        NFP = check_posint(NFP, "NFP")
        self._NFP = NFP if NFP is not None else self.NFP
        if M != self.M or N != self.N or sym != self.sym:
            self._M = check_nonnegint(M, "M", False)
            self._N = check_nonnegint(N, "N", False)
            self._sym = sym if sym is not None else self.sym
            self._modes = self._get_modes(self.M, self.N)
            self._set_up()


class ZernikePolynomial(_Basis):
    """2D basis set for analytic functions in a unit disc.

    Parameters
    ----------
    L : int
        Maximum radial resolution. Use L=-1 for default based on M.
    M : int
        Maximum poloidal resolution.
    sym : {``'cos'``, ``'sin'``, ``False``}
        * ``'cos'`` for cos(m*t-n*z) symmetry
        * ``'sin'`` for sin(m*t-n*z) symmetry
        * ``False`` for no symmetry (Default)
    spectral_indexing : {``'ansi'``, ``'fringe'``}
        Indexing method, default value = ``'ansi'``

        For L=0, all methods are equivalent and give a "chevron" shaped
        basis (only the outer edge of the zernike pyramid of width M).
        For L>0, the indexing scheme defines order of the basis functions:

        ``'ansi'``: ANSI indexing fills in the pyramid with triangles of
        decreasing size, ending in a triangle shape. For L == M,
        the traditional ANSI pyramid indexing is recovered. For L>M, adds rows
        giving a "house" shape.

        ``'fringe'``: Fringe indexing fills in the pyramid with chevrons of
        decreasing size, ending in a diamond shape for L=2*M where
        the traditional fringe/U of Arizona indexing is recovered.
        For L > 2*M, adds chevrons to the bottom, making a hexagonal diamond.

    """

    def __init__(self, L, M, sym=False, spectral_indexing="ansi"):
        self._L = check_nonnegint(L, "L", False)
        self._M = check_nonnegint(M, "M", False)
        self._N = 0
        self._NFP = 1
        self._sym = sym
        self._spectral_indexing = spectral_indexing

        self._modes = self._get_modes(
            L=self.L, M=self.M, spectral_indexing=self.spectral_indexing
        )

        super().__init__()

    def _get_modes(self, L, M, spectral_indexing="ansi"):
        """Get mode numbers for Fourier-Zernike basis functions.

        Parameters
        ----------
        L : int
            Maximum radial resolution.
        M : int
            Maximum poloidal resolution.
        spectral_indexing : {``'ansi'``, ``'fringe'``}
            Indexing method, default value = ``'ansi'``

            For L=0, all methods are equivalent and give a "chevron" shaped
            basis (only the outer edge of the zernike pyramid of width M).
            For L>0, the indexing scheme defines order of the basis functions:

            ``'ansi'``: ANSI indexing fills in the pyramid with triangles of
            decreasing size, ending in a triangle shape. For L == M,
            the traditional ANSI pyramid indexing is recovered. For L>M, adds rows
            to the bottom of the pyramid, increasing L while keeping M constant,
            giving a "house" shape.

            ``'fringe'``: Fringe indexing fills in the pyramid with chevrons of
            decreasing size, ending in a diamond shape for L=2*M where
            the traditional fringe/U of Arizona indexing is recovered.
            For L > 2*M, adds chevrons to the bottom, making a hexagonal diamond.

        Returns
        -------
        modes : ndarray of int, shape(num_modes,3)
            Array of mode numbers [l,m,n].
            Each row is one basis function with modes (l,m,n).

        """
        assert spectral_indexing in [
            "ansi",
            "fringe",
        ], "Unknown spectral_indexing: {}".format(spectral_indexing)

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
                pol_posm += [
                    [(l - m, m) for m in range(0, M + 1)]
                    for l in range(2 * M, L + 1, 2)
                ]

        pol = [
            [(l, m), (l, -m)] if m != 0 else [(l, m)] for l, m in flatten_list(pol_posm)
        ]
        pol = np.array(flatten_list(pol))
        num_pol = len(pol)
        tor = np.zeros((num_pol, 1))

        return np.hstack([pol, tor])

    def evaluate(
        self, nodes, derivatives=np.array([0, 0, 0]), modes=None, unique=False
    ):
        """Evaluate basis functions at specified nodes.

        Parameters
        ----------
        nodes : ndarray of float, size(num_nodes,3)
            Node coordinates, in (rho,theta,zeta).
        derivatives : ndarray of int, shape(num_derivatives,3)
            Order of derivatives to compute in (rho,theta,zeta).
        modes : ndarray of int, shape(num_modes,3), optional
            Basis modes to evaluate (if None, full basis is used).
        unique : bool, optional
            Whether to workload by only calculating for unique values of nodes, modes
            can be faster, but doesn't work with jit or autodiff.

        Returns
        -------
        y : ndarray, shape(num_nodes,num_modes)
            Basis functions evaluated at nodes.

        """
        if modes is None:
            modes = self.modes
        if derivatives[2] != 0:
            return jnp.zeros((nodes.shape[0], modes.shape[0]))
        if not len(modes):
            return np.array([]).reshape((len(nodes), 0))

        r, t, z = nodes.T
        l, m, n = modes.T
        lm = modes[:, :2]

        if unique:
            _, ridx, routidx = np.unique(
                r, return_index=True, return_inverse=True, axis=0
            )
            _, tidx, toutidx = np.unique(
                t, return_index=True, return_inverse=True, axis=0
            )
            _, lmidx, lmoutidx = np.unique(
                lm, return_index=True, return_inverse=True, axis=0
            )
            _, midx, moutidx = np.unique(
                m, return_index=True, return_inverse=True, axis=0
            )
            r = r[ridx]
            t = t[tidx]
            lm = lm[lmidx]
            m = m[midx]

        radial = zernike_radial(r[:, np.newaxis], lm[:, 0], lm[:, 1], dr=derivatives[0])
        poloidal = fourier(t[:, np.newaxis], m, 1, derivatives[1])

        if unique:
            radial = radial[routidx][:, lmoutidx]
            poloidal = poloidal[toutidx][:, moutidx]

        return radial * poloidal

    def change_resolution(self, L, M, sym=None):
        """Change resolution of the basis to the given resolutions.

        Parameters
        ----------
        L : int
            Maximum radial resolution.
        M : int
            Maximum poloidal resolution.
        sym : bool
            Whether to enforce stellarator symmetry.

        Returns
        -------
        None

        """
        if L != self.L or M != self.M or sym != self.sym:
            self._L = check_nonnegint(L, "L", False)
            self._M = check_nonnegint(M, "M", False)
            self._sym = sym if sym is not None else self.sym
            self._modes = self._get_modes(
                self.L, self.M, spectral_indexing=self.spectral_indexing
            )
            self._set_up()


class ChebyshevDoubleFourierBasis(_Basis):
    """3D basis: tensor product of Chebyshev polynomials and two Fourier series.

    Fourier series in both the poloidal and toroidal coordinates.

    Parameters
    ----------
    L : int
        Maximum radial resolution.
    M : int
        Maximum poloidal resolution.
    N : int
        Maximum toroidal resolution.
    NFP : int
        Number of field periods.
    sym : {``'cos'``, ``'sin'``, ``False``}
        * ``'cos'`` for cos(m*t-n*z) symmetry
        * ``'sin'`` for sin(m*t-n*z) symmetry
        * ``False`` for no symmetry (Default)

    """

    def __init__(self, L, M, N, NFP=1, sym=False):
        self._L = check_nonnegint(L, "L", False)
        self._M = check_nonnegint(M, "M", False)
        self._N = check_nonnegint(N, "N", False)
        self._NFP = check_posint(NFP, "NFP", False)
        self._sym = sym
        self._spectral_indexing = "linear"

        self._modes = self._get_modes(L=self.L, M=self.M, N=self.N)

        super().__init__()

    def _get_modes(self, L, M, N):
        """Get mode numbers for Chebyshev-Fourier series.

        Parameters
        ----------
        L : int
            Maximum radial resolution.
        M : int
            Maximum poloidal resolution.
        N : int
            Maximum toroidal resolution.

        Returns
        -------
        modes : ndarray of int, shape(num_modes,3)
            Array of mode numbers [l,m,n].
            Each row is one basis function with modes (l,m,n).

        """
        dim_pol = 2 * M + 1
        dim_tor = 2 * N + 1
        l = np.arange(L + 1)
        m = np.arange(dim_pol) - M
        n = np.arange(dim_tor) - N
        ll, mm, nn = np.meshgrid(l, m, n)
        ll = ll.reshape((-1, 1), order="F")
        mm = mm.reshape((-1, 1), order="F")
        nn = nn.reshape((-1, 1), order="F")
        y = np.hstack([ll, mm, nn])
        return y

    def evaluate(
        self, nodes, derivatives=np.array([0, 0, 0]), modes=None, unique=False
    ):
        """Evaluate basis functions at specified nodes.

        Parameters
        ----------
        nodes : ndarray of float, size(num_nodes,3)
            Node coordinates, in (rho,theta,zeta).
        derivatives : ndarray of int, shape(num_derivatives,3)
            Order of derivatives to compute in (rho,theta,zeta).
        modes : ndarray of in, shape(num_modes,3), optional
            Basis modes to evaluate (if None, full basis is used).
        unique : bool, optional
            whether to reduce workload by only calculating for unique values of nodes,
            modes can be faster, but doesn't work with jit or autodiff

        Returns
        -------
        y : ndarray, shape(num_nodes,num_modes)
            Basis functions evaluated at nodes.

        """
        if modes is None:
            modes = self.modes
        if not len(modes):
            return np.array([]).reshape((len(nodes), 0))

        r, t, z = nodes.T
        l, m, n = modes.T

        radial = chebyshev(r[:, np.newaxis], l, dr=derivatives[0])
        poloidal = fourier(t[:, np.newaxis], m, 1, derivatives[1])
        toroidal = fourier(z[:, np.newaxis], n, self.NFP, derivatives[2])

        return radial * poloidal * toroidal

    def change_resolution(self, L, M, N, NFP=None, sym=None):
        """Change resolution of the basis to the given resolutions.

        Parameters
        ----------
        L : int
            Maximum radial resolution.
        M : int
            Maximum poloidal resolution.
        N : int
            Maximum toroidal resolution.
        NFP : int
            Number of field periods.
        sym : bool
            Whether to enforce stellarator symmetry.

        Returns
        -------
        None

        """
        NFP = check_posint(NFP, "NFP")
        self._NFP = NFP if NFP is not None else self.NFP
        if L != self.L or M != self.M or N != self.N or sym != self.sym:
            self._L = check_nonnegint(L, "L", False)
            self._M = check_nonnegint(M, "M", False)
            self._N = check_nonnegint(N, "N", False)
            self._sym = sym if sym is not None else self.sym
            self._modes = self._get_modes(self.L, self.M, self.N)
            self._set_up()


class FiniteElementBasis(_FE_Basis):
    """3D finite element basis set for analytic functions in a toroidal volume.

    Class for making a 1D, 2D, or 3D set of finite elements on the
    (rho, theta, zeta) grid. 1D finite elements are made on a pure
    theta grid, and 2D finite elements are made in a toroidal cross-section,
    i.e. a grid in (rho, theta).

    Parameters
    ----------
    L : int
        Number of grid points radially
    M : int
        Number of grid points poloidally
    N : int
        Number of grid points toroidally
    K : int
        Order of the finite elements in each interval/triangle/tetrahedron.
    NFP : int
        Number of field periods.
    sym : {``'cos'``, ``'sin'``, ``False``}
        * ``'cos'`` for cos(m*t-n*z) symmetry
        * ``'sin'`` for sin(m*t-n*z) symmetry
        * ``False`` for no symmetry (Default)
    """

    def __init__(self, L, M, N, K=1, NFP=1, sym=False):
        self.L = L
        self.M = M
        self.N = N
        self.K = K
        self._NFP = NFP
        self._sym = sym
        if L == 0 and N == 0:
            self.mesh = FiniteElementMesh1D(M, K=K)
            self.I_2ML = M - 1
            self.Q = K + 1
        elif N == 0:
            self.mesh = FiniteElementMesh2D(L, M, K=K)
            self.I_2ML = 2 * (M - 1) * (L - 1)
            self.Q = int((K + 1) * (K + 2) / 2.0)
        else:
            self.mesh = FiniteElementMesh3D(L, M, N, K=K)
            self.I_2ML = 6 * (M - 1) * (N - 1) * (L - 1)
            self.Q = int((K + 1) * (K + 2) * (K + 3) / 6.0)
        self.nmodes = self.I_2ML * self.Q
        self._modes = self._get_modes()
        super().__init__()

    def _get_modes(self):
        """Get mode numbers for a pure finite element basis.

        Returns
        -------
        modes : ndarray of int, shape(num_modes, 2)
            Array of mode numbers [i, q].
            Each row is one basis function with modes (i, q).

        """
        lij_mesh = np.meshgrid(
            np.arange(self.I_2ML),
            np.arange(self.Q),
            indexing="ij",
        )
        lij_mesh = np.reshape(np.array(lij_mesh, dtype=int), (2, self.nmodes)).T
        return np.unique(lij_mesh, axis=0)

    def evaluate(self, nodes, derivatives=np.array([0, 0, 0]), modes=None, n=-1):
        """Evaluate basis functions at specified nodes.

        Parameters
        ----------
        nodes : ndarray of float, size(num_nodes,3)
            Node coordinates, in (rho,theta,zeta).
        derivatives : ndarray of int, shape(num_derivatives,3)
            Order of derivatives to compute in (rho,theta,zeta).
        modes : ndarray of int, shape(num_modes,2), optional
            Basis modes to evaluate (if None, full basis is used).

        Returns
        -------
        y : ndarray, shape(num_nodes,num_modes)
            Basis functions evaluated at nodes.

        """
        if modes is None:
            modes = self.modes
        if not len(modes):
            return np.array([]).reshape((len(nodes), 0))

        # TODO: avoid duplicate calculations when mixing derivatives
        r, t, z = nodes.T
        i, q = modes.T

        if self.L == 0 and self.N == 0:
            # Get all IQ basis functions from each of the points,
            # and most will be zeros at a given point because of local support.
            basis_functions = self.mesh.full_basis_functions_corresponding_to_points(t)
            basis_functions = np.reshape(basis_functions, (len(t), -1))
            inds = i * self.Q + q
        elif self.N == 0:
            # Tessellate the domain and find the basis functions for theta, zeta
            Rho_Theta = np.array([np.ravel(r), np.ravel(t)]).T
            (
                intervals,
                basis_functions,
            ) = self.mesh.find_triangles_corresponding_to_points(Rho_Theta)

            # Sum the basis functions from each triangle node
            basis_functions = np.reshape(basis_functions, (len(t), -1))
            inds = i * self.Q + q
        else:
            Rho, Theta, Zeta = np.meshgrid(r, t, z, indexing="ij")
            Rho_Theta_Zeta = np.array(
                [np.ravel(Rho), np.ravel(Theta), np.ravel(Zeta)]
            ).T
            (
                intervals,
                basis_functions,
            ) = self.mesh.find_tetrahedron_corresponding_to_points(Rho_Theta_Zeta)
            inds = i * self.Q + q

        return basis_functions[:, inds]

    def change_resolution(self, L, M, N):
        """Change resolution of the basis to the given resolutions.

        Parameters
        ----------
        L : int
            Maximum radial resolution.
        M : int
            Maximum number of poloidal basis functions.
        N : int
            Maximum number of toroidal basis functions.

        Returns
        -------
        None

        """
        if L != self.L or M != self.M or N != self.N:
            self.L = L
            self.M = M
            self.N = N
            self._modes = self._get_modes(self.L, self.M, self.N)
            self._set_up()


class FourierZernikeBasis(_Basis):
    """3D basis set for analytic functions in a toroidal volume.

    Zernike polynomials in the radial & poloidal coordinates, and a Fourier
    series in the toroidal coordinate.

    Parameters
    ----------
    L : int
        Maximum radial resolution. Use L=-1 for default based on M.
    M : int
        Maximum poloidal resolution.
    N : int
        Maximum toroidal resolution.
    NFP : int
        Number of field periods.
    sym : {``'cos'``, ``'sin'``, ``False``}
        * ``'cos'`` for cos(m*t-n*z) symmetry
        * ``'sin'`` for sin(m*t-n*z) symmetry
        * ``False`` for no symmetry (Default)
    spectral_indexing : {``'ansi'``, ``'fringe'``}
        Indexing method, default value = ``'ansi'``

        For L=0, all methods are equivalent and give a "chevron" shaped
        basis (only the outer edge of the zernike pyramid of width M).
        For L>0, the indexing scheme defines order of the basis functions:

        ``'ansi'``: ANSI indexing fills in the pyramid with triangles of
        decreasing size, ending in a triangle shape. For L == M,
        the traditional ANSI pyramid indexing is recovered. For L>M, adds rows
        to the bottom of the pyramid, increasing L while keeping M constant,
        giving a "house" shape.

        ``'fringe'``: Fringe indexing fills in the pyramid with chevrons of
        decreasing size, ending in a diamond shape for L=2*M where
        the traditional fringe/U of Arizona indexing is recovered.
        For L > 2*M, adds chevrons to the bottom, making a hexagonal diamond.

    """

    def __init__(self, L, M, N, NFP=1, sym=False, spectral_indexing="ansi"):
        self._L = check_nonnegint(L, "L", False)
        self._M = check_nonnegint(M, "M", False)
        self._N = check_nonnegint(N, "N", False)
        self._NFP = check_posint(NFP, "NFP", False)
        self._sym = sym
        self._spectral_indexing = spectral_indexing

        self._modes = self._get_modes(
            L=self.L, M=self.M, N=self.N, spectral_indexing=self.spectral_indexing
        )

        super().__init__()

    def _get_modes(self, L, M, N, spectral_indexing="ansi"):
        """Get mode numbers for Fourier-Zernike basis functions.

        Parameters
        ----------
        L : int
            Maximum radial resolution.
        M : int
            Maximum poloidal resolution.
        N : int
            Maximum toroidal resolution.
        spectral_indexing : {``'ansi'``, ``'fringe'``}
            Indexing method, default value = ``'ansi'``

            For L=0, all methods are equivalent and give a "chevron" shaped
            basis (only the outer edge of the zernike pyramid of width M).
            For L>0, the indexing scheme defines order of the basis functions:

            ``'ansi'``: ANSI indexing fills in the pyramid with triangles of
            decreasing size, ending in a triangle shape. For L == M,
            the traditional ANSI pyramid indexing is recovered. For L>M, adds rows
            to the bottom of the pyramid, increasing L while keeping M constant,
            giving a "house" shape.

            ``'fringe'``: Fringe indexing fills in the pyramid with chevrons of
            decreasing size, ending in a diamond shape for L=2*M where
            the traditional fringe/U of Arizona indexing is recovered.
            For L > 2*M, adds chevrons to the bottom, making a hexagonal diamond.

        Returns
        -------
        modes : ndarray of int, shape(num_modes,3)
            Array of mode numbers [l,m,n].
            Each row is one basis function with modes (l,m,n).

        """
        assert spectral_indexing in [
            "ansi",
            "fringe",
        ], "Unknown spectral_indexing: {}".format(spectral_indexing)

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
                pol_posm += [
                    [(l - m, m) for m in range(0, M + 1)]
                    for l in range(2 * M, L + 1, 2)
                ]

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

    def evaluate(
        self, nodes, derivatives=np.array([0, 0, 0]), modes=None, unique=False
    ):
        """Evaluate basis functions at specified nodes.

        Parameters
        ----------
        nodes : ndarray of float, size(num_nodes,3)
            Node coordinates, in (rho,theta,zeta).
        derivatives : ndarray of int, shape(num_derivatives,3)
            Order of derivatives to compute in (rho,theta,zeta).
        modes : ndarray of int, shape(num_modes,3), optional
            Basis modes to evaluate (if None, full basis is used).
        unique : bool, optional
            Whether to workload by only calculating for unique values of nodes, modes
            can be faster, but doesn't work with jit or autodiff.

        Returns
        -------
        y : ndarray, shape(num_nodes,num_modes)
            Basis functions evaluated at nodes.

        """
        if modes is None:
            modes = self.modes
        if not len(modes):
            return np.array([]).reshape((len(nodes), 0))

        # TODO: avoid duplicate calculations when mixing derivatives
        r, t, z = nodes.T
        l, m, n = modes.T
        lm = modes[:, :2]

        if False:
            # TODO: can avoid this here by using grid.unique_idx etc
            # and adding unique_modes attributes to basis
            _, ridx, routidx = np.unique(
                r, return_index=True, return_inverse=True, axis=0
            )
            _, tidx, toutidx = np.unique(
                t, return_index=True, return_inverse=True, axis=0
            )
            _, zidx, zoutidx = np.unique(
                z, return_index=True, return_inverse=True, axis=0
            )
            _, lmidx, lmoutidx = np.unique(
                lm, return_index=True, return_inverse=True, axis=0
            )
            _, midx, moutidx = np.unique(
                m, return_index=True, return_inverse=True, axis=0
            )
            _, nidx, noutidx = np.unique(
                n, return_index=True, return_inverse=True, axis=0
            )
            r = r[ridx]
            t = t[tidx]
            z = z[zidx]
            lm = lm[lmidx]
            m = m[midx]
            n = n[nidx]

        radial = zernike_radial(r[:, np.newaxis], lm[:, 0], lm[:, 1], dr=derivatives[0])
        poloidal = fourier(t[:, np.newaxis], m, dt=derivatives[1])
        toroidal = fourier(z[:, np.newaxis], n, NFP=self.NFP, dt=derivatives[2])
        if unique:
            radial = radial[routidx][:, lmoutidx]
            poloidal = poloidal[toutidx][:, moutidx]
            toroidal = toroidal[zoutidx][:, noutidx]

        return radial * poloidal * toroidal

    def change_resolution(self, L, M, N, NFP=None, sym=None):
        """Change resolution of the basis to the given resolutions.

        Parameters
        ----------
        L : int
            Maximum radial resolution.
        M : int
            Maximum poloidal resolution.
        N : int
            Maximum toroidal resolution.
        NFP : int
            Number of field periods.
        sym : bool
            Whether to enforce stellarator symmetry.

        Returns
        -------
        None

        """
        NFP = check_posint(NFP, "NFP")
        self._NFP = NFP if NFP is not None else self.NFP
        if L != self.L or M != self.M or N != self.N or sym != self.sym:
            self._L = check_nonnegint(L, "L", False)
            self._M = check_nonnegint(M, "M", False)
            self._N = check_nonnegint(N, "N", False)
            self._sym = sym if sym is not None else self.sym
            self._modes = self._get_modes(
                self.L, self.M, self.N, spectral_indexing=self.spectral_indexing
            )
            self._set_up()


class ChebyshevPolynomial(_Basis):
    """Shifted Chebyshev polynomial of the first kind.

    Parameters
    ----------
    L : int
        Maximum radial resolution.

    """

    def __init__(self, L):
        self._L = check_nonnegint(L, "L", False)
        self._M = 0
        self._N = 0
        self._NFP = 1
        self._sym = False
        self._spectral_indexing = "linear"

        self._modes = self._get_modes(L=self.L)

        super().__init__()

    def _get_modes(self, L):
        """Get mode numbers for shifted Chebyshev polynomials of the first kind.

        Parameters
        ----------
        L : int
            Maximum radial resolution.

        Returns
        -------
        modes : ndarray of int, shape(num_modes,3)
            Array of mode numbers [l,m,n].
            Each row is one basis function with modes (l,m,n).

        """
        l = np.arange(L + 1).reshape((-1, 1))
        z = np.zeros((L + 1, 2))
        return np.hstack([l, z])

    def evaluate(
        self, nodes, derivatives=np.array([0, 0, 0]), modes=None, unique=False
    ):
        """Evaluate basis functions at specified nodes.

        Parameters
        ----------
        nodes : ndarray of float, size(num_nodes,3)
            Node coordinates, in (rho,theta,zeta).
        derivatives : ndarray of int, shape(num_derivatives,3)
            Order of derivatives to compute in (rho,theta,zeta).
        modes : ndarray of in, shape(num_modes,3), optional
            Basis modes to evaluate (if None, full basis is used)
        unique : bool, optional
            whether to reduce workload by only calculating for unique values of nodes,
            modes can be faster, but doesn't work with jit or autodiff

        Returns
        -------
        y : ndarray, shape(num_nodes,num_modes)
            basis functions evaluated at nodes

        """
        if modes is None:
            modes = self.modes
        if (derivatives[1] != 0) or (derivatives[2] != 0):
            return jnp.zeros((nodes.shape[0], modes.shape[0]))
        if not len(modes):
            return np.array([]).reshape((len(nodes), 0))

        r, t, z = nodes.T
        l, m, n = modes.T

        radial = chebyshev(r[:, np.newaxis], l, dr=derivatives[0])
        return radial

    def change_resolution(self, L):
        """Change resolution of the basis to the given resolution.

        Parameters
        ----------
        L : int
            Maximum radial resolution.

        """
        if L != self.L:
            self._L = check_nonnegint(L, "L", False)
            self._modes = self._get_modes(self.L)
            self._set_up()


def polyder_vec(p, m, exact=False):
    """Vectorized version of polyder.

    For differentiating multiple polynomials of the same degree

    Parameters
    ----------
    p : ndarray, shape(N,M)
        polynomial coefficients. Each row is 1 polynomial, in descending powers of x,
        each column is a power of x
    m : int >=0
        order of derivative
    exact : bool
        Whether to use exact integer arithmetic (not compatible with JAX, but may be
        needed for very high degree polynomials)

    Returns
    -------
    der : ndarray, shape(N,M)
        polynomial coefficients for derivative in descending order

    """
    if exact:
        return _polyder_exact(p, m)
    else:
        return _polyder_jax(p, m)


def _polyder_exact(p, m):
    m = np.asarray(m, dtype=int)  # order of derivative
    p = np.atleast_2d(p)
    order = p.shape[1] - 1

    D = np.arange(order, -1, -1)
    num = np.array([factorial(i) for i in D], dtype=object)
    den = np.array([factorial(max(i - m, 0)) for i in D], dtype=object)
    D = (num // den).astype(p.dtype)

    p = np.roll(D * p, m, axis=1)
    idx = np.arange(p.shape[1])
    p = np.where(idx < m, 0, p)
    return p


@jit
def _polyder_jax(p, m):
    p = jnp.atleast_2d(jnp.asarray(p))
    m = jnp.asarray(m).astype(int)
    order = p.shape[1] - 1
    D = jnp.arange(order, -1, -1)

    def body(i, Di):
        return Di * jnp.maximum(D - i, 1)

    D = fori_loop(0, m, body, jnp.ones_like(D))

    p = jnp.roll(D * p, m, axis=1)
    idx = jnp.arange(p.shape[1])
    p = jnp.where(idx < m, 0, p)

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
    if prec is not None and prec > 18:
        return _polyval_exact(p, x, prec)
    else:
        return _polyval_jax(p, x)


def _polyval_exact(p, x, prec):
    p = np.atleast_2d(p)
    x = np.atleast_1d(x).flatten()
    # TODO: possibly multithread this bit
    mpmath.mp.dps = prec
    y = np.array([np.asarray(mpmath.polyval(list(pi), x)) for pi in p])
    return y.astype(float)


@jit
def _polyval_jax(p, x):
    p = jnp.atleast_2d(jnp.asarray(p))
    x = jnp.atleast_1d(jnp.asarray(x)).flatten()
    npoly = p.shape[0]  # number of polynomials
    order = p.shape[1]  # order of polynomials
    nx = len(x)  # number of coordinates
    y = jnp.zeros((npoly, nx))

    def body(k, y):
        return y * x + jnp.atleast_2d(p[:, k]).T

    y = fori_loop(0, order, body, y)

    return y.astype(float)


def zernike_radial_coeffs(l, m, exact=True):
    """Polynomial coefficients for radial part of zernike basis.

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
        Polynomial coefficients for Zernike polynomials, in descending powers of r.

    Notes
    -----
    Integer representation is exact up to l~54, so leaving `exact` arg as False
    can speed up evaluation with no loss in accuracy
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


def zernike_radial_poly(r, l, m, dr=0, exact="auto"):
    """Radial part of zernike polynomials.

    Evaluates basis functions using numpy to
    exactly compute the polynomial coefficients
    and Horner's method for low resolution,
    or extended precision arithmetic for high resolution.
    Faster for low resolution, but not differentiable.

    Parameters
    ----------
    r : ndarray, shape(N,)
        radial coordinates to evaluate basis
    l : ndarray of int, shape(K,)
        radial mode number(s)
    m : ndarray of int, shape(K,)
        azimuthal mode number(s)
    dr : int
        order of derivative (Default = 0)
    exact : {"auto", True, False}
        Whether to use exact/extended precision arithmetic. Slower but more accurate.
        "auto" will use higher accuracy when needed.

    Returns
    -------
    y : ndarray, shape(N,K)
        basis function(s) evaluated at specified points

    """
    if exact == "auto":
        exact = np.max(l) > 54
    if exact:
        # this should give accuracy of ~1e-10 in the eval'd polynomials
        lmax = np.max(l)
        prec = int(0.4 * lmax + 8.4)
    else:
        prec = None
    coeffs = zernike_radial_coeffs(l, m, exact=exact)
    coeffs = polyder_vec(coeffs, dr, exact=exact)
    return polyval_vec(coeffs, r, prec=prec).T


@functools.partial(jit, static_argnums=3)
def zernike_radial(r, l, m, dr=0):
    """Radial part of zernike polynomials.

    Evaluates basis functions using JAX and a stable
    evaluation scheme based on jacobi polynomials and
    binomial coefficients. Generally faster for L>24
    and differentiable, but slower for low resolution.

    Parameters
    ----------
    r : ndarray, shape(N,)
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
    m = jnp.abs(m).astype(float)
    alpha = m
    beta = 0
    n = (l - m) // 2
    s = (-1) ** n
    jacobi_arg = 1 - 2 * r**2
    if dr == 0:
        out = r**m * _jacobi(n, alpha, beta, jacobi_arg, 0)
    elif dr == 1:
        f = _jacobi(n, alpha, beta, jacobi_arg, 0)
        df = _jacobi(n, alpha, beta, jacobi_arg, 1)
        out = m * r ** jnp.maximum(m - 1, 0) * f - 4 * r ** (m + 1) * df
    elif dr == 2:
        f = _jacobi(n, alpha, beta, jacobi_arg, 0)
        df = _jacobi(n, alpha, beta, jacobi_arg, 1)
        d2f = _jacobi(n, alpha, beta, jacobi_arg, 2)
        out = (
            (m - 1) * m * r ** jnp.maximum(m - 2, 0) * f
            - 4 * (2 * m + 1) * r**m * df
            + 16 * r ** (m + 2) * d2f
        )
    elif dr == 3:
        f = _jacobi(n, alpha, beta, jacobi_arg, 0)
        df = _jacobi(n, alpha, beta, jacobi_arg, 1)
        d2f = _jacobi(n, alpha, beta, jacobi_arg, 2)
        d3f = _jacobi(n, alpha, beta, jacobi_arg, 3)
        out = (
            (m - 2) * (m - 1) * m * r ** jnp.maximum(m - 3, 0) * f
            - 12 * m**2 * r ** jnp.maximum(m - 1, 0) * df
            + 48 * (m + 1) * r ** (m + 1) * d2f
            - 64 * r ** (m + 3) * d3f
        )
    elif dr == 4:
        f = _jacobi(n, alpha, beta, jacobi_arg, 0)
        df = _jacobi(n, alpha, beta, jacobi_arg, 1)
        d2f = _jacobi(n, alpha, beta, jacobi_arg, 2)
        d3f = _jacobi(n, alpha, beta, jacobi_arg, 3)
        d4f = _jacobi(n, alpha, beta, jacobi_arg, 4)
        out = (
            (m - 3) * (m - 2) * (m - 1) * m * r ** jnp.maximum(m - 4, 0) * f
            - 8 * m * (2 * m**2 - 3 * m + 1) * r ** jnp.maximum(m - 2, 0) * df
            + 48 * (2 * m**2 + 2 * m + 1) * r**m * d2f
            - 128 * (2 * m + 3) * r ** (m + 2) * d3f
            + 256 * r ** (m + 4) * d4f
        )
    else:
        raise NotImplementedError(
            "Analytic radial derivatives of Zernike polynomials for order>4 "
            + "have not been implemented."
        )
    return s * jnp.where((l - m) % 2 == 0, out, 0)


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
        radial coordinates to evaluate basis
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


@functools.partial(jit, static_argnums=2)
def chebyshev(r, l, dr=0):
    """Shifted Chebyshev polynomial.

    Parameters
    ----------
    rho : ndarray, shape(N,)
        radial coordinates to evaluate basis
    l : ndarray of int, shape(K,)
        radial mode number(s)
    dr : int
        order of derivative (Default = 0)

    Returns
    -------
    y : ndarray, shape(N,K)
        basis function(s) evaluated at specified points

    """
    r, l = map(jnp.asarray, (r, l))
    x = 2 * r - 1  # shift
    if dr == 0:
        return jnp.cos(l * jnp.arccos(x))
    else:
        # dy/dr = dy/dx * dx/dr = dy/dx * 2
        raise NotImplementedError(
            "Analytic radial derivatives of Chebyshev polynomials "
            + "have not been implemented."
        )


@jit
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
    theta, m, NFP, dt = map(jnp.asarray, (theta, m, NFP, dt))
    m_pos = (m >= 0).astype(int)
    m_abs = jnp.abs(m) * NFP
    shift = m_pos * jnp.pi / 2 + dt * jnp.pi / 2
    return m_abs**dt * jnp.sin(m_abs * theta + shift)


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


class FiniteElementMesh3D:
    """Class representing a 3D mesh in (rho, theta, zeta) using scikit-fem.

    This class represents a set of I_5LMN = 5LMN tetrahedra obtained by tessalation
    f a uniform  L x M x N mesh in rho, theta, zeta. The point of this class is to
    predefine all the tetrahedra and their associated basis functions so that, given
    a new point (rho_i, theta_i, zeta_i), we can quickly return which tetrahedron
    contains this point and its associated basis functions.

    Parameters
    ----------
    L : int
        Number of mesh points in the rho direction.
    M : int
        Number of mesh points in the theta direction.
    N:  int
        Number of mesh points in the zeta direction
    K: integer
        The order of the finite elements to use, which gives (K+1)(K+2)(K+3) / 6
        basis functions.
    """

    def __init__(self, L, M, N, K=1):
        self.M = M
        self.L = L
        self.N = N
        self.I_5LMN = (
            5 * (L - 1) * (M - 1) * (N - 1)
        )  # Considering how to incorporate n_p
        self.Q = int((K + 1) * (K + 2) * (K + 3) / 6)
        self.K = K

        # Trying a Hex Mesh, and then adding in the tetrahedra to each prism

        mesh = fem.MeshHex.init_tensor(
            np.linspace(0, 1, L),
            np.linspace(0, 2 * np.pi, M),
            np.linspace(0, 2 * np.pi, N),
        )

        vertices = mesh.doflocs

        vertices = vertices.T

        # Plotting the 3D Mesh
        from skfem.visuals.matplotlib import draw, draw_mesh3d

        ax = draw(mesh)
        p = draw_mesh3d(mesh, ax=ax)
        p.show

        if K == 1:
            element = fem.ElementTetP1()

        else:
            element = fem.ElementTetP2()

        basis = fem.CellBasis(mesh, element)

        # Setting up integrals
        from skfem.helpers import dot

        @fem.BilinearForm
        def assembly(u, v, _):
            return dot(u, v)

        self.assembly_matrix = assembly.assemble(basis).todense()

        # We wish to compute the tetrahedral elements for all 6MNL tetrahedra:
        tetrahedra = []
        # Pick four points of tetrahedral elements:

        for i in range(L - 1):
            for j in range(M - 1):
                for k in range(N - 1):

                    # We know that each rectangular prism in the mesh has
                    # six tetrahedra lying in it

                    # There are MNL rectangular prisms in the grid.
                    # Each quad corresponds to ijk
                    # Pick prism:

                    b_l_1 = j + k * L * M + i * M
                    b_r_1 = j + 1 + k * L * M + i * M
                    b_l_2 = j + M + k * L * M + i * M
                    b_r_2 = j + M + 1 + k * L * M + i * M

                    t_l_1 = j + M * L + k * L * M + i * M
                    t_r_1 = j + M * L + 1 + k * L * M + i * M
                    t_l_2 = j + M * L + M + k * L * M + i * M
                    t_r_2 = j + M * L + M + 1 + k * L * M + i * M

                    # Form the rectangular prisms

                    rectangular_prism_vertices = np.zeros([8, 3])
                    rectangular_prism_vertices[0] = vertices[b_l_1]
                    rectangular_prism_vertices[1] = vertices[b_r_1]
                    rectangular_prism_vertices[2] = vertices[b_l_2]
                    rectangular_prism_vertices[3] = vertices[b_r_2]
                    rectangular_prism_vertices[4] = vertices[t_l_1]
                    rectangular_prism_vertices[5] = vertices[t_r_1]
                    rectangular_prism_vertices[6] = vertices[t_l_2]
                    rectangular_prism_vertices[7] = vertices[t_r_2]

                    # Form five tetrahedra out of rectangular prisms.
                    # Basing this on the way MeshTet functions

                    tetrahedra = []
                    tetrahedron_1_vertices = np.zeros([4, 3])
                    tetrahedron_2_vertices = np.zeros([4, 3])
                    tetrahedron_3_vertices = np.zeros([4, 3])
                    tetrahedron_4_vertices = np.zeros([4, 3])
                    tetrahedron_5_vertices = np.zeros([4, 3])

                    tetrahedron_1_vertices[0] = rectangular_prism_vertices[0]
                    tetrahedron_1_vertices[1] = rectangular_prism_vertices[1]
                    tetrahedron_1_vertices[2] = rectangular_prism_vertices[2]
                    tetrahedron_1_vertices[3] = rectangular_prism_vertices[4]

                    tetrahedron_2_vertices[0] = rectangular_prism_vertices[1]
                    tetrahedron_2_vertices[1] = rectangular_prism_vertices[2]
                    tetrahedron_2_vertices[2] = rectangular_prism_vertices[3]
                    tetrahedron_2_vertices[3] = rectangular_prism_vertices[7]

                    tetrahedron_3_vertices[0] = rectangular_prism_vertices[4]
                    tetrahedron_3_vertices[1] = rectangular_prism_vertices[7]
                    tetrahedron_3_vertices[2] = rectangular_prism_vertices[5]
                    tetrahedron_3_vertices[3] = rectangular_prism_vertices[1]

                    tetrahedron_4_vertices[0] = rectangular_prism_vertices[4]
                    tetrahedron_4_vertices[1] = rectangular_prism_vertices[6]
                    tetrahedron_4_vertices[2] = rectangular_prism_vertices[7]
                    tetrahedron_4_vertices[3] = rectangular_prism_vertices[2]

                    tetrahedron_5_vertices[0] = rectangular_prism_vertices[4]
                    tetrahedron_5_vertices[1] = rectangular_prism_vertices[1]
                    tetrahedron_5_vertices[2] = rectangular_prism_vertices[7]
                    tetrahedron_5_vertices[3] = rectangular_prism_vertices[2]

                    # Grabbing the six tetrahedra in each Rectangular prism:
                    tetrahedron1 = TetrahedronFiniteElement(tetrahedron_1_vertices, K=K)
                    tetrahedron2 = TetrahedronFiniteElement(tetrahedron_2_vertices, K=K)
                    tetrahedron3 = TetrahedronFiniteElement(tetrahedron_3_vertices, K=K)
                    tetrahedron4 = TetrahedronFiniteElement(tetrahedron_4_vertices, K=K)
                    tetrahedron5 = TetrahedronFiniteElement(tetrahedron_5_vertices, K=K)
                    tetrahedra.append(tetrahedron1)
                    tetrahedra.append(tetrahedron2)
                    tetrahedra.append(tetrahedron3)
                    tetrahedra.append(tetrahedron4)
                    tetrahedra.append(tetrahedron5)
                    self.vertices = vertices
                    self.tetrahedra = tetrahedra
                    # Setup quadrature points and weights
                    # for numerical integration using scikit-fem

                    [integration_points, weights] = fem.quadrature.get_quadrature(
                        element, K
                    )

                    weights = 6 * weights

                    integration_points = integration_points.T

                    # We want to add a 4th column to the transpose of the
                    # integration_points, where it is one minus the sum of the
                    # first three rows.

                    new_col = np.zeros([integration_points.shape[0], 1])

                    for i in range(integration_points.shape[0]):
                        new_col[i] = (
                            1
                            - integration_points[i, 0]
                            - integration_points[i, 1]
                            - integration_points[i, 2]
                        )

                    # Add new column

                    integration_points = np.append(integration_points, new_col, axis=1)

                    # Integration points, weights, and number of integration points
                    self.integration_points = np.array(integration_points)
                    self.weights = np.array(weights)
                    self.nquad = self.integration_points.shape[0]

    def get_barycentric_coordinates(self, rho_theta_zeta, K):
        """Gets the barycentric coordinates, given a mesh in rho, theta, zeta.

        Parameters
        ----------
        rho_theta_zeta : 3D ndarray, shape (num_points, 3)
            Set of points for which we want to find the tetrahedra that
            they lie inside of in the mesh.
        K : Order of the finite element


        Returns
        -------
        coordinate_matrix: Matrix of volume coordinates for mesh
        (rho_theta_zeta, Q)
        """
        # nodes should be (4,3)

        tetrahedra_location = self.find_tetrahedron_corresponding_to_points(
            rho_theta_zeta
        )[0]

        # Initialize coordinate matrix, shape (num_points,4)
        coordinate_matrix = np.zeros(rho_theta_zeta[0], 4)
        # Looping through points

        for i in range(rho_theta_zeta.shape[0]):
            # grab_tetrahedron corresponding to point
            idx = tetrahedra_location[i]
            tetrahedron_with_point = self.tetrahedra[idx]
            nodes = tetrahedron_with_point
            A = np.ones((4, 4))
            for j in range(3):
                for l in range(4):
                    A[j][l] = nodes[l][j]

            X_vec = np.array(
                [
                    [rho_theta_zeta[i][0]],
                    [rho_theta_zeta[i][1]],
                    [rho_theta_zeta[i][2]],
                    [1],
                ]
            )
            coordinate_matrix[i] = np.dot((np.linalg.inv(A)), X_vec)

        return coordinate_matrix

    def return_quadrature_points(self):
        """Get quadrature points for numerical integration over the mesh.

        Returns
        -------
        quadrature points: 2D ndarray, shape (nquad * 5MLN, 4)
            Points in (rho, theta, zeta) representing the quadrature point
            locations for integration, return in barycentric coordinates. Then
            ultimately converted to real-space coordinates
        """
        nquad = self.nquad
        quadrature_points = np.zeros((self.I_5LMN * nquad, 3))

        M = []
        for tetrahedron in self.tetrahedra:

            vertices = tetrahedron.vertices

            D = np.array(
                [vertices[0, 0], vertices[0, 1], vertices[0, 2], 1],
                [vertices[1, 0], vertices[1, 1], vertices[1, 2], 1],
                [vertices[2, 0], vertices[2, 1], vertices[2, 2], 1],
                [vertices[3, 0], vertices[3, 1], vertices[3, 2], 1],
            )

            six_vol = np.linalg.det(D)

            # Transform points

            integration_points = self.integration_points[:, :3]

            v = vertices

            # Transform points

            A = np.array(
                [
                    [v[2, 0] - v[0, 0], v[1, 0] - v[0, 0], v[3, 0] - v[0, 0]],
                    [v[2, 1] - v[0, 1], v[1, 1] - v[0, 1], v[3, 1] - v[0, 1]],
                    [v[2, 2] - v[0, 2], v[1, 2] - v[0, 2], v[3, 2] - v[0, 2]],
                ]
            )

            new_coor = np.zeros([integration_points.shape[0], 3])

            for i in range(integration_points.shape[0]):
                new_coor[i] = np.dot(A, integration_points[i, :])

            M = np.append(M, new_coor)

        M = np.reshape([self.I_5LMN * nquad, 3])

        new_col = np.zeros([self.I_5LMN * nquad, 1])

        for i in range(self.I_5LMN * nquad):
            new_col[i] = 1 - M[i, 0] - M[i, 1] - M[i, 2]

        # Add new column
        quadrature_points = np.append(M, new_col, axis=1)
        
        q_p = quadrature_points 
        
        # Need to convert to real-space coordinates. quadrature_points
        # has size (I_5LMN * nquad,4), where each row is a set of 
        # barycentric coordinates
        
        barycentric_points = np.zeros([self.I_5LMN * nquad, 3])
        
        for i in range(self.I_5LMN * nquad):
            for j in range(3):
                barycentric_points[i,j] = q_p[i,0]*v[0,j] + q_p[i,1]*v[1,j] + q_p[i,2]*v[2,j] + q_p[i,3]*v[3,j]
            
            
        
        quadrature_points = barycentric_points
        

        return quadrature_points

    def integrate(self, f):
        """Integrates a function over the 3D mesh in (rho, theta, zeta).

        This function allows one to integrate any set of functions of rho, theta
        zeta over the full 3D mesh. Uses numerical quadrature formula for tetrahedra
        in the barycentric coordinates.

        Parameters
        ----------
        f : 3D ndarray, shape (nquad * 5MLN, num_functions)
            Vector function defined on the L x M mesh in (rho, theta, zeta)
            that we would like to integrate component-wise with respect
            to the basis functions. For integration over the barycentric
            coordinates, we need f to be prescribed at the quadrature points
            in a barycentric coordinate system.

        Returns
        -------
        integral: 1D ndarray, shape (num_functions)
            Value of the integral over the mesh for each component of f
        """
        nquad = self.nquad
        if f.shape[1] > 1:
            integral = np.zeros(f.shape[1])
        else:
            integral = 0.0
        for i, tetrahedron in enumerate(self.tetrahedra):
            integral += np.dot(
                abs(tetrahedron.volume6) * self.weights,
                f[i * nquad : (i + 1) * nquad, :],
            )
        return integral / 6.0

    def find_tetrahedra_corresponding_to_points(self, rho_theta_zeta):
        """Given a point on the mesh, find which tetrahedron it lies inside.

        Parameters
        ----------
        rho_theta_zeta : 3D ndarray, shape (num_points, 3)
            Set of points for which we want to find the tetrahedra that
            they lie inside of in the mesh.

        Returns
        -------
        tetrahedra_indices : 1D ndarray, shape (num_points)
            Set of indices that specify the tetrahedra where each point lies.
        basis_functions : 2D ndarray, shape (num_points, Q)
            The basis functions corresponding to the tetrahedra in
            tetrahedra_indices.

        """
        tetrahedra_indices = np.zeros(rho_theta_zeta.shape[0])
        basis_functions = np.zeros((rho_theta_zeta.shape[0], self.Q))
        for i in range(rho_theta_zeta.shape[0]):
            v = rho_theta_zeta[i, :]
            for j, tetrahedron in enumerate(self.tetrahedra):
                v1 = self.tetrahedra.vertices[0, :]
                v2 = self.tetrahedra.vertices[1, :]
                v3 = self.tetrahedra.vertices[2, :]
                v4 = self.tetrahedra.vertices[3, :]
                P = self.tetrahedra_indices[i]
                D0 = np.array(
                    [
                        [v1[0], v1[1], v1[2], 1],
                        [v2[0], v2[1], v2[2], 1],
                        [v3[0], v3[1], v3[2], 1],
                        [v4[0], v4[1], v4[2], 1],
                    ]
                )
                D1 = np.array(
                    [
                        [P[0], P[1], P[2], 1],
                        [v2[0], v2[1], v2[2], 1],
                        [v3[0], v3[1], v3[2], 1],
                        [v4[0], v4[1], v4[2], 1],
                    ]
                )
                D2 = np.array(
                    [
                        [v1[0], v1[1], v1[2], 1],
                        [P[0], P[1], P[2], 1],
                        [v3[0], v3[1], v3[2], 1],
                        [v4[0], v4[1], v4[2], 1],
                    ]
                )
                D3 = np.array(
                    [
                        [v1[0], v1[1], v1[2], 1],
                        [v2[0], v2[1], v2[2], 1],
                        [P[0], P[1], P[2], 1],
                        [v4[0], v4[1], v4[2], 1],
                    ]
                )
                D4 = np.array(
                    [
                        [v1[0], v1[1], v1[2], 1],
                        [v2[0], v2[1], v2[2], 1],
                        [v3[0], v3[1], v3[2], 1],
                        [P[0], P[1], P[2], 1],
                    ]
                )

                Det0 = np.linalg.det(D0)
                Det1 = np.linalg.det(D1)
                Det2 = np.linalg.det(D2)
                Det3 = np.linal.det(D3)
                Det4 = np.linalg.det(D4)

                # Check whether point lies inside tetrahedra:

                if (
                    np.sign(Det0) == np.sign(Det1)
                    and np.sign(Det0) == np.sign(Det2)
                    and np.sign(Det0) == np.sign(Det3)
                    and np.sign(Det0) == np.sign(Det4)
                ):
                    tetrahedra_indices[i] = j
                    basis_functions[i, :], _ = self.tetrahedra.get_basis_functions(v)
        return tetrahedra_indices, basis_functions


class FiniteElementMesh2D:
    """Class representing a 2D mesh in (rho, theta).

    This class represents a set of I_2ML = 2ML triangles obtained by tessellation
    of a UNIFORM rectangular L x M mesh in the (rho, theta) plane.
    The point of this class is to pre-define all the triangles and their
    associated basis functions so that, given a new point (rho_i, theta_i),
    we can quickly return which triangle contains this point & its associated
    basis functions.

    Parameters
    ----------
    L : int
        Number of mesh points in the rho direction.
    M : int
        Number of mesh points in the theta direction.
    K: integer
        The order of the finite elements to use
    """

    def __init__(self, L, M, K=1):
        self.M = M
        self.L = L
        self.I_2ML = 2 * (M - 1) * (L - 1)
        self.Q = int((K + 1) * (K + 2) / 2)
        self.K = K

        # rho_theta mesh
        # Go back later to fix visualization
        # Ensuring 2ML triangles
        mesh = fem.MeshLine(np.linspace(0, 1, L)) * fem.MeshLine(
            np.linspace(0, 2 * np.pi, M)
        )

        # Turn squares into triangles
        mesh = mesh.to_meshtri()

        vertices = mesh.doflocs

        # Turning vertices into shape(number of points, number of coordinates)
        vertices = vertices.T

        # Vertices are enumerated as starting at the bottom, going up M points, and then
        # starting at the bottom again and repeating this process.

        # Plotting the 2D Mesh
        from skfem.visuals.matplotlib import draw, draw_mesh2d

        ax = draw(mesh)
        p = draw_mesh2d(mesh, ax=ax)
        p.show

        if K == 1:
            element = fem.ElementTriP1()

        else:
            element = fem.ElementTriP2()

        basis = fem.CellBasis(mesh, element)

        # record the nodal assembly matrix if need to check these integrals
        from skfem.helpers import dot

        @fem.BilinearForm
        def assembly(u, v, _):
            return dot(u, v)

        self.assembly_matrix = assembly.assemble(basis).todense()

        # Will fix this next section later
        # Compute the triangle elements for all 2ML triangles
        triangles = []
        for i in range(L - 1):
            for j in range(M - 1):

                # Deal with the periodic boundary conditions...??

                # There are ML quadrilaterals in the grid. Each quad corresponds to ij
                # Pick quad:

                b_l = i + j + 1 + i * (M - 1)
                b_r = b_l + M
                t_l = b_l + 1
                t_r = b_r + 1

                # Form the triangles, want vertices to have shape (3,2)

                triangle_1_vertices = np.zeros([3, 2])
                triangle_1_vertices[0, 0] = vertices[b_l - 1, 0]
                triangle_1_vertices[0, 1] = vertices[b_l - 1, 1]

                triangle_1_vertices[1, 0] = vertices[t_l - 1, 0]
                triangle_1_vertices[1, 1] = vertices[t_l - 1, 1]

                triangle_1_vertices[2, 0] = vertices[b_r - 1, 0]
                triangle_1_vertices[2, 1] = vertices[b_r - 1, 1]

                triangle_2_vertices = np.zeros([3, 2])
                triangle_2_vertices[0, 0] = vertices[b_r - 1, 0]
                triangle_2_vertices[0, 1] = vertices[b_r - 1, 1]

                triangle_2_vertices[1, 0] = vertices[t_l - 1, 0]
                triangle_2_vertices[1, 1] = vertices[t_l - 1, 1]

                triangle_2_vertices[2, 0] = vertices[t_r - 1, 0]
                triangle_2_vertices[2, 1] = vertices[t_r - 1, 1]

                # Grabbing the two triangles in each quadrilateral:
                triangle1 = TriangleFiniteElement(triangle_1_vertices, K=K)
                triangle2 = TriangleFiniteElement(triangle_2_vertices, K=K)
                triangles.append(triangle1)
                triangles.append(triangle2)
        self.vertices = vertices
        self.triangles = triangles

        # Setup quadrature points and weights for numerical integration using scikit-fem

        # K = 1 still requires quadrature rule with 3 points so that the
        # assembly matrix integrals are nonsingular. Going to automatically bump up
        # quadrature order.
        [integration_points, weights] = fem.quadrature.get_quadrature(
            element, 3 * K + 1
        )
        weights = 2 * weights

        integration_points = integration_points.T

        # We want to add a 3rdcolumn to the tranpose of the integration_points,
        # where it is one minus the sum of the first two rows.

        new_col = np.zeros([integration_points.shape[0], 1])

        for i in range(integration_points.shape[0]):
            new_col[i] = 1 - integration_points[i, 0] - integration_points[i, 1]

        # Add new column

        integration_points = np.append(integration_points, new_col, axis=1)

        # Integration points, weights, and number of integration points
        self.integration_points = np.array(integration_points)
        self.weights = np.array(weights)
        self.nquad = self.integration_points.shape[0]

    def get_barycentric_coordinates(self, rho_theta, K=1):
        """Gets the barycentric coordinates on rho_theta mesh.

        Return the triangle basis functions,
        evaluated at the 2D rho and theta mesh points.

        Parameters
        ----------
        rho_theta : 2D ndarray, shape (nrho * ntheta, 2)
        Coordinates of the original grid, lying inside this triangle.

        Returns
        -------
        coordinate_matrix: (rho_theta, Q)
        """
        triangle_location = self.find_triangles_corresponding_to_points(rho_theta)[0]

        # Initialize coordinate matrix, shape (num_points,3)
        coordinate_matrix = np.zeros(rho_theta[0], 3)
        # Looping through points

        for i in range(rho_theta.shape[0]):
            # grab triangle corresponding to point
            idx = triangle_location[i]
            triangle_with_point = self.triangle[idx]
            nodes = triangle_with_point
            A = np.ones((3, 3))
            for j in range(2):
                for l in range(3):
                    A[j][l] = nodes[l][j]

            X_vec = np.array([[rho_theta[i][0]], [rho_theta[i][1]], [1]])
            coordinate_matrix[i] = np.dot((np.linalg.inv(A)), X_vec)

        return coordinate_matrix

    def plot_triangles(self, plot_quadrature_points=False):
        """Plot all the triangles in the 2D mesh tessellation."""
        plt.figure(100)
        for i, triangle in enumerate(self.triangles):
            triangle.plot_triangle()
        if plot_quadrature_points:
            quadpoints = self.return_quadrature_points()
            for i in range(self.Q):
                plt.subplot(1, self.Q, i + 1)
                plt.plot(quadpoints[:, 0], quadpoints[:, 1], "ko")
        plt.show()

    def find_triangles_corresponding_to_points(self, rho_theta):
        """Given a point on the mesh, find which triangle it lies inside.

        Parameters
        ----------
        rho_theta : 2D ndarray, shape (num_points, 2)
            Set of points for which we want to find the triangles that
            they lie inside of in the mesh.

        Returns
        -------
        triangle_indices : 1D ndarray, shape (num_points)
            Set of indices that specific the triangles where each point lies.
        basis_functions : 2D ndarray, shape (num_points, Q)
            The basis functions corresponding to the triangles in
            triangle_indices.
        """
        triangle_indices = np.zeros(rho_theta.shape[0])
        basis_functions = np.zeros((rho_theta.shape[0], self.I_2ML * self.Q))
        for i in range(rho_theta.shape[0]):
            v = rho_theta[i, :]
            for j, triangle in enumerate(self.triangles):
                v1 = triangle.vertices[0, :]
                v2 = triangle.vertices[1, :] - triangle.vertices[0, :]
                v3 = triangle.vertices[2, :] - triangle.vertices[0, :]
                a = (np.cross(v, v3) - np.cross(v1, v3)) / np.cross(v2, v3)
                b = -(np.cross(v, v2) - np.cross(v1, v2)) / np.cross(v2, v3)
                if a >= 0 and b >= 0 and (a + b) <= 1:
                    triangle_indices[i] = j
                    basis_functions[i, j * self.Q : (j + 1) * self.Q], _ = (
                        triangle.get_basis_functions(v.reshape(1, 2))
                    )
                    break  # found the right triangle, so break out of j loop

        # Some possible code to vectorize this function in future
        # v = rho_theta
        # v1 = self.vertices[0, :, :]  # shape (num_triangles, 2)
        # v2 = self.vertices[1, :, :]
        # v3 = self.vertices[2, :, :]
        # v_v2_cross = np.outer(v[:, 0], v2[:, 1])
        # v_v3_cross = np.outer(v[:, 0], v3[:, 1])
        # v_v1_cross = np.outer(v[:, 0], v1[:, 1])
        # v1_v2_cross = v1[:, 0] * v2[:, 1]
        # v1_v3_cross = v1[:, 0] * v3[:, 1]
        # v2_v3_cross = v2[:, 0] * v3[:, 1]
        # a = (v_v3_cross - v1_v3_cross) / v2_v3_cross
        # b = -(v_v2_cross - v1_v2_cross) / v2_v3_cross
        # triangle_indices = np.where(np.logical_and(np.logical_and(
        #     a >= 0, b >= 0), (a + b) <= 1))
        # basis_functions, _ = (
        #         self.get_basis_functions(v)
        #     )
        return triangle_indices, basis_functions

    def return_quadrature_points(self):
        """Get quadrature points for numerical integration over the mesh.

        Returns
        -------
        quadrature points: 2D ndarray, shape (nquad * 2ML, 2)
            Points in (rho, theta) representing the quadrature point
            locations for integration, return in real-space coordinates.

        """
        nquad = self.nquad
        quadrature_points = np.zeros((self.I_2ML * nquad, 2))
        q = 0
        for triangle in self.triangles:
            for i in range(nquad):
                A = np.array(
                    [
                        [triangle.b[0], triangle.c[0]],
                        [triangle.b[1], triangle.c[1]],
                        [
                            triangle.b[0] + triangle.b[1] + triangle.b[2],
                            triangle.c[0] + triangle.c[1] + triangle.c[2],
                        ],
                    ]
                )
                b = np.zeros(3)
                b[:2] = triangle.area2 * self.integration_points[i, :2] - triangle.a[:2]
                b[2] = triangle.area2 - triangle.a[0] - triangle.a[1] - triangle.a[2]
                quadrature_points[q, :], _, _, _ = np.linalg.lstsq(A, b)
                q = q + 1
        return quadrature_points

    def integrate(self, f):
        """Integrates a function over the 2D mesh in (rho, theta).

        This function allows one to integrate any set of functions of rho,
        theta over the full 2D L x M mesh. Uses numerical quadrature
        formula for triangles in the barycentric coordinates. Note that
        for K = 1, a single-point quadrature in the triangle is adequate.

        Parameters
        ----------
        f : 2D ndarray, shape (nquad * 2ML, num_functions)
            Vector function defined on the L x M mesh in (rho, theta)
            that we would like to integrate component-wise with respect
            to the basis functions. For integration over the barycentric
            coordinates, we need f to be prescribed at the quadrature points
            in a barycentric coordinate system.

        Returns
        -------
        integral: 1D ndarray, shape (num_functions)
            Value of the integral over the mesh for each component of f.

        """
        nquad = self.nquad
        if f.shape[1] > 1:
            integral = np.zeros(f.shape[1])
        else:
            integral = 0.0
        for i, triangle in enumerate(self.triangles):
            integral += np.dot(
                abs(triangle.area2) * self.weights,
                f[i * nquad : (i + 1) * nquad, :],
            )
        return integral / 2.0


class TetrahedronFiniteElement:
    """Class representing a triangle in a 3D grid of finite elements.

    Parameters
    ----------
    vertices: array-like, shape (4,3)
        The four vertices of the tetrahedron in (rho_i,theta_i, zeta_i)
    K: integer
        The order of the finite elements to use, which gives (K+1)(K+2)(K+3) / 6
        basis functions.
    """

    # Test comment
    def __init__(self, vertices, K=1):
        self.vertices = vertices
        self.Q = int(((K + 1) * (K + 2) * (K + 3)) / 6)
        self.K = K

        D = np.array(
            [vertices[0, 0], vertices[0, 1], vertices[0, 2], 1],
            [vertices[1, 0], vertices[1, 1], vertices[1, 2], 1],
            [vertices[2, 0], vertices[2, 1], vertices[2, 2], 1],
            [vertices[3, 0], vertices[3, 1], vertices[3, 2], 1],
        )

        self.volume6 = np.linalg.det(D)
        self.d = np.linalg.det(D)

        # Computing edge lengths

        e1 = np.sqrt(
            (vertices[0, 0] - vertices[1, 0]) ** 2
            + (vertices[0, 1] - vertices[1, 1]) ** 2
            + (vertices[0, 2] - vertices[1, 2]) ** 2
        )

        e2 = np.sqrt(
            (vertices[0, 0] - vertices[2, 0]) ** 2
            + (vertices[0, 1] - vertices[2, 1]) ** 2
            + (vertices[0, 2] - vertices[2, 2]) ** 2
        )

        e3 = np.sqrt(
            (vertices[0, 0] - vertices[3, 0]) ** 2
            + (vertices[0, 1] - vertices[3, 1]) ** 2
            + (vertices[0, 2] - vertices[3, 2]) ** 2
        )

        e4 = np.sqrt(
            (vertices[1, 0] - vertices[2, 0]) ** 2
            + (vertices[1, 1] - vertices[2, 1]) ** 2
            + (vertices[1, 2] - vertices[2, 2]) ** 2
        )

        e5 = np.sqrt(
            (vertices[1, 0] - vertices[3, 0]) ** 2
            + (vertices[1, 1] - vertices[3, 1]) ** 2
            + (vertices[1, 2] - vertices[3, 2]) ** 2
        )

        e6 = np.sqrt(
            (vertices[2, 0] - vertices[3, 0]) ** 2
            + (vertices[2, 1] - vertices[3, 1]) ** 2
            + (vertices[2, 2] - vertices[3, 2]) ** 2
        )

        # Do we need angles here?

        # We construct equally spaced nodes for order K tetrahedron,
        # which gives Q nodes.
        nodes = []

        # Start with vertices of tetrahedron

        node_mapping = []
        for i in range(4):
            nodes.append(vertices[i, :])
            node_tuple = [0, 0, 0, 0]
            node_tuple[i] = K
            node_mapping.append(node_tuple)

        # If K = 1, we are done. Else, will handle K = 2 separately

        if K == 2:
            # We create one additional midpoint node
            # on each of the six tetrahedron edges
            # for a total of 6 more nodes
            # Checked that the order matches sci-kit

            midpoints = np.zeros([6, 3])

            for i in range(2):
                midpoints[i] = (vertices[i, :] + vertices[i + 1, :]) / 2

            midpoints[5] = (vertices[2, :] + vertices[3, :]) / 2
            midpoints[2] = (vertices[0, :] + vertices[2, :]) / 2
            midpoints[3] = (vertices[0, :] + vertices[3, :]) / 2
            midpoints[4] = (vertices[1, :] + vertices[3, :]) / 2

            # Now, we add all of these nodes to our list of nodes

            for j in range(6):
                nodes.append(midpoints[j, :])
            node_tuple = [K, K, K, K]
            node_mapping.append(node_tuple)

            # We next place the interior nodes in the tetrahedra.

        if K == 3:
            # We place the edge nodes first, spaced out by thirds:

            one_third = np.zeros([6, 3])
            two_thirds = np.zeros([6, 3])

            for i in range(3):
                one_third[i, :] = (1 / 3) * vertices[i + 1, :] + (2 / 3) * vertices[
                    i, :
                ]
                two_thirds[i, :] = (2 / 3) * vertices[i + 1, :] + (1 / 3) * vertices[
                    i, :
                ]

            # p0p2

            one_third[3, :] = (1 / 3) * vertices[2, :] + (2 / 3) * vertices[0, :]
            two_thirds[3, :] = (2 / 3) * vertices[2, :] + (1 / 3) * vertices[0, :]

            # p0p3

            one_third[4, :] = (1 / 3) * vertices[3, :] + (2 / 3) * vertices[0, :]
            two_thirds[4, :] = (2 / 3) * vertices[3, :] + (1 / 3) * vertices[0, :]

            # p1p3

            one_third[5, :] = (1 / 3) * vertices[3, :] + (2 / 3) * vertices[1, :]
            two_thirds[5, :] = (2 / 3) * vertices[3, :] + (1 / 3) * vertices[1, :]

            for i in range(6):
                nodes.append(one_third[i, :])
                nodes.append(two_thirds[i, :])
                node_tuple = [K, K, K, K]
                # Not entirely sure the usage of node_tuple

                node_mapping.append(node_tuple)

            # Next we place the interior nodes. We use the barycenter of
            # each of the four faces.

            barycenters = np.zeros([4, 3])

            # p0p1p2 and p1p2p3

            for j in range(2):
                barycenters[j, :] = (
                    vertices[j, :] + vertices[j + 1, :] + vertices[j + 2, :]
                ) / 3

            # p0p2p3

            barycenters[2, :] = (vertices[0, :] + vertices[2, :] + vertices[3, :]) / 3

            # p0p1p3

            barycenters[3, :] = (vertices[0, :] + vertices[1, :] + vertices[3, :]) / 3

            for j in range(4):
                nodes.append(barycenters[j, :])
                node_tuple = [K, K, K, K]
                node_mapping.append(node_tuple)

        self.nodes = np.array(nodes)
        self.eta_nodes, _ = self.get_barycentric_coordinates(self.nodes)
        self.basis_functions_nodes, _ = self.get_basis_functions(self.nodes)

        if K == 1:
            assert np.allclose(self.eta_nodes, self.basis_functions_nodes)

        # Check basis functions vanish at all nodes except the associated node.
        for i in range(self.Q):
            assert np.allclose(self.basis_functions_nodes[i, i], 1.0)
            for j in range(self.Q):
                if i != j:
                    assert np.allclose(self.basis_functions_nodes[i, j], 0.0)

        # Check we have the same number of basis functions as nodes.
        assert self.nodes.shape[0] == self.Q

    def get_barycentric_coordinates(self, rho_theta_zeta):
        """Gets the barycentic coordinates, given a mesh in rho, theta, zeta.

        Parameters
        ----------
        rho_theta_zeta = 3D ndarray, shape(nrho * ntheta * nzeta, 3)
            Coordinates of the origininal grid, lying inside this tetrahedron.

        Returns
        -------
        eta_final = 3D array, shape (nrho * ntheta * nzeta, 4)
            Barycentric coordinates defined by the tetrahedron and evaluated at the
            points (rho, theta, zeta)
        """
        eta = np.zeros(rho_theta_zeta.shape[[0], 4])

        for i in range(rho_theta_zeta.shape[0]):

            g_v = self.vertices

            D_1 = np.array(
                [rho_theta_zeta[i][0], rho_theta_zeta[i][1], rho_theta_zeta[i][2], 1],
                [g_v[1][0], g_v[1][1], g_v[1][2], 1],
                [g_v[2][0], g_v[2][1], g_v[2][2], 1],
                [g_v[3][0], g_v[3][1], g_v[3][2], 1],
            )

            D_2 = np.array(
                [g_v[0][0], g_v[0][1], g_v[0][2], 1],
                [rho_theta_zeta[i][0], rho_theta_zeta[i][1], rho_theta_zeta[i][2], 1],
                [g_v[2][0], g_v[2][1], g_v[2][2], 1],
                [g_v[3][0], g_v[3][1], g_v[3][2], 1],
            )

            D_3 = np.array(
                [g_v[0][0], g_v[0][1], g_v[0][2], 1],
                [g_v[1][0], g_v[1][1], g_v[1][2], 1],
                [rho_theta_zeta[i][0], rho_theta_zeta[i][1], rho_theta_zeta[i][2], 1],
                [g_v[3][0], g_v[3][1], g_v[3][2], 1],
            )

            D_4 = np.array(
                [g_v[0][0], g_v[0][1], g_v[0][2], 1],
                [g_v[1][0], g_v[1][1], g_v[1][2], 1],
                [g_v[2][0], g_v[2][1], g_v[2][2], 1],
                [rho_theta_zeta[i][0], rho_theta_zeta[i][1], rho_theta_zeta[i][2], 1],
            )

            d_1 = np.linalg.det(D_1)
            d_2 = np.linalg.det(D_2)
            d_3 = np.linalg.det(D_3)
            d_4 = np.linalg.det(D_4)

            eta_1 = d_1 / self.d
            eta_2 = d_2 / self.d
            eta_3 = d_3 / self.d
            eta_4 = d_4 / self.d

            # Check that all points are indeed inside the tetrahedron
            eta_final = []
            rho_theta_zeta_in_tetrahedron = []
            for j in range(4):
                if eta_1[j] < 0 or eta_2[j] < 0 or eta_3[j] < 0 or eta_4[j] < 0:
                    warnings.warn()
                    "Found rho_theta_zeta points outside the tetrahedron ... "
                    "Not using these points to evaluate the barycentric "
                    "coordinates."

                else:
                    eta[j][0] = eta_1
                    eta[j][1] = eta_2
                    eta[j][2] = eta_3
                    eta[j][3] = eta_4

                eta_final.append(eta[j, :])
                rho_theta_zeta_in_tetrahedron.append(rho_theta_zeta[j, :])

        return np.array(eta_final), np.array(rho_theta_zeta_in_tetrahedron)

    def get_basis_functions(self, rho_theta_zeta):
        """
        Gets the barycentric basis functions.

        Return the tetrahedron basis functions, evaluated at the 3D rho, theta,
        and zeta mesh points provided to the function.


        Parameters
        ----------
        rho_theta_zeta = 3D ndarray, shape(nrho * ntheta * nzeta, 3)
            Coordinates of the origininal grid, lying inside this tetrahedron.

        Returns
        -------
        psi_q = (rho_theta_zeta, Q)

        """
        eta, rho_theta_zeta_in_tetrahedron = self.get_barycentric_coordinates(
            rho_theta_zeta
        )
        rho_theta_zeta = rho_theta_zeta_in_tetrahedron
        K = self.K

        # Now, we have the rho_theta_zeta points that lie in the tetrahedron.
        # Recall that eta_nodes is the set of barycentric coordinates for the points
        # lying in the specific tetrahedron.
        basis_functions = np.zeros((rho_theta_zeta.shape[0], self.Q))

        # At each of the rho_theta_zeta points that lie in the given tetrahedron,
        # we wish to evaluate the Q = (K+1)(K+2)(K+3)/6 basis functions.
        K = self.K

        # Compute the vertex basis functions first. This will be the first 4 columns
        # of basis_functions. There are hard-coded from book.

        if K == 1:
            for i in rho_theta_zeta.shape[0]:
                for j in range(4):
                    basis_functions[i, j] = self.eta_nodes[i, j]
        # Next, we deal with the edge nodes. This will provide an additional 6 columns.

        if K == 2:
            for i in rho_theta_zeta.shape[0]:
                for j in range(4):
                    basis_functions[i, j] = self.eta_nodes[i, j] * (
                        2 * self.eta_nodes[i, j] - 1
                    )

            for i in rho_theta_zeta.shape[0]:
                basis_functions[i, 4] = 4 * self.eta_nodes[i, 1] * self.eta_nodes[i, 0]
                basis_functions[i, 5] = 4 * self.eta_nodes[i, 1] * self.eta_nodes[i, 2]
                basis_functions[i, 6] = 4 * self.eta_nodes[i, 0] * self.eta_nodes[i, 2]
                basis_functions[i, 7] = 4 * self.eta_nodes[i, 0] * self.eta_nodes[i, 3]
                basis_functions[i, 8] = 4 * self.eta_nodes[i, 1] * self.eta_nodes[i, 3]
                basis_functions[i, 8] = 4 * self.eta_nodes[i, 1] * self.eta_nodes[i, 3]
                basis_functions[i, 9] = 4 * self.eta_nodes[i, 2] * self.eta_nodes[i, 3]

        if K == 3:
            for i in rho_theta_zeta.shape[0]:
                for j in range(4):
                    basis_functions[i, j] = (
                        (1 / 2)
                        * self.eta_nodes[i, j]
                        * (3 * self.eta_nodes[i, j] - 1)
                        * (3 * self.eta_nodes[i, j] - 2)
                    )
            for i in rho_theta_zeta.shape[0]:
                # Edge nodes
                basis_functions[i, 4] = (
                    (9 / 2)
                    * self.eta_nodes[i, 0]
                    * self.eta_nodes[i, 1]
                    * (3 * self.eta_nodes[i, 0] - 1)
                )
                basis_functions[i, 5] = (
                    (9 / 2)
                    * self.eta_nodes[i, 0]
                    * self.eta_nodes[i, 1]
                    * (3 * self.eta_nodes[i, 1] - 1)
                )
                basis_functions[i, 6] = (
                    (9 / 2)
                    * self.eta_nodes[i, 1]
                    * self.eta_nodes[i, 2]
                    * (3 * self.eta_nodes[i, 1] - 1)
                )
                basis_functions[i, 7] = (
                    (9 / 2)
                    * self.eta_nodes[i, 1]
                    * self.eta_nodes[i, 2]
                    * (3 * self.eta_nodes[i, 2] - 1)
                )
                basis_functions[i, 8] = (
                    (9 / 2)
                    * self.eta_nodes[i, 2]
                    * self.eta_nodes[i, 3]
                    * (3 * self.eta_nodes[i, 2] - 1)
                )
                basis_functions[i, 9] = (
                    (9 / 2)
                    * self.eta_nodes[i, 2]
                    * self.eta_nodes[i, 3]
                    * (3 * self.eta_nodes[i, 3] - 1)
                )
                basis_functions[i, 10] = (
                    (9 / 2)
                    * self.eta_nodes[i, 0]
                    * self.eta_nodes[i, 2]
                    * (3 * self.eta_nodes[i, 0] - 1)
                )
                basis_functions[i, 11] = (
                    (9 / 2)
                    * self.eta_nodes[i, 0]
                    * self.eta_nodes[i, 2]
                    * (3 * self.eta_nodes[i, 2] - 1)
                )
                basis_functions[i, 12] = (
                    (9 / 2)
                    * self.eta_nodes[i, 0]
                    * self.eta_nodes[i, 3]
                    * (3 * self.eta_nodes[i, 0] - 1)
                )
                basis_functions[i, 13] = (
                    (9 / 2)
                    * self.eta_nodes[i, 0]
                    * self.eta_nodes[i, 3]
                    * (3 * self.eta_nodes[i, 3] - 1)
                )
                basis_functions[i, 14] = (
                    (9 / 2)
                    * self.eta_nodes[i, 1]
                    * self.eta_nodes[i, 3]
                    * (3 * self.eta_nodes[i, 1] - 1)
                )
                basis_functions[i, 15] = (
                    (9 / 2)
                    * self.eta_nodes[i, 1]
                    * self.eta_nodes[i, 3]
                    * (3 * self.eta_nodes[i, 3] - 1)
                )

                # Mid-face nodes
                basis_functions[i, 16] = (
                    27
                    * self.eta_nodes[i, 0]
                    * self.eta_nodes[i, 1]
                    * self.eta_nodes[i, 2]
                )
                basis_functions[i, 17] = (
                    27
                    * self.eta_nodes[i, 1]
                    * self.eta_nodes[i, 2]
                    * self.eta_nodes[i, 3]
                )
                basis_functions[i, 18] = (
                    27
                    * self.eta_nodes[i, 0]
                    * self.eta_nodes[i, 2]
                    * self.eta_nodes[i, 3]
                )
                basis_functions[i, 19] = (
                    27
                    * self.eta_nodes[i, 0]
                    * self.eta_nodes[i, 1]
                    * self.eta_nodes[i, 3]
                )

        return basis_functions, rho_theta_zeta_in_tetrahedron


class TriangleFiniteElement:
    """Class representing a triangle in a 2D grid of finite elements.

    Parameters
    ----------
    vertices: array-like, shape(3, 2)
        The three vertices of the triangle in (theta_i, zeta_i)
    K: integer
        The order of the finite elements to use, which gives (K+1)(K+2) / 2
        basis functions.
    """

    def __init__(self, vertices, K=1):
        self.vertices = vertices
        a1 = vertices[1, 0] * vertices[2, 1] - vertices[2, 0] * vertices[1, 1]
        a2 = vertices[2, 0] * vertices[0, 1] - vertices[0, 0] * vertices[2, 1]
        a3 = vertices[0, 0] * vertices[1, 1] - vertices[1, 0] * vertices[0, 1]
        self.a = np.array([a1, a2, a3])
        b1 = vertices[1, 1] - vertices[2, 1]
        b2 = vertices[2, 1] - vertices[0, 1]
        b3 = vertices[0, 1] - vertices[1, 1]
        self.b = np.array([b1, b2, b3])
        c1 = vertices[2, 0] - vertices[1, 0]
        c2 = vertices[0, 0] - vertices[2, 0]
        c3 = vertices[1, 0] - vertices[0, 0]
        self.c = np.array([c1, c2, c3])
        self.area2 = self.vertices[:, 0] @ self.b
        self.Q = int((K + 1) * (K + 2) / 2)
        self.K = K

        # Compute the edge lengths and then the angles in the triangle
        d1 = np.sqrt(
            (vertices[1, 0] - vertices[0, 0]) ** 2
            + (vertices[1, 1] - vertices[0, 1]) ** 2
        )
        d2 = np.sqrt(
            (vertices[2, 0] - vertices[0, 0]) ** 2
            + (vertices[2, 1] - vertices[0, 1]) ** 2
        )
        d3 = np.sqrt(
            (vertices[1, 0] - vertices[2, 0]) ** 2
            + (vertices[1, 1] - vertices[2, 1]) ** 2
        )
        angle1 = np.arccos((d2**2 + d3**2 - d1**2) / (2 * d2 * d3))
        angle2 = np.arccos((d1**2 + d3**2 - d2**2) / (2 * d1 * d3))
        angle3 = np.arccos((d2**2 + d1**2 - d3**2) / (2 * d2 * d1))
        self.angles = np.array([angle1, angle2, angle3]).T

        # Going to construct equally spaced nodes for order K triangle,
        # which gives Q such nodes.
        nodes = []

        # Start with the vertices of the triangle
        node_mapping = []
        for i in range(3):
            nodes.append(vertices[i, :])
            node_tuple = [0, 0, 0]
            node_tuple[i] = K
            node_mapping.append(node_tuple)

        # If K = 1, the vertices are the only nodes for basis functions
        if K > 1:
            # Add (k-1) equally spaced nodes on each triangle edge
            # for k = 1, ..., K - 1
            # for total of 3(K - 1)K / 2 more nodes
            # This is certainly incorrect for K = 3 !!!!
            for i in range(3):
                for j in range(i + 1, 3):
                    for k in range(K - 1):
                        edge_node = (vertices[i, :] + vertices[j, :]) / K * (k + 1)
                        nodes.append(edge_node)
                        node_tuple = [0, 0, 0]
                        node_tuple[i] = k + 1
                        node_tuple[j] = k + 1
                        node_mapping.append(node_tuple)

            # Once all the edge nodes are placed, place the interior nodes
            # This is certainly incorrect for K = 3 !!!!
            for i in range(3):
                # Fill in any nodes within the triangle by drawing rays between
                # the edge nodes that are more than 1 spacing away'
                if i == 0 and K > 2:
                    for k in range(1, K - 1):
                        edge_node1 = (vertices[i, :] + vertices[1, :]) / K * (k + 1)
                        edge_node2 = (vertices[i, :] + vertices[2, :]) / K * (k + 1)
                        center_node = (edge_node1 + edge_node2) / (K - 1) * k
                        nodes.append(center_node)
                        node_tuple = [0, 0, 0]
                        node_tuple[i] = 1
                        node_tuple[j] = 1
                        node_mapping.append(node_tuple)

        self.node_mapping = np.array(node_mapping)
        self.nodes = np.array(nodes)
        self.eta_nodes, _ = self.get_barycentric_coordinates(self.nodes)
        self.basis_functions_nodes, _ = self.get_basis_functions(self.nodes)

        if K == 1:
            assert np.allclose(self.eta_nodes, self.basis_functions_nodes)

        # Check basis functions vanish at all nodes except the associated node.
        for i in range(self.Q):
            assert np.allclose(self.basis_functions_nodes[i, i], 1.0)
            for j in range(self.Q):
                if i != j:
                    assert np.allclose(self.basis_functions_nodes[i, j], 0.0)

        # Check we have the same number of basis functions as nodes.
        assert self.nodes.shape[0] == self.Q

    def get_basis_functions(self, rho_theta):
        """
        Gets the barycentric basis functions.

        Return the triangle basis functions, evaluated at the 2D rho
        and theta mesh points provided to the function.

        Parameters
        ----------
        rho_theta : 2D ndarray, shape (nrho * ntheta, 2)
            Coordinates of the original grid, lying inside this triangle.

        Returns
        -------
        psi_q : (rho_theta, Q)

        """
        eta, rho_theta_in_triangle = self.get_barycentric_coordinates(rho_theta)
        rho_theta = rho_theta_in_triangle
        K = self.K

        # Compute the vertex basis functions first
        basis_functions = np.zeros((rho_theta.shape[0], self.Q))
        for i in range(3):
            inds_x0 = np.ravel(
                np.where(
                    np.logical_not(
                        np.isclose(self.eta_nodes[:, 0], self.eta_nodes[i, 0])
                    )
                )
            )
            inds_y0 = np.ravel(
                np.where(
                    np.logical_not(
                        np.isclose(self.eta_nodes[:, 1], self.eta_nodes[i, 1])
                    )
                )
            )
            inds_z0 = np.ravel(
                np.where(
                    np.logical_not(
                        np.isclose(self.eta_nodes[:, 2], self.eta_nodes[i, 2])
                    )
                )
            )

            # take appropriate intersections of the indices depending on
            # which vertex we have.
            if len(inds_x0) >= len(inds_y0) and len(inds_x0) >= len(inds_z0):
                inds_x0_prime = np.setdiff1d(inds_x0, inds_z0)
                inds_y0_prime = np.setdiff1d(inds_y0, inds_x0)
                inds_z0_prime = np.setdiff1d(inds_z0, inds_x0)
            elif len(inds_y0) >= len(inds_x0) and len(inds_y0) >= len(inds_z0):
                inds_y0_prime = np.setdiff1d(inds_y0, inds_z0)
                inds_x0_prime = np.setdiff1d(inds_x0, inds_y0)
                inds_z0_prime = np.setdiff1d(inds_z0, inds_y0)
            elif len(inds_z0) >= len(inds_y0) and len(inds_z0) >= len(inds_x0):
                inds_z0_prime = np.setdiff1d(inds_z0, inds_x0)
                inds_y0_prime = np.setdiff1d(inds_y0, inds_z0)
                inds_x0_prime = np.setdiff1d(inds_x0, inds_z0)
            # Subtract off the node i from this list of indices
            inds_x0 = np.setdiff1d(inds_x0_prime, [0])
            inds_y0 = np.setdiff1d(inds_y0_prime, [1])
            inds_z0 = np.setdiff1d(inds_z0_prime, [2])

            # Now use these nodes to define the basis function
            node_indices = [0, 0, 0]
            node_indices[i] = K

            basis_functions[:, i] = (
                self.lagrange_polynomial(
                    eta[:, 0], self.eta_nodes[:, 0], node_indices[0], inds_x0, i
                )
                * self.lagrange_polynomial(
                    eta[:, 1], self.eta_nodes[:, 1], node_indices[1], inds_y0, i
                )
                * self.lagrange_polynomial(
                    eta[:, 2], self.eta_nodes[:, 2], node_indices[2], inds_z0, i
                )
            )
            if K == 1:
                assert np.allclose(basis_functions[:, i], eta[:, i])
            elif K == 2:
                assert np.allclose(
                    basis_functions[:, i], eta[:, i] * (2 * eta[:, i] - 1)
                )

        # Now we repeat the basis function calculation for the edge nodes
        # Note that this probably only works for K = 1 and K = 2
        # and we have not dealt with interior nodes when K = 3
        q = 3
        for node in self.node_mapping[3:, :]:
            inds_x0 = np.ravel(
                np.where(
                    np.logical_not(
                        np.isclose(self.eta_nodes[:, 0], self.eta_nodes[q, 0])
                    )
                )
            )
            inds_y0 = np.ravel(
                np.where(
                    np.logical_not(
                        np.isclose(self.eta_nodes[:, 1], self.eta_nodes[q, 1])
                    )
                )
            )
            inds_z0 = np.ravel(
                np.where(
                    np.logical_not(
                        np.isclose(self.eta_nodes[:, 2], self.eta_nodes[q, 2])
                    )
                )
            )

            # take appropriate intersections of the indices depending on
            # which vertex we have.
            if len(inds_x0) <= len(inds_y0) and len(inds_x0) <= len(inds_z0):
                inds_z0_prime = np.setdiff1d(inds_z0, inds_x0)
                inds_y0_prime = np.setdiff1d(inds_y0, inds_x0)
                inds_y0_prime = np.setdiff1d(inds_y0_prime, [1])
                inds_z0_prime = np.setdiff1d(inds_z0_prime, [2])
                inds_x0_prime = []
            elif len(inds_y0) <= len(inds_x0) and len(inds_y0) <= len(inds_z0):
                inds_z0_prime = np.setdiff1d(inds_z0, inds_y0)
                inds_x0_prime = np.setdiff1d(inds_x0, inds_y0)
                inds_x0_prime = np.setdiff1d(inds_x0_prime, [0])
                inds_z0_prime = np.setdiff1d(inds_z0_prime, [2])
                inds_y0_prime = []
            elif len(inds_z0) <= len(inds_y0) and len(inds_z0) <= len(inds_x0):
                inds_y0_prime = np.setdiff1d(inds_y0, inds_z0)
                inds_x0_prime = np.setdiff1d(inds_x0, inds_z0)
                inds_x0_prime = np.setdiff1d(inds_x0_prime, [0])
                inds_y0_prime = np.setdiff1d(inds_y0_prime, [1])
                inds_z0_prime = []
            inds_x0 = inds_x0_prime
            inds_y0 = inds_y0_prime
            inds_z0 = inds_z0_prime

            # Now use these nodes to define the basis function
            basis_functions[:, q] = (
                self.lagrange_polynomial(
                    eta[:, 0], self.eta_nodes[:, 0], node[0], inds_x0, q
                )
                * self.lagrange_polynomial(
                    eta[:, 1], self.eta_nodes[:, 1], node[1], inds_y0, q
                )
                * self.lagrange_polynomial(
                    eta[:, 2], self.eta_nodes[:, 2], node[2], inds_z0, q
                )
            )
            q += 1
        if K == 2:
            assert np.allclose(basis_functions[:, 3], 4 * eta[:, 0] * eta[:, 1])
            assert np.allclose(basis_functions[:, 4], 4 * eta[:, 2] * eta[:, 0])
            assert np.allclose(basis_functions[:, 5], 4 * eta[:, 1] * eta[:, 2])
        return basis_functions, rho_theta_in_triangle

    def lagrange_polynomial(self, eta_i, eta_nodes_i, order, inds_minus_q, q):
        """
        Computes lagrange polynomials.

        Computes the lagrange polynomial given the ith component of the
        Barycentric coordinates on a (theta, zeta) mesh, the ith component
        of the triangle nodes defined for the basis functions, the order
        of the polynomial, and the index q of which node this is.

        Parameters
        ----------
        eta_i : 1D ndarray, shape(ntheta * nzeta)
            The barycentric coordinate i defined at (theta, zeta) points.
        eta_nodes_i : 1D ndarray, shape(Q)
            The barycentric coordinate i defined at the triangle nodes.
        order : integer
            Order of the polynomial.
        q : integer
            The index of the node we are using to define the basis function.
            Options are 0, ..., Q - 1

        Returns
        -------
        lp : 1D ndarray, shape(ntheta * nzeta)
            The lagrange polynomial associated with the barycentric
            coordinate i, the polynomial order (order), and the node q.

        """
        denom = 1.0
        numerator = np.ones(len(eta_i))
        # Avoid choosing the node q associated with this basis function
        for i in inds_minus_q:
            numerator *= eta_i - eta_nodes_i[i]
            denom *= eta_nodes_i[q] - eta_nodes_i[i]
        lp = numerator / denom
        return lp

    def get_barycentric_coordinates(self, rho_theta):
        """
        Gets the barycentric coordinates, given a mesh in rho, theta.

        Parameters
        ----------
        rho_theta : 2D ndarray, shape (nrho * ntheta, 2)
            Coordinates of the original grid, lying inside this triangle.

        Returns
        -------
        eta_u: 2D array, shape (nrho * ntheta, 3)
            Barycentric coordinates defined by the triangle and evaluated
            at the points (theta, zeta).
        """
        # Get the Barycentric coordinates
        eta1 = (
            self.a[0] * np.ones(len(rho_theta[:, 0]))
            + self.b[0] * rho_theta[:, 0]
            + self.c[0] * rho_theta[:, 1]
        ) / self.area2
        eta2 = (
            self.a[1] * np.ones(len(rho_theta[:, 0]))
            + self.b[1] * rho_theta[:, 0]
            + self.c[1] * rho_theta[:, 1]
        ) / self.area2
        eta3 = (
            self.a[2] * np.ones(len(rho_theta[:, 0]))
            + self.b[2] * rho_theta[:, 0]
            + self.c[2] * rho_theta[:, 1]
        ) / self.area2

        # zero out numerical errors or weird minus signs like -0
        eta1[np.isclose(eta1, 0.0)] = 0.0
        eta2[np.isclose(eta2, 0.0)] = 0.0
        eta3[np.isclose(eta3, 0.0)] = 0.0
        eta = np.array([eta1, eta2, eta3]).T

        # Check that all the points are indeed inside the triangle
        eta_final = []
        rho_theta_in_triangle = []
        for i in range(eta.shape[0]):
            if eta1[i] < 0 or eta2[i] < 0 or eta3[i] < 0:
                warnings.warn(
                    "Found rho_theta points outside the triangle ... "
                    "Not using these points to evaluate the barycentric "
                    "coordinates."
                )
            else:
                eta_final.append(eta[i, :])
                rho_theta_in_triangle.append(rho_theta[i, :])
        return np.array(eta_final), np.array(rho_theta_in_triangle)

    def plot_triangle(self):
        """
        Plot the triangle in (rho, theta) and (eta1, eta2, eta3) coordinates.

        Also plots all of the basis functions.

        Parameters
        ----------
        plot_quadrature_points : bool
            Flag to indicate whether or not the quadrature points for
            integration should also be plotted on the mesh.
        """
        # Define uniform grid in (theta, zeta)
        rho = np.linspace(
            np.min(self.vertices[:, 0]), np.max(self.vertices[:, 0]), endpoint=True
        )
        theta = np.linspace(
            np.min(self.vertices[:, 1]), np.max(self.vertices[:, 1]), endpoint=True
        )
        Rho, Theta = np.meshgrid(rho, theta, indexing="ij")
        Rho_Theta = np.array([np.ravel(Rho), np.ravel(Theta)]).T

        # Get the basis functions in the triangle
        psi_q, rho_theta_in_triangle = self.get_basis_functions(Rho_Theta)

        for i in range(self.Q):
            plt.subplot(1, self.Q, i + 1)
            plt.grid()

            # Plot the nodes of the triangle
            plt.plot(self.nodes[:, 0], self.nodes[:, 1], "ro", markersize=2)

            # Plot the ith basis function
            plt.scatter(
                rho_theta_in_triangle[:, 0],
                rho_theta_in_triangle[:, 1],
                c=psi_q[:, i],
                vmin=0,
                vmax=1,
                s=2,
            )
            plt.xlim(0, 1)
            plt.ylim(0, 2 * np.pi)
            tstring = r"$\psi_{" + str(i) + r"}(\theta, \zeta)$"
            plt.ylabel(r"$\theta$")
            plt.xlabel(r"$\rho$")
            plt.title(tstring)


class IntervalFiniteElement:
    """Class representing an interval in a 1D grid of finite elements.

    Takes the range of the isoparametric coordinate eta as [-1, 1],
    instead of other definitions using [0, 1].

    Parameters
    ----------
    vertices: array-like, shape(2)
        The two vertices of the interval in the theta mesh.
    K: integer
        The order of the finite elements to use, which gives (K + 1)
        basis functions.
    """

    def __init__(self, vertices, K=1):
        self.vertices = vertices
        self.length = vertices[1] - vertices[0]
        self.Q = K + 1
        self.K = K

        # if K = 1 or K = 2 with equally-spaced points
        # Jacobian is length / 2 for isoparametric form eta in [-1, 1]
        # still true for K > 2 ??
        self.jacobian = self.length / 2.0

        # Going to construct equally spaced nodes for order K interval,
        # which gives Q such nodes.
        self.nodes = np.linspace(vertices[0], vertices[1], self.Q, endpoint=True)
        self.eta_nodes, _ = self.get_barycentric_coordinates(self.nodes)
        self.basis_functions_nodes, _ = self.get_basis_functions(self.nodes)

        # Check basis functions vanish at all nodes except the associated node.
        for i in range(self.Q):
            assert np.allclose(self.basis_functions_nodes[i, i], 1.0)
            for j in range(self.Q):
                if i != j:
                    assert np.allclose(self.basis_functions_nodes[i, j], 0.0)

        # Check we have the same number of basis functions as nodes.
        assert self.nodes.shape[0] == self.Q

    def get_basis_functions(self, theta):
        """
        Gets the barycentric basis functions.

        Return the interval basis functions, evaluated at the 1D theta
        mesh points provided to the function.

        Parameters
        ----------
        theta : 1D ndarray, shape (M)
            Coordinates of the original grid, lying inside this triangle.

        Returns
        -------
        psi_q : (M, Q)
            Basis functions in each interval.

        """
        eta, theta_in_interval = self.get_barycentric_coordinates(theta)
        theta = theta_in_interval

        # Compute the vertex basis functions first
        basis_functions = np.zeros((theta.shape[0], self.Q))
        for i in range(self.Q):
            basis_functions[:, i] = self.lagrange_polynomial(eta, self.eta_nodes, i)

        if self.K == 1:
            assert np.allclose(basis_functions[:, 0], 0.5 * (1 - eta))
            assert np.allclose(basis_functions[:, 1], 0.5 * (1 + eta))
        if self.K == 2:
            # Note the basis function ordering
            assert np.allclose(basis_functions[:, 0], 0.5 * eta * (eta - 1))
            assert np.allclose(basis_functions[:, 2], 0.5 * eta * (eta + 1))
            assert np.allclose(basis_functions[:, 1], 1 - eta**2)

        return basis_functions, theta_in_interval

    def lagrange_polynomial(self, eta_i, eta_nodes_i, q):
        """
        Computes lagrange polynomials.

        Computes the lagrange polynomial given the ith component of the
        Barycentric coordinates on a theta mesh, the ith component
        of the interval nodes defined for the basis functions,
        and the index q of which node this is.

        Parameters
        ----------
        eta_i : 1D ndarray, shape(M)
            The barycentric coordinate i defined at theta points.
        eta_nodes_i : 1D ndarray, shape(Q)
            The barycentric coordinate i defined at the interval nodes.
        q : integer
            The index of the node we are using to define the basis function.
            Options are 0, ..., Q - 1

        Returns
        -------
        lp : 1D ndarray, shape(M)
            The lagrange polynomial associated with the barycentric
            coordinate i, the polynomial order K + 1, and the node q.

        """
        denom = 1.0
        numerator = np.ones(len(eta_i))
        # Avoid choosing the node q associated with this basis function
        for i in range(self.Q):
            if i != q:
                numerator *= eta_i - eta_nodes_i[i]
                denom *= eta_nodes_i[q] - eta_nodes_i[i]
        lp = numerator / denom
        return lp

    def get_barycentric_coordinates(self, theta):
        """
        Gets the barycentric coordinates, given a mesh in theta.

        Parameters
        ----------
        theta : 1D ndarray, shape (M)
            Coordinates of the original grid, lying inside this interval.

        Returns
        -------
        eta_u: 1D array, shape (M)
            Barycentric coordinates defined by the interval and evaluated
            at the points theta.
        """
        # Get the Barycentric coordinates
        eta = 2 * (theta - self.vertices[0]) / self.length - 1
        try:
            eta_shape = eta.shape[0]
        except IndexError:
            eta = np.array([eta])
            theta = np.array([theta])
            eta_shape = eta.shape[0]

        # zero out numerical errors or weird minus signs like -0
        eta[np.isclose(eta, 0.0)] = 0.0

        # Check that all the points are indeed inside the triangle
        eta_final = []
        theta_in_interval = []
        for i in range(eta_shape):
            if eta[i] < -1 or eta[i] > 1:
                warnings.warn(
                    "Found theta points outside this interval ... "
                    "Not using these points to evaluate the barycentric "
                    "coordinates."
                )
            else:
                eta_final.append(eta[i])
                theta_in_interval.append(theta[i])
        return np.array(eta_final), np.array(theta_in_interval)

    def plot_interval(self):
        """
        Plot the interval.

        Also plots all of the basis functions.

        Parameters
        ----------
        plot_quadrature_points : bool
            Flag to indicate whether or not the quadrature points for
            integration should also be plotted on the mesh.
        """
        # Define uniform grid in (theta, zeta)
        theta = np.linspace(self.vertices[0], self.vertices[1], endpoint=True)

        # Get the basis functions in the triangle
        psi_q, theta_in_interval = self.get_basis_functions(theta)

        for i in range(self.Q):
            plt.subplot(1, self.Q, i + 1)
            plt.grid()

            # Plot the nodes of the interval
            plt.plot(self.nodes, np.zeros(self.Q), "ro", markersize=2)

            # Plot the ith basis function
            plt.plot(theta_in_interval, psi_q[:, i], "b--")
            plt.xlim(0, 2 * np.pi)
            plt.ylabel(r"$\psi_q(\theta)$")
            plt.xlabel(r"$\theta$")


class FiniteElementMesh1D:
    """Class representing a 1D mesh in theta.

    This class represents a 1D FE basis coming from a
    set of M uniform points sampled in theta.

    Parameters
    ----------
    M : int
        Number of mesh points in the theta direction.
    K: integer
        The order of the finite elements to use, which gives (K+1)(K+2) / 2
        basis functions.
    """

    def __init__(self, M, K=1, nquad=2):
        self.M = M
        self.Q = K + 1
        self.K = K
        mesh = fem.MeshLine(np.linspace(0, 2 * np.pi, M, endpoint=True))

        if K == 1:
            e = fem.ElementLineP1()
        else:
            e = fem.ElementLineP2()
        basis = fem.CellBasis(mesh, e)
        vertices = np.ravel(basis.doflocs)

        # record the nodal assembly matrix if need to check these integrals
        from skfem.helpers import dot

        @fem.BilinearForm
        def assembly(u, v, _):
            return dot(u, v)

        self.assembly_matrix = assembly.assemble(basis).todense()

        # K + 1 here so that integral of basis_function * basis_function
        # is nonsingular, which requires a higher order integration
        self.nquad = K + 1

        theta = np.linspace(0, 2 * np.pi, M, endpoint=True)
        self.Theta = theta

        # Compute the basis functions at each node
        intervals = []
        for i in range(M - 1):
            # Deal with the periodic boundary conditions...
            interval = IntervalFiniteElement(
                np.array([vertices[i], vertices[i + 1]]).T, K=K
            )
            intervals.append(interval)

        # Have M vertices and M-1 intervals
        self.vertices = vertices
        self.intervals = intervals

        # Setup quadrature points and weights for numerical integration
        # Using Gauss-Legendre quadrature
        integration_points = []
        weights = []
        # Ordered these from smallest to largest
        if self.nquad == 1:
            integration_points = [0.0]
            weights = [2.0]
        elif self.nquad == 2:
            integration_points = [-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)]
            weights = [1.0, 1.0]
        elif self.nquad == 3:
            integration_points = [-np.sqrt(0.6), 0.0, np.sqrt(0.6)]
            weights = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
        elif self.nquad == 4:
            integration_points = [
                -np.sqrt(3 + np.sqrt(4.8)) / 7.0,
                -np.sqrt(3 - np.sqrt(4.8)) / 7.0,
                np.sqrt(3 - np.sqrt(4.8)) / 7.0,
                np.sqrt(3 + np.sqrt(4.8)) / 7.0,
            ]
            weights = [
                0.5 - 1.0 / (3.0 * np.sqrt(4.8)),
                0.5 + 1.0 / (3.0 * np.sqrt(4.8)),
                0.5 + 1.0 / (3.0 * np.sqrt(4.8)),
                0.5 - 1.0 / (3.0 * np.sqrt(4.8)),
            ]

        self.integration_points = np.array(integration_points)
        self.weights = np.ravel(np.array(weights))

    def plot_intervals(self, plot_quadrature_points=False):
        """Plot all the intervals in the 1D mesh."""
        plt.figure(100)
        for i, interval in enumerate(self.intervals):
            interval.plot_interval()
        if plot_quadrature_points:
            quadpoints = self.return_quadrature_points()
            for i in range(self.Q):
                plt.subplot(1, self.Q, i + 1)
                plt.plot(quadpoints, np.zeros(len(quadpoints)), "ko")
        plt.show()

    def full_basis_functions_corresponding_to_points(self, theta):
        """Given points on the mesh, find all (I, Q) basis functions values.

        Parameters
        ----------
        theta : 1D ndarray, shape (num_points)
            Set of points for which we want to find the intervals that
            they lie inside of in the mesh.

        Returns
        -------
        basis_functions : 3D ndarray, shape (num_points, I, Q)
            All of the IQ basis functions evaluated at the points.
            Most will be zero at a given point.
        """
        basis_functions = np.zeros((theta.shape[0], self.M - 1, self.Q))
        for i in range(theta.shape[0]):
            v = theta[i]
            for j, interval in enumerate(self.intervals):
                v1 = interval.vertices[0]
                v2 = interval.vertices[1]
                if v >= v1 and v <= v2:
                    bfs, _ = interval.get_basis_functions(v)
                    basis_functions[i, j, :] = bfs
                    break
        return basis_functions

    def find_intervals_corresponding_to_points(self, theta):
        """Given a point on the mesh, find which interval it lies inside.

        Parameters
        ----------
        theta : 1D ndarray, shape (num_points)
            Set of points for which we want to find the intervals that
            they lie inside of in the mesh.

        Returns
        -------
        interval_indices : 1D ndarray, shape (num_points)
            Set of indices that specific the intervals where each point lies.
        basis_functions : 2D ndarray, shape (num_points, Q)
            The basis functions corresponding to the intervals in
            interval_indices.
        """
        interval_indices = np.zeros(theta.shape[0])
        basis_functions = np.zeros((theta.shape[0], self.Q))
        for i in range(theta.shape[0]):
            v = theta[i]
            for j, interval in enumerate(self.intervals):
                v1 = interval.vertices[0]
                v2 = interval.vertices[1]
                if v >= v1 and v <= v2:
                    interval_indices[i] = j
                    basis_functions[i, :], _ = interval.get_basis_functions(v)
        return interval_indices, basis_functions

    def return_quadrature_points(self):
        """Get quadrature points for numerical integration over the mesh.

        Returns
        -------
        quadrature points: 1D ndarray, shape (nquad * (M - 1))
            Points in theta representing the quadrature point
            locations for integration in barycentric coordinates.

        """
        nquad = self.nquad
        quadrature_points = np.zeros((self.M - 1) * nquad)
        q = 0
        for interval in self.intervals:
            for i in range(nquad):
                theta1 = interval.vertices[0]
                theta2 = interval.vertices[1]
                quadrature_points[q] = (theta2 - theta1) * (
                    self.integration_points[i] + 1
                ) / 2.0 + theta1
                q = q + 1

        return quadrature_points

    def integrate(self, f):
        """Integrates a function over the 1D mesh in theta.

        This function allows one to integrate any set of functions of theta
        ver the full 1D mesh. Uses Gauss-Legendre quadrature
        formula for in the barycentric coordinates.

        Parameters
        ----------
        f : 1D ndarray, shape (nquad * M, num_functions)
            Vector function defined on the mesh in theta
            that we would like to integrate component-wise with respect
            to the basis functions. For integration over the barycentric
            coordinates, we need f to be prescribed at the quadrature points
            in a barycentric coordinate system.

        Returns
        -------
        integral: 1D ndarray, shape (num_functions)
            Value of the integral over the mesh for each component of f.

        """
        nquad = self.nquad
        if f.shape[1] > 1:
            integral = np.zeros(f.shape[1])
        else:
            integral = 0.0
        for i, interval in enumerate(self.intervals):
            integral += (
                interval.jacobian * self.weights @ f[i * nquad : (i + 1) * nquad, :]
            )
        return integral


@jit
@jnp.vectorize
def _binom(n, k):
    """Binomial coefficient.

    Implementation is only correct for positive integer n,k and n>=k

    Parameters
    ----------
    n : int, array-like
        number of things to choose from
    k : int, array-like
        number of things chosen

    Returns
    -------
    val : int, float, array-like
        number of possible combinations
    """
    # adapted from scipy:
    # https://github.com/scipy/scipy/blob/701ffcc8a6f04509d115aac5e5681c538b5265a2/
    # scipy/special/orthogonal_eval.pxd#L68

    n, k = map(jnp.asarray, (n, k))

    def _binom_body_fun(i, b_n):
        b, n = b_n
        num = n + 1 - i
        den = i
        return (b * num / den, n)

    kx = k.astype(int)
    b, n = fori_loop(1, 1 + kx, _binom_body_fun, (1.0, n))
    return b


@custom_jvp
@jit
@jnp.vectorize
def _jacobi(n, alpha, beta, x, dx=0):
    """Jacobi polynomial evaluation.

    Implementation is only correct for non-negative integer coefficients,
    returns 0 otherwise.

    Parameters
    ----------
    n : int, array_like
        Degree of the polynomial.
    alpha : int, array_like
        Parameter
    beta : int, array_like
        Parameter
    x : float, array_like
        Points at which to evaluate the polynomial

    Returns
    -------
    P : ndarray
        Values of the Jacobi polynomial
    """
    # adapted from scipy:
    # https://github.com/scipy/scipy/blob/701ffcc8a6f04509d115aac5e5681c538b5265a2/
    # scipy/special/orthogonal_eval.pxd#L144

    def _jacobi_body_fun(kk, d_p_a_b_x):
        d, p, alpha, beta, x = d_p_a_b_x
        k = kk + 1.0
        t = 2 * k + alpha + beta
        d = (
            (t * (t + 1) * (t + 2)) * (x - 1) * p + 2 * k * (k + beta) * (t + 2) * d
        ) / (2 * (k + alpha + 1) * (k + alpha + beta + 1) * t)
        p = d + p
        return (d, p, alpha, beta, x)

    n, alpha, beta, x = map(jnp.asarray, (n, alpha, beta, x))

    # coefficient for derivative
    c = (
        gammaln(alpha + beta + n + 1 + dx)
        - dx * jnp.log(2)
        - gammaln(alpha + beta + n + 1)
    )
    c = jnp.exp(c)
    # taking derivative is same as coeff*jacobi but for shifted n,a,b
    n -= dx
    alpha += dx
    beta += dx

    d = (alpha + beta + 2) * (x - 1) / (2 * (alpha + 1))
    p = d + 1
    d, p, alpha, beta, x = fori_loop(
        0, jnp.maximum(n - 1, 0).astype(int), _jacobi_body_fun, (d, p, alpha, beta, x)
    )
    out = _binom(n + alpha, n) * p
    # should be complex for n<0, but it gets replaced elsewhere so just return 0 here
    out = jnp.where(n < 0, 0, out)
    # other edge cases
    out = jnp.where(n == 0, 1.0, out)
    out = jnp.where(n == 1, 0.5 * (2 * (alpha + 1) + (alpha + beta + 2) * (x - 1)), out)
    return c * out


@_jacobi.defjvp
def _jacobi_jvp(x, xdot):
    (n, alpha, beta, x, dx) = x
    (ndot, alphadot, betadot, xdot, dxdot) = xdot
    f = _jacobi(n, alpha, beta, x, dx)
    df = _jacobi(n, alpha, beta, x, dx + 1)
    # in theory n, alpha, beta, dx aren't differentiable (they're integers)
    # but marking them as non-diff argnums seems to cause escaped tracer values.
    # probably a more elegant fix, but just setting those derivatives to zero seems
    # to work fine.
    return f, df * xdot + 0 * ndot + 0 * alphadot + 0 * betadot + 0 * dxdot
