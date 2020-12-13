import numpy as np
import functools
from itertools import permutations, combinations_with_replacement

from desc.backend import jnp, conditional_decorator, jit, use_jax, TextColors, equals
from desc.grid import Grid
from desc.basis import Basis


class Transform():
    """Transform

    Attributes
    ----------
    grid : Grid
        DESCRIPTION
    basis : Basis
        DESCRIPTION
    rcond : float
        relative cutoff for singular values in least squares fit
    derivatives : ndarray
        combinations of derivatives needed
        Each row is one set, columns represent the order of derivatives
        for [rho, theta, zeta]
    matrices : ndarray
        DESCRIPTION
    pinv : ndarray
        DESCRIPTION

    """

    def __init__(self, grid:Grid, basis:Basis, derivs=0, rcond=1e-6) -> None:
        """Initializes a Transform

        Parameters
        ----------
        grid : Grid
            DESCRIPTION
        basis : Basis
            DESCRIPTION
        derivs : int or string
            order of derivatives needed, if an int (Default = 0)
            OR
            type of calculation being performed, if a string
            ``'force'``: all of the derivatives needed to calculate an
            equilibrium from the force balance equations
            ``'qs'``: all of the derivatives needed to calculate quasi-
            symmetry from the triple-product equation
       rcond : float
            relative cutoff for singular values in least squares fit

        Returns
        -------
        None

        """
        self.__grid = grid
        self.__basis = basis
        self.__derivs = derivs
        self.__rcond = rcond

        self.__matrices = {i: {j: {k: {}
                     for k in range(4)} for j in range(4)} for i in range(4)}
        self.__derivatives = self._get_derivatives_(self.__derivs)

        self._sort_derivatives_()
        self._build_()
        self._build_pinv_()
        self._def_save_attrs_()

    def __eq__(self, other) -> bool:
        """Overloads the == operator

        Parameters
        ----------
        other : Transform
            another Transform object to compare to

        Returns
        -------
        bool
            True if other is a Transform with the same attributes as self
            False otherwise

        """
        if self.__class__ != other.__class__:
            return False
        return equals(self.__dict__, other.__dict__)

    def _get_derivatives_(self, derivs):
        """Get array of derivatives needed for calculating objective function

        Parameters
        ----------
        derivs : int or string
            order of derivatives needed, if an int (Default = 0)
            OR
            type of calculation being performed, if a string
            ``'force'``: all of the derivatives needed to calculate an
            equilibrium from the force balance equations
            ``'qs'``: all of the derivatives needed to calculate quasi-
            symmetry from the triple-product equation

        Returns
        -------
        derivatives : ndarray
            combinations of derivatives needed
            Each row is one set, columns represent the order of derivatives
            for [rho, theta, zeta]

        """
        if isinstance(derivs, int) and derivs >= 0:
            derivatives = np.array([[]])
            combos = combinations_with_replacement(range(derivs+1), 3)
            for combo in list(combos):
                perms = set(permutations(combo))
                for perm in list(perms):
                    if derivatives.shape[1] == 3:
                        derivatives = np.vstack([derivatives, np.array(perm)])
                    else:
                        derivatives = np.array([perm])
        elif derivs.lower() == 'force':
            derivatives = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                                    [2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0],
                                    [0, 1, 1], [0, 0, 2]])
            # FIXME: this assumes the Grid is sorted (which it should be)
            if np.all(self.__grid.nodes[:, 0] == np.array([0, 0, 0])):
                axis = np.array([[2, 1, 0], [1, 2, 0], [1, 1, 1], [2, 2, 0]])
                derivatives = np.vstack([derivatives, axis])
        elif derivs.lower() == 'qs':
            derivatives = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                                    [2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0],
                                    [0, 1, 1], [0, 0, 2], [3, 0, 0], [2, 1, 0],
                                    [2, 0, 1], [1, 2, 0], [1, 1, 1], [1, 0, 2],
                                    [0, 3, 0], [0, 2, 1], [0, 1, 2], [0, 0, 3],
                                    [2, 2, 0]])
        else:
            raise NotImplementedError(TextColors.FAIL + 
                  "order options are 'force', 'qs', or a non-negative int"
                  + TextColors.ENDC)
        return derivatives

    def _sort_derivatives_(self) -> None:
        """Sorts derivatives

        Returns
        -------
        None

        """
        sort_idx = np.lexsort((self.__derivatives[:, 0],
                       self.__derivatives[:, 1], self.__derivatives[:, 2]))
        self.__derivatives = self.__derivatives[sort_idx]

    def _build_(self) -> None:
        """Builds the transform matrices for each derivative order
        """
        for d in self.__derivatives:
            self.__matrices[d[0]][d[1]][d[2]] = self.__basis.evaluate(
                                                        self.__grid.nodes, d)

    def _build_pinv_(self) -> None:
        """Builds the transform matrices for each derivative order
        """
        # FIXME: this assumes the derivatives are sorted (which they should be)
        if np.all(self.__derivatives[0, :] == np.array([0, 0, 0])):
            A = self.__matrices[0][0][0]
        else:
            A = self.__basis.evaluate(self.__grid.nodes, np.array([0, 0, 0]))
        self.__pinv = jnp.linalg.pinv(A, rcond=self.__rcond)

    def _def_save_attrs_(self) -> None:
        """Defines attributes to save

        Returns
        -------
        None

        """
        self._save_attrs_ = ['__grid', '__basis', '__derives', '__matrices']

    def transform(self, c, dr=0, dt=0, dz=0):
        """Transform from spectral domain to physical

        Parameters
        ----------
        c : ndarray, shape(N_coeffs,)
            spectral coefficients, indexed as (lm,n) flattened in row major order
        dr : int
            order of radial derivative
        dt : int
            order of poloidal derivative
        dz : int
            order of toroidal derivative

        Returns
        -------
        x : ndarray, shape(N_nodes,)
            array of values of function at node locations

        """
        A = self.__matrices[dr][dt][dz]
        if type(A) is dict:
            raise ValueError(TextColors.FAIL +
                 "Derivative orders are out of initialized bounds" +
                             TextColors.ENDC)
        if A.shape[1] != c.size:
            raise ValueError(TextColors.FAIL +
                 "Coefficients dimension ({}) is incompatible with the number of basis modes({})".format(c.size, A.shape[1]) +
                             TextColors.ENDC)

        return jnp.matmul(A, c)

    @conditional_decorator(functools.partial(jit, static_argnums=(0,)), use_jax)
    def fit(self, x):
        """Transform from physical domain to spectral using least squares fit

        Parameters
        ----------
        x : ndarray, shape(N_nodes,)
            values in real space at coordinates specified by self.grid

        Returns
        -------
        c : ndarray, shape(N_coeffs,)
            spectral coefficients in self.basis

        """
        return jnp.matmul(self.__pinv, x)

    def change_resolution(self, grid:Grid=None, basis:Basis=None) -> None:
        """Re-builds the matrices with a new grid and basis

        Parameters
        ----------
        grid : Grid, optional
            DESCRIPTION
        basis : Basis, optional
            DESCRIPTION

        Returns
        -------
        None

        """
        if grid is None:
            grid = self.__grid
        if basis is None:
            basis = self.__basis

        if self.__grid != grid or self.__basis != basis:
            self.__grid = grid
            self.__basis = basis
            self._build_()
            self._build_pinv_()

    @property
    def grid(self):
        return self.__grid

    @grid.setter
    def grid(self, grid:Grid) -> None:
        """Changes the grid and updates the matrices accordingly

        Parameters
        ----------
        grid : Grid
            DESCRIPTION

        Returns
        -------
        None

        """
        if self.__grid != grid:
            self.__grid = grid
            self._build_()
            self._build_pinv_()

    @property
    def basis(self):
        return self.__basis

    @basis.setter
    def basis(self, basis:Basis) -> None:
        """Changes the basis and updates the matrices accordingly

        Parameters
        ----------
        basis : Basis
            DESCRIPTION

        Returns
        -------
        None

        """
        if self.__basis != basis:
            self.__basis = basis
            self._build_()
            self._build_pinv_()

    @property
    def derivs(self):
        return self.__derivs

    @property
    def derivatives(self):
        return self.__derivatives

    @derivatives.setter
    def derivatives(self, derivs) -> None:
        """Changes the order and updates the matrices accordingly

        Parameters
        ----------
        derivs : int or string
            order of derivatives needed, if an int (Default = 0)
            OR
            type of calculation being performed, if a string
            ``'force'``: all of the derivatives needed to calculate an
            equilibrium from the force balance equations
            ``'qs'``: all of the derivatives needed to calculate quasi-
            symmetry from the triple-product equation

        Returns
        -------
        None

        """
        if derivs != self.__derivs:
            self.__derivs = derivs

            old_derivatives = self.__derivatives
            self.__derivatives = self.get_derivatives(self.__derivs)
            self.sort_derivatives()
            new_derivatives = self.__derivatives

            new_not_in_old = (
                new_derivatives[:, None] == old_derivatives).all(-1).any(-1)
            derivs_to_add = new_derivatives[~new_not_in_old]

            for d in derivs_to_add:
                self.__matrices[d[0]][d[1]][d[2]] = self.__basis.evaluate(
                                                        self.__grid.nodes, d)

    @property
    def matrices(self):
        return self.__matrices

    @property
    def num_nodes(self):
        return self.__grid.num_nodes

    @property
    def num_modes(self):
        return self.__basis.num_modes


# these functions are currently unused ---------------------------------------

def zernike_norm(l, m):
    """Norm of a Zernike polynomial with l, m indexing.
    Returns the integral (Z^m_l)^2 r dr dt, r=[0,1], t=[0,2*pi]

    Parameters
    ----------
    l,m : int
        radial and azimuthal mode numbers.

    Returns
    -------
    norm : float
        norm of Zernike polynomial over unit disk.

    """
    return jnp.sqrt((2 * (l + 1)) / (jnp.pi*(1 + jnp.kronecker(m, 0))))


def lm_to_fringe(l, m):
    """Convert Zernike (l,m) double index to single Fringe index.

    Parameters
    ----------
    l,m : int
        radial and azimuthal mode numbers.

    Returns
    -------
    idx : int
        Fringe index for l,m

    """
    M = (l + np.abs(m)) / 2
    return int(M**2 + M + m)


def fringe_to_lm(idx):
    """Convert single Zernike Fringe index to (l,m) double index.

    Parameters
    ----------
    idx : int
        Fringe index

    Returns
    -------
    l,m : int
        radial and azimuthal mode numbers.

    """
    M = (np.ceil(np.sqrt(idx+1)) - 1)
    m = idx - M**2 - M
    l = 2*M - np.abs(m)
    return int(l), int(m)


def lm_to_ansi(l, m):
    """Convert Zernike (l,m) two term index to ANSI single term index.

    Parameters
    ----------
    l,m : int
        radial and azimuthal mode numbers.

    Returns
    -------
    idx : int
        ANSI index for l,m

    """
    return int((l * (l + 2) + m) / 2)


def ansi_to_lm(idx):
    """Convert Zernike ANSI single term to (l,m) two-term index.

    Parameters
    ----------
    idx : int
        ANSI index

    Returns
    -------
    l,m : int
        radial and azimuthal mode numbers.

    """
    l = int(np.ceil((-3 + np.sqrt(9 + 8*idx))/2))
    m = 2 * idx - l * (l + 2)
    return l, m
