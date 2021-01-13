import numpy as np
from itertools import permutations, combinations_with_replacement
from termcolor import colored

from desc.backend import jnp
from desc.utils import equals
from desc.grid import Grid
from desc.basis import Basis
from desc.io import IOAble


class Transform(IOAble):
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

    _io_attrs_ = ["_grid", "_basis", "_derives", "_matrices"]

    def __init__(
        self,
        grid: Grid = None,
        basis: Basis = None,
        derivs=0,
        rcond=1e-6,
        build=True,
        load_from=None,
        file_format=None,
        obj_lib=None,
    ) -> None:
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
        build : bool
            whether to precompute the transforms now or do it later

         Returns
         -------
         None

        """
        self._file_format_ = file_format

        if load_from is None:
            self._grid = grid
            self._basis = basis
            self._derivs = derivs
            self._rcond = rcond

            self._matrices = {
                i: {j: {k: {} for k in range(4)} for j in range(4)} for i in range(4)
            }
            self._derivatives = self._get_derivatives_(self._derivs)

            self._sort_derivatives_()
            if build:
                self.build()
                self._built = True
            else:
                self._built = False
        else:
            self._init_from_file_(
                load_from=load_from, file_format=file_format, obj_lib=obj_lib
            )

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
            combos = combinations_with_replacement(range(derivs + 1), 3)
            for combo in list(combos):
                perms = set(permutations(combo))
                for perm in list(perms):
                    if derivatives.shape[1] == 3:
                        derivatives = np.vstack([derivatives, np.array(perm)])
                    else:
                        derivatives = np.array([perm])
        elif derivs.lower() == "force":
            derivatives = np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [2, 0, 0],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 2, 0],
                    [0, 1, 1],
                    [0, 0, 2],
                ]
            )
            # FIXME: this assumes the Grid is sorted (which it should be)
            if np.all(self._grid.nodes[:, 0] == np.array([0, 0, 0])):
                axis = np.array([[2, 1, 0], [1, 2, 0], [1, 1, 1], [2, 2, 0]])
                derivatives = np.vstack([derivatives, axis])
        elif derivs.lower() == "qs":
            derivatives = np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [2, 0, 0],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 2, 0],
                    [0, 1, 1],
                    [0, 0, 2],
                    [3, 0, 0],
                    [2, 1, 0],
                    [2, 0, 1],
                    [1, 2, 0],
                    [1, 1, 1],
                    [1, 0, 2],
                    [0, 3, 0],
                    [0, 2, 1],
                    [0, 1, 2],
                    [0, 0, 3],
                    [2, 2, 0],
                ]
            )
        else:
            raise NotImplementedError(
                colored("order options are 'force', 'qs', or a non-negative int", "red")
            )

        return derivatives

    def _sort_derivatives_(self) -> None:
        """Sorts derivatives

        Returns
        -------
        None

        """
        sort_idx = np.lexsort(
            (self._derivatives[:, 0], self._derivatives[:, 1], self._derivatives[:, 2])
        )
        self._derivatives = self._derivatives[sort_idx]

    def build(self) -> None:
        """Builds the transform matrices for each derivative order"""
        for d in self._derivatives:
            self._matrices[d[0]][d[1]][d[2]] = self._basis.evaluate(self._grid.nodes, d)

        # build pinv for fitting
        if np.all(self._derivatives[0, :] == np.array([0, 0, 0])):
            A = self._matrices[0][0][0]
        else:
            A = self._basis.evaluate(self._grid.nodes, np.array([0, 0, 0]))
        if A.size:
            self._pinv = jnp.linalg.pinv(A, rcond=self._rcond)
        else:
            self._pinv = jnp.zeros_like(A.T)
        self._built = True

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
        if not self._built:
            raise AttributeError("Transform must be built before it can be used")

        A = self._matrices[dr][dt][dz]
        if type(A) is dict:
            raise ValueError(
                colored("Derivative orders are out of initialized bounds", "red")
            )
        if A.shape[1] != c.size:
            raise ValueError(
                colored(
                    "Coefficients dimension ({}) is incompatible with the number of basis modes({})".format(
                        c.size, A.shape[1]
                    ),
                    "red",
                )
            )

        return jnp.matmul(A, c)

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
        if not self._built:
            raise AttributeError("Transform must be built before it can be used")
        return jnp.matmul(self._pinv, x)

    def change_resolution(
        self, grid: Grid = None, basis: Basis = None, rebuild: bool = True
    ) -> None:
        """Re-builds the matrices with a new grid and basise

        Parameters
        ----------
        grid : Grid, optional
            DESCRIPTION
        basis : Basis, optional
            DESCRIPTION
        rebuild : bool
            whether to recompute matrices now or wait until requested

        Returns
        -------
        None

        """
        if grid is None:
            grid = self._grid
        if basis is None:
            basis = self._basis

        if (self._grid != grid or self._basis != basis) and rebuild:
            self._grid = grid
            self._basis = basis
            self.build()

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, grid: Grid) -> None:
        """Changes the grid and updates the matrices accordingly

        Parameters
        ----------
        grid : Grid
            DESCRIPTION

        Returns
        -------
        None

        """
        if self._grid != grid:
            self._grid = grid
            self.build()

    @property
    def basis(self):
        return self._basis

    @basis.setter
    def basis(self, basis: Basis) -> None:
        """Changes the basis and updates the matrices accordingly

        Parameters
        ----------
        basis : Basis
            DESCRIPTION

        Returns
        -------
        None

        """
        if self._basis != basis:
            self._basis = basis
            self.build()

    @property
    def derivs(self):
        return self._derivs

    @property
    def derivatives(self):
        return self._derivatives

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
        if derivs != self._derivs:
            self._derivs = derivs

            old_derivatives = self._derivatives
            self._derivatives = self.get_derivatives(self._derivs)
            self.sort_derivatives()
            new_derivatives = self._derivatives

            new_not_in_old = (
                (new_derivatives[:, None] == old_derivatives).all(-1).any(-1)
            )
            derivs_to_add = new_derivatives[~new_not_in_old]

            for d in derivs_to_add:
                self._matrices[d[0]][d[1]][d[2]] = self._basis.evaluate(
                    self._grid.nodes, d
                )

    @property
    def matrices(self):
        return self._matrices

    @property
    def num_nodes(self):
        return self._grid.num_nodes

    @property
    def num_modes(self):
        return self._basis.num_modes
