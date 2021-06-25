import numpy as np
import scipy.linalg
from itertools import permutations, combinations_with_replacement
from termcolor import colored
import warnings

from desc.backend import jnp, put
from desc.utils import issorted, isalmostequal, islinspaced
from desc.io import IOAble


class Transform(IOAble):
    """Transforms from spectral coefficients to real space values.

    Parameters
    ----------
    grid : Grid
        Collocation grid of real space coordinates
    basis : Basis
        Spectral basis of modes
    derivs : int or array-like
        * if an int, order of derivatives needed (default=0)
        * if an array, derivative orders specified explicitly. Shape should be (N,3),
          where each row is one set of partial derivatives [dr, dt, dz]
    rcond : float
         relative cutoff for singular values for inverse fitting
    build : bool
        whether to precompute the transforms now or do it later
    build_pinv : bool
        whether to precompute the pseudoinverse now or do it later
    method : {```'auto'``, `'fft'``, ``'direct1'``, ``'direct2'``}
        * ``'fft'`` uses fast fourier transforms in the zeta direction, and so must have
          equally spaced toroidal nodes, and the same node pattern on each zeta plane
        * ``'direct1'`` uses full matrices and can handle arbitrary node patterns and
          spectral bases.
        * ``'direct2'`` uses a DFT instead of FFT that can be faster in practice
        * ``'auto'`` selects the method based on the grid and basis resolution

    """

    _io_attrs_ = ["_grid", "_basis", "_derivatives", "_rcond", "_method"]

    def __init__(
        self,
        grid,
        basis,
        derivs=0,
        rcond="auto",
        build=False,
        build_pinv=False,
        method="auto",
    ):

        self._grid = grid
        self._basis = basis
        self._rcond = rcond if rcond is not None else "auto"

        self._derivatives = self._get_derivatives(derivs)
        self._sort_derivatives()
        self._method = method

        self._built = False
        self._built_pinv = False
        self._set_up()
        if build:
            self.build()
        if build_pinv:
            self.build_pinv()

    def _set_up(self):

        self.method = self._method
        self._matrices = {
            "direct1": {
                i: {j: {k: {} for k in range(4)} for j in range(4)} for i in range(4)
            },
            "fft": {i: {j: {} for j in range(4)} for i in range(4)},
            "direct2": {i: {} for i in range(4)},
        }

    def _get_derivatives(self, derivs):
        """Get array of derivatives needed for calculating objective function.

        Parameters
        ----------
        derivs : int or string
             order of derivatives needed, if an int (Default = 0)
             OR
             array of derivative orders, shape (N,3)
             [dr, dt, dz]

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
            derivatives = derivatives[
                derivatives.sum(axis=1) <= derivs
            ]  # remove higher orders
        elif np.atleast_1d(derivs).ndim == 1 and len(derivs) == 3:
            derivatives = np.asarray(derivs).reshape((1, 3))
        elif np.atleast_2d(derivs).ndim == 2 and np.atleast_2d(derivs).shape[1] == 3:
            derivatives = np.atleast_2d(derivs)
        else:
            raise NotImplementedError(
                colored(
                    "derivs should be array-like with 3 columns, or a non-negative int",
                    "red",
                )
            )

        return derivatives

    def _sort_derivatives(self):
        """Sort derivatives."""
        sort_idx = np.lexsort(
            (self.derivatives[:, 0], self.derivatives[:, 1], self.derivatives[:, 2])
        )
        self._derivatives = self.derivatives[sort_idx]

    def _check_inputs_fft(self, grid, basis):
        """Check that inputs are formatted correctly for fft method."""
        if grid.num_nodes == 0 or basis.num_modes == 0:
            # trivial case where we just return all zeros, so it doesn't matter
            self._method = "fft"

        zeta_vals, zeta_cts = np.unique(grid.nodes[:, 2], return_counts=True)

        if not isalmostequal(zeta_cts):
            warnings.warn(
                colored(
                    "fft method requires the same number of nodes on each zeta plane, "
                    + "falling back to direct1 method",
                    "yellow",
                )
            )
            self.method = "direct1"
            return

        if not isalmostequal(
            grid.nodes[:, :2].T.reshape((2, zeta_cts[0], -1), order="F")
        ):
            warnings.warn(
                colored(
                    "fft method requires that node pattern is the same on each zeta "
                    + "plane, falling back to direct1 method",
                    "yellow",
                )
            )
            self.method = "direct1"
            return

        id2 = np.lexsort((basis.modes[:, 1], basis.modes[:, 0], basis.modes[:, 2]))
        if not issorted(id2):
            warnings.warn(
                colored(
                    "fft method requires zernike indices to be sorted by toroidal mode "
                    + "number, falling back to direct1 method",
                    "yellow",
                )
            )
            self.method = "direct1"
            return

        if (
            len(zeta_vals) > 1
            and not abs((zeta_vals[-1] + zeta_vals[1]) * basis.NFP - 2 * np.pi) < 1e-14
        ):
            warnings.warn(
                colored(
                    "fft method requires that nodes complete 1 full field period, "
                    + "falling back to direct2 method",
                    "yellow",
                )
            )
            self.method = "direct2"
            return

        n_vals, n_cts = np.unique(basis.modes[:, 2], return_counts=True)
        if len(n_vals) > 1 and not islinspaced(n_vals):
            warnings.warn(
                colored(
                    "fft method requires the toroidal modes are equally spaced in n, "
                    + "falling back to direct1 method",
                    "yellow",
                )
            )
            self.method = "direct1"
            return

        if len(zeta_vals) < len(n_vals):
            warnings.warn(
                colored(
                    "fft method can not undersample in zeta, "
                    + "num_toroidal_modes={}, num_toroidal_angles={}, ".format(
                        len(n_vals), len(zeta_vals)
                    )
                    + "falling back to direct2 method",
                    "yellow",
                )
            )
            self.method = "direct2"
            return

        if len(zeta_vals) % 2 == 0:
            warnings.warn(
                colored(
                    "fft method requires an odd number of toroidal nodes, "
                    + "falling back to direct2 method",
                    "yellow",
                )
            )
            self.method = "direct2"
            return

        if not issorted(grid.nodes[:, 2]):
            warnings.warn(
                colored(
                    "fft method requires nodes to be sorted by toroidal angle in "
                    + "ascending order, falling back to direct2 method",
                    "yellow",
                )
            )
            self.method = "direct2"
            return

        if len(zeta_vals) > 1 and not islinspaced(zeta_vals):
            warnings.warn(
                colored(
                    "fft method requires nodes to be equally spaced in zeta, "
                    + "falling back to direct2 method",
                    "yellow",
                )
            )
            self.method = "direct2"
            return

        self._method = "fft"
        self.lm_modes = np.unique(basis.modes[:, :2], axis=0)
        self.num_lm_modes = self.lm_modes.shape[0]  # number of radial/poloidal modes
        self.num_n_modes = 2 * basis.N + 1  # number of toroidal modes
        self.num_z_nodes = len(zeta_vals)  # number of zeta nodes
        self.N = basis.N  # toroidal resolution of basis
        self.pad_dim = (self.num_z_nodes - 1) // 2 - self.N
        self.dk = basis.NFP * np.arange(-self.N, self.N + 1).reshape((1, -1))
        self.fft_index = np.zeros((basis.num_modes,), dtype=int)
        offset = np.min(basis.modes[:, 2]) + basis.N  # N for sym="cos", 0 otherwise
        for k in range(basis.num_modes):
            row = np.where((basis.modes[k, :2] == self.lm_modes).all(axis=1))[0]
            col = np.where(basis.modes[k, 2] == n_vals)[0]
            self.fft_index[k] = self.num_n_modes * row + col + offset
        self.fft_nodes = np.hstack(
            [
                grid.nodes[:, :2][: grid.num_nodes // self.num_z_nodes],
                np.zeros((grid.num_nodes // self.num_z_nodes, 1)),
            ]
        )

    def _check_inputs_direct2(self, grid, basis):
        """Check that inputs are formatted correctly for direct2 method."""
        if grid.num_nodes == 0 or basis.num_modes == 0:
            # trivial case where we just return all zeros, so it doesn't matter
            self._method = "direct2"
            return

        zeta_vals, zeta_cts = np.unique(grid.nodes[:, 2], return_counts=True)

        if not issorted(grid.nodes[:, 2]):
            warnings.warn(
                colored(
                    "direct2 method requires nodes to be sorted by toroidal angle in "
                    + "ascending order, falling back to direct1 method",
                    "yellow",
                )
            )
            self.method = "direct1"
            return

        if not isalmostequal(zeta_cts):
            warnings.warn(
                colored(
                    "direct2 method requires the same number of nodes on each zeta "
                    + "plane, falling back to direct1 method",
                    "yellow",
                )
            )
            self.method = "direct1"
            return

        if len(zeta_vals) > 1 and not isalmostequal(
            grid.nodes[:, :2].T.reshape((2, zeta_cts[0], -1), order="F")
        ):
            warnings.warn(
                colored(
                    "direct2 method requires that node pattern is the same on each "
                    + "zeta plane, falling back to direct1 method",
                    "yellow",
                )
            )
            self.method = "direct1"
            return

        id2 = np.lexsort((basis.modes[:, 1], basis.modes[:, 0], basis.modes[:, 2]))
        if not issorted(id2):
            warnings.warn(
                colored(
                    "direct2 method requires zernike indices to be sorted by toroidal "
                    + "mode number, falling back to direct1 method",
                    "yellow",
                )
            )
            self.method = "direct1"
            return

        n_vals, n_cts = np.unique(basis.modes[:, 2], return_counts=True)

        self._method = "direct2"
        self.lm_modes = np.unique(basis.modes[:, :2], axis=0)
        self.n_modes = n_vals
        self.zeta_nodes = zeta_vals
        self.num_lm_modes = self.lm_modes.shape[0]  # number of radial/poloidal modes
        self.num_n_modes = self.n_modes.size  # number of toroidal modes
        self.num_z_nodes = len(zeta_vals)  # number of zeta nodes
        self.N = basis.N  # toroidal resolution of basis

        self.fft_index = np.zeros((basis.num_modes,), dtype=int)
        for k in range(basis.num_modes):
            row = np.where((basis.modes[k, :2] == self.lm_modes).all(axis=1))[0]
            col = np.where(basis.modes[k, 2] == n_vals)[0]
            self.fft_index[k] = self.num_n_modes * row + col
        self.fft_nodes = np.hstack(
            [
                grid.nodes[:, :2][: grid.num_nodes // self.num_z_nodes],
                np.zeros((grid.num_nodes // self.num_z_nodes, 1)),
            ]
        )
        self.dft_nodes = np.hstack(
            [np.zeros((self.zeta_nodes.size, 2)), self.zeta_nodes[:, np.newaxis]]
        )

    def build(self):
        """Build the transform matrices for each derivative order."""
        if self.built:
            return

        if self.basis.num_modes == 0:
            self._built = True
            return

        if self.method == "direct1":
            for d in self.derivatives:
                self._matrices["direct1"][d[0]][d[1]][d[2]] = self.basis.evaluate(
                    self.grid.nodes, d
                )

        if self.method in ["fft", "direct2"]:
            temp_d = np.hstack(
                [self.derivatives[:, :2], np.zeros((len(self.derivatives), 1))]
            )
            temp_modes = np.hstack([self.lm_modes, np.zeros((self.num_lm_modes, 1))])
            for d in temp_d:
                self.matrices["fft"][d[0]][d[1]] = self.basis.evaluate(
                    self.fft_nodes, d, modes=temp_modes
                )
        if self.method == "direct2":
            temp_d = np.hstack(
                [np.zeros((len(self.derivatives), 2)), self.derivatives[:, 2:]]
            )
            temp_modes = np.hstack(
                [np.zeros((self.num_n_modes, 2)), self.n_modes[:, np.newaxis]]
            )
            for d in temp_d:
                self.matrices["direct2"][d[2]] = self.basis.evaluate(
                    self.dft_nodes, d, modes=temp_modes
                )

        self._built = True

    def build_pinv(self):
        """Build the pseudoinverse for fitting."""
        if self.built_pinv:
            return
        A = self.basis.evaluate(self.grid.nodes, np.array([0, 0, 0]))
        # for weighted least squares
        A = self.grid.weights[:, np.newaxis] * A
        rcond = None if self.rcond == "auto" else self.rcond
        if A.size:
            self._matrices["pinv"] = scipy.linalg.pinv(A, rcond=rcond)
        else:
            self._matrices["pinv"] = np.zeros_like(A.T)
        self._built_pinv = True

    def transform(self, c, dr=0, dt=0, dz=0):
        """Transform from spectral domain to physical.

        Parameters
        ----------
        c : ndarray, shape(num_coeffs,)
            spectral coefficients, indexed to correspond to the spectral basis
        dr : int
            order of radial derivative
        dt : int
            order of poloidal derivative
        dz : int
            order of toroidal derivative

        Returns
        -------
        x : ndarray, shape(num_nodes,)
            array of values of function at node locations
        """
        if not self.built:
            self.build()

        if self.basis.num_modes != c.size:
            raise ValueError(
                colored(
                    "Coefficients dimension ({}) is incompatible with ".format(c.size)
                    + "the number of basis modes({})".format(self.basis.num_modes),
                    "red",
                )
            )

        if len(c) == 0:
            return np.zeros(self.grid.num_nodes)

        if self.method == "direct1":
            A = self.matrices["direct1"][dr][dt][dz]
            if isinstance(A, dict):
                raise ValueError(
                    colored("Derivative orders are out of initialized bounds", "red")
                )
            return jnp.matmul(A, c)

        elif self.method == "direct2":
            A = self.matrices["fft"][dr][dt]
            B = self.matrices["direct2"][dz]

            if isinstance(A, dict) or isinstance(B, dict):
                raise ValueError(
                    colored("Derivative orders are out of initialized bounds", "red")
                )
            c_mtrx = jnp.zeros((self.num_lm_modes * self.num_n_modes,))
            c_mtrx = put(c_mtrx, self.fft_index, c).reshape((-1, self.num_n_modes))

            cc = jnp.matmul(A, c_mtrx)
            return jnp.matmul(cc, B.T).flatten(order="F")

        elif self.method == "fft":
            A = self.matrices["fft"][dr][dt]
            if isinstance(A, dict):
                raise ValueError(
                    colored("Derivative orders are out of initialized bounds", "red")
                )

            # reshape coefficients
            c_mtrx = jnp.zeros((self.num_lm_modes * self.num_n_modes,))
            c_mtrx = put(c_mtrx, self.fft_index, c).reshape((-1, self.num_n_modes))

            # differentiate
            c_diff = c_mtrx[:, :: (-1) ** dz] * self.dk ** dz * (-1) ** (dz > 1)

            # re-format in complex notation
            c_real = jnp.pad(
                (self.num_z_nodes / 2)
                * (c_diff[:, self.N + 1 :] - 1j * c_diff[:, self.N - 1 :: -1]),
                ((0, 0), (0, self.pad_dim)),
                mode="constant",
            )
            c_cplx = jnp.hstack(
                (
                    self.num_z_nodes * c_diff[:, self.N, jnp.newaxis],
                    c_real,
                    jnp.fliplr(jnp.conj(c_real)),
                )
            )

            # transform coefficients
            c_fft = jnp.real(jnp.fft.ifft(c_cplx))
            return jnp.matmul(A, c_fft).flatten(order="F")

    def fit(self, x):
        """Transform from physical domain to spectral using weighted least squares fit.

        Parameters
        ----------
        x : ndarray, shape(num_nodes,)
            values in real space at coordinates specified by grid

        Returns
        -------
        c : ndarray, shape(num_coeffs,)
            spectral coefficients in basis

        """
        if not self.built_pinv:
            self.build_pinv()
        return jnp.matmul(self.matrices["pinv"], self.grid.weights * x)

    def project(self, y):
        """Project vector y onto basis.

        Equivalent to dotting the transpose of the transform matrix into y, but
        somewhat more efficient in some cases by using FFT instead of full transform

        Parameters
        ----------
        y : ndarray
            vector to project. Should be of size (self.grid.num_nodes,)

        Returns
        -------
        b : ndarray
            vector y projected onto basis, shape (self.basis.num_modes)
        """
        if not self.built:
            self.build()

        if self.grid.num_nodes != y.size:
            raise ValueError(
                colored(
                    "y dimension ({}) is incompatible with ".format(y.size)
                    + "the number of grid nodes({})".format(self.grid.num_nodes),
                    "red",
                )
            )

        if self.method == "direct1":
            A = self.matrices["direct1"][0][0][0]
            return jnp.matmul(A.T, y)

        elif self.method == "direct2":
            A = self.matrices["fft"][0][0]
            B = self.matrices["direct2"][0]
            yy = jnp.matmul(A.T, y.reshape((-1, self.num_z_nodes), order="F"))
            return jnp.matmul(yy, B).flatten()[self.fft_index]

        elif self.method == "fft":
            A = self.matrices["fft"][0][0]
            # this was derived by trial and error, but seems to work correctly
            # there might be a more efficient way...
            a = jnp.fft.fft(A.T @ y.reshape((A.shape[0], -1), order="F"))
            cdn = a[:, 0]
            cr = a[:, 1 : 1 + self.N]
            b = jnp.hstack(
                [-cr.imag[:, ::-1], cdn.real[:, np.newaxis], cr.real]
            ).flatten()[self.fft_index]
            return b

    def change_resolution(
        self, grid=None, basis=None, build=True, build_pinv=False, method="auto"
    ):
        """Re-build the matrices with a new grid and basis.

        Parameters
        ----------
        grid : Grid
            Collocation grid of real space coordinates
        basis : Basis
            Spectral basis of modes
        build : bool
            whether to recompute matrices now or wait until requested
        method : {"auto", "direct1", "direct2", "fft"}
            method to use for computing transforms

        """
        if grid is None:
            grid = self.grid
        if basis is None:
            basis = self.basis

        if not self.grid.eq(grid):
            self._grid = grid
            self._built = False
            self._built_pinv = False
        if not self.basis.eq(basis):
            self._basis = basis
            self._built = False
            self._built_pinv = False
        self.method = method
        if build:
            self.build()
        if build_pinv:
            self.build_pinv()

    @property
    def grid(self):
        """Grid : collocation grid for the transform."""
        if not hasattr(self, "_grid"):
            self._grid = None
        return self._grid

    @grid.setter
    def grid(self, grid):
        if not self.grid.eq(grid):
            self._grid = grid
            if self.method == "fft":
                self._check_inputs_fft(self.grid, self.basis)
            if self.method == "direct2":
                self._check_inputs_direct2(self.grid, self.basis)
            if self.built:
                self._built = False
                self.build()
            if self.built_pinv:
                self._built_pinv = False
                self.build_pinv()

    @property
    def basis(self):
        """Basis : spectral basis for the transform."""
        if not hasattr(self, "_basis"):
            self._basis = None
        return self._basis

    @basis.setter
    def basis(self, basis):
        if not self.basis.eq(basis):
            self._basis = basis
            if self.method == "fft":
                self._check_inputs_fft(self.grid, self.basis)
            if self.method == "direct2":
                self._check_inputs_direct2(self.grid, self.basis)
            if self.built:
                self._built = False
                self.build()
            if self.built_pinv:
                self._built_pinv = False
                self.build_pinv()

    @property
    def derivatives(self):
        """Set of derivatives the transform can compute.

        Returns
        -------
        derivatives : ndarray
            combinations of derivatives needed
            Each row is one set, columns represent the order of derivatives
            for [rho, theta, zeta]
        """
        return self._derivatives

    def change_derivatives(self, derivs, build=True):
        """Change the order and updates the matrices accordingly.

        Doesn't delete any old orders, only adds new ones if not already there

        Parameters
        ----------
        derivs : int or array-like
              * if an int, order of derivatives needed (default=0)
              * if an array, derivative orders specified explicitly.
                shape should be (N,3), where each row is one set of partial derivatives
                [dr, dt, dz]

        build : bool
            whether to build transforms immediately or wait

        """
        new_derivatives = self._get_derivatives(derivs)
        new_not_in_old = (new_derivatives[:, None] == self.derivatives).all(-1).any(-1)
        derivs_to_add = new_derivatives[~new_not_in_old]
        self._derivatives = np.vstack([self.derivatives, derivs_to_add])
        self._sort_derivatives()

        if len(derivs_to_add):
            # if we actually added derivatives and didn't build them, then its not built
            self._built = False
        if build:
            # we don't update self._built here because it is still built from before
            # but it still might have unbuilt matrices from new derivatives
            self.build()

    @property
    def matrices(self):
        """dict of ndarray : transform matrices such that x=A*c."""
        if not hasattr(self, "_matrices"):
            self._matrices = {
                "direct1": {
                    i: {j: {k: {} for k in range(4)} for j in range(4)}
                    for i in range(4)
                },
                "fft": {i: {j: {} for j in range(4)} for i in range(4)},
                "direct2": {i: {} for i in range(4)},
            }
        return self._matrices

    @property
    def num_nodes(self):
        """int : number of nodes in the collocation grid."""
        return self.grid.num_nodes

    @property
    def num_modes(self):
        """int : number of modes in the spectral basis."""
        return self.basis.num_modes

    @property
    def modes(self):
        """ndarray: collocation nodes."""
        return self.grid.nodes

    @property
    def nodes(self):
        """ndarray: spectral mode numbers."""
        return self.basis.nodes

    @property
    def built(self):
        """bool : whether the transform matrices have been built."""
        if not hasattr(self, "_built"):
            self._built = False
        return self._built

    @property
    def built_pinv(self):
        """bool : whether the pseudoinverse matrix has been built."""
        if not hasattr(self, "_built_pinv"):
            self._built_pinv = False
        return self._built_pinv

    @property
    def rcond(self):
        """float: reciprocal condition number for inverse transform."""
        if not hasattr(self, "_rcond"):
            self._rcond = "auto"
        return self._rcond

    @property
    def method(self):
        """{``'direct1'``, ``'direct2'``, ``'fft'``}: method of computing transform."""
        if not hasattr(self, "_method"):
            self._method = "direct1"
        return self._method

    @method.setter
    def method(self, method):
        old_method = self.method
        if method == "auto" and self.basis.N == 0:
            self.method = "direct1"
        elif method == "auto":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.method = "fft"
        elif method == "fft":
            self._check_inputs_fft(self.grid, self.basis)
        elif method == "direct2":
            self._check_inputs_direct2(self.grid, self.basis)
        elif method == "direct1":
            self._method = "direct1"
        else:
            raise ValueError("Unknown transform method: {}".format(method))
        if self.method != old_method:
            self._built = False

    def __repr__(self):
        """String form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (method={}, basis={}, grid={})".format(
                self.method, repr(self.basis), repr(self.grid)
            )
        )
