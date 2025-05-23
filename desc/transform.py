"""Class to transform from spectral basis to real space."""

import warnings

import numpy as np
from termcolor import colored

from desc.backend import jnp, put
from desc.grid import Grid
from desc.io import IOAble
from desc.utils import combination_permutation, warnif


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
    method : {```'auto'``, `'fft'``, ``'direct1'``, ``'direct2'``, ``'jitable'``}
        * ``'fft'`` uses fast fourier transforms in the zeta direction, and so must have
          equally spaced toroidal nodes, and the same node pattern on each zeta plane.
        * ``'direct1'`` uses full matrices and can handle arbitrary node patterns and
          spectral bases.
        * ``'direct2'`` uses a DFT instead of FFT that can be faster in practice.
        * ``'jitable'`` is the same as ``'direct1'`` but avoids some checks, allowing
          you to create transforms inside JIT compiled functions.
        * ``'auto'`` selects the method based on the grid and basis resolution.

    """

    _io_attrs_ = ["_grid", "_basis", "_derivatives", "_rcond", "_method"]
    _static_attrs = ["_derivatives"]

    def __init__(
        self,
        grid,
        basis,
        derivs=0,
        rcond="auto",
        build=True,
        build_pinv=False,
        method="auto",
    ):

        self._grid = grid
        self._basis = basis
        self._rcond = rcond if rcond is not None else "auto"

        warnif(
            self.grid.coordinates != "rtz",
            msg=f"Expected coordinates rtz got {self.grid.coordinates}.",
        )
        # DESC truncates the computational domain to ζ ∈ [0, 2π/grid.NFP)
        # and changes variables to the spectrally condensed ζ* = basis.NFP ζ,
        # so basis.NFP must equal grid.NFP.
        warnif(
            method != "jitable"
            and self.grid.NFP != self.basis.NFP
            and self.basis.N != 0
            and grid.node_pattern != "custom"
            and np.any(self.grid.nodes[:, 2] != 0),
            msg=f"Unequal number of field periods for grid {self.grid.NFP} and "
            f"basis {self.basis.NFP}.",
        )

        self._built = False
        self._built_pinv = False
        self._derivatives = self._get_derivatives(derivs)
        self._sort_derivatives()
        self._method = method
        # assign according to logic in setter function
        self.method = method
        if build:
            self.build()
        if build_pinv:
            self.build_pinv()

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
            derivatives = combination_permutation(3, derivs, False)
        elif np.ndim(derivs) == 1 and len(derivs) == 3:
            derivatives = np.asarray(derivs).reshape((1, 3))
        elif np.ndim(derivs) == 2 and np.atleast_2d(derivs).shape[1] == 3:
            derivatives = np.atleast_2d(derivs)
        else:
            raise NotImplementedError(
                colored(
                    "derivs should be array-like with 3 columns, or a non-negative int",
                    "red",
                )
            )
        # always include the 0,0,0 derivative
        if not (np.array([0, 0, 0]) == derivatives).all(axis=-1).any():
            derivatives = np.concatenate([derivatives, np.array([[0, 0, 0]])])
        return derivatives

    def _sort_derivatives(self):
        """Sort derivatives."""
        sort_idx = np.lexsort(
            (self.derivatives[:, 0], self.derivatives[:, 1], self.derivatives[:, 2])
        )
        self._derivatives = self.derivatives[sort_idx]

    def _get_matrices(self):
        """Get matrices to compute all derivatives."""
        n = 4  # hardcode max derivative order for now,
        matrices = {
            "direct1": {
                i: {j: {k: {} for k in range(n + 1)} for j in range(n + 1)}
                for i in range(n + 1)
            },
            "fft": {i: {j: {} for j in range(n + 1)} for i in range(n + 1)},
            "direct2": {i: {} for i in range(n + 1)},
        }
        return matrices

    def _check_inputs_fft(self, grid, basis):
        """Check that inputs are formatted correctly for fft method."""
        if grid.num_nodes == 0 or basis.num_modes == 0:
            # trivial case where we just return all zeros, so it doesn't matter
            self._method = "direct1"
            return

        if not grid.fft_toroidal:
            warnings.warn(
                colored(
                    "fft method requires compatible grid, got {}".format(grid)
                    + "falling back to direct2 method",
                    "yellow",
                )
            )
            self.method = "direct2"
            return
        if not basis.fft_toroidal:
            warnings.warn(
                colored(
                    "fft method requires compatible basis, got {}".format(basis)
                    + "falling back to direct2 method",
                    "yellow",
                )
            )
            self.method = "direct2"
            return
        if grid.num_zeta < 2 * basis.N + 1:
            warnings.warn(
                colored(
                    "fft method can not undersample in zeta, "
                    + "num_toroidal_modes={}, num_toroidal_angles={}, ".format(
                        basis.N, grid.num_zeta
                    )
                    + "falling back to direct2 method",
                    "yellow",
                )
            )
            self.method = "direct2"
            return
        if (basis.N > 0) and (grid.NFP != basis.NFP):
            warnings.warn(
                colored(
                    "fft method requires grid and basis to have the same NFP, got "
                    + f"grid.NFP={grid.NFP}, basis.NFP={basis.NFP}, "
                    + "falling back to direct2 method",
                    "yellow",
                )
            )
            self.method = "direct2"
            return

        self._method = "fft"
        self.lm_modes = basis.modes[basis.unique_LM_idx, :2]
        self.num_lm_modes = self.lm_modes.shape[0]  # number of radial/poloidal modes
        self.num_n_modes = 2 * basis.N + 1  # number of toroidal modes
        self.pad_dim = self.grid.num_zeta - self.num_n_modes
        self.dk = basis.NFP * np.arange(-basis.N, basis.N + 1).reshape((1, -1))
        row = np.where(
            (basis.modes[:, None, :2] == self.lm_modes[None, :, :]).all(axis=-1)
        )[1]
        col = np.where(
            basis.modes[None, :, 2] == np.arange(-basis.N, basis.N + 1)[:, None, None]
        )[0]
        self.fft_index = np.atleast_1d(np.squeeze(self.num_n_modes * row + col))
        fft_nodes = np.hstack(
            [
                grid.nodes[:, :2][: grid.num_nodes // self.grid.num_zeta],
                np.zeros((grid.num_nodes // self.grid.num_zeta, 1)),
            ]
        )
        # temp grid only used for building transforms, don't need any indexing etc
        self.fft_grid = Grid(fft_nodes, sort=False, jitable=True, axis_shift=0)

    def _check_inputs_direct2(self, grid, basis):
        """Check that inputs are formatted correctly for direct2 method."""
        if grid.num_nodes == 0 or basis.num_modes == 0:
            # trivial case where we just return all zeros, so it doesn't matter
            self._method = "direct1"
            return

        from desc.grid import LinearGrid

        if not (grid.fft_toroidal or isinstance(grid, LinearGrid)):
            warnings.warn(
                colored(
                    "direct2 method requires compatible grid, got {}".format(grid)
                    + "falling back to direct1 method",
                    "yellow",
                )
            )
            self.method = "direct1"
            return
        if not basis.fft_toroidal:  # direct2 and fft have same basis requirements
            warnings.warn(
                colored(
                    "direct2 method requires compatible basis, got {}".format(basis)
                    + "falling back to direct1 method",
                    "yellow",
                )
            )
            self.method = "direct1"
            return

        self._method = "direct2"
        self.lm_modes = basis.modes[basis.unique_LM_idx, :2]
        self.n_modes = basis.modes[basis.unique_N_idx, 2]
        self.zeta_nodes = grid.nodes[grid.unique_zeta_idx, 2]
        self.num_lm_modes = self.lm_modes.shape[0]  # number of radial/poloidal modes
        self.num_n_modes = self.n_modes.size  # number of toroidal modes

        row = np.where(
            (basis.modes[:, None, :2] == self.lm_modes[None, :, :]).all(axis=-1)
        )[1]
        col = np.where(
            basis.modes[None, :, 2] == basis.modes[basis.unique_N_idx, None, 2]
        )[0]
        self.fft_index = np.atleast_1d(np.squeeze(self.num_n_modes * row + col))
        fft_nodes = np.hstack(
            [
                grid.nodes[:, :2][: grid.num_nodes // grid.num_zeta],
                np.zeros((grid.num_nodes // grid.num_zeta, 1)),
            ]
        )
        self.fft_grid = Grid(fft_nodes, sort=False, jitable=True, axis_shift=0)
        dft_nodes = np.hstack(
            [np.zeros((self.zeta_nodes.size, 2)), self.zeta_nodes[:, np.newaxis]]
        )
        self.dft_grid = Grid(dft_nodes, sort=False, jitable=True, axis_shift=0)

    def build(self):
        """Build the transform matrices for each derivative order."""
        if self.built:
            return

        if self.basis.num_modes == 0:
            self._built = True
            return

        if self.method in ["direct1", "jitable"]:
            for d in self.derivatives:
                self.matrices["direct1"][d[0]][d[1]][d[2]] = self.basis.evaluate(
                    self.grid, d
                )

        if self.method in ["fft", "direct2"]:
            temp_d = np.hstack(
                [self.derivatives[:, :2], np.zeros((len(self.derivatives), 1))]
            ).astype(int)
            temp_modes = np.hstack([self.lm_modes, np.zeros((self.num_lm_modes, 1))])
            for d in temp_d:
                self.matrices["fft"][d[0]][d[1]] = self.basis.evaluate(
                    self.fft_grid, d, modes=temp_modes
                )
        if self.method == "direct2":
            temp_d = np.hstack(
                [np.zeros((len(self.derivatives), 2)), self.derivatives[:, 2:]]
            ).astype(int)
            temp_modes = np.hstack(
                [np.zeros((self.num_n_modes, 2)), self.n_modes[:, np.newaxis]]
            )
            for d in temp_d:
                self.matrices["direct2"][d[2]] = self.basis.evaluate(
                    self.dft_grid, d, modes=temp_modes
                )

        self._built = True

    def build_pinv(self):
        """Build the pseudoinverse for fitting."""
        if self.built_pinv:
            return
        rcond = None if self.rcond == "auto" else self.rcond
        if self.method in ["direct1", "jitable"]:
            A = self.basis.evaluate(self.grid, np.array([0, 0, 0]))
            self.matrices["pinv"] = (
                jnp.linalg.pinv(A, rtol=rcond) if A.size else np.zeros_like(A.T)
            )
        elif self.method == "direct2":
            temp_modes = np.hstack([self.lm_modes, np.zeros((self.num_lm_modes, 1))])
            A = self.basis.evaluate(
                self.fft_grid, np.array([0, 0, 0]), modes=temp_modes
            )
            temp_modes = np.hstack(
                [np.zeros((self.num_n_modes, 2)), self.n_modes[:, np.newaxis]]
            )
            B = self.basis.evaluate(
                self.dft_grid, np.array([0, 0, 0]), modes=temp_modes
            )
            self.matrices["pinvA"] = (
                jnp.linalg.pinv(A, rtol=rcond) if A.size else np.zeros_like(A.T)
            )
            self.matrices["pinvB"] = (
                jnp.linalg.pinv(B, rtol=rcond) if B.size else np.zeros_like(B.T)
            )
        elif self.method == "fft":
            temp_modes = np.hstack([self.lm_modes, np.zeros((self.num_lm_modes, 1))])
            A = self.basis.evaluate(
                self.fft_grid, np.array([0, 0, 0]), modes=temp_modes
            )
            self.matrices["pinvA"] = (
                jnp.linalg.pinv(A, rtol=rcond) if A.size else np.zeros_like(A.T)
            )
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
            raise RuntimeError(
                "Transform must be precomputed with transform.build() before being used"
            )

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

        if self.method in ["direct1", "jitable"]:
            A = self.matrices["direct1"].get(dr, {}).get(dt, {}).get(dz, {})
            if isinstance(A, dict):
                raise ValueError(
                    colored("Derivative orders are out of initialized bounds", "red")
                )
            return A @ c

        elif self.method == "direct2":
            A = self.matrices["fft"].get(dr, {}).get(dt, {})
            B = self.matrices["direct2"].get(dz, {})
            if isinstance(A, dict) or isinstance(B, dict):
                raise ValueError(
                    colored("Derivative orders are out of initialized bounds", "red")
                )
            c_mtrx = jnp.zeros((self.num_lm_modes * self.num_n_modes,))
            c_mtrx = put(c_mtrx, self.fft_index, c).reshape((-1, self.num_n_modes))
            cc = A @ c_mtrx
            return (cc @ B.T).flatten(order="F")

        elif self.method == "fft":
            A = self.matrices["fft"].get(dr, {}).get(dt, {})
            if isinstance(A, dict):
                raise ValueError(
                    colored("Derivative orders are out of initialized bounds", "red")
                )
            # reshape coefficients
            c_mtrx = jnp.zeros((self.num_lm_modes * self.num_n_modes,))
            c_mtrx = put(c_mtrx, self.fft_index, c).reshape((-1, self.num_n_modes))
            # differentiate
            c_diff = c_mtrx[:, :: (-1) ** dz] * self.dk**dz * (-1) ** (dz > 1)
            # re-format in complex notation
            c_cplx = (self.grid.num_zeta / 2) * (
                c_diff[:, self.basis.N + 1 :] - 1j * c_diff[:, self.basis.N - 1 :: -1]
            )
            c_pad = jnp.hstack(
                (
                    self.grid.num_zeta * c_diff[:, self.basis.N, jnp.newaxis],
                    c_cplx,
                    jnp.zeros((c_cplx.shape[0], self.pad_dim)),
                    jnp.fliplr(jnp.conj(c_cplx)),
                )
            )
            # transform coefficients
            c_fft = jnp.real(jnp.fft.ifft(c_pad))
            return (A @ c_fft).flatten(order="F")

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
            raise RuntimeError(
                "Transform must be built with transform.build_pinv() before being used"
            )

        if self.method == "direct1":
            Ainv = self.matrices["pinv"]
            c = jnp.matmul(Ainv, x)
        elif self.method == "direct2":
            Ainv = self.matrices["pinvA"]
            Binv = self.matrices["pinvB"]
            yy = jnp.matmul(Ainv, x.reshape((-1, self.grid.num_zeta), order="F"))
            c = jnp.matmul(Binv, yy.T).T.flatten()[self.fft_index]
        elif self.method == "fft":
            Ainv = self.matrices["pinvA"]
            c_fft = jnp.matmul(Ainv, x.reshape((Ainv.shape[1], -1), order="F"))
            c_cplx = jnp.fft.fft(c_fft)
            c_unpad = c_cplx[:, 1 : (c_cplx.shape[1] - self.pad_dim - 1) // 2 + 1]
            c0 = c_cplx[:, :1].real / self.grid.num_zeta
            c2 = c_unpad.real / (self.grid.num_zeta / 2)
            c1 = -c_unpad.imag[:, ::-1] / (self.grid.num_zeta / 2)
            c_diff = jnp.hstack([c1, c0, c2])
            c = c_diff.flatten()[self.fft_index]
        return c

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
            raise RuntimeError(
                "Transform must be precomputed with transform.build() before being used"
            )

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
            yy = jnp.matmul(A.T, y.reshape((-1, self.grid.num_zeta), order="F"))
            return jnp.matmul(yy, B).flatten()[self.fft_index]

        elif self.method == "fft":
            A = self.matrices["fft"][0][0]
            # this was derived by trial and error, but seems to work correctly
            # there might be a more efficient way...
            a = jnp.fft.fft(A.T @ y.reshape((A.shape[0], -1), order="F"))
            cdn = a[:, 0]
            cr = a[:, 1 : 1 + self.basis.N]
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

        if not self.grid.equiv(grid):
            self._grid = grid
            self._built = False
            self._built_pinv = False
        if not self.basis.equiv(basis):
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
        return self.__dict__.setdefault("_grid", None)

    @grid.setter
    def grid(self, grid):
        if not self.grid.equiv(grid):
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
        return self.__dict__.setdefault("_basis", None)

    @basis.setter
    def basis(self, basis):
        if not self.basis.equiv(basis):
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
            # if we actually added derivatives and didn't build them, then it's not
            # built
            self._built = False
        if build:
            # we don't update self._built here because it is still built from before,
            # but it still might have unbuilt matrices from new derivatives
            self.build()

    @property
    def matrices(self):
        """dict: transform matrices such that x=A*c."""
        if not hasattr(self, "_matrices"):
            self._matrices = self._get_matrices()
        return self._matrices

    @property
    def num_nodes(self):
        """int: number of nodes in the collocation grid."""
        return self.grid.num_nodes

    @property
    def num_modes(self):
        """int: number of modes in the spectral basis."""
        return self.basis.num_modes

    @property
    def nodes(self):
        """ndarray: collocation nodes."""
        return self.grid.nodes

    @property
    def modes(self):
        """ndarray: spectral mode numbers."""
        return self.basis.modes

    @property
    def built(self):
        """bool: whether the transform matrices have been built."""
        return self.__dict__.setdefault("_built", False)

    @property
    def built_pinv(self):
        """bool: whether the pseudoinverse matrix has been built."""
        return self.__dict__.setdefault("_built_pinv", False)

    @property
    def rcond(self):
        """float: reciprocal condition number for inverse transform."""
        return self.__dict__.setdefault("_rcond", "auto")

    @property
    def method(self):
        """{``'direct1'``, ``'direct2'``, ``'fft'``}: method of computing transform."""
        return self.__dict__.setdefault("_method", "direct1")

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
        elif method == "jitable":
            self._method = "jitable"
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
