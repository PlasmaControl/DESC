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
        * ``'rpz'`` uses a fast fourier transform in the phi (second) dimension, and
            direct cosine transforms in the R and Z (first and third) dimensions so
            must have a compatible grid (CylindricalGrid) and basis (DoubleChebyshev
            FourierBasis). This method will never be chosen by auto.
        * ``'partialrpz'`` uses a fast fourier transform in the phi (second) dimension,
            but directly evaluates the Chebyshev functions in the first and third
            dimensions. still must have a compatible grid (CylindricalGrid) and basis
            (DoubleChebyshevFourierBasis), but can have arbitrary node patterns in
            the first and third dimensions. This method will never be chosen by auto.
        * ``'directrpz'``  directly evaluates the Chebyshev functions in the first
            and third dimensions and evaluates the Fourier functions in the phi
            dimension, but uses partial sums to improve memory usage compared to
            direct1. still must have a compatible grid (CylindricalGrid) and basis
            (DoubleChebyshevFourierBasis), but can have arbitrary node patterns in each
            dimension. This method will never be chosen by auto.
        * ``'auto'`` selects the method based on the grid and basis resolution.

    """

    _io_attrs_ = ["_grid", "_basis", "_derivatives", "_rcond", "_method"]
    _static_attrs = [
        "_basis",
        "_derivatives",
        "_method",
        "_rcond",
        "_built",
        "_built_pinv",
        "num_n_modes",
        "num_lm_modes",
        "pad_dim",
    ]

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
            self.grid.coordinates not in ["rtz", "rpz"],
            msg=f"Expected coordinates rtz or rpz, got {self.grid.coordinates}.",
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
        n = 4  # hardcode max derivative order for now
        ndi = self.derivatives[:, 0].max()
        ndj = self.derivatives[:, 1].max()
        ndk = self.derivatives[:, 2].max()
        if self.method == "jitable":
            matrices = {
                "direct1": {
                    i: {j: {k: {} for k in range(n + 1)} for j in range(n + 1)}
                    for i in range(n + 1)
                },
            }
        elif self.method == "fft":
            matrices = {
                "fft": {i: {j: {} for j in range(ndj + 1)} for i in range(ndi + 1)},
            }
        elif self.method == "direct2":
            matrices = {
                "fft": {i: {j: {} for j in range(ndj + 1)} for i in range(ndi + 1)},
                "direct2": {k: {} for k in range(ndk + 1)},
            }
        elif self.method == "direct1":
            matrices = {
                "direct1": {
                    i: {j: {k: {} for k in range(ndk + 1)} for j in range(ndj + 1)}
                    for i in range(ndi + 1)
                },
            }
        elif self.method in ["rpz", "partialrpz", "directrpz"]:
            matrices = {
                "rpz": {
                    "dr": {i: {} for i in range(ndi + 1)},
                    "dphi": {j: {} for j in range(ndj + 1)},
                    "dz": {k: {} for k in range(ndk + 1)},
                },
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

    def _check_inputs_partialrpz(self, grid, basis):
        from desc.basis import DoubleChebyshevFourierBasis
        from desc.grid import CylindricalGrid

        if not isinstance(grid, CylindricalGrid) or not isinstance(
            basis, DoubleChebyshevFourierBasis
        ):
            warnings.warn(
                colored(
                    "partialRPZ method requires the compatible basis and grid"
                    + "got {} grid".format(grid)
                    + " and {} basis".format(basis)
                    + "falling back to direct1 method",
                    "yellow",
                )
            )
            self.method = "direct1"
            return
        if grid.NFP != basis.NFP:
            warnings.warn(
                colored(
                    "partialrpz method requires basis and grid to have the same NFP."
                    + "got {} grid NFP".format(grid.NFP)
                    + " and basis NFP {}".format(basis.NFP)
                    + "falling back to direct1 method",
                    "yellow",
                )
            )
            self.method = "direct1"
            return
        # Coefficients for Fourier derivatives in spectral coordinates (phi basis)
        self.dk = self.basis.NFP * jnp.arange(-self.basis.M, self.basis.M + 1)
        self._method = "partialrpz"

    def _check_inputs_directrpz(self, grid, basis):
        if not grid.is_meshgrid:
            warnings.warn(
                colored(
                    "Direct RPZ method requires a tensor product grid."
                    + "falling back to direct1 method",
                    "yellow",
                )
            )
            self.method = "direct1"
            return
        from desc.basis import DoubleChebyshevFourierBasis

        if not isinstance(basis, DoubleChebyshevFourierBasis):
            warnings.warn(
                colored(
                    "Direct RPZ method requires a basis of type"
                    + " DoubleChebyshevFourierBasis."
                    + "falling back to direct1 method",
                    "yellow",
                )
            )
            self.method = "direct1"
            return

    def _check_inputs_rpz(self, grid, basis):
        from desc.basis import DoubleChebyshevFourierBasis
        from desc.grid import CylindricalGrid

        if not isinstance(grid, CylindricalGrid) or not isinstance(
            basis, DoubleChebyshevFourierBasis
        ):
            warnings.warn(
                colored(
                    "RPZ method requires the compatible basis and grid"
                    + " got {} grid".format(grid)
                    + " and {} basis".format(basis)
                    + "falling back to direct1 method",
                    "yellow",
                )
            )
            self.method = "direct1"
            return
        if (grid.L, grid.N) != (basis.L, basis.N):
            warnings.warn(
                colored(
                    "RPZ method requires basis and grid to have same L and N"
                    + " got grid L,N={}".format((grid.L, grid.N))
                    + " and basis L,N={}".format((basis.L, basis.N))
                    + "falling back to direct1 method",
                    "yellow",
                )
            )
            self.method = "direct1"
            return
        if grid.NFP != basis.NFP:
            warnings.warn(
                colored(
                    "rpz method requires basis and grid to have the same NFP."
                    + "got {} grid NFP".format(grid.NFP)
                    + " and basis NFP {}".format(basis.NFP)
                    + "falling back to direct1 method",
                    "yellow",
                )
            )
            self.method = "direct1"
            return
        if not grid.can_fft_dct:
            warnings.warn(
                colored(
                    "RPZ method does not support 0-resolution grids or custom"
                    + " node patterns when defining the CylindricalGrid object"
                    + " falling back to direct1 method."
                    "yellow",
                )
            )
            self.method = "direct1"
            return
        self._method = "rpz"

        # Coefficients for Fourier derivatives in spectral coordinates (phi basis)
        self.dk = self.basis.NFP * jnp.arange(-self.basis.M, self.basis.M + 1)

    def build(self):
        """Build the transform matrices for each derivative order."""
        if self.built:
            return

        if self.method in ["rpz", "partialrpz", "directrpz"]:
            self._build_rpz()
        else:
            self._build_rtz()

        self._built = True

    def _build_rtz(self):
        """Build transforms matrices for each derivative order, for flux coordinates."""
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

    def _build_rpz(self):
        """Build matrices for each derivative order, for cylindrical coordinates."""
        from desc.basis import chebyshev, fourier

        r = self.grid.nodes[self.grid.unique_r_idx, 0]
        z = self.grid.nodes[self.grid.unique_z_idx, 2]

        # Differentiation matrices (only on unique R and Z coordinates)
        for dr in np.unique(self.derivatives[:, 0]):
            if dr > 0 or self.method in ["partialrpz", "directrpz"]:
                self.matrices["rpz"]["dr"][dr] = chebyshev(
                    r[:, jnp.newaxis],
                    self.basis.modes[self.basis.unique_L_idx, 0],
                    dr=dr,
                )
        for dz in np.unique(self.derivatives[:, 2]):
            if dz > 0 or self.method in ["partialrpz", "directrpz"]:
                self.matrices["rpz"]["dz"][dz] = chebyshev(
                    z[:, jnp.newaxis],
                    self.basis.modes[self.basis.unique_N_idx, 2],
                    dr=dz,
                )
        # Differentiation matices for unique phi coordinates
        if self.method == "directrpz":
            phi = self.grid.nodes[self.grid.unique_phi_idx, 1]
            for dphi in np.unique(self.derivatives[:, 1]):
                self.matrices["rpz"]["dphi"][dphi] = fourier(
                    phi[:, jnp.newaxis],
                    self.basis.modes[self.basis.unique_M_idx, 1],
                    NFP=self.basis.NFP,
                    dt=dphi,
                )

    def build_pinv(self):
        """Build the pseudoinverse for fitting."""
        if self.built_pinv:
            return
        rcond = None if self.rcond == "auto" else self.rcond
        if self.method in ["direct1", "jitable", "partialrpz", "directrpz"]:
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
        elif self.method == "rpz":
            pass
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
        elif self.method in ["rpz", "partialrpz", "directrpz"]:
            from desc.basis import ichebfit, ifftfit

            # reshape coefficients (Z,R,phi)
            c_3d = c.reshape(
                self.basis.unique_N_idx.shape[0],
                self.basis.unique_L_idx.shape[0],
                self.basis.unique_M_idx.shape[0],
                -1,
            )
            dphi = dt
            if self.method == "directrpz":
                y = (
                    self.matrices["rpz"]["dphi"][dphi] @ c_3d.swapaxes(2, -2)
                ).swapaxes(2, -2)
            else:
                # differentiate with respect to phi
                y = (
                    c_3d[:, :, :: (-1) ** dphi]
                    * self.dk.reshape((-1, 1)) ** dphi
                    * (-1) ** (dphi > 1)
                )
                y = ifftfit(y, axis=2, n=self.grid.M)

            # differentiate with respect to r
            if dr == 0 and self.method == "rpz":
                y = ichebfit(y, axis=1)
            else:
                y = (self.matrices["rpz"]["dr"][dr] @ y.swapaxes(1, -2)).swapaxes(1, -2)

            # differentiate with respect to z
            if dz == 0 and self.method == "rpz":
                y = ichebfit(y, axis=0)
            else:
                y = (self.matrices["rpz"]["dz"][dz] @ y.swapaxes(0, -2)).swapaxes(0, -2)

            return jnp.squeeze(y.reshape(self.grid.num_nodes, -1))

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

        if self.method in ["direct1", "jitable", "partialrpz"]:
            # Partial RPZ method just defaults to direct1 because it
            # probably won't be necessary to fit a transform to a grid that
            # doesn't have Chebyshev nodes in the R and Z directions
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
        elif self.method == "rpz":
            from desc.basis import chebfit, fftfit

            x_3d = x.reshape(self.grid.num_z, self.grid.num_r, self.grid.num_phi, -1)
            c = fftfit(chebfit(chebfit(x_3d, axis=0), axis=1), axis=2, n=self.basis.M)
            c = (
                c.reshape(self.basis.num_modes)
                if x.ndim == 1
                else c.reshape(self.basis.num_modes, -1)
            )

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

        if self.method in ["direct1", "jitable", "rpz"]:
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
        """{``'direct1'``, ``'direct2'``, ``'fft'``, ``'jitable'``}.

        Transform compute method.
        """
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
        elif method == "rpz":
            self._check_inputs_rpz(self.grid, self.basis)
        elif method == "partialrpz":
            self._check_inputs_partialrpz(self.grid, self.basis)
        elif method == "directrpz":
            self._check_inputs_directrpz(self.grid, self.basis)
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
