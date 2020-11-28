import numpy as np
import functools
from itertools import permutations, combinations_with_replacement

from desc.backend import jnp, conditional_decorator, jit, use_jax, TextColors
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

    def __init__(self, grid:Grid, basis:Basis, order=0, rcond=1e-6) -> None:
        """Initializes a Transform

        Parameters
        ----------
        grid : Grid
            DESCRIPTION
        basis : Basis
            DESCRIPTION
        order : int or string
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
        self.__rcond = rcond
        self.__matrices = {i: {j: {k: {}
                     for k in range(4)} for j in range(4)} for i in range(4)}

        self.__derivatives = self.get_derivatives(order)
        self.sort_derivatives()
        self.build()

    def build(self):
        """"""
        for d in self.__derivatives:
            dr = d[0]
            dv = d[1]
            dz = d[2]
            self.__matrices[dr][dv][dz] = self.__basis.evaluate(
                                        self.__grid.nodes, self.__derivatives)

        # TODO: this assumes the derivatives are sorted (which they should be)
        if np.all(self.__derivatives[0, :] == np.array([0, 0, 0])):
            A = self.__matrices[0][0][0]
        else:
            A = self.__basis.evaluate(self.__grid.nodes, np.array([0, 0, 0]))
        self.__pinv = np.linalg.pinv(A, rcond=self.__rcond)

    def transform(self, c, dr, dv, dz):
        """Transform from spectral domain to physical

        Parameters
        ----------
        c : ndarray, shape(N_coeffs,)
            spectral coefficients, indexed as (lm,n) flattened in row major order
        dr : int
            order of radial derivative
        dv : int
            order of poloidal derivative
        dz : int
            order of toroidal derivative

        Returns
        -------
        x : ndarray, shape(N_nodes,)
            array of values of function at node locations

        """
        return jnp.matmul(self.__matrices[dr][dv][dz], c)

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

    def get_derivatives(self, order):
        """Get array of derivatives needed for calculating objective function

        Parameters
        ----------
        order : int or string
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
        if isinstance(order, int) and order >= 0:
            derivatives = np.array([[]])
            combos = combinations_with_replacement(range(order+1), 3)
            for combo in list(combos):
                perms = set(permutations(combo))
                for perm in list(perms):
                    if derivatives.shape[1] == 3:
                        derivatives = np.vstack([derivatives, np.array(perm)])
                    else:
                        derivatives = np.array([perm])
        elif order.lower() == 'force':
            derivatives = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                                    [2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0],
                                    [0, 1, 1], [0, 0, 2]])
            # TODO: this assumes the Grid is sorted (which it should be)
            if np.all(self.__grid.nodes[:, 0] == np.array([0, 0, 0])):
                axis = np.array([[2, 1, 0], [1, 2, 0], [1, 1, 1], [2, 2, 0]])
                derivatives = np.vstack([derivatives, axis])
        elif order.lower() == 'qs':
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

    def sort_derivatives(self) -> None:
        """Sorts derivatives

        Returns
        -------
        None

        """
        sort_idx = np.lexsort((self.__derivatives[:, 0],
                       self.__derivatives[:, 1], self.__derivatives[:, 2]))
        self.__derivatives = self.__derivatives[sort_idx]

    def change_resolution(self, zern_idx_new):
        """Change the spectral resolution of the transform without full recompute

        Only computes modes that aren't already in the basis

        Parameters
        ----------
        zern_idx_new : ndarray of int, shape(Nc,3)
            new mode numbers for spectral basis.
            each row is one basis function with modes (l,m,n)

        Returns
        -------

        """
        if self.method == 'direct':
            zern_idx_new = jnp.atleast_2d(zern_idx_new)
            # first remove modes that are no longer needed
            old_in_new = (self.zern_idx[:, None] ==
                          zern_idx_new).all(-1).any(-1)
            for d in self.derivatives:
                self.matrices[d[0]][d[1]][d[2]] = self.matrices[d[0]
                                                                ][d[1]][d[2]][:, old_in_new]
            self.zern_idx = self.zern_idx[old_in_new, :]
            # then add new modes
            new_not_in_old = ~(zern_idx_new[:, None]
                               == self.zern_idx).all(-1).any(-1)
            modes_to_add = zern_idx_new[new_not_in_old]
            if len(modes_to_add) > 0:
                for d in self.derivatives:
                    self.matrices[d[0]][d[1]][d[2]] = jnp.hstack([
                        self.matrices[d[0]][d[1]][d[2]],  # old
                        fourzern(self.nodes[0], self.nodes[1], self.nodes[2],  # new
                                 modes_to_add[:, 0], modes_to_add[:, 1], modes_to_add[:, 2], self.NFP, d[0], d[1], d[2])])

            # update indices
            self.zern_idx = np.vstack([self.zern_idx, modes_to_add])
            # permute indexes so they're in the same order as the input
            permute_idx = [self.zern_idx.tolist().index(i)
                           for i in zern_idx_new.tolist()]
            for d in self.derivatives:
                self.matrices[d[0]][d[1]][d[2]] = self.matrices[d[0]
                                                                ][d[1]][d[2]][:, permute_idx]
            self.zern_idx = self.zern_idx[permute_idx]
            self._build_pinv()

        elif self.method == 'fft':
            self._check_inputs_fft(self.nodes, zern_idx_new)
            self.zern_idx = zern_idx_new
            self._build()

    def change_derivatives(self, new_derivatives):
        """Computes new derivative matrices

        Parameters
        ----------
        new_derivatives : ndarray of int, , shape(Nd,3)
            orders of derivatives
            to compute in rho,theta,zeta. Each row of the array should
            contain 3 elements corresponding to derivatives in rho,theta,zeta

        Returns
        -------

        """
        new_not_in_old = (
            new_derivatives[:, None] == self.derivatives).all(-1).any(-1)
        derivs_to_add = new_derivatives[~new_not_in_old]
        if self.method == 'direct':
            for d in derivs_to_add:
                dr = d[0]
                dv = d[1]
                dz = d[2]
                self.matrices[dr][dv][dz] = fourzern(self.nodes[0], self.nodes[1], self.nodes[2],
                                                     self.zern_idx[:, 0], self.zern_idx[:, 1], self.zern_idx[:, 2],
                                                     self.NFP, dr, dv, dz)

        elif self.method == 'fft':
            for d in derivs_to_add:
                dr = d[0]
                dv = d[1]
                dz = 0
                self.matrices[dr][dv][dz] = zern(self.pol_nodes[0], self.pol_nodes[1],
                                                 self.pol_zern_idx[:, 0], self.pol_zern_idx[:, 1], dr, dv)

        self.derivatives = jnp.vstack([self.derivatives, derivs_to_add])

    def change_nodes(self, new_nodes, new_volumes=None):
        """Change the real space resolution by adding new nodes without full recompute

        Only computes basis at spatial nodes that aren't already in the basis

        Parameters
        ----------
        new_nodes : ndarray, shape(3,N)
            new node locations. each column is the location of one node (rho,theta,zeta)
        new_volumes : ndarray, shape(3,N)
            volume elements around each new node (dr,dtheta,dzeta) (Default value = None)

        Returns
        -------

        """
        if new_volumes is None:
            new_volumes = np.ones_like(new_nodes)

        if self.method == 'direct':
            new_nodes = jnp.atleast_2d(new_nodes).T
            # first remove nodes that are no longer needed
            old_in_new = (self.nodes.T[:, None] == new_nodes).all(-1).any(-1)
            for d in self.derivatives:
                self.matrices[d[0]][d[1]][d[2]
                                          ] = self.matrices[d[0]][d[1]][d[2]][old_in_new]
            self.nodes = self.nodes[:, old_in_new]

            # then add new nodes
            new_not_in_old = ~(new_nodes[:, None]
                               == self.nodes.T).all(-1).any(-1)
            nodes_to_add = new_nodes[new_not_in_old]
            if len(nodes_to_add) > 0:
                for d in self.derivatives:
                    self.matrices[d[0]][d[1]][d[2]] = jnp.vstack([
                        self.matrices[d[0]][d[1]][d[2]],  # old
                        fourzern(nodes_to_add[:, 0], nodes_to_add[:, 1], nodes_to_add[:, 2],  # new
                                 self.zern_idx[:, 0], self.zern_idx[:, 1], self.zern_idx[:, 2], self.NFP, d[0], d[1], d[2])])

            # update indices
            self.nodes = np.hstack([self.nodes, nodes_to_add.T])
            # permute indexes so they're in the same order as the input
            permute_idx = [self.nodes.T.tolist().index(i)
                           for i in new_nodes.tolist()]
            for d in self.derivatives:
                self.matrices[d[0]][d[1]][d[2]
                                          ] = self.matrices[d[0]][d[1]][d[2]][permute_idx, :]
            self.nodes = self.nodes[:, permute_idx]
            self.volumes = new_volumes[:, permute_idx]
            self.axn = np.where(self.nodes[0] == 0)[0]
            self._build_pinv()

        elif self.method == 'fft':
            self._check_inputs_fft(new_nodes, self.zern_idx)
            self.nodes = new_nodes
            self.axn = np.where(self.nodes[0] == 0)[0]
            self.volumes = new_volumes
            self._build()

    @property
    def grid(self):
        return self.__grid

    @grid.setter
    def grid(self, grid:Grid):
        self.__grid = grid

    @property
    def basis(self):
        return self.__basis

    @basis.setter
    def basis(self, basis:Basis):
        self.__basis = basis

    @property
    def derivatives(self):
        return self.__derivatives

    @derivatives.setter
    def derivatives(self, derivatives):
        self.__derivatives = derivatives

    @property
    def matrices(self):
        return self.__matrices


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
