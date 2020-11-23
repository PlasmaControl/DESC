import numpy as np
import functools
import warnings
import numba
from desc.backend import jnp, conditional_decorator, jit, use_jax, fori_loop
from desc.backend import issorted, isalmostequal, sign, TextColors
from desc.backend import polyder_vec, polyval_vec, flatten_list


class Transform():
    """Zernike Transform (really a Fourier-Zernike, but whatever)

    Parameters
    ----------
    nodes : ndarray, shape(3,N)
        nodes where basis functions are evaluated.
        First index is (rho,theta,phi), 2nd index is node number
    zern_idx : ndarray of int, shape(Nc,3)
        mode numbers for spectral basis. each row is one basis function with modes (l,m,n)
    NFP : int
        number of field periods
    derivatives : ndarray of int, shape(Nd,3)
        orders of derivatives to compute in rho,theta,zeta.
        Each row of the array should contain 3 elements corresponding to derivatives in rho,theta,zeta
    volumes : ndarray, shape(3,N)
        volume elements at each node, dr,dv,dz
    method : str
        one of 'direct', or 'fft'. 'direct' uses full matrices and can handle arbitrary
        node patterns and spectral basis. 'fft' uses fast fourier transforms in the zeta direction,
        and so must have equally spaced toroidal nodes, and the same node pattern on each zeta plane
    pinv_rcond : float
        relative cutoff for singular values in least squares fit

    Returns
    -------

    """

    def __init__(self, nodes, zern_idx, NFP, derivatives=[0, 0, 0], volumes=None, method='direct', pinv_rcond=1e-6):

        # array of which l,m,n is at which column of the interpolation matrix
        self.zern_idx = zern_idx
        # array of which r,v,z is at which row of the interpolation matrix
        self.nodes = nodes
        self.axn = np.where(nodes[0] == 0)[0]
        self.NFP = NFP
        self.derivatives = np.atleast_2d(derivatives)
        self.volumes = volumes if volumes is not None else np.ones_like(nodes)
        self.pinv_rcond = pinv_rcond
        self.matrices = {i: {j: {k: {}
                                 for k in range(4)} for j in range(4)} for i in range(4)}

        if method in ['direct', 'fft']:
            self.method = method
        else:
            raise ValueError(TextColors.FAIL +
                             "Unknown Zernike Transform method '{}'".format(method) + TextColors.ENDC)
        if self.method == 'fft':
            self._check_inputs_fft(nodes, zern_idx)
        self._build()

    def _build_pinv(self):
        """ """
        A = fourzern(self.nodes[0], self.nodes[1], self.nodes[2], self.zern_idx[:, 0],
                     self.zern_idx[:, 1], self.zern_idx[:, 2], self.NFP, 0, 0, 0)
        self.pinv = jnp.linalg.pinv(A, rcond=self.pinv_rcond)

    def _build(self):
        """helper function to build matrices"""
        self._build_pinv()

        if self.method == 'direct':
            for d in self.derivatives:
                dr = d[0]
                dv = d[1]
                dz = d[2]
                self.matrices[dr][dv][dz] = fourzern(self.nodes[0], self.nodes[1], self.nodes[2],
                                                     self.zern_idx[:, 0], self.zern_idx[:,
                                                                                        1], self.zern_idx[:, 2],
                                                     self.NFP, dr, dv, dz)
        elif self.method == 'fft':
            for d in self.derivatives:
                dr = d[0]
                dv = d[1]
                dz = 0
                self.matrices[dr][dv][dz] = zern(self.pol_nodes[0], self.pol_nodes[1],
                                                 self.pol_zern_idx[:, 0], self.pol_zern_idx[:, 1], dr, dv)

    def _check_inputs_fft(self, nodes, zern_idx):
        """helper function to check that inputs are formatted correctly for fft method

        """
        zeta_vals, zeta_cts = np.unique(nodes[2], return_counts=True)

        if not issorted(nodes[2]):
            warnings.warn(TextColors.WARNING +
                          "fft method requires nodes to be sorted by toroidal angle in ascending order, falling back to direct method" + TextColors.ENDC)
            self.method = 'direct'
            return

        if not isalmostequal(zeta_cts):
            warnings.warn(TextColors.WARNING +
                          "fft method requires the same number of nodes on each zeta plane, falling back to direct method" + TextColors.ENDC)
            self.method = 'direct'
            return

        if len(zeta_vals) > 1:
            if not np.diff(zeta_vals).std() < 1e-14:
                warnings.warn(TextColors.WARNING +
                              "fft method requires nodes to be equally spaced in zeta, falling back to direct method" + TextColors.ENDC)
                self.method = 'direct'
                return

            if not isalmostequal(nodes[:2].reshape((zeta_cts[0], 2, -1), order='F')):
                warnings.warn(TextColors.WARNING +
                              "fft method requires that node pattern is the same on each zeta plane, falling back to direct method" + TextColors.ENDC)
                self.method = 'direct'
                return
            if not abs((zeta_vals[-1] + zeta_vals[1])*self.NFP - 2*np.pi) < 1e-14:
                warnings.warn(TextColors.WARNING +
                              "fft method requires that nodes complete 1 full field period, falling back to direct method" + TextColors.ENDC)
                self.method = 'direct'
                return

        id2 = np.lexsort((zern_idx[:, 1], zern_idx[:, 0], zern_idx[:, 2]))
        if not issorted(id2):
            warnings.warn(TextColors.WARNING +
                          "fft method requires zernike indices to be sorted by toroidal mode number, falling back to direct method" + TextColors.ENDC)
            self.method = 'direct'
            return

        n_vals, n_cts = np.unique(zern_idx[:, 2], return_counts=True)
        if not isalmostequal(n_cts):
            warnings.warn(TextColors.WARNING +
                          "fft method requires that there are the same number of poloidal modes for each toroidal mode, falling back to direct method" + TextColors.ENDC)
            self.method = 'direct'
            return

        if len(n_vals) > 1:
            if not np.diff(n_vals).std() < 1e-14:
                warnings.warn(TextColors.WARNING +
                              "fft method requires the toroidal modes are equally spaced in n, falling back to direct method" + TextColors.ENDC)
                self.method = 'direct'
                return

            if not isalmostequal(zern_idx[:, 0].reshape((n_cts[0], -1), order='F')) \
               or not isalmostequal(zern_idx[:, 1].reshape((n_cts[0], -1), order='F')):
                warnings.warn(TextColors.WARNING +
                              "fft method requires that the poloidal modes are the same for each toroidal mode, falling back to direct method" + TextColors.ENDC)
                self.method = 'direct'
                return

        if not len(zeta_vals) >= len(n_vals):
            warnings.warn(TextColors.WARNING + "fft method can not undersample in zeta, num_zeta_vals={}, num_n_vals={}, falling back to direct method".format(
                len(zeta_vals), len(n_vals)) + TextColors.ENDC)
            self.method = 'direct'
            return

        self.numFour = len(n_vals)  # number of toroidal modes
        self.numFournodes = len(zeta_vals)  # number of toroidal nodes
        self.zeta_pad = (self.numFournodes - self.numFour)//2
        self.pol_zern_idx = zern_idx[:len(zern_idx)//self.numFour, :2]
        self.pol_nodes = nodes[:2, :len(nodes[0])//self.numFournodes]

    def expand_nodes(self, new_nodes, new_volumes=None):
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

    def expand_spectral_resolution(self, zern_idx_new):
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

    def expand_derivatives(self, new_derivatives):
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
                                                     self.zern_idx[:, 0], self.zern_idx[:,
                                                                                        1], self.zern_idx[:, 2],
                                                     self.NFP, dr, dv, dz)

        elif self.method == 'fft':
            for d in derivs_to_add:
                dr = d[0]
                dv = d[1]
                dz = 0
                self.matrices[dr][dv][dz] = zern(self.pol_nodes[0], self.pol_nodes[1],
                                                 self.pol_zern_idx[:, 0], self.pol_zern_idx[:, 1], dr, dv)

        self.derivatives = jnp.vstack([self.derivatives, derivs_to_add])

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
        if self.method == 'direct':
            return self._matmul(self.matrices[dr][dv][dz], c)

        elif self.method == 'fft':
            c_pad = jnp.pad(c.reshape((-1, self.numFour), order='F'),
                            ((0, 0), (self.zeta_pad, self.zeta_pad)), mode='constant')
            dk = self.NFP*jnp.arange(-(self.numFournodes//2),
                                     (self.numFournodes//2)+1).reshape((1, -1))
            c_pad = c_pad[:, ::(-1)**dz]*dk**dz * (-1)**(dz > 1)
            cfft = self._four2phys(c_pad)
            return self._matmul(self.matrices[dr][dv][0], cfft).flatten(order='F')

    @conditional_decorator(functools.partial(jit, static_argnums=(0,)), use_jax)
    def _four2phys(self, c):
        """helper function to do ffts

        """
        K, L = c.shape
        N = (L-1)//2
        # pad with negative wavenumbers
        a = c[:, N:]
        b = c[:, :N][:, ::-1]
        a = jnp.hstack([a[:, 0][:, jnp.newaxis],
                        a[:, 1:]/2,   a[:, 1:][:, ::-1]/2])
        b = jnp.hstack([jnp.zeros((K, 1)), -b[:, 0:]/2, b[:, ::-1]/2])
        # inverse Fourier transform
        a = a*L
        b = b*L
        c = a + 1j*b
        x = jnp.real(jnp.fft.ifft(c, None, 1))
        return x

    @conditional_decorator(functools.partial(jit, static_argnums=(0,)), use_jax)
    def _matmul(self, A, x):
        """helper function for matrix multiplication that we can jit more easily

        """
        return jnp.matmul(A, x)

    @conditional_decorator(functools.partial(jit, static_argnums=(0,)), use_jax)
    def fit(self, x):
        """Transform from physical domain to spectral using least squares fit

        Parameters
        ----------
        x : ndarray, shape(N_nodes,)
            values in real space at coordinates specified by self.nodes

        Returns
        -------
        c : ndarray, shape(N_coeffs,)
            spectral coefficients in Fourier-Zernike basis

        """
        return jnp.matmul(self.pinv, x)


def get_zern_basis_idx_dense(M, N, delta_lm=-1, indexing='fringe'):
    """Gets mode numbers for dense spectral representation in zernike basis.

    Parameters
    ----------
    M : int
        maximum poloidal resolution
    N : int
        maximum toroidal resolution
    delta_lm : int
        maximum difference between poloidal and radial
        resolution (l-m). If < 0, defaults to ``M`` for 'ansi' or
        'chevron' indexing, ``2*M`` for 'fringe' or 'house'.
    indexing : str
        One of ('frige','ansi','house','chevron').
        Zernike indexing method. For delta_lm=0, all methods are equivalent and 
        give a "chevron" shaped basis (only the outer edge of the zernike pyramid 
        of width M).
        For delta_lm>0, the indexing scheme defines how the pyramid is filled in:
        ``'fringe'``: Fringe indexing fills in the pyramid with chevrons of
        decreasing size, ending in a diamond shape. The maximum delta_lm
        is 2*M, for which the traditional "Fringe/ U of Arizona" indexing
        is recovered. Gives a single mode at maximum m and a single mode
        at maximum l and m=0. 
        Total number of modes = (M+1)*(M+2)/2 - (M-delta_lm//2+1)*(M-delta_lm//2)/2
        ``'ansi'``: ANSI indexing fills in the pyramid with triangles of
        decreasing size, ending in a triagle shape. The maximum delta_lm
        is M, at which point the traditional ANSI indexing is recovered.
        Gives a single mode at maximum m, and multiple modes at maximum l,
        from m=0 to m=l.
        Total number of modes = (M-(delta_lm//2)+1)*((delta_lm//2)+1)
        ``'house'``: Fills in the pyramid row by row, with a maximum horizontal
        width of M and a maximum radial resolution of delta_lm. For
        delta_lm = M, it is equivalent to ANSI, while for delta_lm > M
        it takes on a "house" like shape. Gives multiple modes at maximum
        m and maximum l.
        ``'chevron'``: Beginning from the initial chevron of width M, increasing
        delta_lm adds additional chevrons of the same width. Similar to
        "house" but with fewer modes with high l but low m.
        Total number of modes = (M+1)*(2*(delta//2)+1) 
        (Default value = 'fringe')

    Returns
    -------
    zern_idx : ndarray of int, shape(Nmodes,3)
        array of mode numbers [l,m,n]

    """
    default_deltas = {'fringe': 2*M,
                      'ansi': M,
                      'chevron': M,
                      'house': 2*M}

    delta_lm = delta_lm if delta_lm >= 0 else default_deltas[indexing]

    if indexing == 'fringe':
        pol_posm = [[(m+d//2, m-d//2) for m in range(0, M+1) if m-d//2 >= 0]
                    for d in range(0, delta_lm+1, 2)]

    elif indexing == 'ansi':
        pol_posm = [[(m+d, m) for m in range(0, M+1) if m+d < M+1]
                    for d in range(0, delta_lm+1, 2)]

    elif indexing == 'chevron':
        pol_posm = [(m+d, m) for m in range(0, M+1)
                    for d in range(0, delta_lm+1, 2)]

    elif indexing == 'house':
        pol_posm = [[(l, m) for m in range(0, M+1) if l >= m and (l-m) % 2 == 0]
                    for l in range(0, delta_lm+1)] + [(m, m) for m in range(M+1)]
        pol_posm = list(dict.fromkeys(flatten_list(pol_posm)))

    pol = [[(l, m), (l, -m)] if m != 0 else [(l, m)]
           for l, m in flatten_list(pol_posm)]
    pol = np.array(flatten_list(pol))
    num_pol = len(pol)

    pol = np.tile(pol, (2*N+1, 1))
    tor = np.atleast_2d(
        np.tile(np.arange(-N, N+1), (num_pol, 1)).flatten(order='f')).T
    zern_idx = np.hstack([pol, tor])

    sort_idx = np.lexsort((zern_idx[:, 1], zern_idx[:, 0], zern_idx[:, 2]))
    return zern_idx[sort_idx]


def get_double_four_basis_idx_dense(M, N):
    """Gets mode numbers for a dense spectral representation in double fourier basis.

    Parameters
    ----------
    M : int
        maximum poloidal resolution
    N : int
        maximum toroidal resolution

    Returns
    -------
    lambda_idx : ndarray of int, shape(Nmodes,2)
        poloidal and toroidal mode numbers [m,n]

    """

    dimFourM = 2*M+1
    dimFourN = 2*N+1
    return np.array([[m-M, n-N] for m in range(dimFourM) for n in range(dimFourN)])





def eval_four_zern(c, idx, NFP, rho, theta, zeta, dr=0, dv=0, dz=0):
    """Evaluates Fourier-Zernike basis function at a point

    Parameters
    ----------
    c : ndarray, shape(N_coeffs,)
        spectral cofficients
    idx : ndarray, shape(N_coeffs,3)
        indices for spectral basis,
        ie an array of [l,m,n] for each spectral coefficient
    NFP : int
        number of field periods
    rho : float, array-like
        radial coordinates to evaluate
    theta : float, array-like
        poloidal coordinates to evaluate
    zeta : float, array-like
        toroidal coordinates to evaluate
    dr,dv,dz : int
        order of derivatives to take in rho,theta,zeta. (Default value = 0)

    Returns
    -------
    f : ndarray
        function evaluated at specified points

    """
    idx = jnp.atleast_2d(idx)
    rho = jnp.atleast_1d(rho)
    theta = jnp.atleast_1d(theta)
    zeta = jnp.atleast_1d(zeta)
    Z = fourzern(rho, theta, zeta, idx[:, 0],
                 idx[:, 1], idx[:, 2], NFP, dr, dv, dz)
    Z = jnp.atleast_2d(Z)
    f = jnp.matmul(Z, c)
    return f


@conditional_decorator(functools.partial(jit), use_jax)
def eval_double_fourier(c, idx, NFP, theta, phi):
    """Evaluates double fourier series F = sum(F_mn(theta,phi))

    Where F_mn(theta,phi) = c_mn*cos(m*theta)*sin(n*phi)

    Parameters
    ----------
    c : ndarray, shape(N_coeffs,)
        spectral coefficients for double fourier series
    idx : ndarray of int, shape(N_coeffs,2)
        mode numbers for spectral basis.
        idx[i,0] = m, idx[i,1] = n
    NFP : int
        number of field periods
    theta : ndarray, shape(n,)
        theta values where to evaluate
    phi : ndarray, shape(n,)
        phi values where to evaluate

    Returns
    -------
    F : ndarray, shape(n,)
        F(theta,phi) evaluated at specified points

    """

    c = c.flatten()
    interp = double_fourier_basis(theta, phi, idx[:, 0], idx[:, 1], NFP)
    f = jnp.matmul(interp, c)
    return f


def axis_posn(cR, cZ, zern_idx, NFP, zeta=0.0):
    """Finds position of the magnetic axis (R0,Z0)

    Parameters
    ----------
    cR : ndarray, shape(N_coeffs,)
        spectral coefficients of R
    cZ : ndarray, shape(N_coeffs,)
        spectral coefficients of Z
    zern_idx : ndarray, shape(N_coeffs,3)
        array of (l,m,n) indices for each spectral R,Z coeff
    NFP : int
        number of field periods
    zeta : ndarray
        planes to evaluate magnetic axis at (Default value = 0.0)

    Returns
    -------
    R0 : ndarray
        R coordinate of the magnetic axis in the zeta planes specified
    Z0 : ndarray
        Z coordinate of the magnetic axis in the zeta planes specified

    """
    R0 = eval_four_zern(cR, zern_idx, NFP, 0., 0., zeta, dr=0, dv=0, dz=0)
    Z0 = eval_four_zern(cZ, zern_idx, NFP, 0., 0., zeta, dr=0, dv=0, dz=0)

    return R0, Z0



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
