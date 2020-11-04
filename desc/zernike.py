import numpy as np
import functools
import warnings
import numba
from desc.backend import jnp, conditional_decorator, jit, use_jax, fori_loop, factorial
from desc.backend import issorted, isalmostequal, sign, TextColors
from desc.backend import polyder_vec, polyval_vec

# use numba here because the array size depends on the input values, which
# jax can't handle
@numba.jit(forceobj=True)
def get_jacobi_coeffs(l, m):
    """Computes coefficients for Jacobi polynomials used in Zernike basis

    Args:
        l (ndarray of int, shape(N,)): radial mode numbers
        m (ndarray of int, shape(N,)): azimuthal mode numbers

    Returns:
        jacobi_coeffs (ndarray, shape(N,max(l)+1)): matrix of polynomial coefficients in
            order of descending powers. Each row contains the coeffs for a given l,m.
    """
    factorial = np.math.factorial
    l = np.atleast_1d(l).astype(int)
    m = np.atleast_1d(np.abs(m)).astype(int)
    lmax = np.max(l)
    npoly = len(l)
    coeffs = np.zeros((npoly, lmax+1))
    lm_even = ((l-m) % 2 == 0)[:, np.newaxis]
    for ii in range(npoly):
        ll = l[ii]
        mm = m[ii]
        for s in range(mm, ll+1, 2):
            coeffs[ii, s] = (-1)**((ll-s)/2)*factorial((ll+s)/2)/(
                factorial((ll-s)/2)*factorial((s+mm)/2)*factorial((s-mm)/2))
    coeffs = np.where(lm_even, coeffs, 0)
    return np.fliplr(coeffs)


@conditional_decorator(functools.partial(jit, static_argnums=(1, 2)), use_jax)
def zern_radial(rho, l, m, dr):
    """Zernike radial basis function

    Args:
        x (ndarray with shape(N,)): points to evaluate basis 
        l (ndarray of int): radial mode number
        m (ndarray of int): azimuthal mode number
        dr (int): order of derivative

    Returns:
        y (ndarray with shape(N,)): basis function evaluated at specified points
    """
    coeffs = polyder_vec(get_jacobi_coeffs(l, m), dr)
    y = polyval_vec(coeffs, rho).T
    return y


@conditional_decorator(functools.partial(jit), use_jax)
def zern_azimuthal(theta, m, dtheta=0):
    """Zernike azimuthal basis function

    Args:
        theta (ndarray with shape(N,)): points to evaluate basis
        m (ndarray of int, shape(M,)): azimuthal mode number
        dtheta (int): order of derivative

    Returns:
        y (ndarray with shape(N,M)): basis functions evaluated at specified points
    """
    m = jnp.atleast_1d(m)[jnp.newaxis]
    theta = jnp.atleast_1d(theta)[:, jnp.newaxis]
    m_pos = (m >= 0)
    m_neg = (m < 0)
    m = jnp.abs(m)
    der = (1j*m)**dtheta
    exp = der*jnp.exp(1j*m*theta)
    y = jnp.real(m_pos*exp) + jnp.imag(m_neg*exp)
    return y


@conditional_decorator(functools.partial(jit), use_jax)
def four_toroidal(zeta, n, NFP, dzeta=0):
    """Toroidal Fourier basis function

    Args:
        zeta (ndarray with shape(N,)): coordinates to evaluate basis
        n (ndarray of int, shape(M,)): toroidal mode number
        NFP (int): number of field periods
        dzeta (int): order of derivative

    Returns:
        y (ndarray with shape(N,M)): basis function evaluated at specified points
    """
    zeta = jnp.atleast_1d(zeta)[:, jnp.newaxis]
    n = jnp.atleast_1d(n)[jnp.newaxis]
    n_pos = n >= 0
    n_neg = n < 0
    n = jnp.abs(n)
    der = (1j*n*NFP)**dzeta
    exp = der*jnp.exp(1j*n*NFP*zeta)
    y = n_pos*jnp.real(exp) + n_neg*jnp.imag(exp)
    return y


def zern(rho, theta, l, m, dr, dtheta):
    """Zernike 2D basis function

    Args:
        rho (ndarray, shape(N,)): radial coordinates to evaluate basis
        theta (ndarray, shape(N,)): azimuthal coordinates to evaluate basis
        l (ndarray of int, shape(K,)): radial mode number
        m (ndarray of int, shape(K,)): azimuthal mode number
        dr (int): order of radial derivative
        dtheta (int): order of azimuthal derivative

    Returns:
        y (ndarray with shape(N,K)): basis function evaluated at specified points
    """

    radial = zern_radial(rho, tuple(l,), tuple(m,), dr)
    azimuthal = zern_azimuthal(theta, m, dtheta)
    return radial*azimuthal


def fourzern(rho, theta, zeta, l, m, n, NFP, dr, dv, dz):
    """Combined 3D Fourier-Zernike basis function

    Args:
        rho (ndarray with shape(N,)): radial coordinates
        theta (ndarray with shape(N,)): poloidal coordinates
        zeta (ndarray with shape(N,)): toroidal coordinates
        l (ndarray of int, shape(K,)): radial mode number
        m (ndarray of int, shape(K,)): poloidal mode number
        n (ndarray of int, shape(K,)): toroidal mode number
        NFP (int): number of field periods
        dr (int): order of radial derivative
        dv (int): order of poloidal derivative
        dz (int): order of toroidal derivative

    Returns:
        y (ndarray with shape(N,K)): basis function evaluated at specified points
    """
    return zern(rho, theta, l, m, dr, dv)*four_toroidal(zeta, n, NFP, dz)


@conditional_decorator(functools.partial(jit), use_jax)
def double_fourier_basis(theta, phi, m, n, NFP):
    """Double Fourier series for boundary/lambda

    Args:
        theta (ndarray with shape(N,)): poloidal angle to evaluate basis
        phi (ndarray with shape(N,)): toroidal angle to evaluate basis
        m (ndarray of int, shape(M,)): poloidal mode number
        n (ndarray of int, shape(M,)): toroidal mode number
        NFP (int): number of field periods

    Returns:
        y (ndarray with shape(N,M)): basis function evaluated at specified points
    """
    theta = jnp.atleast_1d(theta)
    phi = jnp.atleast_1d(phi)
    m = jnp.atleast_1d(m)
    n = jnp.atleast_1d(n)
    theta = theta[:, jnp.newaxis]
    phi = phi[:, jnp.newaxis]
    m = m[jnp.newaxis]
    n = n[jnp.newaxis]
    m_pos = m >= 0
    m_neg = m < 0
    n_pos = n >= 0
    n_neg = n < 0
    m = jnp.abs(m)
    n = jnp.abs(n)
    m_term = m_pos*jnp.cos(m*theta) + m_neg*jnp.sin(m*theta)
    n_term = n_pos*jnp.cos(n*NFP*phi) + n_neg*jnp.sin(n*NFP*phi)

    return m_term*n_term


class ZernikeTransform():
    """Zernike Transform (really a Fourier-Zernike, but whatever)

    Args:
        nodes (ndarray, shape(3,N)): nodes where basis functions are evaluated. 
            First index is (rho,theta,phi), 2nd index is node number
        zern_idx (ndarray of int, shape(Nc,3)): mode numbers for spectral basis. each row is one basis function with modes (l,m,n)
        NFP (int): number of field periods   
        derivatives (array-like, shape(n,3)): orders of derivatives to compute in rho,theta,zeta.
            Each row of the array should contain 3 elements corresponding to derivatives in rho,theta,zeta
        volumes (ndarray, shape(3,N)): volume elements at each node, dr,dv,dz
        method (str): one of 'direct', or 'fft'. 'direct' uses full matrices and can handle arbitrary
            node patterns and spectral basis. 'fft' uses fast fourier transforms in the zeta direction, 
            and so must have equally spaced toroidal nodes, and the same node pattern on each zeta plane
        pinv_rcond (float): relative cutoff for singular values in least squares fit  

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
        """helper function to check that inputs are formatted correctly for fft method"""
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

        Args:
            new_nodes (ndarray, size(3,N)): new node locations. each column is the location of one node (rho,theta,zeta)
            new_volumes (ndarray, size(3,N)): volume elements around each new node (dr,dtheta,dzeta)
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

        Args:
            zern_idx_new (ndarray of int, shape(Nc,3)): new mode numbers for spectral basis. 
                each row is one basis function with modes (l,m,n)        
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

        Args:
            new_derivatives (array-like, shape(n,3)): orders of derivatives 
                to compute in rho,theta,zeta. Each row of the array should 
                contain 3 elements corresponding to derivatives in rho,theta,zeta
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

        Args:
            c (ndarray, shape(N_coeffs,)): spectral coefficients, indexed as (lm,n) flattened in row major order
            dr (int): order of radial derivative
            dv (int): order of poloidal derivative
            dz (int): order of toroidal derivative

        Returns:
            x (ndarray, shape(N_nodes,)): array of values of function at node locations
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
        """helper function to do ffts"""
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
        """helper function for matrix multiplication that we can jit more easily"""
        return jnp.matmul(A, x)

    @conditional_decorator(functools.partial(jit, static_argnums=(0,)), use_jax)
    def fit(self, x):
        """Transform from physical domain to spectral using least squares fit

        Args:
            x (ndarray, shape(N_nodes,)): values in real space at coordinates specified by self.nodes

        Returns:
            c (ndarray, shape(N_coeffs,)): spectral coefficients in Fourier-Zernike basis
        """
        return jnp.matmul(self.pinv, x)


def get_zern_basis_idx_dense(M, N, indexing='fringe'):
    """Gets mode numbers for dense spectral representation in zernike basis.

    Args: 
        M (int): maximum poloidal resolution
        N (int): maximum toroidal resolution
        indexing (str): one of 'fringe' or 'ansi'. Fringe indexing has the 
            maximum radial resolution = 2M, while ANIS indexing has max radial 
            resolution = M. Fringe is a good up to M~16, while ANSI can avoid 
            numerical issues up to M~30

    Returns: 
        zern_idx (ndarray of int, shape(Nmodes,3)): array of mode numbers [l,m,n]
    """
    if indexing == 'fringe':
        op = fringe_to_lm
        num_lm_modes = (M+1)**2
        if M >= 16:
            warnings.warn(TextColors.WARNING +
                          "Fringe indexing is not recommended for M>=16 due to numerical roundoff at high radial resolution" + TextColors.ENDC)
    elif indexing == 'ansi':
        op = ansi_to_lm
        num_lm_modes = int(M*(M+1)/2)
        if M >= 30:
            warnings.warn(TextColors.WARNING +
                          "ANSI indexing is not recommended for M>=30 due to numerical roundoff at high radial resolution" + TextColors.ENDC)

    num_four = 2*N+1
    zern_idx = np.array([(*op(i), n-N)
                         for i in range(num_lm_modes) for n in range(num_four)])
    sort_idx = np.lexsort((zern_idx[:, 1], zern_idx[:, 0], zern_idx[:, 2]))
    return zern_idx[sort_idx]


def get_double_four_basis_idx_dense(M, N):
    """Gets mode numbers for a dense spectral representation in double fourier basis.

    Args:
        M (int): maximum poloidal resolution
        N (int): maximum toroidal resolution

    Returns:
       lambda_idx (ndarray of int, shape(Nmodes,2)): poloidal and toroidal mode numbers [m,n]
    """

    dimFourM = 2*M+1
    dimFourN = 2*N+1
    return np.array([[m-M, n-N] for m in range(dimFourM) for n in range(dimFourN)])


def zernike_norm(l, m):
    """Norm of a Zernike polynomial with l, m indexing.
    Returns the integral (Z^m_l)^2 r dr dt, r=[0,1], t=[0,2*pi]

    Args:
        l,m (int): radial and azimuthal mode numbers.

    Returns:
        norm (float): norm of Zernike polynomial over unit disk.
    """
    return jnp.sqrt((2 * (l + 1)) / (jnp.pi*(1 + jnp.kronecker(m, 0))))


def lm_to_fringe(l, m):
    """Convert Zernike (l,m) double index to single Fringe index.

    Args:
        l,m (int): radial and azimuthal mode numbers.

    Returns:
        idx (int): Fringe index for l,m
    """
    M = (l + np.abs(m)) / 2
    return int(M**2 + M + m)


def fringe_to_lm(idx):
    """Convert single Zernike Fringe index to (l,m) double index.

    Args:
        idx (int): Fringe index

    Returns: 
        l,m (int): radial and azimuthal mode numbers.
    """
    M = (np.ceil(np.sqrt(idx+1)) - 1)
    m = idx - M**2 - M
    l = 2*M - np.abs(m)
    return int(l), int(m)


def lm_to_ansi(l, m):
    """Convert Zernike (l,m) two term index to ANSI single term index.

    Args:
        l,m (int): radial and azimuthal mode numbers.

    Returns:
        idx (int): ANSI index for l,m
    """
    return int((l * (l + 2) + m) / 2)


def ansi_to_lm(idx):
    """Convert Zernike ANSI single term to (l,m) two-term index.

    Args:
        idx (int): ANSI index

    Returns:
        l,m (int): radial and azimuthal mode numbers.

    """
    l = int(np.ceil((-3 + np.sqrt(9 + 8*idx))/2))
    m = 2 * idx - l * (l + 2)
    return l, m


def eval_four_zern(c, idx, NFP, rho, theta, zeta, dr=0, dv=0, dz=0):
    """Evaluates Fourier-Zernike basis function at a point

    Args:
        c (ndarray, shape(Nc,)): spectral cofficients
        idx (ndarray, shape(Nc,3)): indices for spectral basis, 
            ie an array of [l,m,n] for each spectral coefficient
        NFP (int): number of field periods
        rho (float,array-like): radial coordinates to evaluate
        theta (float,array-like): poloidal coordinates to evaluate
        zeta (float,array-like): toroidal coordinates to evaluate
        dr,dv,dz (int): order of derivatives to take in rho,theta,zeta

    Returns:
        f (ndarray): function evaluated at specified points
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

    Args:
        c (ndarray, shape(Nc,): spectral coefficients for double fourier series
        idx (ndarray of int, shape(Nc,2)): mode numbers for spectral basis. 
            idx[i,0] = m, idx[i,1] = n
        NFP (int): number of field periods
        theta (ndarray, shape(n,)): theta values where to evaluate
        phi (ndarray, shape(n,)): phi values where to evaluate

    Returns:
        F (ndarray, size(n,)): F(theta,phi) evaluated at specified points
    """

    c = c.flatten()
    interp = double_fourier_basis(theta, phi, idx[:, 0], idx[:, 1], NFP)
    f = jnp.matmul(interp, c)
    return f


def axis_posn(cR, cZ, zern_idx, NFP, zeta=0.0):
    """Finds position of the magnetic axis (R0,Z0)

    Args:
        cR (ndarray, shape(N_coeffs,)): spectral coefficients of R
        cZ (ndarray, shape(N_coeffs,)): spectral coefficients of Z
        zern_idx (ndarray, shape(N_coeffs,3)): array of (l,m,n) indices for each spectral R,Z coeff
        NFP (int): number of field periods
        zeta (ndarray): planes to evaluate magnetic axis at

    Returns:
        R0 (float): R coordinate of the magnetic axis in the zeta planes specified
        Z0 (float): Z coordinate of the magnetic axis in the zeta planes specified
    """
    R0 = eval_four_zern(cR, zern_idx, NFP, 0., 0., zeta, dr=0, dv=0, dz=0)
    Z0 = eval_four_zern(cZ, zern_idx, NFP, 0., 0., zeta, dr=0, dv=0, dz=0)

    return R0, Z0


def symmetric_x(zern_idx, lambda_idx):
    """Compute stellarator symmetry linear constraint matrix

    Args:
        zern_idx (ndarray, shape(N_coeffs,3)): array of (l,m,n) indices for each spectral R,Z coeff
        lambda_idx (ndarray, shape(N_coeffs,2)): array of (m,n) indices for each spectral lambda coeff

    Returns:
        A (2D array): matrix such that x=A*y and y=A^T*x
                      where y are the stellarator symmetric components of x
    """

    m_zern = zern_idx[:, 1]
    n_zern = zern_idx[:, 2]
    m_lambda = lambda_idx[:, 0]
    n_lambda = lambda_idx[:, 1]

    # symmetric indices of R, Z, lambda
    sym_R = sign(m_zern)*sign(n_zern) > 0
    sym_Z = sign(m_zern)*sign(n_zern) < 0
    sym_L = sign(m_lambda)*sign(n_lambda) < 0

    sym_x = np.concatenate([sym_R, sym_Z, sym_L])
    A = np.diag(sym_x, k=0).astype(int)
    return A[:, sym_x]
