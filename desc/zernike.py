import numpy as np
import functools
from backend import jnp, conditional_decorator, jit, use_jax, fori_loop, factorial
import warnings


@conditional_decorator(functools.partial(jit), use_jax)
def zern_radial(x, l, m):
    """Zernike radial basis function

    Args:
        x (ndarray with shape(N,)): points to evaluate basis 
        l (int): radial mode number
        m (int): azimuthal mode number

    Returns:
        y (ndarray with shape(N,)): basis function evaluated at specified points
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.zeros_like(x)
    lm_even = (l-m) % 2 == 0
    m = jnp.abs(m)

    def body_fun(k, y):
        coeff = (-1)**k * factorial(l-k)/(factorial(k) *
                                          factorial((l+m)/2-k) * factorial((l-m)/2-k))
        return y + coeff*x**(l-2*k)
    kmax = ((l-m)/2.0)+1.0
    y = fori_loop(0.0, kmax, body_fun, y)

    return y*lm_even


@conditional_decorator(functools.partial(jit), use_jax)
def zern_radial_r(x, l, m):
    """Zernike radial basis function, first derivative in r

    Args:
        x (ndarray with shape(N,)): points to evaluate basis 
        l (int): radial mode number
        m (int): azimuthal mode number

    Returns:
        y (ndarray with shape(N,)): basis function evaluated at specified points
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.zeros_like(x)
    lm_even = (l-m) % 2 == 0
    m = jnp.abs(m)

    def body_fun(k, y):
        coeff = (l-2*k)*(-1)**k * factorial(l-k)/(factorial(k)
                                                  * factorial((l+m)/2-k) * factorial((l-m)/2-k))
        return y + coeff*x**jnp.maximum(l-2*k-1, 0)
    kmax = ((l-m)/2.0)+1.0
    y = fori_loop(0.0, kmax, body_fun, y)

    return y*lm_even


@conditional_decorator(functools.partial(jit), use_jax)
def zern_radial_rr(x, l, m):
    """Zernike radial basis function, second radial derivative

    Args:
        x (ndarray with shape(N,)): points to evaluate basis 
        l (int): radial mode number
        m (int): azimuthal mode numbe

    Returns:
        y (ndarray with shape(N,)): basis function evaluated at specified points
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.zeros_like(x)
    lm_even = (l-m) % 2 == 0
    m = jnp.abs(m)

    def body_fun(k, y):
        coeff = (l-2*k-1)*(l-2*k)*(-1)**k * factorial(l-k) / \
            (factorial(k) * factorial((l+m)/2-k) * factorial((l-m)/2-k))
        return y + coeff*x**jnp.maximum(l-2*k-2, 0)
    kmax = ((l-m)/2.0)+1.0
    y = fori_loop(0.0, kmax, body_fun, y)

    return y*lm_even


@conditional_decorator(functools.partial(jit), use_jax)
def zern_radial_rrr(x, l, m):
    """Zernike radial basis function, third radial derivative

    Args:
        x (ndarray with shape(N,)): points to evaluate basis 
        l (int): radial mode number
        m (int): azimuthal mode numbe

    Returns:
        y (ndarray with shape(N,)): basis function evaluated at specified points
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.zeros_like(x)
    lm_even = (l-m) % 2 == 0
    m = jnp.abs(m)

    def body_fun(k, y):
        coeff = (l-2*k-2)*(l-2*k-1)*(l-2*k)*(-1)**k * factorial(l-k) / \
            (factorial(k) * factorial((l+m)/2-k) * factorial((l-m)/2-k))
        return y + coeff*x**jnp.maximum(l-2*k-3, 0)
    kmax = ((l-m)/2.0)+1.0
    y = fori_loop(0.0, kmax, body_fun, y)

    return y*lm_even


@conditional_decorator(functools.partial(jit), use_jax)
def zern_azimuthal(theta, m):
    """Zernike azimuthal basis function

    Args:
        theta (ndarray with shape(N,)): points to evaluate basis
        m (ndarray of int, shape(M,)): azimuthal mode number

    Returns:
        y (ndarray with shape(N,M)): basis functions evaluated at specified points
    """
    m = jnp.atleast_1d(m)
    theta = jnp.atleast_1d(theta)
    theta = theta[:, jnp.newaxis]
    m = m[jnp.newaxis]
    m_pos = m >= 0
    m_neg = m < 0
    m = jnp.abs(m)
    y = m_pos*jnp.cos(m*theta) + m_neg*jnp.sin(m*theta)

    return y


@conditional_decorator(functools.partial(jit), use_jax)
def zern_azimuthal_v(theta, m):
    """Zernike azimuthal basis function, first azimuthal derivative

    Args:
        theta (ndarray with shape(N,)): points to evaluate basis
        m (ndarray of int, shape(M,)): azimuthal mode number

    Returns:
        y (ndarray with shape(N,M)): basis functions evaluated at specified points
    """
    m = jnp.atleast_1d(m)
    theta = jnp.atleast_1d(theta)
    theta = theta[:, jnp.newaxis]
    m = m[jnp.newaxis]
    m_pos = m >= 0
    m_neg = m < 0
    m = jnp.abs(m)
    y = m_pos*(-m*jnp.sin(m*theta)) + m_neg*(m*jnp.cos(m*theta))

    return y


@conditional_decorator(functools.partial(jit), use_jax)
def zern_azimuthal_vv(theta, m):
    """Zernike azimuthal basis function, second azimuthal derivative

    Args:
        theta (ndarray with shape(N,)): points to evaluate basis
        m (ndarray of int, shape(M,)): azimuthal mode number

    Returns:
        y (ndarray with shape(N,M)): basis function evaluated at specified points
    """
    m = jnp.atleast_1d(m)
    theta = jnp.atleast_1d(theta)
    theta = theta[:, jnp.newaxis]
    m = m[jnp.newaxis]
    m_pos = m >= 0
    m_neg = m < 0
    m = jnp.abs(m)
    y = m_pos*(-m**2*jnp.cos(m*theta)) + m_neg*(-m**2*jnp.sin(m*theta))

    return y


@conditional_decorator(functools.partial(jit), use_jax)
def zern_azimuthal_vvv(theta, m):
    """Zernike azimuthal basis function, third azimuthal derivative

    Args:
        theta (ndarray with shape(N,)): points to evaluate basis
        m (ndarray of int, shape(M,)): azimuthal mode number

    Returns:
        y (ndarray with shape(N,M)): basis function evaluated at specified points
    """
    m = jnp.atleast_1d(m)
    theta = jnp.atleast_1d(theta)
    theta = theta[:, jnp.newaxis]
    m = m[jnp.newaxis]
    m_pos = m >= 0
    m_neg = m < 0
    m = jnp.abs(m)
    y = m_pos*(m**3*jnp.sin(m*theta)) + m_neg*(-m**3*jnp.cos(m*theta))

    return y


@conditional_decorator(functools.partial(jit), use_jax)
def four_toroidal(zeta, n, NFP):
    """Toroidal Fourier basis function

    Args:
        zeta (ndarray with shape(N,)): coordinates to evaluate basis
        n (ndarray of int, shape(M,)): toroidal mode number
        NFP (int): number of field periods

    Returns:
        y (ndarray with shape(N,M)): basis function evaluated at specified points
    """
    zeta = jnp.atleast_1d(zeta)
    n = jnp.atleast_1d(n)
    n = n[jnp.newaxis]
    zeta = zeta[:, jnp.newaxis]
    n_pos = n >= 0
    n_neg = n < 0
    n = jnp.abs(n)
    y = n_pos*(jnp.cos(n*NFP*zeta)) + n_neg*(jnp.sin(n*NFP*zeta))

    return y


@conditional_decorator(functools.partial(jit), use_jax)
def four_toroidal_z(zeta, n, NFP):
    """Toroidal Fourier basis function, first toroidal derivative

    Args:
        zeta (ndarray with shape(N,)): coordinates to evaluate basis
        n (ndarray of int, shape(M,)): toroidal mode number
        NFP (int): number of field periods

    Returns:
        y (ndarray with shape(N,M)): basis function evaluated at specified points
    """
    zeta = jnp.atleast_1d(zeta)
    n = jnp.atleast_1d(n)
    n = n[jnp.newaxis]
    zeta = zeta[:, jnp.newaxis]
    n_pos = n >= 0
    n_neg = n < 0
    n = jnp.abs(n)
    y = n_pos*(-n*NFP*jnp.sin(n*NFP*zeta)) + n_neg*(n*NFP*jnp.cos(n*NFP*zeta))

    return y


@conditional_decorator(functools.partial(jit), use_jax)
def four_toroidal_zz(zeta, n, NFP):
    """Toroidal Fourier basis function, second toroidal derivative

    Args:
        zeta (ndarray with shape(N,)): coordinates to evaluate basis
        n (ndarray of int, shape(M,)): toroidal mode number
        NFP (int): number of field periods

    Returns:
        y (ndarray with shape(N,M)): basis function evaluated at specified points
    """
    zeta = jnp.atleast_1d(zeta)
    n = jnp.atleast_1d(n)
    n = n[jnp.newaxis]
    zeta = zeta[:, jnp.newaxis]
    n_pos = n >= 0
    n_neg = n < 0
    n = jnp.abs(n)
    y = n_pos*(-(n*NFP)**2*jnp.cos(n*NFP*zeta)) + \
        n_neg*(-(n*NFP)**2*jnp.sin(n*NFP*zeta))

    return y


@conditional_decorator(functools.partial(jit), use_jax)
def four_toroidal_zzz(zeta, n, NFP):
    """Toroidal Fourier basis function, third toroidal derivative

    Args:
        zeta (ndarray with shape(N,)): coordinates to evaluate basis
        n (ndarray of int, shape(M,)): toroidal mode number
        NFP (int): number of field periods

    Returns:
        y (ndarray with shape(N,M)): basis function evaluated at specified points
    """
    zeta = jnp.atleast_1d(zeta)
    n = jnp.atleast_1d(n)
    n = n[jnp.newaxis]
    zeta = zeta[:, jnp.newaxis]
    n_pos = n >= 0
    n_neg = n < 0
    n = jnp.abs(n)
    y = n_pos*((n*NFP)**3*jnp.sin(n*NFP*zeta)) + \
        n_neg*(-(n*NFP)**3*jnp.cos(n*NFP*zeta))

    return y


radial_derivatives = {0: zern_radial,
                      1: zern_radial_r,
                      2: zern_radial_rr,
                      3: zern_radial_rrr}
poloidal_derivatives = {0: zern_azimuthal,
                        1: zern_azimuthal_v,
                        2: zern_azimuthal_vv,
                        3: zern_azimuthal_vvv}
toroidal_derivatives = {0: four_toroidal,
                        1: four_toroidal_z,
                        2: four_toroidal_zz,
                        3: four_toroidal_zzz}


def zern(r, theta, l, m, dr, dtheta):
    """Zernike 2D basis function

    Args:
        r (ndarray with shape(N,)): radial coordinates to evaluate basis
        theta (array-like): azimuthal coordinates to evaluate basis
        l (int): radial mode number
        m (int): azimuthal mode number
        dr (int): order of radial derivative
        dtheta (int): order of azimuthal derivative

    Returns:
        y (ndarray with shape(N,)): basis function evaluated at specified points
    """
    radial = radial_derivatives[dr](r, l, m)[:, jnp.newaxis]
    azimuthal = poloidal_derivatives[dtheta](theta, m)

    return radial*azimuthal


def four(zeta, n, NFP, dz):
    """Toroidal Fourier basis function

    Args:
        zeta (ndarray with shape(N,)): coordinates to evaluate basis
        n (ndarray of int, shape(M,)): toroidal mode number
        NFP (int): number of field periods
        dz (int): order of toroidal derivative

    Returns:
        y (ndarray with shape(N,)): basis function evaluated at specified points
    """
    return toroidal_derivatives[dz](zeta, n, NFP)


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


def fourzern(r, theta, zeta, l, m, n, NFP, dr, dv, dz):
    """Combined 3D Fourier-Zernike basis function

    Args:
        r (ndarray with shape(N,)): radial coordinates
        theta (ndarray with shape(N,)): poloidal coordinates
        zeta (ndarray with shape(N,)): toroidal coordinates
        l (int): radial mode number
        m (int): poloidal mode number
        n (int): toroidal mode number
        NFP (int): number of field periods
        dr (int): order of radial derivative
        dv (int): order of poloidal derivative
        dz (int): order of toroidal derivative

    Returns:
        y (ndarray with shape(N,)): basis function evaluated at specified points
    """
    return zern(r, theta, l, m, dr, dv)*four(zeta, n, NFP, dz)


class ZernikeTransform():
    """Zernike Transform (really a Fourier-Zernike, but whatever)

    Args:
        nodes (ndarray, shape(3,N)): nodes where basis functions are evaluated. 
            First index is (rho,theta,phi), 2nd index is node number
        mode_idx (ndarray of int, shape(Nc,3)): mode numbers for spectral basis. each row is one basis function with modes (l,m,n)
        NFP (int): number of field periods   
        derivatives (array-like, shape(n,3)): orders of derivatives to compute in rho,theta,zeta.
            Each row of the array should contain 3 elements corresponding to derivatives in rho,theta,zeta
    """

    def __init__(self, nodes, mode_idx, NFP, derivatives=[0, 0, 0]):
        # array of which l,m,n is at which column of the interpolation matrix
        self.mode_idx = mode_idx
        # array of which r,v,z is at which row of the interpolation matrix
        self.nodes = nodes
        self.NFP = NFP
        self.derivatives = np.atleast_2d(derivatives)
        self.matrices = {i: {j: {k: {}
                                 for k in range(4)} for j in range(4)} for i in range(4)}
        self._build(self.derivatives, self.mode_idx)

    def _build(self, derivs, mode_idx):
        for d in derivs:
            dr = d[0]
            dv = d[1]
            dz = d[2]
            self.matrices[dr][dv][dz] = jnp.hstack([fourzern(self.nodes[0], self.nodes[1], self.nodes[2],
                                                             lmn[0], lmn[1], lmn[2], self.NFP, dr, dv, dz) for lmn in mode_idx])

    def expand_nodes(self, new_nodes):
        """Change the real space resolution by adding new nodes without full recompute

        Only computes basis at spatial nodes that aren't already in the basis

        Args:
            new_nodes (ndarray, side(3,N)): new node locations. each column is the location of one node (rho,theta,zeta)
        """

        new_nodes = jnp.atleast_2d(new_nodes).T
        # first remove nodes that are no longer needed
        old_in_new = (self.nodes.T[:, None] == new_nodes).all(-1).any(-1)
        for d in self.derivatives:
            self.matrices[d[0]][d[1]][d[2]
                                      ] = self.matrices[d[0]][d[1]][d[2]][old_in_new]
        self.nodes = self.nodes[:, old_in_new]

        # then add new nodes
        new_not_in_old = ~(new_nodes[:, None] == self.nodes.T).all(-1).any(-1)
        nodes_to_add = new_nodes[new_not_in_old]
        if len(nodes_to_add) > 0:
            for d in self.derivatives:
                self.matrices[d[0]][d[1]][d[2]] = jnp.vstack([
                    self.matrices[d[0]][d[1]][d[2]],  # old
                    jnp.hstack([fourzern(nodes_to_add[:, 0], nodes_to_add[:, 1], nodes_to_add[:, 2],  # new
                                         lmn[0], lmn[1], lmn[2], self.NFP, d[0], d[1], d[2]) for lmn in self.mode_idx])])

        # update indices
        self.nodes = np.hstack([self.nodes, nodes_to_add.T])
        # permute indexes so they're in the same order as the input
        permute_idx = [self.nodes.T.tolist().index(i)
                       for i in new_nodes.tolist()]
        for d in self.derivatives:
            self.matrices[d[0]][d[1]][d[2]
                                      ] = self.matrices[d[0]][d[1]][d[2]][permute_idx]
        self.nodes = self.nodes[:, permute_idx]

    def expand_spectral_resolution(self, mode_idx_new):
        """Change the spectral resolution of the transform without full recompute

        Only computes modes that aren't already in the basis

        Args:
            mode_idx_new (ndarray of int, shape(Nc,3)): new mode numbers for spectral basis. 
                each row is one basis function with modes (l,m,n)        
        """

        mode_idx_new = jnp.atleast_2d(mode_idx_new)
        # first remove modes that are no longer needed
        old_in_new = (self.mode_idx[:, None] == mode_idx_new).all(-1).any(-1)
        for d in self.derivatives:
            self.matrices[d[0]][d[1]][d[2]] = self.matrices[d[0]
                                                            ][d[1]][d[2]][:, old_in_new]
        self.mode_idx = self.mode_idx[old_in_new]

        # then add new modes
        new_not_in_old = ~(mode_idx_new[:, None]
                           == self.mode_idx).all(-1).any(-1)
        modes_to_add = mode_idx_new[new_not_in_old]
        if len(modes_to_add) > 0:
            for d in self.derivatives:
                self.matrices[d[0]][d[1]][d[2]] = jnp.hstack([
                    self.matrices[d[0]][d[1]][d[2]],  # old
                    jnp.hstack([fourzern(self.nodes[0], self.nodes[1], self.nodes[2],  # new
                                         lmn[0], lmn[1], lmn[2], self.NFP, d[0], d[1], d[2]) for lmn in modes_to_add])])

        # update indices
        self.mode_idx = np.vstack([self.mode_idx, modes_to_add])
        # permute indexes so they're in the same order as the input
        permute_idx = [self.mode_idx.tolist().index(i)
                       for i in mode_idx_new.tolist()]
        for d in self.derivatives:
            self.matrices[d[0]][d[1]][d[2]] = self.matrices[d[0]
                                                            ][d[1]][d[2]][:, permute_idx]
        self.mode_idx = self.mode_idx[permute_idx]

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
        self._build(derivs_to_add, self.mode_idx)
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
        return self._matmul(self.matrices[dr][dv][dz], c)

    @conditional_decorator(functools.partial(jit, static_argnums=(0,)), use_jax)
    def _matmul(self, A, x):
        """helper function for matrix multiplication that we can jit more easily"""
        return jnp.matmul(A, x)

    # TODO: precompute SVD
    @conditional_decorator(functools.partial(jit, static_argnums=(0, 2)), use_jax)
    def fit(self, x, rcond):
        """Transform from physical domain to spectral

        Args:
            x (ndarray, shape(N_nodes,)): values in real space at coordinates specified by self.nodes
            rcond (float): relative cutoff for singular values in least squares fit  

        Returns:
            c (ndarray, shape(N_coeffs,)): spectral coefficients in Fourier-Zernike basis
        """
        return jnp.linalg.lstsq(self.matrices[0][0][0], x, rcond=rcond)[0]


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
            warnings.warn(
                "Fringe indexing is not recommended for M>=16 due to numerical roundoff at high radial resolution")
    elif indexing == 'ansi':
        op = ansi_to_lm
        num_lm_modes = int(M*(M+1)/2)

    num_four = 2*N+1
    return np.array([(*op(i), n-N) for i in range(num_lm_modes) for n in range(num_four)])


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
    rho = jnp.asarray(rho)
    theta = jnp.asarray(theta)
    zeta = jnp.asarray(zeta)
    Z = jnp.stack([fourzern(rho, theta, zeta, lmn[0], lmn[1],
                            lmn[2], NFP, dr, dv, dz) for lmn in idx])
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


def axis_posn(cR, cZ, zern_idx, NFP):
    """Finds position of the magnetic axis (R0,Z0)

    Args:
        cR (ndarray, shape(N_coeffs,)): spectral coefficients of R
        cZ (ndarray, shape(N_coeffs,)): spectral coefficients of Z
        zern_idx (ndarray, shape(N_coeffs,3)): array of (l,m,n) indices for each spectral R,Z coeff
        NFP (int): number of field periods

    Returns:
        R0 (float): R coordinate of the magnetic axis in the zeta=0 plane
        Z0 (float): Z coordinate of the magnetic axis in the zeta=0 plane
    """
    R0 = eval_four_zern(cR, zern_idx, NFP, 0., 0., 0., dr=0, dv=0, dz=0)[0]
    Z0 = eval_four_zern(cZ, zern_idx, NFP, 0., 0., 0., dr=0, dv=0, dz=0)[0]

    return R0, Z0


def symmetric_x(M, N):
    """Compute stellarator symmetry linear constraint matrix

    Args:
        M (int): maximum poloidal mode number of solution
        N (int): maximum toroidal mode number of solution

    Returns:
        A (2D array): matrix such that x=A*y and y=A^T*x
                      where y are the stellarator symmetric components of x
    """
    # TODO: make this work with either ANSI or fringe indexing
    # would be better if instead of M,N, the arg was just the zern_idx
    dimZern = (M+1)**2
    m = np.zeros(dimZern)
    for i in range(dimZern):
        li, mi = fringe_to_lm(i)
        m[i] = mi

    # symmetric indices of R, Z, lambda
    sym_R = np.concatenate([np.tile((m < 0)[np.newaxis].T, (1, N)), np.tile(
        (m >= 0)[np.newaxis].T, (1, N+1))], axis=1).flatten()
    sym_Z = np.concatenate([np.tile((m >= 0)[np.newaxis].T, (1, N)), np.tile(
        (m < 0)[np.newaxis].T, (1, N+1))], axis=1).flatten()
    sym_L = np.concatenate([np.tile(np.concatenate([np.zeros(M, dtype=bool), np.ones(M+1, dtype=bool)])[np.newaxis], (1, N)),
                            np.tile(np.concatenate([np.ones(M, dtype=bool), np.zeros(M+1, dtype=bool)])[np.newaxis], (1, N+1))], axis=1).flatten()
    sym_x = np.concatenate([sym_R, sym_Z, sym_L])

    A = np.diag(sym_x, k=0).astype(int)
    return A[:, sym_x]


def get_jacobi_coeffs(l, m):
    """Computes coefficients for Jacobi polynomials used in Zernike basis

    Args:
        l (ndarray of int, shape(N,)): radial mode numbers
        m (ndarray of int, shape(N,)): azimuthal mode numbers

    Returns:
        jacobi_coeffs (ndarray, shape(N,max(l)+1)): matrix of polynomial coefficients in
            order of descending powers. Each row contains the coeffs for a given l,m.
    """

    l = np.atleast_1d(l)
    m = np.atleast_1d(abs(m))
    lmax = np.max(l)
    npoly = len(l)
    coeffs = np.zeros((npoly, lmax+1))
    for ii, (ll, mm) in enumerate(zip(l, m)):
        if (ll-mm) % 2 != 0:
            continue
        for s in range(mm, ll+1, 2):
            coeffs[ii, s] = (-1)**((ll-s)/2)*factorial((ll+s)/2)/(
                factorial((ll-s)/2)*factorial((s+mm)/2)*factorial((s-mm)/2))
    return np.fliplr(coeffs)
