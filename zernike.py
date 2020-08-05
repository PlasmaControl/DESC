import numpy as np
import functools
from backend import jnp, conditional_decorator, jit, use_jax, fori_loop, factorial


@conditional_decorator(functools.partial(jit), use_jax)
def zern_radial(x,l,m):
    """Zernike radial basis function
    
    Args:
        x (ndarray with shape(N,)): points to evaluate basis 
        l (int): radial mode number
        m (int): azimuthal mode number
    
    Returns:
        y (ndarray with shape(N,)): basis function evaluated at specified points
    """
    x = jnp.asarray(x,dtype=jnp.float64)
    y = jnp.zeros_like(x)
    lm_even = (l-m)%2 == 0
    m = jnp.abs(m)
    
    def body_fun(k,y):
        coeff = (-1)**k* factorial(l-k)/(factorial(k)* factorial((l+m)/2-k)* factorial((l-m)/2-k))
        return y + coeff*x**(l-2*k)
    kmax = ((l-m)/2.0)+1.0
    y = fori_loop(0.0,kmax,body_fun,y)

    return y*lm_even

@conditional_decorator(functools.partial(jit), use_jax)
def zern_radial_r(x,l,m):
    """Zernike radial basis function, first derivative in r
    
    Args:
        x (ndarray with shape(N,)): points to evaluate basis 
        l (int): radial mode number
        m (int): azimuthal mode number
    
    Returns:
        y (ndarray with shape(N,)): basis function evaluated at specified points
    """
    x = jnp.asarray(x,dtype=jnp.float64)
    y = jnp.zeros_like(x)
    lm_even = (l-m)%2 == 0
    m = jnp.abs(m)
    def body_fun(k,y):
        coeff = (l-2*k)*(-1)**k* factorial(l-k)/(factorial(k)* factorial((l+m)/2-k)* factorial((l-m)/2-k))
        return y + coeff*x**jnp.maximum(l-2*k-1,0)
    kmax = ((l-m)/2.0)+1.0
    y = fori_loop(0.0,kmax,body_fun,y)

    return y*lm_even

@conditional_decorator(functools.partial(jit), use_jax)
def zern_radial_rr(x,l,m):
    """Zernike radial basis function, second radial derivative
    
    Args:
        x (ndarray with shape(N,)): points to evaluate basis 
        l (int): radial mode number
        m (int): azimuthal mode numbe
    
    Returns:
        y (ndarray with shape(N,)): basis function evaluated at specified points
    """
    x = jnp.asarray(x,dtype=jnp.float64)
    y = jnp.zeros_like(x)
    lm_even = (l-m)%2 == 0
    m = jnp.abs(m)
    def body_fun(k,y):
        coeff = (l-2*k-1)*(l-2*k)*(-1)**k* factorial(l-k)/(factorial(k)* factorial((l+m)/2-k)* factorial((l-m)/2-k))
        return y + coeff*x**jnp.maximum(l-2*k-2,0)
    kmax = ((l-m)/2.0)+1.0
    y = fori_loop(0.0,kmax,body_fun,y)

    return y*lm_even

@conditional_decorator(functools.partial(jit), use_jax)
def zern_radial_rrr(x,l,m):
    """Zernike radial basis function, third radial derivative
    
    Args:
        x (ndarray with shape(N,)): points to evaluate basis 
        l (int): radial mode number
        m (int): azimuthal mode numbe
    
    Returns:
        y (ndarray with shape(N,)): basis function evaluated at specified points
    """
    x = jnp.asarray(x,dtype=jnp.float64)
    y = jnp.zeros_like(x)
    lm_even = (l-m)%2 == 0
    m = jnp.abs(m)
    def body_fun(k,y):
        coeff = (l-2*k-2)*(l-2*k-1)*(l-2*k)*(-1)**k* factorial(l-k)/(factorial(k)* factorial((l+m)/2-k)* factorial((l-m)/2-k))
        return y + coeff*x**jnp.maximum(l-2*k-3,0)
    kmax = ((l-m)/2.0)+1.0
    y = fori_loop(0.0,kmax,body_fun,y)
    
    return y*lm_even

@conditional_decorator(functools.partial(jit), use_jax)
def zern_azimuthal(theta,l,m):
    """Zernike azimuthal basis function
    
    Args:
        theta (ndarray with shape(N,)): points to evaluate basis
        l (int): radial mode number
        m (int): azimuthal mode number
    
    Returns:
        y (ndarray with shape(N,)): basis function evaluated at specified points
    """
    m_pos = m >= 0
    m_neg = m < 0
    m = jnp.abs(m)
    y = m_pos*jnp.cos(m*theta) + m_neg*jnp.sin(m*theta)

    return y

@conditional_decorator(functools.partial(jit), use_jax)
def zern_azimuthal_v(theta,l,m):
    """Zernike azimuthal basis function, first azimuthal derivative
    
    Args:
        theta (ndarray with shape(N,)): points to evaluate basis
        l (int): radial mode number
        m (int): azimuthal mode number
    
    Returns:
        y (ndarray with shape(N,)): basis function evaluated at specified points
    """
    m_pos = m >= 0
    m_neg = m < 0
    m = jnp.abs(m)
    y = m_pos*(-m*jnp.sin(m*theta)) + m_neg*(m*jnp.cos(m*theta))

    return y
    
@conditional_decorator(functools.partial(jit), use_jax)
def zern_azimuthal_vv(theta,l,m):
    """Zernike azimuthal basis function, second azimuthal derivative
    
    Args:
        theta (ndarray with shape(N,)): points to evaluate basis
        l (int): radial mode number
        m (int): azimuthal mode number
    
    Returns:
        y (ndarray with shape(N,)): basis function evaluated at specified points
    """
    m_pos = m >= 0
    m_neg = m < 0
    m = jnp.abs(m)
    y = m_pos*(-m**2*jnp.cos(m*theta)) + m_neg*(-m**2*jnp.sin(m*theta))

    return y

@conditional_decorator(functools.partial(jit), use_jax)
def zern_azimuthal_vvv(theta,l,m):
    """Zernike azimuthal basis function, third azimuthal derivative
    
    Args:
        theta (ndarray with shape(N,)): points to evaluate basis
        l (int): radial mode number
        m (int): azimuthal mode number
    
    Returns:
        y (ndarray with shape(N,)): basis function evaluated at specified points
    """
    m_pos = m >= 0
    m_neg = m < 0
    m = jnp.abs(m)
    y = m_pos*(m**3*jnp.sin(m*theta)) + m_neg*(-m**3*jnp.cos(m*theta))

    return y

@conditional_decorator(functools.partial(jit), use_jax)
def four_toroidal(zeta,n,NFP):
    """Toroidal Fourier basis function
    
    Args:
        zeta (ndarray with shape(N,)): coordinates to evaluate basis
        n (int): toroidal mode number
        NFP (int): number of field periods
    
    Returns:
        y (ndarray with shape(N,)): basis function evaluated at specified points
    """
    n_pos = n >= 0
    n_neg = n < 0

    n = jnp.abs(n)
    y0 = n_pos*(jnp.cos(n*NFP*zeta)) + n_neg*(jnp.sin(n*NFP*zeta))
    
    return y0

@conditional_decorator(functools.partial(jit), use_jax)
def four_toroidal_z(zeta,n,NFP):
    """Toroidal Fourier basis function, first toroidal derivative
    
    Args:
        zeta (ndarray with shape(N,)): coordinates to evaluate basis
        n (int): toroidal mode number
        NFP (int): number of field periods
    
    Returns:
        y (ndarray with shape(N,)): basis function evaluated at specified points
    """
    n_pos = n >= 0
    n_neg = n < 0
    n = jnp.abs(n)
    y1 = n_pos*(-n*NFP*jnp.sin(n*NFP*zeta)) + n_neg*(n*NFP*jnp.cos(n*NFP*zeta))
    
    return y1

@conditional_decorator(functools.partial(jit), use_jax)
def four_toroidal_zz(zeta,n,NFP):
    """Toroidal Fourier basis function, second toroidal derivative
    
    Args:
        zeta (ndarray with shape(N,)): coordinates to evaluate basis
        n (int): toroidal mode number
        NFP (int): number of field periods
    
    Returns:
        y (ndarray with shape(N,)): basis function evaluated at specified points
    """
    n_pos = n >= 0
    n_neg = n < 0
    n = jnp.abs(n)
    y2 = n_pos*(-(n*NFP)**2*jnp.cos(n*NFP*zeta)) + n_neg*(-(n*NFP)**2*jnp.sin(n*NFP*zeta))
    
    return y2

@conditional_decorator(functools.partial(jit), use_jax)
def four_toroidal_zzz(zeta,n,NFP):
    """Toroidal Fourier basis function, third toroidal derivative
    
    Args:
        zeta (ndarray with shape(N,)): coordinates to evaluate basis
        n (int): toroidal mode number
        NFP (int): number of field periods
    
    Returns:
        y (ndarray with shape(N,)): basis function evaluated at specified points
    """
    n_pos = n >= 0
    n_neg = n < 0
    n = jnp.abs(n)
    y2 = n_pos*((n*NFP)**3*jnp.sin(n*NFP*zeta)) + n_neg*(-(n*NFP)**3*jnp.cos(n*NFP*zeta))
    
    return y2

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

@conditional_decorator(functools.partial(jit,static_argnums=(4,5)), use_jax)
def zern(r,theta,l,m,dr,dtheta):
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
    radial = radial_derivatives[dr](r,l,m)
    azimuthal = poloidal_derivatives[dtheta](theta,l,m)
    
    return radial*azimuthal    

@conditional_decorator(functools.partial(jit,static_argnums=(3,)), use_jax)
def four(zeta,n,NFP,dz):
    """Toroidal Fourier basis function
    
    Args:
        zeta (ndarray with shape(N,)): coordinates to evaluate basis
        n (int): toroidal mode number
        NFP (int): number of field periods
        dz (int): order of toroidal derivative
    
    Returns:
        y (ndarray with shape(N,)): basis function evaluated at specified points
    """
    return toroidal_derivatives[dz](zeta,n,NFP)
    
@conditional_decorator(functools.partial(jit), use_jax)
def double_fourier_basis(theta,phi,m,n,NFP):
    """Double Fourier series for boundary/lambda
    
    Args:
        theta (ndarray with shape(N,)): poloidal angle to evaluate basis
        phi (ndarray with shape(N,)): toroidal angle to evaluate basis
        m (int): poloidal mode number
        n (int): toroidal mode number
        NFP (int): number of field periods
    
    Returns:
        y (ndarray with shape(N,)): basis function evaluated at specified points
    """
    m_pos = m >= 0
    m_neg = m < 0
    n_pos = n >= 0
    n_neg = n < 0
    m = jnp.abs(m)
    n = jnp.abs(n)
    m_term = m_pos*jnp.cos(m*theta) + m_neg*jnp.sin(m*theta)
    n_term = n_pos*jnp.cos(n*NFP*phi) + n_neg*jnp.sin(n*NFP*phi)

    return m_term*n_term

@conditional_decorator(functools.partial(jit,static_argnums=(7,8,9)), use_jax)
def fourzern(r,theta,zeta,l,m,n,NFP,dr,dv,dz):
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
    return zern(r,theta,l,m,dr,dv)*four(zeta,n,NFP,dz)


class ZernikeTransform():
    """Zernike Transform (really a Fourier-Zernike, but whatever)
    
    Args:
        nodes (ndarray, shape(3,N)): nodes where basis functions are evaluated. 
            First index is (rho,theta,phi), 2nd index is node number
        idx (ndarray of int, shape(Nc,3)): mode numbers for spectra basis. each row is one basis function with 
            modes (l,m,n)
        NFP (int): number of field periods   
        derivatives (array-like, shape(n,3)): orders of derivatives to compute in rho,theta,zeta.
            Each row of the array should contain 3 elements corresponding to derivatives in rho,theta,zeta
    """
    def __init__(self,nodes, idx,NFP,derivatives=[0,0,0]):
        # array of which l,m,n is at which column of the interpolation matrix
        self.idx = idx
        # array of which r,v,z is at which row of the interpolation matrix
        self.nodes = nodes
        self.NFP = NFP
        self.derivatives = np.atleast_2d(derivatives)
        self.matrices = {i:{j:{k:{} for k in range(4)} for j in range(4)} for i in range(4)}
        self._build(self.derivatives,self.idx)
    
    def _build(self,derivs, idx):
        for d in derivs:
            dr = d[0]
            dv = d[1]
            dz = d[2]
            self.matrices[dr][dv][dz] = jnp.stack([fourzern(self.nodes[0],self.nodes[1],self.nodes[2],
                                               lmn[0],lmn[1],lmn[2],self.NFP,dr,dv,dz) for lmn in idx]).T 
            
    def expand_nodes(self,new_nodes):
        pass
    
    def expand_M(self,Mnew):
        pass
    
    def expand_N(self,Nnew):
        pass
    
    def expand_derivatives(self,new_derivatives):
        pass
        
    @conditional_decorator(functools.partial(jit,static_argnums=(0,2,3,4)), use_jax)    
    def transform(self,c,dr,dv,dz):
        """Transform from spectral domain to physical
        
        Args:
            c (ndarray, shape(N_coeffs,)): spectral coefficients, indexed as (lm,n) flattened in row major order
            dr (int): order of radial derivative
            dv (int): order of poloidal derivative
            dz (int): order of toroidal derivative
            
        Returns:
            x (ndarray, shape(N_nodes,)): array of values of function at node locations
        """
        return jnp.matmul(self.matrices[dr][dv][dz],c)

    @conditional_decorator(functools.partial(jit,static_argnums=(0,2)), use_jax)    
    def fit(self,x,rcond=1e-6):
        """Transform from physical domain to spectral
        
        Args:
            x (ndarray, shape(N_nodes,)): values in real space at coordinates specified by self.nodes
            rcond (float): relative cutoff for singular values in least squares fit  
            
        Returns:
            c (ndarray, shape(N_coeffs,)): spectral coefficients in Fourier-Zernike basis
        """
        return jnp.linalg.lstsq(self.matrices[0][0][0],x,rcond=rcond)[0]
        
        
def get_zern_basis_idx_dense(M,N):
    num_lm_modes = (M+1)**2
    num_four = 2*N+1
    return jnp.array([(*fringe_to_lm(i),n-N) for i in range(num_lm_modes) for n in range(num_four)])

def get_double_four_basis_idx_dense(M,N):
    dimFourM = 2*M+1
    dimFourN = 2*N+1
    return jnp.array([[m-M,n-N] for m in range(dimFourM) for n in range(dimFourN)])

def zernike_norm(l, m):
    """Norm of a Zernike polynomial with l, m indexing.
    Returns the integral (Z^m_l)^2 r dr dt, r=[0,1], t=[0,2*pi]
    """
    return np.sqrt((2 * (l + 1)) / (np.pi*(1 + np.kronecker(m, 0))))

def lm_to_fringe(l, m):
    """Convert (l,m) double index to single Fringe index.
    """
    M = (l + np.abs(m)) / 2
    return int(M**2 + M + m)

def fringe_to_lm(idx):
    """Convert single Fringe index to (l,m) double index.
    """
    M = (np.ceil(np.sqrt(idx+1)) - 1)
    m = idx - M**2 - M
    l = 2*M - np.abs(m)
    return int(l), int(m)

class JacobiCoeffs():
    def __init__(self,M):
        self.coeffs = {}
        for i in range((M+1)**2):
            l,m = fringe_to_lm(i)
            self._compute_coeffs(l,m)
    def _compute_coeffs(self,l,m):
        lm_even = (l-m)%2 == 0
        m = abs(m)
        self.coeffs[(l,m)] = np.array([lm_even*(-1)**k* factorial(l-k)/(
            factorial(k) * factorial((l+m)//2-k) * factorial((l-m)//2-k)) for k in np.arange(0,(l-m)/2+1,1).astype(np.float64)]) 
    def __call__(self,l,m):
        m = abs(m)
        if (l,m) in self.coeffs:
            return self.coeffs.get((l,m))
        else:
            self._compute_coeffs(l,m)
            return self.coeffs[(l,m)]

def eval_four_zern(c,idx,NFP,rho,theta,zeta,dr=0,dv=0,dz=0):
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
    Z = jnp.stack([fourzern(rho,theta,zeta,lmn[0],lmn[1],lmn[2],NFP,dr,dv,dz) for lmn in idx]).T 
    Z = jnp.atleast_2d(Z)
    f = jnp.matmul(Z,c)
    return f


@conditional_decorator(functools.partial(jit), use_jax)
def eval_double_fourier(c,idx,NFP,theta,phi):
    """Evaluates double fourier series F = sum(F_mn(theta,phi))
    
    Where F_mn(theta,phi) = f_mn*cos(m*theta)*sin(n*phi)
    
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
    f = jnp.zeros_like(theta)
    for k, cc in enumerate(c):
        m = idx[k,0]
        n = idx[k,1]
        f = f + cc*double_fourier_basis(theta,phi,m,n,NFP)
        
    return f


def axis_posn(cR,cZ,zern_idx,NFP):
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
    R0 = eval_four_zern(cR,zern_idx,NFP,0.,0.,0.,dr=0,dv=0,dz=0)[0]
    Z0 = eval_four_zern(cZ,zern_idx,NFP,0.,0.,0.,dr=0,dv=0,dz=0)[0]
    
    return R0,Z0