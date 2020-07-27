import numpy as np
from math import factorial


def zernike_norm(l, m):
    """Norm of a Zernike polynomial with l, m indexing."""
    return np.sqrt((2 * (l + 1)) / (1 + np.kronecker(m, 0)))

def lm_to_fringe(l, m):
    """Convert (l,m) two term index to Fringe index."""
    term1 = (1 + (l + np.abs(m))/2)**2
    term2 = 2 * np.abs(m)
    term3 = (1 + np.sign(m)) / 2
    return int(term1 - term2 - term3)  

def fringe_to_lm(idx):
    """Convert Fringe Z to (l, m) two-term index."""
    idx += 1 # shift 0 base to 1 base
    m_l = 2 * (np.ceil(np.sqrt(idx)) - 1)  # sum of n+m
    g_s = (m_l / 2)**2 + 1  # start of each group of equal n+m given as idx index
    l = m_l / 2 + np.floor((idx - g_s) / 2)
    m = (m_l - l) * (1 - np.mod(idx-g_s, 2) * 2)
    return int(l), int(m)

def zern_radial(x,l,m):
    """Zernike radial basis function
    
    Args:
        x (array-like): points to evaluate basis 
        l (int): radial mode number
        m (int): azimuthal mode number
    """
    m = abs(m)
    y = np.zeros_like(x)
    if (l-m)%2 != 0:
        return y
    else:
        for k in range(int((l-m)/2)+1):
            y += (-1)**k* factorial(l-k)/(
                 factorial(k)* factorial((l+m)/2-k)* factorial((l-m)/2-k))*x**(l-2*k)
        return y
    
def zern_radial_r(x,l,m):
    """Zernike radial basis function, first derivative in r
    
    Args:
        x (array-like): points to evaluate basis 
        l (int): radial mode number
        m (int): azimuthal mode number
    """
    m = abs(m)
    y = np.zeros_like(x)
    if (l-m)%2 != 0:
        return y
    else:
        for k in range(int((l-m)/2)+1):
            y += (l-2*k)*(-1)**k* factorial(l-k)/(
                 factorial(k)* factorial((l+m)/2-k)* factorial((l-m)/2-k))*x**max(l-2*k-1,0)
        return y
    
def zern_radial_rr(x,l,m):
    """Zernike radial basis function, second radial derivative
    
    Args:
        x (array-like): points to evaluate basis 
        l (int): radial mode number
        m (int): azimuthal mode number
    """
    y = np.zeros_like(x)
    if (l-m)%2 != 0:
        return y
    else:
        m = abs(m)
        for k in range(int((l-m)/2)+1):
            y += (l-2*k-1)*(l-2*k)*(-1)**k* factorial(l-k)/(
                 factorial(k)* factorial((l+m)/2-k)* factorial((l-m)/2-k))*x**max(l-2*k-2,0)
        return y
    
def zern_azimuthal(theta,l,m):
    """Zernike azimuthal basis function
    
    Args:
        theta (array-like): points to evaluate basis
        l (int): radial mode number
        m (int): azimuthal mode number
    """
    if m<0:
        return np.sin(abs(m)*theta)
    else:
        return np.cos(abs(m)*theta)
    
def zern_azimuthal_v(theta,l,m):
    """Zernike azimuthal basis function, first azimuthal derivative
    
    Args:
        theta (array-like): points to evaluate basis
        l (int): radial mode number
        m (int): azimuthal mode number
    """
    if m<0:
        return m*np.cos(abs(m)*theta)
    else:
        return -m*np.sin(abs(m)*theta)
    
def zern_azimuthal_vv(theta,l,m):
    """Zernike azimuthal basis function, second azimuthal derivative
    
    Args:
        theta (array-like): points to evaluate basis
        l (int): radial mode number
        m (int): azimuthal mode number
    """
    if m<0:
        return -m**2*np.sin(abs(m)*theta)
    else:
        return -m**2*np.cos(abs(m)*theta)
    
def zern(r,theta,l,m,dr=0,dtheta=0):
    """Zernike 2D basis function
    
    Args:
        r (array-like): radial coordinates to evaluate basis
        theta (array-like): azimuthal coordinates to evaluate basis
        l (int): radial mode number
        m (int): azimuthal mode number
        dr (int): order of radial derivative
        dtheta (int): order of azimuthal derivative
    """
    if dr == 0:
        radial = zern_radial(r,l,m)
    elif dr == 1:
        radial = zern_radial_r(r,l,m)
    elif dr == 2:
        radial = zern_radial_rr(r,l,m)
    else:
        raise NotImplementedError
        
    if dtheta == 0:
        azimuthal = zern_azimuthal(theta,l,m)
    elif dtheta == 1:
        azimuthal = zern_azimuthal_v(theta,l,m)
    elif dtheta == 2:
        azimuthal = zern_azimuthal_vv(theta,l,m)
    else:
        raise NotImplementedError
    return radial*azimuthal

def four(zeta,n,NFP,dz=0):
    """Toroidal Fourier basis function
    
    Args:
        zeta (array-like): coordinates to evaluate basis
        n (int): toroidal mode number
        NFP (int): number of field periods
        dz (int): order or toroidal derivative    
    """
    if dz == 0:
        if n<0:
            return np.sin(abs(n)*NFP*zeta)
        else:
            return np.cos(abs(n)*NFP*zeta)
    if dz == 1:
        if n<0:
            return n*NFP*np.cos(abs(n)*NFP*zeta)
        else:
            return -n*NFP*np.sin(abs(n)*NFP*zeta)
    if dz == 2:
        if n<0:
            return -(n*NFP)**2*np.sin(abs(n)*NFP*zeta)
        else:
            return -(n*NFP)**2*np.cos(abs(n)*NFP*zeta)
        
def F_mn(theta,phi,m,n,NFP):
    """Double Fourier series for boundary/lambda
    
    Args:
        theta (array-like): poloidal angle to evaluate basis
        phi (array-like): toroidal angle to evaluate bais
        m (int): poloidal mode number
        n (int): toroidal mode number
        NFP (int): number of field periods
    """
    if m>=0 and n>=0:
        return np.cos(abs(m)*theta)*np.cos(abs(n)*NFP*phi)
    if m>=0 and n<0:
        return np.cos(abs(m)*theta)*np.sin(abs(n)*NFP*phi)
    if m<0 and n>=0:
        return np.sin(abs(m)*theta)*np.cos(abs(n)*NFP*phi)
    if m<0 and n<0:
        return np.sin(abs(m)*theta)*np.sin(abs(n)*NFP*phi)
        
def fourzern(r,theta,zeta,l,m,n,NFP,dr=0,dv=0,dz=0):
    """Combined 3D Fourier-Zernike basis function
    
    Args:
        r (array-like): radial coordinates
        theta (array-like): poloidal coordinates
        zeta (array-like): toroidal coordinates
        l (int): radial mode number
        m (int): poloidal mode number
        n (int): toroidal mode number
        NFP (int): number of field periods
        dr (int): order of radial derivative
        dv (int): order of poloidal derivative
        dz (int): order of toroidal derivative
    """
    return zern(r,theta,l,m,dr,dv)*four(zeta,n,NFP,dz)


class ZernikeTransform():
    """Zernike Transform (really a Fourier-Zernike, but whatever)
    
    Args:
        nodes (array-like): nodes where basis functions are evaluated
        M (int): maximum poloidal mode number
        N (int): maximum toroidal mode number
        NFP (int): number of field periods    
    """
    def __init__(self,nodes, M,N,NFP):
        num_lm_modes = (M+1)**2
        num_four = 2*N+1
        # array of which l,m,n is at which column of the interpolation matrix
        self.idx = np.array([(*fringe_to_lm(i),n) for i in range(num_lm_modes) for n in range(num_four)])
        # array of which r,v,z is at which row of the interpolation matrix
        self.nodes = nodes
        self.M = M
        self.N = N
        self.NFP = NFP
        self.matrices = {i:{j:{k:np.stack([fourzern(nodes[0],nodes[1],nodes[2],
                                               *fringe_to_lm(lm),n-N,NFP,dr=i,dv=j,dz=k) 
                                           for lm in range(num_lm_modes) for n in range(num_four)]).T 
                                           for k in range(3)} for j in range(3)} for i in range(3)}
    def transform(self,c,dr=0,dv=0,dz=0):
        """Transform from spectral domain to physical
        
        Args:
            c (array-like): spectral coefficients, indexed as (lm,n) flattened in row major order
            dr (int): order of radial derivative
            dv (int): order of poloidal derivative
            dz (int): order of toroidal derivative
        """
        return np.matmul(self.matrices[dr][dv][dz],c)
    
    def fit(self,x,rcond=1e-1):
        """Transform from physical domain to spectral
        
        Args:
            x (array-like): values in real space at coordinates specified by self.nodes
            rcond (float): relative cutoff for singular values in least squares fit        
        """
        return np.linalg.lstsq(self.matrices[0][0][0],x,rcond=rcond)[0]
        