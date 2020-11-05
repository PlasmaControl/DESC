import numpy as np
from desc.backend import TextColors


def get_nodes_pattern(M, N, NFP, index='fringe', surfs='cheb1', sym=False, axis=True):
    """Compute interpolation nodes on a patterned grid

    Args:
        M (int): maximum poloidal mode number
        N (int): maximum toroidal mode number
        NFP (int): number of field periods
        index (string): Zernike polynomial index ordering
            ansi = OSA / ANSI indexing
            fringe = Fringe / University of Arizona indexing
        surfs (string): pattern for radial surfaces
            cheb1 = Chebyshev-Gauss-Lobatto nodes scaled to r=[0,1]
            cheb2 = Chebyshev-Gauss-Lobatto nodes scaled to r=[-1,1]
            any other value defaults to linear spacing in r
        sym (bool): False for nodes to fill the full domain,
                    True for nodes in the stellarator symmetry (half) domain
        axis (bool): True to include a node at the origin (magnetic axis)

    Returns:
        nodes (ndarray, size(3,Nnodes)): node coordinates, in (rho,theta,zeta).
        volumes (ndarray, size(3,Nnodes)): node spacing (drho,dtheta,dzeta) at each node coordinate.
    """

    dimFour = 2*N+1
    if index in ['ansi', 'chevron']:
        dimZern = int((M+1)*(M+2)/2)
        a = 1
    elif index in ['fringe', 'house']:
        dimZern = int((M+1)**2)
        a = 2
    else:
        raise ValueError(
            TextColors.FAIL + "Invalid Zernike polynomial index ordering input." + TextColors.ENDC)

    pattern = {
        'cheb1': (np.cos(np.arange(M, -1, -1)*np.pi/M)+1)/2,
        'cheb2': -np.cos(np.arange(M, 2*M+1, 1)*np.pi/(2*M))
    }
    rho = pattern.get(surfs, np.linspace(0, 1, num=M+1))
    rho = np.sort(rho, axis=None)
    if axis:
        rho[0] = 0
    else:
        rho[0] = rho[1]/4

    drho = np.zeros_like(rho)
    for i in range(rho.size):
        if i == 0:
            drho[i] = (rho[0]+rho[1])/2
        elif i == rho.size-1:
            drho[i] = 1-(rho[-2]+rho[-1])/2
        else:
            drho[i] = (rho[i+1]-rho[i-1])/2

    r = np.zeros(dimZern)
    t = np.zeros(dimZern)
    dr = np.zeros(dimZern)
    dt = np.zeros(dimZern)

    i = 0
    for m in range(M+1):
        dtheta = 2*np.pi/(a*m+1)
        theta = np.arange(0, 2*np.pi, dtheta)
        for j in range(a*m+1):
            r[i] = rho[m]
            t[i] = theta[j]
            dr[i] = drho[m]
            dt[i] = dtheta
            i += 1

    dz = 2*np.pi/(NFP*dimFour)
    z = np.arange(0, 2*np.pi/NFP, dz)

    r = np.tile(r, dimFour)
    t = np.tile(t, dimFour)
    z = np.tile(z[np.newaxis], (dimZern, 1)).flatten(order='F')
    dr = np.tile(dr, dimFour)
    dt = np.tile(dt, dimFour)
    dz = np.ones_like(z)*dz

    nodes = np.stack([r, t, z])
    volumes = np.stack([dr, dt, dz])

    if sym:
        non_sym_idx = np.where(t > np.pi)
        nodes = np.delete(nodes, non_sym_idx, axis=1)
        volumes = np.delete(volumes, non_sym_idx, axis=1)

    sort_idx = np.lexsort((nodes[1], nodes[0], nodes[2]))
    return nodes[:, sort_idx], volumes[:, sort_idx]


def get_nodes_surf(M, N, NFP, surf=1.0, sym=False):
    """Compute interpolation nodes on a single surface

    Args:
        M (int): maximum poloidal mode number
        N (int): maximum toroidal mode number
        NFP (int): number of field periods
        surf (float): radial coordinate of flux surface
        sym (bool): False for nodes to fill the full domain,
                    True for nodes in the stellarator symmetry (half) domain

    Returns:
        nodes (ndarray, size(3,Nnodes)): node coordinates, in (rho,theta,zeta).
        volumes (ndarray, size(3,Nnodes)): node spacing (drho,dtheta,dzeta) at each node coordinate.
    """

    dimFourM = 2*M+1
    dimFourN = 2*N+1

    dt = 2*np.pi/dimFourM
    t = np.arange(0, 2*np.pi, dt)

    dz = 2*np.pi/(NFP*dimFourN)
    z = np.arange(0, 2*np.pi/NFP, dz)

    t, z = np.meshgrid(t, z, indexing='ij')
    t = t.flatten()
    z = z.flatten()
    r = np.ones_like(t)*surf

    dr = np.ones_like(r)
    dt = np.ones_like(t)*dt
    dz = np.ones_like(z)*dz

    nodes = np.stack([r, t, z])
    volumes = np.stack([dr, dt, dz])

    if sym:
        non_sym_idx = np.where(t > np.pi)
        nodes = np.delete(nodes, non_sym_idx, axis=1)
        volumes = np.delete(volumes, non_sym_idx, axis=1)

    sort_idx = np.lexsort((nodes[1], nodes[0], nodes[2]))
    return nodes[:, sort_idx], volumes[:, sort_idx]


def get_nodes_grid(NFP, nr=None, nt=None, nz=None):
    """Compute interpolation nodes on a uniform grid

    Args:
        NFP (int): number of field periods
        nr (int): number of radial grid points
        nt (int): number of poloidal grid points
        nz (int): number of toroidal grid points

    Returns:
        nodes (ndarray, size(3,Nnodes)): node coordinates, in (rho,theta,zeta).
        volumes (ndarray, size(3,Nnodes)): node spacing (drho,dtheta,dzeta) at each node coordinate.
    """

    r = np.linspace(0, 1, nr)
    dr = 1/nr

    t = np.linspace(0, 2*np.pi, nt)
    dt = 2*np.pi/nt

    z = np.linspace(0, 2*np.pi/NFP, nz)
    dz = 2*np.pi/NFP/nz

    r, t, z = np.meshgrid(r, t, z, indexing='ij')
    r = r.flatten()
    t = t.flatten()
    z = z.flatten()

    dr = dr*np.ones_like(r)
    dt = dt*np.ones_like(t)
    dz = dz*np.ones_like(z)

    nodes = np.stack([r, t, z])
    volumes = np.stack([dr, dt, dz])

    return nodes, volumes


# these functions are currently unused
# TODO: finish option for placing nodes at irrational surfaces

def dec_to_cf(x, dmax=6):
    """Compute continued fraction form of a number.

    Args:
        x (float): floating point form of number
        dmax (int): maximum iterations (ie, number of coefficients of continued fraction).

    Returns:
        cf (ndarray of int): coefficients of continued fraction form of x.
    """
    cf = []
    q = np.floor(x)
    cf.append(q)
    x = x - q
    i = 0
    while x != 0 and i < dmax:
        q = np.floor(1 / x)
        cf.append(q)
        x = 1 / x - q
        i = i + 1
    return np.array(cf)


def cf_to_dec(cf):
    """Compute decimal form of a continued fraction.

    Args:
        cf (array-like): coefficients of continued fraction.

    Returns:
        x (float): floating point representation of cf
    """
    if len(cf) == 1:
        return cf[0]
    else:
        return cf[0] + 1/cf_to_dec(cf[1:])


def most_rational(a, b):
    """Compute the most rational number in the range [a,b]

    Args:
        a,b (float): lower and upper bounds

    Returns:
        x (float): most rational number between [a,b]

    """

    # handle empty range
    if a == b:
        return a
    # ensure a < b
    elif a > b:
        c = a
        a = b
        b = c
    # return 0 if in range
    if np.sign(a*b) <= 0:
        return 0
    # handle negative ranges
    elif np.sign(a) < 0:
        s = -1
        a *= -1
        b *= -1
    else:
        s = 1

    a_cf = dec_to_cf(a)
    b_cf = dec_to_cf(b)
    idx = 0  # first idex of dissimilar digits
    for i in range(min(a_cf.size, b_cf.size)):
        if a_cf[i] != b_cf[i]:
            idx = i
            break
    f = 1
    while True:
        dec = cf_to_dec(np.append(a_cf[0:idx], f))
        if dec >= a and dec <= b:
            return dec*s
        f += 1
