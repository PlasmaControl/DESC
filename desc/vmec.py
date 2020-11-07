import numpy as np
from scipy.optimize import fsolve
from netCDF4 import Dataset

from desc.backend import sign
from desc.zernike import ZernikeTransform


# TODO: add other fields including B, rmns, zmnc, lmnc, etc
def read_vmec_output(fname):
    """Reads VMEC data from wout nc file

    Args:
        fname (string): filename of VMEC output file

    Returns:
        vmec_data (dictionary): the VMEC data fields
    """

    file = Dataset(fname, mode='r')

    vmec_data = {
        'NFP': file.variables['nfp'][:],
        'psi': file.variables['phi'][:], # toroidal flux is saved as 'phi'
        'xm': file.variables['xm'][:],
        'xn': file.variables['xn'][:],
        'rmnc': file.variables['rmnc'][:],
        'zmns': file.variables['zmns'][:],
        'lmns': file.variables['lmns'][:]
    }
    try:
        vmec_data['rmns'] = file.variables['rmns'][:]
        vmec_data['zmnc'] = file.variables['zmnc'][:]
        vmec_data['lmnc'] = file.variables['lmnc'][:]
        vmec_data['sym'] = False
    except:
        vmec_data['sym'] = True

    return vmec_data


def vmec_error(equil, vmec_data, Npol=16, Ntor=16):
    """Computes error in SFL coordinates compared to VMEC solution

    Args:
        equil:
        vmec_data:
        Npol (int): number of poloidal angles to sample
        Ntor (int): number of toroidal angles to sample

    Returns:
        average Euclidean distance between VMEC and DESC sample points
    """

    ns = np.size(vmec_data['psi'])
    vartheta = np.linspace(0, 2*np.pi, Npol, endpoint=False)
    zeta = np.linspace(0, 2*np.pi/vmec_data['NFP'], Ntor, endpoint=False)
    phi = zeta

    R_vmec = np.zeros((ns, Npol, Ntor))
    Z_vmec = np.zeros((ns, Npol, Ntor))
    for k in range(Ntor):           # toroidal angle
        for i in range(ns):         # flux surface
            theta = np.zeros((Npol,))
            for j in range(Npol):   # poloidal angle
                f0 = sfl_err(0, vartheta[j], zeta[k], vmec_data, i)
                f2pi = sfl_err(2*np.pi, vartheta[j], zeta[k], vmec_data, i)
                flag = (sign(f0) + sign(f2pi)) / 2
                args = (vartheta[j], zeta[k], vmec_data, i, flag)
                t = fsolve(sfl_err, vartheta[j], args=args)
                if flag != 0:
                    t = np.remainder(t+np.pi, 2*np.pi)
                theta[j] = t;   # theta angle that corresponds to vartheta[j]
            R_vmec[i,:,k] = vmec_transf(vmec_data['rmnc'], vmec_data['xm'], vmec_data['xn'], theta, phi[k], trig='cos')
            Z_vmec[i,:,k] = vmec_transf(vmec_data['zmns'], vmec_data['xm'], vmec_data['xn'], theta, phi[k], trig='sin')
            if not vmec_data['sym']:
                R_vmec[i,:,k] += vmec_transf(vmec_data['rmns'], vmec_data['xm'], vmec_data['xn'], theta, phi[k], trig='sin')
                Z_vmec[i,:,k] += vmec_transf(vmec_data['zmnc'], vmec_data['xm'], vmec_data['xn'], theta, phi[k], trig='cos')

    r = np.tile(np.sqrt(vmec_data['psi']), (1, Npol, Ntor))
    v = np.tile(vartheta, (ns, 1, Ntor))
    z = np.tile(zeta, (ns, Npol, 1))
    nodes = np.stack([r.flatten(), v.flatten(), z.flatten()])
    zernike_transform = ZernikeTransform(nodes, equil['zern_idx'], equil['NFP'], method='fft')
    R_desc = zernike_transform.transform(equil['cR'], 0, 0, 0)
    Z_desc = zernike_transform.transform(equil['cZ'], 0, 0, 0)

    return np.mean(np.sqrt((R_vmec - R_desc)**2 + (Z_vmec - Z_desc)**2))


def sfl_err(theta, vartheta, zeta, vmec_data, s, flag=0):
    """f(theta) = vartheta - theta - lambda(theta)

    Args:
        theta (float): VMEC poloidal angle
        vartheta (float): sfl poloidal angle
        zeta (float): VMEC/sfl toroidal angle
        vmec_data:
        flag (int): offsets theta to ensure f(theta) has one zero

    Returns:
        vartheta - theta - lambda
    """

    theta = theta + np.pi*flag;
    phi = zeta
    l = vmec_transf(vmec_data['lmns'][s,:], vmec_data['xm'], vmec_data['xn'], theta, phi, trig='sin')
    if not vmec_data['sym']:
        l += vmec_transf(vmec_data['lmnc'][s,:], vmec_data['xm'], vmec_data['xn'], theta, phi, trig='cos')
    return vartheta - theta - l;


def vmec_transf(xmna, xm, xn, theta, phi, trig='sin'):
    """Compute Fourier transform of VMEC data

    Args:
        xmns (2d float array): xmnc[:,i] are the sin coefficients at flux surface i
        xm (1d int array): poloidal mode numbers
        xn (1d int array): toroidal mode numbers
        theta (1d float array): poloidal angles
        phi (1d float array): toroidal angles
        trig (string): type of transform, options are 'sin' or 'cos'

    Returns:
        f (3d float array): f[i,j,k] is the transformed data at flux surface i, theta[j], phi[k]
    """

    ns = np.shape(np.atleast_2d(xmna))[0]
    lt = np.size(theta)
    lp = np.size(phi)
    # Create mode x angle arrays
    mtheta = np.atleast_2d(xm).T @ np.atleast_2d(theta)
    nphi   = np.atleast_2d(xn).T @ np.atleast_2d(phi)
    # Create trig arrays
    cosmt = np.cos(mtheta)
    sinmt = np.sin(mtheta)
    cosnp = np.cos(nphi)
    sinnp = np.sin(nphi)
    # Calcualte the transform
    f = np.zeros((ns,lt,lp))
    for k in range(ns):
        xmn = np.tile(np.atleast_2d(np.atleast_2d(xmna)[k,:]).T, (1, lt))
        if trig == 'sin':
            f[k,:,:] = np.tensordot((xmn*sinmt).T, cosnp, axes=1) + np.tensordot((xmn*cosmt).T, sinnp, axes=1)
        elif trig == 'cos':
            f[k,:,:] = np.tensordot((xmn*cosmt).T, cosnp, axes=1) - np.tensordot((xmn*sinmt).T, sinnp, axes=1)
    return f


# TODO: replace this function with vmec_transf
def vmec_interpolate(Cmn, Smn, xm, xn, theta, phi, sym=True):
    """Interpolates VMEC data on a flux surface

    Args:
        Cmn (ndarray, shape(MN,)): cos(mt-np) Fourier coefficients
        Smn (ndarray, shape(MN,)): sin(mt-np) Fourier coefficients
        xm (ndarray, shape(M,)): poloidal mode numbers
        xn (ndarray, shape(N,)): toroidal mode numbers
        theta (ndarray): poloidal angles
        phi (ndarray): toroidal angles
        sym (bool): stellarator symmetry

    Returns:
        if sym = True:
            C, S (tuple of ndarray): VMEC data interpolated at the angles (theta,phi)
            where C has cosine symmetry and S has sine symmetry
        if sym = False:
            X (ndarray): non-symmetric VMEC data interpolated at the angles (theta,phi)
    """

    C_arr = []
    S_arr = []
    dim = Cmn.shape

    for j in range(dim[1]):

        m = xm[j]
        n = xn[j]

        C = [[[Cmn[s, j]*np.cos(m*t - n*p) for p in phi]
              for t in theta] for s in range(dim[0])]
        S = [[[Smn[s, j]*np.sin(m*t - n*p) for p in phi]
              for t in theta] for s in range(dim[0])]
        C_arr.append(C)
        S_arr.append(S)

    C = np.sum(C_arr, axis=0)
    S = np.sum(S_arr, axis=0)
    if sym:
        return C, S
    else:
        return C + S