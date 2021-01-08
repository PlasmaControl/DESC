import numpy as np
from netCDF4 import Dataset, stringtochar
from scipy.optimize import fsolve

from desc.utils import sign
from desc.grid import LinearGrid
from desc.transform import Transform
from desc.configuration import Configuration


class VMECIO():
    """Performs input from VMEC netCDF files to DESC Configurations and vice-versa."""

    @staticmethod
    def load(path:str, L:int=None, index:str='ansi') -> Configuration:
        """Loads a VMEC netCDF file as a Configuration.

        Parameters
        ----------
        path : str
            File path of input data.

        Returns
        -------
        config: Configuration
            Configuration that resembles the VMEC data.

        """
        file = Dataset(path, mode='r')
        inputs = {}

        inputs['sym'] = not bool(file.variables['lasym'][:])
        inputs['Psi'] = file.variables['phi'][:] # toroidal flux is saved as 'phi'
        inputs['NFP'] = file.variables['nfp'][:]
        inputs['M'] = file.variables['mpol'][:] - 1
        inputs['N'] = file.variables['ntor'][:]
        inputs['index'] = index
        if L is None:
            default_L = {'fringe': 2*inputs['M'],
                         'ansi': inputs['M'],
                         'chevron': inputs['M'],
                         'house': 2*inputs['M']}
            inputs['L'] = default_L[inputs['index']]

        # TODO: add profiles, boundary, and x

        file.close

        return Configuration(inputs=inputs)

    @staticmethod
    def save(config:Configuration, path:str, surfs:int=128) -> None:
        """Saves a Configuration as a netCDF file in the VMEC format.

        Parameters
        ----------
        config : Configuration
            Configuration to save.
        path : str
            File path of output data.
        surfs: int (Default = 128)
            Number of flux surfaces to interpolate at.

        Returns
        -------
        None

        """

        p_l = config.p_l
        i_l = config.i_l
        Psi = config.Psi
        NFP = config.NFP
        M = config.M
        N = config.N

        s_full = np.linspace(0, 1, surfs)
        s_half = s_full[0:-1] + 0.5/(surfs-1)
        r_full = np.sqrt(s_full)
        r_half = np.sqrt(s_half)
        full_grid = LinearGrid(rho=r_full)
        half_grid = LinearGrid(rho=r_half)

        p_transform_full = Transform(full_grid, config.p_basis)
        p_transform_half = Transform(half_grid, config.p_basis)
        i_transform_full = Transform(full_grid, config.i_basis)
        i_transform_half = Transform(half_grid, config.i_basis)

        """ VMEC netCDF file is generated in VMEC2000/Sources/Input_Output/wrout.f
            see lines 300+ for full list of included variables
        """

        file = Dataset(path, mode='w')

        # dimensions
        file.createDimension('radius', surfs)   # number of flux surfaces
        file.createDimension('mn_mode', (2*N+1)*M+N+1)  # number of Fourier modes
        file.createDimension('mn_mode_nyq', None)  # used for Nyquist quantities
        file.createDimension('n_tor', 1)    # number of axis guess Fourier modes
        file.createDimension('preset', 21)  # max dimension of profile inputs
        file.createDimension('ndfmax', 101) # used for am_aux & ai_aux
        file.createDimension('time', 100)   # used for fsqrt & wdot
        file.createDimension('dim_00001', 1)
        file.createDimension('dim_00020', 20)
        file.createDimension('dim_00100', 100)
        file.createDimension('dim_00200', 200)

        # variables

        lasym = file.createVariable('lasym', np.int32, ('dim_00001',))
        lasym.long_name = 'asymmetry logical (0 = stellarator symmetry)'
        lasym[:] = int(not config.sym)

        lfreeb = file.createVariable('lfreeb', np.int32, ('dim_00001',))
        lfreeb.long_name = 'free boundary logical (0 = fixed boundary)'
        lfreeb[:] = 0

        ns = file.createVariable('ns', np.int32, ('dim_00001',))
        ns.long_name = 'number of flux surfaces'
        ns[:] = surfs

        mpol = file.createVariable('mpol', np.int32, ('dim_00001',))
        mpol.long_name = 'number of poloidal Fourier modes'
        mpol[:] = M+1

        ntor = file.createVariable('ntor', np.int32, ('dim_00001',))
        ntor.long_name = 'number of positive toroidal Fourier modes'
        ntor[:] = N

        nfp = file.createVariable('nfp', np.int32, ('dim_00001',))
        nfp.long_name = 'number of field periods'
        nfp[:] = NFP

        signgs = file.createVariable('signgs', np.float64, ('dim_00001',))
        signgs.long_name = 'sign of coordinate system jacobian'
        signgs[:] = 1   # TODO: don't hard-code this

        mnmax = file.createVariable('mnmax', np.int32, ('dim_00001',))
        mnmax.long_name = 'total number of Fourier modes'
        mnmax[:] = file.dimensions['mn_mode'].size

        xm = file.createVariable('xm', np.float64, ('mn_mode',))
        xm.long_name = 'poloidal mode numbers'
        xm[:] = np.tile(np.linspace(0, M, M+1), (2*N+1, 1)).T.flatten()[-file.dimensions['mn_mode'].size:]

        xn = file.createVariable('xn', np.float64, ('mn_mode',))
        xn.long_name = 'toroidal mode numbers'
        xn[:] = np.tile(np.linspace(-N, N, 2*N+1)*NFP, M+1)[-file.dimensions['mn_mode'].size:]

        gamma = file.createVariable('gamma', np.float64, ('dim_00001',))
        gamma.long_name = 'compressibility index (0 = pressure prescribed)'
        gamma[:] = 0

        am = file.createVariable('am', np.float64, ('preset',))
        am.long_name = 'pressure coefficients'
        am.units = 'Pa'
        am[:] = np.zeros((file.dimensions['preset'].size,))
        am[0:p_l.size] = p_l

        ai = file.createVariable('ai', np.float64, ('preset',))
        ai.long_name = 'rotational transform coefficients'
        ai[:] = np.zeros((file.dimensions['preset'].size,))
        ai[0:i_l.size] = i_l

        ac = file.createVariable('ac', np.float64, ('preset',))
        ac.long_name = 'normalized toroidal current density coefficients'
        ac[:] = np.zeros((file.dimensions['preset'].size,))

        power_series = stringtochar(np.array(['power_series         '],
                         'S'+str(file.dimensions['preset'].size)))

        pmass_type = file.createVariable('pmass_type', 'S1', ('preset',))
        pmass_type.long_name = 'parameterization of pressure function'
        pmass_type[:] = power_series

        piota_type = file.createVariable('piota_type', 'S1', ('preset',))
        piota_type.long_name = 'parameterization of rotational transform function'
        piota_type[:] = power_series

        pcurr_type = file.createVariable('pcurr_type', 'S1', ('preset',))
        pcurr_type.long_name = 'parameterization of current density function'
        pcurr_type[:] = power_series

        presf = file.createVariable('presf', np.float64, ('radius',))
        presf.long_name = 'pressure on full mesh'
        presf.units = 'Pa'
        presf[:] = p_transform_full.transform(p_l)

        pres = file.createVariable('pres', np.float64, ('radius',))
        pres.long_name = 'pressure on half mesh'
        pres.units = 'Pa'
        pres[0] = 0
        pres[1:] = p_transform_half.transform(p_l)

        mass = file.createVariable('mass', np.float64, ('radius',))
        mass.long_name = 'mass on half mesh'
        mass.units = 'Pa'
        mass[:] = pres[:]

        iotaf = file.createVariable('iotaf', np.float64, ('radius',))
        iotaf.long_name = 'rotational transform on full mesh'
        iotaf[:] = i_transform_full.transform(i_l)

        iotas = file.createVariable('iotas', np.float64, ('radius',))
        iotas.long_name = 'rotational transform on half mesh'
        iotas[0] = 0
        iotas[1:] = i_transform_half.transform(i_l)

        phi = file.createVariable('phi', np.float64, ('radius',))
        phi.long_name = 'toroidal flux'
        phi.units = 'Wb'
        phi[:] = np.linspace(0, Psi, surfs)

        phipf = file.createVariable('phipf', np.float64, ('radius',))
        phipf.long_name = 'd(phi)/ds: toroidal flux derivative'
        phipf[:] = Psi*np.ones((surfs,))

        phips = file.createVariable('phips', np.float64, ('radius',))
        phips.long_name = 'd(phi)/ds * sign(g)/2pi: toroidal flux derivative on half mesh'
        phips[0] = 0
        phips[1:] = phipf[1:]*signgs[:]/(2*np.pi)

        chi = file.createVariable('chi', np.float64, ('radius',))
        chi.long_name = 'poloidal flux'
        chi.units = 'Wb'
        chi[:] = phi[:]*signgs[:]

        chipf = file.createVariable('chipf', np.float64, ('radius',))
        chipf.long_name = 'd(chi)/ds: poloidal flux derivative'
        chipf[:] = phipf[:]*iotaf[:]

        file.close


def convert_vmec_to_desc(vmec_data, zern_idx, lambda_idx, Npol=None, Ntor=None):
    """Computes error in SFL coordinates compared to VMEC solution
    Parameters
    ----------
    vmec_data : dict
        dictionary of VMEC equilibrium parameters
    zern_idx : ndarray, shape(N_coeffs,3)
        indices for R,Z spectral basis,
        ie an array of [l,m,n] for each spectral coefficient
    lambda_idx : ndarray, shape(2M+1)*(2N+1)
        indices for lambda spectral basis,
        ie an array of [m,n] for each spectral coefficient
    Npol : int
        number of poloidal angles to sample per surface (Default value = M)
    Ntor : int
        number of toroidal angles to sample per surface (Default value = N)
    Returns
    -------
    equil : dict
        dictionary of DESC equilibrium parameters
    """
    if Npol is None:
        Npol = 2*np.max(zern_idx[:,1]) + 1
    if Ntor is None:
        Ntor = 2*np.max(zern_idx[:,2]) + 1

    ns = np.size(vmec_data['psi'])
    vartheta = np.linspace(0, 2*np.pi, Npol, endpoint=False)
    zeta = np.linspace(0, 2*np.pi/vmec_data['NFP'], Ntor, endpoint=False)
    phi = zeta

    r = np.tile(np.sqrt(vmec_data['psi'])[..., np.newaxis, np.newaxis], (1, Npol, Ntor))
    v = np.tile(vartheta[np.newaxis, ..., np.newaxis], (ns, 1, Ntor))
    z = np.tile(zeta[np.newaxis, np.newaxis, ...], (ns, Npol, 1))

    nodes = np.stack([r.flatten(), v.flatten(), z.flatten()])
    zernike_transform = ZernikeTransform(
        nodes, zern_idx, vmec_data['NFP'], method='fft')
    four_bdry_interp = double_fourier_basis(v[0,:,:].flatten(), z[0,:,:].flatten(), lambda_idx[:,0], lambda_idx[:,1], vmec_data['NFP'])
    four_bdry_interp_pinv = np.linalg.pinv(four_bdry_interp, rcond=1e-6)

    print('Interpolating VMEC solution to sfl coordinates')
    R = np.zeros((ns, Npol, Ntor))
    Z = np.zeros((ns, Npol, Ntor))
    L = np.zeros((Npol, Ntor))
    for k in range(Ntor):           # toroidal angle
        for i in range(ns):         # flux surface
            theta = np.zeros((Npol,))
            for j in range(Npol):   # poloidal angle
                f0 = sfl_err(np.array([0]), vartheta[j], zeta[k], vmec_data, i)
                f2pi = sfl_err(np.array([2*np.pi]),
                               vartheta[j], zeta[k], vmec_data, i)
                flag = (sign(f0) + sign(f2pi)) / 2
                args = (vartheta[j], zeta[k], vmec_data, i, flag)
                t = fsolve(sfl_err, vartheta[j], args=args)
                if flag != 0:
                    t = np.remainder(t+np.pi, 2*np.pi)
                theta[j] = t   # theta angle that corresponds to vartheta[j]
            R[i, :, k] = vmec_transf(
                vmec_data['rmnc'][i, :], vmec_data['xm'], vmec_data['xn'], theta, phi[k], trig='cos').flatten()
            Z[i, :, k] = vmec_transf(
                vmec_data['zmns'][i, :], vmec_data['xm'], vmec_data['xn'], theta, phi[k], trig='sin').flatten()
            if i == ns-1:
                L[:, k] = vmec_transf(
                    vmec_data['lmns'][i, :], vmec_data['xm'], vmec_data['xn'], theta, phi[k], trig='sin').flatten()
            if not vmec_data['sym']:
                R[i, :, k] += vmec_transf(vmec_data['rmns'][i, :], vmec_data['xm'],
                                          vmec_data['xn'], theta, phi[k], trig='sin').flatten()
                Z[i, :, k] += vmec_transf(vmec_data['zmnc'][i, :], vmec_data['xm'],
                                          vmec_data['xn'], theta, phi[k], trig='cos').flatten()
                if i == ns-1:
                    L[:, k] += vmec_transf(vmec_data['lmnc'][i, :], vmec_data['xm'],
                                           vmec_data['xn'], theta, phi[k], trig='cos').flatten()
        print('{}%'.format((k+1)/Ntor*100))

    cR = zernike_transform.fit(R.flatten())
    cZ = zernike_transform.fit(Z.flatten())
    cL = np.matmul(four_bdry_interp_pinv, L.flatten())
    equil = {
        'cR': cR,
        'cZ': cZ,
        'cL': cL,
        'bdryR': None,
        'bdryZ': None,
        'cP': None,
        'cI': None,
        'Psi_lcfs': vmec_data['psi'],
        'NFP': vmec_data['NFP'],
        'zern_idx': zern_idx,
        'lambda_idx': lambda_idx,
        'bdry_idx': None
    }
    return equil


# TODO: add other fields including B, rmns, zmnc, lmnc, etc
def read_vmec_output(fname):
    """Reads VMEC data from wout nc file

    Parameters
    ----------
    fname : str or path-like
        filename of VMEC output file

    Returns
    -------
    vmec_data : dict
        the VMEC data fields

    """

    file = Dataset(fname, mode='r')

    vmec_data = {
        'NFP': file.variables['nfp'][:],
        'psi': file.variables['phi'][:],  # toroidal flux is saved as 'phi'
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


def vmec_error(equil, vmec_data, Nt=8, Nz=4):
    """Computes error in SFL coordinates compared to VMEC solution

    Parameters
    ----------
    equil : dict
        dictionary of DESC equilibrium parameters
    vmec_data : dict
        dictionary of VMEC equilibrium parameters
    Nt : int
        number of poloidal angles to sample (Default value = 8)
    Nz : int
        number of toroidal angles to sample (Default value = 8)

    Returns
    -------
    err : float
        average Euclidean distance between VMEC and DESC sample points

    """
    ns = np.size(vmec_data['psi'])
    rho = np.sqrt(vmec_data['psi'])
    grid = LinearGrid(L=ns, M=Nt, N=Nz, NFP=equil.NFP, rho=rho)
    R_basis = equil.R_basis
    Z_basis = equil.Z_basis
    R_transf = Transform(grid, R_basis)
    Z_transf = Transform(grid, Z_basis)
    vartheta = np.unique(grid.nodes[:, 1])
    phi = np.unique(grid.nodes[:, 2])

    R_desc = R_transf.transform(equil.cR).reshape((ns, Nt, Nz), order='F')
    Z_desc = Z_transf.transform(equil.cZ).reshape((ns, Nt, Nz), order='F')

    print('Interpolating VMEC solution to sfl coordinates')
    R_vmec = np.zeros((ns, Nt, Nz))
    Z_vmec = np.zeros((ns, Nt, Nz))
    for k in range(Nz):         # toroidal angle
        for i in range(ns):     # flux surface
            theta = np.zeros((Nt,))
            for j in range(Nt): # poloidal angle
                f0 = sfl_err(np.array([0]), vartheta[j], phi[k], vmec_data, i)
                f2pi = sfl_err(np.array([2*np.pi]),
                               vartheta[j], phi[k], vmec_data, i)
                flag = (sign(f0) + sign(f2pi)) / 2
                args = (vartheta[j], phi[k], vmec_data, i, flag)
                t = fsolve(sfl_err, vartheta[j], args=args)
                if flag != 0:
                    t = np.remainder(t+np.pi, 2*np.pi)
                theta[j] = t   # theta angle that corresponds to vartheta[j]
            R_vmec[i, :, k] = vmec_transf(
                vmec_data['rmnc'][i, :], vmec_data['xm'], vmec_data['xn'], theta, phi[k], trig='cos').flatten()
            Z_vmec[i, :, k] = vmec_transf(
                vmec_data['zmns'][i, :], vmec_data['xm'], vmec_data['xn'], theta, phi[k], trig='sin').flatten()
            if not vmec_data['sym']:
                R_vmec[i, :, k] += vmec_transf(vmec_data['rmns'][i, :], vmec_data['xm'],
                                               vmec_data['xn'], theta, phi[k], trig='sin').flatten()
                Z_vmec[i, :, k] += vmec_transf(vmec_data['zmnc'][i, :], vmec_data['xm'],
                                               vmec_data['xn'], theta, phi[k], trig='cos').flatten()
        print('{}%'.format((k+1)/Nz*100))

    return np.mean(np.sqrt((R_vmec - R_desc)**2 + (Z_vmec - Z_desc)**2))


def sfl_err(theta, vartheta, zeta, vmec_data, s, flag=0):
    """f(theta) = vartheta - theta - lambda(theta)

    Parameters
    ----------
    theta : float
        VMEC poloidal angle
    vartheta : float
        sfl poloidal angle
    zeta : float
        VMEC/sfl toroidal angle
    vmec_data : dict
        dictionary of VMEC equilibrium parameters
    flag : int
        offsets theta to ensure f(theta) has one zero (Default value = 0)
    s :


    Returns
    -------
    err : float
        vartheta - theta - lambda

    """

    theta = theta[0] + np.pi*flag
    phi = zeta
    l = vmec_transf(vmec_data['lmns'][s, :], vmec_data['xm'],
                    vmec_data['xn'], theta, phi, trig='sin')
    if not vmec_data['sym']:
        l += vmec_transf(vmec_data['lmnc'][s, :], vmec_data['xm'],
                         vmec_data['xn'], theta, phi, trig='cos')
    return vartheta - theta - l[0][0][0]


def vmec_transf(xmna, xm, xn, theta, phi, trig='sin'):
    """Compute Fourier transform of VMEC data

    Parameters
    ----------
    xmns : 2d float array
        xmnc[:,i] are the sin coefficients at flux surface i
    xm : 1d int array
        poloidal mode numbers
    xn : 1d int array
        toroidal mode numbers
    theta : 1d float array
        poloidal angles
    phi : 1d float array
        toroidal angles
    trig : string
        type of transform, options are 'sin' or 'cos' (Default value = 'sin')
    xmna :


    Returns
    -------
    f : ndarray
        f[i,j,k] is the transformed data at flux surface i, theta[j], phi[k]

    """

    ns = np.shape(np.atleast_2d(xmna))[0]
    lt = np.size(theta)
    lp = np.size(phi)
    # Create mode x angle arrays
    mtheta = np.atleast_2d(xm).T @ np.atleast_2d(theta)
    nphi = np.atleast_2d(xn).T @ np.atleast_2d(phi)
    # Create trig arrays
    cosmt = np.cos(mtheta)
    sinmt = np.sin(mtheta)
    cosnp = np.cos(nphi)
    sinnp = np.sin(nphi)
    # Calcualte the transform
    f = np.zeros((ns, lt, lp))
    for k in range(ns):
        xmn = np.tile(np.atleast_2d(np.atleast_2d(xmna)[k, :]).T, (1, lt))
        if trig == 'sin':
            f[k, :, :] = np.tensordot(
                (xmn*sinmt).T, cosnp, axes=1) + np.tensordot((xmn*cosmt).T, sinnp, axes=1)
        elif trig == 'cos':
            f[k, :, :] = np.tensordot(
                (xmn*cosmt).T, cosnp, axes=1) - np.tensordot((xmn*sinmt).T, sinnp, axes=1)
    return f


# TODO: replace this function with vmec_transf
def vmec_interpolate(Cmn, Smn, xm, xn, theta, phi, sym=True):
    """Interpolates VMEC data on a flux surface

    Parameters
    ----------
    Cmn : ndarray
        cos(mt-np) Fourier coefficients
    Smn : ndarray
        sin(mt-np) Fourier coefficients
    xm : ndarray
        poloidal mode numbers
    xn : ndarray
        toroidal mode numbers
    theta : ndarray
        poloidal angles
    phi : ndarray
        toroidal angles
    sym : bool
        stellarator symmetry (Default value = True)

    Returns
    -------
    if sym = True
        C, S (tuple of ndarray): VMEC data interpolated at the angles (theta,phi)
        where C has cosine symmetry and S has sine symmetry
    if sym = False
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
