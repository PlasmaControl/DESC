import math
import numpy as np
from netCDF4 import Dataset, stringtochar
from scipy.linalg import null_space

from desc.backend import put
from desc.utils import Tristate, sign
from desc.grid import LinearGrid
from desc.basis import FourierSeries, FourierZernikeBasis, jacobi
from desc.transform import Transform
from desc.configuration import Configuration
from desc.boundary_conditions import get_lcfs_bc_matrices


class VMECIO():
    """Performs input from VMEC netCDF files to DESC Configurations and vice-versa."""

    @classmethod
    def load(cls, path:str, L:int=-1, M:int=-1, N:int=-1,
             index:str='ansi') -> Configuration:
        """Loads a VMEC netCDF file as a Configuration.

        Parameters
        ----------
        path : str
            File path of input data.
        L : int, optional
            Radial resolution. Default determined by index.
        M : int, optional
            Poloidal resolution. Default = MPOL-1 from VMEC solution.
        N : int, optional
            Toroidal resolution. Default = NTOR from VMEC solution.
        index : str, optional
            Type of Zernike indexing scheme to use. (Default = 'ansi')

        Returns
        -------
        eq: Configuration
            Configuration that resembles the VMEC data.

        """
        file = Dataset(path, mode='r')
        inputs = {}

        # parameters
        inputs['Psi'] = file.variables['phi'][-1]
        inputs['NFP'] = int(file.variables['nfp'][0])
        inputs['M'] = M if M >  0 else int(file.variables['mpol'][0] - 1)
        inputs['N'] = N if N >= 0 else int(file.variables['ntor'][0])
        inputs['index'] = index
        default_L = {'ansi': inputs['M'],
                     'fringe': 2*inputs['M'],
                     'chevron': inputs['M'],
                     'house': 2*inputs['M']}
        inputs['L'] = L if L >= 0 else default_L[inputs['index']]

        # data
        xm = file.variables['xm'][:].filled()
        xn = file.variables['xn'][:].filled() / inputs['NFP']
        rmnc = file.variables['rmnc'][:].filled()
        zmns = file.variables['zmns'][:].filled()
        lmns = file.variables['lmns'][:].filled()
        try:
            rmns = file.variables['rmns'][:].filled()
            zmnc = file.variables['zmnc'][:].filled()
            lmnc = file.variables['lmnc'][:].filled()
            inputs['sym'] = False
        except:
            rmns = np.zeros_like(rmnc)
            zmnc = np.zeros_like(zmns)
            lmnc = np.zeros_like(lmns)
            inputs['sym'] = True

        # basis symmetry
        if inputs['sym']:
            R_sym = Tristate(True)
            Z_sym = Tristate(False)
        else:
            R_sym = Tristate(None)
            Z_sym = Tristate(None)

        # collocation grid
        surfs = file.dimensions['radius'].size
        rho = np.sqrt(np.linspace(0, 1, surfs))
        grid = LinearGrid(
            M=2*math.ceil(1.5*inputs['M'])+1, N=2*math.ceil(1.5*inputs['N'])+1,
            NFP=inputs['NFP'], sym=inputs['sym'], rho=rho)

        # profiles
        preset = file.dimensions['preset'].size
        p0 = file.variables['presf'][0] / file.variables['am'][0]
        inputs['profiles'] = np.zeros((preset,3))
        inputs['profiles'][:, 0] = np.arange(0, 2*preset, 2)
        inputs['profiles'][:, 1] = file.variables['am'][:]*p0
        inputs['profiles'][:, 2] = file.variables['ai'][:]

        file.close

        # boundary
        m, n, R1_mn = cls._ptolemy_identity(xm, xn, s=rmns[-1, :], c=rmnc[-1, :])
        m, n, Z1_mn = cls._ptolemy_identity(xm, xn, s=zmns[-1, :], c=zmnc[-1, :])
        inputs['boundary'] = np.vstack((m, n, R1_mn, Z1_mn)).T

        # axis
        m, n, R0_mn = cls._ptolemy_identity(xm, xn, s=rmns[0, :], c=rmnc[0, :])
        m, n, Z0_mn = cls._ptolemy_identity(xm, xn, s=zmns[0, :], c=zmnc[0, :])
        R0_basis = FourierSeries(N=inputs['N'], NFP=inputs['NFP'], sym=R_sym)
        Z0_basis = FourierSeries(N=inputs['N'], NFP=inputs['NFP'], sym=Z_sym)
        inputs['R0_n'] = np.zeros((R0_basis.num_modes,))
        inputs['Z0_n'] = np.zeros((Z0_basis.num_modes,))
        for m, n, R0, Z0 in np.vstack((m, n, R0_mn, Z0_mn)).T:
            idx_R = np.where(np.logical_and(R0_basis.modes[:, 1] == m,
                                            R0_basis.modes[:, 2] == n))[0]
            idx_Z = np.where(np.logical_and(Z0_basis.modes[:, 1] == m,
                                            Z0_basis.modes[:, 2] == n))[0]
            inputs['R0_n'] = put(inputs['R0_n'], idx_R, R0)
            inputs['Z0_n'] = put(inputs['Z0_n'], idx_Z, Z0)

        # lambda
        m, n, l_mn = cls._ptolemy_identity(xm, xn, s=lmns, c=lmnc)
        inputs['l_lmn'], l_basis = cls._fourier_to_zernike(m, n, l_mn,
            NFP=inputs['NFP'], L=inputs['L'], M=inputs['M'], N=inputs['N'],
            index=inputs['index'])

        # evaluate flux surface shapes
        m, n, R_mn = cls._ptolemy_identity(xm, xn, s=rmns, c=rmnc)
        m, n, Z_mn = cls._ptolemy_identity(xm, xn, s=zmns, c=zmnc)
        R_lmn, R_basis = cls._fourier_to_zernike(m, n, R_mn,
            NFP=inputs['NFP'], L=inputs['L'], M=inputs['M'], N=inputs['N'],
            index=inputs['index'])
        Z_lmn, Z_basis = cls._fourier_to_zernike(m, n, Z_mn,
            NFP=inputs['NFP'], L=inputs['L'], M=inputs['M'], N=inputs['N'],
            index=inputs['index'])
        R0_transform = Transform(grid=grid, basis=R0_basis)
        Z0_transform = Transform(grid=grid, basis=Z0_basis)
        R_transform = Transform(grid=grid, basis=R_basis)
        Z_transform = Transform(grid=grid, basis=Z_basis)
        R0 = R0_transform.transform(inputs['R0_n'])
        Z0 = Z0_transform.transform(inputs['Z0_n'])
        R = R_transform.transform(R_lmn)
        Z = Z_transform.transform(Z_lmn)

        # r
        r = np.zeros_like(R)
        theta = grid.nodes[:, 1]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        cos_idx = np.where(np.abs(cos_t) >= 1/np.sqrt(2))[0]
        sin_idx = np.where(np.abs(sin_t) >= 1/np.sqrt(2))[0]
        r = put(r, cos_idx, ((R-R0)/cos_t)[cos_idx])
        r = put(r, sin_idx, ((Z-Z0)/sin_t)[sin_idx])
        r_basis = FourierZernikeBasis(
            L=inputs['L'], M=inputs['M'], N=inputs['N'], NFP=inputs['NFP'],
            sym=R_sym, index=inputs['index'])
        r_transform = Transform(grid=grid, basis=r_basis)
        inputs['r_lmn'] = r_transform.fit(r)

        # initialize Configuration
        eq = Configuration(inputs=inputs)

        # enforce LCFS BC on r
        A, b = get_lcfs_bc_matrices(eq.R0_basis, eq.Z0_basis, eq.r_basis,
            eq.l_basis, eq.R1_basis, eq.Z1_basis, eq.R1_mn, eq.Z1_mn)
        A_r = A[:, eq.R0_basis.num_modes+eq.Z0_basis.num_modes:-eq.l_basis.num_modes]
        Z = null_space(A_r)
        r0_lmn = np.linalg.lstsq(A_r, b, rcond=None)[0].flatten()
        eq.r_lmn = r0_lmn + Z.dot(Z.T.dot(eq.r_lmn - r0_lmn))

        return eq

    @classmethod
    def save(cls, eq:Configuration, path:str, surfs:int=128) -> None:
        """Saves a Configuration as a netCDF file in the VMEC format.

        Parameters
        ----------
        eq : Configuration
            Configuration to save.
        path : str
            File path of output data.
        surfs: int (Default = 128)
            Number of flux surfaces to interpolate at.

        Returns
        -------
        None

        """
        file = Dataset(path, mode='w')

        """ VMEC netCDF file is generated in VMEC2000/Sources/Input_Output/wrout.f
            see lines 300+ for full list of included variables
        """

        Psi = eq.Psi
        NFP = eq.NFP
        M = eq.M
        N = eq.N
        p_l = eq.p_l
        i_l = eq.i_l

        s_full = np.linspace(0, 1, surfs)
        s_half = s_full[0:-1] + 0.5/(surfs-1)
        r_full = np.sqrt(s_full)
        r_half = np.sqrt(s_half)
        full_grid = LinearGrid(rho=r_full)
        half_grid = LinearGrid(rho=r_half)

        p_transform_full = Transform(full_grid, eq.p_basis)
        p_transform_half = Transform(half_grid, eq.p_basis)
        i_transform_full = Transform(full_grid, eq.i_basis)
        i_transform_half = Transform(half_grid, eq.i_basis)

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

        lfreeb = file.createVariable('lfreeb', np.int32, ('dim_00001',))
        lfreeb.long_name = 'free boundary logical (0 = fixed boundary)'
        lfreeb[:] = 0

        lasym = file.createVariable('lasym', np.int32, ('dim_00001',))
        lasym.long_name = 'asymmetry logical (0 = stellarator symmetry)'
        lasym[:] = int(not eq.sym)

        nfp = file.createVariable('nfp', np.int32, ('dim_00001',))
        nfp.long_name = 'number of field periods'
        nfp[:] = NFP

        ns = file.createVariable('ns', np.int32, ('dim_00001',))
        ns.long_name = 'number of flux surfaces'
        ns[:] = surfs

        mpol = file.createVariable('mpol', np.int32, ('dim_00001',))
        mpol.long_name = 'number of poloidal Fourier modes'
        mpol[:] = M+1

        ntor = file.createVariable('ntor', np.int32, ('dim_00001',))
        ntor.long_name = 'number of positive toroidal Fourier modes'
        ntor[:] = N

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

        signgs = file.createVariable('signgs', np.float64, ('dim_00001',))
        signgs.long_name = 'sign of coordinate system jacobian'
        signgs[:] = 1   # TODO: don't hard-code this

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

    def _ptolemy_identity(m, n, s, c):
        """Converts from a double Fourier series of the form:
            s*sin(m*theta-n*phi) + c*cos(m*theta-n*phi)
        to the form:
            ss*sin(m*theta)*sin(n*phi) + sc*sin(m*theta)*cos(n*phi) +
            cs*cos(m*theta)*sin(n*phi) + cc*cos(m*theta)*cos(n*phi)
        using Ptolemy's sum and difference formulas.

        Parameters
        ----------
        m : ndarray
            Poloidal mode numbers.
        n : ndarray
            Toroidal mode numbers.
        s : ndarray, optional
            Coefficients of sin(m*theta-n*phi) terms.
            Each row is a separate flux surface.
        c : ndarray, optional
            Coefficients of cos(m*theta-n*phi) terms.
            Each row is a separate flux surface.

        Returns
        -------
        m_new : ndarray, shape(num_modes,)
            Poloidal mode numbers of the double Fourier basis.
        n_new : ndarray, shape(num_modes,)
            Toroidal mode numbers of the double Fourier basis.
        x_mn : ndarray, shape(num_modes,)
            Spectral coefficients in the double Fourier basis.

        """
        s = np.atleast_2d(s)
        c = np.atleast_2d(c)

        M = int(np.max(np.abs(m)))
        N = int(np.max(np.abs(n)))

        mn_new = np.array([[m-M, n-N, 0] for m in range(2*M+1) for n in range(2*N+1)])
        m_new = mn_new[:, 0]
        n_new = mn_new[:, 1]
        x_mn = np.zeros((s.shape[0], m_new.size))

        for i in range(len(m)):
            sin_mn_1 = np.where(np.logical_and(m_new == -np.abs(m[i]),
                                               n_new ==  np.abs(n[i])))[0][0]
            sin_mn_2 = np.where(np.logical_and(m_new ==  np.abs(m[i]),
                                               n_new == -np.abs(n[i])))[0][0]
            cos_mn_1 = np.where(np.logical_and(m_new ==  np.abs(m[i]),
                                               n_new ==  np.abs(n[i])))[0][0]
            cos_mn_2 = np.where(np.logical_and(m_new == -np.abs(m[i]),
                                               n_new == -np.abs(n[i])))[0][0]

            if np.sign(m[i]) != 0:
                x_mn[:, sin_mn_1] += s[:, i]
            x_mn[:, cos_mn_1] += c[:, i]
            if np.sign(n[i]) > 0:
                x_mn[:, sin_mn_2] -= s[:, i]
                if np.sign(m[i]) != 0:
                    x_mn[:, cos_mn_2] += c[:, i]
            elif np.sign(n[i]) < 0:
                x_mn[:, sin_mn_2] += s[:, i]
                if np.sign(m[i]) != 0:
                    x_mn[:, cos_mn_2] -= c[:, i]

        return m_new, n_new, x_mn

    def _fourier_to_zernike(m, n, x_mn, NFP:int=1, L:int=-1, M:int=-1, N:int=-1,
                            index:str='ansi'):
        """Converts from a double Fourier series at each flux surface to a
        Fourier-Zernike basis.

        Parameters
        ----------
        m : ndarray, shape(num_modes,)
            Poloidal mode numbers.
        n : ndarray, shape(num_modes,)
            Toroidal mode numbers.
        x_mn : ndarray, shape(num_modes,)
            Spectral coefficients in the double Fourier basis.
            Each row is a separate flux surface, increasing from the magnetic
            axis to the boundary.
        NFP : int, optional
            Number of toroidal field periods.
        L : int, optional
            Radial resolution. Default determined by index.
        M : int, optional
            Poloidal resolution. Default = MPOL-1 from VMEC solution.
        N : int, optional
            Toroidal resolution. Default = NTOR from VMEC solution.
        index : str, optional
            Type of Zernike indexing scheme to use. (Default = 'ansi')

        Returns
        -------
        x_lmn : ndarray, shape(num_modes,)
            Fourier-Zernike spectral coefficients.
        basis : FourierZernikeBasis
            Basis set for x_lmn

        """
        M = M if M >  0 else int(np.max(np.abs(m)))
        N = N if N >= 0 else int(np.max(np.abs(n)))

        if not np.any(x_mn[:, np.where(sign(m)*sign(n) == -1)[0]]):
            sym = Tristate(True)
        elif not np.any(x_mn[:, np.where(sign(m)*sign(n) == 1)[0]]):
            sym = Tristate(False)
        else:
            sym = Tristate(None)

        basis = FourierZernikeBasis(L=L, M=M, N=N, NFP=NFP, sym=sym, index=index)
        x_lmn = np.zeros((basis.num_modes,))

        surfs = x_mn.shape[0]
        rho = np.sqrt(np.linspace(0, 1, surfs))

        for i in range(len(m)):
            idx = np.where(np.logical_and(basis.modes[:, 1] == m[i],
                                          basis.modes[:, 2] == n[i]))[0]
            if len(idx):
                A = jacobi(rho, basis.modes[idx, 0], basis.modes[idx, 1])
                c = np.linalg.lstsq(A, x_mn[:, i], rcond=None)[0]
                x_lmn = put(x_lmn, idx, c)

        return x_lmn, basis



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
