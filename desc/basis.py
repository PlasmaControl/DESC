import numpy as np
import functools
import warnings
import numba
from desc.backend import jnp, conditional_decorator, jit, use_jax, fori_loop
from desc.backend import issorted, isalmostequal, sign, TextColors
from desc.backend import polyder_vec, polyval_vec, flatten_list


class Basis():
    """
    """

    def __init__(self) -> None:
        pass

    def get_modes(self):
        pass

    def sort_modes(self) -> None:
        pass

    def change_resolution(self) -> None:
        pass


class FourierZernikeBasis(Basis):
    """
    """

    def __init__(self, M, N, delta_lm=-1, indexing='fringe') -> None:
        """Initializes a FourierZernikeBasis

        Returns
        -------
        None.

        """
        self.__M = M
        self.__N = N
        self.__delta_lm = delta_lm
        self.__indexing = indexing

        self.__modes = self.get_modes(M=self.__M, N=self.__N, 
                          delta_lm=self.__delta_lm, indexing=self.__indexing)
        self.sort_nodes()

    def get_modes(M, N, delta_lm=-1, indexing='fringe'):
        """Gets mode numbers for Fourier-Zernike basis functions

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
            Indexing method, one of the following options: 
            ('fourier','ansi','frige','chevron','house').
            ``'fourier'``: Double Fourier series with no radial basis (l=0 for
            all basis functions). This is a 2D subset of the 3D Fourier-Zernike
            basis for use on a single flux surface.
            Total number of modes = (2*M+1)*(2*N+1)
            
            For delta_lm=0, all methods are equivalent and 
            give a "chevron" shaped basis (only the outer edge of the zernike pyramid 
            of width M).
            For delta_lm>0, the indexing scheme defines how the pyramid is filled in:
            ``'ansi'``: ANSI indexing fills in the pyramid with triangles of
            decreasing size, ending in a triagle shape. The maximum delta_lm
            is M, at which point the traditional ANSI indexing is recovered.
            Gives a single mode at maximum m, and multiple modes at maximum l,
            from m=0 to m=l.
            Total number of modes = (M-(delta_lm//2)+1)*((delta_lm//2)+1)
            ``'fringe'``: Fringe indexing fills in the pyramid with chevrons of
            decreasing size, ending in a diamond shape. The maximum delta_lm
            is 2*M, for which the traditional "Fringe/ U of Arizona" indexing
            is recovered. Gives a single mode at maximum m and a single mode
            at maximum l and m=0.
            Total number of modes = (M+1)*(M+2)/2 - (M-delta_lm//2+1)*(M-delta_lm//2)/2
            ``'chevron'``: Beginning from the initial chevron of width M, increasing
            delta_lm adds additional chevrons of the same width. Similar to
            "house" but with fewer modes with high l but low m.
            Total number of modes = (M+1)*(2*(delta//2)+1)
            ``'house'``: Fills in the pyramid row by row, with a maximum horizontal
            width of M and a maximum radial resolution of delta_lm. For
            delta_lm = M, it is equivalent to ANSI, while for delta_lm > M
            it takes on a "house" like shape. Gives multiple modes at maximum
            m and maximum l.
            (Default value = 'fourier')

        Returns
        -------
        modes : ndarray of int, shape(Nmodes,3)
            array of mode numbers [l,m,n]
            each row is one basis function with modes (l,m,n)

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
        return np.hstack([pol, tor])

    def sort_modes(self) -> None:
        """Sorts modes for use with FFT

        Returns
        -------
        None

        """
        sort_idx = np.lexsort((self.__modes[:, 1], self.__modes[:, 0], self.__modes[:, 2]))
        self.__modes = self.__modes[sort_idx]

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

    # use numba here because the array size depends on the input values, which
    # jax can't handle
    @numba.jit(forceobj=True)
    def jacobi_coeffs(l, m):
        """Computes coefficients for Jacobi polynomials used in Zernike basis

        Parameters
        ----------
        l : ndarray of int
            radial mode numbers
        m : ndarray of int
            azimuthal mode numbers

        Returns
        -------
        jacobi_coeffs : ndarray
            matrix of polynomial coefficients in order of descending powers. 
            Each row contains the coeffs for a given l,m.

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
    def jacobi(rho, l, m, dr=0):
        """Jacobi polynomial used for radial basis function

        Parameters
        ----------
        rho : ndarray, shape(N,)
            radial coordiantes to evaluate basis
        l : ndarray of int, shape(K,)
            radial mode number(s)
        m : ndarray of int, shape(K,)
            azimuthal mode number(s)
        dr : int
            order of derivative (Default = 0)

        Returns
        -------
        y : ndarray, shape(N,K)
            basis function evaluated at specified points

        """
        coeffs = polyder_vec(jacobi_coeffs(l, m), dr)
        y = polyval_vec(coeffs, rho).T
        return y

    @conditional_decorator(functools.partial(jit, static_argnums=(1, 2)), use_jax)
    def fourier(theta, m, NFP=1, dt=0):
        """Fourier series used for poloidal and toroidal basis functions

        Parameters
        ----------
        theta : ndarray, shape(N,)
            poloidal/toroidal coordinates to evaluate basis
        m : ndarray of int, shape(K,)
            poloidal/toroidal mode number(s)
        NFP : int
            number of field periods (Default = 1)
        dt : int
            order of derivative (Default = 0)

        Returns
        -------
        y : ndarray, shape(N,K)
            basis functions evaluated at specified points

        """
        theta = jnp.atleast_1d(theta)[:, jnp.newaxis]
        m = jnp.atleast_1d(m)[jnp.newaxis]
        m_pos = (m >= 0)
        m_neg = (m < 0)
        m_abs = jnp.abs(m)
        der = (1j*m_abs*NFP)**dt
        exp = der*jnp.exp(1j*m_abs*NFP*theta)
        y = m_pos*jnp.real(exp) + m_neg*jnp.imag(exp)
        return y

    def zernike(rho, theta, l, m, dr=0, dt=0):
        """2D Zernike basis function

        Parameters
        ----------
        rho : ndarray, shape(N,)
            radial coordinates
        theta : ndarray, shape(N,)
            poloidal coordinates
        l : ndarray of int, shape(K,)
            radial mode number
        m : ndarray of int, shape(K,)
            poloidal mode number
        dr : int
            order of radial derivative (Default = 0)
        dv : int
            order of poloidal derivative (Default = 0)

        Returns
        -------
        y : ndarray, shape(N,K)
            basis function evaluated at specified points

        """
        radial = jacobi(rho, l, m, dr=dr)
        poloidal = fourier(theta, m, dt=dt)
        return radial*poloidal

    def fourzern(rho, theta, zeta, l, m, n, NFP=1, dr=0, dt=0, dz=0):
        """3D Fourier-Zernike basis function

        Parameters
        ----------
        rho : ndarray, shape(N,)
            radial coordinates
        theta : ndarray, shape(N,)
            poloidal coordinates
        zeta : ndarray, shape(N,)
            toroidal coordinates
        l : ndarray of int, shape(K,)
            radial mode number
        m : ndarray of int, shape(K,)
            poloidal mode number
        n : ndarray of int, shape(K,)
            toroidal mode number
        NFP : int
            number of field periods (Default = 1)
        dr : int
            order of radial derivative (Default = 0)
        dv : int
            order of poloidal derivative (Default = 0)
        dz : int
            order of toroidal derivative (Default = 0)

        Returns
        -------
        y : ndarray, shape(N,K)
            basis function evaluated at specified points

        """
        zern = zernike(rho, theta, l, m, dr=dr, dt=dt)
        four = fourier(zeta, n, NFP=NFP, dt=dz)
        return zern*four
