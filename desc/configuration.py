import numpy as np

from desc.backend import jnp, put, opsindex, cross, dot, TextColors, sign
from desc.init_guess import get_initial_guess_scale_bdry
from desc.boundary_conditions import format_bdry


# TODO: can probably remove this function if we enforce symmetry in Basis
def symmetry_matrix(RZ_modes, lambda_modes, sym:bool):
        """Compute stellarator symmetry linear constraint matrix

        Parameters
        ----------
        RZ_modes : ndarray, shape(Nz_coeffs,3)
            array of spectral basis modes (l,m,n) for R and Z
        lambda_modes : ndarray, shape(Nl_coeffs,2)
            array of spectral basis modes (l,m,n) for lambda
        sym : bool
            True for stellarator symmetry, False otherwise

        Returns
        -------
        sym_mat : ndarray of float
            stellarator symmetry linear constraint matrix
            x_full = sym_mat * x_sym
            x_sym = sym_mat.T * x_full

        """
        if sym:
            m_zern = RZ_modes[:, 1]
            n_zern = RZ_modes[:, 2]
            m_lambda = lambda_modes[:, 1]
            n_lambda = lambda_modes[:, 2]

            # symmetric indices of R, Z, lambda
            idx_R = sign(m_zern)*sign(n_zern) > 0
            idx_Z = sign(m_zern)*sign(n_zern) < 0
            idx_L = sign(m_lambda)*sign(n_lambda) < 0

            idx_x = np.concatenate([idx_R, idx_Z, idx_L])
            sym_mat = np.diag(idx_x, k=0).astype(int)[:, idx_x]
        else:
            sym_mat = np.eye(2*RZ_modes.shape[0] + lambda_modes.shape[0])

        return sym_mat


# TODO: can probably replace this function when Configuration interacts with Solver
def change_resolution(x_old, stell_sym, RZ_basis_old, RZ_basis_new, L_basis_old, L_basis_new):
    """

    Parameters
    ----------
    x_old : ndarray
        DESCRIPTION.
    RZ_basis_old : FourierZernikeBasis
        DESCRIPTION.
    RZ_basis_new : FourierZernikeBasis
        DESCRIPTION.
    L_basis_old : DoubleFourierSeries
        DESCRIPTION.
    L_basis_new : DoubleFourierSeries
        DESCRIPTION.

    Returns
    -------
    x_new : ndarray
        

    """
    old_modes = RZ_basis_old.modes
    new_modes = RZ_basis_new.modes

    sym_mat_old = symmetry_matrix(RZ_basis_old.modes, L_basis_old.modes, sym=stell_sym)
    cR_old, cZ_old, cL_old = unpack_state(np.matmul(sym_mat_old, x_old), RZ_basis_old.num_modes)

    cR_new = np.zeros((RZ_basis_new.num_modes,))
    cZ_new = np.zeros((RZ_basis_new.num_modes,))
    cL_new = np.zeros((L_basis_new.num_modes,))

    for i in range(RZ_basis_new.num_modes):
        idx = np.where(np.all(np.array([
                    np.array(old_modes[:, 0] == new_modes[i, 0]),
                    np.array(old_modes[:, 1] == new_modes[i, 1]),
                    np.array(old_modes[:, 2] == new_modes[i, 2])]), axis=0))[0]
        if len(idx):
            cR_new[i] = cR_old[idx[0]]
            cZ_new[i] = cZ_old[idx[0]]

    for i in range(L_basis_new.num_modes):
        idx = np.where(np.all(np.array([
                    np.array(old_modes[:, 0] == new_modes[i, 0]),
                    np.array(old_modes[:, 1] == new_modes[i, 1]),
                    np.array(old_modes[:, 2] == new_modes[i, 2])]), axis=0))[0]
        if len(idx):
            cL_new[i] = cL_old[idx[0]]

    sym_mat_new = symmetry_matrix(RZ_basis_new.modes, L_basis_new.modes, sym=stell_sym)
    x_new = np.concatenate([cR_new, cZ_new, cL_new])
    x_new = jnp.matmul(sym_mat_new.T, x_new)
    return x_new


def unpack_state(x, nRZ):
    """Unpacks the optimization state vector x into cR, cZ, cL components

    Parameters
    ----------
    x : ndarray
        vector to unpack
    nRZ : int
        number of R,Z coeffs

    Returns
    -------
    cR : ndarray
        spectral coefficients of R
    cZ : ndarray
        spectral coefficients of Z
    cL : ndarray
        spectral coefficients of lambda

    """

    cR = x[:nRZ]
    cZ = x[nRZ:2*nRZ]
    cL = x[2*nRZ:]
    return cR, cZ, cL


class Configuration():
    """Configuration constains information about a plasma state, including the 
       shapes of flux surfaces and profile inputs. It can compute additional 
       information, such as the magnetic field and plasma currents. 
    """

    # TODO: replace zern_idx & lambda_idx with Basis objects
    def __init__(self, bdry, cP, cI, Psi, NFP, zern_idx, lambda_idx, sym=False, x=None, axis=None) -> None:
        """Initializes a Configuration

        Parameters
        ----------
        bdry : ndarray of float, shape(Nbdry,4)
            array of boundary Fourier coeffs [m,n,Rcoeff, Zcoeff]
            OR
            array of real space coordinates, [theta,phi,R,Z]
        cP : ndarray
            spectral coefficients of the pressure profile (Pascals)
        cI : ndarray
            spectral coefficients of the rotational transform profile
        Psi : float
            toroidal flux within the last closed flux surface (Webers)
        NFP : int
            number of toroidal field periods
        zern_idx : ndarray of int, shape(N_coeffs,3)
            indices for spectral basis, ie an array of [l,m,n] for each spectral coefficient
        lambda_idx : ndarray of int, shape(Nmodes,2)
            poloidal and toroidal mode numbers [m,n]
        sym : bool
            True for stellarator symmetry, False otherwise
        x : ndarray of float
            state vector of independent variables: [cR, cZ, cL]. If not supplied, 
            the flux surfaces are scaled from the boundary and magnetic axis
        axis : ndarray, shape(Naxis,3)
            array of axis Fourier coeffs [n,Rcoeff, Zcoeff]

        Returns
        -------
        None

        """
        self.__bdry = bdry
        self.__cP = cP
        self.__cI = cI
        self.__Psi = Psi
        self.__NFP = NFP
        self.__zern_idx = zern_idx
        self.__lambda_idx = lambda_idx
        self.__sym = sym
        self.__sym_mat = symmetry_matrix(self.__zern_idx, self.__lambda_idx, sym=self.__sym)

        self.__bdryM, self.__bdryN, self.__bdryR, self.__bdryZ = format_bdry(
            np.max(self.__lambda_idx[:,0]), np.max(self.__lambda_idx[:,1]), self.__NFP, self.__bdry, 'spectral', 'spectral')

        if x is None:
            # TODO: change input reader to make the default axis=None
            if axis is None:
                axis = bdry[np.where(bdry[:, 0] == 0)[0], 1:]
            self.__cR, self.__cZ = get_initial_guess_scale_bdry(
                axis, bdry, 1.0, zern_idx, NFP, mode='spectral', rcond=1e-6)
            self.__cL = np.zeros(len(lambda_idx))
            self.__x = np.concatenate([self.__cR, self.__cZ, self.__cL])
            self.__x = np.matmul(self.__sym_mat, self.__x)
        else:
            self.__x = x
            try:
                self.__cR, self.__cZ, self.__cL = unpack_state(
                    np.matmul(self.__sym_mat, self.__x), len(zern_idx))
            except:
                raise ValueError(TextColors.FAIL + 
                    "State vector dimension is incompatible with other parameters" + TextColors.ENDC)

    @property
    def bdry(self):
        return self.__bdry

    @bdry.setter
    def bdry(self, bdry):
        self.__bdry = bdry
        self.__bdryM, self.__bdryN, self.__bdryR, self.__bdryZ = format_bdry(
            np.max(self.__lambda_idx[:,0]), np.max(self.__lambda_idx[:,1]), self.__NFP, self.__bdry, 'spectral', 'spectral')

    @property
    def cP(self):
        return self.__cP

    @cP.setter
    def cP(self, cP):
        self.__cP = cP

    @property
    def cI(self):
        return self.__cI

    @cI.setter
    def cI(self, cI):
        self.__cI = cI

    @property
    def Psi(self):
        return self.__Psi

    @Psi.setter
    def Psi(self, Psi):
        self.__Psi = Psi

    @property
    def NFP(self):
        return self.__NFP

    @NFP.setter
    def NFP(self, NFP):
        self.__NFP = NFP

    @property
    def sym(self):
        return self.__sym

    @sym.setter
    def sym(self, sym):
        self.__sym = sym
        self.__sym_mat = symmetry_matrix(self.__zern_idx, self.__lambda_idx, sym=self.__sym)
        self.__x = np.matmul(self.__sym_mat, self.__x)

    def attributes(self):
        return (self.x, self.bdryR, self.bdryZ, self.cP, self.cI, self.Psi)

    def compute_coordinate_derivatives(self):
        pass
        # return compute_coordinate_derivatives(self.cR, self.cZ, zernike_transform, zeta_ratio=1.0, mode='equil')

    def compute_covariant_basis(self):
        pass
        # return compute_covariant_basis(coord_der, zernike_transform, mode='equil')

    def compute_contravariant_basis(self):
        pass
        # return compute_contravariant_basis(coord_der, cov_basis, jacobian, zernike_transform)

    def compute_jacobian(self):
        pass
        # return compute_jacobian(coord_der, cov_basis, zernike_transform, mode='equil')

    def compute_magnetic_field(self):
        pass
        # return compute_magnetic_field(cov_basis, jacobian, cI, Psi_lcfs, zernike_transform, mode='equil')

    def compute_plasma_current(self):
        pass
        # return compute_plasma_current(coord_der, cov_basis, jacobian, magnetic_field, cI, Psi_lcfs, zernike_transform)

    def compute_magnetic_field_magnitude(self):
        pass
        # return def compute_magnetic_field_magnitude(cov_basis, magnetic_field, cI, zernike_transform)

    def compute_force_magnitude(self):
        pass
        # return def compute_force_magnitude(coord_der, cov_basis, con_basis, jacobian, magnetic_field, plasma_current, cP, cI, Psi_lcfs, zernike_transform):


class Equilibrium(Configuration):
    """Equilibrium is a decorator design pattern on top of Configuration. 
       It adds information about how the equilibrium configuration was solved. 
    """

    def __init__(self, bdry, cP, cI, Psi, NFP, zern_idx, lambda_idx, sym=False, x=None, axis=None, objective=None, optimizer=None) -> None:
        super().__init__(self, bdry, cP, cI, Psi, NFP, zern_idx, lambda_idx, sym=False, x=None, axis=None)
        self.__objective = objective
        self.__optimizer = optimizer
        self.solved = False

    def optimize(self):
        pass

    @property
    def objective(self):
        return self.__objective

    @objective.setter
    def objective(self, objective):
        self.__objective = objective
        self.solved = False

    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self.__optimizer = optimizer
        self.solved = False


# XXX: Does this inherit from Equilibrium?
class EquiliriaFamily(Equilibrium):
    """EquilibriaFamily stores a list of Equilibria. Its default behavior acts 
       like the last Equilibrium in the list. 
    """

    def __init__(self, equilibria, solver=None) -> None:
        self.__equilibria = equilibria
        self.__solver = solver

    @property
    def equilibria(self):
        return self.__equilibria

    @equilibria.setter
    def equilibria(self, equilibria):
        self.__equilibria = equilibria

    @property
    def solver(self):
        return self.__solver

    @solver.setter
    def solver(self, solver):
        self.__solver = solver


# TODO: overwrite all Equilibrium methods and default to self.__equilibria[-1]

# TODO: eliminate unnecessary derivatives for speedup (eg. R_rrr)
def compute_coordinate_derivatives(cR, cZ, RZ_transform, zeta_ratio=1.0):
    """Converts from spectral to real space and evaluates derivatives of R,Z wrt to SFL coords

    Parameters
    ----------
    cR : ndarray
        spectral coefficients of R
    cZ : ndarray
        spectral coefficients of Z
    RZ_transform : Transform
        object with transform method to go from spectral to physical space with derivatives
    zeta_ratio : float
        scale factor for zeta derivatives. Setting to zero
        effectively solves for individual tokamak solutions at each toroidal plane,
        setting to 1 solves for a stellarator. (Default value = 1.0)

    Returns
    -------
    coord_der : dict
        dictionary of ndarray, shape(N_nodes,) of coordinate derivatives evaluated at node locations
        keys are of the form 'X_y' meaning the derivative of X wrt to y

    """
    # notation: X_y means derivative of X wrt y
    coord_der = {}
    coord_der['R'] = RZ_transform.transform(cR, 0, 0, 0)
    coord_der['Z'] = RZ_transform.transform(cZ, 0, 0, 0)
    coord_der['0'] = jnp.zeros_like(coord_der['R'])

    coord_der['R_r'] = RZ_transform.transform(cR, 1, 0, 0)
    coord_der['Z_r'] = RZ_transform.transform(cZ, 1, 0, 0)
    coord_der['R_v'] = RZ_transform.transform(cR, 0, 1, 0)
    coord_der['Z_v'] = RZ_transform.transform(cZ, 0, 1, 0)
    coord_der['R_z'] = RZ_transform.transform(cR, 0, 0, 1) * zeta_ratio
    coord_der['Z_z'] = RZ_transform.transform(cZ, 0, 0, 1) * zeta_ratio

    coord_der['R_rr'] = RZ_transform.transform(cR, 2, 0, 0)
    coord_der['Z_rr'] = RZ_transform.transform(cZ, 2, 0, 0)
    coord_der['R_rv'] = RZ_transform.transform(cR, 1, 1, 0)
    coord_der['Z_rv'] = RZ_transform.transform(cZ, 1, 1, 0)
    coord_der['R_rz'] = RZ_transform.transform(cR, 1, 0, 1) * zeta_ratio
    coord_der['Z_rz'] = RZ_transform.transform(cZ, 1, 0, 1) * zeta_ratio
    coord_der['R_vv'] = RZ_transform.transform(cR, 0, 2, 0)
    coord_der['Z_vv'] = RZ_transform.transform(cZ, 0, 2, 0)
    coord_der['R_vz'] = RZ_transform.transform(cR, 0, 1, 1) * zeta_ratio
    coord_der['Z_vz'] = RZ_transform.transform(cZ, 0, 1, 1) * zeta_ratio
    coord_der['R_zz'] = RZ_transform.transform(cR, 0, 0, 2) * zeta_ratio
    coord_der['Z_zz'] = RZ_transform.transform(cZ, 0, 0, 2) * zeta_ratio

    # axis or QS terms
    if RZ_transform.grid.axis.size > 0 or RZ_transform.derivs == 'qs':
        coord_der['R_rrr'] = RZ_transform.transform(cR, 3, 0, 0)
        coord_der['Z_rrr'] = RZ_transform.transform(cZ, 3, 0, 0)
        coord_der['R_rrv'] = RZ_transform.transform(cR, 2, 1, 0)
        coord_der['Z_rrv'] = RZ_transform.transform(cZ, 2, 1, 0)
        coord_der['R_rrz'] = RZ_transform.transform(cR, 2, 0, 1) * zeta_ratio
        coord_der['Z_rrz'] = RZ_transform.transform(cZ, 2, 0, 1) * zeta_ratio
        coord_der['R_rvv'] = RZ_transform.transform(cR, 1, 2, 0)
        coord_der['Z_rvv'] = RZ_transform.transform(cZ, 1, 2, 0)
        coord_der['R_rvz'] = RZ_transform.transform(cR, 1, 1, 1) * zeta_ratio
        coord_der['Z_rvz'] = RZ_transform.transform(cZ, 1, 1, 1) * zeta_ratio
        coord_der['R_rzz'] = RZ_transform.transform(cR, 1, 0, 2) * zeta_ratio
        coord_der['Z_rzz'] = RZ_transform.transform(cZ, 1, 0, 2) * zeta_ratio
        coord_der['R_vvv'] = RZ_transform.transform(cR, 0, 3, 0)
        coord_der['Z_vvv'] = RZ_transform.transform(cZ, 0, 3, 0)
        coord_der['R_vvz'] = RZ_transform.transform(cR, 0, 2, 1) * zeta_ratio
        coord_der['Z_vvz'] = RZ_transform.transform(cZ, 0, 2, 1) * zeta_ratio
        coord_der['R_vzz'] = RZ_transform.transform(cR, 0, 1, 2) * zeta_ratio
        coord_der['Z_vzz'] = RZ_transform.transform(cZ, 0, 1, 2) * zeta_ratio
        coord_der['R_zzz'] = RZ_transform.transform(cR, 0, 0, 3) * zeta_ratio
        coord_der['Z_zzz'] = RZ_transform.transform(cZ, 0, 0, 3) * zeta_ratio

        coord_der['R_rrvv'] = RZ_transform.transform(cR, 2, 2, 0)
        coord_der['Z_rrvv'] = RZ_transform.transform(cZ, 2, 2, 0)

    return coord_der


def compute_covariant_basis(coord_der, RZ_transform):
    """Computes covariant basis vectors at grid points

    Parameters
    ----------
    coord_der : dict
        dictionary of ndarray containing the coordinate
        derivatives at each node, such as computed by ``compute_coordinate_derivatives``
    RZ_transform : Transform
        object with transform method to go from spectral to physical space with derivatives
    mode : str
        one of 'equil' or 'qs'. Whether to compute field terms for equilibrium or quasisymmetry optimization (Default value = 'equil')

    Returns
    -------
    cov_basis : dict
        dictionary of ndarray containing covariant basis
        vectors and derivatives at each node. Keys are of the form 'e_x_y',
        meaning the unit vector in the x direction, differentiated wrt to y.

    """
    # notation: subscript word is direction of unit vector, subscript letters denote partial derivatives
    # eg, e_rho_v is the v derivative of the covariant basis vector in the rho direction
    cov_basis = {}
    cov_basis['e_rho'] = jnp.array(
        [coord_der['R_r'],  coord_der['0'],   coord_der['Z_r']])
    cov_basis['e_theta'] = jnp.array(
        [coord_der['R_v'],  coord_der['0'],   coord_der['Z_v']])
    cov_basis['e_zeta'] = jnp.array(
        [coord_der['R_z'],  coord_der['R'],   coord_der['Z_z']])

    cov_basis['e_rho_r'] = jnp.array(
        [coord_der['R_rr'], coord_der['0'],   coord_der['Z_rr']])
    cov_basis['e_rho_v'] = jnp.array(
        [coord_der['R_rv'], coord_der['0'],   coord_der['Z_rv']])
    cov_basis['e_rho_z'] = jnp.array(
        [coord_der['R_rz'], coord_der['0'],   coord_der['Z_rz']])

    cov_basis['e_theta_r'] = jnp.array(
        [coord_der['R_rv'], coord_der['0'],   coord_der['Z_rv']])
    cov_basis['e_theta_v'] = jnp.array(
        [coord_der['R_vv'], coord_der['0'],   coord_der['Z_vv']])
    cov_basis['e_theta_z'] = jnp.array(
        [coord_der['R_vz'], coord_der['0'],   coord_der['Z_vz']])

    cov_basis['e_zeta_r'] = jnp.array(
        [coord_der['R_rz'], coord_der['R_r'], coord_der['Z_rz']])
    cov_basis['e_zeta_v'] = jnp.array(
        [coord_der['R_vz'], coord_der['R_v'], coord_der['Z_vz']])
    cov_basis['e_zeta_z'] = jnp.array(
        [coord_der['R_zz'], coord_der['R_z'], coord_der['Z_zz']])

    # axis or QS terms
    if RZ_transform.grid.axis.size > 0 or RZ_transform.derivs == 'qs':
        cov_basis['e_rho_rr'] = jnp.array(
            [coord_der['R_rrr'], coord_der['0'],   coord_der['Z_rrr']])
        cov_basis['e_rho_rv'] = jnp.array(
            [coord_der['R_rrv'], coord_der['0'],   coord_der['Z_rrv']])
        cov_basis['e_rho_rz'] = jnp.array(
            [coord_der['R_rrz'], coord_der['0'],   coord_der['Z_rrz']])
        cov_basis['e_rho_vv'] = jnp.array(
            [coord_der['R_rvv'], coord_der['0'],   coord_der['Z_rvv']])
        cov_basis['e_rho_vz'] = jnp.array(
            [coord_der['R_rvz'], coord_der['0'],   coord_der['Z_rvz']])
        cov_basis['e_rho_zz'] = jnp.array(
            [coord_der['R_rzz'], coord_der['0'],   coord_der['Z_rzz']])

        cov_basis['e_theta_rr'] = jnp.array(
            [coord_der['R_rrv'], coord_der['0'],   coord_der['Z_rrv']])
        cov_basis['e_theta_rv'] = jnp.array(
            [coord_der['R_rvv'], coord_der['0'],   coord_der['Z_rvv']])
        cov_basis['e_theta_rz'] = jnp.array(
            [coord_der['R_rvz'], coord_der['0'],   coord_der['Z_rvz']])
        cov_basis['e_theta_vv'] = jnp.array(
            [coord_der['R_vvv'], coord_der['0'],   coord_der['Z_vvv']])
        cov_basis['e_theta_vz'] = jnp.array(
            [coord_der['R_vvz'], coord_der['0'],   coord_der['Z_vvz']])
        cov_basis['e_theta_zz'] = jnp.array(
            [coord_der['R_vzz'], coord_der['0'],   coord_der['Z_vzz']])

        cov_basis['e_zeta_rr'] = jnp.array(
            [coord_der['R_rrz'], coord_der['R_rr'], coord_der['Z_rrz']])
        cov_basis['e_zeta_rv'] = jnp.array(
            [coord_der['R_rvz'], coord_der['R_rv'], coord_der['Z_rvz']])
        cov_basis['e_zeta_rz'] = jnp.array(
            [coord_der['R_rzz'], coord_der['R_rz'], coord_der['Z_rzz']])
        cov_basis['e_zeta_vv'] = jnp.array(
            [coord_der['R_vvz'], coord_der['R_vv'], coord_der['Z_vvz']])
        cov_basis['e_zeta_vz'] = jnp.array(
            [coord_der['R_vzz'], coord_der['R_vz'], coord_der['Z_vzz']])
        cov_basis['e_zeta_zz'] = jnp.array(
            [coord_der['R_zzz'], coord_der['R_zz'], coord_der['Z_zzz']])

    return cov_basis


def compute_contravariant_basis(coord_der, cov_basis, jacobian, RZ_transform):
    """Computes contravariant basis vectors and jacobian elements

    Parameters
    ----------
    coord_der : dict
        dictionary of ndarray containing coordinate derivatives
        evaluated at node locations, such as computed by ``compute_coordinate_derivatives``
    cov_basis : dict
        dictionary of ndarray containing covariant basis vectors
        and derivatives at each node, such as computed by ``compute_covariant_basis``
    jacobian : dict
        dictionary of ndarray containing coordinate jacobian
        and partial derivatives, such as computed by ``compute_jacobian``
    RZ_transform : Transform
        object with transform method to go from spectral to physical space with derivatives

    Returns
    -------
    con_basis : dict
        dictionary of ndarray containing contravariant basis vectors and jacobian elements

    """
    # subscripts (superscripts) denote covariant (contravariant) basis vectors
    con_basis = {}

    # contravariant basis vectors
    con_basis['e^rho'] = cross(
        cov_basis['e_theta'], cov_basis['e_zeta'], 0)/jacobian['g']
    con_basis['e^theta'] = cross(
        cov_basis['e_zeta'], cov_basis['e_rho'], 0)/jacobian['g']
    con_basis['e^zeta'] = jnp.array(
        [coord_der['0'], 1/coord_der['R'], coord_der['0']])

    # axis terms
    if RZ_transform.grid.axis.size > 0:
        axn = RZ_transform.grid.axis
        con_basis['e^rho'] = put(con_basis['e^rho'], opsindex[:, axn], (cross(
            cov_basis['e_theta_r'][:, axn], cov_basis['e_zeta'][:, axn], 0)/jacobian['g_r'][axn]))
        # e^theta = infinite at the axis

    # metric coefficients
    con_basis['g^rr'] = dot(con_basis['e^rho'],   con_basis['e^rho'],   0)
    con_basis['g^rv'] = dot(con_basis['e^rho'],   con_basis['e^theta'], 0)
    con_basis['g^rz'] = dot(con_basis['e^rho'],   con_basis['e^zeta'],  0)
    con_basis['g^vv'] = dot(con_basis['e^theta'], con_basis['e^theta'], 0)
    con_basis['g^vz'] = dot(con_basis['e^theta'], con_basis['e^zeta'],  0)
    con_basis['g^zz'] = dot(con_basis['e^zeta'],  con_basis['e^zeta'],  0)

    return con_basis


def compute_jacobian(coord_der, cov_basis, RZ_transform):
    """Computes coordinate jacobian and derivatives

    Parameters
    ----------
    coord_der : dict
        dictionary of ndarray containing of coordinate
        derivatives evaluated at node locations, such as computed by ``compute_coordinate_derivatives``.
    cov_basis : dict
        dictionary of ndarray containing covariant basis
        vectors and derivatives at each node, such as computed by ``compute_covariant_basis``.
    RZ_transform : Transform
        object with transform method to go from spectral to physical space with derivatives

    Returns
    -------
    jacobian : dict
        dictionary of ndarray, shape(N_nodes,) of coordinate
        jacobian and partial derivatives. Keys are of the form `g_x` meaning
        the x derivative of the coordinate jacobian g

    """
    # notation: subscripts denote partial derivatives
    jacobian = {}
    jacobian['g'] = dot(cov_basis['e_rho'], cross(
        cov_basis['e_theta'], cov_basis['e_zeta'], 0), 0)

    jacobian['g_r'] = dot(cov_basis['e_rho_r'], cross(cov_basis['e_theta'],   cov_basis['e_zeta'], 0), 0) \
        + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_r'], cov_basis['e_zeta'], 0), 0) \
        + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta'],
                                          cov_basis['e_zeta_r'], 0), 0)
    jacobian['g_v'] = dot(cov_basis['e_rho_v'], cross(cov_basis['e_theta'],   cov_basis['e_zeta'], 0), 0) \
        + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_v'], cov_basis['e_zeta'], 0), 0) \
        + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta'],
                                          cov_basis['e_zeta_v'], 0), 0)
    jacobian['g_z'] = dot(cov_basis['e_rho_z'], cross(cov_basis['e_theta'],   cov_basis['e_zeta'], 0), 0) \
        + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_z'], cov_basis['e_zeta'], 0), 0) \
        + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta'],
                                          cov_basis['e_zeta_z'], 0), 0)

    # axis or QS terms
    if RZ_transform.grid.axis.size > 0 or RZ_transform.derivs == 'qs':
        jacobian['g_rr'] = dot(cov_basis['e_rho_rr'], cross(cov_basis['e_theta'],   cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho_r'], cross(cov_basis['e_theta_r'], cov_basis['e_zeta'], 0), 0)*2 \
            + dot(cov_basis['e_rho_r'], cross(cov_basis['e_theta'],   cov_basis['e_zeta_r'], 0), 0)*2 \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_rr'], cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_r'], cov_basis['e_zeta_r'], 0), 0)*2 \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta'],
                                              cov_basis['e_zeta_rr'], 0), 0)
        jacobian['g_rv'] = dot(cov_basis['e_rho_rv'], cross(cov_basis['e_theta'],   cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho_r'], cross(cov_basis['e_theta_v'], cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho_r'], cross(cov_basis['e_theta'],   cov_basis['e_zeta_v'], 0), 0) \
            + dot(cov_basis['e_rho_v'], cross(cov_basis['e_theta_r'], cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_rv'], cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_r'], cov_basis['e_zeta_v'], 0), 0) \
            + dot(cov_basis['e_rho_v'], cross(cov_basis['e_theta'],   cov_basis['e_zeta_r'], 0), 0) \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_v'], cov_basis['e_zeta_r'], 0), 0) \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta'],
                                              cov_basis['e_zeta_rv'], 0), 0)
        jacobian['g_rz'] = dot(cov_basis['e_rho_rz'], cross(cov_basis['e_theta'],   cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho_r'], cross(cov_basis['e_theta_z'], cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho_r'], cross(cov_basis['e_theta'],   cov_basis['e_zeta_z'], 0), 0) \
            + dot(cov_basis['e_rho_z'], cross(cov_basis['e_theta_r'], cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_rz'], cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_r'], cov_basis['e_zeta_z'], 0), 0) \
            + dot(cov_basis['e_rho_z'], cross(cov_basis['e_theta'],   cov_basis['e_zeta_r'], 0), 0) \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_z'], cov_basis['e_zeta_r'], 0), 0) \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta'],
                                              cov_basis['e_zeta_rz'], 0), 0)

        jacobian['g_vv'] = dot(cov_basis['e_rho_vv'], cross(cov_basis['e_theta'],   cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho_v'], cross(cov_basis['e_theta_v'], cov_basis['e_zeta'], 0), 0)*2 \
            + dot(cov_basis['e_rho_v'], cross(cov_basis['e_theta'],   cov_basis['e_zeta_v'], 0), 0)*2 \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_vv'], cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_v'], cov_basis['e_zeta_v'], 0), 0)*2 \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta'],
                                              cov_basis['e_zeta_vv'], 0), 0)
        jacobian['g_vz'] = dot(cov_basis['e_rho_vz'], cross(cov_basis['e_theta'],   cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho_v'], cross(cov_basis['e_theta_z'], cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho_v'], cross(cov_basis['e_theta'],   cov_basis['e_zeta_z'], 0), 0) \
            + dot(cov_basis['e_rho_z'], cross(cov_basis['e_theta_v'], cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_vz'], cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_v'], cov_basis['e_zeta_z'], 0), 0) \
            + dot(cov_basis['e_rho_z'], cross(cov_basis['e_theta'],   cov_basis['e_zeta_v'], 0), 0) \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_z'], cov_basis['e_zeta_v'], 0), 0) \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta'],
                                              cov_basis['e_zeta_vz'], 0), 0)
        jacobian['g_zz'] = dot(cov_basis['e_rho_zz'], cross(cov_basis['e_theta'],   cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho_z'], cross(cov_basis['e_theta_z'], cov_basis['e_zeta'], 0), 0)*2 \
            + dot(cov_basis['e_rho_z'], cross(cov_basis['e_theta'],   cov_basis['e_zeta_z'], 0), 0)*2 \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_zz'], cov_basis['e_zeta'], 0), 0) \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta_z'], cov_basis['e_zeta_z'], 0), 0)*2 \
            + dot(cov_basis['e_rho'],   cross(cov_basis['e_theta'],
                                              cov_basis['e_zeta_zz'], 0), 0)

    return jacobian


def compute_magnetic_field(cov_basis, jacobian, cI, Psi_lcfs, iota_transform, mode='force'):
    """Computes magnetic field components at node locations

    Parameters
    ----------
    cov_basis : dict
        dictionary of ndarray containing covariant basis 
        vectors and derivatives at each node, such as computed by ``compute_covariant_basis``.
    jacobian : dict
        dictionary of ndarray containing coordinate jacobian 
        and partial derivatives, such as computed by ``compute_jacobian``.
    cI : ndarray
        coefficients to pass to rotational transform function
    Psi_lcfs : float
        total toroidal flux (in Webers) within LCFS
    iota_transform : Transform
        object with transform method to go from spectral to physical space with derivatives
    mode : str 
        one of 'force' or 'qs'. Whether to compute field terms for equilibrium
        or quasisymmetry optimization (Default value = 'force')

    Returns
    -------
    magnetic_field: dict
        dictionary of ndarray, shape(N_nodes,) of magnetic field
        and derivatives. Keys are of the form 'B_x_y' or 'B^x_y', meaning the
        covariant (B_x) or contravariant (B^x) component of the magnetic field, with the derivative wrt to y.

    """
    # notation: 1 letter subscripts denote derivatives, eg psi_rr = d^2 psi / dr^2
    # subscripts (superscripts) denote covariant (contravariant) components of the field
    magnetic_field = {}
    r = iota_transform.grid.nodes[:, 0]
    axn = iota_transform.grid.axis
    iota = iota_transform.transform(cI, 0)
    iota_r = iota_transform.transform(cI, 1)

    # toroidal flux
    magnetic_field['psi'] = Psi_lcfs*r**2
    magnetic_field['psi_r'] = 2*Psi_lcfs*r
    magnetic_field['psi_rr'] = 2*Psi_lcfs*jnp.ones_like(r)

    # contravariant B components
    magnetic_field['B^rho'] = jnp.zeros_like(r)
    magnetic_field['B^zeta'] = magnetic_field['psi_r'] / \
        (2*jnp.pi*jacobian['g'])
    if len(axn):
        magnetic_field['B^zeta'] = put(
            magnetic_field['B^zeta'], axn, magnetic_field['psi_rr'][axn] / (2*jnp.pi*jacobian['g_r'][axn]))
    magnetic_field['B^theta'] = iota * magnetic_field['B^zeta']
    magnetic_field['B_con'] = magnetic_field['B^rho']*cov_basis['e_rho'] + magnetic_field['B^theta'] * \
        cov_basis['e_theta'] + magnetic_field['B^zeta']*cov_basis['e_zeta']

    # covariant B components
    magnetic_field['B_rho'] = magnetic_field['B^zeta'] * \
        dot(iota*cov_basis['e_theta'] +
            cov_basis['e_zeta'], cov_basis['e_rho'], 0)
    magnetic_field['B_theta'] = magnetic_field['B^zeta'] * \
        dot(iota*cov_basis['e_theta'] +
            cov_basis['e_zeta'], cov_basis['e_theta'], 0)
    magnetic_field['B_zeta'] = magnetic_field['B^zeta'] * \
        dot(iota*cov_basis['e_theta'] +
            cov_basis['e_zeta'], cov_basis['e_zeta'], 0)

    # B^{zeta} derivatives
    magnetic_field['B^zeta_r'] = magnetic_field['psi_rr'] / (2*jnp.pi*jacobian['g']) - \
        (magnetic_field['psi_r']*jacobian['g_r']) / (2*jnp.pi*jacobian['g']**2)
    magnetic_field['B^zeta_v'] = - \
        (magnetic_field['psi_r']*jacobian['g_v']) / (2*jnp.pi*jacobian['g']**2)
    magnetic_field['B^zeta_z'] = - \
        (magnetic_field['psi_r']*jacobian['g_z']) / (2*jnp.pi*jacobian['g']**2)

    # axis terms
    if len(axn):
        magnetic_field['B^zeta_r'] = put(magnetic_field['B^zeta_r'], axn, -(magnetic_field['psi_rr']
                                                                            [axn]*jacobian['g_rr'][axn]) / (4*jnp.pi*jacobian['g_r'][axn]**2))
        magnetic_field['B^zeta_v'] = put(magnetic_field['B^zeta_v'], axn, 0)
        magnetic_field['B^zeta_z'] = put(magnetic_field['B^zeta_z'], axn, -(magnetic_field['psi_rr']
                                                                            [axn]*jacobian['g_rz'][axn]) / (2*jnp.pi*jacobian['g_r'][axn]**2))

    # QS terms
    if mode == 'qs':
        magnetic_field['B^zeta_vv'] = - (magnetic_field['psi_r']*jacobian['g_vv']) / (2*jnp.pi*jacobian['g']**2) \
            + (magnetic_field['psi_r']*jacobian['g_v']
               ** 2) / (jnp.pi*jacobian['g']**3)
        magnetic_field['B^zeta_vz'] = - (magnetic_field['psi_r']*jacobian['g_vz']) / (2*jnp.pi*jacobian['g']**2) \
            + (magnetic_field['psi_r']*jacobian['g_v']*jacobian['g_z']) / \
            (jnp.pi*jacobian['g']**3)
        magnetic_field['B^zeta_zz'] = - (magnetic_field['psi_r']*jacobian['g_zz']) / (2*jnp.pi*jacobian['g']**2) \
            + (magnetic_field['psi_r']*jacobian['g_z']
               ** 2) / (jnp.pi*jacobian['g']**3)

    # covariant B component derivatives
    magnetic_field['B_theta_r'] = magnetic_field['B^zeta_r']*dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_theta'], 0) \
        + magnetic_field['B^zeta']*(dot(iota_r*cov_basis['e_theta']+iota*cov_basis['e_rho_v']+cov_basis['e_zeta_r'], cov_basis['e_theta'], 0)
                                    + dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_rho_v'], 0))
    magnetic_field['B_zeta_r'] = magnetic_field['B^zeta_r']*dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_zeta'], 0) \
        + magnetic_field['B^zeta']*(dot(iota_r*cov_basis['e_theta']+iota*cov_basis['e_rho_v']+cov_basis['e_zeta_r'], cov_basis['e_zeta'], 0)
                                    + dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_zeta_r'], 0))
    magnetic_field['B_rho_v'] = magnetic_field['B^zeta_v']*dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_rho'], 0) \
        + magnetic_field['B^zeta']*(dot(iota*cov_basis['e_theta_v']+cov_basis['e_zeta_v'], cov_basis['e_rho'], 0)
                                    + dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_rho_v'], 0))
    magnetic_field['B_zeta_v'] = magnetic_field['B^zeta_v']*dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_zeta'], 0) \
        + magnetic_field['B^zeta']*(dot(iota*cov_basis['e_theta_v']+cov_basis['e_zeta_v'], cov_basis['e_zeta'], 0)
                                    + dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_zeta_v'], 0))
    magnetic_field['B_rho_z'] = magnetic_field['B^zeta_z']*dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_rho'], 0) \
        + magnetic_field['B^zeta']*(dot(iota*cov_basis['e_theta_z']+cov_basis['e_zeta_z'], cov_basis['e_rho'], 0)
                                    + dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_rho_z'], 0))
    magnetic_field['B_theta_z'] = magnetic_field['B^zeta_z']*dot(iota*cov_basis['e_theta']+cov_basis['e_zeta'], cov_basis['e_theta'], 0) \
        + magnetic_field['B^zeta']*(dot(iota*cov_basis['e_theta_z']+cov_basis['e_zeta_z'], cov_basis['e_theta'], 0)
                                    + dot(iota*cov_basis['e_theta'] + cov_basis['e_zeta'], cov_basis['e_theta_z'], 0))

    return magnetic_field


def compute_plasma_current(coord_der, cov_basis, jacobian, magnetic_field, cI, iota_transform):
    """Computes current density field at node locations

    Parameters
    ----------
    cov_basis : dict
        dictionary of ndarray containing covariant basis 
        vectors and derivatives at each node, such as computed by ``compute_covariant_basis``.
    jacobian : dict
        dictionary of ndarray containing coordinate jacobian 
        and partial derivatives, such as computed by ``compute_jacobian``.
    coord_der : dict
        dictionary of ndarray containing of coordinate 
        derivatives evaluated at node locations, such as computed by ``compute_coordinate_derivatives``.
    magnetic_field : dict
        dictionary of ndarray containing magnetic field and derivatives, 
        such as computed by ``compute_magnetic_field``.
    cI : ndarray
        coefficients to pass to rotational transform function.
    iota_transform : Transform
        object with transform method to go from spectral to physical space with derivatives

    Returns
    -------
    plasma_current : dict
        dictionary of ndarray, shape(N_nodes,) of current field.
        Keys are of the form 'J^x_y' meaning the contravariant (J^x)
        component of the current, with the derivative wrt to y.

    """
    # notation: 1 letter subscripts denote derivatives, eg psi_rr = d^2 psi / dr^2
    # subscripts (superscripts) denote covariant (contravariant) components of the field
    plasma_current = {}
    mu0 = 4*jnp.pi*1e-7
    axn = iota_transform.grid.axis
    iota = iota_transform.transform(cI, 0)

    # axis terms
    if len(axn):
        g_rrv = 2*coord_der['R_rv']*(coord_der['Z_r']*coord_der['R_rv'] - coord_der['R_r']*coord_der['Z_rv']) \
            + 2*coord_der['R_r']*(coord_der['Z_r']*coord_der['R_rvv'] - coord_der['R_r']*coord_der['Z_rvv']) \
            + coord_der['R']*(2*coord_der['Z_rr']*coord_der['R_rvv'] - 2*coord_der['R_rr']*coord_der['Z_rvv']
                              + coord_der['R_rv']*coord_der['Z_rrv'] -
                              coord_der['Z_rv']*coord_der['R_rrv']
                              + coord_der['Z_r']*coord_der['R_rrvv'] - coord_der['R_r']*coord_der['Z_rrvv'])
        Bsup_zeta_rv = magnetic_field['psi_rr']*(2*jacobian['g_rr']*jacobian['g_rv'] -
                                                 jacobian['g_r']*g_rrv) / (4*jnp.pi*jacobian['g_r']**3)
        Bsub_zeta_rv = Bsup_zeta_rv*dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0) + magnetic_field['B^zeta']*dot(
            iota*cov_basis['e_rho_vv'] + 2*cov_basis['e_zeta_rv'], cov_basis['e_zeta'], 0)
        Bsub_theta_rz = magnetic_field['B^zeta_z']*dot(cov_basis['e_zeta'], cov_basis['e_rho_v'], 0) + magnetic_field['B^zeta']*(
            dot(cov_basis['e_zeta_z'], cov_basis['e_rho_v'], 0) + dot(cov_basis['e_zeta'], cov_basis['e_rho_vz'], 0))

    # contravariant J components
    plasma_current['J^rho'] = (magnetic_field['B_zeta_v'] -
                               magnetic_field['B_theta_z']) / (mu0*jacobian['g'])
    plasma_current['J^theta'] = (magnetic_field['B_rho_z'] -
                                 magnetic_field['B_zeta_r']) / (mu0*jacobian['g'])
    plasma_current['J^zeta'] = (magnetic_field['B_theta_r'] -
                                magnetic_field['B_rho_v']) / (mu0*jacobian['g'])

    # axis terms
    if len(axn):
        plasma_current['J^rho'] = put(plasma_current['J^rho'], axn,
                                      (Bsub_zeta_rv[axn] - Bsub_theta_rz[axn]) / (jacobian['g_r'][axn]))

    plasma_current['J_con'] = plasma_current['J^rho']*cov_basis['e_rho'] + plasma_current['J^theta'] * \
        cov_basis['e_theta'] + plasma_current['J^zeta']*cov_basis['e_zeta']

    return plasma_current


def compute_magnetic_field_magnitude(cov_basis, magnetic_field, cI, iota_transform):
    """Computes magnetic field magnitude at node locations

    Parameters
    ----------
    cov_basis : dict
        dictionary of ndarray containing covariant basis
        vectors and derivatives at each node, such as computed by ``compute_covariant_basis``.
    magnetic_field : dict
        dictionary of ndarray containing magnetic field and derivatives,
        such as computed by ``compute_magnetic_field``.
    cI : ndarray
        coefficients to pass to rotational transform function
    iota_transform : Transform
        object with transform method to go from spectral to physical space with derivatives

    Returns
    -------
    B_mag : dict
        dictionary of ndarray, shape(N_nodes,) of magnetic field magnitude and derivatives

    """
    # notation: 1 letter subscripts denote derivatives, eg psi_rr = d^2 psi / dr^2
    # subscripts (superscripts) denote covariant (contravariant) components of the field
    B_mag = {}
    iota = iota_transform.transform(cI, 0)

    B_mag['|B|'] = jnp.abs(magnetic_field['B^zeta'])*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0) +
                                                              2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0) + dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0))

    B_mag['|B|_v'] = jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_v']*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
        + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_v'], 0)+2*iota*(dot(cov_basis['e_theta_v'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_v'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_v'], 0)) \
        / (2*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)))

    B_mag['|B|_z'] = jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_z']*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
        + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_z'], 0)+2*iota*(dot(cov_basis['e_theta_z'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_z'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_z'], 0)) \
        / (2*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)))

    B_mag['|B|_vv'] = jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_vv']*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
        + jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_v']*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_v'], 0)+2*iota*(dot(cov_basis['e_theta_v'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_v'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_v'], 0)) \
        / jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
        + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*(dot(cov_basis['e_theta_v'], cov_basis['e_theta_v'], 0)+dot(cov_basis['e_theta'], cov_basis['e_theta_vv'], 0))+2*iota*(dot(cov_basis['e_theta_vv'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_vv'], 0)+2*dot(cov_basis['e_theta_v'], cov_basis['e_zeta_v'], 0))+2*(dot(cov_basis['e_zeta_v'], cov_basis['e_zeta_v'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta_vv'], 0))) \
        / (2*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0))) \
        + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_v'], 0)+2*iota*(dot(cov_basis['e_theta_v'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_v'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_v'], 0))**2 \
        / (2*(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0))**(3/2))

    B_mag['|B|_zz'] = jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_zz']*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
        + jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_z']*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_z'], 0)+2*iota*(dot(cov_basis['e_theta_z'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_z'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_z'], 0)) \
        / jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
        + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*(dot(cov_basis['e_theta_z'], cov_basis['e_theta_z'], 0)+dot(cov_basis['e_theta'], cov_basis['e_theta_zz'], 0))+2*iota*(dot(cov_basis['e_theta_zz'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_zz'], 0)+2*dot(cov_basis['e_theta_z'], cov_basis['e_zeta_z'], 0))+2*(dot(cov_basis['e_zeta_z'], cov_basis['e_zeta_z'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta_vz'], 0))) \
        / (2*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0))) \
        + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_z'], 0)+2*iota*(dot(cov_basis['e_theta_z'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_z'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_z'], 0))**2 \
        / (2*(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0))**(3/2))

    B_mag['|B|_vz'] = jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_vz']*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
        + jnp.sign(magnetic_field['B^zeta'])*magnetic_field['B^zeta_v']*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_z'], 0)+2*iota*(dot(cov_basis['e_theta_z'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_z'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_z'], 0)) \
        / jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0)) \
        + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*(dot(cov_basis['e_theta_z'], cov_basis['e_theta_v'], 0)+dot(cov_basis['e_theta'], cov_basis['e_theta_vz'], 0))+2*iota*(dot(cov_basis['e_theta_vz'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta_v'], cov_basis['e_zeta_z'], 0)+dot(cov_basis['e_theta_z'], cov_basis['e_zeta_v'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_vz'], 0))+2*(dot(cov_basis['e_zeta_z'], cov_basis['e_zeta_v'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta_vz'], 0))) \
        / (2*jnp.sqrt(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0))) \
        + jnp.abs(magnetic_field['B^zeta'])*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_v'], 0)+2*iota*(dot(cov_basis['e_theta_v'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_v'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_v'], 0))*(2*iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta_z'], 0)+2*iota*(dot(cov_basis['e_theta_z'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_theta'], cov_basis['e_zeta_z'], 0))+2*dot(cov_basis['e_zeta'], cov_basis['e_zeta_z'], 0)) \
        / (2*(iota**2*dot(cov_basis['e_theta'], cov_basis['e_theta'], 0)+2*iota*dot(cov_basis['e_theta'], cov_basis['e_zeta'], 0)+dot(cov_basis['e_zeta'], cov_basis['e_zeta'], 0))**(3/2))

    return B_mag


def compute_force_magnitude(coord_der, cov_basis, con_basis, jacobian, magnetic_field, plasma_current, cP, pres_transform):
    """Computes force error magnitude at node locations

    Parameters
    ----------
    coord_der : dict
        dictionary of ndarray containing of coordinate
        derivatives evaluated at node locations, such as computed by ``compute_coordinate_derivatives``.
    cov_basis : dict
        dictionary of ndarray containing covariant basis
        vectors and derivatives at each node, such as computed by ``compute_covariant_basis``.
    con_basis : dict
        dictionary of ndarray containing contravariant basis
        vectors and metric elements at each node, such as computed by ``compute_contravariant_basis``.
    jacobian : dict
        dictionary of ndarray containing coordinate jacobian
        and partial derivatives, such as computed by ``compute_jacobian``.
    magnetic_field : dict
        dictionary of ndarray containing magnetic field and derivatives, 
        such as computed by ``compute_magnetic_field``.
    plasma_current : dict
        dictionary of ndarray containing current and derivatives, 
        such as computed by ``compute_plasma_current``.
    cP : ndarray
        parameters to pass to pressure function
    Psi_lcfs : float
        total toroidal flux (in Webers) within LCFS
    pres_transform : Transform
        object with transform method to go from spectral to physical space with derivatives

    Returns
    -------
    force_magnitude : ndarray, shape(N_nodes,)
        force error magnitudes at each node.
    p_mag : ndarray, shape(N_nodes,)
        magnitude of pressure gradient at each node.

    """
    mu0 = 4*jnp.pi*1e-7
    axn = pres_transform.grid.axis
    pres_r = pres_transform.transform(cP, 1)

    # force balance error covariant components
    F_rho = jacobian['g']*(plasma_current['J^theta']*magnetic_field['B^zeta'] -
                           plasma_current['J^zeta']*magnetic_field['B^theta']) - pres_r
    F_theta = jacobian['g']*plasma_current['J^rho']*magnetic_field['B^zeta']
    F_zeta = -jacobian['g']*plasma_current['J^rho']*magnetic_field['B^theta']

    # axis terms
    if len(axn):
        Jsup_theta = (magnetic_field['B_rho_z'] -
                      magnetic_field['B_zeta_r']) / mu0
        Jsup_zeta = (magnetic_field['B_theta_r'] -
                     magnetic_field['B_rho_v']) / mu0
        F_rho = put(F_rho, axn, Jsup_theta[axn]*magnetic_field['B^zeta']
                    [axn] - Jsup_zeta[axn]*magnetic_field['B^theta'][axn])
        grad_theta = cross(cov_basis['e_zeta'], cov_basis['e_rho'], 0)
        gsup_vv = dot(grad_theta, grad_theta, 0)
        gsup_rv = dot(con_basis['e^rho'], grad_theta, 0)
        gsup_vz = dot(grad_theta, con_basis['e^zeta'], 0)
        F_theta = put(
            F_theta, axn, plasma_current['J^rho'][axn]*magnetic_field['B^zeta'][axn])
        F_zeta = put(F_zeta, axn, -plasma_current['J^rho']
                     [axn]*magnetic_field['B^theta'][axn])
        con_basis['g^vv'] = put(con_basis['g^vv'], axn, gsup_vv[axn])
        con_basis['g^rv'] = put(con_basis['g^rv'], axn, gsup_rv[axn])
        con_basis['g^vz'] = put(con_basis['g^vz'], axn, gsup_vz[axn])

    # F_i*F_j*g^ij terms
    Fg_rr = F_rho * F_rho * con_basis['g^rr']
    Fg_vv = F_theta*F_theta*con_basis['g^vv']
    Fg_zz = F_zeta * F_zeta * con_basis['g^zz']
    Fg_rv = F_rho * F_theta*con_basis['g^rv']
    Fg_rz = F_rho * F_zeta * con_basis['g^rz']
    Fg_vz = F_theta*F_zeta * con_basis['g^vz']

    # magnitudes
    force_magnitude = jnp.sqrt(
        Fg_rr + Fg_vv + Fg_zz + 2*Fg_rv + 2*Fg_rz + 2*Fg_vz)
    p_mag = jnp.sqrt(pres_r*pres_r*con_basis['g^rr'])

    return force_magnitude, p_mag
