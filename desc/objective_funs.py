import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot

from desc.backend import jnp, put, cross, dot, TextColors
from desc.configuration import unpack_state
from desc.configuration import compute_coordinate_derivatives, compute_covariant_basis
from desc.configuration import compute_contravariant_basis, compute_jacobian
from desc.configuration import compute_magnetic_field, compute_plasma_current, compute_magnetic_field_magnitude
from desc.boundary_conditions import compute_bdry_err, compute_lambda_err
from desc.grid import LinearGrid
from desc.transform import Transform
from desc.equilibrium_io import IOAble


class ObjectiveFunction(IOAble,ABC):

    """Objective function used in the optimization of an Equilibrium

    Attributes
    ----------
    scalar : bool, optional
        True for scalar objectives, False otherwise (Default)
    R_transform : Transform, optional
        transforms R coefficients to real space in the volume
    Z_transform : Transform, optional
        transforms Z coefficients to real space in the volume
    R1_transform : Transform, optional
        transforms R coefficients to real space on the surface
    Z1_transform : Transform, optional
        transforms Z coefficients to real space on the surface
    L_transform : Transform, optional
        transforms lambda coefficients to real space
    P_transform : Transform, optional
        transforms pressure coefficients to real space
    I_transform : Transform, optional
        transforms rotational transform coefficients to real space

    Methods
    -------
    compute(x, bdryR, bdryZ, cP, cI, Psi_lcfs, bdry_ratio=1.0, pres_ratio=1.0, zeta_ratio=1.0, errr_ratio=1.0)
        compute the equilibrium objective function
    callback(x, bdryR, bdryZ, cP, cI, Psi_lcfs, bdry_ratio=1.0, pres_ratio=1.0, zeta_ratio=1.0, errr_ratio=1.0)
        function that prints equilibrium errors
    """
    _save_attrs_ = ['scalar', 'R_transform', 'Z_transform', 'R1_transform',
            'Z1_transform', 'L_transform', 'P_transform', 'I_transform']

    def __init__(self, scalar:bool=False,
                 R_transform:Transform=None, Z_transform:Transform=None,
                 R1_transform:Transform=None, Z1_transform:Transform=None,
                 L_transform:Transform=None, P_transform:Transform=None,
                 I_transform:Transform=None) -> None:
        """Initializes an ObjectiveFunction

        Parameters
        ----------
        scalar : bool, optional
            True for scalar objectives, False otherwise (Default)
        R_transform : Transform, optional
            transforms R coefficients to real space in the volume
        Z_transform : Transform, optional
            transforms Z coefficients to real space in the volume
        R1_transform : Transform, optional
            transforms R coefficients to real space on the surface
        Z1_transform : Transform, optional
            transforms Z coefficients to real space on the surface
        L_transform : Transform, optional
            transforms lambda coefficients to real space
        P_transform : Transform, optional
            transforms pressure coefficients to real space
        I_transform : Transform, optional
            transforms rotational transform coefficients to real space

        Returns
        -------
        None

        """
        self.scalar = scalar
        self.R_transform = R_transform
        self.Z_transform = Z_transform
        self.R1_transform = R1_transform
        self.Z1_transform = Z1_transform
        self.L_transform = L_transform
        self.P_transform = P_transform
        self.I_transform = I_transform

    @abstractmethod
    def compute(self, x, cRb, cZb, cP, cI, Psi_lcfs, bdry_ratio=1.0, pres_ratio=1.0, zeta_ratio=1.0, errr_ratio=1.0):
        pass
    @abstractmethod
    def callback(self, x, cRb, cZb, cP, cI, Psi_lcfs, bdry_ratio=1.0, pres_ratio=1.0, zeta_ratio=1.0, errr_ratio=1.0):
        pass


class ForceErrorNodes(ObjectiveFunction):
    """Minimizes equilibrium force balance error in physical space"""

    def __init__(self, scalar:bool=False,
                 R_transform:Transform=None, Z_transform:Transform=None,
                 R1_transform:Transform=None, Z1_transform:Transform=None,
                 L_transform:Transform=None, P_transform:Transform=None,
                 I_transform:Transform=None) -> None:
        """Initializes a ForceErrorNodes object

        Parameters
        ----------
        scalar : bool, optional
            True for scalar objectives, False otherwise (Default)
        R_transform : Transform, optional
            transforms R coefficients to real space in the volume
        Z_transform : Transform, optional
            transforms Z coefficients to real space in the volume
        R1_transform : Transform, optional
            transforms R coefficients to real space on the surface
        Z1_transform : Transform, optional
            transforms Z coefficients to real space on the surface
        L_transform : Transform, optional
            transforms lambda coefficients to real space
        P_transform : Transform, optional
            transforms pressure coefficients to real space
        I_transform : Transform, optional
            transforms rotational transform coefficients to real space

        Returns
        -------
        None

        """
        super().__init__(scalar, R_transform, Z_transform, R1_transform,
                         Z1_transform, L_transform, P_transform, I_transform)
        self.equil_fun = compute_force_error_nodes
        self.bdry_fun = compute_bdry_err

    def compute(self, x, cRb, cZb, cP, cI, Psi_lcfs, bdry_ratio=1.0, pres_ratio=1.0, zeta_ratio=1.0, errr_ratio=1.0):
        """ Compute force balance error. Overrides the compute method of the parent ObjectiveFunction"""
        cR, cZ, cL = unpack_state(x,
                      self.R_transform.num_modes, self.Z_transform.num_modes)
        errRf, errZf = self.equil_fun(
            cR, cZ, cP, cI, Psi_lcfs, self.R_transform, self.Z_transform, self.P_transform, self.I_transform, pres_ratio, zeta_ratio)
        errRb, errZb = self.bdry_fun(
            cR, cZ, cL, cRb, cZb, self.R1_transform, self.Z1_transform, self.L_transform, bdry_ratio)

        residual = jnp.concatenate([errRf.flatten(),
                                    errZf.flatten(),
                                    errRb.flatten()/errr_ratio,
                                    errZb.flatten()/errr_ratio])

        if not self.L_transform.basis.sym != False:
            errL0 = compute_lambda_err(cL, self.L_transform.basis)
            residual = jnp.concatenate([residual, errL0.flatten()/errr_ratio])

        if self.scalar:
            residual = jnp.log1p(jnp.sum(residual**2))
        return residual

    def callback(self, x, cRb, cZb, cP, cI, Psi_lcfs, bdry_ratio=1.0, pres_ratio=1.0, zeta_ratio=1.0, errr_ratio=1.0)->None:
        """ Print residuals. Overrides callback method of the parent ObjectiveFunction"""
        cR, cZ, cL = unpack_state(x,
                      self.R_transform.num_modes, self.Z_transform.num_modes)
        errRf, errZf = self.equil_fun(
            cR, cZ, cP, cI, Psi_lcfs, self.R_transform, self.Z_transform, self.P_transform, self.I_transform, pres_ratio, zeta_ratio)
        errRb, errZb = self.bdry_fun(
            cR, cZ, cL, cRb, cZb, self.R1_transform, self.Z1_transform, self.L_transform, bdry_ratio)
        errL0 = compute_lambda_err(cL, self.L_transform.basis)

        errRf_rms = jnp.sqrt(jnp.sum(errRf**2))
        errZf_rms = jnp.sqrt(jnp.sum(errZf**2))
        errRb_rms = jnp.sqrt(jnp.sum(errRb**2))
        errZb_rms = jnp.sqrt(jnp.sum(errZb**2))
        errL0_rms = jnp.sqrt(jnp.sum(errL0**2))

        residual = jnp.concatenate([errRf.flatten(),
                                    errZf.flatten(),
                                    errRb.flatten()/errr_ratio,
                                    errZb.flatten()/errr_ratio,
                                    errL0.flatten()/errr_ratio])
        resid_rms = 1/2*jnp.sum(residual**2)
        print('Weighted Loss: {:10.3e}  errFrho: {:10.3e}  errFbeta: {:10.3e}  errRb: {:10.3e}  errZb: {:10.3e}  errL0: {:10.3e}'.format(
                resid_rms, errRf_rms, errZf_rms, errRb_rms, errZb_rms, errL0_rms))


class AccelErrorSpectral(ObjectiveFunction):
    """Minimizes equilibrium acceleration error in spectral space"""

    def __init__(self, scalar:bool=False,
                 R_transform:Transform=None, Z_transform:Transform=None,
                 R1_transform:Transform=None, Z1_transform:Transform=None,
                 L_transform:Transform=None, P_transform:Transform=None,
                 I_transform:Transform=None) -> None:
        """Initializes an AccelErrorNodes object

        Parameters
        ----------
        scalar : bool, optional
            True for scalar objectives, False otherwise (Default)
        R_transform : Transform, optional
            transforms R coefficients to real space in the volume
        Z_transform : Transform, optional
            transforms Z coefficients to real space in the volume
        R1_transform : Transform, optional
            transforms R coefficients to real space on the surface
        Z1_transform : Transform, optional
            transforms Z coefficients to real space on the surface
        L_transform : Transform, optional
            transforms lambda coefficients to real space
        P_transform : Transform, optional
            transforms pressure coefficients to real space
        I_transform : Transform, optional
            transforms rotational transform coefficients to real space

        Returns
        -------
        None

        """
        super().__init__(scalar, R_transform, Z_transform, R1_transform,
                         Z1_transform, L_transform, P_transform, I_transform)
        self.equil_fun = compute_accel_error_spectral
        self.bdry_fun = compute_bdry_err

    def compute(self, x, cRb, cZb, cP, cI, Psi_lcfs, bdry_ratio=1.0, pres_ratio=1.0, zeta_ratio=1.0, errr_ratio=1.0):
        """ Compute spectral space acceleration error. Overrides the compute method of the parent ObjectiveFunction"""
        cR, cZ, cL = unpack_state(x,
                      self.R_transform.num_modes, self.Z_transform.num_modes)

        errRf, errZf = self.equil_fun(
            cR, cZ, cP, cI, Psi_lcfs, self.R_transform, self.Z_transform, self.P_transform, self.I_transform, pres_ratio, zeta_ratio)
        errRb, errZb = self.bdry_fun(
            cR, cZ, cL, cRb, cZb, self.R1_transform, self.Z1_transform, self.L_transform, bdry_ratio)

        residual = jnp.concatenate([errRf.flatten(),
                                    errZf.flatten(),
                                    errRb.flatten()/errr_ratio,
                                    errZb.flatten()/errr_ratio])

        if not self.stell_sym:
            errL0 = compute_lambda_err(cL, self.L_transform.basis)
            residual = jnp.concatenate([residual, errL0.flatten()/errr_ratio])

        if self.scalar:
            residual = jnp.log1p(jnp.sum(residual**2))
        return residual

    def callback(self, x, cRb, cZb, cP, cI, Psi_lcfs, bdry_ratio=1.0, pres_ratio=1.0, zeta_ratio=1.0, errr_ratio=1.0)->None:
        """ Print residuals. Overrides callback method of the parent ObjectiveFunction"""
        cR, cZ, cL = unpack_state(x,
                      self.R_transform.num_modes, self.Z_transform.num_modes)
        errRf, errZf = self.equil_fun(
            cR, cZ, cP, cI, Psi_lcfs, self.R_transform, self.Z_transform, self.P_transform, self.I_transform, pres_ratio, zeta_ratio)
        errRb, errZb = self.bdry_fun(
            cR, cZ, cL, cRb, cZb, self.R1_transform, self.Z1_transform, self.L_transform, bdry_ratio)
        errL0 = compute_lambda_err(cL, self.L_transform.basis)

        errRf_rms = jnp.sqrt(jnp.sum(errRf**2))
        errZf_rms = jnp.sqrt(jnp.sum(errZf**2))
        errRb_rms = jnp.sqrt(jnp.sum(errRb**2))
        errZb_rms = jnp.sqrt(jnp.sum(errZb**2))
        errL0_rms = jnp.sqrt(jnp.sum(errL0**2))

        residual = jnp.concatenate([errRf.flatten(),
                                    errZf.flatten(),
                                    errRb.flatten()/errr_ratio,
                                    errZb.flatten()/errr_ratio,
                                    errL0.flatten()/errr_ratio])
        resid_rms = 1/2*jnp.sum(residual**2)
        print('Weighted Loss: {:10.3e}  errFrho: {:10.3e}  errFbeta: {:10.3e}  errRb: {:10.3e}  errZb: {:10.3e}  errL0: {:10.3e}'.format(
                resid_rms, errRf_rms, errZf_rms, errRb_rms, errZb_rms, errL0_rms))


class ObjectiveFunctionFactory():
    """Factory Class for Objective Functions

    Methods
    -------
    get_equil_obj_fxn(errr_mode, RZ_transform:Transform=None,
                 RZb_transform:Transform=None, L_transform:Transform=None,
                 pres_transform:Transform=None, iota_transform:Transform=None,
                 stell_sym:bool=True, scalar:bool=False)

        Takes type of objective function and attributes of an equilibrium and uses it to compute and return the corresponding objective function

    """

    def get_equil_obj_fun(errr_mode, scalar:bool=False,
                  R_transform:Transform=None, Z_transform:Transform=None,
                  R1_transform:Transform=None, Z1_transform:Transform=None,
                  L_transform:Transform=None, P_transform:Transform=None,
                  I_transform:Transform=None) -> ObjectiveFunction:
        """Accepts parameters necessary to create an objective function, and returns the corresponding ObjectiveFunction object

        Parameters
        ----------
        errr_mode : str
            error mode of the objective function
            one of 'force', 'accel'
        scalar : bool, optional
            True for scalar objectives, False otherwise (Default)
        R_transform : Transform, optional
            transforms R coefficients to real space in the volume
        Z_transform : Transform, optional
            transforms Z coefficients to real space in the volume
        R1_transform : Transform, optional
            transforms R coefficients to real space on the surface
        Z1_transform : Transform, optional
            transforms Z coefficients to real space on the surface
        L_transform : Transform, optional
            transforms lambda coefficients to real space
        P_transform : Transform, optional
            transforms pressure coefficients to real space
        I_transform : Transform, optional
            transforms rotational transform coefficients to real space

        Returns
        -------
        obj_fxn : ObjectiveFunction
            equilibrium objective function object, containing the compute and callback method for the objective function

        """
        if errr_mode == 'force':
            obj_fun = ForceErrorNodes(scalar=scalar,
                R_transform=R_transform, Z_transform=Z_transform,
                R1_transform=R1_transform, Z1_transform=Z1_transform,
                L_transform=L_transform, P_transform=P_transform,
                I_transform=I_transform)
        elif errr_mode == 'accel':
            obj_fun = AccelErrorSpectral(scalar=scalar,
                R_transform=R_transform, Z_transform=Z_transform,
                R1_transform=R1_transform, Z1_transform=Z1_transform,
                L_transform=L_transform, P_transform=P_transform,
                I_transform=I_transform)
        else:
            raise ValueError(TextColors.FAIL + "Requested Objective Function is not implemented." +
                             " Available objective functions are 'force' and 'accel'" + TextColors.ENDC)
        return obj_fun


# TODO: need to turn this into another ObjectiveFun subclass
def get_qisym_obj_fun(stell_sym, M, N, NFP, zernike_transform, zern_idx, lambda_idx, modes_pol, modes_tor):
    """Gets the quasisymmetry objective function

    Parameters
    ----------
    stell_sym : bool
        True if stellarator symmetry is enforced
    M : int
        maximum poloidal resolution
    N : int
        maximum toroidal resolution
    NFP : int
        number of field periods
    zernike_transform : ZernikeTransform
        object with transform method to go from spectral to physical space with derivatives
    zern_idx : ndarray of int
        mode numbers for Zernike basis
    lambda_idx : ndarray of int
        mode numbers for Fourier basis
    modes_pol : ndarray
        poloidal Fourier mode numbers
    modes_tor : ndarray
        toroidal Fourier mode numbers
    Returns
    -------
    qsym_obj : function
        quasisymmetry objective function

    """

    # stellarator symmetry
    sym_mat = symmetry_matrix(zern_idx, lambda_idx, sym=stell_sym)

    def qisym_obj(x, cI, Psi_lcfs):

        cR, cZ, cL = unpack_state(jnp.matmul(sym_mat, x), RZ_transform.num_modes)
        errQS = compute_qs_error_spectral(
            cR, cZ, cI, Psi_lcfs, NFP, zernike_transform, modes_pol, modes_tor, 1.0)

        # normalize weighting by numper of nodes
        residual = errQS.flatten()/jnp.sqrt(errQS.size)
        return residual

    return qisym_obj


def curve_self_intersects(x, y):
    """Checks if a curve intersects itself

    Parameters
    ----------
    x,y : ndarray
        x and y coordinates of points along the curve
    Returns
    -------
    is_intersected : bool
        whether the curve intersects itself

    """

    pts = np.array([x, y])
    pts1 = pts[:, 0:-1]
    pts2 = pts[:, 1:]

    # [start/stop, x/y, segment]
    segments = np.array([pts1, pts2])
    s1, s2 = np.meshgrid(np.arange(len(x)-1), np.arange(len(y)-1))
    idx = np.array([s1.flatten(), s2.flatten()])
    a, b = segments[:, :, idx[0, :]]
    c, d = segments[:, :, idx[1, :]]

    def signed_2d_area(a, b, c):
        return (a[0] - c[0])*(b[1] - c[1]) - (a[1] - c[1])*(b[0] - c[0])

    # signs of areas correspond to which side of ab points c and d are
    a1 = signed_2d_area(a, b, d)  # Compute winding of abd (+ or -)
    a2 = signed_2d_area(a, b, c)  # To intersect, must have sign opposite of a1
    a3 = signed_2d_area(c, d, a)  # Compute winding of cda (+ or -)
    a4 = a3 + a2 - a1  # Since area is constant a1 - a2 = a3 - a4, or a4 = a3 + a2 - a1

    return np.any(np.where(np.logical_and(a1*a2 < 0, a3*a4 < 0), True, False))


def is_nested(cR, cZ, R_basis, Z_basis, L=10, M=361, zeta=0):
    """Checks that an equilibrium has properly nested flux surfaces
        in a given toroidal plane

    Parameters
    ----------
    cR : ndarray, shape(RZ_transform.num_modes,)
        spectral coefficients of R
    cZ : ndarray, shape(RZ_transform.num_modes,)
        spectral coefficients of Z
    basis : FourierZernikeBasis
        spectral basis for R and Z
    L : int
        number of surfaces to check (Default value = 10)
    M : int
        number of poloidal angles to use for the test (Default value = 361)
    zeta : float
        toroidal plane to check (Default value = 0)

    Returns
    -------
    is_nested : bool
        whether or not the surfaces are nested

    """
    grid = LinearGrid(L=L, M=M, N=1, NFP=R_basis.NFP, endpoint=True)
    R_transf = Transform(grid, R_basis)
    Z_transf = Transform(grid, Z_basis)

    Rs = R_transf.transform(cR).reshape((L, -1), order='F')
    Zs = Z_transf.transform(cZ).reshape((L, -1), order='F')

    p = [matplotlib.path.Path(np.stack([R, Z]).T, closed=True) for R, Z in zip(Rs, Zs)]
    nested = np.all([p[i+1].contains_path(p[i]) for i in range(len(p)-1)])
    intersects = np.any([curve_self_intersects(R, Z) for R, Z in zip(Rs, Zs)])
    return nested and not intersects


def compute_force_error_nodes(cR, cZ, cP, cI, Psi_lcfs, R_transform,
                              Z_transform, P_transform, I_transform,
                              pres_ratio, zeta_ratio):
    """Computes force balance error at each node, in radial / helical components

    Parameters
    ----------
    cR : ndarray, shape(N_coeffs,)
        spectral coefficients of R
    cZ : ndarray, shape(N_coeffs,)
        spectral coefficients of Z
    cP : ndarray, shape(N_coeffs,)
        spectral coefficients of pressure
    cI : ndarray, shape(N_coeffs,)
        spectral coefficients of rotational transform
    Psi_lcfs : float
        total toroidal flux within the last closed flux surface
    R_transform : Transform
        transforms cR to physical space
    Z_transform : Transform
        transforms cZ to physical space
    P_transform : Transform
        transforms cP to physical space
    I_transform : Transform
        transforms cI to physical space
    pres_ratio : float
        fraction in range [0,1] of the full pressure profile to use
    zeta_ratio : float
        fraction in range [0,1] of the full toroidal (zeta) derivatives to use

    Returns
    -------
    F_rho : ndarray, shape(N_nodes,)
        radial force balance error at each node
    F_beta : ndarray, shape(N_nodes,)
        helical force balance error at each node

    """
    mu0 = 4*jnp.pi*1e-7
    axn = P_transform.grid.axis
    pres_r = P_transform.transform(cP, 1) * pres_ratio

    # compute fields components
    coord_der = compute_coordinate_derivatives(
        cR, cZ, R_transform, Z_transform, zeta_ratio)
    cov_basis = compute_covariant_basis(
        coord_der, R_transform.grid.axis, R_transform.derivs)
    jacobian = compute_jacobian(
        coord_der, cov_basis, R_transform.grid.axis, R_transform.derivs)
    con_basis = compute_contravariant_basis(
        coord_der, cov_basis, jacobian, R_transform.grid.axis)
    magnetic_field = compute_magnetic_field(
        cov_basis, jacobian, cI, Psi_lcfs, I_transform, R_transform.derivs)
    plasma_current = compute_plasma_current(
        coord_der, cov_basis, jacobian, magnetic_field, cI, I_transform)

    # force balance error components
    F_rho = jacobian['g']*(plasma_current['J^theta']*magnetic_field['B^zeta'] -
                           plasma_current['J^zeta']*magnetic_field['B^theta']) - pres_r
    F_beta = jacobian['g']*plasma_current['J^rho']

    # radial and helical directions
    beta = magnetic_field['B^zeta']*con_basis['e^theta'] - \
        magnetic_field['B^theta']*con_basis['e^zeta']
    radial = jnp.sqrt(
        con_basis['g^rr']) * jnp.sign(dot(con_basis['e^rho'], cov_basis['e_rho'], 0))
    helical = jnp.sqrt(con_basis['g^vv']*magnetic_field['B^zeta']**2 + con_basis['g^zz']*magnetic_field['B^theta']**2 - 2*con_basis['g^vz']*magnetic_field['B^theta']*magnetic_field['B^zeta']) \
        * jnp.sign(dot(beta, cov_basis['e_theta'], 0)) * jnp.sign(dot(beta, cov_basis['e_zeta'], 0))

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
        F_beta = put(F_beta, axn, plasma_current['J^rho'][axn])
        helical = put(helical, axn, jnp.sqrt(
            gsup_vv[axn]*magnetic_field['B^zeta'][axn]**2) * jnp.sign(magnetic_field['B^zeta'][axn]))

    # scalar errors
    f_rho = F_rho * radial
    f_beta = F_beta*helical

    # weight by local volume
    volumes = R_transform.grid.volumes
    vol = jacobian['g']*volumes[:, 0]*volumes[:, 1]*volumes[:, 2]
    if len(axn):
        r = R_transform.grid.nodes[:, 0]
        r1 = jnp.min(r[r != 0])  # value of r one step out from axis
        r1idx = jnp.where(r == r1)[0]
        # volume of axis is zero, but we want to account for nonzero volume in cell around axis
        vol = put(vol, axn, jnp.mean(
            jacobian['g'][r1idx])/2*volumes[axn, 0]*volumes[axn, 1]*volumes[axn, 2])
    f_rho = f_rho * vol
    f_beta = f_beta*vol

    return f_rho, f_beta


def compute_force_error_RphiZ(cR, cZ, cP, cI, Psi_lcfs, R_transform,
                              Z_transform, P_transform, I_transform,
                              pres_ratio, zeta_ratio):
    """Computes force balance error at each node, in R, phi, Z components

    Parameters
    ----------
    cR : ndarray, shape(N_coeffs,)
        spectral coefficients of R
    cZ : ndarray, shape(N_coeffs,)
        spectral coefficients of Z
    cP : ndarray, shape(N_coeffs,)
        spectral coefficients of pressure
    cI : ndarray, shape(N_coeffs,)
        spectral coefficients of rotational transform
    Psi_lcfs : float
        total toroidal flux within the last closed flux surface
    R_transform : Transform
        transforms cR to physical space
    Z_transform : Transform
        transforms cZ to physical space
    P_transform : Transform
        transforms cP to physical space
    I_transform : Transform
        transforms cI to physical space
    pres_ratio : float
        fraction in range [0,1] of the full pressure profile to use
    zeta_ratio : float
        fraction in range [0,1] of the full toroidal (zeta) derivatives to use

    Returns
    -------
    F_err : ndarray, shape(3,N_nodes,)
        F_R, F_phi, F_Z at each node

    """
    mu0 = 4*jnp.pi*1e-7
    axn = P_transform.grid.axis
    pres_r = P_transform.transform(cP, 1) * pres_ratio

    # compute fields components
    coord_der = compute_coordinate_derivatives(cR, cZ, R_transform, Z_transform)
    cov_basis = compute_covariant_basis(
        coord_der, R_transform.grid.axis, R_transform.derivs)
    jacobian = compute_jacobian(
        coord_der, cov_basis, R_transform.grid.axis, R_transform.derivs)
    con_basis = compute_contravariant_basis(
        coord_der, cov_basis, jacobian, R_transform.grid.axis)
    magnetic_field = compute_magnetic_field(
        cov_basis, jacobian, cI, Psi_lcfs, I_transform, R_transform.derivs)
    plasma_current = compute_plasma_current(
        coord_der, cov_basis, jacobian, magnetic_field, cI, I_transform)

    # helical basis vector
    beta = magnetic_field['B^zeta']*con_basis['e^theta'] - \
        magnetic_field['B^theta']*con_basis['e^zeta']

    # force balance error in radial and helical direction
    f_rho = mu0*(plasma_current['J^theta']*magnetic_field['B^zeta'] -
                 plasma_current['J^zeta']*magnetic_field['B^theta']) - mu0*pres_r
    f_beta = mu0*plasma_current['J^rho']

    F_err = f_rho * con_basis['grad_rho'] + f_beta * beta

    # weight by local volume
    volumes = R_transform.grid.volumes
    vol = jacobian['g']*volumes[:, 0]*volumes[:, 1]*volumes[:, 2]
    if len(axn):
        r = R_transform.grid.nodes[:, 0]
        r1 = jnp.min(r[r != 0])  # value of r one step out from axis
        r1idx = jnp.where(r == r1)[0]
        # volume of axis is zero, but we want to account for nonzero volume in cell around axis
        vol = put(vol, axn, jnp.mean(
            jacobian['g'][r1idx])/2*volumes[axn, 0]*volumes[axn, 1]*volumes[axn, 2])
    F_err = F_err*vol

    return F_err


def compute_force_error_RddotZddot(cR, cZ, cP, cI, Psi_lcfs, R_transform,
                                   Z_transform, P_transform, I_transform,
                                   pres_ratio, zeta_ratio):
    """Computes force balance error at each node, projected back onto zernike
    coefficients for R and Z.

    Parameters
    ----------
    cR : ndarray, shape(N_coeffs,)
        spectral coefficients of R
    cZ : ndarray, shape(N_coeffs,)
        spectral coefficients of Z
    cP : ndarray, shape(N_coeffs,)
        spectral coefficients of pressure
    cI : ndarray, shape(N_coeffs,)
        spectral coefficients of rotational transform
    Psi_lcfs : float
        total toroidal flux within the last closed flux surface
    R_transform : Transform
        transforms cR to physical space
    Z_transform : Transform
        transforms cZ to physical space
    P_transform : Transform
        transforms cP to physical space
    I_transform : Transform
        transforms cI to physical space
    pres_ratio : float
        fraction in range [0,1] of the full pressure profile to use
    zeta_ratio : float
        fraction in range [0,1] of the full toroidal (zeta) derivatives to use

    Returns
    -------
    cRddot : ndarray, shape(N_coeffs,)
        spectral coefficients for d^2R/dt^2
    cZddot : ndarray, shape(N_coeffs,)
        spectral coefficients for d^2Z/dt^2

    """
    coord_der = compute_coordinate_derivatives(cR, cZ, R_transform, Z_transform)
    F_err = compute_force_error_RphiZ(
        cR, cZ, cP, cI, Psi_lcfs, R_transform, Z_transform, P_transform, I_transform)
    num_nodes = R_transform.num_nodes  # FIXME: this needs symmetry

    AR = jnp.stack([jnp.ones(num_nodes), -coord_der['R_z'],
                    jnp.zeros(num_nodes)], axis=1)
    AZ = jnp.stack([jnp.zeros(num_nodes), -coord_der['Z_z'],
                    jnp.ones(num_nodes)], axis=1)
    A = jnp.stack([AR, AZ], axis=1)
    Rddot, Zddot = jnp.squeeze(jnp.matmul(A, F_err.T[:, :, jnp.newaxis])).T

    cRddot, cZddot = R_transform.fit(jnp.array([Rddot, Zddot]).T).T

    return cRddot, cZddot


def compute_accel_error_spectral(cR, cZ, cP, cI, Psi_lcfs, R_transform,
                                 Z_transform, P_transform, I_transform,
                                 pres_ratio, zeta_ratio):
    """Computes acceleration error in spectral space

    Parameters
    ----------
    cR : ndarray, shape(N_coeffs,)
        spectral coefficients of R
    cZ : ndarray, shape(N_coeffs,)
        spectral coefficients of Z
    cP : ndarray, shape(N_coeffs,)
        spectral coefficients of pressure
    cI : ndarray, shape(N_coeffs,)
        spectral coefficients of rotational transform
    Psi_lcfs : float
        total toroidal flux within the last closed flux surface
    R_transform : Transform
        transforms cR to physical space
    Z_transform : Transform
        transforms cZ to physical space
    P_transform : Transform
        transforms cP to physical space
    I_transform : Transform
        transforms cI to physical space
    pres_ratio : float
        fraction in range [0,1] of the full pressure profile to use
    zeta_ratio : float
        fraction in range [0,1] of the full toroidal (zeta) derivatives to use

    Returns
    -------
    cR_zz_err : ndarray, shape(N_coeffs,)
        error in cR_zz
    cZ_zz_err : ndarray, shape(N_coeffs,)
        error in cZ_zz

    """
    mu0 = 4*jnp.pi*1e-7
    r = R_transform.grid.nodes[:, 0]
    axn = P_transform.grid.axis

    presr = P_transform.transform(cP, 1) * pres_ratio
    iota = I_transform.transform(cI, 0)
    iotar = I_transform.transform(cI, 1)

    coord_der = compute_coordinate_derivatives(
        cR, cZ, R_transform, Z_transform, zeta_ratio)

    R_zz = -(Psi_lcfs**2*coord_der['R_r']**2*coord_der['Z_v']**2*coord_der['Z_z']**2*r**2 + Psi_lcfs**2*coord_der['R_v']**2*coord_der['Z_r']**2*coord_der['Z_z']**2*r**2 - Psi_lcfs**2*coord_der['R']**3*coord_der['R_r']*coord_der['Z_v']**2*r + Psi_lcfs**2*coord_der['R_r']**2*coord_der['Z_v']**4*r**2*iota**2 + Psi_lcfs**2*coord_der['R']**3*coord_der['R_rr']*coord_der['Z_v']**2*r**2 + Psi_lcfs**2*coord_der['R']**3*coord_der['R_vv']*coord_der['Z_r']**2*r**2 - Psi_lcfs**2*coord_der['R']**2*coord_der['R_r']**2*coord_der['Z_v']**2*r**2 - Psi_lcfs**2*coord_der['R']**2*coord_der['R_v']**2*coord_der['Z_r']**2*r**2 - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['Z_v']**4*r*iota**2 + Psi_lcfs**2*coord_der['R']**3*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']*r + Psi_lcfs**2*coord_der['R_v']**2*coord_der['Z_r']**2*coord_der['Z_v']**2*r**2*iota**2 - coord_der['R']**3*coord_der['R_r']**3*coord_der['Z_v']**4*mu0*jnp.pi**2*presr + Psi_lcfs**2*coord_der['R']*coord_der['R_rr']*coord_der['Z_v']**4*r**2*iota**2 + 2*Psi_lcfs**2*coord_der['R_r']**2*coord_der['Z_v']**3*coord_der['Z_z']*r**2*iota - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_z']**2*coord_der['Z_v']**2*r - Psi_lcfs**2*coord_der['R']**3*coord_der['R_r']*coord_der['Z_r']*coord_der['Z_vv']*r**2 + Psi_lcfs**2*coord_der['R']**3*coord_der['R_r']*coord_der['Z_rv']*coord_der['Z_v']*r**2 - 2*Psi_lcfs**2*coord_der['R']**3*coord_der['R_rv']*coord_der['Z_r']*coord_der['Z_v']*r**2 + Psi_lcfs**2*coord_der['R']**3*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_rv']*r**2 - Psi_lcfs**2*coord_der['R']**3*coord_der['R_v']*coord_der['Z_rr']*coord_der['Z_v']*r**2 - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['Z_v']**2*coord_der['Z_z']**2*r + Psi_lcfs**2*coord_der['R']*coord_der['R_rr']*coord_der['R_z']**2*coord_der['Z_v']**2*r**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_vv']*coord_der['R_z']**2*coord_der['Z_r']**2*r**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_rr']*coord_der['Z_v']**2*coord_der['Z_z']**2*r**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_vv']*coord_der['Z_r']**2*coord_der['Z_z']**2*r**2 - 2*Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['Z_v']**3*coord_der['Z_z']*r*iota + Psi_lcfs**2*coord_der['R']*coord_der['R_rr']*coord_der['R_v']**2*coord_der['Z_v']**2*r**2*iota**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_vv']*coord_der['Z_v']**2*r**2*iota**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_vv']*coord_der['Z_r']**2*coord_der['Z_v']**2*r**2*iota**2 - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['Z_v']**3*coord_der['Z_z']*r**2*iotar + Psi_lcfs**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']**3*r*iota**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_v']**3*coord_der['Z_r']*coord_der['Z_v']*r*iota**2 - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['Z_v']**3*coord_der['Z_rz']*r**2*iota + 2*Psi_lcfs**2*coord_der['R']*coord_der['R_rr']*coord_der['Z_v']**3*coord_der['Z_z']*r**2*iota - Psi_lcfs**2*coord_der['R']*coord_der['R_v']**3*coord_der['Z_r']*coord_der['Z_rz']*r**2*iota + Psi_lcfs**2*coord_der['R']*coord_der['R_v']*coord_der['R_z']**2*coord_der['Z_r']*coord_der['Z_v']*r + Psi_lcfs**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']**2*r + coord_der['R']**3*coord_der['R_v']**3*coord_der['Z_r']**3*coord_der['Z_v']*mu0*jnp.pi**2*presr - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['Z_v']**4*r**2*iota*iotar - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**2*coord_der['Z_v']**2*r*iota**2 + 2*Psi_lcfs**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_vz']*coord_der['Z_v']**2*r**2*iota - 2*Psi_lcfs**2*coord_der['R']*coord_der['R_rv']*coord_der['Z_r']*coord_der['Z_v']**3*r**2*iota**2 - Psi_lcfs**2*coord_der['R']*coord_der['R_v']*coord_der['Z_rr']*coord_der['Z_v']**3*r**2*iota**2 - Psi_lcfs**2*coord_der['R']*coord_der['R_v']**3*coord_der['Z_rr']*coord_der['Z_v']*r**2*iota**2 - 2*Psi_lcfs**2*coord_der['R_r']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']**3*r**2*iota**2 + 2*Psi_lcfs**2*coord_der['R_v']**2*coord_der['Z_r']**2*coord_der['Z_v']*coord_der['Z_z']*r**2*iota - 2*Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_z']*coord_der['R_rz']*coord_der['Z_v']**2*r**2 - 2*Psi_lcfs**2*coord_der['R']*coord_der['R_v']*coord_der['R_z']*coord_der['R_vz']*coord_der['Z_r']**2*r**2 + 2*Psi_lcfs**2*coord_der['R']**2*coord_der['R_r']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']*r**2 - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_z']**2*coord_der['Z_r']*coord_der['Z_vv']*r**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_z']**2*coord_der['Z_rv']*coord_der['Z_v']*r**2 - 2*Psi_lcfs**2*coord_der['R']*coord_der['R_rv']*coord_der['R_z']**2*coord_der['Z_r']*coord_der['Z_v']*r**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_v']*coord_der['R_z']**2*coord_der['Z_r']*coord_der['Z_rv']*r**2 - Psi_lcfs**2*coord_der['R']*coord_der['R_v']*coord_der['R_z']**2*coord_der['Z_rr']*coord_der['Z_v']*r**2 - Psi_lcfs**2*coord_der['R']*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_r']*coord_der['Z_rz']*r**2 - Psi_lcfs**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_z']*coord_der['Z_v']*coord_der['Z_vz']*r**2 - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['Z_r']*coord_der['Z_vv']*coord_der['Z_z']**2*r**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['Z_rv']*coord_der['Z_v']*coord_der['Z_z']**2*r**2 - 2*Psi_lcfs**2*coord_der['R']*coord_der['R_rv']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']**2*r**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_rv']*coord_der['Z_z']**2*r**2 - Psi_lcfs**2*coord_der['R']*coord_der['R_v']*coord_der['Z_rr']*coord_der['Z_v']*coord_der['Z_z']**2*r**2 - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['Z_v']**2*coord_der['Z_z']*coord_der['Z_rz']*r**2 -
             Psi_lcfs**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']**2*coord_der['Z_z']*coord_der['Z_vz']*r**2 - 2*Psi_lcfs**2*coord_der['R_r']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']**2*r**2 - 2*Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_rv']*coord_der['R_v']*coord_der['Z_v']**2*r**2*iota**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']**3*r**2*iota*iotar + Psi_lcfs**2*coord_der['R']*coord_der['R_v']**3*coord_der['Z_r']*coord_der['Z_v']*r**2*iota*iotar + 2*Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**2*coord_der['Z_rv']*coord_der['Z_v']*r**2*iota**2 - Psi_lcfs**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_v']*coord_der['Z_v']*coord_der['Z_vv']*r**2*iota**2 + 2*Psi_lcfs**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_rv']*coord_der['Z_v']**2*r**2*iota**2 - Psi_lcfs**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']**2*coord_der['Z_v']*coord_der['Z_vv']*r**2*iota**2 - 3*coord_der['R']**3*coord_der['R_r']*coord_der['R_v']**2*coord_der['Z_r']**2*coord_der['Z_v']**2*mu0*jnp.pi**2*presr - 2*Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_v']**2*r*iota + 2*Psi_lcfs**2*coord_der['R']*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_r']*coord_der['Z_v']*r*iota + 2*Psi_lcfs**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']**2*coord_der['Z_z']*r*iota - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**2*coord_der['Z_v']**2*r**2*iota*iotar - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_v']**2*r**2*iotar + Psi_lcfs**2*coord_der['R']*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_r']*coord_der['Z_v']*r**2*iotar + Psi_lcfs**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']**2*coord_der['Z_z']*r**2*iotar - 2*Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_rv']*coord_der['R_z']*coord_der['Z_v']**2*r**2*iota - 2*Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_rz']*coord_der['Z_v']**2*r**2*iota + 2*Psi_lcfs**2*coord_der['R']*coord_der['R_rr']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_v']**2*r**2*iota + Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_vz']*r**2*iota + Psi_lcfs**2*coord_der['R']*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_r']*coord_der['Z_rv']*r**2*iota + Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**2*coord_der['Z_v']*coord_der['Z_rz']*r**2*iota - 2*Psi_lcfs**2*coord_der['R']*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_rr']*coord_der['Z_v']*r**2*iota + 2*Psi_lcfs**2*coord_der['R']*coord_der['R_v']**2*coord_der['R_rz']*coord_der['Z_r']*coord_der['Z_v']*r**2*iota - Psi_lcfs**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_v']*coord_der['Z_v']*coord_der['Z_vz']*r**2*iota - Psi_lcfs**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_z']*coord_der['Z_v']*coord_der['Z_vv']*r**2*iota + Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['Z_r']*coord_der['Z_v']**2*coord_der['Z_vz']*r**2*iota + Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['Z_rv']*coord_der['Z_v']**2*coord_der['Z_z']*r**2*iota - 4*Psi_lcfs**2*coord_der['R']*coord_der['R_rv']*coord_der['Z_r']*coord_der['Z_v']**2*coord_der['Z_z']*r**2*iota + Psi_lcfs**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']**2*coord_der['Z_rz']*r**2*iota - 2*Psi_lcfs**2*coord_der['R']*coord_der['R_v']*coord_der['Z_rr']*coord_der['Z_v']**2*coord_der['Z_z']*r**2*iota - Psi_lcfs**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']**2*coord_der['Z_v']*coord_der['Z_vz']*r**2*iota - Psi_lcfs**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']**2*coord_der['Z_vv']*coord_der['Z_z']*r**2*iota + 2*Psi_lcfs**2*coord_der['R']*coord_der['R_vv']*coord_der['Z_r']**2*coord_der['Z_v']*coord_der['Z_z']*r**2*iota - 4*Psi_lcfs**2*coord_der['R_r']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']**2*coord_der['Z_z']*r**2*iota + Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_r']*coord_der['Z_vz']*r**2 + 2*Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_z']*coord_der['R_vz']*coord_der['Z_r']*coord_der['Z_v']*r**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_v']*coord_der['Z_rz']*r**2 + 2*Psi_lcfs**2*coord_der['R']*coord_der['R_v']*coord_der['R_z']*coord_der['R_rz']*coord_der['Z_r']*coord_der['Z_v']*r**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']*coord_der['Z_vz']*r**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']*coord_der['Z_rz']*r**2 + 3*coord_der['R']**3*coord_der['R_r']**2*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']**3*mu0*jnp.pi**2*presr - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_r']*coord_der['Z_vv']*r**2*iota + 3*Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_rv']*coord_der['Z_v']*r**2*iota - 2*Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_vz']*coord_der['Z_r']*coord_der['Z_v']*r**2*iota + 2*Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_vv']*coord_der['R_z']*coord_der['Z_r']*coord_der['Z_v']*r**2*iota - 2*Psi_lcfs**2*coord_der['R']*coord_der['R_rv']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_r']*coord_der['Z_v']*r**2*iota - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_vv']*coord_der['Z_z']*r**2*iota + 3*Psi_lcfs**2*coord_der['R']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_rv']*coord_der['Z_v']*coord_der['Z_z']*r**2*iota) / (Psi_lcfs**2*coord_der['R']*r**2*(coord_der['R_r']*coord_der['Z_v'] - coord_der['R_v']*coord_der['Z_r'])**2)

    Z_zz = (Psi_lcfs**2*coord_der['R']**3*coord_der['R_v']**2*coord_der['Z_r']*r - Psi_lcfs**2*coord_der['R']**3*coord_der['R_v']**2*coord_der['Z_rr']*r**2 - Psi_lcfs**2*coord_der['R']**3*coord_der['R_r']**2*coord_der['Z_vv']*r**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_v']**4*coord_der['Z_r']*r*iota**2 - Psi_lcfs**2*coord_der['R']**3*coord_der['R_r']*coord_der['R_v']*coord_der['Z_v']*r + coord_der['R']**3*coord_der['R_v']**4*coord_der['Z_r']**3*mu0*jnp.pi**2*presr - Psi_lcfs**2*coord_der['R']*coord_der['R_v']**4*coord_der['Z_rr']*r**2*iota**2 + Psi_lcfs**2*coord_der['R_r']**2*coord_der['R_z']*coord_der['Z_v']**3*r**2*iota + Psi_lcfs**2*coord_der['R_v']**3*coord_der['Z_r']**2*coord_der['Z_z']*r**2*iota - Psi_lcfs**2*coord_der['R']**3*coord_der['R_r']*coord_der['R_rv']*coord_der['Z_v']*r**2 + 2*Psi_lcfs**2*coord_der['R']**3*coord_der['R_r']*coord_der['R_v']*coord_der['Z_rv']*r**2 + Psi_lcfs**2*coord_der['R']**3*coord_der['R_r']*coord_der['R_vv']*coord_der['Z_r']*r**2 - Psi_lcfs**2*coord_der['R']**3*coord_der['R_rv']*coord_der['R_v']*coord_der['Z_r']*r**2 + Psi_lcfs**2*coord_der['R']**3*coord_der['R_rr']*coord_der['R_v']*coord_der['Z_v']*r**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_v']**2*coord_der['R_z']**2*coord_der['Z_r']*r + Psi_lcfs**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_z']**2*r + Psi_lcfs**2*coord_der['R_r']**2*coord_der['R_v']*coord_der['Z_v']**3*r**2*iota**2 + Psi_lcfs**2*coord_der['R_v']**3*coord_der['Z_r']**2*coord_der['Z_v']*r**2*iota**2 - Psi_lcfs**2*coord_der['R']*coord_der['R_v']**2*coord_der['R_z']**2*coord_der['Z_rr']*r**2 - Psi_lcfs**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_z']**2*coord_der['Z_vv']*r**2 - Psi_lcfs**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_rr']*coord_der['Z_z']**2*r**2 - Psi_lcfs**2*coord_der['R']*coord_der['R_r']**2*coord_der['Z_vv']*coord_der['Z_z']**2*r**2 + Psi_lcfs**2*coord_der['R_r']**2*coord_der['R_z']*coord_der['Z_v']**2*coord_der['Z_z']*r**2 + Psi_lcfs**2*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_r']**2*coord_der['Z_z']*r**2 + 2*Psi_lcfs**2*coord_der['R']*coord_der['R_v']**3*coord_der['R_z']*coord_der['Z_r']*r*iota - Psi_lcfs**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_v']**2*coord_der['Z_vv']*r**2*iota**2 - Psi_lcfs**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_rr']*coord_der['Z_v']**2*r**2*iota**2 - Psi_lcfs**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']**2*coord_der['Z_vv']*r**2*iota**2 - 2*Psi_lcfs**2*coord_der['R_r']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_v']**2*r**2*iota**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_v']**3*coord_der['R_z']*coord_der['Z_r']*r**2*iotar - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_v']**3*r*iota**2 - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**3*coord_der['Z_v']*r*iota**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_rz']*coord_der['Z_v']**3*r**2*iota - 2*Psi_lcfs**2*coord_der['R']*coord_der['R_v']**3*coord_der['R_z']*coord_der['Z_rr']*r**2*iota + Psi_lcfs**2*coord_der['R']*coord_der['R_v']**3*coord_der['R_rz']*coord_der['Z_r']*r**2*iota - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']**2*coord_der['Z_v']*r - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_v']*coord_der['Z_z']**2*r - coord_der['R']**3*coord_der['R_r']**3*coord_der['R_v']*coord_der['Z_v']**3*mu0*jnp.pi**2*presr + Psi_lcfs**2*coord_der['R']*coord_der['R_v']**4*coord_der['Z_r']*r**2*iota*iotar + 2*Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**3*coord_der['Z_rv']*r**2*iota**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_rr']*coord_der['R_v']*coord_der['Z_v']**3*r**2*iota**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_rr']*coord_der['R_v']**3*coord_der['Z_v']*r**2*iota**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_v']**2*r*iota**2 - 2*Psi_lcfs**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']**2*coord_der['Z_vz']*r**2*iota + Psi_lcfs**2*coord_der['R_r']**2*coord_der['R_v']*coord_der['Z_v']**2*coord_der['Z_z']*r**2*iota + Psi_lcfs**2*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_r']**2*coord_der['Z_v']*r**2*iota - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_rv']*coord_der['R_z']**2*coord_der['Z_v']*r**2 + 2*Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']**2*coord_der['Z_rv']*r**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_vv']*coord_der['R_z']**2*coord_der['Z_r']*r**2 - Psi_lcfs**2*coord_der['R']*coord_der['R_rv']*coord_der['R_v']*coord_der['R_z']**2*coord_der['Z_r']*r**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_rr']*coord_der['R_v']*coord_der['R_z']**2*coord_der['Z_v']*r**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_v']**2*coord_der['R_z']*coord_der['R_rz']*coord_der['Z_r']*r**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_z']*coord_der['R_vz']*coord_der['Z_v']*r**2 - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_rv']*coord_der['Z_v']*coord_der['Z_z']**2*r**2 + 2*Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_rv']*coord_der['Z_z']**2*r**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_vv']*coord_der['Z_r']*coord_der['Z_z']**2*r**2 - Psi_lcfs**2*coord_der['R']*coord_der['R_rv']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_z']**2*r**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_rr']*coord_der['R_v']*coord_der['Z_v']*coord_der['Z_z']**2*r**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_rz']*coord_der['Z_v']**2*coord_der['Z_z']*r**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_v']*coord_der['R_vz']*coord_der['Z_r']**2*coord_der['Z_z']*r**2 + 2*Psi_lcfs**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_z']*coord_der['Z_rz']*r**2 + 2*Psi_lcfs**2*coord_der['R']*coord_der['R_r']**2*coord_der['Z_v']*coord_der['Z_z'] *
            coord_der['Z_vz']*r**2 - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_v']**3*r**2*iota*iotar - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**3*coord_der['Z_v']*r**2*iota*iotar - 2*Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_rv']*coord_der['R_v']**2*coord_der['Z_v']*r**2*iota**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_v']*coord_der['R_vv']*coord_der['Z_v']*r**2*iota**2 - 2*Psi_lcfs**2*coord_der['R']*coord_der['R_rv']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']**2*r**2*iota**2 + Psi_lcfs**2*coord_der['R']*coord_der['R_v']*coord_der['R_vv']*coord_der['Z_r']**2*coord_der['Z_v']*r**2*iota**2 + 2*Psi_lcfs**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_rv']*coord_der['Z_v']*r**2*iota**2 + 3*coord_der['R']**3*coord_der['R_r']**2*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_v']**2*mu0*jnp.pi**2*presr - 2*Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_v']*r*iota - 2*Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_v']**2*coord_der['Z_z']*r*iota + 2*Psi_lcfs**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']*r*iota + Psi_lcfs**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_v']**2*r**2*iota*iotar - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_v']*r**2*iotar - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_v']**2*coord_der['Z_z']*r**2*iotar + Psi_lcfs**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']*r**2*iotar + 4*Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_rv']*r**2*iota - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**2*coord_der['R_vz']*coord_der['Z_r']*r**2*iota - Psi_lcfs**2*coord_der['R']*coord_der['R_rv']*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_r']*r**2*iota - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']**2*coord_der['R_rz']*coord_der['Z_v']*r**2*iota + 2*Psi_lcfs**2*coord_der['R']*coord_der['R_rr']*coord_der['R_v']**2*coord_der['R_z']*coord_der['Z_v']*r**2*iota - 2*Psi_lcfs**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_v']*coord_der['R_z']*coord_der['Z_vv']*r**2*iota + Psi_lcfs**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_v']*coord_der['R_vz']*coord_der['Z_v']*r**2*iota + Psi_lcfs**2*coord_der['R']*coord_der['R_r']**2*coord_der['R_vv']*coord_der['R_z']*coord_der['Z_v']*r**2*iota - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_rv']*coord_der['Z_v']**2*coord_der['Z_z']*r**2*iota - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_vz']*coord_der['Z_r']*coord_der['Z_v']**2*r**2*iota - 2*Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_v']**2*coord_der['Z_rz']*r**2*iota + 2*Psi_lcfs**2*coord_der['R']*coord_der['R_rr']*coord_der['R_v']*coord_der['Z_v']**2*coord_der['Z_z']*r**2*iota - Psi_lcfs**2*coord_der['R']*coord_der['R_v']*coord_der['R_rz']*coord_der['Z_r']*coord_der['Z_v']**2*r**2*iota + Psi_lcfs**2*coord_der['R']*coord_der['R_v']*coord_der['R_vv']*coord_der['Z_r']**2*coord_der['Z_z']*r**2*iota + Psi_lcfs**2*coord_der['R']*coord_der['R_v']*coord_der['R_vz']*coord_der['Z_r']**2*coord_der['Z_v']*r**2*iota - 2*Psi_lcfs**2*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_r']*coord_der['Z_v']**2*r**2*iota + 2*Psi_lcfs**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_rv']*coord_der['Z_z']*r**2*iota + 2*Psi_lcfs**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_rz']*r**2*iota - 2*Psi_lcfs**2*coord_der['R']*coord_der['R_v']**2*coord_der['Z_rr']*coord_der['Z_v']*coord_der['Z_z']*r**2*iota - 2*Psi_lcfs**2*coord_der['R_r']*coord_der['R_v']**2*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']*r**2*iota - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']*coord_der['R_vz']*coord_der['Z_r']*r**2 - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']*coord_der['R_rz']*coord_der['Z_v']*r**2 - 2*Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_z']*coord_der['Z_vz']*r**2 - Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_vz']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']*r**2 - 2*Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_v']*coord_der['Z_z']*coord_der['Z_rz']*r**2 - Psi_lcfs**2*coord_der['R']*coord_der['R_v']*coord_der['R_rz']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']*r**2 - 2*Psi_lcfs**2*coord_der['R_r']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']*r**2 - 3*coord_der['R']**3*coord_der['R_r']*coord_der['R_v']**3*coord_der['Z_r']**2*coord_der['Z_v']*mu0*jnp.pi**2*presr - 3*Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_rv']*coord_der['R_v']*coord_der['R_z']*coord_der['Z_v']*r**2*iota + Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['R_vv']*coord_der['R_z']*coord_der['Z_r']*r**2*iota + 2*Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_vz']*r**2*iota - 2*Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_vv']*coord_der['Z_z']*r**2*iota + 2*Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_v']*coord_der['Z_rv']*coord_der['Z_v']*coord_der['Z_z']*r**2*iota + Psi_lcfs**2*coord_der['R']*coord_der['R_r']*coord_der['R_vv']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']*r**2*iota - 3*Psi_lcfs**2*coord_der['R']*coord_der['R_rv']*coord_der['R_v']*coord_der['Z_r']*coord_der['Z_v']*coord_der['Z_z']*r**2*iota) / (Psi_lcfs**2*coord_der['R']*r**2*(coord_der['R_r']*coord_der['Z_v'] - coord_der['R_v']*coord_der['Z_r'])**2)

    if len(axn):
        R_zz = put(R_zz, axn, (24*Psi_lcfs**2*coord_der['R_rv'][axn]**2*coord_der['Z_r'][axn]**2*coord_der['R'][axn]**2 - 24*Psi_lcfs**2*coord_der['Z_z'][axn]**2*coord_der['Z_rv'][axn]**2*coord_der['R_r'][axn]**2 - 24*Psi_lcfs**2*coord_der['R_rv'][axn]**2*coord_der['Z_z'][axn]**2*coord_der['Z_r'][axn]**2 + 24*Psi_lcfs**2*coord_der['Z_rv'][axn]**2*coord_der['R_r'][axn]**2*coord_der['R'][axn]**2 - 24*Psi_lcfs**2*coord_der['R_rr'][axn]*coord_der['Z_rv'][axn]**2*coord_der['R'][axn]**3 - 12*Psi_lcfs**2*coord_der['R_rrvv'][axn]*coord_der['Z_r'][axn]**2*coord_der['R'][axn]**3 + 24*Psi_lcfs**2*coord_der['R_z'][axn]**2*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['Z_r'][axn] - 24*Psi_lcfs**2*coord_der['Z_z'][axn]**2*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]**2 - 12*Psi_lcfs**2*coord_der['R_rrvv'][axn]*coord_der['Z_z'][axn]**2*coord_der['Z_r'][axn]**2*coord_der['R'][axn] + 24*Psi_lcfs**2*coord_der['Z_z'][axn]**2*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['Z_r'][axn] - 72*Psi_lcfs**2*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]**2*coord_der['R'][axn]**2 + 72*Psi_lcfs**2*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['Z_r'][axn]*coord_der['R'][axn]**2 + 12*Psi_lcfs**2*coord_der['Z_rrvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]**3 + 24*Psi_lcfs**2*coord_der['R_rr'][axn]*coord_der['Z_rvv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]**3 + 24*Psi_lcfs**2*coord_der['R_rv'][axn]*coord_der['Z_rr'][axn]*coord_der['Z_rv'][axn]*coord_der['R'][axn]**3 - 12*Psi_lcfs**2*coord_der['R_rv'][axn]*coord_der['Z_rrv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]**3 - 48*Psi_lcfs**2*coord_der['Z_rr'][axn]*coord_der['R_rvv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]**3 + 24*Psi_lcfs**2*coord_der['Z_rr'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn]**3 + 24*Psi_lcfs**2*coord_der['Z_rv'][axn]*coord_der['R_rrv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]**3 - 12*Psi_lcfs**2*coord_der['Z_rv'][axn]*coord_der['Z_rrv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn]**3 - 24*Psi_lcfs**2*coord_der['R_z'][axn]**2*coord_der['R_rr'][axn]*coord_der['Z_rv'][axn]**2*coord_der['R'][axn] - 24*Psi_lcfs**2*coord_der['R_rr'][axn]*coord_der['Z_z'][axn]**2*coord_der['Z_rv'][axn]**2*coord_der['R'][axn] - 24*Psi_lcfs**2*coord_der['R_z'][axn]**2*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]**2 - 12*Psi_lcfs**2*coord_der['R_rrvv'][axn]*coord_der['R_z'][axn]**2*coord_der['Z_r'][axn]**2*coord_der['R'][axn] + 24*Psi_lcfs**2*iota[axn]*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]**3*coord_der['R_r'][axn]*coord_der['R'][axn] + 12*Psi_lcfs**2*coord_der['Z_rrvv'][axn]*coord_der['R_z'][axn]**2*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 48*Psi_lcfs**2*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['R_rvz'][axn]*coord_der['Z_r'][axn]**2*coord_der['R'][axn] - 48*Psi_lcfs**2*coord_der['R_z'][axn]*coord_der['R_rz'][axn]*coord_der['R_rvv'][axn]*coord_der['Z_r'][axn]**2*coord_der['R'][axn] + 24*Psi_lcfs**2*coord_der['R_z'][axn]**2*coord_der['R_rr'][axn]*coord_der['Z_rvv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_lcfs**2*coord_der['R_z'][axn]**2*coord_der['R_rv'][axn]*coord_der['Z_rr'][axn]*coord_der['Z_rv'][axn]*coord_der['R'][axn] - 12*Psi_lcfs**2*coord_der['R_z'][axn]**2*coord_der['R_rv'][axn]*coord_der['Z_rrv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_lcfs**2*coord_der['R_z'][axn]**2*coord_der['Z_rr'][axn]*coord_der['R_rvv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_lcfs**2*coord_der['R_z'][axn]**2*coord_der['Z_rr'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] + 24*Psi_lcfs**2*coord_der['R_z'][axn]**2*coord_der['Z_rv'][axn]*coord_der['R_rrv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 12*Psi_lcfs**2*coord_der['R_z'][axn]**2*coord_der['Z_rv'][axn]*coord_der['Z_rrv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] + 48*Psi_lcfs**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]**2*coord_der['Z_rv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn] + 12*Psi_lcfs**2*coord_der['Z_rrvv'][axn]*coord_der['Z_z'][axn]**2*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_lcfs**2*coord_der['R_z'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_rvz'][axn]*coord_der['R_r'][axn]**2*coord_der['R']
                               [axn] + 24*Psi_lcfs**2*coord_der['R_rr'][axn]*coord_der['Z_z'][axn]**2*coord_der['Z_rvv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_lcfs**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]**2*coord_der['Z_rr'][axn]*coord_der['Z_rv'][axn]*coord_der['R'][axn] - 12*Psi_lcfs**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]**2*coord_der['Z_rrv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_lcfs**2*coord_der['Z_z'][axn]**2*coord_der['Z_rr'][axn]*coord_der['R_rvv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_lcfs**2*coord_der['Z_z'][axn]**2*coord_der['Z_rr'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] + 24*Psi_lcfs**2*coord_der['Z_z'][axn]**2*coord_der['Z_rv'][axn]*coord_der['R_rrv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 12*Psi_lcfs**2*coord_der['Z_z'][axn]**2*coord_der['Z_rv'][axn]*coord_der['Z_rrv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] + 24*Psi_lcfs**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['Z_rvz'][axn]*coord_der['Z_r'][axn]**2*coord_der['R'][axn] - 48*Psi_lcfs**2*coord_der['Z_z'][axn]*coord_der['Z_rz'][axn]*coord_der['R_rvv'][axn]*coord_der['Z_r'][axn]**2*coord_der['R'][axn] - 48*Psi_lcfs**2*coord_der['R_rv'][axn]*coord_der['Z_rv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]**2 + 48*Psi_lcfs**2*coord_der['R_z'][axn]*coord_der['R_rz'][axn]*coord_der['Z_rv'][axn]**2*coord_der['R_r'][axn]*coord_der['R'][axn] + 24*Psi_lcfs**2*coord_der['R_z'][axn]*coord_der['R_rv'][axn]**2*coord_der['Z_rz'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_lcfs**2*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]**2*coord_der['Z_rz'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] + 24*Psi_lcfs**2*iota[axn]*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['Z_rv'][axn]**2*coord_der['R_r'][axn]*coord_der['R'][axn] - 24*Psi_lcfs**2*iota[axn]*coord_der['R_z'][axn]*coord_der['R_rv'][axn]**2*coord_der['Z_rv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 24*Psi_lcfs**2*iota[axn]*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]**2*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_lcfs**2*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['R_rz'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 24*Psi_lcfs**2*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_rz'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] - 24*Psi_lcfs**2*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['Z_rvz'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 48*Psi_lcfs**2*coord_der['R_z'][axn]*coord_der['R_rz'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_lcfs**2*coord_der['R_z'][axn]*coord_der['Z_rv'][axn]*coord_der['R_rvz'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 24*Psi_lcfs**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_rz'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 24*Psi_lcfs**2*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_rvz'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 48*Psi_lcfs**2*coord_der['Z_z'][axn]*coord_der['Z_rz'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_lcfs**2*iota[axn]*coord_der['R_z'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['R'][axn] + 24*Psi_lcfs**2*iota[axn]*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['Z_rvv'][axn]*coord_der['Z_r'][axn]**2*coord_der['R'][axn] - 48*Psi_lcfs**2*iota[axn]*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]*coord_der['R_rvv'][axn]*coord_der['Z_r'][axn]**2*coord_der['R'][axn] + 24*Psi_lcfs**2*iota[axn]*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_lcfs**2*iota[axn]*coord_der['R_z'][axn]*coord_der['Z_rv'][axn]*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_lcfs**2*iota[axn]*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]) / (24*Psi_lcfs**2*(coord_der['R_rv'][axn]*coord_der['Z_r'][axn] - coord_der['Z_rv'][axn]*coord_der['R_r'][axn])**2*coord_der['R'][axn]))

        Z_zz = put(Z_zz, axn, (24*Psi_lcfs**2*coord_der['Z_z'][axn]**2*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['Z_r'][axn] - 24*Psi_lcfs**2*coord_der['Z_z'][axn]**2*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]**3 - 24*Psi_lcfs**2*coord_der['R_rv'][axn]**2*coord_der['Z_rr'][axn]*coord_der['R'][axn]**3 - 72*Psi_lcfs**2*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]**3*coord_der['R'][axn]**2 - 12*Psi_lcfs**2*coord_der['Z_rrvv'][axn]*coord_der['R_r'][axn]**2*coord_der['R'][axn]**3 - 24*Psi_lcfs**2*coord_der['R_z'][axn]**2*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]**3 - 12*Psi_lcfs**2*coord_der['Z_rrvv'][axn]*coord_der['Z_z'][axn]**2*coord_der['R_r'][axn]**2*coord_der['R'][axn] + 72*Psi_lcfs**2*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['Z_r'][axn]*coord_der['R'][axn]**2 + 12*Psi_lcfs**2*coord_der['R_rrvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]**3 + 24*Psi_lcfs**2*coord_der['R_rr'][axn]*coord_der['R_rv'][axn]*coord_der['Z_rv'][axn]*coord_der['R'][axn]**3 + 24*Psi_lcfs**2*coord_der['R_rr'][axn]*coord_der['R_rvv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]**3 - 48*Psi_lcfs**2*coord_der['R_rr'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn]**3 - 12*Psi_lcfs**2*coord_der['R_rv'][axn]*coord_der['R_rrv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]**3 + 24*Psi_lcfs**2*coord_der['R_rv'][axn]*coord_der['Z_rrv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn]**3 + 24*Psi_lcfs**2*coord_der['Z_rr'][axn]*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn]**3 - 12*Psi_lcfs**2*coord_der['Z_rv'][axn]*coord_der['R_rrv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn]**3 - 24*Psi_lcfs**2*coord_der['R_z'][axn]**2*coord_der['R_rv'][axn]**2*coord_der['Z_rr'][axn]*coord_der['R'][axn] - 24*Psi_lcfs**2*coord_der['R_rv'][axn]**2*coord_der['Z_z'][axn]**2*coord_der['Z_rr'][axn]*coord_der['R'][axn] + 24*Psi_lcfs**2*coord_der['R_z'][axn]**2*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['Z_r'][axn] + 24*Psi_lcfs**2*coord_der['R_z'][axn]*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]**2*coord_der['R_r'][axn]**2 + 24*Psi_lcfs**2*coord_der['R_z'][axn]*coord_der['R_rv'][axn]**2*coord_der['Z_z'][axn]*coord_der['Z_r'][axn]**2 - 12*Psi_lcfs**2*coord_der['Z_rrvv'][axn]*coord_der['R_z'][axn]**2*coord_der['R_r'][axn]**2*coord_der['R'][axn] + 24*Psi_lcfs**2*iota[axn]*coord_der['R_z'][axn]*coord_der['R_rv'][axn]**3*coord_der['Z_r'][axn]*coord_der['R'][axn] + 12*Psi_lcfs**2*coord_der['R_rrvv'][axn]*coord_der['R_z'][axn]**2*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_lcfs**2*coord_der['R_z'][axn]**2*coord_der['R_rr'][axn]*coord_der['R_rv'][axn]*coord_der['Z_rv'][axn]*coord_der['R'][axn] + 24*Psi_lcfs**2*coord_der['R_z'][axn]**2*coord_der['R_rr'][axn]*coord_der['R_rvv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_lcfs**2*coord_der['R_z'][axn]**2*coord_der['R_rr'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] - 12*Psi_lcfs**2*coord_der['R_z'][axn]**2*coord_der['R_rv'][axn]*coord_der['R_rrv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_lcfs**2*coord_der['R_z'][axn]**2*coord_der['R_rv'][axn]*coord_der['Z_rrv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] + 24*Psi_lcfs**2*coord_der['R_z'][axn]**2*coord_der['Z_rr'][axn]*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] - 12*Psi_lcfs**2*coord_der['R_z'][axn]**2*coord_der['Z_rv'][axn]*coord_der['R_rrv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] + 12*Psi_lcfs**2*coord_der['R_rrvv'][axn]*coord_der['Z_z'][axn]**2*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_lcfs**2*coord_der['R_z'][axn]*coord_der['R_rz'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['R'][axn] + 24*Psi_lcfs**2*coord_der['R_z'][axn]*coord_der['Z_rv'][axn]*coord_der['R_rvz'][axn]*coord_der['R_r'][axn]**2*coord_der['R'][axn] + 24*Psi_lcfs**2*coord_der['R_rr'][axn]*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]**2*coord_der['Z_rv'][axn]*coord_der['R'][axn] + 24*Psi_lcfs**2*coord_der['R_rr'][axn]*coord_der['Z_z'][axn]**2*coord_der['R_rvv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_lcfs**2*coord_der['R_rr'][axn]
                               * coord_der['Z_z'][axn]**2*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] - 12*Psi_lcfs**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]**2*coord_der['R_rrv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_lcfs**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]**2*coord_der['Z_rrv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] + 24*Psi_lcfs**2*coord_der['Z_z'][axn]**2*coord_der['Z_rr'][axn]*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] - 12*Psi_lcfs**2*coord_der['Z_z'][axn]**2*coord_der['Z_rv'][axn]*coord_der['R_rrv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] + 24*Psi_lcfs**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['R_rvz'][axn]*coord_der['Z_r'][axn]**2*coord_der['R'][axn] + 48*Psi_lcfs**2*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_rvz'][axn]*coord_der['R_r'][axn]**2*coord_der['R'][axn] - 48*Psi_lcfs**2*coord_der['Z_z'][axn]*coord_der['Z_rz'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['R'][axn] + 24*Psi_lcfs**2*coord_der['R_z'][axn]*coord_der['R_rv'][axn]**2*coord_der['R_rz'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_lcfs**2*coord_der['Z_z'][axn]*coord_der['R_rz'][axn]*coord_der['Z_rv'][axn]**2*coord_der['R_r'][axn]*coord_der['R'][axn] + 48*Psi_lcfs**2*coord_der['R_rv'][axn]**2*coord_der['Z_z'][axn]*coord_der['Z_rz'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_lcfs**2*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn] - 24*Psi_lcfs**2*iota[axn]*coord_der['R_z'][axn]*coord_der['R_rv'][axn]**2*coord_der['Z_rv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] - 24*Psi_lcfs**2*iota[axn]*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]**2*coord_der['R_r'][axn]*coord_der['R'][axn] + 24*Psi_lcfs**2*iota[axn]*coord_der['R_rv'][axn]**2*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 24*Psi_lcfs**2*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['R_rz'][axn]*coord_der['Z_rv'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] - 24*Psi_lcfs**2*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['R_rvz'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 48*Psi_lcfs**2*coord_der['R_z'][axn]*coord_der['R_rz'][axn]*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 24*Psi_lcfs**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['R_rz'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_lcfs**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]*coord_der['Z_rz'][axn]*coord_der['R_r'][axn]*coord_der['R'][axn] - 48*Psi_lcfs**2*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['Z_rvz'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 24*Psi_lcfs**2*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]*coord_der['R_rvz'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 48*Psi_lcfs**2*coord_der['Z_z'][axn]*coord_der['Z_rz'][axn]*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_lcfs**2*iota[axn]*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['R'][axn] + 24*Psi_lcfs**2*iota[axn]*coord_der['R_z'][axn]*coord_der['Z_rv'][axn]*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]**2*coord_der['R'][axn] + 24*Psi_lcfs**2*iota[axn]*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['R_rvv'][axn]*coord_der['Z_r'][axn]**2*coord_der['R'][axn] + 24*Psi_lcfs**2*iota[axn]*coord_der['R_z'][axn]*coord_der['R_rv'][axn]*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] - 48*Psi_lcfs**2*iota[axn]*coord_der['R_rv'][axn]*coord_der['Z_z'][axn]*coord_der['Z_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn] + 24*Psi_lcfs**2*iota[axn]*coord_der['Z_z'][axn]*coord_der['Z_rv'][axn]*coord_der['R_rvv'][axn]*coord_der['R_r'][axn]*coord_der['Z_r'][axn]*coord_der['R'][axn]) / (24*Psi_lcfs**2*(coord_der['R_rv'][axn]*coord_der['Z_r'][axn] - coord_der['Z_rv'][axn]*coord_der['R_r'][axn])**2*coord_der['R'][axn]))

    R_zz_err = coord_der['R_zz'] - R_zz
    Z_zz_err = coord_der['Z_zz'] - Z_zz

    cR_zz_err = R_transform.fit(R_zz_err)
    cZ_zz_err = Z_transform.fit(Z_zz_err)

    return cR_zz_err, cZ_zz_err


def compute_qs_error_spectral(cR, cZ, cP, cI, Psi_lcfs, R_transform,
                              Z_transform, P_transform, I_transform,
                              pres_ratio, zeta_ratio):
    """Computes quasisymmetry error in spectral space

    Parameters
    ----------
    cR : ndarray, shape(N_coeffs,)
        spectral coefficients of R
    cZ : ndarray, shape(N_coeffs,)
        spectral coefficients of Z
    cP : ndarray, shape(N_coeffs,)
        spectral coefficients of pressure
    cI : ndarray, shape(N_coeffs,)
        spectral coefficients of rotational transform
    Psi_lcfs : float
        total toroidal flux within the last closed flux surface
    R_transform : Transform
        transforms cR to physical space
    Z_transform : Transform
        transforms cZ to physical space
    P_transform : Transform
        transforms cP to physical space
    I_transform : Transform
        transforms cI to physical space
    pres_ratio : float
        fraction in range [0,1] of the full pressure profile to use
    zeta_ratio : float
        fraction in range [0,1] of the full toroidal (zeta) derivatives to use

    Returns
    -------
    cQS : ndarray
        quasisymmetry error Fourier coefficients

    """
    iota = I_transform.transform(cI, 0)

    coord_der = compute_coordinate_derivatives(cR, cZ, R_transform, Z_transform)
    cov_basis = compute_covariant_basis(
        coord_der, R_transform.grid.axis, R_transform.derivs)
    jacobian = compute_jacobian(
        coord_der, cov_basis, R_transform.grid.axis, R_transform.derivs)
    magnetic_field = compute_magnetic_field(
        cov_basis, jacobian, cI, Psi_lcfs, I_transform, R_transform.derivs)
    B_mag = compute_magnetic_field_magnitude(
        cov_basis, magnetic_field, cI, I_transform)

    # B-tilde derivatives
    Bt_v = magnetic_field['B^zeta_v']*(iota*B_mag['|B|_v']+B_mag['|B|_z']) + \
        magnetic_field['B^zeta']*(iota*B_mag['|B|_vv']+B_mag['|B|_vz'])
    Bt_z = magnetic_field['B^zeta_z']*(iota*B_mag['|B|_v']+B_mag['|B|_z']) + \
        magnetic_field['B^zeta']*(iota*B_mag['|B|_vz']+B_mag['|B|_zz'])

    # quasisymmetry
    QS = B_mag['|B|_v']*Bt_z - B_mag['|B|_z']*Bt_v
    # FIXME:  this transform should have the same grid as RZ, but a DoubleFourier basis
    return transform.fit(QS)
