import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot

from desc.backend import TextColors, jnp
from desc.utils import unpack_state, dot

from desc.grid import LinearGrid
from desc.transform import Transform
from desc.equilibrium_io import IOAble
from desc.derivatives import Derivative
from desc.compute_funs import compute_force_error_magnitude


class ObjectiveFunction(IOAble, ABC):

    """Objective function used in the optimization of an Equilibrium

    Attributes
    ----------
    R0_transform : Transform
        transforms R0_n coefficients to real space
    Z0_transform : Transform
        transforms Z0_n coefficients to real space
    r_transform : Transform
        transforms r_lmn coefficients to real space
    l_transform : Transform
        transforms l_lmn coefficients to real space
    R1_transform : Transform
        transforms R1_mn coefficients to real space
    Z1_transform : Transform
        transforms Z1_mn coefficients to real space
    p_transform : Transform
        transforms p_l coefficients to real space
    i_transform : Transform
        transforms i_l coefficients to real space

    Methods
    -------
    compute(x, R1_mn, Z1_mn, p_l, i_l, Psi, zeta_ratio=1.0)
        compute the equilibrium objective function
    callback(x, R1_mn, Z1_mn, p_l, i_l, Psi, zeta_ratio=1.0)
        function that prints equilibrium errors
    """
    _io_attrs_ = ['scalar', 'R0_transform', 'Z0_transform', 'r_transform',
                  'l_transform', 'R1_transform', 'Z1_transform',
                  'p_transform', 'i_transform']

    def __init__(self,
                 R0_transform: Transform = None, Z0_transform: Transform = None,
                 r_transform: Transform = None,  l_transform: Transform = None,
                 R1_transform: Transform = None, Z1_transform: Transform = None,
                 p_transform: Transform = None,  i_transform: Transform = None) -> None:
        """Initializes an ObjectiveFunction

        Parameters
        ----------
        R0_transform : Transform
            transforms R0_n coefficients to real space
        Z0_transform : Transform
            transforms Z0_n coefficients to real space
        r_transform : Transform
            transforms r_lmn coefficients to real space
        l_transform : Transform
            transforms l_lmn coefficients to real space
        R1_transform : Transform
            transforms R1_mn coefficients to real space
        Z1_transform : Transform
            transforms Z1_mn coefficients to real space
        p_transform : Transform
            transforms p_l coefficients to real space
        i_transform : Transform
            transforms i_l coefficients to real space

        Returns
        -------
        None

        """
        self.R0_transform = R0_transform
        self.Z0_transform = Z0_transform
        self.r_transform = r_transform
        self.l_transform = l_transform
        self.R1_transform = R1_transform
        self.Z1_transform = Z1_transform
        self.p_transform = p_transform
        self.i_transform = i_transform

        self._grad = Derivative(self.compute_scalar, mode='grad')
        self._hess = Derivative(self.compute_scalar, mode='hess')
        self._jac = Derivative(self.compute, mode='fwd')

    @abstractmethod
    def compute(self, x, R1_mn, Z1_mn, p_l, i_l, Psi, zeta_ratio=1.0):
        pass

    @abstractmethod
    def compute_scalar(self, x, R1_mn, Z1_mn, p_l, i_l, Psi, zeta_ratio=1.0):
        pass

    @abstractmethod
    def callback(self, x, R1_mn, Z1_mn, p_l, i_l, Psi, zeta_ratio=1.0):
        pass

    def grad(self, x, R1_mn, Z1_mn, p_l, i_l, Psi, zeta_ratio=1.0):
        return self._grad.compute(x, R1_mn, Z1_mn, p_l, i_l, Psi, zeta_ratio)

    def hess(self, x, R1_mn, Z1_mn, p_l, i_l, Psi, zeta_ratio=1.0):
        return self._hess.compute(x, R1_mn, Z1_mn, p_l, i_l, Psi, zeta_ratio)

    def jac(self, x, R1_mn, Z1_mn, p_l, i_l, Psi, zeta_ratio=1.0):
        return self._jac.compute(x, R1_mn, Z1_mn, p_l, i_l, Psi, zeta_ratio)


class ForceErrorNodes(ObjectiveFunction):
    """Minimizes equilibrium force balance error in physical space"""

    def __init__(self,
                 R0_transform: Transform = None, Z0_transform: Transform = None,
                 r_transform: Transform = None,  l_transform: Transform = None,
                 R1_transform: Transform = None, Z1_transform: Transform = None,
                 p_transform: Transform = None,  i_transform: Transform = None) -> None:
        """Initializes a ForceErrorNodes object

        Parameters
        ----------
        R0_transform : Transform
            transforms R0_n coefficients to real space
        Z0_transform : Transform
            transforms Z0_n coefficients to real space
        r_transform : Transform
            transforms r_lmn coefficients to real space
        l_transform : Transform
            transforms l_lmn coefficients to real space
        R1_transform : Transform
            transforms R1_mn coefficients to real space
        Z1_transform : Transform
            transforms Z1_mn coefficients to real space
        p_transform : Transform
            transforms p_l coefficients to real space
        i_transform : Transform
            transforms i_l coefficients to real space

        Returns
        -------
        None

        """
        super().__init__(R0_transform, Z0_transform, r_transform, l_transform,
                         R1_transform, Z1_transform, p_transform, i_transform)
        self.scalar = False

    def compute(self, x, R1_mn, Z1_mn, p_l, i_l, Psi, zeta_ratio=1.0):
        """Compute force balance error."""

        R0_n, Z0_n, r_lmn, l_lmn = unpack_state(
                x, self._R0_transform.basis.num_modes, self._Z0_transform.basis.num_modes,
                self._r_transform.basis.num_modes, self._l_transform.basis.num_modes)

        (force_error, current_density, magnetic_field, profiles, con_basis,
         jacobian, cov_basis, toroidal_coords, polar_coords) =  compute_force_error_magnitude(
             Psi, R0_n, Z0_n, r_lmn, l_lmn, p_l, i_l, self.R0_transform,
             self.Z0_transform, self.r_transform, self.l_transform,
             self.p_transform, self.i_transform, zeta_ratio)

        volumes = self.R0_transform.grid.volumes
        dr = volumes[:, 0]
        dt = volumes[:, 1]
        dz = volumes[:, 2]

        f_rho = force_error['F_rho']*force_error['|grad(rho)|']*jacobian['g']*dr*dt*dz*\
            jnp.sign(dot(con_basis['e^rho'], cov_basis['e_rho'], 0))
        f_beta = force_error['F_beta']*force_error['|beta|']*jacobian['g']*dr*dt*dz*\
            jnp.sign(dot(force_error['beta'], cov_basis['e_theta'], 0))*\
            jnp.sign(dot(force_error['beta'], cov_basis['e_zeta'], 0))

        residual = jnp.concatenate([f_rho.flatten(), f_beta.flatten()])
        return residual

    def compute_scalar(self, x, R1_mn, Z1_mn, p_l, i_l, Psi, zeta_ratio=1.0):
        residual = self.compute(x, R1_mn, Z1_mn, p_l, i_l, Psi, zeta_ratio)
        residual = jnp.sum(residual**2)
        return residual

    def callback(self, x, R1_mn, Z1_mn, p_l, i_l, Psi, zeta_ratio=1.0) -> bool:
        """Prints the rms errors."""

        R0_n, Z0_n, r_lmn, l_lmn = unpack_state(
                x, self._R0_transform.basis.num_modes, self._Z0_transform.basis.num_modes,
                self._r_transform.basis.num_modes, self._l_transform.basis.num_modes)

        (force_error, current_density, magnetic_field, profiles, con_basis,
         jacobian, cov_basis, toroidal_coords, polar_coords) =  compute_force_error_magnitude(
             Psi, R0_n, Z0_n, r_lmn, l_lmn, p_l, i_l, self.R0_transform,
             self.Z0_transform, self.r_transform, self.l_transform,
             self.p_transform, self.i_transform, zeta_ratio)

        volumes = self.R0_transform.grid.volumes
        dr = volumes[:, 0]
        dt = volumes[:, 1]
        dz = volumes[:, 2]

        f_rho = force_error['F_rho']*force_error['|grad(rho)|']*jacobian['g']*dr*dt*dz*\
            jnp.sign(dot(con_basis['e^rho'], cov_basis['e_rho'], 0))
        f_beta = force_error['F_beta']*force_error['|beta|']*jacobian['g']*dr*dt*dz*\
            jnp.sign(dot(force_error['beta'], cov_basis['e_theta'], 0))*\
            jnp.sign(dot(force_error['beta'], cov_basis['e_zeta'], 0))

        f_rho_rms  = jnp.sqrt(jnp.sum(f_rho**2))
        f_beta_rms = jnp.sqrt(jnp.sum(f_beta**2))

        residual = jnp.concatenate([f_rho.flatten(), f_beta.flatten()])
        resid_rms = 1/2*jnp.sum(residual**2)

        print('Total residual: {:10.3e}  f_rho: {:10.3e}  f_beta: {:10.3e}'.format(
            resid_rms, f_rho_rms, f_beta_rms))
        return False


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

    def get_equil_obj_fun(errr_mode,
                          R0_transform: Transform = None, Z0_transform: Transform = None,
                          R1_transform: Transform = None, Z1_transform: Transform = None,
                          L_transform: Transform = None, p_transform: Transform = None,
                          i_transform: Transform = None) -> ObjectiveFunction:
        """Accepts parameters necessary to create an objective function, and returns the corresponding ObjectiveFunction object

        Parameters
        ----------
        errr_mode : str
            error mode of the objective function
            one of 'force', 'accel'
        R0_transform : Transform, optional
            transforms R coefficients to real space in the volume
        Z0_transform : Transform, optional
            transforms Z coefficients to real space in the volume
        R1_transform : Transform, optional
            transforms R coefficients to real space on the surface
        Z1_transform : Transform, optional
            transforms Z coefficients to real space on the surface
        L_transform : Transform, optional
            transforms lambda coefficients to real space
        p_transform : Transform, optional
            transforms pressure coefficients to real space
        i_transform : Transform, optional
            transforms rotational transform coefficients to real space

        Returns
        -------
        obj_fun : ObjectiveFunction
            equilibrium objective function object, containing the compute and callback method for the objective function

        """
        if len(R0_transform.grid.axis):
            raise ValueError(TextColors.FAIL + "Objective cannot be evaluated at the magnetic axis. " +
                             "Yell at Daniel to implement this!" + TextColors.ENDC)

        if errr_mode == 'force':
            obj_fun = ForceErrorNodes(R0_transform=R0_transform, Z0_transform=Z0_transform,
                                      R1_transform=R1_transform, Z1_transform=Z1_transform,
                                      L_transform=L_transform, p_transform=p_transform,
                                      i_transform=i_transform)
        else:
            raise ValueError(TextColors.FAIL + "Requested Objective Function is not implemented. " +
                             "Available objective functions are: 'force'" + TextColors.ENDC)

        return obj_fun


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

    p = [matplotlib.path.Path(np.stack([R, Z]).T, closed=True)
         for R, Z in zip(Rs, Zs)]
    nested = np.all([p[i+1].contains_path(p[i]) for i in range(len(p)-1)])
    intersects = np.any([curve_self_intersects(R, Z) for R, Z in zip(Rs, Zs)])
    return nested and not intersects
