from abc import ABC, abstractmethod
from termcolor import colored

from desc.backend import jnp
from desc.utils import unpack_state, dot
from desc.transform import Transform
from desc.equilibrium_io import IOAble
from desc.derivatives import Derivative
from desc.compute_funs import compute_force_error_magnitude
from desc.boundary_conditions import BoundaryConstraint, RadialConstraint


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
                 p_transform: Transform = None,  i_transform: Transform = None,
                 bc_constraint: BoundaryConstraint = None,
                 radial_constraint: RadialConstraint = None) -> None:
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
        bc_constraint : BoundaryConstraint
            linear constraint to enforce boundary conditions

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
        if bc_constraint is None:
            bc_constraint = BoundaryConstraint(R0_transform.basis, Z0_transform.basis,
                                               r_transform.basis, l_transform.basis)
        self.bc_constraint = bc_constraint
        if radial_constraint is None:
            radial_constraint = RadialConstraint(
                r_transform.basis)
        self.radial_constraint = radial_constraint

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

    def grad_x(self, x, R1_mn, Z1_mn, p_l, i_l, Psi, zeta_ratio=1.0):
        """Computes gradient vector of scalar form of the objective wrt to x"""
        return self._grad.compute(x, R1_mn, Z1_mn, p_l, i_l, Psi, zeta_ratio)

    def hess_x(self, x, R1_mn, Z1_mn, p_l, i_l, Psi, zeta_ratio=1.0):
        """Computes hessian matrix of scalar form of the objective wrt to x"""
        return self._hess.compute(x, R1_mn, Z1_mn, p_l, i_l, Psi, zeta_ratio)

    def jac_x(self, x, R1_mn, Z1_mn, p_l, i_l, Psi, zeta_ratio=1.0):
        """Computes jacobian matrx of vector form of the objective wrt to x"""
        return self._jac.compute(x, R1_mn, Z1_mn, p_l, i_l, Psi, zeta_ratio)

    def derivative(self, argnums, x, R1_mn, Z1_mn, p_l, i_l, Psi, zeta_ratio=1.0):
        """Computes arbitrary derivatives of the objective

        Parameters
        ----------
        argnums : int, tuple
            integer or tuple of integers describing which argument numbers
            of the objective should be differentiated. Passing a tuple with
            multiple values will compute a higher order derivative. Eg,
            argnums=(0,0) would compute the 2nd derivative with respect to the
            zeroth argument, while argnums=(3,5) would compute a mixed second
            derivative, first with respect to the third argument and then with
            respect to the fifth.
        """
        if not isinstance(argnums, tuple):
            argnums = tuple(argnums)

        f = self.compute
        for a in argnums:
            f = Derivative(f, argnum=a)
        y = f(x, R1_mn, Z1_mn, p_l, i_l, Psi, zeta_ratio=1.0)
        return y


class ForceErrorNodes(ObjectiveFunction):
    """Minimizes equilibrium force balance error in physical space"""

    def __init__(self,
                 R0_transform: Transform = None, Z0_transform: Transform = None,
                 r_transform: Transform = None,  l_transform: Transform = None,
                 R1_transform: Transform = None, Z1_transform: Transform = None,
                 p_transform: Transform = None,  i_transform: Transform = None,
                 bc_constraint: BoundaryConstraint = None,
                 radial_constraint: RadialConstraint = None) -> None:
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
        bc_constraint : BoundaryConstraint
            linear constraint to enforce boundary conditions

        Returns
        -------
        None

        """
        super().__init__(R0_transform, Z0_transform, r_transform, l_transform,
                         R1_transform, Z1_transform, p_transform, i_transform,
                         bc_constraint, radial_constraint)
        self.scalar = False

    def compute(self, x, R1_mn, Z1_mn, p_l, i_l, Psi, zeta_ratio=1.0):
        """Compute force balance error."""

        # input x is really "y", need to convert to full x
        x = self.bc_constraint.recover(x)

        R0_n, Z0_n, r_lmn, l_lmn = unpack_state(
            x, self.R0_transform.basis.num_modes, self.Z0_transform.basis.num_modes,
            self.r_transform.basis.num_modes, self.l_transform.basis.num_modes)

        (force_error, current_density, magnetic_field, profiles, con_basis,
         jacobian, cov_basis, toroidal_coords, polar_coords) = compute_force_error_magnitude(
             Psi, R0_n, Z0_n, r_lmn, l_lmn, R1_mn, Z1_mn, p_l, i_l,
             self.R0_transform, self.Z0_transform, self.r_transform,
             self.l_transform, self.R1_transform, self.Z1_transform,
             self.p_transform, self.i_transform, zeta_ratio)

        volumes = self.R0_transform.grid.volumes
        dr = volumes[:, 0]
        dt = volumes[:, 1]
        dz = volumes[:, 2]

        f_rho = force_error['F_rho']*force_error['|grad(rho)|']*jacobian['g']*dr*dt*dz *\
            jnp.sign(dot(con_basis['e^rho'], cov_basis['e_rho'], 0))
        f_beta = force_error['F_beta']*force_error['|beta|']*jacobian['g']*dr*dt*dz *\
            jnp.sign(dot(force_error['beta'], cov_basis['e_theta'], 0)) *\
            jnp.sign(dot(force_error['beta'], cov_basis['e_zeta'], 0))

        dr_penalty = self.radial_constraint.compute(r_lmn)

        residual = jnp.concatenate(
            [f_rho.flatten(), f_beta.flatten(), dr_penalty.flatten()])
        return residual

    def compute_scalar(self, x, R1_mn, Z1_mn, p_l, i_l, Psi, zeta_ratio=1.0):
        residual = self.compute(x, R1_mn, Z1_mn, p_l, i_l, Psi, zeta_ratio)
        residual = jnp.sum(residual**2)
        return residual

    def callback(self, x, R1_mn, Z1_mn, p_l, i_l, Psi, zeta_ratio=1.0) -> bool:
        """Prints the rms errors."""

        # input x is really "y", need to convert to full x
        x = self.bc_constraint.recover(x)

        R0_n, Z0_n, r_lmn, l_lmn = unpack_state(
            x, self.R0_transform.basis.num_modes, self.Z0_transform.basis.num_modes,
            self.r_transform.basis.num_modes, self.l_transform.basis.num_modes)

        (force_error, current_density, magnetic_field, profiles, con_basis,
         jacobian, cov_basis, toroidal_coords, polar_coords) = compute_force_error_magnitude(
             Psi, R0_n, Z0_n, r_lmn, l_lmn, R1_mn, Z1_mn, p_l, i_l,
             self.R0_transform, self.Z0_transform, self.r_transform,
             self.l_transform, self.R1_transform, self.Z1_transform,
             self.p_transform, self.i_transform, zeta_ratio)

        volumes = self.R0_transform.grid.volumes
        dr = volumes[:, 0]
        dt = volumes[:, 1]
        dz = volumes[:, 2]

        f_rho = force_error['F_rho']*force_error['|grad(rho)|']*jacobian['g']*dr*dt*dz *\
            jnp.sign(dot(con_basis['e^rho'], cov_basis['e_rho'], 0))
        f_beta = force_error['F_beta']*force_error['|beta|']*jacobian['g']*dr*dt*dz *\
            jnp.sign(dot(force_error['beta'], cov_basis['e_theta'], 0)) *\
            jnp.sign(dot(force_error['beta'], cov_basis['e_zeta'], 0))

        f_rho_rms = jnp.sqrt(jnp.sum(f_rho**2))
        f_beta_rms = jnp.sqrt(jnp.sum(f_beta**2))

        residual = jnp.concatenate([f_rho.flatten(), f_beta.flatten()])
        resid_rms = 1/2*jnp.sum(residual**2)

        print('Total residual: {:10.3e}  f_rho: {:10.3e}  f_beta: {:10.3e}'.format(
            resid_rms, f_rho_rms, f_beta_rms))

        return None


class ObjectiveFunctionFactory():
    """Factory Class for Objective Functions

    Methods
    -------
    get_equil_obj_fun()

    """

    def get_equil_obj_fun(errr_mode,
                          R0_transform: Transform = None, Z0_transform: Transform = None,
                          r_transform: Transform = None, l_transform: Transform = None,
                          R1_transform: Transform = None, Z1_transform: Transform = None,
                          p_transform: Transform = None, i_transform: Transform = None,
                          bc_constraint: BoundaryConstraint = None,
                          radial_constraint: RadialConstraint = None) -> ObjectiveFunction:
        """Accepts parameters necessary to create an objective function, and returns the corresponding ObjectiveFunction object

        Parameters
        ----------
        errr_mode : str
            error mode of the objective function
            one of 'force', 'accel'
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
        bc_constraint : BoundaryConstraint
            linear constraint to enforce BC
        Returns
        -------
        obj_fun : ObjectiveFunction
            equilibrium objective function object, containing the compute and callback method for the objective function

        """
        if len(R0_transform.grid.axis):
            raise ValueError(colored("Objective cannot be evaluated at the magnetic axis. " +
                                     "Yell at Daniel to implement this!", 'red'))

        if errr_mode == 'force':
            obj_fun = ForceErrorNodes(
                R0_transform=R0_transform, Z0_transform=Z0_transform,
                r_transform=r_transform, l_transform=l_transform,
                R1_transform=R1_transform, Z1_transform=Z1_transform,
                p_transform=p_transform, i_transform=i_transform,
                bc_constraint=bc_constraint)
        else:
            raise ValueError(colored("Requested Objective Function is not implemented. " +
                                     "Available objective functions are: 'force'", 'red'))

        return obj_fun
