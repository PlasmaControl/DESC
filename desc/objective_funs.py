import numpy as np
from abc import ABC, abstractmethod
from termcolor import colored

from desc.backend import jnp
from desc.utils import unpack_state
from desc.transform import Transform
from desc.io import IOAble
from desc.derivatives import Derivative
from desc.compute_funs import compute_force_error_magnitude, dot, compute_energy
from desc.boundary_conditions import BoundaryConstraint

__all__ = ["ForceErrorNodes", "EnergyVolIntegral", "get_objective_function"]


class ObjectiveFunction(IOAble, ABC):

    """Objective function used in the optimization of an Equilibrium

    Attributes
    ----------
    R_transform : Transform
        transforms R_lmn coefficients to real space
    Z_transform : Transform
        transforms Z_lmn coefficients to real space
    L_transform : Transform
        transforms L_mn coefficients to real space
    Rb_transform : Transform
        transforms Rb_mn coefficients to real space
    Zb_transform : Transform
        transforms Zb_mn coefficients to real space
    p_transform : Transform
        transforms p_l coefficients to real space
    i_transform : Transform
        transforms i_l coefficients to real space
    BC_constraint : BoundaryConstraint
            linear constraint to enforce boundary conditions

    """

    _io_attrs_ = [
        "scalar",
        "R_transform",
        "Z_transform",
        "L_transform",
        "Rb_transform",
        "Zb_transform",
        "p_transform",
        "i_transform",
        "BC_constraint",
    ]

    arg_names = {"Rb_mn": 1, "Zb_mn": 2, "p_l": 3, "i_l": 4, "Psi": 5, "zeta_ration": 6}

    def __init__(
        self,
        R_transform: Transform = None,
        Z_transform: Transform = None,
        L_transform: Transform = None,
        Rb_transform: Transform = None,
        Zb_transform: Transform = None,
        p_transform: Transform = None,
        i_transform: Transform = None,
        BC_constraint: BoundaryConstraint = None,
    ) -> None:
        """Initializes an ObjectiveFunction

        Parameters
        ----------
        R_transform : Transform
            transforms R_lmn coefficients to real space
        Z_transform : Transform
            transforms Z_lmn coefficients to real space
        L_transform : Transform
            transforms L_mn coefficients to real space
        Rb_transform : Transform
            transforms Rb_mn coefficients to real space
        Zb_transform : Transform
            transforms Zb_mn coefficients to real space
        p_transform : Transform
            transforms p_l coefficients to real space
        i_transform : Transform
            transforms i_l coefficients to real space
        BC_constraint : BoundaryConstraint
            linear constraint to enforce boundary conditions

        Returns
        -------
        None

        """
        self.R_transform = R_transform
        self.Z_transform = Z_transform
        self.L_transform = L_transform
        self.Rb_transform = Rb_transform
        self.Zb_transform = Zb_transform
        self.p_transform = p_transform
        self.i_transform = i_transform
        self.BC_constraint = BC_constraint

        self._grad = Derivative(self.compute_scalar, mode="grad")
        self._hess = Derivative(self.compute_scalar, mode="hess")
        self._jac = Derivative(self.compute, mode="fwd")

    @property
    @abstractmethod
    def scalar(self):
        """boolean of whether default "compute" method is a scalar or vector"""

    @property
    @abstractmethod
    def name(self):
        """return a string indicator of the type of objective function"""

    @property
    @abstractmethod
    def derivatives(self):
        """return arrays indicating which derivatives are needed to compute"""

    @abstractmethod
    def compute(self, x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio=1.0):
        """compute the objective function"""

    @abstractmethod
    def compute_scalar(self, x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio=1.0):
        """compute the scalar form of the objective"""

    @abstractmethod
    def callback(self, x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio=1.0):
        """print the value of the objective"""

    def grad_x(self, x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio=1.0):
        """Computes gradient vector of scalar form of the objective wrt to x"""
        return self._grad.compute(x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio)

    def hess_x(self, x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio=1.0):
        """Computes hessian matrix of scalar form of the objective wrt to x"""
        return self._hess.compute(x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio)

    def jac_x(self, x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio=1.0):
        """Computes jacobian matrx of vector form of the objective wrt to x"""
        return self._jac.compute(x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio)

    def jvp(self, argnum, v, x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio=1.0):
        """Computes jacobian-vector product of the objective function

        Parameters
        ----------
        argnum : int
            integer describing which argument of the objective should be differentiated.
        v : ndarray
            vector to multiply the jacobian matrix by

        Returns
        -------
        df : ndarray
            Jacobian vector product
        """
        f = Derivative(self.compute, argnum=argnum, mode="jvp")
        return f(v, x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio)

    def derivative(self, argnums, x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio=1.0):
        """Computes arbitrary derivatives of the objective function

        Parameters
        ----------
        argnums : int, str, tuple
            integer or str or tuple of integers/strings describing which arguments
            of the objective should be differentiated.
            Passing a tuple with multiple values will compute a higher order derivative.
            Eg, argnums=(0,0) would compute the 2nd derivative with respect to the
            zeroth argument, while argnums=(3,5) would compute a mixed second
            derivative, first with respect to the third argument and then with
            respect to the fifth.

        Returns
        -------
        df : ndarray
            specified derivative of the objective
        """
        if not isinstance(argnums, tuple):
            argnums = (argnums,)

        f = self.compute
        for a in argnums:
            if isinstance(a, int) and a < 7:
                f = Derivative(f, argnum=a)
            elif isinstance(a, str) and a in ObjectiveFunction.arg_names:
                a = ObjectiveFunction.arg_names.get(a)
                f = Derivative(f, argnum=a)
            else:
                raise ValueError(
                    "argnums should be integers between 0 and 6 or one of {}, got {}".format(
                        ObjectiveFunction.arg_names, a
                    )
                )

        return f(x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio)


class ForceErrorNodes(ObjectiveFunction):
    """Minimizes equilibrium force balance error in physical space"""

    def __init__(
        self,
        R_transform: Transform = None,
        Z_transform: Transform = None,
        L_transform: Transform = None,
        Rb_transform: Transform = None,
        Zb_transform: Transform = None,
        p_transform: Transform = None,
        i_transform: Transform = None,
        BC_constraint: BoundaryConstraint = None,
    ) -> None:
        """Initializes a force error objective function

        Parameters
        ----------
        R_transform : Transform
            transforms R_lmn coefficients to real space
        Z_transform : Transform
            transforms Z_lmn coefficients to real space
        L_transform : Transform
            transforms L_mn coefficients to real space
        Rb_transform : Transform
            transforms Rb_mn coefficients to real space
        Zb_transform : Transform
            transforms Zb_mn coefficients to real space
        p_transform : Transform
            transforms p_l coefficients to real space
        i_transform : Transform
            transforms i_l coefficients to real space
        BC_constraint : BoundaryConstraint
            linear constraint to enforce boundary conditions

        """
        super().__init__(
            R_transform,
            Z_transform,
            L_transform,
            Rb_transform,
            Zb_transform,
            p_transform,
            i_transform,
            BC_constraint,
        )

    @property
    def scalar(self):
        """boolean of whether default "compute" method is a scalar or vector"""
        return False

    @property
    def name(self):
        """return a string indicator of the type of objective function"""
        return "force"

    @property
    def derivatives(self):
        """return arrays indicating which derivatives are needed to compute"""
        # TODO: different derivatives for R,Z,L,p,i ?
        # old axis derivatives
        axis = np.array([[2, 1, 0], [1, 2, 0], [1, 1, 1], [2, 2, 0]])
        derivatives = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [2, 0, 0],
                [1, 1, 0],
                [1, 0, 1],
                [0, 2, 0],
                [0, 1, 1],
                [0, 0, 2],
            ]
        )
        return derivatives

    def compute(self, x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio=1.0):
        """Compute force balance error

        Parameters
        ----------
        x : ndarray
            optimization state vector
        Rb_mn : ndarray
            array of fourier coefficients for R boundary
        Zb_mn : ndarray
            array of fourier coefficients for Z boundary
        p_l : ndarray
            series coefficients for pressure profile
        i_l : ndarray
            series coefficients for iota profile
        Psi : float
            toroidal flux within the last closed flux surface in webers
        zeta_ratio : float
            multiplier on the toroidal derivatives.

        Returns
        -------
        f : ndarray
            force error in radial and helical directions at each node
        """

        if self.BC_constraint is not None:
            # x is really 'y', need to recover full state vector
            x = self.BC_constraint.recover_from_bdry(x, Rb_mn, Zb_mn)

        R_lmn, Z_lmn, L_mn = unpack_state(
            x,
            self.R_transform.basis.num_modes,
            self.Z_transform.basis.num_modes,
        )

        (
            force_error,
            current_density,
            magnetic_field,
            con_basis,
            jacobian,
            cov_basis,
            toroidal_coords,
            profiles,
        ) = compute_force_error_magnitude(
            Psi,
            R_lmn,
            Z_lmn,
            L_mn,
            p_l,
            i_l,
            self.R_transform,
            self.Z_transform,
            self.L_transform,
            self.p_transform,
            self.i_transform,
            zeta_ratio,
        )

        weights = self.R_transform.grid.weights

        f_rho = (
            force_error["F_rho"]
            * force_error["|grad(rho)|"]
            * jacobian["g"]
            * weights
            * jnp.sign(dot(con_basis["e^rho"], cov_basis["e_rho"], 0))
        )
        f_beta = (
            force_error["F_beta"]
            * force_error["|beta|"]
            * jacobian["g"]
            * weights
            * jnp.sign(dot(force_error["beta"], cov_basis["e_theta"], 0))
            * jnp.sign(dot(force_error["beta"], cov_basis["e_zeta"], 0))
        )
        residual = jnp.concatenate([f_rho.flatten(), f_beta.flatten()])

        return residual

    def compute_scalar(self, x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio=1.0):
        """Compute the total force balance error

        eg 1/2 sum(f**2)

        Parameters
        ----------
        x : ndarray
            optimization state vector
        Rb_mn : ndarray
            array of fourier coefficients for R boundary
        Zb_mn : ndarray
            array of fourier coefficients for Z boundary
        p_l : ndarray
            series coefficients for pressure profile
        i_l : ndarray
            series coefficients for iota profile
        Psi : float
            toroidal flux within the last closed flux surface in webers
        zeta_ratio : float
            multiplier on the toroidal derivatives.

        Returns
        -------
        f : float
            total force balance error
        """
        residual = self.compute(x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio)
        residual = 1 / 2 * jnp.sum(residual ** 2)
        return residual

    def callback(self, x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio=1.0) -> bool:
        """Prints the rms errors for radial and helical force balance

        Parameters
        ----------
        x : ndarray
            optimization state vector
        Rb_mn : ndarray
            array of fourier coefficients for R boundary
        Zb_mn : ndarray
            array of fourier coefficients for Z boundary
        p_l : ndarray
            series coefficients for pressure profile
        i_l : ndarray
            series coefficients for iota profile
        Psi : float
            toroidal flux within the last closed flux surface in webers
        zeta_ratio : float
            multiplier on the toroidal derivatives.
        """

        if self.BC_constraint is not None:
            # x is really 'y', need to recover full state vector
            x = self.BC_constraint.recover_from_bdry(x, Rb_mn, Zb_mn)

        R_lmn, Z_lmn, L_mn = unpack_state(
            x,
            self.R_transform.basis.num_modes,
            self.Z_transform.basis.num_modes,
        )

        (
            force_error,
            current_density,
            magnetic_field,
            con_basis,
            jacobian,
            cov_basis,
            toroidal_coords,
            profiles,
        ) = compute_force_error_magnitude(
            Psi,
            R_lmn,
            Z_lmn,
            L_mn,
            p_l,
            i_l,
            self.R_transform,
            self.Z_transform,
            self.L_transform,
            self.p_transform,
            self.i_transform,
            zeta_ratio,
        )

        weights = self.R_transform.grid.weights

        f_rho = (
            force_error["F_rho"]
            * force_error["|grad(rho)|"]
            * jacobian["g"]
            * weights
            * jnp.sign(dot(con_basis["e^rho"], cov_basis["e_rho"], 0))
        )
        f_beta = (
            force_error["F_beta"]
            * force_error["|beta|"]
            * jacobian["g"]
            * weights
            * jnp.sign(dot(force_error["beta"], cov_basis["e_theta"], 0))
            * jnp.sign(dot(force_error["beta"], cov_basis["e_zeta"], 0))
        )

        f_rho_rms = jnp.sqrt(jnp.sum(f_rho ** 2))
        f_beta_rms = jnp.sqrt(jnp.sum(f_beta ** 2))

        residual = jnp.concatenate([f_rho.flatten(), f_beta.flatten()])
        resid_rms = 1 / 2 * jnp.sum(residual ** 2)

        print(
            "Total residual: {:10.3e}  f_rho: {:10.3e}  f_beta: {:10.3e}".format(
                resid_rms, f_rho_rms, f_beta_rms
            )
        )

        return None


class EnergyVolIntegral(ObjectiveFunction):
    """Minimizes the volume integral of MHD energy (B^2 / (2*mu0) - p) in physical space"""

    def __init__(
        self,
        R_transform: Transform = None,
        Z_transform: Transform = None,
        L_transform: Transform = None,
        Rb_transform: Transform = None,
        Zb_transform: Transform = None,
        p_transform: Transform = None,
        i_transform: Transform = None,
        BC_constraint: BoundaryConstraint = None,
    ) -> None:
        """Initializes an EnergyVolintegral object

        Parameters
        ----------
        R_transform : Transform
            transforms R_lmn coefficients to real space
        Z_transform : Transform
            transforms Z_lmn coefficients to real space
        L_transform : Transform
            transforms L_lmn coefficients to real space
        Rb_transform : Transform
            transforms Rb_mn coefficients to real space
        Zb_transform : Transform
            transforms Zb_mn coefficients to real space
        p_transform : Transform
            transforms p_l coefficients to real space
        i_transform : Transform
            transforms i_l coefficients to real space
        BC_constraint : BoundaryConstraint
            linear constraint to enforce boundary conditions

        """
        super().__init__(
            R_transform,
            Z_transform,
            L_transform,
            Rb_transform,
            Zb_transform,
            p_transform,
            i_transform,
            BC_constraint,
        )

    @property
    def scalar(self):
        """boolean of whether default "compute" method is a scalar or vector"""
        return True

    @property
    def name(self):
        """return a string indicator of the type of objective function"""
        return "energy"

    @property
    def derivatives(self):
        """return arrays indicating which derivatives are needed to compute"""
        # TODO: different derivatives for R,Z,L,p,i ?
        derivatives = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )
        return derivatives

    def compute(self, x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio=1.0):
        """Compute MHD energy

        Parameters
        ----------
        x : ndarray
            optimization state vector
        Rb_mn : ndarray
            array of fourier coefficients for R boundary
        Zb_mn : ndarray
            array of fourier coefficients for Z boundary
        p_l : ndarray
            series coefficients for pressure profile
        i_l : ndarray
            series coefficients for iota profile
        Psi : float
            toroidal flux within the last closed flux surface in webers
        zeta_ratio : float
            multiplier on the toroidal derivatives.

        Returns
        -------
        W : float
            total MHD energy in the plasma volume
        """

        if self.BC_constraint is not None:
            # x is really 'y', need to recover full state vector
            x = self.BC_constraint.recover_from_bdry(x, Rb_mn, Zb_mn)

        R_lmn, Z_lmn, L_lmn = unpack_state(
            x,
            self.R_transform.basis.num_modes,
            self.Z_transform.basis.num_modes,
        )

        (
            energy,
            magnetic_field,
            jacobian,
            cov_basis,
            toroidal_coords,
            profiles,
        ) = compute_energy(
            Psi,
            R_lmn,
            Z_lmn,
            L_lmn,
            p_l,
            i_l,
            self.R_transform,
            self.Z_transform,
            self.L_transform,
            self.p_transform,
            self.i_transform,
            zeta_ratio,
        )

        residual = energy["W"]

        return residual

    def compute_scalar(self, x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio=1.0):
        """Compute MHD energy

        Parameters
        ----------
        x : ndarray
            optimization state vector
        Rb_mn : ndarray
            array of fourier coefficients for R boundary
        Zb_mn : ndarray
            array of fourier coefficients for Z boundary
        p_l : ndarray
            series coefficients for pressure profile
        i_l : ndarray
            series coefficients for iota profile
        Psi : float
            toroidal flux within the last closed flux surface in webers
        zeta_ratio : float
            multiplier on the toroidal derivatives.

        Returns
        -------
        W : float
            total MHD energy in the plasma volume
        """
        residual = self.compute(x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio)
        return residual

    def callback(self, x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio=1.0) -> bool:
        """Print the MHD energy

        Parameters
        ----------
        x : ndarray
            optimization state vector
        Rb_mn : ndarray
            array of fourier coefficients for R boundary
        Zb_mn : ndarray
            array of fourier coefficients for Z boundary
        p_l : ndarray
            series coefficients for pressure profile
        i_l : ndarray
            series coefficients for iota profile
        Psi : float
            toroidal flux within the last closed flux surface in webers
        zeta_ratio : float
            multiplier on the toroidal derivatives.

        """
        if self.BC_constraint is not None:
            # x is really 'y', need to recover full state vector
            x = self.BC_constraint.recover_from_bdry(x, Rb_mn, Zb_mn)

        R_lmn, Z_lmn, L_lmn = unpack_state(
            x,
            self.R_transform.basis.num_modes,
            self.Z_transform.basis.num_modes,
        )

        (
            energy,
            magnetic_field,
            jacobian,
            cov_basis,
            toroidal_coords,
            profiles,
        ) = compute_energy(
            Psi,
            R_lmn,
            Z_lmn,
            L_lmn,
            p_l,
            i_l,
            self.R_transform,
            self.Z_transform,
            self.L_transform,
            self.p_transform,
            self.i_transform,
            zeta_ratio,
        )

        residual = energy["W"]

        print(
            "Total MHD energy: {:10.3e}, Magnetic Energy: {:10.3e}, Pressure Energy: {:10.3e}".format(
                energy["W"], energy["W_B"], energy["W_p"]
            )
        )


def get_objective_function(
    errr_mode,
    R_transform: Transform = None,
    Z_transform: Transform = None,
    L_transform: Transform = None,
    Rb_transform: Transform = None,
    Zb_transform: Transform = None,
    p_transform: Transform = None,
    i_transform: Transform = None,
    BC_constraint: BoundaryConstraint = None,
) -> ObjectiveFunction:
    """Get an objective function by name

    Parameters
    ----------
    errr_mode : str
        name of the desired objective function, eg 'force' or 'energy'
    R_transform : Transform
        transforms R_lmn coefficients to real space
    Z_transform : Transform
        transforms Z_lmn coefficients to real space
    L_transform : Transform
        transforms L_mn coefficients to real space
    Rb_transform : Transform
        transforms Rb_mn coefficients to real space
    Zb_transform : Transform
        transforms Zb_mn coefficients to real space
    p_transform : Transform
        transforms p_l coefficients to real space
    i_transform : Transform
        transforms i_l coefficients to real space
    BC_constraint : BoundaryConstraint
        linear constraint to enforce boundary conditions

    Returns
    -------
    objective : ObjectiveFunction
        objective initialized with the given transforms and constraints
    """

    if errr_mode == "force":
        obj_fun = ForceErrorNodes(
            R_transform=R_transform,
            Z_transform=Z_transform,
            L_transform=L_transform,
            Rb_transform=Rb_transform,
            Zb_transform=Zb_transform,
            p_transform=p_transform,
            i_transform=i_transform,
            BC_constraint=BC_constraint,
        )
    elif errr_mode == "energy":
        obj_fun = EnergyVolIntegral(
            R_transform=R_transform,
            Z_transform=Z_transform,
            L_transform=L_transform,
            Rb_transform=Rb_transform,
            Zb_transform=Zb_transform,
            p_transform=p_transform,
            i_transform=i_transform,
            BC_constraint=BC_constraint,
        )
    else:
        raise ValueError(
            colored(
                "Requested Objective Function is not implemented. "
                + "Available objective functions are: 'force'",
                "red",
            )
        )

    return obj_fun


"""
QS derivatives (just here so we don't loose them)
            derivatives = np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [2, 0, 0],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 2, 0],
                    [0, 1, 1],
                    [0, 0, 2],
                    [3, 0, 0],
                    [2, 1, 0],
                    [2, 0, 1],
                    [1, 2, 0],
                    [1, 1, 1],
                    [1, 0, 2],
                    [0, 3, 0],
                    [0, 2, 1],
                    [0, 1, 2],
                    [0, 0, 3],
                    [2, 2, 0],
                ]
            )
"""
