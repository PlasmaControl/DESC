import numpy as np
from abc import ABC, abstractmethod
from termcolor import colored
import warnings
import scipy.special

from desc.backend import jnp, jit
from desc.utils import unpack_state
from desc.io import IOAble
from desc.derivatives import Derivative
from desc.compute_funs import compute_force_error_magnitude, dot, compute_energy

__all__ = ["ForceErrorNodes", "EnergyVolIntegral", "get_objective_function"]


class ObjectiveFunction(IOAble, ABC):

    """Objective function used in the optimization of an Equilibrium

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
    use_jit : bool, optional
        whether to just-in-time compile the objective and derivatives
    devices : jax.device or list of jax.device, optional
        devices to jit compile to

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

    arg_names = {"Rb_mn": 1, "Zb_mn": 2, "p_l": 3, "i_l": 4, "Psi": 5, "zeta_ratio": 6}

    def __init__(
        self,
        R_transform,
        Z_transform,
        L_transform,
        Rb_transform,
        Zb_transform,
        p_transform,
        i_transform,
        BC_constraint=None,
        use_jit=True,
        devices=None,
    ):

        self.R_transform = R_transform
        self.Z_transform = Z_transform
        self.L_transform = L_transform
        self.Rb_transform = Rb_transform
        self.Zb_transform = Zb_transform
        self.p_transform = p_transform
        self.i_transform = i_transform
        self.BC_constraint = BC_constraint
        self._use_jit = use_jit

        if not isinstance(devices, (list, tuple)):
            devices = [devices]

        self._grad = Derivative(
            self.compute_scalar, mode="grad", use_jit=use_jit, devices=devices
        )
        self._hess = Derivative(
            self.compute_scalar, mode="hess", use_jit=use_jit, devices=devices
        )
        self._jac = Derivative(
            self.compute, mode="fwd", use_jit=use_jit, devices=devices
        )

        if self._use_jit:
            self.compute = jit(self.compute, device=devices[0])
            self.compute_scalar = jit(self.compute_scalar, device=devices[0])

    @property
    @abstractmethod
    def scalar(self):
        """bool : whether default "compute" method is a scalar or vector"""

    @property
    @abstractmethod
    def name(self):
        """str : type of objective function"""

    @property
    @abstractmethod
    def derivatives(self):
        """ndarray : which derivatives are needed to compute"""

    @abstractmethod
    def compute(self, *args):
        """compute the objective function

        Parameters
        ----------
        args : list
            (x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio)

        """

    @abstractmethod
    def compute_scalar(self, *args):
        """compute the scalar form of the objective"""

    @abstractmethod
    def callback(self, *args):
        """print the value of the objective"""

    def grad_x(self, *args):
        """Computes gradient vector of scalar form of the objective wrt to x"""
        return self._grad.compute(*args)

    def hess_x(self, *args):
        """Computes hessian matrix of scalar form of the objective wrt to x"""
        return self._hess.compute(*args)

    def jac_x(self, *args):
        """Computes jacobian matrx of vector form of the objective wrt to x"""
        return self._jac.compute(*args)

    def jvp(self, argnum, v, *args):
        """Computes jacobian-vector product of the objective function

        Eg, df/dx*v

        Parameters
        ----------
        argnum : int or tuple of int
            integer describing which argument of the objective should be differentiated.
        v : ndarray or tuple of ndarray
            vector to multiply the jacobian matrix by, one per argnum
        args : list
            (x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio)

        Returns
        -------
        df : ndarray
            Jacobian vector product, summed over different argnums
        """
        f = Derivative.compute_jvp(self.compute, argnum, v, *args)
        return f

    def jvp2(self, argnum1, argnum2, v1, v2, *args):
        """Computes 2nd derivative jacobian-vector product of the objective function

        Eg, d^2f/dx^2*v1*v2

        Parameters
        ----------
        argnum1, argnum2 : int or tuple of int
            integer describing which argument of the objective should be differentiated.
        v1, v2 : ndarray or tuple of ndarray
            vector to multiply the jacobian matrix by, one per argnum
        args : list
            (x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio)

        Returns
        -------
        d2f : ndarray
            Jacobian vector product
        """
        f = Derivative.compute_jvp2(self.compute, argnum1, argnum2, v1, v2, *args)
        return f

    def derivative(self, argnums, *args):
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
        args : list
            (x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio)

        Returns
        -------
        df : ndarray
            specified derivative of the objective
        """
        if not isinstance(argnums, tuple):
            argnums = (argnums,)

        f = self.compute
        dims = [f(*args).size]
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
            dims.append(args[a].size)

        return f(*args).reshape(tuple(dims))


class ForceErrorNodes(ObjectiveFunction):
    """Minimizes equilibrium force balance error in physical space

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
    use_jit : bool, optional
        whether to just-in-time compile the objective and derivatives
    devices : jax.device or list of jax.device, optional
        devices to jit compile to

    """

    def __init__(
        self,
        R_transform,
        Z_transform,
        L_transform,
        Rb_transform,
        Zb_transform,
        p_transform,
        i_transform,
        BC_constraint=None,
        use_jit=True,
        devices=None,
    ):

        super().__init__(
            R_transform,
            Z_transform,
            L_transform,
            Rb_transform,
            Zb_transform,
            p_transform,
            i_transform,
            BC_constraint,
            use_jit,
            devices,
        )
        self.dimx = (
            R_transform.num_modes + Z_transform.num_modes + L_transform.num_modes
        )
        self.dimy = self.dimx if BC_constraint is None else BC_constraint.dimy
        self.dimf = 2 * R_transform.num_nodes
        block_size = 3000
        if not isinstance(devices, (list, tuple)):
            devices = [devices]

        self._grad = Derivative(self.compute_scalar, mode="grad", use_jit=self._use_jit)

        if self.dimy > block_size:
            self._hess = Derivative(
                self.compute_scalar,
                mode="hess",
                use_jit=self._use_jit,
                devices=devices,
                block_size=block_size,
                shape=(self.dimy, self.dimy),
            )
        else:
            self._hess = Derivative(
                self.compute_scalar,
                mode="hess",
                use_jit=self._use_jit,
                devices=devices,
            )
        if self.dimf > block_size:
            self._jac = Derivative(
                self.compute,
                mode="fwd",
                use_jit=self._use_jit,
                devices=devices,
                block_size=block_size,
                shape=(self.dimf, self.dimy),
            )
        else:
            self._jac = Derivative(
                self.compute, mode="fwd", use_jit=self._use_jit, devices=devices
            )

    @property
    def scalar(self):
        """bool : whether default "compute" method is a scalar or vector"""
        return False

    @property
    def name(self):
        """str : type of objective function"""
        return "force"

    @property
    def derivatives(self):
        """ndarray : which derivatives are needed to compute"""
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

        if self.BC_constraint is not None and x.size == self.dimy:
            # x is really 'y', need to recover full state vector
            x = self.BC_constraint.recover_from_bdry(x, Rb_mn, Zb_mn)

        R_lmn, Z_lmn, L_lmn = unpack_state(
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

        if self.BC_constraint is not None and x.size == self.dimy:
            # x is really 'y', need to recover full state vector
            x = self.BC_constraint.recover_from_bdry(x, Rb_mn, Zb_mn)

        R_lmn, Z_lmn, L_lmn = unpack_state(
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
    """Minimizes the volume integral of MHD energy (B^2 / (2*mu0) - p) in physical space

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
    use_jit : bool
        whether to just-in-time compile the objective and derivatives

    """

    def __init__(
        self,
        R_transform,
        Z_transform,
        L_transform,
        Rb_transform,
        Zb_transform,
        p_transform,
        i_transform,
        BC_constraint=None,
        use_jit=True,
        devices=None,
    ):

        super().__init__(
            R_transform,
            Z_transform,
            L_transform,
            Rb_transform,
            Zb_transform,
            p_transform,
            i_transform,
            BC_constraint,
            use_jit,
            devices,
        )
        self.dimx = (
            R_transform.num_modes + Z_transform.num_modes + L_transform.num_modes
        )
        self.dimy = self.dimx if BC_constraint is None else BC_constraint.dimy
        self.dimf = self.dimy
        block_size = 1000
        if not isinstance(devices, (list, tuple)):
            devices = [devices]

        self._grad = Derivative(
            self.compute_scalar, mode="grad", use_jit=self._use_jit, device=devices[0]
        )

        if self.dimy > block_size:
            self._hess = Derivative(
                self.compute_scalar,
                mode="hess",
                use_jit=self._use_jit,
                devices=devices,
                block_size=block_size,
                shape=(self.dimy, self.dimy),
            )
        else:
            self._hess = Derivative(
                self.compute_scalar,
                mode="hess",
                use_jit=self._use_jit,
                devices=devices,
            )
        # scalar objective -> jac = grad
        self._jac = Derivative(
            self.compute, mode="rev", use_jit=self._use_jit, devices=devices
        )

        rho = R_transform.grid.nodes[:, 0]
        N_radial_roots = len(jnp.unique(rho))
        roots, _ = scipy.special.js_roots(N_radial_roots, 2, 2)
        if not np.all(np.unique(rho) == roots):
            warnings.warn(
                colored(
                    "Quadrature energy integration method requires 'quad' pattern nodes, MHD energy calculated will be incorrect",
                    "yellow",
                )
            )

    @property
    def scalar(self):
        """bool : whether default "compute" method is a scalar or vector"""
        return True

    @property
    def name(self):
        """str : type of objective function"""
        return "energy"

    @property
    def derivatives(self):
        """ndarray : which derivatives are needed to compute"""
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

        if self.BC_constraint is not None and x.size == self.dimy:
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
        if self.BC_constraint is not None and x.size == self.dimy:
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
    R_transform,
    Z_transform,
    L_transform,
    Rb_transform,
    Zb_transform,
    p_transform,
    i_transform,
    BC_constraint=None,
    use_jit=True,
    devices=None,
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
    use_jit : bool
        whether to just-in-time compile the objective and derivatives
    devices : jax.device or list of jax.device, optional
        devices to jit compile to

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
            use_jit=use_jit,
            devices=devices,
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
            use_jit=use_jit,
            devices=devices,
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
