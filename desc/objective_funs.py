import numpy as np
from abc import ABC, abstractmethod
from termcolor import colored
import warnings
import scipy.special

from desc.backend import jnp, jit
from desc.utils import unpack_state, equals, Timer
from desc.io import IOAble
from desc.derivatives import Derivative
from desc.transform import Transform
from desc.boundary_conditions import BoundaryConstraint
from desc.compute_funs import (
    compute_force_error,
    compute_force_error_magnitude,
    dot,
    compute_energy,
)
from desc.grid import Grid, LinearGrid, ConcentricGrid, QuadratureGrid
from desc.basis import (
    PowerSeries,
    FourierSeries,
    DoubleFourierSeries,
    FourierZernikeBasis,
)

__all__ = [
    "ForceErrorNodes",
    "ForceErrorGalerkin",
    "ForceConstraintNodes",
    "EnergyVolIntegral",
    "get_objective_function",
]


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
        devices to jit compile to. If None, use the default devices

    """

    _io_attrs_ = [
        "R_transform",
        "Z_transform",
        "L_transform",
        "Rb_transform",
        "Zb_transform",
        "p_transform",
        "i_transform",
        "BC_constraint",
    ]

    _object_lib_ = {"Transform": Transform, "BoundaryConstraint": BoundaryConstraint}
    _object_lib_.update(Transform._object_lib_)

    arg_names = {"Rb_mn": 1, "Zb_mn": 2, "p_l": 3, "i_l": 4, "Psi": 5, "zeta_ratio": 6}

    def __init__(
        self,
        R_transform=None,
        Z_transform=None,
        L_transform=None,
        Rb_transform=None,
        Zb_transform=None,
        p_transform=None,
        i_transform=None,
        BC_constraint=None,
        use_jit=True,
        devices=None,
        load_from=None,
        file_format=None,
        obj_lib=None,
    ):

        if load_from is None:
            self.R_transform = R_transform
            self.Z_transform = Z_transform
            self.L_transform = L_transform
            self.Rb_transform = Rb_transform
            self.Zb_transform = Zb_transform
            self.p_transform = p_transform
            self.i_transform = i_transform
            self.BC_constraint = BC_constraint

        else:
            self._init_from_file_(
                load_from=load_from, file_format=file_format, obj_lib=obj_lib
            )
        self.dimx = (
            self.R_transform.num_modes
            + self.Z_transform.num_modes
            + self.L_transform.num_modes
        )
        self.dimy = self.dimx if self.BC_constraint is None else self.BC_constraint.dimy
        self.dimf = self.R_transform.num_nodes + self.Z_transform.num_nodes
        if not isinstance(devices, (list, tuple)):
            devices = [devices]
        self._check_transforms()
        self.set_derivatives(use_jit, devices)
        self.compiled = False
        if not use_jit:
            self.compiled = True

    def _check_transforms(self):
        """makes sure transforms can compute the correct derivatives"""
        if not all(
            (self.derivatives[:, None] == self.R_transform.derivatives).all(-1).any(-1)
        ):
            self.R_transform.change_derivatives(self.derivatives)
        if not all(
            (self.derivatives[:, None] == self.Z_transform.derivatives).all(-1).any(-1)
        ):
            self.Z_transform.change_derivatives(self.derivatives)
        if not all(
            (self.derivatives[:, None] == self.L_transform.derivatives).all(-1).any(-1)
        ):
            self.L_transform.change_derivatives(self.derivatives)
        if not all(
            (self.derivatives[:, None] == self.Rb_transform.derivatives).all(-1).any(-1)
        ):
            self.Rb_transform.change_derivatives(self.derivatives)
        if not all(
            (self.derivatives[:, None] == self.Zb_transform.derivatives).all(-1).any(-1)
        ):
            self.Zb_transform.change_derivatives(self.derivatives)
        if not all(
            (self.derivatives[:, None] == self.p_transform.derivatives).all(-1).any(-1)
        ):
            self.p_transform.change_derivatives(self.derivatives)
        if not all(
            (self.derivatives[:, None] == self.i_transform.derivatives).all(-1).any(-1)
        ):
            self.i_transform.change_derivatives(self.derivatives)

    def set_derivatives(self, use_jit=True, devices=[None], block_size="auto"):
        """Set up derivatives of the objective function

        Parameters
        ----------
        use_jit : bool, optional
            whether to just-in-time compile the objective and derivatives
        devices : jax.device or list of jax.device, optional
            devices to jit compile to. If None, use the default devices

        """
        if block_size == "auto":
            block_size = np.inf  # TODO: correct automatic sizing based on avail mem

        if not isinstance(devices, (list, tuple)):
            devices = [devices]

        self._grad = Derivative(
            self.compute_scalar, mode="grad", use_jit=use_jit, device=devices[0]
        )

        self._hess = Derivative(
            self.compute_scalar,
            mode="hess",
            use_jit=use_jit,
            devices=devices,
            block_size=block_size,
            shape=(self.dimy, self.dimy),
        )
        self._jac = Derivative(
            self.compute,
            mode="fwd",
            use_jit=use_jit,
            devices=devices,
            block_size=block_size,
            shape=(self.dimf, self.dimy),
        )

        if use_jit:
            self.compute = jit(self.compute, device=devices[0])
            self.compute_scalar = jit(self.compute_scalar, device=devices[0])

    def compile(self, x, args, verbose=1, mode="auto"):
        """Calls the necessary functions to ensure the function is compiled

        Parameters
        ----------
        x : ndarray
            any array of the correct shape to trigger jit compilation
        args : tuple
            additional arguments passed to objective function and derivatives
        verbose : int, optional
            level of output
        mode : {"auto", "lsq", "scalar", "all"}
            whether to compile for least squares optimization or scalar optimization.
            "auto" compiles based on the type of objective, "all" compiles all derivatives


        """
        if not hasattr(self, "_grad"):
            self.set_derivatives()

        timer = Timer()
        if mode == "auto" and self.scalar:
            mode = "scalar"
        elif mode == "auto":
            mode = "lsq"

        if verbose > 0:
            print("Compiling objective function and derivatives")
        timer.start("Total compilation time")

        if mode in ["scalar", "all"]:
            timer.start("Objective compilation time")
            f0 = self.compute_scalar(x, *args).block_until_ready()
            timer.stop("Objective compilation time")
            if verbose > 1:
                timer.disp("Objective compilation time")
            timer.start("Gradient compilation time")
            g0 = self.grad_x(x, *args).block_until_ready()
            timer.stop("Gradient compilation time")
            if verbose > 1:
                timer.disp("Gradient compilation time")
            timer.start("Hessian compilation time")
            H0 = self.hess_x(x, *args).block_until_ready()
            timer.stop("Hessian compilation time")
            if verbose > 1:
                timer.disp("Hessian compilation time")
        if mode in ["lsq", "all"]:
            timer.start("Objective compilation time")
            f0 = self.compute(x, *args).block_until_ready()
            timer.stop("Objective compilation time")
            if verbose > 1:
                timer.disp("Objective compilation time")
            timer.start("Jacobian compilation time")
            J0 = self.jac_x(x, *args).block_until_ready()
            timer.stop("Jacobian compilation time")
            if verbose > 1:
                timer.disp("Jacobian compilation time")

        timer.stop("Total compilation time")
        if verbose > 1:
            timer.disp("Total compilation time")
        self.compiled = True

    # note: we can't override __eq__ here because that breaks the hashing that jax uses
    # when jitting functions
    def eq(self, other):
        """Test for equivalence between objectives

        Parameters
        ----------
        other : ObjectiveFunction
            another ObjectiveFunction object to compare to

        Returns
        -------
        bool
            True if other is an ObjectiveFunction with the same attributes as self
            False otherwise

        """
        if self.__class__ != other.__class__:
            return False
        ignore_keys = [
            "_grad",
            "_jac",
            "_hess",
            "compute",
            "compute_scalar",
        ]
        dict1 = {
            key: val for key, val in self.__dict__.items() if key not in ignore_keys
        }
        dict2 = {
            key: val for key, val in other.__dict__.items() if key not in ignore_keys
        }
        return equals(dict1, dict2)

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


class ForceErrorGalerkin(ObjectiveFunction):
    """Minimizes spectral coefficients of force balance residual

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
        devices to jit compile to. If None, use the default devices

    """

    def __init__(
        self,
        R_transform=None,
        Z_transform=None,
        L_transform=None,
        Rb_transform=None,
        Zb_transform=None,
        p_transform=None,
        i_transform=None,
        BC_constraint=None,
        use_jit=True,
        devices=None,
        load_from=None,
        file_format=None,
        obj_lib=None,
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
            load_from,
            file_format,
            obj_lib,
        )

        if self.R_transform.grid.node_pattern != "quad":
            warnings.warn(
                colored(
                    "Galerkin method requires 'quad' pattern nodes, force error calculated will be incorrect",
                    "yellow",
                )
            )

    @property
    def scalar(self):
        """bool : whether default "compute" method is a scalar or vector"""
        return False

    @property
    def name(self):
        """str : type of objective function"""
        return "galerkin"

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

    def _residuals(self, x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio=1.0):
        """Compute force balance residuals at collocation nodes.

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
            jacobian,
            cov_basis,
            toroidal_coords,
            profiles,
        ) = compute_force_error(
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

        F_R = (
            force_error["F_rho"] * toroidal_coords["R_r"]
            + force_error["F_theta"] * toroidal_coords["R_t"]
            + force_error["F_zeta"] * toroidal_coords["R_z"]
        )
        F_Z = (
            force_error["F_rho"] * toroidal_coords["Z_r"]
            + force_error["F_theta"] * toroidal_coords["Z_t"]
            + force_error["F_zeta"] * toroidal_coords["Z_z"]
        )

        return F_R, F_Z

    def compute(self, x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio=1.0):
        """Compute spectral coefficients of force balance residual by quadrature.

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
        F_R, F_Z = self._residuals(x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio)
        weights = self.R_transform.grid.weights

        f_R = jnp.dot(self.R_transform._matrices[0][0][0].T, F_R * weights)
        f_Z = jnp.dot(self.Z_transform._matrices[0][0][0].T, F_Z * weights)

        residual = jnp.concatenate([f_R.flatten(), f_Z.flatten()])
        return residual

    def compute_scalar(self, x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio=1.0):
        """Compute the integral of the force balance residual by quadrature.

        eg int(|F_R| + |F_Z|)

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
        F_R, F_Z = self._residuals(x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio)
        weights = self.R_transform.grid.weights
        return jnp.sum((jnp.abs(F_R) + jnp.abs(F_Z)) * weights)

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
        F_R, F_Z = self._residuals(x, Rb_mn, Zb_mn, p_l, i_l, Psi, zeta_ratio)
        weights = self.R_transform.grid.weights
        F_R_int = jnp.sum(jnp.abs(F_R) * weights)
        F_Z_int = jnp.sum(jnp.abs(F_Z) * weights)
        F_T_int = jnp.sum((jnp.abs(F_R) + jnp.abs(F_Z)) * weights)

        print(
            "int(|F_R|+|F_Z|): {:10.3e}  int(|F_R|): {:10.3e}  int(|F_Z|): {:10.3e}".format(
                F_T_int, F_R_int, F_Z_int
            )
        )

        return None


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
        devices to jit compile to. If None, use the default devices

    """

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


class ForceConstraintNodes(ObjectiveFunction):
    """Minimizes equilibrium force balance error and lambda constraint in physical space

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
        devices to jit compile to. If None, use the default devices

    """

    @property
    def scalar(self):
        """bool : whether default "compute" method is a scalar or vector"""
        return False

    @property
    def name(self):
        """str : type of objective function"""
        return "lambda"

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
        f_lambda = (
            toroidal_coords["R_t"] * toroidal_coords["R_tt"]
            + toroidal_coords["Z_t"] * toroidal_coords["Z_tt"]
        )
        residual = jnp.concatenate(
            [f_rho.flatten(), f_beta.flatten(), f_lambda.flatten()]
        )

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
        f_lambda = (
            toroidal_coords["R_t"] * toroidal_coords["R_tt"]
            + toroidal_coords["Z_t"] * toroidal_coords["Z_tt"]
        )

        f_rho_rms = jnp.sqrt(jnp.sum(f_rho ** 2))
        f_beta_rms = jnp.sqrt(jnp.sum(f_beta ** 2))
        f_lambda_rms = jnp.sqrt(jnp.sum(f_lambda ** 2))

        residual = jnp.concatenate(
            [f_rho.flatten(), f_beta.flatten(), f_lambda.flatten()]
        )
        resid_rms = 1 / 2 * jnp.sum(residual ** 2)

        print(
            "Total residual: {:10.3e}  f_rho: {:10.3e}  f_beta: {:10.3e}  f_lambda: {:10.3e}".format(
                resid_rms, f_rho_rms, f_beta_rms, f_lambda_rms
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
    use_jit : bool, optional
        whether to just-in-time compile the objective and derivatives
    devices : jax.device or list of jax.device, optional
        devices to jit compile to. If None, use the default devices

    """

    def __init__(
        self,
        R_transform=None,
        Z_transform=None,
        L_transform=None,
        Rb_transform=None,
        Zb_transform=None,
        p_transform=None,
        i_transform=None,
        BC_constraint=None,
        use_jit=True,
        devices=None,
        load_from=None,
        file_format=None,
        obj_lib=None,
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
            load_from,
            file_format,
            obj_lib,
        )

        if self.R_transform.grid.node_pattern != "quad":
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
    objective,
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
    objective : str
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

    if objective == "force":
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
    elif objective == "galerkin":
        obj_fun = ForceErrorGalerkin(
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
    elif objective == "lambda":
        obj_fun = ForceConstraintNodes(
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
    elif objective == "energy":
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
                + "Available objective functions are: 'force', 'lambda', 'galerkin', 'energy'",
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
