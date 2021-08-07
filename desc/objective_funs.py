import numpy as np
from abc import ABC, abstractmethod
from termcolor import colored
import warnings

from desc.backend import jnp, jit, use_jax
from desc.utils import unpack_state, Timer
from desc.io import IOAble
from desc.derivatives import Derivative
from desc.compute_funs import (
    compute_force_error_magnitude,
    compute_energy,
    compute_quasisymmetry,
)

__all__ = [
    "ForceErrorNodes",
    "ForceErrorGalerkin",
    "EnergyVolIntegral",
    "get_objective_function",
]


class ObjectiveFunction(IOAble, ABC):
    """Objective function used in the optimization of an Equilibrium.

    Parameters
    ----------
    R_transform : Transform
        transforms R_lmn coefficients to real space
    Z_transform : Transform
        transforms Z_lmn coefficients to real space
    L_transform : Transform
        transforms L_lmn coefficients to real space
    p_profile: Profile
        transforms p_l coefficients to real space
    i_profile: Profile
        transforms i_l coefficients to real space
    BC_constraint : BoundaryCondition
            linear constraint to enforce boundary conditions
    use_jit : bool, optional
        whether to just-in-time compile the objective and derivatives

    """

    _io_attrs_ = [
        "R_transform",
        "Z_transform",
        "L_transform",
        "p_profile",
        "i_profile",
        "BC_constraint",
        "use_jit",
    ]

    arg_names = {"Rb_lmn": 1, "Zb_lmn": 2, "p_l": 3, "i_l": 4, "Psi": 5}

    def __init__(
        self,
        R_transform,
        Z_transform,
        L_transform,
        p_profile,
        i_profile,
        BC_constraint,
        use_jit=True,
    ):

        self.R_transform = R_transform
        self.Z_transform = Z_transform
        self.L_transform = L_transform
        self.p_profile = p_profile
        self.i_profile = i_profile
        self.BC_constraint = BC_constraint
        self.use_jit = use_jit
        self._set_up()

    def _set_up(self):
        self.dimx = (
            self.R_transform.num_modes
            + self.Z_transform.num_modes
            + self.L_transform.num_modes
        )
        self.dimy = self.dimx if self.BC_constraint is None else self.BC_constraint.dimy
        self.dimf = self.R_transform.num_nodes + self.Z_transform.num_nodes
        self._check_transforms()
        self.set_derivatives(self.use_jit)
        self.compiled = False
        self.built = False
        if not self.use_jit:
            self.compiled = True

    def _check_transforms(self):
        """Make sure transforms can compute the correct derivatives."""
        if not all(
            (self.derivatives[:, None] == self.R_transform.derivatives).all(-1).any(-1)
        ):
            self.R_transform.change_derivatives(self.derivatives, build=False)
        if not all(
            (self.derivatives[:, None] == self.Z_transform.derivatives).all(-1).any(-1)
        ):
            self.Z_transform.change_derivatives(self.derivatives, build=False)
        if not all(
            (self.derivatives[:, None] == self.L_transform.derivatives).all(-1).any(-1)
        ):
            self.L_transform.change_derivatives(self.derivatives, build=False)

    def set_derivatives(self, use_jit=True, block_size="auto"):
        """Set up derivatives of the objective function.

        Parameters
        ----------
        use_jit : bool, optional
            whether to just-in-time compile the objective and derivatives

        """
        self._grad = Derivative(self.compute_scalar, mode="grad", use_jit=use_jit)

        self._hess = Derivative(
            self.compute_scalar,
            mode="hess",
            use_jit=use_jit,
            block_size=block_size,
            shape=(self.dimy, self.dimy),
        )
        self._jac = Derivative(
            self.compute,
            mode="fwd",
            use_jit=use_jit,
            block_size=block_size,
            shape=(self.dimf, self.dimy),
        )

        if use_jit:
            self.compute = jit(self.compute)
            self.compute_scalar = jit(self.compute_scalar)

    def build(self, rebuild=False, verbose=1):
        """Precompute the transform matrices to be used in optimization

        Parameters
        ----------
        rebuild : bool
            whether to force recalculation of transforms that have already been built
        verbose : int, optional
            level of output
        """
        if self.built and not rebuild:
            return
        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")
        if not self.R_transform.built:
            self.R_transform.build()
        if not self.Z_transform.built:
            self.Z_transform.build()
        if not self.L_transform.built:
            self.L_transform.build()
        if hasattr(self.p_profile, "_transform") and not (
            self.p_profile._transform.built
        ):
            self.p_profile._transform.build()
        if hasattr(self.i_profile, "_transform") and not (
            self.i_profile._transform.built
        ):
            self.i_profile._transform.build()
        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")
        self.built = True

    def compile(self, x, args, verbose=1, mode="auto"):
        """Call the necessary functions to ensure the function is compiled.

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
            "auto" compiles based on the type of objective,
            "all" compiles all derivatives

        """
        if not hasattr(self, "_grad"):
            self.set_derivatives()
        if not use_jax:
            self.compiled = True
            return

        timer = Timer()
        if mode == "auto" and self.scalar:
            mode = "scalar"
        elif mode == "auto":
            mode = "lsq"

        self.build(verbose=verbose)

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

    @property
    @abstractmethod
    def scalar(self):
        """Whether default "compute" method is a scalar or vector (bool)."""

    @property
    @abstractmethod
    def name(self):
        """Name of objective function (str)."""

    @property
    @abstractmethod
    def derivatives(self):
        """Which derivatives are needed to compute (ndarray)."""

    @abstractmethod
    def compute(self, *args):
        """Compute the objective function.

        Parameters
        ----------
        args : list
            (x, Rb_lmn, Zb_lmn, p_l, i_l, Psi)

        """

    @abstractmethod
    def compute_scalar(self, *args):
        """Compute the scalar form of the objective."""

    @abstractmethod
    def callback(self, *args):
        """Print the value of the objective."""

    def grad_x(self, *args):
        """Compute gradient vector of scalar form of the objective wrt to x."""
        return self._grad.compute(*args)

    def hess_x(self, *args):
        """Compute hessian matrix of scalar form of the objective wrt to x."""
        return self._hess.compute(*args)

    def jac_x(self, *args):
        """Compute jacobian matrx of vector form of the objective wrt to x."""
        return self._jac.compute(*args)

    def jvp(self, argnum, v, *args):
        """Compute jacobian-vector product of the objective function.

        Eg, df/dx*v

        Parameters
        ----------
        argnum : int or tuple of int
            integer describing which argument of the objective should be differentiated.
        v : ndarray or tuple of ndarray
            vector to multiply the jacobian matrix by, one per argnum
        args : list
            (x, Rb_lmn, Zb_lmn, p_l, i_l, Psi)

        Returns
        -------
        df : ndarray
            Jacobian vector product, summed over different argnums

        """
        f = Derivative.compute_jvp(self.compute, argnum, v, *args)
        return f

    def jvp2(self, argnum1, argnum2, v1, v2, *args):
        """Compute 2nd derivative jacobian-vector product of the objective function.

        Eg, d^2f/dx^2*v1*v2

        Parameters
        ----------
        argnum1, argnum2 : int or tuple of int
            integer describing which argument of the objective should be differentiated.
        v1, v2 : ndarray or tuple of ndarray
            vector to multiply the jacobian matrix by, one per argnum
        args : list
            (x, Rb_lmn, Zb_lmn, p_l, i_l, Psi)

        Returns
        -------
        d2f : ndarray
            Jacobian vector product

        """
        f = Derivative.compute_jvp2(self.compute, argnum1, argnum2, v1, v2, *args)
        return f

    def jvp3(self, argnum1, argnum2, argnum3, v1, v2, v3, *args):
        """Compute 3rd derivative jacobian-vector product of the objective function.

        Eg, d^3f/d3^2*v1*v2*v3

        Parameters
        ----------
        argnum1, argnum2, argnum2 : int or tuple of int
            integer describing which argument of the objective should be differentiated.
        v1, v2, v3 : ndarray or tuple of ndarray
            vector to multiply the jacobian matrix by, one per argnum
        args : list
            (x, Rb_lmn, Zb_lmn, p_l, i_l, Psi)

        Returns
        -------
        d3f : ndarray
            Jacobian vector product

        """
        f = Derivative.compute_jvp3(
            self.compute, argnum1, argnum2, argnum3, v1, v2, v3, *args
        )
        return f

    def derivative(self, argnums, *args):
        """Compute arbitrary derivatives of the objective function.

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
            (x, Rb_lmn, Zb_lmn, p_l, i_l, Psi)

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
            if isinstance(a, int) and a < 6:
                f = Derivative(f, argnum=a)
            elif isinstance(a, str) and a in ObjectiveFunction.arg_names:
                a = ObjectiveFunction.arg_names.get(a)
                f = Derivative(f, argnum=a)
            else:
                raise ValueError(
                    "argnums should be integers between 0 and 5 "
                    + "or one of {}, got {}".format(ObjectiveFunction.arg_names, a)
                )
            dims.append(args[a].size)

        return f(*args).reshape(tuple(dims))


class ForceErrorNodes(ObjectiveFunction):
    """Minimizes equilibrium force balance error in physical space.

    Parameters
    ----------
    R_transform : Transform
        transforms R_lmn coefficients to real space
    Z_transform : Transform
        transforms Z_lmn coefficients to real space
    L_transform : Transform
        transforms L_lmn coefficients to real space
    p_profile: Profile
        transforms p_l coefficients to real space
    i_profile: Profile
        transforms i_l coefficients to real space
    BC_constraint : BoundaryCondition
        linear constraint to enforce boundary conditions
    use_jit : bool, optional
        whether to just-in-time compile the objective and derivatives

    """

    @property
    def scalar(self):
        """Whether default "compute" method is a scalar or vector (bool)."""
        return False

    @property
    def name(self):
        """Name of objective function (str)."""
        return "force"

    @property
    def derivatives(self):
        """Which derivatives are needed to compute (ndarray)."""
        # TODO: different derivatives for R,Z,L,p,i ?
        # old axis derivatives
        # axis = np.array([[2, 1, 0], [1, 2, 0], [1, 1, 1], [2, 2, 0]])
        derivatives = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [2, 0, 0],
                [0, 2, 0],
                [0, 0, 2],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
            ]
        )
        return derivatives

    def compute(self, x, Rb_lmn, Zb_lmn, p_l, i_l, Psi):
        """Compute force balance error.

        Parameters
        ----------
        x : ndarray
            optimization state vector
        Rb_lmn : ndarray
            array of fourier coefficients for R boundary
        Zb_lmn : ndarray
            array of fourier coefficients for Z boundary
        p_l : ndarray
            series coefficients for pressure profile
        i_l : ndarray
            series coefficients for iota profile
        Psi : float
            toroidal flux within the last closed flux surface in webers

        Returns
        -------
        f : ndarray
            force error in radial and helical directions at each node

        """
        if self.BC_constraint is not None and x.size == self.dimy:
            # x is really 'y', need to recover full state vector
            x = self.BC_constraint.recover_from_constraints(x, Rb_lmn, Zb_lmn)

        R_lmn, Z_lmn, L_lmn = unpack_state(
            x, self.R_transform.basis.num_modes, self.Z_transform.basis.num_modes
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
            self.p_profile,
            self.i_profile,
        )

        weights = self.R_transform.grid.weights

        f_rho = (
            force_error["F_rho"] * force_error["|grad(rho)|"] * jacobian["g"] * weights
        )
        f_beta = force_error["F_beta"] * force_error["|beta|"] * jacobian["g"] * weights
        residual = jnp.concatenate([f_rho.flatten(), f_beta.flatten()])

        return residual

    def compute_scalar(self, x, Rb_lmn, Zb_lmn, p_l, i_l, Psi):
        """Compute the total force balance error.

        eg 1/2 sum(f**2)

        Parameters
        ----------
        x : ndarray
            optimization state vector
        Rb_lmn : ndarray
            array of fourier coefficients for R boundary
        Zb_lmn : ndarray
            array of fourier coefficients for Z boundary
        p_l : ndarray
            series coefficients for pressure profile
        i_l : ndarray
            series coefficients for iota profile
        Psi : float
            toroidal flux within the last closed flux surface in webers

        Returns
        -------
        f : float
            total force balance error
        """
        residual = self.compute(x, Rb_lmn, Zb_lmn, p_l, i_l, Psi)
        residual = 1 / 2 * jnp.sum(residual ** 2)
        return residual

    def callback(self, x, Rb_lmn, Zb_lmn, p_l, i_l, Psi):
        """Print the rms errors for radial and helical force balance.

        Parameters
        ----------
        x : ndarray
            optimization state vector
        Rb_lmn : ndarray
            array of fourier coefficients for R boundary
        Zb_lmn : ndarray
            array of fourier coefficients for Z boundary
        p_l : ndarray
            series coefficients for pressure profile
        i_l : ndarray
            series coefficients for iota profile
        Psi : float
            toroidal flux within the last closed flux surface in webers

        """
        if self.BC_constraint is not None and x.size == self.dimy:
            # x is really 'y', need to recover full state vector
            x = self.BC_constraint.recover_from_constraints(x, Rb_lmn, Zb_lmn)

        R_lmn, Z_lmn, L_lmn = unpack_state(
            x, self.R_transform.basis.num_modes, self.Z_transform.basis.num_modes
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
            self.p_profile,
            self.i_profile,
        )

        weights = self.R_transform.grid.weights

        f_rho = (
            force_error["F_rho"] * force_error["|grad(rho)|"] * jacobian["g"] * weights
        )
        f_beta = force_error["F_beta"] * force_error["|beta|"] * jacobian["g"] * weights

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
    """Minimizes the volume integral of MHD energy in physical space.

    W = integral of (B^2 / (2*mu0) - p) dV

    Parameters
    ----------
    R_transform : Transform
        transforms R_lmn coefficients to real space
    Z_transform : Transform
        transforms Z_lmn coefficients to real space
    L_transform : Transform
        transforms L_lmn coefficients to real space
    p_profile: Profile
        transforms p_l coefficients to real space
    i_profile: Profile
        transforms i_l coefficients to real space
    BC_constraint : BoundaryCondition
        linear constraint to enforce boundary conditions
    use_jit : bool, optional
        whether to just-in-time compile the objective and derivatives

    """

    def __init__(
        self,
        R_transform,
        Z_transform,
        L_transform,
        p_profile,
        i_profile,
        BC_constraint,
        use_jit=True,
    ):

        super().__init__(
            R_transform,
            Z_transform,
            L_transform,
            p_profile,
            i_profile,
            BC_constraint,
            use_jit,
        )

        if self.R_transform.grid.node_pattern != "quad":
            warnings.warn(
                colored(
                    "Energy method requires 'quad' pattern nodes, "
                    + "force error calculated will be incorrect.",
                    "yellow",
                )
            )

    @property
    def scalar(self):
        """Whether default "compute" method is a scalar or vector (bool)."""
        return True

    @property
    def name(self):
        """Name of objective function (str)."""
        return "energy"

    @property
    def derivatives(self):
        """Which derivatives are needed to compute (ndarray)."""
        # TODO: different derivatives for R,Z,L,p,i ?
        derivatives = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        return derivatives

    def compute(self, x, Rb_lmn, Zb_lmn, p_l, i_l, Psi):
        """Compute MHD energy.

        Parameters
        ----------
        x : ndarray
            optimization state vector
        Rb_lmn : ndarray
            array of fourier coefficients for R boundary
        Zb_lmn : ndarray
            array of fourier coefficients for Z boundary
        p_l : ndarray
            series coefficients for pressure profile
        i_l : ndarray
            series coefficients for iota profile
        Psi : float
            toroidal flux within the last closed flux surface in webers

        Returns
        -------
        W : float
            total MHD energy in the plasma volume

        """
        if self.BC_constraint is not None and x.size == self.dimy:
            # x is really 'y', need to recover full state vector
            x = self.BC_constraint.recover_from_constraints(x, Rb_lmn, Zb_lmn)

        R_lmn, Z_lmn, L_lmn = unpack_state(
            x, self.R_transform.basis.num_modes, self.Z_transform.basis.num_modes
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
            self.p_profile,
            self.i_profile,
        )

        residual = energy["W"]

        return residual

    def compute_scalar(self, x, Rb_lmn, Zb_lmn, p_l, i_l, Psi):
        """Compute MHD energy.

        Parameters
        ----------
        x : ndarray
            optimization state vector
        Rb_lmn : ndarray
            array of fourier coefficients for R boundary
        Zb_lmn : ndarray
            array of fourier coefficients for Z boundary
        p_l : ndarray
            series coefficients for pressure profile
        i_l : ndarray
            series coefficients for iota profile
        Psi : float
            toroidal flux within the last closed flux surface in webers

        Returns
        -------
        W : float
            total MHD energy in the plasma volume

        """
        residual = self.compute(x, Rb_lmn, Zb_lmn, p_l, i_l, Psi)
        return residual

    def callback(self, x, Rb_lmn, Zb_lmn, p_l, i_l, Psi):
        """Print the MHD energy.

        Parameters
        ----------
        x : ndarray
            optimization state vector
        Rb_lmn : ndarray
            array of fourier coefficients for R boundary
        Zb_lmn : ndarray
            array of fourier coefficients for Z boundary
        p_l : ndarray
            series coefficients for pressure profile
        i_l : ndarray
            series coefficients for iota profile
        Psi : float
            toroidal flux within the last closed flux surface in webers

        """
        if self.BC_constraint is not None and x.size == self.dimy:
            # x is really 'y', need to recover full state vector
            x = self.BC_constraint.recover_from_constraints(x, Rb_lmn, Zb_lmn)

        R_lmn, Z_lmn, L_lmn = unpack_state(
            x, self.R_transform.basis.num_modes, self.Z_transform.basis.num_modes
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
            self.p_profile,
            self.i_profile,
        )

        print(
            "Total MHD energy: {:10.3e}, ".format(energy["W"])
            + "Magnetic Energy: {:10.3e}, Pressure Energy: {:10.3e}".format(
                energy["W_B"], energy["W_p"]
            )
        )
        return None


class QuasisymmetryTripleProduct(ObjectiveFunction):
    """Maximizes quasisymmetry with the triple product definition.

    Parameters
    ----------
    R_transform : Transform
        transforms R_lmn coefficients to real space
    Z_transform : Transform
        transforms Z_lmn coefficients to real space
    L_transform : Transform
        transforms L_lmn coefficients to real space
    p_profile: Profile
        transforms p_l coefficients to real space
    i_profile: Profile
        transforms i_l coefficients to real space
    BC_constraint : BoundaryCondition
        linear constraint to enforce boundary conditions
    use_jit : bool, optional
        whether to just-in-time compile the objective and derivatives

    """

    @property
    def scalar(self):
        """Whether default "compute" method is a scalar or vector (bool)."""
        return False

    @property
    def name(self):
        """Name of objective function (str)."""
        return "qs_tp"

    @property
    def derivatives(self):
        """Which derivatives are needed to compute (ndarray)."""
        derivatives = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [2, 0, 0],
                [0, 2, 0],
                [0, 0, 2],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [0, 3, 0],
                [0, 0, 3],
                [1, 1, 1],
                [1, 2, 0],
                [1, 0, 2],
                [0, 2, 1],
                [0, 1, 2],
            ]
        )
        return derivatives

    def compute(self, x, Rb_lmn, Zb_lmn, p_l, i_l, Psi):
        """Compute quasisymmetry error.

        Parameters
        ----------
        x : ndarray
            optimization state vector
        Rb_lmn : ndarray
            array of fourier coefficients for R boundary
        Zb_lmn : ndarray
            array of fourier coefficients for Z boundary
        p_l : ndarray
            series coefficients for pressure profile
        i_l : ndarray
            series coefficients for iota profile
        Psi : float
            toroidal flux within the last closed flux surface in webers

        Returns
        -------
        f : ndarray
            force error in radial and helical directions at each node

        """
        if self.BC_constraint is not None and x.size == self.dimy:
            # x is really 'y', need to recover full state vector
            x = self.BC_constraint.recover_from_constraints(x, Rb_lmn, Zb_lmn)

        R_lmn, Z_lmn, L_lmn = unpack_state(
            x, self.R_transform.basis.num_modes, self.Z_transform.basis.num_modes
        )

        (
            quasisymmetry,
            current_density,
            magnetic_field,
            con_basis,
            jacobian,
            cov_basis,
            toroidal_coords,
            profiles,
        ) = compute_quasisymmetry(
            Psi,
            R_lmn,
            Z_lmn,
            L_lmn,
            p_l,
            i_l,
            self.R_transform,
            self.Z_transform,
            self.L_transform,
            self.p_profile,
            self.i_profile,
        )

        # QS triple product (T^4/m^2)
        QS = (
            profiles["psi_r"]
            * (
                magnetic_field["|B|_t"] * quasisymmetry["B*grad(|B|)_z"]
                - magnetic_field["|B|_z"] * quasisymmetry["B*grad(|B|)_t"]
            )
            / jacobian["g"]
        )

        # normalization factor = <|B|>^4 / R^2
        R0 = Rb_lmn[
            jnp.where((self.Rb_transform.basis.modes == [0, 0, 0]).all(axis=1))[0]
        ]
        norm = jnp.mean(magnetic_field["|B|"] * jacobian["g"]) / jnp.mean(jacobian["g"])
        return QS * R0 ** 2 / norm ** 4  # normalized QS error

    def compute_scalar(self, x, Rb_lmn, Zb_lmn, p_l, i_l, Psi):
        """Compute the volume averaged quasi-symmetry error.

        Parameters
        ----------
        x : ndarray
            optimization state vector
        Rb_lmn : ndarray
            array of fourier coefficients for R boundary
        Zb_lmn : ndarray
            array of fourier coefficients for Z boundary
        p_l : ndarray
            series coefficients for pressure profile
        i_l : ndarray
            series coefficients for iota profile
        Psi : float
            toroidal flux within the last closed flux surface in webers

        Returns
        -------
        f : float
            average quasi-symmetry error

        """
        if self.BC_constraint is not None and x.size == self.dimy:
            # x is really 'y', need to recover full state vector
            x = self.BC_constraint.recover_from_constraints(x, Rb_lmn, Zb_lmn)

        R_lmn, Z_lmn, L_lmn = unpack_state(
            x, self.R_transform.basis.num_modes, self.Z_transform.basis.num_modes
        )

        (
            quasisymmetry,
            current_density,
            magnetic_field,
            con_basis,
            jacobian,
            cov_basis,
            toroidal_coords,
            profiles,
        ) = compute_quasisymmetry(
            Psi,
            R_lmn,
            Z_lmn,
            L_lmn,
            p_l,
            i_l,
            self.R_transform,
            self.Z_transform,
            self.L_transform,
            self.p_profile,
            self.i_profile,
        )

        # QS triple product (T^4/m^2)
        QS = (
            profiles["psi_r"]
            * (
                magnetic_field["|B|_t"] * quasisymmetry["B*grad(|B|)_z"]
                - magnetic_field["|B|_z"] * quasisymmetry["B*grad(|B|)_t"]
            )
            / jacobian["g"]
        )

        # normalization factor = <|B|>^4 / R^2
        R0 = Rb_lmn[
            jnp.where((self.Rb_transform.basis.modes == [0, 0, 0]).all(axis=1))[0]
        ]
        norm = jnp.mean(magnetic_field["|B|"] * jacobian["g"]) / jnp.mean(jacobian["g"])
        f = QS * R0 ** 2 / norm ** 4  # normalized QS error
        return jnp.mean(jnp.abs(f) * jacobian["g"]) / jnp.mean(jacobian["g"])

    def callback(self, x, Rb_lmn, Zb_lmn, p_l, i_l, Psi):
        """Print the rms errors for quasisymmetry.

        Parameters
        ----------
        x : ndarray
            optimization state vector
        Rb_lmn : ndarray
            array of fourier coefficients for R boundary
        Zb_lmn : ndarray
            array of fourier coefficients for Z boundary
        p_l : ndarray
            series coefficients for pressure profile
        i_l : ndarray
            series coefficients for iota profile
        Psi : float
            toroidal flux within the last closed flux surface in webers

        """
        residual = self.compute(x, Rb_lmn, Zb_lmn, p_l, i_l, Psi)
        resid_rms = 1 / 2 * jnp.sum(residual ** 2)

        print("Residual: {:10.3e}".format(resid_rms))
        return None


class QuasisymmetryFluxFunction(ObjectiveFunction):
    """Maximizes quasisymmetry with the flux function definition.

    Parameters
    ----------
    R_transform : Transform
        transforms R_lmn coefficients to real space
    Z_transform : Transform
        transforms Z_lmn coefficients to real space
    L_transform : Transform
        transforms L_lmn coefficients to real space
    p_profile: Profile
        transforms p_l coefficients to real space
    i_profile: Profile
        transforms i_l coefficients to real space
    BC_constraint : BoundaryCondition
        linear constraint to enforce boundary conditions
    use_jit : bool, optional
        whether to just-in-time compile the objective and derivatives

    """

    def __init__(
        self,
        R_transform,
        Z_transform,
        L_transform,
        p_profile,
        i_profile,
        BC_constraint,
        use_jit=True,
    ):

        super().__init__(
            R_transform,
            Z_transform,
            L_transform,
            p_profile,
            i_profile,
            BC_constraint,
            use_jit,
        )

        rho_vals = np.unique(self.R_transform.grid.nodes[:, 0])
        if rho_vals.size != 1:
            warnings.warn(
                colored(
                    "QS Flux Function requires nodes on a single flux surface, "
                    + "quasisymmetry error calculated will be incorrect.",
                    "yellow",
                )
            )

    @property
    def scalar(self):
        """Whether default "compute" method is a scalar or vector (bool)."""
        return False

    @property
    def name(self):
        """Name of objective function (str)."""
        return "qs_ff"

    @property
    def derivatives(self):
        """Which derivatives are needed to compute (ndarray)."""
        derivatives = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [2, 0, 0],
                [0, 2, 0],
                [0, 0, 2],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [0, 3, 0],
                [0, 0, 3],
                [1, 1, 1],
                [1, 2, 0],
                [1, 0, 2],
                [0, 2, 1],
                [0, 1, 2],
            ]
        )
        return derivatives

    def compute(self, x, Rb_lmn, Zb_lmn, p_l, i_l, Psi):
        """Compute quasisymmetry error.

        Parameters
        ----------
        x : ndarray
            optimization state vector
        Rb_lmn : ndarray
            array of fourier coefficients for R boundary
        Zb_lmn : ndarray
            array of fourier coefficients for Z boundary
        p_l : ndarray
            series coefficients for pressure profile
        i_l : ndarray
            series coefficients for iota profile
        Psi : float
            toroidal flux within the last closed flux surface in webers

        Returns
        -------
        f : ndarray
            force error in radial and helical directions at each node

        """
        if self.BC_constraint is not None and x.size == self.dimy:
            # x is really 'y', need to recover full state vector
            x = self.BC_constraint.recover_from_constraints(x, Rb_lmn, Zb_lmn)

        R_lmn, Z_lmn, L_lmn = unpack_state(
            x, self.R_transform.basis.num_modes, self.Z_transform.basis.num_modes
        )

        (
            quasisymmetry,
            current_density,
            magnetic_field,
            con_basis,
            jacobian,
            cov_basis,
            toroidal_coords,
            profiles,
        ) = compute_quasisymmetry(
            Psi,
            R_lmn,
            Z_lmn,
            L_lmn,
            p_l,
            i_l,
            self.R_transform,
            self.Z_transform,
            self.L_transform,
            self.p_profile,
            self.i_profile,
        )

        # M/N (type of QS)
        helicity = 1.0 / 1.0

        # covariant Boozer components
        G = jnp.mean(magnetic_field["B_zeta"] * jacobian["g"]) / jnp.mean(
            jacobian["g"]
        )  # poloidal current
        I = jnp.mean(magnetic_field["B_theta"] * jacobian["g"]) / jnp.mean(
            jacobian["g"]
        )  # toroidal current

        # flux function C=C(rho)
        C = (helicity * G + I) / (helicity * profiles["iota"] - 1)

        # QS flux function (T^3)
        QS = (
            profiles["psi_r"]
            / jacobian["g"]
            * (
                magnetic_field["B_zeta"] * magnetic_field["|B|_t"]
                - magnetic_field["B_theta"] * magnetic_field["|B|_z"]
            )
            - C * quasisymmetry["B*grad(|B|)"]
        )

        # normalization factor = <|B|>^3
        norm = jnp.mean(magnetic_field["|B|"] * jacobian["g"]) / jnp.mean(jacobian["g"])
        return QS / norm ** 3  # normalized QS error

    def compute_scalar(self, x, Rb_lmn, Zb_lmn, p_l, i_l, Psi):
        """Compute the volume averaged quasi-symmetry error.

        Parameters
        ----------
        x : ndarray
            optimization state vector
        Rb_lmn : ndarray
            array of fourier coefficients for R boundary
        Zb_lmn : ndarray
            array of fourier coefficients for Z boundary
        p_l : ndarray
            series coefficients for pressure profile
        i_l : ndarray
            series coefficients for iota profile
        Psi : float
            toroidal flux within the last closed flux surface in webers

        Returns
        -------
        f : float
            average quasi-symmetry error

        """
        if self.BC_constraint is not None and x.size == self.dimy:
            # x is really 'y', need to recover full state vector
            x = self.BC_constraint.recover_from_constraints(x, Rb_lmn, Zb_lmn)

        R_lmn, Z_lmn, L_lmn = unpack_state(
            x, self.R_transform.basis.num_modes, self.Z_transform.basis.num_modes
        )

        (
            quasisymmetry,
            current_density,
            magnetic_field,
            con_basis,
            jacobian,
            cov_basis,
            toroidal_coords,
            profiles,
        ) = compute_quasisymmetry(
            Psi,
            R_lmn,
            Z_lmn,
            L_lmn,
            p_l,
            i_l,
            self.R_transform,
            self.Z_transform,
            self.L_transform,
            self.p_profile,
            self.i_profile,
        )

        # covariant Boozer components
        G = jnp.mean(magnetic_field["B_zeta"] * jacobian["g"]) / jnp.mean(
            jacobian["g"]
        )  # poloidal current
        I = jnp.mean(magnetic_field["B_theta"] * jacobian["g"]) / jnp.mean(
            jacobian["g"]
        )  # toroidal current

        helicity = 1.0 / 1.0  # M/N (type of QS)
        # flux function C=C(rho)
        C = (helicity * G + I) / (helicity * profiles["iota"] - 1)

        # QS flux function (T^3)
        QS = (
            profiles["psi_r"]
            / jacobian["g"]
            * (
                magnetic_field["B_zeta"] * magnetic_field["|B|_t"]
                - magnetic_field["B_theta"] * magnetic_field["|B|_z"]
            )
            - C * quasisymmetry["B*grad(|B|)"]
        )

        # normalization factor = <|B|>^3
        norm = jnp.mean(magnetic_field["|B|"] * jacobian["g"]) / jnp.mean(jacobian["g"])
        f = QS / norm ** 3  # normalized QS error
        return jnp.mean(jnp.abs(f) * jacobian["g"]) / jnp.mean(jacobian["g"])

    def callback(self, x, Rb_lmn, Zb_lmn, p_l, i_l, Psi):
        """Print the rms errors for quasisymmetry.

        Parameters
        ----------
        x : ndarray
            optimization state vector
        Rb_lmn : ndarray
            array of fourier coefficients for R boundary
        Zb_lmn : ndarray
            array of fourier coefficients for Z boundary
        p_l : ndarray
            series coefficients for pressure profile
        i_l : ndarray
            series coefficients for iota profile
        Psi : float
            toroidal flux within the last closed flux surface in webers

        """
        residual = self.compute(x, Rb_lmn, Zb_lmn, p_l, i_l, Psi)
        resid_rms = 1 / 2 * jnp.sum(residual ** 2)

        print("Residual: {:10.3e}".format(resid_rms))
        return None


def get_objective_function(
    objective,
    R_transform,
    Z_transform,
    L_transform,
    p_profile,
    i_profile,
    BC_constraint=None,
    use_jit=True,
):
    """Get an objective function by name.

    Parameters
    ----------
    objective : str
        name of the desired objective function, eg ``'force'`` or ``'energy'``
    R_transform : Transform
        transforms R_lmn coefficients to real space
    Z_transform : Transform
        transforms Z_lmn coefficients to real space
    L_transform : Transform
        transforms L_lmn coefficients to real space
    p_profile: Profile
        transforms p_l coefficients to real space
    i_profile: Profile
        transforms i_l coefficients to real space
    BC_constraint : BoundaryCondition
        linear constraint to enforce boundary conditions
    use_jit : bool
        whether to just-in-time compile the objective and derivatives

    Returns
    -------
    obj_fun : ObjectiveFunction
        objective initialized with the given transforms and constraints

    """
    if objective == "force":
        obj_fun = ForceErrorNodes(
            R_transform=R_transform,
            Z_transform=Z_transform,
            L_transform=L_transform,
            p_profile=p_profile,
            i_profile=i_profile,
            BC_constraint=BC_constraint,
            use_jit=use_jit,
        )
    elif objective == "galerkin":
        obj_fun = ForceErrorGalerkin(
            R_transform=R_transform,
            Z_transform=Z_transform,
            L_transform=L_transform,
            p_profile=p_profile,
            i_profile=i_profile,
            BC_constraint=BC_constraint,
            use_jit=use_jit,
        )
    elif objective == "energy":
        obj_fun = EnergyVolIntegral(
            R_transform=R_transform,
            Z_transform=Z_transform,
            L_transform=L_transform,
            p_profile=p_profile,
            i_profile=i_profile,
            BC_constraint=BC_constraint,
            use_jit=use_jit,
        )
    else:
        raise ValueError(
            colored(
                "Requested Objective Function is not implemented. "
                + "Available objective functions are: "
                + "'force', 'lambda', 'galerkin', 'energy'",
                "red",
            )
        )

    return obj_fun
