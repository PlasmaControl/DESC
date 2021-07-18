import numpy as np
from abc import ABC, abstractmethod
from termcolor import colored
import warnings
from scipy.constants import mu_0

from desc.backend import jnp, jit, use_jax
from desc.utils import Timer
from desc.io import IOAble
from desc.derivatives import Derivative
from desc.grid import QuadratureGrid
from desc.transform import Transform
from desc.geometry import FourierRZToroidalSurface, ZernikeRZToroidalSection
from desc.compute_funs import (
    compute_pressure,
    compute_jacobian,
    compute_magnetic_field_magnitude,
)

__all__ = [
    "FixedBoundary",
    "LCFSBoundary",
    "Volume",
    "Energy",
]


class ObjectiveFunction(IOAble):
    """Objective function comprised of one or more Objectives."""

    _io_attrs_ = ["objectives", "constraints"]
    _arg_order_ = ("R", "Z", "L", "Rb", "Zb", "pressure", "iota", "Psi")

    def __init__(self, objectives, constraints, eq=None, use_jit=True):
        """Initialize an Objective Function.

        Parameters
        ----------
        objectives : Objective, tuple
            List of objectives to be targeted during optimization.
        constraints : Objective, tuple
            List of objectives to be used as constraints during optimization.
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.

        """
        if not isinstance(objectives, tuple):
            objectives = (objectives,)
        if not isinstance(constraints, tuple):
            constraints = (constraints,)

        self._objectives = objectives
        self._constraints = constraints
        self._use_jit = use_jit
        self._built = False
        self._compiled = False

        if eq is not None:
            self.build(eq)

    def _set_constraint_derivatives(self, use_jit=True, block_size="auto"):
        """Set up derivatives of the constraint functions.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.

        """
        self._derivatives = {}
        self._derivatives["R"] = Derivative(
            self.compute_constraints,
            argnum=0,
            mode="fwd",
            use_jit=use_jit,
            block_size=block_size,
            shape=(self._dim_c, self._dim_R),
        )
        self._derivatives["Z"] = Derivative(
            self.compute_constraints,
            argnum=1,
            mode="fwd",
            use_jit=use_jit,
            block_size=block_size,
            shape=(self._dim_c, self._dim_Z),
        )
        self._derivatives["L"] = Derivative(
            self.compute_constraints,
            argnum=2,
            mode="fwd",
            use_jit=use_jit,
            block_size=block_size,
            shape=(self._dim_c, self._dim_L),
        )
        self._derivatives["Rb"] = Derivative(
            self.compute_constraints,
            argnum=3,
            mode="fwd",
            use_jit=use_jit,
            block_size=block_size,
            shape=(self._dim_c, self._dim_Rb),
        )
        self._derivatives["Zb"] = Derivative(
            self.compute_constraints,
            argnum=4,
            mode="fwd",
            use_jit=use_jit,
            block_size=block_size,
            shape=(self._dim_c, self._dim_Zb),
        )
        self._derivatives["pressure"] = Derivative(
            self.compute_constraints,
            argnum=5,
            mode="fwd",
            use_jit=use_jit,
            block_size=block_size,
            shape=(self._dim_c, self._dim_p),
        )
        self._derivatives["iota"] = Derivative(
            self.compute_constraints,
            argnum=6,
            mode="fwd",
            use_jit=use_jit,
            block_size=block_size,
            shape=(self._dim_c, self._dim_i),
        )
        self._derivatives["Psi"] = Derivative(
            self.compute_constraints,
            argnum=7,
            mode="fwd",
            use_jit=use_jit,
            block_size=block_size,
            shape=(self._dim_c, 1),
        )

        if use_jit:
            self.compute_constraints = jit(self.compute_constraints)

    def _set_objective_derivatives(self, use_jit=True, block_size="auto"):
        """Set up derivatives of the objective functions.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.

        """
        self._grad = Derivative(self.compute_scalar, mode="grad", use_jit=use_jit)
        self._hess = Derivative(
            self.compute_scalar,
            mode="hess",
            use_jit=use_jit,
            block_size=block_size,
            shape=(self._dim_y, self._dim_y),
        )
        self._jac = Derivative(
            self.compute,
            mode="fwd",
            use_jit=use_jit,
            block_size=block_size,
            shape=(self._dim_f, self._dim_y),
        )

        if use_jit:
            self.compute = jit(self.compute)
            self.compute_scalar = jit(self.compute_scalar)

    def _build_linear_constraints(self):
        """Compute and factorize A to get pseudoinverse and nullspace."""
        # constraints are linear so variable values are irrelevant
        args = (
            np.zeros((self._dim_R,)),
            np.zeros((self._dim_Z,)),
            np.zeros((self._dim_L,)),
            np.zeros((self._dim_Rb,)),
            np.zeros((self._dim_Zb,)),
            np.zeros((self._dim_p,)),
            np.zeros((self._dim_i,)),
            np.zeros((self._dim_Psi,)),
        )

        # A = dc/dx
        self._A = np.array([])
        for arg in self._arg_order_:
            A = self._derivatives[arg].compute(*args)
            self._A = np.hstack((self._A, A)) if self._A.size else A

        # c = A*x - b
        self._b = np.array([])
        for obj in self._constraints:
            b = obj.target
            self._b = np.hstack((self._b, b)) if self._b.size else b

        # remove duplicate constraints
        temp = np.hstack([self._A, self._b.reshape((-1, 1))])
        temp = np.unique(temp, axis=0)
        self._A = np.atleast_2d(temp[:, :-1])
        self._b = temp[:, -1].flatten()

        # SVD of A
        u, s, vh = np.linalg.svd(self._A, full_matrices=True)
        M, N = u.shape[0], vh.shape[1]
        K = min(M, N)
        rcond = np.finfo(self._A.dtype).eps * max(M, N)

        # Z = null space of A
        tol = np.amax(s) * rcond
        large = s > tol
        num = np.sum(large, dtype=int)
        self._Z = vh[num:, :].T.conj()
        self._dim_y = self._Z.shape[1]

        uk = u[:, :K]
        vhk = vh[:K, :]
        s = np.divide(1, s, where=large, out=s)
        s[(~large,)] = 0
        self._Ainv = np.matmul(vhk.T, np.multiply(s[..., np.newaxis], uk.T))
        self._x0 = np.dot(self._Ainv, self._b)

    def build(self, eq, verbose=1):
        """Build the constraints and objectives.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        verbose : int, optional
            Level of output.

        """
        timer = Timer()
        timer.start("Objecive build")

        # state vector component dimensions
        self._dim_R = eq.R_basis.num_modes
        self._dim_Z = eq.Z_basis.num_modes
        self._dim_L = eq.L_basis.num_modes
        self._dim_Rb = eq.surface.R_basis.num_modes
        self._dim_Zb = eq.surface.Z_basis.num_modes
        self._dim_p = eq.pressure.params.size
        self._dim_i = eq.iota.params.size
        self._dim_Psi = 1
        self._dim_x = (
            self._dim_R
            + self._dim_Z
            + self._dim_L
            + self._dim_Rb
            + self._dim_Zb
            + self._dim_p
            + self._dim_i
            + self._dim_Psi
        )

        # build constraints
        self._dim_c = 0
        for constraint in self._constraints:
            if not constraint.linear:
                raise NotImplementedError("Constraints must be linear.")
            if not constraint.built:
                if verbose > 0:
                    print("Building constraint: " + constraint.name)
                constraint.build(eq, verbose=verbose)
            self._dim_c += constraint.dim_f

        # build objectives
        self._dim_f = 0
        self._scalar = True
        for objective in self._objectives:
            if not objective.scalar:
                self._scalar = False
            if not objective.built:
                if verbose > 0:
                    print("Building objective: " + objective.name)
                objective.build(eq, verbose=verbose)
            self._dim_f += objective.dim_f

        self._set_constraint_derivatives(self._use_jit)

        # build linear constraint matrices
        if verbose > 0:
            print("Building linear constraints")
        timer.start("linear constraint build")
        self._build_linear_constraints()
        timer.stop("linear constraint build")
        if verbose > 1:
            timer.disp("linear constraint build")

        self._set_objective_derivatives(self._use_jit)

        self._built = True
        timer.stop("Objecive build")
        if verbose > 1:
            timer.disp("Objecive build")

    def compute_constraints(self, R_lmn, Z_lmn, L_lmn, Rb_lmn, Zb_lmn, p_l, i_l, Psi):
        """Compute the constraint equations.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.
        L_lmn : ndarray
            Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
        Rb_lmn : ndarray
            Spectral coefficients of Rb(rho,theta,zeta) -- boundary R coordinate.
        Zb_lmn : ndarray
            Spectral coefficients of Zb(rho,theta,zeta) -- boundary Z coordiante.
        p_l : ndarray
            Spectral coefficients of p(rho) -- pressure profile.
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.
        Psi : float
            Total toroidal flux within the last closed flux surface, in Webers.

        Returns
        -------
        f : float, ndarray
            Constraint equation errors.

        """
        kwargs = {
            "R_lmn": R_lmn,
            "Z_lmn": Z_lmn,
            "L_lmn": L_lmn,
            "Rb_lmn": Rb_lmn,
            "Zb_lmn": Zb_lmn,
            "p_l": p_l,
            "i_l": i_l,
            "Psi": Psi,
        }
        f = jnp.array([obj.compute(**kwargs) for obj in self._constraints])
        return jnp.concatenate(f)

    def compute(self, x):
        """Compute the objective function.

        Parameters
        ----------
        x : ndarray
            Full state vector x or optimization variable y.

        Returns
        -------
        f : float, ndarray
            Objective function value(s).

        """
        if x.size == self._dim_y:
            x = self.recover(x)  # x is really y
        if x.size != self._dim_x:
            raise ValueError("State vector is not the proper size.")
        kwargs = self.unpack_state(x)

        f = jnp.array([obj.compute(**kwargs) for obj in self._objectives])
        return jnp.concatenate(f)

    def compute_scalar(self, x):
        """Compute the scalar form of the objective.

        Parameters
        ----------
        x : ndarray
            Full state vector x or optimization variable y.

        Returns
        -------
        f : float, ndarray
            Objective function scalar value.

        """
        if x.size == self._dim_y:
            x = self.recover(x)  # x is really y
        if x.size != self._dim_x:
            raise ValueError("State vector is not the proper size.")
        kwargs = self.unpack_state(x)

        f = jnp.array([obj.compute_scalar(**kwargs) for obj in self._objectives])
        return jnp.sum(f)

    def callback(self, x):
        """Print the value(s) of the objective.

        Parameters
        ----------
        x : ndarray
            Full state vector x or optimization variable y.

        """
        if x.size == self._dim_y:
            x = self.recover(x)  # x is really y
        if x.size != self._dim_x:
            raise ValueError("State vector is not the proper size.")
        kwargs = self.unpack_state(x)

        for obj in self._objectives:
            obj.callback(**kwargs)
        return None

    def unpack_state(self, x):
        """Unpack the full state vector x into its components.

        Parameters
        ----------
        x : ndarray
            Full state vector of optimization variables.
            x = [R_lmn, Z_lmn, L_lmn, Rb_lmn, Zb_lmn, p_l, i_l, Psi]

        Returns
        -------
        kwargs : dict
            Dictionary of the state components with the following keys:
                "R_lmn", "Z_lmn", "L_lmn", "Rb_lmn", "Zb_lmn", "p_l", "i_l", "Psi"

        """
        if not self._built:
            raise RuntimeError("ObjectiveFunction must be built first.")
        kwargs = {}
        kwargs["R_lmn"] = x[: self._dim_R]
        kwargs["Z_lmn"] = x[self._dim_R : self._dim_R + self._dim_Z]
        kwargs["L_lmn"] = x[
            self._dim_R + self._dim_Z : self._dim_R + self._dim_Z + self._dim_L
        ]
        kwargs["Rb_lmn"] = x[
            self._dim_R
            + self._dim_Z
            + self._dim_L : self._dim_R
            + self._dim_Z
            + self._dim_L
            + self._dim_Rb
        ]
        kwargs["Zb_lmn"] = x[
            self._dim_R
            + self._dim_Z
            + self._dim_L
            + self._dim_Rb : self._dim_R
            + self._dim_Z
            + self._dim_L
            + self._dim_Rb
            + self._dim_Zb
        ]
        kwargs["p_l"] = x[
            self._dim_R
            + self._dim_Z
            + self._dim_L
            + self._dim_Rb
            + self._dim_Zb : self._dim_R
            + self._dim_Z
            + self._dim_L
            + self._dim_Rb
            + self._dim_Zb
            + self._dim_p
        ]
        kwargs["i_l"] = x[
            self._dim_R
            + self._dim_Z
            + self._dim_L
            + self._dim_Rb
            + self._dim_Zb
            + self._dim_p : -self._dim_Psi
        ]
        kwargs["Psi"] = x[-self._dim_Psi :]
        return kwargs

    def project(self, x):
        """Project a full state vector x into the optimization variable y."""
        if not self._built:
            raise RuntimeError("ObjectiveFunction must be built first.")
        dx = x - self._x0
        y = jnp.dot(self._Z.T, dx)
        return jnp.squeeze(y)

    def recover(self, y):
        """Recover the full state vector x from the optimization variable y."""
        if not self._built:
            raise RuntimeError("ObjectiveFunction must be built first.")
        x = self._x0 + jnp.dot(self._Z, y)
        return jnp.squeeze(x)

    def make_feasible(self, x):
        """Return a full state vector x that satisfies the linear constraints."""
        y = self.project(x)
        return self._x0 + np.dot(self._Z, y)

    def grad(self, x):
        """Compute gradient vector of scalar form of the objective."""
        return self._grad.compute(x)

    def hess(self, x):
        """Compute Hessian matrix of scalar form of the objective wrt to x."""
        return self._hess.compute(x)

    def jac(self, x):
        """Compute Jacobian matrx of vector form of the objective wrt to x."""
        return self._jac.compute(x)

    def jvp(self, x, v):
        """Compute Jacobian-vector product of the objective function."""
        return Derivative.compute_jvp(self.compute, 0, v, x)

    def jvp2(self, argnum1, argnum2, v1, v2, x):
        """Compute 2nd derivative Jacobian-vector product of the objective function."""
        return Derivative.compute_jvp2(self.compute, 0, 0, v1, v2, x)

    def jvp3(self, argnum1, argnum2, argnum3, v1, v2, v3, x):
        """Compute 3rd derivative jacobian-vector product of the objective function."""
        return Derivative.compute_jvp3(self.compute, 0, 0, 0, v1, v2, v3, x)

    def compile(self, x, verbose=1, mode="auto"):
        """Call the necessary functions to ensure the function is compiled.

        Parameters
        ----------
        x : ndarray
            Full state vector x or optimization variable y.
        verbose : int, optional
            Level of output.
        mode : {"auto", "lsq", "scalar", "all"}
            Whether to compile for least squares optimization or scalar optimization.
            "auto" compiles based on the type of objective,
            "all" compiles all derivatives.

        """
        if not self._built:
            raise RuntimeError("ObjectiveFunction must be built first.")
        if not use_jax:
            self._compiled = True
            return

        timer = Timer()
        if mode == "auto" and self._scalar:
            mode = "scalar"
        elif mode == "auto":
            mode = "lsq"

        if verbose > 0:
            print("Compiling objective function and derivatives")
        timer.start("Total compilation time")

        if mode in ["scalar", "all"]:
            timer.start("Objective compilation time")
            f0 = self.compute_scalar(x).block_until_ready()
            timer.stop("Objective compilation time")
            if verbose > 1:
                timer.disp("Objective compilation time")
            timer.start("Gradient compilation time")
            g0 = self.grad_x(x).block_until_ready()
            timer.stop("Gradient compilation time")
            if verbose > 1:
                timer.disp("Gradient compilation time")
            timer.start("Hessian compilation time")
            H0 = self.hess_x(x).block_until_ready()
            timer.stop("Hessian compilation time")
            if verbose > 1:
                timer.disp("Hessian compilation time")
        if mode in ["lsq", "all"]:
            timer.start("Objective compilation time")
            f0 = self.compute(x).block_until_ready()
            timer.stop("Objective compilation time")
            if verbose > 1:
                timer.disp("Objective compilation time")
            timer.start("Jacobian compilation time")
            J0 = self.jac_x(x).block_until_ready()
            timer.stop("Jacobian compilation time")
            if verbose > 1:
                timer.disp("Jacobian compilation time")

        timer.stop("Total compilation time")
        if verbose > 1:
            timer.disp("Total compilation time")
        self._compiled = True

    @property
    def use_jit(self):
        """bool: Whether to just-in-time compile the objective and derivatives."""
        return self._use_jit

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        if not self._built:
            raise RuntimeError("ObjectiveFunction must be built first.")
        return self._scalar

    @property
    def built(self):
        """bool: Whether the objectives have been built (or not)."""
        return self._built

    @property
    def compiled(self):
        """bool: Whether the functions have been compiled (or not)."""
        return self._compiled

    @property
    def dim_x(self):
        """int: Dimensional of the full state vector."""
        if not self._built:
            raise RuntimeError("ObjectiveFunction must be built first.")
        return self._dim_x

    @property
    def dim_y(self):
        """int: Number of independent optimization variables."""
        if not self._built:
            raise RuntimeError("ObjectiveFunction must be built first.")
        return self._dim_y

    @property
    def dim_c(self):
        """int: Number of constraint equations."""
        if not self._built:
            raise RuntimeError("ObjectiveFunction must be built first.")
        return self._dim_c

    @property
    def dim_f(self):
        """int: Number of objective equations."""
        if not self._built:
            raise RuntimeError("ObjectiveFunction must be built first.")
        return self._dim_f

    @property
    def A(self):
        """ndarray: Linear constraint matrix: A*x = b."""
        if not self._built:
            raise RuntimeError("ObjectiveFunction must be built first.")
        return self._A

    @property
    def b(self):
        """ndarray: Linear constraint vector: A*x = b."""
        if not self._built:
            raise RuntimeError("ObjectiveFunction must be built first.")
        return self._b

    @property
    def x0(self):
        """ndarray: Feasible state vector."""
        if not self._built:
            raise RuntimeError("ObjectiveFunction must be built first.")
        return self._x0


class _Objective(IOAble, ABC):
    """Objective (or constraint) used in the optimization of an Equilibrium."""

    _io_attrs_ = [
        "grid",
        "target",
        "weight",
    ]

    def __init__(self, geometry=None, eq=None, target=0, weight=1):
        """Initialize an Objective.

        Parameters
        ----------
        geometry : TBD, optional
            Geometry defining where the objective is evaluated. Often a Grid or Surface.
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        target : float, ndarray
            Target value(s) of the objective.
            len(target) must be equal to Objective.dim_f
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            len(weight) must be equal to Objective.dim_f

        """
        self._target = np.atleast_1d(target)
        self._weight = np.atleast_1d(weight)
        self._built = False

        if self.scalar:
            self._dim_f = 1
        else:
            self._dim_f = None

        if eq is not None:
            self.build(eq, geometry)

    def _check_dimensions(self):
        """Check that self.target = self.weight = self.dim_f."""
        if self._target.size == 1:
            self._target = self._target * np.ones((self._dim_f,))
        if self._weight.size == 1:
            self._weight = self._weight * np.ones((self._dim_f,))

        if self._target.size != self._dim_f:
            raise ValueError("len(target) != dim_f")
        if self._weight.size != self._dim_f:
            raise ValueError("len(weight) != dim_f")

        return None

    @abstractmethod
    def build(self, eq, grid=None, verbose=1):
        """Precompute the transforms."""

    @abstractmethod
    def compute(self, **kwargs):
        """Compute the objective function."""

    @abstractmethod
    def compute_scalar(self, **kwargs):
        """Compute the scalar form of the objective."""

    @abstractmethod
    def callback(self, **kwargs):
        """Print the value(s) of the objective."""

    @property
    def target(self):
        """float: Target value(s) of the objective."""
        return self._target

    @target.setter
    def target(self, target):
        self._target = target
        self._check_dimensions()

    @property
    def weight(self):
        """float: Weighting to apply to the Objective, relative to other Objectives."""
        return self._weight

    @weight.setter
    def weight(self, weight):
        self._weight = weight
        self._check_dimensions()

    @property
    def built(self):
        """bool: Whether the transforms have been precomputed (or not)."""
        return self._built

    @property
    def dim_f(self):
        """int: Number of objective equations."""
        return self._dim_f

    @property
    @abstractmethod
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""

    @property
    @abstractmethod
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""

    @property
    @abstractmethod
    def name(self):
        """Name of objective function (str)."""


class FixedBoundary(_Objective):
    """Fixes boundary coefficients."""

    def __init__(
        self,
        surface=None,
        eq=None,
        target=(None, None),
        weight=(1, 1),
        modes=(True, True),
    ):
        """Initialize a FixedBoundary Objective.

        Parameters
        ----------
        surface : Surface, optional
            Toroidal surface containing the Fourier modes to evaluate at.
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        target : tuple, float, ndarray, optional
            Target value(s) of the objective. Tuple = (R_target, Z_target)
            len(target) = len(weight) = len(modes). If None, uses surface coefficients.
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            Tuple = (R_target, Z_target). len(target) = len(weight) = len(modes)
        modes : ndarray, optional
            Basis modes numbers [l,m,n] of boundary modes to fix.
            Tuple = (R_modes, Z_modes). len(target) = len(weight) = len(modes).
            If True/False uses all/none of the surface modes.

        """
        self._surface = surface
        self._R_target = target[0]
        self._Z_target = target[1]
        self._R_weight = np.atleast_1d(weight[0])
        self._Z_weight = np.atleast_1d(weight[1])
        self._R_modes = modes[0]
        self._Z_modes = modes[1]
        super().__init__(surface, eq=eq, target=target, weight=weight)

    def build(self, eq, surface=None, verbose=1):
        """Precompute the transforms.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        surface : Surface, optional
            Toroidal surface containing the Fourier modes to evaluate at.
        verbose : int, optional
            Level of output.

        """
        if surface is not None:
            self._surface = surface
        if self._surface is None:
            self._surface = eq.surface

        # find inidies of R boundary modes to fix
        if self._R_modes is False or self._R_modes is None:  # no modes
            self._idx_Rb = np.array([], dtype=int)
        elif self._R_modes is True:  # all modes in surface
            self._idx_Rb = np.arange(self._surface.R_basis.num_modes)
            idx_R = self._idx_Rb
        else:  # specified modes
            dtype = {
                "names": ["f{}".format(i) for i in range(3)],
                "formats": 3 * [self._R_modes.dtype],
            }
            _, self._idx_Rb, idx_R = np.intersect1d(
                self._surface.R_basis.modes.view(dtype), self._R_modes.view(dtype)
            )
            if self._idx_Rb.size < self._R_modes.shape[0]:
                warnings.warn(
                    colored(
                        "Some of the given R modes are not in the boundary surface, ",
                        +"these modes will not be fixed.",
                        "yellow",
                    )
                )

        # find inidies of Z boundary modes to fix
        if self._Z_modes is False or self._Z_modes is None:  # no modes
            self._idx_Zb = np.array([], dtype=int)
        elif self._Z_modes is True:  # all modes in surface
            self._idx_Zb = np.arange(self._surface.Z_basis.num_modes)
            idx_Z = self._idx_Zb
        else:  # specified modes
            dtype = {
                "names": ["f{}".format(i) for i in range(3)],
                "formats": 3 * [self._Z_modes.dtype],
            }
            _, self._idx_Zb, idx_Z = np.intersect1d(
                self._surface.Z_basis.modes.view(dtype), self._Z_modes.view(dtype)
            )
            if self._idx_Zb.size < self._Z_modes.shape[0]:
                warnings.warn(
                    colored(
                        "Some of the given Z modes are not in the boundary surface, ",
                        +"these modes will not be fixed.",
                        "yellow",
                    )
                )

        dim_Rb = self._idx_Rb.size
        dim_Zb = self._idx_Zb.size
        self._dim_f = dim_Rb + dim_Zb

        # set target values for R boundary coefficients
        if self._R_target is None:  # use surface coefficients
            self._R_target = self._surface.R_lmn[self._idx_Rb]
        elif self._R_target.size == 1:  # use scalar target for all modes
            self._R_target = self._R_target * np.ones((dim_Rb,))
        elif self._R_target.size == self._R_modes.shape[0]:  # use given array target
            self._R_target = self._R_target[idx_R]
        else:
            raise ValueError("R target must be the same size as R modes.")

        # set target values for Z boundary coefficients
        if self._Z_target is None:  # use surface coefficients
            self._Z_target = self._surface.Z_lmn[self._idx_Zb]
        elif self._Z_target.size == 1:  # use scalar target for all modes
            self._Z_target = self._Z_target * np.ones((dim_Zb,))
        elif self._Z_target.size == self._Z_modes.shape[0]:  # use given array target
            self._Z_target = self._Z_target[idx_Z]
        else:
            raise ValueError("Z target must be the same size as Z modes.")

        # check R boundary weights
        if self._R_weight.size == 1:  # use scalar weight for all modes
            self._R_weight = self._R_weight * np.ones((dim_Rb,))
        elif self._R_weight.size == self._R_modes.shape[0]:  # use given array weight
            self._R_weight = self._R_weight[idx_R]
        else:
            raise ValueError("R weight must be the same size as R modes.")

        # check Z boundary weights
        if self._Z_weight.size == 1:  # use scalar weight for all modes
            self._Z_weight = self._Z_weight * np.ones((dim_Zb,))
        elif self._Z_weight.size == self._Z_modes.shape[0]:  # use given array weight
            self._Z_weight = self._Z_weight[idx_Z]
        else:
            raise ValueError("Z weight must be the same size as Z modes.")

        self._target = np.concatenate((self._R_target, self._Z_target))
        self._weight = np.concatenate((self._R_weight, self._Z_weight))
        self._built = True
        self._check_dimensions()

    def _compute(self, Rb_lmn, Zb_lmn):
        """Compute fixed-boundary errors.

        Parameters
        ----------
        Rb_lmn : ndarray
            Spectral coefficients of Rb(rho,theta,zeta) -- boundary R coordinate.
        Zb_lmn : ndarray
            Spectral coefficients of Zb(rho,theta,zeta) -- boundary Z coordiante.

        Returns
        -------
        Rb : ndarray
            Boundary surface errors in R coordinate, in meters.
        Zb : ndarray
            Boundary surface errors in Z coordinate, in meters.

        """
        Rb = Rb_lmn[self._idx_Rb] - self._R_target
        Zb = Zb_lmn[self._idx_Zb] - self._Z_target
        return Rb, Zb

    def compute(self, Rb_lmn, Zb_lmn, **kwargs):
        """Compute fixed-boundary errors.

        Parameters
        ----------
        Rb_lmn : ndarray
            Spectral coefficients of Rb(rho,theta,zeta) -- boundary R coordinate.
        Zb_lmn : ndarray
            Spectral coefficients of Zb(rho,theta,zeta) -- boundary Z coordiante.

        Returns
        -------
        f : ndarray
            Boundary surface errors, in meters.

        """
        Rb, Zb = self._compute(Rb_lmn, Zb_lmn)
        return jnp.concatenate((Rb * self._R_weight, Zb * self._Z_weight))

    def compute_scalar(self, Rb_lmn, Zb_lmn, **kwargs):
        """Compute total fixed-boundary error.

        Parameters
        ----------
        Rb_lmn : ndarray
            Spectral coefficients of Rb(rho,theta,zeta) -- boundary R coordinate.
        Zb_lmn : ndarray
            Spectral coefficients of Zb(rho,theta,zeta) -- boundary Z coordiante.

        Returns
        -------
        f : ndarray
            Boundary surface errors, in meters.

        """
        return jnp.linalg.norm(self.compute(Rb_lmn, Zb_lmn))

    def callback(self, Rb_lmn, Zb_lmn, **kwargs):
        """Print fixed-boundary errors.

        Parameters
        ----------
        Rb_lmn : ndarray
            Spectral coefficients of Rb(rho,theta,zeta) -- boundary R coordinate.
        Zb_lmn : ndarray
            Spectral coefficients of Zb(rho,theta,zeta) -- boundary Z coordiante.

        """
        Rb, Zb = self._compute(Rb_lmn, Zb_lmn)
        f = jnp.concatenate(np.array((Rb, Zb)))
        print(
            "Total fixed-boundary error: {:10.3e}, ".format(jnp.linalg.norm(f))
            + "R fixed-boundary error: {:10.3e}, ".format(jnp.linalg.norm(Rb))
            + "Z fixed-boundary error: {:10.3e} ".format(jnp.linalg.norm(Zb))
            + "(m)"
        )
        return None

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return False

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return True

    @property
    def name(self):
        """Name of objective function (str)."""
        return "fixed-boundary"


class LCFSBoundary(_Objective):
    """Boundary condition on the last closed flux surface."""

    def __init__(self, surface=None, eq=None, target=0, weight=1):
        """Initialize a LCFSBoundary Objective.

        Parameters
        ----------
        surface : FourierRZToroidalSurface, optional
            Toroidal surface containing the Fourier modes to evaluate at.
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        target : float, ndarray, optional
            This target always gets overridden to be 0.
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            len(weight) must be equal to Objective.dim_f

        """
        target = 0
        self._surface = surface
        super().__init__(surface, eq=eq, target=target, weight=weight)

    def build(self, eq, surface=None, verbose=1):
        """Precompute the transforms.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        surface : FourierRZToroidalSurface, optional
            Toroidal surface containing the Fourier modes to evaluate at.
        verbose : int, optional
            Level of output.

        """
        if surface is not None:
            self._surface = surface
        if self._surface is None:
            self._surface = eq.surface

        R_modes = eq.R_basis.modes
        Z_modes = eq.Z_basis.modes
        Rb_modes = self._surface.R_basis.modes
        Zb_modes = self._surface.Z_basis.modes

        dim_R = eq.R_basis.num_modes
        dim_Z = eq.Z_basis.num_modes
        dim_Rb = self._surface.R_basis.num_modes
        dim_Zb = self._surface.Z_basis.num_modes
        self._dim_f = dim_Rb + dim_Zb

        self._A_R = np.zeros((dim_Rb, dim_R))
        self._A_Z = np.zeros((dim_Zb, dim_Z))

        for i, (l, m, n) in enumerate(R_modes):
            j = np.argwhere(np.logical_and(Rb_modes[:, 1] == m, Rb_modes[:, 2] == n))
            self._A_R[j, i] = 1

        for i, (l, m, n) in enumerate(Z_modes):
            j = np.argwhere(np.logical_and(Zb_modes[:, 1] == m, Zb_modes[:, 2] == n))
            self._A_Z[j, i] = 1

        self._built = True
        self._check_dimensions()

    def _compute(self, R_lmn, Z_lmn, Rb_lmn, Zb_lmn):
        """Compute last closed flux surface boundary errors.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.
        Rb_lmn : ndarray
            Spectral coefficients of Rb(rho,theta,zeta) -- boundary R coordinate.
        Zb_lmn : ndarray
            Spectral coefficients of Zb(rho,theta,zeta) -- boundary Z coordiante.

        Returns
        -------
        Rb : ndarray
            Boundary surface errors in R coordinate, in meters.
        Zb : ndarray
            Boundary surface errors in Z coordinate, in meters.

        """
        Rb = jnp.dot(self._A_R, R_lmn) - Rb_lmn
        Zb = jnp.dot(self._A_Z, Z_lmn) - Zb_lmn
        return Rb, Zb

    def compute(self, R_lmn, Z_lmn, Rb_lmn, Zb_lmn, **kwargs):
        """Compute last closed flux surface boundary errors.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.
        Rb_lmn : ndarray
            Spectral coefficients of Rb(rho,theta,zeta) -- boundary R coordinate.
        Zb_lmn : ndarray
            Spectral coefficients of Zb(rho,theta,zeta) -- boundary Z coordiante.

        Returns
        -------
        f : ndarray
            Boundary surface errors, in meters.

        """
        Rb, Zb = self._compute(R_lmn, Z_lmn, Rb_lmn, Zb_lmn)
        return jnp.concatenate((Rb, Zb)) * self._weight

    def compute_scalar(self, R_lmn, Z_lmn, Rb_lmn, Zb_lmn, **kwargs):
        """Compute total last closed flux surface boundary error.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.
        Rb_lmn : ndarray
            Spectral coefficients of Rb(rho,theta,zeta) -- boundary R coordinate.
        Zb_lmn : ndarray
            Spectral coefficients of Zb(rho,theta,zeta) -- boundary Z coordiante.

        Returns
        -------
        f : ndarray
            Boundary surface errors, in meters.

        """
        return jnp.linalg.norm(self.compute(R_lmn, Z_lmn, Rb_lmn, Zb_lmn))

    def callback(self, R_lmn, Z_lmn, Rb_lmn, Zb_lmn, **kwargs):
        """Print last closed flux surface boundary errors.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.
        Rb_lmn : ndarray
            Spectral coefficients of Rb(rho,theta,zeta) -- boundary R coordinate.
        Zb_lmn : ndarray
            Spectral coefficients of Zb(rho,theta,zeta) -- boundary Z coordiante.

        """
        Rb, Zb = self._compute(R_lmn, Z_lmn, Rb_lmn, Zb_lmn)
        f = jnp.concatenate(np.array((Rb, Zb)))
        print(
            "Total boundary error: {:10.3e}, ".format(jnp.linalg.norm(f))
            + "R boundary error: {:10.3e}, ".format(jnp.linalg.norm(Rb))
            + "Z boundary error: {:10.3e} ".format(jnp.linalg.norm(Zb))
            + "(m)"
        )
        return None

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return False

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return True

    @property
    def name(self):
        """Name of objective function (str)."""
        return "lcfs"


class Volume(_Objective):
    """Plasma volume."""

    def __init__(self, grid=None, eq=None, target=0, weight=1):
        """Initialize a Volume Objective.

        Parameters
        ----------
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        target : float, ndarray, optional
            Target value(s) of the objective.
            len(target) must be equal to Objective.dim_f
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            len(weight) must be equal to Objective.dim_f

        """
        self._grid = grid
        super().__init__(grid, eq=eq, target=target, weight=weight)

        if self._grid is not None and self._grid.node_pattern != "quad":
            warnings.warn(
                colored(
                    "Volume objective requires 'quad' node pattern, "
                    + "integration will be incorrect.",
                    "yellow",
                )
            )

    def build(self, eq, grid=None, verbose=1):
        """Precompute the transforms.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.
        verbose : int, optional
            Level of output.

        """
        if grid is not None:
            self._grid = grid
        if self._grid is None:
            self._grid = QuadratureGrid(L=eq.L, M=eq.M, N=eq.N, NFP=eq.NFP)

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._R_transform = Transform(self._grid, eq.R_basis, derivs=1, build=True)
        self._Z_transform = Transform(self._grid, eq.Z_basis, derivs=1, build=True)

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._built = True
        self._check_dimensions()

    def _compute(self, R_lmn, Z_lmn):
        """Compute plasma volume.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.

        Returns
        -------
        V : float
            Plasma volume, in cubic meters.

        """
        data = compute_jacobian(R_lmn, Z_lmn, self._R_transform, self._Z_transform)
        V = jnp.sum(jnp.abs(data["sqrt(g)"]) * self._grid.weights)
        return V

    def compute(self, R_lmn, Z_lmn, **kwargs):
        """Compute plasma volume.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.

        Returns
        -------
        V : float
            Plasma volume, in cubic meters.

        """
        V = self._compute(R_lmn, Z_lmn)
        return jnp.atleast_1d((V - self._target) * self._weight)

    def compute_scalar(self, R_lmn, Z_lmn, **kwargs):
        """Compute plasma volume.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.

        Returns
        -------
        V : float
            Plasma volume, in cubic meters.

        """
        return self.compute(R_lmn, Z_lmn)

    def callback(self, R_lmn, Z_lmn, **kwargs):
        """Print plamsa volume.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.

        """
        V = self._compute(R_lmn, Z_lmn)
        print("Plasma volume: {:10.3e} (m^3)".format(V))
        return None

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return True

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return False

    @property
    def name(self):
        """Name of objective function (str)."""
        return "volume"


class Energy(_Objective):
    """MHD energy: W = integral( B^2 / (2*mu0) + p / (gamma - 1) ) dV."""

    _io_attrs_ = _Objective._io_attrs_ + ["gamma"]

    def __init__(self, grid=None, eq=None, target=0, weight=1, gamma=0):
        """Initialize an Energy Objective.

        Parameters
        ----------
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        target : float, ndarray, optional
            Target value(s) of the objective.
            len(target) must be equal to Objective.dim_f
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            len(weight) must be equal to Objective.dim_f
        gamma : float, optional
            Adiabatic (compressional) index. Default = 0.

        """
        self._grid = grid
        self._gamma = gamma
        super().__init__(grid, eq=eq, target=target, weight=weight)

        if self._grid is not None and self._grid.node_pattern != "quad":
            warnings.warn(
                colored(
                    "Energy objective requires 'quad' node pattern, "
                    + "integration will be incorrect.",
                    "yellow",
                )
            )

    def build(self, eq, grid=None, verbose=1):
        """Precompute the transforms.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.
        verbose : int, optional
            Level of output.

        """
        if grid is not None:
            self._grid = grid
        if self._grid is None:
            self._grid = QuadratureGrid(L=eq.L, M=eq.M, N=eq.N, NFP=eq.NFP)

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._iota = eq.iota.copy()
        self._pressure = eq.pressure.copy()
        self._iota.grid = self._grid
        self._pressure.grid = self._grid

        self._R_transform = Transform(self._grid, eq.R_basis, derivs=1, build=True)
        self._Z_transform = Transform(self._grid, eq.Z_basis, derivs=1, build=True)
        self._L_transform = Transform(self._grid, eq.L_basis, derivs=1, build=True)

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._built = True
        self._check_dimensions()

    def _compute(self, R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi):
        """Compute MHD energy components.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.
        L_lmn : ndarray
            Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.
        p_l : ndarray
            Spectral coefficients of p(rho) -- pressure profile.
        Psi : float
            Total toroidal flux within the last closed flux surface, in Webers.

        Returns
        -------
        W : float
            Total MHD energy in the plasma volume, in Joules.
        W_B : float
            Magnetic energy, in Joules.
        W_p : float
            Pressure energy, in Joules.

        """
        data = compute_pressure(p_l, self._pressure)
        data = compute_magnetic_field_magnitude(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
            data=data,
        )
        W_B = jnp.sum(
            data["|B|"] ** 2 * jnp.abs(data["sqrt(g)"]) * self._grid.weights
        ) / (2 * mu_0)
        W_p = jnp.sum(data["p"] * jnp.abs(data["sqrt(g)"]) * self._grid.weights) / (
            self._gamma - 1
        )
        W = W_B + W_p
        return W, W_B, W_p

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi, **kwargs):
        """Compute MHD energy.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.
        L_lmn : ndarray
            Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.
        p_l : ndarray
            Spectral coefficients of p(rho) -- pressure profile.
        Psi : float
            Total toroidal flux within the last closed flux surface, in Webers.

        Returns
        -------
        W : float
            Total MHD energy in the plasma volume, in Joules.

        """
        W, W_B, B_p = self._compute(R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi)
        return jnp.atleast_1d((W - self._target) * self._weight)

    def compute_scalar(self, R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi, **kwargs):
        """Compute MHD energy.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.
        L_lmn : ndarray
            Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.
        p_l : ndarray
            Spectral coefficients of p(rho) -- pressure profile.
        Psi : float
            Total toroidal flux within the last closed flux surface, in Webers.

        Returns
        -------
        W : float
            Total MHD energy in the plasma volume, in Joules.

        """
        return self.compute(R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi)

    def callback(self, R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi, **kwargs):
        """Print MHD energy.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate.
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordiante.
        L_lmn : ndarray
            Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.
        p_l : ndarray
            Spectral coefficients of p(rho) -- pressure profile.
        Psi : float
            Total toroidal flux within the last closed flux surface, in Webers.

        """
        W, W_B, W_p = self._compute(R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi)
        print(
            "Total MHD energy: {:10.3e}, ".format(W)
            + "Magnetic Energy: {:10.3e}, Pressure Energy: {:10.3e} ".format(W_B, W_p)
            + "(J)"
        )
        return None

    @property
    def gamma(self):
        """float: Adiabatic (compressional) index."""
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return True

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return False

    @property
    def name(self):
        """Name of objective function (str)."""
        return "energy"


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
