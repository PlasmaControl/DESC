import numpy as np
from abc import ABC, abstractmethod
from termcolor import colored
import warnings
from scipy.constants import mu_0
from inspect import getfullargspec

from desc import config
from desc.backend import jnp, jit, use_jax
from desc.utils import Timer
from desc.io import IOAble
from desc.derivatives import Derivative
from desc.grid import QuadratureGrid, ConcentricGrid, LinearGrid
from desc.transform import Transform
from desc.profiles import PowerSeriesProfile
from desc.compute_funs import (
    compute_rotational_transform,
    compute_covariant_metric_coefficients,
    compute_magnetic_field_magnitude,
    compute_contravariant_current_density,
    compute_force_error,
    compute_quasisymmetry_error,
    compute_volume,
    compute_energy,
)

__all__ = [
    "FixedBoundary",
    "FixedPressure",
    "FixedIota",
    "FixedPsi",
    "LCFSBoundary",
    "TargetIota",
    "Volume",
    "Energy",
    "RadialForceBalance",
    "HelicalForceBalance",
    "RadialCurrent",
    "PoloidalCurrent",
    "ToroidalCurrent",
    "QuasisymmetryFluxFunction",
    "QuasisymmetryTripleProduct",
]

# XXX: could use `indicies` instead of `arg_order` in ObjectiveFunction loops
_arg_order_ = ("R_lmn", "Z_lmn", "L_lmn", "Rb_lmn", "Zb_lmn", "p_l", "i_l", "Psi")


class ObjectiveFunction(IOAble):
    """Objective function comprised of one or more Objectives."""

    _io_attrs_ = ["objectives", "constraints"]

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
            self.build(eq, use_jit=self._use_jit)

    def _set_state_vector(self):
        """Set state vector components, dimensions, and indicies."""
        self._args = np.unique(
            np.concatenate(
                [
                    np.concatenate([obj.args for obj in self._constraints]),
                    np.concatenate([obj.args for obj in self._objectives]),
                ]
            )
        )

        self._dimensions = self._objectives[0]._dimensions

        idx = 0
        self._indicies = {}
        for arg in _arg_order_:
            if arg in self._args:
                self._indicies[arg] = np.arange(idx, idx + self._dimensions[arg])
                idx += self._dimensions[arg]

        self._dim_y = idx

    def _build_linear_constraints(self):
        """Compute and factorize A to get pseudoinverse and nullspace."""
        # A = dc/dx
        self._A = np.array([[]])
        for obj in self._constraints:
            A = np.array([[]])
            for arg in _arg_order_:
                if arg in self._args:
                    a = np.atleast_2d(obj.derivatives[arg])
                    A = np.hstack((A, a)) if A.size else a
            self._A = np.vstack((self._A, A)) if self._A.size else A

        # c = A*y - b
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
        self._Zinv = np.linalg.pinv(self._Z, rcond=rcond)
        self._dim_x = self._Z.shape[1]

        uk = u[:, :K]
        vhk = vh[:K, :]
        s = np.divide(1, s, where=large, out=s)
        s[(~large,)] = 0
        self._Ainv = np.matmul(vhk.T, np.multiply(s[..., np.newaxis], uk.T))
        self._y0 = np.dot(self._Ainv, self._b)

    def _set_derivatives(self, use_jit=True, block_size="auto"):
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
            shape=(self._dim_x, self._dim_x),
        )
        self._jac = Derivative(
            self.compute,
            mode="fwd",
            use_jit=use_jit,
            block_size=block_size,
            shape=(self._dim_f, self._dim_x),
        )

        if use_jit:
            self.compute = jit(self.compute)
            self.compute_scalar = jit(self.compute_scalar)

    def build(self, eq, use_jit=True, verbose=1):
        """Build the constraints and objectives.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        self._use_jit = use_jit
        timer = Timer()
        timer.start("Objecive build")

        # build constraints
        self._dim_c = 0
        for constraint in self._constraints:
            if not constraint.linear:
                raise NotImplementedError("Constraints must be linear.")
            if not constraint.built:
                if verbose > 0:
                    print("Building constraint: " + constraint.name)
                constraint.build(eq, use_jit=self._use_jit, verbose=verbose)
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
                objective.build(eq, use_jit=self._use_jit, verbose=verbose)
            self._dim_f += objective.dim_f

        self._set_state_vector()

        # build linear constraint matrices
        if verbose > 0:
            print("Building linear constraints")
        timer.start("linear constraint build")
        self._build_linear_constraints()
        timer.stop("linear constraint build")
        if verbose > 1:
            timer.disp("linear constraint build")

        self._set_derivatives(self._use_jit)

        self._built = True
        timer.stop("Objecive build")
        if verbose > 1:
            timer.disp("Objecive build")

    def compute(self, x):
        """Compute the objective function.

        Parameters
        ----------
        x : ndarray
            Optimization variables.

        Returns
        -------
        f : ndarray
            Objective function value(s).

        """
        if x.size != self._dim_x:
            raise ValueError("Optimization vector is not the proper size.")
        y = self.recover(x)
        kwargs = self.unpack_state(y)

        return jnp.concatenate([obj.compute(**kwargs) for obj in self._objectives])

    def compute_scalar(self, x):
        """Compute the scalar form of the objective.

        Parameters
        ----------
        x : ndarray
            Optimization variables.

        Returns
        -------
        f : float
            Objective function scalar value.

        """
        return jnp.sum(self.compute(x) ** 2)

    def callback(self, x):
        """Print the value(s) of the objective.

        Parameters
        ----------
        x : ndarray
            Optimization variables.

        """
        if x.size != self._dim_x:
            raise ValueError("Optimization vector is not the proper size.")
        y = self.recover(x)
        kwargs = self.unpack_state(y)

        for obj in self._objectives:
            obj.callback(**kwargs)
        return None

    def unpack_state(self, y):
        """Unpack the full state vector y into its components.

        Parameters
        ----------
        y : ndarray
            Full state vector.

        Returns
        -------
        kwargs : dict
            Dictionary of the state components with the following keys:
                "R_lmn", "Z_lmn", "L_lmn", "Rb_lmn", "Zb_lmn", "p_l", "i_l", "Psi"

        """
        if not self._built:
            raise RuntimeError("ObjectiveFunction must be built first.")
        if y.size != self._dim_y:
            raise ValueError("State vector is not the proper size.")

        kwargs = {}
        for arg in self._args:
            kwargs[arg] = y[self._indicies[arg]]
        return kwargs

    def project(self, y):
        """Project a full state vector y into the optimization vector x."""
        if not self._built:
            raise RuntimeError("ObjectiveFunction must be built first.")
        dy = y - self._y0
        x = jnp.dot(self._Z.T, dy)
        return jnp.squeeze(x)

    def recover(self, x):
        """Recover the full state vector y from the optimization vector x."""
        if not self._built:
            raise RuntimeError("ObjectiveFunction must be built first.")
        y = self._y0 + jnp.dot(self._Z, x)
        return jnp.squeeze(y)

    def make_feasible(self, y):
        """Return a full state vector y that satisfies the linear constraints."""
        x = self.project(y)
        return self._y0 + np.dot(self._Z, x)

    def y(self, eq):
        """Return the full state vector y from the Equilibrium eq."""
        kwargs = eq.get_args()
        y = np.zeros((self._dim_y,))

        for arg in self._args:
            y[self._indicies[arg]] = kwargs[arg]
        return y

    def x(self, eq):
        """Return the optimization variable x from the Equilibrium eq."""
        return self.project(self.y(eq))

    def get_args(self, x):
        """Get arguments from the optimization vector x (or y)."""
        if x.size == self._dim_x:
            y = self.recover(x)
        elif x.size == self._dim_y:
            y = x

        kwargs = {}
        for arg in self._args:
            kwargs[arg] = y[self._indicies[arg]]
        return kwargs

    def grad(self, x):
        """Compute gradient vector of scalar form of the objective wrt x."""
        # TODO: add block method
        return self._grad.compute(x)

    def hess(self, x):
        """Compute Hessian matrix of scalar form of the objective wrt x."""
        # TODO: add block method
        return self._hess.compute(x)

    def jac(self, x):
        """Compute Jacobian matrx of vector form of the objective wrt x."""
        if config.get("device") == "gpu":
            y = self.recover(x)
            kwargs = self.unpack_state(y)

            jac = np.array([[]])
            for obj in self._objectives:
                A = np.array([[]])  # A = df/dy
                for arg in _arg_order_:
                    if arg in self._args:
                        a = obj.derivatives[arg]
                        if isinstance(a, Derivative):
                            args = [kwargs[arg] for arg in obj.args]
                            a = a.compute(*args)
                        a = np.atleast_2d(a)
                        A = np.hstack((A, a)) if A.size else a
                jac = np.vstack((jac, A)) if jac.size else A

            return np.dot(jac, self._Z)  # Z = dy/dx

        else:
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

    # TODO: add function to compute derivatives wrt args
    # give jacobian as optional argument

    def compile(self, mode="auto", verbose=1):
        """Call the necessary functions to ensure the function is compiled.

        Parameters
        ----------
        mode : {"auto", "lsq", "scalar", "all"}
            Whether to compile for least squares optimization or scalar optimization.
            "auto" compiles based on the type of objective,
            "all" compiles all derivatives.
        verbose : int, optional
            Level of output.

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

        # variable values are irrelevant for compilation
        x = np.zeros((self._dim_x,))

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
            g0 = self.grad(x).block_until_ready()
            timer.stop("Gradient compilation time")
            if verbose > 1:
                timer.disp("Gradient compilation time")
            timer.start("Hessian compilation time")
            H0 = self.hess(x).block_until_ready()
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
            J0 = self.jac(x)
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
    def args(self):
        """list: Names (str) of arguments to the compute functions."""
        return self._args

    @property
    def dim_y(self):
        """int: Dimensional of the full state vector y."""
        if not self._built:
            raise RuntimeError("ObjectiveFunction must be built first.")
        return self._dim_y

    @property
    def dim_x(self):
        """int: Dimension of the optimization vector x."""
        if not self._built:
            raise RuntimeError("ObjectiveFunction must be built first.")
        return self._dim_x

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
    def Ainv(self):
        """ndarray: Linear constraint matrix inverse: y0 = Ainv*b."""
        if not self._built:
            raise RuntimeError("ObjectiveFunction must be built first.")
        return self._Ainv

    @property
    def b(self):
        """ndarray: Linear constraint vector: A*x = b."""
        if not self._built:
            raise RuntimeError("ObjectiveFunction must be built first.")
        return self._b

    @property
    def y0(self):
        """ndarray: Feasible state vector."""
        if not self._built:
            raise RuntimeError("ObjectiveFunction must be built first.")
        return self._y0

    @property
    def Z(self):
        """ndarray: Linear constraint nullspace: y = y0 + Z*x, dy/dx = Z."""
        if not self._built:
            raise RuntimeError("ObjectiveFunction must be built first.")
        return self._Z

    @property
    def Zinv(self):
        """ndarray: Linear constraint nullspace inverse: dx/dy = Zinv."""
        if not self._built:
            raise RuntimeError("ObjectiveFunction must be built first.")
        return self._Zinv


class _Objective(IOAble, ABC):
    """Objective (or constraint) used in the optimization of an Equilibrium."""

    _io_attrs_ = ["grid", "target", "weight"]

    def __init__(self, eq=None, target=0, weight=1):
        """Initialize an Objective.

        Parameters
        ----------
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

        if eq is not None:
            self.build(eq)

    def _set_dimensions(self, eq):
        """Set state vector component dimensions."""
        self._dimensions = {}
        self._dimensions["R_lmn"] = eq.R_basis.num_modes
        self._dimensions["Z_lmn"] = eq.Z_basis.num_modes
        self._dimensions["L_lmn"] = eq.L_basis.num_modes
        self._dimensions["Rb_lmn"] = eq.surface.R_basis.num_modes
        self._dimensions["Zb_lmn"] = eq.surface.Z_basis.num_modes
        self._dimensions["p_l"] = eq.pressure.params.size
        self._dimensions["i_l"] = eq.iota.params.size
        self._dimensions["Psi"] = 1

    def _set_derivatives(self, use_jit=True, block_size="auto"):
        """Set up derivatives of the objective wrt each argument."""
        self._derivatives = {}
        self._scalar_derivatives = {}
        self._args = getfullargspec(self.compute)[0][1:]

        # only used for linear objectives so variable values are irrelevant
        kwargs = {
            "R_lmn": np.zeros((self._dimensions["R_lmn"],)),
            "Z_lmn": np.zeros((self._dimensions["Z_lmn"],)),
            "L_lmn": np.zeros((self._dimensions["L_lmn"],)),
            "Rb_lmn": np.zeros((self._dimensions["Rb_lmn"],)),
            "Zb_lmn": np.zeros((self._dimensions["Zb_lmn"],)),
            "p_l": np.zeros((self._dimensions["p_l"],)),
            "i_l": np.zeros((self._dimensions["i_l"],)),
            "Psi": np.zeros((self._dimensions["Psi"],)),
        }
        args = [kwargs[arg] for arg in self._args]

        # constant derivatives are pre-computed, otherwise set up Derivative instance
        for arg in _arg_order_:
            if arg in self._args:  # derivative wrt arg
                self._derivatives[arg] = Derivative(
                    self.compute,
                    argnum=self._args.index(arg),
                    mode="fwd",
                    use_jit=use_jit,
                    block_size=block_size,
                    shape=(self._dim_f, self._dimensions[arg]),
                )
                self._scalar_derivatives[arg] = Derivative(
                    self.compute_scalar,
                    argnum=self._args.index(arg),
                    mode="fwd",
                    use_jit=use_jit,
                    block_size=block_size,
                    shape=(self._dim_f, self._dimensions[arg]),
                )
                if self.linear:  # linear objectives have constant derivatives
                    self._derivatives[arg] = self._derivatives[arg].compute(*args)
                    self._scalar_derivatives[arg] = self._scalar_derivatives[
                        arg
                    ].compute(*args)
            else:  # these derivatives are always zero
                self._derivatives[arg] = np.zeros((self._dim_f, self._dimensions[arg]))
                self._scalar_derivatives[arg] = np.zeros((1, self._dimensions[arg]))

        if use_jit:
            self.compute = jit(self.compute)
            self.compute_scalar = jit(self.compute_scalar)

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
    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays."""
        # TODO: most transforms are pre-computing more derivatives than required

    @abstractmethod
    def compute(self, *args, **kwargs):
        """Compute the objective function."""

    def compute_scalar(self, *args, **kwargs):
        """Compute the scalar form of the objective."""
        return jnp.sum(self.compute(*args, **kwargs) ** 2)

    @abstractmethod
    def callback(self, *args):
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
    def args(self):
        """list: Names (str) of arguments to the compute functions."""
        return self._args

    @property
    def derivatives(self):
        """dict: Derivatives of the function wrt the argument given by the dict keys."""
        return self._derivatives

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
        eq=None,
        target=(None, None),
        weight=(1, 1),
        surface=None,
        modes=(True, True),
    ):
        """Initialize a FixedBoundary Objective.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        target : tuple, float, ndarray, optional
            Target value(s) of the objective. Tuple = (R_target, Z_target)
            len(target) = len(weight) = len(modes). If None, uses surface coefficients.
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            Tuple = (R_target, Z_target). len(target) = len(weight) = len(modes)
        surface : Surface, optional
            Toroidal surface containing the Fourier modes to evaluate at.
        modes : ndarray, optional
            Basis modes numbers [l,m,n] of boundary modes to fix.
            Tuple = (R_modes, Z_modes). len(target) = len(weight) = len(modes).
            If True/False uses all/none of the surface modes.

        """
        self._R_target = target[0]
        self._Z_target = target[1]
        self._R_weight = np.atleast_1d(weight[0])
        self._Z_weight = np.atleast_1d(weight[1])
        self._surface = surface
        self._R_modes = modes[0]
        self._Z_modes = modes[1]
        super().__init__(eq=eq, target=target, weight=weight)

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
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
        self._check_dimensions()

        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, Rb_lmn, Zb_lmn):
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
        f = jnp.concatenate((Rb, Zb))
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


class FixedPressure(_Objective):
    """Fixes pressure coefficients."""

    def __init__(
        self, eq=None, target=None, weight=1, profile=None, modes=True,
    ):
        """Initialize a FixedPressure Objective.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        target : tuple, float, ndarray, optional
            Target value(s) of the objective.
            len(target) = len(weight) = len(modes). If None, uses profile coefficients.
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            len(target) = len(weight) = len(modes)
        profile : Profile, optional
            Profile containing the radial modes to evaluate at.
        modes : ndarray, optional
            Basis modes numbers [l,m,n] of boundary modes to fix.
            len(target) = len(weight) = len(modes).
            If True/False uses all/none of the profile modes.

        """
        self._profile = profile
        self._modes = modes
        super().__init__(eq=eq, target=target, weight=weight)

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        if self._profile is None:
            self._profile = eq.pressure
        if not isinstance(self._profile, PowerSeriesProfile):
            raise NotImplementedError("profile must be of type `PowerSeriesProfile`")
            # TODO: add implementation for SplineProfile & MTanhProfile

        # find inidies of profile modes to fix
        if self._modes is False or self._modes is None:  # no modes
            self._idx = np.array([], dtype=int)
        elif self._modes is True:  # all modes in profile
            self._idx = np.arange(self._profile.basis.num_modes)
            idx = self._idx
        else:  # specified modes
            dtype = {
                "names": ["f{}".format(i) for i in range(3)],
                "formats": 3 * [self._modes.dtype],
            }
            _, self._idx, idx = np.intersect1d(
                self._profile.basis.modes.view(dtype), self._modes.view(dtype)
            )
            if self._idx.size < self._modes.shape[0]:
                warnings.warn(
                    colored(
                        "Some of the given modes are not in the pressure profile, ",
                        +"these modes will not be fixed.",
                        "yellow",
                    )
                )

        self._dim_f = self._idx.size

        # set target values for pressure coefficients
        if self._target[0] is None:  # use profile coefficients
            self._target = self._profile.params[self._idx]
        elif self._target.size == 1:  # use scalar target for all modes
            self._target = self._target * np.ones((self._dim_f,))
        elif self._target.size == self._modes.shape[0]:  # use given array target
            self._target = self._target[idx]
        else:
            raise ValueError("target must be the same size as modes")

        # check weights
        if self._weight.size == 1:  # use scalar weight for all modes
            self._weight = self._weight * np.ones((self._dim_f,))
        elif self._weight.size == self._modes.shape[0]:  # use given array weight
            self._weight = self._weight[idx]
        else:
            raise ValueError("weight must be the same size as modes")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, p_l):
        return p_l[self._idx] - self._target

    def compute(self, p_l, **kwargs):
        """Compute fixed-pressure profile errors.

        Parameters
        ----------
        p_l : ndarray
            Spectral coefficients of p(rho) -- pressure profile.

        Returns
        -------
        f : ndarray
            Pressure profile errors, in Pascals.

        """
        return self._compute(p_l) * self._weight

    def callback(self, p_l, **kwargs):
        """Print fixed-pressure profile errors.

        Parameters
        ----------
        p_l : ndarray
            Spectral coefficients of p(rho) -- pressure profile.

        """
        f = self._compute(p_l)
        print("Fixed-pressure profile error: {:10.3e} (Pa)".format(jnp.linalg.norm(f)))
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
        return "fixed-pressure"


class FixedIota(_Objective):
    """Fixes rotational transform coefficients."""

    def __init__(
        self, eq=None, target=None, weight=1, profile=None, modes=True,
    ):
        """Initialize a FixedIota Objective.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        target : tuple, float, ndarray, optional
            Target value(s) of the objective.
            len(target) = len(weight) = len(modes). If None, uses profile coefficients.
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            len(target) = len(weight) = len(modes)
        profile : Profile, optional
            Profile containing the radial modes to evaluate at.
        modes : ndarray, optional
            Basis modes numbers [l,m,n] of boundary modes to fix.
            len(target) = len(weight) = len(modes).
            If True/False uses all/none of the profile modes.

        """
        self._profile = profile
        self._modes = modes
        super().__init__(eq=eq, target=target, weight=weight)

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        if self._profile is None:
            self._profile = eq.iota
        if not isinstance(self._profile, PowerSeriesProfile):
            raise NotImplementedError("profile must be of type `PowerSeriesProfile`")
            # TODO: add implementation for SplineProfile & MTanhProfile

        # find inidies of profile modes to fix
        if self._modes is False or self._modes is None:  # no modes
            self._idx = np.array([], dtype=int)
        elif self._modes is True:  # all modes in profile
            self._idx = np.arange(self._profile.basis.num_modes)
            idx = self._idx
        else:  # specified modes
            dtype = {
                "names": ["f{}".format(i) for i in range(3)],
                "formats": 3 * [self._modes.dtype],
            }
            _, self._idx, idx = np.intersect1d(
                self._profile.basis.modes.view(dtype), self._modes.view(dtype)
            )
            if self._idx.size < self._modes.shape[0]:
                warnings.warn(
                    colored(
                        "Some of the given modes are not in the iota profile, ",
                        +"these modes will not be fixed.",
                        "yellow",
                    )
                )

        self._dim_f = self._idx.size

        # set target values for iota coefficients
        if self._target[0] is None:  # use profile coefficients
            self._target = self._profile.params[self._idx]
        elif self._target.size == 1:  # use scalar target for all modes
            self._target = self._target * np.ones((self._dim_f,))
        elif self._target.size == self._modes.shape[0]:  # use given array target
            self._target = self._target[idx]
        else:
            raise ValueError("target must be the same size as modes")

        # check weights
        if self._weight.size == 1:  # use scalar weight for all modes
            self._weight = self._weight * np.ones((self._dim_f,))
        elif self._weight.size == self._modes.shape[0]:  # use given array weight
            self._weight = self._weight[idx]
        else:
            raise ValueError("weight must be the same size as modes")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, i_l):
        return i_l[self._idx] - self._target

    def compute(self, i_l, **kwargs):
        """Compute fixed-iota profile errors.

        Parameters
        ----------
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.

        Returns
        -------
        f : ndarray
            Rotational transform profile errors.

        """
        return self._compute(i_l) * self._weight

    def callback(self, i_l, **kwargs):
        """Print fixed-iota profile errors.

        Parameters
        ----------
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.

        """
        f = self._compute(i_l)
        print("Fixed-iota profile error: {:10.3e}".format(jnp.linalg.norm(f)))
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
        return "fixed-iota"


class FixedPsi(_Objective):
    """Fixes total toroidal magnetic flux within the last closed flux surface."""

    def __init__(self, eq=None, target=None, weight=1):
        """Initialize a FixedIota Objective.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        target : float, optional
            Target value(s) of the objective. If None, uses Equilibrium value.
        weight : float, optional
            Weighting to apply to the Objective, relative to other Objectives.

        """
        super().__init__(eq=eq, target=target, weight=weight)

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        # set target value for Psi
        if self._target[0] is None:  # use Equilibrium value
            self._target = np.atleast_1d(eq.Psi)

        self._dim_f = 1

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, Psi):
        return Psi - self._target

    def compute(self, Psi, **kwargs):
        """Compute fixed-Psi error.

        Parameters
        ----------
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface, in Webers.

        Returns
        -------
        f : ndarray
            Total toroidal magnetic flux error, in Webers.

        """
        return self._compute(Psi) * self._weight

    def callback(self, Psi, **kwargs):
        """Print fixed-Psi error.

        Parameters
        ----------
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface, in Webers.

        """
        f = self._compute(Psi)
        print("Fixed-Psi error: {:10.3e} (Wb)".format(f))
        return None

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return True

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return True

    @property
    def name(self):
        """Name of objective function (str)."""
        return "fixed-Psi"


class LCFSBoundary(_Objective):
    """Boundary condition on the last closed flux surface."""

    def __init__(self, eq=None, target=0, weight=1, surface=None):
        """Initialize a LCFSBoundary Objective.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        target : float, ndarray, optional
            This target always gets overridden to be 0.
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            len(weight) must be equal to Objective.dim_f
        surface : FourierRZToroidalSurface, optional
            Toroidal surface containing the Fourier modes to evaluate at.

        """
        target = 0
        self._surface = surface
        super().__init__(eq=eq, target=target, weight=weight)

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
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

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, R_lmn, Z_lmn, Rb_lmn, Zb_lmn):
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
        f = jnp.concatenate((Rb, Zb))
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


class TargetIota(_Objective):
    """Targets a rotational transform profile."""

    def __init__(self, eq=None, target=1, weight=1, profile=None, grid=None):
        """Initialize a TargetIota Objective.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        target : tuple, float, ndarray, optional
            Target value(s) of the objective.
            len(target) = len(weight) = len(modes). If None, uses profile coefficients.
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            len(target) = len(weight) = len(modes)
        profile : Profile, optional
            Profile containing the radial modes to evaluate at.
        grid : Grid, optional
            Collocation grid containing the nodes to evaluate at.

        """
        self._profile = profile
        self._grid = grid
        super().__init__(eq=eq, target=target, weight=weight)

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        if self._profile is None:
            self._profile = eq.iota
        if self._grid is None:
            self._grid = LinearGrid(L=2, NFP=eq.NFP, axis=True, rho=[0, 1])

        self._dim_f = self._grid.num_nodes

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._iota = eq.iota.copy()
        self._iota.grid = self._grid

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, i_l):
        data = compute_rotational_transform(i_l, self._iota)
        return data["iota"] - self._target

    def compute(self, i_l, **kwargs):
        """Compute rotational transform profile errors.

        Parameters
        ----------
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.

        Returns
        -------
        f : ndarray
            Rotational transform profile errors.

        """
        return self._compute(i_l) * self._weight

    def callback(self, i_l, **kwargs):
        """Print rotational transform profile errors.

        Parameters
        ----------
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.

        """
        f = self._compute(i_l)
        print("Target-iota profile error: {:10.3e}".format(jnp.linalg.norm(f)))
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
        return "target-iota"


class Volume(_Objective):
    """Plasma volume."""

    def __init__(self, eq=None, target=0, weight=1, grid=None):
        """Initialize a Volume Objective.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        target : float, ndarray, optional
            Target value(s) of the objective.
            len(target) must be equal to Objective.dim_f
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            len(weight) must be equal to Objective.dim_f
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.

        """
        self._grid = grid
        super().__init__(eq=eq, target=target, weight=weight)

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        if self._grid is None:
            self._grid = QuadratureGrid(L=eq.L, M=eq.M, N=eq.N, NFP=eq.NFP)

        self._dim_f = 1

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._R_transform = Transform(self._grid, eq.R_basis, derivs=1, build=True)
        self._Z_transform = Transform(self._grid, eq.Z_basis, derivs=1, build=True)

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, R_lmn, Z_lmn):
        data = compute_volume(R_lmn, Z_lmn, self._R_transform, self._Z_transform)
        return data["V"]

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
        return (jnp.atleast_1d(V) - self._target) * self._weight

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
    """MHD energy.

    W = integral( B^2 / (2*mu0) + p / (gamma - 1) ) dV  (J)

    """

    _io_attrs_ = _Objective._io_attrs_ + ["gamma"]

    def __init__(self, eq=None, target=0, weight=1, grid=None, gamma=0):
        """Initialize an Energy Objective.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        target : float, ndarray, optional
            Target value(s) of the objective.
            len(target) must be equal to Objective.dim_f
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            len(weight) must be equal to Objective.dim_f
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.
        gamma : float, optional
            Adiabatic (compressional) index. Default = 0.

        """
        self._grid = grid
        self._gamma = gamma
        super().__init__(eq=eq, target=target, weight=weight)

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        if self._grid is None:
            self._grid = QuadratureGrid(L=eq.L, M=eq.M, N=eq.N, NFP=eq.NFP)

        self._dim_f = 1

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

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi):
        data = compute_energy(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            p_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
            self._pressure,
            self._gamma,
        )
        return data["W"], data["W_B"], data["W_p"]

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
            Total toroidal magnetic flux within the last closed flux surface, in Webers.

        Returns
        -------
        W : float
            Total MHD energy in the plasma volume, in Joules.

        """
        W, W_B, W_p = self._compute(R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi)
        return (jnp.atleast_1d(W) - self._target) * self._weight

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
            Total toroidal magnetic flux within the last closed flux surface, in Webers.

        """
        W, W_B, W_p = self._compute(R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi)
        print(
            "Total MHD energy: {:10.3e}, ".format(W)
            + "Magnetic Energy: {:10.3e}, ".format(W_B)
            + "Pressure Energy: {:10.3e} ".format(W_p)
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


class RadialForceBalance(_Objective):
    """Radial MHD force balance.

    F_rho = sqrt(g) (B^zeta J^theta - B^theta J^zeta) - grad(p)
    f_rho = F_rho |grad(rho)| dV  (N)

    """

    def __init__(self, eq=None, target=0, weight=1, grid=None, norm=False):
        """Initialize a RadialForceBalance Objective.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        target : float, ndarray, optional
            Target value(s) of the objective.
            len(target) must be equal to Objective.dim_f
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            len(weight) must be equal to Objective.dim_f
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.
        norm : bool, optional
            Whether to normalize the objective values (make dimensionless).

        """
        self._grid = grid
        self._norm = norm
        super().__init__(eq=eq, target=target, weight=weight)

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        if self._grid is None:
            self._grid = ConcentricGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
                rotation="cos",
                node_pattern=eq.node_pattern,
            )

        self._dim_f = self._grid.num_nodes

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._iota = eq.iota.copy()
        self._pressure = eq.pressure.copy()
        self._iota.grid = self._grid
        self._pressure.grid = self._grid

        self._R_transform = Transform(self._grid, eq.R_basis, derivs=2, build=True)
        self._Z_transform = Transform(self._grid, eq.Z_basis, derivs=2, build=True)
        self._L_transform = Transform(self._grid, eq.L_basis, derivs=2, build=True)

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi):
        data = compute_force_error(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            p_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
            self._pressure,
        )
        f_rho = data["F_rho"] * data["|grad(rho)|"]
        f = data["|F|"]
        if self._norm:
            f_rho = f_rho / data["|grad(p)|"]
            f = f / data["|grad(p)|"]
        f_rho = f_rho * data["sqrt(g)"] * self._grid.weights
        f = f * data["sqrt(g)"] * self._grid.weights
        return f_rho, f

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi, **kwargs):
        """Compute radial MHD force balance errors.

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
            Total toroidal magnetic flux within the last closed flux surface, in Webers.

        Returns
        -------
        f_rho : ndarray
            Radial MHD force balance error at each node (N).

        """
        f_rho, f = self._compute(R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi)
        return (f_rho - self._target) * self._weight

    def callback(self, R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi, **kwargs):
        """Print radial MHD force balance error.

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
            Total toroidal magnetic flux within the last closed flux surface, in Webers.

        """
        f_rho, f = self._compute(R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi)
        if self._norm:
            units = "(normalized)"
        else:
            units = "(N)"
        print(
            "Radial force: {:10.3e}, ".format(jnp.linalg.norm(f_rho))
            + "Total force: {:10.3e} ".format(jnp.linalg.norm(f))
            + units
        )
        return None

    @property
    def norm(self):
        """bool: Whether the objective values are normalized."""
        return self._norm

    @norm.setter
    def norm(self, norm):
        self._norm = norm

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return False

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return False

    @property
    def name(self):
        """Name of objective function (str)."""
        return "radial force"


class HelicalForceBalance(_Objective):
    """Helical MHD force balance.

    F_beta = sqrt(g) J^rho
    beta = B^zeta grad(theta) - B^theta grad(zeta)
    f_beta = F_beta |beta| dV  (N)

    """

    def __init__(self, eq=None, target=0, weight=1, grid=None, norm=False):
        """Initialize a HelicalForceBalance Objective.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        target : float, ndarray, optional
            Target value(s) of the objective.
            len(target) must be equal to Objective.dim_f
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            len(weight) must be equal to Objective.dim_f
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.
        norm : bool, optional
            Whether to normalize the objective values (make dimensionless).

        """
        self._grid = grid
        self._norm = norm
        super().__init__(eq=eq, target=target, weight=weight)

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        if self._grid is None:
            self._grid = ConcentricGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
                rotation="sin",
                node_pattern=eq.node_pattern,
            )

        self._dim_f = self._grid.num_nodes

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._iota = eq.iota.copy()
        self._pressure = eq.pressure.copy()
        self._iota.grid = self._grid
        self._pressure.grid = self._grid

        self._R_transform = Transform(self._grid, eq.R_basis, derivs=2, build=True)
        self._Z_transform = Transform(self._grid, eq.Z_basis, derivs=2, build=True)
        self._L_transform = Transform(self._grid, eq.L_basis, derivs=2, build=True)

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi):
        data = compute_force_error(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            p_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
            self._pressure,
        )
        f_beta = data["F_beta"] * data["|beta|"]
        f = data["|F|"]
        if self._norm:
            f_beta = f_beta / data["|grad(p)|"]
            f = f / data["|grad(p)|"]
        f_beta = f_beta * data["sqrt(g)"] * self._grid.weights
        f = f * data["sqrt(g)"] * self._grid.weights
        return f_beta, f

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi, **kwargs):
        """Compute helical MHD force balance errors.

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
            Total toroidal magnetic flux within the last closed flux surface, in Webers.

        Returns
        -------
        f_beta : ndarray
            Helical MHD force balance error at each node (N).

        """
        f_beta, f = self._compute(R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi)
        return (f_beta - self._target) * self._weight

    def callback(self, R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi, **kwargs):
        """Print helical MHD force balance error.

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
            Total toroidal magnetic flux within the last closed flux surface, in Webers.

        """
        f_beta, f = self._compute(R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi)
        if self._norm:
            units = "(normalized)"
        else:
            units = "(N)"
        print(
            "Helical force: {:10.3e}, ".format(jnp.linalg.norm(f_beta))
            + "Total force: {:10.3e} ".format(jnp.linalg.norm(f))
            + units
        )
        return None

    @property
    def norm(self):
        """bool: Whether the objective values are normalized."""
        return self._norm

    @norm.setter
    def norm(self, norm):
        self._norm = norm

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return False

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return False

    @property
    def name(self):
        """Name of objective function (str)."""
        return "helical force"


class RadialCurrent(_Objective):
    """Radial current."""

    def __init__(self, eq=None, target=0, weight=1, grid=None, norm=False):
        """Initialize a RadialCurrent Objective.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        target : float, ndarray, optional
            Target value(s) of the objective.
            len(target) must be equal to Objective.dim_f
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            len(weight) must be equal to Objective.dim_f
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.
        norm : bool, optional
            Whether to normalize the objective values (make dimensionless).

        """
        self._grid = grid
        self._norm = norm
        super().__init__(eq=eq, target=target, weight=weight)

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        if self._grid is None:
            self._grid = ConcentricGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
                rotation="sin",
                node_pattern=eq.node_pattern,
            )

        self._dim_f = self._grid.num_nodes

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._iota = eq.iota.copy()
        self._iota.grid = self._grid

        self._R_transform = Transform(self._grid, eq.R_basis, derivs=2, build=True)
        self._Z_transform = Transform(self._grid, eq.Z_basis, derivs=2, build=True)
        self._L_transform = Transform(self._grid, eq.L_basis, derivs=2, build=True)

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, R_lmn, Z_lmn, L_lmn, i_l, Psi):
        data = compute_contravariant_current_density(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
        )
        data = compute_covariant_metric_coefficients(
            R_lmn, Z_lmn, self._R_transform, self._Z_transform, data=data
        )
        f = data["J^rho"] * jnp.sqrt(data["g_rr"])
        if self._norm:
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
            )
            B = jnp.mean(data["|B|"] * data["sqrt(g)"]) / jnp.mean(data["sqrt(g)"])
            R = jnp.mean(data["R"] * data["sqrt(g)"]) / jnp.mean(data["sqrt(g)"])
            f = f * mu_0 / (B * R ** 2)
        f = f * data["sqrt(g)"] * self._grid.weights
        return f

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, Psi, **kwargs):
        """Compute radial current.

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
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface, in Webers.

        Returns
        -------
        f : ndarray
            Radial current at each node (A*m).

        """
        f = self._compute(R_lmn, Z_lmn, L_lmn, i_l, Psi)
        return (f - self._target) * self._weight

    def callback(self, R_lmn, Z_lmn, L_lmn, i_l, Psi, **kwargs):
        """Print radial current.

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
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface, in Webers.

        """
        f = self._compute(R_lmn, Z_lmn, L_lmn, i_l, Psi)
        if self._norm:
            units = "(normalized)"
        else:
            units = "(A*m)"
        print("Radial current: {:10.3e} ".format(jnp.linalg.norm(f)) + units)
        return None

    @property
    def norm(self):
        """bool: Whether the objective values are normalized."""
        return self._norm

    @norm.setter
    def norm(self, norm):
        self._norm = norm

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return False

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return False

    @property
    def name(self):
        """Name of objective function (str)."""
        return "radial current"


class PoloidalCurrent(_Objective):
    """Poloidal current."""

    def __init__(self, eq=None, target=0, weight=1, grid=None, norm=False):
        """Initialize a PoloidalCurrent Objective.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        target : float, ndarray, optional
            Target value(s) of the objective.
            len(target) must be equal to Objective.dim_f
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            len(weight) must be equal to Objective.dim_f
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.
        norm : bool, optional
            Whether to normalize the objective values (make dimensionless).

        """
        self._grid = grid
        self._norm = norm
        super().__init__(eq=eq, target=target, weight=weight)

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        if self._grid is None:
            self._grid = ConcentricGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
                rotation="cos",
                node_pattern=eq.node_pattern,
            )

        self._dim_f = self._grid.num_nodes

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._iota = eq.iota.copy()
        self._iota.grid = self._grid

        self._R_transform = Transform(self._grid, eq.R_basis, derivs=2, build=True)
        self._Z_transform = Transform(self._grid, eq.Z_basis, derivs=2, build=True)
        self._L_transform = Transform(self._grid, eq.L_basis, derivs=2, build=True)

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, R_lmn, Z_lmn, L_lmn, i_l, Psi):
        data = compute_contravariant_current_density(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
        )
        data = compute_covariant_metric_coefficients(
            R_lmn, Z_lmn, self._R_transform, self._Z_transform, data=data
        )
        f = data["J^theta"] * jnp.sqrt(data["g_tt"])
        if self._norm:
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
            )
            B = jnp.mean(data["|B|"] * data["sqrt(g)"]) / jnp.mean(data["sqrt(g)"])
            R = jnp.mean(data["R"] * data["sqrt(g)"]) / jnp.mean(data["sqrt(g)"])
            f = f * mu_0 / (B * R ** 2)
        f = f * data["sqrt(g)"] * self._grid.weights
        return f

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, Psi, **kwargs):
        """Compute poloidal current.

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
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface, in Webers.

        Returns
        -------
        f : ndarray
            Poloidal current at each node (A*m).

        """
        f = self._compute(R_lmn, Z_lmn, L_lmn, i_l, Psi)
        return (f - self._target) * self._weight

    def callback(self, R_lmn, Z_lmn, L_lmn, i_l, Psi, **kwargs):
        """Print poloidal current.

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
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface, in Webers.

        """
        f = self._compute(R_lmn, Z_lmn, L_lmn, i_l, Psi)
        if self._norm:
            units = "(normalized)"
        else:
            units = "(A*m)"
        print("Poloidal current: {:10.3e} ".format(jnp.linalg.norm(f)) + units)
        return None

    @property
    def norm(self):
        """bool: Whether the objective values are normalized."""
        return self._norm

    @norm.setter
    def norm(self, norm):
        self._norm = norm

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return False

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return False

    @property
    def name(self):
        """Name of objective function (str)."""
        return "poloidal current"


class ToroidalCurrent(_Objective):
    """Toroidal current."""

    def __init__(self, eq=None, target=0, weight=1, grid=None, norm=False):
        """Initialize a ToroidalCurrent Objective.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        target : float, ndarray, optional
            Target value(s) of the objective.
            len(target) must be equal to Objective.dim_f
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            len(weight) must be equal to Objective.dim_f
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.
        norm : bool, optional
            Whether to normalize the objective values (make dimensionless).

        """
        self._grid = grid
        self._norm = norm
        super().__init__(eq=eq, target=target, weight=weight)

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        if self._grid is None:
            self._grid = ConcentricGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
                rotation="cos",
                node_pattern=eq.node_pattern,
            )

        self._dim_f = self._grid.num_nodes

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._iota = eq.iota.copy()
        self._iota.grid = self._grid

        self._R_transform = Transform(self._grid, eq.R_basis, derivs=2, build=True)
        self._Z_transform = Transform(self._grid, eq.Z_basis, derivs=2, build=True)
        self._L_transform = Transform(self._grid, eq.L_basis, derivs=2, build=True)

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, R_lmn, Z_lmn, L_lmn, i_l, Psi):
        data = compute_contravariant_current_density(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
        )
        data = compute_covariant_metric_coefficients(
            R_lmn, Z_lmn, self._R_transform, self._Z_transform, data=data
        )
        f = data["J^zeta"] * jnp.sqrt(data["g_zz"])
        if self._norm:
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
            )
            B = jnp.mean(data["|B|"] * data["sqrt(g)"]) / jnp.mean(data["sqrt(g)"])
            R = jnp.mean(data["R"] * data["sqrt(g)"]) / jnp.mean(data["sqrt(g)"])
            f = f * mu_0 / (B * R ** 2)
        f = f * data["sqrt(g)"] * self._grid.weights
        return f

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, Psi, **kwargs):
        """Compute toroidal current.

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
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface, in Webers.

        Returns
        -------
        f : ndarray
            Toroidal current at each node (A*m).

        """
        f = self._compute(R_lmn, Z_lmn, L_lmn, i_l, Psi)
        return (f - self._target) * self._weight

    def callback(self, R_lmn, Z_lmn, L_lmn, i_l, Psi, **kwargs):
        """Print toroidal current.

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
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface, in Webers.

        """
        f = self._compute(R_lmn, Z_lmn, L_lmn, i_l, Psi)
        if self._norm:
            units = "(normalized)"
        else:
            units = "(A*m)"
        print("Toroidal current: {:10.3e} ".format(jnp.linalg.norm(f)) + units)
        return None

    @property
    def norm(self):
        """bool: Whether the objective values are normalized."""
        return self._norm

    @norm.setter
    def norm(self, norm):
        self._norm = norm

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return False

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return False

    @property
    def name(self):
        """Name of objective function (str)."""
        return "toroidal current"


class QuasisymmetryFluxFunction(_Objective):
    """Quasi-symmetry flux function error."""

    def __init__(
        self, eq=None, target=0, weight=1, grid=None, helicity=(1, 0), norm=False
    ):
        """Initialize a QuasisymmetryFluxFunction Objective.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        target : float, ndarray, optional
            Target value(s) of the objective.
            len(target) must be equal to Objective.dim_f
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            len(weight) must be equal to Objective.dim_f
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.
        helicity : tuple, optional
            Type of quasi-symmetry (M, N).
        norm : bool, optional
            Whether to normalize the objective values (make dimensionless).

        """
        self._grid = grid
        self._helicity = helicity
        self._norm = norm
        super().__init__(eq=eq, target=target, weight=weight)

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        if self._grid is None:
            self._grid = LinearGrid(
                L=1,
                M=2 * eq.M_grid + 1,
                N=2 * eq.N_grid + 1,
                NFP=eq.NFP,
                sym=eq.sym,
                rho=0.75,
            )

        self._dim_f = self._grid.num_nodes

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._iota = eq.iota.copy()
        self._iota.grid = self._grid

        self._R_transform = Transform(self._grid, eq.R_basis, derivs=3, build=True)
        self._Z_transform = Transform(self._grid, eq.Z_basis, derivs=3, build=True)
        self._L_transform = Transform(self._grid, eq.L_basis, derivs=3, build=True)

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, R_lmn, Z_lmn, L_lmn, i_l, Psi):
        data = compute_quasisymmetry_error(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
            self._helicity,
        )
        f = data["QS_FF"] * self._grid.weights
        if self._norm:
            B = jnp.mean(data["|B|"] * data["sqrt(g)"]) / jnp.mean(data["sqrt(g)"])
            f = f / B ** 3
        return f

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, Psi, **kwargs):
        """Compute quasi-symmetry flux function errors.

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
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface, in Webers.

        Returns
        -------
        f : ndarray
            Quasi-symmetry flux function error at each node (T^3).

        """
        f = self._compute(R_lmn, Z_lmn, L_lmn, i_l, Psi)
        return (f - self._target) * self._weight

    def callback(self, R_lmn, Z_lmn, L_lmn, i_l, Psi, **kwargs):
        """Print quasi-symmetry flux function error.

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
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface, in Webers.

        """
        f = self._compute(R_lmn, Z_lmn, L_lmn, i_l, Psi)
        if self._norm:
            units = "(normalized)"
        else:
            units = "(T^3)"
        print(
            "Quasi-symmetry ({},{}) error: {:10.3e} ".format(
                self._helicity[0], self._helicity[1], jnp.linalg.norm(f)
            )
            + units
        )
        return None

    @property
    def helicity(self):
        """tuple: Type of quasi-symmetry (M, N)."""
        return self._helicity

    @helicity.setter
    def helicity(self, helicity):
        self._helicity = helicity

    @property
    def norm(self):
        """bool: Whether the objective values are normalized."""
        return self._norm

    @norm.setter
    def norm(self, norm):
        self._norm = norm

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return False

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return False

    @property
    def name(self):
        """Name of objective function (str)."""
        return "QS flux function"


class QuasisymmetryTripleProduct(_Objective):
    """Quasi-symmetry triple product error."""

    def __init__(self, eq=None, target=0, weight=1, grid=None, norm=False):
        """Initialize a QuasisymmetryTripleProduct Objective.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        target : float, ndarray, optional
            Target value(s) of the objective.
            len(target) must be equal to Objective.dim_f
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            len(weight) must be equal to Objective.dim_f
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.
        norm : bool, optional
            Whether to normalize the objective values (make dimensionless).

        """
        self._grid = grid
        self._norm = norm
        super().__init__(eq=eq, target=target, weight=weight)

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        if self._grid is None:
            self._grid = LinearGrid(
                L=1,
                M=2 * eq.M_grid + 1,
                N=2 * eq.N_grid + 1,
                NFP=eq.NFP,
                sym=eq.sym,
                rho=0.75,
            )

        self._dim_f = self._grid.num_nodes

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._iota = eq.iota.copy()
        self._iota.grid = self._grid

        self._R_transform = Transform(self._grid, eq.R_basis, derivs=3, build=True)
        self._Z_transform = Transform(self._grid, eq.Z_basis, derivs=3, build=True)
        self._L_transform = Transform(self._grid, eq.L_basis, derivs=3, build=True)

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def _compute(self, R_lmn, Z_lmn, L_lmn, i_l, Psi):
        data = compute_quasisymmetry_error(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
        )
        f = data["QS_TP"] * self._grid.weights
        if self._norm:
            B = jnp.mean(data["|B|"] * data["sqrt(g)"]) / jnp.mean(data["sqrt(g)"])
            R = jnp.mean(data["R"] * data["sqrt(g)"]) / jnp.mean(data["sqrt(g)"])
            f = f * R ** 2 / B ** 4
        return f

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, Psi, **kwargs):
        """Compute quasi-symmetry triple product errors.

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
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface, in Webers.

        Returns
        -------
        f : ndarray
            Quasi-symmetry flux function error at each node (T^4/m^2).

        """
        f = self._compute(R_lmn, Z_lmn, L_lmn, i_l, Psi)
        return (f - self._target) * self._weight

    def callback(self, R_lmn, Z_lmn, L_lmn, i_l, Psi, **kwargs):
        """Print quasi-symmetry triple product error.

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
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface, in Webers.

        """
        f = self._compute(R_lmn, Z_lmn, L_lmn, i_l, Psi)
        if self._norm:
            units = "(normalized)"
        else:
            units = "(T^4/m^2)"
        print("Quasi-symmetry error: {:10.3e} ".format(jnp.linalg.norm(f)) + units)
        return None

    @property
    def norm(self):
        """bool: Whether the objective values are normalized."""
        return self._norm

    @norm.setter
    def norm(self, norm):
        self._norm = norm

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar (or vector)."""
        return False

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return False

    @property
    def name(self):
        """Name of objective function (str)."""
        return "QS triple product"


def get_objective_function(method):
    """Get an objective function by name.

    Parameters
    ----------
    method : str
        Name of the desired objective function, eg ``'force'`` or ``'energy'``.

    Returns
    -------
    obj_fun : ObjectiveFunction
        Objective function with the desired objectives and constraints.

    """
    if method == "force":
        objectives = (RadialForceBalance(), HelicalForceBalance())
    elif method == "energy":
        objectives = Energy()

    constraints = (
        FixedBoundary(),
        FixedPressure(),
        FixedIota(),
        FixedPsi(),
        LCFSBoundary(),
    )
    return ObjectiveFunction(objectives, constraints)
