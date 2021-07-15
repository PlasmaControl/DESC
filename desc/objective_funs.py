import numpy as np
from abc import ABC, abstractmethod
from termcolor import colored
import warnings
from scipy.constants import mu_0

from desc.backend import jnp, jit, use_jax
from desc.utils import unpack_state, Timer
from desc.io import IOAble
from desc.derivatives import Derivative
from desc.grid import QuadratureGrid
from desc.transform import Transform
from desc.compute_funs import (
    compute_pressure,
    compute_jacobian,
    compute_magnetic_field_magnitude,
)

__all__ = [
    "Energy",
    "get_objective_function",
]


class ObjectiveFunction(IOAble):
    """Objective function comprised of one or more Objectives."""

    _io_attrs_ = ["objectives", "constraints"]

    def __init__(self, objectives, constraints, eq=None, use_jit=True):
        """Initialize an Objective Function.

        Parameters
        ----------
        objectives : Objective, tuple
            List of objectives to be targeted during optimization.
        constraints : BoundaryCondition
            Boundary condition.
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.

        """
        if not isinstance(objectives, tuple):
            objectives = (objectives,)
        # TODO: generalize constraints to be Objectives like "objectives"

        self._objectives = objectives
        self._constraints = constraints
        self._use_jit = use_jit
        self._built = False
        self._compiled = False

        self._dim_x = self._constraints.dimy
        self._dim_f = 0
        for obj in self._objectives:
            self._dim_f += obj.dim_f

        self.set_derivatives(self._use_jit)

        if eq is not None:
            self.build(eq)

    def set_derivatives(self, use_jit=True, block_size="auto"):
        """Set up derivatives of the objective function.

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

    def build(self, eq, verbose=1):
        """Precompute the transforms.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        verbose : int, optional
            Level of output.

        """
        self._nR = eq.R_basis.num_modes
        self._nZ = eq.Z_basis.num_modes

        for obj in self._objectives:
            if not obj.built:
                if verbose > 0:
                    print("Building objective: " + obj.name)
                obj.build(eq, verbose=verbose)

        self._built = True

    # XXX: maybe have a method to set which variables are free/fixed for optimization?

    # TODO: use 'target' & 'weight'
    # TODO: generalize params to (x, **kwargs)
    def compute(self, x, Rb_lmn, Zb_lmn, p_l, i_l, Psi):
        """Compute the objective function.

        Parameters
        ----------
        x : ndarray
            State vector of optimization variables.
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
            Objective function value(s).

        """
        # x is really 'y', need to recover full state vector
        x = self._constraints.recover_from_constraints(x, Rb_lmn, Zb_lmn)
        R_lmn, Z_lmn, L_lmn = unpack_state(x, self._nR, self._nZ)

        kwargs = {
            "R_lmn": R_lmn,
            "Z_lmn": Z_lmn,
            "L_lmn": L_lmn,
            "Rb_lmn": Rb_lmn,
            "Zb_lmn": Zb_lmn,
            "i_l": i_l,
            "p_l": p_l,
            "Psi": Psi,
        }
        f = jnp.array([obj.compute(**kwargs) for obj in self._objectives])
        return jnp.concatenate(f)

    def compute_scalar(self, x, Rb_lmn, Zb_lmn, p_l, i_l, Psi):
        """Compute the scalar form of the objective.

        Parameters
        ----------
        x : ndarray
            State vector of optimization variables.
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
            Objective function value(s).

        """
        # x is really 'y', need to recover full state vector
        x = self._constraints.recover_from_constraints(x, Rb_lmn, Zb_lmn)
        R_lmn, Z_lmn, L_lmn = unpack_state(x, self._nR, self._nZ)

        kwargs = {
            "R_lmn": R_lmn,
            "Z_lmn": Z_lmn,
            "L_lmn": L_lmn,
            "Rb_lmn": Rb_lmn,
            "Zb_lmn": Zb_lmn,
            "i_l": i_l,
            "p_l": p_l,
            "Psi": Psi,
        }
        f = jnp.array([obj.compute_scalar(**kwargs) for obj in self._objectives])
        return jnp.sum(f)

    def callback(self, x, Rb_lmn, Zb_lmn, p_l, i_l, Psi):
        """Print the value(s) of the objective.

        Parameters
        ----------
        x : ndarray
            State vector of optimization variables.
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

        """
        # x is really 'y', need to recover full state vector
        x = self._constraints.recover_from_constraints(x, Rb_lmn, Zb_lmn)
        R_lmn, Z_lmn, L_lmn = unpack_state(x, self._nR, self._nZ)

        kwargs = {
            "R_lmn": R_lmn,
            "Z_lmn": Z_lmn,
            "L_lmn": L_lmn,
            "Rb_lmn": Rb_lmn,
            "Zb_lmn": Zb_lmn,
            "i_l": i_l,
            "p_l": p_l,
            "Psi": Psi,
        }
        for obj in self._objectives:
            obj.callback(**kwargs)
        return None

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
            self._compiled = True
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
        self._compiled = True

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
        """Compute Jacobian-vector product of the objective function.

        Eg, df/dx*v

        Parameters
        ----------
        argnum : int or tuple of int
            Integer describing which argument of the objective should be differentiated.
        v : ndarray or tuple of ndarray
            Vector to multiply the Jacobian matrix by, one per argnum.
        args : list
            List of arguments to the objective function.

        Returns
        -------
        df : ndarray
            Jacobian-vector product.

        """
        return Derivative.compute_jvp(self.compute, argnum, v, *args)

    def jvp2(self, argnum1, argnum2, v1, v2, *args):
        """Compute 2nd derivative Jacobian-vector product of the objective function.

        Eg, d^2f/dx^2*v1*v2

        Parameters
        ----------
        argnum1, argnum2 : int or tuple of int
            Integer describing which argument of the objective should be differentiated.
        v1, v2 : ndarray or tuple of ndarray
            Vector to multiply the Jacobian matrix by, one per argnum.
        args : list
            List of arguments to the objective function.

        Returns
        -------
        d2f : ndarray
            Jacobian-vector product.

        """
        return Derivative.compute_jvp2(self.compute, argnum1, argnum2, v1, v2, *args)

    def jvp3(self, argnum1, argnum2, argnum3, v1, v2, v3, *args):
        """Compute 3rd derivative jacobian-vector product of the objective function.

        Eg, d^3f/d3^2*v1*v2*v3

        Parameters
        ----------
        argnum1, argnum2, argnum2 : int or tuple of int
            Integer describing which argument of the objective should be differentiated.
        v1, v2, v3 : ndarray or tuple of ndarray
            Vector to multiply the Jacobian matrix by, one per argnum.
        args : list
            List of arguments to the objective function.

        Returns
        -------
        d3f : ndarray
            Jacobian-vector product.

        """
        return Derivative.compute_jvp3(
            self.compute, argnum1, argnum2, argnum3, v1, v2, v3, *args
        )

    def derivative(self, argnums, *args):
        """Compute arbitrary derivatives of the objective function.

        Parameters
        ----------
        argnums : int, str, tuple
            Integer or str or tuple of integers/strings describing which arguments
            of the objective should be differentiated.
            Passing a tuple with multiple values will compute a higher order derivative.
            Eg, argnums=(0,0) would compute the 2nd derivative with respect to the
            zeroth argument, while argnums=(3,5) would compute a mixed second
            derivative, first with respect to the third argument and then with
            respect to the fifth.
        args : list
            List of arguments to the objective function.

        Returns
        -------
        df : ndarray
            Specified derivative of the objective.

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

    @property
    def use_jit(self):
        """bool: Whether to just-in-time compile the objective and derivatives."""
        return self._use_jit

    @property
    def compiled(self):
        """bool: Whether the functions have been compiled."""
        return self._compiled

    @property
    def dim_x(self):
        """int: Number of optimization variables."""
        return self._dim_x

    @property
    def dim_f(self):
        """int: Number of objective equations."""
        return self._dim_f


class _Objective(IOAble, ABC):
    """Objective (or constraint) used in the optimization of an Equilibrium."""

    _io_attrs_ = [
        "grid",
        "target",
        "weight",
    ]

    def __init__(self, eq=None, grid=None, target=0, weight=1):
        """Initialize an Objective.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        grid : Grid, ndarray
            Collocation grid containing the (rho, theta, zeta) coordinates of the nodes
            to evaluate at.
        target : float, ndarray
            Target value(s) of the objective.
            len(target) must be equal to Objective.dim_f
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            len(weight) must be equal to Objective.dim_f

        """
        self._grid = grid
        self._target = target
        self._weight = weight
        self._built = False

        if self.scalar:
            self._dim_f = 1
        else:
            self._dim_f = None

        if eq is not None:
            self.build(eq, self._grid)

    @property
    def grid(self):
        """Grid: Collocation grid containing the nodes to evaluate at."""
        return self._grid

    @property
    def target(self):
        """float: Target value(s) of the objective."""
        return self._target

    @target.setter
    def target(self, target):
        self._target = target

    @property
    def weight(self):
        """float: Weighting to apply to the Objective, relative to other Objectives."""
        return self._weight

    @weight.setter
    def weight(self, weight):
        self._weight = weight

    @property
    def built(self):
        """bool: Whether the transforms have been precomputed."""
        return self._built

    @property
    def dim_f(self):
        """int: Number of objective equations."""
        return self._dim_f

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
    @abstractmethod
    def scalar(self):
        """Whether default "compute" method is a scalar or vector (bool)."""

    @property
    @abstractmethod
    def name(self):
        """Name of objective function (str)."""


class Volume(_Objective):
    """Plasma volume."""

    def __init__(self, eq=None, grid=None, target=0, weight=1, gamma=0):
        """Initialize a Volume Objective.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        grid : Grid, ndarray
            Collocation grid containing the (rho, theta, zeta) coordinates of the nodes
            to evaluate at.
        target : float, ndarray
            Target value(s) of the objective.
            len(target) must be equal to Objective.dim_f
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            len(weight) must be equal to Objective.dim_f

        """
        super().__init__(eq=eq, grid=grid, target=target, weight=weight)

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
        grid : Grid, ndarray
            Collocation grid containing the (rho, theta, zeta) coordinates of the nodes
            to evaluate at.
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

    def callback(self, R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi, **kwargs):
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
        """Whether default "compute" method is a scalar or vector (bool)."""
        return True

    @property
    def name(self):
        """Name of objective function (str)."""
        return "volume"


class Energy(_Objective):
    """MHD energy: W = integral( B^2 / (2*mu0) + p / (gamma - 1) ) dV."""

    _io_attrs_ = _Objective._io_attrs_ + ["gamma"]

    def __init__(self, eq=None, grid=None, target=0, weight=1, gamma=0):
        """Initialize an Energy Objective.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        grid : Grid, ndarray
            Collocation grid containing the (rho, theta, zeta) coordinates of the nodes
            to evaluate at.
        target : float, ndarray
            Target value(s) of the objective.
            len(target) must be equal to Objective.dim_f
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            len(weight) must be equal to Objective.dim_f
        gamma : float, optional
            Adiabatic (compressional) index. Default = 0.

        """
        self._gamma = gamma
        super().__init__(eq=eq, grid=grid, target=target, weight=weight)

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
        grid : Grid, ndarray
            Collocation grid containing the (rho, theta, zeta) coordinates of the nodes
            to evaluate at.
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
        """Whether default "compute" method is a scalar or vector (bool)."""
        return True

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
