"""Base classes for objectives."""

from abc import ABC, abstractmethod
from inspect import getfullargspec

import numpy as np

from desc.backend import block_diag, jit, jnp, use_jax
from desc.compute import arg_order
from desc.derivatives import Derivative
from desc.io import IOAble
from desc.utils import Timer, is_broadcastable

# XXX: could use `indices` instead of `arg_order` in ObjectiveFunction loops


class ObjectiveFunction(IOAble):
    """Objective function comprised of one or more Objectives.

    Parameters
    ----------
    objectives : tuple of Objective
        List of objectives to be minimized.
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the objectives.
    use_jit : bool, optional
        Whether to just-in-time compile the objectives and derivatives.
    deriv_mode : {"batched", "blocked"}
        method for computing derivatives. "batched" is generally faster, "blocked" may
        use less memory. Note that the "blocked" Hessian will only be block diagonal.
    verbose : int, optional
        Level of output.

    """

    _io_attrs_ = ["_objectives"]

    def __init__(
        self, objectives, eq=None, use_jit=True, deriv_mode="batched", verbose=1
    ):

        if not isinstance(objectives, tuple):
            objectives = (objectives,)

        assert use_jit in {True, False}
        assert deriv_mode in {"batched", "blocked"}

        self._objectives = objectives
        self._use_jit = use_jit
        self._deriv_mode = deriv_mode
        self._built = False
        self._compiled = False

        if eq is not None:
            self.build(eq, use_jit=self._use_jit, verbose=verbose)

    def _set_state_vector(self):
        """Set state vector components, dimensions, and indices."""
        self._args = np.concatenate([obj.args for obj in self.objectives])
        self._args = [arg for arg in arg_order if arg in self._args]

        self._dimensions = self.objectives[0].dimensions

        self._dim_x = 0
        self._x_idx = {}
        for arg in self.args:
            self.x_idx[arg] = np.arange(self._dim_x, self._dim_x + self.dimensions[arg])
            self._dim_x += self.dimensions[arg]

    def _set_derivatives(self):
        """Set up derivatives of the objective functions."""
        self._derivatives = {"jac": {}, "grad": {}, "hess": {}}
        for arg in self.args:
            self._derivatives["jac"][arg] = lambda x, arg=arg: jnp.vstack(
                [
                    obj.derivatives["jac"][arg](
                        *self._kwargs_to_args(self.unpack_state(x), obj.args)
                    )
                    for obj in self.objectives
                ]
            )
            self._derivatives["grad"][arg] = lambda x, arg=arg: jnp.sum(
                jnp.array(
                    [
                        obj.derivatives["grad"][arg](
                            *self._kwargs_to_args(self.unpack_state(x), obj.args)
                        )
                        for obj in self.objectives
                    ]
                ),
                axis=0,
            )
            self._derivatives["hess"][arg] = lambda x, arg=arg: jnp.sum(
                jnp.array(
                    [
                        obj.derivatives["hess"][arg](
                            *self._kwargs_to_args(self.unpack_state(x), obj.args)
                        )
                        for obj in self.objectives
                    ]
                ),
                axis=0,
            )

        if self._deriv_mode == "blocked":
            self._grad = lambda x: jnp.concatenate(
                [jnp.atleast_1d(self._derivatives["grad"][arg](x)) for arg in self.args]
            )
            self._jac = lambda x: jnp.hstack(
                [self._derivatives["jac"][arg](x) for arg in self.args]
            )
            self._hess = lambda x: block_diag(
                *[self._derivatives["hess"][arg](x) for arg in self.args]
            )
        if self._deriv_mode == "batched":
            self._grad = Derivative(self.compute_scalar, mode="grad")
            self._hess = Derivative(self.compute_scalar, mode="hess")
            self._jac = Derivative(self.compute, mode="fwd")

    def jit(self):
        """Apply JIT to compute methods, or re-apply after updating self."""
        # can't loop here because del doesn't work on getattr
        # main idea is that when jitting a method, jax replaces that method
        # with a CompiledFunction object, with self compiled in. To re-jit
        # (ie, after updating attributes of self), we just need to delete the jax
        # CompiledFunction object, which will then leave the raw method in its place,
        # and then jit the raw method with the new self

        # doing str name type checking to avoid importing weird jax private stuff
        # for proper isinstance check
        if "CompiledFunction" in str(type(self.compute)):
            del self.compute
        self.compute = jit(self.compute)
        if "CompiledFunction" in str(type(self.compute_scalar)):
            del self.compute_scalar
        self.compute_scalar = jit(self.compute_scalar)
        if "CompiledFunction" in str(type(self.jac)):
            del self.jac
        self.jac = jit(self.jac)
        if "CompiledFunction" in str(type(self.hess)):
            del self.hess
        self.hess = jit(self.hess)
        if "CompiledFunction" in str(type(self.grad)):
            del self.grad
        self.grad = jit(self.grad)
        if "CompiledFunction" in str(type(self.jvp)):
            del self.jvp
        self.jvp = jit(self.jvp)

    def build(self, eq, use_jit=None, verbose=1):
        """Build the objective.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        if use_jit is not None:
            self._use_jit = use_jit
        timer = Timer()
        timer.start("Objective build")

        # build objectives
        self._dim_f = 0
        for objective in self.objectives:
            if verbose > 0:
                print("Building objective: " + objective.name)
            objective.build(eq, use_jit=self.use_jit, verbose=verbose)
            self._dim_f += objective.dim_f
        if self._dim_f == 1:
            self._scalar = True
        else:
            self._scalar = False

        self._set_state_vector()
        self._set_derivatives()
        if self.use_jit:
            self.jit()

        self._built = True
        timer.stop("Objective build")
        if verbose > 1:
            timer.disp("Objective build")

    def compute(self, x):
        """Compute the objective function.

        Parameters
        ----------
        x : ndarray
            State vector.

        Returns
        -------
        f : ndarray
            Objective function value(s).

        """
        kwargs = self.unpack_state(x)
        f = jnp.concatenate(
            [
                obj.compute(*self._kwargs_to_args(kwargs, obj.args))
                for obj in self.objectives
            ]
        )
        return f

    def compute_scalar(self, x):
        """Compute the scalar form of the objective function.

        Parameters
        ----------
        x : ndarray
            State vector.

        Returns
        -------
        f : float
            Objective function scalar value.

        """
        f = jnp.sum(self.compute(x) ** 2) / 2
        return f

    def print_value(self, x):
        """Print the value(s) of the objective.

        Parameters
        ----------
        x : ndarray
            State vector.

        """
        if self.compiled and self._compile_mode in {"scalar", "all"}:
            f = self.compute_scalar(x)
        else:
            f = jnp.sum(self.compute(x) ** 2) / 2
        print("Total (sum of squares): {:10.3e}, ".format(f))
        kwargs = self.unpack_state(x)
        for obj in self.objectives:
            obj.print_value(**kwargs)
        return None

    def unpack_state(self, x):
        """Unpack the state vector into its components.

        Parameters
        ----------
        x : ndarray
            State vector.

        Returns
        -------
        kwargs : dict
            Dictionary of the state components with argument names as keys.

        """
        if not self.built:
            raise RuntimeError("ObjectiveFunction must be built first.")

        x = jnp.atleast_1d(x)
        if x.size != self.dim_x:
            raise ValueError("Input vector dimension is invalid.")

        kwargs = {}
        for arg in self.args:
            kwargs[arg] = jnp.atleast_1d(x[self.x_idx[arg]])
        return kwargs

    def _kwargs_to_args(self, kwargs, args):
        tuple_args = (kwargs[arg] for arg in args)
        return tuple_args

    def x(self, eq):
        """Return the full state vector from the Equilibrium eq."""
        x = np.zeros((self.dim_x,))
        for arg in self.args:
            x[self.x_idx[arg]] = getattr(eq, arg)
        return x

    def grad(self, x):
        """Compute gradient vector of scalar form of the objective wrt x."""
        return jnp.atleast_1d(self._grad(x).squeeze())

    def hess(self, x):
        """Compute Hessian matrix of scalar form of the objective wrt x."""
        return jnp.atleast_2d(self._hess(x).squeeze())

    def jac(self, x):
        """Compute Jacobian matrx of vector form of the objective wrt x."""
        return jnp.atleast_2d(self._jac(x).squeeze())

    def jvp(self, v, x):
        """Compute Jacobian-vector product of the objective function.

        Parameters
        ----------
        v : tuple of ndarray
            Vectors to right-multiply the Jacobian by.
            The number of vectors given determines the order of derivative taken.
        x : ndarray
            Optimization variables.

        """
        if not isinstance(v, tuple):
            v = (v,)
        if len(v) == 1:
            return Derivative.compute_jvp(self.compute, 0, v[0], x)
        elif len(v) == 2:
            return Derivative.compute_jvp2(self.compute, 0, 0, v[0], v[1], x)
        elif len(v) == 3:
            return Derivative.compute_jvp3(self.compute, 0, 0, 0, v[0], v[1], v[2], x)
        else:
            raise NotImplementedError("Cannot compute JVP higher than 3rd order.")

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
        if not self.built:
            raise RuntimeError("ObjectiveFunction must be built first.")
        if not use_jax:
            self._compiled = True
            return

        timer = Timer()
        if mode == "auto" and self.scalar:
            mode = "scalar"
        elif mode == "auto":
            mode = "lsq"
        self._compile_mode = mode
        # variable values are irrelevant for compilation
        x = np.zeros((self.dim_x,))

        if verbose > 0:
            print("Compiling objective function and derivatives")
        timer.start("Total compilation time")

        if mode in ["scalar", "all"]:
            timer.start("Objective compilation time")
            _ = self.compute_scalar(x).block_until_ready()
            timer.stop("Objective compilation time")
            if verbose > 1:
                timer.disp("Objective compilation time")
            timer.start("Gradient compilation time")
            _ = self.grad(x).block_until_ready()
            timer.stop("Gradient compilation time")
            if verbose > 1:
                timer.disp("Gradient compilation time")
            timer.start("Hessian compilation time")
            _ = self.hess(x).block_until_ready()
            timer.stop("Hessian compilation time")
            if verbose > 1:
                timer.disp("Hessian compilation time")
        if mode in ["lsq", "all"]:
            timer.start("Objective compilation time")
            _ = self.compute(x).block_until_ready()
            timer.stop("Objective compilation time")
            if verbose > 1:
                timer.disp("Objective compilation time")
            timer.start("Jacobian compilation time")
            _ = self.jac(x).block_until_ready()
            timer.stop("Jacobian compilation time")
            if verbose > 1:
                timer.disp("Jacobian compilation time")

        timer.stop("Total compilation time")
        if verbose > 1:
            timer.disp("Total compilation time")
        self._compiled = True

    @property
    def objectives(self):
        """list: List of objectives."""
        return self._objectives

    @property
    def use_jit(self):
        """bool: Whether to just-in-time compile the objective and derivatives."""
        return self._use_jit

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar or vector."""
        if not self._built:
            raise RuntimeError("ObjectiveFunction must be built first.")
        return self._scalar

    @property
    def built(self):
        """bool: Whether the objectives have been built or not."""
        return self._built

    @property
    def compiled(self):
        """bool: Whether the functions have been compiled or not."""
        return self._compiled

    @property
    def args(self):
        """list: Names (str) of arguments to the compute functions."""
        return self._args

    @property
    def dimensions(self):
        """dict: Dimensions of the argument given by the dict keys."""
        return self._dimensions

    @property
    def x_idx(self):
        """dict: Indices of the components of the state vector."""
        return self._x_idx

    @property
    def dim_x(self):
        """int: Dimensional of the state vector."""
        if not self.built:
            raise RuntimeError("ObjectiveFunction must be built first.")
        return self._dim_x

    @property
    def dim_f(self):
        """int: Number of objective equations."""
        if not self.built:
            raise RuntimeError("ObjectiveFunction must be built first.")
        return self._dim_f


class _Objective(IOAble, ABC):
    """Objective (or constraint) used in the optimization of an Equilibrium.

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
    name : str
        Name of the objective function.

    """

    _io_attrs_ = [
        "_target",
        "_weight",
        "_name",
        "_args",
        "_normalize",
        "_normalize_target",
        "_normalization",
    ]

    def __init__(
        self,
        eq=None,
        target=0,
        weight=1,
        normalize=False,
        normalize_target=False,
        name=None,
    ):

        assert np.all(np.asarray(weight) > 0)
        assert normalize in {True, False}
        assert normalize_target in {True, False}
        self._target = target
        self._weight = weight
        self._normalize = normalize
        self._normalize_target = normalize_target
        self._normalization = 1
        self._name = name
        self._use_jit = None
        self._built = False
        # if args is already set don't overwrite it
        self._args = getattr(
            self,
            "_args",
            [arg for arg in getfullargspec(self.compute)[0] if arg != "self"],
        )
        if eq is not None:
            self.build(eq)

    def _set_dimensions(self, eq):
        """Set state vector component dimensions."""
        self._dimensions = {}
        self._dimensions["R_lmn"] = eq.R_basis.num_modes
        self._dimensions["Z_lmn"] = eq.Z_basis.num_modes
        self._dimensions["L_lmn"] = eq.L_basis.num_modes
        self._dimensions["p_l"] = eq.pressure.params.size
        try:
            self._dimensions["i_l"] = eq.iota.params.size
            self._dimensions["c_l"] = 0
        except AttributeError:
            self._dimensions["i_l"] = 0
            self._dimensions["c_l"] = eq.current.params.size
        self._dimensions["Psi"] = 1
        self._dimensions["Rb_lmn"] = eq.surface.R_basis.num_modes
        self._dimensions["Zb_lmn"] = eq.surface.Z_basis.num_modes

    def _set_derivatives(self):
        """Set up derivatives of the objective wrt each argument."""
        self._derivatives = {"jac": {}, "grad": {}, "hess": {}}

        for arg in arg_order:
            if arg in self.args:  # derivative wrt arg
                self._derivatives["jac"][arg] = Derivative(
                    self.compute,
                    argnum=self.args.index(arg),
                    mode="fwd",
                )
                self._derivatives["grad"][arg] = Derivative(
                    self.compute_scalar,
                    argnum=self.args.index(arg),
                    mode="grad",
                )
                self._derivatives["hess"][arg] = Derivative(
                    self.compute_scalar,
                    argnum=self.args.index(arg),
                    mode="hess",
                )
            else:  # these derivatives are always zero
                self._derivatives["jac"][arg] = lambda *args, **kwargs: jnp.zeros(
                    (self.dim_f, self.dimensions[arg])
                )
                self._derivatives["grad"][arg] = lambda *args, **kwargs: jnp.zeros(
                    (1, self.dimensions[arg])
                )
                self._derivatives["hess"][arg] = lambda *args, **kwargs: jnp.zeros(
                    (self.dimensions[arg], self.dimensions[arg])
                )

    def jit(self):
        """Apply JIT to compute methods, or re-apply after updating self."""
        # doing str name type checking to avoid importing weird jax private stuff
        # for proper isinstance check
        if "CompiledFunction" in str(type(self.compute)):
            del self.compute
        self.compute = jit(self.compute)
        if "CompiledFunction" in str(type(self.compute_scalar)):
            del self.compute_scalar
        self.compute_scalar = jit(self.compute_scalar)
        del self._derivatives
        self._set_derivatives()
        for mode, val in self._derivatives.items():
            for arg, deriv in val.items():
                self._derivatives[mode][arg] = jit(self._derivatives[mode][arg])

    def _check_dimensions(self):
        """Check that len(target) = len(weight) = dim_f."""
        self._target = np.asarray(self._target)
        self._weight = np.asarray(self._weight)
        if not is_broadcastable((self.dim_f,), self.target.shape):
            raise ValueError("len(target) != dim_f")
        if not is_broadcastable((self.dim_f,), self.weight.shape):
            raise ValueError("len(weight) != dim_f")

    def update_target(self, eq):
        """Update target values using an Equilibrium.

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium that will be optimized to satisfy the Objective.

        """
        self.target = np.atleast_1d(getattr(eq, self.target_arg, self.target))
        if self._use_jit:
            self.jit()

    @abstractmethod
    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays."""
        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives()
        if use_jit is not None:
            self._use_jit = use_jit
        if self._use_jit:
            self.jit()
        self._built = True

    @abstractmethod
    def compute(self, *args, **kwargs):
        """Compute the objective function."""

    def compute_scalar(self, *args, **kwargs):
        """Compute the scalar form of the objective."""
        if self.scalar:
            f = self.compute(*args, **kwargs)
        else:
            f = jnp.sum(self.compute(*args, **kwargs) ** 2) / 2
        return f.squeeze()

    def print_value(self, *args, **kwargs):
        """Print the value of the objective."""
        x = self._unshift_unscale(self.compute(*args, **kwargs))
        print(self._print_value_fmt.format(jnp.linalg.norm(x)) + self._units)
        if self._normalize:
            print(
                self._print_value_fmt.format(jnp.linalg.norm(x / self.normalization))
                + "(normalized)"
            )

    def _shift_scale(self, x):
        """Apply target and weighting."""
        target = (
            self.target / self.normalization if self._normalize_target else self.target
        )
        return (jnp.atleast_1d(x) / self.normalization - target) * self.weight

    def _unshift_unscale(self, x):
        """Undo target and weighting."""
        target = (
            self.target / self.normalization if self._normalize_target else self.target
        )
        return (x / self.weight + target) * self.normalization

    def xs(self, eq):
        """Return a tuple of args required by this objective from the Equilibrium eq."""
        return tuple(getattr(eq, arg) for arg in self.args)

    def _parse_args(self, *args, **kwargs):
        assert (len(args) == 0) or (len(kwargs) == 0), (
            "compute should be called with either positional or keyword arguments,"
            + " not both"
        )
        if len(args):
            assert len(args) == len(
                self.args
            ), f"compute expected {len(self.args)} arguments, got {len(args)}"
            params = {key: val for key, val in zip(self.args, args)}
        else:
            assert all([arg in kwargs for arg in self.args]), (
                "compute missing required keyword arguments "
                + f"{set(self.args).difference(kwargs.keys())}"
            )
            params = kwargs
        return params

    @property
    def target(self):
        """float: Target value(s) of the objective."""
        return self._target

    @target.setter
    def target(self, target):
        self._target = np.atleast_1d(target)
        self._check_dimensions()

    @property
    def weight(self):
        """float: Weighting to apply to the Objective, relative to other Objectives."""
        return self._weight

    @weight.setter
    def weight(self, weight):
        assert np.all(np.asarray(weight) > 0)
        self._weight = np.atleast_1d(weight)
        self._check_dimensions()

    @property
    def normalization(self):
        """float: normalizing scale factor."""
        if self._normalize and not self.built:
            raise ValueError("Objective must be built first")
        return self._normalization

    @property
    def built(self):
        """bool: Whether the transforms have been precomputed (or not)."""
        return self._built

    @property
    def args(self):
        """list: Names (str) of arguments to the compute functions."""
        return self._args

    @property
    def target_arg(self):
        """str: Name of argument corresponding to the target."""
        return ""

    @property
    def dimensions(self):
        """dict: Dimensions of the argument given by the dict keys."""
        return self._dimensions

    @property
    def derivatives(self):
        """dict: Derivatives of the function wrt the argument given by the dict keys."""
        return self._derivatives

    @property
    def dim_f(self):
        """int: Number of objective equations."""
        return self._dim_f

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar or vector."""
        return self._scalar

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return self._linear

    @property
    def fixed(self):
        """bool: Whether the objective fixes individual parameters (or linear combo)."""
        if self.linear:
            return self._fixed
        else:
            return False

    @property
    def name(self):
        """Name of objective function (str)."""
        return self._name
