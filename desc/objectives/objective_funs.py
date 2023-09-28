"""Base classes for objectives."""

import warnings
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
    use_jit : bool, optional
        Whether to just-in-time compile the objectives and derivatives.
    deriv_mode : {"batched", "blocked"}
        method for computing derivatives. "batched" is generally faster, "blocked" may
        use less memory. Note that the "blocked" Hessian will only be block diagonal.
    verbose : int, optional
        Level of output.

    """

    _io_attrs_ = ["_objectives"]

    def __init__(self, objectives, use_jit=True, deriv_mode="batched", verbose=1):
        if not isinstance(objectives, (tuple, list)):
            objectives = (objectives,)
        assert all(
            isinstance(obj, _Objective) for obj in objectives
        ), "members of ObjectiveFunction should be instances of _Objective"
        assert use_jit in {True, False}
        assert deriv_mode in {"batched", "blocked", "looped"}

        self._objectives = objectives
        self._use_jit = use_jit
        self._deriv_mode = deriv_mode
        self._built = False
        self._compiled = False

    def set_args(self, *args):
        """Set which arguments the objective should expect.

        Defaults to args from all sub-objectives. Additional arguments can be passed in.
        """
        self._args = list(np.concatenate([obj.args for obj in self.objectives]))
        self._args += list(args)
        self._args = [arg for arg in arg_order if arg in self._args]
        self._set_state_vector()

    def _set_state_vector(self):
        """Set state vector components, dimensions, and indices."""
        self._dimensions = self.objectives[0].dimensions

        self._dim_x = 0
        self._x_idx = {}
        for arg in self.args:
            self.x_idx[arg] = np.arange(self._dim_x, self._dim_x + self.dimensions[arg])
            self._dim_x += self.dimensions[arg]

    def _set_derivatives(self):
        """Set up derivatives of the objective functions."""
        self._derivatives = {
            "jac_scaled": {},
            "jac_unscaled": {},
            "grad": {},
            "hess": {},
        }
        for arg in self.args:
            self._derivatives["jac_scaled"][
                arg
            ] = lambda x, constants=None, arg=arg: jnp.vstack(
                [
                    obj.derivatives["jac_scaled"][arg](
                        *self._kwargs_to_args(self.unpack_state(x), obj.args),
                        constants=const,
                    )
                    for obj, const in zip(self.objectives, constants)
                ]
            )
            self._derivatives["jac_unscaled"][
                arg
            ] = lambda x, constants=None, arg=arg: jnp.vstack(
                [
                    obj.derivatives["jac_unscaled"][arg](
                        *self._kwargs_to_args(self.unpack_state(x), obj.args),
                        constants=const,
                    )
                    for obj, const in zip(self.objectives, constants)
                ]
            )
            self._derivatives["grad"][arg] = lambda x, constants=None, arg=arg: jnp.sum(
                jnp.array(
                    [
                        obj.derivatives["grad"][arg](
                            *self._kwargs_to_args(self.unpack_state(x), obj.args),
                            constants=const,
                        )
                        for obj, const in zip(self.objectives, constants)
                    ]
                ),
                axis=0,
            )
            self._derivatives["hess"][arg] = lambda x, constants=None, arg=arg: jnp.sum(
                jnp.array(
                    [
                        obj.derivatives["hess"][arg](
                            *self._kwargs_to_args(self.unpack_state(x), obj.args),
                            constants=const,
                        )
                        for obj, const in zip(self.objectives, constants)
                    ]
                ),
                axis=0,
            )

        if self._deriv_mode == "blocked":
            self._grad = lambda x, constants=None: jnp.concatenate(
                [
                    jnp.atleast_1d(
                        self._derivatives["grad"][arg](x, constants=constants)
                    )
                    for arg in self.args
                ]
            )
            self._jac_scaled = lambda x, constants=None: jnp.hstack(
                [
                    self._derivatives["jac_scaled"][arg](x, constants=constants)
                    for arg in self.args
                ]
            )
            self._jac_unscaled = lambda x, constants=None: jnp.hstack(
                [
                    self._derivatives["jac_unscaled"][arg](x, constants=constants)
                    for arg in self.args
                ]
            )
            self._hess = lambda x, constants=None: block_diag(
                *[
                    self._derivatives["hess"][arg](x, constants=constants)
                    for arg in self.args
                ]
            )
        if self._deriv_mode in {"batched", "looped"}:
            self._grad = Derivative(self.compute_scalar, mode="grad")
            self._hess = Derivative(self.compute_scalar, mode="hess")
        if self._deriv_mode == "batched":
            self._jac_scaled = Derivative(self.compute_scaled, mode="fwd")
            self._jac_unscaled = Derivative(self.compute_unscaled, mode="fwd")
        if self._deriv_mode == "looped":
            self._jac_scaled = Derivative(self.compute_scaled, mode="looped")
            self._jac_unscaled = Derivative(self.compute_unscaled, mode="looped")

    def jit(self):  # noqa: C901
        """Apply JIT to compute methods, or re-apply after updating self."""
        # can't loop here because del doesn't work on getattr
        # main idea is that when jitting a method, jax replaces that method
        # with a CompiledFunction object, with self compiled in. To re-jit
        # (ie, after updating attributes of self), we just need to delete the jax
        # CompiledFunction object, which will then leave the raw method in its place,
        # and then jit the raw method with the new self

        self._use_jit = True

        try:
            del self.compute_scaled
        except AttributeError:
            pass
        self.compute_scaled = jit(self.compute_scaled)

        try:
            del self.compute_scaled_error
        except AttributeError:
            pass
        self.compute_scaled_error = jit(self.compute_scaled_error)

        try:
            del self.compute_unscaled
        except AttributeError:
            pass
        self.compute_unscaled = jit(self.compute_unscaled)

        try:
            del self.compute_scalar
        except AttributeError:
            pass
        self.compute_scalar = jit(self.compute_scalar)

        try:
            del self.jac_scaled
        except AttributeError:
            pass
        self.jac_scaled = jit(self.jac_scaled)

        try:
            del self.jac_unscaled
        except AttributeError:
            pass
        self.jac_unscaled = jit(self.jac_unscaled)

        try:
            del self.hess
        except AttributeError:
            pass
        self.hess = jit(self.hess)

        try:
            del self.grad
        except AttributeError:
            pass
        self.grad = jit(self.grad)

        try:
            del self.jvp_scaled
        except AttributeError:
            pass
        self.jvp_scaled = jit(self.jvp_scaled)

        try:
            del self.jvp_unscaled
        except AttributeError:
            pass
        self.jvp_unscaled = jit(self.jvp_unscaled)

        try:
            del self.vjp_scaled
        except AttributeError:
            pass
        self.vjp_scaled = jit(self.vjp_scaled)

        try:
            del self.vjp_unscaled
        except AttributeError:
            pass
        self.vjp_unscaled = jit(self.vjp_unscaled)

        for obj in self._objectives:
            if obj._use_jit:
                obj.jit()

    def build(self, eq=None, use_jit=None, verbose=1):
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

        self.set_args()
        self._set_derivatives()
        if self.use_jit:
            self.jit()

        self._built = True
        timer.stop("Objective build")
        if verbose > 1:
            timer.disp("Objective build")

    def compute_unscaled(self, x, constants=None):
        """Compute the raw value of the objective function.

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives.

        Returns
        -------
        f : ndarray
            Objective function value(s).

        """
        kwargs = self.unpack_state(x)
        if constants is None:
            constants = self.constants
        f = jnp.concatenate(
            [
                obj.compute_unscaled(
                    *self._kwargs_to_args(kwargs, obj.args), constants=const
                )
                for obj, const in zip(self.objectives, constants)
            ]
        )
        return f

    def compute_scaled(self, x, constants=None):
        """Compute the objective function and apply weighting and normalization.

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives.

        Returns
        -------
        f : ndarray
            Objective function value(s).

        """
        kwargs = self.unpack_state(x)
        if constants is None:
            constants = self.constants
        f = jnp.concatenate(
            [
                obj.compute_scaled(
                    *self._kwargs_to_args(kwargs, obj.args), constants=const
                )
                for obj, const in zip(self.objectives, constants)
            ]
        )
        return f

    def compute_scaled_error(self, x, constants=None):
        """Compute and apply the target/bounds, weighting, and normalization.

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives.

        Returns
        -------
        f : ndarray
            Objective function value(s).

        """
        kwargs = self.unpack_state(x)
        if constants is None:
            constants = self.constants
        f = jnp.concatenate(
            [
                obj.compute_scaled_error(
                    *self._kwargs_to_args(kwargs, obj.args), constants=const
                )
                for obj, const in zip(self.objectives, constants)
            ]
        )
        return f

    def compute_scalar(self, x, constants=None):
        """Compute the sum of squares error.

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives.

        Returns
        -------
        f : float
            Objective function scalar value.

        """
        f = jnp.sum(self.compute_scaled_error(x, constants=constants) ** 2) / 2
        return f

    def print_value(self, x, constants=None):
        """Print the value(s) of the objective.

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives.

        """
        if constants is None:
            constants = self.constants
        if self.compiled and self._compile_mode in {"scalar", "all"}:
            f = self.compute_scalar(x, constants=constants)
        else:
            f = jnp.sum(self.compute_scaled_error(x, constants=constants) ** 2) / 2
        print("Total (sum of squares): {:10.3e}, ".format(f))
        kwargs = self.unpack_state(x)
        for obj, const in zip(self.objectives, constants):
            obj.print_value(**kwargs, constants=const)
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
            raise ValueError(
                "Input vector dimension is invalid, expected "
                + f"{self.dim_x} got {x.size}."
            )

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

    def grad(self, x, constants=None):
        """Compute gradient vector of scalar form of the objective wrt x."""
        if constants is None:
            constants = self.constants
        return jnp.atleast_1d(self._grad(x, constants).squeeze())

    def hess(self, x, constants=None):
        """Compute Hessian matrix of scalar form of the objective wrt x."""
        if constants is None:
            constants = self.constants
        return jnp.atleast_2d(self._hess(x, constants).squeeze())

    def jac_scaled(self, x, constants=None):
        """Compute Jacobian matrix of vector form of the objective wrt x."""
        if constants is None:
            constants = self.constants
        return jnp.atleast_2d(self._jac_scaled(x, constants).squeeze())

    def jac_unscaled(self, x, constants=None):
        """Compute Jacobian matrix of vector form of the objective wrt x, unweighted."""
        if constants is None:
            constants = self.constants
        return jnp.atleast_2d(self._jac_unscaled(x, constants).squeeze())

    def jvp_scaled(self, v, x):
        """Compute Jacobian-vector product of the objective function.

        Uses the scaled form of the objective.

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
            return Derivative.compute_jvp(self.compute_scaled, 0, v[0], x)
        elif len(v) == 2:
            return Derivative.compute_jvp2(self.compute_scaled, 0, 0, v[0], v[1], x)
        elif len(v) == 3:
            return Derivative.compute_jvp3(
                self.compute_scaled, 0, 0, 0, v[0], v[1], v[2], x
            )
        else:
            raise NotImplementedError("Cannot compute JVP higher than 3rd order.")

    def jvp_unscaled(self, v, x):
        """Compute Jacobian-vector product of the objective function.

        Uses the unscaled form of the objective.

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
            return Derivative.compute_jvp(self.compute_unscaled, 0, v[0], x)
        elif len(v) == 2:
            return Derivative.compute_jvp2(self.compute_unscaled, 0, 0, v[0], v[1], x)
        elif len(v) == 3:
            return Derivative.compute_jvp3(
                self.compute_unscaled, 0, 0, 0, v[0], v[1], v[2], x
            )
        else:
            raise NotImplementedError("Cannot compute JVP higher than 3rd order.")

    def vjp_scaled(self, v, x):
        """Compute vector-Jacobian product of the objective function.

        Uses the scaled form of the objective.

        Parameters
        ----------
        v : ndarray
            Vector to left-multiply the Jacobian by.
        x : ndarray
            Optimization variables.

        """
        return Derivative.compute_vjp(self.compute_scaled, 0, v, x)

    def vjp_unscaled(self, v, x):
        """Compute vector-Jacobian product of the objective function.

        Uses the unscaled form of the objective.

        Parameters
        ----------
        v : ndarray
            Vector to left-multiply the Jacobian by.
        x : ndarray
            Optimization variables.

        """
        return Derivative.compute_vjp(self.compute_unscaled, 0, v, x)

    def compile(self, mode="auto", verbose=1):
        """Call the necessary functions to ensure the function is compiled.

        Parameters
        ----------
        mode : {"auto", "lsq", "scalar", "bfgs", "all"}
            Whether to compile for least squares optimization or scalar optimization.
            "auto" compiles based on the type of objective, either scalar or lsq
            "bfgs" compiles only scalar objective and gradient,
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
            print(
                "Compiling objective function and derivatives: "
                + f"{[obj.name for obj in self.objectives]}"
            )
        timer.start("Total compilation time")

        if mode in ["scalar", "bfgs", "all"]:
            timer.start("Objective compilation time")
            _ = self.compute_scalar(x, self.constants).block_until_ready()
            timer.stop("Objective compilation time")
            if verbose > 1:
                timer.disp("Objective compilation time")
            timer.start("Gradient compilation time")
            _ = self.grad(x, self.constants).block_until_ready()
            timer.stop("Gradient compilation time")
            if verbose > 1:
                timer.disp("Gradient compilation time")
        if mode in ["scalar", "all"]:
            timer.start("Hessian compilation time")
            _ = self.hess(x, self.constants).block_until_ready()
            timer.stop("Hessian compilation time")
            if verbose > 1:
                timer.disp("Hessian compilation time")
        if mode in ["lsq", "all"]:
            timer.start("Objective compilation time")
            _ = self.compute_scaled(x, self.constants).block_until_ready()
            timer.stop("Objective compilation time")
            if verbose > 1:
                timer.disp("Objective compilation time")
            timer.start("Jacobian compilation time")
            _ = self.jac_scaled(x, self.constants).block_until_ready()
            timer.stop("Jacobian compilation time")
            if verbose > 1:
                timer.disp("Jacobian compilation time")

        timer.stop("Total compilation time")
        if verbose > 1:
            timer.disp("Total compilation time")
        self._compiled = True

    @property
    def constants(self):
        """list: constant parameters for each sub-objective."""
        return [obj.constants for obj in self.objectives]

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

    @property
    def target_scaled(self):
        """ndarray: target vector."""
        target = []
        for obj in self.objectives:
            if obj.target is not None:
                target_i = jnp.ones(obj.dim_f) * obj.target
            else:
                # need to return something, so use midpoint of bounds as approx target
                target_i = jnp.ones(obj.dim_f) * (obj.bounds[0] + obj.bounds[1]) / 2
            target_i = obj._scale(target_i)
            if not obj._normalize_target:
                target_i *= obj.normalization
            target += [target_i]
        return jnp.concatenate(target)

    @property
    def bounds_scaled(self):
        """tuple: lower and upper bounds for residual vector."""
        lb, ub = [], []
        for obj in self.objectives:
            if obj.bounds is not None:
                lb_i = jnp.ones(obj.dim_f) * obj.bounds[0]
                ub_i = jnp.ones(obj.dim_f) * obj.bounds[1]
            else:
                lb_i = jnp.ones(obj.dim_f) * obj.target
                ub_i = jnp.ones(obj.dim_f) * obj.target
            lb_i = obj._scale(lb_i)
            ub_i = obj._scale(ub_i)
            if not obj._normalize_target:
                lb_i *= obj.normalization
                ub_i *= obj.normalization
            lb += [lb_i]
            ub += [ub_i]
        return (jnp.concatenate(lb), jnp.concatenate(ub))

    @property
    def weights(self):
        """ndarray: weight vector."""
        return jnp.concatenate(
            [jnp.ones(obj.dim_f) * obj.weight for obj in self.objectives]
        )


class _Objective(IOAble, ABC):
    """Objective (or constraint) used in the optimization of an Equilibrium.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    name : str, optional
        Name of the objective function.

    """

    _scalar = False
    _linear = False
    _coordinates = ""
    _units = "(Unknown)"
    _equilibrium = False
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
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        name=None,
    ):
        if self._scalar:
            assert self._coordinates == ""
        assert np.all(np.asarray(weight) > 0)
        assert normalize in {True, False}
        assert normalize_target in {True, False}
        assert (bounds is None) or (isinstance(bounds, tuple) and len(bounds) == 2)
        assert (bounds is None) or (target is None), "Cannot use both bounds and target"
        self._target = target
        self._bounds = bounds
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
        self._eq = eq
        if eq is None:
            warnings.warn(
                FutureWarning(
                    "Creating an Objective without specifying the Equilibrium to"
                    " optimize is deprecated, in the future this will raise an error."
                )
            )

    def _set_dimensions(self, eq):
        """Set state vector component dimensions."""
        self._dimensions = {}
        for arg in arg_order:
            self._dimensions[arg] = np.atleast_1d(getattr(eq, arg)).size

    def _set_derivatives(self):
        """Set up derivatives of the objective wrt each argument."""
        self._derivatives = {
            "jac_scaled": {},
            "jac_unscaled": {},
            "grad": {},
            "hess": {},
        }

        for arg in arg_order:
            if arg in self.args:  # derivative wrt arg
                self._derivatives["jac_unscaled"][arg] = Derivative(
                    self.compute_unscaled,
                    argnum=self.args.index(arg),
                    mode="fwd",
                )
                self._derivatives["jac_scaled"][arg] = Derivative(
                    self.compute_scaled,
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
                self._derivatives["jac_unscaled"][
                    arg
                ] = lambda *args, arg=arg, **kwargs: jnp.zeros(
                    (self.dim_f, self.dimensions[arg])
                )
                self._derivatives["jac_scaled"][
                    arg
                ] = lambda *args, arg=arg, **kwargs: jnp.zeros(
                    (self.dim_f, self.dimensions[arg])
                )
                self._derivatives["grad"][
                    arg
                ] = lambda *args, arg=arg, **kwargs: jnp.zeros(
                    (1, self.dimensions[arg])
                )
                self._derivatives["hess"][
                    arg
                ] = lambda *args, arg=arg, **kwargs: jnp.zeros(
                    (self.dimensions[arg], self.dimensions[arg])
                )

    def jit(self):
        """Apply JIT to compute methods, or re-apply after updating self."""
        self._use_jit = True

        try:
            del self.compute_scaled
        except AttributeError:
            pass
        self.compute_scaled = jit(self.compute_scaled)

        try:
            del self.compute_scaled_error
        except AttributeError:
            pass
        self.compute_scaled_error = jit(self.compute_scaled_error)

        try:
            del self.compute_unscaled
        except AttributeError:
            pass
        self.compute_unscaled = jit(self.compute_unscaled)

        try:
            del self.compute_scalar
        except AttributeError:
            pass
        self.compute_scalar = jit(self.compute_scalar)

        del self._derivatives
        self._set_derivatives()
        for mode, val in self._derivatives.items():
            for arg, deriv in val.items():
                self._derivatives[mode][arg] = jit(self._derivatives[mode][arg])

    def _check_dimensions(self):
        """Check that len(target) = len(bounds) = len(weight) = dim_f."""
        if self.bounds is not None:  # must be a tuple of length 2
            self._bounds = tuple([np.asarray(bound) for bound in self._bounds])
            for bound in self.bounds:
                if not is_broadcastable((self.dim_f,), bound.shape):
                    raise ValueError("len(bounds) != dim_f")
            if np.any(self.bounds[1] < self.bounds[0]):
                raise ValueError("bounds must be: (lower bound, upper bound)")
        else:  # target only gets used if bounds is None
            self._target = np.asarray(self._target)
            if not is_broadcastable((self.dim_f,), self.target.shape):
                raise ValueError("len(target) != dim_f")

        self._weight = np.asarray(self._weight)
        if not is_broadcastable((self.dim_f,), self.weight.shape):
            raise ValueError("len(weight) != dim_f")

    @abstractmethod
    def build(self, eq=None, use_jit=True, verbose=1):
        """Build constant arrays."""
        eq = eq or self._eq
        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives()

        # set quadrature weights if they haven't been
        if hasattr(self, "_constants") and ("quad_weights" not in self._constants):
            if self._coordinates == "":
                w = jnp.ones((self.dim_f,))
            elif self._coordinates == "rtz":
                w = self._constants["transforms"]["grid"].weights
                w *= jnp.sqrt(self._constants["transforms"]["grid"].num_nodes)
            elif self._coordinates == "r":
                w = self._constants["transforms"]["grid"].compress(
                    self._constants["transforms"]["grid"].spacing[:, 0],
                    surface_label="rho",
                )
                w = jnp.sqrt(w)
            if w.size:
                w = jnp.tile(w, self.dim_f // w.size)
            self._constants["quad_weights"] = w

        if use_jit is not None:
            self._use_jit = use_jit
        if self._use_jit:
            self.jit()
        self._built = True

    @abstractmethod
    def compute(self, *args, **kwargs):
        """Compute the objective function."""

    def compute_unscaled(self, *args, **kwargs):
        """Compute the raw value of the objective."""
        return jnp.atleast_1d(self.compute(*args, **kwargs))

    def compute_scaled(self, *args, **kwargs):
        """Compute and apply weighting and normalization."""
        f = self.compute(*args, **kwargs)
        return self._scale(f, **kwargs)

    def compute_scaled_error(self, *args, **kwargs):
        """Compute and apply the target/bounds, weighting, and normalization."""
        f = self.compute(*args, **kwargs)
        return self._scale(self._shift(f), **kwargs)

    def _shift(self, f):
        """Subtract target or clamp to bounds."""
        if self.bounds is not None:  # using lower/upper bounds instead of target
            if self._normalize_target:
                bounds = self.bounds
            else:
                bounds = tuple([bound * self.normalization for bound in self.bounds])
            f_target = jnp.where(  # where f is within target bounds, return 0 error
                jnp.logical_and(f >= bounds[0], f <= bounds[1]),
                jnp.zeros_like(f),
                jnp.where(  # otherwise return error = f - bound
                    jnp.abs(f - bounds[0]) < jnp.abs(f - bounds[1]),
                    f - bounds[0],  # errors below lower bound are negative
                    f - bounds[1],  # errors above upper bound are positive
                ),
            )
        else:  # using target instead of lower/upper bounds
            if self._normalize_target:
                target = self.target
            else:
                target = self.target * self.normalization
            f_target = f - target
        return f_target

    def _scale(self, f, *args, **kwargs):
        """Apply weighting, normalization etc."""
        constants = kwargs.get("constants", self.constants)
        if constants is None:
            w = jnp.ones_like(f)
        else:
            w = constants["quad_weights"]
        f_norm = jnp.atleast_1d(f) / self.normalization  # normalization
        return f_norm * w * self.weight

    def compute_scalar(self, *args, **kwargs):
        """Compute the scalar form of the objective."""
        if self.scalar:
            f = self.compute_scaled_error(*args, **kwargs)
        else:
            f = jnp.sum(self.compute_scaled_error(*args, **kwargs) ** 2) / 2
        return f.squeeze()

    def print_value(self, *args, **kwargs):
        """Print the value of the objective."""
        # compute_unscaled is jitted so better to use than than bare compute
        f = self.compute_unscaled(*args, **kwargs)
        if self.linear:
            # probably a Fixed* thing, just need to know norm
            f = jnp.linalg.norm(self._shift(f))
            print(self._print_value_fmt.format(f) + self._units)

        elif self.scalar:
            # dont need min/max/mean of a scalar
            print(self._print_value_fmt.format(f.squeeze()) + self._units)
            if self._normalize and self._units != "(dimensionless)":
                print(
                    self._print_value_fmt.format(self._scale(self._shift(f)).squeeze())
                    + "(normalized error)"
                )

        else:
            # try to do weighted mean if possible
            constants = kwargs.get("constants", self.constants)
            if constants is None:
                w = jnp.ones_like(f)
            else:
                w = constants["quad_weights"]

            # target == 0 probably indicates f is some sort of error metric,
            # mean abs makes more sense than mean
            abserr = jnp.all(self.target == 0)
            f = jnp.abs(f) if abserr else f
            fmax = jnp.max(f)
            fmin = jnp.min(f)
            fmean = jnp.mean(f * w) / jnp.mean(w)

            print(
                "Maximum "
                + ("absolute " if abserr else "")
                + self._print_value_fmt.format(fmax)
                + self._units
            )
            print(
                "Minimum "
                + ("absolute " if abserr else "")
                + self._print_value_fmt.format(fmin)
                + self._units
            )
            print(
                "Average "
                + ("absolute " if abserr else "")
                + self._print_value_fmt.format(fmean)
                + self._units
            )

            if self._normalize and self._units != "(dimensionless)":
                print(
                    "Maximum "
                    + ("absolute " if abserr else "")
                    + self._print_value_fmt.format(fmax / self.normalization)
                    + "(normalized)"
                )
                print(
                    "Minimum "
                    + ("absolute " if abserr else "")
                    + self._print_value_fmt.format(fmin / self.normalization)
                    + "(normalized)"
                )
                print(
                    "Average "
                    + ("absolute " if abserr else "")
                    + self._print_value_fmt.format(fmean / self.normalization)
                    + "(normalized)"
                )

    def xs(self, eq):
        """Return a tuple of args required by this objective from the Equilibrium eq."""
        return tuple(getattr(eq, arg) for arg in self.args)

    def _parse_args(self, *args, **kwargs):
        constants = kwargs.pop("constants", None)
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
        return params, constants

    @property
    def constants(self):
        """dict: Constant parameters such as transforms and profiles."""
        if hasattr(self, "_constants"):
            return self._constants
        return None

    @property
    def target(self):
        """float: Target value(s) of the objective."""
        return self._target

    @target.setter
    def target(self, target):
        self._target = np.atleast_1d(target)
        self._check_dimensions()

    @property
    def bounds(self):
        """tuple: Lower and upper bounds of the objective."""
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        assert (bounds is None) or (isinstance(bounds, tuple) and len(bounds) == 2)
        self._bounds = bounds
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
