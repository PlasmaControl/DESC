import numpy as np
from abc import ABC, abstractmethod
from inspect import getfullargspec


from desc.backend import use_jax, jnp, jit
from desc.utils import Timer
from desc.io import IOAble
from desc.derivatives import Derivative
from desc.compute import arg_order

# XXX: could use `indicies` instead of `arg_order` in ObjectiveFunction loops


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
    verbose : int, optional
        Level of output.

    """

    _io_attrs_ = ["_objectives"]

    def __init__(self, objectives, eq=None, use_jit=True, verbose=1):

        if not isinstance(objectives, tuple):
            objectives = (objectives,)

        self._objectives = objectives
        self._use_jit = use_jit
        self._built = False
        self._compiled = False

        if eq is not None:
            self.build(eq, use_jit=self._use_jit, verbose=verbose)

    def _set_state_vector(self):
        """Set state vector components, dimensions, and indicies."""
        self._args = np.concatenate([obj.args for obj in self.objectives])
        self._args = [arg for arg in arg_order if arg in self._args]

        self._dimensions = self.objectives[0].dimensions

        self._dim_x = 0
        self._x_idx = {}
        for arg in self.args:
            self.x_idx[arg] = np.arange(self._dim_x, self._dim_x + self.dimensions[arg])
            self._dim_x += self.dimensions[arg]

    def _set_derivatives(self, use_jit=True):
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
        )
        self._jac = Derivative(
            self.compute,
            mode="fwd",
            use_jit=use_jit,
        )

        if use_jit:
            self.compute = jit(self.compute)
            self.compute_scalar = jit(self.compute_scalar)
            # TODO: add jit for jac, hess, jvp, etc.
            # then can remove jit from Derivatives class

    def build(self, eq, use_jit=True, verbose=1):
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
        self._use_jit = use_jit
        timer = Timer()
        timer.start("Objecive build")

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

        # build linear constraint matrices

        self._set_derivatives(self.use_jit)

        self._built = True
        timer.stop("Objecive build")
        if verbose > 1:
            timer.disp("Objecive build")

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
        f = jnp.concatenate([obj.compute(**kwargs) for obj in self.objectives])
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

    def callback(self, x):
        """Print the value(s) of the objective.

        Parameters
        ----------
        x : ndarray
            State vector.

        """
        f = self.compute_scalar(x)
        print("Total (sum of squares): {:10.3e}, ".format(f))
        kwargs = self.unpack_state(x)
        for obj in self.objectives:
            obj.callback(**kwargs)
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
            kwargs[arg] = x[self.x_idx[arg]]
        return kwargs

    def x(self, eq):
        """Return the full state vector from the Equilibrium eq."""
        x = np.zeros((self.dim_x,))
        for arg in self.args:
            x[self.x_idx[arg]] = getattr(eq, arg)
        return x

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
        J = self._jac.compute(x)
        return jnp.atleast_2d(J)

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

        # variable values are irrelevant for compilation
        x = np.zeros((self.dim_x,))

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
            J0 = self.jac(x).block_until_ready()
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
    def dimensions(self):
        """dict: Dimensions of the argument given by the dict keys."""
        return self._dimensions

    @property
    def x_idx(self):
        """dict: Indicies of the components of the state vector."""
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

    _io_attrs_ = ["_target", "_weight", "_name"]

    def __init__(self, eq=None, target=0, weight=1, name=None):

        assert np.all(np.asarray(weight) > 0)
        self._target = np.atleast_1d(target)
        self._weight = np.atleast_1d(weight)
        self._name = name
        self._built = False

        if eq is not None:
            self.build(eq)

    def _set_dimensions(self, eq):
        """Set state vector component dimensions."""
        self._dimensions = {}
        self._dimensions["R_lmn"] = eq.R_basis.num_modes
        self._dimensions["Z_lmn"] = eq.Z_basis.num_modes
        self._dimensions["L_lmn"] = eq.L_basis.num_modes
        self._dimensions["p_l"] = eq.pressure.params.size
        self._dimensions["i_l"] = eq.iota.params.size
        self._dimensions["Psi"] = 1
        self._dimensions["Rb_lmn"] = eq.surface.R_basis.num_modes
        self._dimensions["Zb_lmn"] = eq.surface.Z_basis.num_modes

    def _set_derivatives(self, use_jit=True):
        """Set up derivatives of the objective wrt each argument."""
        self._derivatives = {}
        self._scalar_derivatives = {}
        self._args = [arg for arg in getfullargspec(self.compute)[0] if arg != "self"]

        # only used for linear objectives so variable values are irrelevant
        kwargs = dict(  # FIXME: need to use dim_x
            [(arg, np.zeros((self.dimensions[arg],))) for arg in self.dimensions.keys()]
        )
        args = [kwargs[arg] for arg in self.args]

        # constant derivatives are pre-computed, otherwise set up Derivative instance
        for arg in arg_order:
            if arg in self.args:  # derivative wrt arg
                self._derivatives[arg] = Derivative(
                    self.compute,
                    argnum=self.args.index(arg),
                    mode="fwd",
                    use_jit=use_jit,
                )
                self._scalar_derivatives[arg] = Derivative(
                    self.compute_scalar,
                    argnum=self.args.index(arg),
                    mode="fwd",
                    use_jit=use_jit,
                )
                if self.linear:  # linear objectives have constant derivatives
                    self._derivatives[arg] = self._derivatives[arg].compute(*args)
                    self._scalar_derivatives[arg] = self._scalar_derivatives[
                        arg
                    ].compute(*args)
            else:  # these derivatives are always zero
                self._derivatives[arg] = np.zeros((self.dim_f, self.dimensions[arg]))
                self._scalar_derivatives[arg] = np.zeros((1, self.dimensions[arg]))

        if use_jit:
            self.compute = jit(self.compute)
            self.compute_scalar = jit(self.compute_scalar)

    def _check_dimensions(self):
        """Check that len(target) = len(weight) = dim_f."""
        if np.unique(self.target).size == 1:
            self._target = np.repeat(self.target[0], self.dim_f)
        if np.unique(self.weight).size == 1:
            self._weight = np.repeat(self.weight[0], self.dim_f)

        if self.target.size != self.dim_f:
            raise ValueError("len(target) != dim_f")
        if self.weight.size != self.dim_f:
            raise ValueError("len(weight) != dim_f")

        return None

    def update_target(self, eq):
        """Update target values using an Equilibrium.

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium that will be optimized to satisfy the Objective.

        """
        self.target = np.atleast_1d(getattr(eq, self.target_arg, self.target))

    @abstractmethod
    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays."""

    @abstractmethod
    def compute(self, *args, **kwargs):
        """Compute the objective function."""

    def compute_scalar(self, *args, **kwargs):
        """Compute the scalar form of the objective."""
        if self.scalar:
            f = self.compute(*args, **kwargs)
        else:
            f = jnp.sum(self.compute(*args, **kwargs) ** 2) / 2
        return f

    def callback(self, *args, **kwargs):
        """Print the value of the objective."""
        x = self._unshift_unscale(self.compute(*args, **kwargs))
        print(self._callback_fmt.format(jnp.linalg.norm(x)))

    def _shift_scale(self, x):
        """Apply target and weighting."""
        return (x - self.target) * self.weight

    def _unshift_unscale(self, x):
        """Undo target and weighting."""
        return x / self.weight + self.target

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
        """bool: Whether default "compute" method is a scalar (or vector)."""
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
