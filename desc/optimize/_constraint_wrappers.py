"""Wrappers for doing STELLOPT/SIMSOPT like optimization."""

import numpy as np

from desc.backend import jnp
from desc.compute import arg_order
from desc.objectives import (
    BoundaryRSelfConsistency,
    BoundaryZSelfConsistency,
    ObjectiveFunction,
    get_fixed_boundary_constraints,
    maybe_add_self_consistency,
)
from desc.objectives.utils import align_jacobian, factorize_linear_constraints
from desc.utils import Timer, get_instance

from .utils import compute_jac_scale, f_where_x


class LinearConstraintProjection(ObjectiveFunction):
    """Remove linear constraints via orthogonal projection.

    Given a problem of the form

    min_x f(x)  subject to A*x=b

    We can write any feasible x=xp + Z*x_reduced where xp is a particular solution to
    Ax=b (taken to be the least norm solution), Z is a representation for the null
    space of A (A*Z=0) and x_reduced is unconstrained. This transforms the problem into

    min_x_reduced f(x_reduced)

    Parameters
    ----------
    objective : ObjectiveFunction
        Objective function to optimize.
    constraints : tuple of Objective
        Linear constraints to enforce. Should be a tuple or list of Objective,
        and must all be linear.
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the objectives.
    verbose : int, optional
        Level of output.
    """

    def __init__(self, objective, constraints, eq=None, verbose=1):
        assert isinstance(objective, ObjectiveFunction), (
            "objective should be instance of ObjectiveFunction." ""
        )
        for con in constraints:
            if not con.linear:
                raise ValueError(
                    "LinearConstraintProjection method "
                    + "cannot handle nonlinear constraint {}.".format(con)
                )
        self._objective = objective
        self._objectives = [objective]
        self._constraints = constraints
        self._built = False
        # don't want to compile this, just use the compiled objective
        self._use_jit = False
        self._compiled = False
        self._eq = eq

    def build(self, eq=None, use_jit=None, verbose=1):
        """Build the objective.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
            Note: unused by this class, should pass to sub-objectives directly.
        verbose : int, optional
            Level of output.

        """
        eq = eq or self._eq
        timer = Timer()
        timer.start("Linear constraint projection build")

        # we don't always build here because in ~all cases the user doesn't interact
        # with this directly, so if the user wants to manually rebuild they should
        # do it before this wrapper is created for them.
        if not self._objective.built:
            self._objective.build(eq, verbose=verbose)
        for con in self._constraints:
            if not con.built:
                con.build(eq, verbose=verbose)

        args = np.concatenate([obj.args for obj in self._constraints])
        args = np.concatenate((args, self._objective.args))
        # this is all args used by both constraints and objective
        self._args = [arg for arg in arg_order if arg in args]
        self._dim_f = self._objective.dim_f
        self._dimensions = self._objective.dimensions
        self._scalar = self._objective.scalar
        self._dimensions = self._objective.dimensions
        self._objective.set_args(*self._args)
        self.set_args()
        (
            self._xp,
            self._A,
            self._b,
            self._Z,
            self._unfixed_idx,
            self._project,
            self._recover,
        ) = factorize_linear_constraints(
            self._constraints,
            self._args,
        )
        self._dim_x_full = self._dim_x
        self._dim_x = self._Z.shape[1]

        self._built = True
        timer.stop("Linear constraint projection build")
        if verbose > 1:
            timer.disp("Linear constraint projection build")

    def compile(self, mode="lsq", verbose=1):
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
        self._objective.compile(mode, verbose)

    def project(self, x):
        """Project full vector x into x_reduced that satisfies constraints."""
        return self._project(x)

    def recover(self, x_reduced):
        """Recover the full state vector from the reduced optimization vector."""
        return self._recover(x_reduced)

    def x(self, eq):
        """Return the reduced state vector from the Equilibrium eq."""
        x = np.zeros((self._dim_x_full,))
        for arg in self.args:
            x[self.x_idx[arg]] = getattr(eq, arg)
        return self.project(x)

    def unpack_state(self, x):
        """Unpack the state vector into its components.

        Parameters
        ----------
        x : ndarray
            State vector, either full or reduced.

        Returns
        -------
        kwargs : dict
            Dictionary of the state components with argument names as keys.

        """
        if x.size != self.dim_x:
            raise ValueError(
                "Input vector dimension is invalid, expected "
                + f"{self.dim_x} got {x.size}."
            )
        x = self.recover(x)
        return self._objective.unpack_state(x)

    def compute_unscaled(self, x_reduced, constants=None):
        """Compute the unscaled form of the objective function.

        Parameters
        ----------
        x_reduced : ndarray
            Reduced state vector that satisfies linear constraints.
        constants : list
            Constant parameters passed to sub-objectives.

        Returns
        -------
        f : ndarray
            Objective function value(s).

        """
        x = self.recover(x_reduced)
        f = self._objective.compute_unscaled(x, constants)
        return f

    def compute_scaled(self, x_reduced, constants=None):
        """Compute the objective function and apply weighting / normalization.

        Parameters
        ----------
        x_reduced : ndarray
            Reduced state vector that satisfies linear constraints.
        constants : list
            Constant parameters passed to sub-objectives.

        Returns
        -------
        f : ndarray
            Objective function value(s).

        """
        x = self.recover(x_reduced)
        f = self._objective.compute_scaled(x, constants)
        return f

    def compute_scaled_error(self, x_reduced, constants=None):
        """Compute the objective function and apply weighting / bounds.

        Parameters
        ----------
        x_reduced : ndarray
            Reduced state vector that satisfies linear constraints.
        constants : list
            Constant parameters passed to sub-objectives.

        Returns
        -------
        f : ndarray
            Objective function value(s).

        """
        x = self.recover(x_reduced)
        f = self._objective.compute_scaled_error(x, constants)
        return f

    def compute_scalar(self, x_reduced, constants=None):
        """Compute the scalar form of the objective function.

        Parameters
        ----------
        x_reduced : ndarray
            Reduced state vector that satisfies linear constraints.
        constants : list
            Constant parameters passed to sub-objectives.

        Returns
        -------
        f : float
            Objective function value.

        """
        x = self.recover(x_reduced)
        return self._objective.compute_scalar(x, constants)

    def grad(self, x_reduced, constants=None):
        """Compute gradient of the sum of squares of residuals.

        Parameters
        ----------
        x_reduced : ndarray
            Reduced state vector that satisfies linear constraints.
        constants : list
            Constant parameters passed to sub-objectives.

        Returns
        -------
        g : ndarray
            gradient vector.
        """
        x = self.recover(x_reduced)
        df = self._objective.grad(x, constants)
        return df[self._unfixed_idx] @ self._Z

    def hess(self, x_reduced, constants=None):
        """Compute Hessian of the sum of squares of residuals.

        Parameters
        ----------
        x_reduced : ndarray
            Reduced state vector that satisfies linear constraints.
        constants : list
            Constant parameters passed to sub-objectives.

        Returns
        -------
        H : ndarray
            Hessian matrix.
        """
        x = self.recover(x_reduced)
        df = self._objective.hess(x, constants)
        return self._Z.T @ df[self._unfixed_idx, :][:, self._unfixed_idx] @ self._Z

    def jac_unscaled(self, x_reduced, constants):
        """Compute Jacobian of the vector objective function without weighting / bounds.

        Parameters
        ----------
        x_reduced : ndarray
            Reduced state vector that satisfies linear constraints.
        constants : list
            Constant parameters passed to sub-objectives.

        Returns
        -------
        J : ndarray
            Jacobian matrix.
        """
        x = self.recover(x_reduced)
        df = self._objective.jac_unscaled(x, constants)
        return df[:, self._unfixed_idx] @ self._Z

    def jac_scaled(self, x_reduced, constants=None):
        """Compute Jacobian of the vector objective function with weighting / bounds.

        Parameters
        ----------
        x_reduced : ndarray
            Reduced state vector that satisfies linear constraints.
        constants : list
            Constant parameters passed to sub-objectives.

        Returns
        -------
        J : ndarray
            Jacobian matrix.
        """
        x = self.recover(x_reduced)
        df = self._objective.jac_scaled(x, constants)
        return df[:, self._unfixed_idx] @ self._Z

    def vjp_scaled(self, v, x_reduced):
        """Compute vector-Jacobian product of the objective function.

        Uses the scaled form of the objective.

        Parameters
        ----------
        v : ndarray
            Vector to left-multiply the Jacobian by.
        x_reduced : ndarray
            Optimization variables.

        """
        x = self.recover(x_reduced)
        df = self._objective.vjp_scaled(v, x)
        return df[self._unfixed_idx] @ self._Z

    def vjp_unscaled(self, v, x_reduced):
        """Compute vector-Jacobian product of the objective function.

        Uses the unscaled form of the objective.

        Parameters
        ----------
        v : ndarray
            Vector to left-multiply the Jacobian by.
        x_reduced : ndarray
            Optimization variables.

        """
        x = self.recover(x_reduced)
        df = self._objective.vjp_unscaled(v, x)
        return df[self._unfixed_idx] @ self._Z

    @property
    def constants(self):
        """list: constant parameters for each sub-objective."""
        return self._objective.constants

    @property
    def target_scaled(self):
        """ndarray: target vector."""
        return self._objective.target_scaled

    @property
    def bounds_scaled(self):
        """tuple: lower and upper bounds for residual vector."""
        return self._objective.bounds_scaled

    @property
    def weights(self):
        """ndarray: weight vector."""
        return self._objective.weights


class ProximalProjection(ObjectiveFunction):
    """Remove equilibrium constraint by projecting onto constraint at each step.

    Combines objective and equilibrium constraint into a single objective to then pass
    to an unconstrained optimizer.

    At each iteration, after a step is taken to reduce the objective, the equilibrium
    is perturbed and re-solved to bring it back into force balance. This is analogous
    to a proximal method where each iterate is projected back onto the feasible set.

    Parameters
    ----------
    objective : ObjectiveFunction
        Objective function to optimize.
    constraint : ObjectiveFunction
        Equilibrium constraint to enforce. Should be an ObjectiveFunction with one or
        more of the following objectives: {ForceBalance, CurrentDensity,
        RadialForceBalance, HelicalForceBalance}
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the objectives.
    verbose : int, optional
        Level of output.
    perturb_options, solve_options : dict
        dictionary of arguments passed to Equilibrium.perturb and Equilibrium.solve
        during the projection step.
    """

    def __init__(
        self,
        objective,
        constraint,
        eq=None,
        verbose=1,
        perturb_options=None,
        solve_options=None,
    ):
        assert isinstance(objective, ObjectiveFunction), (
            "objective should be instance of ObjectiveFunction." ""
        )
        assert isinstance(constraint, ObjectiveFunction), (
            "constraint should be instance of ObjectiveFunction." ""
        )
        for con in constraint.objectives:
            if not con._equilibrium:
                raise ValueError(
                    "ProximalProjection method "
                    + "cannot handle general nonlinear constraint {}.".format(con)
                )
        self._objective = objective
        self._constraint = constraint
        solve_options = {} if solve_options is None else solve_options
        perturb_options = {} if perturb_options is None else perturb_options
        self._objectives = [objective, constraint]
        perturb_options.setdefault("verbose", 0)
        perturb_options.setdefault("include_f", False)
        solve_options.setdefault("verbose", 0)
        self._perturb_options = perturb_options
        self._solve_options = solve_options
        self._built = False
        # don't want to compile this, just use the compiled objective and constraint
        self._use_jit = False
        self._compiled = False
        self._eq = eq

    def set_args(self, *args):
        """Set which arguments the objective should expect.

        Defaults to args from all sub-objectives. Additional arguments can be passed in.
        """
        # this is everything taken by either objective
        self._full_args = (
            self._constraint.args
            + self._objective.args
            + list(np.concatenate([obj.args for obj in self._linear_constraints]))
        )
        self._full_args = [arg for arg in arg_order if arg in self._full_args]

        # arguments being optimized are all args, but with internal degrees of freedom
        # replaced by boundary terms
        self._args = self._full_args.copy() + list(args)
        self._args = [arg for arg in arg_order if arg in self._args]
        if "L_lmn" in self._args:
            self._args.remove("L_lmn")
        if "Ra_n" in self._args:
            self._args.remove("Ra_n")
        if "Za_n" in self._args:
            self._args.remove("Za_n")
        if "R_lmn" in self._args:
            self._args.remove("R_lmn")
            if "Rb_lmn" not in self._args:
                self._args.append("Rb_lmn")
        if "Z_lmn" in self._args:
            self._args.remove("Z_lmn")
            if "Zb_lmn" not in self._args:
                self._args.append("Zb_lmn")
        self._set_state_vector()

    def _set_state_vector(self):
        self._dimensions = self._objective.dimensions
        self._dim_x = 0
        self._x_idx = {}
        for arg in self.args:
            self.x_idx[arg] = np.arange(self._dim_x, self._dim_x + self.dimensions[arg])
            self._dim_x += self.dimensions[arg]
        self._dim_x_full = 0
        self._x_idx_full = {}
        for arg in self._full_args:
            self._x_idx_full[arg] = np.arange(
                self._dim_x_full, self._dim_x_full + self.dimensions[arg]
            )
            self._dim_x_full += self.dimensions[arg]

        (
            xp,
            A,
            b,
            self._Z,
            self._unfixed_idx,
            project,
            recover,
        ) = factorize_linear_constraints(self._linear_constraints, self._full_args)

        # dx/dc - goes from the full state to optimization variables
        x_idx = np.concatenate(
            [
                self._x_idx_full[arg]
                for arg in self._args
                if arg not in ["Rb_lmn", "Zb_lmn"]
            ]
        )
        x_idx.sort(kind="mergesort")
        self._dxdc = np.eye(self._dim_x_full)[:, x_idx]
        if "Rb_lmn" in self._args:
            c = get_instance(self._linear_constraints, BoundaryRSelfConsistency)
            A = c.derivatives["jac_unscaled"]["R_lmn"](
                *[jnp.zeros(c.dimensions[arg]) for arg in c.args]
            )
            Ainv = np.linalg.pinv(A)
            dxdRb = np.eye(self._dim_x_full)[:, self._x_idx_full["R_lmn"]] @ Ainv
            self._dxdc = np.hstack((self._dxdc, dxdRb))
        if "Zb_lmn" in self._args:
            c = get_instance(self._linear_constraints, BoundaryZSelfConsistency)
            A = c.derivatives["jac_unscaled"]["Z_lmn"](
                *[jnp.zeros(c.dimensions[arg]) for arg in c.args]
            )
            Ainv = np.linalg.pinv(A)
            dxdZb = np.eye(self._dim_x_full)[:, self._x_idx_full["Z_lmn"]] @ Ainv
            self._dxdc = np.hstack((self._dxdc, dxdZb))

    def build(self, eq=None, use_jit=None, verbose=1):  # noqa: C901
        """Build the objective.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
            Note: unused by this class, should pass to sub-objectives directly.
        verbose : int, optional
            Level of output.

        """
        eq = eq or self._eq
        timer = Timer()
        timer.start("Proximal projection build")

        self._eq = eq.copy()
        self._linear_constraints = get_fixed_boundary_constraints(
            eq=eq,
            iota=self._eq.iota is not None,
            kinetic=self._eq.electron_temperature is not None,
        )
        self._linear_constraints = maybe_add_self_consistency(
            eq, self._linear_constraints
        )

        # we don't always build here because in ~all cases the user doesn't interact
        # with this directly, so if the user wants to manually rebuild they should
        # do it before this wrapper is created for them.
        if not self._objective.built:
            self._objective.build(self._eq, verbose=verbose)
        if not self._constraint.built:
            self._constraint.build(self._eq, verbose=verbose)

        for constraint in self._linear_constraints:
            constraint.build(self._eq, verbose=verbose)

        self._dim_f = self._objective.dim_f
        if self._dim_f == 1:
            self._scalar = True
        else:
            self._scalar = False

        self.set_args()
        self._objective.set_args(*self._full_args)
        self._constraint.set_args(*self._full_args)
        # history and caching
        self._x_old = np.zeros((self._dim_x,))
        for arg in self.args:
            self._x_old[self.x_idx[arg]] = getattr(eq, arg)

        self._allx = [self._x_old]
        self._allxopt = [self._objective.x(eq)]
        self._allxeq = [self._constraint.x(eq)]
        self.history = {}
        for arg in arg_order:
            self.history[arg] = [np.asarray(getattr(self._eq, arg)).copy()]

        self._built = True
        timer.stop("Proximal projection build")
        if verbose > 1:
            timer.disp("Proximal projection build")

    def compile(self, mode="lsq", verbose=1):
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
        self._objective.compile(mode, verbose)
        self._constraint.compile(mode, verbose)

    def _update_equilibrium(self, x, store=False):
        """Update the internal equilibrium with new boundary, profile etc.

        Parameters
        ----------
        x : ndarray
            New values of optimization variables.
        store : bool
            Whether the new x should be stored in self.history

        Notes
        -----
        After updating, if store=False, self._eq will revert back to the previous
        solution when store was True
        """
        # first check if its something we've seen before, if it is just return
        # cached value, no need to perturb + resolve
        xopt = f_where_x(x, self._allx, self._allxopt)
        xeq = f_where_x(x, self._allx, self._allxeq)
        if xopt.size > 0 and xeq.size > 0:
            pass
        else:
            x_dict = self.unpack_state(x)
            x_dict_old = self.unpack_state(self._x_old)
            deltas = {str(key): x_dict[key] - x_dict_old[key] for key in x_dict}
            self._eq = self._eq.perturb(
                objective=self._constraint,
                constraints=self._linear_constraints,
                deltas=deltas,
                **self._perturb_options,
            )
            self._eq.solve(
                objective=self._constraint,
                constraints=self._linear_constraints,
                **self._solve_options,
            )
            xopt = self._objective.x(self._eq)
            xeq = self._constraint.x(self._eq)
            self._allx.append(x)
            self._allxopt.append(xopt)
            self._allxeq.append(xeq)

        if store:
            self._x_old = x
            xd = self.unpack_state(x)
            xod = self._objective.unpack_state(xopt)
            xed = self._constraint.unpack_state(xeq)
            xd.update(xod)
            xd.update(xed)
            for arg in arg_order:
                val = xd.get(arg, self.history[arg][-1])
                self.history[arg] += [np.asarray(val).copy()]
                # ensure eq has correct values if we didn't go into else block above.
                if val.size:
                    setattr(self._eq, arg, val)
            for con in self._linear_constraints:
                if hasattr(con, "update_target"):
                    con.update_target(self._eq)
        else:
            for arg in arg_order:
                val = self.history[arg][-1].copy()
                if val.size:
                    setattr(self._eq, arg, val)
            for con in self._linear_constraints:
                if hasattr(con, "update_target"):
                    con.update_target(self._eq)
        return xopt, xeq

    def compute_scaled(self, x, constants=None):
        """Compute the objective function and apply weights/normalization.

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
        if constants is None:
            constants = self.constants
        xopt, _ = self._update_equilibrium(x, store=False)
        return self._objective.compute_scaled(xopt, constants[0])

    def compute_scaled_error(self, x, constants=None):
        """Compute the error between target and objective and apply weights etc.

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
        if constants is None:
            constants = self.constants
        xopt, _ = self._update_equilibrium(x, store=False)
        return self._objective.compute_scaled_error(xopt, constants[0])

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
        if constants is None:
            constants = self.constants
        xopt, _ = self._update_equilibrium(x, store=False)
        return self._objective.compute_unscaled(xopt, constants[0])

    def grad(self, x, constants=None):
        """Compute gradient of the sum of squares of residuals.

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives.

        Returns
        -------
        g : ndarray
            gradient vector.
        """
        f = jnp.atleast_1d(self.compute_scaled_error(x, constants))
        J = self.jac_scaled(x, constants)
        return f.T @ J

    def jac_unscaled(self, x, constants=None):
        """Compute Jacobian of the vector objective function without weights / bounds.

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives.

        Returns
        -------
        J : ndarray
            Jacobian matrix.
        """
        raise NotImplementedError("Unscaled jacobian of proximal projection is hard.")

    def jac_scaled(self, x, constants=None):
        """Compute Jacobian of the vector objective function with weights / bounds.

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives.

        Returns
        -------
        J : ndarray
            Jacobian matrix.
        """
        if constants is None:
            constants = self.constants
        xg, xf = self._update_equilibrium(x, store=True)

        # Jacobian matrices wrt combined state vectors
        Fx = self._constraint.jac_scaled(xf, constants[1])
        Gx = self._objective.jac_scaled(xg, constants[0])
        Fx = align_jacobian(Fx, self._constraint, self._objective)
        Gx = align_jacobian(Gx, self._objective, self._constraint)

        # projections onto optimization space
        # possibly better way: Gx @ np.eye(Gx.shape[1])[:,self._unfixed_idx] @ self._Z
        Fx_reduced = Fx[:, self._unfixed_idx] @ self._Z
        Gx_reduced = Gx[:, self._unfixed_idx] @ self._Z
        Fc = Fx @ self._dxdc
        Gc = Gx @ self._dxdc

        # some scaling to improve conditioning
        wf, _ = compute_jac_scale(Fx_reduced)
        wg, _ = compute_jac_scale(Gx_reduced)
        wx = wf + wg
        Fxh = Fx_reduced * wx
        Gxh = Gx_reduced * wx

        cutoff = np.finfo(Fxh.dtype).eps * np.max(Fxh.shape)
        uf, sf, vtf = np.linalg.svd(Fxh, full_matrices=False)
        sf += sf[-1]  # add a tiny bit of regularization
        sfi = np.where(sf < cutoff * sf[0], 0, 1 / sf)
        Fxh_inv = vtf.T @ (sfi[..., np.newaxis] * uf.T)

        # TODO: make this more efficient for finite differences etc. Can probably
        # reduce the number of operations and tangents
        LHS = Gxh @ (Fxh_inv @ Fc) - Gc
        return -LHS

    def hess(self, x, constants=None):
        """Compute Hessian of the sum of squares of residuals.

        Uses the "small residual approximation" where the Hessian is replaced by
        the square of the Jacobian: H = J.T @ J

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives.

        Returns
        -------
        H : ndarray
            Hessian matrix.
        """
        J = self.jac_scaled(x, constants)
        return J.T @ J

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
        raise NotImplementedError

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
        raise NotImplementedError

    @property
    def constants(self):
        """list: constant parameters for each sub-objective."""
        return [self._objective.constants, self._constraint.constants]

    @property
    def target_scaled(self):
        """ndarray: target vector."""
        return self._objective.target_scaled

    @property
    def bounds_scaled(self):
        """tuple: lower and upper bounds for residual vector."""
        return self._objective.bounds_scaled

    @property
    def weights(self):
        """ndarray: weight vector."""
        return self._objective.weights
