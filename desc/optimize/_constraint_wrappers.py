"""Wrappers for doing STELLOPT/SIMSOPT like optimization."""

import numpy as np

from desc.backend import jnp
from desc.compute import arg_order
from desc.derivatives import Derivative
from desc.objectives import (
    CurrentDensity,
    ForceBalance,
    HelicalForceBalance,
    ObjectiveFunction,
    RadialForceBalance,
)
from desc.objectives.utils import (
    align_jacobian,
    factorize_linear_constraints,
    get_fixed_boundary_constraints,
)

from .utils import compute_jac_scale, f_where_x


class ProximalProjection(ObjectiveFunction):
    """Create a proximal projection operator for equilibirum constraints.

    Combines objective and constraints into a single objective to then pass to an
    unconstrained optimizer.

    At each iteration, after a step is taken to reduce the objective, the equilibrium
    is perturbed and re-solved to bring it back into force balance. This is analagous
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
        perturb_options={},
        solve_options={},
    ):
        assert isinstance(objective, ObjectiveFunction), (
            "objective should be instance of ObjectiveFunction." ""
        )
        assert isinstance(constraint, ObjectiveFunction), (
            "constraint should be instance of ObjectiveFunction." ""
        )
        for con in constraint.objectives:
            if not isinstance(
                con,
                (
                    ForceBalance,
                    RadialForceBalance,
                    HelicalForceBalance,
                    CurrentDensity,
                ),
            ):
                raise ValueError(
                    "ProximalProjection method "
                    + "cannot handle general nonlinear constraint {}.".format(con)
                )
        self._objective = objective
        self._constraint = constraint
        perturb_options.setdefault("verbose", 0)
        solve_options.setdefault("verbose", 0)
        self._perturb_options = perturb_options
        self._solve_options = solve_options
        self._built = False
        # don't want to compile this, just use the compiled objective and constraint
        self._use_jit = False
        self._compiled = False

        if eq is not None:
            self.build(eq, verbose=verbose)

    # TODO: add timing and verbose statements
    def build(self, eq, use_jit=None, verbose=1):
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
        self._eq = eq.copy()
        self._linear_constraints = get_fixed_boundary_constraints(
            iota=not isinstance(self._constraint.objectives[0], CurrentDensity)
            and self._eq.iota is not None,
            kinetic=eq.electron_temperature is not None,
        )

        self._objective.build(self._eq, verbose=verbose)
        self._constraint.build(self._eq, verbose=verbose)
        # remove constraints that aren't necessary
        self._linear_constraints = tuple(
            [
                con
                for con in self._linear_constraints
                if con.args[0] in self._constraint.args
            ]
        )
        for constraint in self._linear_constraints:
            constraint.build(self._eq, verbose=verbose)
        self._objectives = self._objective.objectives

        self._dim_f = self._objective.dim_f
        if self._dim_f == 1:
            self._scalar = True
        else:
            self._scalar = False

        # this is everything taken by either objective
        self._full_args = self._constraint.args + self._objective.args
        self._full_args = [arg for arg in arg_order if arg in self._full_args]

        # arguments being optimized are all args, but with internal degrees of freedom
        # replaced by boundary terms
        self._args = self._full_args.copy()
        if "L_lmn" in self._args:
            self._args.remove("L_lmn")
        if "R_lmn" in self._args:
            self._args.remove("R_lmn")
            if "Rb_lmn" not in self._args:
                self._args.append("Rb_lmn")
        if "Z_lmn" in self._args:
            self._args.remove("Z_lmn")
            if "Zb_lmn" not in self._args:
                self._args.append("Zb_lmn")

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
            self._Ainv,
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
            dxdRb = (
                np.eye(self._dim_x_full)[:, self._x_idx_full["R_lmn"]]
                @ self._Ainv["R_lmn"]
            )
            self._dxdc = np.hstack((self._dxdc, dxdRb))
        if "Zb_lmn" in self._args:
            dxdZb = (
                np.eye(self._dim_x_full)[:, self._x_idx_full["Z_lmn"]]
                @ self._Ainv["Z_lmn"]
            )
            self._dxdc = np.hstack((self._dxdc, dxdZb))

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
                **self._perturb_options
            )
            self._eq.solve(
                objective=self._constraint,
                constraints=self._linear_constraints,
                **self._solve_options
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
            for arg in self._full_args:
                val = xd[arg]
                self.history[arg] += [np.asarray(val).copy()]
                # ensure eq has correct values if we didn't go into else block above.
                if val.size:
                    setattr(self._eq, arg, val)
            for con in self._linear_constraints:
                con.update_target(self._eq)
        else:
            for arg in self._full_args:
                val = self.history[arg][-1].copy()
                if val.size:
                    setattr(self._eq, arg, val)
            for con in self._linear_constraints:
                con.update_target(self._eq)
        return xopt, xeq

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
        xopt, _ = self._update_equilibrium(x, store=False)
        return self._objective.compute(xopt)

    def grad(self, x):
        """Compute gradient of the sum of squares of residuals.

        Parameters
        ----------
        x : ndarray
            State vector.

        Returns
        -------
        g : ndarray
            gradient vector.
        """
        f = jnp.atleast_1d(self.compute(x))
        J = self.jac(x)
        return f.T @ J

    def jac(self, x):
        """Compute Jacobian of the vector objective function.

        Parameters
        ----------
        x : ndarray
            State vector.

        Returns
        -------
        J : ndarray
            Jacobian matrix.
        """
        xg, xf = self._update_equilibrium(x, store=True)

        # Jacobian matrices wrt combined state vectors
        Fx = self._constraint.jac(xf)
        Gx = self._objective.jac(xg)
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

    def hess(self, x):
        """Compute Hessian of the sum of squares of residuals.

        Uses the "small residual approximation" where the Hessian is replaced by
        the square of the Jacobian: H = J.T @ J

        Parameters
        ----------
        x : ndarray
            State vector.

        Returns
        -------
        H : ndarray
            Hessian matrix.
        """
        J = self.jac(x)
        return J.T @ J


class Lagrangian(ObjectiveFunction):
    """Create Lagrangian function for objective + constraints.

    Combines objective and constraints and lagrange multipliers
    into a single objective to then pass to an unconstrained
    optimizer.

    Parameters
    ----------
    objective : ObjectiveFunction
        Objective to minimize.
    constraint : ObjectiveFunction
        Constraint to enforce.
    multipliers : ndarray, optional
        initial guess for lagrange multipliers. Defaults to zeros
        Must be equal in length to constraint.dim_f
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the objectives.
    use_jit : bool
        whether to jit compile compute methods
    verbose : int, optional
        Level of output.

    Notes
    -----
    Optimality conditions for the Lagrangian happen at a saddle point,
    meaning we cannot minimize the Lagrangian directly. Instead,
    we seek the minimum of the gradient squared. Lagrangian.compute_scalar,
    Lagrangian.grad, and Lagrangian.hess return values the squared gradient
    and its derivatives rather than the actual Lagrangian value.
    Lagrangian.compute and Lagrangian.jac return the gradient and hessian
    of the scalar lagrangian for use in a least squares or root finding routine.
    To compute the value of the Lagrangian, use
    Lagrangian.compute_lagrangian.
    """

    def __init__(
        self, objective, constraint, multipliers=None, eq=None, use_jit=True, verbose=1
    ):

        assert isinstance(objective, ObjectiveFunction), (
            "objective should be instance of ObjectiveFunction." ""
        )
        assert isinstance(constraint, ObjectiveFunction), (
            "constraint should be instance of ObjectiveFunction." ""
        )
        self._objective = objective
        self._constraint = constraint
        self._multipliers = multipliers

        assert use_jit in {True, False}
        self._use_jit = use_jit
        self._built = False
        self._compiled = False

        if eq is not None:
            self.build(eq, use_jit=self._use_jit, verbose=verbose)

    def _set_state_vector(self):
        """Set state vector components, dimensions, and indices."""
        self._args = np.concatenate([self._objective.args, self._constraint.args])
        self._args = [arg for arg in arg_order if arg in self._args]
        self._args.append("multipliers")
        self._dimensions = self._objective.dimensions
        self._dimensions["multipliers"] = self._multipliers.size

        self._dim_x = 0
        self._x_idx = {}
        for arg in self.args:
            self.x_idx[arg] = np.arange(self._dim_x, self._dim_x + self.dimensions[arg])
            self._dim_x += self.dimensions[arg]

    def _set_derivatives(self):
        self._lagrangian_grad = Derivative(self.compute_lagrangian, mode="grad")
        self._lagrangian_hess = Derivative(self.compute_lagrangian, mode="hess")
        self._jac = self._lagrangian_hess
        self._grad = Derivative(self.compute_scalar, mode="grad")
        self._hess = Derivative(self.compute_scalar, mode="hess")

    def _x_to_xo_xc(self, x):
        """Unpack full x into sub-x taken by objective and constraints."""
        kwargs = self.unpack_state(x)
        xo = jnp.concatenate([kwargs[arg] for arg in self._objective.args])
        xc = jnp.concatenate([kwargs[arg] for arg in self._constraint.args])
        mu = kwargs["multipliers"]
        return xo, xc, mu

    def x(self, eq):
        """Return the full state vector from the Equilibrium eq."""
        xo = self._objective.unpack_state(self._objective.x(eq))
        xc = self._constraint.unpack_state(self._constraint.x(eq))
        xo.update(xc)
        xo["multipliers"] = self._multipliers
        x = jnp.concatenate([xo[arg] for arg in self.args])
        return x

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
        if use_jit is not None:
            self._use_jit = use_jit

        self._objective.build(eq, use_jit, verbose)
        self._constraint.build(eq, use_jit, verbose)
        if self._multipliers is None:
            self._multipliers = self._estimate_lagrange_multipliers(eq)
        self._multipliers = np.atleast_1d(self._multipliers)
        assert (
            self._multipliers.ndim == 1
            and len(self._multipliers) == self._constraint.dim_f
        )
        self._set_state_vector()
        self._dim_f = self._dim_x
        self._set_derivatives()
        if self.use_jit:
            self.jit()
        self._scalar = False

        self._built = True

    def _estimate_lagrange_multipliers(self, eq):
        A = self._constraint.jac(self._constraint.x(eq))
        g = self._objective.grad(self._objective.x(eq))
        A = align_jacobian(A, self._constraint, self._objective)
        g = align_jacobian(jnp.atleast_2d(g), self._objective, self._constraint)[0]
        return jnp.linalg.lstsq(A.T, g, rcond=None)[0]

    def compute_lagrangian(self, x):
        """Compute the value of the Lagrangian."""
        xo, xc, mu = self._x_to_xo_xc(x)
        fo = self._objective.compute_scalar(xo)
        fc = self._constraint.compute(xc)
        return fo + jnp.dot(mu, fc)

    def compute_scalar(self, x):
        """Compute the scalar form of the objective = norm**2 of grad of Lagrangian."""
        return 1 / 2 * jnp.sum(self.compute(x) ** 2)

    def compute(self, x):
        """Compute the vector form of the objective = grad of Langrangian."""
        return self._lagrangian_grad(x)
