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
        # need compiled=True to avoid calling objective.compile which calls
        # compute with all zeros, leading to error in perturb/resolve
        self._compiled = True

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
            and self._eq.iota is not None
        )

        self._objective.build(self._eq, verbose=verbose)
        self._constraint.build(self._eq, verbose=verbose)
        for constraint in self._linear_constraints:
            constraint.build(self._eq, verbose=verbose)
        self._objectives = self._objective.objectives

        self._dim_f = self._objective.dim_f
        if self._dim_f == 1:
            self._scalar = True
        else:
            self._scalar = False

        # set_state_vector
        self._args = ["p_l", "i_l", "c_l", "Psi", "Rb_lmn", "Zb_lmn"]
        if isinstance(self._constraint.objectives[0], CurrentDensity):
            self._args.remove("p_l")
        self._dimensions = self._objective.dimensions
        self._dim_x = 0
        self._x_idx = {}
        for arg in self.args:
            self.x_idx[arg] = np.arange(self._dim_x, self._dim_x + self.dimensions[arg])
            self._dim_x += self.dimensions[arg]

        self._full_args = np.concatenate((self.args, self._constraint.args))
        self._full_args = [arg for arg in arg_order if arg in self._full_args]

        (
            xp,
            A,
            self._Ainv,
            b,
            self._Z,
            self._unfixed_idx,
            project,
            recover,
        ) = factorize_linear_constraints(
            self._linear_constraints, self._constraint.args
        )

        self._x_old = np.zeros((self._dim_x,))
        for arg in self.args:
            self._x_old[self.x_idx[arg]] = getattr(eq, arg)

        self._allx = [self._x_old]
        self.history = {}
        for arg in self._full_args:
            self.history[arg] = [np.asarray(getattr(self._eq, arg)).copy()]

        self._built = True

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
        if jnp.allclose(x, self._x_old, rtol=1e-14, atol=1e-14):
            pass
        else:
            x_dict = self.unpack_state(x)
            x_dict_old = self.unpack_state(self._x_old)
            deltas = {
                "d" + str(key).split("_")[0]: x_dict[key] - x_dict_old[key]
                for key in x_dict
            }
            self._eq = self._eq.perturb(
                objective=self._constraint,
                constraints=self._linear_constraints,
                **deltas,
                **self._perturb_options
            )
            self._eq.solve(
                objective=self._constraint,
                constraints=self._linear_constraints,
                **self._solve_options
            )

        xopt = self._objective.x(self._eq)
        xeq = self._constraint.x(self._eq)
        if store:
            self._x_old = x
            self._allx.append(x)
            for arg in self._full_args:
                self.history[arg] += [np.asarray(getattr(self._eq, arg)).copy()]
        else:
            for arg in self._full_args:
                val = self.history[arg][-1].copy()
                if val.size:
                    setattr(self._eq, arg, val)
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

        # dx/dc
        x_idx = np.concatenate(
            [
                self._constraint.x_idx[arg]
                for arg in ["p_l", "i_l", "c_l", "Psi"]
                if arg in self._constraint.args
            ]
        )
        x_idx.sort(kind="mergesort")
        dxdc = np.eye(self._constraint.dim_x)[:, x_idx]
        dxdRb = (
            np.eye(self._constraint.dim_x)[:, self._constraint.x_idx["R_lmn"]]
            @ self._Ainv["R_lmn"]
        )
        dxdc = np.hstack((dxdc, dxdRb))
        dxdZb = (
            np.eye(self._constraint.dim_x)[:, self._constraint.x_idx["Z_lmn"]]
            @ self._Ainv["Z_lmn"]
        )
        dxdc = np.hstack((dxdc, dxdZb))

        # Jacobian matrices wrt combined state vectors
        Fx = self._constraint.jac(xf)
        Gx = self._objective.jac(xg)
        Fx = align_jacobian(Fx, self._constraint, self._objective)
        Gx = align_jacobian(Gx, self._objective, self._constraint)
        # possibly better way: Gx @ np.eye(Gx.shape[1])[:,self._unfixed_idx] @ self._Z
        Fx_reduced = Fx[:, self._unfixed_idx] @ self._Z
        Gx_reduced = Gx[:, self._unfixed_idx] @ self._Z

        Fc = Fx @ dxdc
        Fx_reduced_inv = jnp.linalg.pinv(Fx_reduced, rcond=1e-6)

        Gc = Gx @ dxdc
        # TODO: make this more efficient for finite differences etc. Can probably
        # reduce the number of operations and tangents
        LHS = Gx_reduced @ (Fx_reduced_inv @ Fc) - Gc
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
            self._multipliers = np.zeros(self._constraint.dim_f)
        self._multipliers = np.atleast_1d(self._multipliers)
        assert (
            self._multipliers.ndim == 1
            and len(self._multipliers) == self._constraint.dim_f
        )
        self._dim_f = self._objective.dim_f + self._constraint.dim_f

        self._set_state_vector()
        self._set_derivatives()
        if self.use_jit:
            self.jit()
        self._scalar = False

        self._built = True

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
