"""Wrappers for doing STELLOPT/SIMSOPT like optimization."""

import numpy as np

from desc.backend import jnp
from desc.compute import arg_order

from ._equilibrium import CurrentDensity
from .objective_funs import ObjectiveFunction
from .utils import (
    factorize_linear_constraints,
    get_equilibrium_objective,
    get_fixed_boundary_constraints,
)


class WrappedEquilibriumObjective(ObjectiveFunction):
    """Evaluate an objective subject to equilibrium constraint.

    Parameters
    ----------
    objective : ObjectiveFunction
        Objective function to optimize.
    eq_objective : ObjectiveFunction
        Equilibrium objective to enforce.
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the objectives.
    verbose : int, optional
        Level of output.

    """

    def __init__(
        self,
        objective,
        eq_objective=None,
        eq=None,
        verbose=1,
        perturb_options={},
        solve_options={},
    ):

        self._objective = objective
        self._eq_objective = eq_objective
        self._perturb_options = perturb_options
        self._solve_options = solve_options
        self._built = False
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
        if self._eq_objective is None:
            self._eq_objective = get_equilibrium_objective()
        self._constraints = get_fixed_boundary_constraints(
            iota=not isinstance(self._eq_objective.objectives[0], CurrentDensity)
            and self._eq.iota is not None
        )

        self._objective.build(self._eq, verbose=verbose)
        self._eq_objective.build(self._eq, verbose=verbose)
        for constraint in self._constraints:
            constraint.build(self._eq, verbose=verbose)
        self._objectives = self._objective.objectives

        self._dim_f = self._objective.dim_f
        if self._dim_f == 1:
            self._scalar = True
        else:
            self._scalar = False

        # set_state_vector
        self._args = ["p_l", "i_l", "c_l", "Psi", "Rb_lmn", "Zb_lmn"]
        if isinstance(self._eq_objective.objectives[0], CurrentDensity):
            self._args.remove("p_l")
        self._dimensions = self._objective.dimensions
        self._dim_x = 0
        self._x_idx = {}
        for arg in self.args:
            self.x_idx[arg] = np.arange(self._dim_x, self._dim_x + self.dimensions[arg])
            self._dim_x += self.dimensions[arg]

        self._full_args = np.concatenate((self.args, self._eq_objective.args))
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
        ) = factorize_linear_constraints(self._constraints, self._eq_objective.args)

        self._x_old = np.zeros((self._dim_x,))
        for arg in self.args:
            self._x_old[self.x_idx[arg]] = getattr(eq, arg)

        self._allx = [self._x_old]
        self.history = {}
        for arg in self._full_args:
            self.history[arg] = [np.asarray(getattr(self._eq, arg)).copy()]

        self._built = True

    def _update_equilibrium(self, x):
        """Update the internal equilibrium with new boundary, profile etc."""
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
                objective=self._eq_objective,
                constraints=self._constraints,
                **deltas,
                **self._perturb_options
            )
            self._eq.solve(
                objective=self._eq_objective,
                constraints=self._constraints,
                **self._solve_options
            )
            self._x_old = x
            self._allx.append(x)

            for arg in self._full_args:
                self.history[arg] += [np.asarray(getattr(self._eq, arg)).copy()]

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
        self._update_equilibrium(x)
        x_obj = self._objective.x(self._eq)
        return self._objective.compute(x_obj)

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
        self._update_equilibrium(x)

        # dx/dc
        x_idx = np.concatenate(
            [
                self._eq_objective.x_idx[arg]
                for arg in ["p_l", "i_l", "c_l", "Psi"]
                if arg in self._eq_objective.args
            ]
        )
        x_idx.sort(kind="mergesort")
        dxdc = np.eye(self._eq_objective.dim_x)[:, x_idx]
        dxdRb = (
            np.eye(self._eq_objective.dim_x)[:, self._eq_objective.x_idx["R_lmn"]]
            @ self._Ainv["R_lmn"]
        )
        dxdc = np.hstack((dxdc, dxdRb))
        dxdZb = (
            np.eye(self._eq_objective.dim_x)[:, self._eq_objective.x_idx["Z_lmn"]]
            @ self._Ainv["Z_lmn"]
        )
        dxdc = np.hstack((dxdc, dxdZb))

        # state vectors
        xf = self._eq_objective.x(self._eq)
        xg = self._objective.x(self._eq)

        # Jacobian matrices wrt combined state vectors
        Fx = self._eq_objective.jac(xf)
        Gx = self._objective.jac(xg)
        Fx = {
            arg: Fx[:, self._eq_objective.x_idx[arg]] for arg in self._eq_objective.args
        }
        Gx = {arg: Gx[:, self._objective.x_idx[arg]] for arg in self._objective.args}
        for arg in self._eq_objective.args:
            if arg not in Fx.keys():
                Fx[arg] = jnp.zeros((self._eq_objective.dim_f, self.dimensions[arg]))
            if arg not in Gx.keys():
                Gx[arg] = jnp.zeros((self._objective.dim_f, self.dimensions[arg]))
        Fx = jnp.hstack([Fx[arg] for arg in arg_order if arg in Fx])
        Gx = jnp.hstack([Gx[arg] for arg in arg_order if arg in Gx])
        Fx_reduced = Fx[:, self._unfixed_idx] @ self._Z
        Gx_reduced = Gx[:, self._unfixed_idx] @ self._Z

        Fc = Fx @ dxdc
        Fx_reduced_inv = jnp.linalg.pinv(Fx_reduced, rcond=1e-6)

        Gc = Gx @ dxdc

        GxFx = Gx_reduced @ Fx_reduced_inv
        LHS = GxFx @ Fc - Gc
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
