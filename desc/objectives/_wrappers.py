import numpy as np

from desc.backend import jnp
from desc.compute import arg_order
from .objective_funs import ObjectiveFunction
from .utils import (
    get_equilibrium_objective,
    get_fixed_boundary_constraints,
    factorize_linear_constraints,
)


class WrappedEquilibriumObjective(ObjectiveFunction):
    """Evaluate an objective subject to equilibrium constraint."""

    def __init__(
        self,
        objective,
        eq_objective=None,
        eq=None,
        use_jit=True,
        verbose=1,
        perturb_options={},
        solve_options={},
    ):
        """Initialize a WrappedEquilibriumObjective.

        Parameters
        ----------
        objective : ObjectiveFunction
            Objective function to optimize.
        eq_objective : ObjectiveFunction
            Equilibrium objective to enforce.
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the objectives.
        use_jit : bool, optional
            Whether to just-in-time compile the objectives and derivatives.
        verbose : int, optional
            Level of output.

        """
        self._objective = objective
        self._eq_objective = eq_objective
        self._perturb_options = perturb_options
        self._solve_options = solve_options
        self._use_jit = use_jit
        self._built = False
        self._compiled = True

        if eq is not None:
            self.build(eq, use_jit=self._use_jit, verbose=verbose)

    # TODO: add timing and verbose statements
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
        self._eq = eq.copy()
        if self._eq_objective is None:
            self._eq_objective = get_equilibrium_objective()
        self._constraints = get_fixed_boundary_constraints()

        self._objective.build(self._eq, use_jit=self.use_jit, verbose=verbose)
        self._eq_objective.build(self._eq, use_jit=self.use_jit, verbose=verbose)
        for constraint in self._constraints:
            constraint.build(self._eq, use_jit=self.use_jit, verbose=verbose)
        self._objectives = self._objective.objectives

        self._dim_f = self._objective.dim_f
        if self._dim_f == 1:
            self._scalar = True
        else:
            self._scalar = False

        # set_state_vector
        self._args = ["p_l", "i_l", "Psi", "Rb_lmn", "Zb_lmn"]
        self._dimensions = self._objective.dimensions
        self._dim_x = 0
        self._x_idx = {}
        for arg in self.args:
            self.x_idx[arg] = np.arange(self._dim_x, self._dim_x + self.dimensions[arg])
            self._dim_x += self.dimensions[arg]

        (
            xp,
            A,
            self._Ainv,
            b,
            self._Z,
            self._unfixed_idx,
            project,
            recover,
        ) = factorize_linear_constraints(self._constraints)

        self._x_old = np.zeros((self._dim_x,))
        for arg in self.args:
            self._x_old[self.x_idx[arg]] = getattr(eq, arg)

        self._built = True

    def _update_equilibrium(self, x):
        """Description."""
        if jnp.all(x == self._x_old):
            pass
        else:
            x_dict = self.unpack_state(x)
            x_dict_old = self.unpack_state(self._x_old)
            deltas = {
                "d" + str(key).split("_")[0]: x_dict[key] - x_dict_old[key]
                for key in x_dict
            }
            self._eq.perturb(
                self._eq_objective, self._constraints, **deltas, **self._perturb_options
            )
            self._eq.solve(objective=self._eq_objective, **self._solve_options)
            self._x_old = x

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

        f = jnp.atleast_1d(self.compute(x))
        J = self.jac(x)
        return f.T @ J

    def jac(self, x):

        self._update_equilibrium(x)

        # dx/dc
        x_idx = np.concatenate(
            [self._eq_objective.x_idx[arg] for arg in ["p_l", "i_l", "Psi"]]
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

        Fc = Fx @ dxdc
        Fx_inv = jnp.linalg.pinv(Fx, rcond=1e-6)

        Gc = Gx @ dxdc

        GxFx = Gx @ Fx_inv
        LHS = GxFx @ Fc - Gc
        return LHS

    def hess(self, x):

        J = self.jac(x)
        return J.T @ J
