import numpy as np
from desc.backend import jnp
from .utils import get_force_balance_objective, factorize_linear_constraints


class WrappedForceObjective:
    """Evaluate an objective subject to equilibrium constraint

    Parameters
    ----------
    objetive : ObjectiveFunction
        objective to evaluate
    eq : Equilibrium
        starting equilibrium for optimization
    perturb_options : dict
        keyword options passed to eq.perturb after each update
    solve_options : dict
        keyword arguments passed to eq.solve after each update
    """

    def __init__(self, objective, eq=None, perturb_options={}, solve_options={}):

        self._objective = objective
        self._force_objective, self._constraints = get_force_balance_objective()
        self._perturb_options = perturb_options
        self._solve_options = solve_options

        if eq is not None:
            self.build(eq)

    def build(self, eq):

        self._eq = eq
        self._objective.build(self._eq)
        self._force_objective.build(self._eq)
        for constraint in self._constraints:
            constraint.build(self._eq)

        xp, A, Ainv, b, Z, unfixed_idx, project, recover = factorize_linear_constraints(
            self._constraints, self._force_objective.dim_x, self._force_objective.x_idx
        )
        self._dimensions = {}
        self._dimensions["R_lmn"] = self._eq.R_basis.num_modes
        self._dimensions["Z_lmn"] = self._eq.Z_basis.num_modes
        self._dimensions["L_lmn"] = self._eq.L_basis.num_modes
        self._dimensions["Rb_lmn"] = self._eq.surface.R_basis.num_modes
        self._dimensions["Zb_lmn"] = self._eq.surface.Z_basis.num_modes
        self._dimensions["p_l"] = self._eq.pressure.params.size
        self._dimensions["i_l"] = self._eq.iota.params.size
        self._dimensions["Psi"] = 1
        self._args = ["Rb_lmn", "Zb_lmn", "p_l", "i_l", "Psi"]
        self._x_idx = {}
        idx = 0
        for arg in self._args:
            self._x_idx[arg] = np.arange(idx, idx + self._dimensions[arg])
            idx += self._dimensions[arg]

        self._unfixed_idx = unfixed_idx
        self._fixed_idx = np.setdiff_1d(
            np.arange(self._force_objective.dim_x), unfixed_idx
        )
        self._A = A
        self._Ainv = Ainv
        # full x is everything FB takes, which should be everything?
        self._xfull = self._force_objective.x(eq)
        # optimization x is all the stuff thats fixed during a normal eq solve
        self._xopt_old = jnp.concatenate(
            [self._A @ self._xfull[self._unfixed_idx], self._xfull[self._fixed_idx]]
        )

    def _unpack_xopt(self, xopt):
        kwargs = {}
        for arg in self._args:
            kwargs[arg] = xopt[self._x_idx[arg]]
        return kwargs

    def _update(self, xopt):
        if jnp.all(xopt == self._xopt_old):
            pass
        else:
            xoptd = self._unpack_xopt(xopt)
            xoptd_old = self._unpack_xopt(self._xopt_old)
            deltas = {"d" + str(key): xoptd[key] - xoptd_old[key] for key in xoptd}
            self._eq.perturb(
                self._force_objective,
                self._constraints,
                **deltas,
                **self._perturb_options
            )
            self._eq.solve(objective=self._force_objective, **self._solve_options)
            self._xopt_old = xopt

    def compute(self, xopt):

        self._update(xopt)
        f = self._objective.compute(self._objective.x(self._eq))
        return f

    def grad(self, xopt):

        f = self.compute(xopt)
        J = self.jac(xopt)
        return f.T @ J

    def jac(self, xopt):

        self._update(xopt)

        dxdc = np.zeros((self._force_objective.dim_x, 0))
        x_idx = np.concatenate([self._force_objective.x_idx[arg] for arg in arg_order])
        x_idx.sort(kind="mergesort")
        dxdc = np.eye(self._force_objective.dim_x)[:, x_idx]
        dxdRb = (
            np.eye(self._force_objective.dim_x)[:, self._force_objective.x_idx["R_lmn"]]
            @ self._Ainv["R_lmn"]
        )
        dxdc = np.hstack((dxdc, dxdRb))
        dxdZb = (
            np.eye(self._force_objective.dim_x)[:, self._force_objective.x_idx["Z_lmn"]]
            @ self._Ainv["Z_lmn"]
        )
        dxdc = np.hstack((dxdc, dxdZb))

        xf = self._force_objective.x(self._eq)
        Fx = self._force_objective.jac(xf)
        Fx_reduced = Fx[:, self._unfixed_idx] @ self._Z
        Fc = Fx @ dxdc
        Fx_reduced_inv = jnp.linalg.pinv(Fx_reduced, rcond=1e-6)

        # 1st partial derivatives of g objective wrt both x and c
        xg = self._objective.x(self._eq)
        Gx = self._objective.jac(xg)
        Gx_reduced = Gx[:, self._unfixed_idx] @ self._Z
        Gc = Gx @ dxdc

        GxFx = Gx_reduced @ Fx_reduced_inv
        LHS = GxFx @ Fc - Gc
        return LHS

    def hess(self, xopt):

        J = self.jac(xopt)
        return J.T @ J
