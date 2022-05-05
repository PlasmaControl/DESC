import numpy as np
from desc.backend import jnp
from .utils import (
    get_equilibrium_objective,
    get_fixed_boundary_constraints,
    factorize_linear_constraints,
)
from desc.compute import arg_order


class WrappedEquilibriumObjective:
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

    def __init__(
        self,
        objective,
        equilibrium_objective=None,
        eq=None,
        verbose=1,
        perturb_options={},
        solve_options={},
    ):

        self._objective = objective
        self._perturb_options = perturb_options
        self._solve_options = solve_options
        self._verbose = verbose

        if eq is not None:
            self.build(eq)

    def build(self, eq):

        self._eq = eq.copy()
        if self._equilibrium_objective is None:
            self._equilibrium_objective = get_equilibrium_objective()
        self._constraints = get_fixed_boundary_constraints(self._eq.bdry_mode)

        self._objective.build(self._eq)
        self._equilibrium_objective.build(self._eq)
        for constraint in self._constraints:
            constraint.build(self._eq)

        xp, A, Ainv, b, Z, unfixed_idx, project, recover = factorize_linear_constraints(
            self._constraints,
            self._equilibrium_objective.dim_x,
            self._equilibrium_objective.x_idx,
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
        self._fixed_idx = np.setdiff1d(
            np.arange(self._equilibrium_objective.dim_x), unfixed_idx
        )
        self._A = A
        self._Ainv = Ainv
        self._Z = Z
        # full x is everything FB takes, which should be everything?
        self._xfull = self._equilibrium_objective.x(eq)
        # optimization x is all the stuff thats fixed during a normal eq solve
        self._xopt_old = jnp.concatenate(
            [jnp.atleast_1d(getattr(self._eq, arg)) for arg in self._args]
        )

    def x(self, eq):
        return jnp.concatenate([jnp.atleast_1d(getattr(eq, arg)) for arg in self._args])

    def _unpack_xopt(self, xopt):
        kwargs = {}
        for arg in self._args:
            kwargs[arg] = xopt[self._x_idx[arg]]
        return kwargs

    def _update(self, xopt):
        if jnp.all(xopt == self._xopt_old):
            pass
        else:
            if self._verbose:
                print("Updating equilibrium")
            xoptd = self._unpack_xopt(xopt)
            xoptd_old = self._unpack_xopt(self._xopt_old)
            deltas = {"d" + str(key): xoptd[key] - xoptd_old[key] for key in xoptd}
            self._eq.perturb(
                self._equilibrium_objective,
                self._constraints,
                **deltas,
                **self._perturb_options
            )
            self._eq.solve(objective=self._equilibrium_objective, **self._solve_options)
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

        dxdc = np.zeros((self._equilibrium_objective.dim_x, 0))
        x_idx = np.concatenate(
            [self._equilibrium_objective.x_idx[arg] for arg in self._args[2:]]
        )
        x_idx.sort(kind="mergesort")
        dxdc = np.eye(self._equilibrium_objective.dim_x)[:, x_idx]
        dxdRb = (
            np.eye(self._equilibrium_objective.dim_x)[
                :, self._equilibrium_objective.x_idx["R_lmn"]
            ]
            @ self._Ainv["R_lmn"]
        )
        dxdc = np.hstack((dxdc, dxdRb))
        dxdZb = (
            np.eye(self._equilibrium_objective.dim_x)[
                :, self._equilibrium_objective.x_idx["Z_lmn"]
            ]
            @ self._Ainv["Z_lmn"]
        )
        dxdc = np.hstack((dxdc, dxdZb))

        xf = self._equilibrium_objective.x(self._eq)
        Fx = self._equilibrium_objective.jac(xf)

        # 1st partial derivatives of g objective wrt both x and c
        xg = self._objective.x(self._eq)
        Gx = self._objective.jac(xg)

        Fx = {
            arg: Fx[:, self._equilibrium_objective.x_idx[arg]]
            for arg in self._equilibrium_objective.args
        }
        Gx = {arg: Gx[:, self._objective.x_idx[arg]] for arg in self._objective.args}

        for arg in arg_order:
            if (arg in self._objective.args) and not (
                arg in self._equilibrium_objective.args
            ):
                Fx[arg] = jnp.zeros(
                    (self._equilibrium_objective.dim_f, self._objective.dimensions[arg])
                )
            if (arg in self._equilibrium_objective.args) and not (
                arg in self._objective.args
            ):
                Gx[arg] = jnp.zeros(
                    (self._objective.dim_f, self._equilibrium_objective.dimensions[arg])
                )

        Fx = jnp.hstack([Fx[arg] for arg in arg_order if arg in Fx])
        Gx = jnp.hstack([Gx[arg] for arg in arg_order if arg in Gx])

        Fc = Fx @ dxdc
        Fx_inv = jnp.linalg.pinv(Fx, rcond=1e-6)

        Gc = Gx @ dxdc

        GxFx = Gx @ Fx_inv
        LHS = GxFx @ Fc - Gc
        return LHS

    def hess(self, xopt):

        J = self.jac(xopt)
        return J.T @ J
