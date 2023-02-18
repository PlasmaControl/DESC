"""Wrappers for doing STELLOPT/SIMSOPT like optimization."""

import numpy as np

from desc.backend import jnp
from desc.compute import arg_order

from ._equilibrium import CurrentDensity
from .objective_funs import ObjectiveFunction
from .utils import (
    align_jacobian,
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
            and self._eq.iota is not None,
            kinetic=eq.electron_temperature is not None,
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

        # this is everything taken by either objective
        self._other_args = [arg for arg in self._objective.args if arg not in arg_order]
        self._full_args = self._eq_objective.args + self._objective.args
        self._full_args = [arg for arg in arg_order if arg in self._full_args]
        self._full_args += self._other_args

        # remove constraints that aren't necessary
        self._constraints = tuple(
            [con for con in self._constraints if con.args[0] in self._eq_objective.args]
        )

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

        self._QI_dict = self._objective._QI_dict
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
        ) = factorize_linear_constraints(
            self._constraints, self._full_args, self._dimensions
        )

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
            if arg in arg_order:
                self._x_old[self.x_idx[arg]] = getattr(eq, arg)
            else:
                arg_name = arg.split(" ")
                self._x_old[self.x_idx[arg]] = getattr(
                    self._objective.objectives[self._QI_dict[arg_name[1]]],
                    arg_name[0],
                )

        self._allx = [self._x_old]
        self._allxopt = [self._objective.x(eq)]
        self._allxeq = [self._eq_objective.x(eq)]
        self.history = {}
        for arg in arg_order:
            self.history[arg] = [np.asarray(getattr(self._eq, arg)).copy()]
        for arg in self._other_args:
            arg_name = arg.split(" ")
            self.history[arg] = [
                np.asarray(
                    getattr(
                        self._objective.objectives[
                            self._QI_dict[arg_name[1]]
                        ],
                        arg_name[0],
                    )
                ).copy()
            ]

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
        from desc.optimize.utils import f_where_x

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
                objective=self._eq_objective,
                constraints=self._constraints,
                deltas=deltas,
                **self._perturb_options
            )
            self._eq.solve(
                objective=self._eq_objective,
                constraints=self._constraints,
                **self._solve_options
            )
            xopt = self._objective.x(self._eq)
            xeq = self._eq_objective.x(self._eq)
            self._allx.append(x)
            self._allxopt.append(xopt)
            self._allxeq.append(xeq)

        if store:
            self._x_old = x
            xd = self.unpack_state(x)
            xod = self._objective.unpack_state(xopt)
            xed = self._eq_objective.unpack_state(xeq)
            xd.update(xod)
            xd.update(xed)
            for arg in arg_order:
                val = xd.get(arg, self.history[arg][-1])
                self.history[arg] += [np.asarray(val).copy()]
                # ensure eq has correct values if we didn't go into else block above.
                if val.size:
                    setattr(self._eq, arg, val)
            for con in self._constraints:
                con.update_target(self._eq)
        else:
            for arg in arg_order:
                val = self.history[arg][-1].copy()
                if val.size:
                    setattr(self._eq, arg, val)
            for con in self._constraints:
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
        Fx = self._eq_objective.jac(xf)
        Gx = self._objective.jac(xg)
        Fx = align_jacobian(Fx, self._eq_objective, self._full_args, self._dimensions)
        Gx = align_jacobian(Gx, self._objective, self._full_args, self._dimensions)

        # projections onto optimization space
        # possibly better way: Gx @ np.eye(Gx.shape[1])[:,self._unfixed_idx] @ self._Z
        Fx_reduced = Fx[:, self._unfixed_idx] @ self._Z
        Gx_reduced = Gx[:, self._unfixed_idx] @ self._Z
        Fc = Fx @ self._dxdc
        Gc = Gx @ self._dxdc

        # FIXME (@f0uriest): need to import here to avoid circular dependencies
        from desc.optimize.utils import compute_jac_scale

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
