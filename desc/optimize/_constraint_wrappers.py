"""Wrappers for doing STELLOPT/SIMSOPT like optimization."""

import numpy as np

from desc.backend import jnp
from desc.objectives import (
    BoundaryRSelfConsistency,
    BoundaryZSelfConsistency,
    ObjectiveFunction,
    get_fixed_boundary_constraints,
    maybe_add_self_consistency,
)
from desc.objectives.utils import factorize_linear_constraints
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
    verbose : int, optional
        Level of output.
    """

    def __init__(self, objective, constraints, verbose=1):
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
        self._constraints = constraints
        self._built = False
        # don't want to compile this, just use the compiled objective
        self._use_jit = False
        self._compiled = False

    def build(self, use_jit=None, verbose=1):
        """Build the objective.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
            Note: unused by this class, should pass to sub-objectives directly.
        verbose : int, optional
            Level of output.

        """
        timer = Timer()
        timer.start("Linear constraint projection build")

        # we don't always build here because in ~all cases the user doesn't interact
        # with this directly, so if the user wants to manually rebuild they should
        # do it before this wrapper is created for them.
        if not self._objective.built:
            self._objective.build(verbose=verbose)
        for con in self._constraints:
            if not con.built:
                con.build(verbose=verbose)

        self._dim_f = self._objective.dim_f
        self._scalar = self._objective.scalar
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
            self._objective,
        )
        self._dim_x = self._objective.dim_x
        self._dim_x_reduced = self._Z.shape[1]

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

    def x(self, *things):
        """Return the reduced state vector from the Equilibrium eq."""
        x = self._objective.x(*things)
        return self.project(x)

    def unpack_state(self, x, per_objective=True):
        """Unpack the state vector into its components.

        Parameters
        ----------
        x : ndarray
            State vector, either full or reduced.
        per_objective : bool
            Whether to return param dicts for each objective (default) or for each
            unique optimizable thing.

        Returns
        -------
        params : pytree of dict
            if per_objective is True, this is a nested list of of parameters for each
            sub-Objective, such that self.objectives[i] has parameters params[i].
            Otherwise, it is a list of parameters tied to each optimizable thing
            such that params[i] = self.things[i].params_dict

        """
        if x.size != self._dim_x_reduced:
            raise ValueError(
                "Input vector dimension is invalid, expected "
                + f"{self._dim_x_reduced} got {x.size}."
            )
        x = self.recover(x)
        return self._objective.unpack_state(x, per_objective)

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

    def __getattr__(self, name):
        """For other attributes we defer to the base objective."""
        return getattr(self._objective, name)


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
    eq : Equilibrium
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
        eq,
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

    def _set_eq_state_vector(self):

        full_args = self._eq.optimizable_params.copy()
        self._args = self._eq.optimizable_params.copy()
        for arg in ["R_lmn", "Z_lmn", "L_lmn", "Ra_n", "Za_n"]:
            self._args.remove(arg)
        (
            xp,
            A,
            b,
            self._Z,
            self._unfixed_idx,
            project,
            recover,
        ) = factorize_linear_constraints(self._linear_constraints, self._constraint)

        # dx/dc - goes from the full state to optimization variables
        dxdc = []
        xz = {arg: np.zeros(self._eq.dimensions[arg]) for arg in full_args}

        for arg in self._args:
            if arg not in ["Rb_lmn", "Zb_lmn"]:
                x_idx = self._eq.x_idx[arg]
                dxdc.append(np.eye(self._eq.dim_x)[:, x_idx])
            if arg == "Rb_lmn":
                c = get_instance(self._linear_constraints, BoundaryRSelfConsistency)
                A = c.jac_unscaled(xz)[0]["R_lmn"]
                Ainv = np.linalg.pinv(A)
                dxdRb = np.eye(self._eq.dim_x)[:, self._eq.x_idx["R_lmn"]] @ Ainv
                dxdc.append(dxdRb)
            if arg == "Zb_lmn":
                c = get_instance(self._linear_constraints, BoundaryZSelfConsistency)
                A = c.jac_unscaled(xz)[0]["Z_lmn"]
                Ainv = np.linalg.pinv(A)
                dxdZb = np.eye(self._eq.dim_x)[:, self._eq.x_idx["Z_lmn"]] @ Ainv
                dxdc.append(dxdZb)
        self._dxdc = np.hstack(dxdc)

    def build(self, use_jit=None, verbose=1):  # noqa: C901
        """Build the objective.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
            Note: unused by this class, should pass to sub-objectives directly.
        verbose : int, optional
            Level of output.

        """
        eq = self._eq
        timer = Timer()
        timer.start("Proximal projection build")

        self._eq = eq
        self._linear_constraints = get_fixed_boundary_constraints(eq=eq)
        self._linear_constraints = maybe_add_self_consistency(
            self._eq, self._linear_constraints
        )

        # we don't always build here because in ~all cases the user doesn't interact
        # with this directly, so if the user wants to manually rebuild they should
        # do it before this wrapper is created for them.
        if not self._objective.built:
            self._objective.build(use_jit=use_jit, verbose=verbose)
        if not self._constraint.built:
            self._constraint.build(use_jit=use_jit, verbose=verbose)

        for constraint in self._linear_constraints:
            constraint.build(use_jit=use_jit, verbose=verbose)

        assert self._constraint.things == [
            eq
        ], "ProximalProjection can only handle constraints on the equilibrium."

        self._objectives = [self._objective, self._constraint]
        self._set_things(self._all_things)

        self._eq_idx = self.things.index(self._eq)

        self._dim_f = self._objective.dim_f
        if self._dim_f == 1:
            self._scalar = True
        else:
            self._scalar = False

        self._set_eq_state_vector()
        # history and caching
        self._x_old = self.x(self.things)
        self._allx = [self._x_old]
        self._allxopt = [self._objective.x(*self.things)]
        self._allxeq = [self._eq.pack_params(self._eq.params_dict)]
        self.history = [[t.params_dict.copy() for t in self.things]]

        self._built = True
        timer.stop("Proximal projection build")
        if verbose > 1:
            timer.disp("Proximal projection build")

    def unpack_state(self, x, per_objective=True):
        """Unpack the state vector into its components.

        Parameters
        ----------
        x : ndarray
            State vector.
        per_objective : bool
            Whether to return param dicts for each objective (default) or for each
            unique optimizable thing.

        Returns
        -------
        params : dict
            Parameter dictionary for equilibrium, with just external degrees of freedom
            visible to the optimizer.

        """
        if not self.built:
            raise RuntimeError("ObjectiveFunction must be built first.")

        x = jnp.atleast_1d(x)
        if x.size != self.dim_x:
            raise ValueError(
                "Input vector dimension is invalid, expected "
                + f"{self.dim_x} got {x.size}."
            )

        xs_splits = [t.dim_x for t in self.things]
        xs_splits[self._eq_idx] = np.sum(
            [self._eq.dimensions[arg] for arg in self._args]
        )
        xs_splits = np.cumsum(xs_splits)
        xs = jnp.split(x, xs_splits)
        params = []
        for t, xi in zip(self.things, xs):
            if t is self._eq:
                xi_splits = np.cumsum([self._eq.dimensions[arg] for arg in self._args])
                p = {arg: xis for arg, xis in zip(self._args, jnp.split(xi, xi_splits))}
                params += [p]
            else:
                params += [t.unpack_params(xi)]

        if per_objective:
            params = self._unflatten(params)
        return params

    def x(self, *things):
        """Return the full state vector from the Optimizable objects things."""
        # TODO: also check resolution etc?
        things = things or self.things
        assert [type(t1) == type(t2) for t1, t2 in zip(things, self.things)]
        xs = []
        for t in self.things:
            if t is self._eq:
                xs += [
                    jnp.concatenate(
                        [jnp.atleast_1d(t.params_dict[arg]) for arg in self._args]
                    )
                ]
            else:
                xs += [t.pack_params(t.params_dict)]

        return jnp.concatenate(xs)

    @property
    def dim_x(self):
        """int: Dimension of the state vector."""
        s = 0
        for t in self.things:
            if t is self._eq:
                s += sum(self._eq.dimensions[arg] for arg in self._args)
            else:
                s += t.dim_x
        return s

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
            x_list = self.unpack_state(x, False)
            x_list_old = self.unpack_state(self._x_old, False)
            x_dict = x_list[self._eq_idx]
            x_dict_old = x_list_old[self._eq_idx]
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
            xeq = self._eq.pack_params(self._eq.params_dict)
            x_list[self._eq_idx] = self._eq.params_dict.copy()
            xopt = jnp.concatenate(
                [t.pack_params(xi) for t, xi in zip(self.things, x_list)]
            )
            self._allx.append(x)
            self._allxopt.append(xopt)
            self._allxeq.append(xeq)

        if store:
            self._x_old = x
            x_list = self.unpack_state(x, False)
            xeq_dict = self._eq.unpack_params(xeq)
            self._eq.params_dict = xeq_dict
            x_list[self._eq_idx] = xeq_dict
            self.history.append(x_list)
            for con in self._linear_constraints:
                if hasattr(con, "update_target"):
                    con.update_target(self._eq)
        else:
            # reset to last good params
            self._eq.params_dict = self.history[-1][self._eq_idx]
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

        # f depends only on eq, g can depend on other things
        # all the fancy projection/prox stuff only applies to eq dofs
        # so we split Gx into parts that depend only on eq, and stuff on other things
        Gxs = jnp.split(Gx, np.cumsum([t.dim_x for t in self.things]), axis=-1)
        Gx = Gxs[self._eq_idx]
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
        uf, sf, vtf = jnp.linalg.svd(Fxh, full_matrices=False)
        sf += sf[-1]  # add a tiny bit of regularization
        sfi = np.where(sf < cutoff * sf[0], 0, 1 / sf)
        Fxh_inv = vtf.T @ (sfi[..., np.newaxis] * uf.T)

        # TODO: make this more efficient for finite differences etc. Can probably
        # reduce the number of operations and tangents
        LHS = Gxh @ (Fxh_inv @ Fc) - Gc
        # now add back in non-eq dofs
        Gxs[self._eq_idx] = -LHS
        return jnp.hstack(Gxs)

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

    def __getattr__(self, name):
        """For other attributes we defer to the base objective."""
        return getattr(self._objective, name)
