"""Wrappers for doing STELLOPT/SIMSOPT like optimization."""

import functools

import numpy as np

from desc.backend import jit, jnp
from desc.batching import batched_vectorize
from desc.objectives import (
    BoundaryRSelfConsistency,
    BoundaryZSelfConsistency,
    ObjectiveFunction,
    get_fixed_boundary_constraints,
    maybe_add_self_consistency,
)
from desc.objectives.utils import factorize_linear_constraints
from desc.utils import Timer, errorif, get_instance, setdefault

from .utils import f_where_x


class LinearConstraintProjection(ObjectiveFunction):
    """Remove linear constraints via orthogonal projection.

    Given a problem of the form

    min_x f(x) subject to A*x=b

    We can write any feasible x=xp + Z*x_reduced where xp is a particular solution to
    Ax=b (taken to be the least norm solution), Z is a representation for the null
    space of A (A*Z=0) and x_reduced is unconstrained. This transforms the problem into

    min_x_reduced f(x_reduced)

    Parameters
    ----------
    objective : ObjectiveFunction
        Objective function to optimize.
    constraint : ObjectiveFunction
        Objective function of linear constraints to enforce.
    name : str
        Name of the objective function.
    """

    def __init__(self, objective, constraint, name="LinearConstraintProjection"):
        errorif(
            not isinstance(objective, ObjectiveFunction),
            ValueError,
            "Objective should be instance of ObjectiveFunction.",
        )
        errorif(
            not isinstance(constraint, ObjectiveFunction),
            ValueError,
            "Constraint should be instance of ObjectiveFunction.",
        )
        for con in constraint.objectives:
            errorif(
                not con.linear,
                ValueError,
                "LinearConstraintProjection method cannot handle "
                + f"nonlinear constraint {con}.",
            )
            errorif(
                con.bounds is not None,
                ValueError,
                f"Linear constraint {con} must use target instead of bounds.",
            )

        self._objective = objective
        self._constraint = constraint
        self._built = False
        # don't want to compile this, just use the compiled objective
        self._use_jit = False
        self._compiled = False
        self._name = name

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
        if not self._constraint.built:
            self._constraint.build(verbose=verbose)

        self._dim_f = self._objective.dim_f
        self._scalar = self._objective.scalar
        (
            self._xp,
            self._A,
            self._b,
            self._Z,
            self._D,
            self._unfixed_idx,
            self._project,
            self._recover,
        ) = factorize_linear_constraints(
            self._objective,
            self._constraint,
        )
        self._dim_x = self._objective.dim_x
        self._dim_x_reduced = self._Z.shape[1]

        # equivalent matrix for A[unfixed_idx] @ D @ Z == A @ unfixed_idx_mat
        self._unfixed_idx_mat = jnp.diag(self._D)[:, self._unfixed_idx] @ self._Z

        self._built = True
        timer.stop("Linear constraint projection build")
        if verbose > 1:
            timer.disp("Linear constraint projection build")

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
            Reduced state vector (e.g. from calling self.x(*things)).
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
        """Compute gradient of self.compute_scalar.

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
        return df[self._unfixed_idx] @ (self._Z * self._D[self._unfixed_idx, None])

    def hess(self, x_reduced, constants=None):
        """Compute Hessian of self.compute_scalar.

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
        return (
            (self._Z.T * (1 / self._D)[None, self._unfixed_idx])
            @ df[self._unfixed_idx, :][:, self._unfixed_idx]
            @ (self._Z * self._D[self._unfixed_idx, None])
        )

    def _jac(self, x_reduced, constants=None, op="scaled"):
        x = self.recover(x_reduced)
        v = self._unfixed_idx_mat
        df = getattr(self._objective, "jvp_" + op)(v.T, x, constants)
        return df.T

    def jac_scaled(self, x_reduced, constants=None):
        """Compute Jacobian of self.compute_scaled.

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
        return self._jac(x_reduced, constants, "scaled")

    def jac_scaled_error(self, x_reduced, constants=None):
        """Compute Jacobian of self.compute_scaled_error.

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
        return self._jac(x_reduced, constants, "scaled_error")

    def jac_unscaled(self, x_reduced, constants=None):
        """Compute Jacobian of self.compute_unscaled.

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
        return self._jac(x_reduced, constants, "unscaled")

    def _jvp(self, v, x_reduced, constants=None, op="jvp_scaled"):
        x = self.recover(x_reduced)
        v = self._unfixed_idx_mat @ v
        df = getattr(self._objective, op)(v, x, constants)
        return df

    def jvp_scaled(self, v, x_reduced, constants=None):
        """Compute Jacobian-vector product of self.compute_scaled.

        Parameters
        ----------
        v : tuple of ndarray
            Vectors to right-multiply the Jacobian by.
        x_reduced : ndarray
            Optimization variables with linear constraints removed.
        constants : list
            Constant parameters passed to sub-objectives.

        """
        return self._jvp(v, x_reduced, constants, "jvp_scaled")

    def jvp_scaled_error(self, v, x_reduced, constants=None):
        """Compute Jacobian-vector product of self.compute_scaled_error.

        Parameters
        ----------
        v : tuple of ndarray
            Vectors to right-multiply the Jacobian by.
        x_reduced : ndarray
            Optimization variables with linear constraints removed.
        constants : list
            Constant parameters passed to sub-objectives.

        """
        return self._jvp(v, x_reduced, constants, "jvp_scaled_error")

    def jvp_unscaled(self, v, x_reduced, constants=None):
        """Compute Jacobian-vector product of self.compute_unscaled.

        Parameters
        ----------
        v : tuple of ndarray
            Vectors to right-multiply the Jacobian by.
        x_reduced : ndarray
            Optimization variables with linear constraints removed.
        constants : list
            Constant parameters passed to sub-objectives.

        """
        return self._jvp(v, x_reduced, constants, "jvp_unscaled")

    def _vjp(self, v, x_reduced, constants=None, op="vjp_scaled"):
        x = self.recover(x_reduced)
        df = getattr(self._objective, op)(v, x, constants)
        return df[self._unfixed_idx] @ (self._Z * self._D[self._unfixed_idx, None])

    def vjp_scaled(self, v, x_reduced, constants=None):
        """Compute vector-Jacobian product of self.compute_scaled.

        Parameters
        ----------
        v : ndarray
            Vector to left-multiply the Jacobian by.
        x_reduced : ndarray
            Optimization variables with linear constraints removed.
        constants : list
            Constant parameters passed to sub-objectives.

        """
        return self._vjp(v, x_reduced, constants, "vjp_scaled")

    def vjp_scaled_error(self, v, x_reduced, constants=None):
        """Compute vector-Jacobian product of self.compute_scaled_error.

        Parameters
        ----------
        v : ndarray
            Vector to left-multiply the Jacobian by.
        x_reduced : ndarray
            Optimization variables with linear constraints removed.
        constants : list
            Constant parameters passed to sub-objectives.

        """
        return self._vjp(v, x_reduced, constants, "vjp_scaled_error")

    def vjp_unscaled(self, v, x_reduced, constants=None):
        """Compute vector-Jacobian product of self.compute_unscaled.

        Parameters
        ----------
        v : ndarray
            Vector to left-multiply the Jacobian by.
        x_reduced : ndarray
            Optimization variables with linear constraints removed.
        constants : list
            Constant parameters passed to sub-objectives.

        """
        return self._vjp(v, x_reduced, constants, "vjp_unscaled")

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
    perturb_options, solve_options : dict
        dictionary of arguments passed to Equilibrium.perturb and Equilibrium.solve
        during the projection step.
    name : str
        Name of the objective function.
    """

    def __init__(
        self,
        objective,
        constraint,
        eq,
        perturb_options=None,
        solve_options=None,
        name="ProximalProjection",
    ):
        assert isinstance(objective, ObjectiveFunction), (
            "objective should be instance of ObjectiveFunction." ""
        )
        assert isinstance(constraint, ObjectiveFunction), (
            "constraint should be instance of ObjectiveFunction." ""
        )
        for con in constraint.objectives:
            errorif(
                not con._equilibrium,
                ValueError,
                "ProximalProjection method cannot handle general "
                + f"nonlinear constraint {con}.",
            )
            # can't have bounds on constraint bc if constraint is satisfied then
            # Fx == 0, and that messes with Gx @ Fx^-1 Fc etc.
            errorif(
                con.bounds is not None,
                ValueError,
                "ProximalProjection can only handle equality constraints, "
                + f"got bounds for constraint {con}",
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
        self._name = name

    def _set_eq_state_vector(self):
        full_args = self._eq.optimizable_params.copy()
        self._args = self._eq.optimizable_params.copy()
        # Proximal projection cannot use these parameters
        # Remove them from the list of parameters to optimize
        for arg in [
            "R_lmn",
            "Z_lmn",
            "L_lmn",
            "Ra_n",
            "Za_n",
            "Rp_lmn",
            "Zp_lmn",
            "Lp_lmn",
        ]:
            if arg in self._args:
                self._args.remove(arg)
        linear_constraint = ObjectiveFunction(self._linear_constraints)
        linear_constraint.build()
        _, _, _, self._Z, self._D, self._unfixed_idx, _, _ = (
            factorize_linear_constraints(self._constraint, linear_constraint)
        )

        # dx/dc - goes from the full state to optimization variables for eq
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

        errorif(
            self._constraint.things != [eq],
            ValueError,
            "ProximalProjection can only handle constraints on the equilibrium.",
        )

        self._objectives = [self._objective, self._constraint]
        self._set_things()

        self._eq_idx = self.things.index(self._eq)

        self._dim_f = self._objective.dim_f
        if self._dim_f == 1:
            self._scalar = True
        else:
            self._scalar = False

        self._set_eq_state_vector()

        # map from eq c to full c
        self._dimc_per_thing = [t.dim_x for t in self.things]
        self._dimc_per_thing[self._eq_idx] = np.sum(
            [self._eq.dimensions[arg] for arg in self._args]
        )
        self._dimx_per_thing = [t.dim_x for t in self.things]

        # equivalent matrix for A[unfixed_idx] @ D @ Z == A @ unfixed_idx_mat
        self._unfixed_idx_mat = jnp.eye(self._objective.dim_x)
        self._unfixed_idx_mat = jnp.split(
            self._unfixed_idx_mat, np.cumsum([t.dim_x for t in self.things]), axis=-1
        )
        self._unfixed_idx_mat[self._eq_idx] = self._unfixed_idx_mat[self._eq_idx][
            :, self._unfixed_idx
        ] @ (self._Z * self._D[self._unfixed_idx, None])
        self._unfixed_idx_mat = np.concatenate(
            [np.atleast_2d(foo) for foo in self._unfixed_idx_mat], axis=-1
        )

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

        x = jnp.atleast_1d(jnp.asarray(x))
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
                p.update(  # add in dummy values for missing parameters
                    {
                        arg: jnp.zeros_like(xis)
                        for arg, xis in t.params_dict.items()
                        if arg not in self._args  # R_lmn, Z_lmn, L_lmn, Ra_n, Za_n
                    }
                )
                params += [p]
            else:
                params += [t.unpack_params(xi)]

        if per_objective:
            # params is a list of lists of dicts, for each thing and for each objective
            params = self._unflatten(params)
            # this filters out the params of things that are unused by each objective
            params = [
                [par for par, thing in zip(param, self.things) if thing in obj.things]
                for param, obj in zip(params, self.objectives)
            ]
        return params

    def x(self, *things):
        """Return the full state vector from the Optimizable objects things."""
        # TODO (#1392): also check resolution etc?
        things = things or self.things
        assert [type(t1) is type(t2) for t1, t2 in zip(things, self.things)]
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
        constants = setdefault(constants, self.constants)
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
        constants = setdefault(constants, self.constants)
        xopt, _ = self._update_equilibrium(x, store=False)
        return self._objective.compute_scaled_error(xopt, constants[0])

    def compute_scalar(self, x, constants=None):
        """Compute the sum of squares error.

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives.

        Returns
        -------
        f : float
            Objective function scalar value.

        """
        f = jnp.sum(self.compute_scaled_error(x, constants=constants) ** 2) / 2
        return f

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
        constants = setdefault(constants, self.constants)
        xopt, _ = self._update_equilibrium(x, store=False)
        return self._objective.compute_unscaled(xopt, constants[0])

    def grad(self, x, constants=None):
        """Compute gradient of self.compute_scalar.

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
        # TODO (#1393): figure out projected vjp to make this better
        f = jnp.atleast_1d(self.compute_scaled_error(x, constants))
        J = self.jac_scaled_error(x, constants)
        return f.T @ J

    def hess(self, x, constants=None):
        """Compute Hessian of self.compute_scalar.

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
        J = self.jac_scaled_error(x, constants)
        return J.T @ J

    def jac_scaled(self, x, constants=None):
        """Compute Jacobian of self.compute_scaled.

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
        v = jnp.eye(x.shape[0])
        return self.jvp_scaled(v, x, constants).T

    def jac_scaled_error(self, x, constants=None):
        """Compute Jacobian of self.compute_scaled_error.

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
        v = jnp.eye(x.shape[0])
        return self.jvp_scaled_error(v, x, constants).T

    def jac_unscaled(self, x, constants=None):
        """Compute Jacobian of self.compute_unscaled.

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
        v = jnp.eye(x.shape[0])
        return self.jvp_unscaled(v, x, constants).T

    def jvp_scaled(self, v, x, constants=None):
        """Compute Jacobian-vector product of self.compute_scaled.

        Parameters
        ----------
        v : ndarray or tuple of ndarray
            Vectors to right-multiply the Jacobian by.
            This method only works for first order jvps.
        x : ndarray
            Optimization variables.
        constants : list
            Constant parameters passed to sub-objectives.

        """
        v = v[0] if isinstance(v, (tuple, list)) else v
        constants = setdefault(constants, self.constants)
        xg, xf = self._update_equilibrium(x, store=True)
        jvpfun = lambda u: self._jvp(u, xf, xg, constants, op="scaled")
        return batched_vectorize(
            jvpfun, signature="(n)->(k)", chunk_size=self._objective._jac_chunk_size
        )(v)

    def jvp_scaled_error(self, v, x, constants=None):
        """Compute Jacobian-vector product of self.compute_scaled_error.

        Parameters
        ----------
        v : ndarray or tuple of ndarray
            Vectors to right-multiply the Jacobian by.
            This method only works for first order jvps.
        x : ndarray
            Optimization variables.
        constants : list
            Constant parameters passed to sub-objectives.

        """
        v = v[0] if isinstance(v, (tuple, list)) else v
        constants = setdefault(constants, self.constants)
        xg, xf = self._update_equilibrium(x, store=True)
        jvpfun = lambda u: self._jvp(u, xf, xg, constants, op="scaled_error")
        return batched_vectorize(
            jvpfun, signature="(n)->(k)", chunk_size=self._objective._jac_chunk_size
        )(v)

    def jvp_unscaled(self, v, x, constants=None):
        """Compute Jacobian-vector product of self.compute_unscaled.

        Parameters
        ----------
        v : ndarray or tuple of ndarray
            Vectors to right-multiply the Jacobian by.
            This method only works for first order jvps.
        x : ndarray
            Optimization variables.
        constants : list
            Constant parameters passed to sub-objectives.

        """
        v = v[0] if isinstance(v, (tuple, list)) else v
        constants = setdefault(constants, self.constants)
        xg, xf = self._update_equilibrium(x, store=True)
        jvpfun = lambda u: self._jvp(u, xf, xg, constants, op="unscaled")
        return batched_vectorize(
            jvpfun, signature="(n)->(k)", chunk_size=self._objective._jac_chunk_size
        )(v)

    def _jvp(self, v, xf, xg, constants, op):
        # we're replacing stuff like this with jvps
        # Fx_reduced = Fx[:, unfixed_idx] @ Z               # noqa: E800
        # Gx_reduced = Gx[:, unfixed_idx] @ Z               # noqa: E800
        # Fc = Fx @ dxdc @ v                                # noqa: E800
        # Gc = Gx @ dxdc @ v                                # noqa: E800
        # LHS = Gx_reduced @ (Fx_reduced_inv @ Fc) - Gc     # noqa: E800

        # v contains "boundary" dofs from eq and other objects
        # want jvp_f to only get parts from equilibrium, not other things
        vs = jnp.split(v, np.cumsum(self._dimc_per_thing))
        # this is Fx_reduced_inv @ Fc
        dfdc = _proximal_jvp_f_pure(
            self._constraint,
            xf,
            constants[1],
            vs[self._eq_idx],
            self._unfixed_idx,
            self._Z,
            self._D,
            self._dxdc,
            op,
        )
        # broadcasting against multiple things
        dfdcs = [jnp.zeros(dim) for dim in self._dimc_per_thing]
        dfdcs[self._eq_idx] = dfdc
        dfdc = jnp.concatenate(dfdcs)

        # dG/dc = Gx_reduced @ (Fx_reduced_inv @ Fc) - Gc
        # = Gx @ (unfixed_idx @ Z @ dfdc - dxdc @ v)
        # unfixed_idx_mat includes Z already
        dxdcv = jnp.concatenate(
            [
                *vs[: self._eq_idx],
                self._dxdc @ vs[self._eq_idx],
                *vs[self._eq_idx + 1 :],
            ]
        )
        tangent = self._unfixed_idx_mat @ dfdc - dxdcv
        if self._objective._deriv_mode in ["batched"]:
            out = getattr(self._objective, "jvp_" + op)(tangent, xg, constants[0])
        else:  # deriv_mode == "blocked"
            vgs = jnp.split(tangent, np.cumsum(self._dimx_per_thing))
            xgs = jnp.split(xg, np.cumsum(self._dimx_per_thing))
            out = _proximal_jvp_blocked_pure(self._objective, vgs, xgs, op)
        return -out

    @property
    def constants(self):
        """list: constant parameters for each sub-objective."""
        return [self._objective.constants, self._constraint.constants]

    def __getattr__(self, name):
        """For other attributes we defer to the base objective."""
        return getattr(self._objective, name)


# in ProximalProjection we have an explicit state that we keep track of (and add
# to as we go) meaning if we jit anything with self static it doesn't update
# correctly, while if we leave self unstatic then it recompiles every time because
# the pytree structure of ProximalProjection is changing. To get around that we
# define these helper functions that are stateless so we can safely jit them


@functools.partial(jit, static_argnames=["op"])
def _proximal_jvp_f_pure(constraint, xf, constants, dc, unfixed_idx, Z, D, dxdc, op):
    Fx = getattr(constraint, "jac_" + op)(xf, constants)
    Fx_reduced = Fx @ jnp.diag(D)[:, unfixed_idx] @ Z
    Fc = Fx @ (dxdc @ dc)
    Fxh = Fx_reduced
    cutoff = jnp.finfo(Fxh.dtype).eps * max(Fxh.shape)
    uf, sf, vtf = jnp.linalg.svd(Fxh, full_matrices=False)
    sf += sf[-1]  # add a tiny bit of regularization
    sfi = jnp.where(sf < cutoff * sf[0], 0, 1 / sf)
    Fxh_inv = vtf.T @ (sfi[..., None] * uf.T)
    return Fxh_inv @ Fc


@functools.partial(jit, static_argnames=["op"])
def _proximal_jvp_blocked_pure(objective, vgs, xgs, op):
    out = []
    for k, (obj, const) in enumerate(zip(objective.objectives, objective.constants)):
        thing_idx = objective._things_per_objective_idx[k]
        xi = [xgs[i] for i in thing_idx]
        vi = [vgs[i] for i in thing_idx]
        assert len(xi) > 0
        assert len(vi) > 0
        assert len(xi) == len(vi)
        if obj._deriv_mode == "rev":
            # obj might not allow fwd mode, so compute full rev mode jacobian
            # and do matmul manually. This is slightly inefficient, but usually
            # when rev mode is used, dim_f <<< dim_x, so its not too bad.
            Ji = getattr(obj, "jac_" + op)(*xi, constants=const)
            outi = jnp.array([Jii @ vii.T for Jii, vii in zip(Ji, vi)]).sum(axis=0)
            out.append(outi)
        else:
            outi = getattr(obj, "jvp_" + op)([_vi for _vi in vi], xi, constants=const).T
            out.append(outi)
    out = jnp.concatenate(out)
    return out
