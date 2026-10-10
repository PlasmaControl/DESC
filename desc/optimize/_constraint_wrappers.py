"""Wrappers for doing STELLOPT/SIMSOPT like optimization."""

import functools

import numpy as np

from desc.backend import jit, jnp, put
from desc.objectives import (
    BoundaryRSelfConsistency,
    BoundaryZSelfConsistency,
    ObjectiveFunction,
    get_fixed_boundary_constraints,
    maybe_add_self_consistency,
)
from desc.objectives.utils import (
    _Project,
    _Recover,
    factorize_linear_constraints,
    remove_fixed_parameters,
)
from desc.utils import Timer, errorif, get_instance, setdefault, svd_inv_null, warnif

from .least_squares import lsqtr
from .tr_subproblems import trust_region_step_exact_svd
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
    x_scale : array_like or ``'auto'``, optional
        Characteristic scale of each variable. Setting ``x_scale`` is equivalent
        to reformulating the problem in scaled variables ``xs = x / x_scale``.
        If set to ``'auto'``, the scale is determined from the initial state vector.
        This can be passed through optimizer options as
        solve_options["linear_constraint_options"]["x_scale"].
    name : str
        Name of the objective function.

    """

    def __init__(
        self, objective, constraint, x_scale="auto", name="LinearConstraintProjection"
    ):
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
        self._x_scale = x_scale
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
        timer.start(f"{self.name} build")

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
            self._ADinv,
            self._A_nondegenerate,
            self._degenerate_idx,
        ) = factorize_linear_constraints(
            self._objective,
            self._constraint,
            self._x_scale,
        )
        # inverse of the linear constraint matrix A without any scaling
        self._Ainv = self._D[self._unfixed_idx, None] * self._ADinv
        # nullspace of the linear constraint matrix A without any scaling
        self._ZA = self._D[self._unfixed_idx, None] * self._Z
        self._ZA = self._ZA / jnp.linalg.norm(self._ZA, axis=0)
        self._dim_x = self._objective.dim_x
        self._dim_x_reduced = self._Z.shape[1]

        # equivalent matrix for A[unfixed_idx] @ D @ Z == A @ feasible_tangents
        # Represents the tangent directions of the reduced parameters in full space
        # During optimization, we have the reduced parameters x_reduced, and we need
        # to compute the derivatives for that, but since compute functions are written
        # for the full state vector, we have to compute the derivatives with
        # these tangents.
        # For example, let's say the full state vector X has constraints X1=X2 and
        # X = [X1 X2 X3]. The reduced state vector of this is Y = [Y1 Y2]. We can take
        # Y1=X1=X2 and Y2=X3. Then df/dY1 = df/dX1 + df/dX2 and df/dY2 = df/dX3.
        # in this case, feasible_tangents = [ [1 , 0], [1, 0], [0,1]]
        # and is a shape 3x2 matrix equivalent to dx/dy
        # s.t. df/dy = df/dx @ dx/dy

        # df/dx_reduced = df/dx_full_unscaled @ dx_full_unscaled/dx_reduced # noqa: E800
        # x_full_unscaled = D(xp + Z @ x_reduced)                           # noqa: E800
        # So, the feasible tangents (aka. dx_full_unscaled/dx_reduced) is D@Z
        # Since the fixed parameters stay constant, we add 0 rows by below operation
        self._feasible_tangents = jnp.diag(self._D)[:, self._unfixed_idx] @ self._Z

        self._built = True
        timer.stop(f"{self.name} build")
        if verbose > 1:
            timer.disp(f"{self.name} build")

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

    def update_constraint_target(self, eq_new):
        """Update the target of the constraint.

        Updates the particular solution (xp), nullspace (Z), scaling (D) and
        the inverse of the scaled linear constraint matrix (ADinv) to reflect the new
        equilibrium a.k.a. the new target of the constraint of system Ax=b. This
        also updates the project and recover methods. Updating quantities in this way
        is faster than calling factorize_linear_constraints again.

        Parameters
        ----------
        eq_new : Equilibrium
            New equilibrium to target for the constraints.
        """
        for con in self._constraint.objectives:
            if hasattr(con, "update_target"):
                con.update_target(eq_new)

        dim_x = self._objective.dim_x
        # particular solution to Ax=b
        xp = jnp.zeros(dim_x)
        x0 = jnp.zeros(dim_x)
        A = self._A_nondegenerate
        b = -self._constraint.compute_scaled_error(x0)
        b = np.delete(b, self._degenerate_idx)

        # There is probably a more clever way of doing this, but for now we just
        # remove fixed parameters from A and b again by the same loop as in factorize
        # Actually A (unscaled linear constraint matrix without any degenerate rows)
        # does not change here, but still recompute it while updating others
        A, b, xp, unfixed_idx, fixed_idx = remove_fixed_parameters(A, b, xp)

        # if user specified x_scale, don't dynamically change it
        if self._x_scale == "auto":
            x_scale = self._objective.x(*self._objective.things)
            self._D = jnp.where(jnp.abs(x_scale) < 1e2, 1, jnp.abs(x_scale))

            # since D has changed, we need to update the ADinv
            # as mentioned above A does not change, so we can use the same Ainv
            # pinv(A) = Ainv, ADinv = pinv(A @ D) = Dinv @ Ainv, Dinv = 1 / D
            self._ADinv = (1 / self._D)[unfixed_idx, None] * self._Ainv
            # we also need to update the nullspace Z of AD in a similar way
            # A @ ZA = 0 -> (A @ D) @ ((1 / D) @ ZA) = 0 -> Z = (1 / D) @ ZA
            # where ZA is the nullspace of A, and Z is the nullspace of AD
            self._Z = (1 / self._D)[self._unfixed_idx, None] * self._ZA
            # we also normalize Z to make each column have unit norm
            self._Z = self._Z / jnp.linalg.norm(self._Z, axis=0)

        xp = put(xp, unfixed_idx, self._ADinv @ b)
        xp = put(xp, fixed_idx, ((1 / self._D) * xp)[fixed_idx])
        # cast to jnp arrays
        self._xp = jnp.asarray(xp)

        self._project = _Project(self._Z, self._D, self._xp, self._unfixed_idx)
        self._recover = _Recover(self._Z, self._D, self._xp, self._unfixed_idx, dim_x)

    def compute_unscaled(self, x_reduced, constants=None):
        """Compute the unscaled form of the objective function.

        Parameters
        ----------
        x_reduced : ndarray
            Reduced state vector that satisfies linear constraints.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

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
            Constant parameters passed to sub-objectives. (Deprecated)

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
            Constant parameters passed to sub-objectives. (Deprecated)

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
            Constant parameters passed to sub-objectives. (Deprecated)

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
            Constant parameters passed to sub-objectives. (Deprecated)

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
            Constant parameters passed to sub-objectives. (Deprecated)

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
        v = self._feasible_tangents
        df = getattr(self._objective, "jvp_" + op)(v.T, x, constants)
        return df.T

    def jac_scaled(self, x_reduced, constants=None):
        """Compute Jacobian of self.compute_scaled.

        Parameters
        ----------
        x_reduced : ndarray
            Reduced state vector that satisfies linear constraints.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

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
            Constant parameters passed to sub-objectives. (Deprecated)

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
            Constant parameters passed to sub-objectives. (Deprecated)

        Returns
        -------
        J : ndarray
            Jacobian matrix.

        """
        return self._jac(x_reduced, constants, "unscaled")

    def _jvp(self, v, x_reduced, constants=None, op="jvp_scaled"):
        x = self.recover(x_reduced)
        v = self._feasible_tangents @ v
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
            Constant parameters passed to sub-objectives. (Deprecated)

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
            Constant parameters passed to sub-objectives. (Deprecated)

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
            Constant parameters passed to sub-objectives. (Deprecated)

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
            Constant parameters passed to sub-objectives. (Deprecated)

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
            Constant parameters passed to sub-objectives. (Deprecated)

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
            Constant parameters passed to sub-objectives. (Deprecated)

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
        self._solve_during_proximal_build = solve_options.pop(
            "solve_during_proximal_build", True
        )  # If user does not want the solve during build, mainly for debug purposes
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
        # the eq optimizable variables for proximal are the Rb, Zb and profile
        # coefficients. Once these are chosen, we will solve the equilibrium to
        # find the R_lmn, Z_lmn, L_lmn, Ra_n, Za_n. That is why we remove them
        # from the list of optimizable variables. This is accompanied by not including
        # self-consistency constraints (see get_combined_constraint_objectives in
        # desc.optimize.optimizer) and also removing columns corresponding to these
        # variables from the constraint matrix A in
        # desc.objectives.utils.factorize_linear_constraints.
        for arg in ["R_lmn", "Z_lmn", "L_lmn", "Ra_n", "Za_n"]:
            self._args.remove(arg)

        dxdc = []
        xz = {arg: np.zeros(self._eq.dimensions[arg]) for arg in full_args}

        for arg in self._args:
            if arg not in ["Rb_lmn", "Zb_lmn"]:
                x_idx = self._eq.x_idx[arg]
                dxdc.append(np.eye(self._eq.dim_x)[:, x_idx])
            if arg == "Rb_lmn":
                c = get_instance(self._eq_linear_constraints, BoundaryRSelfConsistency)
                # We have A @ R_lmn = Rb_lmn
                A = c.jac_unscaled(xz)[0]["R_lmn"]
                Ainv = np.linalg.pinv(A)
                # Once this is multipled by Rb_lmn, we get the full eq state vector
                # with the R_lmn but rest is 0
                dxdRb = np.eye(self._eq.dim_x)[:, self._eq.x_idx["R_lmn"]] @ Ainv
                dxdc.append(dxdRb)
            if arg == "Zb_lmn":
                c = get_instance(self._eq_linear_constraints, BoundaryZSelfConsistency)
                A = c.jac_unscaled(xz)[0]["Z_lmn"]
                Ainv = np.linalg.pinv(A)
                dxdZb = np.eye(self._eq.dim_x)[:, self._eq.x_idx["Z_lmn"]] @ Ainv
                dxdc.append(dxdZb)
        # dxdc is a matrix that when multiplied by the optimization variables (only
        # Rb_lmn, Zb_lmn) gives the full state vector of the equilibrium (Rb_lmn and
        # Zb_lmn part will be 0, but they will be represented by the equivalent
        # R_lmn and Z_lmn). For example, let's say the eq optimization variables are
        # ceq = [Rb_lmn, Zb_lmn, p_l, i_l].T                      # noqa : E800
        # Then, we will use dxdc for the following:
        # xeq = dxdc @ ceq                                        # noqa : E800
        # And xeq will be,
        # xeq = [                                                 # noqa : E800
        #     R_lmn, Z_lmn, jnp.zeros_like(L_lmn)                 # noqa : E800
        #     jnp.zeros_like(Rb_lmn), jnp.zeros_like(Zb_lmn),     # noqa : E800
        #     p_l, i_l,                                           # noqa : E800
        # ]                                                       # noqa : E800
        self._dxdc = jnp.hstack(dxdc)

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
        timer = Timer()
        timer.start("Proximal projection build")

        self._eq_linear_constraints = get_fixed_boundary_constraints(eq=self._eq)
        self._eq_linear_constraints = maybe_add_self_consistency(
            self._eq, self._eq_linear_constraints
        )

        # we don't always build here because in ~all cases the user doesn't interact
        # with this directly, so if the user wants to manually rebuild they should
        # do it before this wrapper is created for them.
        if not self._objective.built:
            self._objective.build(use_jit=use_jit, verbose=verbose)
        if not self._constraint.built:
            self._constraint.build(use_jit=use_jit, verbose=verbose)

        for constraint in self._eq_linear_constraints:
            constraint.build(use_jit=use_jit, verbose=verbose)

        # Here we create and build the LinearConstraintProjection
        # for the equilibrium subproblem using the self._constraint as objective
        # and our fixed-bdry constraints we just made. This will
        # be passed as the objective for the eq subproblem, which saves
        # some time as by building it here we can avoid re-computing the
        # constraint matrix A and its SVD for the feasible direction method
        self._eq_solve_objective = LinearConstraintProjection(
            self._constraint,
            ObjectiveFunction(self._eq_linear_constraints),
            name="Eq Update LinearConstraintProjection",
        )
        self._eq_solve_objective.build(use_jit=use_jit, verbose=verbose)

        errorif(
            self._constraint.things != [self._eq],
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

        # the full state vector includes all the parameters from all the things
        # however, sub-objectives only need the part for their thing. We will
        # use this to split the state vector into its components
        self._dimx_per_thing = [t.dim_x for t in self.things]
        # we remove the R_lmn, Z_lmn, L_lmn, Ra_n, Za_n from the equilibrium params
        # dimc_per_thing accounts for that, don't confuse it with reduced state vector
        self._dimc_per_thing = [t.dim_x for t in self.things]
        self._dimc_per_thing[self._eq_idx] = int(
            np.sum([self._eq.dimensions[arg] for arg in self._args])
        )
        # we will need to set this static attribute, only possible if tuple
        self._dimc_per_thing = tuple(self._dimc_per_thing)

        ## history and caching
        # first, ensure equilibrium is solved to the
        # specified tolerances, necessary as we assume
        # eq is solved when taking the derivatives later
        if self._solve_during_proximal_build:
            self._eq.solve(
                objective=self._eq_solve_objective,
                constraints=None,
                **self._solve_options,
            )
        # then store the now-solved eq state as the initial state
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

        xs = jnp.split(x, np.cumsum(self._dimc_per_thing))
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
        """Return the full state vector from the Optimizable objects things.

        Note that we remove the R_lmn, Z_lmn, L_lmn, Ra_n, Za_n from the equilibrium
        params.
        """
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
        """int: Dimension of the state vector.

        Note that we remove the R_lmn, Z_lmn, L_lmn, Ra_n, Za_n from the equilibrium
        params.
        """
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
            New values of the state vector of equilibrium (except R_lmn, Z_lmn,
            L_lmn, Ra_n, Za_n) and all the parameters of the other things.
        store : bool
            Whether the new x should be stored in self.history

        Notes
        -----
        After updating, if store=False, self._eq will revert back to the previous
        solution when store was True

        """
        # xopt is the full state vector of all the things
        # xeq is the full state vector of the equilibrium only

        # TODO (#1720): We don't need to check the whole state vector, just the
        # equilibrium parameters should be enough.
        # first check if its something we've seen before, if it is just return
        # cached value, no need to perturb + resolve
        xopt = f_where_x(x, self._allx, self._allxopt)
        xeq = f_where_x(x, self._allx, self._allxeq)
        if xopt.size > 0 and xeq.size > 0:
            pass
        else:
            # After unpack_state, R_lmn, Z_lmn, L_lmn, Ra_n and Za_n in below lists
            # will be 0s
            x_list = self.unpack_state(x, False)
            x_list_old = self.unpack_state(self._x_old, False)
            xeq_dict = x_list[self._eq_idx]
            xeq_dict_old = x_list_old[self._eq_idx]
            deltas = {str(key): xeq_dict[key] - xeq_dict_old[key] for key in xeq_dict}
            # We pass in the LinearConstraintProjection object to skip some redundant
            # computations in the perturb and solve methods
            self._eq = self._eq.perturb(
                objective=self._eq_solve_objective,
                constraints=None,
                deltas=deltas,
                **self._perturb_options,
            )
            self._eq.solve(
                objective=self._eq_solve_objective,
                constraints=None,
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
            self._eq_solve_objective.update_constraint_target(self._eq)

        return xopt, xeq

    def compute_scaled(self, x, constants=None):
        """Compute the objective function and apply weights/normalization.

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        Returns
        -------
        f : ndarray
            Objective function value(s).

        """
        constants = setdefault(constants, [None, None])
        xopt, _ = self._update_equilibrium(x, store=False)
        return self._objective.compute_scaled(xopt, constants[0])

    def compute_scaled_error(self, x, constants=None):
        """Compute the error between target and objective and apply weights etc.

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        Returns
        -------
        f : ndarray
            Objective function value(s).

        """
        constants = setdefault(constants, [None, None])
        xopt, _ = self._update_equilibrium(x, store=False)
        return self._objective.compute_scaled_error(xopt, constants[0])

    def compute_scalar(self, x, constants=None):
        """Compute the sum of squares error.

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

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
            Constant parameters passed to sub-objectives. (Deprecated)

        Returns
        -------
        f : ndarray
            Objective function value(s).

        """
        constants = setdefault(constants, [None, None])
        xopt, _ = self._update_equilibrium(x, store=False)
        return self._objective.compute_unscaled(xopt, constants[0])

    def grad(self, x, constants=None):
        """Compute gradient of self.compute_scalar.

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        Returns
        -------
        g : ndarray
            gradient vector.

        """
        # We are looking for the gradient of L = 0.5 * Gᵀ @ G
        # Then, the gradient is ∇L = Gᵀ @ J_of_G
        # where J_of_G is the Jacobian of G with respect to the optimization variables
        # We explained getting J_of_G in the _jvp method. It is basically,
        # J_of_G = ∇G @ [dc_tangents - (∇F @ dx_tangents)⁻¹ @ (∇F @ dc_tangents)]
        # where ∇G is the Jacobian of G with respect to full state vector
        # and ∇F is the Jacobian of F with respect to full state vector. Then,
        # ∇L = Gᵀ @ ∇G @ [dc_tangents - (∇F @ dx_tangents)⁻¹ @ (∇F @ dc_tangents)]
        # We get the part in [] using the _proximal_get_tangents.
        v = jnp.eye(x.shape[0])
        constants = setdefault(constants, [None, None])
        xg, xf = self._update_equilibrium(x, store=True)
        tangents = _proximal_get_tangents(
            self._constraint,
            xf,
            v,
            constants[1],
            self._eq_solve_objective._feasible_tangents,
            self._dxdc,
            self._dimc_per_thing,
            self._eq_idx,
            "scaled_error",
        )
        g = self._objective.compute_scaled_error(xg, constants[0])
        g_vjp = self._objective.vjp_scaled_error(g, xg, constants[0])
        return tangents @ g_vjp

    def hess(self, x, constants=None):
        """Compute Hessian of self.compute_scalar.

        Uses the "small residual approximation" where the Hessian is replaced by
        the square of the Jacobian: H = J.T @ J

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

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
            Constant parameters passed to sub-objectives. (Deprecated)

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
            Constant parameters passed to sub-objectives. (Deprecated)

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
            Constant parameters passed to sub-objectives. (Deprecated)

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
            Constant parameters passed to sub-objectives. (Deprecated)

        """
        op = "scaled"
        return self._jvp(v, x, constants, op)

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
            Constant parameters passed to sub-objectives. (Deprecated)

        """
        op = "scaled_error"
        return self._jvp(v, x, constants, op)

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
            Constant parameters passed to sub-objectives. (Deprecated)

        """
        op = "unscaled"
        return self._jvp(v, x, constants, op)

    def _jvp(self, v, x, constants=None, op="scaled_error"):
        # The goal is to compute the Jacobian of the objective function with respect to
        # the optimization variables (c). Before taking the Jacobian, we update the
        # equilibrium such that
        # F(x+dx, c+dc) = 0 = F(x, c) + dF/dx * dx + dF/dc * dc
        # so that we can set F(x, c) = 0, from here we can solve for dx and get
        # dx = - (dF/dx)⁻¹ * dF/dc * dc     # noqa : E800
        # We can then compute the Jacobian of the objective function with respect to c
        # G(x+dx, c+dc) = G(x, c) + dG/dx * dx + dG/dc * dc
        # substituting in dx we get
        # G(x+dx, c+dc) = G(x, c) + [ dG/dc - dG/dx * (dF/dx)⁻¹ * dF/dc ] * dc
        # and the Jacobian we want is dG/dc - dG/dx * (dF/dx)⁻¹ * dF/dc

        # Note: This Jacobian can be obtained using JVPs in proper tangent directions.
        # First we will compute the tangent direction (see _proximal_get_tangents),
        # then we will compute the Jacobian.
        v = v[0] if isinstance(v, (tuple, list)) else v
        constants = setdefault(constants, [None, None])
        xg, xf = self._update_equilibrium(x, store=True)

        # we don't need to divide this part into blocked and batched because
        # self._constraint._deriv_mode will handle it
        tangents = _proximal_get_tangents(
            self._constraint,
            xf,
            v,
            constants[1],
            self._eq_solve_objective._feasible_tangents,
            self._dxdc,
            self._dimc_per_thing,
            self._eq_idx,
            op,
        )

        if self._objective._deriv_mode == "batched":
            # objective's method already know about its jac_chunk_size
            return getattr(self._objective, "jvp_" + op)(tangents, xg, constants[0])
        else:
            return _proximal_jvp_blocked_pure(
                self._objective,
                jnp.split(tangents, np.cumsum(self._dimx_per_thing), axis=-1),
                jnp.split(xg, np.cumsum(self._dimx_per_thing)),
                op,
            )

    @property
    def constants(self):
        """list: constant parameters for each sub-objective."""
        warnif(
            True,
            FutureWarning,
            "constants is deprecated and will be removed in a future "
            "release. Users should not include constants in the arguments "
            "of their objective compute methods. Instead declare all the "
            "constants in the build method and use as obj._constants.",
        )
        return [self._objective.constants, self._constraint.constants]

    def __getattr__(self, name):
        """For other attributes we defer to the base objective."""
        return getattr(self._objective, name)


# in ProximalProjection we have an explicit state that we keep track of (and add
# to as we go) meaning if we jit anything with self static it doesn't update
# correctly, while if we leave self unstatic then it recompiles every time because
# the pytree structure of ProximalProjection is changing. To get around that we
# define these helper functions that are stateless so we can safely jit them


def jit_if_possible(func=None, *, static_argnames=("op",)):
    """Jit a function if use_jit."""
    if func is None:
        return functools.partial(jit_if_possible, static_argnames=static_argnames)
    jitted_func = functools.partial(jit, static_argnames=list(static_argnames))(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # first arg has to be ObjectiveFunction
        obj = args[0]
        if getattr(obj, "_use_jit", False):
            return jitted_func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return wrapper


@jit_if_possible(static_argnames=("dimc_per_thing", "eq_idx", "op"))
def _proximal_get_tangents(
    constraint,
    xf,
    v,
    constants,
    eq_feasible_tangents,
    dxdc,
    dimc_per_thing,
    eq_idx,
    op="scaled_error",
):
    # We try to find dG/dc - dG/dx * (dF/dx)⁻¹ * dF/dc
    # where G is the objective function. Since DESC stores x and c in the same
    # vector, instead of multiple JVP calls, we will just find a tangent direction
    # that will give us the same result.
    # For making the explanation clear, assume J is the Jacobian of the objective
    # function with respect to the full state vector (both x and c). Then,
    # dG/dc = J @ (tangent vectors in c direction)
    # dG/dx = J @ (tangent vectors in x direction)
    # So, dG/dc - dG/dx * (dF/dx)⁻¹ * dF/dc can be written as
    # J @ [(tangent vectors in c direction) - (tangent vectors in x direction)@dfdc]
    # Note: We will never form full Jacobian J, we will just compute the above
    # expression by JVPs.

    # v contains prox._args DoFs from eq and other objects (like coils, surfaces
    # etc). Only the eq block changes when the equilibrium is re-solved.
    vs = jnp.split(v, np.cumsum(dimc_per_thing)[:-1], axis=-1)
    # JVPs are taken for the dxdc @ v tangents, but at most dimc of them are useful,
    # so take whichever combination needs fewer of them. Applying v after the JVPs
    # only pays off with a lot of coil, surface etc DoFs, ie. single stage.
    if vs[eq_idx].ndim == 2 and vs[eq_idx].shape[0] > dimc_per_thing[eq_idx]:
        eq_tangents = vs[eq_idx] @ _proximal_eq_tangents(
            constraint, xf, constants, eq_feasible_tangents, dxdc.T, op
        )
    else:
        dxdcv = vs[eq_idx] @ dxdc.T
        # atleast_2d and reshape are to also handle a single (1D) direction
        eq_tangents = _proximal_eq_tangents(
            constraint, xf, constants, eq_feasible_tangents, jnp.atleast_2d(dxdcv), op
        )
        eq_tangents = eq_tangents.reshape(dxdcv.shape)
    return jnp.concatenate([*vs[:eq_idx], eq_tangents, *vs[eq_idx + 1 :]], axis=-1)


@jit_if_possible
def _proximal_eq_tangents(
    constraint, xf, constants, eq_feasible_tangents, dxdcv, op="scaled_error"
):
    # Note: dxdcv holds the directions in c, mapped to the full eq state vector, as
    # rows. It is either dxdc.T or v @ dxdc.T, the return has the same shape.

    # here Fxh is dF/dx in the reduced (feasible) eq coordinates and Fc is dF/dc. A
    # single batched JVP gives both, so the SVD below is computed once by
    # construction, instead of relying on the compiler to hoist it out of a loop.
    # Our compute functions never include variables like Rb_lmn, Zb_lmn etc. So,
    # taking the JVP in just dc direction will give 0. To prevent this, we use dxdc
    # which is the dx/dc matrix and convert the Rb_lmn to R_lmn entries etc.
    # For example, if we want the derivative wrt Rb_023, we should take the derivative
    # wrt all R_lmn coefficients that contribute to Rb_023. See BoundaryRSelfConsistency
    # for the relation between Rb_lmn and R_lmn.
    dim_x_reduced = eq_feasible_tangents.shape[-1]
    tangents = jnp.concatenate([eq_feasible_tangents.T, dxdcv], axis=0)
    J = getattr(constraint, "jvp_" + op)(tangents, xf, constants)
    Fxh, Fc = J[:dim_x_reduced].T, J[dim_x_reduced:].T
    cutoff = jnp.finfo(Fxh.dtype).eps * max(Fxh.shape)
    uf, sf, vtf = jnp.linalg.svd(Fxh, full_matrices=False)
    sf += sf[-1]  # add a tiny bit of regularization
    sfi = jnp.where(sf < cutoff * sf[0], 0, 1 / sf)
    # this is (dF/dx)⁻¹ @ dF/dc for all the directions at once  # noqa : E800
    dfdc = vtf.T @ (sfi[:, None] * (uf.T @ Fc))
    # feasible_tangents maps the reduced eq state vector back to the full one
    return dxdcv - (eq_feasible_tangents @ dfdc).T


@jit_if_possible
def _proximal_jvp_blocked_pure(objective, vgs, xgs, op):
    # Note: This function is not vectorized and takes the full set of tangents, and
    # returns a matrix.

    # vgs and xgs are list of arrays (each element of the list is not same size
    # necessarily), that are split by the things in the objective. If there are multiple
    # things for the ObjectiveFunction, each split belongs to a different thing. The
    # information about which thing is used by which sub-objective is stored in
    # _things_per_objective_idx.

    # Note: This function is very similar to _jvp_blocked in ObjectiveFunction with
    # some naming differences to account for ProximalProjection.
    out = []
    for k, obj in enumerate(objective.objectives):
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
            Ji = getattr(obj, "jac_" + op)(*xi)
            outi = jnp.array([Jii @ vii.T for Jii, vii in zip(Ji, vi)]).sum(axis=0)
            out.append(outi)
        else:
            outi = getattr(obj, "jvp_" + op)([_vi for _vi in vi], xi).T
            out.append(outi)
    return jnp.concatenate(out).T


# Equilibrium parameters that are unknowns of the free boundary problem. Given the
# external field and the profiles etc., they are found by solving the free boundary
# conditions instead of being chosen by the optimizer.
_FREE_BOUNDARY_ARGS = ["Rb_lmn", "Zb_lmn", "I", "G", "Phi_mn"]
# Equilibrium parameters found by solving force balance for a given boundary.
_INTERIOR_ARGS = ["R_lmn", "Z_lmn", "L_lmn", "Ra_n", "Za_n"]


class ProximalProjectionFreeBoundary(ProximalProjection):
    """Remove free boundary equilibrium constraints by projecting at each step.

    This is the free boundary analog of ``ProximalProjection``. The optimization
    variables are only the inputs to the free boundary equilibrium problem: the
    parameters of the external field (eg. coil shapes and currents), the profiles,
    Psi, and the parameters of any other things. The interior of the equilibrium is
    determined by force balance (as in ``ProximalProjection``), and the boundary shape
    (and sheet current, if the equilibrium has one) is determined by the free boundary
    conditions.

    At each iteration, after a step is taken to reduce the objective, the boundary is
    predicted to first order, the equilibrium is perturbed and re-solved, and then the
    free boundary problem is re-solved to bring it back into free boundary equilibrium.

    The derivatives are found by nesting the implicit function theorem. With F the
    force balance error, B the free boundary error, x the interior of the equilibrium,
    b the boundary, q the optimization variables and G the objective:

    - dx/dc = -(∂F/∂x)⁺ ∂F/∂c for any input c to the fixed boundary problem, which
      gives the total derivatives (at fixed boundary) dB/db, dB/dq, dG/db and dG/dq.
    - db/dq = -(dB/db)⁺ dB/dq, which uses the Gauss-Newton approximation to the
      optimality condition of the least squares free boundary problem.
    - The Jacobian is then dG/dq + dG/db db/dq.

    Parameters
    ----------
    objective : ObjectiveFunction
        Objective function to optimize.
    constraint : ObjectiveFunction
        Free boundary equilibrium constraints to enforce. Should contain one or more
        equilibrium objectives {ForceBalance, CurrentDensity, RadialForceBalance,
        HelicalForceBalance}, and one or more free boundary objectives
        {BoundaryError, VacuumBoundaryError}. If the external field is being
        optimized, the free boundary objectives should be built with
        ``field_fixed=False``.
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the objectives.
    perturb_options, solve_options : dict
        dictionary of arguments passed to Equilibrium.perturb and Equilibrium.solve
        during the projection step.
    free_boundary_options : dict
        Options for the free boundary part of the projection step. Can contain:

        - ``"constraints"`` : tuple of linear constraints on the free boundary
          unknowns, eg. ``FixBoundaryR(eq, modes=...)`` to only allow some boundary
          modes to change. These are found automatically when passed as constraints
          to ``Optimizer.optimize``.
        - ``"predict"`` : bool, whether to predict the boundary change from the
          derivatives as the initial guess for the free boundary solve. Default True.
        - ``"predict_order"`` : int, 1 or 2, order of the boundary prediction. The
          second order term needs a few second order JVPs along the step, but no new
          factorizations. Default 1.
        - ``"predict_tr_ratio"`` : float, the second order term is found with a trust
          region of this size relative to the first order term, which keeps it from
          being dominated by poorly determined directions of dB/db. Default 0.1.
        - ``"rcond"`` : float, relative cutoff for small singular values when
          inverting dB/db. Defaults to machine precision times the largest dimension.
        - ``"maxiter"``, ``"ftol"``, ``"xtol"``, ``"gtol"``, ``"x_scale"``,
          ``"verbose"``, ``"options"`` : passed to ``desc.optimize.lsqtr`` when solving
          the free boundary problem. Defaults are ``maxiter=20``, ``ftol=1e-2``,
          ``xtol=1e-6``, ``gtol=1e-8``, ``x_scale="jac"``, ``verbose=0``. The free
          boundary problem is warm started from the previous solution and the
          predicted boundary at each step, so it doesn't need to be solved tightly.
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
        free_boundary_options=None,
        name="ProximalProjectionFreeBoundary",
    ):
        errorif(
            not isinstance(constraint, ObjectiveFunction),
            ValueError,
            "constraint should be instance of ObjectiveFunction.",
        )
        eq_cons = [con for con in constraint.objectives if con._equilibrium]
        fb_cons = [con for con in constraint.objectives if con._free_boundary]
        others = [
            con
            for con in constraint.objectives
            if con not in eq_cons and con not in fb_cons
        ]
        errorif(
            len(others) > 0,
            ValueError,
            "ProximalProjectionFreeBoundary method cannot handle general "
            + f"nonlinear constraints {others}.",
        )
        errorif(
            len(eq_cons) == 0,
            ValueError,
            "ProximalProjectionFreeBoundary needs an equilibrium constraint such as "
            + "ForceBalance.",
        )
        errorif(
            len(fb_cons) == 0,
            ValueError,
            "ProximalProjectionFreeBoundary needs a free boundary constraint such as "
            + "BoundaryError or VacuumBoundaryError.",
        )
        for con in fb_cons:
            errorif(
                con.bounds is not None,
                ValueError,
                "ProximalProjectionFreeBoundary can only handle equality constraints, "
                + f"got bounds for constraint {con}",
            )
        super().__init__(
            objective,
            ObjectiveFunction(eq_cons, use_jit=constraint.use_jit),
            eq,
            perturb_options=perturb_options,
            solve_options=solve_options,
            name=name,
        )
        self._constraint_fb = ObjectiveFunction(fb_cons, use_jit=constraint.use_jit)

        free_boundary_options = (
            {} if free_boundary_options is None else free_boundary_options.copy()
        )
        fb_constraints = free_boundary_options.pop("constraints", ())
        self._fb_linear_constraints = tuple(fb_constraints)
        self._fb_predict = free_boundary_options.pop("predict", True)
        self._fb_predict_order = free_boundary_options.pop("predict_order", 1)
        self._fb_predict_tr_ratio = free_boundary_options.pop("predict_tr_ratio", 0.1)
        errorif(
            self._fb_predict_order not in [1, 2],
            ValueError,
            f"predict_order should be 1 or 2, got {self._fb_predict_order}",
        )
        self._fb_rcond = free_boundary_options.pop("rcond", None)
        free_boundary_options.setdefault("maxiter", 20)
        free_boundary_options.setdefault("ftol", 1e-2)
        free_boundary_options.setdefault("xtol", 1e-6)
        free_boundary_options.setdefault("gtol", 1e-8)
        free_boundary_options.setdefault("x_scale", "jac")
        free_boundary_options.setdefault("verbose", 0)
        free_boundary_options.setdefault("options", {})
        self._fb_solve_options = free_boundary_options

    def _separate_free_boundary_constraints(self, constraints):
        """Take the linear constraints on the free boundary unknowns from constraints.

        The boundary is not an optimization variable, so linear constraints on it (eg.
        FixBoundaryR to only allow some modes to change) are instead enforced when
        solving the free boundary problem.

        Parameters
        ----------
        constraints : tuple of _Objective
            Linear constraints passed to the optimizer.

        Returns
        -------
        constraints : tuple of _Objective
            The constraints that do not act on the free boundary unknowns.

        """
        fb_cons, rest = [], []
        for con in constraints:
            if len(con.things) != 1 or con.things[0] is not self._eq:
                rest.append(con)
                continue
            if not con.built:
                # same as ObjectiveFunction.build would do later
                con.build(use_jit=True, verbose=0)
            J = con.jac_unscaled(*con.xs(self._eq))[0]
            args = {arg for arg, Ji in J.items() if np.any(np.asarray(Ji) != 0)}
            if not args & set(_FREE_BOUNDARY_ARGS) or args & set(_INTERIOR_ARGS):
                # self-consistency etc. constraints that touch the interior are
                # handled by the fixed boundary problem
                rest.append(con)
                continue
            errorif(
                len(args - set(_FREE_BOUNDARY_ARGS)) > 0,
                ValueError,
                f"Linear constraint {con} acts on both the free boundary unknowns "
                + f"{_FREE_BOUNDARY_ARGS} and the optimization variables, which is "
                + "not supported with ProximalProjectionFreeBoundary.",
            )
            fb_cons.append(con)
        self._fb_linear_constraints = self._fb_linear_constraints + tuple(fb_cons)
        return tuple(rest)

    def _set_things(self, things=None):
        old_things = getattr(self, "_things", None)
        super()._set_things(things)
        errorif(
            self._built
            and (
                len(old_things) != len(self._things)
                or any(t1 is not t2 for t1, t2 in zip(old_things, self._things))
            ),
            ValueError,
            "Cannot add things to ProximalProjectionFreeBoundary after it is built, "
            + "all things should be used by the objective or nonlinear constraints.",
        )
        # the objective is evaluated with the state of all the things, eg. coils may
        # only enter through the free boundary constraint
        self._objective._set_things(self._things)

    def _set_eq_state_vector(self):
        super()._set_eq_state_vector()
        eq = self._eq
        # inputs to the fixed boundary problem, called c in ProximalProjection
        self._c_args = self._args
        self._fb_args = [arg for arg in self._c_args if arg in _FREE_BOUNDARY_ARGS]
        # optimization variables
        self._args = [arg for arg in self._c_args if arg not in _FREE_BOUNDARY_ARGS]

        c_idx = {}
        offset = 0
        for arg in self._c_args:
            c_idx[arg] = np.arange(offset, offset + eq.dimensions[arg])
            offset += eq.dimensions[arg]
        self._c_idx_fb = np.concatenate([c_idx[arg] for arg in self._fb_args])
        self._c_idx_q = np.concatenate(
            [np.array([], dtype=int)] + [c_idx[arg] for arg in self._args]
        )
        self._x_idx_fb = np.concatenate([eq.x_idx[arg] for arg in self._fb_args])

        # ProximalProjection maps Rb_lmn, Zb_lmn to only R_lmn, Z_lmn since the
        # equilibrium compute functions don't use Rb_lmn, Zb_lmn directly. Here we also
        # keep them, since BoundaryError uses them for the sheet current.
        dxdc = np.array(self._dxdc)
        for arg in ["Rb_lmn", "Zb_lmn"]:
            dxdc[eq.x_idx[arg], c_idx[arg]] = 1
        self._dxdc = jnp.asarray(dxdc)

    def _set_free_boundary_constraints(self):
        """Factorize linear constraints on the free boundary unknowns b.

        Feasible values are b = bp + Z y where y is unconstrained and Z has orthonormal
        columns.
        """
        dim_b = self._x_idx_fb.size
        for con in self._fb_linear_constraints:
            errorif(
                len(con.things) != 1 or con.things[0] is not self._eq,
                ValueError,
                f"Free boundary constraint {con} should only act on the equilibrium.",
            )
        if len(self._fb_linear_constraints):
            con = ObjectiveFunction(self._fb_linear_constraints)
            con.build(verbose=0)
            x0 = jnp.zeros(con.dim_x)
            A = con.jac_scaled(x0)[:, self._x_idx_fb]
            rhs = -con.compute_scaled_error(x0)
            Ainv, Z = svd_inv_null(A)
            bp = Ainv @ rhs
        else:
            Z = jnp.eye(dim_b)
            bp = jnp.zeros(dim_b)
        errorif(
            Z.shape[1] == 0,
            ValueError,
            "All of the free boundary unknowns are fixed by linear constraints.",
        )
        self._fb_Z = Z
        self._fb_bp = bp

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
        timer = Timer()
        timer.start("Proximal projection build")
        eq = self._eq

        self._eq_linear_constraints = get_fixed_boundary_constraints(eq=eq)
        self._eq_linear_constraints = maybe_add_self_consistency(
            eq, self._eq_linear_constraints
        )
        for obj in [self._objective, self._constraint, self._constraint_fb]:
            if not obj.built:
                obj.build(use_jit=use_jit, verbose=verbose)
        for constraint in self._eq_linear_constraints:
            constraint.build(use_jit=use_jit, verbose=verbose)

        # objective for solving the fixed boundary problem, see ProximalProjection
        self._eq_solve_objective = LinearConstraintProjection(
            self._constraint,
            ObjectiveFunction(self._eq_linear_constraints),
            name="Eq Update LinearConstraintProjection",
        )
        self._eq_solve_objective.build(use_jit=use_jit, verbose=verbose)

        errorif(
            self._constraint.things != [eq],
            ValueError,
            "ProximalProjectionFreeBoundary can only handle equilibrium constraints "
            + "on the equilibrium.",
        )
        errorif(
            not any(t is eq for t in self._constraint_fb.things),
            ValueError,
            "Free boundary constraints should be on the equilibrium being optimized.",
        )

        self._built = False
        self._objectives = [self._objective, self._constraint, self._constraint_fb]
        self._set_things()
        self._check_field_fixed()
        self._eq_idx = self.things.index(eq)
        # indices of the things the free boundary constraints depend on
        self._fb_things_idx = [
            [i for i, t in enumerate(self.things) if t is t_fb][0]
            for t_fb in self._constraint_fb.things
        ]

        self._dim_f = self._objective.dim_f
        self._scalar = self._dim_f == 1

        self._set_eq_state_vector()
        self._set_free_boundary_constraints()

        # full state vector of each thing
        self._dimx_per_thing = [t.dim_x for t in self.things]
        # optimization variables of each thing, for the equilibrium we remove the
        # interior and the free boundary unknowns
        self._dimc_per_thing = [t.dim_x for t in self.things]
        self._dimc_per_thing[self._eq_idx] = int(
            np.sum([eq.dimensions[arg] for arg in self._args])
        )
        self._dimc_per_thing = tuple(self._dimc_per_thing)

        # derivatives of the free boundary problem at the most recent state
        self._linearization = None
        # (x, v, db, linearization) from the last jvp, where db is the boundary change
        # for each direction v in the optimization variables, for predicting the new
        # boundary
        self._dbdv = None

        if self._solve_during_proximal_build:
            # derivatives assume we start from a free boundary equilibrium
            eq.solve(
                objective=self._eq_solve_objective,
                constraints=None,
                **self._solve_options,
            )
            self._solve_free_boundary([t.params_dict for t in self.things])
        self._x_old = self.x(*self.things)
        self._allx = [self._x_old]
        self._allxopt = [self._objective.x(*self.things)]
        self._allxeq = [eq.pack_params(eq.params_dict)]
        self.history = [[t.params_dict.copy() for t in self.things]]

        self._built = True
        timer.stop("Proximal projection build")
        if verbose > 1:
            timer.disp("Proximal projection build")

    def _check_field_fixed(self):
        # if the external field is optimized, the free boundary constraints need to
        # know about it, otherwise we miss how the boundary changes with the field
        for con in self._constraint_fb.objectives:
            for field in getattr(con, "_field", []):
                errorif(
                    any(field is t for t in self.things)
                    and not any(field is t for t in con.things),
                    ValueError,
                    f"The external field of {con} is being optimized, so it should be "
                    + "built with field_fixed=False.",
                )

    def _xopt(self, xeq, x_list):
        """Full state vector of all things from eq state xeq and other params."""
        return jnp.concatenate(
            [
                xeq if i == self._eq_idx else t.pack_params(x_list[i])
                for i, t in enumerate(self.things)
            ]
        )

    def _fb_x(self, xopt):
        """State vector for the free boundary constraints from the full state."""
        xs = jnp.split(xopt, np.cumsum(self._dimx_per_thing)[:-1], axis=-1)
        return jnp.concatenate([xs[i] for i in self._fb_things_idx], axis=-1)

    def _get_b(self):
        """Free boundary unknowns of the equilibrium as a single vector."""
        return jnp.concatenate(
            [jnp.atleast_1d(self._eq.params_dict[arg]) for arg in self._fb_args]
        )

    def _split_b(self, b):
        """Split vector of free boundary unknowns into dict by parameter name."""
        splits = np.cumsum([self._eq.dimensions[arg] for arg in self._fb_args])[:-1]
        return dict(zip(self._fb_args, jnp.split(b, splits)))

    def _linearize(self, xopt):
        """Compute derivatives of the free boundary problem at the current state.

        Parameters
        ----------
        xopt : ndarray
            Full state vector of all the things. The equilibrium should be in force
            balance at this state.

        Returns
        -------
        linearization : dict
            Containing ``"T_c"``, the change in the full equilibrium state vector
            (with force balance maintained) for unit changes in each of the inputs to
            the fixed boundary problem, ``"J_Bb"``, the derivative of the free boundary
            error wrt the free boundary unknowns, and ``"P"``, the pseudo-inverse of
            ``J_Bb @ Z`` where Z is the null space of the linear constraints on the
            free boundary unknowns, ``"JZ_svd"``, the SVD (u, s, vt) of ``J_Bb @ Z``,
            and ``"F_factors"``, the SVD (u, 1/s, vt) of dF/dx in the reduced
            equilibrium coordinates.

        """
        lin = self._linearization
        if lin is not None and np.array_equal(lin["x"], xopt):
            return lin
        xeq = jnp.split(xopt, np.cumsum(self._dimx_per_thing)[:-1])[self._eq_idx]
        # this is dx/dc = dxdc - (dF/dx)⁺ dF/dc as rows, for all c at once, and the
        # SVD of dF/dx in the reduced eq coordinates for the second order prediction
        T_c, F_factors = _fb_eq_tangents(
            self._constraint,
            xeq,
            self._eq_solve_objective._feasible_tangents,
            self._dxdc.T,
        )
        T_b = T_c[self._c_idx_fb]
        J_Bb = self._constraint_fb.jvp_scaled(
            self._fb_tangents(T_b), self._fb_x(xopt)
        ).T
        P, JZ_svd = _pinv(J_Bb @ self._fb_Z, self._fb_rcond)
        self._linearization = {
            "x": xopt,
            "T_c": T_c,
            "J_Bb": J_Bb,
            "P": P,
            "JZ_svd": JZ_svd,
            "F_factors": F_factors,
        }
        return self._linearization

    def _fb_tangents(self, eq_tangents, tangents_per_thing=None):
        """Assemble tangents for the free boundary constraints.

        Parameters
        ----------
        eq_tangents : ndarray
            Tangents for the equilibrium as rows.
        tangents_per_thing : list of ndarray, optional
            Tangents for all of the things. If None, all except the equilibrium are 0.

        """
        n = eq_tangents.shape[0]
        out = []
        for i in self._fb_things_idx:
            if i == self._eq_idx:
                out.append(eq_tangents)
            elif tangents_per_thing is None:
                out.append(jnp.zeros((n, self._dimx_per_thing[i])))
            else:
                out.append(tangents_per_thing[i])
        return jnp.concatenate(out, axis=-1)

    def _solve_free_boundary(self, x_list):
        """Solve the free boundary problem with the external field etc. fixed.

        Changes self._eq in place. The equilibrium should start in force balance.

        Parameters
        ----------
        x_list : list of dict
            Parameters of each thing. The equilibrium entry is not used, the current
            state of self._eq is used instead.

        """
        eq = self._eq
        Z, bp = self._fb_Z, self._fb_bp
        # (y, eq params) for each evaluated point, lsqtr only asks for the Jacobian at
        # points it has evaluated the function at
        cache = []
        # perturbations are taken from the last point the Jacobian was evaluated at,
        # which is the last accepted point
        anchor = {"b": self._get_b(), "params": eq.params_dict.copy()}

        def goto(y):
            for yi, params in reversed(cache):
                if np.array_equal(yi, y):
                    eq.params_dict = params
                    self._eq_solve_objective.update_constraint_target(eq)
                    return
            eq.params_dict = anchor["params"]
            self._eq_solve_objective.update_constraint_target(eq)
            db = bp + Z @ y - anchor["b"]
            if np.any(db != 0):
                eq.perturb(
                    objective=self._eq_solve_objective,
                    constraints=None,
                    deltas=self._split_b(db),
                    **self._perturb_options,
                )
                eq.solve(
                    objective=self._eq_solve_objective,
                    constraints=None,
                    **self._solve_options,
                )
            cache.append((y, eq.params_dict.copy()))

        def xopt():
            return self._xopt(eq.pack_params(eq.params_dict), x_list)

        def fun(y):
            goto(y)
            return self._constraint_fb.compute_scaled_error(self._fb_x(xopt()))

        def jac(y):
            goto(y)
            anchor["b"], anchor["params"] = self._get_b(), eq.params_dict.copy()
            return self._linearize(xopt())["J_Bb"] @ Z

        options = self._fb_solve_options.copy()
        options["options"] = options["options"].copy()
        y0 = Z.T @ (self._get_b() - bp)
        result = lsqtr(fun, y0, jac, **options)
        goto(result["x"])
        self._fb_result = result
        return result

    def _predict_boundary(self, x):
        """Predict the change in the free boundary unknowns from x_old to x.

        Uses the derivatives from the last jvp at x_old, which are known in the
        directions v of that jvp (usually the full space, or the null space of the
        linear constraints which contains x - x_old).

        Returns
        -------
        db : ndarray or None
            Predicted change in the free boundary unknowns, or None if no prediction
            is available.

        """
        if (
            not self._fb_predict
            or self._dbdv is None
            or not np.array_equal(self._dbdv[0], self._x_old)
        ):
            return None
        _, v, db, lin = self._dbdv
        dq = np.asarray(x - self._x_old)
        a = np.linalg.lstsq(v.T, dq, rcond=None)[0]
        b1 = db.T @ a
        if self._fb_predict_order == 1:
            return b1
        return b1 + self._second_order_boundary(dq, b1, lin)

    def _second_order_boundary(self, dq, b1, lin):
        """Second order term of the boundary change along the step dq.

        With t the first order change of the full state (force balance and free
        boundary conditions maintained), the fixed boundary second order state change
        is x2 = -(dF/dx)⁺ ½ F''[t, t], and the second order boundary change solves
        dB/db b2 = -(½ B''[t, t] + B' x2) in the least squares sense, with
        |b2| <= predict_tr_ratio |b1|. Without the trust region, components of b1 in
        poorly determined directions of dB/db get amplified again and b2 can be larger
        than b1, which makes the prediction worse.

        Parameters
        ----------
        dq : ndarray
            Step in the optimization variables.
        b1 : ndarray
            First order change in the free boundary unknowns for dq.
        lin : dict
            Linearization at the start of the step, from self._linearize.

        """
        T_c, (uf, sfi, vtf), xopt = lin["T_c"], lin["F_factors"], lin["x"]
        splits = np.cumsum(self._dimx_per_thing)[:-1]
        t = jnp.split(jnp.asarray(dq), np.cumsum(self._dimc_per_thing)[:-1])
        t[self._eq_idx] = (
            t[self._eq_idx] @ T_c[self._c_idx_q] + b1 @ T_c[self._c_idx_fb]
        )
        t_eq = t[self._eq_idx]
        xeq = jnp.split(xopt, splits)[self._eq_idx]
        F2 = self._constraint.jvp_scaled((t_eq, t_eq), xeq)
        x2 = -self._eq_solve_objective._feasible_tangents @ (
            vtf.T @ (sfi * (uf.T @ (F2 / 2)))
        )
        xB = self._fb_x(xopt)
        tB = self._fb_tangents(t_eq[None], [ti[None] for ti in t])[0]
        B2 = self._constraint_fb.jvp_scaled((tB, tB), xB)
        Bx2 = self._constraint_fb.jvp_scaled(self._fb_tangents(x2[None])[0], xB)
        u, s, vt = lin["JZ_svd"]
        y1 = self._fb_Z.T @ b1
        y2, _, _ = trust_region_step_exact_svd(
            B2 / 2 + Bx2, u, s, vt.T, self._fb_predict_tr_ratio * jnp.linalg.norm(y1)
        )
        return self._fb_Z @ y2

    def _update_equilibrium(self, x, store=False):
        """Update the internal equilibrium with new field, profiles etc.

        Parameters
        ----------
        x : ndarray
            New values of the optimization variables.
        store : bool
            Whether the new x should be stored in self.history

        Notes
        -----
        After updating, if store=False, self._eq will revert back to the previous
        solution when store was True

        """
        # xopt is the full state vector of all the things
        # xeq is the full state vector of the equilibrium only
        xopt = f_where_x(x, self._allx, self._allxopt)
        xeq = f_where_x(x, self._allx, self._allxeq)
        if xopt.size > 0 and xeq.size > 0:
            pass
        else:
            x_list = self.unpack_state(x, False)
            x_list_old = self.unpack_state(self._x_old, False)
            xeq_dict = x_list[self._eq_idx]
            xeq_dict_old = x_list_old[self._eq_idx]
            deltas = {arg: xeq_dict[arg] - xeq_dict_old[arg] for arg in self._args}
            db = self._predict_boundary(x)
            if db is not None:
                deltas.update(self._split_b(db))
            self._eq.perturb(
                objective=self._eq_solve_objective,
                constraints=None,
                deltas=deltas,
                **self._perturb_options,
            )
            self._eq.solve(
                objective=self._eq_solve_objective,
                constraints=None,
                **self._solve_options,
            )
            self._solve_free_boundary(x_list)
            xeq = self._eq.pack_params(self._eq.params_dict)
            x_list[self._eq_idx] = self._eq.params_dict.copy()
            xopt = self._xopt(xeq, x_list)
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
        self._eq_solve_objective.update_constraint_target(self._eq)

        return xopt, xeq

    def _get_tangents(self, v, x, xopt):
        """Tangents in the full state space for directions v in the optimization space.

        Parameters
        ----------
        v : ndarray
            Directions in the space of optimization variables, as rows.
        x : ndarray
            Optimization variables.
        xopt : ndarray
            Full state vector of all the things at x, in free boundary equilibrium.

        Returns
        -------
        tangents : ndarray
            Change in the full state vector of all things for each direction in v,
            with force balance and the free boundary conditions maintained.

        """
        lin = self._linearize(xopt)
        T_c, P, Z = lin["T_c"], lin["P"], self._fb_Z
        v2 = jnp.atleast_2d(v)
        vs = jnp.split(v2, np.cumsum(self._dimc_per_thing)[:-1], axis=-1)
        # changes from the optimization variables at fixed boundary
        tangents = list(vs)
        tangents[self._eq_idx] = vs[self._eq_idx] @ T_c[self._c_idx_q]
        # change in free boundary error from the optimization variables
        dB = self._constraint_fb.jvp_scaled(
            self._fb_tangents(tangents[self._eq_idx], tangents), self._fb_x(xopt)
        )
        # change in the boundary to keep free boundary equilibrium: db = -dB/db⁺ dB
        db = -(jnp.atleast_2d(dB) @ P.T) @ Z.T
        tangents[self._eq_idx] = tangents[self._eq_idx] + db @ T_c[self._c_idx_fb]
        if v.ndim == 2:
            self._dbdv = (np.asarray(x), np.asarray(v), np.asarray(db), lin)
        tangents = jnp.concatenate(tangents, axis=-1)
        return tangents if v.ndim == 2 else tangents[0]

    def grad(self, x, constants=None):
        """Compute gradient of self.compute_scalar.

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        Returns
        -------
        g : ndarray
            gradient vector.

        """
        constants = setdefault(constants, [None, None, None])
        xg, _ = self._update_equilibrium(x, store=True)
        tangents = self._get_tangents(jnp.eye(x.shape[0]), x, xg)
        g = self._objective.compute_scaled_error(xg, constants[0])
        g_vjp = self._objective.vjp_scaled_error(g, xg, constants[0])
        return tangents @ g_vjp

    def _jvp(self, v, x, constants=None, op="scaled_error"):
        v = v[0] if isinstance(v, (tuple, list)) else v
        constants = setdefault(constants, [None, None, None])
        xg, _ = self._update_equilibrium(x, store=True)
        tangents = self._get_tangents(v, x, xg)
        if self._objective._deriv_mode == "batched":
            return getattr(self._objective, "jvp_" + op)(tangents, xg, constants[0])
        else:
            return _proximal_jvp_blocked_pure(
                self._objective,
                jnp.split(tangents, np.cumsum(self._dimx_per_thing), axis=-1),
                jnp.split(xg, np.cumsum(self._dimx_per_thing)),
                op,
            )

    @property
    def constants(self):
        """list: constant parameters for each sub-objective."""
        warnif(
            True,
            FutureWarning,
            "constants is deprecated and will be removed in a future "
            "release. Users should not include constants in the arguments "
            "of their objective compute methods. Instead declare all the "
            "constants in the build method and use as obj._constants.",
        )
        return [
            self._objective.constants,
            self._constraint.constants,
            self._constraint_fb.constants,
        ]


@jit_if_possible(static_argnames=())
def _fb_eq_tangents(constraint, xf, eq_feasible_tangents, dxdcv):
    """Same as _proximal_eq_tangents but also returns the SVD of dF/dx."""
    dim_x_reduced = eq_feasible_tangents.shape[-1]
    tangents = jnp.concatenate([eq_feasible_tangents.T, dxdcv], axis=0)
    J = constraint.jvp_scaled(tangents, xf)
    Fxh, Fc = J[:dim_x_reduced].T, J[dim_x_reduced:].T
    cutoff = jnp.finfo(Fxh.dtype).eps * max(Fxh.shape)
    uf, sf, vtf = jnp.linalg.svd(Fxh, full_matrices=False)
    sf += sf[-1]  # add a tiny bit of regularization
    sfi = jnp.where(sf < cutoff * sf[0], 0, 1 / sf)
    dfdc = vtf.T @ (sfi[:, None] * (uf.T @ Fc))
    return dxdcv - (eq_feasible_tangents @ dfdc).T, (uf, sfi, vtf)


@jit
def _pinv(A, rcond=None):
    """Pseudo-inverse of A, ignoring singular values below rcond * largest one.

    Also returns the SVD (u, s, vt) of A.
    """
    u, s, vt = jnp.linalg.svd(A, full_matrices=False)
    rcond = setdefault(rcond, jnp.finfo(A.dtype).eps * max(A.shape))
    sinv = jnp.where(s > rcond * s[0], 1 / s, 0)
    return (vt.T * sinv) @ u.T, (u, s, vt)
