"""Wrappers for doing STELLOPT/SIMSOPT like optimization."""

import functools

import numpy as np

from desc.backend import jit, jnp, put
from desc.batching import batched_vectorize
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

        (self._eq_Z, self._eq_D, self._eq_unfixed_idx) = (
            self._eq_solve_objective._Z,
            self._eq_solve_objective._D,
            self._eq_solve_objective._unfixed_idx,
        )

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
        self._dimc_per_thing[self._eq_idx] = np.sum(
            [self._eq.dimensions[arg] for arg in self._args]
        )

        # equivalent matrix for A[unfixed_idx] @ D @ Z == A @ feasible_tangents
        self._feasible_tangents = jnp.eye(self._objective.dim_x)
        self._feasible_tangents = jnp.split(
            self._feasible_tangents, np.cumsum(self._dimx_per_thing), axis=-1
        )
        # dg/dxeq_reduced = dg/dx_eq_unscaled @ dx_eq_unscaled/dxeq_reduced # noqa: E800
        # x_eq_unscaled = Deq(xp_eq + Zeq @ xeq_reduced)                    # noqa: E800
        # So, the feasible tangents (aka. dx_eq_unscaled/dx_reduced) is Deq@Zeq
        # Since here we are setting the feasible direction for eq parameters only,
        # we need to add 0 rows for eq fixed parameters and non-eq parameters which we
        # handle by below operation
        self._feasible_tangents[self._eq_idx] = self._feasible_tangents[self._eq_idx][
            :, self._eq_unfixed_idx
        ] @ (self._eq_Z * self._eq_D[self._eq_unfixed_idx, None])
        self._feasible_tangents = jnp.concatenate(
            [np.atleast_2d(foo) for foo in self._feasible_tangents], axis=-1
        )

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
            Constant parameters passed to sub-objectives.

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
            Constant parameters passed to sub-objectives.

        """
        op = "unscaled"
        return self._jvp(v, x, constants, op)

    def _jvp(self, v, x, constants=None, op="scaled_error"):
        # The goal is to compute the Jacobian of the objective function with respect to
        # the optimization variables (c). Before taking the Jacobian, we update the
        # equilibrium such that
        # F(x+dx, c+dc) = 0 = F(x, c) + dF/dx * dx + dF/dc * dc
        # so that we can set F(x, c) = 0, from here we can solve for dx and get
        # dx = - (dF/dx)^-1 * dF/dc * dc     # noqa : E800
        # We can then compute the Jacobian of the objective function with respect to c
        # G(x+dx, c+dc) = G(x, c) + dG/dx * dx + dG/dc * dc
        # substituting in dx we get
        # G(x+dx, c+dc) = G(x, c) + [ dG/dc - dG/dx * (dF/dx)^-1 * dF/dc ]* dc
        # and the Jacobian we want is dG/dc - dG/dx * (dF/dx)^-1 * dF/dc

        # Note: This Jacobian can be obtained using JVPs in proper tangent directions.
        # First we will compute the tangent direction (see _get_tangent for details),
        # then we will compute the Jacobian.
        v = v[0] if isinstance(v, (tuple, list)) else v
        constants = setdefault(constants, self.constants)
        xg, xf = self._update_equilibrium(x, store=True)

        # we don't need to divide this part into blocked and batched because
        # self._constraint._deriv_mode will handle it
        jvpfun = lambda u: self._get_tangent(u, xf, constants, op=op)
        tangents = batched_vectorize(
            jvpfun,
            signature="(n)->(k)",
            chunk_size=self._constraint._jac_chunk_size,
        )(v)

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

    def _get_tangent(self, v, xf, constants, op):
        # Note: This function is vectorized over v. So, v is expected to be 1D array
        # of size self.dim_x.

        # v contains self._args DoFs from eq and other objects (like coils, surfaces
        # etc), we want jvp_f to only get parts from equilibrium, not other things
        vs = jnp.split(v, np.cumsum(self._dimc_per_thing))
        # This is (dF/dx)^-1 * dF/dc  # noqa : E800
        dfdc = _proximal_jvp_f_pure(
            self._constraint,
            xf,
            constants[1],
            vs[self._eq_idx],
            self._eq_solve_objective._feasible_tangents,
            self._dxdc,
            op,
        )
        # broadcasting against multiple things
        dfdcs = [jnp.zeros(dim) for dim in self._dimc_per_thing]
        dfdcs[self._eq_idx] = dfdc
        # note that dfdc.size != vs[self._eq_idx].size
        # dfdc has the size of reduced state vector of the equilibrium
        # but vs[self._eq_idx] has the size of self._args DoFs
        dfdc = jnp.concatenate(dfdcs)

        # We try to find dG/dc - dG/dx * (dF/dx)^-1 * dF/dc
        # where G is the objective function. Since DESC stores x and c in the same
        # vector, instead of multiple JVP calls, we will just find a tangent direction
        # that will give us the same result.
        # For making the explanation clear, assume J is the Jacobian of the objective
        # function with respect to the full state vector (both x and c). Then,
        # dG/dc = J @ (tangent vectors in c direction)
        # dG/dx = J @ (tangent vectors in x direction)
        # So, dG/dc - dG/dx * (dF/dx)^-1 * dF/dc can be written as
        # J @ [(tangent vectors in c direction) - (tangent vectors in x direction)@dfdc]
        # Note: We will never form full Jacobian J, we will just compute the above
        # expression by JVPs.
        dxdcv = jnp.concatenate(
            [
                *vs[: self._eq_idx],
                self._dxdc @ vs[self._eq_idx],  # Rb_lmn, Zb_lmn to full eq state vector
                *vs[self._eq_idx + 1 :],
            ]
        )
        tangent = dxdcv - self._feasible_tangents @ dfdc
        return tangent

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
def _proximal_jvp_f_pure(constraint, xf, constants, dc, eq_feasible_tangents, dxdc, op):
    # Note: This function is called by _get_tangent which is vectorized over v
    # (v is called dc in this function). So, dc is expected to be 1D array
    # of same size as full equilibrium state vector. This function returns a 1D array.

    # here we are forming (dF/dx)^-1 @ dF/dc
    # where Fxh is dF/dx and Fc is dF/dc
    Fxh = getattr(constraint, "jvp_" + op)(eq_feasible_tangents.T, xf, constants).T
    # Our compute functions never include variables like Rb_lmn, Zb_lmn etc. So,
    # taking the JVP in just dc direction will give 0. To prevent this, we use dxdc
    # which is the dx/dc matrix and convert the Rb_lmn to R_lmn entries etc.
    # For example, if we want the derivative wrt Rb_023, we should take the derivative
    # wrt all R_lmn coefficients that contribute to Rb_023. See BoundaryRSelfConsistency
    # for the relation between Rb_lmn and R_lmn.
    Fc = getattr(constraint, "jvp_" + op)(dxdc @ dc, xf, constants)
    cutoff = jnp.finfo(Fxh.dtype).eps * max(Fxh.shape)
    uf, sf, vtf = jnp.linalg.svd(Fxh, full_matrices=False)
    sf += sf[-1]  # add a tiny bit of regularization
    sfi = jnp.where(sf < cutoff * sf[0], 0, 1 / sf)
    return vtf.T @ (sfi * (uf.T @ Fc))


@functools.partial(jit, static_argnames=["op"])
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
    return jnp.concatenate(out).T
