import numpy as np
from scipy.linalg import block_diag

from desc.backend import jnp, put
from desc.utils import svd_inv_null
from desc.compute import arg_order
from .objective_funs import ObjectiveFunction
from .linear_objectives import (
    FixBoundaryR,
    FixBoundaryZ,
    LambdaGauge,
    FixPressure,
    FixIota,
    FixPsi,
)
from ._equilibrium import ForceBalance, RadialForceBalance, HelicalForceBalance, Energy


def get_fixed_boundary_constraints():
    """Get the constraints necessary for a typical fixed-boundary equilibrium problem.

    Returns
    -------
    constraints, tuple of _Objectives
        A list of the linear constraints used in fixed-boundary problems.

    """
    constraints = (
        FixBoundaryR(fixed_boundary=True),
        FixBoundaryZ(fixed_boundary=True),
        LambdaGauge(),
        FixPressure(),
        FixIota(),
        FixPsi(),
    )
    return constraints


def get_equilibrium_objective(mode="force"):
    """Get the objective function for a typical force balance equilibrium problem.

    Returns
    -------
    objective, ObjectiveFunction
        An objective function with default force balance objectives.

    """
    if mode == "force":
        objectives = ForceBalance()
    elif mode == "force2":
        objectives = (RadialForceBalance(), HelicalForceBalance())
    elif mode == "energy":
        objectives = Energy()
    else:
        raise ValueError("got an unknown equilibrium objective type '{}'".format(mode))
    return ObjectiveFunction(objectives)


def factorize_linear_constraints(constraints, extra_args=[]):
    """Compute and factorize A to get pseudoinverse and nullspace."""
    # set state vector
    args = np.concatenate([obj.args for obj in constraints])
    args = np.concatenate((args, extra_args))
    args = [arg for arg in arg_order if arg in args]
    dimensions = constraints[0].dimensions
    dim_x = 0
    x_idx = {}
    for arg in args:
        x_idx[arg] = np.arange(dim_x, dim_x + dimensions[arg])
        dim_x += dimensions[arg]

    A = {}
    b = {}
    Ainv = {}
    xp = jnp.zeros(dim_x)  # particular solution to Ax=b
    constraint_args = []  # all args used in constraints
    unfixed_args = []  # subset of constraint args for unfixed objectives

    # linear constraint matrices for each objective
    for obj in constraints:
        if len(obj.args) > 1:
            raise ValueError("Linear constraints must have only 1 argument.")
        arg = obj.args[0]
        constraint_args.append(arg)
        if obj.fixed and obj.dim_f == obj.dimensions[obj.target_arg]:
            # if all coefficients are fixed the constraint matrices are not needed
            xp = put(xp, x_idx[obj.target_arg], obj.target)
        else:
            unfixed_args.append(arg)
            A_ = obj.derivatives[arg]
            b_ = obj.target
            if A_.shape[0]:
                Ainv_, Z_ = svd_inv_null(A_)
            else:
                Ainv_ = A_.T
            A[arg] = A_
            b[arg] = b_
            Ainv[arg] = Ainv_

    # catch any arguments that are not constrained
    for arg in x_idx.keys():
        if arg not in constraint_args:
            unfixed_args.append(arg)
            A[arg] = jnp.zeros((1, constraints[0].dimensions[arg]))
            b[arg] = jnp.zeros((1,))

    # full A matrix for all unfixed constraints
    if len(A):
        unfixed_idx = jnp.concatenate(
            [x_idx[arg] for arg in arg_order if arg in A.keys()]
        )
        A_full = block_diag(*[A[arg] for arg in arg_order if arg in A.keys()])
        b_full = jnp.concatenate([b[arg] for arg in arg_order if arg in b.keys()])
        Ainv_full, Z = svd_inv_null(A_full)
        xp = put(xp, unfixed_idx, Ainv_full @ b_full)

    def project(x):
        """Project a full state vector into the reduced optimization vector."""
        x_reduced = jnp.dot(Z.T, (x - xp)[unfixed_idx])
        return jnp.atleast_1d(jnp.squeeze(x_reduced))

    def recover(x_reduced):
        """Recover the full state vector from the reducted optimization vector."""
        dx = put(jnp.zeros(dim_x), unfixed_idx, Z @ x_reduced)
        return jnp.atleast_1d(jnp.squeeze(xp + dx))

    return xp, A, Ainv, b, Z, unfixed_idx, project, recover
