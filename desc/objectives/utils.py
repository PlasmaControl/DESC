from desc.backend import jnp, put
from scipy.linalg import block_diag
from desc.utils import svd_inv_null
from desc.compute import arg_order
from .objective_funs import ObjectiveFunction
from .linear_objectives import (
    LCFSBoundaryR,
    LCFSBoundaryZ,
    LambdaGauge,
    PoincareLambda,
    FixedPressure,
    FixedIota,
    FixedPsi,
    PoincareBoundaryR,
    PoincareBoundaryZ,
)
from ._equilibrium import ForceBalance, RadialForceBalance, HelicalForceBalance, Energy


def get_fixed_boundary_constraints(mode="lcfs"):
    """Get the constraints necessary for a typical fixed-boundary equilibrium problem.

    Returns
    -------
    constraints, tuple of _Objectives
        A list of the linear constraints used in fixed-boundary problems.

    """

    constraints = (
        LambdaGauge(),
        FixedPressure(),
        FixedIota(),
        FixedPsi(),
    )
    if mode == "lcfs":
        constraints += (LCFSBoundaryR(), LCFSBoundaryZ())
    elif mode == "poincare":
        constraints += (PoincareBoundaryR(), PoincareBoundaryZ())
    else:
        raise ValueError("got an unknown boundary condition type '{}'".format(mode))
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


def factorize_linear_constraints(constraints, dim_x, x_idx):
    """Compute and factorize A to get pseudoinverse and nullspace."""
    A = {}
    b = {}
    Ainv = {}
    xp = jnp.zeros(dim_x)  # particular solution to Ax=b

    # A matrices for each unfixed constraint
    for obj in constraints:
        if obj.fixed:
            # if we're fixing something that's not in x, ignore it
            if obj.target_arg in x_idx.keys():
                xp = put(xp, x_idx[obj.target_arg], obj.target)
        else:
            if len(obj.args) > 1:
                raise ValueError("Non-fixed constraints must have only 1 argument.")
            arg = obj.args[0]
            A_ = obj.derivatives[arg]
            b_ = obj.target
            if A_.shape[0]:
                Ainv_, Z_ = svd_inv_null(A_)
            else:
                Ainv_ = A_.T
            A[arg] = A_
            b[arg] = b_
            Ainv[arg] = Ainv_

    # full A matrix for all unfixed constraints
    unfixed_idx = jnp.concatenate([x_idx[arg] for arg in arg_order if arg in A.keys()])
    A_full = block_diag(*[A[arg] for arg in arg_order if arg in A.keys()])
    b_full = jnp.concatenate([b[arg] for arg in arg_order if arg in b.keys()])
    Ainv_full, Z = svd_inv_null(A_full)
    xp = put(xp, unfixed_idx, Ainv_full @ b_full)

    def project(x):
        """Project a full state vector into the reduced optimization vector."""
        x_reduced = jnp.dot(Z.T, (x - xp)[unfixed_idx])
        return jnp.squeeze(x_reduced)

    def recover(x_reduced):
        """Recover the full state vector from the reducted optimization vector."""
        dx = put(jnp.zeros(dim_x), unfixed_idx, Z @ x_reduced)
        return jnp.squeeze(xp + dx)

    return xp, A, Ainv, b, Z, unfixed_idx, project, recover
