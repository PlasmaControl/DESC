"""Functions for getting common objectives and constraints."""

import numpy as np
from scipy.constants import mu_0

from desc.backend import block_diag, jnp, put
from desc.compute import arg_order
from desc.utils import svd_inv_null


def factorize_linear_constraints(constraints, objective_args):
    """Compute and factorize A to get pseudoinverse and nullspace.

    Given constraints of the form Ax=b, factorize A to find a particular solution xp
    and the null space Z st. Axp=b and AZ=0, so that the full space of solutions to
    Ax=b can be written as x=xp + Zy where y is now unconstrained.


    Parameters
    ----------
    constraints : tuple of Objectives
        linear objectives/constraints to factorize for projection method.
    objective_args : list of str
        names of all arguments used by the desired objective.

    Returns
    -------
    xp : ndarray
        particular solution to Ax=b
    A : dict of ndarray
        Individual constraint matrices, keyed by argument
    Ainv : dict of ndarray
        Individual pseudoinverses of constraint matrices
    b : dict of ndarray
        Individual rhs vectors
    Z : ndarray
        Null space operator for full combined A
    unfixed_idx : ndarray
        indices of x that correspond to non-fixed values
    project, recover : function
        functions to project full vector x into reduced vector y,
        and recovering x from y.

    """
    # set state vector
    args = np.concatenate([obj.args for obj in constraints])
    args = np.concatenate((args, objective_args))
    # this is all args used by both constraints and objective
    args = [arg for arg in arg_order if arg in args]
    dimensions = constraints[0].dimensions
    dim_x = 0
    x_idx = {}
    for arg in objective_args:
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
        if arg not in objective_args:
            continue
        constraint_args.append(arg)
        if obj.fixed and obj.dim_f == obj.dimensions[obj.target_arg]:
            # if all coefficients are fixed the constraint matrices are not needed
            xp = put(xp, x_idx[obj.target_arg], obj.target)
        else:
            unfixed_args.append(arg)
            A_ = obj.derivatives["jac"][arg](jnp.zeros(obj.dimensions[arg]))
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
        """Recover the full state vector from the reduced optimization vector."""
        dx = put(jnp.zeros(dim_x), unfixed_idx, Z @ x_reduced)
        return jnp.atleast_1d(jnp.squeeze(xp + dx))

    return xp, A, Ainv, b, Z, unfixed_idx, project, recover


def compute_scaling_factors(eq):
    """Compute dimensional quantities for normalizations."""
    scales = {}
    R10 = eq.Rb_lmn[eq.surface.R_basis.get_idx(M=1, N=0)]
    Z10 = eq.Zb_lmn[eq.surface.Z_basis.get_idx(M=-1, N=0)]
    R00 = eq.Rb_lmn[eq.surface.R_basis.get_idx(M=0, N=0)]

    scales["R0"] = R00
    scales["a"] = np.sqrt(np.abs(R10 * Z10))
    scales["V"] = 2 * np.pi * scales["R0"] * scales["a"]
    scales["A"] = np.pi * scales["a"] ** 2
    scales["B_T"] = eq.Psi / scales["A"]
    iota = eq.get_profile("iota")(np.linspace(0, 1, 20))
    scales["B_P"] = scales["B_T"] * np.mean(np.abs(iota))
    scales["B"] = np.sqrt(scales["B_T"] ** 2 + scales["B_P"] ** 2)
    scales["p"] = scales["B"] ** 2 / (2 * mu_0)
    scales["W"] = scales["p"] * scales["V"]
    scales["J"] = scales["B"] / scales["a"] / mu_0
    scales["F"] = scales["p"] / scales["a"]
    scales["f"] = scales["F"] * scales["V"]
    return scales
