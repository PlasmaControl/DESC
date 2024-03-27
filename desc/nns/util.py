from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import orbax, flax

import desc


def get_scale(eq, objective, x_init):
    # scaling factor for NN output
    # first unpack the state
    state_scale_dict = objective.unpack_state(x_init)[0][0]
    # now calculate scales
    scaler = 1 / (abs(eq.R_basis.modes[:, :2]).sum(axis=1) + 1)
    scalel = 1 / (abs(eq.L_basis.modes[:, :2]).sum(axis=1) + 1)
    scalez = 1 / (abs(eq.Z_basis.modes[:, :2]).sum(axis=1) + 1)

    # set scales to init_state
    state_scale_dict["R_lmn"] = scaler
    state_scale_dict["L_lmn"] = scalel
    state_scale_dict["Z_lmn"] = scalez
    flat_state_scale = jax.flatten_util.ravel_pytree(state_scale_dict)[0]
    # project into nullspace/lin.constr. space
    x_scale = objective.project(flat_state_scale)
    return x_scale


def get_desc_opt(
    name: str,
    params: jax.Array,
    loss_fun: Callable,
    df_dx: Callable,
    tols: tuple[float, float, float],
    verbose: int = 2,
    maxiter: int = 2e3,
) -> Callable:
    assert name in ["sgd", "bfgs"]

    if name == "sgd":
        opt = jax.tree_util.Partial(
            desc.optimize.sgd,
            fun=loss_fun,
            x0=params,
            grad=df_dx,
            verbose=verbose,
            maxiter=maxiter,
            ftol=tols[0],
            xtol=tols[1],
            gtol=tols[2],
        )
    elif name == "bfgs":
        opt = jax.tree_util.Partial(
            desc.optimize.fmintr,
            hess="bfgs",
            fun=loss_fun,
            x0=params,
            grad=df_dx,
            verbose=verbose,
            maxiter=maxiter,
            ftol=tols[0],
            xtol=tols[1],
            gtol=tols[2],
        )
    else:
        raise NotImplementedError(f"optimizer {name} not implemented")
    return opt


def get_nn_opt(name: str):
    assert name in ["adamw", "lbfgs"]


def get_train_state():
    raise NotImplementedError


def get_filter_params(params, keys_to_keep: list):
    """

    Args:
        params: some non-frozen parameters of a jax model
        keys_to_keep: keys in this frozen parameters to keep

    Returns: a filter-tree with keys-to-keep==True and all other set to False,
                this filter can be used e.g. for eqx.partition()

    """
    fs = {}
    for param_group, group_params in params["params"].items():
        fs.update({param_group: {}})
        for gk, _ in group_params.items():
            if gk in keys_to_keep:
                fs[param_group].update({gk: True})
            else:
                fs[param_group].update({gk: False})
    filter_spec = {"params": fs}
    return filter_spec


def linalg_norm_op(x, bias=False, kernel=False):
    # TODO-1(@timt) check if this could be deprecated for jax.tree_util.tree_l2_norm
    if bias and kernel:
        pass
    elif bias:
        filter_spec = get_filter_params(x, keys_to_keep=["bias"])
        x, _ = eqx.partition(x, filter_spec=filter_spec)
    elif kernel:
        filter_spec = get_filter_params(x, keys_to_keep=["kernel"])
        x, _ = eqx.partition(x, filter_spec=filter_spec)

    leaves, _ = jax.tree_util.tree_flatten(x)
    return sum(jnp.linalg.norm(l) for l in leaves)
