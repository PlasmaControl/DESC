from typing import Callable

import jax

from desc.objectives import ObjectiveFunction

from mlps import SomeNN


def loss_fn_constr_bdry(
    m_params: jax.Array,
    unflat_fn: Callable,
    module: SomeNN,
    module_inp: tuple,
    objective: ObjectiveFunction,
    dimy: int,
):
    m_params = unflat_fn(m_params)
    y = module.apply(m_params, module_inp)
    x = objective.x()
    assert x.shape == y.shape
    x = x.at[:dimy].set(y)

    force_err = objective.compute_scalar(x)
    return force_err


def loss_fn_soft_bdry(
    m_params: jax.Array,
    unflat_fn: Callable,
    module: SomeNN,
    module_inp: tuple,
    objective: ObjectiveFunction,
):
    # todo-1(@tthun) broken
    raise NotImplementedError
    m_params = unflat_fn(m_params)
    y_mod = module.apply(m_params, module_inp)
    # y = jnp.concat((y_mod[0], y_mod[1], y_mod[2]))
    x = objective.x()
    # 80% sure this operation is correct if sym=True ; 99% sure if sym=False
    x = x.at[:].set(y_mod)
    force_err = objective.compute_scalar(x)
    # bdry_err = objective.  # ...
    return force_err
