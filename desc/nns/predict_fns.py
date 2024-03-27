import jax
import jax.numpy as jnp
import flax


def expand_module(
    expansion: jax.Array,
    module_params: dict,
    module: flax.linen.Module,
    static_params: dict[str, jax.Array],  # needed?
):
    """

    Parameters
    ----------
    expansion : [Rb_lmn, Lb_lmn, Zb_lmn]  where  Lb_lmn == 0
        Input to Module (ie NN)
    module_params : dictionary which is a prefix for the instantiated Module
        parameters of module
    module : child of flax linen module
    static_params : dict
        static parameters for possible use (e.g. rho^xm array)

    Returns
    -------
        f_z_modes : dimy with shape eq.R_lmn.size + eq.Z_lmn.size + eq.L_lmn.size - eq.Rb_lmn.size - eq.Zb_lmn.size
    """
    assert expansion.ndim == 2
    rlmn, llmn, zlmn = module.apply(module_params, expansion)
    y = jnp.stack((rlmn, llmn, zlmn), axis=-1)
    return y
