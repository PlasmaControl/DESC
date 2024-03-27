import datetime
import typing
import pprint

import jax
from jax import random
import optax
import matplotlib.pyplot as plt

import desc.grid
from desc.input_reader import InputReader
from desc.objectives import (
    ObjectiveFunction,
    ForceBalance,
    get_fixed_boundary_constraints,
    maybe_add_self_consistency,
    # GenericObjective,
)
from desc.equilibrium import Equilibrium
from desc.plotting import plot_section, plot_surfaces

from mlps import SomeNN, MultiMLP, SingleMLP
from loss_fns import loss_fn_constr_bdry
from util import get_scale, get_desc_opt


def init_fn_learning(
    eq: desc.equilibrium,
    module_config: dict,
    seed: int = 42,
) -> tuple[SomeNN, dict, typing.Any, dict, desc.objectives.ObjectiveFunction]:
    """
    Get objective and module

    Args:
        module_config: has name and layers
        eq:
        seed:

    Returns:

    """
    # force balance
    objective_force = ForceBalance(eq)

    # constrain the jacobian to be positive with high weights
    # objective_jac = GenericObjective(
    #     "sqrt(g)", eq, weight=1, name="positive_jacobian"  # bounds=(0, 1e12),
    # )

    # combine objectives
    objective = ObjectiveFunction((objective_force,))  # objective_jac,

    # this fixes the rlmn and zlmn at each iteration step,
    # which changes the fixed point during training = very bad
    # constraint = ObjectiveFunction(
    #     (BoundaryRSelfConsistency(eq), BoundaryZSelfConsistency(eq))
    # )

    # this does not, it projects the optimisation problem into a
    # sub-domain where R and Z LCFS are always satisfied. Better
    constraint = ObjectiveFunction(
        maybe_add_self_consistency(eq, constraints=get_fixed_boundary_constraints(eq)),
    )

    # constrain fixed boundary Ax-b=0 with x=x_p + Zy where y is prediction of module
    objective = desc.optimize.LinearConstraintProjection(objective, constraint)

    # always build the objective
    objective.build()

    # get random seed for module init
    init_key = random.PRNGKey(seed)

    # get the initial guess encoded in the spectral
    # domain with linear constraint projection
    # we predict in spectral with x = x_nn + x_init
    x_init = objective.x()

    # scaling of output of NN in projected space
    x_scale = get_scale(eq, objective, x_init)

    # update the module_config
    module_config.update({"x_init": x_init, "x_scale": x_scale})

    # initialize the module with input dummy, one for each Rblmn, Lblmn, Zblmn
    if module_config["name"] == "multiMLP":
        module_config = MultiMLP.configure(eq, module_config)
        module = MultiMLP(**module_config)
    elif module_config["name"] == "singleMLP":
        module_config = SingleMLP.configure(eq, module_config)
        module = SingleMLP(**module_config)
    else:
        raise NotImplementedError

    module_input = module.create_input(eq, seed=seed)
    module_params = module.init(init_key, module_input)

    return module, module_params, module_input, module_config, objective


def optimize_module(
    module: SomeNN,
    module_params: dict,
    module_inp: jax.Array,
    objective: desc.objectives.ObjectiveFunction,
    dimy: int,
    tols: tuple[float, float, float],
) -> dict:
    flat_params, unflat_fn = jax.flatten_util.ravel_pytree(module_params)

    loss_fn_constr_fb = jax.tree_util.Partial(
        loss_fn_constr_bdry,
        unflat_fn=unflat_fn,
        module=module,
        module_inp=module_inp,
        objective=objective,
        dimy=dimy,
    )

    df_dx = jax.jacfwd(loss_fn_constr_fb)

    opt = get_desc_opt(
        name="sgd",
        params=flat_params,
        loss_fun=loss_fn_constr_fb,
        df_dx=df_dx,
        tols=tols,
        verbose=2,
        maxiter=10000,
    )
    res = opt()

    module_params = unflat_fn(res.x)

    return module_params


def plot_solution(
    eq: Equilibrium,
    module: SomeNN,
    objective: ObjectiveFunction,
    module_params: dict,
    module_inp: jax.Array,
    inp_str_name: str,
):
    y_final = module.apply(module_params, module_inp)
    unmapped_state = objective.unpack_state(y_final, per_objective=False)
    eq.params_dict = unmapped_state[-1]
    plot_surfaces(eq)
    plot_section(eq, "|F|", norm_F=True, log=True)
    plt.show()

    inp_str = inp_str_name.replace("./", "")
    fig, ax = plot_section(eq, "|F|", norm_F=True, log=True)
    cur_time = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    fig.savefig(f"./plots/{cur_time}_{inp_str}_F")

    fig, ax = plot_surfaces(eq)
    fig.savefig(f"./plots/{cur_time}_{inp_str}_surf")


def main(inp, module_name="singleMLP"):
    seed = 42
    module_config = {
        "name": module_name,
        "mlp_layers": [32, 32],
        "apply_scale": True,
    }

    ir = InputReader(cl_args=inp)
    input_idx = 1  # which continuation step to take

    input_spec = ir.inputs[
        input_idx
    ]  # optimization only on final specification, no continuation for now
    eq = Equilibrium(**input_spec, check_kwargs=False)

    # assert not eq.sym, "Only non-symmetric eq solutions"
    assert eq.sym, "Only symmetric eq solutions"
    assert eq.bdry_mode == "lcfs", "Only fixed boundary eq"

    print("Module config:")
    pprint.pprint(module_config)

    module, module_params, module_input, module_config, objective = init_fn_learning(
        eq=eq,
        module_config=module_config,
        seed=seed,
    )

    module_params = optimize_module(
        module=module,
        module_params=module_params,
        module_inp=module_input,
        objective=objective,
        dimy=module.dimy,
        tols=(
            ir.inputs[input_idx]["ftol"],
            ir.inputs[input_idx]["xtol"],
            ir.inputs[input_idx]["gtol"],
        ),
    )

    plot_solution(
        eq=eq,
        module=module,
        objective=objective,
        module_inp=module_input,
        module_params=module_params,
        inp_str_name=module_name + "_" + inp + "_" + input_idx,
    )


if __name__ == "__main__":
    # main(sys.argv[1:])
    main("./DSHAPE_test")

    #### Some tests:

    """ multiMLP with or without scaling, [32, 32], 100 steps, SGD
    without:     2.593e-01      4.783e-04      1.728e-04      3.284e+00
    with:        4.098e-01      5.533e-04      3.687e-04      1.158e+00
        "with" has more stable decrease
    """

    """ multiMLP with SGD or with BFGS, [32, 32], 100 steps, scaling=False
    SGD:         2.593e-01      4.783e-04      1.728e-04      3.284e+00
    BFGS:        takes 2 long
    """

    """ multiMLP with SGD or with BFGS, [32, 32], 100 steps, scaling=True
    SGD:         4.098e-01      5.533e-04      3.687e-04      1.158e+00
    BFGS:        takes 2 long
    """

    """ singleMLP with SGD or with BFGS, [32, 32], 100 steps, scaling=True
    SGD:         4.246e-01      5.435e-04      3.158e-04      1.305e+00
    BFGS:
    BFGS with [6,6]:       divergence
    """

    """ singleMLP with or without scaling, [32, 32], 100 steps, SGD
    without:    divergence!?
    with:       4.246e-01      5.435e-04      3.158e-04      1.305e+00
    """

    """ both with scaling , [32, 32], 25 steps, SGD
    multiMLP after 25 steps: 4.826e-01      1.902e-03      8.368e-04      4.343e+00
    singleMLP after 25 steps: 4.936e-01      2.484e-03      7.290e-04      5.591e+00
    """
