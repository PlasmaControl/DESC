import numpy as np
from desc.backend import jnp
from desc.io import load
from desc.grid import LinearGrid
from desc.plotting import *
from desc.objectives import ObjectiveFunction, Bxdl, XPointDistanceBound, FixCoilCurrent
from desc.optimize import Optimizer
from desc.coils import FourierRZCoil


def load_inputs(coil_paths, eq_path):
    # load inputs
    small_coils = load(coil_paths[0])
    large_coils = load(coil_paths[1])
    eq = load(eq_path)

    return eq, large_coils, small_coils


def find_xpt(
    l, small_coils, large_coils, save_path, eq, dist=None, curve_N=24, eval_grid_N=360
):
    # Optimize X point location
    if dist is None:
        offset = 0.4
    else:
        offset = np.maximum(0.02, dist)
    coords = jnp.stack([l["R"], l["phi"], l["Z"] - offset]).T
    c = FourierRZCoil.from_values(
        current=0, coords=coords, N=curve_N, basis="rpz", NFP=eq.NFP, name="x-point"
    )
    optimizer = Optimizer("lsq-exact")

    eval_grid = LinearGrid(N=eval_grid_N)
    weights = {"bxdl": 1, "linking": 100, "dist": 100}
    bxdl = Bxdl(
        curve=c,
        eq=eq,
        field=[small_coils, large_coils],
        field_grid=LinearGrid(N=32),
        eq_kwargs={"method": "biot-savart"},
        bs_chunk_size=512,
        eval_grid=eval_grid,
        curve_fixed=False,
        field_fixed=True,
        weight=weights["dist"],
        target=0,
    )
    # linking = PlasmaLinkingNumber(c,eq,eval_grid,target=0,weight=weights["linking"])
    o = XPointDistanceBound(
        eq,
        c,
        N_grid=36,
        M_grid=36,
        eq_fixed=True,
        bounds=(0, 0.4),
        weight=weights["dist"],
    )

    obj = ObjectiveFunction((o, bxdl))  # ((o,bxdl,linking))
    obj.build()
    constraints = FixCoilCurrent(c)
    (optimized_coilset,), _ = optimizer.optimize(
        c,
        objective=obj,
        constraints=constraints,
        maxiter=100,
        verbose=3,
        ftol=1e-4,
        copy=True,
    )

    # coords = optimized_coilset.compute(['R','Z'],grid=LinearGrid())#[0]
    optimized_coilset.save(save_path)
