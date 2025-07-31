import numpy as np
from desc.backend import jnp
from desc.io import load
from desc.grid import LinearGrid
from desc.plotting import *
from desc.objectives import ObjectiveFunction, Bxdl, XPointDistanceBound, FixCoilCurrent
from desc.optimize import Optimizer
from desc.coils import FourierRZCoil

def load_inputs(coil_paths,eq_path):
    # load inputs
    small_coils = load(coil_paths[0])
    large_coils = load(coil_paths[1])
    eq = load(eq_path)

    return eq, large_coils,small_coils
def find_xpt(l,small_coils,large_coils,dir,dist=None):
    # Optimize X point location
    if dist is None:
        offset = 0.4
    else:
        offset = np.maximum(0.02,dist)
    coords = jnp.stack([l['R'],l['phi'],l['Z']-offset]).T
    c = FourierRZCoil.from_values(current=0,coords=coords,basis='rpz',NFP=eq.NFP,name='x-point')
    optimizer = Optimizer("lsq-exact")
    eval_grid_N = 24
    eval_grid = LinearGrid(N=eval_grid_N)
    weights = {
        'bxdl': 1,
        'linking': 100,
        'dist': 100
    }
    bxdl = Bxdl(curve=c,
                    eq=eq,
                    field=[small_coils, large_coils],
                    field_grid=LinearGrid(N=32),
                    eq_kwargs={"method": 'biot-savart'},
                    bs_chunk_size=512,
                    eval_grid=eval_grid,
                    curve_fixed=False,
                    field_fixed=True,
                    weight=weights['dist'],
                    target=0)
    #linking = PlasmaLinkingNumber(c,eq,eval_grid,target=0,weight=weights["linking"])
    o = XPointDistanceBound(eq, c, N_grid=36,M_grid=36, eq_fixed=True,bounds=(0,0.4),weight=weights['dist'])

    obj = ObjectiveFunction((o,bxdl))#((o,bxdl,linking))
    obj.build()
    constraints = (FixCoilCurrent(c))
    (optimized_coilset,), _ = optimizer.optimize(
        c,
        objective=obj,
        constraints=constraints,
        maxiter=100,
        verbose=3,
        ftol=1e-4,
        copy=True,
    )

    #coords = optimized_coilset.compute(['R','Z'],grid=LinearGrid())#[0]
    np.save(f'{dir}x_point_Rn.npy',optimized_coilset.R_n)
    np.save(f'{dir}x_point_Zn.npy',optimized_coilset.Z_n)

root_dir = "/global/homes/m/mavida/DESC/xpt_dist_sweep/G1600_E/G1600_E/"
eq_path = root_dir + "equil_G1600.h5"
eq = load(eq_path)
grid = LinearGrid(rho=[1.0],theta=[3*np.pi/2],N=24,NFP=eq.NFP)
l = eq.compute(['R','phi','Z'],grid=grid)


for dist in [0,20,40]:
    dir = root_dir + f"G1600_E_{dist}/"
    coil_paths = (
            f"{dir}encircling_G1600_E_{dist}.h5",
            f"{dir}shaping_G1600_E_{dist}.h5",
        )
    line_path = f"{dir}G1600_xpt_coil_location_{dist}cm.txt"
    eq, large_coils, small_coils = load_inputs(coil_paths,eq_path)
    find_xpt(l,small_coils,large_coils,root_dir+f"{dist}cm_",dist=dist/100)
    
coil_paths = (f"{root_dir}encircling_G1600_B.h5", f"{root_dir}shaping_G1600_B.h5")
eq, large_coils, small_coils = load_inputs(coil_paths,eq_path)
find_xpt(l,small_coils,large_coils,root_dir+'null_hyp_',dist=dist/100)
