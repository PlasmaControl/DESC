from find_xpt import *

weights = {
    'b_dot_n': 10,
    'bxdl': 2e-2,
    'xpt_dist': 1e-1
}

root_dir = "/global/homes/m/mavida/DESC/xpt_dist_sweep/xpt_dist_sweep/"
eq_path = root_dir + "equil_div-opt_98_DESC_fixed.h5"
eq = load(eq_path)

grid = LinearGrid(rho=[1.0], theta=[3 * np.pi / 2], N=24, NFP=eq.NFP)
l = eq.compute(["R", "phi", "Z"], grid=grid)
rad = 0.6
offset = 0.3
max_current = 6e6

out_tag = (
        f"shaping_div_98_{int(rad * 100)}cm_{int(max_current / 1e6)}MA_"
        + "_".join([f"{key}{value}" for key, value in weights.items()])
    )
encircling = load(root_dir + "encircling_div_98.h5")
shaping = load(root_dir + "xpt_param_scan/" + out_tag + "_shaping_coils_final.h5")

find_xpt(l, shaping, encircling, root_dir + out_tag, eq)
