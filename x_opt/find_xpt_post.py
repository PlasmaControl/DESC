from find_xpt import *

root_dir = "/global/homes/m/mavida/DESC/xpt_dist_sweep/xpt_dist_sweep/"
eq_path = root_dir + "equil_div-opt_98_DESC_fixed.h5"
eq = load(eq_path)

grid = LinearGrid(rho=[1.0], theta=[3 * np.pi / 2], N=24, NFP=eq.NFP)
l = eq.compute(["R", "phi", "Z"], grid=grid)
out_tag = "shaping_div_98_60cm_6MA_b_dot_n10_bxdl0.02_xpt_dist1"
encircling = load(root_dir + "encircling_div_98.h5")
shaping = load(root_dir + "xpt_param_scan/" + out_tag + "_shaping_coils_final.h5")

find_xpt(l, shaping, encircling, root_dir + out_tag, eq)
