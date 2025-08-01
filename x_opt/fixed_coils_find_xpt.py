from find_xpt import *
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
    eq, large_coils, small_coils = load_inputs(coil_paths,eq_path)
    find_xpt(l,small_coils,large_coils,root_dir+f"{dist}cm_",dist=dist/100)
    
coil_paths = (f"{root_dir}encircling_G1600_B.h5", f"{root_dir}shaping_G1600_B.h5")
eq, large_coils, small_coils = load_inputs(coil_paths,eq_path)
find_xpt(l,small_coils,large_coils,root_dir+'null_hyp_',dist=dist/100)
