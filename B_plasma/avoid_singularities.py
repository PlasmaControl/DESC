from netCDF4 import Dataset
import numpy as np
import os

from desc.grid import QuadratureGrid, LinearGrid, CylindricalGrid
from desc.io import load
from desc.utils import safenorm, Timer

# Load the equilibrium
file_path = '/global/homes/m/mavida/DESC/B_plasma/'
save_path = file_path+'grid_scans_shifted/'
os.makedirs(save_path,exist_ok=True)
eq = load(file_path+"equil_div-opt_64_DESC_fixed.h5")

# Timer object to quantify perforamcnce
timer = Timer()

# Define the resolutions to use
start = 6
stop = 7
bs_resolutions = np.logspace(start=start, stop=stop, num=(stop-start+1), endpoint=True, base=2, dtype=int)
start = 2
stop = 9
vc_resolutions = np.logspace(start=start, stop=stop, num=(stop-start+1), endpoint=True, base=2, dtype=int)
resolutions = {'virtual casing':vc_resolutions,'biot-savart':bs_resolutions}

# Timer object to quantify performance
timer = Timer()

# Coordinates on which to calculate the magnetic field
out_coords = np.load(file_path+'bmw_coords_high_res.npy')

# Grids on which to discretize the integral
res = 64
zeta = np.unique(out_coords[:,2])
zeta_lr = zeta[::2]
zeta_lr = zeta_lr + (zeta[1]-zeta[0])/2
zeta_lr = zeta_lr % (2 * np.pi/eq.NFP)
source_grid = QuadratureGrid(L=res, M=res, zeta=zeta_lr, NFP=eq.NFP)

# Calculate B on the grid using the three different methods
method = 'biot-savart'
save_name = save_path+method.replace(' ','_')+str(res)+'_grid.npy'
if not os.path.exists(save_name):
    print('starting '+method+' and resolution'+str(res))
    timer.start(method+str(res))
    B = eq.compute_magnetic_field(out_coords, source_grid=source_grid, method=method)
    timer.stop(method+str(res))
    np.save(save_name,B)
    print(method + ' finished with time ' + str(timer[method+str(res)]) + 'and resolution' + str(res))
