from curses.ascii import SI
from desc import set_device
set_device("gpu")
from desc.objectives import ParticleTracer
from desc.grid import Grid
import desc.io
from desc.backend import jnp
import scipy.constants
import matplotlib.pyplot as plt
import numpy as np
from time import time as timet

def output_to_file(solution, name):
    if SINGLE_PARTICLE:
        list1 = solution[:, 0]
        list2 = solution[:, 1]
        list3 = solution[:, 2]
        list4 = solution[:, 3]

        combined_lists = zip(list1, list2, list3, list4)
        
        file_name = f'{name}.txt'

        with open(file_name, 'w') as file:        
            for row in combined_lists:
                row_str = '\t'.join(map(str, row))
                file.write(row_str + '\n')
    else:
        reshaped_array = jnp.hstack([solution[i] for i in range(solution.shape[0])])

        # Save to a text file
        with open(f"{name}.txt", 'w') as f:
            for row in reshaped_array:
                f.write('\t'.join(f'{x}' for x in row) + '\n')

####################################################### Set if single or multi particle #################################################################
"""
Set to True to simulate a single particle, False to simulate multiple particles

IF TRUE: Single Particle
IF FALSE: Multiple Particles

Differences:
Multi particle - 5 particles with the same theta and zeta, but different psi (and consequently different vpar); this will make the mu different for each particle since it depends on psi and vpar
Single particle - 1 particle with a single set of psi, theta, zeta, and vpar;
"""
SINGLE_PARTICLE = False
##########################################################################################################################################################

initial_time = timet()

filename = "eq_0112_M1_N1.h5"
eq = desc.io.load(filename)
save_text_name = "solution_test_" + filename

print("*************** Start ***************")
print("Particle Tracer")
print(f"Filename: {filename}")
print("*************************************")

eq._iota = eq.get_profile("iota").to_powerseries(order=eq.L, sym=True)
eq._current = None

# Energy and Mass info
Energy_eV = 100
Energy_SI = Energy_eV*scipy.constants.elementary_charge
Mass = 4*scipy.constants.proton_mass
Charge = 2*scipy.constants.elementary_charge

# Initial State - (psi, theta, zeta, vpar)
zeta_i = 0.5
theta_i = jnp.pi/2
vpar_i = 0.7*jnp.sqrt(2*Energy_SI/Mass)

if SINGLE_PARTICLE:
    psi_i = 0.9
    ini_cond = jnp.array([psi_i, theta_i, zeta_i, vpar_i])
    gridnodes = jnp.array([psi_i, theta_i, zeta_i])
else:
    psi_i = jnp.linspace(0.1, 0.9, 5) 
    ini_cond = jnp.array([[float(psi_i), theta_i, zeta_i, float(vpar_i)] for psi_i in psi_i])
    gridnodes = jnp.array([[float(psi_i), theta_i, zeta_i] for psi_i in psi_i])

grid = Grid(nodes=gridnodes.T, jitable=False, sort=False)


# Time
tmin = 0
tmax = 1e-4
nt = 100
time = jnp.linspace(tmin, tmax, nt)

Mass_Charge_Ratio = Mass/Charge

data = eq.compute(["|B|", "R"], grid=grid)

mu = Energy_SI/(Mass*data["|B|"]) - (vpar_i**2)/(2*data["|B|"])

if SINGLE_PARTICLE:
    ini_param = jnp.array([mu[0], Mass_Charge_Ratio])
else:
    ini_param = jnp.array([[mu, Mass_Charge_Ratio] for mu in mu])


objective = ParticleTracer(eq=eq, output_time=time, initial_conditions=ini_cond, initial_parameters=ini_param, compute_option="tracer", tolerance=1.e-7)

intermediate_time = timet()
print(f"Time from beginning until here: {intermediate_time - initial_time}s")
objective.build()
solution = objective.compute(*objective.xs(eq))
intermediate_time_2 = timet()
print(f"Time to build and compute: {intermediate_time_2 - intermediate_time}s")

print("*************** SOLUTION .compute() ***************")
print(solution)
print("***************************************************")

output_to_file(solution=solution, name=save_text_name)


print(f"Solution File Name: {save_text_name}.txt")
final_time = timet()
print(f"Total Time: {final_time - initial_time}s")
print("*************** END ***************")