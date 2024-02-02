from desc import set_device
set_device("gpu")
import desc.equilibrium
from desc.objectives import ParticleTracer, ObjectiveFunction
from desc.grid import Grid
import desc.io
from desc.objectives import ForceBalance, FixBoundaryR, FixBoundaryZ, FixPressure, FixIota, FixPsi
from desc.backend import jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy
from time import time as timet

initial_time = timet()

# filename = "input.final_freeb_output.h5"
# filename = "DESC_ellipse.vacuum.0609.a_fixed_bdry_L_15_M_15_N_15_nfev_300_Mgrid_26_ftol_1e-4.h5"
filename = "input.LandremanPaul2021_QA_scaled_output.h5"
savename = "optimized_" + filename

print("*************** START ***************")
print("Optimization")
print(f"Filename: {filename}")
print("****************************************")

# Load Equilibrium
eq = desc.io.load(filename)
eq._iota = eq.get_profile("iota").to_powerseries(order=eq.L, sym=True)
eq._current = None
# eq.solve()

# Energy and Mass info
Energy_eV = 3.52e6
Proton_Mass = scipy.constants.proton_mass
Proton_Charge = scipy.constants.elementary_charge

Energy_SI = Energy_eV*Proton_Charge

# Particle Info
Mass = 4*Proton_Mass
Charge = 2*Proton_Charge

# Initial State
psi_i = 0.2
zeta_i = 0
theta_i = 0
vpar_i = 0.7*jnp.sqrt(2*Energy_SI/Mass)
ini_cond = [float(psi_i), theta_i, zeta_i, float(vpar_i)]

# Time
tmin = 0
tmax = 1e-4
nt = 250
time = jnp.linspace(tmin, tmax, nt)

initial_conditions = ini_cond
Mass_Charge_Ratio = Mass/Charge

grid = Grid(jnp.array([jnp.sqrt(psi_i), theta_i, zeta_i]).T, jitable=True, sort=False)
data = eq.compute("|B|", grid=grid)

mu = Energy_SI/(Mass*data["|B|"]) - (vpar_i**2)/(2*data["|B|"])

ini_param = [float(mu), Mass_Charge_Ratio]

intermediate_time = timet()
print(f"Time from beginning until here: {intermediate_time - initial_time}s")

objective = ParticleTracer(eq=eq, output_time=time, initial_conditions=ini_cond, initial_parameters=ini_param, compute_option="optimization", tolerance=1.4e-8)

objective.build()
solution = objective.compute(*objective.xs(eq))

intermediate_time_2 = timet()
print(f"Time to build and compute: {intermediate_time_2 - intermediate_time}s")

print("*************** SOLUTION .compute() ***************")
print(solution)
print("***************************************************")

ObjFunction = ObjectiveFunction([objective])
ObjFunction.build()

print("*************** ObjFunction.compile() ***************")
ObjFunction.compile(mode="bfgs")
print("*****************************************************")

intermediate_time_3 = timet()
print(f"Time to build and compile: {intermediate_time_3 - intermediate_time_2}s")

#print(ObjFunction.x(eq))
#xs = objective.xs(eq)
#print("*************** xs **************")
#print(xs)
#print("*********************************")

R_modes = np.array([[0, 0, 0]])
constraints = (ForceBalance(eq), FixBoundaryR(eq, modes=R_modes), FixBoundaryZ(eq, modes=False), FixPressure(eq), FixIota(eq), FixPsi(eq))
eq.optimize(objective=ObjFunction, optimizer = "fmin-auglag-bfgs", constraints=constraints, verbose=3, maxiter=100) # Mudar o número de iterações para 3, 10, 100
eq.save(savename)

intermediate_time_4 = timet()
print(f"Time to optimize: {intermediate_time_4 - intermediate_time_3}s")

print("Optimization Completed")
print(f"Optimized Filename: {savename}")
final_time = timet()
print(f"Total time: {final_time - initial_time}s")
print("*********************** END ***********************")

