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

filename = "eq_0108_M1_N1.h5"
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
Energy_eV = 100
Proton_Mass = scipy.constants.proton_mass
Proton_Charge = scipy.constants.elementary_charge
Energy_SI = Energy_eV*Proton_Charge

# Particle Info
Mass = 4*Proton_Mass
Charge = 2*Proton_Charge

# Initial State
psi_i = jnp.linspace(0.1, 0.9, 1000)
zeta_i = 0.5
theta_i = jnp.pi/2
vpar_i = 0.6*jnp.sqrt(2*Energy_SI/Mass)

# Initial Conditions
ini_cond = jnp.array([[float(psi_i), theta_i, zeta_i, float(vpar_i)] for psi_i in psi_i])
gridnodes = jnp.array([[float(psi_i), theta_i, zeta_i] for psi_i in psi_i])

# Time
tmin = 0
tmax = 1e-3
nt = 1500
time = jnp.linspace(tmin, tmax, nt)

initial_conditions = ini_cond
Mass_Charge_Ratio = Mass/Charge

# grid = Grid(nodes=jnp.array([jnp.sqrt(psi_i), theta_i, zeta_i]).T, jitable=False, sort=False)
grid = Grid(nodes=gridnodes.T, jitable=False, sort=False)
data = eq.compute(["|B|", "R"], grid=grid)

mu = Energy_SI/(Mass*data["|B|"]) - (vpar_i**2)/(2*data["|B|"])

ini_param = jnp.array([[mu, Mass_Charge_Ratio] for mu in mu])

intermediate_time = timet()

print(f"Time from beginning until here: {intermediate_time - initial_time}s")

objective = ParticleTracer(eq=eq, output_time=time, initial_conditions=ini_cond, initial_parameters=ini_param, compute_option="optimization", tolerance=1.4e-8)

objective.build()
solution = objective.compute(*objective.xs(eq))

intermediate_time_2 = timet()
print(f"Time to build and compute: {intermediate_time_2 - intermediate_time}s")

ObjFunction = ObjectiveFunction((objective), deriv_mode="looped")
ObjFunction.build()

intermediate_time_3 = timet()
print(f"Time to build and compile: {intermediate_time_3 - intermediate_time_2}s")

R_modes = jnp.array([[0, 0, 0]])
constraints = (ForceBalance(eq, bounds=(-1e-2, 1e-2)), FixBoundaryR(eq, modes=R_modes), FixBoundaryZ(eq, modes=False), FixPsi(eq), FixPressure(eq)) #FixPressure(eq), FixCurrent(eq), FixIota(eq), ForceBalance(eq, bounds=(-1e-3, 1e-3))
eq.optimize(objective=ObjFunction, optimizer = "fmin-auglag-bfgs", constraints=constraints, verbose=3, maxiter=100, copy=False)
eq.save(savename)

intermediate_time_4 = timet()
print(f"Time to optimize: {intermediate_time_4 - intermediate_time_3}s")

print("Optimization Completed")
print(f"Optimized Filename: {savename}")
final_time = timet()
print(f"Total time: {final_time - initial_time}s")
print("*********************** END ***********************")

