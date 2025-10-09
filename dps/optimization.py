from time import time as timet

import matplotlib.pyplot as plt
import numpy as np
import scipy

# set_device("gpu")
import desc.equilibrium
import desc.io
from desc import set_device
from desc.backend import jnp
from desc.grid import Grid
from desc.objectives import (
    FixBoundaryR,
    FixBoundaryZ,
    FixIota,
    FixPressure,
    FixPsi,
    ForceBalance,
    ObjectiveFunction,
    ParticleTracer,
)

####################################################### Set if single or multi particle #################################################################
"""
Set to True to simulate a single particle, False to simulate multiple particles

IF TRUE: Single Particle
IF FALSE: Multiple Particles

Differences:
Multi particle - 5 particles with the same theta and zeta, but different psi (and consequently different vpar); this will make the mu different for each particle since it depends on psi and vpar
Single particle - 1 particle with a single set of psi, theta, zeta, and vpar;
"""
SINGLE_PARTICLE = True
##########################################################################################################################################################

initial_time = timet()

filename = "eq_0108_M1_N1.h5"
savename = "optimized_" + filename

print("*************** START ***************")
print("Optimization")
print(f"Filename: {filename}")
print("****************************************")

# Load Equilibrium
eq = desc.io.load(filename)
eq._iota = eq.get_profile("iota").to_powerseries(order=1, sym=True)
eq._current = None
# eq.solve()

# Energy and Mass info
Energy_eV = 100
Energy_SI = Energy_eV * scipy.constants.elementary_charge
Mass = 4 * scipy.constants.proton_mass
Charge = 2 * scipy.constants.elementary_charge

# Initial State - (psi, theta, zeta, vpar)
zeta_i = 0.5
theta_i = jnp.pi / 2
vpar_i = 0.7 * jnp.sqrt(2 * Energy_SI / Mass)

if SINGLE_PARTICLE:
    psi_i = 0.9
    ini_cond = jnp.array([psi_i, theta_i, zeta_i, vpar_i])
    gridnodes = jnp.array([psi_i, theta_i, zeta_i])
else:
    psi_i = jnp.linspace(0.1, 0.9, 5)
    ini_cond = jnp.array(
        [[float(psi_i), theta_i, zeta_i, float(vpar_i)] for psi_i in psi_i]
    )
    gridnodes = jnp.array([[float(psi_i), theta_i, zeta_i] for psi_i in psi_i])

grid = Grid(nodes=gridnodes.T, jitable=False, sort=False)

# Time
tmin = 0
tmax = 1e-3
nt = 15
time = jnp.linspace(tmin, tmax, nt)

Mass_Charge_Ratio = Mass / Charge

data = eq.compute(["|B|", "R"], grid=grid)

mu = Energy_SI / (Mass * data["|B|"]) - (vpar_i**2) / (2 * data["|B|"])

if SINGLE_PARTICLE:
    ini_param = jnp.array([mu[0], Mass_Charge_Ratio])
else:
    ini_param = jnp.array([[mu, Mass_Charge_Ratio] for mu in mu])

intermediate_time = timet()

print(f"Time from beginning until here: {intermediate_time - initial_time}s")

objective = ParticleTracer(
    eq=eq,
    output_time=time,
    initial_conditions=ini_cond,
    initial_parameters=ini_param,
    compute_option="optimization",
    tolerance=1.4e-8,
)

objective.build()
solution = objective.compute(*objective.xs(eq))

intermediate_time_2 = timet()
print(f"Time to build and compute: {intermediate_time_2 - intermediate_time}s")

ObjFunction = ObjectiveFunction((objective), deriv_mode="looped")
ObjFunction.build()

intermediate_time_3 = timet()
print(f"Time to build and compile: {intermediate_time_3 - intermediate_time_2}s")

R_modes = jnp.array([[0, 0, 0]])
constraints = (
    ForceBalance(eq, bounds=(-1e-2, 1e-2)),
    FixBoundaryR(eq, modes=R_modes),
    FixBoundaryZ(eq, modes=False),
    FixPsi(eq),
    FixPressure(eq),
)  # FixPressure(eq), FixCurrent(eq), FixIota(eq), ForceBalance(eq, bounds=(-1e-3, 1e-3))
eq.optimize(
    objective=ObjFunction,
    optimizer="fmin-auglag-bfgs",
    constraints=constraints,
    verbose=3,
    maxiter=100,
    copy=False,
)
eq.save(savename)

intermediate_time_4 = timet()
print(f"Time to optimize: {intermediate_time_4 - intermediate_time_3}s")

print("Optimization Completed")
print(f"Optimized Filename: {savename}")
final_time = timet()
print(f"Total time: {final_time - initial_time}s")
print("*********************** END ***********************")
