from desc import set_device
set_device("gpu")
from desc.objectives import ParticleTracer, ObjectiveFunction
from desc.grid import Grid
import desc.io
from desc.objectives import ForceBalance, FixBoundaryR, FixBoundaryZ, FixPressure, FixIota, FixPsi
from desc.backend import jnp
import matplotlib.pyplot as plt
import numpy as np
import jax.random

# Load Equilibrium
eq = desc.io.load("input.final_freeb_output.h5")[-1]
eq._iota = eq.get_profile("iota").to_powerseries(order=eq.L, sym=True)
eq._current = None
eq.solve()

# Energy and Mass info
Energy_eV = 1
Proton_Mass = 1.673e-27
Proton_Charge = 1.6e-19

Energy_SI = Energy_eV*Proton_Charge

# Particle Info
Mass = 4*Proton_Mass
Charge = 2*Proton_Charge

# Initial State
psi_i = 0.8
zeta_i = 0.1
theta_i = 0.2
vpar_i = 0.7*jnp.sqrt(2*Energy_SI/Mass)
ini_cond = [float(psi_i), theta_i, zeta_i, float(vpar_i)]

# Time
tmin = 0
tmax = 1e-5
nt = 2000
time = jnp.linspace(tmin, tmax, nt)

initial_conditions = ini_cond
Mass_Charge_Ratio = Mass/Charge

grid = Grid(jnp.array([jnp.sqrt(psi_i), theta_i, zeta_i]).T, jitable=True, sort=False)
data = eq.compute("|B|", grid=grid)

mu = Energy_SI/(Mass*data["|B|"]) - (vpar_i**2)/(2*data["|B|"])

ini_param = [float(mu), Mass_Charge_Ratio]

objective = ParticleTracer(eq=eq, output_time=time, initial_conditions=ini_cond, initial_parameters=ini_param, compute_option="optimization")

objective.build()
solution = objective.compute(*objective.xs(eq))

print("*************** SOLUTION .compute() ***************")
print(solution)
print("***************************************************")


ObjFunction = ObjectiveFunction([objective])
ObjFunction.build()

print("*************** ObjFunction.compile() ***************")
ObjFunction.compile(mode="bfgs")
print("*****************************************************")

gradient = ObjFunction.grad(ObjFunction.x(eq))
print("*************** GRADIENT ***************")
print(gradient)
print("****************************************")

#print(ObjFunction.x(eq))
#xs = objective.xs(eq)
#print("*************** xs **************")
#print(xs)
#print("*********************************")

R_modes = np.array([[0, 0, 0]])
constraints = (ForceBalance(eq), FixBoundaryR(eq, modes=R_modes), FixBoundaryZ(eq, modes=False), FixPressure(eq), FixIota(eq), FixPsi(eq))
eq.optimize(objective=ObjFunction, optimizer = "fmin-auglag-bfgs", constraints=constraints, verbose=3)
eq.save("test_run_optimized.h5")
