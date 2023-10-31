from desc import set_device
set_device("gpu")
from desc.grid import Grid
import desc.io
from desc.backend import jnp
import scipy.constants
import matplotlib.pyplot as plt
import numpy as np
from time import time as timet
import desc.equilibrium
from desc.objectives import ParticleTracer, ObjectiveFunction, ForceBalance, FixBoundaryR, FixBoundaryZ, FixPressure, FixIota, FixPsi
import matplotlib.pyplot as plt
import scipy
from desc.geometry import FourierRZToroidalSurface
from desc.continuation import solve_continuation_automatic

initial_time = timet()

# INITIAL STATE AND PARAMETERS
# Energy and Mass info
Energy_eV = 1 #3.52e6
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

r01 = np.arange(-0.2, 0.2, 0.025)

f_values = []

for R01 in r01:
    print(f"R01: {R01}; Tteration: {r01.tolist().index(R01)} of {len(r01)}")
    surf = FourierRZToroidalSurface(
        R_lmn=[1, 0.125, R01], #alterar 0.1
        Z_lmn=[-0.125, -0.1],
        modes_R=[[0, 0], [1, 0], [0, 1]],
        modes_Z=[[-1, 0], [0, -1]],
        NFP=3,
    )
    eq = desc.equilibrium.Equilibrium(M=2, N=2, Psi=1, surface=surf)
    eq = solve_continuation_automatic(eq, objective="force", bdry_step=0.5, verbose=3)[-1]
    eq._iota = eq.get_profile("iota").to_powerseries(order=eq.L, sym=True)
    eq._current = None
    # eq.solve()

    grid = Grid(jnp.array([jnp.sqrt(psi_i), theta_i, zeta_i]).T, jitable=True, sort=False)
    data = eq.compute("|B|", grid=grid)

    mu = Energy_SI/(Mass*data["|B|"]) - (vpar_i**2)/(2*data["|B|"])

    ini_param = [float(mu), Mass_Charge_Ratio]

    intermediate_time = timet()

    objective = ParticleTracer(eq=eq, output_time=time, initial_conditions=ini_cond, initial_parameters=ini_param, compute_option="optimization", tolerance=1.4e-8)

    objective.build()
    solution = objective.compute(*objective.xs(eq))

    intermediate_time_2 = timet()
    print(f"Time to build and compute R01{R01}: {intermediate_time_2 - intermediate_time}s")

    print("*************** SOLUTION .compute() ***************")
    print(solution)
    print("***************************************************")
    f_values.append(solution)

f = np.asarray(f_values)
plt.plot(r01, f, 'o', color='red', label='F function values for R01')
plt.xlabel('R01')
plt.ylabel('F function values')
plt.savefig('f_values.png')

np.savetxt('r01_f_values.txt', np.c_[r01, f], delimiter=' ')
final_time = timet()
print(f"Total time: {final_time - initial_time}s")
print("*********************** END ***********************")

