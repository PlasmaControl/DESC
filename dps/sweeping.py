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
nt = 200
time = jnp.linspace(tmin, tmax, nt)

initial_conditions = ini_cond
Mass_Charge_Ratio = Mass/Charge

r11 = np.arange(-0.05, 0.05, 0.005)

f_values = []

for R11 in r11:
    print(f"R11: {R11}; Tteration: {r11.tolist().index(R11)} of {len(r11)}")
    surf = FourierRZToroidalSurface(
    R_lmn=jnp.array([1, -0.1, R11, 0.03]),  # boundary coefficients
    Z_lmn=jnp.array([0.1, -0.03, -0.03]),
    modes_R=jnp.array(
        [[0, 0], [1, 0], [1, 1], [-1, -1]]
    ),  # [M, N] boundary Fourier modes
    modes_Z=jnp.array([[-1, 0], [-1, 1], [1, -1]]),
    NFP=5,  # number of (toroidal) field periods
)
    eq = desc.equilibrium.Equilibrium(M=1, N=1, Psi=1, surface=surf)
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
    print(f"Time to build and compute R11{R11}: {intermediate_time_2 - intermediate_time}s")

    print("*************** SOLUTION .compute() ***************")
    print(solution)
    print("***************************************************")
    f_values.append(solution)

f = np.asarray(f_values)
plt.plot(r11, f, 'o', color='red', label='F function values for R11')
plt.xlabel('R11')
plt.ylabel('F function values')
plt.savefig('f_values.png')

np.savetxt('r01_f_values.txt', np.c_[r11, f], delimiter=' ')
final_time = timet()
print(f"Total time: {final_time - initial_time}s")
print("*********************** END ***********************")

