from desc import set_device
set_device("gpu")
from desc.objectives import ParticleTracer
from desc.grid import Grid
import desc.io
from desc.backend import jnp
import scipy.constants
import matplotlib.pyplot as plt
import numpy as np

# Load Equilibrium
eq = desc.io.load("input.nearaxis_6.82_output.h5")[-1]
eq._iota = eq.get_profile("iota").to_powerseries(order=eq.L, sym=True)
eq._current = None
eq.solve()

# Output the resulting solution to a .txt file, in 4 columns (psi, theta, zeta, vpar)
def output_to_file(solution, name):
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


# Energy and Mass info
Energy_eV = 3.52e6
Proton_Mass = scipy.constants.proton_mass
Proton_Charge = scipy.constants.elementary_charge
Energy_SI = Energy_eV*Proton_Charge

# Particle Info
Mass = 4*Proton_Mass
Charge = 2*Proton_Charge

# Initial State
psi_i = 0.5
zeta_i = 0
theta_i = 0
vpar_i = 0.3*jnp.sqrt(2*Energy_SI/Mass)
ini_cond = [float(psi_i), theta_i, zeta_i, float(vpar_i)]

# Time
tmin = 0
tmax = 1e-4
nt = 100
time = jnp.linspace(tmin, tmax, nt)

initial_conditions = ini_cond
Mass_Charge_Ratio = Mass/Charge

grid = Grid(jnp.array([jnp.sqrt(psi_i), theta_i, zeta_i]).T, jitable=True, sort=False)
data = eq.compute(["|B|", "R"], grid=grid)

mu = Energy_SI/(Mass*data["|B|"]) - (vpar_i**2)/(2*data["|B|"])

ini_param = [float(mu), Mass_Charge_Ratio]

objective = ParticleTracer(eq=eq, output_time=time, initial_conditions=ini_cond, initial_parameters=ini_param, compute_option="tracer", tolerance=1.4e-8)

print(f"Initial Velocity (parallel component): {vpar_i}")
print(f"Radius: {data['R']}")
print(f"Magnetic Field (abs): {data['|B|']}")
print(f"μ: {mu}")
print(f"Gyroradius: {Mass/Charge*jnp.sqrt(2*mu/data['|B|'])}") #GyroRadius
print(f"Gyrofrequency: {Charge*data['|B|']/Mass}") #Gyrofrequency

objective.build()
solution = objective.compute(*objective.xs(eq))

print("*************** SOLUTION .compute() ***************")
print(solution)
print("***************************************************")

output_to_file(solution=solution, name="comparison_output_nearaxis_6.82")

# PLOT FUNCTIONS

def Trajectory_Plot():
    fig, ax = plt.subplots()
    ax.plot(np.sqrt(solution[:, 0]) * np.cos(solution[:, 1]), np.sqrt(solution[:, 0]) * np.sin(solution[:, 1]))
    ax.set_aspect("equal", adjustable='box')
    plt.xlabel(r'$\sqrt$($\psi$)cos($\theta$)')
    plt.ylabel(r'$\sqrt$($\psi$)sin($\theta$)')
    fig.savefig("Trajectory_Plot.png")

    
def Quantity_Plot():
    fig, axs = plt.subplots(2, 2)
    axs[0, 1].plot(time, solution[:, 0], 'tab:orange')
    axs[0, 1].set_title(r'$\psi$ (t)')
    axs[1, 0].plot(time, solution[:, 1], 'tab:green')
    axs[1, 0].set_title(r'$\theta$ (t)')
    axs[1, 1].plot(time, solution[:, 2], 'tab:red')
    axs[1, 1].set_title(r'$\zeta$ (t)')
    axs[0, 0].plot(time, solution[:, 3], 'tab:blue')
    axs[0, 0].set_title(r"$v_{\parallel}$ (t)")
    fig = plt.gcf()
    fig.set_size_inches(10.5, 10.5)
    fig.savefig("Quantity_Plot.png")

def Energy_Plot():
    grid = Grid(np.vstack((np.sqrt(solution[:, 0]), solution[:, 1], solution[:, 2])).T,sort=False)
    B_field = eq.compute("|B|", grid=grid)
    Energy = 0.5*(solution[:, 3]**2 + 2*B_field["|B|"]*mu)*Mass

    plt.plot(time, (Energy-Energy_SI)/Energy_SI)
    plt.title(r"(E - E$_0$)/E$_0$")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.savefig("Energy_Plot.png")