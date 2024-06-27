# from desc import set_device
# set_device("gpu")
from desc.objectives import ParticleTracer
from desc.grid import Grid
import desc.io
from desc.backend import jnp
import scipy.constants
import matplotlib.pyplot as plt
import numpy as np
from time import time as timet

initial_time = timet()

filename = "eq_2411_M1_N1.h5"
eq = desc.io.load(filename)
save_text_name = "solution" + filename

print("*************** Start ***************")
print("Particle Tracer")
print(f"Filename: {filename}")
print("*************************************")

eq._iota = eq.get_profile("iota").to_powerseries(order=eq.L, sym=True)
eq._current = None
# eq.solve()
# eq.Psi = eq.Psi/6

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
Energy_eV = 100
Proton_Mass = scipy.constants.proton_mass
Proton_Charge = scipy.constants.elementary_charge
Energy_SI = Energy_eV*Proton_Charge

# Particle Info
Mass = 4*Proton_Mass
Charge = 2*Proton_Charge

# Initial State
psi_i = 0.8
zeta_i = 0.5
theta_i = jnp.pi/2
vpar_i = 0.7*jnp.sqrt(2*Energy_SI/Mass)
ini_cond = jnp.array([float(psi_i), theta_i, zeta_i, float(vpar_i)])

# Time
tmin = 0
tmax = 1e-6
nt = 500
time = jnp.linspace(tmin, tmax, nt)

initial_conditions = ini_cond
Mass_Charge_Ratio = Mass/Charge

grid = Grid(nodes=jnp.array([jnp.sqrt(psi_i), theta_i, zeta_i]).T, jitable=False, sort=False)
data = eq.compute(["|B|", "R"], grid=grid)

mu = Energy_SI/(Mass*data["|B|"]) - (vpar_i**2)/(2*data["|B|"])

ini_param = jnp.array([mu[0], Mass_Charge_Ratio])

objective = ParticleTracer(eq=eq, output_time=time, initial_conditions=ini_cond, initial_parameters=ini_param, compute_option="tracer", tolerance=1.e-7)

print(f"Initial Velocity (parallel component): {vpar_i}")
print(f"Radius: {data['R']}")
print(f"Magnetic Field (abs): {data['|B|']}")
print(f"μ: {mu}")
print(f"Gyroradius: {Mass/Charge*jnp.sqrt(2*mu/data['|B|'])}") #GyroRadius
print(f"Gyrofrequency: {Charge*data['|B|']/Mass}") #Gyrofrequency

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

# PLOT FUNCTIONS

def Trajectory_Plot(solution=solution, save_name="Trajectory_Plot.png"):
    fig, ax = plt.subplots()
    ax.plot(np.sqrt(solution[:, 0]) * np.cos(solution[:, 1]), np.sqrt(solution[:, 0]) * np.sin(solution[:, 1]))
    ax.set_aspect("equal", adjustable='box')
    plt.xlabel(r'$\sqrt{\psi}cos(\theta)$')
    plt.ylabel(r'$\sqrt{\psi}sin(\theta)$')
    fig.savefig(save_name, bbox_inches="tight", dpi=300)
    print(f"Trajectory Plot Saved: {save_name}")

def Quantity_Plot(solution=solution, save_name="Quantity_Plot.png"):
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
    fig.savefig(save_name, bbox_inches="tight", dpi=300)
    print(f"Quantity Plot Saved: {save_name}")

def Energy_Plot(solution=solution, save_name="Energy_Plot.png"):
    plt.figure()
    grid = Grid(np.vstack((np.sqrt(solution[:, 0]), solution[:, 1], solution[:, 2])).T,sort=False)
    B_field = eq.compute("|B|", grid=grid)
    Energy = 0.5*(solution[:, 3]**2 + 2*B_field["|B|"]*mu)*Mass

    plt.plot(time, (Energy-Energy_SI)/Energy_SI)
    plt.title(r"(E - E$_0$)/E$_0$")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.savefig(save_name, bbox_inches="tight", dpi=300)
    print(f"Energy Plot Saved: {save_name}")

Trajectory_Plot()
Quantity_Plot()
Energy_Plot()

print("Particle Tracer Complete")
if Trajectory_Plot or Quantity_Plot or Energy_Plot:
    print("Plots Saved")
print(f"Solution File Name: {save_text_name}.txt")
final_time = timet()
print(f"Total Time: {final_time - initial_time}s")
print("*************** END ***************")