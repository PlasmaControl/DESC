from desc import set_device
set_device("gpu")
from desc.grid import Grid
import desc.io
from desc.backend import jnp
import scipy.constants
import matplotlib.pyplot as plt
from time import time as timet
import desc.equilibrium
from desc.objectives import (ParticleTracer, AspectRatio, ObjectiveFunction, ForceBalance, 
                                FixBoundaryR, FixBoundaryZ, FixPressure, FixIota, FixPsi, FixParameters)
from desc.geometry import FourierRZToroidalSurface
from desc.equilibrium import Equilibrium
from desc.continuation import solve_continuation_automatic


initial_time = timet()

# Functions

def output_to_file(sol, name):
    print(sol.shape)
    list1 = sol[:, 0]
    list2 = sol[:, 1]
    list3 = sol[:, 2]
    list4 = sol[:, 3]

    combined_lists = zip(list1, list2, list3, list4)
    
    file_name = f'{name}.txt'

    with open(file_name, 'w') as file:        
        for row in combined_lists:
            row_str = '\t'.join(map(str, row))
            file.write(row_str + '\n')

def savetxt(data, name):
    print(data.shape)

    combined_lists = zip(data)
    
    file_name = f'{name}.txt'

    with open(file_name, 'w') as file:        
        for row in combined_lists:
            row_str = '\t'.join(map(str, row))
            file.write(row_str + '\n')

def Trajectory_Plot(solution, save_name="Trajectory_Plot.png"):
    fig, ax = plt.subplots()
    ax.plot(jnp.sqrt(solution[:, 0]) * jnp.cos(solution[:, 1]), jnp.sqrt(solution[:, 0]) * jnp.sin(solution[:, 1]))
    ax.set_aspect("equal", adjustable='box')
    plt.xlabel(r'$\sqrt{\psi}cos(\theta)$')
    plt.ylabel(r'$\sqrt{\psi}sin(\theta)$')
    fig.savefig(save_name, bbox_inches="tight", dpi=300)
    print(f"Trajectory Plot Saved: {save_name}")

def Quantity_Plot(solution, save_name="Quantity_Plot.png"):
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

def Energy_Plot(solution, save_name="Energy_Plot.png"):
    plt.figure()
    grid = Grid(jnp.vstack((jnp.sqrt(solution[:, 0]), solution[:, 1], solution[:, 2])).T,sort=False)
    B_field = eq.compute("|B|", grid=grid)
    Energy = 0.5*(solution[:, 3]**2 + 2*B_field["|B|"]*mu)*Mass

    plt.plot(time, (Energy-Energy_SI)/Energy_SI)
    plt.title(r"(E - E$_0$)/E$_0$")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.savefig(save_name, bbox_inches="tight", dpi=300)
    print(f"Energy Plot Saved: {save_name}")

print("*************** START ***************")

# Load Equilibrium

print("\nStarting Equilibrium")
# eq_file = "input.LandremanPaul2021_QA_scaled_output.h5"
# eq_file = "test_equilibrium.h5"
eq_file = "eq_0112_M1_N1.h5"

opt_file = "optimized_" + eq_file
print(f"Loaded Equilibrium: {eq_file}\n")

eq = desc.io.load(eq_file)
eq._iota = eq.get_profile("iota").to_powerseries(order=eq.L, sym=True)
eq._current = None
# eq._pressure = None

# Energy and Mass info
Energy_eV = 10 #1 # eV (3.52e6 eV proton energy)
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
vpar_i = -0.1*jnp.sqrt(2*Energy_SI/Mass)
ini_cond = jnp.array([float(psi_i), theta_i, zeta_i, float(vpar_i)])

# Time
tmin = 0
tmax = 5e-2
nt = 7500
time = jnp.linspace(tmin, tmax, nt)

# Initial State
initial_conditions = ini_cond
Mass_Charge_Ratio = Mass/Charge

# Mu 
grid = Grid(jnp.array([jnp.sqrt(psi_i), theta_i, zeta_i]).T, jitable=False, sort=False)
data = eq.compute("|B|", grid=grid)
mu = Energy_SI/(Mass*data["|B|"]) - (vpar_i**2)/(2*data["|B|"])

# Initial Parameters
# ini_param = jnp.array([float(mu), Mass_Charge_Ratio]) # this is deprecated!
ini_param = jnp.array([mu[0], Mass_Charge_Ratio])       # this works


intermediate_time = timet()
print(f"\nTime from beginning until here: {intermediate_time - initial_time}s\n")

# Objective Function
objective = ParticleTracer(eq=eq, output_time=time, initial_conditions=ini_cond, initial_parameters=ini_param, compute_option="optimization", tolerance=1e-7, deriv_mode="rev")
objective.build()

# ar = jnp.copy(eq.compute("R0/a")["R0/a"])
# aspect_ratio = AspectRatio(eq=eq, target=ar, weight=1e1)
# Compute optimization
solution = objective.compute(*objective.xs(eq))

intermediate_time_2 = timet()
print(f"\nTime to build and compute solution: {intermediate_time_2 - intermediate_time}s\n")

# Objective Object
ObjFunction = ObjectiveFunction((objective), deriv_mode="looped")
ObjFunction.build()

# gradient = ObjFunction.grad(ObjFunction.x(eq))

# savetxt(gradient, "gradient")

intermediate_time_3 = timet()
print(f"\nTime to build and compile ObjFunction: {intermediate_time_3 - intermediate_time_2}s\n")

################################################################################################################
################################################################################################################
################################################# Optimization #################################################
################################################################################################################
################################################################################################################

# R_modes = jnp.vstack(([0, 0, 0], eq.surface.R_basis.modes[jnp.max(jnp.abs(eq.surface.R_basis.modes), 1), :]))
# Z_modes = eq.surface.Z_basis.modes[jnp.max(jnp.abs(eq.surface.Z_basis.modes), 1), :]

R_modes = jnp.array([[0, 0, 0]])
constraints = (ForceBalance(eq, bounds=(-1e-3, 1e-3)), FixBoundaryR(eq, modes=R_modes), FixBoundaryZ(eq, modes=False), FixPsi(eq), FixPressure(eq)) #FixPressure(eq), FixCurrent(eq), FixIota(eq), ForceBalance(eq, bounds=(-1e-3, 1e-3))
eq.optimize(objective=ObjFunction, optimizer = "fmin-auglag-bfgs", constraints=constraints, verbose=3, maxiter=100, copy=False)
eq.save(opt_file)

intermediate_time_4 = timet()
print(f"\nTime to optimize: {intermediate_time_4 - intermediate_time_3}s\n")

print("\nOptimization Completed")
print(f"Optimized Filename: {opt_file}")
optimization_final_time = timet()
print(f"Total time: {optimization_final_time - initial_time}s")
print("*********************** OPTIMIZATION END ***********************\n")

print("\n*************** TRACING ***************")

################################################################################################################
################################################################################################################
################################################## Tracing #####################################################
################################################################################################################
################################################################################################################

tracing_original = ParticleTracer(eq=eq, output_time=time, initial_conditions=ini_cond, initial_parameters=ini_param, compute_option="tracer", tolerance=1.4e-8)

# Compute tracing original equilibrium
eq_again = desc.io.load(eq_file)
eq_again._iota = eq_again.get_profile("iota").to_powerseries(order=eq_again.L, sym=True)
eq_again._current = None

intermediate_time_5 = timet()
tracing_original.build()
tracer_solution_original = tracing_original.compute(*tracing_original.xs(eq_again))
intermediate_time_6 = timet()
print(f"\nTime to build and trace (original): {intermediate_time_6 - intermediate_time_5}s\n")

output_to_file(tracer_solution_original, name="tracing_original")

# Compute tracing optimized equilibrium
opt_eq = desc.io.load(opt_file)
opt_eq._iota = opt_eq.get_profile("iota").to_powerseries(order=opt_eq.L, sym=True)
opt_eq._current = None

tracing_optimized = ParticleTracer(eq=opt_eq, output_time=time, initial_conditions=ini_cond, initial_parameters=ini_param, compute_option="tracer", tolerance=1.4e-8)

intermediate_time_7 = timet()
tracing_optimized.build()
tracer_solution_optimized = tracing_optimized.compute(*tracing_optimized.xs(opt_eq))
intermediate_time_8 = timet()
print(f"\nTime to build and trace (optimized): {intermediate_time_8 - intermediate_time_7}s\n")

output_to_file(tracer_solution_optimized, name="tracing_optimized")

# Comparison
difference = tracer_solution_original - tracer_solution_optimized

output_to_file(difference, name="tracing_difference")

print("\n*********************** TRACING END ***********************\n")

print("\n*************** PLOTTING ***************\n")

print("Original Equilibrium")
Trajectory_Plot(solution=tracer_solution_original, save_name="Trajectory_Plot_original.png")
Quantity_Plot(solution=tracer_solution_original, save_name="Quantity_Plot_original.png")
Energy_Plot(solution=tracer_solution_original, save_name="Energy_Plot_original.png")

print("Optimized Equilibrium")
Trajectory_Plot(solution=tracer_solution_optimized, save_name="Trajectory_Plot_optimized.png")
Quantity_Plot(solution=tracer_solution_optimized, save_name="Quantity_Plot_optimized.png")
Energy_Plot(solution=tracer_solution_optimized, save_name="Energy_Plot_optimized.png")

print("*************** PLOTTING END ***************")

final_time = timet()
print(f"\n\nTotal time: {final_time - initial_time}s\n\n")
print("*************** END ***************")