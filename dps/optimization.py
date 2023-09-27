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

eq = desc.io.load("input.final_freeb_output.h5")[-1]
eq._iota = eq.get_profile("iota").to_powerseries(order=eq.L, sym=True)
eq._current = None
eq.solve()


def output_to_file(solution, filename):
    list1 = solution[:, 0]
    list2 = solution[:, 1]
    list3 = solution[:, 2]
    list4 = solution[:, 3]

    combined_lists = zip(list1, list2, list3, list4)

    file_name = filename

    with open(file_name, 'w') as file:
        for row in combined_lists:
            row_str = '\t'.join(map(str, row))
            file.write(row_str + '\n')


mass = 1.673e-27
Energy = 3.52e6*1.6e-19

psi_init = 0.8
zeta_init = 0.1
theta_init = 0.2
v_init = 0.7*jnp.sqrt(2*Energy/mass)

ini_cond = [float(psi_init), theta_init, zeta_init, float(v_init)]
print(ini_cond)

tmin = 0
tmax = 0.00007
nt = 200
time = jnp.linspace(tmin, tmax, nt)

mass = 4*1.673e-27
Energy = 3.52e6*1.6e-19
psi_i = ini_cond[0]
theta_i = ini_cond[1]
zeta_i = ini_cond[2]
vpar = ini_cond[3]

initial_conditions = ini_cond
mass_charge = mass/1.6e-19

grid = Grid(jnp.array([jnp.sqrt(psi_i), theta_i, zeta_i]).T, jitable=True, sort=False)
data = eq.compute("|B|", grid=grid)

mu = Energy/(mass*data["|B|"]) - (vpar**2)/(2*data["|B|"])

ini_param = [float(mu), mass_charge]

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
