from desc.objectives import ParticleTracer, ObjectiveFunction
from desc.examples import get
from desc.grid import Grid
import desc.io
from desc.backend import jnp
import matplotlib.pyplot as plt
import numpy as np

eq = desc.io.load("test_run.h5")
eq._iota = eq.get_profile("iota").to_powerseries(order=eq.L, sym=True)
eq._current = None
# eq.solve()

tmin = 0
tmax = 0.0007
nt = 500
time = jnp.linspace(tmin, tmax, nt)
psi_i = 0.7
theta_i = 0.2
zeta_i = 0.2

mass = 1.673e-27
Energy = 3.52e6*1.6e-19
ini_vpar = 0.5*jnp.sqrt(2*Energy/mass)
ini_cond = [psi_i, theta_i, zeta_i, float(ini_vpar)]

mass_charge = mass/1.6e-19

grid = Grid(jnp.array([jnp.sqrt(psi_i), theta_i, zeta_i]).T, jitable=True, sort=False)
data = eq.compute("|B|", grid=grid)

mu = Energy/(mass*data["|B|"]) - (ini_vpar**2)/(2*data["|B|"])

ini_param = [float(mu), mass_charge]

objective = ParticleTracer(eq=eq, output_time=time, initial_conditions=ini_cond, initial_parameters=ini_param, compute_option="tracer")

objective.build()
solution = objective.compute(*objective.xs(eq))

print("*************** SOLUTION .compute() ***************")
print(solution)
print("***************************************************")

f = open("output_file.txt", "w")
f.write(f"{solution[:, 0]}, {solution[:, 1]}, {solution[:, 2]}, {solution[:, 3]}, {mu}")
f.close()

plt.plot(np.sqrt(solution[:, 0]) * np.cos(solution[:, 1]), np.sqrt(solution[:, 0]) * np.sin(solution[:, 1]))
plt.savefig("trajectory.png")

fig, axs = plt.subplots(2, 2)
axs[0, 1].plot(t, solution[:, 0], 'tab:orange')
axs[0, 1].set_title(r'$\psi$ (t)')
axs[1, 0].plot(t, solution[:, 1], 'tab:green')
axs[1, 0].set_title(r'$\theta$ (t)')
axs[1, 1].plot(t, solution[:, 2], 'tab:red')
axs[1, 1].set_title(r'$\zeta$ (t)')
axs[0, 0].plot(t, solution[:, 3], 'tab:blue')
axs[0, 0].set_title(r"$v_{\parallel}$ (t)")

fig = plt.gcf()
fig.set_size_inches(10.5, 10.5)

fig.savefig("quantities.png", dpi = 300)


"""
objective = ParticleTracer(eq=eq, output_time=time, initial_conditions=ini_cond, initial_parameters=ini_param)
objective.compute()
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
xs = objective.xs(eq)
print("*************** xs **************")
print(xs)
print("*********************************")
"""