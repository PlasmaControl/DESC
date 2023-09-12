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
tmax = 0.00007
nt = 20
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

objective = ParticleTracer(eq=eq, output_time=time, initial_conditions=ini_cond, initial_parameters=ini_param)

objective.build()
solution = objective.compute(*objective.xs(eq))
print(solution)


objective = ParticleTracer(eq=eq, output_time=time, initial_conditions=ini_cond, initial_parameters=ini_param)
ObjFunction = ObjectiveFunction([objective])
ObjFunction.build()

ObjFunction.compile()

gradient = ObjFunction.grad(ObjFunction.x(eq))
print(gradient)
#print(ObjFunction.x(eq))
xs = objective.xs(eq)
print(xs)
