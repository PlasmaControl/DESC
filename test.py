from desc import set_device

set_device("gpu")

from desc.equilibrium import Equilibrium
from desc.objectives import ObjectiveFunction, QuasiIsodynamic

eq = Equilibrium()
obj = ObjectiveFunction(QuasiIsodynamic(M_booz=eq.M, N_booz=eq.N), eq)
f = obj.compute(obj.x(eq))
