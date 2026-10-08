"""Example script for creating a QA configuration at the scale of the ARIES-CS reactor.

It has the boundary of the "precise QA" configuration, but is solved at finite-beta
and optimized for a self-consistent bootstratp current using the temperature and density
profiles of Landreman, Buller, and Drevlak.
"""

from desc import set_device

set_device("gpu")

import numpy as np

from desc.compat import rescale
from desc.examples import get
from desc.grid import LinearGrid
from desc.objectives import (
    BootstrapRedlConsistency,
    FixAtomicNumber,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixElectronDensity,
    FixElectronTemperature,
    FixIonTemperature,
    FixPsi,
    ForceBalance,
    ObjectiveFunction,
)
from desc.profiles import PowerSeriesProfile

# same boundary as the precise_QA equilibrium, but at finite beta
eq = get("precise_QA")

# kinetic profiles
a = 1.7  # minor radius (ARIES-CS scale)
B = 5.86  # volume average B (ARIES-CS scale)
n0 = 2.38e20  # density on axis (<beta>=2.5%)
T0 = 9.45e3  # density on axis (<beta>=2.5%)
eq.pressure = None
eq.current = PowerSeriesProfile(np.zeros(eq.L + 1), np.arange(eq.L + 1), sym=False)
eq.atomic_number = PowerSeriesProfile([1], [0])
eq.electron_density = PowerSeriesProfile(
    n0 * np.array([1, -1]), np.array([0, 10]), sym=True
)
eq.electron_temperature = PowerSeriesProfile(
    T0 * np.array([1, -1]), np.array([0, 2]), sym=True
)
eq.ion_temperature = PowerSeriesProfile(
    T0 * np.array([1, -1]), np.array([0, 2]), sym=True
)

# solve at finite beta
eq, _ = eq.solve(verbose=3, copy=True)

# scale to ARIES-CS reactor size
eq = rescale(eq, L=("a", a), B=("<B>", B), scale_pressure=False, copy=True, verbose=1)

# optimize for self-consistent bootstrap current
grid = LinearGrid(
    rho=np.linspace(1 / eq.L_grid, 1, eq.L_grid) - 1 / (2 * eq.L_grid),
    M=eq.M_grid,
    N=eq.N_grid,
    NFP=eq.NFP,
    sym=True,
)
objective = ObjectiveFunction(
    BootstrapRedlConsistency(eq=eq, grid=grid, helicity=(1, 0))
)
constraints = (
    FixAtomicNumber(eq=eq),
    FixBoundaryR(eq=eq),
    FixBoundaryZ(eq=eq),
    FixCurrent(eq=eq, indices=np.array([0, 1])),
    FixElectronDensity(eq=eq),
    FixElectronTemperature(eq=eq),
    FixIonTemperature(eq=eq),
    FixPsi(eq=eq),
    ForceBalance(eq=eq),
)
eq, _ = eq.optimize(
    objective=objective,
    constraints=constraints,
    optimizer="proximal-lsq-exact",
    maxiter=50,
    gtol=1e-16,
    verbose=3,
    copy=True,
)

# save equilibrium
eq.save("reactor_QA_output.h5")
