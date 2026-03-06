import sys
import time
from pathlib import Path

import numpy as np

from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.field.trajectory_helpers import TrappedPoincare
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from firm3d.util.functions import proc0_print, setup_logging
from firm3d.util.mpi import comm_size, comm_world, verbose

equil_filename = sys.argv[1] if len(sys.argv) > 1 else "wout_precise_QH_output.nc"
# Resolve to absolute path so the file is found regardless of cwd
equil_path = Path(equil_filename)
if not equil_path.is_absolute():
    equil_path = Path.cwd() / equil_path
if not equil_path.exists():
    raise FileNotFoundError(f"Equilibrium file not found: {equil_path}")
equil_filename = str(equil_path.resolve())

# Output prefix from filename (without .nc) for unique output files
output_prefix = equil_path.stem

charge = ALPHA_PARTICLE_CHARGE
mass = ALPHA_PARTICLE_MASS
Ekin = FUSION_ALPHA_PARTICLE_ENERGY

resolution = 48  # Resolution for field interpolation
neta_poinc = 5  # Number of eta initial conditions for poincare
ns_poinc = 120  # Number of s initial conditions for poincare
Nmaps = 3000  # Number of Poincare return maps to compute
ns_interp = resolution  # number of radial grid points for interpolation
ntheta_interp = resolution  # number of poloidal grid points for interpolation
nzeta_interp = resolution  # number of toroidal grid points for interpolation
order = 3  # order for interpolation
tol = 1e-8  # Tolerance for ODE solver
helicity_M = 0  # helicity of field strength contours
degree = 3  # Degree for Lagrange interpolation

# Setup logging to redirect output to file
setup_logging(f"stdout_trapped_map_{output_prefix}_{resolution}_{comm_size}.txt")

time1 = time.time()

bri = BoozerRadialInterpolant(equil_filename, order, no_K=True, comm=comm_world)
helicity_N = bri.nfp

field = InterpolatedBoozerField(
    bri,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,
)

s_grid = np.linspace(0, 1, 100)
theta_grid = np.linspace(0, 2*np.pi, 100)
zeta_grid = np.linspace(0, 2*np.pi/field.nfp, 100)

s_grid, theta_grid, zeta_grid = np.meshgrid(s_grid, theta_grid, zeta_grid, indexing="ij")

points = np.zeros((s_grid.size, 3))
points[:, 0] = s_grid.flatten()
points[:, 1] = theta_grid.flatten()
points[:, 2] = zeta_grid.flatten()

field.set_points(points)
modB = field.modB()
minB = np.min(modB)
maxB = np.max(modB)

proc0_print("minB: ", minB)
proc0_print("maxB: ", maxB)

lam_range = np.linspace(1/maxB, 1/minB, 10)

for lam in lam_range:
    poinc = TrappedPoincare(
        field,
        helicity_M,
        helicity_N,
        mass,
        charge,
        Ekin,
        lam = lam,
        ns_poinc=ns_poinc,
        neta_poinc=neta_poinc,
        Nmaps=Nmaps,
        comm=comm_world,
        solver_options={"reltol": tol, "abstol": tol, "axis": 0},
        tmax=1e-3,
        )

    if verbose:
        poinc.plot_poincare(filename=f"{output_prefix}_trapped_map_{1/lam:.2f}.pdf")
        np.savez(
            f"{output_prefix}_poincare_data_Bcrit_{1/lam:.2f}.npz",
            s_all=poinc.s_all,
            chis_all=poinc.chis_all,
            etas_all=poinc.etas_all,
            t_all=poinc.t_all,
            lam=poinc.lam,
            modBcrit=poinc.modBcrit,
        )

time2 = time.time()

proc0_print("poincare time: ", time2 - time1)
