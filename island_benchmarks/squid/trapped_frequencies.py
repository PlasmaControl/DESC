import time

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
from firm3d.field.coordinates import boozer_to_vmec 
from pathlib import Path
import sys

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
neta_poinc = 20  # Number of eta initial conditions for poincare
ns_poinc = 50  # Number of s initial conditions for poincare
Nmaps = 1000  # Number of Poincare return maps to compute
ns_interp = resolution  # number of radial grid points for interpolation
ntheta_interp = resolution  # number of poloidal grid points for interpolation
nzeta_interp = resolution  # number of toroidal grid points for interpolation
order = 3  # order for interpolation
tol = 1e-8  # Tolerance for ODE solver
helicity_M = 0  # helicity of field strength contours
degree = 3  # Degree for Lagrange interpolation

# Setup logging to redirect output to file
setup_logging(f"stdout_trapped_frequencies_{resolution}_{comm_size}.txt")

time1 = time.time()

time1_interp = time.time()
bri = BoozerRadialInterpolant(
    equil_filename, order, no_K=True, comm=comm_world,
)
helicity_N = bri.nfp

field = InterpolatedBoozerField(
    bri,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,
)
time2_interp = time.time()
proc0_print("interpolation time: ", time2_interp - time1_interp)

# Compute min and maxB over the volume 

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

if verbose:
    import matplotlib

    matplotlib.use("Agg")  # Don't use interactive backend
    import matplotlib.pyplot as plt

    plt.figure()
    plt.xlabel("s")
    plt.ylabel(r"$\omega_{\alpha}/\omega_b$")

s_prof_list = []
omega_b_prof_list = []
omega_alpha_prof_list = []
for lam in lam_range:
    proc0_print(f"Computing trapped frequencies for lam = {lam}")
    poinc = TrappedPoincare(
        field,
        helicity_M,
        helicity_N,
        mass,
        charge,
        Ekin,
        lam=lam,
        theta_mirror=0,
        zeta_mirror=0,
        ns_poinc=ns_poinc,
        neta_poinc=neta_poinc,
        Nmaps=Nmaps,
        comm=comm_world,
        solver_options={"reltol": tol, "abstol": tol, "axis": 0},
        tmax=1e-3,
    )
    s_map = poinc.s_all
    chis_map = poinc.chis_all
    etas_map = poinc.etas_all
    t_map = poinc.t_all

    omega_alpha_list = []
    omega_b_list = []
    init_s_list = []
    for s_traj, chis_traj, etas_traj, t_traj in zip(s_map, chis_map, etas_map, t_map):
        if (len(s_traj) < 2):  # Need at least one full Poincare return maps to compute frequency
            continue
        # Convert to Boozer coordinates
        theta_traj, zeta_traj = poinc.chi_eta_to_theta_zeta(np.array(chis_traj), np.array(etas_traj))
        # Convert to PEST coordinates
        points_boozer = np.zeros((len(s_traj), 3))
        points_boozer[:, 0] = s_traj
        points_boozer[:, 1] = theta_traj
        points_boozer[:, 2] = zeta_traj
        field.set_points(points_boozer)
        nu = field.nu()[:, 0]
        iota = field.iota()[:, 0]
        phi_traj = zeta_traj - nu 
        vartheta_traj = theta_traj - iota * nu
        alpha_traj = theta_traj - iota * phi_traj
        delta_phi = np.array(phi_traj[1::]) - np.array(phi_traj[0:-1])
        delta_vartheta = np.array(vartheta_traj[1::]) - np.array(vartheta_traj[0:-1])
        delta_alpha = np.array(alpha_traj[1::]) - np.array(alpha_traj[0:-1])
        delta_t = t_traj[1::]
        # Average over wells along one field line
        omega_alpha_list.append(np.mean(delta_alpha) / np.mean(delta_t))
        omega_b_list.append(2 * np.pi / np.mean(delta_t))  # bounce frequency
        init_s_list.append(s_traj[0])

    omega_b = np.array(omega_b_list)
    omega_alpha = np.array(omega_alpha_list)
    init_s = np.array(init_s_list)
    s_init_unique = np.unique(poinc.s_init)

    indices = []
    for i in range(len(init_s)):
        index = np.argmin(np.abs(init_s[i] - s_init_unique))
        indices.append(index)
    indices = np.array(indices)

    # Average profile over eta_init 
    s_profile = []
    omega_b_profile = []
    omega_alpha_profile = []
    for i in range(len(s_init_unique)):
        this_omega_b = omega_b[indices == i]
        this_omega_alpha = omega_alpha[indices == i]
        if len(this_omega_b) > 0:
            omega_b_mean = np.mean(this_omega_b)
            omega_alpha_mean = np.mean(this_omega_alpha)
            s_profile.append(s_init_unique[i])
            omega_b_profile.append(omega_b_mean)
            omega_alpha_profile.append(omega_alpha_mean)
    s_profile = np.array(s_profile)
    omega_b_profile = np.array(omega_b_profile)
    omega_alpha_profile = np.array(omega_alpha_profile)

    if verbose:
        plt.plot(s_profile, omega_alpha_profile/ omega_b_profile, label=f"Bcrit = {poinc.modBcrit}")
        s_prof_list.append(s_profile)
        omega_b_prof_list.append(omega_b_profile)
        omega_alpha_prof_list.append(omega_alpha_profile)

if verbose:
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_trapped_frequencies.png", bbox_inches='tight')

    np.savez(
    f"{output_prefix}_trapped_frequencies.npz",
    lam_values=lam_range,
    s_prof=np.array(s_prof_list, dtype=object),
    omega_b_prof=np.array(omega_b_prof_list, dtype=object),
    omega_alpha_prof=np.array(omega_alpha_prof_list, dtype=object),
    )

time2 = time.time()

proc0_print("poincare time: ", time2 - time1)
