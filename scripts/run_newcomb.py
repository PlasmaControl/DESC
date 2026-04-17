# A script to set up and solve a high aspect-ratio tokamak equilibrium using DESC,
# and then evaluate its stability using Newcomb's procedure.
from desc import set_device
set_device("gpu")

import pdb
import time
from desc.examples import get
from desc.grid import LinearGrid, Grid
from desc.diffmat_utils import *
from desc.equilibrium.coords import map_coordinates
from desc.equilibrium import Equilibrium
from desc.backend import jnp, jax
from matplotlib import pyplot as plt
from desc.utils import dot
from desc.integrals.quad_utils import leggauss_lob, automorphism_staircase1
from desc.io import load
import numpy as np
from scipy.interpolate import RegularGridInterpolator


# Make or load an ultra high aspect-ratio tokamak (essentially a screw pinch)
from desc.equilibrium import Equilibrium
from desc.continuation import solve_continuation_automatic
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile
from desc.grid import QuadratureGrid
import os

from newcomb import *

# Input parameters
a = 1  # Minor radius
aspect_ratio = 200  # Aspect ratio of the tokamak
R = aspect_ratio * a  # Major radius
NFP = 1
axisym = False  # Whether to enforce axisymmetry in the eigenvalue solve
n_mode_axisym = 0  # If axisym is True, the toroidal mode number to solve for

# Quadratic iota profile: iota(rho) = iota_0 - 0.05*rho^2
# => d^2 iota / d rho^2 = -0.1 (decreasing, as requested)
iota_on_axis_values = np.linspace(0.8, 1.25, 10)

save_path = "./high_aspect_ratio_tokamak/"
os.makedirs(save_path, exist_ok=True)

results_iota0 = []
results_lambda_min = []

stabilities = []

for iota_0 in iota_on_axis_values:
    iota_coeffs = np.array([iota_0, -0.5])
    iota_modes  = np.array([0, 2])
    iota_profile = PowerSeriesProfile(iota_coeffs, modes=iota_modes)
    I_profile = None

    p_coeffs = np.array([0.125, 0, 0, 0, -0.125])
    p_profile = PowerSeriesProfile(p_coeffs)

    # Save directory and filename
    save_tag = (
        f"axisym_{axisym}_ar_{aspect_ratio}_NFP_{NFP}"
        f"_p_{'_'.join(p_coeffs.astype(str))}"
        f"_iota0_{iota_0:.4f}_d2iota_{-0.1:.4f}"
    )
    save_name = f"equilibrium_{save_tag}.h5"

    print(f"\n=== iota_0 = {iota_0:.4f} ===")
    print("solving equilibrium")
    eq = Equilibrium(
        L=12,
        M=12,
        N=0,
        surface=FourierRZToroidalSurface.from_shape_parameters(
            major_radius=R,
            aspect_ratio=aspect_ratio,
            elongation=1,
            triangularity=0,
            squareness=0,
            eccentricity=0,
            torsion=0,
            twist=0,
            NFP=NFP,
            sym=True,
        ),
        NFP=NFP,
        iota=iota_profile,
        current=I_profile,
        pressure=p_profile,
        Psi=1,
    )

    eq = solve_continuation_automatic(eq, ftol=1E-13, gtol=1E-13, xtol=1E-13, verbose=0)[-1]
    eq.save(save_path + save_name)
    print("equilibrium solved")

    # Evaluate stability using Newcomb's procedure  
    tic = time.time()
    stabilities.append(evaluate_stability(eq))
    toc = time.time()
    print(f"stability evaluation took {toc - tic:.2f} seconds")

# Plotting results
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(iota_on_axis_values, np.array(stabilities).astype(int), marker='.', linestyle="dashed")
ax.set_xlabel("$\iota_0$ (on-axis iota)")
ax.set_ylabel("Stability (1=stable, 0=unstable)")
ax.vlines(1.0, 0, 1, colors='r', linestyles='dashed', label='$\iota_0=1$')
ax.set_title("Stability of straight tokamak from Newcomb's Procedure")
ax.legend()
fig.savefig(save_path + "stability_vs_iota0.png", dpi=300)