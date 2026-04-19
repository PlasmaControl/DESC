# A script to set up and solve a high aspect-ratio tokamak equilibrium using DESC,
# and then evaluate its stability using Newcomb's procedure.
#from desc import set_device
#set_device("gpu")

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
from desc.compute._stability import term_by_term_stability
from desc.compute.utils import get_params, get_transforms

# Input parameters
a = 1  # Minor radius
aspect_ratio = 200  # Aspect ratio of the tokamak
R = aspect_ratio * a  # Major radius
NFP = 1
axisym = True  # Whether to enforce axisymmetry in the eigenvalue solve
n_mode_axisym = 1  # toroidal mode number n (axisym case)
m = 1             # poloidal mode number for the kink trial function

# Low-res solve for eigenfunction guess
n_rho = 36
n_theta = 36
if axisym:
    n_zeta = 1
else:
    n_zeta = 14

# Quadratic iota profile: iota(rho) = iota_0 - 0.5*rho^2
iota_on_axis_values = np.linspace(0.8, 1.25, 10)

save_path = "./high_aspect_ratio_tokamak/"
os.makedirs(save_path, exist_ok=True)

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
        f"_iota0_{iota_0:.4f}_d2iota_{2*iota_coeffs[-1]:.4f}"
        f"n_rho_{n_rho}_n_theta_{n_theta}_n_zeta_{n_zeta}"
        f"analytic_trial_function"
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

    
    
    print("making input grid and diffmats")
    x, w = leggauss_lob(n_rho)

    rho = automorphism_staircase1(x, eps=1e-2, x_0=0.5, m_1=2.0, m_2=2.0)
    dx_f = jax.vmap(
        lambda x_val: jax.grad(automorphism_staircase1, argnums=0)(
            x_val, eps=1e-2, x_0=0.5, m_1=2.0, m_2=2.0
        )
    )

    scale_vector = 1 / (dx_f(x)[:, None])
    scale_vector_inv = dx_f(x)[:, None]

    D0, W0 = legendre_diffmat(n_rho)
    D0 = D0 * scale_vector
    W0 = W0 * scale_vector_inv

    theta = jnp.linspace(0.0, 2 * jnp.pi, n_theta, endpoint=False)
    D1, W1 = fourier_diffmat(n_theta)

    zeta = jnp.linspace(0.0, 2 * jnp.pi / eq.NFP, n_zeta, endpoint=False)
    D2, W2 = fourier_diffmat(n_zeta)

    grid0 = LinearGrid(rho=rho, theta=theta, zeta=zeta, NFP=1, sym=False)
    diffmat = DiffMat(D_rho=D0, W_rho=W0, D_theta=D1, W_theta=W1, D_zeta=D2, W_zeta=W2)

    reshaped_nodes = jnp.reshape(
        grid0.meshgrid_reshape(grid0.nodes, order="rtz"), (n_rho * n_theta * n_zeta, 3)
    )
    print("mapping coordinates")
    rtz_nodes = map_coordinates(
        eq,
        reshaped_nodes,
        inbasis=("rho", "theta_PEST", "zeta"),
        outbasis=("rho", "theta", "zeta"),
        period=(jnp.inf, 2 * jnp.pi, jnp.inf),
        tol=1e-12,
        maxiter=50,
    )
    print("coordinates mapped")

    grid = Grid(rtz_nodes)

    # ── q=1 surface ───────────────────────────────────────────────────────────
    # iota(rho) = iota_0 - 0.5*rho^2  =>  q=1 at rho_q1 = sqrt((iota_0-1)/0.5)
    if iota_0 <= 1.0:
        print(f"  iota_0={iota_0:.4f}: no q=1 surface inside plasma, skipping")
        results_lambda_min.append(np.nan)
        continue
    rho_q1 = np.sqrt((iota_0 - 1.0) / 0.5)
    print(f"  q=1 surface at rho_q1 = {rho_q1:.4f}")

    # ── Screw-pinch equilibrium quantities on 1-D radial grid ────────────────
    grid1d = LinearGrid(rho=np.array(rho), theta=0.0, zeta=0.0, NFP=NFP)
    data1d = eq.compute(data_keys + ["a", "R0", "p", "psi_r"], grid=grid1d)
    a_val  = float(data1d["a"])
    R0_val = float(data1d["R0"])
    r_arr  = np.array(rho) * a_val               # physical minor radius (n_rho,)
    k_sc   = -n_mode_axisym / R0_val             # toroidal wavenumber (newcomb sign)

    B_theta_1d = np.array(B_theta_pinch(data1d, r_arr))  # (n_rho,)
    B_z_1d     = np.array(B_z_pinch(data1d, R0_val))      # (n_rho,)
    B_mag_1d   = np.sqrt(B_theta_1d**2 + B_z_1d**2)
    psi_r_1d   = np.array(data1d["psi_r"])                # (n_rho,)
    p_1d       = np.array(data1d["p"])                    # (n_rho,) pressure

    k0sq_1d = k_sc**2 + (m / r_arr)**2                   # (n_rho,)
    Fs_1d, _ = F(np.array(rho), data1d, m=m, k=k_sc, a=a_val, R0=R0_val)
    Fs_1d    = np.array(Fs_1d)
    f_1d, _  = fg(np.array(rho), data1d, m=m, k=k_sc, a=a_val, R0=R0_val)
    f_1d     = np.array(f_1d)
    G_1d     = f_1d / r_arr                               # G = F²/k₀²

    gamma_ad  = 5 / 3
    delta_sq  = gamma_ad * p_1d * k0sq_1d                 # δ²

    # ── Trial function ξ₀(ρ): smooth step, 1 inside q=1, 0 outside ──────────
    eps_step = 0.05  # smoothing width in rho
    xi0   = 0.5 * (1.0 - np.tanh((np.array(rho) - rho_q1) / eps_step))  # (n_rho,)
    # BC: vanish at axis and edge
    xi0[0]  = 0.0
    xi0[-1] = 0.0

    # dξ₀/dρ analytically, then d(rξ₀)/dr = ξ₀ + ρ dξ₀/dρ
    dxi0_drho  = -1.0 / (2.0 * eps_step) / np.cosh((np.array(rho) - rho_q1) / eps_step)**2
    d_r_xi0_dr = xi0 + np.array(rho) * dxi0_drho   # = d(rξ₀)/dr  (units: 1, since r=ρ·a cancels)

    # ── Formula 2: η₀(ρ) = 1/(r k₀²) [2k B_θ ξ₀ + G d(rξ₀)/dr] ────────────
    eta0 = (2 * k_sc * B_theta_1d * xi0 + G_1d * d_r_xi0_dr) / (r_arr * k0sq_1d)  # (n_rho,)

    # ── Formula 1: ξ_∥₀(ρ) = B F/(F²+δ²) ∇·ξ_⊥ ─────────────────────────────
    # ∇·ξ_⊥ for mode e^{i(mθ-nζ)} at amplitude level (angular phase factors absorbed):
    #   D₀ = (1/r) d(rξ₀)/dr - m η₀ B_z/(r B) + n η₀ B_θ/(R B)
    div_xi_perp = (
        d_r_xi0_dr / r_arr
        - m * eta0 * B_z_1d / (r_arr * B_mag_1d)
        + n_mode_axisym * eta0 * B_theta_1d / (R0_val * B_mag_1d)
    )
    xi_par0 = B_mag_1d * Fs_1d / (Fs_1d**2 + delta_sq) * div_xi_perp  # (n_rho,)

    # ── Build 2D (ρ, θ, ζ) arrays ─────────────────────────────────────────────
    # Physical convention on the real grid:
    #   ξ_r(ρ,θ)      = ξ₀(ρ) cos(mθ)
    #   η_phys(ρ,θ)   = -η₀(ρ) sin(mθ)    [factor i → phase shift]
    #   ξ_∥_phys(ρ,θ) = -ξ_∥₀(ρ) sin(mθ)
    #
    # Physical θ/z components:
    #   ξ_θ_phys = η_phys B_z/B + ξ_∥_phys B_θ/B  = -(η₀ B_z + ξ_∥₀ B_θ)/B · sin(mθ)
    #   ξ_z_phys = -η_phys B_θ/B + ξ_∥_phys B_z/B  = (η₀ B_θ - ξ_∥₀ B_z)/B · sin(mθ)
    #
    # Contravariant PEST (screw-pinch approx: e_ρ≈a r̂, e_θ≈r θ̂, e_ζ≈R ẑ):
    #   ξ^ρ = ξ_r / a
    #   ξ^θ = ξ_θ_phys / r
    #   ξ^ζ = ξ_z_phys / R
    #
    # Storage convention: xi_rho_stored = ξ^ρ / psi_r

    theta_arr = np.array(theta)                    # (n_theta,)
    cos_mt    = np.cos(m * theta_arr)              # (n_theta,)
    sin_mt    = np.sin(m * theta_arr)

    # Radial amplitudes, shape (n_rho,) — broadcast over theta below
    amp_rho   = xi0 / (a_val * psi_r_1d)                                  # ξ^ρ / psi_r
    amp_theta = -(eta0 * B_z_1d + xi_par0 * B_theta_1d) / (r_arr * B_mag_1d)   # ξ^θ amplitude
    amp_zeta  = (eta0 * B_theta_1d - xi_par0 * B_z_1d) / (R0_val * B_mag_1d)   # ξ^ζ amplitude

    # (n_rho, n_theta, n_zeta=1)
    xi_rho_low   = (amp_rho[:, None, None]   * cos_mt[None, :, None]).reshape(n_rho, n_theta, n_zeta)
    xi_theta_low = (amp_theta[:, None, None] * (-sin_mt[None, :, None])).reshape(n_rho, n_theta, n_zeta)
    xi_zeta_low  = (amp_zeta[:, None, None]  * (-sin_mt[None, :, None])).reshape(n_rho, n_theta, n_zeta)

    # Enforce Dirichlet BC: ξ^ρ = 0 at ρ=0 and ρ=1
    xi_rho_low[0,  :, :] = 0.0
    xi_rho_low[-1, :, :] = 0.0

    np.save(save_path + f"xi_rho_low_{save_tag}.npy",   xi_rho_low)
    np.save(save_path + f"xi_theta_low_{save_tag}.npy", xi_theta_low)
    np.save(save_path + f"xi_zeta_low_{save_tag}.npy",  xi_zeta_low)

    # ── term_by_term_stability ────────────────────────────────────────────────
    print("  running term_by_term_stability")
    tic = time.time()

    tbt_data_keys = [
        "g_rr|PEST", "g_rv|PEST", "g_rp|PEST",
        "g_vv|PEST", "g_vp|PEST", "g_pp|PEST",
        "g^rr", "g^rv", "g^rz",
        "J^theta_PEST", "J^zeta", "|J|",
        "sqrt(g)_PEST",
        "(sqrt(g)_PEST_r)|PEST",
        "(sqrt(g)_PEST_v)|PEST",
        "(sqrt(g)_PEST_p)|PEST",
        "finite-n instability drive",
        "iota", "psi_r", "psi_rr", "p", "a",
    ]
    tbt_data   = eq.compute(tbt_data_keys, grid=grid)
    params     = get_params("finite-n lambda", eq)
    transforms = get_transforms("finite-n lambda", eq, grid, diffmat=diffmat)

    v = jnp.concatenate([xi_rho_low.flatten(), xi_theta_low.flatten(), xi_zeta_low.flatten()])
    v = v / jnp.linalg.norm(v)

    energy = term_by_term_stability(
        v, params, transforms, tbt_data,
        diffmat=diffmat,
        gamma=100,
        incompressible=False,
        axisym=axisym,
        n_mode_axisym=n_mode_axisym,
        sigma=0,
    )
    toc = time.time()
    print(f"  term_by_term_stability took {toc-tic:.1f} s,  Rayleigh quotient = {energy:.6e}")

    results_lambda_min.append(float(energy))


# ── Save summary ──────────────────────────────────────────────────────────────
results_lambda_min = np.array(results_lambda_min)

np.savez(
    save_path + "iota_scan_results_analytic.npz",
    iota_on_axis=iota_on_axis_values,
    lambda_min=results_lambda_min,
)

print("Done. Results saved to", save_path)
print("Run analyze_stability.py for Mercier + delta_W breakdown.")
