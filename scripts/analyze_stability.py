"""
Post-processing script: loads saved equilibria and eigenfunctions from the
iota scan, then computes:
  1. Mercier criterion (D_Mercier and its four sub-terms vs rho)
  2. delta_W term decomposition (xi^T A_term xi for each energy group)
  3. Sanity check: sum of energy terms == lambda * ||xi||^2

Saves all results and produces summary plots. Run after run_newcomb.py.
"""

from desc import set_device

set_device("gpu")

import time
import os
import numpy as np
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from desc.io import load
from desc.grid import LinearGrid, Grid
from desc.diffmat_utils import DiffMat, legendre_diffmat, fourier_diffmat
from desc.equilibrium.coords import map_coordinates
from desc.integrals.quad_utils import leggauss_lob, automorphism_staircase1
from desc.compute._stability import term_by_term_stability
from desc.compute.utils import get_params, get_transforms

# ── Settings (must match run_newcomb.py) ─────────────────────────────────────
save_path = "./high_aspect_ratio_tokamak/"
plot_path = save_path + "eigenfunction_plots/"
os.makedirs(plot_path, exist_ok=True)

aspect_ratio = 200
NFP = 1
axisym = False
n_mode_axisym = 0
p_coeffs = np.array([0.125, 0, 0, 0, -0.125])

n_rho_hi = 64
n_theta_hi = 64
n_zeta_hi = 12

iota_on_axis_values = np.linspace(0.6, 1.5, 10)

ENERGY_KEYS = [
    "Q²",
    "ξ^ρ (𝐉 × ∇ρ)/|∇ ρ|² ⋅ 𝐐",
    "𝐐 ⋅(𝐉 × ∇ρ)/|∇ ρ|² ξ^ρ",
    "|J|² drive",
    "compressibility",
    "finite-n instability drive term",
]
ENERGY_LABELS = {
    "Q²": r"$|\mathbf{Q}|^2$",
    "ξ^ρ (𝐉 × ∇ρ)/|∇ ρ|² ⋅ 𝐐": r"$\xi^\rho(\mathbf{J}\times\nabla\rho)/|\nabla\rho|^2\cdot\mathbf{Q}$",
    "𝐐 ⋅(𝐉 × ∇ρ)/|∇ ρ|² ξ^ρ": r"$\mathbf{Q}\cdot(\mathbf{J}\times\nabla\rho)/|\nabla\rho|^2\,\xi^\rho$",
    "|J|² drive": r"$|J|^2$ drive",
    "compressibility": r"$\gamma p_0\,|\nabla\cdot\xi|^2$ (compressibility)",
    "finite-n instability drive term": r"instability drive",
}
ENERGY_COLORS = {
    "Q²": "steelblue",
    "ξ^ρ (𝐉 × ∇ρ)/|∇ ρ|² ⋅ 𝐐": "darkorange",
    "𝐐 ⋅(𝐉 × ∇ρ)/|∇ ρ|² ξ^ρ": "goldenrod",
    "|J|² drive": "mediumpurple",
    "compressibility": "seagreen",
    "finite-n instability drive term": "crimson",
}

# ── Build the high-res diffmat (same as run_newcomb.py) ──────────────────────
x_gl, _ = leggauss_lob(n_rho_hi)
rho_hi = automorphism_staircase1(x_gl, eps=1e-2, x_0=0.5, m_1=2.0, m_2=2.0)
dx_f = jax.vmap(
    lambda x_val: jax.grad(automorphism_staircase1, argnums=0)(
        x_val, eps=1e-2, x_0=0.5, m_1=2.0, m_2=2.0
    )
)
scale_vector = 1 / (dx_f(x_gl)[:, None])
scale_vector_inv = dx_f(x_gl)[:, None]

D0, W0 = legendre_diffmat(n_rho_hi)
D0 = D0 * scale_vector
W0 = W0 * scale_vector_inv

theta_hi = jnp.linspace(0.0, 2 * jnp.pi, n_theta_hi, endpoint=False)
D1, W1 = fourier_diffmat(n_theta_hi)

# zeta grid uses NFP=1 (overridden per-equilibrium below via eq.NFP)
# We rebuild per-case so NFP is correct — placeholder here
D2_placeholder, W2_placeholder = fourier_diffmat(n_zeta_hi)

# ── Accumulation containers ───────────────────────────────────────────────────
results_iota0 = []
results_lambda_min = []
results_mercier = []  # (rho, D_M, D_sh, D_cu, D_we, D_ge) per case

# ── Main loop ─────────────────────────────────────────────────────────────────
for iota_0 in iota_on_axis_values:
    save_tag = (
        f"axisym_{axisym}_ar_{aspect_ratio}_NFP_{NFP}"
        f"_p_{'_'.join(p_coeffs.astype(str))}"
        f"_iota0_{iota_0:.4f}_d2iota_{-0.1:.4f}"
    )

    eq_path = save_path + f"equilibrium_{save_tag}.h5"
    xi_rho_path = save_path + f"xi_rho_{save_tag}.npy"
    xi_theta_path = save_path + f"xi_theta_{save_tag}.npy"
    xi_zeta_path = save_path + f"xi_zeta_{save_tag}.npy"
    lam_path = save_path + f"lambda_{save_tag}.npy"

    missing = [
        p
        for p in [eq_path, xi_rho_path, xi_theta_path, xi_zeta_path, lam_path]
        if not os.path.exists(p)
    ]
    if missing:
        print(f"Skipping iota_0={iota_0:.4f}: missing files {missing}")
        continue

    print(f"\n=== iota_0 = {iota_0:.4f} ===")
    eq = load(eq_path)

    lambda_arr = np.load(lam_path)
    lambda_min = float(np.min(lambda_arr))
    results_iota0.append(iota_0)
    results_lambda_min.append(lambda_min)
    print(f"  lambda_min = {lambda_min:.6e}")

    # ── 1. Mercier criterion ──────────────────────────────────────────────────
    mercier_npz = save_path + f"mercier_{save_tag}.npz"
    if os.path.exists(mercier_npz):
        print("  loading cached Mercier data")
        m = np.load(mercier_npz)
        rho_m, D_M, D_sh, D_cu, D_we, D_ge = (
            m["rho"],
            m["D_Mercier"],
            m["D_shear"],
            m["D_current"],
            m["D_well"],
            m["D_geodesic"],
        )
    else:
        print("  computing Mercier criterion")
        mercier_grid = LinearGrid(rho=np.linspace(0.02, 1.0, 200), M=12, N=0)
        md = eq.compute(
            ["D_Mercier", "D_shear", "D_current", "D_well", "D_geodesic", "rho"],
            grid=mercier_grid,
        )
        rho_m = np.array(md["rho"])
        D_M = np.array(md["D_Mercier"])
        D_sh = np.array(md["D_shear"])
        D_cu = np.array(md["D_current"])
        D_we = np.array(md["D_well"])
        D_ge = np.array(md["D_geodesic"])
        np.savez(
            mercier_npz,
            rho=rho_m,
            D_Mercier=D_M,
            D_shear=D_sh,
            D_current=D_cu,
            D_well=D_we,
            D_geodesic=D_ge,
        )

    unstable_frac = np.mean(D_M < 0)
    print(f"  Mercier: {unstable_frac*100:.1f}% of domain unstable (D_Mercier < 0)")
    results_mercier.append((rho_m, D_M, D_sh, D_cu, D_we, D_ge))

    # ── 2. delta_W energy decomposition ──────────────────────────────────────
    energy_npz = save_path + f"energy_terms_{save_tag}.npz"
    print("  computing energy term decomposition")

    # Rebuild zeta grid with correct NFP
    zeta_hi = jnp.linspace(0.0, 2 * jnp.pi / eq.NFP, n_zeta_hi, endpoint=False)
    D2, W2 = fourier_diffmat(n_zeta_hi)

    diffmat = DiffMat(
        D_rho=D0,
        W_rho=W0,
        D_theta=D1,
        W_theta=W1,
        D_zeta=D2,
        W_zeta=W2,
    )

    grid0 = LinearGrid(rho=rho_hi, theta=theta_hi, zeta=zeta_hi, NFP=1, sym=False)
    reshaped_nodes = jnp.reshape(
        grid0.meshgrid_reshape(grid0.nodes, order="rtz"),
        (n_rho_hi * n_theta_hi * n_zeta_hi, 3),
    )

    print("  mapping coordinates")
    rtz_nodes = map_coordinates(
        eq,
        reshaped_nodes,
        inbasis=("rho", "theta_PEST", "zeta"),
        outbasis=("rho", "theta", "zeta"),
        period=(jnp.inf, 2 * jnp.pi, jnp.inf),
        tol=1e-12,
        maxiter=50,
    )
    grid = Grid(rtz_nodes)

    # Load saved eigenfunction components and flatten to preconditioned vector
    xi_rho = np.load(xi_rho_path)  # (n_rho, n_theta, n_zeta)
    xi_theta = np.load(xi_theta_path)
    xi_zeta = np.load(xi_zeta_path)
    v = jnp.concatenate([xi_rho.flatten(), xi_theta.flatten(), xi_zeta.flatten()])
    v = v / jnp.linalg.norm(v)

    print("  running xi^T A_term xi for each delta_W group")
    tic = time.time()
    params = get_params("finite-n lambda", eq)
    data_keys = [
        "g_rr|PEST",
        "g_rv|PEST",
        "g_rp|PEST",
        "g_vv|PEST",
        "g_vp|PEST",
        "g_pp|PEST",
        "g^rr",
        "g^rv",
        "g^rz",
        "J^theta_PEST",
        "J^zeta",
        "|J|",
        "sqrt(g)_PEST",
        "(sqrt(g)_PEST_r)|PEST",
        "(sqrt(g)_PEST_v)|PEST",
        "(sqrt(g)_PEST_p)|PEST",
        "finite-n instability drive",
        "iota",
        "psi_r",
        "psi_rr",
        "p",
        "a",
    ]
    data = eq.compute(data_keys, grid=grid)
    transforms = get_transforms("finite-n lambda", eq, grid, diffmat=diffmat)
    energy_data = term_by_term_stability(
        v,
        params,
        transforms,
        data,
        diffmat=diffmat,
        gamma=100,
        incompressible=False,
        axisym=axisym,
    )
    toc = time.time()
    print(f"  done in {toc-tic:.1f} s")
    print(energy_data)
    np.savez(
        energy_npz,
        **energy_data,
        iota_0=iota_0,
    )
    print(
        f"iota= {iota_0}, lambda = {lambda_min}, sum={np.sum(np.fromiter(energy_data[k] for k in energy_data.keys()))}"
    )
    print(
        f"  sanity check: sum of energy terms = {np.sum(np.fromiter(energy_data[k] for k in energy_data.keys())):.6e} vs lambda*||xi||^2 = {lambda_min * jnp.linalg.norm(v)**2:.6e}"
    )
# ── Convert to arrays ─────────────────────────────────────────────────────────
results_iota0 = np.array(results_iota0)
results_lambda_min = np.array(results_lambda_min)

# ── Plot 1: Mercier D_Mercier vs rho ─────────────────────────────────────────
cmap_lines = plt.get_cmap("plasma")
colors_scan = [cmap_lines(v) for v in np.linspace(0.1, 0.9, len(results_iota0))]

fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(0, color="gray", lw=0.8, ls="--")
for (rho_m, D_M, *_), iota_0, color in zip(results_mercier, results_iota0, colors_scan):
    ax.plot(rho_m, D_M, color=color, lw=1.8, label=rf"$\iota_0={iota_0:.2f}$")
ax.set_xlabel(r"$\rho$", fontsize=14)
ax.set_ylabel(r"$D_{\mathrm{Mercier}}$  (Wb$^{-2}$)", fontsize=13)
ax.set_title(
    r"Mercier criterion vs $\rho$  (positive = stable)" + "\n"
    r"$\iota(\rho)=\iota_0-0.05\,\rho^2$",
    fontsize=12,
)
ax.tick_params(labelsize=12)
ax.legend(fontsize=9, ncol=2, loc="upper right")
plt.tight_layout()
fig.savefig(plot_path + "iota_scan_mercier.png", dpi=150)
plt.show()

# ── Plot 2: delta_W energy terms vs iota_0 ───────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.axvline(1, color="gray", lw=0.8, ls=":", alpha=0.6)

for k in energy_data.keys():
    normalized = energy_data[k] / jnp.linalg.norm(v) ** 2
    ax.plot(
        results_iota0,
        normalized,
        "o-",
        lw=2,
        ms=6,
        color=ENERGY_COLORS[k],
        label=ENERGY_LABELS[k],
    )

ax.plot(
    results_iota0,
    results_lambda_min,
    "k^--",
    lw=1.5,
    ms=7,
    label=r"$\lambda_{\min}$ (eigenvalue)",
)

ax.set_xlabel(r"$\iota_0$", fontsize=14)
ax.set_ylabel(r"$\xi^T A_{\mathrm{term}} \xi \;/\; \|\xi\|^2$", fontsize=13)
ax.set_title(
    r"$\delta W$ term contributions vs $\iota_0$" + "\n"
    r"$\iota(\rho)=\iota_0-0.05\,\rho^2$,  stabilizing $>0$, destabilizing $<0$",
    fontsize=12,
)
ax.tick_params(labelsize=12)
ax.legend(fontsize=10, loc="best")
plt.tight_layout()
fig.savefig(plot_path + "iota_scan_energy_terms.png", dpi=150)
plt.show()

print("Done. Plots saved to", plot_path)
