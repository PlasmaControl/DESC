"""
Load saved eigenfunctions from the iota scan and plot:
  - xi^rho, xi^theta, xi^zeta vs rho  at fixed theta=0  (one plot per equilibrium)
  - xi^rho, xi^theta, xi^zeta vs theta at a few rho indices (one plot per equilibrium)
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from desc.integrals.quad_utils import leggauss_lob, automorphism_staircase1

# ── Settings ──────────────────────────────────────────────────────────────────
save_path   = "./high_aspect_ratio_tokamak/"
plot_path   = save_path + "eigenfunction_plots/"
os.makedirs(plot_path, exist_ok=True)

# These must match what run_newcomb.py used for the high-res solve
n_rho   = 64
n_theta = 64
n_zeta  = 1

# rho grid (reproduced from run_newcomb.py)
x, _ = leggauss_lob(n_rho)
rho   = automorphism_staircase1(x, eps=1e-2, x_0=0.5, m_1=2.0, m_2=2.0)
theta = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)

# Fixed parameters used when building the save_tag
aspect_ratio = 200
NFP          = 1
axisym       = True
p_coeffs     = np.array([0.125, 0, 0, 0, -0.125])

# rho indices to use for the "vs theta" plots
rho_plot_indices = [8, 20, 35, 50]
rho_colors       = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

labels     = [r"$\xi^\rho$", r"$\xi^\theta$", r"$\xi^\zeta$"]
comp_colors = ["steelblue", "darkorange", "seagreen"]

# ── Load scan metadata ────────────────────────────────────────────────────────
scan = np.load(save_path + "iota_scan_results.npz")
iota_on_axis_values = scan["iota_on_axis"]

# ── Loop over equilibria ──────────────────────────────────────────────────────
for iota_0 in iota_on_axis_values:
    save_tag = (
        f"axisym_{axisym}_ar_{aspect_ratio}_NFP_{NFP}"
        f"_p_{'_'.join(p_coeffs.astype(str))}"
        f"_iota0_{iota_0:.4f}_d2iota_{-0.1:.4f}"
    )

    xi_rho_path   = save_path + f"xi_rho_{save_tag}.npy"
    xi_theta_path = save_path + f"xi_theta_{save_tag}.npy"
    xi_zeta_path  = save_path + f"xi_zeta_{save_tag}.npy"

    if not all(os.path.exists(p) for p in [xi_rho_path, xi_theta_path, xi_zeta_path]):
        print(f"Skipping iota_0={iota_0:.4f}: files not found")
        continue

    xi_rho   = np.load(xi_rho_path)    # (n_rho, n_theta, n_zeta)
    xi_theta = np.load(xi_theta_path)
    xi_zeta  = np.load(xi_zeta_path)

    title_base = rf"$\iota_0 = {iota_0:.3f}$,  $\iota(\rho) = \iota_0 - 0.05\,\rho^2$"

    # ── Plot 1: xi vs rho at fixed theta=0, zeta=0 ──────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for xi, label, color in zip([xi_rho, xi_theta, xi_zeta], labels, comp_colors):
        ax.plot(rho, xi[:, 0, 0], color=color, lw=2, label=label)
    ax.set_xlabel(r"$\rho$", fontsize=14)
    ax.set_ylabel(r"$\xi$ (arb.)", fontsize=14)
    ax.set_title(title_base + r",  $\theta=0$", fontsize=12)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=13)
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    plt.tight_layout()
    fig.savefig(plot_path + f"xi_vs_rho_{save_tag}.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: xi vs theta at a few rho values, zeta=0 ─────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, xi, label in zip(axes, [xi_rho, xi_theta, xi_zeta], labels):
        for i_rho, color in zip(rho_plot_indices, rho_colors):
            ax.plot(
                theta, xi[i_rho, :, 0],
                color=color, lw=2,
                label=rf"$\rho = {rho[i_rho]:.2f}$",
            )
        ax.set_xlabel(r"$\theta$", fontsize=14)
        ax.set_ylabel(label, fontsize=14)
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
        ax.tick_params(labelsize=11)
        ax.axhline(0, color="gray", lw=0.7, ls="--")
        ax.legend(fontsize=10)
    fig.suptitle(title_base, fontsize=13)
    plt.tight_layout()
    fig.savefig(plot_path + f"xi_vs_theta_{save_tag}.png", dpi=150)
    plt.close(fig)

    print(f"iota_0={iota_0:.4f}: plots saved")

print("Done. Plots saved to", plot_path)
