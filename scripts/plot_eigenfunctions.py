"""
Load saved eigenfunctions from the iota scan and plot:
  - xi vs rho at theta=0          (one plot per equilibrium)
  - xi vs rho at theta=pi/2       (one plot per equilibrium)
  - xi vs theta at a few rho      (one plot per equilibrium)
  - xi vs theta at rho=1 (BC check, one plot per equilibrium)
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from desc.integrals.quad_utils import leggauss_lob, automorphism_staircase1

# ── Settings ──────────────────────────────────────────────────────────────────
save_path   = "./high_aspect_ratio_tokamak/"
plot_path   = save_path + "eigenfunction_plots/"
os.makedirs(plot_path, exist_ok=True)


# Fixed parameters used when building the save_tag
aspect_ratio = 200
NFP          = 1
axisym       = False
p_coeffs     = np.array([0.125, 0, 0, 0, -0.125])

# These must match what mat-free.py used for the high-res solve
n_rho = 24
n_theta = 24
if axisym:
    n_zeta = 1
else:
    n_zeta = 14

# rho grid (reproduced from mat-free.py)
x, _ = leggauss_lob(n_rho)
rho   = automorphism_staircase1(x, eps=1e-2, x_0=0.5, m_1=2.0, m_2=2.0)
theta = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)


# rho indices to use for the "vs theta" plots
rho_plot_indices = [8, 20, 35, 50]
rho_colors       = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

labels     = [r"$\xi^\rho$", r"$\xi^\theta$", r"$\xi^\zeta$"]
comp_colors = ["steelblue", "darkorange", "seagreen"]

# ── Load scan metadata ────────────────────────────────────────────────────────
scan = np.load(save_path + "iota_scan_results.npz")
iota_on_axis_values = np.linspace(0.6, 1.5, 10)#scan["iota_on_axis"]

# ── Loop over equilibria ──────────────────────────────────────────────────────
for iota_0 in iota_on_axis_values:
    save_tag = (
        f"axisym_{axisym}_ar_{aspect_ratio}_NFP_{NFP}"
        f"_p_{'_'.join(p_coeffs.astype(str))}"
        f"_iota0_{iota_0:.4f}_d2iota_{-0.1:.4f}"
        f"n_rho_{n_rho}_n_theta_{n_theta}_n_zeta_{n_zeta}"
    )

    xi_rho_path   = save_path + f"xi_rho_low_{save_tag}.npy"
    xi_theta_path = save_path + f"xi_theta_low_{save_tag}.npy"
    xi_zeta_path  = save_path + f"xi_zeta_low_{save_tag}.npy"

    if not all(os.path.exists(p) for p in [xi_rho_path, xi_theta_path, xi_zeta_path]):
        print(f"Skipping iota_0={iota_0:.4f}: files not found")
        continue

    xi_rho   = np.load(xi_rho_path)    # (n_rho, n_theta, n_zeta)
    xi_theta = np.load(xi_theta_path)
    xi_zeta  = np.load(xi_zeta_path)

    title_base = rf"$\iota_0 = {iota_0:.3f}$,  $\iota(\rho) = \iota_0 - 0.05\,\rho^2$"

    # nearest theta index to pi/2
    i_theta_half_pi = int(np.argmin(np.abs(theta - np.pi / 2)))

    def _plot_vs_rho(i_theta, theta_label, fname_suffix):
        fig, ax = plt.subplots(figsize=(7, 5))
        for xi, label, color in zip([xi_rho, xi_theta, xi_zeta], labels, comp_colors):
            ax.plot(rho, xi[:, i_theta, 0], color=color, lw=2, label=label)
        ax.set_xlabel(r"$\rho$", fontsize=14)
        ax.set_ylabel(r"$\xi$ (arb.)", fontsize=14)
        ax.set_title(title_base + f",  {theta_label}", fontsize=12)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=13)
        ax.axhline(0, color="gray", lw=0.7, ls="--")
        plt.tight_layout()
        fig.savefig(plot_path + f"{fname_suffix}_{save_tag}.png", dpi=150)
        plt.close(fig)

    # ── Plot 1: xi vs rho at theta=0 ────────────────────────────────────────
    _plot_vs_rho(0, r"$\theta=0$", "xi_vs_rho_theta0")

    # ── Plot 2: xi vs rho at theta=pi/2 ─────────────────────────────────────
    _plot_vs_rho(
        i_theta_half_pi,
        rf"$\theta \approx \pi/2$  ($\theta={theta[i_theta_half_pi]:.3f}$)",
        "xi_vs_rho_theta_half_pi",
    )

    def _plot_vs_theta(rho_indices, suptitle_extra, fname_suffix):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
        for ax, xi, label in zip(axes, [xi_rho, xi_theta, xi_zeta], labels):
            for i_rho, color in zip(rho_indices, rho_colors):
                ax.plot(
                    theta, xi[i_rho, :, 0],
                    color=color, lw=2,
                    label=rf"$\rho = {rho[i_rho]:.2f}$",
                )
            ax.set_xlabel(r"$\theta$", fontsize=14)
            ax.set_ylabel(label, fontsize=14)
            ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
            ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
            ax.tick_params(labelsize=11)
            ax.axhline(0, color="gray", lw=0.7, ls="--")
            ax.legend(fontsize=10)
        fig.suptitle(title_base + suptitle_extra, fontsize=13)
        plt.tight_layout()
        fig.savefig(plot_path + f"{fname_suffix}_{save_tag}.png", dpi=150)
        plt.close(fig)

    # ── Plot 3: xi vs theta at a few interior rho values ────────────────────
    _plot_vs_theta(rho_plot_indices, "", "xi_vs_theta")

    # ── Plot 4: xi vs theta at rho=1 (Dirichlet BC check) ───────────────────
    # rho[-1] is the outermost grid point; xi^rho should be ~0 there
    _plot_vs_theta(
        [n_rho - 1],
        rf"  —  $\rho = {rho[-1]:.3f}$ (boundary BC check)",
        "xi_vs_theta_bc",
    )

    print(f"iota_0={iota_0:.4f}: plots saved")

print("Done. Plots saved to", plot_path)

# ── Lambda vs iota_0 summary plot ────────────────────────────────────────────
all_lambda_min = scan["lambda_min"]

fig, ax = plt.subplots(figsize=(7, 5))
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.axvline(1, color="gray", lw=0.8, ls="--", label=r"$\iota_0 = 1$")
ax.plot(iota_on_axis_values, all_lambda_min, linestyle="-",marker=".",
        color="steelblue", lw=2, ms=7)
ax.set_xlabel(r"$\iota_0$", fontsize=14)
ax.set_ylabel(r"$\lambda_{\min}$", fontsize=14)
ax.set_title(
    r"Stability eigenvalue vs $\iota_0$" + "\n"
    r"$\iota(\rho) = \iota_0 - 0.05\,\rho^2$",
    fontsize=12,
)
ax.tick_params(labelsize=12)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(plot_path + "lambda_vs_iota0.png", dpi=150)
plt.show()
print(f"Lambda plot saved to {plot_path}lambda_vs_iota0.png")
