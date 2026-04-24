from desc import set_device
set_device("gpu")

import numpy as np
from desc.geometry import FourierRZToroidalSurface
from desc.utils import rpz2xyz, dot
from desc.grid import LinearGrid, Grid, QuadratureGrid
from desc.magnetic_fields import SourceFreeField
from desc.coils import FourierRZCoil, CoilSet
from desc.equilibrium import Equilibrium
from desc.continuation import solve_continuation_automatic
from desc.examples import get
from desc.backend import jnp
from matplotlib import pyplot as plt
from desc.io import load
from desc.equilibrium.coords import map_coordinates
import os
from scipy.constants import mu_0
from desc.profiles import PowerSeriesProfile

# ── Parameters ────────────────────────────────────────────────────────────────
M      = 28
N      = 28
n_theta = 2 * M
n_zeta  = 2 * N
n_surf  = n_theta * n_zeta
eq_tag  = "circular_tokamak"
eq_name = "Circular tokamak"
NFP     = 1

save_path = "results/phi_matrix/"
eq_save_name = f"{eq_tag}_M_{M}_N_{N}_equilibrium.h5"
phi_save_name = f"{eq_tag}_M_{M}_N_{N}_phi_matrix.npy"
os.makedirs(save_path, exist_ok=True)

I_coil = 1e6  # fixed coil current (A)

# vertical coil offset scan: Delta_h / a from 0.01 to 0.1
delta_h_fracs = np.linspace(0.01, 0.1, 20)

# ── Equilibrium ───────────────────────────────────────────────────────────────
R0 = 2
a = 1

if os.path.exists(save_path + eq_save_name):
    eq = load(save_path + eq_save_name)
    eq.change_resolution(NFP=1)
    eq.surface = SourceFreeField(eq.surface, M, N, NFP=1)
else:
    surface = FourierRZToroidalSurface.from_shape_parameters(
        major_radius=R0,
        aspect_ratio=int(R0/a),
        elongation=1,
        triangularity=0,
        squareness=0,
        eccentricity=0,
        torsion=0,
        NFP=1,
    )
    surface = SourceFreeField(surface, M, N, NFP=1)
    iota_coeffs = np.array([0.9, 0, 0.1, 0, 0.1])
    p_profile = np.array([0.125, 0, 0, 0, -0.125])
    eq = Equilibrium(
            L=12,
            M=12,
            N=0,
            surface=surface,
            NFP=NFP,
            iota=PowerSeriesProfile(iota_coeffs),
            pressure=PowerSeriesProfile(p_profile),
            Psi=1,
        )

    eq = solve_continuation_automatic(
        eq, ftol=1e-13, gtol=1e-13, xtol=1e-13, verbose=0
    )[-1]
    eq.save(save_path + eq_save_name)
a  = eq.compute("a")["a"]
R0 = eq.compute("R0")["R0"]

# ── Coil builder ─────────────────────────────────────────────────────────────
c = eq.axis
Z_modes = np.hstack([c.Z_basis.modes[:, 2], 0])

def make_coil(delta_h):
    Z_n_pos = np.hstack([c.Z_n,  delta_h * a])
    Z_n_neg = np.hstack([c.Z_n, -delta_h * a])
    return CoilSet([
        FourierRZCoil(current=I_coil,  R_n=c.R_n, Z_n=Z_n_pos,
                      NFP=c.NFP, modes_R=c.R_basis.modes[:, 2], modes_Z=Z_modes),
        FourierRZCoil(current=-I_coil, R_n=c.R_n, Z_n=Z_n_neg,
                      NFP=c.NFP, modes_R=c.R_basis.modes[:, 2], modes_Z=Z_modes),
    ])

# ── PEST grid & coordinate mapping ────────────────────────────────────────────
pest_grid = LinearGrid(rho=1.0, theta=n_theta, zeta=n_zeta, NFP=NFP, sym=False)

rho   = np.array([1.0])
theta = pest_grid.unique_theta
zeta  = pest_grid.unique_zeta
grid0 = LinearGrid(rho=rho, theta=theta, zeta=zeta, NFP=NFP, sym=False)
reshaped_nodes = jnp.reshape(
    grid0.meshgrid_reshape(grid0.nodes, order="rtz"), (n_surf, 3)
)
print("mapping coordinates …")
rtz_nodes = map_coordinates(
    eq, reshaped_nodes,
    inbasis=("rho", "theta_PEST", "zeta"),
    outbasis=("rho", "theta", "zeta"),
    period=(jnp.inf, 2 * jnp.pi, jnp.inf),
    tol=1e-12, maxiter=50,
)
grid = Grid(rtz_nodes, NFP=NFP)

# Surface grid in BIEST order (zeta outermost, theta fastest)
surf_nodes = (
    np.array(rtz_nodes)
    .reshape(n_theta, n_zeta, 3)
    .transpose(1, 0, 2)
    .reshape(n_surf, 3)
)
rtz_surface_grid = Grid(surf_nodes, NFP=NFP)

# ── phi_matrix ────────────────────────────────────────────────────────────────
if os.path.exists(save_path + phi_save_name):
    print(f"loading phi_matrix from {save_path + phi_save_name}")
    phi_matrix = np.load(save_path + phi_save_name)
else:
    print("computing phi_matrix …")
    data_phi = eq.compute(
        ["phi_matrix_pest"], rtz_surface_grid,
        pest_grid=pest_grid, problem="exterior Neumann", chunk_size=1,
    )
    phi_matrix = np.array(data_phi["phi_matrix_pest"])
    phi_matrix = (
        phi_matrix
        .reshape(n_zeta, n_theta, n_zeta, n_theta)
        .transpose(1, 0, 3, 2)
        .reshape(n_surf, n_surf)
    )
    np.save(save_path + phi_save_name, phi_matrix)
    print("phi_matrix saved.")

# ── Equilibrium quantities at the surface (fixed) ────────────────────────────
data_eq = eq.compute(["x", "n_rho", "g^rr", "sqrt(g)_PEST"], grid=grid, basis="xyz")
a_N = a
sqrtg = data_eq["sqrt(g)_PEST"]
g_sup_rr = data_eq["g^rr"]
sqrtg_grad_rho = sqrtg * np.sqrt(g_sup_rr)
#  pre-build the weighted phi_matrix (independent of coil)
integration_weights = sqrtg_grad_rho * grid0.meshgrid_reshape(grid0.weights, order="rtz").reshape(n_surf)
full_matrix = phi_matrix * integration_weights[:, None] / (2 * mu_0)

# ── W_V_true: direct volume integral ─────────────────────────────────────────
def compute_external_coil_energy(field, R0, a, L=50, M_quad=50, r_max_factor=20.0):
    """(2π/2μ₀) ∫_{r>a} B²(R,Z) r(R₀ + r cosθ) dr dθ  [one toroidal slice]."""
    quad_grid = QuadratureGrid(L=L, M=M_quad, N=0)
    r_flat = quad_grid.nodes[:, 0] * r_max_factor + a
    t_flat = quad_grid.nodes[:, 1]
    R_flat = R0 + r_flat * np.cos(t_flat)
    Z_flat = r_flat * np.sin(t_flat)
    valid  = R_flat > 0
    R_v, Z_v = R_flat[valid], Z_flat[valid]
    r_v, t_v = r_flat[valid], t_flat[valid]
    coords  = np.column_stack([R_v, np.zeros(valid.sum()), Z_v])
    B       = np.array(field.compute_magnetic_field(
        coords, basis="rpz", source_grid=LinearGrid(N=36)
    ))
    B_sq    = np.sum(B**2, axis=-1)
    jac     = r_v * (R0 + r_v * np.cos(t_v))
    weights = quad_grid.spacing[valid].prod(axis=-1) * r_max_factor
    return (2 * np.pi) / (2 * mu_0) * np.dot(B_sq, jac * weights)

# ── Delta_h scan ─────────────────────────────────────────────────────────────
W_V_vals      = np.zeros(len(delta_h_fracs))
W_V_true_vals = np.zeros(len(delta_h_fracs))

for k, frac in enumerate(delta_h_fracs):
    delta_h = frac * a
    coil_k  = make_coil(delta_h)

    # W_V from phi_matrix
    B_k  = coil_k.compute_magnetic_field(data_eq["x"],
                                          source_grid=LinearGrid(N=pest_grid.N))
    Bn_k = dot(B_k, data_eq["n_rho"])
    W_V_vals[k] = - float(Bn_k @ full_matrix @ Bn_k)

    # W_V_true from direct integration
    W_V_true_vals[k] = compute_external_coil_energy(coil_k, R0, a, r_max_factor=10000, L=2**12 , M_quad=2**12)
                                                    

    print(f"  Δh/a = {frac:.3f}  W_V = {W_V_vals[k]:.4e}  W_V_true = {W_V_true_vals[k]:.4e}")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(delta_h_fracs, W_V_vals,      marker="o", label=r"$W_V$ (BIE, $M{=}36,\,N{=}72$)")
ax.plot(delta_h_fracs, W_V_true_vals, marker="s", linestyle="--",
        label=r"$W_V^{\rm true}$ (direct $\int B^2\,dV$)")
ax.set_xlabel(r"$\Delta_h / a$", fontsize=13)
ax.set_ylabel("Vacuum energy (J)",  fontsize=13)
ax.set_title(f"Vacuum energy vs coil offset — {eq_name} ($I = {I_coil:.0e}$ A)", fontsize=13)
ax.legend(fontsize=12)
fig.tight_layout()
fig.savefig(save_path + "W_V_vs_delta_h.png", dpi=150)
plt.show()
print("done.")
