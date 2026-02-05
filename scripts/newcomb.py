from scipy.constants import mu_0
# 1. Setup Grid and Compute necessary quantities
grid = LinearGrid(rho=jnp.linspace(1E-5,1,10), theta=0, zeta=0)
data = eq.compute(["p_r", "iota", "iota_r", "B^theta", "B^zeta", "a"], grid=grid)

# Define wavenumbers
m = 1  # Poloidal mode number
n = 1  # Toroidal mode number


a = data["a"]

r = grid.nodes[:, 0] * a
R0 = eq.compute("R0")["R0"]
k = n / R0

# Map DESC variables to Screw Pinch variables
# B_z ~ B^zeta * R0, B_theta ~ B^theta * r
B_z = data["B^zeta"] * R0
B_theta = data["B^theta"] * r
p_prime = data["p_r"] / a # dp/dr
q = 1.0 / data["iota"]
q_prime = -data["iota_r"] / (data["iota"]**2 * a) # dq/dr

# Definitions from Friedberg pg. 463
F = k * B_z + m * B_theta / r
F_dagger = k * B_z - m * B_theta / r
G = m * B_z / r - k * B_theta
k0_sq = k**2 + (m/r)**2

# --- STEP 1: Suydam's Criterion ---
# Formula: r * B_z^2 * (q'/q)^2 + 8 * mu_0 * p' > 0
suydam = r * (B_z**2) * (q_prime / q)**2 + 8 * mu_0 * p_prime
is_suydam_stable = jnp.all(suydam > 0)

print(f"--- Stability Analysis for mode (m={m}, n={n}) ---")
print(f"Suydam's Criterion: {'PASSED' if is_suydam_stable else 'VIOLATED'}")


