from scipy.constants import mu_0
from desc.grid import LinearGrid, Grid
from desc.backend import jnp
from desc.utils import dot
from diffrax import (
    Event,
    ODETerm,
    PIDController,
    RecursiveCheckpointAdjoint,
    SaveAt,
    Tsit5,
    diffeqsolve,
)
from desc.compute.utils import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from scipy.optimize import brentq
import numpy as np

data_keys = ["B", "e^vartheta", "grad(phi)", "p_r"]


# --- STEP 1: Suydam's Criterion ---
# Formula: r * B_z^2 * (q'/q)^2 + 8 * mu_0 * p' > 0
def B_theta_pinch(data, r):
    """
    Compute B_theta in screw pinch approximation from
    stellarator variables.
    """
    return dot(data["B"], data["e^vartheta"]) * r


def B_z_pinch(data, R0):
    """
    Compute B_z in screw pinch approximation from
    stellarator variables.
    """
    return dot(data["B"], data["grad(phi)"]) * R0


def evaluate_suydam(eq, npoints=50):
    """
    Evaluate Suydam's criterion for MHD stability on a DESC equilibrium.
    Parameters
    ----------
    eq : desc.equilibrium.Equilibrium
        The DESC equilibrium to evaluate.
    npoints : int
        Number of radial points to evaluate Suydam's criterion.
    Returns
    -------
    is_suydam_stable : bool
        True if the equilibrium is Suydam stable, False otherwise.
    suydam : jnp.ndarray
        The values of Suydam's criterion at each radial point.
    """
    # 1. Setup Grid and Compute necessary quantities
    grid = LinearGrid(rho=jnp.linspace(1e-5, 1, npoints), theta=0, zeta=0)
    data = eq.compute(data_keys + ["iota", "iota_r", "a", "R0"], grid=grid)

    # 2. Extract major and minor radii
    a = data["a"]
    r = grid.nodes[:, 0] * a
    R0 = data["R0"]

    # Map DESC variables to Screw Pinch variables
    # B_z ~ B_zeta / R0, B_theta ~ B_theta / r
    B_z = B_z_pinch(data, R0)
    p_prime = data["p_r"] / a  # dp/dr
    q = 1.0 / data["iota"]  # Safety factor
    q_prime = -data["iota_r"] / (data["iota"] ** 2 * a)  # dq/dr

    suydam = r * (B_z**2) * (q_prime / q) ** 2 + 8 * mu_0 * p_prime
    is_suydam_stable = jnp.all(suydam > 0)

    return is_suydam_stable, suydam


# --- STEP 2: Integrate minimizing differential equation with no resonant surfaces ---


def _odefun(rho, u, args):
    """
    ODE function for the minimizing perturbation xi.
    d/dr [xi, f * xi_r] = [ xi_r, g * xi ]

    u = [xi, f * xi_r]
    """
    eq, m, k, a, R0 = args
    params = eq.params_dict
    grid = Grid(
        jnp.array([rho, 0.0, 0.0]).T,
        spacing=jnp.zeros((3,)).T,
        jitable=True,
    )
    transforms = get_transforms(data_keys, eq, grid, jitable=True)
    profiles = {
        # "current": eq.current,
        "iota": eq.iota,  # iota right now is totally wrong so anything related to B_theta is wrong
        "electron_density": eq.electron_density,
        "pressure": eq.pressure,
        "atomic_number": eq.atomic_number,
        "electron_temperature": eq.electron_temperature,
        "ion_temperature": eq.ion_temperature,
    }

    data = compute_fun(
        "desc.equilibrium.equilibrium.Equilibrium",
        data_keys,
        params,
        transforms,
        profiles,
    )
    f, g = fg(rho, data, m, k, a, R0)

    """
    # Compute df/dr
    dFdr = m / r * (B_theta_r - B_theta / r) + k * B_z_r
    dfdr = (Fs / k0_sq) * (F + 2 * r * dFdr + (2 * m**2 * F) / (r**2 * k0_sq))
    """
    #
    return jnp.stack([u[1] / f[0], g[0] * u[0]]) * (1 / a)


def F(rho, data, m, k, a, R0):
    """
    Compute F function from Friedberg pg. 463.
    Used in the ODE for the minimizing perturbation xi.
    Inputs:
    rho: normalized minor radius
    data: dictionary containing necessary
        (B^theta, B^zeta, p_r)
    Returns:
    F: function F evaluated at rho
    """
    # Map DESC variables to Screw Pinch variables
    # B_z ~ B^zeta * R0, B_theta ~ B^theta * r
    r = rho * a  # rhos * a
    B_z = B_z_pinch(data, R0)
    B_theta = B_theta_pinch(data, r)

    # Definitions from Friedberg pg. 463
    Fs = k * B_z + m * B_theta / r
    F_daggers = k * B_z - m * B_theta / r

    return Fs, F_daggers


def fg(rho, data, m, k, a, R0):
    """
    Compute f and g functions from Friedberg pg. 463.
    Used in the ODE for the minimizing perturbation xi.
    Inputs:
    rho: normalized minor radius
    data: dictionary containing necessary
        (B^theta, B^zeta, p_r)
    Returns:
    f, g: functions f and g evaluated at rho
    """
    r = rho * a  # minor radius
    k0_sq = k**2 + (m / r) ** 2

    # Compute F and F_daggers
    Fs, F_daggers = F(rho, data, m, k, a, R0)

    # Compute f
    f = r * Fs**2 / k0_sq

    # Compute dp/dr
    p_prime = data["p_r"] / a  # dp/dr

    # Compute g
    term1 = 2 * mu_0 * (k**2 / k0_sq) * p_prime
    term2 = ((k0_sq * r**2 - 1) / (k0_sq * r**2)) * r * (Fs**2)
    term3 = (2 * k**2 / (r * k0_sq**2)) * Fs * F_daggers
    g = term1 + term2 + term3

    return f, g


def compute_minimizing_perturbation(
    eq,
    m,
    n,
    from_axis=True,
    r_s=0,
    rhos=jnp.linspace(1e-2, 1.0, 1000),
    rtol=1e-13,
    atol=1e-16,
):
    """
    Compute the minimizing perturbation xi for mode (m, n)
    Inputs:
    eq: DESC Equilibrium object
    m: poloidal mode number
    n: toroidal mode number
    rhos: normalized minor radius grid points

    Returns:
    xi: minimizing perturbation evaluated at rhos
    f dxi/dr values at rhos
    """

    # Compute necessary parameters
    R0 = eq.compute("R0")["R0"]
    k = -n / R0
    a = eq.compute("a")["a"]

    # Initial conditions
    rho_0 = rhos.min()

    # Compute power law for xi near the axis or resonant surface
    if from_axis:
        # Starting from the magnetic axis
        p1 = (
            (jnp.abs(m) - 1) if m != 0 else 1
        )  # power of nonsingular solution at r -> 0
    else:
        # Starting from a resonant surface (rho = r_s)
        grid = LinearGrid(rho=jnp.array([r_s]), theta=0.0, zeta=0.0)

        data = eq.compute(
            data_keys + ["iota", "iota_r"],
            grid=grid,
        )
        q = 1.0 / data["iota"]
        q_prime = -data["iota_r"] / (data["iota"] ** 2 * a)  # dq/dr
        p_r = data["p_r"] / a  # dp/dr
        B_z = B_z_pinch(data, R0)
        D_s = -(2 * mu_0 * p_r * q**2) / (
            r_s * a * B_z**2 * q_prime**2
        )  # Suydam parameter at the edge
        p1 = (-0.5 + 0.5 * jnp.sqrt(1 - 4 * D_s))[0]

    grid = LinearGrid(rho=jnp.array([rho_0]), theta=0.0, zeta=0.0)
    data = eq.compute(
        data_keys,
        grid=grid,
    )
    xi0 = (rho_0 - r_s) ** p1
    xi_r0 = (p1 / a) * (rho_0 - r_s) ** (p1 - 1)
    f0, g0 = fg(rho_0, data, m, k, a, R0)
    u0 = jnp.array([xi0, f0[0] * xi_r0])

    # Event to stop integration if xi crosses zero
    def event_fn(t, y, args, **kwargs):
        return False  # y[0] < atol # xi starts positive, stop when it goes negative

    event = Event(event_fn)
    # Solver parameters
    saveat = SaveAt(ts=rhos)
    solution = diffeqsolve(
        terms=ODETerm(_odefun),
        solver=Tsit5(),
        y0=u0,
        t0=rho_0,
        t1=rhos.max(),
        saveat=saveat,
        dt0=1e-15,
        args=(eq, m, k, a, R0),
        stepsize_controller=PIDController(rtol=rtol, atol=atol),
        event=event,
        max_steps=int(1e10),
    )

    # Return xi and f * dxi/dr
    return solution.ys[:, 0], solution.ys[:, 1], solution.ts


# --- STEP 3: Handle resonant surfaces ---
def find_resonant_surfaces(eq, m, n, n_points=1000):
    """
    Find resonant surfaces using q profile directly.
    Inputs:
    eq: DESC Equilibrium object
    m: poloidal mode number
    n: toroidal mode number
    n_points: number of radial points to sample
    Returns:
    resonant_rhos: array of resonant surface locations (in rho)
    """
    rho_array = jnp.linspace(0, 1, n_points)

    # Compute q = 1/iota at all points (vectorized)
    grid = LinearGrid(
        rho=rho_array,
    )

    data = eq.compute(
        data_keys + ["a", "R0"],
        grid=grid,
    )
    R0 = data["R0"]
    a = data["a"]
    residual, _ = F(rho_array, data, m=m, k=-n / R0, a=a, R0=R0)

    # Vectorized sign change detection
    sign_changes = residual[:-1] * residual[1:] < 0

    # Get indices where sign changes occur
    change_indices = jnp.where(sign_changes)[0]

    # Check Fs for sign changes
    resonant_rhos = []
    for i in change_indices:
        # Use more precise root finding
        def Fs_func(rho):
            grid = Grid(
                jnp.array([[rho, 0.0, 0.0]]),
                spacing=jnp.zeros((3,)),
                jitable=True,
            )
            transforms = get_transforms(data_keys, eq, grid, jitable=True)
            profiles = {
                # "current": eq.current,
                "iota": eq.iota,  # iota right now is totally wrong so anything related to B_theta is wrong
                "electron_density": eq.electron_density,
                "pressure": eq.pressure,
                "atomic_number": eq.atomic_number,
                "electron_temperature": eq.electron_temperature,
                "ion_temperature": eq.ion_temperature,
            }
            data = compute_fun(
                "desc.equilibrium.equilibrium.Equilibrium",
                data_keys,
                eq.params_dict,
                transforms,
                profiles,
            )
            Fs, _ = F(rho, data, m, -n / R0, a, R0)
            return float(Fs[0])

        rho_root = brentq(
            Fs_func, float(rho_array[i]), float(rho_array[i + 1]), xtol=1e-13
        )
        resonant_rhos.append(rho_root)

    return jnp.array(resonant_rhos)


def find_xi(eq, m, n, nrho=5000, nlog=100):

    # Find segments to integrate over
    resonant_rhos = find_resonant_surfaces(eq, m=m, n=n, n_points=1000)
    resonant_rhos = jnp.hstack([0.0, resonant_rhos, 1.0])  # include axis and edge
    rho_segments = jnp.vstack([resonant_rhos[:-1], resonant_rhos[1:]]).T

    # Create empty arrays
    xi_segments = np.zeros((len(rho_segments), nrho + nlog))
    all_rhos = np.zeros((len(rho_segments), nrho + nlog))

    # Loop through each segment
    for i, segment in enumerate(rho_segments):
        rho_start, rho_end = segment
        from_axis = i == 0
        # Rhos to save at
        drho = np.minimum(
            (rho_end - rho_start) * 1e-1, 1e-2
        )  # distance from resonant surface to start/end logging
        rhos_segment = jnp.hstack([
            rho_start + jnp.logspace(jnp.log10(drho) - 2, jnp.log10(drho), nlog, endpoint=False),
            jnp.linspace(rho_start + drho, rho_end - drho, nrho)
        ])
        # Compute xi over this segment
        xi_segment, fxi_r_segment, ts = compute_minimizing_perturbation(
            eq, m=m, n=n, r_s=rho_start, from_axis=from_axis, rhos=rhos_segment
        )

        # Store results
        all_rhos[i, :] = ts
        xi_segments[i, :] = xi_segment

    return all_rhos, xi_segments, resonant_rhos

# --- STEP 4: Putting it all together ---
def evaluate_stability(eq):
    """
    Evaluate MHD stability of a DESC equilibrium using Suydam's criterion
    and the minimizing perturbation method.
    Inputs:
    eq: DESC Equilibrium object
    Returns:
    is_stable: bool
        True if the equilibrium is stable, False otherwise.
    """

    # Step 1: Check Suydam's criterion
    is_suydam_stable, suydam_values = evaluate_suydam(eq)
    if not is_suydam_stable:
        print("Equilibrium is Suydam unstable.")
        return False
    else:
        print("Equilibrium is Suydam stable.")

    # Step 2 & 3: Compute minimizing perturbation with resonant surfaces
    # Friedberg pg. 476: only m = 0, k^2 -> 0 & m=1, -infty<k<infty need to be checked
    m = 0  # poloidal mode number
    ns = [-1, 1]
    for n in ns:
        xi_rhos, xi_values, resonant_rhos = find_xi(eq, m=m, n=n)
        if (xi_values<=0).any():
            print(f"Equilibrium is unstable for mode (m={m}, n={n}).")
            return False
        else:
            print(f"Equilibrium is stable for mode (m={m}, n={n}).")
    m = 1
    ns = np.arange(-10,11)
    for n in ns:
        xi_rhos, xi_values, resonant_rhos = find_xi(eq, m=m, n=n)
        if (xi_values<=0).any():
            print(f"Equilibrium is unstable for mode (m={m}, n={n}).")
            return False
        else:
            print(f"Equilibrium is stable for mode (m={m}, n={n}).")
    return True 