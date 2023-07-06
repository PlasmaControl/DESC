"""Compute functions for bootstrap current."""

from scipy.constants import elementary_charge
from scipy.special import roots_legendre

from ..backend import fori_loop, jnp
from .data_index import register_compute_fun
from .utils import compress, expand, surface_averages_map


@register_compute_fun(
    name="trapped fraction",
    label="1 - \\frac{3}{4} \\langle B^2 \\rangle \\int_0^{1/Bmax} "
    "\\frac{\\lambda\\; d\\lambda}{\\langle \\sqrt{1 - \\lambda B} \\rangle}",
    units="~",
    units_long="None",
    description="Neoclassical effective trapped particle fraction",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["sqrt(g)", "V_r(r)", "|B|", "<B^2>", "max_tz |B|"],
    n_gauss="n_gauss",
)
def _trapped_fraction(params, transforms, profiles, data, **kwargs):
    r"""
    Evaluate the effective trapped particle fraction.

    Compute the effective fraction of trapped particles, which enters
    several formulae for neoclassical transport. The trapped fraction
    ``f_t`` has a standard definition in neoclassical theory:

    .. math::
        f_t = 1 - \frac{3}{4} \langle B^2 \rangle \int_0^{1/Bmax}
            \frac{\lambda\; d\lambda}{\langle \sqrt{1 - \lambda B} \rangle}

    where :math:`\langle \ldots \rangle` is a flux surface average.
    """
    # Get nodes and weights for Gauss-Legendre integration:
    n_gauss = kwargs.get("n_gauss", 20)
    base_nodes, base_weights = roots_legendre(n_gauss)
    # Rescale for integration on [0, 1], not [-1, 1]:
    lambd = jnp.asarray((base_nodes + 1) * 0.5)
    lambda_weights = jnp.asarray(base_weights * 0.5)

    grid = transforms["grid"]
    Bmax = data["max_tz |B|"]
    modB_over_Bmax = data["|B|"] / Bmax
    sqrt_g = data["sqrt(g)"]
    Bmax_squared = compress(grid, Bmax * Bmax)
    V_r = compress(grid, data["V_r(r)"])
    compute_surface_averages = surface_averages_map(grid, expand_out=False)

    # Sum over the lambda grid points, using fori_loop for efficiency.
    def body_fun(jlambda, lambda_integral):
        flux_surf_avg_term = compute_surface_averages(
            jnp.sqrt(1 - lambd[jlambda] * modB_over_Bmax),
            sqrt_g,
            denominator=V_r,
        )
        return lambda_integral + lambda_weights[jlambda] * lambd[jlambda] / (
            Bmax_squared * flux_surf_avg_term
        )

    lambda_integral = fori_loop(0, n_gauss, body_fun, jnp.zeros(grid.num_rho))

    trapped_fraction = 1 - 0.75 * compress(grid, data["<B^2>"]) * lambda_integral
    data["trapped fraction"] = expand(grid, trapped_fraction)
    return data


def j_dot_B_Redl(
    geom_data,
    profile_data,
    helicity_N=None,
):
    r"""Compute the bootstrap current.

    (specifically :math:`\langle\vec{J}\cdot\vec{B}\rangle`) using the formulae in
    Redl et al, Physics of Plasmas 28, 022502 (2021). This formula for
    the bootstrap current is valid in axisymmetry, quasi-axisymmetry,
    and quasi-helical symmetry, but not in other stellarators.

    The argument ``geom_data`` is a Dictionary that should contain the
    following items:

    - G: 1D array with the Boozer ``G`` coefficient.
    - R: 1D array with the effective value of ``R`` to use in the Redl formula,
      not necessarily the major radius.
    - iota: 1D array with the rotational transform.
    - epsilon: 1D array with the effective inverse aspect ratio to use in
      the Redl formula.
    - psi_edge: float, the boundary toroidal flux, divided by (2 pi).
    - f_t: 1D array with the effective trapped particle fraction

    The argument ``profile_data`` is a Dictionary that should contain the
    following items, all 1D arrays with quantities on the same radial grid:

    - rho: effective minor radius.
    - ne: electron density, in meters^{-3}.
    - Te: electron temperature, in eV.
    - Ti: ion temperature, in eV.
    - Zeff: effective atomic charge.
    - ne_r: derivative of electron density with respect to rho.
    - Te_r: derivative of electron temperature with respect to rho.
    - Ti_r: derivative of ion temperature with respect to rho.

    Parameters
    ----------
    geom_data : dict
        Dictionary containing the data described above.
    profile_data : dict
        Dictionary containing the data described above.
    helicity_N : int
        Set to 0 for quasi-axisymmetry, or +/- NFP for quasi-helical symmetry.
        This quantity is used to apply the quasisymmetry isomorphism to map the
        collisionality and bootstrap current from the tokamak expressions to
        quasi-helical symmetry.

    Returns
    -------
    J_dot_B_data : dict
        Dictionary containing the computed data listed above.
    """
    G = geom_data["G"]
    R = geom_data["R"]
    iota = geom_data["iota"]
    epsilon = geom_data["epsilon"]
    psi_edge = geom_data["psi_edge"]
    f_t = geom_data["f_t"]

    # Set profiles:
    rho = profile_data["rho"]
    ne = profile_data["ne"]
    Te = profile_data["Te"]
    Ti = profile_data["Ti"]
    # Since Zeff appears in the Redl formula via sqrt(Zeff - 1), when
    # Zeff = 1 the gradient can sometimes evaluate to NaN. This
    # problem is avoided by adding a tiny number here:
    Zeff = jnp.maximum(1 + 1.0e-14, profile_data["Zeff"])
    ni = ne / Zeff
    pe = ne * Te
    d_ne_d_s = profile_data["ne_r"] / (2 * rho)
    d_Te_d_s = profile_data["Te_r"] / (2 * rho)
    d_Ti_d_s = profile_data["Ti_r"] / (2 * rho)

    # Eq (18d)-(18e) in Sauter, Angioni, and Lin-Liu, Physics of Plasmas 6, 2834 (1999).
    ln_Lambda_e = 31.3 - jnp.log(jnp.sqrt(ne) / Te)
    ln_Lambda_ii = 30 - jnp.log(Zeff**3 * jnp.sqrt(ni) / (Ti**1.5))

    # Eq (18b)-(18c) in Sauter:
    geometry_factor = abs(R / (iota - helicity_N))
    nu_e = (
        geometry_factor
        * (6.921e-18)
        * ne
        * Zeff
        * ln_Lambda_e
        / (Te * Te * (epsilon**1.5))
    )
    nu_i = (
        geometry_factor
        * (4.90e-18)
        * ni
        * (Zeff**4)
        * ln_Lambda_ii
        / (Ti * Ti * (epsilon**1.5))
    )

    # Redl eq (11):
    X31 = f_t / (
        1
        + (0.67 * (1 - 0.7 * f_t) * jnp.sqrt(nu_e)) / (0.56 + 0.44 * Zeff)
        + (0.52 + 0.086 * jnp.sqrt(nu_e))
        * (1 + 0.87 * f_t)
        * nu_e
        / (1 + 1.13 * jnp.sqrt(Zeff - 1))
    )

    # Redl eq (10):
    Zfac = Zeff**1.2 - 0.71
    L31 = (
        (1 + 0.15 / Zfac) * X31
        - 0.22 / Zfac * (X31**2)
        + 0.01 / Zfac * (X31**3)
        + 0.06 / Zfac * (X31**4)
    )

    # Redl eq (14):
    X32e = f_t / (
        (
            1
            + 0.23 * (1 - 0.96 * f_t) * jnp.sqrt(nu_e) / jnp.sqrt(Zeff)
            + 0.13
            * (1 - 0.38 * f_t)
            * nu_e
            / (Zeff * Zeff)
            * (
                jnp.sqrt(1 + 2 * jnp.sqrt(Zeff - 1))
                + f_t * f_t * jnp.sqrt((0.075 + 0.25 * (Zeff - 1) ** 2) * nu_e)
            )
        )
    )

    # Redl eq (13):
    F32ee = (
        (0.1 + 0.6 * Zeff)
        * (X32e - X32e**4)
        / (Zeff * (0.77 + 0.63 * (1 + (Zeff - 1) ** 1.1)))
        + 0.7
        / (1 + 0.2 * Zeff)
        * (X32e**2 - X32e**4 - 1.2 * (X32e**3 - X32e**4))
        + 1.3 / (1 + 0.5 * Zeff) * (X32e**4)
    )

    # Redl eq (16):
    X32ei = f_t / (
        1
        + 0.87 * (1 + 0.39 * f_t) * jnp.sqrt(nu_e) / (1 + 2.95 * (Zeff - 1) ** 2)
        + 1.53 * (1 - 0.37 * f_t) * nu_e * (2 + 0.375 * (Zeff - 1))
    )

    # Redl eq (15):
    F32ei = (
        -(0.4 + 1.93 * Zeff) / (Zeff * (0.8 + 0.6 * Zeff)) * (X32ei - X32ei**4)
        + 5.5
        / (1.5 + 2 * Zeff)
        * (X32ei**2 - X32ei**4 - 0.8 * (X32ei**3 - X32ei**4))
        - 1.3 / (1 + 0.5 * Zeff) * (X32ei**4)
    )

    # Redl eq (12):
    L32 = F32ei + F32ee

    # Redl eq (19):
    L34 = L31

    # Redl eq (20):
    alpha0 = (
        -(0.62 + 0.055 * (Zeff - 1))
        * (1 - f_t)
        / (
            (0.53 + 0.17 * (Zeff - 1))
            * (1 - (0.31 - 0.065 * (Zeff - 1)) * f_t - 0.25 * f_t * f_t)
        )
    )
    # Redl eq (21):
    alpha = (
        (alpha0 + 0.7 * Zeff * jnp.sqrt(f_t * nu_i)) / (1 + 0.18 * jnp.sqrt(nu_i))
        - 0.002 * nu_i * nu_i * (f_t**6)
    ) / (1 + 0.004 * nu_i * nu_i * (f_t**6))

    # Factor of elementary_charge is included below to convert temperatures from eV to J
    dnds_term = (
        -G
        * elementary_charge
        * (ne * Te + ni * Ti)
        * L31
        * (d_ne_d_s / ne)
        / (psi_edge * (iota - helicity_N))
    )
    dTeds_term = (
        -G
        * elementary_charge
        * pe
        * (L31 + L32)
        * (d_Te_d_s / Te)
        / (psi_edge * (iota - helicity_N))
    )
    dTids_term = (
        -G
        * elementary_charge
        * ni
        * Ti
        * (L31 + L34 * alpha)
        * (d_Ti_d_s / Ti)
        / (psi_edge * (iota - helicity_N))
    )
    J_dot_B = dnds_term + dTeds_term + dTids_term

    # Store all results in the J_dot_B_data dictionary:
    nu_e_star = nu_e
    nu_i_star = nu_i
    variables = [
        "rho",
        "ne",
        "ni",
        "Zeff",
        "Te",
        "Ti",
        "d_ne_d_s",
        "d_Te_d_s",
        "d_Ti_d_s",
        "ln_Lambda_e",
        "ln_Lambda_ii",
        "nu_e_star",
        "nu_i_star",
        "X31",
        "X32e",
        "X32ei",
        "F32ee",
        "F32ei",
        "L31",
        "L32",
        "L34",
        "alpha0",
        "alpha",
        "dnds_term",
        "dTeds_term",
        "dTids_term",
    ]
    J_dot_B_data = geom_data.copy()
    for v in variables:
        J_dot_B_data[v] = eval(v)
    J_dot_B_data["<J*B>"] = J_dot_B
    return J_dot_B_data


@register_compute_fun(
    name="<J*B> Redl",
    label="\\langle\\mathbf{J}\\cdot\\mathbf{B}\\rangle_{Redl}",
    units="T A m^{-2}",
    units_long="Tesla Ampere / meter^2",
    description="Bootstrap current profile, Redl model for quasisymmetry",
    dim=1,
    params=["Psi"],
    transforms={"grid": []},
    profiles=["atomic_number"],
    coordinates="r",
    data=[
        "trapped fraction",
        "G",
        "I",
        "iota",
        "<1/|B|>",
        "effective r/R0",
        "ne",
        "ne_r",
        "Te",
        "Te_r",
        "Ti",
        "Ti_r",
        "Zeff",
    ],
    helicity="helicity",
)
def _compute_J_dot_B_Redl(params, transforms, profiles, data, **kwargs):
    r"""Compute the bootstrap current.

    (specifically :math:`\langle\vec{J}\cdot\vec{B}\rangle`) using the formulae in
    Redl et al, Physics of Plasmas 28, 022502 (2021). This formula for
    the bootstrap current is valid in axisymmetry, quasi-axisymmetry,
    and quasi-helical symmetry, but not in other stellarators.
    """
    grid = transforms["grid"]

    # Note that the geom_data dictionary provided to j_dot_B_Redl()
    # contains info only as a function of rho, not theta or zeta,
    # i.e. on the compressed grid. In contrast, "data" contains
    # quantities on a 3D grid even for quantities that are flux
    # functions.
    geom_data = {}
    geom_data["f_t"] = compress(grid, data["trapped fraction"])
    geom_data["epsilon"] = compress(grid, data["effective r/R0"])
    geom_data["G"] = compress(grid, data["G"])
    geom_data["I"] = compress(grid, data["I"])
    geom_data["iota"] = compress(grid, data["iota"])
    geom_data["<1/|B|>"] = compress(grid, data["<1/|B|>"])
    geom_data["R"] = (geom_data["G"] + geom_data["iota"] * geom_data["I"]) * geom_data[
        "<1/|B|>"
    ]
    geom_data["psi_edge"] = params["Psi"] / (2 * jnp.pi)

    profile_data = {}
    profile_data["rho"] = compress(grid, data["rho"])
    profile_data["ne"] = compress(grid, data["ne"])
    profile_data["ne_r"] = compress(grid, data["ne_r"])
    profile_data["Te"] = compress(grid, data["Te"])
    profile_data["Te_r"] = compress(grid, data["Te_r"])
    profile_data["Ti"] = compress(grid, data["Ti"])
    profile_data["Ti_r"] = compress(grid, data["Ti_r"])
    if profiles["atomic_number"] is None:
        Zeff = jnp.ones(grid.num_rho)
    else:
        Zeff = compress(grid, data["Zeff"])
    profile_data["Zeff"] = Zeff

    helicity = kwargs.get("helicity", (1, 0))
    helicity_N = helicity[1]

    j_dot_B_data = j_dot_B_Redl(
        geom_data,
        profile_data,
        helicity_N,
    )
    data["<J*B> Redl"] = expand(grid, j_dot_B_data["<J*B>"])
    return data
