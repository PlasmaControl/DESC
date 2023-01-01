"""Compute functions for bootstrap current."""

import warnings
from scipy.special import roots_legendre
from scipy.constants import elementary_charge

from ..backend import jnp, fori_loop
from ..profiles import Profile, PowerSeriesProfile
from .data_index import register_compute_fun
from .utils import (
    compress,
    expand,
    surface_integrals,
    surface_averages,
    surface_min,
    surface_max,
)


def trapped_fraction(grid, modB, sqrt_g, n_gauss=20):
    r"""
    Evaluate the effective trapped particle fraction.

    Compute the effective fraction of trapped particles, which enters
    several formulae for neoclassical transport, as well as several
    quantities that go into its calculation.  The input data can be
    provided on a uniform grid of arbitrary toroidal and poloidal
    angles that need not be straight-field-line angles.

    The trapped fraction ``f_t`` has a standard definition in neoclassical theory:

    .. math::
        f_t = 1 - \frac{3}{4} \left< B^2 \right> \int_0^{1/Bmax}
            \frac{\lambda\; d\lambda}{\left< \sqrt{1 - \lambda B} \right>}

    where :math:`\left< \ldots \right>` is a flux surface average.

    The effective inverse aspect ratio epsilon is defined by

    .. math::
        \frac{Bmax}{Bmin} = \frac{1 + \epsilon}{1 - \epsilon}

    This definition is motivated by the fact that this formula would
    be true in the case of circular cross-section surfaces in
    axisymmetry with :math:`B \propto 1/R` and :math:`R = (1 +
    \epsilon \cos\theta) R_0`.

    This function operates on plain numpy/jax arrays, not the
    Equilibrium attributes used as arguments to other ``compute_*``
    functions. This is so this function can be applied to and tested
    on analytic magnetic fields.

    This function returns a Dictionary containing the following data,
    all 1D arrays of shape ``(grid.num_rho,)``:
    - ``"Bmin"``: The minimum of :math:`|B|` on each surface.
    - ``"Bmax"``: The maximum of :math:`|B|` on each surface.
    - ``"epsilon"``: The effective inverse aspect ratio on each surface.
    - ``"<1/B>"``: :math:`\left<B^2\right>` on each surface,
      where :math:`\left< \ldots \right>` denotes a flux surface average.
    - ``"<1/B>"``: :math:`\left<1/B\right>` on each surface,
      where :math:`\left< \ldots \right>` denotes a flux surface average.
    - ``"f_t"``: The effective trapped fraction on each surface.

    Parameters
    ----------
    grid : A ``Grid`` object.
    modB : Magnetic field strength :math:`|B|` on the grid points.
    sqrt_g : The Jacobian :math:`1/(\nabla\rho\times\nabla\theta\cdot\nabla\zeta)`
        on the grid points.
    n_gauss : int
        Number of Gauss-Legendre integration points for the :math:`\lambda` integral.

    Returns
    -------
    f_t_data : dict
        Dictionary containing the computed data listed above.
    """

    denominator = surface_integrals(grid, sqrt_g)
    fsa_B2 = compress(
        grid, surface_averages(grid, modB * modB, sqrt_g, denominator=denominator)
    )
    fsa_1overB = compress(
        grid, surface_averages(grid, 1 / modB, sqrt_g, denominator=denominator)
    )

    Bmax = surface_max(grid, modB)
    Bmin = surface_min(grid, modB)
    w = Bmax / Bmin
    epsilon = (w - 1) / (w + 1)

    # Get nodes and weights for Gauss-Legendre integration:
    base_nodes, base_weights = roots_legendre(n_gauss)
    # Rescale for integration on [0, 1], not [-1, 1]:
    lambd = (base_nodes + 1) * 0.5
    lambda_weights = base_weights * 0.5

    modB_over_Bmax = modB / expand(grid, Bmax)

    # Sum over the lambda grid points, using fori_loop for efficiency.
    lambd = jnp.asarray(lambd)
    lambda_weights = jnp.asarray(lambda_weights)

    def body_fun(jlambda, lambda_integral):
        flux_surf_avg_term = surface_averages(
            grid,
            jnp.sqrt(1 - lambd[jlambda] * modB_over_Bmax),
            sqrt_g,
            denominator=denominator,
        )
        return lambda_integral + lambda_weights[jlambda] * lambd[jlambda] / (
            Bmax * Bmax * compress(grid, flux_surf_avg_term)
        )

    lambda_integral = fori_loop(0, n_gauss, body_fun, jnp.zeros(grid.num_rho))

    f_t = 1 - 0.75 * fsa_B2 * lambda_integral
    f_t_data = {
        "<B**2>": fsa_B2,
        "<1/B>": fsa_1overB,
        "Bmin": Bmin,
        "Bmax": Bmax,
        "epsilon": epsilon,
        "f_t": f_t,
        "rho": grid.nodes[grid.unique_rho_idx, 0],
    }
    return f_t_data


def j_dot_B_Redl(
    geom_data,
    ne,
    Te,
    Ti,
    Zeff=None,
    helicity_N=None,
    plot=False,
):
    r"""
    Compute the bootstrap current (specifically
    :math:`\left<\vec{J}\cdot\vec{B}\right>`) using the formulae in
    Redl et al, Physics of Plasmas 28, 022502 (2021).

    The profiles of ne, Te, Ti, and Zeff should all be instances of
    subclasses of :obj:`desc.Profile`, i.e. they should
    have ``__call__()`` and ``dfds()`` functions. If ``Zeff == None``, a
    constant 1 is assumed. If ``Zeff`` is a float, a constant profile will
    be assumed.

    ``ne`` should have units of 1/m^3. ``Ti`` and ``Te`` should have
    units of eV.

    Geometric data can be specified in one of two ways. In the first
    approach, the arguments ``s``, ``G``, ``R``, ``iota``,
    ``epsilon``, ``f_t``, ``psi_edge``, and ``nfp`` are specified,
    while the argument ``geom`` is not. In the second approach, the
    argument ``geom`` is set to an instance of either
    :obj:`RedlGeomVmec` or :obj:`RedlGeomBoozer`, and this object will
    be used to set all the other geometric quantities. In this case,
    the arguments ``s``, ``G``, ``R``, ``iota``, ``epsilon``, ``f_t``,
    ``psi_edge``, and ``nfp`` should not be specified.

    The input variable ``s`` is a 1D array of values of normalized
    toroidal flux.  The input arrays ``G``, ``R``, ``iota``,
    ``epsilon``, and ``f_t``, should be 1d arrays evaluated on this
    same ``s`` grid. The bootstrap current
    :math:`\left<\vec{J}\cdot\vec{B}\right>` will be computed on this
    same set of flux surfaces.

    If you provide a :obj:`RedlGeomBoozer` object for ``geom``, then
    it is not necessary to specify the argument ``helicity_n`` here,
    in which case ``helicity_n`` will be taken from ``geom``.

    Parameters
    ----------
    ne: A :obj:`~Profile` object with the electron density profile.
    Te: A :obj:`~Profile` object with the electron temperature profile.
    Ti: A :obj:`~Profile` object with the ion temperature profile.
    Zeff: A :obj:`~Profile` object with the profile of the average
        impurity charge :math:`Z_{eff}`. Or, a single number can be provided if this profile is
        constant. Or, if ``None``, Zeff = 1 will be used.
    helicity_N: 0 for quasi-axisymmetry, or +/- NFP for quasi-helical symmetry.
        This quantity is used to apply the quasisymmetry isomorphism to map the collisionality
        and bootstrap current from the tokamak expressions to quasi-helical symmetry.
    plot: Whether to make a plot of many of the quantities computed.

    Returns
    -------
    jdotB_data : dict
        Dictionary containing the computed data listed above.
    """
    rho = geom_data["rho"]
    G = geom_data["G"]
    R = geom_data["R"]
    iota = geom_data["iota"]
    epsilon = geom_data["epsilon"]
    psi_edge = geom_data["psi_edge"]
    f_t = geom_data["f_t"]

    if Zeff is None:
        Zeff = PowerSeriesProfile(1.0, modes=[0])
    if not isinstance(Zeff, Profile):
        # Zeff is presumably a number. Convert it to a constant profile.
        Zeff = PowerSeriesProfile([Zeff], modes=[0])

    # Evaluate profiles on the grid:
    ne_rho = ne(rho)
    Te_rho = Te(rho)
    Ti_rho = Ti(rho)
    Zeff_rho = Zeff(rho)
    ni_rho = ne_rho / Zeff_rho
    pe_rho = ne_rho * Te_rho
    pi_rho = ni_rho * Ti_rho
    d_ne_d_s = ne(rho, dr=1) / (2 * rho)
    d_Te_d_s = Te(rho, dr=1) / (2 * rho)
    d_Ti_d_s = Ti(rho, dr=1) / (2 * rho)

    # Profiles may go to 0 at rho=1, so exclude the last few grid points:
    # These if statements are incompatible with jit:
    """
    if jnp.any(ne_rho[:-2] < 1e17):
        warnings.warn("ne is surprisingly low. It should have units 1/meters^3")
    if jnp.any(Te_rho[:-2] < 50):
        warnings.warn("Te is surprisingly low. It should have units of eV")
    if jnp.any(Ti_rho[:-2] < 50):
        warnings.warn("Ti is surprisingly low. It should have units of eV")
    """
    # Eq (18d)-(18e) in Sauter.
    # Check that we do not need to convert units of n or T!
    ln_Lambda_e = 31.3 - jnp.log(jnp.sqrt(ne_rho) / Te_rho)
    ln_Lambda_ii = 30 - jnp.log(Zeff_rho**3 * jnp.sqrt(ni_rho) / (Ti_rho**1.5))

    # Eq (18b)-(18c) in Sauter:
    geometry_factor = abs(R / (iota - helicity_N))
    nu_e = (
        geometry_factor
        * (6.921e-18)
        * ne_rho
        * Zeff_rho
        * ln_Lambda_e
        / (Te_rho * Te_rho * (epsilon**1.5))
    )
    nu_i = (
        geometry_factor
        * (4.90e-18)
        * ni_rho
        * (Zeff_rho**4)
        * ln_Lambda_ii
        / (Ti_rho * Ti_rho * (epsilon**1.5))
    )
    # These if statements are incompatible with jit:
    """
    if jnp.any(nu_e[:-2] < 1e-6):
        warnings.warn(
            "nu_*e is surprisingly low. Check that the density and temperature are correct."
        )
    if jnp.any(nu_i[:-2] < 1e-6):
        warnings.warn(
            "nu_*i is surprisingly low. Check that the density and temperature are correct."
        )
    if jnp.any(nu_e[:-2] > 1e5):
        warnings.warn(
            "nu_*e is surprisingly large. Check that the density and temperature are correct."
        )
    if jnp.any(nu_i[:-2] > 1e5):
        warnings.warn(
            "nu_*i is surprisingly large. Check that the density and temperature are correct."
        )
    """
    # Redl eq (11):
    X31 = f_t / (
        1
        + (0.67 * (1 - 0.7 * f_t) * jnp.sqrt(nu_e)) / (0.56 + 0.44 * Zeff_rho)
        + (0.52 + 0.086 * jnp.sqrt(nu_e))
        * (1 + 0.87 * f_t)
        * nu_e
        / (1 + 1.13 * jnp.sqrt(Zeff_rho - 1))
    )

    # Redl eq (10):
    Zfac = Zeff_rho**1.2 - 0.71
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
            + 0.23 * (1 - 0.96 * f_t) * jnp.sqrt(nu_e) / jnp.sqrt(Zeff_rho)
            + 0.13
            * (1 - 0.38 * f_t)
            * nu_e
            / (Zeff_rho * Zeff_rho)
            * (
                jnp.sqrt(1 + 2 * jnp.sqrt(Zeff_rho - 1))
                + f_t * f_t * jnp.sqrt((0.075 + 0.25 * (Zeff_rho - 1) ** 2) * nu_e)
            )
        )
    )

    # Redl eq (13):
    F32ee = (
        (0.1 + 0.6 * Zeff_rho)
        * (X32e - X32e**4)
        / (Zeff_rho * (0.77 + 0.63 * (1 + (Zeff_rho - 1) ** 1.1)))
        + 0.7
        / (1 + 0.2 * Zeff_rho)
        * (X32e**2 - X32e**4 - 1.2 * (X32e**3 - X32e**4))
        + 1.3 / (1 + 0.5 * Zeff_rho) * (X32e**4)
    )

    # Redl eq (16):
    X32ei = f_t / (
        1
        + 0.87 * (1 + 0.39 * f_t) * jnp.sqrt(nu_e) / (1 + 2.95 * (Zeff_rho - 1) ** 2)
        + 1.53 * (1 - 0.37 * f_t) * nu_e * (2 + 0.375 * (Zeff_rho - 1))
    )

    # Redl eq (15):
    F32ei = (
        -(0.4 + 1.93 * Zeff_rho)
        / (Zeff_rho * (0.8 + 0.6 * Zeff_rho))
        * (X32ei - X32ei**4)
        + 5.5
        / (1.5 + 2 * Zeff_rho)
        * (X32ei**2 - X32ei**4 - 0.8 * (X32ei**3 - X32ei**4))
        - 1.3 / (1 + 0.5 * Zeff_rho) * (X32ei**4)
    )

    # Redl eq (12):
    L32 = F32ei + F32ee

    # Redl eq (19):
    L34 = L31

    # Redl eq (20):
    alpha0 = (
        -(0.62 + 0.055 * (Zeff_rho - 1))
        * (1 - f_t)
        / (
            (0.53 + 0.17 * (Zeff_rho - 1))
            * (1 - (0.31 - 0.065 * (Zeff_rho - 1)) * f_t - 0.25 * f_t * f_t)
        )
    )
    # Redl eq (21):
    alpha = (
        (alpha0 + 0.7 * Zeff_rho * jnp.sqrt(f_t * nu_i)) / (1 + 0.18 * jnp.sqrt(nu_i))
        - 0.002 * nu_i * nu_i * (f_t**6)
    ) / (1 + 0.004 * nu_i * nu_i * (f_t**6))

    # Factor of elementary_charge is included below to convert temperatures from eV to J
    dnds_term = (
        -G
        * elementary_charge
        * (ne_rho * Te_rho + ni_rho * Ti_rho)
        * L31
        * (d_ne_d_s / ne_rho)
        / (psi_edge * (iota - helicity_N))
    )
    dTeds_term = (
        -G
        * elementary_charge
        * pe_rho
        * (L31 + L32)
        * (d_Te_d_s / Te_rho)
        / (psi_edge * (iota - helicity_N))
    )
    dTids_term = (
        -G
        * elementary_charge
        * pi_rho
        * (L31 + L34 * alpha)
        * (d_Ti_d_s / Ti_rho)
        / (psi_edge * (iota - helicity_N))
    )
    jdotB = dnds_term + dTeds_term + dTids_term

    # Store all results in the jdotB_data dictionary:
    nu_e_star = nu_e
    nu_i_star = nu_i
    variables = [
        "rho",
        "ne_rho",
        "ni_rho",
        "Zeff_rho",
        "Te_rho",
        "Ti_rho",
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
        "jdotB",
    ]
    jdotB_data = geom_data.copy()
    for v in variables:
        jdotB_data[v] = eval(v)

    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(14, 7))
        plt.rcParams.update({"font.size": 8})
        nrows = 5
        ncols = 5
        variables = [
            "Bmax",
            "Bmin",
            "epsilon",
            "<B**2>",
            "<1/B>",
            "f_t",
            "iota",
            "G",
            "R",
            "ne_rho",
            "ni_rho",
            "Zeff_rho",
            "Te_rho",
            "Ti_rho",
            "ln_Lambda_e",
            "ln_Lambda_ii",
            "nu_e_star",
            "nu_i_star",
            "dnds_term",
            "dTeds_term",
            "dTids_term",
            "L31",
            "L32",
            "alpha",
            "jdotB",
        ]
        for j, variable in enumerate(variables):
            plt.subplot(nrows, ncols, j + 1)
            plt.plot(rho, jdotB_data[variable])
            plt.title(variable)
            plt.xlabel(r"$\rho$")
        plt.tight_layout()
        plt.show()

    return jdotB_data


@register_compute_fun(
    name="<J*B> Redl",
    label="\\langle\\mathbf{J}\\cdot\\mathbf{B}\\rangle_{Redl}",
    units="T A m^{-2}",
    units_long="Tesla Ampere / meter^2",
    description="Bootstrap current profile, Redl model for quasisymmetry",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["|B|", "sqrt(g)", "G", "I", "iota"],
)
def _compute_J_dot_B_Redl(params, transforms, profiles, data, **kwargs):
    """
    Compute the geometric quantities needed for the Redl bootstrap
    formula using the global Bmax and Bmin on each surface.

    The effective trapped particle fraction :math:`f_t` will also be
    computed using the full nonaxisymmetric field strength on each
    flux surface.

    The advantage of this approach over :func:`Redl_geom_Boozer` is that no
    transformation to Boozer coordinates is involved in this
    method. However, the approach here may over-estimate ``epsilon``.
    """
    grid = transforms["grid"]

    # Note that geom_data contains info only as a function of rho, not
    # theta or zeta, i.e. on the compressed grid. In contrast, data
    # contains quantities on a 3D grid even for quantities that are
    # flux functions.
    geom_data = trapped_fraction(grid, data["|B|"], data["sqrt(g)"])
    geom_data["G"] = compress(grid, data["G"])
    geom_data["I"] = compress(grid, data["I"])
    geom_data["iota"] = compress(grid, data["iota"])
    geom_data["R"] = (geom_data["G"] + geom_data["iota"] * geom_data["I"]) * geom_data[
        "<1/B>"
    ]
    geom_data["psi_edge"] = params["Psi"] / (2 * jnp.pi)

    # The "backup" PowerSeriesProfiles here are necessary for test_compute_funs.py::test_compute_everything
    ne = kwargs.get("ne", PowerSeriesProfile([1e20]))
    Te = kwargs.get("Te", PowerSeriesProfile([1e3]))
    Ti = kwargs.get("Ti", PowerSeriesProfile([1e3]))
    Zeff = kwargs.get("Zeff", 1.0)
    helicity_N = kwargs.get("helicity_N", 0)

    j_dot_B_data = j_dot_B_Redl(
        geom_data,
        ne,
        Te,
        Ti,
        Zeff,
        helicity_N,
        plot=False,
    )
    data["<J*B> Redl"] = expand(grid, j_dot_B_data["jdotB"])
    return data
