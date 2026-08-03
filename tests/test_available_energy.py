"""Tests for available-energy analytic limits."""

import math

import numpy as np
import pytest
from qsc import Qsc
from quadax import quadgk
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ellipe, ellipk, erf, gamma, gammainc

from desc.backend import jnp
from desc.compute._turbulence import _ae_kernel, _ae_precompute, _ae_reduce
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid
from desc.integrals._ae_plot import _ae_well_data
from desc.integrals.bounce_integral import Bounce2D
from desc.objectives import ForceBalance, ObjectiveFunction, get_NAE_constraints
from desc.profiles import PowerSeriesProfile


def _paper_F(c1):
    """Weighting function F(c1) from Eq. (4.6) of Rodriguez & Mackenbach."""
    c1 = np.asarray(c1, dtype=float)
    return (
        2 * np.sqrt(c1) * (15 + 4 * c1) * np.exp(-c1)
        + 3 * np.sqrt(np.pi) * (2 * c1 - 5) * erf(np.sqrt(c1))
    ) / (8 * c1**2)


def _analytic_ae_kernel(energy, omega_alpha, omega_star):
    """Evaluate DESC's local AE kernel in the omnigenous density-gradient limit."""
    G = jnp.array([[[1.0]]])
    G_omega_alpha = jnp.array([[[omega_alpha]]])
    G_omega_psi = jnp.array([[[0.0]]])
    data = {
        "ae psi width": jnp.array([1.0]),
        "ae alpha width": jnp.array([1.0]),
        "ae grad(density)": jnp.array([omega_star]),
        "ae grad(temperature)": jnp.array([0.0]),
    }
    ae_data = _ae_precompute(G, G_omega_alpha, G_omega_psi, data)
    return np.asarray(_ae_kernel(*ae_data, energy)).ravel()


def _ae_inputs(c1, omega_alpha):
    """Return simple analytic-limit AE inputs."""
    G = jnp.array([[[1.0]]])
    G_omega_alpha = jnp.array([[[omega_alpha]]])
    G_omega_psi = jnp.array([[[0.0]]])
    data = {
        "ae psi width": jnp.array([1.0]),
        "ae alpha width": jnp.array([1.0]),
        "ae grad(density)": jnp.array([c1 * omega_alpha]),
        "ae grad(temperature)": jnp.array([0.0]),
    }
    return G, G_omega_alpha, G_omega_psi, data


def _adaptive_energy_integral(c1, omega_alpha, abs_err=1e-11, rel_err=1e-11):
    """Integrate the analytic-limit AE kernel with adaptive quadrature."""
    ae_data = _ae_precompute(*_ae_inputs(c1, omega_alpha))
    value = quadgk(
        lambda energy: (energy**1.5 * jnp.exp(-energy))
        * _ae_reduce(*ae_data, jnp.ones(1), energy).squeeze(-1),
        jnp.array([0.0, jnp.inf]),
        epsabs=abs_err,
        epsrel=rel_err,
    )[0]
    return np.asarray(value).squeeze()


def _strong_drive_precession_integral():
    """Evaluate the dimensionless integral in Eq. (F14)."""

    def precession_shape(k):
        return 2 * ellipe(k**2) / ellipk(k**2) - 1

    k0 = brentq(precession_shape, 0.0, 1 - 1e-12)
    return quad(
        lambda k: k * ellipk(k**2) * precession_shape(k),
        0.0,
        k0,
        epsabs=1e-13,
        epsrel=1e-13,
    )[0]


def _exp_quadratic_profile(kappa, order=8):
    """Return exp(−κρ²/2) as an even power-series profile."""
    modes = np.arange(0, 2 * order + 1, 2)
    params = np.asarray(
        [(-kappa / 2) ** j / math.factorial(j) for j in range(order + 1)]
    )
    return PowerSeriesProfile(params, modes=modes, sym=True)


def _near_axis_qs_ae(r, eta, omega_star, num_pitch):
    """Evaluate the leading-order QS model in Appendix F with DESC's AE kernel."""
    nodes, weights = np.polynomial.legendre.leggauss(num_pitch)
    k = (nodes + 1) / 2
    weights = weights / 2
    complete_elliptic_k = ellipk(k**2)
    precession_shape = 2 * ellipe(k**2) / complete_elliptic_k - 1

    # Equations (F2)-(F3), with the common dimensional factors set to one.
    normalized_bounce_time = 2 * np.sqrt(2) * complete_elliptic_k / np.sqrt(r * eta)
    normalized_omega_alpha = eta * precession_shape
    G = jnp.asarray(normalized_bounce_time[None, :, None])
    G_omega_alpha = G * jnp.asarray(normalized_omega_alpha[None, :, None])
    data = {
        "ae psi width": jnp.ones(1),
        "ae alpha width": jnp.ones(1),
        "ae grad(density)": jnp.asarray([omega_star]),
        "ae grad(temperature)": jnp.zeros(1),
    }
    ae_data = _ae_precompute(G, G_omega_alpha, jnp.zeros_like(G), data)
    pitch_weights = jnp.asarray(weights * 4 * r * eta * k)
    energy_integral = quadgk(
        lambda energy: (energy**1.5 * jnp.exp(-energy))
        * _ae_reduce(*ae_data, pitch_weights, energy).squeeze(-1),
        jnp.asarray([0.0, jnp.inf]),
        epsabs=1e-11,
        epsrel=1e-10,
    )[0]

    # This is DESC's final 3nT/2 normalization for this analytic model.
    desc_ae = float(energy_integral / (12 * np.pi**1.5))

    # Equation (F12), evaluated independently after the energy integration.
    co_precessing = normalized_omega_alpha > 0
    c1 = omega_star / normalized_omega_alpha[co_precessing]
    paper_F = (
        c1 * gammainc(2.5, c1) * gamma(2.5) - gammainc(3.5, c1) * gamma(3.5)
    ) / c1**2
    paper_exact = (
        2
        * np.sqrt(2)
        / np.pi**1.5
        * omega_star**2
        * np.sqrt(r * eta)
        * np.sum(
            weights[co_precessing]
            * k[co_precessing]
            * complete_elliptic_k[co_precessing]
            * paper_F
        )
    )
    return desc_ae, paper_exact


@pytest.mark.unit
def test_available_energy_kernel_matches_ramp_form():
    """In the omnigenous, density-gradient limit, _ae reduces to a ramp."""
    omega_alpha = 0.7
    c1 = 2.0
    omega_star = c1 * omega_alpha
    energy = jnp.asarray([0.1, 1.0, 1.9, 2.0, 2.1, 5.0])

    actual = _analytic_ae_kernel(energy, omega_alpha, omega_star)
    expected = 2 * omega_alpha**2 * np.maximum(c1 - np.asarray(energy), 0.0)

    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9)


@pytest.mark.unit
def test_counter_rotating_particles_do_not_contribute_to_ae():
    """The Heaviside factor in Eq. (4.5) is attained in the kernel."""
    energy = jnp.asarray([0.1, 1.0, 10.0])
    np.testing.assert_allclose(
        _analytic_ae_kernel(energy, omega_alpha=-0.7, omega_star=1.4),
        0.0,
        rtol=1e-7,
        atol=1e-9,
    )


@pytest.mark.unit
def test_available_energy_quadgk_integral_matches_paper_weight_function():
    """Adaptive energy quadrature matches Eq. (4.6)."""
    omega_alpha = 0.7
    c1 = np.asarray([0.5, 1.0, 2.0, 5.0, 10.0])
    np.testing.assert_allclose(
        [_adaptive_energy_integral(c, omega_alpha) for c in c1],
        [2 * (c * omega_alpha) ** 2 * _paper_F(c) for c in c1],
        rtol=1e-7,
        atol=1e-9,
    )


@pytest.mark.unit
def test_available_energy_sums_distinct_wells():
    """Every well in one retained field-line domain contributes to AE."""
    energy = jnp.asarray([0.4, 1.3])
    G = jnp.asarray([[[1.0, 1.7]]])
    G_omega_alpha = jnp.asarray([[[0.6, 1.1]]])
    G_omega_psi = jnp.asarray([[[0.2, 0.4]]])
    data = {
        "ae psi width": jnp.asarray([1.0]),
        "ae alpha width": jnp.asarray([1.0]),
        "ae grad(density)": jnp.asarray([2.0]),
        "ae grad(temperature)": jnp.asarray([0.3]),
    }
    combined = _ae_reduce(
        *_ae_precompute(G, G_omega_alpha, G_omega_psi, data),
        jnp.ones(1),
        energy,
    )
    separate = sum(
        _ae_reduce(
            *_ae_precompute(
                G[..., well : well + 1],
                G_omega_alpha[..., well : well + 1],
                G_omega_psi[..., well : well + 1],
                data,
            ),
            jnp.ones(1),
            energy,
        )
        for well in range(G.shape[-1])
    )
    np.testing.assert_allclose(combined, separate, rtol=1e-7, atol=1e-9)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("omega_star", "num_pitch", "limit", "kernel_rtol", "asymptote_rtol"),
    [
        pytest.param(1e-2, 2048, "weak", 2e-2, 6e-2, id="weakly-driven"),
        pytest.param(1e2, 512, "strong", 1e-8, 2e-2, id="strongly-driven"),
    ],
)
def test_available_energy_matches_near_axis_qs_asymptotes(
    omega_star, num_pitch, limit, kernel_rtol, asymptote_rtol
):
    """DESC attains the weakly and strongly driven limits in Appendix F."""
    r = 1e-2
    eta = 1.0
    actual, paper_exact = _near_axis_qs_ae(r, eta, omega_star, num_pitch)

    # The factor ⅔ converts the paper's nT normalization to DESC's 3nT/2.
    np.testing.assert_allclose(actual, (2 / 3) * paper_exact, rtol=kernel_rtol, atol=0)

    if limit == "weak":
        # Equation (F10), equivalent to Eq. (4.8) in this normalization.
        asymptote = 2 * np.sqrt(2) / (9 * np.pi) * omega_star**3 * np.sqrt(r / eta)
    else:
        # Equations (F13)-(F15), equivalent to Eq. (4.9).
        asymptote = (
            3
            / (np.pi * np.sqrt(2))
            * omega_star
            * eta
            * np.sqrt(r * eta)
            * _strong_drive_precession_integral()
        )
    np.testing.assert_allclose(paper_exact, asymptote, rtol=asymptote_rtol, atol=0)


@pytest.mark.skip
def test_available_energy_from_optimized_near_axis_tokamak():
    """Test the full equilibrium-to-AE pipeline against Eqs. (4.8) and (4.9)."""
    major_radius = 10.0
    boundary_radius = 0.1
    qsc = Qsc(
        rc=[major_radius],
        zs=[0.0],
        nfp=1,
        etabar=-1 / major_radius,
        B0=1.0,
        # Keep ι just above one so one complete poloidal transit lies strictly
        # inside one toroidal field period.
        I2=0.10001,
        nphi=61,
    )
    eq = Equilibrium.from_near_axis(qsc, r=boundary_radius, L=4, M=4, N=0, ntheta=41)
    eq.solve(
        objective=ObjectiveFunction(ForceBalance(eq=eq)),
        constraints=get_NAE_constraints(eq, qsc, order=1, fix_lambda=False, N=eq.N),
        ftol=1e-2,
        xtol=1e-7,
        maxiter=50,
        verbose=0,
    )

    rho = 0.05
    grid = LinearGrid(
        rho=np.asarray([rho]),
        M=eq.M_grid,
        N=0,
        NFP=eq.NFP,
        sym=False,
    )
    geometry = eq.compute(["iota", "psi_r"], grid=grid)
    iota = float(grid.compress(geometry["iota"])[0])
    np.testing.assert_allclose(
        grid.compress(geometry["psi_r"])[0],
        qsc.Bbar * boundary_radius**2 * rho,
        rtol=1e-12,
    )

    num_field_periods = math.ceil(eq.NFP / abs(iota))
    assert num_field_periods == 1
    num_complete_transits = math.floor(num_field_periods * abs(iota) / eq.NFP)
    assert num_complete_transits == 1
    domain_length = num_field_periods * 2 * np.pi / eq.NFP
    transit_length = 2 * np.pi / abs(iota)
    zeta_first_max = (domain_length - transit_length) / 2

    # At fixed ζ, locate the global |B| maximum and choose α so the field line
    # encounters it at ζ₁. The next global maximum occurs one poloidal transit later.
    maximum_grid = LinearGrid(
        rho=np.asarray([rho]),
        theta=np.linspace(0, 2 * np.pi, 2048, endpoint=False),
        zeta=np.asarray([zeta_first_max]),
        NFP=eq.NFP,
        sym=False,
    )
    maximum_data = eq.compute(["|B|", "theta_PEST"], grid=maximum_grid)
    maximum_index = int(np.argmax(maximum_data["|B|"]))
    alpha = float(
        np.mod(
            maximum_data["theta_PEST"][maximum_index] - iota * zeta_first_max,
            2 * np.pi,
        )
    )

    eq.pressure = None
    eq.electron_temperature = PowerSeriesProfile([1.0], modes=[0], sym=True)
    eq.ion_temperature = PowerSeriesProfile([1.0], modes=[0], sym=True)
    eq.atomic_number = 1.0
    angle = Bounce2D.angle(eq, X=32, Y=32, rho=np.asarray([rho]))
    common = {
        "grid": grid,
        "angle": angle,
        "alpha": np.asarray([alpha]),
        "radial_scale": 1.0,
        "binormal_scale": 1.0,
        # The caller selects and normalizes the retained complete poloidal domain.
        "fieldline_normalization": abs(iota) / num_complete_transits,
        "num_field_periods": num_field_periods,
        "num_well": 2,
        "num_quad": 32,
        "Y_B": 256,
        "nufft_eps": 0,
        "quad_atol": 1e-18,
        "quad_rtol": 1e-8,
    }

    actual_by_regime = {}
    for regime, kappa, num_pitch, rtol in [
        ("weak", 1e-3, 2049, 1e-1),
        ("strong", 1e2, 513, 2e-2),
    ]:
        eq.electron_density = _exp_quadratic_profile(kappa)
        data = eq.compute(
            ["available energy", "ne_r", "ne"],
            num_pitch=num_pitch,
            **common,
        )
        actual = float(grid.compress(data["available energy"])[0])
        gradient = float(-grid.compress(data["ne_r"] / data["ne"])[0])
        eta = abs(qsc.etabar)
        if regime == "weak":
            paper = (
                2
                * np.sqrt(2)
                / (9 * np.pi)
                * gradient**3
                * np.sqrt(rho / (boundary_radius * eta))
            )
        else:
            paper = (
                3
                * _strong_drive_precession_integral()
                / (np.pi * np.sqrt(2))
                * gradient
                * (boundary_radius * eta) ** 1.5
                * np.sqrt(rho)
            )
        # Equation (4.2) uses nT; DESC reports the result relative to 3nT/2.
        np.testing.assert_allclose(actual, (2 / 3) * paper, rtol=rtol, atol=0)
        actual_by_regime[regime] = actual

    well_data = _ae_well_data(
        eq,
        alpha=alpha,
        density_gradient=-5.0,
        temperature_gradient=0.0,
        num_pitch=129,
        **{key: value for key, value in common.items() if key != "alpha"},
    )
    np.testing.assert_array_equal(well_data.valid.sum(axis=1), 1)

    # A longer integer toroidal window retains two complete poloidal transits.
    # Normalize by the number of transits, while still summing every well in each.
    repeated_num_field_periods = 2
    repeated_num_transits = math.floor(repeated_num_field_periods * abs(iota) / eq.NFP)
    assert repeated_num_transits == 2
    repeated = eq.compute(
        "available energy",
        num_field_periods=repeated_num_field_periods,
        num_well=3,
        num_pitch=513,
        fieldline_normalization=abs(iota) / repeated_num_transits,
        **{
            key: value
            for key, value in common.items()
            if key not in ("fieldline_normalization", "num_field_periods", "num_well")
        },
    )
    np.testing.assert_allclose(
        grid.compress(repeated["available energy"])[0],
        actual_by_regime["strong"],
        rtol=2e-6,
    )
