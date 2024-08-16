"""Test interpolation to Clebsch coordinates and Fourier bounce integration."""

import numpy as np
import pytest
from matplotlib import pyplot as plt
from numpy.polynomial.legendre import leggauss
from tests.test_bounce_integral import _drift_analytic
from tests.test_plotting import tol_1d

from desc.backend import jnp
from desc.compute.bounce_integral import get_pitch
from desc.compute.fourier_bounce_integral import (
    FourierChebyshevBasis,
    alpha_sequence,
    bounce_integral,
    required_names,
)
from desc.equilibrium import Equilibrium
from desc.equilibrium.coords import get_rtz_grid, map_coordinates
from desc.examples import get
from desc.grid import LinearGrid


@pytest.mark.unit
@pytest.mark.parametrize(
    "alpha_0, iota, num_period, period",
    [(0, np.sqrt(2), 1, 2 * np.pi), (0, np.arange(1, 3) * np.sqrt(2), 5, 2 * np.pi)],
)
def test_alpha_sequence(alpha_0, iota, num_period, period):
    """Test field line poloidal label tracking utility."""
    iota = np.atleast_1d(iota)
    alphas = alpha_sequence(alpha_0, iota, num_period, period)
    assert alphas.shape == (iota.size, num_period)
    for i in range(iota.size):
        assert np.unique(alphas[i]).size == num_period, "Is iota irrational?"
    print(alphas)


@pytest.mark.unit
def test_fourier_chebyshev(rho=1, M=8, N=32, f=lambda B, pitch: B * pitch):
    """Test bounce points..."""
    eq = get("W7-X")
    clebsch = FourierChebyshevBasis.nodes(M, N, rho=rho)
    desc_from_clebsch = map_coordinates(
        eq,
        clebsch,
        inbasis=("rho", "alpha", "zeta"),
        period=(np.inf, 2 * np.pi, np.inf),
    )
    grid = LinearGrid(
        rho=rho, M=eq.M_grid, N=eq.N_grid, sym=False, NFP=eq.NFP
    )  # check if NFP!=1 works
    data = eq.compute(names=required_names() + ["min_tz |B|", "max_tz |B|"], grid=grid)
    bounce_integrate, _ = bounce_integral(
        grid, data, M, N, desc_from_clebsch, check=True, warn=False
    )  # TODO check true
    pitch = get_pitch(
        grid.compress(data["min_tz |B|"]), grid.compress(data["max_tz |B|"]), 10
    )
    result = bounce_integrate(f, [], pitch)  # noqa: F841


@pytest.mark.unit
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_drift():
    """Test bounce-averaged drift with analytical expressions."""
    eq = Equilibrium.load(".//tests//inputs//low-beta-shifted-circle.h5")
    psi_boundary = eq.Psi / (2 * np.pi)
    psi = 0.25 * psi_boundary
    rho = np.sqrt(psi / psi_boundary)
    np.testing.assert_allclose(rho, 0.5)

    # Make a set of nodes along a single fieldline.
    grid_fsa = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, sym=eq.sym, NFP=eq.NFP)
    data = eq.compute(["iota"], grid=grid_fsa)
    iota = grid_fsa.compress(data["iota"]).item()
    alpha = 0
    zeta = np.linspace(-np.pi / iota, np.pi / iota, (2 * eq.M_grid) * 4 + 1)
    grid = get_rtz_grid(
        eq,
        rho,
        alpha,
        zeta,
        coordinates="raz",
        period=(np.inf, 2 * np.pi, np.inf),
        iota=np.array([iota]),
    )
    data = eq.compute(
        required_names()
        + [
            "cvdrift",
            "gbdrift",
            "grad(psi)",
            "grad(alpha)",
            "shear",
            "iota",
            "psi",
            "a",
        ],
        grid=grid,
    )
    np.testing.assert_allclose(data["psi"], psi)
    np.testing.assert_allclose(data["iota"], iota)
    assert np.all(data["B^zeta"] > 0)
    B_ref = 2 * np.abs(psi_boundary) / data["a"] ** 2
    data["B ref"] = B_ref
    data["rho"] = rho
    data["alpha"] = alpha
    data["zeta"] = zeta
    data["psi"] = grid.compress(data["psi"])
    data["iota"] = grid.compress(data["iota"])
    data["shear"] = grid.compress(data["shear"])

    # Compute analytic approximation.
    drift_analytic, cvdrift, gbdrift, pitch = _drift_analytic(data)
    # Compute numerical result.
    bounce_integrate, _ = bounce_integral(
        data,
        knots=zeta,
        B_ref=B_ref,
        L_ref=data["a"],
        quad=leggauss(28),  # converges to absolute and relative tolerance of 1e-7
        check=True,
    )

    def integrand_num(cvdrift, gbdrift, B, pitch):
        g = jnp.sqrt(1 - pitch * B)
        return (cvdrift * g) - (0.5 * g * gbdrift) + (0.5 * gbdrift / g)

    def integrand_den(B, pitch):
        return 1 / jnp.sqrt(1 - pitch * B)

    drift_numerical_num = bounce_integrate(
        integrand=integrand_num,
        f=[cvdrift, gbdrift],
        pitch=pitch[:, np.newaxis],
        num_well=1,
    )
    drift_numerical_den = bounce_integrate(
        integrand=integrand_den,
        f=[],
        pitch=pitch[:, np.newaxis],
        num_well=1,
        weight=np.ones(zeta.size),
    )

    drift_numerical_num = np.squeeze(drift_numerical_num)
    drift_numerical_den = np.squeeze(drift_numerical_den)
    drift_numerical = drift_numerical_num / drift_numerical_den
    msg = "There should be one bounce integral per pitch in this example."
    assert drift_numerical.size == drift_analytic.size, msg
    np.testing.assert_allclose(drift_numerical, drift_analytic, atol=5e-3, rtol=5e-2)

    fig, ax = plt.subplots()
    ax.plot(1 / pitch, drift_analytic)
    ax.plot(1 / pitch, drift_numerical)
    return fig
