"""Test interpolation to Clebsch coordinates and Fourier bounce integration."""

import numpy as np
import pytest
from matplotlib import pyplot as plt
from numpy.polynomial.chebyshev import chebinterpolate, chebroots
from numpy.polynomial.legendre import leggauss
from tests.test_bounce_integral import _drift_analytic
from tests.test_plotting import tol_1d

from desc.backend import jnp
from desc.equilibrium import Equilibrium
from desc.equilibrium.coords import get_rtz_grid, map_coordinates
from desc.examples import get
from desc.grid import Grid, LinearGrid
from desc.integrals._interp_utils import fourier_pts
from desc.integrals.bounce_integral import filter_bounce_points, get_pitch
from desc.integrals.fourier_bounce_integral import (
    FourierChebyshevBasis,
    alpha_sequence,
    bounce_integral,
    required_names,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "alpha_0, iota, num_period, period",
    [(0, np.sqrt(2), 1, 2 * np.pi), (0, np.arange(1, 3) * np.sqrt(2), 5, 2 * np.pi)],
)
def test_alpha_sequence(alpha_0, iota, num_period, period):
    """Test field line poloidal label tracking."""
    iota = np.atleast_1d(iota)
    alphas = alpha_sequence(alpha_0, iota, num_period, period)
    assert alphas.shape == (iota.size, num_period)
    for i in range(iota.size):
        assert np.unique(alphas[i]).size == num_period, f"{iota} is irrational"
    print(alphas)


class TestBouncePoints:
    """Test that bounce points are computed correctly."""

    @staticmethod
    def _cheb_intersect(cheb, k):
        cheb = cheb.copy()
        cheb[0] = cheb[0] - k
        roots = chebroots(cheb)
        intersect = roots[
            np.logical_and(np.isreal(roots), np.abs(roots.real) <= 1)
        ].real
        return intersect

    @staticmethod
    def _periodic_fun(nodes, M, N):
        alpha, zeta = nodes.T
        f = -2 * np.cos(1 / (0.1 + zeta**2)) + 2
        return f.reshape(M, N)

    @pytest.mark.unit
    def test_bp1_first(self):
        """Test that bounce points are computed correctly."""
        pitch = 1 / np.linspace(1, 4, 20).reshape(20, 1)
        M, N = 1, 10
        domain = (-1, 1)
        nodes = FourierChebyshevBasis.nodes(M, N, domain=domain)
        f = self._periodic_fun(nodes, M, N)
        fcb = FourierChebyshevBasis(f, domain=domain)
        pcb = fcb.compute_cheb(fourier_pts(M))
        bp1, bp2 = pcb.bounce_points(*pcb.intersect(1 / pitch))
        pcb.check_bounce_points(bp1, bp2, pitch.ravel())
        bp1, bp2 = filter_bounce_points(bp1, bp2)

        def f(z):
            return -2 * np.cos(1 / (0.1 + z**2)) + 2

        r = self._cheb_intersect(chebinterpolate(f, N), 1 / pitch)
        np.testing.assert_allclose(bp1, r[::2], rtol=1e-3)
        np.testing.assert_allclose(bp2, r[1::2], rtol=1e-3)


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
    grid_rtz = Grid.create_meshgrid(
        [
            rho,
            np.linspace(0, 2 * np.pi, eq.M_grid),
            np.linspace(0, 2 * np.pi, eq.N_grid + 1),
        ],
    )
    data = eq.compute(["iota"], grid=grid_rtz)
    iota = grid_rtz.compress(data["iota"]).item()
    alpha = 0
    zeta = np.linspace(-np.pi / iota, np.pi / iota, (2 * eq.M_grid) * 4 + 1)
    grid_raz = get_rtz_grid(
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
        grid=grid_raz,
    )
    np.testing.assert_allclose(data["psi"], psi)
    np.testing.assert_allclose(data["iota"], iota)
    assert np.all(data["B^zeta"] > 0)
    B_ref = 2 * np.abs(psi_boundary) / data["a"] ** 2
    data["B ref"] = B_ref
    data["rho"] = rho
    data["alpha"] = alpha
    data["zeta"] = zeta
    data["psi"] = grid_raz.compress(data["psi"])
    data["iota"] = grid_raz.compress(data["iota"])
    data["shear"] = grid_raz.compress(data["shear"])
    # Compute analytic approximation.
    drift_analytic, cvdrift, gbdrift, pitch = _drift_analytic(data)

    # Compute numerical result.
    M, N = eq.M_grid, 100
    clebsch = FourierChebyshevBasis.nodes(M=eq.M_grid, N=N, rho=rho)
    data_2 = eq.compute(names=required_names() + ["cvdrift", "gbdrift"], grid=grid_rtz)
    normalization = -np.sign(data["psi"]) * data["B ref"] * data["a"] ** 2
    cvdrift = data_2["cvdrift"] * normalization
    gbdrift = data_2["gbdrift"] * normalization
    bounce_integrate, _ = bounce_integral(
        grid_rtz,
        data_2,
        M,
        N,
        desc_from_clebsch=map_coordinates(
            eq,
            clebsch,
            inbasis=("rho", "alpha", "zeta"),
            period=(np.inf, 2 * np.pi, np.inf),
            iota=np.broadcast_to(data["iota"], (M * N)),
        ),
        alpha_0=data["alpha"],
        num_transit=5,
        B_ref=data["B ref"],
        L_ref=data["a"],
        quad=leggauss(28),  # converges to absolute and relative tolerance of 1e-7
        check=True,
        plot=True,
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
    )
    drift_numerical = np.squeeze(drift_numerical_num / drift_numerical_den)
    msg = "There should be one bounce integral per pitch in this example."
    assert drift_numerical.size == drift_analytic.size, msg
    np.testing.assert_allclose(drift_numerical, drift_analytic, atol=5e-3, rtol=5e-2)

    fig, ax = plt.subplots()
    ax.plot(1 / pitch, drift_analytic)
    ax.plot(1 / pitch, drift_numerical)
    plt.show()
    return fig
