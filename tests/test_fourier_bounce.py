"""Test interpolation to Clebsch coordinates and Fourier bounce integration."""

import numpy as np
import pytest

from desc.compute.bounce_integral import get_pitch
from desc.compute.fourier_bounce_integral import (
    FourierChebyshevBasis,
    _alpha_sequence,
    bounce_integral,
    required_names,
)
from desc.equilibrium.coords import map_coordinates
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
    alphas = _alpha_sequence(alpha_0, iota, num_period, period)
    assert alphas.shape == (iota.size, num_period)
    for i in range(iota.size):
        assert np.unique(alphas[i]).size == num_period, "Is iota irrational?"
    print(alphas)


@pytest.mark.unit
def test_fourier_chebyshev(rho=1, M=8, N=32, f=lambda x: x):
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
