"""Test for neoclassical transport compute functions."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from tests.test_plotting import tol_1d

from desc import examples
from desc.equilibrium.coords import rtz_grid


@pytest.mark.unit
def test_field_line_average():
    """Test that field line average converges to surface average."""
    eq = examples.get("W7-X")
    grid = rtz_grid(
        eq,
        np.array([0, 0.5]),
        np.array([0]),
        np.linspace(0, 40 * np.pi, 200),
        coordinates="raz",
        period=(np.inf, 2 * np.pi, np.inf),
    )
    data = eq.compute(["<L|r,a>", "<G|r,a>", "V_r(r)"], grid=grid)
    np.testing.assert_allclose(
        data["<L|r,a>"] / data["<G|r,a>"], data["V_r(r)"] / (4 * np.pi**2), rtol=1e-3
    )


@pytest.mark.unit
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_effective_ripple():
    """Test effective ripple with W7-X."""
    eq = examples.get("W7-X")
    rho = np.linspace(0, 1, 10)
    grid = rtz_grid(
        eq,
        rho,
        np.array([0]),
        np.linspace(0, 20 * np.pi, 1000),
        coordinates="raz",
        period=(np.inf, 2 * np.pi, np.inf),
    )
    data = eq.compute("effective ripple", grid=grid)
    assert np.isfinite(data["effective ripple"]).all()
    fig, ax = plt.subplots()
    ax.plot(rho, grid.compress(data["effective ripple"]), marker="o")
    return fig
