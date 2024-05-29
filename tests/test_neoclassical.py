"""Test for neoclassical transport compute functions."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from desc.compute._neoclassical import poloidal_leggauss
from desc.equilibrium import Equilibrium
from desc.equilibrium.coords import rtz_grid
from desc.equilibrium.equilibrium import compute_raz_data


@pytest.mark.unit
def test_integration_on_field_line():
    """Test that V_psi(r)*L / S(r)*L = V_psi(r) / S(r) as L → ∞."""
    eq = Equilibrium.load(
        "tests/inputs/DESC_from_NAE_O_r1_precise_QI_plunk_fixed_bdry_r0"
        ".15_L_9_M_9_N_24_output.h5"
    )
    resolution = 10
    rho = np.linspace(0, 1, resolution)

    # Surface average field lines truncated at 1 toroidal transit.
    alpha, w = poloidal_leggauss(resolution)
    L = 2 * np.pi
    zeta = np.linspace(0, L, resolution)
    grid = rtz_grid(eq, rho, alpha, zeta, coordinates="raz")
    grid.source_grid.poloidal_weight = w
    data = compute_raz_data(
        eq, grid, ["V_psi(r)*L", "S(r)*L"], names_1dr=["psi_r", "V_r(r)", "S(r)"]
    )
    np.testing.assert_allclose(
        data["V_psi(r)*L"] / data["S(r)*L"],
        data["V_r(r)"] / data["psi_r"] / data["S(r)"],
        rtol=0.01,
    )

    # Now for field line with large L.
    # The drawback of this approach is that convergence can regress
    # unless resolution is increased linearly with L.
    L = 100 * np.pi
    zeta = np.linspace(0, L, resolution**2)
    grid = rtz_grid(eq, rho, 0, zeta, coordinates="raz")
    data = compute_raz_data(
        eq, grid, ["V_psi(r)*L", "S(r)*L"], names_1dr=["psi_r", "V_r(r)", "S(r)"]
    )
    np.testing.assert_allclose(
        data["V_psi(r)*L"] / data["S(r)*L"],
        data["V_r(r)"] / data["psi_r"] / data["S(r)"],
        rtol=0.01,
    )


@pytest.mark.unit
def test_effective_ripple():
    """Compare DESC effective ripple against neo stellopt."""
    eq = Equilibrium.load(
        "tests/inputs/DESC_from_NAE_O_r1_precise_QI_plunk_fixed_bdry_r0"
        ".15_L_9_M_9_N_24_output.h5"
    )
    grid = rtz_grid(
        eq,
        radial=np.linspace(0, 1, 20),
        poloidal=np.array([0]),
        toroidal=np.linspace(0, 100 * np.pi, 500),
        coordinates="raz",
    )
    data = compute_raz_data(
        eq,
        grid,
        [
            "B^zeta",
            "|B|_z|r,a",
            "|B|",
            "|grad(psi)|",
            "cvdrift0",
            "V_psi(r)*L",
            "S(r)*L",
        ],
        names_0d=["R0"],
        names_1dr=["min_tz |B|", "max_tz |B|"],
    )
    data = eq.compute(
        "effective ripple raw",
        grid=grid,
        data=data,
        override_grid=False,
        # batch=False,  # noqa: E800
        # quad=vec_quadax(quadax.quadgk),  # noqa: E800
    )
    assert np.isfinite(data["effective ripple raw"]).all()
    rho = grid.compress(grid.nodes[:, 0])
    ripple = grid.compress(data["effective ripple raw"])
    fig, ax = plt.subplots(2)
    ax[0].plot(rho, ripple, marker="o", label="∫ db ∑ⱼ Hⱼ² / Iⱼ")
    ax[0].set_xlabel(r"$\rho$")
    ax[0].set_ylabel("effective ripple raw")
    ax[0].set_title("effective ripple raw, defined as ∫ db ∑ⱼ Hⱼ² / Iⱼ")

    # Workaround until eq.compute() is fixed to only compute dependencies
    # that are needed for the requested computation. (So don't compute
    # dependencies of things already in data).
    data_R0 = eq.compute("R0")
    for key in data_R0:
        if key not in data:
            # Need to add R0's dependencies which are surface functions of zeta
            # aren't attempted to be recomputed on grid_desc.
            data[key] = data_R0[key]
    data = eq.compute("effective ripple", grid=grid, data=data)
    assert np.isfinite(data["effective ripple"]).all()
    eff_ripple = grid.compress(data["effective ripple"])
    ax[1].plot(rho, eff_ripple, marker="o", label=r"$\epsilon_{\text{effective}}$")
    ax[1].set_xlabel(r"$\rho$")
    ax[1].set_ylabel(r"$\epsilon_{\text{effective}}$")
    ax[1].set_title("Effective ripple (not raised to 3/2 power)")
    plt.tight_layout()
    plt.show()
    plt.close()
