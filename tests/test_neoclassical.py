"""Test for neoclassical transport compute functions."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from desc.compute._neoclassical import _poloidal_average, poloidal_leggauss
from desc.equilibrium import Equilibrium
from desc.equilibrium.coords import rtz_grid
from desc.equilibrium.equilibrium import compute_raz_data


@pytest.mark.unit
def test_field_line_average():
    """Test that field line average converges to surface average."""
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
    data = compute_raz_data(eq, grid, ["L|r,a", "G|r,a"], names_1dr=["V_r(r)"])
    np.testing.assert_allclose(
        _poloidal_average(grid.source_grid, data["L|r,a"] / data["G|r,a"]),
        grid.compress(data["V_r(r)"]),
        rtol=3e-2,
    )

    # Now for field line with large L.
    L = 20 * np.pi
    zeta = np.linspace(0, L, resolution * 2)
    grid = rtz_grid(eq, rho, 0, zeta, coordinates="raz")
    data = compute_raz_data(eq, grid, ["L|r,a", "G|r,a"], names_1dr=["V_r(r)"])
    np.testing.assert_allclose(
        np.squeeze(data["L|r,a"] / data["G|r,a"]),
        grid.compress(data["V_r(r)"]),
        rtol=3e-2,
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
        ["B^zeta", "|B|", "|B|_z|r,a", "|grad(psi)|", "kappa_g", "L|r,a"],
        names_0d=["R0"],
        names_1dr=["min_tz |B|", "max_tz |B|", "V_r(r)", "psi_r", "S(r)"],
    )
    data = eq.compute(
        "effective ripple raw",
        grid=grid,
        data=data,
        override_grid=False,
        batch=False,  # noqa: E800
        # quad=vec_quadax(quadax.quadgk),  # noqa: E800
    )
    assert np.isfinite(data["effective ripple raw"]).all()
    rho = grid.compress(grid.nodes[:, 0])
    ripple = grid.compress(data["effective ripple raw"])
    fig, ax = plt.subplots(2)
    ax[0].plot(rho, ripple, marker="o")
    ax[0].set_xlabel(r"$\rho$")
    ax[0].set_ylabel("effective ripple raw")
    ax[0].set_title(r"∫ dλ λ⁻² $\langle$ ∑ⱼ Hⱼ²/Iⱼ $\rangle$")

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
    ax[1].plot(rho, eff_ripple, marker="o")
    ax[1].set_xlabel(r"$\rho$")
    ax[1].set_ylabel(r"$\epsilon_{\text{eff}}^{3/2}$")
    ax[1].set_title(
        r"ε¹ᐧ⁵ = π/(8√2) (R₀(∂_ψ V)/S)² ∫ dλ λ⁻² $\langle$ ∑ⱼ Hⱼ²/Iⱼ $\rangle$"
    )
    plt.tight_layout()
    plt.show()
    plt.close()
