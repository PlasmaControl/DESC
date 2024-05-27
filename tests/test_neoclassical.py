"""Test for neoclassical transport compute functions."""

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pytest
import quadax
from orthax.legendre import leggauss

from desc.compute import data_index, get_data_deps
from desc.compute._neoclassical import vec_quadax
from desc.compute.bounce_integral import bounce_integral
from desc.equilibrium import Equilibrium
from desc.equilibrium.coords import desc_grid_from_field_line_coords
from desc.grid import LinearGrid
from desc.utils import Timer


def _compute_field_line_data(eq, grid_desc, names_field_line, names_0d_or_1dr=None):
    """Compute field line quantities on correct grids.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to compute on.
    grid_desc : Grid
        Grid on which the field line quantities should be computed.
    names_field_line : list
        Field line quantities that will be computed on the returned field line grid.
        Should not include 0d or 1dr quantities.
    names_0d_or_1dr : list
        Things to compute that are constant throughout volume or over flux surface.

    Returns
    -------
    data : dict
        Computed quantities.

    """
    # TODO: https://github.com/PlasmaControl/DESC/issues/719
    if names_0d_or_1dr is None:
        names_0d_or_1dr = []
    p = "desc.equilibrium.equilibrium.Equilibrium"
    # Gather dependencies of given quantities.
    deps = (
        get_data_deps(names_field_line + names_0d_or_1dr, obj=p, has_axis=False)
        + names_0d_or_1dr
    )
    deps = list(set(deps))
    # Create grid with given flux surfaces.
    rho = grid_desc.compress(grid_desc.nodes[:, 0])
    grid1dr = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, sym=eq.sym, NFP=eq.NFP)
    # Compute dependencies on correct grids.
    seed_data = eq.compute(deps, grid=grid1dr)
    dep1dr = {dep for dep in deps if data_index[p][dep]["coordinates"] == "r"}
    dep0d = {dep for dep in deps if data_index[p][dep]["coordinates"] == ""}

    # Collect quantities that can be used as a seed to compute the
    # field line quantities over the grid mapped from field line coordinates.
    # (Single field line grid won't have enough poloidal resolution to
    # compute these quantities accurately).
    data0d = {key: val for key, val in seed_data.items() if key in dep0d}
    data1d = {
        key: grid_desc.copy_data_from_other(val, grid1dr)
        for key, val in seed_data.items()
        if key in dep1dr
    }
    data = data0d | data1d
    # Compute field line quantities with precomputed dependencies.
    for name in names_field_line:
        if name in data:
            del data[name]
    data = eq.compute(
        names=names_field_line, grid=grid_desc, data=data, override_grid=False
    )
    return data


@pytest.mark.unit
def test_effective_ripple():
    """Compare DESC effective ripple against neo stellopt."""
    timer = Timer()
    eq = Equilibrium.load(
        "tests/inputs/DESC_from_NAE_O_r1_precise_QI_plunk_fixed_bdry_r0"
        ".15_L_9_M_9_N_24_output.h5"
    )
    timer.start("map")
    grid = desc_grid_from_field_line_coords(
        eq,
        rho=np.linspace(0.01, 1, 20),
        alpha=np.array([0]),
        zeta=np.linspace(-10 * np.pi, 10 * np.pi, 100),
    )
    timer.stop("map")
    timer.disp("map")
    timer.start("dependency compute")
    data = _compute_field_line_data(
        eq,
        grid,
        ["B^zeta", "|B|_z|r,a", "|B|", "|grad(psi)|", "cvdrift0"],
        ["min_tz |B|", "max_tz |B|", "R0", "V_r(r)", "psi_r", "S(r)"],
    )
    timer.stop("dependency compute")
    timer.disp("dependency compute")
    timer.start("ripple compute")
    data = eq.compute(
        "ripple",
        grid=grid,
        data=data,
        override_grid=False,
        bounce_integral=partial(bounce_integral, quad=leggauss(28)),
        quad=vec_quadax(quadax.quadgk),
    )
    timer.stop("ripple compute")
    timer.disp("ripple compute")
    assert np.isfinite(data["ripple"]).all()
    rho = grid.compress(grid.nodes[:, 0])
    ripple = grid.compress(data["ripple"])
    fig, ax = plt.subplots(2)
    ax[0].plot(rho, ripple, marker="o", label="∫ db ∑ⱼ Hⱼ² / Iⱼ")
    ax[0].set_xlabel(r"$\rho$")
    ax[0].set_ylabel("ripple")
    ax[0].set_title("Ripple, defined as ∫ db ∑ⱼ Hⱼ² / Iⱼ")
    # Workaround until eq.compute() is fixed to only compute dependencies
    # that are needed for the requested computation. (So don't compute
    # dependencies of things already in data).
    data_R0 = eq.compute("R0")
    for key in data_R0:
        if key not in data:
            # Need to add R0's dependencies which are surface functions of zeta
            # aren't attempted to be recomputed on grid_desc.
            data[key] = data_R0[key]
    data = eq.compute("effective ripple", grid=grid, data=data, override_grid=False)
    assert np.isfinite(data["effective ripple"]).all()
    eff_ripple = grid.compress(data["effective ripple"])
    ax[1].plot(rho, eff_ripple, marker="o", label=r"$\epsilon_{\text{effective}}$")
    ax[1].set_xlabel(r"$\rho$")
    ax[1].set_ylabel(r"$\epsilon_{\text{effective}}$")
    ax[1].set_title("Effective ripple (not raised to 3/2 power)")
    plt.tight_layout()
    plt.show()
    plt.close()
