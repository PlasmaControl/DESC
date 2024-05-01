"""Test for neoclassical transport compute functions."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from desc.compute import data_index, get_data_deps
from desc.compute.bounce_integral import (
    desc_grid_from_field_line_coords,
    tanh_sinh_quad,
)
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid


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
    eq = Equilibrium.load(
        "tests/inputs/DESC_from_NAE_O_r1_precise_QI_plunk_fixed_bdry_r0"
        ".15_L_9_M_9_N_24_output.h5"
    )
    grid_desc, grid_fl = desc_grid_from_field_line_coords(
        eq,
        rho=np.linspace(0.01, 1, 20),
        zeta=np.linspace(-10 * np.pi, 10 * np.pi, 100),
    )
    data = _compute_field_line_data(
        eq,
        grid_desc,
        ["B^zeta", "|B|_z|r,a", "|B|", "|grad(psi)|", "cvdrift0"],
        ["max_tz |B|", "R0", "V_r(r)", "psi_r", "S(r)"],
    )
    data = eq.compute(
        "ripple",
        grid=grid_desc,
        data=data,
        override_grid=False,
        grid_fl=grid_fl,
        b_quad_res=5,
        # Gauss-Legendre quadrature with sin automorph ~28 nodes.
        # But the real advantage is that Gauss-Legendre with sin
        # allows for composite Gauss-Legendre quadrature which
        # will be able to match 30 node quadrature with maybe ~10 nodes.
        # So very large memory savings.
        quad=lambda: tanh_sinh_quad(41),
        batched=False,
        # check=True,  # noqa: E800
        # plot=True,  # noqa: E800
    )
    assert np.isfinite(data["ripple"]).all()
    rho = grid_desc.compress(grid_desc.nodes[:, 0])
    ripple = grid_desc.compress(data["ripple"])
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
    data = eq.compute(
        "effective ripple",
        grid=grid_desc,
        data=data,
        grid_fl=grid_fl,
        override_grid=False,
    )
    assert np.isfinite(data["effective ripple"]).all()
    eff_ripple = grid_desc.compress(data["effective ripple"])
    ax[1].plot(rho, eff_ripple, marker="o", label=r"$\epsilon_{\text{effective}}$")
    ax[1].set_xlabel(r"$\rho$")
    ax[1].set_ylabel(r"$\epsilon_{\text{effective}}$")
    ax[1].set_title("Effective ripple (not raised to 3/2 power)")
    plt.tight_layout()
    plt.show()
    plt.close()
