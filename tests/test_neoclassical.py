"""Test for neoclassical transport compute functions."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from desc.compute import data_index, get_data_deps
from desc.compute.bounce_integral import desc_grid_from_field_line_coords
from desc.examples import get
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
    names_0d_or_1dr.append("iota")
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
    data = {}
    data.update(data0d)
    data.update(data1d)
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
    eq = get("HELIOTRON")
    grid_desc, grid_fl = desc_grid_from_field_line_coords(
        eq, rho=np.linspace(1e-7, 1, 10)
    )
    data = _compute_field_line_data(
        eq,
        grid_desc,
        ["B^zeta", "|B|_z|r,a", "|B|", "|grad(psi)|", "cvdrift0"],
        ["max_tz |B|"],
    )
    data = eq.compute(
        "ripple", grid=grid_desc, data=data, override_grid=False, grid_fl=grid_fl
    )
    assert np.isfinite(data["ripple"]).all()
    rho = grid_desc.compress(grid_desc.nodes[:, 0])
    ripple = grid_desc.compress(data["ripple"])
    fig, ax = plt.subplots()
    ax.plot(rho, ripple, label="ripple")
    plt.show()
    plt.close()
    data = eq.compute("effective ripple", grid=grid_desc, data=data)
    assert np.isfinite(data["effective ripple"]).all()
    eff_ripple = grid_desc.compress(data["effective ripple"])
    fig, ax = plt.subplots()
    ax.plot(rho, eff_ripple, label="Effective ripple")
    plt.show()
    plt.close()
