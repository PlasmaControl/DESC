"""Tests for compatability functions."""

import numpy as np
import pytest

from desc.compat import flip_helicity, rescale
from desc.examples import get
from desc.grid import Grid, LinearGrid, QuadratureGrid


@pytest.mark.unit
def test_flip_helicity_axisym():
    """Test flip_helicity on an axisymmetric Equilibrium."""
    eq = get("DSHAPE")

    grid = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    data_keys = ["current", "|F|", "D_Mercier"]

    data_old = eq.compute(data_keys, grid=grid)
    eq = flip_helicity(eq)
    data_new = eq.compute(data_keys, grid=grid)

    # check that basis vectors did not change
    np.testing.assert_allclose(data_old["e_rho"], data_new["e_rho"])
    np.testing.assert_allclose(data_old["e_theta"], data_new["e_theta"])
    np.testing.assert_allclose(data_old["e^zeta"], data_new["e^zeta"])

    # check that Jacobian is still positive
    np.testing.assert_array_less(0, grid.compress(data_new["sqrt(g)"]))

    # check that iota changed sign
    np.testing.assert_array_less(0, grid.compress(data_old["iota"]))  # old: +
    np.testing.assert_array_less(grid.compress(data_new["iota"]), 0)  # new: -

    # check that current changed sign
    np.testing.assert_array_less(0, grid.compress(data_old["current"]))  # old: +
    np.testing.assert_array_less(grid.compress(data_new["current"]), 0)  # new: -

    # check that stability did not change
    np.testing.assert_allclose(
        grid.compress(data_old["D_Mercier"]), grid.compress(data_new["D_Mercier"])
    )

    # check that force balance did not change
    # (the collocation points are the same because of axisymmetry)
    np.testing.assert_allclose(data_old["|F|"], data_new["|F|"])


@pytest.mark.unit
@pytest.mark.solve
def test_flip_helicity_iota():
    """Test flip_helicity on an Equilibrium with an iota profile."""
    eq = get("HELIOTRON")

    grid = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    nodes = grid.nodes.copy()
    nodes[:, -1] *= -1
    grid_flip = Grid(nodes)  # grid with negative zeta values
    data_keys = ["current", "|F|", "D_Mercier"]

    data_old = eq.compute(data_keys, grid=grid)
    eq = flip_helicity(eq)
    data_new = eq.compute(data_keys, grid=grid)
    data_flip = eq.compute(data_keys, grid=grid_flip)

    # check that basis vectors did not change
    np.testing.assert_allclose(data_old["e_rho"], data_flip["e_rho"])
    np.testing.assert_allclose(data_old["e_theta"], data_flip["e_theta"])
    np.testing.assert_allclose(data_old["e^zeta"], data_flip["e^zeta"])

    # check that Jacobian is still positive
    np.testing.assert_array_less(0, grid.compress(data_new["sqrt(g)"]))

    # check that iota changed sign
    np.testing.assert_array_less(0, grid.compress(data_old["iota"]))  # old: +
    np.testing.assert_array_less(grid.compress(data_new["iota"]), 0)  # new: -

    # check that current changed sign
    np.testing.assert_array_less(0, grid.compress(data_old["current"]))  # old: +
    np.testing.assert_array_less(grid.compress(data_new["current"]), 0)  # new: -

    # check that stability did not change
    np.testing.assert_allclose(
        grid.compress(data_old["D_Mercier"]), grid.compress(data_new["D_Mercier"])
    )

    # check that the total force balance error on each surface did not change
    # (equivalent collocation points now corresond to the opposite zeta values)
    np.testing.assert_allclose(data_old["|F|"], data_flip["|F|"], rtol=1e-3)


@pytest.mark.unit
@pytest.mark.solve
def test_flip_helicity_current():
    """Test flip_helicity on an Equilibrium with a current profile."""
    eq = get("HSX")

    grid = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    nodes = grid.nodes.copy()
    nodes[:, -1] *= -1
    grid_flip = Grid(nodes)  # grid with negative zeta values
    data_keys = ["current", "|F|", "D_Mercier", "f_C"]

    data_old = eq.compute(data_keys, grid=grid, helicity=(1, eq.NFP))
    eq = flip_helicity(eq)
    data_new = eq.compute(data_keys, grid=grid, helicity=(-1, eq.NFP))
    data_flip = eq.compute(data_keys, grid=grid_flip, helicity=(-1, eq.NFP))
    # note that now the QH helicity is reversed: (M, N) -> (-M, N)

    # check that basis vectors did not change
    np.testing.assert_allclose(data_old["e_rho"], data_flip["e_rho"])
    np.testing.assert_allclose(data_old["e_theta"], data_flip["e_theta"])
    np.testing.assert_allclose(data_old["e^zeta"], data_flip["e^zeta"])

    # check that Jacobian is still positive
    np.testing.assert_array_less(0, grid.compress(data_new["sqrt(g)"]))

    # check that iota changed sign
    np.testing.assert_array_less(grid.compress(data_old["iota"]), 0)  # old: -
    np.testing.assert_array_less(0, grid.compress(data_new["iota"]))  # new: +

    # current=0 so do not need to check sign change

    # check that stability did not change
    np.testing.assert_allclose(
        grid.compress(data_old["D_Mercier"]), grid.compress(data_new["D_Mercier"])
    )

    # check that the total force balance error on each surface did not change
    # (equivalent collocation points now corresond to the opposite zeta values)
    np.testing.assert_allclose(data_old["|F|"], data_flip["|F|"], rtol=1e-3)

    # check that the QH errors now need the opposite helicity
    # (equivalent collocation points now corresond to the opposite zeta values)
    np.testing.assert_allclose(data_old["f_C"], data_flip["f_C"], atol=1e-8)


@pytest.mark.unit
def test_rescale():
    """Test rescale function."""

    def fun(eq):
        """Compute the quantities that can be scaled."""
        grid_quad = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        data_quad = eq.compute(
            ["R0", "a", "R0/a", "V", "<|B|>_vol", "|F|", "|grad(|B|^2)|/2mu0"],
            grid_quad,
        )
        grid_axis = LinearGrid(N=eq.N_grid, NFP=eq.NFP, rho=0)
        data_axis = eq.compute("|B|", grid_axis)
        grid_lcfs = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=1)
        data_lcfs = eq.compute("|B|", grid_lcfs)
        return {
            "R0": data_quad["R0"],
            "a": data_quad["a"],
            "R0/a": data_quad["R0/a"],
            "V": data_quad["V"],
            "B0": np.mean(data_axis["|B|"]),
            "<|B|>_vol": data_quad["<|B|>_vol"],
            "B_max": np.max(data_lcfs["|B|"]),
            "err": data_quad["|F|"] / data_quad["|grad(|B|^2)|/2mu0"],
        }

    eq0 = get("DSHAPE")
    old_vals = fun(eq0)
    # original values: R0=3.5, a=1.2, V=100, B0=0.2, <|B|>_vol=0.22, B_max=0.3

    # check that defaults do no scaling
    eq = eq0.copy()
    eq = rescale(eq)
    new_vals = fun(eq)
    for key, val in old_vals.items():
        np.testing.assert_allclose(new_vals[key], val)

    # scale R0 and B0
    eq = eq0.copy()
    eq = rescale(eq, L=("R0", 1), B=("B0", 0.5))
    new_vals = fun(eq)
    np.testing.assert_allclose(new_vals["R0"], 1)
    np.testing.assert_allclose(new_vals["B0"], 0.5)
    np.testing.assert_allclose(new_vals["R0/a"], old_vals["R0/a"])
    np.testing.assert_allclose(new_vals["err"], old_vals["err"], atol=1e-10)

    # scale a and <|B|>_vol
    eq = eq0.copy()
    eq = rescale(eq, L=("a", 0.5), B=("<|B|>_vol", 1))
    new_vals = fun(eq)
    np.testing.assert_allclose(new_vals["a"], 0.5)
    np.testing.assert_allclose(new_vals["<|B|>_vol"], 1)
    np.testing.assert_allclose(new_vals["R0/a"], old_vals["R0/a"])
    np.testing.assert_allclose(new_vals["err"], old_vals["err"], atol=1e-10)

    # scale V and B_max
    eq = eq0.copy()
    eq = rescale(eq, L=("V", 200), B=("B_max", 2))
    new_vals = fun(eq)
    np.testing.assert_allclose(new_vals["V"], 200)
    np.testing.assert_allclose(new_vals["B_max"], 2)
    np.testing.assert_allclose(new_vals["R0/a"], old_vals["R0/a"])
    np.testing.assert_allclose(new_vals["err"], old_vals["err"], atol=1e-10)
