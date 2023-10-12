"""Tests for compatability functions."""

import numpy as np
import pytest

from desc.compat import flip_helicity
from desc.examples import get
from desc.grid import Grid, QuadratureGrid


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
    np.testing.assert_array_less(grid.compress(data_old["iota"]), 0)
    np.testing.assert_array_less(0, grid.compress(data_new["iota"]))

    # check that current changed sign
    np.testing.assert_array_less(grid.compress(data_old["current"]), 0)
    np.testing.assert_array_less(0, grid.compress(data_new["current"]))

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
    np.testing.assert_allclose(data_old["|F|"], data_flip["|F|"])


@pytest.mark.unit
@pytest.mark.solve
def test_flip_helicity_current():
    """Test flip_helicity on an Equilibrium with a current profile."""
    # TODO: change this to use HSX example
    eq = get("precise_QH")

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
    np.testing.assert_array_less(0, grid.compress(data_old["iota"]))  # old: +
    np.testing.assert_array_less(grid.compress(data_new["iota"]), 0)  # new: -

    # check that current changed sign
    """
    # FIXME: include these tests for a QH case that has finite current, like HSX
    np.testing.assert_array_less(0, grid.compress(data_old["current"]))  # old: +
    np.testing.assert_array_less(grid.compress(data_new["current"]), 0)  # new: -
    """

    # check that stability did not change
    np.testing.assert_allclose(
        grid.compress(data_old["D_Mercier"]), grid.compress(data_new["D_Mercier"])
    )

    # check that the total force balance error on each surface did not change
    # (equivalent collocation points now corresond to the opposite zeta values)
    np.testing.assert_allclose(data_old["|F|"], data_flip["|F|"])

    # check that the QH errors now need the opposite helicity
    # (equivalent collocation points now corresond to the opposite zeta values)
    np.testing.assert_allclose(data_old["f_C"], data_flip["f_C"], atol=1e-12)
    # FIXME: can probably set atol=0 when switch to HSX example (b/c not precise QH)
