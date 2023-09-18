"""Tests for compatability functions."""

import numpy as np
import pytest

from desc.compat import ensure_positive_iota
from desc.compute.utils import surface_integrals
from desc.equilibrium import Equilibrium
from desc.grid import QuadratureGrid
from desc.vmec import VMECIO


@pytest.mark.unit
def test_ensure_positive_iota_axisym():
    """Test ensure_positive_iota on an axisymmetric Equilibrium."""
    eq = VMECIO.load("tests/inputs/wout_DSHAPE.nc")
    grid = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    data_keys = ["iota", "current", "|F|", "D_Mercier"]

    data_old = eq.compute(data_keys, grid=grid)
    eq = ensure_positive_iota(eq)
    data_new = eq.compute(data_keys, grid=grid)

    # check that Jacobian is still positive
    np.testing.assert_array_less(0, grid.compress(data_new["sqrt(g)"]))

    # check that iota actually changed sign
    np.testing.assert_array_less(grid.compress(data_old["iota"]), 0)
    np.testing.assert_array_less(0, grid.compress(data_new["iota"]))

    # check that current actually changed sign
    np.testing.assert_array_less(grid.compress(data_old["current"]), 0)
    np.testing.assert_array_less(0, grid.compress(data_new["current"]))

    # check that force balance did not change
    # (the collocation points are the same because of axisymmetry)
    np.testing.assert_allclose(data_old["|F|"], data_new["|F|"])

    # check that stability did not change
    np.testing.assert_allclose(
        grid.compress(data_old["D_Mercier"]), grid.compress(data_new["D_Mercier"])
    )


@pytest.mark.unit
@pytest.mark.solve
def test_ensure_positive_iota():
    """Test ensure_positive_iota on an Equilibrium with an iota profile."""
    # this is the DummyStellarator, except with a negative iota profile
    inputs = {
        "sym": True,
        "NFP": 3,
        "Psi": 1.0,
        "L": 4,
        "M": 4,
        "N": 4,
        "pressure": np.array([[0, 1e4], [2, -2e4], [4, 1e4]]),
        "iota": np.array([[0, -0.5], [2, -0.5]]),
        "surface": np.array(
            [
                [0, 0, 0, 3, 0],
                [0, 1, 0, 1, 0],
                [0, -1, 0, 0, -1],
                [0, 1, 1, 0.3, 0],
                [0, -1, -1, 0.3, 0],
                [0, 1, -1, 0, -0.3],
                [0, -1, 1, 0, 0.3],
            ],
        ),
    }
    eq = Equilibrium(**inputs)
    eq.solve()

    grid = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    data_keys = ["iota", "current", "|F|", "D_Mercier"]

    data_old = eq.compute(data_keys, grid=grid)
    eq = ensure_positive_iota(eq)
    data_new = eq.compute(data_keys, grid=grid)

    # check that Jacobian is still positive
    np.testing.assert_array_less(0, grid.compress(data_new["sqrt(g)"]))

    # check that iota actually changed sign
    np.testing.assert_array_less(grid.compress(data_old["iota"]), 0)
    np.testing.assert_array_less(0, grid.compress(data_new["iota"]))

    # check that current actually changed sign
    np.testing.assert_array_less(grid.compress(data_old["current"]), 0)
    np.testing.assert_array_less(0, grid.compress(data_new["current"]))

    # check that the total force balance error on each surface did not change
    # (the collocation points are different because the boundary surface changed)
    np.testing.assert_allclose(
        surface_integrals(grid, data_old["|F|"], expand_out=False),
        surface_integrals(grid, data_new["|F|"], expand_out=False),
    )

    # check that stability did not change
    np.testing.assert_allclose(
        grid.compress(data_old["D_Mercier"]), grid.compress(data_new["D_Mercier"])
    )


@pytest.mark.unit
@pytest.mark.solve
def test_ensure_positive_current():
    """Test ensure_positive_iota on an Equilibrium with a current profile."""
    # this is the DummyStellarator, except with a negative current profile
    inputs = {
        "sym": True,
        "NFP": 3,
        "Psi": 1.0,
        "L": 4,
        "M": 4,
        "N": 4,
        "pressure": np.array([[0, 1e4], [2, -2e4], [4, 1e4]]),
        "current": np.array([[2, -1e5]]),
        "surface": np.array(
            [
                [0, 0, 0, 3, 0],
                [0, 1, 0, 1, 0],
                [0, -1, 0, 0, -1],
                [0, 1, 1, 0.3, 0],
                [0, -1, -1, -0.3, 0],
                [0, 1, -1, 0, 0.3],
                [0, -1, 1, 0, 0.3],
            ],
        ),
    }
    eq = Equilibrium(**inputs)
    eq.solve()

    grid = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    data_keys = ["iota", "current", "|F|", "D_Mercier"]

    data_old = eq.compute(data_keys, grid=grid)
    eq = ensure_positive_iota(eq)
    data_new = eq.compute(data_keys, grid=grid)

    # check that Jacobian is still positive
    np.testing.assert_array_less(0, grid.compress(data_new["sqrt(g)"]))

    # check that iota actually changed sign
    np.testing.assert_array_less(grid.compress(data_old["iota"]), 0)
    np.testing.assert_array_less(0, grid.compress(data_new["iota"]))

    # check that current actually changed sign
    np.testing.assert_array_less(grid.compress(data_old["current"]), 0)
    np.testing.assert_array_less(0, grid.compress(data_new["current"]))

    # check that the total force balance error on each surface did not change
    # (the collocation points are different because the boundary surface changed)
    np.testing.assert_allclose(
        surface_integrals(grid, data_old["|F|"], expand_out=False),
        surface_integrals(grid, data_new["|F|"], expand_out=False),
    )

    # check that stability did not change
    np.testing.assert_allclose(
        grid.compress(data_old["D_Mercier"]), grid.compress(data_new["D_Mercier"])
    )
