"""Tests for compute functions evaluated at limits."""

import numpy as np
import pytest

import desc.io
from desc.compute import data_index
from desc.compute.utils import surface_integrals
from desc.equilibrium import Equilibrium
from desc.examples import get
from desc.grid import LinearGrid


limit_does_not_exist = {
    "D_shear",  # may exist for some iota profiles
    "D_current",
    "D_well",
    "D_geodesic",
    "D_Mercier",
    "e^theta",
    "grad(alpha)",
    "J^theta",
    "e^helical",
    "|e^helical|",
    "g^tt",
    "g^rt",
    "g^tz",
    "|grad(theta)|",
    # TODO: contravariant basis vector derivatives
}


class TestAxisLimits:
    """Tests for compute functions evaluated at limits."""

    @pytest.mark.unit
    def test_axis_limit_api(self):
        """Test that axis limit dependencies are computed only when necessary."""
        eq = Equilibrium()
        grid = LinearGrid(L=2, M=2, N=2, axis=False)
        assert not grid.axis.size
        data = eq.compute("B0", grid=grid)
        assert "B0" in data and "psi_r" in data and "sqrt(g)" in data
        # assert axis limit dependencies are not in data
        assert "psi_rr" not in data and "sqrt(g)_r" not in data
        grid = LinearGrid(L=2, M=2, N=2, axis=True)
        assert grid.axis.size
        data = eq.compute("B0", grid=grid)
        assert "B0" in data and "psi_r" in data and "sqrt(g)" in data
        # assert axis limit dependencies are in data
        assert "psi_rr" in data and "sqrt(g)_r" in data
        assert np.all(np.isfinite(data["B0"]))

    @pytest.mark.unit
    def test_limit_existence(self):
        """Test that only quantities which lack limits do not evaluate at axis."""
        for eq in ("W7-X", "QAS"):
            eq = get(eq)
            grid = LinearGrid(L=2, M=2, N=2, sym=eq.sym, NFP=eq.NFP, axis=True)
            assert grid.axis.size
            data = eq.compute(data_index.keys(), grid=grid)
            assert data.keys() == data_index.keys()
            for key, val in data.items():
                if key in limit_does_not_exist:
                    assert np.all(~np.isfinite(val[grid.axis]))
                else:
                    assert np.all(np.isfinite(val))

    @staticmethod
    def continuity(eq, name, expected_at_axis=None):
        """Test that the rho=0 axis limit of name is a continuous extension."""
        if data_index[name]["coordinates"] == "":
            return
        delta = 1e-5  # any smaller accumulates finite precision errors
        epsilon = 1e-5
        rho = np.linspace(0, 1, 10) * delta
        grid = LinearGrid(rho=rho, M=7, N=7, NFP=eq.NFP, sym=eq.sym)
        assert grid.axis.size

        quantity = eq.compute(name, grid=grid)[name]
        if data_index[name]["coordinates"] == "r":
            quantity = grid.compress(quantity)
        else:
            quantity = surface_integrals(grid, np.abs(quantity), expand_out=False)
        # check continuity
        np.testing.assert_allclose(quantity[:-1], quantity[1:], atol=epsilon)
        # check expected value at axis
        if expected_at_axis is None:
            # fit the data (except axis pt) to a polynomial to extrapolate to axis
            poly = np.polyfit(rho[1:], quantity[1:], 6)
            expected_at_axis = poly[-1]  # constant term is same as eval poly at rho=0
        np.testing.assert_allclose(quantity[0], expected_at_axis, atol=epsilon)

    @pytest.mark.unit
    def test_zero_limits(self):
        """Test limits of basic quantities that should be 0 at magnetic axis."""
        w7x = get("W7-X")  # fixed iota
        qas = get("QAS")  # fixed current
        for key in ("rho", "psi", "psi_r", "e_theta", "sqrt(g)"):
            TestAxisLimits.continuity(w7x, key, expected_at_axis=0)
            TestAxisLimits.continuity(qas, key, expected_at_axis=0)

    @pytest.mark.unit
    def test_limit_value(self):
        """Heuristic to test correctness of all quantities with limits."""
        w7x = get("W7-X")  # fixed iota
        qas = get("QAS")  # fixed current
        for key in data_index.keys() - limit_does_not_exist:
            TestAxisLimits.continuity(w7x, key)
            TestAxisLimits.continuity(qas, key)

    @pytest.mark.unit
    @pytest.mark.solve
    def test_rotational_transform(self, DSHAPE_current):
        """Test axis limit of iota."""
        # test should be done on equilibria with fixed current profiles
        TestAxisLimits.continuity(
            desc.io.load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1],
            "iota",
        )
