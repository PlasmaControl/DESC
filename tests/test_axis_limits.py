"""Tests for compute functions evaluated at limits."""

import numpy as np
import pytest

import desc.io
from desc.compute import data_index
from desc.compute.utils import compress, surface_averages
from desc.equilibrium import Equilibrium
from desc.examples import get
from desc.grid import LinearGrid


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
        assert np.isfinite(data["B0"]).all()

    @pytest.mark.unit
    def test_nonexistent_limits(self):
        """Test that quantities whose limit does not exist evaluates not finite."""
        eq = get("W7-X")
        grid = LinearGrid(L=5, M=5, N=5, sym=eq.sym, NFP=eq.NFP, axis=True)
        axis_mask = grid.nodes[:, 0] == 0
        no_limits = ["e^theta", "grad(alpha)"]
        data = eq.compute(names=no_limits, grid=grid)
        for quantity in no_limits:
            assert np.all(~np.isfinite(data[quantity][axis_mask]))

    @staticmethod
    def continuity(eq, name, expected_at_axis=None):
        """Test that the rho=0 axis limit of name is computed accurately."""
        delta = 1e-5
        epsilon = 1e-6
        grid = LinearGrid(
            rho=np.linspace(0, 1, 10) * delta, M=7, N=7, NFP=eq.NFP, sym=eq.sym
        )
        assert grid.axis.size
        quantity = eq.compute(name, grid=grid)[name]
        if data_index[name]["coordinates"] == "r":
            quantity = compress(grid, quantity)
        elif data_index[name]["coordinates"] != "":
            quantity = surface_averages(grid, quantity, expand_out=False)
        assert np.isfinite(quantity).all()
        # check continuity
        np.testing.assert_allclose(quantity[:-1], quantity[1:], atol=epsilon)
        if expected_at_axis is not None:
            # check value
            np.testing.assert_allclose(quantity[0], expected_at_axis, atol=epsilon)

    @pytest.mark.unit
    def test_e_theta(self):
        """Test that e_theta goes to 0 at magnetic axis."""
        # All limits rely on this.
        TestAxisLimits.continuity(get("W7-X"), "e_theta", expected_at_axis=0)

    @pytest.mark.unit
    def test_b_fsa(self):
        """Test axis limit of B."""
        TestAxisLimits.continuity(get("W7-X"), "B")

    @pytest.mark.unit
    @pytest.mark.solve
    def test_rotational_transform(self, DSHAPE_current):
        """Test axis limit of iota."""
        # test should be done on equilibria with fixed current profiles
        computed_close_to_axis = -0.994167
        TestAxisLimits.continuity(
            desc.io.load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1],
            "iota",
            expected_at_axis=computed_close_to_axis,
        )
        computed_close_to_axis = -0.360675
        TestAxisLimits.continuity(
            get("QAS"), "iota", expected_at_axis=computed_close_to_axis
        )
