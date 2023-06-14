"""Tests for compute functions evaluated at limits."""

import numpy as np
import pytest

import desc.io
from desc.compute.utils import compress, surface_averages
from desc.equilibrium import Equilibrium
from desc.examples import get
from desc.grid import LinearGrid


class TestAxisLimits:
    """Tests for compute functions evaluated at limits."""

    @pytest.mark.unit
    def test_compute_axis_limit_api(self):
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
    def test_b_mag_fsa(self):
        """Test continuity of <|B|>. Failure indicates B0 limit is wrong."""

        def test(eq, expected_at_axis):
            delta = 1e-3
            epsilon = 1e-6
            rho = np.linspace(0, delta, 10)
            lg = LinearGrid(rho=rho, M=7, N=7, NFP=eq.NFP, sym=eq.sym)
            b_mag_fsa_no_sqrt_g = compress(
                lg, surface_averages(lg, eq.compute("|B|", grid=lg)["|B|"])
            )
            # check continuity
            assert np.isfinite(b_mag_fsa_no_sqrt_g).all()
            np.testing.assert_allclose(
                b_mag_fsa_no_sqrt_g[:-1], b_mag_fsa_no_sqrt_g[1:], atol=epsilon
            )
            # check value
            np.testing.assert_allclose(
                b_mag_fsa_no_sqrt_g[0], expected_at_axis, atol=epsilon
            )

        value_computed_close_to_axis = 2.708108
        test(get("W7-X"), value_computed_close_to_axis)

    @pytest.mark.unit
    @pytest.mark.solve
    def test_rotational_transform(self, DSHAPE_current):
        """Test that the limit at rho=0 axis is computed accurately."""
        # test should be done on equilibria with fixed current profiles
        def test(eq, expected_at_axis):
            delta = 1e-3
            epsilon = 1e-6
            rho = np.linspace(0, delta, 10)
            lg = LinearGrid(rho=rho, M=5, N=5, NFP=eq.NFP, sym=eq.sym)
            iota = compress(lg, eq.compute("iota", grid=lg)["iota"])
            # check continuity
            assert np.isfinite(iota).all()
            np.testing.assert_allclose(iota[:-1], iota[1:], atol=epsilon)
            # check value
            np.testing.assert_allclose(iota[0], expected_at_axis, atol=epsilon)

        eq = desc.io.load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
        value_computed_close_to_axis = -0.994167
        test(eq, value_computed_close_to_axis)
        value_computed_close_to_axis = -0.360675
        test(get("QAS"), value_computed_close_to_axis)
