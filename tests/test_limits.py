"""Tests for compute functions evaluated at limits."""

import numpy as np
import pytest

from desc.compute.utils import compress, surface_averages
from desc.examples import get
from desc.grid import LinearGrid


class TestLimits:
    """Tests for compute functions evaluated at limits."""

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

        test(get("W7-X"), 2.708108)
