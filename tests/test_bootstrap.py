"""Tests for bootstrap current functions."""

import numpy as np
import pytest

from desc.compute._bootstrap import trapped_fraction


class TestBootstrap:
    """Tests for bootstrap current functions"""

    @pytest.mark.unit
    def test_trapped_fraction(self):
        """
        Confirm that the quantities computed by trapped_fraction()
        match analytic results for a model magnetic field.
        """
        ns = 2
        ntheta = 200
        nfp = 3
        results = []
        for nphi in [1, 50]:
            modB = np.zeros((ntheta, nphi, ns))
            sqrtg = np.zeros((ntheta, nphi, ns))

            theta1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
            phi1d = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
            phi, theta = np.meshgrid(phi1d, theta1d)

            sqrtg[:, :, 0] = 10.0
            sqrtg[:, :, 1] = -25.0

            modB[:, :, 0] = 13.0 + 2.6 * np.cos(theta)
            modB[:, :, 1] = 9.0 + 3.7 * np.sin(theta - nfp * phi)

            results = trapped_fraction(modB, sqrtg)
            # The average of (b0 + b1 cos(theta))^2 is b0^2 + (1/2) * b1^2
            np.testing.assert_allclose(
                results["<B**2>"],
                [13.0 ** 2 + 0.5 * 2.6 ** 2, 9.0 ** 2 + 0.5 * 3.7 ** 2]
            )
            np.testing.assert_allclose(
                results["<1/B>"],
                [1 / np.sqrt(13.0 ** 2 - 2.6 ** 2), 1 / np.sqrt(9.0 ** 2 - 3.7 ** 2)]
            )
            np.testing.assert_allclose(results["Bmin"], [13.0 - 2.6, 9.0 - 3.7], rtol=1e-4)
            np.testing.assert_allclose(results["Bmax"], [13.0 + 2.6, 9.0 + 3.7], rtol=1e-4)
            np.testing.assert_allclose(results["epsilon"], [2.6 / 13.0, 3.7 / 9.0], rtol=1e-3)
