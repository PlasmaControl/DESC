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
        nr = 2
        ntheta = 200
        nfp = 3
        results = []
        for nzeta in [1, 50]:
            modB = np.zeros((ntheta, nzeta, nr))
            sqrt_g = np.zeros((ntheta, nzeta, nr))

            theta1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
            zeta1d = np.linspace(0, 2 * np.pi / nfp, nzeta, endpoint=False)
            zeta, theta = np.meshgrid(zeta1d, theta1d)

            sqrt_g[:, :, 0] = 10.0
            sqrt_g[:, :, 1] = -25.0

            modB[:, :, 0] = 13.0 + 2.6 * np.cos(theta)
            modB[:, :, 1] = 9.0 + 3.7 * np.sin(theta - nfp * zeta)

            results = trapped_fraction(modB, sqrt_g)
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
            
    @pytest.mark.unit
    def test_trapped_fraction_Kim(self):
        """
        Compare the trapped fraction to eq (C18) in Kim, Diamond, &
        Groebner, Physics of Fluids B 3, 2050 (1991)
        """
        nr = 50
        ntheta = 100
        B0 = 7.5
        epsilon_in = np.linspace(0, 1, nr, endpoint=False)  # Avoid divide-by-0 when epsilon=1
        theta1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        nfp = 3
        for nzeta in [1, 13]:
            zeta1d = np.linspace(0, 2 * np.pi / nfp, nzeta, endpoint=False)
            zeta, theta = np.meshgrid(zeta1d, theta1d)
            modB = np.zeros((ntheta, nzeta, nr))
            sqrt_g = np.zeros((ntheta, nzeta, nr))
            for jr in range(nr):
                # Eq (A6)
                modB[:, :, jr] = B0 / (1 + epsilon_in[jr] * np.cos(theta))
                # For Jacobian, use eq (A7) for the theta dependence,
                # times an arbitrary overall scale factor
                sqrt_g[:, :, jr] = 6.7 * (1 + epsilon_in[jr] * np.cos(theta))

            results = trapped_fraction(modB, sqrt_g)

            f_t_Kim = 1.46 * np.sqrt(epsilon_in) - 0.46 * epsilon_in  # Eq (C18) in Kim et al

            np.testing.assert_allclose(results["Bmin"], B0 / (1 + epsilon_in))
            np.testing.assert_allclose(results["Bmax"], B0 / (1 - epsilon_in))
            np.testing.assert_allclose(epsilon_in, results["epsilon"])
            # Eq (A8):
            np.testing.assert_allclose(
                results["<B**2>"],
                B0 * B0 / np.sqrt(1 - epsilon_in ** 2),
                rtol=1e-6,
            )
            # Note the loose tolerance for this next test since we do not expect precise agreement.
            np.testing.assert_allclose(results["f_t"], f_t_Kim, rtol=0.1, atol=0.07)
            np.testing.assert_allclose(results["<1/B>"], (2 + epsilon_in ** 2) / (2 * B0))
