"""Tests for bootstrap current functions."""

import numpy as np
import pytest
from scipy.constants import elementary_charge
from scipy.integrate import quad

import desc.io
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid, QuadratureGrid
from desc.optimize import Optimizer
from desc.profiles import PowerSeriesProfile, SplineProfile
from desc.compute._bootstrap import trapped_fraction, j_dot_B_Redl
from desc.compute.utils import compress
from desc.objectives import *


class TestBootstrapCompute:
    """Tests for bootstrap current compute functions"""

    @pytest.mark.unit
    def test_trapped_fraction(self):
        """
        Confirm that the quantities computed by trapped_fraction()
        match analytic results for a model magnetic field.
        """
        M = 100
        NFP = 3
        for N in [0, 25]:
            grid = LinearGrid(
                rho=[0.5, 1],
                M=M,
                N=N,
                NFP=NFP,
            )
            theta = grid.nodes[:, 1]
            zeta = grid.nodes[:, 2]
            modB = np.zeros_like(theta)
            sqrt_g = np.zeros_like(theta)

            sqrt_g[grid.inverse_rho_idx == 0] = 10.0
            sqrt_g[grid.inverse_rho_idx == 1] = -25.0

            modB = 13.0 + 2.6 * np.cos(theta)
            modB_1 = 9.0 + 3.7 * np.sin(theta - NFP * zeta)
            mask = grid.inverse_rho_idx == 1
            modB[mask] = modB_1[mask]

            f_t_data = trapped_fraction(grid, modB, sqrt_g)
            # The average of (b0 + b1 cos(theta))^2 is b0^2 + (1/2) * b1^2
            np.testing.assert_allclose(
                f_t_data["<B**2>"],
                [13.0**2 + 0.5 * 2.6**2, 9.0**2 + 0.5 * 3.7**2],
            )
            np.testing.assert_allclose(
                f_t_data["<1/B>"],
                [1 / np.sqrt(13.0**2 - 2.6**2), 1 / np.sqrt(9.0**2 - 3.7**2)],
            )
            np.testing.assert_allclose(
                f_t_data["Bmin"], [13.0 - 2.6, 9.0 - 3.7], rtol=1e-4
            )
            np.testing.assert_allclose(
                f_t_data["Bmax"], [13.0 + 2.6, 9.0 + 3.7], rtol=1e-4
            )
            np.testing.assert_allclose(
                f_t_data["epsilon"], [2.6 / 13.0, 3.7 / 9.0], rtol=1e-3
            )

    @pytest.mark.unit
    def test_trapped_fraction_Kim(self):
        """
        Compare the trapped fraction to eq (C18) in Kim, Diamond, &
        Groebner, Physics of Fluids B 3, 2050 (1991), which gives
        a rough approximation for f_t.
        """
        L = 50
        M = 200
        B0 = 7.5

        # Maximum inverse aspect ratio to consider.
        # It is slightly less than 1 to avoid divide-by-0.
        epsilon_max = 0.96

        NFP = 3

        def test(N, grid_type):
            grid = grid_type(
                L=L,
                M=M,
                N=N,
                NFP=NFP,
            )
            rho = grid.nodes[:, 0]
            theta = grid.nodes[:, 1]
            epsilon_3D = rho * epsilon_max
            # Pick out unique values:
            epsilon = np.array(sorted(list(set(epsilon_3D))))

            # Eq (A6)
            modB = B0 / (1 + epsilon_3D * np.cos(theta))
            # For Jacobian, use eq (A7) for the theta dependence,
            # times an arbitrary overall scale factor
            sqrt_g = 6.7 * (1 + epsilon_3D * np.cos(theta))

            f_t_data = trapped_fraction(grid, modB, sqrt_g)

            # Eq (C18) in Kim et al:
            f_t_Kim = 1.46 * np.sqrt(epsilon) - 0.46 * epsilon

            np.testing.assert_allclose(f_t_data["Bmin"], B0 / (1 + epsilon))
            # Looser tolerance for Bmax since there is no grid point there:
            Bmax = B0 / (1 - epsilon)
            np.testing.assert_allclose(f_t_data["Bmax"], Bmax, rtol=0.001)
            np.testing.assert_allclose(epsilon, f_t_data["epsilon"], rtol=1e-4)
            # Eq (A8):
            fsa_B2 = B0 * B0 / np.sqrt(1 - epsilon**2)
            np.testing.assert_allclose(
                f_t_data["<B**2>"],
                fsa_B2,
                rtol=1e-6,
            )
            np.testing.assert_allclose(f_t_data["<1/B>"], (2 + epsilon**2) / (2 * B0))
            # Note the loose tolerance for this next test since we do not expect precise agreement.
            np.testing.assert_allclose(f_t_data["f_t"], f_t_Kim, rtol=0.1, atol=0.07)

            # Now compute f_t numerically by a different algorithm:
            modB = modB.reshape((grid.num_zeta, grid.num_rho, grid.num_theta))
            sqrt_g = sqrt_g.reshape((grid.num_zeta, grid.num_rho, grid.num_theta))
            fourpisq = 4 * np.pi * np.pi
            d_V_d_rho = np.mean(sqrt_g, axis=(0, 2)) / fourpisq
            f_t = np.zeros(grid.num_rho)
            for jr in range(grid.num_rho):

                def integrand(lambd):
                    # This function gives lambda / <sqrt(1 - lambda B)>:
                    return lambd / (
                        np.mean(np.sqrt(1 - lambd * modB[:, jr, :]) * sqrt_g[:, jr, :])
                        / (fourpisq * d_V_d_rho[jr])
                    )

                integral = quad(integrand, 0, 1 / Bmax[jr])
                f_t[jr] = 1 - 0.75 * fsa_B2[jr] * integral[0]
            np.testing.assert_allclose(
                f_t_data["f_t"][1:], f_t[1:], rtol=0.001, atol=0.001
            )

            # Use True in this "if" statement to plot extra info:
            if False:
                import matplotlib.pyplot as plt

                plt.plot(epsilon, f_t_Kim, label="Kim")
                plt.plot(epsilon, f_t_data["f_t"], label="desc")
                plt.plot(epsilon, f_t, ":", label="Alternative algorithm")
                plt.xlabel("epsilon")
                plt.ylabel("Effective trapped fraction $f_t$")
                plt.legend(loc=0)
                plt.show()

        for N in [0, 13]:
            for grid_type in [LinearGrid, QuadratureGrid]:
                test(N, grid_type)

    @pytest.mark.unit
    def test_Redl_second_pass(self):
        """
        A second pass through coding up the equations from Redl et al,
        Phys Plasmas (2021) to make sure I didn't make transcription
        errors.
        """
        # Make up some arbitrary functions to use for input:
        ne = PowerSeriesProfile(1.0e20 * np.array([1, -0.8]), modes=[0, 4])
        Te = PowerSeriesProfile(25e3 * np.array([1, -0.9]), modes=[0, 2])
        Ti = PowerSeriesProfile(20e3 * np.array([1, -0.9]), modes=[0, 2])
        Zeff = PowerSeriesProfile(np.array([1.5, 0.5]), modes=[0, 2])
        rho = np.linspace(0, 1, 20)
        rho = rho[1:]  # Avoid divide-by-0 on axis
        s = rho * rho
        NFP = 4
        helicity_n = 1
        helicity_N = NFP * helicity_n
        G = 32.0 - s
        iota = 0.95 - 0.7 * s
        R = 6.0 - 0.1 * s
        epsilon = 0.3 * rho
        f_t = 1.46 * np.sqrt(epsilon)
        psi_edge = 68 / (2 * np.pi)

        # Evaluate profiles on the rho grid:
        ne_rho = ne(rho)
        Te_rho = Te(rho)
        Ti_rho = Ti(rho)
        Zeff_s = Zeff(rho)
        ni_rho = ne_rho / Zeff_s
        d_ne_d_s = ne(rho, dr=1) / (2 * rho)
        d_Te_d_s = Te(rho, dr=1) / (2 * rho)
        d_Ti_d_s = Ti(rho, dr=1) / (2 * rho)

        # Sauter eq (18d)-(18e):
        ln_Lambda_e = 31.3 - np.log(np.sqrt(ne_rho) / Te_rho)
        ln_Lambda_ii = 30.0 - np.log((Zeff_s**3) * np.sqrt(ni_rho) / (Ti_rho**1.5))

        # Sauter eq (18b)-(18c):
        nu_e = abs(
            R
            * (6.921e-18)
            * ne_rho
            * Zeff_s
            * ln_Lambda_e
            / ((iota - helicity_N) * (Te_rho**2) * (epsilon**1.5))
        )
        nu_i = abs(
            R
            * (4.90e-18)
            * ni_rho
            * (Zeff_s**4)
            * ln_Lambda_ii
            / ((iota - helicity_N) * (Ti_rho**2) * (epsilon**1.5))
        )

        # Redl eq (11):
        X31 = f_t / (
            1
            + 0.67 * (1 - 0.7 * f_t) * np.sqrt(nu_e) / (0.56 + 0.44 * Zeff_s)
            + (0.52 + 0.086 * np.sqrt(nu_e))
            * (1 + 0.87 * f_t)
            * nu_e
            / (1 + 1.13 * np.sqrt(Zeff_s - 1))
        )

        # Redl eq (10):
        Zfac = Zeff_s**1.2 - 0.71
        L31 = (
            (1 + 0.15 / Zfac) * X31
            - 0.22 / Zfac * (X31**2)
            + 0.01 / Zfac * (X31**3)
            + 0.06 / Zfac * (X31**4)
        )

        # Redl eq (14):
        X32e = f_t / (
            1
            + 0.23 * (1 - 0.96 * f_t) * np.sqrt(nu_e / Zeff_s)
            + 0.13
            * (1 - 0.38 * f_t)
            * nu_e
            / (Zeff_s**2)
            * (
                np.sqrt(1 + 2 * np.sqrt(Zeff_s - 1))
                + f_t * f_t * np.sqrt((0.075 + 0.25 * ((Zeff_s - 1) ** 2)) * nu_e)
            )
        )

        # Redl eq (13):
        F32ee = (
            (0.1 + 0.6 * Zeff_s)
            / (Zeff_s * (0.77 + 0.63 * (1 + (Zeff_s - 1) ** 1.1)))
            * (X32e - X32e**4)
            + 0.7
            / (1 + 0.2 * Zeff_s)
            * (X32e**2 - X32e**4 - 1.2 * (X32e**3 - X32e**4))
            + 1.3 / (1 + 0.5 * Zeff_s) * (X32e**4)
        )

        # Redl eq (16)
        X32ei = f_t / (
            1
            + 0.87 * (1 + 0.39 * f_t) * np.sqrt(nu_e) / (1 + 2.95 * ((Zeff_s - 1) ** 2))
            + 1.53 * (1 - 0.37 * f_t) * nu_e * (2 + 0.375 * (Zeff_s - 1))
        )

        # Redl eq (15)
        F32ei = (
            -(0.4 + 1.93 * Zeff_s)
            / (Zeff_s * (0.8 + 0.6 * Zeff_s))
            * (X32ei - X32ei**4)
            + 5.5
            / (1.5 + 2 * Zeff_s)
            * (X32ei**2 - X32ei**4 - 0.8 * (X32ei**3 - X32ei**4))
            - 1.3 / (1 + 0.5 * Zeff_s) * (X32ei**4)
        )

        # Redl eq (12)
        L32 = F32ei + F32ee

        # Redl eq (20):
        alpha0 = (
            -(0.62 + 0.055 * (Zeff_s - 1))
            * (1 - f_t)
            / (
                (0.53 + 0.17 * (Zeff_s - 1))
                * (1 - (0.31 - 0.065 * (Zeff_s - 1)) * f_t - 0.25 * (f_t**2))
            )
        )

        # Redl eq (21):
        alpha = (
            (alpha0 + 0.7 * Zeff_s * np.sqrt(f_t * nu_i)) / (1 + 0.18 * np.sqrt(nu_i))
            - 0.002 * (nu_i**2) * (f_t**6)
        ) / (1 + 0.004 * (nu_i**2) * (f_t**6))

        # Redl eq (19):
        L34 = L31

        Te_J = Te_rho * elementary_charge
        Ti_J = Ti_rho * elementary_charge
        d_Te_d_s_J = d_Te_d_s * elementary_charge
        d_Ti_d_s_J = d_Ti_d_s * elementary_charge
        pe = ne_rho * Te_J
        pi = ni_rho * Ti_J
        p = pe + pi
        Rpe = pe / p
        d_ni_d_s = d_ne_d_s / Zeff_s
        d_p_d_s = (
            ne_rho * d_Te_d_s_J
            + Te_J * d_ne_d_s
            + ni_rho * d_Ti_d_s_J
            + Ti_J * d_ni_d_s
        )
        # Equation from the bottom of the right column of Sauter errata:
        factors = -G * pe / (psi_edge * (iota - helicity_N))
        dnds_term = factors * L31 * p / pe * ((Te_J * d_ne_d_s + Ti_J * d_ni_d_s) / p)
        dTeds_term = factors * (
            L31 * p / pe * (ne_rho * d_Te_d_s_J) / p + L32 * (d_Te_d_s_J / Te_J)
        )
        dTids_term = factors * (
            L31 * p / pe * (ni_rho * d_Ti_d_s_J) / p
            + L34 * alpha * (1 - Rpe) / Rpe * (d_Ti_d_s_J / Ti_J)
        )
        J_dot_B_pass2 = (
            -G
            * pe
            * (
                L31 * p / pe * (d_p_d_s / p)
                + L32 * (d_Te_d_s_J / Te_J)
                + L34 * alpha * (1 - Rpe) / Rpe * (d_Ti_d_s_J / Ti_J)
            )
            / (psi_edge * (iota - helicity_N))
        )

        geom_data = {
            "rho": rho,
            "G": G,
            "R": R,
            "iota": iota,
            "epsilon": epsilon,
            "f_t": f_t,
            "psi_edge": psi_edge,
        }
        J_dot_B_data = j_dot_B_Redl(geom_data, ne, Te, Ti, Zeff, helicity_N)

        atol = 1e-13
        rtol = 1e-13
        np.testing.assert_allclose(
            J_dot_B_data["nu_e_star"], nu_e, atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            J_dot_B_data["nu_i_star"], nu_i, atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(J_dot_B_data["X31"], X31, atol=atol, rtol=rtol)
        np.testing.assert_allclose(J_dot_B_data["L31"], L31, atol=atol, rtol=rtol)
        np.testing.assert_allclose(J_dot_B_data["X32e"], X32e, atol=atol, rtol=rtol)
        np.testing.assert_allclose(J_dot_B_data["F32ee"], F32ee, atol=atol, rtol=rtol)
        np.testing.assert_allclose(J_dot_B_data["X32ei"], X32ei, atol=atol, rtol=rtol)
        np.testing.assert_allclose(J_dot_B_data["F32ei"], F32ei, atol=atol, rtol=rtol)
        np.testing.assert_allclose(J_dot_B_data["L32"], L32, atol=atol, rtol=rtol)
        np.testing.assert_allclose(J_dot_B_data["alpha"], alpha, atol=atol, rtol=rtol)
        np.testing.assert_allclose(
            J_dot_B_data["dTeds_term"], dTeds_term, atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            J_dot_B_data["dTids_term"], dTids_term, atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            J_dot_B_data["dnds_term"], dnds_term, atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            J_dot_B_data["<J*B>"], J_dot_B_pass2, atol=atol, rtol=rtol
        )

    @pytest.mark.unit
    def test_Redl_figures_2_3(self):
        """
        Make sure the implementation here can roughly recover the plots
        from figures 2 and 3 in the Redl paper.
        """
        for Zeff in [1, 1.8]:
            target_nu_e_star = 4.0e-5
            target_nu_i_star = 1.0e-5
            # Make up some profiles
            ne = PowerSeriesProfile([1.0e17])
            Te = PowerSeriesProfile([1.0e5])
            Ti_over_Te = np.sqrt(
                4.9 * Zeff * Zeff * target_nu_e_star / (6.921 * target_nu_i_star)
            )
            Ti = PowerSeriesProfile([1.0e5 * Ti_over_Te])
            rho = np.linspace(0, 1, 100) ** 2
            rho = rho[1:]  # Avoid divide-by-0 on axis
            NFP = 1
            helicity_N = 0
            epsilon = rho
            f_t = 1.46 * np.sqrt(epsilon) - 0.46 * epsilon
            psi_edge = 68 / (2 * np.pi)
            G = 32.0 - rho * rho  # Doesn't matter
            R = 5.0 + 0.1 * rho * rho  # Doesn't matter
            # Redl uses fixed values of nu_e*. To match this, I'll use
            # a contrived iota profile that is chosen just to give the
            # desired nu_*.

            ne_rho = ne(rho)
            Te_rho = Te(rho)
            Zeff_rho = Zeff

            # Sauter eq (18d):
            ln_Lambda_e = 31.3 - np.log(np.sqrt(ne_rho) / Te_rho)

            # Sauter eq (18b), but without the iota factor:
            nu_e_without_iota = (
                R
                * (6.921e-18)
                * ne_rho
                * Zeff_rho
                * ln_Lambda_e
                / ((Te_rho**2) * (epsilon**1.5))
            )

            iota = nu_e_without_iota / target_nu_e_star
            # End of determining the qR profile that gives the desired nu*.

            geom_data = {
                "rho": rho,
                "G": G,
                "R": R,
                "iota": iota,
                "epsilon": epsilon,
                "f_t": f_t,
                "psi_edge": psi_edge,
            }
            J_dot_B_data = j_dot_B_Redl(geom_data, ne, Te, Ti, Zeff, helicity_N)

            # Change False to True in the next line to plot the data for debugging.
            if False:
                # Make a plot, matching the axis ranges of Redl's
                # figures 2 and 3 as best as possible.
                import matplotlib.pyplot as plt

                plt.figure(figsize=(6.5, 5.5))
                nrows = 2
                ncols = 2
                xlim = [-0.05, 1.05]

                plt.subplot(nrows, ncols, 1)
                plt.semilogy(f_t, J_dot_B_data["nu_e_star"], label=r"$\nu_{e*}$")
                plt.semilogy(f_t, J_dot_B_data["nu_i_star"], label=r"$\nu_{i*}$")
                plt.xlabel("f_t")
                plt.legend(loc=0, fontsize=8)

                plt.subplot(nrows, ncols, 2)
                plt.plot(f_t, J_dot_B_data["L31"], "g")
                plt.title("L31")
                plt.xlabel("f_t")
                plt.xlim(xlim)
                plt.ylim(-0.05, 1.05)

                plt.subplot(nrows, ncols, 3)
                plt.plot(f_t, J_dot_B_data["L32"], "g")
                plt.title("L32")
                plt.xlabel("f_t")
                plt.xlim(xlim)
                if Zeff == 1:
                    plt.ylim(-0.25, 0.025)
                else:
                    plt.ylim(-0.2, 0.05)

                plt.subplot(nrows, ncols, 4)
                plt.plot(f_t, J_dot_B_data["alpha"], "g")
                plt.title("alpha")
                plt.xlabel("f_t")
                plt.xlim(xlim)
                plt.ylim(-1.25, 0.04)

                plt.tight_layout()
                plt.show()

            # Make sure L31, L32, and alpha are within the right range:
            np.testing.assert_array_less(J_dot_B_data["L31"], 1.05)
            np.testing.assert_array_less(0, J_dot_B_data["L31"])
            np.testing.assert_array_less(J_dot_B_data["L32"], 0.01)
            np.testing.assert_array_less(-0.25, J_dot_B_data["L32"])
            np.testing.assert_array_less(J_dot_B_data["alpha"], 0.05)
            np.testing.assert_array_less(-1.2, J_dot_B_data["alpha"])
            if Zeff > 1:
                np.testing.assert_array_less(-1.0, J_dot_B_data["alpha"])
            assert J_dot_B_data["L31"][0] < 0.1
            assert J_dot_B_data["L31"][-1] > 0.9
            assert J_dot_B_data["L32"][0] > -0.05
            assert J_dot_B_data["L32"][-1] > -0.05
            if Zeff == 0:
                assert np.min(J_dot_B_data["L32"]) < -0.2
                assert J_dot_B_data["alpha"][0] < -1.05
            else:
                assert np.min(J_dot_B_data["L32"]) < -0.13
                assert J_dot_B_data["alpha"][0] < -0.9
            assert J_dot_B_data["alpha"][-1] > -0.1

    @pytest.mark.unit
    def test_Redl_figures_4_5(self):
        """
        Make sure the implementation here can roughly recover the plots
        from figures 4 and 5 in the Redl paper.
        """
        for Zeff in [1, 1.8]:
            n_nu_star = 30
            n_f_t = 3
            target_nu_stars = 10.0 ** np.linspace(-4, 4, n_nu_star)
            f_ts = np.array([0.24, 0.45, 0.63])
            L31s = np.zeros((n_nu_star, n_f_t))
            L32s = np.zeros((n_nu_star, n_f_t))
            alphas = np.zeros((n_nu_star, n_f_t))
            nu_e_stars = np.zeros((n_nu_star, n_f_t))
            nu_i_stars = np.zeros((n_nu_star, n_f_t))
            for j_nu_star, target_nu_star in enumerate(target_nu_stars):
                target_nu_e_star = target_nu_star
                target_nu_i_star = target_nu_star
                # Make up some profiles
                ne = PowerSeriesProfile([1.0e17], modes=[0])
                Te = PowerSeriesProfile([1.0e5], modes=[0])
                Ti_over_Te = np.sqrt(
                    4.9 * Zeff * Zeff * target_nu_e_star / (6.921 * target_nu_i_star)
                )
                Ti = PowerSeriesProfile([1.0e5 * Ti_over_Te], modes=[0])
                rho = np.ones(n_f_t)
                NFP = 0
                helicity_N = 0
                G = 32.0 - rho * rho  # Doesn't matter
                R = 5.0 + 0.1 * rho * rho  # Doesn't matter
                epsilon = rho * rho  # Doesn't matter
                f_t = f_ts
                psi_edge = 68 / (2 * np.pi)
                # Redl uses fixed values of nu_e*. To match this, I'll use
                # a contrived iota profile that is chosen just to give the
                # desired nu_*.

                ne_rho = ne(rho)
                Te_rho = Te(rho)
                Zeff_rho = Zeff

                # Sauter eq (18d):
                ln_Lambda_e = 31.3 - np.log(np.sqrt(ne_rho) / Te_rho)

                # Sauter eq (18b), but without the q = 1/iota factor:
                nu_e_without_iota = (
                    R
                    * (6.921e-18)
                    * ne_rho
                    * Zeff_rho
                    * ln_Lambda_e
                    / ((Te_rho**2) * (epsilon**1.5))
                )

                iota = nu_e_without_iota / target_nu_e_star
                # End of determining the qR profile that gives the desired nu*.

                geom_data = {
                    "rho": rho,
                    "G": G,
                    "R": R,
                    "iota": iota,
                    "epsilon": epsilon,
                    "f_t": f_t,
                    "psi_edge": psi_edge,
                }
                J_dot_B_data = j_dot_B_Redl(geom_data, ne, Te, Ti, Zeff, helicity_N)

                L31s[j_nu_star, :] = J_dot_B_data["L31"]
                L32s[j_nu_star, :] = J_dot_B_data["L32"]
                alphas[j_nu_star, :] = J_dot_B_data["alpha"]
                nu_e_stars[j_nu_star, :] = J_dot_B_data["nu_e_star"]
                nu_i_stars[j_nu_star, :] = J_dot_B_data["nu_i_star"]
                # np.testing.assert_allclose(J_dot_B_data["nu_e_star"], target_nu_e_star)
                # np.testing.assert_allclose(J_dot_B_data["nu_i_star"], target_nu_i_star)

            # Change False to True in the next line to plot the data for debugging.
            if False:
                # Make a plot, matching the axis ranges of Redl's
                # figures 4 and 5 as best as possible.
                import matplotlib.pyplot as plt

                plt.figure(figsize=(6.5, 5.5))
                nrows = 2
                ncols = 2
                xlim = [3.0e-5, 1.5e4]

                plt.subplot(nrows, ncols, 2)
                for j in range(n_f_t):
                    plt.semilogx(nu_e_stars[:, j], L31s[:, j], label=f"f_t={f_ts[j]}")
                plt.legend(loc=0, fontsize=8)
                plt.title(f"L31, Zeff={Zeff}")
                plt.xlabel("nu_{*e}")
                plt.xlim(xlim)
                if Zeff == 1:
                    plt.ylim(-0.05, 0.85)
                else:
                    plt.ylim(-0.05, 0.75)

                plt.subplot(nrows, ncols, 3)
                for j in range(n_f_t):
                    plt.semilogx(nu_e_stars[:, j], L32s[:, j], label=f"f_t={f_ts[j]}")
                plt.legend(loc=0, fontsize=8)
                plt.title(f"L32, Zeff={Zeff}")
                plt.xlabel("nu_{*e}")
                plt.xlim(xlim)
                if Zeff == 1:
                    plt.ylim(-0.26, 0.21)
                else:
                    plt.ylim(-0.18, 0.2)

                plt.subplot(nrows, ncols, 4)
                for j in range(n_f_t):
                    plt.semilogx(nu_i_stars[:, j], alphas[:, j], label=f"f_t={f_ts[j]}")
                plt.legend(loc=0, fontsize=8)
                plt.title(f"alpha, Zeff={Zeff}")
                plt.xlabel("nu_{*i}")
                plt.xlim(xlim)
                if Zeff == 1:
                    plt.ylim(-1.1, 2.2)
                else:
                    plt.ylim(-1.1, 2.35)

                plt.tight_layout()
                plt.show()

            # Make sure L31, L32, and alpha are within the right range:
            if Zeff == 1:
                np.testing.assert_array_less(L31s, 0.71)
                np.testing.assert_array_less(0, L31s)
                np.testing.assert_array_less(L31s[-1, :], 1.0e-5)
                np.testing.assert_array_less(L32s, 0.2)
                np.testing.assert_array_less(-0.23, L32s)
                np.testing.assert_array_less(L32s[-1, :], 3.0e-5)
                np.testing.assert_array_less(-3.0e-5, L32s[-1, :])
                np.testing.assert_array_less(L32s[0, :], -0.17)
                np.testing.assert_array_less(alphas, 1.2)
                np.testing.assert_array_less(alphas[0, :], -0.58)
                np.testing.assert_array_less(-1.05, alphas)
                np.testing.assert_array_less(0.8, np.max(alphas, axis=0))
                np.testing.assert_array_less(L31s[:, 0], 0.33)
                assert L31s[0, 0] > 0.3
                np.testing.assert_array_less(L31s[0, 1], 0.55)
                assert L31s[0, 1] > 0.51
                np.testing.assert_array_less(L31s[0, 2], 0.7)
                assert L31s[0, 2] > 0.68
            else:
                np.testing.assert_array_less(L31s, 0.66)
                np.testing.assert_array_less(0, L31s)
                np.testing.assert_array_less(L31s[-1, :], 1.5e-5)
                np.testing.assert_array_less(L32s, 0.19)
                np.testing.assert_array_less(-0.15, L32s)
                np.testing.assert_array_less(L32s[-1, :], 5.0e-5)
                np.testing.assert_array_less(0, L32s[-1, :])
                np.testing.assert_array_less(L32s[0, :], -0.11)
                np.testing.assert_array_less(alphas, 2.3)
                np.testing.assert_array_less(alphas[0, :], -0.4)
                np.testing.assert_array_less(-0.9, alphas)
                np.testing.assert_array_less(1.8, np.max(alphas, axis=0))
                np.testing.assert_array_less(L31s[:, 0], 0.27)
                assert L31s[0, 0] > 0.24
                np.testing.assert_array_less(L31s[0, 1], 0.49)
                assert L31s[0, 1] > 0.45
                np.testing.assert_array_less(L31s[0, 2], 0.66)
                assert L31s[0, 2] > 0.63

    @pytest.mark.unit
    def test_Redl_sfincs_tokamak_benchmark(self):
        """
        Compare the Redl <J*B> to a SFINCS calculation for a
        model tokamak.

        The SFINCS calculation is on Matt Landreman's laptop in
        /Users/mattland/Box/work21/20211225-01-sfincs_tokamak_bootstrap_for_Redl_benchmark
        """
        ne = PowerSeriesProfile(5.0e20 * np.array([1, -1]), modes=[0, 8])
        Te = PowerSeriesProfile(8e3 * np.array([1, -1]), modes=[0, 2])
        Ti = Te
        s_surfaces = np.linspace(0.01, 0.99, 99)
        rho = np.sqrt(s_surfaces)
        helicity = (1, 0)
        filename = ".//tests//inputs//circular_model_tokamak_output.h5"
        eq = desc.io.load(filename)[-1]
        J_dot_B_sfincs = np.array(
            [
                -577720.30718026,
                -737097.14851563,
                -841877.1731213,
                -924690.37927967,
                -996421.14965534,
                -1060853.54247997,
                -1120000.15051496,
                -1175469.30096585,
                -1228274.42232883,
                -1279134.94084881,
                -1328502.74017954,
                -1376746.08281939,
                -1424225.7135264,
                -1471245.54499716,
                -1518022.59582135,
                -1564716.93168823,
                -1611473.13548435,
                -1658436.14166984,
                -1705743.3966606,
                -1753516.75354018,
                -1801854.51072685,
                -1850839.64964612,
                -1900546.86009713,
                -1951047.40424607,
                -2002407.94774638,
                -2054678.30773555,
                -2107880.19135161,
                -2162057.48184046,
                -2217275.94462326,
                -2273566.0131982,
                -2330938.65226651,
                -2389399.44803491,
                -2448949.45267694,
                -2509583.82212581,
                -2571290.69542303,
                -2634050.8642164,
                -2697839.22372799,
                -2762799.43321187,
                -2828566.29269343,
                -2895246.32116721,
                -2962784.4499046,
                -3031117.70888815,
                -3100173.19345621,
                -3169866.34773162,
                -3240095.93569359,
                -3310761.89170199,
                -3381738.85511963,
                -3452893.53199984,
                -3524079.68661978,
                -3595137.36934266,
                -3665892.0942594,
                -3736154.01439094,
                -3805717.10211429,
                -3874383.57672975,
                -3941857.83476556,
                -4007909.78112076,
                -4072258.58610167,
                -4134609.67635966,
                -4194641.53357309,
                -4252031.55214378,
                -4306378.23970987,
                -4357339.70206557,
                -4404503.4788238,
                -4447364.62100875,
                -4485559.83633318,
                -4518524.39965094,
                -4545649.39588513,
                -4566517.96382113,
                -4580487.82371991,
                -4586917.13595789,
                -4585334.66419017,
                -4574935.10788554,
                -4555027.85929442,
                -4524904.81212564,
                -4483819.87563906,
                -4429820.99499252,
                -4364460.04545626,
                -4285813.80804979,
                -4193096.44129549,
                -4085521.92933703,
                -3962389.92116629,
                -3822979.23919869,
                -3666751.38186485,
                -3493212.37971975,
                -3302099.71461769,
                -3093392.43317121,
                -2867475.54470152,
                -2625108.03673121,
                -2367586.06128219,
                -2096921.32817857,
                -1815701.10075496,
                -1527523.11762782,
                -1237077.39816553,
                -950609.3080458,
                -677002.74349353,
                -429060.85924996,
                -224317.60933134,
                -82733.32462396,
                -22233.12804732,
            ]
        )

        grid = LinearGrid(rho=rho, M=eq.M, N=eq.N, NFP=eq.NFP)
        data = eq.compute(
            "<J*B> Redl",
            grid=grid,
            helicity=helicity,
            ne=ne,
            Te=Te,
            Ti=Ti,
            Zeff=1,
        )
        J_dot_B_Redl = compress(grid, data["<J*B> Redl"])

        # Use True in this "if" statement to plot the bootstrap
        # current profiles:
        if False:
            import matplotlib.pyplot as plt

            plt.plot(rho, J_dot_B_Redl, "+-", label="Redl")
            plt.plot(rho, J_dot_B_sfincs, ".-", label="sfincs")
            plt.xlabel("rho")
            plt.title("<J*B> [T A / m^2]")
            plt.legend(loc=0)
            plt.show()

        # The relative error is a bit larger at the boundary, where the
        # absolute magnitude is quite small, so drop those points.
        np.testing.assert_allclose(J_dot_B_Redl[:-5], J_dot_B_sfincs[:-5], rtol=0.1)

    @pytest.mark.unit
    def test_Redl_sfincs_QA(self):
        """
        Compare the Redl <J*B> to a SFINCS calculation for a
        reactor-scale quasi-axisymmetric configuration.

        This test reproduces figure 1.a of Landreman Buller
        Drevlak, Physics of Plasmas 29, 082501 (2022)
        https://doi.org/10.1063/5.0098166

        The SFINCS calculation is on the IPP Cobra machine in
        /ptmp/mlan/20211226-01-sfincs_for_precise_QS_for_Redl_benchmark/20211226-01-012_QA_Ntheta25_Nzeta39_Nxi60_Nx7_manySurfaces
        """
        ne = PowerSeriesProfile(4.13e20 * np.array([1, -1]), modes=[0, 10])
        Te = PowerSeriesProfile(12.0e3 * np.array([1, -1]), modes=[0, 2])
        Ti = Te
        Zeff = 1
        helicity = (1, 0)
        filename = ".//tests//inputs//LandremanPaul2022_QA_reactorScale_lowRes.h5"
        eq = desc.io.load(filename)
        s_surfaces = np.linspace(0.025, 0.975, 39)
        rho = np.sqrt(s_surfaces)
        J_dot_B_sfincs = np.array(
            [
                -2164875.78234086,
                -3010997.004258,
                -3586912.40439179,
                -4025873.78974165,
                -4384855.40656673,
                -4692191.91608418,
                -4964099.33007648,
                -5210508.61474677,
                -5442946.68999908,
                -5657799.82786579,
                -5856450.57370037,
                -6055808.19817868,
                -6247562.80014873,
                -6431841.43078959,
                -6615361.81912527,
                -6793994.01503932,
                -6964965.34953497,
                -7127267.47873969,
                -7276777.92844458,
                -7409074.62499181,
                -7518722.07866914,
                -7599581.37772525,
                -7644509.67670812,
                -7645760.36382036,
                -7594037.38147436,
                -7481588.70786642,
                -7299166.08742784,
                -7038404.20002745,
                -6691596.45173419,
                -6253955.52847633,
                -5722419.58059673,
                -5098474.47777983,
                -4390147.20699043,
                -3612989.71633149,
                -2793173.34162084,
                -1967138.17518374,
                -1192903.42248978,
                -539990.088677,
                -115053.37380415,
            ]
        )

        grid = LinearGrid(rho=rho, M=eq.M, N=eq.N, NFP=eq.NFP)
        data = eq.compute(
            "<J*B> Redl",
            grid=grid,
            helicity=helicity,
            ne=ne,
            Te=Te,
            Ti=Ti,
            Zeff=1,
        )
        J_dot_B_Redl = compress(grid, data["<J*B> Redl"])

        # Use True in this "if" statement to plot the bootstrap
        # current profiles, reproducing figure 1.a of Landreman Buller
        # Drevlak, Physics of Plasmas 29, 082501 (2022)
        # https://doi.org/10.1063/5.0098166
        if False:
            import matplotlib.pyplot as plt

            plt.plot(rho**2, J_dot_B_Redl, "+-", label="Redl")
            plt.plot(rho**2, J_dot_B_sfincs, ".-", label="sfincs")
            plt.legend(loc=0)
            plt.xlabel(r"$\rho^2 = s$")
            plt.ylabel("<J*B> [T A / m^2]")
            plt.xlim([0, 1])
            plt.show()

        np.testing.assert_allclose(J_dot_B_Redl[1:-1], J_dot_B_sfincs[1:-1], rtol=0.1)

    @pytest.mark.unit
    def test_Redl_sfincs_QH(self):
        """
        Compare the Redl <J*B> to a SFINCS calculation for a
        reactor-scale quasi-helically symmetric configuration.

        This test reproduces figure 1.b of Landreman Buller
        Drevlak, Physics of Plasmas 29, 082501 (2022)
        https://doi.org/10.1063/5.0098166

        The SFINCS calculation is on the IPP Cobra machine in
        /ptmp/mlan/20211226-01-sfincs_for_precise_QS_for_Redl_benchmark/20211226-01-019_QH_Ntheta25_Nzeta39_Nxi\
60_Nx7_manySurfaces
        """
        ne = PowerSeriesProfile(4.13e20 * np.array([1, -1]), modes=[0, 10])
        Te = PowerSeriesProfile(12.0e3 * np.array([1, -1]), modes=[0, 2])
        Ti = Te
        Zeff = 1
        helicity = (1, -4)
        filename = ".//tests//inputs//LandremanPaul2022_QH_reactorScale_lowRes.h5"
        eq = desc.io.load(filename)
        s_surfaces = np.linspace(0.025, 0.975, 39)
        rho = np.sqrt(s_surfaces)
        J_dot_B_sfincs = np.array(
            [
                -1086092.9561775,
                -1327299.73501589,
                -1490400.04894085,
                -1626634.32037339,
                -1736643.64671843,
                -1836285.33939607,
                -1935027.3099312,
                -2024949.13178129,
                -2112581.50178861,
                -2200196.92359437,
                -2289400.72956248,
                -2381072.32897262,
                -2476829.87345286,
                -2575019.97938908,
                -2677288.45525839,
                -2783750.09013764,
                -2894174.68898196,
                -3007944.74771214,
                -3123697.37793226,
                -3240571.57445779,
                -3356384.98579004,
                -3468756.64908024,
                -3574785.02500657,
                -3671007.37469685,
                -3753155.07811322,
                -3816354.48636373,
                -3856198.2242986,
                -3866041.76391937,
                -3839795.40512069,
                -3770065.26594065,
                -3649660.76253605,
                -3471383.501417,
                -3228174.23182819,
                -2914278.54799143,
                -2525391.54652021,
                -2058913.26485519,
                -1516843.60879267,
                -912123.395174,
                -315980.89711036,
            ]
        )

        grid = LinearGrid(rho=rho, M=eq.M, N=eq.N, NFP=eq.NFP)
        data = eq.compute(
            "<J*B> Redl",
            grid=grid,
            helicity=helicity,
            ne=ne,
            Te=Te,
            Ti=Ti,
            Zeff=1,
        )
        J_dot_B_Redl = compress(grid, data["<J*B> Redl"])

        # Use True in this "if" statement to plot the bootstrap
        # current profiles, reproducing figure 1.b of Landreman Buller
        # Drevlak, Physics of Plasmas 29, 082501 (2022)
        # https://doi.org/10.1063/5.0098166
        if False:
            import matplotlib.pyplot as plt

            plt.plot(rho**2, J_dot_B_Redl, "+-", label="Redl")
            plt.plot(rho**2, J_dot_B_sfincs, ".-", label="sfincs")
            plt.legend(loc=0)
            plt.xlabel(r"$\rho^2 = s$")
            plt.ylabel("<J*B> [T A / m^2]")
            plt.xlim([0, 1])
            plt.show()

        np.testing.assert_allclose(J_dot_B_Redl[1:-1], J_dot_B_sfincs[1:-1], rtol=0.1)


class TestBootstrapObjectives:
    """Tests for bootstrap current objective functions"""

    @pytest.mark.unit
    def test_BootstrapRedlConsistency_12(self):
        """
        The BootstrapRedlConsistency objective function should be
        1/2 if the equilibrium is a vacuum field.
        """
        # n and T profiles do not go to 0 at rho=0 to avoid divide-by-0 there:
        ne = PowerSeriesProfile(5.0e20 * np.array([1, -0.9]), modes=[0, 8])
        Te = PowerSeriesProfile(8e3 * np.array([1, -0.9]), modes=[0, 2])
        Ti = PowerSeriesProfile(9e3 * np.array([1, -0.9]), modes=[0, 2])
        Zeff = PowerSeriesProfile(np.array([1.2, 0.2]), modes=[0, 2])
        helicity = (1, -4)
        filename = ".//tests//inputs//LandremanPaul2022_QH_reactorScale_lowRes.h5"
        eq = desc.io.load(filename)

        def test(grid_type, kwargs, L_factor, M_factor, N_factor, rho_exponent):
            grid = grid_type(
                L=eq.L * L_factor,
                M=eq.M * M_factor,
                N=eq.N * N_factor,
                NFP=eq.NFP,
                **kwargs,
            )
            obj = ObjectiveFunction(
                BootstrapRedlConsistency(
                    grid=grid,
                    helicity=helicity,
                    ne=ne,
                    Te=Te,
                    Ti=Ti,
                    Zeff=Zeff,
                    rho_exponent=rho_exponent,
                ),
                eq,
            )
            scalar_objective = obj.compute_scalar(obj.x(eq))
            print("Should be approximately 0.5:", scalar_objective)
            return scalar_objective

        results = []

        # Loop over grid trypes. For LinearGrid we need to drop the point at rho=0 to avoid a divide-by-0
        grid_types = [LinearGrid, QuadratureGrid]
        kwargs = [{"axis": False}, {}]

        # Loop over grid resolutions:
        L_factors = [1, 2, 1, 1, 1]
        M_factors = [1, 1, 2, 1, 1]
        N_factors = [1, 1, 1, 2, 1]
        rho_exponents = [0, 1, 1, 1, 2]

        for grid_type, kwargs in zip(grid_types, kwargs):
            for L_factor, M_factor, N_factor, rho_exponent in zip(
                L_factors, M_factors, N_factors, rho_exponents
            ):
                results.append(
                    test(grid_type, kwargs, L_factor, M_factor, N_factor, rho_exponent)
                )

        np.testing.assert_allclose(np.array(results), 0.5, rtol=2e-4)

    @pytest.mark.unit
    def test_BootstrapRedlConsistency_resolution(self):
        """
        Confirm that the BootstrapRedlConsistency objective function is
        approximately independent of grid resolution.
        """
        ne = PowerSeriesProfile(2.0e19 * np.array([1, -0.85]), modes=[0, 4])
        Te = PowerSeriesProfile(1e3 * np.array([1.02, -3, 3, -1]), modes=[0, 2, 4, 6])
        Ti = PowerSeriesProfile(1.1e3 * np.array([1.02, -3, 3, -1]), modes=[0, 2, 4, 6])
        Zeff = 1.4
        helicity = (1, 0)
        filename = ".//tests//inputs//DSHAPE_output_saved_without_current.h5"
        eq = desc.io.load(filename)[-1]

        # Set to True to plot the profiles of <J*B>
        if False:
            grid = LinearGrid(L=eq.L, M=eq.M, N=0, axis=False)
            data = eq.compute(
                ["<J*B>", "<J*B> Redl", "rho"],
                grid=grid,
                ne=ne,
                Te=Te,
                Ti=Ti,
                Zeff=Zeff,
                helicity=helicity,
            )
            import matplotlib.pyplot as plt

            plt.plot(data["rho"], data["<J*B>"], label="MHD")
            plt.plot(data["rho"], data["<J*B> Redl"], label="Redl")
            plt.xlabel("rho")
            plt.title("<J*B>")
            plt.legend(loc=0)
            plt.show()

        def test(grid_type, kwargs, L, M, N):
            grid = grid_type(
                L=L,
                M=M,
                N=N,
                NFP=eq.NFP,
                **kwargs,
            )
            obj = ObjectiveFunction(
                BootstrapRedlConsistency(
                    grid=grid,
                    helicity=helicity,
                    ne=ne,
                    Te=Te,
                    Ti=Ti,
                    Zeff=Zeff,
                    rho_exponent=1,
                ),
                eq,
            )
            scalar_objective = obj.compute_scalar(obj.x(eq))
            return scalar_objective

        results = []

        # Loop over grid trypes. For LinearGrid we need to drop the point at rho=0 to avoid a divide-by-0
        grid_types = [LinearGrid, QuadratureGrid]
        kwargss = [{"axis": False}, {}]

        # Loop over grid resolutions:
        Ls = [10, 20, 10, 10]
        Ms = [10, 10, 20, 10]
        Ns = [0, 0, 0, 2]

        for grid_type, kwargs in zip(grid_types, kwargss):
            for L, M, N in zip(Ls, Ms, Ns):
                results.append(test(grid_type, kwargs, L, M, N))

        results = np.array(results)
        np.testing.assert_allclose(results, np.mean(results), rtol=0.02)

    @pytest.mark.unit
    @pytest.mark.slow
    def test_bootstrap_consistency_iota(self):
        """Try optimizing for bootstrap consistency in axisymmetry, at fixed boundary shape.

        This version of the test covers an equilibrium with an iota profile.
        """

        ne0 = 4.0e20
        T0 = 12.0e3
        ne = PowerSeriesProfile(ne0 * np.array([1, -1]), modes=[0, 10])
        Te = PowerSeriesProfile(T0 * np.array([1, -1]), modes=[0, 2])
        Ti = Te
        Zeff = 1
        helicity = (1, 0)
        pressure = PowerSeriesProfile(
            2 * ne0 * T0 * elementary_charge * np.array([1, -1, 0, 0, 0, -1, 1]),
            modes=[0, 2, 4, 6, 8, 10, 12],
        )
        B0 = 5.0  # Mean |B|
        LM_resolution = 8

        initial_iota = 0.61
        num_iota_points = 21
        iota = SplineProfile(np.full(num_iota_points, initial_iota))

        # Set initial condition:
        Rmajor = 6.0
        aminor = 2.0
        NFP = 1
        surface = FourierRZToroidalSurface(
            R_lmn=np.array([Rmajor, aminor]),
            modes_R=[[0, 0], [1, 0]],
            Z_lmn=np.array([aminor]),
            modes_Z=[[-1, 0]],
            NFP=NFP,
        )

        eq = Equilibrium(
            surface=surface,
            pressure=pressure,
            iota=iota,
            Psi=B0 * np.pi * (aminor**2),
            NFP=NFP,
            L=LM_resolution,
            M=LM_resolution,
            N=0,
            L_grid=2 * LM_resolution,
            M_grid=2 * LM_resolution,
            N_grid=0,
            sym=True,
        )

        eq.solve(
            verbose=3,
            ftol=1e-8,
            constraints=get_fixed_boundary_constraints(),
            optimizer=Optimizer("lsq-exact"),
            objective=ObjectiveFunction(objectives=ForceBalance()),
        )

        eq.save("test_bootstrap_consistency_iota_initial.h5")
        # eq = desc.io.load("test_bootstrap_consistency_iota_initial.h5")

        # Done establishing the initial condition. Now set up the optimization.

        constraints = (
            ForceBalance(),
            FixBoundaryR(),
            FixBoundaryZ(),
            FixPressure(),
            FixPsi(),
        )

        # grid for bootstrap consistency objective:
        grid = LinearGrid(
            rho=np.linspace(1e-4, 1 - 1e-4, (num_iota_points - 1) * 2 + 1),
            M=eq.M * 2,
            N=eq.N * 2,
            NFP=eq.NFP,
        )
        objective = ObjectiveFunction(
            BootstrapRedlConsistency(
                grid=grid,
                helicity=helicity,
                ne=ne,
                Te=Te,
                Ti=Ti,
                Zeff=Zeff,
            )
        )
        eq, result = eq.optimize(
            verbose=3,
            objective=objective,
            constraints=constraints,
            optimizer=Optimizer("scipy-trf"),
            ftol=1e-6,
        )

        eq.save("test_bootstrap_consistency_iota_final.h5")

        scalar_objective = objective.compute_scalar(objective.x(eq))
        assert scalar_objective < 3e-5
        data = eq.compute(
            ["<J*B>", "<J*B> Redl"],
            grid=grid,
            ne=ne,
            Te=Te,
            Ti=Ti,
            Zeff=Zeff,
            helicity=helicity,
        )
        J_dot_B_MHD = compress(grid, data["<J*B>"])
        J_dot_B_Redl = compress(grid, data["<J*B> Redl"])

        assert np.max(J_dot_B_MHD) < 4e5
        assert np.max(J_dot_B_MHD) > 0
        assert np.min(J_dot_B_MHD) < -5.1e6
        assert np.min(J_dot_B_MHD) > -5.4e6
        np.testing.assert_allclose(J_dot_B_MHD, J_dot_B_Redl, atol=5e5)

    @pytest.mark.unit
    @pytest.mark.slow
    def test_bootstrap_consistency_current(self):
        """Try optimizing for bootstrap consistency in axisymmetry, at fixed boundary shape.

        This version of the test covers the case of an equilibrium with a current profile.
        """

        ne0 = 4.0e20
        T0 = 12.0e3
        ne = PowerSeriesProfile(ne0 * np.array([1, -1]), modes=[0, 10])
        Te = PowerSeriesProfile(T0 * np.array([1, -1]), modes=[0, 2])
        Ti = Te
        Zeff = 1
        helicity = (1, 0)
        pressure = PowerSeriesProfile(
            2 * ne0 * T0 * elementary_charge * np.array([1, -1, 0, 0, 0, -1, 1]),
            modes=[0, 2, 4, 6, 8, 10, 12],
        )
        B0 = 5.0  # Mean |B|
        LM_resolution = 8

        current = PowerSeriesProfile([0, -1.2e7], modes=[0, 2], sym=False)

        # Set initial condition:
        Rmajor = 6.0
        aminor = 2.0
        NFP = 1
        surface = FourierRZToroidalSurface(
            R_lmn=np.array([Rmajor, aminor]),
            modes_R=[[0, 0], [1, 0]],
            Z_lmn=np.array([aminor]),
            modes_Z=[[-1, 0]],
            NFP=NFP,
        )

        eq = Equilibrium(
            surface=surface,
            pressure=pressure,
            current=current,
            Psi=B0 * np.pi * (aminor**2),
            NFP=NFP,
            L=LM_resolution,
            M=LM_resolution,
            N=0,
            L_grid=2 * LM_resolution,
            M_grid=2 * LM_resolution,
            N_grid=0,
            sym=True,
        )
        current_L = 16
        eq.current.change_resolution(current_L)

        eq.solve(
            verbose=3,
            ftol=1e-8,
            constraints=get_fixed_boundary_constraints(iota=False),
            optimizer=Optimizer("lsq-exact"),
            objective=ObjectiveFunction(objectives=ForceBalance()),
        )

        eq.save("test_bootstrap_consistency_current_initial.h5")
        # eq = desc.io.load("test_bootstrap_consistency_current_initial.h5")

        # Done establishing the initial condition. Now set up the optimization.

        constraints = (
            ForceBalance(),
            FixBoundaryR(),
            FixBoundaryZ(),
            FixPressure(),
            FixCurrent(indices=[0]),
            FixPsi(),
        )

        # grid for bootstrap consistency objective:
        grid = QuadratureGrid(
            L=current_L * 2,
            M=eq.M * 2,
            N=eq.N * 2,
            NFP=eq.NFP,
        )
        objective = ObjectiveFunction(
            BootstrapRedlConsistency(
                grid=grid,
                helicity=helicity,
                ne=ne,
                Te=Te,
                Ti=Ti,
                Zeff=Zeff,
            )
        )
        eq, result = eq.optimize(
            verbose=3,
            objective=objective,
            constraints=constraints,
            optimizer=Optimizer("scipy-trf"),
            ftol=1e-6,
            gtol=0,  # Critical to set gtol=0 when optimizing current profile!
        )

        eq.save("test_bootstrap_consistency_current_final.h5")

        scalar_objective = objective.compute_scalar(objective.x(eq))
        assert scalar_objective < 3e-5
        data = eq.compute(
            ["<J*B>", "<J*B> Redl"],
            grid=grid,
            ne=ne,
            Te=Te,
            Ti=Ti,
            Zeff=Zeff,
            helicity=helicity,
        )
        J_dot_B_MHD = compress(grid, data["<J*B>"])
        J_dot_B_Redl = compress(grid, data["<J*B> Redl"])

        assert np.max(J_dot_B_MHD) < 4e5
        assert np.max(J_dot_B_MHD) > 0
        assert np.min(J_dot_B_MHD) < -5.1e6
        assert np.min(J_dot_B_MHD) > -5.4e6
        np.testing.assert_allclose(J_dot_B_MHD, J_dot_B_Redl, atol=5e5)
