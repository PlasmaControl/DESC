"""Tests for bootstrap current functions."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.constants import elementary_charge
from scipy.integrate import quad

import desc.io
from desc.compat import rescale
from desc.compute._bootstrap import _trapped_fraction, compute_J_dot_B_Redl
from desc.compute._field import (
    _1_over_B_fsa,
    _B2_fsa,
    _effective_r_over_R0,
    _max_tz_modB,
    _min_tz_modB,
)
from desc.compute._geometry import _V_r_of_r, _V_rr_of_r
from desc.equilibrium import Equilibrium
from desc.examples import get
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid, QuadratureGrid
from desc.objectives import (
    BootstrapRedlConsistency,
    FixAtomicNumber,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixElectronDensity,
    FixElectronTemperature,
    FixIonTemperature,
    FixPsi,
    ForceBalance,
    ObjectiveFunction,
    get_fixed_boundary_constraints,
)
from desc.objectives.normalization import compute_scaling_factors
from desc.optimize import Optimizer
from desc.profiles import PowerSeriesProfile, SplineProfile

pytest_mpl_tol = 7.8
pytest_mpl_remove_text = True


def trapped_fraction(grid, modB, sqrt_g, sqrt_g_r):
    """
    Helper function to test trapped fraction calculation.

    Function to help test the trapped fraction calculation on
    analytic B fields rather than Equilibrium objects.
    """
    data = {"|B|": modB, "|B|^2": modB**2, "sqrt(g)": sqrt_g, "sqrt(g)_r": sqrt_g_r}
    params = None
    transforms = {"grid": grid}
    profiles = None
    data = _V_r_of_r(params, transforms, profiles, data)
    data = _V_rr_of_r(params, transforms, profiles, data)
    data = _B2_fsa(params, transforms, profiles, data)
    data = _1_over_B_fsa(params, transforms, profiles, data)
    data = _max_tz_modB(params, transforms, profiles, data)
    data = _min_tz_modB(params, transforms, profiles, data)
    data = _effective_r_over_R0(params, transforms, profiles, data)
    data = _trapped_fraction(params, transforms, profiles, data)
    return data


class TestBootstrapCompute:
    """Tests for bootstrap current compute functions."""

    @pytest.mark.unit
    def test_trapped_fraction_analytic(self):
        """Confirm that trapped_fraction() matches analytic results for model field."""
        M = 100
        NFP = 3
        for N in [0, 25]:
            grid = LinearGrid(rho=[0.5, 1], M=M, N=N, NFP=NFP)
            theta = grid.nodes[:, 1]
            zeta = grid.nodes[:, 2]

            sqrt_g = np.where(grid.inverse_rho_idx == 0, 10.0, 0.0)
            mask = grid.inverse_rho_idx == 1
            sqrt_g[mask] = -25.0
            modB = np.where(
                mask, 9.0 + 3.7 * np.sin(theta - NFP * zeta), 13.0 + 2.6 * np.cos(theta)
            )
            # TODO (#671): find value for sqrt_g_r to test axis limit
            f_t_data = trapped_fraction(grid, modB, sqrt_g, sqrt_g_r=np.nan)
            # The average of (b0 + b1 cos(theta))^2 is b0^2 + (1/2) * b1^2
            np.testing.assert_allclose(
                f_t_data["<|B|^2>"],
                grid.expand(np.array([13.0**2 + 0.5 * 2.6**2, 9.0**2 + 0.5 * 3.7**2])),
            )
            np.testing.assert_allclose(
                f_t_data["<1/|B|>"],
                grid.expand(1 / np.sqrt([13.0**2 - 2.6**2, 9.0**2 - 3.7**2])),
            )
            np.testing.assert_allclose(
                f_t_data["min_tz |B|"],
                grid.expand(np.array([13.0 - 2.6, 9.0 - 3.7])),
                rtol=1e-4,
            )
            np.testing.assert_allclose(
                f_t_data["max_tz |B|"],
                grid.expand(np.array([13.0 + 2.6, 9.0 + 3.7])),
                rtol=1e-4,
            )
            np.testing.assert_allclose(
                f_t_data["effective r/R0"],
                grid.expand(np.array([2.6 / 13.0, 3.7 / 9.0])),
                rtol=1e-3,
            )

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(
        remove_text=pytest_mpl_remove_text, tolerance=pytest_mpl_tol
    )
    def test_trapped_fraction_Kim(self):
        """Analytic test for trapped fraction calculation.

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

        fig = plt.figure()

        def test(N, grid_type):
            grid = grid_type(L=L, M=M, N=N, NFP=NFP)
            rho = grid.nodes[:, 0]
            theta = grid.nodes[:, 1]
            epsilon_3D = rho * epsilon_max
            epsilon = np.unique(epsilon_3D)

            # Eq (A6)   # noqa: E800
            modB = B0 / (1 + epsilon_3D * np.cos(theta))
            # For Jacobian, use eq (A7) for the theta dependence,
            # times an arbitrary overall scale factor
            sqrt_g = 6.7 * (1 + epsilon_3D * np.cos(theta))
            # Above "Jacobian" is nonzero at magnetic axis, so set
            # sqrt(g)_r as sqrt(g) to nullify automatic computation of
            # limit which assumes sqrt(g) is true Jacobian and zero at the
            # magnetic axis.
            f_t_data = trapped_fraction(grid, modB, sqrt_g, sqrt_g_r=sqrt_g)

            # Eq (C18) in Kim et al:
            f_t_Kim = 1.46 * np.sqrt(epsilon) - 0.46 * epsilon

            np.testing.assert_allclose(
                f_t_data["min_tz |B|"], grid.expand(B0 / (1 + epsilon))
            )
            # Looser tolerance for Bmax since there is no grid point there:
            Bmax = B0 / (1 - epsilon)
            np.testing.assert_allclose(
                f_t_data["max_tz |B|"], grid.expand(Bmax), rtol=0.001
            )
            np.testing.assert_allclose(
                f_t_data["effective r/R0"], grid.expand(epsilon), rtol=1e-4
            )
            # Eq (A8):
            fsa_B2 = B0**2 / np.sqrt(1 - epsilon**2)
            np.testing.assert_allclose(
                f_t_data["<|B|^2>"], grid.expand(fsa_B2), rtol=1e-6
            )
            np.testing.assert_allclose(
                f_t_data["<1/|B|>"], grid.expand((2 + epsilon**2) / (2 * B0))
            )
            # Note the loose tolerance for this next test since we do not expect precise
            # agreement.
            np.testing.assert_allclose(
                f_t_data["trapped fraction"], grid.expand(f_t_Kim), rtol=0.1, atol=0.07
            )

            # Now compute f_t numerically by a different algorithm:
            modB = modB.reshape((grid.num_zeta, grid.num_rho, grid.num_theta))
            sqrt_g = sqrt_g.reshape((grid.num_zeta, grid.num_rho, grid.num_theta))
            fourpisq = 4 * np.pi**2
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
                grid.compress(f_t_data["trapped fraction"])[1:],
                f_t[1:],
                rtol=0.001,
                atol=0.001,
            )

            plt.plot(epsilon, f_t_Kim, "b", label="Kim")
            plt.plot(
                epsilon, grid.compress(f_t_data["trapped fraction"]), "r", label="desc"
            )
            plt.plot(epsilon, f_t, ":g", label="Alternative algorithm")

        for N in [0, 13]:
            for grid_type in [LinearGrid, QuadratureGrid]:
                test(N, grid_type)

        plt.xlabel("epsilon")
        plt.ylabel("Effective trapped fraction $f_t$")
        plt.legend(loc=0)
        return fig

    @pytest.mark.unit
    def test_Redl_second_pass(self):
        """Alternate implementation of Redl calculations for verification.

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
        Zeff_rho = Zeff(rho)
        ni_rho = ne_rho / Zeff_rho
        d_ne_d_s = ne(rho, dr=1) / (2 * rho)
        d_Te_d_s = Te(rho, dr=1) / (2 * rho)
        d_Ti_d_s = Ti(rho, dr=1) / (2 * rho)

        # Sauter eq (18d)-(18e):
        ln_Lambda_e = 31.3 - np.log(np.sqrt(ne_rho) / Te_rho)
        ln_Lambda_ii = 30.0 - np.log((Zeff_rho**3) * np.sqrt(ni_rho) / (Ti_rho**1.5))

        # Sauter eq (18b)-(18c):
        nu_e = abs(
            R
            * 6.921e-18
            * ne_rho
            * Zeff_rho
            * ln_Lambda_e
            / ((iota - helicity_N) * (Te_rho**2) * (epsilon**1.5))
        )
        nu_i = abs(
            R
            * 4.90e-18
            * ni_rho
            * (Zeff_rho**4)
            * ln_Lambda_ii
            / ((iota - helicity_N) * (Ti_rho**2) * (epsilon**1.5))
        )

        # Redl eq (11):
        X31 = f_t / (
            1
            + 0.67 * (1 - 0.7 * f_t) * np.sqrt(nu_e) / (0.56 + 0.44 * Zeff_rho)
            + (0.52 + 0.086 * np.sqrt(nu_e))
            * (1 + 0.87 * f_t)
            * nu_e
            / (1 + 1.13 * np.sqrt(Zeff_rho - 1))
        )

        # Redl eq (10):
        Zfac = Zeff_rho**1.2 - 0.71
        L31 = (
            (1 + 0.15 / Zfac) * X31
            - 0.22 / Zfac * (X31**2)
            + 0.01 / Zfac * (X31**3)
            + 0.06 / Zfac * (X31**4)
        )

        # Redl eq (14):
        X32e = f_t / (
            1
            + 0.23 * (1 - 0.96 * f_t) * np.sqrt(nu_e / Zeff_rho)
            + 0.13
            * (1 - 0.38 * f_t)
            * nu_e
            / (Zeff_rho**2)
            * (
                np.sqrt(1 + 2 * np.sqrt(Zeff_rho - 1))
                + f_t * f_t * np.sqrt((0.075 + 0.25 * ((Zeff_rho - 1) ** 2)) * nu_e)
            )
        )

        # Redl eq (13):
        F32ee = (
            (0.1 + 0.6 * Zeff_rho)
            / (Zeff_rho * (0.77 + 0.63 * (1 + (Zeff_rho - 1) ** 1.1)))
            * (X32e - X32e**4)
            + 0.7
            / (1 + 0.2 * Zeff_rho)
            * (X32e**2 - X32e**4 - 1.2 * (X32e**3 - X32e**4))
            + 1.3 / (1 + 0.5 * Zeff_rho) * (X32e**4)
        )

        # Redl eq (16)
        X32ei = f_t / (
            1
            + 0.87
            * (1 + 0.39 * f_t)
            * np.sqrt(nu_e)
            / (1 + 2.95 * ((Zeff_rho - 1) ** 2))
            + 1.53 * (1 - 0.37 * f_t) * nu_e * (2 + 0.375 * (Zeff_rho - 1))
        )

        # Redl eq (15)
        F32ei = (
            -(0.4 + 1.93 * Zeff_rho)
            / (Zeff_rho * (0.8 + 0.6 * Zeff_rho))
            * (X32ei - X32ei**4)
            + 5.5
            / (1.5 + 2 * Zeff_rho)
            * (X32ei**2 - X32ei**4 - 0.8 * (X32ei**3 - X32ei**4))
            - 1.3 / (1 + 0.5 * Zeff_rho) * (X32ei**4)
        )

        # Redl eq (12)
        L32 = F32ei + F32ee

        # Redl eq (20):
        alpha0 = (
            -(0.62 + 0.055 * (Zeff_rho - 1))
            * (1 - f_t)
            / (
                (0.53 + 0.17 * (Zeff_rho - 1))
                * (1 - (0.31 - 0.065 * (Zeff_rho - 1)) * f_t - 0.25 * (f_t**2))
            )
        )

        # Redl eq (21):
        alpha = (
            (alpha0 + 0.7 * Zeff_rho * np.sqrt(f_t * nu_i)) / (1 + 0.18 * np.sqrt(nu_i))
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
        d_ni_d_s = d_ne_d_s / Zeff_rho
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
            "G": G,
            "R": R,
            "iota": iota,
            "epsilon": epsilon,
            "f_t": f_t,
            "psi_edge": psi_edge,
        }
        profile_data = {
            "rho": rho,
            "ne": ne_rho,
            "Te": Te_rho,
            "Ti": Ti_rho,
            "Zeff": Zeff_rho,
            "ne_r": ne(rho, dr=1),
            "Te_r": Te(rho, dr=1),
            "Ti_r": Ti(rho, dr=1),
        }
        J_dot_B_data = compute_J_dot_B_Redl(geom_data, profile_data, helicity_N)

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
    @pytest.mark.mpl_image_compare(
        remove_text=pytest_mpl_remove_text, tolerance=pytest_mpl_tol
    )
    def test_Redl_figures_2_3(self):
        """Recreate plots to verify correctness.

        Make sure the implementation here can roughly recover the plots
        from figures 2 and 3 in the Redl paper.
        """
        fig = plt.figure(figsize=(7, 8))
        nrows = 3
        ncols = 2
        xlim = [-0.05, 1.05]

        for j_Zeff, Zeff in enumerate([1, 1.8]):
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
                * 6.921e-18
                * ne_rho
                * Zeff_rho
                * ln_Lambda_e
                / ((Te_rho**2) * (epsilon**1.5))
            )

            iota = nu_e_without_iota / target_nu_e_star
            # End of determining the qR profile that gives the desired nu*.

            geom_data = {
                "G": G,
                "R": R,
                "iota": iota,
                "epsilon": epsilon,
                "f_t": f_t,
                "psi_edge": psi_edge,
            }
            profile_data = {
                "rho": rho,
                "ne": ne(rho),
                "Te": Te(rho),
                "Ti": Ti(rho),
                "Zeff": Zeff_rho,
                "ne_r": ne(rho, dr=1),
                "Te_r": Te(rho, dr=1),
                "Ti_r": Ti(rho, dr=1),
            }
            J_dot_B_data = compute_J_dot_B_Redl(geom_data, profile_data, helicity_N)

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

            # Make a plot, matching the axis ranges of Redl's
            # figures 2 and 3 as best as possible.
            Zeff_str = f" for Zeff = {Zeff}"

            plt.subplot(nrows, ncols, 1 + j_Zeff)
            plt.plot(f_t, J_dot_B_data["L31"], "g")
            plt.title("L31" + Zeff_str)
            plt.xlabel("f_t")
            plt.xlim(xlim)
            plt.ylim(-0.05, 1.05)

            plt.subplot(nrows, ncols, 3 + j_Zeff)
            plt.plot(f_t, J_dot_B_data["L32"], "g")
            plt.title("L32" + Zeff_str)
            plt.xlabel("f_t")
            plt.xlim(xlim)
            if Zeff == 1:
                plt.ylim(-0.25, 0.025)
            else:
                plt.ylim(-0.2, 0.05)

            plt.subplot(nrows, ncols, 5 + j_Zeff)
            plt.plot(f_t, J_dot_B_data["alpha"], "g")
            plt.title("alpha" + Zeff_str)
            plt.xlabel("f_t")
            plt.xlim(xlim)
            plt.ylim(-1.25, 0.04)

        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(
        remove_text=pytest_mpl_remove_text, tolerance=pytest_mpl_tol
    )
    def test_Redl_figures_4_5(self):
        """Recreate plots to verify correctness.

        Make sure the implementation here can roughly recover the plots
        from figures 4 and 5 in the Redl paper.
        """
        fig = plt.figure(figsize=(7, 8))
        nrows = 3
        ncols = 2
        xlim = [3.0e-5, 1.5e4]

        for j_Zeff, Zeff in enumerate([1, 1.8]):
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
                    * 6.921e-18
                    * ne_rho
                    * Zeff_rho
                    * ln_Lambda_e
                    / ((Te_rho**2) * (epsilon**1.5))
                )

                iota = nu_e_without_iota / target_nu_e_star
                # End of determining the qR profile that gives the desired nu*.

                geom_data = {
                    "G": G,
                    "R": R,
                    "iota": iota,
                    "epsilon": epsilon,
                    "f_t": f_t,
                    "psi_edge": psi_edge,
                }
                profile_data = {
                    "rho": rho,
                    "ne": ne(rho),
                    "Te": Te(rho),
                    "Ti": Ti(rho),
                    "Zeff": Zeff_rho,
                    "ne_r": ne(rho, dr=1),
                    "Te_r": Te(rho, dr=1),
                    "Ti_r": Ti(rho, dr=1),
                }
                J_dot_B_data = compute_J_dot_B_Redl(geom_data, profile_data, helicity_N)

                L31s[j_nu_star, :] = J_dot_B_data["L31"]
                L32s[j_nu_star, :] = J_dot_B_data["L32"]
                alphas[j_nu_star, :] = J_dot_B_data["alpha"]
                nu_e_stars[j_nu_star, :] = J_dot_B_data["nu_e_star"]
                nu_i_stars[j_nu_star, :] = J_dot_B_data["nu_i_star"]
                np.testing.assert_allclose(J_dot_B_data["nu_e_star"], target_nu_e_star)
                # nu*i is tiny bit different from the target since
                # lnLambda_i != lnLambda_e:
                np.testing.assert_allclose(
                    J_dot_B_data["nu_i_star"], target_nu_i_star, rtol=0.2
                )

            # Make a plot, matching the axis ranges of Redl's
            # figures 4 and 5 as best as possible.

            plt.subplot(nrows, ncols, 1 + j_Zeff)
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

            plt.subplot(nrows, ncols, 3 + j_Zeff)
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

            plt.subplot(nrows, ncols, 5 + j_Zeff)
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

        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(
        remove_text=pytest_mpl_remove_text, tolerance=pytest_mpl_tol
    )
    def test_Redl_sfincs_tokamak_benchmark(self):
        """Compare the Redl <J*B> to a SFINCS calculation for a model tokamak.

        The SFINCS calculation is on Matt Landreman's laptop in
        /Users/mattland/Box/work21/20211225-01-sfincs_tokamak_bootstrap_for_Redl_benchmark
        """
        fig = plt.figure()
        s_surfaces = np.linspace(0.01, 0.99, 99)
        rho = np.sqrt(s_surfaces)
        helicity = (1, 0)
        filename = ".//tests//inputs//circular_model_tokamak_output.h5"
        eq = desc.io.load(filename)[-1]
        eq.electron_density = PowerSeriesProfile(
            5.0e20 * np.array([1, -1]), modes=[0, 8]
        )
        eq.electron_temperature = PowerSeriesProfile(
            8e3 * np.array([1, -1]), modes=[0, 2]
        )
        eq.ion_temperature = PowerSeriesProfile(8e3 * np.array([1, -1]), modes=[0, 2])
        # Flip sign of sfincs data, since it was computed for a
        # configuration with iota>0 in vmec's left-handed coordinate
        # system, whereas here we use a config with iota>0 in desc's
        # right-handed system.
        J_dot_B_sfincs = -np.array(
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
        data = eq.compute("<J*B> Redl", grid=grid, helicity=helicity)
        J_dot_B_Redl = grid.compress(data["<J*B> Redl"])

        # The relative error is a bit larger at the boundary, where the
        # absolute magnitude is quite small, so drop those points.
        np.testing.assert_allclose(J_dot_B_Redl[:-5], J_dot_B_sfincs[:-5], rtol=0.1)

        plt.plot(rho, J_dot_B_Redl, "+-", label="Redl")
        plt.plot(rho, J_dot_B_sfincs, ".-", label="sfincs")
        plt.xlabel("rho")
        plt.title("<J*B> [T A / m^2]")
        plt.legend(loc=0)
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(
        remove_text=pytest_mpl_remove_text, tolerance=pytest_mpl_tol
    )
    def test_Redl_sfincs_QA(self):
        """Compare the Redl <J*B> to a SFINCS calculation for reactor-scale QA.

        This test reproduces figure 1.a of Landreman Buller
        Drevlak, Physics of Plasmas 29, 082501 (2022)
        https://doi.org/10.1063/5.0098166

        The SFINCS calculation is on the IPP Cobra machine in
        /ptmp/mlan/20211226-01-sfincs_for_precise_QS_for_Redl_benchmark/20211226
        -01-012_QA_Ntheta25_Nzeta39_Nxi60_Nx7_manySurfaces
        """
        fig = plt.figure()
        helicity = (1, 0)
        filename = ".//tests//inputs//LandremanPaul2022_QA_reactorScale_lowRes.h5"
        eq = desc.io.load(filename)[-1]
        eq.electron_density = PowerSeriesProfile(
            4.13e20 * np.array([1, -1]), modes=[0, 10]
        )
        eq.electron_temperature = PowerSeriesProfile(
            12.0e3 * np.array([1, -1]), modes=[0, 2]
        )
        eq.ion_temperature = eq.electron_temperature
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
        data = eq.compute("<J*B> Redl", grid=grid, helicity=helicity)
        J_dot_B_Redl = grid.compress(data["<J*B> Redl"])

        np.testing.assert_allclose(J_dot_B_Redl[1:-1], J_dot_B_sfincs[1:-1], rtol=0.1)

        # This plot below reproduces figure 1.a of Landreman Buller
        # Drevlak, Physics of Plasmas 29, 082501 (2022)
        # https://doi.org/10.1063/5.0098166
        plt.plot(rho**2, J_dot_B_Redl, "+-", label="Redl")
        plt.plot(rho**2, J_dot_B_sfincs, ".-r", label="sfincs")
        plt.legend(loc=0)
        plt.xlabel(r"$\rho^2 = s$")
        plt.ylabel("<J*B> [T A / m^2]")
        plt.xlim([0, 1])
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(
        remove_text=pytest_mpl_remove_text, tolerance=pytest_mpl_tol
    )
    def test_Redl_sfincs_QH(self):
        """Compare the Redl <J*B> to a SFINCS calculation for a reactor-scale QH.

        This test reproduces figure 1.b of Landreman Buller
        Drevlak, Physics of Plasmas 29, 082501 (2022)
        https://doi.org/10.1063/5.0098166

        The SFINCS calculation is on the IPP Cobra machine in
        /ptmp/mlan/20211226-01-sfincs_for_precise_QS_for_Redl_benchmark/20211226
        -01-019_QH_Ntheta25_Nzeta39_Nxi60_Nx7_manySurfaces
        """
        fig = plt.figure()
        helicity = (1, 4)
        filename = ".//tests//inputs//LandremanPaul2022_QH_reactorScale_lowRes.h5"
        eq = desc.io.load(filename)[-1]
        eq.electron_density = PowerSeriesProfile(
            4.13e20 * np.array([1, -1]), modes=[0, 10]
        )
        eq.electron_temperature = PowerSeriesProfile(
            12.0e3 * np.array([1, -1]), modes=[0, 2]
        )
        eq.ion_temperature = eq.electron_temperature
        eq.atomic_number = 1
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
        data = eq.compute("<J*B> Redl", grid=grid, helicity=helicity)
        grid2 = LinearGrid(rho=0.0, M=eq.M, N=eq.N, NFP=eq.NFP)
        grid3 = LinearGrid(rho=1.0, M=eq.M, N=eq.N, NFP=eq.NFP)
        with pytest.warns(UserWarning, match="rho=0"):
            eq.compute("<J*B> Redl", grid=grid2, helicity=helicity)
        with pytest.warns(UserWarning, match="vanish"):
            eq.compute("<J*B> Redl", grid=grid3, helicity=helicity)

        J_dot_B_Redl = grid.compress(data["<J*B> Redl"])

        np.testing.assert_allclose(J_dot_B_Redl[1:-1], J_dot_B_sfincs[1:-1], rtol=0.1)

        # The plot below reproduces figure 1.b of Landreman Buller
        # Drevlak, Physics of Plasmas 29, 082501 (2022)
        # https://doi.org/10.1063/5.0098166
        plt.plot(rho**2, J_dot_B_Redl, "+-", label="Redl")
        plt.plot(rho**2, J_dot_B_sfincs, ".-r", label="sfincs")
        plt.legend(loc=0)
        plt.xlabel(r"$\rho^2 = s$")
        plt.ylabel("<J*B> [T A / m^2]")
        plt.xlim([0, 1])
        return fig


class TestBootstrapObjectives:
    """Tests for bootstrap current objective functions."""

    @pytest.mark.unit
    def test_BootstrapRedlConsistency_normalization(self):
        """Objective function should be invariant under scaling |B| or size."""
        Rmajor = 1.7
        aminor = 0.2
        Delta = 0.3
        NFP = 5
        Psi0 = 1.2
        LMN_resolution = 5
        helicity = (1, NFP)
        ne0 = 3.0e20
        Te0 = 15e3
        Ti0 = 14e3
        ne = PowerSeriesProfile(ne0 * np.array([1, -0.85]), modes=[0, 4])
        Te = PowerSeriesProfile(Te0 * np.array([1.02, -3, 3, -1]), modes=[0, 2, 4, 6])
        Ti = PowerSeriesProfile(Ti0 * np.array([1.02, -3, 3, -1]), modes=[0, 2, 4, 6])
        Zeff = 1.4

        surface = FourierRZToroidalSurface(
            R_lmn=np.array([Rmajor, aminor, Delta]),
            modes_R=[[0, 0], [1, 0], [0, 1]],
            Z_lmn=np.array([-aminor, Delta]),
            modes_Z=[[-1, 0], [0, -1]],
            NFP=NFP,
        )

        eq = Equilibrium(
            surface=surface,
            electron_density=ne,
            electron_temperature=Te,
            ion_temperature=Ti,
            atomic_number=Zeff,
            iota=PowerSeriesProfile([1.1]),
            Psi=Psi0,
            NFP=NFP,
            L=LMN_resolution,
            M=LMN_resolution,
            N=LMN_resolution,
            L_grid=2 * LMN_resolution,
            M_grid=2 * LMN_resolution,
            N_grid=2 * LMN_resolution,
            sym=True,
        )
        # The equilibrium need not be in force balance, so no need to solve().
        grid = QuadratureGrid(
            L=LMN_resolution, M=LMN_resolution, N=LMN_resolution, NFP=eq.NFP
        )
        obj = ObjectiveFunction(
            BootstrapRedlConsistency(eq=eq, grid=grid, helicity=helicity)
        )
        obj.build()
        scalar_objective1 = obj.compute_scalar(obj.x(eq))

        # Scale |B|, changing <J*B> and <J*B>_Redl by "factor":
        factor = 2.0
        eq.Psi *= np.sqrt(factor)
        # Scale n and T to vary neoclassical <J*B> at fixed nu*:
        eq.electron_density = PowerSeriesProfile(
            factor ** (3.0 / 5) * ne0 * np.array([1, -0.85]), modes=[0, 4]
        )
        eq.electron_temperature = PowerSeriesProfile(
            factor ** (2.0 / 5) * Te0 * np.array([1.02, -3, 3, -1]), modes=[0, 2, 4, 6]
        )
        eq.ion_temperature = PowerSeriesProfile(
            factor ** (2.0 / 5) * Ti0 * np.array([1.02, -3, 3, -1]), modes=[0, 2, 4, 6]
        )
        obj = ObjectiveFunction(
            BootstrapRedlConsistency(eq=eq, grid=grid, helicity=helicity)
        )
        obj.build()
        scalar_objective2 = obj.compute_scalar(obj.x(eq))

        # Scale size, changing <J*B> and <J*B>_Redl by "factor":
        factor = 3.0
        eq.Psi = Psi0 / factor**2
        eq.R_lmn /= factor
        eq.Z_lmn /= factor
        eq.Rb_lmn /= factor
        eq.Zb_lmn /= factor
        # Scale n and T to vary neoclassical <J*B> at fixed nu*:
        eq.electron_density = PowerSeriesProfile(
            factor ** (2.0 / 5) * ne0 * np.array([1, -0.85]), modes=[0, 4]
        )
        eq.electron_temperature = PowerSeriesProfile(
            factor ** (-2.0 / 5) * Te0 * np.array([1.02, -3, 3, -1]), modes=[0, 2, 4, 6]
        )
        eq.ion_temperature = PowerSeriesProfile(
            factor ** (-2.0 / 5) * Ti0 * np.array([1.02, -3, 3, -1]), modes=[0, 2, 4, 6]
        )
        obj = ObjectiveFunction(
            BootstrapRedlConsistency(eq=eq, grid=grid, helicity=helicity)
        )
        obj.build()
        scalar_objective3 = obj.compute_scalar(obj.x(eq))

        results = np.array([scalar_objective1, scalar_objective2, scalar_objective3])

        # Compute the expected objective from
        # ½ ∫dρ [(⟨J⋅B⟩_MHD - ⟨J⋅B⟩_Redl) / (J_ref B_ref)]²
        scales = compute_scaling_factors(eq)
        data = eq.compute(["<J*B>", "<J*B> Redl"], grid=grid, helicity=helicity)
        integrand = (data["<J*B>"] - data["<J*B> Redl"]) / (scales["B"] * scales["J"])
        expected = 0.5 * sum(grid.weights * integrand**2) / (4 * np.pi**2)
        print(
            "bootstrap objectives for scaled configs:", results, " expected:", expected
        )

        # Results are not perfectly identical because ln(Lambda) is not quite invariant.
        np.testing.assert_allclose(results, expected, rtol=3e-3)

    @pytest.mark.regression
    def test_bootstrap_consistency_iota(self, TmpDir):
        """Try optimizing for bootstrap consistency in axisymmetry, at fixed shape.

        This version of the test covers an equilibrium with an iota
        profile rather than a current profile.
        """
        ne0 = 4.0e20
        T0 = 12.0e3
        ne = PowerSeriesProfile(ne0 * np.array([1, -1]), modes=[0, 10])
        Te = PowerSeriesProfile(T0 * np.array([1, -1]), modes=[0, 2])
        Ti = Te
        Zeff = 1

        helicity = (1, 0)
        B0 = 5.0  # Desired average |B|
        LM_resolution = 8

        initial_iota = -0.61
        num_iota_points = 21
        iota = SplineProfile(np.full(num_iota_points, initial_iota))

        # Set initial condition:
        Rmajor = 6.0
        aminor = 2.0
        NFP = 1
        surface = FourierRZToroidalSurface(
            R_lmn=np.array([Rmajor, aminor]),
            modes_R=[[0, 0], [1, 0]],
            Z_lmn=np.array([-aminor]),
            modes_Z=[[-1, 0]],
            NFP=NFP,
        )

        eq = Equilibrium(
            surface=surface,
            electron_density=ne,
            electron_temperature=Te,
            ion_temperature=Ti,
            atomic_number=Zeff,
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
            constraints=get_fixed_boundary_constraints(eq=eq),
            optimizer=Optimizer("lsq-exact"),
            objective=ObjectiveFunction(objectives=ForceBalance(eq=eq)),
        )

        initial_output_file = str(
            TmpDir.join("test_bootstrap_consistency_iota_initial.h5")
        )
        print("initial_output_file:", initial_output_file)
        eq.save(initial_output_file)

        # Done establishing the initial condition. Now set up the optimization.

        constraints = (
            ForceBalance(eq=eq),
            FixBoundaryR(eq=eq),
            FixBoundaryZ(eq=eq),
            FixElectronDensity(eq=eq),
            FixElectronTemperature(eq=eq),
            FixIonTemperature(eq=eq),
            FixAtomicNumber(eq=eq),
            FixPsi(eq=eq),
        )

        # grid for bootstrap consistency objective:
        grid = LinearGrid(
            rho=np.linspace(1e-4, 1 - 1e-4, (num_iota_points - 1) * 2 + 1),
            M=eq.M * 2,
            N=eq.N * 2,
            NFP=eq.NFP,
        )
        objective = ObjectiveFunction(
            BootstrapRedlConsistency(eq=eq, grid=grid, helicity=helicity)
        )
        eq, _ = eq.optimize(
            verbose=3,
            objective=objective,
            constraints=constraints,
            optimizer=Optimizer("proximal-scipy-trf"),
            ftol=1e-6,
        )

        final_output_file = str(TmpDir.join("test_bootstrap_consistency_iota_final.h5"))
        print("final_output_file:", final_output_file)
        eq.save(final_output_file)

        scalar_objective = objective.compute_scalar(objective.x(eq))
        assert scalar_objective < 3e-5
        data = eq.compute(["<J*B>", "<J*B> Redl"], grid=grid, helicity=helicity)
        J_dot_B_MHD = grid.compress(data["<J*B>"])
        J_dot_B_Redl = grid.compress(data["<J*B> Redl"])

        assert np.max(J_dot_B_MHD) < 4e5
        assert np.max(J_dot_B_MHD) > 0
        assert np.min(J_dot_B_MHD) < -5.1e6
        assert np.min(J_dot_B_MHD) > -5.4e6
        np.testing.assert_allclose(J_dot_B_MHD, J_dot_B_Redl, atol=5e5)

    @pytest.mark.regression
    def test_bootstrap_consistency_current(self, TmpDir):
        """
        Try optimizing for bootstrap consistency in axisymmetry, at fixed shape.

        This version of the test covers the case of an equilibrium
        with a current profile rather than an iota profile.
        """
        ne0 = 4.0e20
        T0 = 12.0e3
        ne = PowerSeriesProfile(ne0 * np.array([1, -1]), modes=[0, 10])
        Te = PowerSeriesProfile(T0 * np.array([1, -1]), modes=[0, 2])
        Ti = Te

        helicity = (1, 0)
        B0 = 5.0  # Desired average |B|
        LM_resolution = 8

        current = PowerSeriesProfile([0, -1.2e7], modes=[0, 2], sym=False)

        # Set initial condition:
        Rmajor = 6.0
        aminor = 2.0
        NFP = 1
        surface = FourierRZToroidalSurface(
            R_lmn=np.array([Rmajor, aminor]),
            modes_R=[[0, 0], [1, 0]],
            Z_lmn=np.array([-aminor]),
            modes_Z=[[-1, 0]],
            NFP=NFP,
        )
        with pytest.warns(UserWarning, match="current profile is not an even"):
            eq = Equilibrium(
                surface=surface,
                electron_density=ne,
                electron_temperature=Te,
                ion_temperature=Ti,
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
            constraints=get_fixed_boundary_constraints(eq=eq),
            optimizer=Optimizer("lsq-exact"),
            objective=ObjectiveFunction(objectives=ForceBalance(eq=eq)),
        )

        initial_output_file = str(
            TmpDir.join("test_bootstrap_consistency_current_initial.h5")
        )
        print("initial_output_file:", initial_output_file)
        eq.save(initial_output_file)

        # Done establishing the initial condition. Now set up the optimization.

        constraints = (
            ForceBalance(eq=eq),
            FixBoundaryR(eq=eq),
            FixBoundaryZ(eq=eq),
            FixElectronDensity(eq=eq),
            FixElectronTemperature(eq=eq),
            FixIonTemperature(eq=eq),
            FixAtomicNumber(eq=eq),
            FixCurrent(eq=eq, indices=[0]),
            FixPsi(eq=eq),
        )

        # grid for bootstrap consistency objective:
        grid = QuadratureGrid(L=current_L * 2, M=eq.M * 2, N=eq.N * 2, NFP=eq.NFP)
        objective = ObjectiveFunction(
            BootstrapRedlConsistency(eq=eq, grid=grid, helicity=helicity)
        )
        eq, _ = eq.optimize(
            verbose=3,
            objective=objective,
            constraints=constraints,
            optimizer=Optimizer("proximal-scipy-trf"),
            ftol=1e-6,
            gtol=0,  # It is critical to set gtol=0 when optimizing current profile!
        )

        final_output_file = str(
            TmpDir.join("test_bootstrap_consistency_current_final.h5")
        )
        print("final_output_file:", final_output_file)
        eq.save(final_output_file)

        scalar_objective = objective.compute_scalar(objective.x(eq))
        assert scalar_objective < 3e-5
        data = eq.compute(["<J*B>", "<J*B> Redl"], grid=grid, helicity=helicity)
        J_dot_B_MHD = grid.compress(data["<J*B>"])
        J_dot_B_Redl = grid.compress(data["<J*B> Redl"])

        assert np.max(J_dot_B_MHD) < 4e5
        assert np.max(J_dot_B_MHD) > 0
        assert np.min(J_dot_B_MHD) < -5.1e6
        assert np.min(J_dot_B_MHD) > -5.4e6
        np.testing.assert_allclose(J_dot_B_MHD, J_dot_B_Redl, atol=5e5)


@pytest.mark.unit
def test_bootstrap_objective_build():
    """Test defaults and warnings in bootstrap objective build method."""
    # can't build without profiles
    eq = Equilibrium(L=3, M=3, N=3, NFP=2)
    with pytest.raises(RuntimeError):
        BootstrapRedlConsistency(eq=eq).build()
    eq = Equilibrium(
        L=3, M=3, N=3, NFP=2, electron_temperature=1e3, electron_density=1e21
    )
    eq.electron_density = None
    with pytest.raises(RuntimeError):
        BootstrapRedlConsistency(eq=eq).build()
    eq = Equilibrium(
        L=3, M=3, N=3, NFP=2, electron_temperature=1e3, electron_density=1e21
    )
    eq.ion_temperature = None
    with pytest.raises(RuntimeError):
        BootstrapRedlConsistency(eq=eq).build()
    eq = Equilibrium(
        L=3, M=3, N=3, NFP=2, electron_temperature=1e3, electron_density=1e25
    )
    # density too high
    with pytest.raises(UserWarning):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            BootstrapRedlConsistency(eq=eq).build()
    eq = Equilibrium(
        L=3, M=3, N=3, NFP=2, electron_temperature=1e5, electron_density=1e21
    )
    # electron temperature too high
    with pytest.raises(UserWarning):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            BootstrapRedlConsistency(eq=eq).build()
    eq = Equilibrium(
        L=3,
        M=3,
        N=3,
        NFP=2,
        electron_temperature=1e3,
        electron_density=1e21,
        ion_temperature=1e5,
    )
    # ion temperature too high
    with pytest.raises(UserWarning):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            BootstrapRedlConsistency(eq=eq).build()
    eq = Equilibrium(
        L=3,
        M=3,
        N=3,
        NFP=2,
        electron_temperature=1e3,
        electron_density=1e1,
        ion_temperature=1e3,
    )
    # density too low
    with pytest.raises(UserWarning):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            BootstrapRedlConsistency(eq=eq).build()
    eq = Equilibrium(
        L=3,
        M=3,
        N=3,
        NFP=2,
        electron_temperature=3,
        electron_density=1e21,
        ion_temperature=1e3,
    )
    # electron temperature too low
    with pytest.raises(UserWarning):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            BootstrapRedlConsistency(eq=eq).build()
    eq = Equilibrium(
        L=3,
        M=3,
        N=3,
        NFP=2,
        electron_temperature=1e3,
        electron_density=1e21,
        ion_temperature=1,
    )
    # ion temperature too low
    with pytest.raises(UserWarning):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            BootstrapRedlConsistency(eq=eq).build()

    eq = Equilibrium(
        L=4,
        M=4,
        N=4,
        NFP=2,
        electron_temperature=1e3,
        electron_density=1e21,
        ion_temperature=1e3,
    )
    obj = BootstrapRedlConsistency(eq=eq)
    obj.build()
    # make sure default grid has the right nodes
    assert obj.constants["transforms"]["grid"].num_theta == 17
    assert obj.constants["transforms"]["grid"].num_zeta == 17
    assert obj.constants["transforms"]["grid"].num_rho == 4
    np.testing.assert_allclose(
        obj.constants["transforms"]["grid"].nodes[
            obj.constants["transforms"]["grid"].unique_rho_idx, 0
        ],
        np.array([0.125, 0.375, 0.625, 0.875]),
    )


@pytest.mark.slow
@pytest.mark.regression
def test_bootstrap_optimization_comparison_qa():
    """Test that both methods of bootstrap optimization agree."""
    # this same example is used in docs/notebooks/tutorials/bootstrap_current

    # initial equilibrium
    eq0 = get("precise_QA")
    eq0 = rescale(eq0, L=("R0", 10), B=("B0", 5.86))
    eq0.pressure = None
    eq0.atomic_number = PowerSeriesProfile(np.array([1]), sym=True)
    eq0.electron_density = (
        PowerSeriesProfile(np.array([1.0, 0.0, 0.0, 0.0, 0.0, -1.0]), sym=True)
        * 2.38e20
    )
    eq0.electron_temperature = (
        PowerSeriesProfile(np.array([1.0, -1.0]), sym=True) * 9.45e3
    )
    eq0.ion_temperature = PowerSeriesProfile(np.array([1.0, -1.0]), sym=True) * 9.45e3
    with pytest.warns(UserWarning, match="not an even power series"):
        eq0.current = PowerSeriesProfile(np.zeros((eq0.L + 1,)), sym=False)
    eq0, _ = eq0.solve(objective="force", optimizer="lsq-exact", verbose=3)
    eq1 = eq0.copy()
    eq2 = eq0.copy()

    grid = LinearGrid(
        M=eq0.M_grid,
        N=eq0.N_grid,
        NFP=eq0.NFP,
        sym=eq0.sym,
        rho=np.linspace(1 / eq0.L_grid, 1, eq0.L_grid) - 1 / (2 * eq0.L_grid),
    )

    # method 1
    objective = ObjectiveFunction(
        BootstrapRedlConsistency(eq=eq1, grid=grid, helicity=(1, 0)),
    )
    constraints = (
        FixAtomicNumber(eq=eq1),
        FixBoundaryR(eq=eq1),
        FixBoundaryZ(eq=eq1),
        FixCurrent(eq=eq1, indices=[0, 1]),
        FixElectronDensity(eq=eq1),
        FixElectronTemperature(eq=eq1),
        FixIonTemperature(eq=eq1),
        FixPsi(eq=eq1),
        ForceBalance(eq=eq1),
    )
    eq1, _ = eq1.optimize(
        objective=objective,
        constraints=constraints,
        optimizer="proximal-lsq-exact",
        maxiter=10,
        verbose=3,
    )

    # method 2
    niters = 4
    for k in range(niters):
        eq2 = eq2.copy()
        data = eq2.compute("current Redl", grid)
        current = grid.compress(data["current Redl"])
        rho = grid.compress(data["rho"])
        XX = np.fliplr(np.vander(rho, eq2.L + 1)[:, :-2])
        eq2.c_l = np.pad(np.linalg.lstsq(XX, current, rcond=None)[0], (2, 0))
        eq2, _ = eq2.solve(objective="force", optimizer="lsq-exact", verbose=3)

    grid = LinearGrid(
        M=eq0.M_grid,
        N=eq0.N_grid,
        NFP=eq0.NFP,
        sym=eq0.sym,
        rho=np.linspace(0.25, 0.9, 14),
    )
    data1 = eq1.compute(["<J*B> Redl", "<J*B>"], grid)
    data2 = eq2.compute(["<J*B> Redl", "<J*B>"], grid)

    np.testing.assert_allclose(
        grid.compress(data1["<J*B>"]), grid.compress(data1["<J*B> Redl"]), rtol=2e-2
    )
    np.testing.assert_allclose(
        grid.compress(data2["<J*B>"]), grid.compress(data2["<J*B> Redl"]), rtol=2e-2
    )
    np.testing.assert_allclose(
        grid.compress(data1["<J*B>"]), grid.compress(data2["<J*B>"]), rtol=2e-2
    )
