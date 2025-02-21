"""Test integration algorithms."""

from functools import partial

import numpy as np
import pytest
from jax import grad
from matplotlib import pyplot as plt
from numpy.polynomial.chebyshev import chebinterpolate, chebroots
from numpy.polynomial.legendre import leggauss
from scipy import integrate
from scipy.interpolate import CubicHermiteSpline
from scipy.special import ellipe, ellipk, ellipkm1
from tests.test_interp_utils import _f_1d, _f_1d_nyquist_freq, _f_2d, _f_2d_nyquist_freq
from tests.test_plotting import tol_1d

from desc.backend import jnp, vmap
from desc.basis import FourierZernikeBasis
from desc.equilibrium import Equilibrium
from desc.equilibrium.coords import get_rtz_grid
from desc.examples import get
from desc.geometry import FourierRZToroidalSurface
from desc.grid import ConcentricGrid, Grid, LinearGrid, QuadratureGrid
from desc.integrals import (
    Bounce1D,
    Bounce2D,
    DFTInterpolator,
    FFTInterpolator,
    line_integrals,
    singular_integral,
    surface_averages,
    surface_integrals,
    surface_integrals_transform,
    surface_max,
    surface_min,
    surface_variance,
    virtual_casing_biot_savart,
)
from desc.integrals._bounce_utils import (
    _check_bounce_points,
    _get_extrema,
    bounce_points,
)
from desc.integrals._interp_utils import fourier_pts
from desc.integrals._laplace import _compute_dPhi_dn, _compute_Phi_mn, compute_laplace
from desc.integrals.basis import FourierChebyshevSeries
from desc.integrals.quad_utils import (
    automorphism_sin,
    bijection_from_disc,
    chebgauss1,
    chebgauss2,
    get_quadrature,
    grad_automorphism_sin,
    grad_bijection_from_disc,
    leggauss_lob,
    tanh_sinh,
)
from desc.integrals.singularities import (
    _get_quadrature_nodes,
    _kernel_nr_over_r3,
    best_ratio,
    heuristic_support_params,
)
from desc.integrals.surface_integral import _get_grid_surface
from desc.magnetic_fields import ToroidalMagneticField
from desc.plotting import plot_boundary
from desc.transform import Transform
from desc.utils import dot, errorif, safediv


class TestSurfaceIntegral:
    """Tests for non-singular surface integrals."""

    # arbitrary choice
    L = 5
    M = 5
    N = 2
    NFP = 3

    @staticmethod
    def _surface_integrals(grid, q=np.array([1.0]), surface_label="rho"):
        """Compute a surface integral for each surface in the grid."""
        _, _, spacing, has_endpoint_dupe, _ = _get_grid_surface(
            grid, grid.get_label(surface_label)
        )
        weights = (spacing.prod(axis=1) * np.nan_to_num(q).T).T
        surfaces = {}
        nodes = grid.nodes[:, {"rho": 0, "theta": 1, "zeta": 2}[surface_label]]
        for grid_row_idx, surface_label_value in enumerate(nodes):
            surfaces.setdefault(surface_label_value, []).append(grid_row_idx)
        integrals = [weights[surfaces[key]].sum(axis=0) for key in sorted(surfaces)]
        if has_endpoint_dupe:
            integrals[0] = integrals[-1] = integrals[0] + integrals[-1]
        return np.asarray(integrals)

    @pytest.mark.unit
    def test_unknown_unique_grid_integral(self):
        """Test that averages are invariant to whether grids have unique_idx."""
        lg = LinearGrid(L=self.L, M=self.M, N=self.N, NFP=self.NFP, endpoint=False)
        q = np.arange(lg.num_nodes) ** 2
        result = surface_integrals(lg, q, surface_label="rho")
        del lg._unique_rho_idx
        np.testing.assert_allclose(
            surface_integrals(lg, q, surface_label="rho"), result
        )
        result = surface_averages(lg, q, surface_label="theta")
        del lg._unique_poloidal_idx
        np.testing.assert_allclose(
            surface_averages(lg, q, surface_label="theta"), result
        )
        result = surface_variance(lg, q, surface_label="zeta")
        del lg._unique_zeta_idx
        np.testing.assert_allclose(
            surface_variance(lg, q, surface_label="zeta"), result
        )

    @pytest.mark.unit
    def test_surface_integrals_transform(self):
        """Test surface integral of a kernel function."""

        def test(surface_label, grid):
            ints = np.arange(grid.num_nodes)
            # better to test when all elements have the same sign
            q = np.abs(np.outer(np.cos(ints), np.sin(ints)))
            # This q represents the kernel function
            # K_{u_1} = |cos(x(u_1, u_2, u_3)) * sin(x(u_4, u_5, u_6))|
            # The first dimension of q varies the domain u_1, u_2, and u_3
            # and the second dimension varies the codomain u_4, u_5, u_6.
            integrals = surface_integrals_transform(grid, surface_label)(q)
            unique_size = {
                "rho": grid.num_rho,
                "theta": grid.num_theta,
                "zeta": grid.num_zeta,
            }[surface_label]
            assert integrals.shape == (unique_size, grid.num_nodes), surface_label

            desired = self._surface_integrals(grid, q, surface_label)
            np.testing.assert_allclose(integrals, desired, err_msg=surface_label)

        cg = ConcentricGrid(L=self.L, M=self.M, N=self.N, sym=True, NFP=self.NFP)
        lg = LinearGrid(
            L=self.L, M=self.M, N=self.N, sym=True, NFP=self.NFP, endpoint=True
        )
        test("rho", cg)
        test("theta", lg)
        test("zeta", cg)

    @pytest.mark.unit
    def test_surface_averages_vector_functions(self):
        """Test surface averages of vector-valued, function-valued integrands."""

        def test(surface_label, grid):
            g_size = grid.num_nodes  # not a choice; required
            f_size = g_size // 10 + (g_size < 10)
            # arbitrary choice, but f_size != v_size != g_size is better to test
            v_size = g_size // 20 + (g_size < 20)
            g = np.cos(np.arange(g_size))
            fv = np.sin(np.arange(f_size * v_size).reshape(f_size, v_size))
            # better to test when all elements have the same sign
            q = np.abs(np.einsum("g,fv->gfv", g, fv))
            sqrt_g = np.arange(g_size).astype(float)

            averages = surface_averages(grid, q, sqrt_g, surface_label)
            assert averages.shape == q.shape == (g_size, f_size, v_size), surface_label

            desired = (
                self._surface_integrals(grid, (sqrt_g * q.T).T, surface_label).T
                / self._surface_integrals(grid, sqrt_g, surface_label)
            ).T
            np.testing.assert_allclose(
                grid.compress(averages, surface_label), desired, err_msg=surface_label
            )

        cg = ConcentricGrid(L=self.L, M=self.M, N=self.N, sym=True, NFP=self.NFP)
        lg = LinearGrid(
            L=self.L, M=self.M, N=self.N, sym=True, NFP=self.NFP, endpoint=True
        )
        test("rho", cg)
        test("theta", lg)
        test("zeta", cg)

    @pytest.mark.unit
    def test_surface_area(self):
        """Test that surface_integrals(ds) is 4π² for rho, 2pi for theta, zeta.

        This test should ensure that surfaces have the correct area on grids
        constructed by specifying L, M, N and by specifying an array of nodes.
        Each test should also be done on grids with duplicate nodes
        (e.g. endpoint=True) and grids with symmetry.
        """

        def test(surface_label, grid):
            areas = surface_integrals(
                grid, surface_label=surface_label, expand_out=False
            )
            correct_area = 4 * np.pi**2 if surface_label == "rho" else 2 * np.pi
            np.testing.assert_allclose(areas, correct_area, err_msg=surface_label)

        lg = LinearGrid(
            L=self.L, M=self.M, N=self.N, NFP=self.NFP, sym=False, endpoint=False
        )
        lg_sym = LinearGrid(
            L=self.L, M=self.M, N=self.N, NFP=self.NFP, sym=True, endpoint=False
        )
        lg_endpoint = LinearGrid(
            L=self.L, M=self.M, N=self.N, NFP=self.NFP, sym=False, endpoint=True
        )
        lg_sym_endpoint = LinearGrid(
            L=self.L, M=self.M, N=self.N, NFP=self.NFP, sym=True, endpoint=True
        )
        rho = np.linspace(1, 0, self.L)[::-1]
        theta = np.linspace(0, 2 * np.pi, self.M, endpoint=False)
        theta_endpoint = np.linspace(0, 2 * np.pi, self.M, endpoint=True)
        zeta = np.linspace(0, 2 * np.pi / self.NFP, self.N, endpoint=False)
        zeta_endpoint = np.linspace(0, 2 * np.pi / self.NFP, self.N, endpoint=True)
        lg_2 = LinearGrid(
            rho=rho, theta=theta, zeta=zeta, NFP=self.NFP, sym=False, endpoint=False
        )
        lg_2_sym = LinearGrid(
            rho=rho, theta=theta, zeta=zeta, NFP=self.NFP, sym=True, endpoint=False
        )
        lg_2_endpoint = LinearGrid(
            rho=rho,
            theta=theta_endpoint,
            zeta=zeta_endpoint,
            NFP=self.NFP,
            sym=False,
            endpoint=True,
        )
        lg_2_sym_endpoint = LinearGrid(
            rho=rho,
            theta=theta_endpoint,
            zeta=zeta_endpoint,
            NFP=self.NFP,
            sym=True,
            endpoint=True,
        )
        cg = ConcentricGrid(L=self.L, M=self.M, N=self.N, NFP=self.NFP, sym=False)
        cg_sym = ConcentricGrid(L=self.L, M=self.M, N=self.N, NFP=self.NFP, sym=True)

        for label in ("rho", "theta", "zeta"):
            test(label, lg)
            test(label, lg_sym)
            test(label, lg_endpoint)
            test(label, lg_sym_endpoint)
            test(label, lg_2)
            test(label, lg_2_sym)
            test(label, lg_2_endpoint)
            test(label, lg_2_sym_endpoint)
            if label != "theta":
                # theta integrals are poorly defined on concentric grids
                test(label, cg)
                test(label, cg_sym)

    @pytest.mark.unit
    def test_line_length(self):
        """Test that line_integrals(dl) is 1 for rho, 2π for theta, zeta.

        This test should ensure that lines have the correct length on grids
        constructed by specifying L, M, N and by specifying an array of nodes.
        """

        def test(grid):
            if not isinstance(grid, ConcentricGrid):
                for theta_val in grid.nodes[grid.unique_theta_idx, 1]:
                    result = line_integrals(
                        grid,
                        line_label="rho",
                        fix_surface=("theta", theta_val),
                        expand_out=False,
                    )
                    np.testing.assert_allclose(result, 1)
                for rho_val in grid.nodes[grid.unique_rho_idx, 0]:
                    result = line_integrals(
                        grid,
                        line_label="zeta",
                        fix_surface=("rho", rho_val),
                        expand_out=False,
                    )
                    np.testing.assert_allclose(result, 2 * np.pi)
            for zeta_val in grid.nodes[grid.unique_zeta_idx, 2]:
                result = line_integrals(
                    grid,
                    line_label="theta",
                    fix_surface=("zeta", zeta_val),
                    expand_out=False,
                )
                np.testing.assert_allclose(result, 2 * np.pi)

        lg = LinearGrid(L=self.L, M=self.M, N=self.N, NFP=self.NFP, sym=False)
        lg_sym = LinearGrid(L=self.L, M=self.M, N=self.N, NFP=self.NFP, sym=True)
        rho = np.linspace(1, 0, self.L)[::-1]
        theta = np.linspace(0, 2 * np.pi, self.M, endpoint=False)
        zeta = np.linspace(0, 2 * np.pi / self.NFP, self.N, endpoint=False)
        lg_2 = LinearGrid(rho=rho, theta=theta, zeta=zeta, NFP=self.NFP, sym=False)
        lg_2_sym = LinearGrid(rho=rho, theta=theta, zeta=zeta, NFP=self.NFP, sym=True)
        cg = ConcentricGrid(L=self.L, M=self.M, N=self.N, NFP=self.NFP, sym=False)
        cg_sym = ConcentricGrid(L=self.L, M=self.M, N=self.N, NFP=self.NFP, sym=True)

        test(lg)
        test(lg_sym)
        test(lg_2)
        test(lg_2_sym)
        test(cg)
        test(cg_sym)

    @pytest.mark.unit
    def test_surface_averages_identity_op(self):
        """Test flux surface averages of surface functions are identity operations."""
        eq = get("W7-X")
        with pytest.warns(UserWarning, match="Reducing radial"):
            eq.change_resolution(3, 3, 3, 6, 6, 6)
        grid = ConcentricGrid(L=self.L, M=self.M, N=self.N, NFP=eq.NFP, sym=eq.sym)
        data = eq.compute(["p", "sqrt(g)"], grid=grid)
        pressure_average = surface_averages(grid, data["p"], data["sqrt(g)"])
        np.testing.assert_allclose(data["p"], pressure_average)

    @pytest.mark.unit
    def test_surface_averages_homomorphism(self):
        """Test flux surface averages of surface functions are additive homomorphisms.

        Meaning average(a + b) = average(a) + average(b).
        """
        eq = get("W7-X")
        with pytest.warns(UserWarning, match="Reducing radial"):
            eq.change_resolution(3, 3, 3, 6, 6, 6)
        grid = ConcentricGrid(L=self.L, M=self.M, N=self.N, NFP=eq.NFP, sym=eq.sym)
        data = eq.compute(["|B|", "|B|_t", "sqrt(g)"], grid=grid)
        a = surface_averages(grid, data["|B|"], data["sqrt(g)"])
        b = surface_averages(grid, data["|B|_t"], data["sqrt(g)"])
        a_plus_b = surface_averages(grid, data["|B|"] + data["|B|_t"], data["sqrt(g)"])
        np.testing.assert_allclose(a_plus_b, a + b)

    @pytest.mark.unit
    def test_surface_integrals_against_shortcut(self):
        """Test integration against less general methods."""
        grid = ConcentricGrid(L=self.L, M=self.M, N=self.N, NFP=self.NFP)
        ds = grid.spacing[:, :2].prod(axis=-1)
        # something arbitrary that will give different sum across surfaces
        q = np.arange(grid.num_nodes) ** 2
        # The predefined grids sort nodes in zeta surface chunks.
        # To compute a quantity local to a surface, we can reshape it into zeta
        # surface chunks and compute across the chunks.
        result = grid.expand(
            (ds * q).reshape((grid.num_zeta, -1)).sum(axis=-1),
            surface_label="zeta",
        )
        np.testing.assert_allclose(
            surface_integrals(grid, q, surface_label="zeta"),
            desired=result,
        )

    @pytest.mark.unit
    def test_surface_averages_against_shortcut(self):
        """Test averaging against less general methods."""
        # test on zeta surfaces
        grid = LinearGrid(L=self.L, M=self.M, N=self.N, NFP=self.NFP)
        # something arbitrary that will give different average across surfaces
        q = np.arange(grid.num_nodes) ** 2
        # The predefined grids sort nodes in zeta surface chunks.
        # To compute a quantity local to a surface, we can reshape it into zeta
        # surface chunks and compute across the chunks.
        mean = grid.expand(
            q.reshape((grid.num_zeta, -1)).mean(axis=-1),
            surface_label="zeta",
        )
        # number of nodes per surface
        n = grid.num_rho * grid.num_theta
        np.testing.assert_allclose(np.bincount(grid.inverse_zeta_idx), desired=n)
        ds = grid.spacing[:, :2].prod(axis=-1)
        np.testing.assert_allclose(
            surface_integrals(grid, q / ds, surface_label="zeta") / n,
            desired=mean,
        )
        np.testing.assert_allclose(
            surface_averages(grid, q, surface_label="zeta"),
            desired=mean,
        )

        # test on grids with a single rho surface
        eq = get("W7-X")
        with pytest.warns(UserWarning, match="Reducing radial"):
            eq.change_resolution(3, 3, 3, 6, 6, 6)
        rho = np.array((1 - 1e-4) * np.random.default_rng().random() + 1e-4)
        grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
        data = eq.compute(["|B|", "sqrt(g)"], grid=grid)
        np.testing.assert_allclose(
            surface_averages(grid, data["|B|"], data["sqrt(g)"]),
            np.mean(data["sqrt(g)"] * data["|B|"]) / np.mean(data["sqrt(g)"]),
            err_msg="average with sqrt(g) fail",
        )
        np.testing.assert_allclose(
            surface_averages(grid, data["|B|"]),
            np.mean(data["|B|"]),
            err_msg="average without sqrt(g) fail",
        )

    @pytest.mark.unit
    def test_symmetry_surface_average_1(self):
        """Test surface average of a symmetric function."""

        def test(grid):
            r = grid.nodes[:, 0]
            t = grid.nodes[:, 1]
            z = grid.nodes[:, 2] * grid.NFP
            true_surface_avg = 5
            function_of_rho = 1 / (r + 0.35)
            f = (
                true_surface_avg
                + np.cos(t)
                - 0.5 * np.cos(z)
                + 3 * np.cos(t) * np.cos(z) ** 2
                - 2 * np.sin(z) * np.sin(t)
            ) * function_of_rho
            np.testing.assert_allclose(
                surface_averages(grid, f),
                true_surface_avg * function_of_rho,
                rtol=1e-15,
                err_msg=type(grid),
            )

        # these tests should be run on relatively low resolution grids,
        # or at least low enough so that the asymmetric spacing test fails
        L = [3, 3, 5, 3]
        M = [3, 6, 5, 7]
        N = [2, 2, 2, 2]
        NFP = [5, 3, 5, 3]
        sym = np.array([True, True, False, False])
        # to test code not tested on grids made with M=.
        even_number = 4
        n_theta = even_number - sym

        # asymmetric spacing
        with pytest.raises(AssertionError):
            theta = 2 * np.pi * np.array([t**2 for t in np.linspace(0, 1, max(M))])
            test(LinearGrid(L=max(L), theta=theta, N=max(N), sym=False))

        for i in range(len(L)):
            test(LinearGrid(L=L[i], M=M[i], N=N[i], NFP=NFP[i], sym=sym[i]))
            test(LinearGrid(L=L[i], theta=n_theta[i], N=N[i], NFP=NFP[i], sym=sym[i]))
            test(
                LinearGrid(
                    L=L[i],
                    theta=np.linspace(0, 2 * np.pi, n_theta[i]),
                    N=N[i],
                    NFP=NFP[i],
                    sym=sym[i],
                )
            )
            test(
                LinearGrid(
                    L=L[i],
                    theta=np.linspace(0, 2 * np.pi, n_theta[i] + 1),
                    N=N[i],
                    NFP=NFP[i],
                    sym=sym[i],
                )
            )
            test(QuadratureGrid(L=L[i], M=M[i], N=N[i], NFP=NFP[i]))
            test(ConcentricGrid(L=L[i], M=M[i], N=N[i], NFP=NFP[i], sym=sym[i]))
            # nonuniform spacing when sym is False, but spacing is still symmetric
            test(
                LinearGrid(
                    L=L[i],
                    theta=np.linspace(0, np.pi, n_theta[i]),
                    N=N[i],
                    NFP=NFP[i],
                    sym=sym[i],
                )
            )
            test(
                LinearGrid(
                    L=L[i],
                    theta=np.linspace(0, np.pi, n_theta[i] + 1),
                    N=N[i],
                    NFP=NFP[i],
                    sym=sym[i],
                )
            )

    @pytest.mark.unit
    def test_symmetry_surface_average_2(self):
        """Tests that surface averages are correct using specified basis."""

        def test(grid, basis, true_avg=1):
            transform = Transform(grid, basis)

            # random data with specified average on each surface
            coeffs = np.random.rand(basis.num_modes)
            coeffs[np.all(basis.modes[:, 1:] == [0, 0], axis=1)] = 0
            coeffs[np.all(basis.modes == [0, 0, 0], axis=1)] = true_avg

            # compute average for each surface in grid
            values = transform.transform(coeffs)
            numerical_avg = surface_averages(grid, values, expand_out=False)
            np.testing.assert_allclose(
                # values closest to axis are never accurate enough
                numerical_avg[isinstance(grid, ConcentricGrid) :],
                true_avg,
                err_msg=str(type(grid)) + " " + str(grid.sym),
            )

        M = 5
        M_grid = 13
        test(
            QuadratureGrid(L=M_grid, M=M_grid, N=0), FourierZernikeBasis(L=M, M=M, N=0)
        )
        test(
            LinearGrid(L=M_grid, M=M_grid, N=0, sym=True),
            FourierZernikeBasis(L=M, M=M, N=0, sym="cos"),
        )
        test(
            ConcentricGrid(L=M_grid, M=M_grid, N=0), FourierZernikeBasis(L=M, M=M, N=0)
        )
        test(
            ConcentricGrid(L=M_grid, M=M_grid, N=0, sym=True),
            FourierZernikeBasis(L=M, M=M, N=0, sym="cos"),
        )

    @pytest.mark.unit
    def test_surface_variance(self):
        """Test correctness of variance against less general methods."""
        grid = LinearGrid(L=self.L, M=self.M, N=self.N, NFP=self.NFP)
        # something arbitrary that will give different variance across surfaces
        q = np.arange(grid.num_nodes) ** 2

        # Test weighted sample variance with different weights.
        # positive weights to prevent cancellations that may hide implementation error
        weights = np.cos(q) * np.sin(q) + 5
        biased = surface_variance(
            grid, q, weights, bias=True, surface_label="zeta", expand_out=False
        )
        unbiased = surface_variance(
            grid, q, weights, surface_label="zeta", expand_out=False
        )
        # The predefined grids sort nodes in zeta surface chunks.
        # To compute a quantity local to a surface, we can reshape it into zeta
        # surface chunks and compute across the chunks.
        chunks = q.reshape((grid.num_zeta, -1))
        # The ds weights are built into the surface variance function.
        # So weights for np.cov should be ds * weights. Since ds is constant on
        # LinearGrid, we need to get the same result if we don't multiply by ds.
        weights = weights.reshape((grid.num_zeta, -1))
        for i in range(grid.num_zeta):
            np.testing.assert_allclose(
                biased[i],
                desired=np.cov(chunks[i], bias=True, aweights=weights[i]),
            )
            np.testing.assert_allclose(
                unbiased[i],
                desired=np.cov(chunks[i], aweights=weights[i]),
            )

        # Test weighted sample variance converges to unweighted sample variance
        # when all weights are equal.
        chunks = grid.expand(chunks, surface_label="zeta")
        np.testing.assert_allclose(
            surface_variance(grid, q, np.e, bias=True, surface_label="zeta"),
            desired=chunks.var(axis=-1),
        )
        np.testing.assert_allclose(
            surface_variance(grid, q, np.e, surface_label="zeta"),
            desired=chunks.var(axis=-1, ddof=1),
        )

    @pytest.mark.unit
    def test_surface_min_max(self):
        """Test the surface_min and surface_max functions."""
        for grid_type in [LinearGrid, QuadratureGrid, ConcentricGrid]:
            grid = grid_type(L=self.L, M=self.M, N=self.N, NFP=self.NFP)
            rho = grid.nodes[:, 0]
            theta = grid.nodes[:, 1]
            zeta = grid.nodes[:, 2]
            # Make up an arbitrary function of the coordinates:
            B = (
                1.7
                + 0.4 * rho * np.cos(theta)
                + 0.8 * rho * rho * np.cos(2 * theta - 3 * zeta)
            )
            Bmax_alt = np.zeros(grid.num_rho)
            Bmin_alt = np.zeros(grid.num_rho)
            for j in range(grid.num_rho):
                mask = grid.inverse_rho_idx == j
                Bmax_alt[j] = np.max(B[mask])
                Bmin_alt[j] = np.min(B[mask])
            np.testing.assert_allclose(Bmax_alt, grid.compress(surface_max(grid, B)))
            np.testing.assert_allclose(Bmin_alt, grid.compress(surface_min(grid, B)))


class TestSingularities:
    """Tests for high order singular integration."""

    @pytest.mark.unit
    @pytest.mark.parametrize("interpolator", [FFTInterpolator, DFTInterpolator])
    def test_biest_interpolators(self, interpolator):
        """Test that FFT and DFT interpolation gives same result for standard grids."""
        grid = LinearGrid(0, *_f_2d_nyquist_freq())
        h_t = 2 * np.pi / grid.num_theta
        h_z = 2 * np.pi / grid.num_zeta / grid.NFP

        st = 3
        sz = 5
        q = 4
        r, w, _, _ = _get_quadrature_nodes(q)
        dt = st / 2 * h_t * r * np.sin(w)
        dz = sz / 2 * h_z * r * np.cos(w)

        interp = interpolator(grid, grid, st, sz, q)
        theta = grid.nodes[:, 1]
        zeta = grid.nodes[:, 2]
        f = _f_2d(theta, zeta)
        for i in range(dt.size):
            truth = _f_2d(theta + dt[i], zeta + dz[i])
            np.testing.assert_allclose(interp(f, i), truth)

    @pytest.mark.unit
    def test_singular_integral_greens_id(self):
        """Test high order singular integration using greens identity.

        Any harmonic function can be represented as the sum of a single layer and double
        layer potential:

        Φ(r) = -1/2π ∫ Φ(r) n⋅(r-r')/|r-r'|³ da + 1/2π ∫ dΦ/dn 1/|r-r'| da

        If we choose Φ(r) == 1, then we get

        1 + 1/2π ∫ n⋅(r-r')/|r-r'|³ da = 0

        So we integrate the kernel n⋅(r-r')/|r-r'|³ and can benchmark the residual.

        Hyperparams from Dhairya for greens ID test:

         M  q  Nu Nv   N=Nu*Nv    error
        13 10  15 13       195 0.246547
        13 10  30 13       390 0.0313301
        13 12  45 13       585 0.0022925
        13 12  60 13       780 0.00024359
        13 12  75 13       975 1.97686e-05
        19 16  90 19      1710 1.2541e-05
        19 16 105 19      1995 2.91152e-06
        19 18 120 19      2280 7.03463e-07
        19 18 135 19      2565 1.60672e-07
        25 20 150 25      3750 7.59613e-09
        31 22 210 31      6510 1.04357e-09
        37 24 240 37      8880 1.80728e-11
        43 28 300 43     12900 2.14129e-12
        """
        eq = Equilibrium()
        Nu = [13, 13, 13, 19, 19, 25, 37]
        Nv = [30, 45, 60, 90, 120, 150, 240]
        ss = [13, 13, 13, 19, 19, 25, 37]
        qs = [10, 12, 12, 16, 18, 20, 24]
        es = [0.4, 2e-2, 3e-3, 5e-5, 4e-6, 1e-6, 1e-9]

        for i in range(len(Nu)):
            grid = LinearGrid(M=Nu[i] // 2, N=Nv[i] // 2, NFP=eq.NFP)
            interpolator = FFTInterpolator(grid, grid, ss[i], ss[i], qs[i])
            data = eq.compute(
                _kernel_nr_over_r3.keys + ["|e_theta x e_zeta|"], grid=grid
            )
            err = singular_integral(
                data, data, "nr_over_r3", interpolator, chunk_size=50
            )
            np.testing.assert_array_less(np.abs(2 * np.pi + err), es[i])

        eq = get("W7-X")
        Nu = 100
        Nv = 100
        es = 6e-7
        grid = LinearGrid(M=Nu // 2, N=Nv // 2, NFP=eq.NFP)
        st, sz, q = heuristic_support_params(grid, best_ratio(data)[0])
        interpolator = FFTInterpolator(grid, grid, st, sz, q)
        data = eq.compute(_kernel_nr_over_r3.keys + ["|e_theta x e_zeta|"], grid=grid)
        err = singular_integral(data, data, "nr_over_r3", interpolator, chunk_size=50)
        np.testing.assert_array_less(np.abs(2 * np.pi + err), es)

    @pytest.mark.unit
    @pytest.mark.parametrize("interpolator", [FFTInterpolator, DFTInterpolator])
    def test_singular_integral_vac_estell(self, interpolator):
        """Test calculating Bplasma for vacuum estell, which should be near 0."""
        eq = get("ESTELL")
        grid = LinearGrid(M=25, N=25, NFP=eq.NFP)
        keys = [
            "K_vc",
            "B",
            "|B|^2",
            "R",
            "phi",
            "Z",
            "e^rho",
            "n_rho",
            "|e_theta x e_zeta|",
        ]
        data = eq.compute(keys, grid=grid)
        st, sz, q = heuristic_support_params(grid, best_ratio(data)[0])
        interp = interpolator(grid, grid, st, sz, q)
        Bplasma = virtual_casing_biot_savart(data, data, interp, chunk_size=50)
        # need extra factor of B/2 bc we're evaluating on plasma surface
        Bplasma += data["B"] / 2
        Bplasma = np.linalg.norm(Bplasma, axis=-1)
        # scale by total field magnitude
        B = Bplasma / np.linalg.norm(data["B"], axis=-1).mean()
        np.testing.assert_allclose(B, 0, atol=0.0054)

    @pytest.mark.unit
    def test_laplace_harmonic_simple(self, chunk_size=2000, resolution=50, atol=1e-4):
        """Test that Laplace solution recovers expected analytic result.

        Define boundary R_b(θ,ζ) = R₀ + a cos θ and Z_b(θ,ζ) = -a sin θ.
        θ = 0 is outboard side and θ increases clockwise.
        Define harmonic map Φ: ρ,θ,ζ ↦ Z(ρ,θ,ζ).
        Choose b.c. B₀⋅n = -∇ϕ⋅n
                         = -[0, 0, 1]⋅[cos(θ)cos(ζ), cos(θ)sin(ζ), -sin(θ)] = sin(θ)
        and test that ‖ Φ − Z ‖_∞ → 0.
        """
        a = 1
        eq = Equilibrium()  # Choosing a = 1.
        evl_grid = LinearGrid(M=5, N=5, NFP=eq.NFP)
        phi_grid = LinearGrid(M=1, N=0, NFP=eq.NFP)
        src_grid = LinearGrid(M=resolution, N=resolution, NFP=eq.NFP)

        src_theta = src_grid.nodes[:, 1]
        B0n = np.sin(src_theta)
        laplace = compute_laplace(eq, B0n, evl_grid, phi_grid, src_grid)
        np.testing.assert_allclose(laplace["src_data"]["Z"], -a * np.sin(src_theta))
        np.testing.assert_allclose(-laplace["src_data"]["n_rho"][:, 2], B0n, atol=1e-12)

        Phi_mn = _compute_Phi_mn(laplace, chunk_size, warn_fft=False)
        Phi = Transform(evl_grid, laplace["phi_transform"].basis).transform(Phi_mn)
        np.testing.assert_allclose(np.ptp(Phi - laplace["evl_data"]["Z"]), 0, atol=atol)

        B0n = np.sin(evl_grid.nodes[:, 1])
        dPhi_dn = _compute_dPhi_dn(laplace, Phi_mn, chunk_size, warn_fft=False)
        Bn = B0n + dPhi_dn
        np.testing.assert_allclose(Bn, 0, atol=0.03)

    @pytest.mark.unit
    def test_laplace_harmonic_general(self, chunk_size=1000, resolution=70, atol=1e-3):
        """Test that Laplace solution recovers expected analytic result.

        Define boundary R_b(θ,ζ) and Z_b(θ,ζ).
        Define harmonic map Φ: ρ,θ,ζ ↦ Z(ρ,θ,ζ).
        Choose b.c. B₀⋅n = -∇ϕ⋅n and test that ‖ Φ − Z ‖_∞ → 0.
        """
        eq = Equilibrium(
            # elliptic cross-section with torsion
            surface=FourierRZToroidalSurface(
                R_lmn=[10, 1, 0.2],
                Z_lmn=[-2, -0.2],
                modes_R=[[0, 0], [1, 0], [0, 1]],
                modes_Z=[[-1, 0], [0, -1]],
            )
        )
        evl_grid = LinearGrid(M=min(20, resolution), N=min(20, resolution), NFP=eq.NFP)
        phi_grid = evl_grid
        src_grid = LinearGrid(M=resolution, N=resolution, NFP=eq.NFP)
        B0n = -eq.compute("n_rho", grid=src_grid)["n_rho"][:, 2]
        laplace = compute_laplace(eq, B0n, evl_grid, phi_grid, src_grid)

        Phi_mn = _compute_Phi_mn(laplace, chunk_size, warn_fft=False)
        Phi = Transform(evl_grid, laplace["phi_transform"].basis).transform(Phi_mn)
        np.testing.assert_allclose(np.ptp(Phi - laplace["evl_data"]["Z"]), 0, atol=atol)

        B0n = -laplace["evl_data"]["n_rho"][:, 2]
        dPhi_dn = _compute_dPhi_dn(laplace, Phi_mn, chunk_size, warn_fft=False)
        Bn = B0n + dPhi_dn
        np.testing.assert_allclose(Bn, 0, atol=0.03)

    @pytest.mark.unit
    @pytest.mark.parametrize("eq", [get("ESTELL")])
    def test_laplace_bdotn_toroidal(
        self, eq, chunk_size=1000, resolution=60, atol=1e-3, plot=True
    ):
        """Test that Laplace solution satisfies toroidal field boundary condition."""
        source_grid = LinearGrid(M=resolution, N=resolution, NFP=eq.NFP)
        source_data = eq.compute(["G", "R0"], grid=source_grid)
        B0 = ToroidalMagneticField(
            B0=source_grid.compress(source_data["G"])[-1] / source_data["R0"],
            R0=source_data["R0"],
        )
        B0n, _ = B0.compute_Bnormal(
            eq.surface,
            eval_grid=source_grid,
            source_grid=source_grid,
            vc_source_grid=source_grid,
            chunk_size=chunk_size,
        )

        Phi_grid = LinearGrid(M=min(20, resolution), N=min(20, resolution), NFP=eq.NFP)
        Phi_mn, Phi_transform = _compute_Phi_mn(
            eq=eq,
            B0n=B0n,
            Phi_grid=Phi_grid,
            source_grid=source_grid,
            chunk_size=chunk_size,
            warn_fft=False,
        )
        eval_grid = Phi_grid
        dPhi_dn = _compute_dPhi_dn(
            eq=eq,
            eval_grid=eval_grid,
            source_grid=source_grid,
            Phi_mn=Phi_mn,
            basis=Phi_transform.basis,
            chunk_size=chunk_size,
            warn_fft=False,
        )
        B0n, _ = B0.compute_Bnormal(
            eq.surface,
            eval_grid=eval_grid,
            source_grid=source_grid,
            vc_source_grid=source_grid,
            chunk_size=chunk_size,
        )
        Bn = B0n + dPhi_dn

        if plot:
            Bn = eval_grid.meshgrid_reshape(Bn, "rtz")[0]
            theta = eval_grid.meshgrid_reshape(eval_grid.nodes[:, 1], "rtz")[0]
            zeta = eval_grid.meshgrid_reshape(eval_grid.nodes[:, 2], "rtz")[0]
            fig, ax = plt.subplots()
            contour = ax.contourf(theta, zeta, Bn)
            fig.colorbar(contour, ax=ax)
            ax.set_title(r"$(\nabla \Phi + B_0) \cdot n$ on $\partial D$")
            plt.show()

        np.testing.assert_allclose(Bn, 0, atol=atol)

    @staticmethod
    def manual_transform(coef, m, n, theta, zeta):
        """Evaluates Double Fourier Series of form G_n^m at theta and zeta pts."""
        op_four = np.where(
            ((m < 0) & (n < 0))[:, np.newaxis],
            np.sin(np.abs(m)[:, np.newaxis] * theta)
            * np.sin(np.abs(n)[:, np.newaxis] * zeta),
            n[:, np.newaxis] * zeta * np.nan,
        )
        op_three = np.where(
            ((m < 0) & (n >= 0))[:, np.newaxis],
            np.sin(np.abs(m)[:, np.newaxis] * theta) * np.cos(n[:, np.newaxis] * zeta),
            op_four,
        )
        op_two = np.where(
            ((m >= 0) & (n < 0))[:, np.newaxis],
            np.cos(m[:, np.newaxis] * theta) * np.sin(np.abs(n)[:, np.newaxis] * zeta),
            op_three,
        )
        op_one = np.where(
            ((m >= 0) & (n >= 0))[:, np.newaxis],
            np.cos(m[:, np.newaxis] * theta) * np.cos(n[:, np.newaxis] * zeta),
            op_two,
        )
        return np.sum(coef[:, np.newaxis] * op_one, axis=0)

    @staticmethod
    def merkel_transform(coef, m, n, theta, zeta, fun=np.cos):
        """Evaluates double Fourier series of form cos(m theta + n zeta)."""
        return np.sum(
            coef[:, np.newaxis]
            * fun(m[:, np.newaxis] * theta + n[:, np.newaxis] * zeta),
            axis=0,
        )

    @staticmethod
    def _merkel_surf(C_r, C_z):
        """Convert merkel coefficients to DESC coefficients and check."""
        m_b = max(
            max(abs(mn[0]) for mn in C_r.keys()), max(abs(mn[0]) for mn in C_z.keys())
        )
        n_b = max(
            max(abs(mn[1]) for mn in C_r.keys()), max(abs(mn[1]) for mn in C_z.keys())
        )
        R_lmn = {
            (m, n): 0.0 for m in range(-m_b, m_b + 1) for n in range(-n_b, n_b + 1)
        }
        Z_lmn = {
            (m, n): 0.0 for m in range(-m_b, m_b + 1) for n in range(-n_b, n_b + 1)
        }
        for m in range(0, m_b + 1):
            for n in range(-n_b, n_b + 1):
                if (m, n) in C_r:
                    R_lmn[(m, abs(n))] += C_r[(m, n)]
                    if n < 0:
                        R_lmn[(-m, n)] += C_r[(m, n)]
                    elif n > 0:
                        R_lmn[(-m, -n)] -= C_r[(m, n)]
                if (m, n) in C_z:
                    Z_lmn[(-m, abs(n))] += C_z[(m, n)]
                    if n < 0:
                        Z_lmn[(m, n)] -= C_z[(m, n)]
                    elif n > 0:
                        Z_lmn[(m, -n)] += C_z[(m, n)]

        grid = LinearGrid(rho=1, M=5, N=5)
        R_bench = TestSingularities.manual_transform(
            np.array(list(R_lmn.values())),
            np.array([mn[0] for mn in R_lmn.keys()]),
            np.array([mn[1] for mn in R_lmn.keys()]),
            -grid.nodes[:, 1],  # theta is flipped
            grid.nodes[:, 2],
        )
        R_merk = TestSingularities.merkel_transform(
            np.array(list(C_r.values())),
            np.array([mn[0] for mn in C_r.keys()]),
            np.array([mn[1] for mn in C_r.keys()]),
            -grid.nodes[:, 1],  # theta is flipped
            grid.nodes[:, 2],
        )
        Z_bench = TestSingularities.manual_transform(
            np.array(list(Z_lmn.values())),
            np.array([mn[0] for mn in Z_lmn.keys()]),
            np.array([mn[1] for mn in Z_lmn.keys()]),
            -grid.nodes[:, 1],  # theta is flipped
            grid.nodes[:, 2],
        )
        Z_merk = TestSingularities.merkel_transform(
            np.array(list(C_z.values())),
            np.array([mn[0] for mn in C_z.keys()]),
            np.array([mn[1] for mn in C_z.keys()]),
            -grid.nodes[:, 1],  # theta is flipped
            grid.nodes[:, 2],
            fun=np.sin,
        )
        np.testing.assert_allclose(R_bench, R_merk)
        np.testing.assert_allclose(Z_bench, Z_merk)
        surf = FourierRZToroidalSurface(
            R_lmn=list(R_lmn.values()),
            Z_lmn=list(Z_lmn.values()),
            modes_R=list(R_lmn.keys()),
            modes_Z=list(Z_lmn.keys()),
        )
        surf_data = surf.compute(["R", "Z"], grid=grid)
        np.testing.assert_allclose(surf_data["R"], R_merk)
        np.testing.assert_allclose(surf_data["Z"], Z_merk)
        return surf

    @pytest.mark.unit
    def test_laplace_dommaschk(self):
        """Use Dommaschk potentials to generate benchmark test and compare."""
        C_r = {
            (0, -2): 0.000056,
            (0, -1): -0.000921,
            (0, 0): 0.997922,
            (0, 1): -0.000921,
            (0, 2): 0.000056,
            (1, -2): -0.000067,
            (1, -1): -0.034645,
            (1, 0): 0.093260,
            (1, 1): 0.000880,
            (1, 2): 0.000178,
            (2, -2): 0.000373,
            (2, -1): 0.000575,
            (2, 0): 0.002916,
            (2, 1): -0.000231,
            (2, 2): 0.000082,
            (3, -2): 0.000462,
            (3, -1): -0.001509,
            (3, 0): 0.001748,
            (3, 1): -0.000239,
            (3, 2): 0.000052,
        }
        C_z = {
            (0, -2): 0.000076,
            (0, -1): 0.000923,
            (0, 0): 0.000000,
            (0, 1): -0.000923,
            (0, 2): -0.000076,
            (1, -2): 0.000069,
            (1, -1): 0.035178,
            (1, 0): 0.099830,
            (1, 1): 0.000860,
            (1, 2): -0.000179,
            (2, -2): -0.000374,
            (2, -1): 0.000257,
            (2, 0): 0.003096,
            (2, 1): 0.003021,
            (2, 2): 0.000007,
            (3, -2): -0.000518,
            (3, -1): 0.002233,
            (3, 0): 0.001828,
            (3, 1): 0.000257,
            (3, 2): 0.000035,
        }
        surf = self._merkel_surf(C_r, C_z)
        eq = Equilibrium(surface=surf)
        plot_boundary(eq, phi=[0, 1, 2])
        plt.show()
        return self.test_laplace_bdotn_toroidal(eq)


class TestBouncePoints:
    """Test that bounce points are computed correctly."""

    @staticmethod
    def _cheb_intersect(cheb, k):
        cheb = cheb.copy()
        cheb[0] = cheb[0] - k
        roots = chebroots(cheb)
        intersect = roots[np.logical_and(np.isreal(roots), np.abs(roots.real) < 1)].real
        return intersect

    @staticmethod
    def filter(z1, z2):
        """Remove bounce points whose integrals have zero measure."""
        mask = (z1 - z2) != 0.0
        return z1[mask], z2[mask]

    @pytest.mark.unit
    def test_z1_first(self):
        """Case where straight line through first two intersects is in epigraph."""
        start = np.pi / 3
        end = 6 * np.pi
        k = np.linspace(start, end, 5)
        B = CubicHermiteSpline(k, np.cos(k), -np.sin(k))
        pitch_inv = 0.5
        intersect = B.solve(pitch_inv, extrapolate=False)
        z1, z2 = bounce_points(pitch_inv, k, B.c.T, B.derivative().c.T)
        _check_bounce_points(z1, z2, pitch_inv, k, B.c.T, plot=True, include_knots=True)
        z1, z2 = TestBouncePoints.filter(z1, z2)
        assert z1.size and z2.size
        np.testing.assert_allclose(z1, intersect[0::2])
        np.testing.assert_allclose(z2, intersect[1::2])

    @pytest.mark.unit
    def test_z2_first(self):
        """Case where straight line through first two intersects is in hypograph."""
        start = -3 * np.pi
        end = -start
        k = np.linspace(start, end, 5)
        B = CubicHermiteSpline(k, np.cos(k), -np.sin(k))
        pitch_inv = 0.5
        intersect = B.solve(pitch_inv, extrapolate=False)
        z1, z2 = bounce_points(pitch_inv, k, B.c.T, B.derivative().c.T)
        _check_bounce_points(z1, z2, pitch_inv, k, B.c.T, plot=True, include_knots=True)
        z1, z2 = TestBouncePoints.filter(z1, z2)
        assert z1.size and z2.size
        np.testing.assert_allclose(z1, intersect[1:-1:2])
        np.testing.assert_allclose(z2, intersect[0::2][1:])

    @pytest.mark.unit
    def test_z1_before_extrema(self):
        """Case where local maximum is the shared intersect between two wells."""
        # To make sure both regions in epigraph left and right of extrema are
        # integrated over.
        start = -np.pi
        end = -2 * start
        k = np.linspace(start, end, 5)
        B = CubicHermiteSpline(
            k, np.cos(k) + 2 * np.sin(-2 * k), -np.sin(k) - 4 * np.cos(-2 * k)
        )
        dB_dz = B.derivative()
        pitch_inv = B(dB_dz.roots(extrapolate=False))[3] - 1e-13
        z1, z2 = bounce_points(pitch_inv, k, B.c.T, dB_dz.c.T)
        _check_bounce_points(z1, z2, pitch_inv, k, B.c.T, plot=True, include_knots=True)
        z1, z2 = TestBouncePoints.filter(z1, z2)
        assert z1.size and z2.size
        intersect = B.solve(pitch_inv, extrapolate=False)
        np.testing.assert_allclose(z1[1], 1.982767, rtol=1e-6)
        np.testing.assert_allclose(z1, intersect[[1, 2]], rtol=1e-6)
        # intersect array could not resolve double root as single at index 2,3
        np.testing.assert_allclose(intersect[2], intersect[3], rtol=1e-6)
        np.testing.assert_allclose(z2, intersect[[3, 4]], rtol=1e-6)

    @pytest.mark.unit
    def test_z2_before_extrema(self):
        """Case where local minimum is the shared intersect between two wells."""
        # To make sure both regions in hypograph left and right of extrema are not
        # integrated over.
        start = -1.2 * np.pi
        end = -2 * start
        k = np.linspace(start, end, 7)
        B = CubicHermiteSpline(
            k,
            np.cos(k) + 2 * np.sin(-2 * k) + k / 4,
            -np.sin(k) - 4 * np.cos(-2 * k) + 1 / 4,
        )
        dB_dz = B.derivative()
        pitch_inv = B(dB_dz.roots(extrapolate=False))[2]
        z1, z2 = bounce_points(pitch_inv, k, B.c.T, dB_dz.c.T)
        _check_bounce_points(z1, z2, pitch_inv, k, B.c.T, plot=True, include_knots=True)
        z1, z2 = TestBouncePoints.filter(z1, z2)
        assert z1.size and z2.size
        intersect = B.solve(pitch_inv, extrapolate=False)
        np.testing.assert_allclose(z1, intersect[[0, -2]])
        np.testing.assert_allclose(z2, intersect[[1, -1]])

    @pytest.mark.unit
    def test_extrema_first_and_before_z1(self):
        """Case where first intersect is extrema and second enters epigraph."""
        # To make sure we don't perform integral between first pair of intersects.
        start = -1.2 * np.pi
        end = -2 * start
        k = np.linspace(start, end, 7)
        B = CubicHermiteSpline(
            k,
            np.cos(k) + 2 * np.sin(-2 * k) + k / 20,
            -np.sin(k) - 4 * np.cos(-2 * k) + 1 / 20,
        )
        dB_dz = B.derivative()
        pitch_inv = B(dB_dz.roots(extrapolate=False))[2] + 1e-13
        z1, z2 = bounce_points(pitch_inv, k[2:], B.c[:, 2:].T, dB_dz.c[:, 2:].T)
        _check_bounce_points(
            z1,
            z2,
            pitch_inv,
            k[2:],
            B.c[:, 2:].T,
            plot=True,
            include_knots=True,
            start=k[2],
        )
        z1, z2 = TestBouncePoints.filter(z1, z2)
        assert z1.size and z2.size
        intersect = B.solve(pitch_inv, extrapolate=False)
        np.testing.assert_allclose(z1[0], 0.835319, rtol=1e-6)
        intersect = intersect[intersect >= k[2]]
        np.testing.assert_allclose(z1, intersect[[0, 2, 4]], rtol=1e-6)
        np.testing.assert_allclose(z2, intersect[[0, 3, 5]], rtol=1e-6)

    @pytest.mark.unit
    def test_extrema_first_and_before_z2(self):
        """Case where first intersect is extrema and second exits epigraph."""
        # To make sure we do perform integral between first pair of intersects.
        start = -1.2 * np.pi
        end = -2 * start + 1
        k = np.linspace(start, end, 7)
        B = CubicHermiteSpline(
            k,
            np.cos(k) + 2 * np.sin(-2 * k) + k / 10,
            -np.sin(k) - 4 * np.cos(-2 * k) + 1 / 10,
        )
        dB_dz = B.derivative()
        pitch_inv = B(dB_dz.roots(extrapolate=False))[1] - 1e-13
        z1, z2 = bounce_points(pitch_inv, k, B.c.T, dB_dz.c.T)
        _check_bounce_points(z1, z2, pitch_inv, k, B.c.T, plot=True, include_knots=True)
        z1, z2 = TestBouncePoints.filter(z1, z2)
        assert z1.size and z2.size
        # Our routine correctly detects intersection, while scipy, jnp.root fails.
        intersect = B.solve(pitch_inv, extrapolate=False)
        np.testing.assert_allclose(z1[0], -0.671904, rtol=1e-6)
        np.testing.assert_allclose(z1, intersect[[0, 3, 5]], rtol=1e-5)
        # intersect array could not resolve double root as single at index 0,1
        np.testing.assert_allclose(intersect[0], intersect[1], rtol=1e-5)
        np.testing.assert_allclose(z2, intersect[[2, 4, 6]], rtol=1e-5)

    @pytest.mark.unit
    def test_get_extrema(self):
        """Test computation of extrema of |B|."""
        start = -np.pi
        end = -2 * start
        k = np.linspace(start, end, 5)
        B = CubicHermiteSpline(
            k, np.cos(k) + 2 * np.sin(-2 * k), -np.sin(k) - 4 * np.cos(-2 * k)
        )
        dB_dz = B.derivative()
        ext, B_ext = _get_extrema(k, B.c.T, dB_dz.c.T)
        mask = ~np.isnan(ext)
        ext, B_ext = ext[mask], B_ext[mask]
        idx = np.argsort(ext)

        ext_scipy = np.sort(dB_dz.roots(extrapolate=False))
        B_ext_scipy = B(ext_scipy)
        assert ext.size == ext_scipy.size
        np.testing.assert_allclose(ext[idx], ext_scipy)
        np.testing.assert_allclose(B_ext[idx], B_ext_scipy)

    @pytest.mark.unit
    def test_z1_first_chebyshev(self):
        """Test that bounce points are computed correctly."""

        def f(z):
            return -2 * np.cos(1 / (0.1 + z**2)) + 2

        M, N = 1, 10
        alpha, zeta = FourierChebyshevSeries.nodes(M, N).T
        cheb = FourierChebyshevSeries(f(zeta).reshape(M, N)).compute_cheb(
            fourier_pts(M)
        )
        pitch_inv = 3
        z1, z2 = cheb.intersect1d(pitch_inv)
        cheb.check_intersect1d(z1, z2, pitch_inv)
        z1, z2 = TestBouncePoints.filter(z1, z2)

        r = self._cheb_intersect(chebinterpolate(f, N - 1), pitch_inv)
        np.testing.assert_allclose(z1, r[np.isclose(r, -0.24, atol=1e-1)])
        np.testing.assert_allclose(z2, r[np.isclose(r, 0.24, atol=1e-1)])


auto_sin = (automorphism_sin, grad_automorphism_sin)


class TestBounceQuadrature:
    """Test bounce quadrature."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "is_strong, quad, automorphism",
        [
            (True, tanh_sinh(30), None),
            (False, tanh_sinh(20), None),
            (True, leggauss(25), auto_sin),
            # chebgauss1 without c.o.v. is sensitive to approximation error
            (True, chebgauss1(25), auto_sin),
            (False, leggauss_lob(8, interior_only=True), auto_sin),
            (False, chebgauss2(21), None),
        ],
    )
    def test_bounce_quadrature(self, is_strong, quad, automorphism):
        """Test quadrature matches singular (strong and weak) elliptic integrals."""
        # Some prime number that doesn't appear anywhere in calculation.
        # Ensures no lucky cancellation occurs from ζ₂ − ζ₁ / π = π / (ζ₂ − ζ₁)
        # which could mask errors since π appears often in transformations.
        v = 3
        knots = np.linspace(-np.pi / 2 * v, np.pi / 2 * v, 50)
        B = np.sin(knots / v) ** 2 + 1
        dB_dz = np.sin(2 * knots / v) / v
        bounce = Bounce1D(
            Grid.create_meshgrid([1, 0, knots], coordinates="raz"),
            data={"B^zeta": B, "B^zeta_z|r,a": dB_dz, "|B|": B, "|B|_z|r,a": dB_dz},
            quad=quad,
            automorphism=automorphism,
            check=True,
        )

        m = 1 - 1e-3
        # integral(integrand) = truth iff pitch_inv = 2
        pitch_inv = 2 - 1e-12
        k = pitch_inv * m  # m = k * pitch
        if is_strong:
            integrand = lambda data, B, pitch: 1 / jnp.sqrt(1 - k * pitch * (B - 1))
            truth = v * 2 * ellipkm1(1 - m)
        else:
            integrand = lambda data, B, pitch: jnp.sqrt(1 - k * pitch * (B - 1))
            truth = v * 2 * ellipe(m)
        np.testing.assert_allclose(
            bounce.integrate(integrand, pitch_inv, check=True, plot=True).sum(),
            truth,
            rtol=1e-4,
        )

    @staticmethod
    @partial(np.vectorize, excluded={0})
    def _adaptive_elliptic(integrand, k):
        a = 0
        b = 2 * np.arcsin(k)
        return integrate.quad(
            integrand, a, b, args=(k,), points=b, epsrel=1e-10, epsabs=1e-10
        )[0]

    @staticmethod
    def _fixed_elliptic(integrand, k, deg):
        k = np.atleast_1d(k)
        a = -2 * np.arcsin(k)
        b = 2 * np.arcsin(k)
        x, w = get_quadrature(leggauss(deg), (automorphism_sin, grad_automorphism_sin))
        Z = bijection_from_disc(x, a[..., np.newaxis], b[..., np.newaxis])
        k = k[..., np.newaxis]
        quad = integrand(Z, k).dot(w) * grad_bijection_from_disc(a, b) / 2
        return quad

    @staticmethod
    def elliptic_incomplete(k2):
        """Calculate elliptic integrals for bounce averaged binormal drift.

        The test is nice because it is independent of all the bounce integrals
        and splines. One can test performance of different quadrature methods
        by using that method in the ``_fixed_elliptic`` method above.

        """
        K_integrand = lambda Z, k: 2 / np.sqrt(k**2 - np.sin(Z / 2) ** 2) * (k / 4)
        E_integrand = lambda Z, k: 2 * np.sqrt(k**2 - np.sin(Z / 2) ** 2) / (k * 4)
        # Scipy's elliptic integrals are broken.
        # https://github.com/scipy/scipy/issues/20525.
        k = np.sqrt(k2)
        K = TestBounceQuadrature._adaptive_elliptic(K_integrand, k)
        E = TestBounceQuadrature._adaptive_elliptic(E_integrand, k)
        # Make sure scipy's adaptive quadrature is not broken.
        np.testing.assert_allclose(
            K, TestBounceQuadrature._fixed_elliptic(K_integrand, k, 12)
        )
        np.testing.assert_allclose(
            E, TestBounceQuadrature._fixed_elliptic(E_integrand, k, 10)
        )

        E0 = ellipe(k2)
        K0 = ellipk(k2)

        I_00 = 4 / k * K  # Incomplete integral
        I_0 = 4 * K0  # Complete integral

        I_10 = 4 * k * E
        I_1 = 4 * (E0 + (k2 - 1) * K0)

        I_20 = 16 * k * E
        I_2 = 16 * (E0 + (k2 - 1) * K0)

        I_30 = 16 * k / 9 * (2 * (-1 + 2 * k2) * E - (-1 + k2) * K)
        I_3 = 16 / 9 * (2 * (-1 + 2 * k2) * (E0 + (k2 - 1) * K0) - (-1 + k2) * k2 * K0)

        I_40 = 16 * k / 3 * ((-1 + 2 * k2) * E - 2 * (-1 + k2) * K)
        I_4 = 16 / 3 * ((-1 + 2 * k2) * E0 - (-1 + k2) * K0)

        I_50 = 32 * k / 30 * (2 * (1 - k2 + k2**2) * E - (1 - 3 * k2 + 2 * k2**2) * K)
        I_5 = (
            32
            / 30
            * (
                2 * (1 - k2 + k2**2) * (E0 + (k2 - 1) * K0)
                - (1 - 3 * k2 + 2 * k2**2) * k2 * K0
            )
        )

        I_60 = 4 / k * (2 * k2 * E + (1 - 2 * k2) * K)
        I_6 = 4 * (2 * E0 - K0)

        I_70 = 2 * k / 3 * ((-2 + 4 * k2) * E - 4 * (-1 + k2) * K)
        I_7 = 4 / 3 * ((2 * k2 - 1) * E0 - (k2 - 1) * K0)

        # Check if incomplete integral expressions match the complete integrals
        np.testing.assert_allclose(I_0, I_00)
        np.testing.assert_allclose(I_1, I_10)
        np.testing.assert_allclose(I_2, I_20)
        np.testing.assert_allclose(I_3, I_30)
        np.testing.assert_allclose(I_4, I_40)
        np.testing.assert_allclose(I_5, I_50)
        np.testing.assert_allclose(I_6, I_60)
        np.testing.assert_allclose(I_7, I_70)

        # Check for math mistakes.
        np.testing.assert_allclose(
            I_2,
            TestBounceQuadrature._adaptive_elliptic(
                lambda Z, k: 2 / np.sqrt(k**2 - np.sin(Z / 2) ** 2) * Z * np.sin(Z), k
            ),
        )
        np.testing.assert_allclose(
            I_3,
            TestBounceQuadrature._adaptive_elliptic(
                lambda Z, k: 2 * np.sqrt(k**2 - np.sin(Z / 2) ** 2) * Z * np.sin(Z), k
            ),
        )
        np.testing.assert_allclose(
            I_4,
            TestBounceQuadrature._adaptive_elliptic(
                lambda Z, k: 2 / np.sqrt(k**2 - np.sin(Z / 2) ** 2) * np.sin(Z) ** 2, k
            ),
        )
        np.testing.assert_allclose(
            I_5,
            TestBounceQuadrature._adaptive_elliptic(
                lambda Z, k: 2 * np.sqrt(k**2 - np.sin(Z / 2) ** 2) * np.sin(Z) ** 2, k
            ),
        )
        # scipy fails
        np.testing.assert_allclose(
            I_6,
            TestBounceQuadrature._fixed_elliptic(
                lambda Z, k: 2 / np.sqrt(k**2 - np.sin(Z / 2) ** 2) * np.cos(Z),
                k,
                deg=12,
            ),
        )
        np.testing.assert_allclose(
            I_7,
            TestBounceQuadrature._adaptive_elliptic(
                lambda Z, k: 2 * np.sqrt(k**2 - np.sin(Z / 2) ** 2) * np.cos(Z), k
            ),
        )
        return I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7


class TestBounce:
    """Test bounce integration with one-dimensional local spline methods."""

    @staticmethod
    def _example_numerator(data, B, pitch):
        f = (1 - 0.5 * pitch * B) * data["g_zz"]
        return safediv(f, jnp.sqrt(jnp.abs(1 - pitch * B)))

    @staticmethod
    def _example_denominator(data, B, pitch):
        return safediv(1, jnp.sqrt(jnp.abs(1 - pitch * B)))

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d * 4)
    def test_bounce1d_checks(self):
        """Test that all the internal correctness checks pass for real example."""
        # noqa: D202
        # Suppose we want to compute a bounce average of the function
        # f(ℓ) = (1 − λ|B|/2) * g_zz, where g_zz is the squared norm of the
        # toroidal basis vector on some set of field lines specified by (ρ, α)
        # coordinates. This is defined as
        # [∫ f(ℓ) / √(1 − λ|B|) dℓ] / [∫ 1 / √(1 − λ|B|) dℓ]

        # 1. Define python functions for the integrands. We do that above.
        # 2. Pick flux surfaces, field lines, and how far to follow the field
        #    line in Clebsch coordinates ρ, α, ζ.
        rho = np.linspace(0.1, 1, 6)
        alpha = np.array([0, 0.5])
        zeta = np.linspace(-2 * np.pi, 2 * np.pi, 200)
        eq = get("HELIOTRON")

        # 3. Convert above coordinates to DESC computational coordinates.
        grid = get_rtz_grid(eq, rho, alpha, zeta, coordinates="raz")
        # 4. Compute input data.
        data = eq.compute(
            Bounce1D.required_names + ["min_tz |B|", "max_tz |B|", "g_zz"], grid=grid
        )
        # 5. Make the bounce integration operator.
        bounce = Bounce1D(grid.source_grid, data, check=True)
        pitch_inv, _ = bounce.get_pitch_inv_quad(
            min_B=grid.compress(data["min_tz |B|"]),
            max_B=grid.compress(data["max_tz |B|"]),
            num_pitch=10,
        )
        # 6. Compute bounce points.
        points = bounce.points(pitch_inv)
        # 7. Optionally check for correctness of bounce points.
        bounce.check_points(points, pitch_inv, plot=False)
        # 8. Integrate.
        num = bounce.integrate(
            integrand=TestBounce._example_numerator,
            pitch_inv=pitch_inv,
            data={"g_zz": Bounce1D.reshape(grid.source_grid, data["g_zz"])},
            points=points,
            check=True,
        )
        den = bounce.integrate(
            integrand=TestBounce._example_denominator,
            pitch_inv=pitch_inv,
            points=points,
            check=True,
            batch=False,
        )
        avg = safediv(num, den)
        errorif(not np.isfinite(avg).all())
        errorif(
            np.count_nonzero(avg) == 0,
            msg="Detected 0 wells on this cut of the field line. Make sure enough "
            "toroidal transits were followed for this test, or plot the field line "
            "to see if this is expected.",
        )

        # 9. Example manipulation of the output
        # Sum all bounce averages on a particular field line, for every field line.
        result = avg.sum(axis=-1)
        # The result stored at
        l, m, p = 1, 0, 3
        print("Result(ρ, α, λ):", result[l, m, p])
        # corresponds to the 1/λ value
        print("1/λ(ρ, λ):", pitch_inv[l, p])
        # for the Clebsch field line coordinates
        nodes = bounce.reshape(grid.source_grid, grid.source_grid.nodes[:, :2])
        print("(ρ, α):", nodes[l, m, 0])

        # 10. Plotting
        fig, ax = bounce.plot(l, m, pitch_inv[l], include_legend=False, show=False)
        return fig

    @pytest.mark.unit
    def test_interp_to_argmin(self):
        """Test interpolation of h to argmin g."""  # noqa: D202

        # Test functions chosen with purpose; don't change unless plotted and compared.
        def h(z):
            return np.cos(3 * z) * np.sin(2 * np.cos(z)) + np.cos(1.2 * z)

        def g(z):
            return np.sin(3 * z) * np.cos(1 / (1 + z)) * np.cos(z**2) * z

        def dg_dz(z):
            return (
                3 * z * np.cos(3 * z) * np.cos(z**2) * np.cos(1 / (1 + z))
                - 2 * z**2 * np.sin(3 * z) * np.sin(z**2) * np.cos(1 / (1 + z))
                + z * np.sin(3 * z) * np.sin(1 / (1 + z)) * np.cos(z**2) / (1 + z) ** 2
                + np.sin(3 * z) * np.cos(z**2) * np.cos(1 / (1 + z))
            )

        argmin_g = 5.61719
        zeta = np.linspace(0, 3 * np.pi, 175)
        data = dict.fromkeys(Bounce1D.required_names, g(zeta))
        data["|B|_z|r,a"] = dg_dz(zeta)
        bounce = Bounce1D(Grid.create_meshgrid([1, 0, zeta], coordinates="raz"), data)
        points = np.array(0, ndmin=2), np.array(2 * np.pi, ndmin=2)
        np.testing.assert_allclose(
            bounce.interp_to_argmin(h(zeta), points), h(argmin_g), rtol=1e-3
        )

    @staticmethod
    def get_drift_analytic_data():
        """Get data to compute bounce averaged binormal drift analytically."""
        eq = Equilibrium.load(".//tests//inputs//low-beta-shifted-circle.h5")
        psi_boundary = eq.Psi / (2 * np.pi)
        psi = 0.25 * psi_boundary
        rho = np.sqrt(psi / psi_boundary)
        np.testing.assert_allclose(rho, 0.5)

        # Make a set of nodes along a single fieldline.
        grid_fsa = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, sym=eq.sym, NFP=eq.NFP)
        data = eq.compute(["iota"], grid=grid_fsa)
        iota = grid_fsa.compress(data["iota"]).item()
        alpha = 0
        zeta = np.linspace(-np.pi / iota, np.pi / iota, (2 * eq.M_grid) * 4 + 1)
        grid = get_rtz_grid(eq, rho, alpha, zeta, coordinates="raz", iota=iota)
        data = eq.compute(
            Bounce1D.required_names
            + [
                "cvdrift",
                "gbdrift",
                "grad(psi)",
                "grad(alpha)",
                "shear",
                "iota",
                "psi",
                "a",
                "theta_PEST",
            ],
            grid=grid,
        )
        np.testing.assert_allclose(data["psi"], psi)
        np.testing.assert_allclose(data["iota"], iota)
        assert np.all(data["B^zeta"] > 0)
        data["Bref"] = 2 * np.abs(psi_boundary) / data["a"] ** 2
        data["rho"] = rho
        data["alpha"] = alpha
        data["zeta"] = zeta
        data["psi"] = grid.compress(data["psi"])
        data["iota"] = grid.compress(data["iota"])
        data["shear"] = grid.compress(data["shear"])

        things = {"grid": grid, "eq": eq}
        return data, things

    @staticmethod
    def drift_analytic(data):
        """Compute analytic approximation for bounce-averaged binormal drift.

        Returns
        -------
        drift_analytic : jnp.ndarray
            Analytic approximation for the true result that the numerical computation
            should attempt to match.
        cvdrift, gbdrift : jnp.ndarray
            Numerically computed ``data["cvdrift"]` and ``data["gbdrift"]`` normalized
            by some scale factors for this unit test. These should be fed to the bounce
            integration as input.
        pitch_inv : jnp.ndarray
            Shape (P, ).
            1/λ values used.

        """
        B = data["|B|"] / data["Bref"]
        B0 = np.mean(B)
        # epsilon should be changed to dimensionless, and computed in a way that
        # is independent of normalization length scales, like "effective r/R0".
        epsilon = data["a"] * data["rho"]  # Aspect ratio of the flux surface.
        np.testing.assert_allclose(epsilon, 0.05)
        # same as 1 / (1 + epsilon cos(theta)) assuming epsilon << 1
        B_analytic = B0 * (1 - epsilon * np.cos(data["theta_PEST"]))
        np.testing.assert_allclose(B, B_analytic, atol=3e-3)

        gradpar = data["a"] * data["B^zeta"] / data["|B|"]
        # This method of computing G0 suggests a fixed point iteration.
        G0 = data["a"]
        gradpar_analytic = G0 * (1 - epsilon * np.cos(data["theta_PEST"]))
        gradpar_theta_analytic = data["iota"] * gradpar_analytic
        G0 = np.mean(gradpar_theta_analytic)
        np.testing.assert_allclose(gradpar, gradpar_analytic, atol=5e-3)

        # Comparing coefficient calculation here with coefficients from compute/_metric
        data["normalization"] = -np.sign(data["psi"]) * data["Bref"] * data["a"] ** 2
        cvdrift = data["cvdrift"] * data["normalization"]
        gbdrift = data["gbdrift"] * data["normalization"]
        dPdrho = np.mean(-0.5 * (cvdrift - gbdrift) * data["|B|"] ** 2)
        alpha_MHD = -0.5 * dPdrho / data["iota"] ** 2
        gds21 = (
            -np.sign(data["iota"])
            * data["shear"]
            * dot(data["grad(psi)"], data["grad(alpha)"])
            / data["Bref"]
        )
        gds21_analytic = -data["shear"] * (
            data["shear"] * data["theta_PEST"]
            - alpha_MHD / B**4 * np.sin(data["theta_PEST"])
        )
        gds21_analytic_low_order = -data["shear"] * (
            data["shear"] * data["theta_PEST"]
            - alpha_MHD / B0**4 * np.sin(data["theta_PEST"])
        )
        np.testing.assert_allclose(gds21, gds21_analytic, atol=2e-2)
        np.testing.assert_allclose(gds21, gds21_analytic_low_order, atol=2.7e-2)

        fudge_1 = 0.19
        gbdrift_analytic = fudge_1 * (
            -data["shear"]
            + np.cos(data["theta_PEST"])
            - gds21_analytic / data["shear"] * np.sin(data["theta_PEST"])
        )
        gbdrift_analytic_low_order = fudge_1 * (
            -data["shear"]
            + np.cos(data["theta_PEST"])
            - gds21_analytic_low_order / data["shear"] * np.sin(data["theta_PEST"])
        )
        fudge_2 = 0.07
        cvdrift_analytic = gbdrift_analytic + fudge_2 * alpha_MHD / B**2
        cvdrift_analytic_low_order = (
            gbdrift_analytic_low_order + fudge_2 * alpha_MHD / B0**2
        )
        np.testing.assert_allclose(gbdrift, gbdrift_analytic, atol=1e-2)
        np.testing.assert_allclose(cvdrift, cvdrift_analytic, atol=2e-2)
        np.testing.assert_allclose(gbdrift, gbdrift_analytic_low_order, atol=1e-2)
        np.testing.assert_allclose(cvdrift, cvdrift_analytic_low_order, atol=2e-2)

        # Exclude singularity not captured by analytic approximation for pitch near
        # the maximum |B|. (This is captured by the numerical integration).
        pitch_inv = Bounce1D.get_pitch_inv_quad(np.min(B), np.max(B), 100)[0][:-1]
        k2 = 0.5 * ((1 - B0 / pitch_inv) / (epsilon * B0 / pitch_inv) + 1)
        I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7 = (
            TestBounceQuadrature.elliptic_incomplete(k2)
        )
        y = np.sqrt(2 * epsilon * B0 / pitch_inv)
        I_0, I_2, I_4, I_6 = map(lambda I: I / y, (I_0, I_2, I_4, I_6))
        I_1, I_3, I_5, I_7 = map(lambda I: I * y, (I_1, I_3, I_5, I_7))

        drift_analytic_num = (
            fudge_2 * alpha_MHD / B0**2 * I_1
            - 0.5
            * fudge_1
            * (
                data["shear"] * (I_0 + I_1 - I_2 - I_3)
                + alpha_MHD / B0**4 * (I_4 + I_5)
                - (I_6 + I_7)
            )
        ) / G0
        drift_analytic_den = I_0 / G0
        drift_analytic = drift_analytic_num / drift_analytic_den
        return drift_analytic, cvdrift, gbdrift, pitch_inv

    @staticmethod
    def drift_num_integrand(data, B, pitch):
        """Integrand of numerator of bounce averaged binormal drift."""
        g = jnp.sqrt(jnp.abs(1 - pitch * B))
        return (
            (data["cvdrift"] * g)
            - (0.5 * g * data["gbdrift"])
            + (0.5 * data["gbdrift"] / g)
        )

    @staticmethod
    def drift_den_integrand(data, B, pitch):
        """Integrand of denominator of bounce averaged binormal drift."""
        return 1 / jnp.sqrt(jnp.abs(1 - pitch * B))

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
    def test_binormal_drift_bounce1d(self):
        """Test bounce-averaged drift with analytical expressions."""
        data, things = TestBounce.get_drift_analytic_data()
        drift_analytic, cvdrift, gbdrift, pitch_inv = TestBounce.drift_analytic(data)

        bounce = Bounce1D(
            things["grid"].source_grid,
            data,
            Bref=data["Bref"],
            Lref=data["a"],
            check=True,
        )
        points = bounce.points(pitch_inv, num_well=1)
        bounce.check_points(points, pitch_inv, plot=False)
        interp_data = {
            "cvdrift": Bounce1D.reshape(things["grid"].source_grid, cvdrift),
            "gbdrift": Bounce1D.reshape(things["grid"].source_grid, gbdrift),
        }
        drift_numerical_num = bounce.integrate(
            integrand=TestBounce.drift_num_integrand,
            pitch_inv=pitch_inv,
            data=interp_data,
            points=points,
            check=True,
        )
        drift_numerical_den = bounce.integrate(
            integrand=TestBounce.drift_den_integrand,
            pitch_inv=pitch_inv,
            points=points,
            check=True,
        )
        drift_numerical = np.squeeze(drift_numerical_num / drift_numerical_den)
        assert np.isfinite(drift_numerical).all()
        msg = "There should be one bounce integral per pitch in this example."
        assert drift_numerical.size == drift_analytic.size, msg
        np.testing.assert_allclose(
            drift_numerical, drift_analytic, atol=5e-3, rtol=5e-2
        )

        TestBounce._test_bounce_autodiff(
            bounce, TestBounce.drift_num_integrand, interp_data
        )

        fig, ax = plt.subplots()
        ax.plot(pitch_inv, drift_analytic)
        ax.plot(pitch_inv, drift_numerical)
        return fig

    @staticmethod
    def _test_bounce_autodiff(bounce, integrand, data):
        """Make sure reverse mode AD works correctly on this algorithm.

        Non-differentiable operations (e.g. ``take_mask``) are used in computation.
        See https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
        and https://jax.readthedocs.io/en/latest/faq.html#
        why-are-gradients-zero-for-functions-based-on-sort-order.

        If the AD tool works properly, then these operations should be assigned
        zero gradients while the gradients wrt parameters of our physics computations
        accumulate correctly. Less mature AD tools may have subtle bugs that cause
        the gradients to not accumulate correctly. (There's a few
        GitHub issues that JAX has fixed related to this in the past.)

        This test first confirms the gradients computed by reverse mode AD matches
        the analytic approximation of the true gradient. Then we confirm that the
        partial gradients wrt the integrand and bounce points are correct.

        Apply the Leibniz integral rule
        https://en.wikipedia.org/wiki/Leibniz_integral_rule, with
        the label w summing over the magnetic wells:

        ∂_λ ∑_w ∫_ζ₁^ζ₂ f dζ  (λ) = ∑_w [
             ∫_ζ₁^ζ₂ (∂f/∂λ)(λ) dζ
             + f(λ,ζ₂) (∂ζ₂/∂λ)(λ)
             - f(λ,ζ₁) (∂ζ₁/∂λ)(λ)
        ]
        where (∂ζ₁/∂λ)(λ) = -λ² / (∂|B|/∂ζ|ρ,α)(ζ₁)
              (∂ζ₂/∂λ)(λ) = -λ² / (∂|B|/∂ζ|ρ,α)(ζ₂)

        All terms in these expressions are known analytically.
        If we wanted, it's simple to check explicitly that AD takes each derivative
        correctly because |w| = 1 is constant and our tokamak has symmetry
        (∂|B|/∂ζ|ρ,α)(ζ₁) = - (∂|B|/∂ζ|ρ,α)(ζ₂).

        After confirming the left hand side is correct, we just check that derivative
        wrt bounce points of the right hand side doesn't vanish due to some zero
        gradient issue mentioned above.

        """

        def integrand_grad(data, B, pitch):
            grad_fun = grad(integrand, -1)
            for _ in range(B.ndim):
                grad_fun = vmap(grad_fun)
            return grad_fun(data, B, jnp.broadcast_to(pitch, B.shape))

        def fun1(pitch):
            return bounce.integrate(
                integrand=integrand, pitch_inv=1 / pitch, data=data, check=False
            ).sum()

        def fun2(pitch):
            return bounce.integrate(
                integrand=integrand_grad, pitch_inv=1 / pitch, data=data, check=True
            ).sum()

        pitch = 1.0
        # can obtain from math or just extrapolate from analytic expression plot
        analytic_approximation_of_gradient = 650
        np.testing.assert_allclose(
            grad(fun1)(pitch), analytic_approximation_of_gradient, rtol=5e-3
        )
        # It is expected that this is much larger because the integrand is singular
        # wrt λ but the boundary derivative: f(λ,ζ₂) (∂ζ₂/∂λ)(λ) - f(λ,ζ₁) (∂ζ₁/∂λ)(λ).
        # smooths out because the bounce points ζ₁ and ζ₂ are smooth functions of λ.
        np.testing.assert_allclose(fun2(pitch), -171500, rtol=1e-1)


class TestBounce2D:
    """Test bounce integration that uses 2D pseudo-spectral methods."""

    @pytest.mark.unit
    def test_interp_to_argmin(self):
        """Test interpolation of h to argmin of g."""  # noqa: D202

        def g(z):
            return np.cos(z) ** 2 + np.sin(7 * z) - np.sin(z)

        argmin_g = np.pi / 2
        h = _f_1d
        nyquist = 2 * max(_f_1d_nyquist_freq(), 7) + 1
        grid = LinearGrid(theta=1, zeta=nyquist, sym=False)
        bounce = Bounce2D(
            grid,
            dict.fromkeys(Bounce2D.required_names, g(grid.nodes[:, 2])),
            # dummy value; h depends on ζ alone,so doesn't matter what θ(α, ζ) is
            theta=Bounce2D.reshape(grid, grid.nodes[:, 1]),
            Y_B=2 * nyquist,
            num_transit=1,
        )
        points = np.array(0, ndmin=2), np.array(2 * np.pi, ndmin=2)
        np.testing.assert_allclose(
            bounce.interp_to_argmin(bounce.reshape(grid, h(grid.nodes[:, 2])), points),
            h(argmin_g),
            rtol=1e-6,
        )

    @pytest.mark.slow
    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d * 4)
    def test_bounce2d_checks(self):
        """Test that all the internal correctness checks pass for real example."""
        # noqa: D202
        # Suppose we want to compute a bounce average of the function
        # f(ℓ) = (1 − λ|B|/2) * g_zz, where g_zz is the squared norm of the
        # toroidal basis vector on some set of field lines specified by (ρ, α)
        # coordinates. This is defined as
        # [∫ f(ℓ) / √(1 − λ|B|) dℓ] / [∫ 1 / √(1 − λ|B|) dℓ]

        # 1. Define python functions for the integrands. We do that above.
        # 2. Pick flux surfaces and grid resolution.
        rho = np.linspace(0.1, 1, 6)
        alpha = np.array([0, 0.5])
        eq = get("HELIOTRON")
        grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
        # 3. Compute input data.
        data = eq.compute(
            Bounce2D.required_names + ["min_tz |B|", "max_tz |B|", "g_zz", "1"],
            grid=grid,
        )
        # 4. Compute DESC coordinates of optimal interpolation nodes.
        theta = Bounce2D.compute_theta(eq, X=16, Y=64, rho=rho)
        # 5. Make the bounce integration operator.
        bounce = Bounce2D(
            grid,
            data,
            theta,
            alpha=alpha,
            num_transit=2,
            check=True,
            spline=False,
        )
        pitch_inv, _ = bounce.get_pitch_inv_quad(
            min_B=grid.compress(data["min_tz |B|"]),
            max_B=grid.compress(data["max_tz |B|"]),
            num_pitch=10,
        )
        # 6. Compute bounce points.
        points = bounce.points(pitch_inv)
        # 7. Optionally check for correctness of bounce points.
        bounce.check_points(points, pitch_inv, plot=False)
        # 8. Integrate.
        num = bounce.integrate(
            integrand=TestBounce._example_numerator,
            pitch_inv=pitch_inv,
            data={"g_zz": Bounce2D.reshape(grid, data["g_zz"])},
            points=points,
            check=True,
        )
        den = bounce.integrate(
            integrand=TestBounce._example_denominator,
            pitch_inv=pitch_inv,
            points=points,
            check=True,
        )
        avg = safediv(num, den)
        errorif(not np.isfinite(avg).all())
        errorif(
            np.count_nonzero(avg) == 0,
            msg="Detected 0 wells on this cut of the field line. Make sure enough "
            "toroidal transits were followed for this test, or plot the field line "
            "to see if this is expected.",
        )

        # 9. Example manipulation of the output
        # Sum all bounce averages on a particular field line, for every field line.
        result = avg.sum(axis=-1)
        # The result stored at
        l, m, p = 1, 0, 3
        print("Result(ρ, α, λ):", result[l, m, p])
        # corresponds to the 1/λ value
        print("1/λ(ρ, λ):", pitch_inv[l, p])
        # for the Clebsch field line coordinates
        print("(ρ, α):", rho[l], alpha[m])

        np.testing.assert_allclose(
            bounce.compute_fieldline_length(),
            # Crossref w/ "fieldline length" in data index with 1000 points.
            [
                385.20520905,
                361.94836747,
                345.64342394,
                333.29485717,
                352.15451128,
                440.10036239,
            ],
            rtol=1e-2,
        )

        # 10. Plotting
        fig, ax = bounce.plot(l, m, pitch_inv[l], include_legend=False, show=False)
        _, _ = bounce.plot_theta(l, m, show=False)

        # make sure tests pass when spline=True
        b = Bounce2D(
            grid, data, theta, alpha=alpha, num_transit=2, check=True, spline=True
        )
        b.check_points(b.points(pitch_inv), pitch_inv, plot=False)
        _, _ = b.plot(l, m, pitch_inv[l], show=False)

        return fig

    @staticmethod
    def drift_num_integrand(data, B, pitch):
        """Integrand of numerator of bounce averaged binormal drift."""
        g = jnp.sqrt(jnp.abs(1 - pitch * B))
        cvdrift = (
            data["cvdrift (periodic)"] + data["gbdrift (secular)/phi"] * data["zeta"]
        )
        gbdrift = (
            data["gbdrift (periodic)"] + data["gbdrift (secular)/phi"] * data["zeta"]
        )
        return (cvdrift * g) - (0.5 * g * gbdrift) + (0.5 * gbdrift / g)

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
    def test_binormal_drift_bounce2d(self):
        """Test bounce-averaged drift with analytical expressions."""
        data, things = TestBounce.get_drift_analytic_data()
        drift_analytic, _, _, pitch_inv = TestBounce.drift_analytic(data)

        eq = things["eq"]
        grid = LinearGrid(
            rho=data["rho"], M=eq.M_grid, N=max(1, eq.N_grid), NFP=eq.NFP, sym=False
        )
        names = ["cvdrift (periodic)", "gbdrift (periodic)", "gbdrift (secular)/phi"]
        grid_data = eq.compute(names=Bounce2D.required_names + names, grid=grid)
        for name in names:
            grid_data[name] = grid_data[name] * data["normalization"]

        bounce = Bounce2D(
            grid,
            grid_data,
            Bounce2D.compute_theta(eq, X=8, Y=8, rho=data["rho"], iota=data["iota"]),
            alpha=data["alpha"] - 2.5 * np.pi * data["iota"],
            num_transit=3,
            Bref=data["Bref"],
            Lref=data["a"],
            check=True,
            spline=False,
        )
        points = bounce.points(pitch_inv, num_well=1)
        bounce.check_points(points, pitch_inv, plot=False)
        interp_data = {name: Bounce2D.reshape(grid, grid_data[name]) for name in names}
        drift_numerical_num = bounce.integrate(
            integrand=TestBounce2D.drift_num_integrand,
            pitch_inv=pitch_inv,
            data=interp_data,
            points=points,
            check=True,
        )
        drift_numerical_den = bounce.integrate(
            integrand=TestBounce.drift_den_integrand,
            pitch_inv=pitch_inv,
            points=points,
            check=True,
        )
        drift_numerical = np.squeeze(drift_numerical_num / drift_numerical_den)
        assert np.isfinite(drift_numerical).all()
        msg = "There should be one bounce integral per pitch in this example."
        assert drift_numerical.size == drift_analytic.size, msg
        np.testing.assert_allclose(
            drift_numerical, drift_analytic, atol=5e-3, rtol=5e-2
        )

        TestBounce._test_bounce_autodiff(
            bounce, TestBounce2D.drift_num_integrand, interp_data
        )

        fig, ax = plt.subplots()
        ax.plot(pitch_inv, drift_analytic)
        ax.plot(pitch_inv, drift_numerical)
        return fig
