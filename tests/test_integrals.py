"""Test integration algorithms."""

import numpy as np
import pytest

from desc.basis import FourierZernikeBasis
from desc.equilibrium import Equilibrium
from desc.examples import get
from desc.grid import ConcentricGrid, LinearGrid, QuadratureGrid
from desc.integrals import (
    DFTInterpolator,
    FFTInterpolator,
    _get_grid_surface,
    _get_quadrature_nodes,
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
from desc.transform import Transform


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
    """Tests for high order singular integration.

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

    @pytest.mark.unit
    def test_singular_integral_greens_id(self):
        """Test high order singular integration using greens identity.

        Any harmonic function can be represented as the sum of a single layer and double
        layer potential:

        Φ(r) = -1/2π ∫ Φ(r) n⋅(r-r')/|r-r'|³ da + 1/2π ∫ dΦ/dn 1/|r-r'| da

        If we choose Φ(r) == 1, then we get

        1 + 1/2π ∫ n⋅(r-r')/|r-r'|³ da = 0

        So we integrate the kernel n⋅(r-r')/|r-r'|³ and can benchmark the residual.

        """
        eq = Equilibrium()
        Nv = np.array([30, 45, 60, 90, 120, 150, 240])
        Nu = np.array([13, 13, 13, 19, 19, 25, 37])
        ss = np.array([13, 13, 13, 19, 19, 25, 37])
        qs = np.array([10, 12, 12, 16, 18, 20, 24])
        es = np.array([0.4, 2e-2, 3e-3, 5e-5, 4e-6, 1e-6, 1e-9])
        eval_grid = LinearGrid(M=5, N=6, NFP=eq.NFP)

        for i, (m, n) in enumerate(zip(Nu, Nv)):
            source_grid = LinearGrid(M=m // 2, N=n // 2, NFP=eq.NFP)
            source_data = eq.compute(
                ["R", "Z", "phi", "e^rho", "|e_theta x e_zeta|"], grid=source_grid
            )
            eval_data = eq.compute(
                ["R", "Z", "phi", "e^rho", "|e_theta x e_zeta|"], grid=eval_grid
            )
            s = ss[i]
            q = qs[i]
            interpolator = FFTInterpolator(eval_grid, source_grid, s, q)

            err = singular_integral(
                eval_data,
                source_data,
                "nr_over_r3",
                interpolator,
                loop=True,
            )
            np.testing.assert_array_less(np.abs(2 * np.pi + err), es[i])

    @pytest.mark.unit
    def test_singular_integral_vac_estell(self):
        """Test calculating Bplasma for vacuum estell, which should be near 0."""
        eq = get("ESTELL")
        eval_grid = LinearGrid(M=8, N=8, NFP=eq.NFP)

        source_grid = LinearGrid(M=18, N=18, NFP=eq.NFP)

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

        source_data = eq.compute(keys, grid=source_grid)
        eval_data = eq.compute(keys, grid=eval_grid)

        k = min(source_grid.num_theta, source_grid.num_zeta)
        s = k // 2 + int(np.sqrt(k))
        q = k // 2 + int(np.sqrt(k))

        interpolator = FFTInterpolator(eval_grid, source_grid, s, q)
        Bplasma = virtual_casing_biot_savart(
            eval_data,
            source_data,
            interpolator,
            loop=True,
        )
        # need extra factor of B/2 bc we're evaluating on plasma surface
        Bplasma += eval_data["B"] / 2
        Bplasma = np.linalg.norm(Bplasma, axis=-1)
        # scale by total field magnitude
        B = Bplasma / np.mean(np.linalg.norm(eval_data["B"], axis=-1))
        # this isn't a perfect vacuum equilibrium (|J| ~ 1e3 A/m^2), so increasing
        # resolution of singular integral won't really make Bplasma less.
        np.testing.assert_array_less(B, 0.05)

    @pytest.mark.unit
    def test_biest_interpolators(self):
        """Test that FFT and DFT interpolation gives same result for standard grids."""
        sgrid = LinearGrid(0, 5, 6)
        egrid = LinearGrid(0, 4, 7)
        s = 3
        q = 4
        r, w, dr, dw = _get_quadrature_nodes(q)
        interp1 = FFTInterpolator(egrid, sgrid, s, q)
        interp2 = DFTInterpolator(egrid, sgrid, s, q)

        f = lambda t, z: np.sin(4 * t) + np.cos(3 * z)

        source_dtheta = sgrid.spacing[:, 1]
        source_dzeta = sgrid.spacing[:, 2] / sgrid.NFP
        source_theta = sgrid.nodes[:, 1]
        source_zeta = sgrid.nodes[:, 2]
        eval_theta = egrid.nodes[:, 1]
        eval_zeta = egrid.nodes[:, 2]

        h_t = np.mean(source_dtheta)
        h_z = np.mean(source_dzeta)

        for i in range(len(r)):
            dt = s / 2 * h_t * r[i] * np.sin(w[i])
            dz = s / 2 * h_z * r[i] * np.cos(w[i])
            theta_i = eval_theta + dt
            zeta_i = eval_zeta + dz
            ff = f(theta_i, zeta_i)

            g1 = interp1(f(source_theta, source_zeta), i)
            g2 = interp2(f(source_theta, source_zeta), i)
            np.testing.assert_allclose(g1, g2)
            np.testing.assert_allclose(g1, ff)
