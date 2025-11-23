"""Regression tests for plotting functions, by comparing to saved baseline images."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

from desc.basis import (
    ChebyshevDoubleFourierBasis,
    ChebyshevPolynomial,
    DoubleFourierSeries,
    FourierSeries,
    FourierZernikeBasis,
    PowerSeries,
    ZernikePolynomial,
)
from desc.coils import CoilSet, FourierXYZCoil, MixedCoilSet
from desc.compute import data_index
from desc.examples import get
from desc.geometry import FourierRZToroidalSurface, FourierXYZCurve
from desc.grid import ConcentricGrid, Grid, LinearGrid, QuadratureGrid
from desc.integrals import surface_averages
from desc.io import load
from desc.magnetic_fields import (
    OmnigenousField,
    PoloidalMagneticField,
    SumMagneticField,
    ToroidalMagneticField,
)
from desc.plotting import (
    plot_1d,
    plot_2d,
    plot_3d,
    plot_basis,
    plot_boozer_modes,
    plot_boozer_surface,
    plot_boundaries,
    plot_boundary,
    plot_coefficients,
    plot_coils,
    plot_comparison,
    plot_field_lines,
    plot_fsa,
    plot_gammac,
    plot_grid,
    plot_logo,
    plot_qs_error,
    plot_section,
    plot_surfaces,
    poincare_plot,
)
from desc.utils import isalmostequal, xyz2rpz

tol_1d = 4.5
tol_2d = 10
tol_3d = 10


@pytest.mark.unit
def test_kwarg_warning(DummyStellarator):
    """Test that passing in unknown kwargs throws an error."""
    eq = load(load_from=str(DummyStellarator["output_path"]))
    with pytest.raises(AssertionError):
        fig, ax = plot_1d(eq, "psi_rr", not_a_kwarg=True)
    return None


@pytest.mark.unit
def test_kwarg_future_warning(DummyStellarator):
    """Test that passing in deprecated kwargs throws a warning."""
    eq = load(load_from=str(DummyStellarator["output_path"]))
    with pytest.raises(FutureWarning):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fig, ax = plot_surfaces(eq, zeta=2)
    return None


class TestPlot1D:
    """Tests for plot_1d."""

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
    def test_1d_p(self):
        """Test plotting 1d pressure profile."""
        eq = get("SOLOVEV")
        fig, ax, data = plot_1d(eq, "p", figsize=(4, 4), return_data=True)
        assert "p" in data.keys()
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
    def test_1d_elongation(self):
        """Test plotting 1d elongation as a function of toroidal angle."""
        eq = get("precise_QA")
        grid = LinearGrid(M=eq.M_grid, N=20, NFP=eq.NFP)
        fig, ax, data = plot_1d(
            eq, "a_major/a_minor", grid=grid, figsize=(4, 4), return_data=True
        )
        assert "a_major/a_minor" in data.keys()
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
    def test_1d_iota(self):
        """Test plotting 1d rotational transform."""
        eq = get("DSHAPE_current")
        grid = LinearGrid(rho=0.5, theta=100, zeta=0.0)
        fig, ax, data = plot_1d(eq, "iota", grid=grid, figsize=(4, 4), return_data=True)
        assert "theta" in data.keys()
        assert "iota" in data.keys()
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
    def test_1d_iota_radial(self):
        """Test plotting 1d rotational transform."""
        eq = get("DSHAPE_current")
        fig, ax, data = plot_1d(eq, "iota", figsize=(4, 4), return_data=True)
        assert "rho" in data.keys()
        assert "iota" in data.keys()
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
    def test_1d_logpsi(self):
        """Test plotting 1d flux function with log scale."""
        eq = get("DSHAPE_current")
        fig, ax, data = plot_1d(eq, "psi", log=True, figsize=(4, 4), return_data=True)
        ax.set_ylim([1e-5, 1e0])
        assert "rho" in data.keys()
        assert "psi" in data.keys()
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
    def test_plot_1d_curve(self):
        """Test plot_1d function for Curve objects."""
        curve = FourierXYZCurve([0, 10, 1])
        fig, ax = plot_1d(curve, "curvature")
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
    def test_plot_1d_surface(self):
        """Test plot_1d function for Surface objects."""
        surf = FourierRZToroidalSurface()
        fig, ax = plot_1d(surf, "curvature_H_rho", grid=LinearGrid(M=50))
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
    def test_plot_1d_normalized(self):
        """Test plotting normalized flux surface average <|B|> on log scale."""
        eq = get("DSHAPE_CURRENT")
        fig, ax = plot_1d(eq, "<|B|>", log=True, normalize="<|B|>_vol")
        ax.set_ylim([9e-1, 1.1e0])
        return fig


class TestPlot2D:
    """Tests for plot_2d."""

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=10)
    def test_2d_logF(self):
        """Test plotting 2d force error with log scale."""
        eq = get("DSHAPE_CURRENT")
        grid = LinearGrid(rho=100, theta=100, zeta=0.0)
        fig, ax, data = plot_2d(
            eq, "|F|", log=True, grid=grid, figsize=(4, 4), return_data=True
        )
        assert "|F|" in data.keys()
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_2d_g_tz(self):
        """Test plotting 2d metric coefficients vs theta/zeta."""
        eq = get("DSHAPE_CURRENT")
        grid = LinearGrid(rho=0.5, theta=100, zeta=100)
        fig, ax, data = plot_2d(
            eq, "sqrt(g)", grid=grid, figsize=(4, 4), return_data=True
        )
        assert "theta" in data.keys()
        assert "zeta" in data.keys()

        assert "sqrt(g)" in data.keys()
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_2d_g_rz(self):
        """Test plotting 2d metric coefficients vs rho/zeta."""
        eq = get("DSHAPE_CURRENT")
        grid = LinearGrid(rho=100, theta=0.0, zeta=100)
        fig, ax, data = plot_2d(
            eq, "sqrt(g)", grid=grid, figsize=(4, 4), return_data=True
        )
        assert "rho" in data.keys()
        assert "zeta" in data.keys()
        assert "sqrt(g)" in data.keys()

        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_plot_con_basis(self):
        """Test 2d plot of R component of e^rho."""
        eq = get("DSHAPE_CURRENT")
        fig, ax, data = plot_2d(
            eq, "e^rho", component="R", figsize=(4, 4), return_data=True
        )
        for string in ["e^rho", "normalization", "theta", "zeta"]:
            assert string in data.keys()
        assert data["normalization"] == 1

        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_plot_cov_basis(self):
        """Test 2d plot of norm of e_rho."""
        eq = get("DSHAPE_CURRENT")
        fig, ax, data = plot_2d(eq, "e_rho", figsize=(4, 4), return_data=True)
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_plot_normF_2d(self):
        """Test 2d plot of normalized force."""
        grid = LinearGrid(rho=np.array(0.8), M=20, N=2)
        eq = get("DSHAPE_CURRENT")
        fig, ax, data = plot_2d(
            eq, "|F|_normalized", figsize=(4, 4), return_data=True, grid=grid
        )
        for string in ["|F|_normalized", "theta", "zeta"]:
            assert string in data.keys()
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_plot_normF_2d_deprecated(self):
        """Test deprecated 2d plot of normalized force."""
        grid = LinearGrid(rho=np.array(0.8), M=20, N=2)
        eq = get("DSHAPE_CURRENT")
        with pytest.raises((ValueError, FutureWarning)):
            _, _ = plot_2d(eq, "|F|", norm_F=True, normalize="<|grad(|B|^2)|/2mu0>_vol")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            fig, ax = plot_2d(eq, "|F|", figsize=(4, 4), norm_F=True, grid=grid)
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_plot_2d_surface(self):
        """Test plot_2d function for Surface objects."""
        surf = FourierRZToroidalSurface()
        fig, ax = plot_2d(surf, "curvature_H_rho")
        return fig

    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    @pytest.mark.unit
    def test_2d_plot_Bn(self):
        """Test 2d plotting of Bn on equilibrium surface."""
        eq = get("HELIOTRON")
        fig, _ = plot_2d(
            eq,
            "B*n",
            field=ToroidalMagneticField(1, 1),
            field_grid=LinearGrid(M=10, N=10),
            grid=LinearGrid(M=30, N=30, NFP=eq.NFP, endpoint=True),
        )
        return fig


class TestPlot3D:
    """Tests for plot_3d."""

    @pytest.mark.unit
    def test_3d_tz(self):
        """Test 3d plot of force on interior surface."""
        eq = get("DSHAPE_CURRENT")
        grid = LinearGrid(rho=0.5, theta=100, zeta=100)
        fig, data = plot_3d(eq, "|F|", log=True, grid=grid, return_data=True)
        assert "X" in data.keys()
        assert "Y" in data.keys()
        assert "Z" in data.keys()

        assert "|F|" in data.keys()

    @pytest.mark.unit
    def test_3d_tz_normalized(self):
        """Test 3d plot of normalized force on interior surface."""
        eq = get("DSHAPE_CURRENT")
        grid = LinearGrid(rho=0.5, theta=100, zeta=100)
        fig, data = plot_3d(
            eq,
            "|F|",
            log=True,
            grid=grid,
            return_data=True,
            normalize="<|grad(p)|>_vol",
        )
        assert "X" in data.keys()
        assert "Y" in data.keys()
        assert "Z" in data.keys()

        assert "|F|" in data.keys()
        assert "normalization" in data.keys()

    @pytest.mark.unit
    def test_3d_rz(self):
        """Test 3d plotting of pressure on toroidal cross section."""
        eq = get("DSHAPE_CURRENT")
        grid = LinearGrid(rho=100, theta=0.0, zeta=100)
        _ = plot_3d(eq, "p", grid=grid)

    @pytest.mark.unit
    def test_3d_rt(self):
        """Test 3d plotting of flux on poloidal ribbon."""
        eq = get("DSHAPE_CURRENT")
        grid = LinearGrid(rho=100, theta=100, zeta=0.0)
        _ = plot_3d(eq, "psi", grid=grid)

    @pytest.mark.unit
    def test_plot_3d_surface(self):
        """Test 3d plotting of surface object."""
        surf = FourierRZToroidalSurface()
        _ = plot_3d(
            surf,
            "curvature_H_rho",
            showgrid=False,
            showscale=False,
            zeroline=False,
            showticklabels=False,
            showaxislabels=False,
        )

    @pytest.mark.unit
    def test_3d_plot_Bn(self):
        """Test 3d plotting of Bn on equilibrium surface."""
        eq = get("precise_QA")
        with pytest.warns(UserWarning, match="Reducing radial"):
            eq.change_resolution(M=4, N=4, L=4, M_grid=8, N_grid=8, L_grid=8)
        _ = plot_3d(
            eq,
            "B*n",
            field=ToroidalMagneticField(1, 1),
            grid=LinearGrid(M=30, N=30, NFP=1, endpoint=True),
        )


class TestPlotFSA:
    """Tests for plot_fsa."""

    @pytest.mark.unit
    def test_plot_fsa_axis_limit(self):
        """Test magnetic axis limit of flux surface average is plotted."""
        eq = get("W7-X")
        rho = np.linspace(0, 1, 10)
        grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=rho)
        assert grid.axis.size

        name = "J*B"
        assert (
            "<" + name + ">" in data_index["desc.equilibrium.equilibrium.Equilibrium"]
        ), "Test with a different quantity."
        # should forward computation to compute function
        _, _, plot_data = plot_fsa(
            eq=eq,
            name=name,
            rho=rho,
            M=eq.M_grid,
            N=eq.N_grid,
            with_sqrt_g=True,
            return_data=True,
        )
        desired = grid.compress(
            eq.compute(names="<" + name + ">", grid=grid)["<" + name + ">"]
        )
        np.testing.assert_allclose(
            plot_data["<" + name + ">"], desired, equal_nan=False
        )

        name = "psi_r/sqrt(g)"
        assert (
            "<" + name + ">"
            not in data_index["desc.equilibrium.equilibrium.Equilibrium"]
        ), "Test with a different quantity."
        # should automatically compute axis limit
        _, _, plot_data = plot_fsa(
            eq=eq,
            name=name,
            rho=rho,
            M=eq.M_grid,
            N=eq.N_grid,
            with_sqrt_g=True,
            return_data=True,
        )
        data = eq.compute(names=[name, "sqrt(g)", "sqrt(g)_r"], grid=grid)
        desired = surface_averages(
            grid=grid,
            q=data[name],
            sqrt_g=grid.replace_at_axis(data["sqrt(g)"], data["sqrt(g)_r"], copy=True),
            expand_out=False,
        )
        np.testing.assert_allclose(
            plot_data["<" + name + ">_fsa"], desired, equal_nan=False
        )

        name = "|B|"
        assert (
            "<" + name + ">" in data_index["desc.equilibrium.equilibrium.Equilibrium"]
        ), "Test with a different quantity."
        _, _, plot_data = plot_fsa(
            eq=eq,
            name=name,
            rho=rho,
            M=eq.M_grid,
            N=eq.N_grid,
            with_sqrt_g=False,  # test that does not compute data_index["<|B|>"]
            return_data=True,
        )
        data = eq.compute(names=name, grid=grid)
        desired = surface_averages(grid=grid, q=data[name], expand_out=False)
        np.testing.assert_allclose(
            plot_data["<" + name + ">_fsa"], desired, equal_nan=False
        )

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
    def test_fsa_I(self):
        """Test plotting of flux surface average toroidal current."""
        eq = get("DSHAPE_CURRENT")
        fig, ax, data = plot_fsa(eq, "B_theta", with_sqrt_g=False, return_data=True)
        assert "rho" in data.keys()
        assert "<B_theta>_fsa" in data.keys()
        assert "normalization" in data.keys()
        assert data["normalization"] == 1

        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
    def test_fsa_F_normalized(self):
        """Test plotting flux surface average normalized force error on log scale."""
        eq = get("DSHAPE_CURRENT")
        fig, ax = plot_fsa(eq, "|F|", log=True, normalize="<|grad(p)|>_vol")
        ax.set_ylim([1e-6, 1e-3])
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
    def test_fsa_F_normalized_deprecated(self):
        """Test plotting deprecated fsa normalized force error on log scale."""
        eq = get("DSHAPE_CURRENT")
        with pytest.raises((ValueError, FutureWarning)):
            _, _ = plot_fsa(
                eq, "|F|", log=True, norm_F=True, normalize="<|grad(|B|^2)|/2mu0>_vol"
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            fig, ax = plot_fsa(eq, "|F|", log=True, norm_F=True)
        return fig


class TestPlotSection:
    """Tests for plot_section."""

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_section_J(self):
        """Test plotting poincare section of radial current."""
        eq = get("DSHAPE_CURRENT")
        fig, ax, data = plot_section(eq, "J^rho", return_data=True)
        assert "R" in data.keys()
        assert "Z" in data.keys()
        assert "J^rho" in data.keys()
        assert "normalization" in data.keys()
        assert data["normalization"] == 1

        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_section_chi_contour(self):
        """Test plotting poincare section of poloidal flux, with fill=False."""
        eq = get("DSHAPE_CURRENT")
        fig, ax = plot_section(eq, "chi", fill=False, levels=20)
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_section_F(self):
        """Test plotting poincare section of radial force."""
        eq = get("DSHAPE_CURRENT")
        fig, ax = plot_section(eq, "F_rho")
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=50)
    def test_section_logF(self):
        """Test plotting poincare section of force magnitude on log scale."""
        eq = get("DSHAPE_CURRENT")
        fig, ax = plot_section(eq, "|F|", log=True)
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_plot_normF_section(self):
        """Test Poincare section plot of normalized force on log scale."""
        eq = get("DSHAPE_CURRENT")
        fig, ax = plot_section(eq, "|F|_normalized", log=True)
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_plot_normF_section_deprecated(self):
        """Test old section plot of normalized force on log scale."""
        eq = get("DSHAPE_CURRENT")
        with pytest.raises((ValueError, FutureWarning)):
            _, _ = plot_section(
                eq, "|F|", log=True, norm_F=True, normalize="<|grad(|B|^2)|/2mu0>_vol"
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            fig, ax = plot_section(eq, "|F|", log=True, norm_F=True)

        return fig


class TestPlotSurfaces:
    """Tests for plot_surfaces."""

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_plot_surfaces(self):
        """Test plotting flux surfaces."""
        eq = get("DSHAPE_CURRENT")
        fig, ax, data = plot_surfaces(eq, return_data=True)
        for string in [
            "rho_R_coords",
            "rho_Z_coords",
            "vartheta_R_coords",
            "vartheta_Z_coords",
        ]:
            assert string in data.keys()

        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_plot_surfaces_no_theta(self):
        """Test plotting flux surfaces without theta contours."""
        eq = get("DSHAPE_CURRENT")
        fig, ax, data = plot_surfaces(eq, theta=False, return_data=True)
        for string in ["rho_R_coords", "rho_Z_coords"]:
            assert string in data.keys()

        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_plot_surfaces_HELIOTRON(self):
        """Test plot surfaces of equilibrium for correctness of vartheta lines."""
        fig, ax = plot_surfaces(get("HELIOTRON"))
        return fig


class TestPlotBoundary:
    """Tests for plot_boundary and plot_boundaries."""

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_plot_boundary(self):
        """Test plotting boundary."""
        eq = get("W7-X")
        fig, ax, data = plot_boundary(eq, plot_axis=True, return_data=True)
        assert "R" in data.keys()
        assert "Z" in data.keys()

        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
    def test_plot_boundary_surface(self):
        """Test plot_boundary function for Surface objects."""
        surf = FourierRZToroidalSurface()
        fig, ax = plot_boundary(surf)
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
    def test_plot_boundaries(self):
        """Test plotting boundaries."""
        eq1 = get("SOLOVEV")
        eq2 = get("DSHAPE")
        eq3 = get("W7-X")
        eq4 = get("ESTELL")
        with pytest.raises(ValueError, match="differing field periods"):
            fig, ax = plot_boundaries([eq3, eq4], theta=0)
        _, _, data1 = plot_boundaries(
            (eq1, eq2, eq3),
            phi=4,
            return_data=True,
        )
        fig, ax, data = plot_boundaries(
            (eq1, eq2, eq3),
            phi=np.linspace(0, 2 * np.pi / eq3.NFP, 4, endpoint=False),
            return_data=True,
        )
        assert "R" in data.keys()
        assert "Z" in data.keys()
        assert len(data["R"]) == 3
        assert len(data["Z"]) == 3
        assert (
            data["R"][-1].shape == data1["R"][-1].shape
        ), "Passing phi as an integer or array results in different behavior"
        assert (
            data["Z"][-1].shape == data1["Z"][-1].shape
        ), "Passing phi as an integer or array results in different behavior"

        return fig


class TestPlotComparison:
    """Tests for plot_comparison."""

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_plot_comparison(self):
        """Test plotting comparison of flux surfaces."""
        eqf = get("DSHAPE_CURRENT", "all")
        fig, ax, data = plot_comparison(eqf, return_data=True)
        for string in [
            "rho_R_coords",
            "rho_Z_coords",
            "vartheta_R_coords",
            "vartheta_Z_coords",
        ]:
            assert string in data.keys()
            assert len(data[string]) == len(eqf)

        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_plot_comparison_no_theta(self):
        """Test plotting comparison of flux surfaces without theta contours."""
        eqf = get("DSHAPE_CURRENT", "all")
        fig, ax = plot_comparison(eqf, theta=0)
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_plot_comparison_different_NFPs(self):
        """Test plotting comparison of flux surfaces with differing NFPs."""
        eq = get("SOLOVEV")
        eq_nonax = get("HELIOTRON")
        eq_nonax2 = get("ESTELL")
        with pytest.raises(ValueError, match="differing field periods"):
            fig, ax = plot_comparison([eq_nonax, eq_nonax2], theta=0)
        fig, ax = plot_comparison(
            [eq, eq_nonax],
            phi=np.linspace(0, 2 * np.pi / eq_nonax.NFP, 6, endpoint=False),
            theta=0,
        )
        return fig


class TestPlotGrid:
    """Tests for the plot_grid function."""

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_plot_grid_linear(self):
        """Test plotting linear grid."""
        grid = LinearGrid(rho=10, theta=10, zeta=1)
        fig, ax, data = plot_grid(grid, return_data=True)
        for string in ["theta", "rho"]:
            assert string in data.keys()
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_plot_grid_quad(self):
        """Test plotting quadrature grid."""
        grid = QuadratureGrid(L=10, M=10, N=1)
        fig, ax = plot_grid(grid, figsize=(6, 6))
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_plot_grid_jacobi(self):
        """Test plotting concentric grid with jacobi nodes."""
        grid = ConcentricGrid(L=20, M=10, N=1, node_pattern="jacobi")
        fig, ax = plot_grid(grid)
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_plot_grid_cheb1(self):
        """Test plotting concentric grid with chebyshev 1 nodes."""
        grid = ConcentricGrid(L=20, M=10, N=1, node_pattern="cheb1")
        fig, ax = plot_grid(grid)
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_plot_grid_cheb2(self):
        """Test plotting concentric grid with chebyshev 2 nodes."""
        grid = ConcentricGrid(L=20, M=10, N=1, node_pattern="cheb2")
        fig, ax = plot_grid(grid)
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_plot_grid_ocs(self):
        """Test plotting concentric grid with optimal concentric sampling nodes."""
        grid = ConcentricGrid(L=20, M=10, N=1, node_pattern="ocs")
        fig, ax = plot_grid(grid)
        return fig


class TestPlotBasis:
    """Tests for plot_basis function."""

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_plot_basis_powerseries(self):
        """Test plotting power series basis."""
        basis = PowerSeries(L=6)
        fig, ax, data = plot_basis(basis, return_data=True)
        for string in ["amplitude", "rho", "l"]:
            assert string in data.keys()
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_plot_basis_chebyshevpoly(self):
        """Test plotting Chebyshev polynomial."""
        basis = ChebyshevPolynomial(L=6)
        fig, ax, data = plot_basis(basis, return_data=True)
        for string in ["amplitude", "rho", "l"]:
            assert string in data.keys()
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=26)
    def test_plot_basis_zernikepoly(self):
        """Test plotting zernike polynomial."""
        basis = ZernikePolynomial(L=6, M=4)
        fig, ax, data = plot_basis(basis, return_data=True)
        for string in ["amplitude", "rho", "l"]:
            assert string in data.keys()
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=26)
    def test_plot_basis_zernikepoly_derivative(self):
        """Test plotting zernike polynomial derivative."""
        basis = ZernikePolynomial(L=6, M=4)
        fig, ax = plot_basis(basis, derivative=[2, 0, 0])
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_plot_basis_fourierseries(self):
        """Test plotting fourier series basis."""
        basis = FourierSeries(N=3)
        fig, ax, data = plot_basis(basis, return_data=True)
        for string in ["amplitude", "n", "zeta"]:
            assert string in data.keys()
        return fig

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_plot_basis_doublefourierseries(self):
        """Test plotting double fourier series basis."""
        basis = DoubleFourierSeries(M=3, N=2)
        fig, ax, data = plot_basis(basis, return_data=True)
        for string in ["amplitude", "n", "zeta", "m", "theta"]:
            assert string in data.keys()
        return fig

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=26)
    def test_plot_basis_fourierzernike(self):
        """Test plotting fourier-zernike basis."""
        basis = FourierZernikeBasis(L=8, M=3, N=2)
        fig, ax, data = plot_basis(basis, return_data=True)
        for string in ["amplitude", "l", "rho", "m", "theta"]:
            assert string in data.keys()
        return fig

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=26)
    def test_plot_basis_fourierzernike_derivative(self):
        """Test plotting fourier-zernike basis derivative."""
        basis = FourierZernikeBasis(L=8, M=3, N=2)
        fig, ax = plot_basis(basis, derivative=[1, 0, 0])
        return fig

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=26)
    def test_plot_basis_chebyshevdoublefourier(self):
        """Test plotting chebyshev-double fourier basis."""
        basis = ChebyshevDoubleFourierBasis(L=8, M=3, N=2)
        fig, ax, data = plot_basis(basis, return_data=True)
        for string in ["amplitude", "l", "rho", "m", "theta"]:
            assert string in data.keys()
        return fig


class TestPlotBoozerModes:
    """Tests for plot_boozer_modes."""

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
    def test_plot_boozer_modes(self):
        """Test plotting boozer spectrum."""
        eq = get("WISTELL-A")
        fig, ax, data = plot_boozer_modes(
            eq,
            M_booz=eq.M,
            N_booz=eq.N,
            num_modes=7,
            return_data=True,
            norm=True,
            rho=5,
        )
        ax.set_ylim([1e-6, 5e0])
        for string in ["|B|_mn_B", "B modes", "rho"]:
            assert string in data.keys()
        return fig

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
    def test_plot_boozer_modes_breaking_only(self):
        """Test plotting symmetry breaking boozer spectrum."""
        eq = get("WISTELL-A")
        fig, ax = plot_boozer_modes(
            eq,
            M_booz=eq.M,
            N_booz=eq.N,
            helicity=(1, -eq.NFP),
            norm=True,
            num_modes=7,
            rho=5,
        )
        ax.set_ylim([1e-6, 5e0])
        return fig

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
    def test_plot_boozer_modes_max(self):
        """Test plotting symmetry breaking boozer spectrum."""
        eq = get("WISTELL-A")
        fig, ax, data = plot_boozer_modes(
            eq,
            M_booz=eq.M,
            N_booz=eq.N,
            helicity=(1, -eq.NFP),
            max_only=True,
            label="WISTELL-A",
            color="r",
            norm=True,
            rho=5,
            return_data=True,
        )
        ax.set_ylim([1e-6, 5e0])
        for string in ["|B|_mn_B", "B modes", "rho"]:
            assert string in data.keys()
        return fig

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
    def test_plot_boozer_modes_no_norm(self):
        """Test plotting boozer spectrum without B0 and norm."""
        eq = get("ESTELL")
        fig, ax = plot_boozer_modes(
            eq,
            M_booz=eq.M,
            N_booz=eq.N,
            num_modes=7,
            B0=False,
            log=False,
            rho=5,
        )
        return fig


class TestPlotBoozerSurface:
    """Tests for plot_boozer_surface."""

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_plot_boozer_surface(self):
        """Test plotting B in boozer coordinates."""
        eq = get("WISTELL-A")
        fig, ax, data = plot_boozer_surface(
            eq, M_booz=eq.M, N_booz=eq.N, return_data=True, rho=0.5, fieldlines=4
        )
        for string in [
            "|B|",
            "theta_B",
            "zeta_B",
        ]:
            assert string in data.keys()
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_plot_omnigenous_field(self):
        """Test plot omnigenous magnetic field."""
        field = OmnigenousField(
            L_B=0,
            M_B=4,
            L_x=0,
            M_x=1,
            N_x=1,
            NFP=4,
            helicity=(1, 4),
            B_lm=np.array([0.8, 0.9, 1.1, 1.2]),
            x_lmn=np.array([0, -np.pi / 8, 0, np.pi / 8, 0, np.pi / 4]),
        )
        fig, ax = plot_boozer_surface(field, iota=0.6, fieldlines=4)
        return fig


@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_plot_qs_error():
    """Test plotting qs error metrics."""
    eq = get("WISTELL-A")
    fig, ax, data = plot_qs_error(
        eq,
        helicity=(1, -eq.NFP),
        M_booz=eq.M,
        N_booz=eq.N,
        log=True,
        return_data=True,
        rho=5,
    )
    ax.set_ylim([1e-3, 2e-1])
    for string in ["rho", "f_T", "f_B", "f_C"]:
        assert string in data.keys()
        if string != "rho":
            # ensure that there is different QS data for each surface
            # related to gh PR #400
            assert not isalmostequal(data[string])
    return fig


@pytest.mark.unit
def test_plot_coils():
    """Test 3d plotting of coils with currents."""
    N = 48
    NFP = 4
    I = 1
    coil = FourierXYZCoil()
    coil.rotate(angle=np.pi / N)
    coils = CoilSet.linspaced_angular(coil, I, [0, 0, 1], np.pi / NFP, N // NFP // 2)
    coils2 = MixedCoilSet.from_symmetry(coils, NFP, True)
    with pytest.raises(ValueError, match="Expected `coils`"):
        plot_coils("not a coil")
    fig, data = plot_coils(coils2, return_data=True)

    def flatten_coils(coilset):
        if hasattr(coilset, "__len__"):
            return [a for i in coilset for a in flatten_coils(i)]
        else:
            return [coilset]

    coil_list = flatten_coils(coils2)
    for string in ["X", "Y", "Z"]:
        assert string in data.keys()
        assert len(data[string]) == len(coil_list)


@pytest.mark.unit
def test_plot_coils_no_grid():
    """Test 3d plotting of coils with currents without any gridlines."""
    N = 48
    NFP = 4
    I = 1
    coil = FourierXYZCoil()
    coil.rotate(angle=np.pi / N)
    coils = CoilSet.linspaced_angular(coil, I, [0, 0, 1], np.pi / NFP, N // NFP // 2)
    with pytest.raises(ValueError, match="Expected `coils`"):
        plot_coils("not a coil")
    fig, data = plot_coils(
        coils,
        unique=True,
        return_data=True,
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        showaxislabels=False,
    )

    def flatten_coils(coilset):
        if hasattr(coilset, "__len__"):
            return [a for i in coilset for a in flatten_coils(i)]
        else:
            return [coilset]

    coil_list = flatten_coils(coils)
    for string in ["X", "Y", "Z"]:
        assert string in data.keys()
        assert len(data[string]) == len(coil_list)


@pytest.mark.unit
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_plot_b_mag():
    """Test plot of |B| on longer field lines for gyrokinetic simulations."""
    psi = 0.5
    rho = np.sqrt(psi)
    npol = 2
    nzgrid = 128
    alpha = 0
    # compute iota
    eq = get("W7-X")
    iota = eq.compute("iota", grid=LinearGrid(rho=rho, NFP=eq.NFP))["iota"][0]

    # get flux tube coordinate system
    zeta = np.linspace(
        -np.pi * npol / np.abs(iota), np.pi * npol / np.abs(iota), 2 * nzgrid + 1
    )
    thetas = alpha * np.ones_like(zeta) + iota * zeta

    rhoa = rho * np.ones_like(zeta)
    c = np.vstack([rhoa, thetas, zeta]).T
    coords = eq.map_coordinates(c, inbasis=("rho", "theta_PEST", "zeta"))
    grid = Grid(coords)

    # compute |B| normalized in the usual flux tube way
    psib = np.abs(eq.compute("psi")["psi"][-1])
    Lref = eq.compute("a")["a"]
    Bref = 2 * psib / Lref**2
    bmag = eq.compute("|B|", grid=grid)["|B|"] / Bref
    fig, ax = plt.subplots()
    ax.plot(bmag)
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
def test_plot_coefficients():
    """Test scatter plot of spectral coefficients."""
    eq = get("DSHAPE_CURRENT")
    fig, ax = plot_coefficients(eq, color="b", marker="o")
    ax[0, 0].set_ylim([1e-8, 1e1])
    return fig


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
@pytest.mark.unit
def test_plot_logo():
    """Test plotting the DESC logo."""
    fig, ax = plot_logo()
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
def test_plot_poincare():
    """Test making a poincare plot."""
    ext_field = load("tests/inputs/precise_QA_helical_coils.h5")
    eq = get("precise_QA")
    grid_trace = LinearGrid(rho=np.linspace(0, 1, 9))
    r0 = eq.compute("R", grid=grid_trace)["R"]
    z0 = eq.compute("Z", grid=grid_trace)["Z"]

    fig, ax = poincare_plot(ext_field, r0, z0, ntransit=50, NFP=eq.NFP)
    return fig


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
@pytest.mark.unit
def test_plot_gammac():
    """Test plotting gamma_c."""
    eq = get("W7-X")
    fig, ax = plot_gammac(eq, rho=0.5)
    return fig


@pytest.mark.unit
def test_plot_field_lines():
    """Test plotting field lines."""
    field = ToroidalMagneticField(B0=-1.0, R0=1.0)
    fig, data = plot_field_lines(
        field, [1.0], [0.0], nphi_per_transit=10, ntransit=0.8, return_data=True
    )
    assert all(data["Z"][0] == 0)
    assert np.allclose((data["X"][0] ** 2 + data["Y"][0] ** 2), 1)

    field1 = ToroidalMagneticField(B0=1.0, R0=1.0)
    field2 = PoloidalMagneticField(B0=1.0, R0=1.0, iota=3.0)
    field = SumMagneticField([field1, field2])
    _ = plot_field_lines(
        field,
        R0=[1.1, 1.3],
        Z0=[0.0, 0.1],
        nphi_per_transit=100,
        ntransit=2,
        endpoint=True,
        bs_chunk_size=10,
    )


@pytest.mark.unit
def test_plot_field_lines_reversed():
    """Test plotting field lines with reversed direction."""
    field = load("tests/inputs/precise_QA_helical_coils.h5")
    eq = get("precise_QA")
    grid_trace = LinearGrid(rho=[1])
    r0 = eq.compute("R", grid=grid_trace)["R"]
    z0 = eq.compute("Z", grid=grid_trace)["Z"]

    fig, data1 = plot_field_lines(
        field, r0, z0, nphi_per_transit=100, ntransit=1, return_data=True
    )
    # get the last point of the field line
    pt_end = np.array([data1["X"][0][-1], data1["Y"][0][-1], data1["Z"][0][-1]])
    # convert to RPZ coordinates (actually not necessary since phi0=0)
    pt_end_rpz = xyz2rpz(pt_end)
    # plot the field line in the reversed direction starting from the end point
    # and going backwards, this should overlap with the previous field line
    fig, data2 = plot_field_lines(
        field,
        pt_end_rpz[0],
        pt_end_rpz[2],
        nphi_per_transit=100,
        ntransit=-1,
        return_data=True,
        fig=fig,
        color="red",
        lw=10,
        # test that passing options as dict works
        options={"throw": True, "made_jump": None},
    )
    x1 = data1["X"][0]
    y1 = data1["Y"][0]
    z1 = data1["Z"][0]
    x2 = data2["X"][0]
    y2 = data2["Y"][0]
    z2 = data2["Z"][0]
    # we should flip the second field line to compare with the first one
    assert np.allclose(x1, np.flip(x2), atol=1e-7)
    assert np.allclose(y1, np.flip(y2), atol=1e-7)
    assert np.allclose(z1, np.flip(z2), atol=1e-7)
