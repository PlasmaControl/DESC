"""Regression tests for plotting functions, by comparing to saved baseline images."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.interpolate import interp1d

from desc.basis import (
    DoubleFourierSeries,
    FourierSeries,
    FourierZernikeBasis,
    PowerSeries,
)
from desc.coils import CoilSet, FourierXYZCoil, MixedCoilSet
from desc.compute import data_index
from desc.compute.utils import surface_averages
from desc.examples import get
from desc.grid import ConcentricGrid, Grid, LinearGrid, QuadratureGrid
from desc.io import load
from desc.plotting import (
    _find_idx,
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
    plot_field_lines_sfl,
    plot_fsa,
    plot_grid,
    plot_logo,
    plot_qs_error,
    plot_section,
    plot_surfaces,
)
from desc.utils import isalmostequal

tol_1d = 7.8
tol_2d = 15
tol_3d = 15


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
    with pytest.warns(FutureWarning):
        fig, ax = plot_surfaces(eq, zeta=2)
    return None


@pytest.mark.unit
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_1d_p(SOLOVEV):
    """Test plotting 1d pressure profile."""
    eq = load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax, data = plot_1d(eq, "p", figsize=(4, 4), return_data=True)
    assert "p" in data.keys()
    return fig


@pytest.mark.unit
def test_1d_fsa_consistency():
    """Test that plot_1d uses 2d grid to compute quantities with surface averages."""
    eq = get("W7-X")

    def test(name, with_sqrt_g=True, grid=None):
        _, ax_0 = plot_1d(eq, name, grid=grid)
        # 100 rho points is plot_1d default
        _, ax_1 = plot_fsa(
            eq,
            name,
            with_sqrt_g=with_sqrt_g,
            rho=100 if grid is None else grid.nodes[:, 0],
        )
        np.testing.assert_allclose(
            ax_0.lines[0].get_xydata(), ax_1.lines[0].get_xydata()
        )

    lg = LinearGrid(rho=np.linspace(0, 1, 30))
    test("magnetic well")
    test("magnetic well", grid=lg)
    test("current", with_sqrt_g=False)
    test("current", with_sqrt_g=False, grid=lg)


@pytest.mark.unit
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_1d_dpdr(DSHAPE_current):
    """Test plotting 1d pressure derivative."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    fig, ax, data = plot_1d(eq, "p_r", figsize=(4, 4), return_data=True)
    assert "p_r" in data.keys()
    return fig


@pytest.mark.unit
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_1d_iota(DSHAPE_current):
    """Test plotting 1d rotational transform."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    grid = LinearGrid(rho=0.5, theta=100, zeta=0.0)
    fig, ax, data = plot_1d(eq, "iota", grid=grid, figsize=(4, 4), return_data=True)
    assert "theta" in data.keys()
    assert "iota" in data.keys()
    return fig


@pytest.mark.unit
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_1d_iota_radial(DSHAPE_current):
    """Test plotting 1d rotational transform."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    fig, ax, data = plot_1d(eq, "iota", figsize=(4, 4), return_data=True)
    assert "rho" in data.keys()
    assert "iota" in data.keys()
    return fig


@pytest.mark.unit
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_1d_logpsi(DSHAPE_current):
    """Test plotting 1d flux function with log scale."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    fig, ax, data = plot_1d(eq, "psi", log=True, figsize=(4, 4), return_data=True)
    ax.set_ylim([1e-5, 1e0])
    assert "rho" in data.keys()
    assert "psi" in data.keys()
    return fig


@pytest.mark.unit
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=10)
def test_2d_logF(DSHAPE_current):
    """Test plotting 2d force error with log scale."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    grid = LinearGrid(rho=100, theta=100, zeta=0.0)
    fig, ax, data = plot_2d(
        eq, "|F|", log=True, grid=grid, figsize=(4, 4), return_data=True
    )
    assert "|F|" in data.keys()
    return fig


@pytest.mark.unit
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
def test_2d_g_tz(DSHAPE_current):
    """Test plotting 2d metric coefficients vs theta/zeta."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    grid = LinearGrid(rho=0.5, theta=100, zeta=100)
    fig, ax, data = plot_2d(eq, "sqrt(g)", grid=grid, figsize=(4, 4), return_data=True)
    assert "theta" in data.keys()
    assert "zeta" in data.keys()

    assert "sqrt(g)" in data.keys()
    return fig


@pytest.mark.unit
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
def test_2d_g_rz(DSHAPE_current):
    """Test plotting 2d metric coefficients vs rho/zeta."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    grid = LinearGrid(rho=100, theta=0.0, zeta=100)
    fig, ax, data = plot_2d(eq, "sqrt(g)", grid=grid, figsize=(4, 4), return_data=True)
    assert "rho" in data.keys()
    assert "zeta" in data.keys()
    assert "sqrt(g)" in data.keys()

    return fig


@pytest.mark.unit
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
def test_2d_lambda(DSHAPE_current):
    """Test plotting lambda on 2d grid."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    fig, ax, data = plot_2d(eq, "lambda", figsize=(4, 4), return_data=True)
    assert "lambda" in data.keys()

    return fig


@pytest.mark.unit
@pytest.mark.solve
def test_3d_B(DSHAPE_current):
    """Test 3d plot of toroidal field."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    fig, data = plot_3d(eq, "B^zeta", return_data=True)
    assert "X" in data.keys()
    assert "Y" in data.keys()
    assert "Z" in data.keys()

    assert "B^zeta" in data.keys()

    return fig


@pytest.mark.unit
@pytest.mark.solve
def test_3d_J(DSHAPE_current):
    """Test 3d plotting of poloidal current."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    grid = LinearGrid(rho=1.0, theta=100, zeta=100)
    fig = plot_3d(eq, "J^theta", grid=grid)
    return fig


@pytest.mark.unit
@pytest.mark.solve
def test_3d_tz(DSHAPE_current):
    """Test 3d plot of force on interior surface."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    grid = LinearGrid(rho=0.5, theta=100, zeta=100)
    fig = plot_3d(eq, "|F|", log=True, grid=grid)
    return fig


@pytest.mark.unit
@pytest.mark.solve
def test_3d_rz(DSHAPE_current):
    """Test 3d plotting of pressure on toroidal cross section."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    grid = LinearGrid(rho=100, theta=0.0, zeta=100)
    fig = plot_3d(eq, "p", grid=grid)
    return fig


@pytest.mark.unit
@pytest.mark.solve
def test_3d_rt(DSHAPE_current):
    """Test 3d plotting of flux on poloidal ribbon."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    grid = LinearGrid(rho=100, theta=100, zeta=0.0)
    fig = plot_3d(eq, "psi", grid=grid)
    return fig


@pytest.mark.unit
def test_plot_fsa_axis_limit():
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
    np.testing.assert_allclose(plot_data["<" + name + ">"], desired, equal_nan=False)

    name = "B0"
    assert (
        "<" + name + ">" not in data_index["desc.equilibrium.equilibrium.Equilibrium"]
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
        with_sqrt_g=False,  # Test that does not compute data_index["<|B|>"]
        return_data=True,
    )
    data = eq.compute(names=name, grid=grid)
    desired = surface_averages(grid=grid, q=data[name], expand_out=False)
    np.testing.assert_allclose(
        plot_data["<" + name + ">_fsa"], desired, equal_nan=False
    )


@pytest.mark.unit
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_fsa_I(DSHAPE_current):
    """Test plotting of flux surface average toroidal current."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    fig, ax, data = plot_fsa(eq, "B_theta", with_sqrt_g=False, return_data=True)
    assert "rho" in data.keys()
    assert "<B_theta>_fsa" in data.keys()
    assert "normalization" in data.keys()
    assert data["normalization"] == 1

    return fig


@pytest.mark.unit
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_fsa_G(DSHAPE_current):
    """Test plotting of flux surface average poloidal current."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    fig, ax = plot_fsa(eq, "B_zeta", with_sqrt_g=False, log=True)
    ax.set_ylim([1e-1, 1e0])
    return fig


@pytest.mark.unit
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_fsa_F_normalized(DSHAPE_current):
    """Test plotting flux surface average normalized force error on log scale."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    fig, ax = plot_fsa(eq, "|F|", log=True, norm_F=True, norm_name="<|grad(p)|>_vol")
    ax.set_ylim([1e-6, 1e-3])
    return fig


@pytest.mark.unit
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
def test_section_J(DSHAPE_current):
    """Test plotting poincare section of radial current."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    fig, ax, data = plot_section(eq, "J^rho", return_data=True)
    assert "R" in data.keys()
    assert "Z" in data.keys()
    assert "J^rho" in data.keys()
    assert "normalization" in data.keys()
    assert data["normalization"] == 1

    return fig


@pytest.mark.unit
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=24)
def test_section_Z(DSHAPE_current):
    """Test plotting poincare section of Z coordinate."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    fig, ax = plot_section(eq, "Z")
    return fig


@pytest.mark.unit
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
def test_section_R(DSHAPE_current):
    """Test plotting poincare section of R coordinate."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    fig, ax = plot_section(eq, "R")
    return fig


@pytest.mark.unit
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
def test_section_F(DSHAPE_current):
    """Test plotting poincare section of radial force."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    fig, ax = plot_section(eq, "F_rho")
    return fig


@pytest.mark.unit
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
def test_section_F_normalized_vac(DSHAPE_current):
    """Test plotting poincare section of normalized vacuum force error."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[1]
    fig, ax = plot_section(eq, "|F|", norm_F=True)
    return fig


@pytest.mark.unit
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=50)
def test_section_logF(DSHAPE_current):
    """Test plotting poincare section of force magnitude on log scale."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    fig, ax = plot_section(eq, "|F|", log=True)
    return fig


@pytest.mark.slow
@pytest.mark.unit
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
def test_plot_surfaces(DSHAPE_current):
    """Test plotting flux surfaces."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    fig, ax, data = plot_surfaces(eq, return_data=True)
    for string in [
        "rho_R_coords",
        "rho_Z_coords",
        "vartheta_R_coords",
        "vartheta_Z_coords",
    ]:
        assert string in data.keys()

    return fig


@pytest.mark.slow
@pytest.mark.unit
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
def test_plot_surfaces_no_theta(DSHAPE_current):
    """Test plotting flux surfaces without theta contours."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    fig, ax, data = plot_surfaces(eq, theta=False, return_data=True)
    for string in ["rho_R_coords", "rho_Z_coords"]:
        assert string in data.keys()

    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
def test_plot_boundary():
    """Test plotting boundary."""
    eq = get("W7-X")
    fig, ax, data = plot_boundary(eq, plot_axis=True, return_data=True)
    assert "R" in data.keys()
    assert "Z" in data.keys()

    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
def test_plot_boundaries():
    """Test plotting boundaries."""
    eq1 = get("SOLOVEV")
    eq2 = get("DSHAPE")
    eq3 = get("W7-X")
    fig, ax, data = plot_boundaries((eq1, eq2, eq3), return_data=True)
    assert "R" in data.keys()
    assert "Z" in data.keys()
    assert len(data["R"]) == 3
    assert len(data["Z"]) == 3

    return fig


@pytest.mark.slow
@pytest.mark.unit
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
def test_plot_comparison(DSHAPE_current):
    """Test plotting comparison of flux surfaces."""
    eqf = load(load_from=str(DSHAPE_current["desc_h5_path"]))
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


@pytest.mark.slow
@pytest.mark.unit
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
def test_plot_comparison_no_theta(DSHAPE_current):
    """Test plotting comparison of flux surfaces without theta contours."""
    eqf = load(load_from=str(DSHAPE_current["desc_h5_path"]))
    fig, ax = plot_comparison(eqf, theta=0)
    return fig


@pytest.mark.unit
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
def test_plot_con_basis(DSHAPE_current):
    """Test 2d plot of R component of e^rho."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    fig, ax, data = plot_2d(
        eq, "e^rho", component="R", figsize=(4, 4), return_data=True
    )
    for string in ["e^rho", "normalization", "theta", "zeta"]:
        assert string in data.keys()
    assert data["normalization"] == 1

    return fig


@pytest.mark.unit
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
def test_plot_cov_basis(DSHAPE_current):
    """Test 2d plot of norm of e_rho."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    fig, ax, data = plot_2d(eq, "e_rho", figsize=(4, 4), return_data=True)
    return fig


@pytest.mark.unit
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
def test_plot_magnetic_tension(DSHAPE_current):
    """Test 2d plot of magnetic tension."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    fig, ax, data = plot_2d(eq, "|(B*grad)B|", figsize=(4, 4), return_data=True)
    return fig


@pytest.mark.unit
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
def test_plot_magnetic_pressure(DSHAPE_current):
    """Test 2d plot of magnetic pressure."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    fig, ax, data = plot_2d(eq, "|grad(|B|^2)|/2mu0", figsize=(4, 4), return_data=True)
    return fig


@pytest.mark.unit
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
def test_plot_gradpsi(DSHAPE_current):
    """Test 2d plot of norm of grad(rho)."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    fig, ax, data = plot_2d(eq, "|grad(rho)|", figsize=(4, 4), return_data=True)
    return fig


@pytest.mark.unit
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
def test_plot_normF_2d(DSHAPE_current):
    """Test 2d plot of normalized force."""
    grid = LinearGrid(rho=np.array(0.8), M=20, N=2)
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    fig, ax, data = plot_2d(
        eq, "|F|", norm_F=True, figsize=(4, 4), return_data=True, grid=grid
    )
    for string in ["|F|", "theta", "zeta"]:
        assert string in data.keys()
    return fig


@pytest.mark.unit
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
def test_plot_normF_section(DSHAPE_current):
    """Test Poincare section plot of normalized force on log scale."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    fig, ax = plot_section(eq, "|F|", norm_F=True, log=True)
    return fig


@pytest.mark.unit
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
def test_plot_coefficients(DSHAPE_current):
    """Test scatter plot of spectral coefficients."""
    eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    fig, ax = plot_coefficients(eq)
    ax[0, 0].set_ylim([1e-8, 1e1])
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
def test_plot_logo():
    """Test plotting the DESC logo."""
    fig, ax = plot_logo()
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


class TestPlotFieldLines:
    """Tests for plotting field lines."""

    @pytest.mark.unit
    def test_find_idx(self):
        """Test finding the index of the node closest to a given point."""
        # pick the first grid node point, add epsilon to it, check it returns idx of 0
        grid = LinearGrid(L=1, M=2, N=2, axis=False)
        epsilon = np.finfo(float).eps
        test_point = grid.nodes[0, :] + epsilon
        idx = _find_idx(*test_point, grid=grid)
        assert idx == 0

    @pytest.mark.unit
    @pytest.mark.solve
    @pytest.mark.slow
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
    def test_plot_field_line(self, DSHAPE_current):
        """Test plotting single field line over 1 transit."""
        eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
        fig, ax, data = plot_field_lines_sfl(
            eq, rho=1, seed_thetas=0, phi_end=2 * np.pi, return_data=True
        )
        for string in ["R", "Z", "phi", "seed_thetas"]:
            assert string in data.keys()
        return fig

    @pytest.mark.unit
    @pytest.mark.solve
    @pytest.mark.slow
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_3d)
    def test_plot_field_lines(self, DSHAPE_current):
        """Test plotting multiple field lines over 1 transit."""
        eq = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
        fig, ax = plot_field_lines_sfl(
            eq, rho=1, seed_thetas=np.linspace(0, 2 * np.pi, 4), phi_end=2 * np.pi
        )
        return fig


@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_plot_boozer_modes():
    """Test plotting boozer spectrum."""
    eq = get("WISTELL-A")
    fig, ax, data = plot_boozer_modes(
        eq, M_booz=eq.M, N_booz=eq.N, num_modes=7, return_data=True, norm=True
    )
    ax.set_ylim([1e-6, 5e0])
    for string in ["|B|_mn", "B modes", "rho"]:
        assert string in data.keys()
    return fig


@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_plot_boozer_modes_breaking_only():
    """Test plotting symmetry breaking boozer spectrum."""
    eq = get("WISTELL-A")
    fig, ax = plot_boozer_modes(
        eq,
        M_booz=eq.M,
        N_booz=eq.N,
        helicity=(1, -eq.NFP),
        norm=True,
        num_modes=7,
    )
    ax.set_ylim([1e-6, 5e0])
    return fig


@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_plot_boozer_modes_max():
    """Test plotting symmetry breaking boozer spectrum."""
    eq = get("WISTELL-A")
    fig, ax = plot_boozer_modes(
        eq,
        M_booz=eq.M,
        N_booz=eq.N,
        helicity=(1, -eq.NFP),
        max_only=True,
        label="WISTELL-A",
        color="r",
        norm=True,
    )
    ax.set_ylim([1e-6, 5e0])
    return fig


@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_plot_boozer_modes_no_norm():
    """Test plotting boozer spectrum without B0 and norm."""
    eq = get("ESTELL")
    fig, ax = plot_boozer_modes(
        eq, M_booz=eq.M, N_booz=eq.N, num_modes=7, B0=False, log=False
    )
    return fig


@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_2d)
def test_plot_boozer_surface():
    """Test plotting B in boozer coordinates."""
    eq = get("WISTELL-A")
    fig, ax, data = plot_boozer_surface(
        eq, M_booz=eq.M, N_booz=eq.N, return_data=True, rho=0.5, fieldlines=4
    )
    for string in [
        "|B|",
        "theta_Boozer",
        "zeta_Boozer",
    ]:
        assert string in data.keys()
    return fig


@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_plot_qs_error():
    """Test plotting qs error metrics."""
    eq = get("WISTELL-A")
    fig, ax, data = plot_qs_error(
        eq, helicity=(1, -eq.NFP), M_booz=eq.M, N_booz=eq.N, log=True, return_data=True
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
    coils.grid = 100
    coils2 = MixedCoilSet.from_symmetry(coils, NFP, True)
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
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_plot_b_mag():
    """Test plot of |B| on longer field lines for gyrokinetic simulations."""
    psi = 0.5
    npol = 2
    nzgrid = 128
    alpha = 0
    # compute and fit iota profile
    eq = get("W7-X")
    data = eq.compute("iota")
    fi = interp1d(data["rho"], data["iota"])

    # get flux tube coordinate system
    rho = np.sqrt(psi)
    iota = fi(rho)
    zeta = np.linspace(
        -np.pi * npol / np.abs(iota), np.pi * npol / np.abs(iota), 2 * nzgrid + 1
    )
    thetas = alpha * np.ones_like(zeta) + iota * zeta

    rhoa = rho * np.ones_like(zeta)
    c = np.vstack([rhoa, thetas, zeta]).T
    coords = eq.compute_theta_coords(c)
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
def test_plot_surfaces_HELIOTRON():
    """Test plot surfaces of equilibrium for correctness of vartheta lines."""
    fig, ax = plot_surfaces(get("HELIOTRON"))
    return fig
