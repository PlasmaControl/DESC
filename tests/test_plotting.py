import pytest
import numpy as np
from desc.plotting import (
    plot_1d,
    plot_2d,
    plot_3d,
    plot_fsa,
    plot_section,
    plot_surfaces,
    plot_comparison,
    plot_logo,
    plot_grid,
    plot_basis,
    plot_coefficients,
    _find_idx,
    plot_field_lines_sfl,
    plot_boozer_modes,
    plot_boozer_surface,
    plot_qs_error,
    plot_coils,
)
from desc.grid import LinearGrid, ConcentricGrid, QuadratureGrid
from desc.basis import (
    PowerSeries,
    FourierSeries,
    DoubleFourierSeries,
    FourierZernikeBasis,
)
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.coils import FourierXYZCoil, CoilSet


@pytest.mark.unit
def test_kwarg_warning(DummyStellarator):
    """Test that passing in unknown kwargs throws an error."""
    eq = Equilibrium.load(load_from=str(DummyStellarator["output_path"]))
    with pytest.raises(AssertionError):
        fig, ax = plot_1d(eq, "p", not_a_kwarg=True)
    return None


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_1d_p(SOLOVEV):
    """Test plotting 1d pressure profile."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_1d(eq, "p", figsize=(4, 4))
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_1d_dpdr(SOLOVEV):
    """Test plotting 1d pressure derivative."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_1d(eq, "p_r", figsize=(4, 4))
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_1d_iota(SOLOVEV):
    """Test plotting 1d rotational transform."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    grid = LinearGrid(rho=0.5, theta=100, zeta=0.0)
    fig, ax = plot_1d(eq, "iota", grid=grid, figsize=(4, 4))
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_1d_logpsi(SOLOVEV):
    """Test plotting 1d flux funciton with log scale."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_1d(eq, "psi", log=True, figsize=(4, 4))
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=72)
def test_2d_logF(SOLOVEV):
    """Test plotting 2d force error with log scale."""
    # plot test is inconsistent
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    grid = LinearGrid(rho=100, theta=100, zeta=0.0)
    fig, ax = plot_2d(eq, "|F|", log=True, grid=grid, figsize=(4, 4))
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_2d_g_tz(SOLOVEV):
    """Test plotting 2d metric coefficients vs theta/zeta."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    grid = LinearGrid(rho=0.5, theta=100, zeta=100)
    fig, ax = plot_2d(eq, "sqrt(g)", grid=grid, figsize=(4, 4))
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_2d_g_rz(SOLOVEV):
    """Test plotting 2d metric coefficients vs rho/zeta."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    grid = LinearGrid(rho=100, theta=0.0, zeta=100)
    fig, ax = plot_2d(eq, "sqrt(g)", grid=grid, figsize=(4, 4))
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_2d_lambda(SOLOVEV):
    """Test plotting lambda on 2d grid."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_2d(eq, "lambda", figsize=(4, 4))
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_3d_B(SOLOVEV):
    """Test 3d plot of toroidal field."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_3d(eq, "B^zeta")
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_3d_J(SOLOVEV):
    """Test 3d plotting of poloidal current."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    grid = LinearGrid(rho=1.0, theta=100, zeta=100)
    fig, ax = plot_3d(eq, "J^theta", grid=grid)
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_3d_tz(SOLOVEV):
    """Test 3d plot of force on interior surface."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    grid = LinearGrid(rho=0.5, theta=100, zeta=100)
    fig, ax = plot_3d(eq, "|F|", log=True, grid=grid)
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_3d_rz(SOLOVEV):
    """Test 3d plotting of pressure on toroidal cross section."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    grid = LinearGrid(rho=100, theta=0.0, zeta=100)
    fig, ax = plot_3d(eq, "p", grid=grid)
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_3d_rt(SOLOVEV):
    """Test 3d plotting of flux on poloidal ribbon."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    grid = LinearGrid(rho=100, theta=100, zeta=0.0)
    fig, ax = plot_3d(eq, "psi", grid=grid)
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_fsa_I(SOLOVEV):
    """Test plotting of flux surface average toroidal current."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_fsa(eq, "B_theta")
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_fsa_G(SOLOVEV):
    """Test plotting of flux surface average poloidal current."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_fsa(eq, "B_zeta", log=True)
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=60)
def test_section_J(SOLOVEV):
    """Test plotting poincare section of radial current."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_section(eq, "J^rho")
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_section_Z(SOLOVEV):
    """Test plotting poincare section of Z coordinate."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_section(eq, "Z")
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_section_R(SOLOVEV):
    """Test plotting poincare section of R coordinate."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_section(eq, "R")
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_section_F(SOLOVEV):
    """Test plotting poincare section of radial force."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_section(eq, "F_rho")
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_section_logF(SOLOVEV):
    """Test plotting poincare section of force magnitude on log scale."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_section(eq, "|F|", log=True)
    return fig


@pytest.mark.slow
@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_surfaces(SOLOVEV):
    """Test plotting flux surfaces."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_surfaces(eq)
    return fig


@pytest.mark.slow
@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_surfaces_no_theta(SOLOVEV):
    """Test plotting flux surfaces without theta contours."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_surfaces(eq, theta=False)
    return fig


@pytest.mark.slow
@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_comparison(DSHAPE):
    """Test plotting comparison of flux surfaces."""
    eqf = EquilibriaFamily.load(load_from=str(DSHAPE["desc_h5_path"]))
    fig, ax = plot_comparison(eqf)
    return fig


@pytest.mark.slow
@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_comparison_no_theta(DSHAPE):
    """Test plotting comparison of flux surfaces without theta contours."""
    eqf = EquilibriaFamily.load(load_from=str(DSHAPE["desc_h5_path"]))
    fig, ax = plot_comparison(eqf, theta=0)
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_con_basis(SOLOVEV):
    """Test 2d plot of R component of e^rho."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_2d(eq, "e^rho", component="R", figsize=(4, 4))
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_cov_basis(SOLOVEV):
    """Test 2d plot of norm of e_rho."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_2d(eq, "e_rho", figsize=(4, 4))
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_magnetic_tension(SOLOVEV):
    """Test 2d plot of magnetic tension."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_2d(eq, "|(B*grad)B|", figsize=(4, 4))
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_magnetic_pressure(SOLOVEV):
    """Test 2d plot of magnetic pressure."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_2d(eq, "|grad(|B|^2)|/2mu0", figsize=(4, 4))
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_gradpsi(SOLOVEV):
    """Test 2d plot of norm of grad(rho)."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_2d(eq, "|grad(rho)|", figsize=(4, 4))
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=55)
def test_plot_normF_2d(SOLOVEV):
    """Test 2d plot of normalized force magnitude."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_2d(eq, "|F|", norm_F=True, figsize=(4, 4))
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_normF_section(SOLOVEV):
    """Test poincare section plot of normalized force magnitude."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_section(eq, "|F|", norm_F=True, log=True)
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=60)
def test_plot_coefficients(SOLOVEV):
    """Test scatter plot of spectral coefficients."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_coefficients(eq)
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_logo():
    """Test plotting the DESC logo."""
    fig, ax = plot_logo()
    return fig


class TestPlotGrid:
    """Tests for the plot_grid function."""

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(tolerance=50)
    def test_plot_grid_linear(self):
        """Test plotting linear grid."""
        grid = LinearGrid(rho=10, theta=10, zeta=1)
        fig, ax = plot_grid(grid)
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(tolerance=50)
    def test_plot_grid_quad(self):
        """Test plotting quadrature grid."""
        grid = QuadratureGrid(L=10, M=10, N=1)
        fig, ax = plot_grid(grid, figsize=(6, 6))
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(tolerance=50)
    def test_plot_grid_jacobi(self):
        """Test plotting concentric grid with jacobi nodes."""
        grid = ConcentricGrid(L=20, M=10, N=1, node_pattern="jacobi")
        fig, ax = plot_grid(grid)
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(tolerance=50)
    def test_plot_grid_cheb1(self):
        """Test plotting concentric grid with chebyshev 1 nodes."""
        grid = ConcentricGrid(L=20, M=10, N=1, node_pattern="cheb1")
        fig, ax = plot_grid(grid)
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(tolerance=50)
    def test_plot_grid_cheb2(self):
        """Test plotting concentric grid with chebyshev 2 nodes."""
        grid = ConcentricGrid(L=20, M=10, N=1, node_pattern="cheb2")
        fig, ax = plot_grid(grid)
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(tolerance=50)
    def test_plot_grid_ocs(self):
        """Test plotting concentric grid with optimal concentric sampling nodes."""
        grid = ConcentricGrid(L=20, M=10, N=1, node_pattern="ocs")
        fig, ax = plot_grid(grid)
        return fig


class TestPlotBasis:
    """Tests for plot_basis function."""

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(tolerance=50)
    def test_plot_basis_powerseries(self):
        """Test plotting power series basis."""
        basis = PowerSeries(L=6)
        fig, ax = plot_basis(basis)
        return fig

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(tolerance=50)
    def test_plot_basis_fourierseries(self):
        """Test plotting fourier series basis."""
        basis = FourierSeries(N=3)
        fig, ax = plot_basis(basis)
        return fig

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.mpl_image_compare(tolerance=50)
    def test_plot_basis_doublefourierseries(self):
        """Test plotting double fourier series basis."""
        basis = DoubleFourierSeries(M=3, N=2)
        fig, ax = plot_basis(basis)
        return fig

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.mpl_image_compare(tolerance=50)
    def test_plot_basis_fourierzernike(self):
        """Test plotting fourier-zernike basis."""
        basis = FourierZernikeBasis(L=8, M=3, N=2)
        fig, ax = plot_basis(basis)
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
    def test_field_line_Rbf(self):
        pass

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.mpl_image_compare(tolerance=50)
    def test_plot_field_line(self, SOLOVEV):
        """Test plotting single field line over 1 transit."""
        eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
        fig, ax, _ = plot_field_lines_sfl(eq, rho=1, seed_thetas=0, phi_end=2 * np.pi)
        return fig

    @pytest.mark.unit
    @pytest.mark.slow
    @pytest.mark.mpl_image_compare(tolerance=50)
    def test_plot_field_lines(self, SOLOVEV):
        """Test plotting multiple field lines over 1 transit."""
        eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
        fig, ax, _ = plot_field_lines_sfl(
            eq, rho=1, seed_thetas=np.linspace(0, 2 * np.pi, 4), phi_end=2 * np.pi
        )
        return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_boozer_modes(SOLOVEV):
    """Test plotting boozer spectrum."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_boozer_modes(eq)
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_boozer_surface(SOLOVEV):
    """Test plotting B in boozer coordinates."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_boozer_surface(eq, figsize=(4, 4))
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_qs_error(SOLOVEV):
    """Test plotting qs error metrics."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_qs_error(eq, helicity=(0, 0), log=False)
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_qh_optimization(precise_QH):
    """Test plotting B in boozer coordinates from precise QH optimization."""
    eq = EquilibriaFamily.load(load_from=str(precise_QH["optimal_h5_path"]))[-1]
    fig, ax = plot_boozer_surface(eq)
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_coils():
    """Test 3d plotting of coils with currents."""
    N = 48
    NFP = 4
    I = 1
    coil = FourierXYZCoil()
    coil.rotate(angle=np.pi / N)
    coils = CoilSet.linspaced_angular(coil, I, [0, 0, 1], np.pi / NFP, N // NFP // 2)
    coils.grid = 100
    coils2 = CoilSet.from_symmetry(coils, NFP, True)
    fig, ax = plot_coils(coils2)

    return fig
