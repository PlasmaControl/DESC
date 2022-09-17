import pytest
import unittest
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

tol_1d = 1
tol_2d = 1
tol_3d = 5


def test_kwarg_warning(DummyStellarator):
    eq = Equilibrium.load(load_from=str(DummyStellarator["output_path"]))
    with pytest.raises(AssertionError):
        fig, ax = plot_1d(eq, "p", not_a_kwarg=True)
    return None


@pytest.mark.mpl_image_compare(tolerance=tol_1d)
def test_1d_p(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_1d(eq, "p", figsize=(4, 4))
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_1d)
def test_1d_dpdr(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_1d(eq, "p_r", figsize=(4, 4))
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_1d)
def test_1d_iota(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    grid = LinearGrid(rho=0.5, theta=100, zeta=0.0)
    fig, ax = plot_1d(eq, "iota", grid=grid, figsize=(4, 4))
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_1d)
def test_1d_logpsi(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_1d(eq, "psi", log=True, figsize=(4, 4))
    return fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_2d_logF(SOLOVEV):
    # plot test is inconsistent
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    grid = LinearGrid(rho=100, theta=100, zeta=0.0)
    fig, ax = plot_2d(eq, "|F|", log=True, grid=grid, figsize=(4, 4))
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_2d)
def test_2d_g_tz(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    grid = LinearGrid(rho=0.5, theta=100, zeta=100)
    fig, ax = plot_2d(eq, "sqrt(g)", grid=grid, figsize=(4, 4))
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_2d)
def test_2d_g_rz(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    grid = LinearGrid(rho=100, theta=0.0, zeta=100)
    fig, ax = plot_2d(eq, "sqrt(g)", grid=grid, figsize=(4, 4))
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_2d)
def test_2d_lambda(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_2d(eq, "lambda", figsize=(4, 4))
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_3d)
def test_3d_B(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_3d(eq, "B^zeta")
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_3d)
def test_3d_J(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    grid = LinearGrid(rho=1.0, theta=100, zeta=100)
    fig, ax = plot_3d(eq, "J^theta", grid=grid)
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_3d)
def test_3d_tz(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    grid = LinearGrid(rho=0.5, theta=100, zeta=100)
    fig, ax = plot_3d(eq, "|F|", log=True, grid=grid)
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_3d)
def test_3d_rz(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    grid = LinearGrid(rho=100, theta=0.0, zeta=100)
    fig, ax = plot_3d(eq, "p", grid=grid)
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_3d)
def test_3d_rt(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    grid = LinearGrid(rho=100, theta=100, zeta=0.0)
    fig, ax = plot_3d(eq, "psi", grid=grid)
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_1d)
def test_fsa_I(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_fsa(eq, "B_theta")
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_1d)
def test_fsa_G(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_fsa(eq, "B_zeta", log=True)
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_1d)
def test_fsa_F_normalized(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_fsa(eq, "|F|", log=True, norm_F=True)
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_2d)
def test_section_J(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_section(eq, "J^rho")
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_2d)
def test_section_Z(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_section(eq, "Z")
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_2d)
def test_section_R(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_section(eq, "R")
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_2d)
def test_section_F(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_section(eq, "F_rho")
    return fig


@pytest.mark.mpl_image_compare(tolerance=50)
def test_section_logF(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_section(eq, "|F|", log=True)
    return fig


@pytest.mark.slow
@pytest.mark.mpl_image_compare(tolerance=tol_2d)
def test_plot_surfaces(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_surfaces(eq)
    return fig


@pytest.mark.slow
@pytest.mark.mpl_image_compare(tolerance=tol_2d)
def test_plot_surfaces_no_theta(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_surfaces(eq, theta=False)
    return fig


@pytest.mark.slow
@pytest.mark.mpl_image_compare(tolerance=tol_2d)
def test_plot_comparison(DSHAPE):
    eqf = EquilibriaFamily.load(load_from=str(DSHAPE["desc_h5_path"]))
    fig, ax = plot_comparison(eqf)
    return fig


@pytest.mark.slow
@pytest.mark.mpl_image_compare(tolerance=tol_2d)
def test_plot_comparison_no_theta(DSHAPE):
    eqf = EquilibriaFamily.load(load_from=str(DSHAPE["desc_h5_path"]))
    fig, ax = plot_comparison(eqf, theta=0)
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_2d)
def test_plot_con_basis(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_2d(eq, "e^rho", component="R", figsize=(4, 4))
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_2d)
def test_plot_cov_basis(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_2d(eq, "e_rho", figsize=(4, 4))
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_2d)
def test_plot_magnetic_tension(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_2d(eq, "|(B*grad)B|", figsize=(4, 4))
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_2d)
def test_plot_magnetic_pressure(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_2d(eq, "|grad(|B|^2)|/2mu0", figsize=(4, 4))
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_2d)
def test_plot_gradpsi(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_2d(eq, "|grad(rho)|", figsize=(4, 4))
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_2d)
def test_plot_normF_2d(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_2d(eq, "|F|", norm_F=True, figsize=(4, 4))
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_2d)
def test_plot_normF_section(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_section(eq, "|F|", norm_F=True, log=True)
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_2d)
def test_plot_coefficients(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_coefficients(eq)
    ax[0, 0].set_ylim((1e-9, 1e1))
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_2d)
def test_plot_logo():
    fig, ax = plot_logo()
    return fig


class TestPlotGrid(unittest.TestCase):
    @pytest.mark.mpl_image_compare(tolerance=tol_2d)
    def test_plot_grid_linear(self):
        grid = LinearGrid(rho=10, theta=10, zeta=1)
        fig, ax = plot_grid(grid)
        return fig

    @pytest.mark.mpl_image_compare(tolerance=tol_2d)
    def test_plot_grid_quad(self):
        grid = QuadratureGrid(L=10, M=10, N=1)
        fig, ax = plot_grid(grid, figsize=(6, 6))
        return fig

    @pytest.mark.mpl_image_compare(tolerance=tol_2d)
    def test_plot_grid_jacobi(self):
        grid = ConcentricGrid(L=20, M=10, N=1, node_pattern="jacobi")
        fig, ax = plot_grid(grid)
        return fig

    @pytest.mark.mpl_image_compare(tolerance=tol_2d)
    def test_plot_grid_cheb1(self):
        grid = ConcentricGrid(L=20, M=10, N=1, node_pattern="cheb1")
        fig, ax = plot_grid(grid)
        return fig

    @pytest.mark.mpl_image_compare(tolerance=tol_2d)
    def test_plot_grid_cheb2(self):
        grid = ConcentricGrid(L=20, M=10, N=1, node_pattern="cheb2")
        fig, ax = plot_grid(grid)
        return fig

    @pytest.mark.mpl_image_compare(tolerance=tol_2d)
    def test_plot_grid_ocs(self):
        grid = ConcentricGrid(L=20, M=10, N=1, node_pattern="ocs")
        fig, ax = plot_grid(grid)
        return fig


class TestPlotBasis(unittest.TestCase):
    @pytest.mark.mpl_image_compare(tolerance=tol_2d)
    def test_plot_basis_powerseries(self):
        basis = PowerSeries(L=6)
        fig, ax = plot_basis(basis)
        return fig

    @pytest.mark.mpl_image_compare(tolerance=tol_2d)
    def test_plot_basis_fourierseries(self):
        basis = FourierSeries(N=3)
        fig, ax = plot_basis(basis)
        return fig

    @pytest.mark.slow
    @pytest.mark.mpl_image_compare(tolerance=tol_2d)
    def test_plot_basis_doublefourierseries(self):
        basis = DoubleFourierSeries(M=3, N=2)
        fig, ax = plot_basis(basis)
        return fig

    @pytest.mark.slow
    @pytest.mark.mpl_image_compare(tolerance=tol_2d)
    def test_plot_basis_fourierzernike(self):
        basis = FourierZernikeBasis(L=8, M=3, N=2)
        fig, ax = plot_basis(basis)
        return fig


class TestPlotFieldLines(unittest.TestCase):
    def test_find_idx(self):
        # pick the first grid node point, add epsilon to it, check it returns idx of 0
        grid = LinearGrid(L=1, M=2, N=2, axis=False)
        epsilon = np.finfo(float).eps
        test_point = grid.nodes[0, :] + epsilon
        idx = _find_idx(*test_point, grid=grid)
        self.assertEqual(idx, 0)

    def test_field_line_Rbf(self):
        pass


@pytest.mark.slow
@pytest.mark.mpl_image_compare(tolerance=tol_2d)
def test_plot_field_line(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax, _ = plot_field_lines_sfl(eq, rho=1, seed_thetas=0, phi_end=2 * np.pi)
    return fig


@pytest.mark.slow
@pytest.mark.mpl_image_compare(tolerance=tol_3d)
def test_plot_field_lines(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax, _ = plot_field_lines_sfl(
        eq, rho=1, seed_thetas=np.linspace(0, 2 * np.pi, 4), phi_end=2 * np.pi
    )
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_1d)
def test_plot_boozer_modes(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_boozer_modes(eq)
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_2d)
def test_plot_boozer_surface(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_boozer_surface(eq, figsize=(4, 4))
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_1d)
def test_plot_qs_error(SOLOVEV):
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    fig, ax = plot_qs_error(eq, helicity=(0, 0), log=False)
    return fig


@pytest.mark.mpl_image_compare(tolerance=tol_3d)
def test_plot_coils():
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
