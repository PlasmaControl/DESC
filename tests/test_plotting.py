import pytest
import unittest
import numpy as np
from desc.plotting import (
    plot_1d,
    plot_2d,
    plot_3d,
    plot_surfaces,
    plot_section,
    plot_logo,
    plot_grid,
    plot_basis,
)
from desc.grid import LinearGrid, ConcentricGrid, QuadratureGrid
from desc.basis import (
    PowerSeries,
    FourierSeries,
    DoubleFourierSeries,
    FourierZernikeBasis,
)
from desc import plotting as dplt


class TestPlot(unittest.TestCase):
    def setUp(self):
        self.names = [
            "B",
            "|B|",
            "B^zeta",
            "B_zeta",
            "B_r",
            "B^zeta_r",
            "B_zeta_r",
            "B**2",
            "B_r**2",
            "B^zeta**2",
            "B_zeta**2",
            "B^zeta_r**2",
            "B_zeta_r**2",
        ]
        self.bases = ["B", "|B|", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B"]
        self.sups = ["", "", "zeta", "", "", "zeta", "", "", "", "zeta", "", "zeta", ""]
        self.subs = ["", "", "", "zeta", "", "", "zeta", "", "", "", "zeta", "", "zeta"]
        self.ds = ["", "", "", "", "r", "r", "r", "", "r", "", "", "r", "r"]
        self.pows = ["", "", "", "", "", "", "", "2", "2", "2", "2", "2", "2"]
        self.name_dicts = []
        for name in self.names:
            self.name_dicts.append(dplt._format_name(name))

    def test_name_dict(self):
        self.assertTrue(
            all(
                [
                    self.name_dicts[i]["base"] == self.bases[i]
                    for i in range(len(self.names))
                ]
            )
        )
        self.assertTrue(
            all(
                [
                    self.name_dicts[i]["sups"] == self.sups[i]
                    for i in range(len(self.names))
                ]
            )
        )
        self.assertTrue(
            all(
                [
                    self.name_dicts[i]["subs"] == self.subs[i]
                    for i in range(len(self.names))
                ]
            )
        )
        self.assertTrue(
            all([self.name_dicts[i]["d"] == self.ds[i] for i in range(len(self.names))])
        )
        self.assertTrue(
            all(
                [
                    self.name_dicts[i]["power"] == self.pows[i]
                    for i in range(len(self.names))
                ]
            )
        )

    def test_name_label(self):
        labels = [dplt._name_label(nd) for nd in self.name_dicts]
        print(labels)
        self.assertTrue(all([label[0] == "$" and label[-1] == "$" for label in labels]))
        self.assertTrue(
            all(
                [
                    "/dr" in labels[i]
                    for i in range(len(labels))
                    if self.name_dicts[i]["d"] != ""
                ]
            )
        )
        self.assertTrue(
            all(
                [
                    "^{" not in labels[i]
                    for i in range(len(labels))
                    if self.name_dicts[i]["sups"] == ""
                    and self.name_dicts[i]["power"] == ""
                ]
            )
        )
        self.assertTrue(
            all(
                [
                    "_{" not in labels[i]
                    for i in range(len(labels))
                    if self.name_dicts[i]["subs"] == ""
                ]
            )
        )


@pytest.mark.mpl_image_compare(tolerance=50)
def test_1d_p(plot_eq):
    fig, ax = plot_1d(plot_eq, "p")
    return fig


@pytest.mark.mpl_image_compare(tolerance=50)
def test_1d_dpdr(plot_eq):
    fig, ax = plot_1d(plot_eq, "p_r")
    return fig


@pytest.mark.mpl_image_compare(tolerance=50)
def test_1d_iota(plot_eq):
    grid = LinearGrid(rho=0.5, theta=np.linspace(0, 2 * np.pi, 100), zeta=0, axis=True)
    fig, ax = plot_1d(plot_eq, "iota", grid=grid)
    return fig


@pytest.mark.mpl_image_compare(tolerance=50)
def test_1d_logpsi(plot_eq):
    fig, ax = plot_1d(plot_eq, "psi", log=True)
    return fig


@pytest.mark.mpl_image_compare(tolerance=50)
def test_2d_logF(plot_eq):
    grid = LinearGrid(
        rho=np.linspace(0, 1, 100),
        theta=np.linspace(0, 2 * np.pi, 100),
        zeta=0,
        axis=True,
    )
    fig, ax = plot_2d(plot_eq, "|F|", log=True, grid=grid)
    return fig


@pytest.mark.mpl_image_compare(tolerance=50)
def test_2d_g_tz(plot_eq):
    grid = LinearGrid(
        rho=0.5,
        theta=np.linspace(0, 2 * np.pi, 100),
        zeta=np.linspace(0, 2 * np.pi, 100),
        axis=True,
    )
    fig, ax = plot_2d(plot_eq, "g", grid=grid)
    return fig


@pytest.mark.mpl_image_compare(tolerance=50)
def test_2d_g_rz(plot_eq):
    grid = LinearGrid(
        rho=np.linspace(0, 1, 100),
        theta=0,
        zeta=np.linspace(0, 2 * np.pi, 100),
        axis=True,
    )
    fig, ax = plot_2d(plot_eq, "g", grid=grid)
    return fig


@pytest.mark.mpl_image_compare(tolerance=50)
def test_2d_lambda(plot_eq):
    fig, ax = plot_2d(plot_eq, "lambda")
    return fig


@pytest.mark.mpl_image_compare(tolerance=50)
def test_3d_B(plot_eq):
    fig, ax = plot_3d(plot_eq, "B^zeta")
    return fig


@pytest.mark.mpl_image_compare(tolerance=50)
def test_3d_J(plot_eq):
    grid = LinearGrid(
        rho=1,
        theta=np.linspace(0, 2 * np.pi, 100),
        zeta=np.linspace(0, 2 * np.pi, 100),
        axis=True,
    )
    fig, ax = plot_3d(plot_eq, "J^theta", grid=grid)
    return fig


@pytest.mark.mpl_image_compare(tolerance=50)
def test_3d_tz(plot_eq):
    grid = LinearGrid(
        rho=0.5,
        theta=np.linspace(0, 2 * np.pi, 100),
        zeta=np.linspace(0, 2 * np.pi, 100),
        axis=True,
    )
    fig, ax = plot_3d(plot_eq, "|F|", log=True, grid=grid)
    return fig


@pytest.mark.mpl_image_compare(tolerance=50)
def test_3d_rz(plot_eq):
    grid = LinearGrid(
        rho=np.linspace(0, 1, 100),
        theta=0,
        zeta=np.linspace(0, 2 * np.pi, 100),
        axis=True,
    )
    fig, ax = plot_3d(plot_eq, "p", grid=grid)
    return fig


@pytest.mark.mpl_image_compare(tolerance=50)
def test_3d_rt(plot_eq):
    grid = LinearGrid(
        rho=np.linspace(0, 1, 100), theta=np.linspace(0, 2 * np.pi, 100), zeta=0
    )
    fig, ax = plot_3d(plot_eq, "psi", grid=grid)
    return fig


@pytest.mark.mpl_image_compare(tolerance=50)
def test_section_J(plot_eq):
    fig, ax = plot_section(plot_eq, "J^rho")
    return fig


@pytest.mark.mpl_image_compare(tolerance=50)
def test_section_Z(plot_eq):
    fig, ax = plot_section(plot_eq, "Z")
    return fig


@pytest.mark.mpl_image_compare(tolerance=50)
def test_section_R(plot_eq):
    fig, ax = plot_section(plot_eq, "R")
    return fig


@pytest.mark.mpl_image_compare(tolerance=50)
def test_section_F(plot_eq):
    fig, ax = plot_section(plot_eq, "F_rho")
    return fig


@pytest.mark.mpl_image_compare(tolerance=50)
def test_section_logF(plot_eq):
    fig, ax = plot_section(plot_eq, "|F|", log=True)
    return fig


@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_surfaces(plot_eq):
    fig, ax = plot_surfaces(plot_eq)
    return fig


@pytest.mark.mpl_image_compare(tolerance=50)
def test_plot_logo():
    fig, ax = plot_logo()
    return fig


class TestPlotGrid(unittest.TestCase):
    @pytest.mark.mpl_image_compare(tolerance=50)
    def test_plot_grid_linear(self):
        grid = LinearGrid(L=10, M=10, N=1)
        fig, ax = plot_grid(grid)
        return fig

    @pytest.mark.mpl_image_compare(tolerance=50)
    def test_plot_grid_quad(self):
        grid = QuadratureGrid(L=10, M=10, N=1)
        fig, ax = plot_grid(grid)
        return fig

    @pytest.mark.mpl_image_compare(tolerance=50)
    def test_plot_grid_jacobi(self):
        grid = ConcentricGrid(L=20, M=10, N=1, node_pattern="jacobi")
        fig, ax = plot_grid(grid)
        return fig

    @pytest.mark.mpl_image_compare(tolerance=50)
    def test_plot_grid_cheb1(self):
        grid = ConcentricGrid(L=20, M=10, N=1, node_pattern="cheb1")
        fig, ax = plot_grid(grid)
        return fig

    @pytest.mark.mpl_image_compare(tolerance=50)
    def test_plot_grid_cheb2(self):
        grid = ConcentricGrid(L=20, M=10, N=1, node_pattern="cheb2")
        fig, ax = plot_grid(grid)
        return fig

    @pytest.mark.mpl_image_compare(tolerance=50)
    def test_plot_grid_ocs(self):
        grid = ConcentricGrid(L=20, M=10, N=1, node_pattern="ocs")
        fig, ax = plot_grid(grid)
        return fig


class TestPlotBasis(unittest.TestCase):
    @pytest.mark.mpl_image_compare(tolerance=50)
    def test_plot_basis_powerseries(self):
        basis = PowerSeries(L=6)
        fig, ax = plot_basis(basis)
        return fig

    @pytest.mark.mpl_image_compare(tolerance=50)
    def test_plot_basis_fourierseries(self):
        basis = FourierSeries(N=3)
        fig, ax = plot_basis(basis)
        return fig

    @pytest.mark.mpl_image_compare(tolerance=50)
    def test_plot_basis_doublefourierseries(self):
        basis = DoubleFourierSeries(M=3, N=2)
        fig, ax = plot_basis(basis)
        return fig

    @pytest.mark.mpl_image_compare(tolerance=50)
    def test_plot_basis_fourierzernike(self):
        basis = FourierZernikeBasis(L=8, M=3, N=2)
        fig, ax = plot_basis(basis)
        return fig
