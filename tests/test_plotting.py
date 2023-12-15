"""Regression tests for plotting functions, by comparing to saved baseline images."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.interpolate import interp1d

from desc.basis import (
    DoubleFourierSeries,
    FourierSeries,
    FourierZernikeBasis,
    ChebyshevZernikeBasis,
    PowerSeries,
)
from desc.coils import CoilSet, FourierXYZCoil
from desc.compute import data_index
from desc.compute.utils import surface_averages
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.examples import get
from desc.grid import ConcentricGrid, Grid, LinearGrid, QuadratureGrid

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

@pytest.mark.mirror_unit
def test_section_J():
    """Test plotting poincare section of radial current."""
    eq = Equilibrium(L=3, M=3, N=3, mirror = True)
    fig, ax, data = plot_section(eq, "J^rho", return_data=True)
    assert "R" in data.keys()
    assert "Z" in data.keys()
    assert "J^rho" in data.keys()
    assert "normalization" in data.keys()
    assert data["normalization"] == 1

    return fig


@pytest.mark.mirror_unit
def test_section_Z():
    """Test plotting poincare section of Z coordinate."""
    eq = Equilibrium(L=3, M=3, N=3, mirror = True)
    fig, ax = plot_section(eq, "Z")
    return fig


@pytest.mark.mirror_unit
def test_section_R():
    """Test plotting poincare section of R coordinate."""
    eq = Equilibrium(L=3, M=3, N=3, mirror = True)
    fig, ax = plot_section(eq, "R")
    return fig


@pytest.mark.mirror_unit
def test_section_F():
    """Test plotting poincare section of radial force."""
    eq = Equilibrium(L=3, M=3, N=3, mirror = True)
    fig, ax = plot_section(eq, "F_rho")
    return fig


@pytest.mark.mirror_unit
def test_section_F_normalized_vac():
    """Test plotting poincare section of normalized vacuum force error."""
    eq = Equilibrium(L=3, M=3, N=3, mirror = True)
    fig, ax = plot_section(eq, "|F|", norm_F=True)
    return fig


# @pytest.mark.broken_unit
# def test_section_logF():
#     """Test plotting poincare section of force magnitude on log scale."""
#     eq = Equilibrium(L=3, M=3, N=3, mirror = True)
#     fig, ax = plot_section(eq, "|F|", log=True)
#     return fig

@pytest.mark.mirror_unit
def test_3d_mirror():
    """Testing the 2d_plot with mirror."""
    eq = Equilibrium(L=3, M=3, N=3, mirror = True)
    fig, ax = plot_3d(eq, "|F|")
    return fig

@pytest.mark.mirror_unit
def test_3d_B():
    """Test 3d plot of toroidal field."""
    eq = Equilibrium(L=3, M=3, N=3, mirror = True)
    fig, ax, data = plot_3d(eq, "B^zeta", return_data=True)
    assert "X" in data.keys()
    assert "Y" in data.keys()
    assert "Z" in data.keys()
    assert "B^zeta" in data.keys()
    return fig


@pytest.mark.mirror_unit
def test_3d_J():
    """Test 3d plotting of poloidal current."""
    eq = Equilibrium(L=3, M=3, N=3, mirror = True)
    grid = LinearGrid(rho=1.0, theta=100, zeta=100)
    fig, ax = plot_3d(eq, "J^theta", grid=grid)
    return fig


# @pytest.mark.unit
# def test_3d_tz():
#     """Test 3d plot of force on interior surface."""
#     eq = Equilibrium(L=3, M=3, N=3, mirror = True)
#     grid = LinearGrid(rho=0.5, theta=100, zeta=100)
#     fig, ax = plot_3d(eq, "|F|", log=True, grid=grid)
#     return fig


# @pytest.mark.unit
# def test_3d_rz():
#     """Test 3d plotting of pressure on toroidal cross section."""
#     eq = Equilibrium(L=3, M=3, N=3, mirror = True)
#     grid = LinearGrid(rho=100, theta=0.0, zeta=100)
#     fig, ax = plot_3d(eq, "p", grid=grid)
#     return fig


@pytest.mark.mirror_unit
def test_3d_rt():
    """Test 3d plotting of flux on poloidal ribbon."""
    eq = Equilibrium(L=3, M=3, N=3, mirror = True)
    grid = LinearGrid(rho=100, theta=100, zeta=0.0)
    fig, ax = plot_3d(eq, "psi", grid=grid)
    return fig


@pytest.mark.mirror_unit
def test_plot_basis_chebyshevzernike():
    """Test plotting chebyshev-zernike basis."""
    basis = ChebyshevZernikeBasis(L=8, M=3, N=2)
    fig, ax, data = plot_basis(basis, return_data=True)
    for string in ["amplitude", "l", "rho", "m", "theta"]:
        assert string in data.keys()



# @pytest.mark.unit
# def test_2d_logF(DSHAPE_current):
#     """Test plotting 2d force error with log scale."""
#     eq = EquilibriaFamily.load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
#     grid = LinearGrid(rho=100, theta=100, zeta=0.0)
#     fig, ax, data = plot_2d(
#         eq, "|F|", log=True, grid=grid, figsize=(4, 4), return_data=True
#     )
#     assert "|F|" in data.keys()
#     return fig


#Need increasing contours
# @pytest.mark.unit
# def test_2d_g_tz():
#     """Test plotting 2d metric coefficients vs theta/zeta."""
#     eq = Equilibrium(L=3, M=3, N=3, mirror = True)
#     grid = LinearGrid(rho=0.5, theta=100, zeta=100)
#     fig, ax, data = plot_2d(eq, "sqrt(g)", grid=grid, figsize=(4, 4), return_data=True)
#     assert "theta" in data.keys()
#     assert "zeta" in data.keys()
#     assert "sqrt(g)" in data.keys()
#     return fig


@pytest.mark.unit
def test_2d_g_rz():
    """Test plotting 2d metric coefficients vs rho/zeta."""
    eq = Equilibrium(L=3, M=3, N=3, mirror = True)
    grid = LinearGrid(rho=100, theta=0.0, zeta=100)
    fig, ax, data = plot_2d(eq, "sqrt(g)", grid=grid, figsize=(4, 4), return_data=True)
    assert "rho" in data.keys()
    assert "zeta" in data.keys()
    assert "sqrt(g)" in data.keys()
    return fig


# @pytest.mark.unit
# def test_2d_lambda():
#     """Test plotting lambda on 2d grid."""
#     eq = Equilibrium(L=3, M=3, N=3, mirror = True)
#     fig, ax, data = plot_2d(eq, "lambda", figsize=(4, 4), return_data=True)
#     assert "lambda" in data.keys()
#     return fig


# @pytest.mark.unit
# def test_plot_cov_basis():
#     """Test 2d plot of norm of e_rho."""
#     eq = Equilibrium(L=3, M=3, N=3, mirror = True)
#     fig, ax, data = plot_2d(eq, "e_rho", figsize=(4, 4), return_data=True)
#     return fig


@pytest.mark.unit
def test_plot_magnetic_tension():
    """Test 2d plot of magnetic tension."""
    eq = Equilibrium(L=3, M=3, N=3, mirror = True)
    fig, ax, data = plot_2d(eq, "|(B*grad)B|", figsize=(4, 4), return_data=True)
    return fig


# @pytest.mark.unit
# def test_plot_magnetic_pressure():
#     """Test 2d plot of magnetic pressure."""
#     eq = Equilibrium(L=3, M=3, N=3, mirror = True)
#     fig, ax, data = plot_2d(eq, "|grad(|B|^2)|/2mu0", figsize=(4, 4), return_data=True)
#     return fig


# @pytest.mark.unit
# def test_plot_gradpsi():
#     """Test 2d plot of norm of grad(rho)."""
#     eq = Equilibrium(L=3, M=3, N=3, mirror = True)
#     fig, ax, data = plot_2d(eq, "|grad(rho)|", figsize=(4, 4), return_data=True)
#     return fig


# @pytest.mark.unit
# def test_plot_normF_2d():
#     """Test 2d plot of normalized force."""
#     grid = LinearGrid(rho=np.array(0.8), M=20, N=2)
#     eq = Equilibrium(L=3, M=3, N=3, mirror = True)
#     fig, ax, data = plot_2d(
#         eq, "|F|", norm_F=True, figsize=(4, 4), return_data=True, grid=grid
#     )
#     for string in ["|F|", "theta", "zeta"]:
#         assert string in data.keys()
#     return fig