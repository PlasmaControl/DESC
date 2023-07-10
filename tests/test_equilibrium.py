"""Tests for Equilibrium class."""

import os
import pickle
import warnings

import numpy as np
import pytest
from netCDF4 import Dataset

import desc.examples
from desc.__main__ import main
from desc.backend import sign
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.grid import Grid, LinearGrid
from desc.io import InputReader
from desc.objectives import get_equilibrium_objective

from .utils import area_difference, compute_coords


@pytest.mark.unit
@pytest.mark.solve
def test_compute_geometry(DSHAPE_current):
    """Test computation of plasma geometric values."""

    def test(stellarator):
        # VMEC values
        file = Dataset(str(stellarator["vmec_nc_path"]), mode="r")
        V_vmec = float(file.variables["volume_p"][-1])
        R0_vmec = float(file.variables["Rmajor_p"][-1])
        a_vmec = float(file.variables["Aminor_p"][-1])
        ar_vmec = float(file.variables["aspect"][-1])
        file.close()

        # DESC values
        eq = EquilibriaFamily.load(load_from=str(stellarator["desc_h5_path"]))[-1]
        data = eq.compute("R0/a")
        V_desc = data["V"]
        R0_desc = data["R0"]
        a_desc = data["a"]
        ar_desc = data["R0/a"]

        assert abs(V_vmec - V_desc) < 5e-3
        assert abs(R0_vmec - R0_desc) < 5e-3
        assert abs(a_vmec - a_desc) < 5e-3
        assert abs(ar_vmec - ar_desc) < 5e-3

    test(DSHAPE_current)


@pytest.mark.slow
@pytest.mark.unit
@pytest.mark.solve
def test_compute_theta_coords(DSHAPE_current):
    """Test root finding for theta(theta*,lambda(theta))."""
    eq = EquilibriaFamily.load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]

    rho = np.linspace(0.01, 0.99, 200)
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi, 200, endpoint=False)

    nodes = np.vstack([rho, theta, zeta]).T
    coords = eq.compute("lambda", grid=Grid(nodes, sort=False))
    flux_coords = nodes.copy()
    flux_coords[:, 1] += coords["lambda"]

    geom_coords = eq.compute_theta_coords(flux_coords)
    geom_coords = np.array(geom_coords)

    # catch difference between 0 and 2*pi
    if geom_coords[0, 1] > np.pi:  # theta[0] = 0
        geom_coords[0, 1] = geom_coords[0, 1] - 2 * np.pi

    np.testing.assert_allclose(nodes, geom_coords, rtol=1e-5, atol=1e-5)


@pytest.mark.slow
@pytest.mark.unit
@pytest.mark.solve
def test_compute_flux_coords(DSHAPE_current):
    """Test root finding for (rho,theta,zeta) from (R,phi,Z)."""
    eq = EquilibriaFamily.load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]

    rho = np.linspace(0.01, 0.99, 200)
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi, 200, endpoint=False)

    nodes = np.vstack([rho, theta, zeta]).T
    coords = eq.compute(["R", "Z"], grid=Grid(nodes, sort=False))
    real_coords = np.vstack([coords["R"].flatten(), zeta, coords["Z"].flatten()]).T

    flux_coords = eq.compute_flux_coords(real_coords)
    flux_coords = np.array(flux_coords)

    # catch difference between 0 and 2*pi
    if flux_coords[0, 1] > np.pi:  # theta[0] = 0
        flux_coords[0, 1] = flux_coords[0, 1] - 2 * np.pi

    np.testing.assert_allclose(nodes, flux_coords, rtol=1e-5, atol=1e-5)


@pytest.mark.unit
def test_map_coordinates():
    """Test root finding for (rho,theta,zeta) from (R,phi,Z)."""
    eq = desc.examples.get("DSHAPE")

    inbasis = ["alpha", "phi", "rho"]
    outbasis = ["rho", "theta_PEST", "zeta"]

    rho = np.linspace(0.01, 0.99, 100)
    theta = np.linspace(0, np.pi, 100, endpoint=False)
    zeta = np.linspace(0, np.pi, 100, endpoint=False)

    grid = Grid(np.vstack([rho, theta, zeta]).T, sort=False)
    in_data = eq.compute(inbasis, grid=grid)
    in_coords = np.stack([in_data[k] for k in inbasis], axis=-1)
    out_data = eq.compute(outbasis, grid=grid)
    out_coords = np.stack([out_data[k] for k in outbasis], axis=-1)

    out = eq.map_coordinates(in_coords, inbasis, outbasis)
    np.testing.assert_allclose(out, out_coords, rtol=1e-4, atol=1e-4)


@pytest.mark.unit
def test_map_coordinates2():
    """Test root finding for (rho,theta,zeta) for common use cases."""
    eq = desc.examples.get("W7-X")

    n = 100
    # finding coordinates along a single field line
    coords = np.array([np.ones(n), np.zeros(n), np.linspace(0, 10 * np.pi, n)]).T
    out = eq.map_coordinates(
        coords,
        ["rho", "alpha", "zeta"],
        ["rho", "theta", "zeta"],
        period=(np.inf, 2 * np.pi, 10 * np.pi),
    )
    assert not np.any(np.isnan(out))

    # contours of const theta for plotting
    grid_kwargs = {
        "rho": np.linspace(0, 1, 10),
        "NFP": eq.NFP,
        "theta": np.linspace(0, 2 * np.pi, 3, endpoint=False),
        "zeta": np.linspace(0, 2 * np.pi / eq.NFP, 2, endpoint=False),
    }
    t_grid = LinearGrid(**grid_kwargs)

    out = eq.map_coordinates(
        t_grid.nodes,
        ["rho", "theta_PEST", "phi"],
        ["rho", "theta", "zeta"],
        period=(np.inf, 2 * np.pi, 2 * np.pi),
    )
    assert not np.any(np.isnan(out))


@pytest.mark.slow
@pytest.mark.unit
@pytest.mark.solve
def test_to_sfl(DSHAPE_current):
    """Test converting an equilibrium to straight field line coordinates."""
    eq = EquilibriaFamily.load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]

    Rr1, Zr1, Rv1, Zv1 = compute_coords(eq)
    Rr2, Zr2, Rv2, Zv2 = compute_coords(eq.to_sfl())
    rho_err, theta_err = area_difference(Rr1, Rr2, Zr1, Zr2, Rv1, Rv2, Zv1, Zv2)

    np.testing.assert_allclose(rho_err, 0, atol=2.5e-4)
    np.testing.assert_allclose(theta_err, 0, atol=1e-4)


@pytest.mark.slow
@pytest.mark.regression
def test_continuation_resolution(tmpdir_factory):
    """Test that stepping resolution in continuation method works correctly."""
    input_path = ".//tests//inputs//res_test"
    output_dir = tmpdir_factory.mktemp("result")
    desc_h5_path = output_dir.join("res_test_out.h5")

    cwd = os.path.dirname(__file__)
    exec_dir = os.path.join(cwd, "..")
    input_filename = os.path.join(exec_dir, input_path)

    args = ["-o", str(desc_h5_path), input_filename, "-vv"]
    with pytest.warns((UserWarning, DeprecationWarning)):
        main(args)


@pytest.mark.unit
def test_grid_resolution_warning():
    """Test that a warning is thrown if grid resolution is too low."""
    eq = Equilibrium(L=3, M=3, N=3)
    eqN = eq.copy()
    eqN.change_resolution(N=1, N_grid=0)
    with pytest.warns(Warning):
        eqN.solve(ftol=1e-2, maxiter=2)
    eqM = eq.copy()
    eqM.change_resolution(M=eq.M, M_grid=eq.M - 1)
    with pytest.warns(Warning):
        eqM.solve(ftol=1e-2, maxiter=2)
    eqL = eq.copy()
    eqL.change_resolution(L=eq.L, L_grid=eq.L - 1)
    with pytest.warns(Warning):
        eqL.solve(ftol=1e-2, maxiter=2)


@pytest.mark.unit
def test_eq_change_grid_resolution():
    """Test changing equilibrium grid resolution."""
    eq = Equilibrium(L=2, M=2, N=2)
    eq.change_resolution(L_grid=10, M_grid=10, N_grid=10)
    assert eq.L_grid == 10
    assert eq.M_grid == 10
    assert eq.N_grid == 10


@pytest.mark.unit
def test_eq_change_symmetry():
    """Test changing stellarator symmetry."""
    eq = Equilibrium(L=2, M=2, N=2, NFP=2, sym=False)
    idx_sin = np.nonzero(
        sign(eq.R_basis.modes[:, 1]) * sign(eq.R_basis.modes[:, 2]) < 0
    )[0]
    idx_cos = np.nonzero(
        sign(eq.R_basis.modes[:, 1]) * sign(eq.R_basis.modes[:, 2]) > 0
    )[0]
    sin_modes = eq.R_basis.modes[idx_sin, :]
    cos_modes = eq.R_basis.modes[idx_cos, :]

    # stellarator symmetric
    eq.change_resolution(sym=True)
    assert eq.sym
    assert eq.R_basis.sym == "cos"
    assert not np.any(
        [np.any(np.all(i == eq.R_basis.modes, axis=-1)) for i in sin_modes]
    )
    assert eq.Z_basis.sym == "sin"
    assert not np.any(
        [np.any(np.all(i == eq.Z_basis.modes, axis=-1)) for i in cos_modes]
    )
    assert eq.L_basis.sym == "sin"
    assert not np.any(
        [np.any(np.all(i == eq.L_basis.modes, axis=-1)) for i in cos_modes]
    )
    assert eq.surface.sym
    assert eq.surface.R_basis.sym == "cos"
    assert eq.surface.Z_basis.sym == "sin"

    # undo symmetry
    eq.change_resolution(sym=False)
    assert not eq.sym
    assert not eq.R_basis.sym
    assert np.all([np.any(np.all(i == eq.R_basis.modes, axis=-1)) for i in sin_modes])
    assert not eq.Z_basis.sym
    assert np.all([np.any(np.all(i == eq.Z_basis.modes, axis=-1)) for i in cos_modes])
    assert not eq.L_basis.sym
    assert np.all([np.any(np.all(i == eq.L_basis.modes, axis=-1)) for i in cos_modes])
    assert not eq.surface.sym
    assert not eq.surface.R_basis.sym
    assert not eq.surface.Z_basis.sym


@pytest.mark.unit
def test_resolution():
    """Test changing equilibrium spectral resolution."""
    eq1 = Equilibrium(L=5, M=6, N=7, L_grid=8, M_grid=9, N_grid=10)
    eq2 = Equilibrium()

    assert eq1.resolution != eq2.resolution
    eq2.change_resolution(**eq1.resolution)
    assert eq1.resolution == eq2.resolution

    eq1.L = 2
    eq1.M = 3
    eq1.N = 4
    eq1.NFP = 5
    assert eq1.R_basis.L == 2
    assert eq1.R_basis.M == 3
    assert eq1.R_basis.N == 4
    assert eq1.R_basis.NFP == 5


@pytest.mark.unit
def test_equilibrium_from_near_axis():
    """Test loading a solution from pyQSC/pyQIC."""
    qsc_path = "./tests/inputs/qsc_r2section5.5.pkl"
    file = open(qsc_path, "rb")
    na = pickle.load(file)
    file.close()

    r = 1e-2
    eq = Equilibrium.from_near_axis(na, r=r, M=8, N=8)
    grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
    data = eq.compute("|B|", grid=grid)

    assert eq.is_nested()
    assert eq.NFP == na.nfp
    np.testing.assert_allclose(eq.Ra_n[:2], na.rc, atol=1e-10)
    np.testing.assert_allclose(eq.Za_n[-2:], na.zs, atol=1e-10)
    np.testing.assert_allclose(data["|B|"][0], na.B_mag(r, 0, 0), rtol=2e-2)


@pytest.mark.unit
def test_poincare_solve_not_implemented():
    """Test that solving with fixed poincare section doesn't work yet."""
    inputs = {
        "L": 4,
        "M": 2,
        "N": 2,
        "NFP": 3,
        "sym": False,
        "spectral_indexing": "ansi",
        "axis": np.array([[0, 10, 0]]),
        "pressure": np.array([[0, 10], [2, 5]]),
        "iota": np.array([[0, 1], [2, 3]]),
        "surface": np.array([[0, 0, 0, 10, 0], [1, 1, 0, 1, 1]]),
    }

    eq = Equilibrium(**inputs)
    np.testing.assert_allclose(
        eq.Rb_lmn, [10.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    with pytest.raises(NotImplementedError):
        eq.solve()


@pytest.mark.unit
def test_equilibriafamily_constructor():
    """Test that correct errors are thrown when making EquilibriaFamily."""
    eq = Equilibrium()
    ir = InputReader(["./tests/inputs/DSHAPE"])
    eqf = EquilibriaFamily(eq, *ir.inputs)
    assert len(eqf) == 4

    with pytest.raises(TypeError):
        _ = EquilibriaFamily(4, 5, 6)


@pytest.mark.unit
def test_change_NFP():
    """Test that changing the eq NFP correctly changes everything."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        eq = desc.examples.get("HELIOTRON")
        eq.change_resolution(NFP=4)
        obj = get_equilibrium_objective(eq=eq)
        obj.build()
