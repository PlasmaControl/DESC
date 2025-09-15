"""Tests for Equilibrium class."""

import os
import warnings

import numpy as np
import pytest
from qic import Qic

from desc.__main__ import main
from desc.backend import sign
from desc.compute.utils import get_transforms
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.equilibrium.coords import _map_clebsch_coordinates
from desc.examples import get
from desc.grid import Grid, LinearGrid
from desc.io import InputReader, load
from desc.objectives import ForceBalance, ObjectiveFunction, get_equilibrium_objective
from desc.profiles import PowerSeriesProfile

from .utils import area_difference, compute_coords


@pytest.mark.unit
def test_map_PEST_coordinates():
    """Test root finding for theta(theta_PEST,lambda(theta))."""
    eq = get("DSHAPE_CURRENT")
    with pytest.warns(UserWarning, match="Reducing radial"):
        eq.change_resolution(3, 3, 0, 6, 6, 0)
    rho = np.linspace(0.01, 0.99, 200)
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi, 200, endpoint=False)

    nodes = np.vstack([rho, theta, zeta]).T
    coords = eq.compute("lambda", grid=Grid(nodes, sort=False))
    flux_coords = nodes.copy()
    flux_coords[:, 1] += coords["lambda"]

    geom_coords = eq.map_coordinates(flux_coords, inbasis=("rho", "theta_PEST", "zeta"))
    geom_coords = np.array(geom_coords)

    # catch difference between 0 and 2*pi
    if geom_coords[0, 1] > np.pi:  # theta[0] = 0
        geom_coords[0, 1] = geom_coords[0, 1] - 2 * np.pi

    np.testing.assert_allclose(nodes, geom_coords, rtol=1e-5, atol=1e-5)


@pytest.mark.unit
def test_map_coordinates():
    """Test root finding for (rho,theta,zeta) for common use cases."""
    # finding coordinates along a single field line
    eq = get("NCSX")
    with pytest.warns(UserWarning, match="Reducing radial"):
        eq.change_resolution(3, 3, 3, 6, 6, 6)
    n = 100
    coords = np.array([np.ones(n), np.zeros(n), np.linspace(0, 10 * np.pi, n)]).T
    out = eq.map_coordinates(coords, inbasis=["rho", "alpha", "zeta"])
    assert np.isfinite(out).all()

    eq = get("DSHAPE")

    inbasis = ["R", "phi", "Z"]
    outbasis = ["rho", "theta_PEST", "zeta"]

    rho = np.linspace(0.01, 0.99, 20)
    theta = np.linspace(0, np.pi, 20, endpoint=False)
    zeta = np.linspace(0, np.pi, 20, endpoint=False)

    grid = Grid(np.vstack([rho, theta, zeta]).T, sort=False)
    in_data = eq.compute(inbasis, grid=grid)
    in_coords = np.column_stack([in_data[k] for k in inbasis])
    out_data = eq.compute(outbasis, grid=grid)
    out_coords = np.column_stack([out_data[k] for k in outbasis])

    out = eq.map_coordinates(
        in_coords,
        inbasis,
        outbasis,
        period=(np.inf, 2 * np.pi, np.inf),
        maxiter=40,
    )
    np.testing.assert_allclose(out, out_coords, rtol=1e-4, atol=1e-4)


@pytest.mark.unit
def test_map_clebsch_coordinates():
    """Test root finding for (rho,alpha,zeta)."""
    eq = get("NCSX")
    assert eq.NFP > 1
    rho = np.linspace(0.5, 1, 2)
    alpha = np.linspace(0, 2 * np.pi, 3)
    zeta = np.array([2 * np.pi, 2 * np.pi - 0.1, np.e, 0.24, 0.2])
    iota = eq.compute("iota", grid=LinearGrid(rho=rho))["iota"]

    grid = Grid.create_meshgrid([rho, alpha, zeta], coordinates="raz")
    out = eq.map_coordinates(
        grid.nodes, inbasis=("rho", "alpha", "zeta"), iota=grid.expand(iota)
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Unequal number of field periods")
        lmbda = get_transforms(
            "lambda", eq, LinearGrid(rho=rho, M=eq.L_basis.M, zeta=zeta)
        )["L"]
    assert lmbda.basis.NFP == eq.NFP
    np.testing.assert_allclose(
        lmbda.grid.meshgrid_reshape(lmbda.grid.nodes[:, 2], "rtz")[0, 0, ::-1], zeta
    )
    np.testing.assert_allclose(
        _map_clebsch_coordinates(iota, alpha, zeta[::-1], eq.L_lmn, lmbda)[..., ::-1],
        grid.meshgrid_reshape(out[:, 1], "raz"),
    )


@pytest.mark.unit
def test_map_coordinates_derivative():
    """Test root finding for (rho,theta,zeta) from (R,phi,Z)."""
    eq = get("DSHAPE")
    with pytest.warns(UserWarning, match="Reducing radial"):
        eq.change_resolution(3, 3, 0, 6, 6, 0)
    inbasis = ["rho", "alpha", "phi"]

    rho = np.linspace(0.01, 0.99, 20)
    theta = np.linspace(0, np.pi, 20, endpoint=False)
    zeta = np.linspace(0, np.pi, 20, endpoint=False)

    grid = Grid(np.vstack([rho, theta, zeta]).T, sort=False)
    in_data = eq.compute(inbasis, grid=grid)
    in_coords = np.stack([in_data[k] for k in inbasis], axis=-1)

    import jax

    @jax.jit
    def foo(params, in_coords):
        out = eq.map_coordinates(
            in_coords,
            inbasis,
            ("rho", "theta_PEST", "zeta"),
            np.array([rho, theta, zeta]).T,
            params,
            period=(2 * np.pi, 2 * np.pi, np.inf),
            maxiter=40,
        )
        return out

    J1 = jax.jit(jax.jacfwd(foo))(eq.params_dict, in_coords)
    J2 = jax.jit(jax.jacrev(foo))(eq.params_dict, in_coords)
    for j1, j2 in zip(J1.values(), J2.values()):
        assert ~np.any(np.isnan(j1))
        assert ~np.any(np.isnan(j2))
        np.testing.assert_allclose(j1, j2, atol=1e-12)

    rho = np.linspace(0.01, 0.99, 200)
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi, 200, endpoint=False)

    nodes = np.vstack([rho, theta, zeta]).T
    coords = eq.compute("lambda", grid=Grid(nodes, sort=False))
    flux_coords = nodes.copy()
    flux_coords[:, 1] += coords["lambda"]

    # this will call _map_PEST_coordinates inside map_coordinates
    @jax.jit
    def bar(L_lmn):
        params = {"L_lmn": L_lmn}
        geom_coords = eq.map_coordinates(
            flux_coords,
            inbasis=("rho", "theta_PEST", "zeta"),
            params=params,
        )
        return geom_coords

    J1 = jax.jit(jax.jacfwd(bar))(eq.params_dict["L_lmn"])
    J2 = jax.jit(jax.jacrev(bar))(eq.params_dict["L_lmn"])

    assert ~np.any(np.isnan(J1))
    assert ~np.any(np.isnan(J2))
    np.testing.assert_allclose(J1, J2)


@pytest.mark.slow
@pytest.mark.unit
def test_to_sfl():
    """Test converting an equilibrium to straight field line coordinates."""
    eq = get("DSHAPE_CURRENT")
    Rr1, Zr1, Rv1, Zv1 = compute_coords(eq)
    Rr2, Zr2, Rv2, Zv2 = compute_coords(eq.to_sfl())
    rho_err, theta_err = area_difference(Rr1, Rr2, Zr1, Zr2, Rv1, Rv2, Zv1, Zv2)

    np.testing.assert_allclose(rho_err, 0, atol=1e-4)
    np.testing.assert_allclose(theta_err, 0, atol=1e-6)


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
    # if we first raise warnings to errors then check for error we can avoid
    # actually running the full solve
    with pytest.raises(UserWarning):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            eqN.solve(ftol=1e-2, maxiter=2)
    eqM = eq.copy()
    eqM.change_resolution(M=eq.M, M_grid=eq.M - 1)
    with pytest.raises(UserWarning):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            eqM.solve(ftol=1e-2, maxiter=2)
    eqL = eq.copy()
    eqL.change_resolution(L=eq.L, L_grid=eq.L - 1)
    with pytest.raises(UserWarning):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            eqL.solve(ftol=1e-2, maxiter=2)


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
    assert eq.axis.sym
    assert eq.axis.R_basis.sym == "cos"
    assert eq.axis.Z_basis.sym == "sin"

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
    assert eq.axis.sym is False
    assert eq.axis.R_basis.sym is False
    assert eq.axis.Z_basis.sym is False


@pytest.mark.unit
def test_resolution():
    """Test changing equilibrium spectral resolution."""
    eq1 = Equilibrium(L=5, M=6, N=7, L_grid=8, M_grid=9, N_grid=10)
    eq2 = Equilibrium()

    assert eq1.resolution != eq2.resolution
    eq2.change_resolution(**eq1.resolution)
    assert eq1.resolution == eq2.resolution


@pytest.mark.unit
def test_equilibrium_from_near_axis():
    """Test loading a solution from pyQSC/pyQIC."""
    na = Qic.from_paper("r2 section 5.5", rs=[0, 1e-5], zc=[0, 1e-5])

    r = 1e-2
    eq = Equilibrium.from_near_axis(na, r=r, M=8, N=8)
    grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
    data = eq.compute("|B|", grid=grid)

    # get the sin/cos modes
    eq_rc = eq.Ra_n[
        np.where(
            np.logical_and(
                eq.axis.R_basis.modes[:, 2] >= 0,
                eq.axis.R_basis.modes[:, 2] < na.nfourier,
            )
        )
    ]
    eq_zc = eq.Za_n[
        np.where(
            np.logical_and(
                eq.axis.Z_basis.modes[:, 2] >= 0,
                eq.axis.Z_basis.modes[:, 2] < na.nfourier,
            )
        )
    ]
    eq_rs = np.flipud(
        eq.Ra_n[
            np.where(
                np.logical_and(
                    eq.axis.R_basis.modes[:, 2] < 0,
                    eq.axis.R_basis.modes[:, 2] > -na.nfourier,
                )
            )
        ]
    )
    eq_zs = np.flipud(
        eq.Za_n[
            np.where(
                np.logical_and(
                    eq.axis.Z_basis.modes[:, 2] < 0,
                    eq.axis.Z_basis.modes[:, 2] > -na.nfourier,
                )
            )
        ]
    )

    assert eq.is_nested()
    assert eq.NFP == na.nfp
    np.testing.assert_allclose(eq_rc, na.rc, atol=1e-10)
    # na.zs[0] is always 0, which DESC doesn't include
    np.testing.assert_allclose(eq_zs, na.zs[1:], atol=1e-10)
    np.testing.assert_allclose(eq_rs, na.rs[1:], atol=1e-10)
    np.testing.assert_allclose(eq_zc, na.zc, atol=1e-10)
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
        "surface": np.array(
            [
                [0, 0, 0, 10, 0],
                [1, 1, 0, 1, 0.1],
                [1, -1, 0, 0.2, -1],
            ]
        ),
    }

    eq = Equilibrium(**inputs)
    assert eq.bdry_mode == "poincare"
    np.testing.assert_allclose(
        eq.Rb_lmn, [10.0, 0.2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        eq.Zb_lmn, [0.0, -1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
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
        eq = get("HELIOTRON")
        with pytest.warns(UserWarning, match="Reducing radial"):
            eq.change_resolution(3, 3, 1, 6, 6, 2)
        eq.change_resolution(NFP=4)
        obj = get_equilibrium_objective(eq=eq)
        obj.build()


@pytest.mark.unit
def test_error_when_ndarray_or_integer_passed():
    """Test that errors raise correctly when a non-Grid object is passed."""
    eq = get("DSHAPE")
    with pytest.raises(TypeError):
        eq.compute("R", grid=1)
    with pytest.raises(TypeError):
        eq.compute("R", grid=np.linspace(0, 1, 10))


@pytest.mark.unit
def test_equilibrium_unused_kwargs():
    """Test that invalid kwargs raise an error, for gh issue #850."""
    pres = PowerSeriesProfile()
    curr = PowerSeriesProfile()
    with pytest.raises(TypeError):
        _ = Equilibrium(pres=pres, curr=curr)
    _ = Equilibrium(pressure=pres, current=curr)


@pytest.mark.unit
@pytest.mark.solve
def test_backward_compatible_load_and_resolve():
    """Test backwards compatibility of load and re-solve."""
    with pytest.warns(RuntimeWarning):
        eq = EquilibriaFamily.load(load_from=".//tests//inputs//NCSX_older.h5")[-1]

    # reducing resolution since we only want to test eq.solve
    with pytest.warns(UserWarning, match="Reducing radial"):
        eq.change_resolution(4, 4, 4, 4, 4, 4)

    f_obj = ForceBalance(eq=eq)
    obj = ObjectiveFunction(f_obj, use_jit=False)
    eq.solve(maxiter=1, objective=obj)


@pytest.mark.unit
def test_assigning_profile_iota_current():
    """Test assigning current to iota-constrained eq and vice-versa."""
    eq = get("HELIOTRON")  # iota-constrained
    with pytest.warns(UserWarning, match="existing rotational"):
        eq.current = PowerSeriesProfile()
    assert eq.iota is None
    with pytest.warns(UserWarning, match="existing toroidal"):
        eq.iota = PowerSeriesProfile()
    assert eq.current is None


@pytest.mark.unit
def test_eq_optimize_default_constraints_warning(DummyStellarator):
    """Tests default constraints warning for eq.optimize."""
    eq = load(load_from=str(DummyStellarator["output_path"]), file_format="hdf5")
    eq.change_resolution(M=1, N=0, M_grid=2, N_grid=0)
    with pytest.warns(UserWarning, match="no equil"):
        eq.optimize(
            ObjectiveFunction(ForceBalance(eq)),
            constraints=(),
            optimizer="lsq-exact",
            maxiter=0,
        )
