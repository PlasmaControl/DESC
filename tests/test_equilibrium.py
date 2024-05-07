"""Tests for Equilibrium class."""

import os
import pickle
import warnings

import numpy as np
import pytest

from desc.__main__ import main
from desc.backend import sign
from desc.compute.utils import cross, dot
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.examples import get
from desc.grid import Grid, LinearGrid
from desc.io import InputReader
from desc.objectives import ForceBalance, ObjectiveFunction, get_equilibrium_objective
from desc.profiles import PowerSeriesProfile

from .utils import area_difference, compute_coords


@pytest.mark.unit
def test_compute_theta_coords():
    """Test root finding for theta(theta*,lambda(theta))."""
    eq = get("DSHAPE_CURRENT")
    eq.change_resolution(3, 3, 0, 6, 6, 0)
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


@pytest.mark.unit
def test_map_coordinates():
    """Test root finding for (rho,theta,zeta) for common use cases."""
    # finding coordinates along a single field line
    eq = get("NCSX")
    eq.change_resolution(3, 3, 3, 6, 6, 6)
    n = 100
    coords = np.array([np.ones(n), np.zeros(n), np.linspace(0, 10 * np.pi, n)]).T
    out = eq.map_coordinates(
        coords,
        ["rho", "alpha", "zeta"],
        ["rho", "theta", "zeta"],
        period=(np.inf, 2 * np.pi, 10 * np.pi),
    )
    assert not np.any(np.isnan(out))

    eq = get("DSHAPE")

    inbasis = ["R", "phi", "Z"]
    outbasis = ["rho", "theta_PEST", "zeta"]

    rho = np.linspace(0.01, 0.99, 20)
    theta = np.linspace(0, np.pi, 20, endpoint=False)
    zeta = np.linspace(0, np.pi, 20, endpoint=False)

    grid = Grid(np.vstack([rho, theta, zeta]).T, sort=False)
    in_data = eq.compute(inbasis, grid=grid)
    in_coords = np.stack([in_data[k] for k in inbasis], axis=-1)
    out_data = eq.compute(outbasis, grid=grid)
    out_coords = np.stack([out_data[k] for k in outbasis], axis=-1)

    out = eq.map_coordinates(
        in_coords,
        inbasis,
        outbasis,
        period=(np.inf, 2 * np.pi, np.inf),
        maxiter=40,
    )
    np.testing.assert_allclose(out, out_coords, rtol=1e-4, atol=1e-4)


@pytest.mark.unit
def test_map_coordinates_derivative():
    """Test root finding for (rho,theta,zeta) from (R,phi,Z)."""
    eq = get("DSHAPE")
    eq.change_resolution(3, 3, 0, 6, 6, 0)
    inbasis = ["alpha", "phi", "rho"]
    outbasis = ["rho", "theta_PEST", "zeta"]

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
            outbasis,
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
        np.testing.assert_allclose(j1, j2)

    rho = np.linspace(0.01, 0.99, 200)
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi, 200, endpoint=False)

    nodes = np.vstack([rho, theta, zeta]).T
    coords = eq.compute("lambda", grid=Grid(nodes, sort=False))
    flux_coords = nodes.copy()
    flux_coords[:, 1] += coords["lambda"]

    @jax.jit
    def bar(L_lmn):
        geom_coords = eq.compute_theta_coords(flux_coords, L_lmn)
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
    eq.change_resolution(6, 6, 0, 12, 12, 0)
    Rr1, Zr1, Rv1, Zv1 = compute_coords(eq)
    Rr2, Zr2, Rv2, Zv2 = compute_coords(eq.to_sfl())
    rho_err, theta_err = area_difference(Rr1, Rr2, Zr1, Zr2, Rv1, Rv2, Zv1, Zv2)

    np.testing.assert_allclose(rho_err, 0, atol=1e-2)
    np.testing.assert_allclose(theta_err, 0, atol=2e-4)


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
    eq.change_resolution(4, 4, 4, 4, 4, 4)

    f_obj = ForceBalance(eq=eq)
    obj = ObjectiveFunction(f_obj, use_jit=False)
    eq.solve(maxiter=1, objective=obj)


@pytest.mark.unit
def test_shifted_circle_geometry():
    """
    In this test, we calculate a low-beta shifted circle equilibrium with DESC.

    We then compare the various geometric coefficients with their respective analytical
    expressions. These expression are available in Edmund Highcock's thesis on arxiv
    https://arxiv.org/pdf/1207.4419.pdf  (Table 3.5)
    """
    eq = Equilibrium.load(".//tests//inputs//low-beta-shifted-circle.h5")

    eq_keys = ["iota", "iota_r", "a", "rho", "psi"]

    psi = 0.25  # rho^2 (or normalized psi)
    alpha = 0

    eq_keys = ["iota", "iota_r", "a", "rho", "psi"]

    data_eq = eq.compute(eq_keys)

    iotas = np.interp(np.sqrt(psi), data_eq["rho"], data_eq["iota"])
    shears = np.interp(np.sqrt(psi), data_eq["rho"], data_eq["iota_r"])

    N = int((2 * eq.M_grid) * 4 + 1)

    zeta = np.linspace(-1.0 * np.pi / iotas, 1.0 * np.pi / iotas, N)
    theta_PEST = alpha * np.ones(N, dtype=int) + iotas * zeta

    coords1 = np.zeros((N, 3))
    coords1[:, 0] = np.sqrt(psi) * np.ones(N, dtype=int)
    coords1[:, 1] = theta_PEST
    coords1[:, 2] = zeta

    # Creating a grid along a field line
    c1 = eq.compute_theta_coords(coords1)
    grid = Grid(c1, sort=False)

    data_keys = [
        "kappa",
        "|grad(psi)|^2",
        "grad(|B|)",
        "grad(alpha)",
        "grad(psi)",
        "B",
        "grad(|B|)",
        "iota",
        "|B|",
        "B^zeta",
        "cvdrift0",
        "cvdrift",
        "gbdrift",
    ]

    data = eq.compute(data_keys, grid=grid, override_grid=False)

    psib = data_eq["psi"][-1]

    # signs
    sign_psi = psib / np.abs(psib)
    sign_iota = iotas / np.abs(iotas)

    # normalizations
    Lref = data_eq["a"]
    Bref = 2 * np.abs(psib) / Lref**2

    modB = data["|B|"]
    bmag = modB / Bref

    x = Lref * np.sqrt(psi)
    s_hat = -x / iotas * shears / Lref

    grad_psi = data["grad(psi)"]
    grad_alpha = data["grad(alpha)"]

    iota = data["iota"]

    gradpar = Lref * data["B^zeta"] / modB

    gds21 = -sign_iota * np.array(dot(grad_psi, grad_alpha)) * s_hat / Bref

    gbdrift = np.array(dot(cross(data["B"], data["grad(|B|)"]), grad_alpha))
    gbdrift *= -sign_psi * 2 * Bref * Lref**2 / modB**3 * np.sqrt(psi)

    cvdrift = (
        -sign_psi
        * 2
        * Bref
        * Lref**2
        * np.sqrt(psi)
        * dot(cross(data["B"], data["kappa"]), grad_alpha)
        / modB**2
    )

    cvdrift0 = np.array(dot(cross(data["B"], data["grad(|B|)"]), grad_psi))
    cvdrift0 *= sign_iota * sign_psi * s_hat * 2 / modB**3 / np.sqrt(psi)

    ## Comparing coefficient calculation here with coefficients from compute/_mtric
    cvdrift_2 = -2 * sign_psi * Bref * Lref**2 * np.sqrt(psi) * data["cvdrift"]
    gbdrift_2 = -2 * sign_psi * Bref * Lref**2 * np.sqrt(psi) * data["gbdrift"]

    # The error here should be of the same order as the max force error
    np.testing.assert_allclose(gbdrift, gbdrift_2, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(cvdrift, cvdrift_2, atol=8e-4, rtol=9e-5)

    a0_over_R0 = Lref * np.sqrt(psi)

    # For the rest of the expressions, the error ~ a0_over_R0
    fudge_factor1 = -3.8
    cvdrift0_an = fudge_factor1 * a0_over_R0 * s_hat * np.sin(theta_PEST)
    np.testing.assert_allclose(cvdrift0, cvdrift0_an, atol=5e-3, rtol=5e-3)

    bmag_an = np.mean(bmag) * (1 - a0_over_R0 * np.cos(theta_PEST))
    np.testing.assert_allclose(bmag, bmag_an, atol=5e-3, rtol=5e-3)

    gradpar_an = 2 * Lref * iota * (1 - a0_over_R0 * np.cos(theta_PEST))
    np.testing.assert_allclose(gradpar, gradpar_an, atol=9e-3, rtol=5e-3)

    dPdrho = np.mean(-0.5 * (cvdrift - gbdrift) * modB**2)
    alpha_MHD = -dPdrho * 1 / iota**2 * 0.5

    gds21_an = (
        -1 * s_hat * (s_hat * theta_PEST - alpha_MHD / bmag**4 * np.sin(theta_PEST))
    )
    np.testing.assert_allclose(gds21, gds21_an, atol=1.7e-2, rtol=5e-4)

    fudge_factor2 = 0.19
    gbdrift_an = fudge_factor2 * (
        -1 * s_hat + (np.cos(theta_PEST) - 1.0 * gds21 / s_hat * np.sin(theta_PEST))
    )

    fudge_factor3 = 0.07
    cvdrift_an = gbdrift_an + fudge_factor3 * alpha_MHD / bmag**2

    # Comparing coefficients with their analytical expressions
    np.testing.assert_allclose(gbdrift, gbdrift_an, atol=1.5e-2, rtol=5e-3)
    np.testing.assert_allclose(cvdrift, cvdrift_an, atol=9e-3, rtol=5e-3)
