"""Tests for Mercier stability functions."""

import numpy as np
import pytest
from netCDF4 import Dataset
from scipy.interpolate import CubicSpline as cubspl
from scipy.interpolate import interp1d

import desc.examples
import desc.io
from desc.backend import jnp
from desc.compute.utils import cross, dot
from desc.equilibrium import Equilibrium
from desc.equilibrium.coords import compute_theta_coords
from desc.grid import Grid, LinearGrid
from desc.objectives import MagneticWell, MercierStability

DEFAULT_RANGE = (0.05, 1)
DEFAULT_RTOL = 1e-2
DEFAULT_ATOL = 1e-6
MAX_SIGN_DIFF = 5


def assert_all_close(
    y1, y2, rho, rho_range=DEFAULT_RANGE, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL
):
    """Test that the values of y1 and y2, over a given range are close enough.

    Parameters
    ----------
    y1 : ndarray
        values to compare
    y2 : ndarray
        values to compare
    rho : ndarray
        rho values
    rho_range : (float, float)
        the range of rho values to compare
    rtol : float
        relative tolerance
    atol : float
        absolute tolerance

    """
    minimum, maximum = rho_range
    interval = (minimum < rho) & (rho < maximum)
    np.testing.assert_allclose(y1[interval], y2[interval], rtol=rtol, atol=atol)


def get_vmec_data(path, quantity):
    """Get data from a VMEC wout.nc file.

    Parameters
    ----------
    path : str
        Path to VMEC file.
    quantity: str
        Name of the quantity to return.

    Returns
    -------
    rho : ndarray
        Radial coordinate.
    q : ndarray
        Variable from VMEC output.

    """
    f = Dataset(path)
    rho = np.sqrt(f.variables["phi"] / np.array(f.variables["phi"])[-1])
    q = np.array(f.variables[quantity])
    f.close()
    return rho, q


@pytest.mark.unit
def test_mercier_vacuum():
    """Test that the Mercier stability criteria are 0 without pressure."""
    eq = Equilibrium()
    data = eq.compute(["D_shear", "D_current", "D_well", "D_geodesic", "D_Mercier"])
    np.testing.assert_allclose(data["D_shear"], 0)
    np.testing.assert_allclose(data["D_current"], 0)
    np.testing.assert_allclose(data["D_well"], 0)
    np.testing.assert_allclose(data["D_geodesic"], 0)
    np.testing.assert_allclose(data["D_Mercier"], 0)


@pytest.mark.unit
def test_compute_d_shear():
    """Test that D_shear has a stabilizing effect and matches VMEC."""

    def test(eq, vmec, rho_range=(0, 1), rtol=1e-12, atol=0.0):
        rho, d_shear_vmec = get_vmec_data(vmec, "DShear")
        grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=rho)
        d_shear = grid.compress(eq.compute("D_shear", grid=grid)["D_shear"])

        assert np.all(
            d_shear[bool(grid.axis.size) :] >= 0
        ), "D_shear should always have a stabilizing effect."
        assert_all_close(d_shear, d_shear_vmec, rho, rho_range, rtol, atol)

    test(
        desc.examples.get("DSHAPE_CURRENT"),
        ".//tests//inputs//wout_DSHAPE.nc",
        (0.3, 0.9),
        atol=0.01,
        rtol=0.1,
    )
    test(desc.examples.get("HELIOTRON"), ".//tests//inputs//wout_HELIOTRON.nc")


@pytest.mark.unit
def test_compute_d_current():
    """Test calculation of D_current stability criterion against VMEC."""

    def test(eq, vmec, rho_range=DEFAULT_RANGE, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
        rho, d_current_vmec = get_vmec_data(vmec, "DCurr")
        grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=rho)
        d_current = grid.compress(eq.compute("D_current", grid=grid)["D_current"])

        assert (
            np.nonzero(np.sign(d_current) != np.sign(d_current_vmec))[0].size
            <= MAX_SIGN_DIFF
        )
        assert_all_close(d_current, d_current_vmec, rho, rho_range, rtol, atol)

    test(
        desc.examples.get("DSHAPE_CURRENT"),
        ".//tests//inputs//wout_DSHAPE.nc",
        (0.3, 0.9),
        rtol=1e-1,
        atol=1e-2,
    )
    test(
        desc.examples.get("HELIOTRON"),
        ".//tests//inputs//wout_HELIOTRON.nc",
        (0.25, 0.85),
        rtol=1e-1,
    )


@pytest.mark.unit
def test_compute_d_well():
    """Test calculation of D_well stability criterion against VMEC."""

    def test(eq, vmec, rho_range=DEFAULT_RANGE, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
        rho, d_well_vmec = get_vmec_data(vmec, "DWell")
        grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=rho)
        d_well = grid.compress(eq.compute("D_well", grid=grid)["D_well"])

        assert (
            np.nonzero(np.sign(d_well) != np.sign(d_well_vmec))[0].size <= MAX_SIGN_DIFF
        )
        assert_all_close(d_well, d_well_vmec, rho, rho_range, rtol, atol)

    test(
        desc.examples.get("DSHAPE_CURRENT"),
        ".//tests//inputs//wout_DSHAPE.nc",
        (0.3, 0.9),
        rtol=1e-1,
    )
    test(
        desc.examples.get("HELIOTRON"),
        ".//tests//inputs//wout_HELIOTRON.nc",
        (0.01, 0.45),
        rtol=1.75e-1,
    )
    test(
        desc.examples.get("HELIOTRON"),
        ".//tests//inputs//wout_HELIOTRON.nc",
        (0.45, 0.6),
        atol=7.2e-1,
    )
    test(
        desc.examples.get("HELIOTRON"),
        ".//tests//inputs//wout_HELIOTRON.nc",
        (0.6, 0.99),
        rtol=2e-2,
    )


@pytest.mark.unit
def test_compute_d_geodesic():
    """Test that D_geodesic has a destabilizing effect and matches VMEC."""

    def test(eq, vmec, rho_range=DEFAULT_RANGE, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
        rho, d_geodesic_vmec = get_vmec_data(vmec, "DGeod")
        grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=rho)
        d_geodesic = grid.compress(eq.compute("D_geodesic", grid=grid)["D_geodesic"])

        assert np.all(
            d_geodesic[bool(grid.axis.size) :] <= 0
        ), "D_geodesic should always have a destabilizing effect."
        assert_all_close(d_geodesic, d_geodesic_vmec, rho, rho_range, rtol, atol)

    test(
        desc.examples.get("DSHAPE_CURRENT"),
        ".//tests//inputs//wout_DSHAPE.nc",
        (0.3, 0.9),
        rtol=1e-1,
    )
    test(
        desc.examples.get("HELIOTRON"),
        ".//tests//inputs//wout_HELIOTRON.nc",
        (0.15, 0.825),
        rtol=1.2e-1,
    )
    test(
        desc.examples.get("HELIOTRON"),
        ".//tests//inputs//wout_HELIOTRON.nc",
        (0.85, 0.95),
        atol=1.2e-1,
    )


@pytest.mark.unit
def test_compute_d_mercier():
    """Test calculation of D_Mercier stability criterion against VMEC."""

    def test(eq, vmec, rho_range=DEFAULT_RANGE, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
        rho, d_mercier_vmec = get_vmec_data(vmec, "DMerc")
        grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=rho)
        d_mercier = grid.compress(eq.compute("D_Mercier", grid=grid)["D_Mercier"])

        assert (
            np.nonzero(np.sign(d_mercier) != np.sign(d_mercier_vmec))[0].size
            <= MAX_SIGN_DIFF
        )
        assert_all_close(d_mercier, d_mercier_vmec, rho, rho_range, rtol, atol)

    test(
        desc.examples.get("DSHAPE_CURRENT"),
        ".//tests//inputs//wout_DSHAPE.nc",
        (0.3, 0.9),
        rtol=1e-1,
        atol=1e-2,
    )
    test(
        desc.examples.get("HELIOTRON"),
        ".//tests//inputs//wout_HELIOTRON.nc",
        (0.1, 0.325),
        rtol=1.3e-1,
    )
    test(
        desc.examples.get("HELIOTRON"),
        ".//tests//inputs//wout_HELIOTRON.nc",
        (0.325, 0.95),
        rtol=5e-2,
    )


@pytest.mark.unit
def test_compute_magnetic_well():
    """Test that D_well and magnetic_well match signs under finite pressure."""

    def test(eq, rho=np.linspace(0, 1, 128)):
        grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=rho)
        d_well = grid.compress(eq.compute("D_well", grid=grid)["D_well"])
        magnetic_well = grid.compress(
            eq.compute("magnetic well", grid=grid)["magnetic well"]
        )
        assert (
            np.nonzero(np.sign(d_well) != np.sign(magnetic_well))[0].size
            <= MAX_SIGN_DIFF
        )

    test(desc.examples.get("DSHAPE_CURRENT"))
    test(desc.examples.get("HELIOTRON"))


@pytest.mark.unit
def test_mercier_print(capsys):
    """Test that the Mercier stability criteria prints correctly."""
    eq = Equilibrium()
    grid = LinearGrid(L=10, M=10, N=5, axis=False)

    Dmerc = eq.compute("D_Mercier", grid=grid)["D_Mercier"]

    mercier_obj = MercierStability(eq=eq, grid=grid)
    mercier_obj.build()
    np.testing.assert_allclose(mercier_obj.compute(*mercier_obj.xs(eq)), 0)
    mercier_obj.print_value(*mercier_obj.xs(eq))
    out = capsys.readouterr()

    corr_out = str(
        "Precomputing transforms\n"
        + "Maximum "
        + mercier_obj._print_value_fmt.format(np.max(Dmerc))
        + mercier_obj._units
        + "\n"
        + "Minimum "
        + mercier_obj._print_value_fmt.format(np.min(Dmerc))
        + mercier_obj._units
        + "\n"
        + "Average "
        + mercier_obj._print_value_fmt.format(np.mean(Dmerc))
        + mercier_obj._units
        + "\n"
        + "Maximum "
        + mercier_obj._print_value_fmt.format(np.max(Dmerc / mercier_obj.normalization))
        + "(normalized)"
        + "\n"
        + "Minimum "
        + mercier_obj._print_value_fmt.format(np.min(Dmerc / mercier_obj.normalization))
        + "(normalized)"
        + "\n"
        + "Average "
        + mercier_obj._print_value_fmt.format(
            np.mean(Dmerc / mercier_obj.normalization)
        )
        + "(normalized)"
        + "\n"
    )
    assert out.out == corr_out


@pytest.mark.unit
def test_magwell_print(capsys):
    """Test that the magnetic well stability criteria prints correctly."""
    eq = desc.examples.get("HELIOTRON")
    grid = LinearGrid(L=12, M=12, N=6, NFP=eq.NFP, axis=False)
    obj = MagneticWell(eq=eq, grid=grid)
    obj.build()

    magwell = grid.compress(eq.compute("magnetic well", grid=grid)["magnetic well"])
    f = obj.compute(*obj.xs(eq))
    np.testing.assert_allclose(f, magwell)

    obj.print_value(*obj.xs(eq))
    out = capsys.readouterr()

    corr_out = str(
        "Precomputing transforms\n"
        + "Maximum "
        + obj._print_value_fmt.format(np.max(magwell))
        + obj._units
        + "\n"
        + "Minimum "
        + obj._print_value_fmt.format(np.min(magwell))
        + obj._units
        + "\n"
        + "Average "
        + obj._print_value_fmt.format(np.mean(magwell))
        + obj._units
        + "\n"
    )
    assert out.out == corr_out


@pytest.mark.unit
def test_ballooning_geometry(tmpdir_factory):
    """Test the geometry coefficients used for the adjoint-ballooning solver.

    The same coefficients are used for local gyrokinetic solvers which would
    be useful when we couple DESC with GX/GS2 etc.
    Observation: The larger the force error, the worse the tests behave. For
    example, HELIOTRON coefficients are hard to match
    """
    psi = 0.5  # Actually rho^2 (normalized)
    alpha = 0
    ntor = 2.0

    try:
        eq0 = desc.examples.get("W7-X")[-1]
    except TypeError:
        eq0 = desc.examples.get("W7-X")

    eq_list = [eq0]
    fac_list = [4]

    for eq, fac in zip(eq_list, fac_list):
        print(eq)
        eq_keys = ["iota", "iota_r", "a", "rho", "psi"]

        data_eq = eq.compute(eq_keys)

        fi = interp1d(data_eq["rho"], data_eq["iota"])
        fs = interp1d(data_eq["rho"], data_eq["iota_r"])

        iotas = fi(np.sqrt(psi))
        shears = fs(np.sqrt(psi))

        N = int((2 * eq.M_grid * eq.N_grid) * ntor * int(fac) + 1)
        coords1 = np.zeros((N, 3))
        coords1[:, 0] = np.sqrt(psi) * np.ones(N, dtype=int)
        coords1[:, 1] = alpha * np.ones(N, dtype=int) + iotas * np.linspace(
            -ntor * np.pi, ntor * np.pi, N
        )
        zeta = np.linspace(-ntor * np.pi, ntor * np.pi, N)
        coords1[:, 2] = zeta

        c1 = eq.compute_theta_coords(coords1)
        grid = Grid(c1, sort=False)

        data_keys = [
            "p_r",
            "psi_r",
            "sqrt(g)_PEST",
            "|grad(psi)|^2",
            "grad(|B|)",
            "grad(alpha)",
            "grad(psi)",
            "B",
            "grad(|B|)",
            "kappa",
            "iota",
            "lambda_t",
            "lambda_z",
            "lambda_tt",
            "lambda_zz",
            "lambda_tz",
            "g^aa",
            "g^ra",
            "g^rr",
            "g^aa_z",
            "g^aa_t",
            "g^aa_zz",
            "g^aa_tt",
            "g^aa_tz",
            "g^ra_z",
            "g^ra_t",
            "g^ra_zz",
            "g^ra_tt",
            "g^ra_tz",
            "g^rr_z",
            "g^rr_t",
            "g^rr_zz",
            "g^rr_tt",
            "g^rr_tz",
            "cvdrift",
            "cvdrift0",
            "|B|",
            "|B|_z",
            "|B|_t",
            "|B|_zz",
            "|B|_tt",
            "|B|_tz",
            "B^zeta",
            "B^zeta_t",
            "B^zeta_z",
            "B^zeta_tt",
            "B^zeta_zz",
            "B^zeta_tz",
        ]

        data = eq.compute(data_keys, grid=grid)

        psib = data_eq["psi"][-1]
        sign_psi = psib / np.abs(psib)
        sign_iota = iotas / np.abs(iotas)
        # normalizations
        Lref = data_eq["a"]
        Bref = 2 * np.abs(psib) / Lref**2

        modB = data["|B|"]
        x = Lref * np.sqrt(psi)
        shat = -x / iotas * shears / Lref

        psi_r = data["psi_r"]

        grad_psi = data["grad(psi)"]
        grad_psi_sq = data["|grad(psi)|^2"]
        grad_alpha = data["grad(alpha)"]

        iota = data["iota"]

        temp_fac1 = 1 / (1 + data["lambda_t"])
        temp_fac2 = (iota - data["lambda_z"]) * temp_fac1

        g_sup_rr = data["g^rr"]
        g_sup_rr_z0 = data["g^rr_z"] + data["g^rr_t"] * temp_fac2
        g_sup_rr_zz0 = (
            data["g^rr_zz"]
            + 2 * data["g^rr_tz"] * temp_fac2
            - data["g^rr_t"]
            * (
                data["lambda_zz"] * temp_fac1
                + 2 * temp_fac2 * data["lambda_tz"] * temp_fac1
                + temp_fac2**2 * temp_fac1 * data["lambda_tt"]
            )
            + data["g^rr_tt"] * temp_fac2**2
        )

        g_sup_ra = data["g^ra"]
        g_sup_ra_z0 = data["g^ra_z"] + data["g^ra_t"] * temp_fac2

        g_sup_aa = data["g^aa"]
        g_sup_aa_z0 = data["g^aa_z"] + data["g^aa_t"] * temp_fac2
        g_sup_aa_zz0 = (
            data["g^aa_zz"]
            + 2 * data["g^aa_tz"] * temp_fac2
            - data["g^aa_t"]
            * (
                data["lambda_zz"] * temp_fac1
                + 2 * temp_fac2 * data["lambda_tz"] * temp_fac1
                + temp_fac2**2 * temp_fac1 * data["lambda_tt"]
            )
            + data["g^aa_tt"] * temp_fac2**2
        )

        modB = data["|B|"]
        modB_z0 = data["|B|_z"] + data["|B|_t"] * temp_fac2
        modB_zz0 = (
            data["|B|_zz"]
            + 2 * data["|B|_tz"] * temp_fac2
            - data["|B|_t"]
            * (
                data["lambda_zz"] * temp_fac1
                + 2 * temp_fac2 * data["lambda_tz"] * temp_fac1
                + temp_fac2**2 * temp_fac1 * data["lambda_tt"]
            )
            + data["|B|_tt"] * temp_fac2**2
        )

        B_sup_zeta = data["B^zeta"]
        B_sup_zeta_z0 = data["B^zeta_z"] + temp_fac2 * data["B^zeta_t"]
        B_sup_zeta_zz0 = (
            data["B^zeta_zz"]
            + 2 * data["B^zeta_tz"] * temp_fac2
            - data["B^zeta_t"]
            * (
                data["lambda_zz"] * temp_fac1
                + 2 * temp_fac2 * data["lambda_tz"] * temp_fac1
                + temp_fac2**2 * temp_fac1 * data["lambda_tt"]
            )
            + data["B^zeta_tt"] * temp_fac2**2
        )

        gds2 = np.array(dot(grad_alpha, grad_alpha)) * Lref**2 * psi
        gds2_alt = g_sup_aa * Lref**2 * psi

        gds21 = -sign_iota * np.array(dot(grad_psi, grad_alpha)) * shat / Bref
        gds21_alt = -sign_iota * g_sup_ra * shat / Bref * (psi_r)

        gds22 = grad_psi_sq * (1 / psi) * (shat / (Lref * Bref)) ** 2
        gds22_alt = g_sup_rr * (psi_r) ** 2 * (1 / psi) * (shat / (Lref * Bref)) ** 2

        gbdrift = np.array(dot(cross(data["B"], data["grad(|B|)"]), grad_alpha))
        gbdrift *= -sign_psi * 2 * Bref * Lref**2 / modB**3 * np.sqrt(psi)
        gbdrift_alt = -sign_psi * data["gbdrift"] * 2 * Bref * Lref**2 * np.sqrt(psi)

        cvdrift = (
            -sign_psi
            * 2
            * Bref
            * Lref**2
            * np.sqrt(psi)
            * dot(cross(data["B"], data["kappa"]), grad_alpha)
            / modB**2
        )
        cvdrift_alt = -sign_psi * data["cvdrift"] * 2 * Bref * Lref**2 * np.sqrt(psi)

        np.testing.assert_allclose(gds2, gds2_alt)
        np.testing.assert_allclose(gds22, gds22_alt)
        np.testing.assert_allclose(gds21, gds21_alt)
        np.testing.assert_allclose(gbdrift, gbdrift_alt)
        np.testing.assert_allclose(cvdrift, cvdrift_alt, atol=1e-2)

        sqrt_g_PEST = data["sqrt(g)_PEST"]
        spl0 = cubspl(zeta, sqrt_g_PEST)
        np.testing.assert_allclose(sqrt_g_PEST, 1 / (B_sup_zeta / psi_r))
        np.testing.assert_allclose(
            spl0.derivative()(zeta),
            -psi_r * B_sup_zeta_z0 / (B_sup_zeta) ** 2,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            spl0.derivative(nu=2)(zeta),
            psi_r
            * (
                -B_sup_zeta_zz0 / (B_sup_zeta) ** 2
                + 2 * B_sup_zeta_z0**2 / (B_sup_zeta) ** 3
            ),
            atol=2e-1,
        )

        spl1 = cubspl(zeta, g_sup_aa)
        spl11 = cubspl(zeta, g_sup_aa_z0)
        np.testing.assert_allclose(g_sup_aa_z0, spl1.derivative()(zeta), atol=5e-2)
        np.testing.assert_allclose(g_sup_aa_zz0, spl1.derivative(nu=2)(zeta), atol=1e2)
        np.testing.assert_allclose(g_sup_aa_zz0, spl11.derivative()(zeta), atol=1e2)

        spl2 = cubspl(zeta, g_sup_ra)
        spl21 = cubspl(zeta, g_sup_ra_z0)
        np.testing.assert_allclose(g_sup_ra_z0, spl2.derivative(nu=1)(zeta), atol=5e-2)
        np.testing.assert_allclose(
            spl2.derivative(nu=2)(zeta), spl21.derivative(nu=1)(zeta), atol=1e2
        )

        spl3 = cubspl(zeta, g_sup_rr)
        spl31 = cubspl(zeta, g_sup_rr_z0)
        np.testing.assert_allclose(g_sup_rr_z0, spl3.derivative(nu=1)(zeta), atol=5e-2)
        np.testing.assert_allclose(g_sup_rr_zz0, spl3.derivative(nu=2)(zeta), atol=1e2)
        np.testing.assert_allclose(g_sup_rr_zz0, spl31.derivative(nu=1)(zeta), atol=1e2)

        spl4 = cubspl(zeta, modB)
        np.testing.assert_allclose(modB_z0, spl4.derivative(nu=1)(zeta), atol=1e-3)
        np.testing.assert_allclose(modB_zz0, spl4.derivative(nu=2)(zeta), atol=3e-1)


@pytest.mark.unit
def test_ballooning_stability_eval():
    """Cross-compare all the stability functions.

    We calculated the ideal ballooning growth rate and Newcomb metric for
    the HELIOTRON case at different radii.
    """
    try:
        eq = desc.examples.get("HELIOTRON")[-1]
    except TypeError:
        eq = desc.examples.get("HELIOTRON")

    # Flux surfaces on which to evaluate ballooning stability
    surfaces = [0.01, 0.5, 1.0]

    grid = LinearGrid(rho=jnp.array(surfaces), NFP=eq.NFP)
    eq_data_keys = ["iota"]

    data = eq.compute(eq_data_keys, grid=grid)
    iota = data["iota"]

    Nalpha = int(8)  # Number of field lines

    assert Nalpha == int(8), "Nalpha in the compute function hard-coded to 8!"

    # Field lines on which to evaluate ballooning stability
    alpha = jnp.linspace(0, np.pi, Nalpha + 1)[:Nalpha]

    # Number of toroidal transits of the field line
    ntor = int(3)

    # Number of point along a field line in ballooning space
    N0 = int(2.0 * ntor * eq.M_grid * eq.N_grid + 1)

    # range of the ballooning coordinate zeta
    zeta = np.linspace(-jnp.pi * ntor, jnp.pi * ntor, N0)

    for i in range(len(surfaces)):
        rho = surfaces[i] * np.ones((N0 * Nalpha,))

        theta_PEST = alpha[:, None] + iota[0] * zeta
        zeta_full = jnp.tile(zeta, Nalpha)

        theta_PEST = theta_PEST.flatten()
        zeta_full = zeta_full.flatten()

        theta_coords = jnp.array([rho, theta_PEST, zeta_full]).T

        # Rootfinding theta for a given theta_PEST
        desc_coords = compute_theta_coords(
            eq, theta_coords, L_lmn=eq.L_lmn, tol=1e-8, maxiter=25
        )

        sfl_grid = Grid(desc_coords, sort=False)

        data_keys = ["ideal_ball_gamma1", "ideal_ball_gamma2", "Newcomb_metric"]
        data = eq.compute(data_keys, grid=sfl_grid)

        lam1 = data["ideal_ball_gamma1"]
        lam2 = data["ideal_ball_gamma2"]
        Newcomb_metric = data["Newcomb_metric"]

        np.testing.assert_allclose(lam1, lam2, atol=5e-3, rtol=1e-8)

        if lam2 > 0:
            assert (
                Newcomb_metric >= 1
            ), "Newcomb metric indicates stabiliy for an unstable equilibrium"
        else:
            assert (
                Newcomb_metric < 1
            ), "Newcomb metric indicates instabiliy for a stable equilibrium"


@pytest.mark.unit
def test_compare_with_COBRAVMEC():
    """Compare DESC ballooning solved with COBRAVMEC."""

    def find_root_simple(x, y):
        sign_changes = np.where(np.diff(np.sign(y)))[0]

        if len(sign_changes) == 0:
            return None  # No zero crossing found

        # Get the indices where y changes sign
        i = sign_changes[0]

        # Linear interpolation
        x0, x1 = x[i], x[i + 1]
        y0, y1 = y[i], y[i + 1]

        # Calculate the zero crossing
        x_zero = x0 + (0 - y0) * (x1 - x0) / (y1 - y0)

        return x_zero

    A = np.loadtxt("./tests/inputs/cobra_grate.HELIOTRON_L24_M16_N12")

    ns1 = int(A[0, 2])
    nangles = int(np.shape(A)[0] / (ns1 + 1))

    B = np.zeros((ns1,))
    for i in range(nangles):
        if i == 0:
            B = A[i + 1 : (i + 1) * ns1 + 1, 2]
        else:
            B = np.vstack((B, A[i * ns1 + i + 1 : (i + 1) * ns1 + i + 1, 2]))

    gamma1 = np.amax(B, axis=0)

    s1 = np.linspace(0, 1, ns1)
    s1 = s1 + np.diff(s1)[0]

    # COBRAVMEC calculated everything in s(=rho^2),
    # DESC calculates in rho(=sqrt(s))
    rho1 = np.sqrt(s1)

    root_COBRAVMEC = find_root_simple(rho1, gamma1)

    try:
        eq = desc.examples.get("HELIOTRON")[-1]
    except TypeError:
        eq = desc.examples.get("HELIOTRON")

    # Flux surfaces on which to evaluate ballooning stability
    surfaces = [0.98, 0.985, 0.99, 0.995, 1.0]

    grid = LinearGrid(rho=jnp.array(surfaces), NFP=eq.NFP)
    eq_data_keys = ["iota"]

    data = eq.compute(eq_data_keys, grid=grid)
    iota = data["iota"]

    Nalpha = int(8)  # Number of field lines

    assert Nalpha == int(8), "Nalpha in the compute function hard-coded to 8!"

    # Field lines on which to evaluate ballooning stability
    alpha = jnp.linspace(0, np.pi, Nalpha + 1)[:Nalpha]

    # Number of toroidal transits of the field line
    ntor = int(3)

    # Number of point along a field line in ballooning space
    N0 = int(2.0 * ntor * eq.M_grid * eq.N_grid + 1)

    # range of the ballooning coordinate zeta
    zeta = np.linspace(-jnp.pi * ntor, jnp.pi * ntor, N0)

    lam2_array = np.zeros(
        len(surfaces),
    )

    for i in range(len(surfaces)):
        rho = surfaces[i] * np.ones((N0 * Nalpha,))

        theta_PEST = alpha[:, None] + iota[0] * zeta
        zeta_full = jnp.tile(zeta, Nalpha)

        theta_PEST = theta_PEST.flatten()
        zeta_full = zeta_full.flatten()

        theta_coords = jnp.array([rho, theta_PEST, zeta_full]).T

        # Rootfinding theta for a given theta_PEST
        desc_coords = compute_theta_coords(
            eq, theta_coords, L_lmn=eq.L_lmn, tol=1e-8, maxiter=25
        )

        sfl_grid = Grid(desc_coords, sort=False)

        data_keys = ["ideal_ball_gamma2"]
        data = eq.compute(data_keys, grid=sfl_grid)

        lam2_array[i] = data["ideal_ball_gamma2"]

    root_DESC = find_root_simple(np.array(surfaces), lam2_array)

    # Comparing the points of marginal stability from COBRAVMEC and DESC
    np.testing.assert_allclose(root_COBRAVMEC, root_DESC, atol=5e-4, rtol=1e-8)
