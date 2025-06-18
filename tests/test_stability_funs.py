"""Tests for stability functions."""

import numpy as np
import pytest
from netCDF4 import Dataset
from scipy.constants import mu_0
from scipy.interpolate import interp1d

import desc.examples
import desc.io
from desc.backend import jnp
from desc.equilibrium import Equilibrium
from desc.grid import Grid, LinearGrid, QuadratureGrid
from desc.objectives import MagneticWell, MercierStability
from desc.utils import PRINT_WIDTH, cross, dot

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
    phi = np.array(f.variables["phi"])
    rho = np.sqrt(phi / phi[-1])
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
        (0.07, 0.45),
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
    mercier_obj.print_value(mercier_obj.xs(eq))
    out = capsys.readouterr()
    pre_width = len("Maximum ")

    corr_out = str(
        "Precomputing transforms\n"
        + "Maximum "
        + f"{mercier_obj._print_value_fmt:<{PRINT_WIDTH - pre_width}}"
        + "{:10.3e} ".format(np.max(Dmerc))
        + mercier_obj._units
        + "\n"
        + "Minimum "
        + f"{mercier_obj._print_value_fmt:<{PRINT_WIDTH - pre_width}}"
        + "{:10.3e} ".format(np.min(Dmerc))
        + mercier_obj._units
        + "\n"
        + "Average "
        + f"{mercier_obj._print_value_fmt:<{PRINT_WIDTH - pre_width}}"
        + "{:10.3e} ".format(np.mean(Dmerc))
        + mercier_obj._units
        + "\n"
        + "Maximum "
        + f"{mercier_obj._print_value_fmt:<{PRINT_WIDTH - pre_width}}"
        + "{:10.3e} ".format(np.max(Dmerc / mercier_obj.normalization))
        + "(normalized)"
        + "\n"
        + "Minimum "
        + f"{mercier_obj._print_value_fmt:<{PRINT_WIDTH - pre_width}}"
        + "{:10.3e} ".format(np.min(Dmerc / mercier_obj.normalization))
        + "(normalized)"
        + "\n"
        + "Average "
        + f"{mercier_obj._print_value_fmt:<{PRINT_WIDTH - pre_width}}"
        + "{:10.3e} ".format(np.mean(Dmerc / mercier_obj.normalization))
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

    obj.print_value(obj.xs(eq))
    out = capsys.readouterr()
    pre_width = len("Maximum ")

    corr_out = str(
        "Precomputing transforms\n"
        + "Maximum "
        + f"{obj._print_value_fmt:<{PRINT_WIDTH - pre_width}}"
        + "{:10.3e} ".format(np.max(magwell))
        + obj._units
        + "\n"
        + "Minimum "
        + f"{obj._print_value_fmt:<{PRINT_WIDTH - pre_width}}"
        + "{:10.3e} ".format(np.min(magwell))
        + obj._units
        + "\n"
        + "Average "
        + f"{obj._print_value_fmt:<{PRINT_WIDTH - pre_width}}"
        + "{:10.3e} ".format(np.mean(magwell))
        + obj._units
        + "\n"
    )
    assert out.out == corr_out


@pytest.mark.unit
def test_ballooning_geometry(tmpdir_factory):
    """Test the geometry coefficients used for the adjoint-ballooning solver.

    We calculate coefficients using two different method and cross-compare them.
    The same coefficients are used for local gyrokinetic solvers which would
    be useful when we couple DESC with GX/GS2 etc.
    Observation: The larger the force error, the worse the tests behave. For
    example, HELIOTRON coefficients are hard to match
    """
    psi = 0.5  # Actually rho^2 (normalized)
    alpha = 0.0
    ntor = 2

    eq0 = desc.examples.get("W7-X")
    eq1 = desc.examples.get("precise_QA")

    eq_list = [eq0, eq1]

    for eq in eq_list:
        eq_keys = ["iota", "iota_r", "a", "rho", "psi"]

        grid = QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid, NFP=eq.NFP)
        data_eq = eq.compute(eq_keys, grid=grid)

        fi = interp1d(
            data_eq["rho"][grid.unique_rho_idx],
            data_eq["iota"][grid.unique_rho_idx],
        )
        fs = interp1d(
            data_eq["rho"][grid.unique_rho_idx],
            data_eq["iota_r"][grid.unique_rho_idx],
        )

        iotas = fi(np.sqrt(psi))
        shears = fs(np.sqrt(psi))

        rho = np.sqrt(psi)
        N = (8 * eq.M_grid * eq.N_grid) * ntor + 1
        zeta = np.linspace(-ntor * np.pi, ntor * np.pi, N)

        data_keys = [
            "p_r",
            "psi",
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
            "cvdrift",
            "gbdrift",
            "cvdrift0",
            "|B|",
            "B^zeta",
            "theta_PEST_t",
            "g^tt",
            "g^zz",
            "g^rt",
            "g^tz",
            "g^rz",
        ]

        grid = Grid.create_meshgrid([rho, alpha, zeta], coordinates="raz")

        # Fieldline data
        data = eq.compute(data_keys, grid=grid)

        # Data used to defining normalization
        data01 = eq.compute(["Psi", "a"], grid=LinearGrid(rho=np.array([1.0])))
        # normalizations
        psi_b = data01["Psi"][-1] / (2 * jnp.pi)
        Lref = data01["a"]
        Bref = 2 * abs(psi_b) / Lref**2

        # Data used only used for signs
        psi_s = data_eq["psi"][-1]
        sign_psi = psi_s / np.abs(psi_s)
        sign_iota = iotas / np.abs(iotas)

        x = Lref * np.sqrt(psi)
        shat = -x / iotas * shears / Lref

        grad_alpha = data["grad(alpha)"]

        g_sup_rr = data["g^rr"]
        g_sup_ra = data["g^ra"]
        g_sup_aa = data["g^aa"]

        lambda_r = data["lambda_r"]
        lambda_t = data["lambda_t"]
        lambda_z = data["lambda_z"]

        grad_alpha_r = lambda_r - zeta * shears
        grad_alpha_t = 1 + lambda_t
        grad_alpha_z = -iotas + lambda_z

        mod_grad_alpha_alt = np.sqrt(
            grad_alpha_r**2 * data["g^rr"]
            + grad_alpha_t**2 * data["g^tt"]
            + grad_alpha_z**2 * data["g^zz"]
            + 2 * grad_alpha_r * grad_alpha_t * data["g^rt"]
            + 2 * grad_alpha_r * grad_alpha_z * data["g^rz"]
            + 2 * grad_alpha_t * grad_alpha_z * data["g^tz"]
        )

        psi_r = 2 * psi_b * rho
        grad_psi_dot_grad_alpha_alt = (
            psi_r * grad_alpha_r * data["g^rr"]
            + psi_r * grad_alpha_t * data["g^rt"]
            + psi_r * grad_alpha_z * data["g^rz"]
        )

        grad_psi_dot_grad_psi_alt = psi_r**2 * data["g^rr"]

        modB = data["|B|"]
        B_sup_zeta = data["B^zeta"]

        gds2_alt = mod_grad_alpha_alt**2 * Lref**2 * psi
        gds2 = g_sup_aa * Lref**2 * psi

        gds21_alt = -sign_iota * grad_psi_dot_grad_alpha_alt * shat / Bref
        gds21 = -sign_iota * g_sup_ra * shat / Bref * (psi_r)

        gds22 = grad_psi_dot_grad_psi_alt * (1 / psi) * (shat / (Lref * Bref)) ** 2
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

        np.testing.assert_allclose(gds2, gds2_alt, rtol=6e-3)
        np.testing.assert_allclose(gds22, gds22_alt)
        # gds21 is a zero crossing quantity, rtol won't work,
        # shifting the grid slightly can change rtol requirement significantly
        np.testing.assert_allclose(gds21, gds21_alt, atol=2e-3)
        np.testing.assert_allclose(gbdrift, gbdrift_alt)
        # cvdrift is a zero crossing quantity, rtol won't work,
        # shifting the grid slightly can change rtol requirement significantly
        np.testing.assert_allclose(cvdrift, cvdrift_alt, atol=5e-3)

        sqrt_g_PEST = data["sqrt(g)_PEST"]
        np.testing.assert_allclose(sqrt_g_PEST, 1 / (B_sup_zeta / psi_r))


@pytest.mark.unit
def test_grad_alpha_zeta0_maps():
    """Test computation of gds2 and c ballooning which redefine ∇α."""
    eq = desc.examples.get("W7-X")

    # This is ι ζ₀ and not ι ζ₀ sign ι.
    iota_zeta0 = np.linspace(-np.pi / 2, np.pi / 2, 15)[:, np.newaxis]
    data = eq.compute(
        [
            "alpha_r (secular)",
            "iota",
            "iota_r",
            "rho",
            "a",
            "psi",
            "psi_r",
            "p_r",
            "B^zeta",
            "gds2",
            "c ballooning",
        ],
        zeta0=iota_zeta0,
    )

    g_sup_aa = eq.compute(
        ["g^aa"],
        # Redefine ∇α to ∇(α + ι ζ₀ sign ι)
        data={
            "alpha_r (secular)": data["alpha_r (secular)"]
            + data["iota_r"] / jnp.abs(data["iota"]) * iota_zeta0
        },
    )["g^aa"]

    np.testing.assert_allclose(data["gds2"], g_sup_aa * data["rho"] ** 2)
    assert jnp.any(jnp.sign(data["iota"]) < 0), "The test is better if ι < 0."

    cvdrift = eq.compute(
        ["cvdrift"],
        # Redefine ∇α to ∇(α + ι ζ₀)
        data={
            "alpha_r (secular)": data["alpha_r (secular)"]
            + data["iota_r"] / data["iota"] * iota_zeta0
        },
    )["cvdrift"]

    psi_boundary = eq.Psi / (2 * jnp.pi)
    np.testing.assert_allclose(
        data["c ballooning"],
        (2 * psi_boundary * data["a"] * mu_0)  # a³ Bₙ μ₀
        * jnp.sign(data["psi"])
        * data["p_r"]
        / data["psi_r"]
        / data["B^zeta"]
        * cvdrift
        * data["rho"] ** 2
        * 2,
    )
    assert jnp.any(jnp.sign(data["psi"]) < 0), "The test is better if ψ < 0."


@pytest.mark.unit
def test_ballooning_stability_eval():
    """Cross-compare all the stability functions.

    We calculated the ideal ballooning growth rate and Newcomb ball
    metric for the HELIOTRON case at different radii.
    """
    from scipy.constants import mu_0

    eq = desc.examples.get("HELIOTRON")

    # Flux surfaces on which to evaluate ballooning stability
    surfaces = [0.01, 0.8, 1.0]

    grid = LinearGrid(rho=jnp.array(surfaces), NFP=eq.NFP)
    eq_data_keys = ["iota"]

    data = eq.compute(eq_data_keys, grid=grid)

    N_alpha = 8

    # Field lines on which to evaluate ballooning stability
    alpha = jnp.linspace(0, np.pi, N_alpha, endpoint=False)

    # Number of toroidal transits of the field line
    ntor = 3

    # Number of point along a field line in ballooning space
    N_zeta = 2 * ntor * eq.M_grid * eq.N_grid + 1

    # range of the ballooning coordinate zeta
    zeta = np.linspace(-jnp.pi * ntor, jnp.pi * ntor, N_zeta)

    for i in range(len(surfaces)):
        rho = np.array([surfaces[i]])

        grid = Grid.create_meshgrid([rho, alpha, zeta], coordinates="raz")

        data_keys0 = [
            "g^aa",
            "g^ra",
            "g^rr",
            "cvdrift",
            "cvdrift0",
            "|B|",
            "B^zeta",
            "p_r",
            "iota",
            "iota_r",
            "shear",
            "psi",
            "psi_r",
            "rho",
            "Psi",
        ]
        data0 = eq.compute(data_keys0, grid=grid)

        rho = jnp.mean(data0["rho"])

        data_keys01 = ["Psi", "a"]
        data01 = eq.compute(data_keys01, grid=LinearGrid(rho=np.array([1.0])))

        # here we use a different method for calculating the growth rate that uses
        # different numerics than "ideal ballooning lambda" so that we can verify them
        # against one another
        psi_b = data01["Psi"][-1] / (2 * jnp.pi)
        # Calculating a_N accurately requires a QuadratureGrid
        # which is automatically accounted for inside of eq.compute
        a_N = data01["a"]
        B_N = 2 * psi_b / a_N**2

        N_zeta0 = 15
        zeta0 = jnp.linspace(-0.5 * jnp.pi, 0.5 * jnp.pi, N_zeta0)

        iota = jnp.mean(data0["iota"])
        shear = jnp.mean(data0["shear"])
        psi = jnp.mean(data0["psi"])
        sign_psi = jnp.sign(psi)
        sign_iota = jnp.sign(iota)

        phi = zeta

        B = grid.meshgrid_reshape(data0["|B|"], "arz")
        B_sup_zeta = grid.meshgrid_reshape(data0["B^zeta"], "arz")
        gradpar = B_sup_zeta / B

        dpdpsi = jnp.mean(mu_0 * data0["p_r"] / data0["psi_r"])

        g_sup_aa = grid.meshgrid_reshape(data0["g^aa"], "arz")[None, ...]
        g_sup_ra = grid.meshgrid_reshape(data0["g^ra"], "arz")[None, ...]
        g_sup_rr = grid.meshgrid_reshape(data0["g^rr"], "arz")[None, ...]

        gds2 = jnp.reshape(
            jnp.transpose(
                rho**2
                * (
                    g_sup_aa
                    - 2
                    * sign_iota
                    * shear
                    / rho
                    * zeta0[:, None, None, None]
                    * g_sup_ra
                    + zeta0[:, None, None, None] ** 2 * (shear / rho) ** 2 * g_sup_rr
                ),
                axes=(1, 0, 2, 3),
            ),
            (N_alpha, N_zeta0, N_zeta),
        )

        f = a_N * B_N**3 * gds2 / B**3 * 1 / gradpar
        g = a_N**3 * B_N * gds2 / B * gradpar
        g_half = (g[:, :, 1:] + g[:, :, :-1]) / 2

        cvdrift = grid.meshgrid_reshape(data0["cvdrift"], "arz")[None, ...]
        cvdrift0 = grid.meshgrid_reshape(data0["cvdrift0"], "arz")[None, ...]

        c = (
            a_N**3
            * B_N
            * jnp.reshape(
                jnp.transpose(
                    2
                    / B_sup_zeta[None, ...]
                    * sign_psi
                    * rho**2
                    * dpdpsi
                    * (
                        cvdrift
                        - shear / (2 * rho**2) * zeta0[:, None, None, None] * cvdrift0
                    ),
                    axes=(1, 0, 2, 3),
                ),
                (N_alpha, N_zeta0, N_zeta),
            )
        )

        h = phi[1] - phi[0]

        A = jnp.zeros((N_alpha, N_zeta0, N_zeta - 2, N_zeta - 2))

        i = jnp.arange(N_alpha)[:, None, None, None]
        l = jnp.arange(N_zeta0)[None, :, None, None]
        j = jnp.arange(N_zeta - 2)[None, None, :, None]
        k = jnp.arange(N_zeta - 2)[None, None, None, :]

        # This is an alternate way of solving the same eigenvalue problem
        # The definition of this matrix is provided in Appendix A of
        # Gaur et al. https://doi.org/10.1017/S0022377823000995
        A = A.at[i, l, j, k].set(
            g_half[i, l, k] / f[i, l, k] / h**2 * (j - k == -1)
            + (
                -(g_half[i, l, j + 1] + g_half[i, l, j]) / f[i, l, j + 1] / h**2
                + c[i, l, j + 1] / f[i, l, j + 1]
            )
            * (j - k == 0)
            + g_half[i, l, j] / f[i, l, j + 1] / h**2 * (j - k == 1)
        )

        w = jnp.linalg.eigvals(jnp.where(jnp.isfinite(A), A, 0))

        lam1 = jnp.max(jnp.real(jnp.max(w, axis=(2,))))

        # now compute our regular metrics and compare them
        data_keys = ["ideal ballooning lambda", "Newcomb ballooning metric"]
        data = eq.compute(data_keys, grid=grid)

        lam2_full = data["ideal ballooning lambda"]

        X0_full = data["ideal ballooning eigenfunction"]

        assert np.shape(lam2_full) == (
            1,
            N_alpha,
            N_zeta0,
            1,
        ), "output eigenvalue spectrum does not have the right shape"

        assert np.shape(X0_full) == (
            1,
            N_alpha,
            N_zeta0,
            N_zeta - 2,
            1,
        ), "output eigenfunction spectrum does not have the right shape"

        lam2 = jnp.max(lam2_full)

        Newcomb_metric = data["Newcomb ballooning metric"]

        np.testing.assert_allclose(lam1, lam2, rtol=5e-5)

        if lam2 > 0:
            assert Newcomb_metric <= 0, (
                "Newcomb metric indicates stability for an unstable equilibrium, "
                f"surface = {rho}, lam = {lam2}, newcomb = {Newcomb_metric}"
            )
        else:
            assert Newcomb_metric > 0, (
                "Newcomb metric indicates instability for a stable equilibrium, "
                f"surface = {rho}, lam = {lam2}, newcomb = {Newcomb_metric}"
            )


@pytest.mark.unit
def test_ballooning_compare_with_COBRAVMEC():
    """Compare marginal stability points from DESC ballooning solve with COBRAVMEC.

    COBRAVMEC uses some higher order techniques to refine the growth rate value,
    so we don't expect the actual values to agree, but the sign should be the same.
    so instead of comparing growth rates, we compare the radial point where it
    becomes unstable. Recall that it should be stable on axis (zero pressure gradient)
    but can become unstable elsewhere.
    """

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

    eq = desc.examples.get("HELIOTRON")
    surfaces = np.array([0.98, 0.985, 0.99, 0.995, 1.0])
    Nalpha = 8
    alpha = jnp.linspace(0, np.pi, Nalpha + 1)[:Nalpha]
    ntor = 3
    N0 = 4 * ntor * eq.M_grid * eq.N_grid + 1
    zeta = np.linspace(-jnp.pi * ntor, jnp.pi * ntor, N0)
    lam2_array = []
    for i in range(surfaces.size):
        grid = Grid.create_meshgrid([surfaces[i], alpha, zeta], coordinates="raz")
        data = eq.compute("ideal ballooning lambda", grid=grid)
        lam2_array.append(data["ideal ballooning lambda"].max())
    lam2_array = np.array(lam2_array)
    root_DESC = find_root_simple(surfaces, lam2_array)
    np.testing.assert_allclose(root_COBRAVMEC, root_DESC, rtol=2e-3)
