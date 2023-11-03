"""Tests for Mercier stability functions."""

import numpy as np
import pytest
from netCDF4 import Dataset

import desc.examples
import desc.io
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid
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


def get_vmec_data(stellarator, quantity):
    """Get data from a VMEC wout.nc file.

    Parameters
    ----------
    stellarator : str
        The equilibrium's fixture.
    quantity: str
        Name of the quantity to return.

    Returns
    -------
    rho : ndarray
        Radial coordinate.
    q : ndarray
        Variable from VMEC output.

    """
    f = Dataset(str(stellarator["vmec_nc_path"]))
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
@pytest.mark.solve
def test_compute_d_shear(DSHAPE_current, HELIOTRON_ex):
    """Test that D_shear has a stabilizing effect and matches VMEC."""

    def test(stellarator, rho_range=(0, 1), rtol=1e-12, atol=0.0):
        eq = desc.io.load(load_from=str(stellarator["desc_h5_path"]))[-1]
        rho, d_shear_vmec = get_vmec_data(stellarator, "DShear")
        grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=rho)
        d_shear = grid.compress(eq.compute("D_shear", grid=grid)["D_shear"])

        assert np.all(
            d_shear[bool(grid.axis.size) :] >= 0
        ), "D_shear should always have a stabilizing effect."
        assert_all_close(d_shear, d_shear_vmec, rho, rho_range, rtol, atol)

    test(DSHAPE_current, (0.3, 0.9), atol=0.01, rtol=0.1)
    test(HELIOTRON_ex)


@pytest.mark.unit
@pytest.mark.solve
def test_compute_d_current(DSHAPE_current, HELIOTRON_ex):
    """Test calculation of D_current stability criterion against VMEC."""

    def test(
        stellarator, rho_range=DEFAULT_RANGE, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL
    ):
        eq = desc.io.load(load_from=str(stellarator["desc_h5_path"]))[-1]
        rho, d_current_vmec = get_vmec_data(stellarator, "DCurr")
        grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=rho)
        d_current = grid.compress(eq.compute("D_current", grid=grid)["D_current"])

        assert (
            np.nonzero(np.sign(d_current) != np.sign(d_current_vmec))[0].size
            <= MAX_SIGN_DIFF
        )
        assert_all_close(d_current, d_current_vmec, rho, rho_range, rtol, atol)

    test(DSHAPE_current, (0.3, 0.9), rtol=1e-1, atol=1e-2)
    test(HELIOTRON_ex, (0.25, 0.85), rtol=1e-1)


@pytest.mark.unit
@pytest.mark.solve
def test_compute_d_well(DSHAPE_current, HELIOTRON_ex):
    """Test calculation of D_well stability criterion against VMEC."""

    def test(
        stellarator, rho_range=DEFAULT_RANGE, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL
    ):
        eq = desc.io.load(load_from=str(stellarator["desc_h5_path"]))[-1]
        rho, d_well_vmec = get_vmec_data(stellarator, "DWell")
        grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=rho)
        d_well = grid.compress(eq.compute("D_well", grid=grid)["D_well"])

        assert (
            np.nonzero(np.sign(d_well) != np.sign(d_well_vmec))[0].size <= MAX_SIGN_DIFF
        )
        assert_all_close(d_well, d_well_vmec, rho, rho_range, rtol, atol)

    test(DSHAPE_current, (0.3, 0.9), rtol=1e-1)
    test(HELIOTRON_ex, (0.01, 0.45), rtol=1.75e-1)
    test(HELIOTRON_ex, (0.45, 0.6), atol=7.2e-1)
    test(HELIOTRON_ex, (0.6, 0.99), rtol=2e-2)


@pytest.mark.unit
@pytest.mark.solve
def test_compute_d_geodesic(DSHAPE_current, HELIOTRON_ex):
    """Test that D_geodesic has a destabilizing effect and matches VMEC."""

    def test(
        stellarator, rho_range=DEFAULT_RANGE, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL
    ):
        eq = desc.io.load(load_from=str(stellarator["desc_h5_path"]))[-1]
        rho, d_geodesic_vmec = get_vmec_data(stellarator, "DGeod")
        grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=rho)
        d_geodesic = grid.compress(eq.compute("D_geodesic", grid=grid)["D_geodesic"])

        assert np.all(
            d_geodesic[bool(grid.axis.size) :] <= 0
        ), "D_geodesic should always have a destabilizing effect."
        assert_all_close(d_geodesic, d_geodesic_vmec, rho, rho_range, rtol, atol)

    test(DSHAPE_current, (0.3, 0.9), rtol=1e-1)
    test(HELIOTRON_ex, (0.15, 0.825), rtol=1.2e-1)
    test(HELIOTRON_ex, (0.85, 0.95), atol=1.2e-1)


@pytest.mark.unit
@pytest.mark.solve
def test_compute_d_mercier(DSHAPE_current, HELIOTRON_ex):
    """Test calculation of D_Mercier stability criterion against VMEC."""

    def test(
        stellarator, rho_range=DEFAULT_RANGE, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL
    ):
        eq = desc.io.load(load_from=str(stellarator["desc_h5_path"]))[-1]
        rho, d_mercier_vmec = get_vmec_data(stellarator, "DMerc")
        grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=rho)
        d_mercier = grid.compress(eq.compute("D_Mercier", grid=grid)["D_Mercier"])

        assert (
            np.nonzero(np.sign(d_mercier) != np.sign(d_mercier_vmec))[0].size
            <= MAX_SIGN_DIFF
        )
        assert_all_close(d_mercier, d_mercier_vmec, rho, rho_range, rtol, atol)

    test(DSHAPE_current, (0.3, 0.9), rtol=1e-1, atol=1e-2)
    test(HELIOTRON_ex, (0.1, 0.325), rtol=1.3e-1)
    test(HELIOTRON_ex, (0.325, 0.95), rtol=5e-2)


@pytest.mark.unit
@pytest.mark.solve
def test_compute_magnetic_well(DSHAPE_current, HELIOTRON_ex):
    """Test that D_well and magnetic_well match signs under finite pressure."""

    def test(stellarator, rho=np.linspace(0, 1, 128)):
        eq = desc.io.load(load_from=str(stellarator["desc_h5_path"]))[-1]
        grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=rho)
        d_well = grid.compress(eq.compute("D_well", grid=grid)["D_well"])
        magnetic_well = grid.compress(
            eq.compute("magnetic well", grid=grid)["magnetic well"]
        )
        assert (
            np.nonzero(np.sign(d_well) != np.sign(magnetic_well))[0].size
            <= MAX_SIGN_DIFF
        )

    test(DSHAPE_current)
    test(HELIOTRON_ex)


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
