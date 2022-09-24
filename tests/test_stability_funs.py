import numpy as np
from netCDF4 import Dataset
import pytest

import desc.io
from desc.compute.utils import compress
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid

default_range = (0.05, 1)
default_rtol = 1e-2
default_atol = 1e-6


def all_close(
    y1, y2, rho, rho_range=default_range, rtol=default_rtol, atol=default_atol
):
    """
    Test that the values of y1 and y2, over the indices defined by the given range,
    are closer than the given tolerance.

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
    interval = np.where((minimum < rho) & (rho < maximum))[0]
    np.testing.assert_allclose(y1[interval], y2[interval], rtol=rtol, atol=atol)


def get_vmec_data(name, quantity):
    """
    Parameters
    ----------
    name : str
        Name of the equilibrium.
    quantity: str
        Name of the quantity to return.

    Returns
    -------
    rho : ndarray
        Radial coordinate.
    q : ndarray
        Variable from VMEC output.

    """
    f = Dataset("tests/inputs/wout_" + name + ".nc")
    rho = np.sqrt(f.variables["phi"] / np.array(f.variables["phi"])[-1])
    q = np.asarray(f.variables[quantity])
    f.close()
    return rho, q


@pytest.mark.unit
@pytest.mark.solve
def test_compute_d_mercier(DSHAPE, HELIOTRON):
    eq = Equilibrium()
    DMerc = eq.compute("D_Mercier")["D_Mercier"]
    np.testing.assert_allclose(DMerc, 0, err_msg="should be 0 in vacuum")

    def test(
        stellarator, name, rho_range=default_range, rtol=default_rtol, atol=default_atol
    ):
        eq = desc.io.load(load_from=str(stellarator["desc_h5_path"]))[-1]
        rho, vmec = get_vmec_data(name, "DMerc")
        grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=rho)
        DMerc = compress(grid, eq.compute("D_Mercier", grid)["D_Mercier"])
        all_close(DMerc, vmec, rho, rho_range, rtol, atol)

    test(DSHAPE, "DSHAPE", (0.175, 1))
    test(HELIOTRON, "HELIOTRON", (0.1, 0.325), rtol=13e-2)
    test(HELIOTRON, "HELIOTRON", (0.325, 0.95), rtol=4e-2)


@pytest.mark.unit
@pytest.mark.solve    
def test_compute_d_shear(DSHAPE, HELIOTRON):
    eq = Equilibrium()
    DShear = eq.compute("D_shear")["D_shear"]
    np.testing.assert_allclose(DShear, 0, err_msg="should be 0 in vacuum")

    def test(
        stellarator, name, rho_range=default_range, rtol=default_rtol, atol=default_atol
    ):
        eq = desc.io.load(load_from=str(stellarator["desc_h5_path"]))[-1]
        rho, vmec = get_vmec_data(name, "DShear")
        grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=rho)
        DShear = compress(grid, eq.compute("D_shear", grid)["D_shear"])

        assert np.all(
            DShear[np.isfinite(DShear)] >= 0
        ), "D_shear should always have a stabilizing effect."
        all_close(DShear, vmec, rho, rho_range, rtol, atol)

    test(DSHAPE, "DSHAPE", (0, 1), 1e-12, 0)
    test(HELIOTRON, "HELIOTRON", (0, 1), 1e-12, 0)


@pytest.mark.unit
@pytest.mark.solve    
def test_compute_d_current(DSHAPE, HELIOTRON):
    eq = Equilibrium()
    DCurr = eq.compute("D_current")["D_current"]
    np.testing.assert_allclose(DCurr, 0, err_msg="should be 0 in vacuum")

    def test(
        stellarator, name, rho_range=default_range, rtol=default_rtol, atol=default_atol
    ):
        eq = desc.io.load(load_from=str(stellarator["desc_h5_path"]))[-1]
        rho, vmec = get_vmec_data(name, "DCurr")
        grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=rho)
        DCurr = compress(grid, eq.compute("D_current", grid)["D_current"])
        all_close(DCurr, vmec, rho, rho_range, rtol, atol)

    test(DSHAPE, "DSHAPE", (0.075, 0.975))
    test(HELIOTRON, "HELIOTRON", (0.25, 0.85), rtol=1e-1)

    
@pytest.mark.unit
@pytest.mark.solve
def test_compute_d_well(DSHAPE, HELIOTRON):
    eq = Equilibrium()
    DWell = eq.compute("D_well")["D_well"]
    np.testing.assert_allclose(DWell, 0, err_msg="should be 0 in vacuum")

    def test(
        stellarator, name, rho_range=default_range, rtol=default_rtol, atol=default_atol
    ):
        eq = desc.io.load(load_from=str(stellarator["desc_h5_path"]))[-1]
        rho, vmec = get_vmec_data(name, "DWell")
        grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=rho)
        DWell = compress(grid, eq.compute("D_well", grid)["D_well"])
        all_close(DWell, vmec, rho, rho_range, rtol, atol)

    test(DSHAPE, "DSHAPE", (0.11, 1))
    test(HELIOTRON, "HELIOTRON", (0.01, 0.45), rtol=175e-3)
    test(HELIOTRON, "HELIOTRON", (0.45, 0.6), atol=72e-2)
    test(HELIOTRON, "HELIOTRON", (0.6, 0.99), rtol=13e-3)


@pytest.mark.unit
@pytest.mark.solve    
def test_compute_d_geodesic(DSHAPE, HELIOTRON):
    eq = Equilibrium()
    DGeod = eq.compute("D_geodesic")["D_geodesic"]
    np.testing.assert_allclose(DGeod, 0, err_msg="should be 0 in vacuum")

    def test(
        stellarator, name, rho_range=default_range, rtol=default_rtol, atol=default_atol
    ):
        eq = desc.io.load(load_from=str(stellarator["desc_h5_path"]))[-1]
        rho, vmec = get_vmec_data(name, "DGeod")
        grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=rho)
        DGeod = compress(grid, eq.compute("D_geodesic", grid)["D_geodesic"])

        assert np.all(
            DGeod[np.isfinite(DGeod)] <= 0
        ), "DGeod should always have a destabilizing effect."
        all_close(DGeod, vmec, rho, rho_range, rtol, atol)

    test(DSHAPE, "DSHAPE", (0.15, 0.975))
    test(HELIOTRON, "HELIOTRON", (0.15, 0.825), rtol=12e-2)
    test(HELIOTRON, "HELIOTRON", (0.85, 0.95), atol=12e-2)


@pytest.mark.unit
@pytest.mark.solve    
def test_compute_magnetic_well(DSHAPE, HELIOTRON):
    def test(stellarator, name):
        eq = desc.io.load(load_from=str(stellarator["desc_h5_path"]))[-1]
        rho, vmec = get_vmec_data(name, "DWell")
        grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, rho=rho)
        magnetic_well = compress(
            grid, eq.compute("magnetic well", grid)["magnetic well"]
        )
        # sign should match for finite non-zero pressure cases
        assert len(np.where(np.sign(magnetic_well) != np.sign(vmec))[0]) <= 5

    test(DSHAPE, "DSHAPE")
    test(HELIOTRON, "HELIOTRON")
