import numpy as np
from netCDF4 import Dataset

from desc.equilibrium import EquilibriaFamily
from desc.vmec import VMECIO


# test for sucessful runs


def test_SOLOVEV_run(SOLOVEV):
    """Tests that the SOLOVEV example runs without errors."""
    output = SOLOVEV["output"]
    assert output.returncode == 0


def test_DSHAPE_run(DSHAPE):
    """Tests that the DSHAPE example runs without errors."""
    output = DSHAPE["output"]
    assert output.returncode == 0


# compare results to VMEC solution


def test_SOLOVEV_results(SOLOVEV):
    """Tests that the SOLOVEV example gives the same result as VMEC."""

    equil = EquilibriaFamily(load_from=str(SOLOVEV["output_path"]))[-1]
    err = VMECIO.area_difference_vmec(equil, SOLOVEV["vmec_nc_path"])

    assert err < 1e-3


def test_DSHAPE_results(DSHAPE):
    """Tests that the DSHAPE example gives the same result as VMEC."""

    equil = EquilibriaFamily(load_from=str(DSHAPE["output_path"]))[-1]
    err = VMECIO.area_difference_vmec(equil, DSHAPE["vmec_nc_path"])

    assert err < 1e-3
