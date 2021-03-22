import numpy as np
from netCDF4 import Dataset

from desc.equilibrium import EquilibriaFamily
from desc.vmec import VMECIO


# compare results to VMEC solution


def test_SOLOVEV_results(SOLOVEV):
    """Tests that the SOLOVEV example gives the same result as VMEC."""

    equil = EquilibriaFamily.load(
        load_from=str(SOLOVEV["output_path"]), file_format="hdf5"
    )[-1]
    err = VMECIO.area_difference_vmec(equil, SOLOVEV["vmec_nc_path"])

    assert err < 1e-3


def test_DSHAPE_results(DSHAPE):
    """Tests that the DSHAPE example gives the same result as VMEC."""

    equil = EquilibriaFamily.load(
        load_from=str(DSHAPE["output_path"]), file_format="hdf5"
    )[-1]
    err = VMECIO.area_difference_vmec(equil, DSHAPE["vmec_nc_path"])

    assert err < 1e-3
