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
    equil = EquilibriaFamily(load_from=str(SOLOVEV["output_path"]))
    VMECIO.save(equil[-1], str(SOLOVEV["desc_nc_path"]))

    file_vmec = Dataset(str(SOLOVEV["vmec_nc_path"]), mode="r")
    file_desc = Dataset(str(SOLOVEV["desc_nc_path"]), mode="r")

    rmnc_vmec = file_vmec.variables["rmnc"][:]
    rmnc_desc = file_desc.variables["rmnc"][:]
    zmns_vmec = file_vmec.variables["zmns"][:]
    zmns_desc = file_desc.variables["zmns"][:]
    lmns_vmec = file_vmec.variables["lmns"][:]
    lmns_desc = file_desc.variables["lmns"][:]

    file_vmec.close
    file_desc.close

    np.testing.assert_allclose(rmnc_desc, rmnc_vmec, atol=1e-3)
    np.testing.assert_allclose(zmns_desc, zmns_vmec, atol=1e-3)
    np.testing.assert_allclose(lmns_desc, lmns_vmec, atol=1e-3)


def test_DSHAPE_results(DSHAPE):
    """Tests that the DSHAPE example gives the same result as VMEC."""
    equil = EquilibriaFamily(load_from=str(DSHAPE["output_path"]))
    VMECIO.save(equil[-1], str(DSHAPE["desc_nc_path"]))

    file_vmec = Dataset(str(DSHAPE["vmec_nc_path"]), mode="r")
    file_desc = Dataset(str(DSHAPE["desc_nc_path"]), mode="r")

    rmnc_vmec = file_vmec.variables["rmnc"][:]
    rmnc_desc = file_desc.variables["rmnc"][:]
    zmns_vmec = file_vmec.variables["zmns"][:]
    zmns_desc = file_desc.variables["zmns"][:]
    lmns_vmec = file_vmec.variables["lmns"][:]
    lmns_desc = file_desc.variables["lmns"][:]

    file_vmec.close
    file_desc.close

    np.testing.assert_allclose(rmnc_desc, rmnc_vmec, atol=2e-2)
    np.testing.assert_allclose(zmns_desc, zmns_vmec, atol=2e-2)
    np.testing.assert_allclose(lmns_desc, lmns_vmec, atol=3e-2)
