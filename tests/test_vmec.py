import numpy as np
from netCDF4 import Dataset

from desc.vmec import VMECIO


def test_VMECIO(SOLOVEV):
    """Tests if loading and then saving a VMEC equilibrium gives the original file.
    """

    input_path = "examples//VMEC//wout_SOLOVEV.nc"
    output_path = SOLOVEV["output_path"] + "_vmec.nc"

    eq = VMECIO.load(input_path)
    VMECIO.save(eq, output_path)

    file1 = Dataset(input_path, mode="r")
    file2 = Dataset(output_path, mode="r")

    rmnc1 = file1.variables["rmnc"][:]
    rmnc2 = file2.variables["rmnc"][:]
    zmns1 = file1.variables["zmns"][:]
    zmns2 = file2.variables["zmns"][:]

    np.testing.assert_allclose(rmnc2[(0, -1), :], rmnc1[(0, -1), :], atol=1e-6)
    np.testing.assert_allclose(zmns2[(0, -1), :], zmns1[(0, -1), :], atol=1e-6)
