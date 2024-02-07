"""Tests for EFIT-> DESC interface."""

import numpy as np
import pytest

from desc.efit import EFITIO
from desc.input_reader import InputReader


@pytest.mark.unit
def test_efit_to_desc_input(tmpdir_factory):
    """
    Test EFIT equilibrium to DESC input conversion.

    Test the efit to desc converted by comparing the
    converted input DESC file with the correct desc input file
    """
    efit_file_path = "./tests/inputs/eqdsk_cocos.out"

    tmpdir = tmpdir_factory.mktemp("efit_to_desc_inputs")
    tmp_path1 = tmpdir.join("desc_from_eqdsk_true")
    tmp_path2 = tmpdir.join("desc_from_eqdsk_bdry_dist_true")

    with pytest.warns(UserWarning):
        eq = EFITIO.load(str(efit_file_path), M=20)
        ir1 = EFITIO.efit_to_desc_input(eq, tmp_path1)

    ir1 = InputReader(cl_args=[str(tmp_path1)])
    arr1 = ir1.parse_inputs()[-1]["surface"]
    arr1 = arr1[arr1[:, 1].argsort()]
    arr1mneg = arr1[arr1[:, 1] < 0]
    arr1mpos = arr1[arr1[:, 1] >= 0]

    desc_input_truth = "./tests/inputs/desc_from_eqdsk_cocos"

    # with pytest.warns(UserWarning):
    ir2 = InputReader(cl_args=[str(desc_input_truth)])
    arr2 = ir2.parse_inputs()[-1]["surface"]
    arr2 = arr2[arr2[:, 1].argsort()]
    arr2mneg = arr2[arr2[:, 1] < 0]
    arr2mpos = arr2[arr2[:, 1] >= 0]

    np.testing.assert_allclose(
        np.minimum(
            np.linalg.norm(arr1mneg[:, 3:] - arr2mneg[:, 3:]),
            np.linalg.norm(arr1mneg[:, 3:] + arr2mneg[:, 3:]),
        ),
        0,
        atol=1e-8,
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        np.minimum(
            np.linalg.norm(arr1mpos[:, 3:] - arr2mpos[:, 3:]),
            np.linalg.norm(arr1mpos[:, 3:] + arr2mpos[:, 3:]),
        ),
        0,
        atol=1e-8,
        rtol=1e-8,
    )

    ir1 = InputReader()

    eq = EFITIO.load(str(efit_file_path), M=20, bdry_dist=0.99)
    ir1 = EFITIO.efit_to_desc_input(eq, tmp_path2)

    ir1 = InputReader(cl_args=[str(tmp_path2)])
    arr1 = ir1.parse_inputs()[-1]["surface"]
    arr1 = arr1[arr1[:, 1].argsort()]
    arr1mneg = arr1[arr1[:, 1] < 0]
    arr1mpos = arr1[arr1[:, 1] >= 0]

    desc_input_truth = "./tests/inputs/desc_from_eqdsk_cocos_bdry_dist"

    ir2 = InputReader(cl_args=[str(desc_input_truth)])
    arr2 = ir2.parse_inputs()[-1]["surface"]
    arr2 = arr2[arr2[:, 1].argsort()]
    arr2mneg = arr2[arr2[:, 1] < 0]
    arr2mpos = arr2[arr2[:, 1] >= 0]

    np.testing.assert_allclose(
        np.minimum(
            np.linalg.norm(arr1mneg[:, 3:] - arr2mneg[:, 3:]),
            np.linalg.norm(arr1mneg[:, 3:] + arr2mneg[:, 3:]),
        ),
        0,
        atol=1e-8,
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        np.minimum(
            np.linalg.norm(arr1mpos[:, 3:] - arr2mpos[:, 3:]),
            np.linalg.norm(arr1mpos[:, 3:] + arr2mpos[:, 3:]),
        ),
        0,
        atol=1e-8,
        rtol=1e-8,
    )
