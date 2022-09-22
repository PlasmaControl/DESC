import unittest
import pytest
import os
import pathlib
import h5py
import shutil
import numpy as np

from desc.io import InputReader, load
from desc.io import hdf5Writer, hdf5Reader
from desc.io.ascii_io import write_ascii, read_ascii
from desc.utils import equals
from desc.grid import LinearGrid
from desc.basis import FourierZernikeBasis
from desc.transform import Transform
from desc.equilibrium import Equilibrium


def test_vmec_input(tmpdir_factory):
    input_path = "./tests/inputs/input.DSHAPE"
    tmpdir = tmpdir_factory.mktemp("desc_inputs")
    tmp_path = tmpdir.join("input.DSHAPE")
    shutil.copyfile(input_path, tmp_path)
    ir = InputReader(cl_args=[str(tmp_path)])
    vmec_inputs = ir.inputs
    path = tmpdir.join("desc_from_vmec")
    ir.write_desc_input(path, ir.inputs)
    ir2 = InputReader(cl_args=[str(path)])
    desc_inputs = ir2.inputs
    for d, v in zip(desc_inputs, vmec_inputs):
        d.pop("output_path")
        v.pop("output_path")
    assert all([equals(in1, in2) for in1, in2 in zip(vmec_inputs, desc_inputs)])


class TestInputReader(unittest.TestCase):
    def setUp(self):
        self.argv0 = []
        self.argv1 = ["nonexistant_input_file"]
        self.argv2 = ["./tests/inputs/MIN_INPUT"]

    def test_no_input_file(self):
        with self.assertRaises(NameError):
            InputReader(cl_args=self.argv0)

    def test_nonexistant_input_file(self):
        with self.assertRaises(FileNotFoundError):
            InputReader(cl_args=self.argv1)

    def test_min_input(self):
        ir = InputReader(cl_args=self.argv2)
        # self.assertEqual(ir.args.prog, 'DESC', 'Program is incorrect.')
        self.assertEqual(
            ir.args.input_file[0], self.argv2[0], "Input file name does not match"
        )
        # self.assertEqual(ir.output_path, self.argv2[0] + '.output',
        #        'Default output file does not match.')
        self.assertEqual(
            ir.input_path,
            str(pathlib.Path("./" + self.argv2[0]).resolve()),
            "Path to input file is incorrect.",
        )
        # Test defaults
        self.assertFalse(ir.args.plot, "plot is not default False")
        self.assertFalse(ir.args.quiet, "quiet is not default False")
        self.assertEqual(ir.args.verbose, 1, "verbose is not default 1")
        # self.assertEqual(ir.args.vmec_path, '', "vmec path is not default ''")
        # self.assertFalse(ir.args.gpuID, 'gpu argument was given')
        self.assertFalse(ir.args.numpy, "numpy is not default False")
        self.assertEqual(
            os.environ["DESC_BACKEND"],
            "jax",
            "numpy environment variable incorrect with default argument",
        )
        self.assertFalse(ir.args.version, "version is not default False")
        self.assertEqual(
            len(ir.inputs[0]),
            28,
            "number of inputs does not match number expected in MIN_INPUT",
        )
        # test equality of arguments

    def test_np_environ(self):
        argv = self.argv2 + ["--numpy"]
        InputReader(cl_args=argv)
        self.assertEqual(
            os.environ["DESC_BACKEND"],
            "numpy",
            "numpy environment variable incorrect on use",
        )

    def test_quiet_verbose(self):
        ir = InputReader(self.argv2)
        self.assertEqual(
            ir.inputs[0]["verbose"],
            1,
            "value of inputs['verbose'] incorrect on no arguments",
        )
        argv = self.argv2 + ["-v"]
        ir = InputReader(argv)
        self.assertEqual(
            ir.inputs[0]["verbose"],
            2,
            "value of inputs['verbose'] incorrect on verbose argument",
        )
        argv = self.argv2 + ["-vv"]
        ir = InputReader(argv)
        self.assertEqual(
            ir.inputs[0]["verbose"],
            3,
            "value of inputs['verbose'] incorrect on double verbose argument",
        )
        argv = self.argv2 + ["-q"]
        ir = InputReader(argv)
        self.assertEqual(
            ir.inputs[0]["verbose"],
            0,
            "value of inputs['verbose'] incorrect on quiet argument",
        )

    def test_vmec_to_desc_input(self):
        # FIXME: maybe just store a file we know is converted correctly,
        #  and checksum compare a live conversion to it
        pass


class MockObject:
    def __init__(self):
        self._io_attrs_ = ["a", "b", "c"]


def test_writer_given_filename(writer_test_file):
    writer = hdf5Writer(writer_test_file, "w")
    assert writer.check_type(writer.target) is False
    assert writer.check_type(writer.base) is True
    assert writer._close_base_ is True
    writer.close()
    assert writer._close_base_ is False


def test_writer_given_file(writer_test_file):
    f = h5py.File(writer_test_file, "w")
    writer = hdf5Writer(f, "w")
    assert writer.check_type(writer.target) is True
    assert writer.check_type(writer.base) is True
    assert writer._close_base_ is False
    # with self.assertWarns(RuntimeWarning):
    #    writer.close()
    assert writer._close_base_ is False
    f.close()


def test_writer_close_on_delete(writer_test_file):
    writer = hdf5Writer(writer_test_file, "w")
    with pytest.raises(OSError):
        newwriter = hdf5Writer(writer_test_file, "w")
    del writer
    newwriter = hdf5Writer(writer_test_file, "w")
    del newwriter


def test_writer_write_dict(writer_test_file):
    thedict = {"1": 1, "2": 2, "3": 3}
    writer = hdf5Writer(writer_test_file, "w")
    writer.write_dict(thedict)
    with pytest.raises(SyntaxError):
        writer.write_dict(thedict, where="not a writable type")
    writer.close()
    f = h5py.File(writer_test_file, "r")
    for key in thedict.keys():
        assert key in f.keys()
        assert f[key][()] == thedict[key]
    f.close()
    reader = hdf5Reader(writer_test_file)

    dict1 = reader.read_dict()
    assert dict1 == thedict
    reader.close()


def test_writer_write_list(writer_test_file):
    thelist = ["1", 1, "2", 2, "3", 3]
    writer = hdf5Writer(writer_test_file, "w")
    writer.write_list(thelist)
    with pytest.raises(SyntaxError):
        writer.write_list(thelist, where="not a writable type")
    writer.close()
    reader = hdf5Reader(writer_test_file)

    list1 = reader.read_list()
    assert list1 == thelist
    reader.close()


def test_writer_write_obj(writer_test_file):
    mo = MockObject()
    writer = hdf5Writer(writer_test_file, "w")
    # writer should throw runtime warning if any save_attrs are undefined
    with pytest.warns(RuntimeWarning):
        writer.write_obj(mo)
    writer.close()
    writer = hdf5Writer(writer_test_file, "w")
    for name in mo._io_attrs_:
        setattr(mo, name, name)
    writer.write_obj(mo)
    groupname = "initial"
    writer.write_obj(mo, where=writer.sub(groupname))
    writer.close()
    f = h5py.File(writer_test_file, "r")
    for key in mo._io_attrs_:
        assert key in f.keys()
    assert groupname in f.keys()
    initial = f[groupname]
    for key in mo._io_attrs_:
        assert key in initial.keys()
    f.close()


def test_reader_given_filename(reader_test_file):
    reader = hdf5Reader(reader_test_file)
    assert reader.check_type(reader.target) is False
    assert reader.check_type(reader.base) is True
    assert reader._close_base_ is True
    reader.close()
    assert reader._close_base_ is False


def test_reader_given_file(reader_test_file):
    f = h5py.File(reader_test_file, "r")
    reader = hdf5Reader(f)
    assert reader.check_type(reader.target) is True
    assert reader.check_type(reader.base) is True
    assert reader._close_base_ is False
    assert reader._close_base_ is False
    f.close()


def test_reader_read_obj(reader_test_file):
    mo = MockObject()
    reader = hdf5Reader(reader_test_file)
    reader.read_obj(mo)
    mo._io_attrs_ += "4"
    with pytest.warns(RuntimeWarning):
        reader.read_obj(mo)
    del mo._io_attrs_[-1]
    submo = MockObject()
    reader.read_obj(submo, where=reader.sub("subgroup"))
    for key in mo._io_attrs_:
        assert hasattr(mo, key)
        assert hasattr(submo, key)


def test_pickle_io(SOLOVEV, tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("desc_inputs")
    tmp_path = tmpdir.join("solovev_test.pkl")
    eqf = load(load_from=str(SOLOVEV["desc_h5_path"]))
    eqf.save(tmp_path, file_format="pickle")
    peqf = load(tmp_path, file_format="pickle")
    assert equals(eqf, peqf)


def test_ascii_io(SOLOVEV, tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("desc_inputs")
    tmp_path = tmpdir.join("solovev_test.txt")
    eq1 = load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    write_ascii(tmp_path, eq1)
    with pytest.warns(UserWarning):
        eq2 = read_ascii(tmp_path)
    assert np.allclose(eq1.R_lmn, eq2.R_lmn)
    assert np.allclose(eq1.Z_lmn, eq2.Z_lmn)
    assert np.allclose(eq1.L_lmn, eq2.L_lmn)


def test_copy():
    basis = FourierZernikeBasis(2, 2, 2)
    grid = LinearGrid(2, 2, 2)
    transform1 = Transform(grid, basis, method="direct1")
    transform2 = transform1.copy(deepcopy=False)

    assert transform1.basis is transform2.basis
    np.testing.assert_allclose(
        transform1.matrices["direct1"][0][0][0],
        transform2.matrices["direct1"][0][0][0],
        rtol=1e-10,
        atol=1e-10,
    )

    transform3 = transform1.copy(deepcopy=True)
    assert transform1.basis is not transform3.basis
    assert transform1.basis.eq(transform3.basis)
    np.testing.assert_allclose(
        transform1.matrices["direct1"][0][0][0],
        transform3.matrices["direct1"][0][0][0],
        rtol=1e-10,
        atol=1e-10,
    )


def test_save_none(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("none_test")
    eq = Equilibrium()
    eq._iota = None
    eq.save(tmpdir + "none_test.h5")
    eq1 = load(tmpdir + "none_test.h5")
    assert eq1.iota is None
