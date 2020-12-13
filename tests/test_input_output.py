import unittest
import os
import pathlib
import h5py
from desc.input_reader import InputReader
from desc.equilibrium_io import hdf5Writer, hdf5Reader
from desc.configuration import Configuration, Equilibrium
#from desc.input_output import read_input


#class TestIO(unittest.TestCase):
#    """tests for input/output functions"""
#
#    def test_min_input(self):
#        dirname = os.path.dirname(__file__)
#        filename = os.path.join(dirname, 'MIN_INPUT')
#        inputs = read_input(filename)
#
#        self.assertEqual(len(inputs), 26)

class TestInputReader(unittest.TestCase):

    def setUp(self):
        self.argv0 = []
        self.argv1 = ['nonexistant_input_file']
        self.argv2 = ['./tests/MIN_INPUT']

    def test_no_input_file(self):
        with self.assertRaises(NameError):
            ir = InputReader(cl_args=self.argv0)

    def test_nonexistant_input_file(self):
        with self.assertRaises(FileNotFoundError):
            ir = InputReader(cl_args=self.argv1)

    def test_min_input(self):
        ir = InputReader(cl_args=self.argv2)
        #self.assertEqual(ir.args.prog, 'DESC', 'Program is incorrect.')
        self.assertEqual(ir.args.input_file[0], self.argv2[0],
                'Input file name does not match')
        #self.assertEqual(ir.output_path, self.argv2[0] + '.output',
        #        'Default output file does not match.')
        self.assertEqual(ir.input_path,
                str(pathlib.Path('./'+self.argv2[0]).resolve()),
                'Path to input file is incorrect.')
        #Test defaults
        self.assertFalse(ir.args.plot, 'plot is not default False')
        self.assertFalse(ir.args.quiet, 'quiet is not default False')
        self.assertFalse(ir.args.verbose, 'verbose is not default False')
        #self.assertEqual(ir.args.vmec_path, '', "vmec path is not default ''")
        #self.assertFalse(ir.args.gpuID, 'gpu argument was given')
        self.assertFalse(ir.args.numpy, 'numpy is not default False')
        self.assertEqual(os.environ['DESC_USE_NUMPY'], '', 'numpy environment '
            'variable incorrect with default argument')
        self.assertFalse(ir.args.version, 'version is not default False')
        self.assertEqual(len(ir.inputs), 28, 'number of inputs does not match '

            'number expected in MIN_INPUT')
        # test equality of arguments

    def test_np_environ(self):
        argv = self.argv2 + ['--numpy']
        ir = InputReader(cl_args=argv)
        self.assertEqual(os.environ['DESC_USE_NUMPY'], 'True', 'numpy '
            'environment variable incorrect on use')

    def test_quiet_verbose(self):
        ir = InputReader(self.argv2)
        self.assertEqual(ir.inputs['verbose'], 1, "value of inputs['verbose'] "
            "incorrect on no arguments")
        argv = self.argv2 + ['-v']
        ir = InputReader(argv)
        self.assertEqual(ir.inputs['verbose'], 2, "value of inputs['verbose'] "
            "incorrect on verbose argument")
        argv.append('-q')
        ir = InputReader(argv)
        self.assertEqual(ir.inputs['verbose'], 0, "value of inputs['verbose'] "
            "incorrect on quiet argument")

    def test_vmec_to_desc_input(self):
        pass

class MockObject:
    def __init__(self):
        self._save_attrs_ = ['a', 'b', 'c']

class Testhdf5Writer(unittest.TestCase):

    def setUp(self):
        self.filename = 'writer_test_file'
        self.file_mode = 'w'

    def test_given_filename(self):
        writer = hdf5Writer(self.filename, self.file_mode)
        self.assertFalse(writer.check_type(writer.target))
        self.assertTrue(writer.check_type(writer.base))
        self.assertTrue(writer._close_base_)
        writer.close()
        self.assertFalse(writer._close_base_)

    def test_given_file(self):
        f = h5py.File(self.filename, self.file_mode)
        writer = hdf5Writer(f, self.file_mode)
        self.assertTrue(writer.check_type(writer.target))
        self.assertTrue(writer.check_type(writer.base))
        self.assertFalse(writer._close_base_)
        #with self.assertWarns(RuntimeWarning):
        #    writer.close()
        self.assertFalse(writer._close_base_)
        f.close()

    def test_close_on_delete(self):
        writer = hdf5Writer(self.filename, self.file_mode)
        with self.assertRaises(OSError):
            newwriter = hdf5Writer(self.filename, self.file_mode)
        del writer
        newwriter = hdf5Writer(self.filename, self.file_mode)
        del newwriter

    def test_write_dict(self):
        thedict = {'1':1, '2':2, '3':3}
        writer = hdf5Writer(self.filename, self.file_mode)
        writer.write_dict(thedict)
        writer.write_dict(thedict, where=writer.sub('subgroup'))
        with self.assertRaises(SyntaxError):
            writer.write_dict(thedict, where='not a writable type')
        writer.close()
        f = h5py.File(self.filename, 'r')
        g = f['subgroup']
        for key in thedict.keys():
            self.assertTrue(key in f.keys())
            self.assertTrue(key in g.keys())
        f.close()

    def test_write_obj(self):
        mo = MockObject()
        writer = hdf5Writer(self.filename, self.file_mode)
        #writer should throw runtime warning if any save_attrs are undefined
        with self.assertWarns(RuntimeWarning):
            writer.write_obj(mo)
        writer.close()
        writer = hdf5Writer(self.filename, self.file_mode)
        for name in mo._save_attrs_:
            setattr(mo, name, name)
        writer.write_obj(mo)
        groupname = 'initial'
        writer.write_obj(mo, where=writer.sub(groupname))
        writer.close()
        f = h5py.File(self.filename, 'r')
        for key in mo._save_attrs_:
            self.assertTrue(key in f.keys())
        self.assertTrue(groupname in f.keys())
        initial = f[groupname]
        for key in mo._save_attrs_:
            self.assertTrue(key in initial.keys())
        f.close()

class Testhdf5Reader(unittest.TestCase):

    def setUp(self):
        self.filename = 'reader_test_file'
        self.file_mode = 'r'
        self.thedict = {'a':'a', 'b':'b', 'c':'c'}
        f = h5py.File(self.filename, 'w')
        self.subgroup = 'subgroup'
        g = f.create_group(self.subgroup)
        for key in self.thedict.keys():
            f.create_dataset(key, data=self.thedict[key])
            g.create_dataset(key, data=self.thedict[key])
        f.close()

    def test_given_filename(self):
        reader = hdf5Reader(self.filename)
        self.assertFalse(reader.check_type(reader.target))
        self.assertTrue(reader.check_type(reader.base))
        self.assertTrue(reader._close_base_)
        reader.close()
        self.assertFalse(reader._close_base_)

    def test_given_file(self):
        f = h5py.File(self.filename, self.file_mode)
        reader = hdf5Reader(f)
        self.assertTrue(reader.check_type(reader.target))
        self.assertTrue(reader.check_type(reader.base))
        self.assertFalse(reader._close_base_)
        #with self.assertWarns(RuntimeWarning):
        #    reader.close()
        self.assertFalse(reader._close_base_)
        f.close()

    #def test_close_on_delete(self):
    #    reader = hdf5Reader(self.filename)
    #    with self.assertRaises(OSError):
    #        newreader = hdf5Reader(self.filename)
    #    del reader
    #    newreader = hdf5Reader(self.filename)
    #    del newreader

    def test_read_dict(self):
        reader = hdf5Reader(self.filename)
        newdict = {}
        newsubdict = {}
        otherdict = {}
        reader.read_dict(newdict)
        reader.read_dict(newsubdict, where=reader.sub(self.subgroup))
        with self.assertRaises(SyntaxError):
            reader.read_dict(otherdict, where='not a readable type')
        reader.close()
        self.assertTrue(self.thedict == newdict)
        self.assertTrue(self.thedict == newsubdict)

    def test_read_obj(self):
        mo = MockObject()
        reader = hdf5Reader(self.filename)
        reader.read_obj(mo)
        mo._save_attrs_  += '4'
        with self.assertWarns(RuntimeWarning):
            reader.read_obj(mo)
        del mo._save_attrs_[-1]
        print(mo._save_attrs_)
        submo = MockObject()
        reader.read_obj(submo, where=reader.sub(self.subgroup))
        for key in mo._save_attrs_:
            self.assertTrue(hasattr(mo, key))
            self.assertTrue(hasattr(submo, key))

    def test_load_configuration(self):
        pass

    def test_load_equilibrium(self):
        pass
