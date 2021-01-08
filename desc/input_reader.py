import argparse
import pathlib
import sys
import warnings
import os
import re
import numpy as np
from datetime import datetime
from termcolor import colored


class InputReader:
    """
    Reads command line arguments and parses input files.

    Arguments
    ---------
    cl_args (optional): list
        explicit command line arguments

    Attributes
    ----------
    args : Namespace
        parsed namespace of all command line arguments
    inputs: dict
        dictionary of values from input file
    input_path: string
        path to input file
    output_path: string
        path to output file

    Methods
    -------
    parse_args
    parse_inputs
    write_desc_input

    """

    def __init__(self, cl_args=None):
        """Initialize InputReader instance.

        Parameters
        ----------
        cl_args : None or list (Default = None)
            command line arguments to parse. Default (=None) is to use command line arguments from sys.argv.

        Returns
        -------
        None

        """
        self.args = self.parse_args(cl_args=cl_args)

        if not self.args.version:
            print("Reading input from {}".format(self.input_path))
            print("Outputs will be written to {}".format(self.output_path))

            self.inputs = self.parse_inputs()

    def parse_args(self, cl_args=None):
        """Parse command line arguments.

        Parameters
        ----------
        cl_args : None or list (Default = None)
            command line arguments to parse. Default (=None) is to use command line arguments from sys.argv.

        Returns
        -------
        args : namespace
            parsed arguments

        """
        self.parser = self._get_parser_()

        if cl_args is None:
            cl_args = sys.argv[1:]
        else:
            pass
        args = self.parser.parse_args(cl_args)

        if args.version:
            import desc
            print(desc.__version__)
            return args

        if len(args.input_file) == 0:
            raise NameError('Input file path must be specified')
            #print('Input file path must be specified')
            # return None

        # ''.join(args.input_file)).resolve()
        self.input_path = pathlib.Path(args.input_file[0]).resolve()
        if self.input_path.is_file():
            self.input_path = str(self.input_path)
        else:
            raise FileNotFoundError("Input file '{}' does not exist.".format(
                str(self.input_path)))

        if args.output:
            self.output_path = args.output
        else:
            self.output_path = self.input_path+'.output'

        if args.numpy:
            os.environ['DESC_USE_NUMPY'] = 'True'
        else:
            os.environ['DESC_USE_NUMPY'] = ''

        return args

    def _get_parser_(self):
        """Gets parser for command line arguments.

        Parameters
        ----------

        Returns
        -------
        parser : argparse object
            argument parser

        """

        parser = argparse.ArgumentParser(prog='DESC',
                                         description='DESC computes equilibria by solving the force balance equations. '
                                         + 'It can also be used for perturbation analysis and sensitivity studies '
                                         + 'to see how the equilibria change as input parameters are varied.')
        parser.add_argument('input_file', nargs='*',
                            help='Path to input file')
        parser.add_argument('-o', '--output', metavar='output_file',
                            help='Path to output file. If not specified, defaults to <input_name>.output')
        parser.add_argument('-p', '--plot', action='store_true',
                            help='Plot results after solver finishes')
        parser.add_argument('-q', '--quiet', action='store_true',
                            help='Do not display any progress information')
        parser.add_argument('-v', '--verbose', action='store_true',
                            help='Display detailed progress information')
        parser.add_argument('--vmec', metavar='vmec_path',
                            help='Path to VMEC data for comparison plot')
        parser.add_argument('--gpu', '-g', action='store', nargs='?', default=False, const=True, metavar='gpuID',
                            help='Use GPU if available, and an optional device ID to use a specific GPU.'
                            + ' If no ID is given, default is to select the GPU with most available memory.'
                            + ' Note that not all of the computation will be done '
                            + 'on the gpu, only the most expensive parts where the I/O efficiency is worth it.')
        parser.add_argument('--numpy', action='store_true', help="Use numpy backend.Performance will be much slower,"
                            + " and autodiff won't work but may be useful for debugging")
        parser.add_argument('--version', action='store_true',
                            help='Display version number and exit')
        return parser

    def parse_inputs(self):
        """Reads input from DESC input file, converts from VMEC input if necessary

        Parameters
        ----------
        fname : string
            filename of input file

        Returns
        -------
        inputs : dict
            all the input parameters and options

        """

        # default values
        inputs = {
            'sym': False,
            'Psi': 1.0,
            'NFP': 1,
            'L': np.atleast_1d(None),
            'M': np.atleast_1d(0),
            'N': np.atleast_1d(0),
            'M_grid': np.atleast_1d(0),
            'N_grid': np.atleast_1d(0),
            'bdry_ratio': np.atleast_1d(1.0),
            'pres_ratio': np.atleast_1d(1.0),
            'zeta_ratio': np.atleast_1d(1.0),
            'pert_order': np.atleast_1d(1),
            'ftol': np.atleast_1d(1e-6),
            'xtol': np.atleast_1d(1e-6),
            'gtol': np.atleast_1d(1e-6),
            'nfev': np.atleast_1d(None),
            'optim_method': 'trf',
            'errr_mode': 'force',
            'bdry_mode': 'spectral',
            'zern_mode': 'fringe',
            'node_mode': 'cheb1',
            'profiles': np.atleast_2d((0, 0.0, 0.0)),
            'boundary': np.atleast_2d((0, 0, 0.0, 0.0)),
            'axis':     np.atleast_2d((0, 0.0, 0.0)),
        }

        inputs['output_path'] = self.output_path

        if self.args.quiet:
            inputs['verbose'] = 0
        elif self.args.verbose:
            inputs['verbose'] = 2
        else:
            inputs['verbose'] = 1

        file = open(self.input_path, 'r')
        num_form = r'[-+]?\ *\d*\.?\d*(?:[Ee]\ *[-+]?\ *\d+)?'

        for line in file:

            # check if VMEC input file format
            isVMEC = re.search(r'&INDATA', line)
            if isVMEC:
                print('Converting VMEC input to DESC input')
                path = self.input_path + '_desc'
                self._vmec_to_desc_input_(self.input_path, path)
                print('Generated DESC input file {}:'.format(path))
                return self.parse_input(path)

            # extract numbers & words
            match = re.search(r'[!#]', line)
            if match:
                comment = match.start()
            else:
                comment = len(line)
            match = re.search(r'=', line)
            if match:
                equals = match.start()
            else:
                equals = len(line)
            command = (line.strip()+' ')[0:comment]
            argument = (command.strip()+' ')[0:equals]
            numbers = [float(x) for x in re.findall(
                num_form, command) if re.search(r'\d', x)]
            words = command[equals+1:].split()

            # global parameters
            match = re.search(r'sym', argument, re.IGNORECASE)
            if match:
                inputs['sym'] = int(numbers[0])
            match = re.search(r'Psi', argument, re.IGNORECASE)
            if match:
                inputs['Psi'] = numbers[0]
            match = re.search(r'NFP', argument, re.IGNORECASE)
            if match:
                inputs['NFP'] = int(numbers[0])

            # spectral resolution
            match = re.search(r'L_rad', argument, re.IGNORECASE)
            if match:
                inputs['L'] = np.array(numbers).astype(int)
            match = re.search(r'M_pol', argument, re.IGNORECASE)
            if match:
                inputs['M'] = np.array(numbers).astype(int)
            match = re.search(r'N_tor', argument, re.IGNORECASE)
            if match:
                inputs['N'] = np.array(numbers).astype(int)
            match = re.search(r'M_grid', argument, re.IGNORECASE)
            if match:
                inputs['M_grid'] = np.array(numbers).astype(int)
            match = re.search(r'N_grid', argument, re.IGNORECASE)
            if match:
                inputs['N_grid'] = np.array(numbers).astype(int)

            # continuation parameters
            match = re.search(r'bdry_ratio', argument, re.IGNORECASE)
            if match:
                inputs['bdry_ratio'] = np.array(numbers).astype(float)
            match = re.search(r'pres_ratio', argument, re.IGNORECASE)
            if match:
                inputs['pres_ratio'] = np.array(numbers).astype(float)
            match = re.search(r'zeta_ratio', argument, re.IGNORECASE)
            if match:
                inputs['zeta_ratio'] = np.array(numbers).astype(float)
            match = re.search(r'pert_order', argument, re.IGNORECASE)
            if match:
                inputs['pert_order'] = np.array(numbers).astype(int)

            # solver tolerances
            match = re.search(r'ftol', argument, re.IGNORECASE)
            if match:
                inputs['ftol'] = np.array(numbers).astype(float)
            match = re.search(r'xtol', argument, re.IGNORECASE)
            if match:
                inputs['xtol'] = np.array(numbers).astype(float)
            match = re.search(r'gtol', argument, re.IGNORECASE)
            if match:
                inputs['gtol'] = np.array(numbers).astype(float)
            match = re.search(r'nfev', argument, re.IGNORECASE)
            if match:
                inputs['nfev'] = np.array(
                    [None if i == 0 else i for i in numbers]).astype(int)

            # solver methods
            match = re.search(r'optim_method', argument, re.IGNORECASE)
            if match:
                inputs['optim_method'] = words[0]
            match = re.search(r'errr_mode', argument, re.IGNORECASE)
            if match:
                inputs['errr_mode'] = words[0]
            match = re.search(r'bdry_mode', argument, re.IGNORECASE)
            if match:
                inputs['bdry_mode'] = words[0]
            match = re.search(r'zern_mode', argument, re.IGNORECASE)
            if match:
                inputs['zern_mode'] = words[0]
            match = re.search(r'node_mode', argument, re.IGNORECASE)
            if match:
                inputs['node_mode'] = words[0]

            # coefficient indicies
            match = re.search(r'l\s*:\s*'+num_form, command, re.IGNORECASE)
            if match:
                l = [int(x) for x in re.findall(num_form, match.group(0))
                     if re.search(r'\d', x)][0]
            match = re.search(r'm\s*:\s*'+num_form, command, re.IGNORECASE)
            if match:
                m = [int(x) for x in re.findall(num_form, match.group(0))
                     if re.search(r'\d', x)][0]
            match = re.search(r'n\s*:\s*'+num_form, command, re.IGNORECASE)
            if match:
                n = [int(x) for x in re.findall(num_form, match.group(0))
                     if re.search(r'\d', x)][0]

            # profile coefficients
            match = re.search(r'\sp\s*=\s*'+num_form, command, re.IGNORECASE)
            if match:
                p_l = [float(x) for x in re.findall(
                    num_form, match.group(0)) if re.search(r'\d', x)][0]
                prof_idx = np.where(inputs['profiles'][:, 0] == l)[0]
                if prof_idx.size == 0:
                    prof_idx = np.atleast_1d(inputs['profiles'].shape[0])
                    inputs['profiles'] = np.pad(
                        inputs['profiles'], ((0, 1), (0, 0)), mode='constant')
                    inputs['profiles'][prof_idx[0], 0] = l
                inputs['profiles'][prof_idx[0], 1] = p_l
            match = re.search(r'\si\s*=\s*'+num_form, command, re.IGNORECASE)
            if match:
                i_l = [float(x) for x in re.findall(
                    num_form, match.group(0)) if re.search(r'\d', x)][0]
                prof_idx = np.where(inputs['profiles'][:, 0] == l)[0]
                if prof_idx.size == 0:
                    prof_idx = np.atleast_1d(inputs['profiles'].shape[0])
                    inputs['profiles'] = np.pad(
                        inputs['profiles'], ((0, 1), (0, 0)), mode='constant')
                    inputs['profiles'][prof_idx[0], 0] = l
                inputs['profiles'][prof_idx[0], 2] = i_l

            # boundary surface coefficients
            match = re.search(r'R1\s*=\s*'+num_form, command, re.IGNORECASE)
            if match:
                R1 = [float(x) for x in re.findall(
                    num_form, match.group(0)) if re.search(r'\d', x)][1]
                bdry_m = np.where(inputs['boundary'][:, 0] == m)[0]
                bdry_n = np.where(inputs['boundary'][:, 1] == n)[0]
                bdry_idx = bdry_m[np.in1d(bdry_m, bdry_n)]
                if bdry_idx.size == 0:
                    bdry_idx = np.atleast_1d(inputs['boundary'].shape[0])
                    inputs['boundary'] = np.pad(
                        inputs['boundary'], ((0, 1), (0, 0)), mode='constant')
                    inputs['boundary'][bdry_idx[0], 0] = m
                    inputs['boundary'][bdry_idx[0], 1] = n
                inputs['boundary'][bdry_idx[0], 2] = R1
            match = re.search(r'Z1\s*=\s*'+num_form, command, re.IGNORECASE)
            if match:
                Z1 = [float(x) for x in re.findall(
                    num_form, match.group(0)) if re.search(r'\d', x)][1]
                bdry_m = np.where(inputs['boundary'][:, 0] == m)[0]
                bdry_n = np.where(inputs['boundary'][:, 1] == n)[0]
                bdry_idx = bdry_m[np.in1d(bdry_m, bdry_n)]
                if bdry_idx.size == 0:
                    bdry_idx = np.atleast_1d(inputs['boundary'].shape[0])
                    inputs['boundary'] = np.pad(
                        inputs['boundary'], ((0, 1), (0, 0)), mode='constant')
                    inputs['boundary'][bdry_idx[0], 0] = m
                    inputs['boundary'][bdry_idx[0], 1] = n
                inputs['boundary'][bdry_idx[0], 3] = Z1

            # magnetic axis coefficients
            match = re.search(r'R0\s*=\s*'+num_form, command, re.IGNORECASE)
            if match:
                R0_n = [float(x) for x in re.findall(
                    num_form, match.group(0)) if re.search(r'\d', x)][1]
                axis_idx = np.where(inputs['axis'][:, 0] == n)[0]
                if axis_idx.size == 0:
                    axis_idx = np.atleast_1d(inputs['axis'].shape[0])
                    inputs['axis'] = np.pad(
                        inputs['axis'], ((0, 1), (0, 0)), mode='constant')
                    inputs['axis'][axis_idx[0], 0] = n
                inputs['axis'][axis_idx[0], 1] = R0_n
            match = re.search(r'Z0\s*=\s*'+num_form, command, re.IGNORECASE)
            if match:
                Z0_n = [float(x) for x in re.findall(
                    num_form, match.group(0)) if re.search(r'\d', x)][1]
                axis_idx = np.where(inputs['axis'][:, 0] == n)[0]
                if axis_idx.size == 0:
                    axis_idx = np.atleast_1d(inputs['axis'].shape[0])
                    inputs['axis'] = np.pad(
                        inputs['axis'], ((0, 1), (0, 0)), mode='constant')
                    inputs['axis'][axis_idx[0], 0] = n
                inputs['axis'][axis_idx[0], 2] = Z0_n

        # error handling
        if np.any(inputs['M'] == 0):
            raise IOError(colored('M_pol is not assigned', 'red'))
        if np.sum(inputs['boundary']) == 0:
            raise IOError(
                colored('Fixed-boundary surface is not assigned', 'red'))
        arrs = ['L', 'M', 'N', 'M_grid', 'N_grid', 'bdry_ratio',
                'pres_ratio', 'zeta_ratio', 'pert_order',
                'ftol', 'xtol', 'gtol', 'nfev']
        arr_len = 0
        for a in arrs:
            arr_len = max(arr_len, len(inputs[a]))
        for a in arrs:
            if inputs[a].size == 1:
                inputs[a] = np.broadcast_to(
                    inputs[a], arr_len, subok=True).copy()
            elif inputs[a].size != arr_len:
                raise IOError(colored(
                              'Continuation parameter arrays are not proper lengths', 'red'))

        # unsupplied values
        if np.sum(inputs['M_grid']) == 0:
            inputs['M_grid'] = np.rint(1.5*inputs['M']).astype(int)
        if np.sum(inputs['N_grid']) == 0:
            inputs['N_grid'] = np.rint(1.5*inputs['N']).astype(int)
        if np.sum(inputs['axis']) == 0:
            axis_idx = np.where(inputs['boundary'][:, 0] == 0)[0]
            inputs['axis'] = inputs['boundary'][axis_idx, 1:]
        if None in inputs['L']:
            default_L = {'fringe': 2*inputs['M'],
                         'ansi': inputs['M'],
                         'chevron': inputs['M'],
                         'house': 2*inputs['M']}
            inputs['L'] = default_L[inputs['zern_mode']]

        return inputs

    def write_desc_input(self, filename, inputs=None):
        """Generates a DESC input file from a dictionary of parameters

        Parameters
        ----------
        filename : str or path-like
            name of the file to create
        inputs : dict
            dictionary of input parameters

        Returns
        -------

        """

        # default to use self.inputs
        if inputs is None:
            inputs = self.inputs
        else:
            pass

        f = open(filename, 'w+')

        f.write('# global parameters \n')
        f.write('sym = {} \n'.format(inputs['sym']))
        f.write('NFP = {} \n'.format(inputs['NFP']))
        f.write('Psi_lcfs = {} \n'.format(inputs['Psi_lcfs']))

        f.write('\n# spectral resolution \n')
        f.write('Mpol = {} \n'.format(
            ', '.join([str(i) for i in np.atleast_1d(inputs['Mpol'])])))
        f.write('Ntor = {} \n'.format(
            ', '.join([str(i) for i in np.atleast_1d(inputs['Ntor'])])))
        f.write('M_grid = {} \n'.format(
            ', '.join([str(i) for i in np.atleast_1d(inputs['M_grid'])])))
        f.write('N_grid = {} \n'.format(
            ', '.join([str(i) for i in np.atleast_1d(inputs['N_grid'])])))

        f.write('\n# continuation parameters \n')
        f.write('bdry_ratio = {} \n'.format(
            ', '.join([str(i) for i in np.atleast_1d(inputs['bdry_ratio'])])))
        f.write('pres_ratio = {} \n'.format(
            ', '.join([str(i) for i in np.atleast_1d(inputs['pres_ratio'])])))
        f.write('zeta_ratio = {} \n'.format(
            ', '.join([str(i) for i in np.atleast_1d(inputs['zeta_ratio'])])))
        f.write('pert_order = {} \n'.format(
            ', '.join([str(i) for i in np.atleast_1d(inputs['pert_order'])])))

        f.write('\n# solver tolerances \n')
        f.write('ftol = {} \n'.format(
            ', '.join([str(i) for i in np.atleast_1d(inputs['ftol'])])))
        f.write('xtol = {} \n'.format(
            ', '.join([str(i) for i in np.atleast_1d(inputs['xtol'])])))
        f.write('gtol = {} \n'.format(
            ', '.join([str(i) for i in np.atleast_1d(inputs['gtol'])])))
        f.write('nfev = {} \n'.format(
            ', '.join([str(i) for i in np.atleast_1d(inputs['nfev'])])))

        f.write('\n# solver methods \n')
        f.write('optim_method = {} \n'.format(inputs['optim_method']))
        f.write('errr_mode = {} \n'.format(inputs['errr_mode']))
        f.write('bdry_mode = {} \n'.format(inputs['bdry_mode']))
        f.write('zern_mode = {} \n'.format(inputs['zern_mode']))
        f.write('node_mode = {} \n'.format(inputs['node_mode']))

        f.write('\n# pressure and rotational transform profiles \n')
        for (l, p, i) in inputs['axis']:
            f.write('l: {:3d}\tp = {:16.8E}\ti = {:16.8E}\n'.format(
                int(l), p, i))

        f.write('\n# fixed-boundary surface shape \n')
        for (m, n, R1, Z1) in inputs['boundary']:
            f.write('m: {:3d}\tn: {:3d}\tR1 = {:16.8E}\tZ1 = {:16.8E}\n'.format(
                int(m), int(n), R1, Z1))

        f.write('\n# magnetic axis initial guess \n')
        for (n, R0, Z0) in inputs['axis']:
            f.write('n: {:3d}\tR0 = {:16.8E}\tZ0 = {:16.8E}\n'.format(
                int(n), R0, Z0))

        f.close()

    def _vmec_to_desc_input_(self, vmec_fname, desc_fname):
        """Converts a VMEC input file to an equivalent DESC input file

        Parameters
        ----------
        vmec_fname : str or path-like
            filename of VMEC input file
        desc_fname : str or path-like
            filename of DESC input file. If it already exists it is overwritten.

        Returns
        -------

        """

        # file objects
        vmec_file = open(vmec_fname, 'r')
        desc_file = open(desc_fname, 'w')

        desc_file.seek(0)
        now = datetime.now()
        date = now.strftime('%m/%d/%Y')
        time = now.strftime('%H:%M:%S')
        desc_file.write('# This DESC input file was auto generated from the VMEC input file\n# {}\n# on {} at {}.\n\n'
                        .format(vmec_fname, date, time))

        num_form = r'[-+]?\ *\d*\.?\d*(?:[Ee]\ *[-+]?\ *\d+)?'
        Ntor = 99

        pres_scale = 1.0
        p_l = np.array([0.0])
        i_l = np.array([0.0])
        axis = np.array([[0, 0, 0.0]])
        bdry = np.array([[0, 0, 0.0, 0.0]])

        for line in vmec_file:
            comment = line.find('!')
            command = (line.strip()+' ')[0:comment]

            # global parameters
            if re.search(r'LRFP\s*=\s*T', command, re.IGNORECASE):
                warnings.warn(
                    colored('Using poloidal flux instead of toroidal flux!', 'yellow'))
            match = re.search('LASYM\s*=\s*[TF]', command, re.IGNORECASE)
            if match:
                if re.search(r'T', match.group(0), re.IGNORECASE):
                    desc_file.write('sym \t=   0\n')
                else:
                    desc_file.write('sym \t=   1\n')
            match = re.search(r'NFP\s*=\s*'+num_form, command, re.IGNORECASE)
            if match:
                numbers = [int(x) for x in re.findall(
                    num_form, match.group(0)) if re.search(r'\d', x)]
                desc_file.write('NFP\t\t\t= {:3d}\n'.format(numbers[0]))
            match = re.search(r'PHIEDGE\s*=\s*'+num_form,
                              command, re.IGNORECASE)
            if match:
                numbers = [float(x) for x in re.findall(
                    num_form, match.group(0)) if re.search(r'\d', x)]
                desc_file.write('Psi\t= {:16.8E}\n'.format(numbers[0]))
            match = re.search(r'MPOL\s*=\s*'+num_form, command, re.IGNORECASE)
            if match:
                numbers = [int(x) for x in re.findall(
                    num_form, match.group(0)) if re.search(r'\d', x)]
                desc_file.write('M_pol\t\t= {:3d}\n'.format(numbers[0]))
            match = re.search(r'NTOR\s*=\s*'+num_form, command, re.IGNORECASE)
            if match:
                numbers = [int(x) for x in re.findall(
                    num_form, match.group(0)) if re.search(r'\d', x)]
                desc_file.write('N_tor\t\t= {:3d}\n'.format(numbers[0]))
                Ntor = numbers[0]

            # pressure profile
            match = re.search(r'bPMASS_TYPE\s*=\s*\w*', command, re.IGNORECASE)
            if match:
                if not re.search(r'\bpower_series\b', match.group(0), re.IGNORECASE):
                    warnings.warn(
                        colored('Pressure is not a power series!', 'yellow'))
            match = re.search(r'GAMMA\s*=\s*'+num_form, command, re.IGNORECASE)
            if match:
                numbers = [float(x) for x in re.findall(
                    num_form, match.group(0)) if re.search(r'\d', x)]
                if numbers[0] != 0:
                    warnings.warn(colored('GAMMA is not 0.0', 'yellow'))
            match = re.search(r'BLOAT\s*=\s*'+num_form, command, re.IGNORECASE)
            if match:
                numbers = [float(x) for x in re.findall(
                    num_form, match.group(0)) if re.search(r'\d', x)]
                if numbers[0] != 1:
                    warnings.warn(colored('BLOAT is not 1.0', 'yellow'))
            match = re.search(r'SPRES_PED\s*=\s*'+num_form,
                              command, re.IGNORECASE)
            if match:
                numbers = [float(x) for x in re.findall(
                    num_form, match.group(0)) if re.search(r'\d', x)]
                if numbers[0] != 1:
                    warnings.warn(colored('SPRES_PED is not 1.0', 'yellow'))
            match = re.search(r'PRES_SCALE\s*=\s*'+num_form,
                              command, re.IGNORECASE)
            if match:
                numbers = [float(x) for x in re.findall(
                    num_form, match.group(0)) if re.search(r'\d', x)]
                pres_scale = numbers[0]
            match = re.search(r'AM\s*=(\s*'+num_form+')*',
                              command, re.IGNORECASE)
            if match:
                numbers = [float(x) for x in re.findall(
                    num_form, match.group(0)) if re.search(r'\d', x)]
                for k in range(len(numbers)):
                    l = 2*k
                    if p_l.size < l+1:
                        p_l = np.pad(p_l, (0, l+1-p_l.size), mode='constant')
                    p_l[l] = numbers[k]

            # rotational transform
            match = re.search(r'NCURR\s*=(\s*'+num_form+')*',
                              command, re.IGNORECASE)
            if match:
                numbers = [float(x) for x in re.findall(
                    num_form, match.group(0)) if re.search(r'\d', x)]
                if numbers[0] != 0:
                    warnings.warn(
                        colored('Not using rotational transform!', 'yellow'))
            if re.search(r'\bPIOTA_TYPE\b', command, re.IGNORECASE):
                if not re.search(r'\bpower_series\b', command, re.IGNORECASE):
                    warnings.warn(
                        colored('Iota is not a power series!', 'yellow'))
            match = re.search(r'AI\s*=(\s*'+num_form+')*',
                              command, re.IGNORECASE)
            if match:
                numbers = [float(x) for x in re.findall(
                    num_form, match.group(0)) if re.search(r'\d', x)]
                for k in range(len(numbers)):
                    l = 2*k
                    if i_l.size < l+1:
                        i_l = np.pad(i_l, (0, l+1-i_l.size), mode='constant')
                    i_l[l] = numbers[k]

            # magnetic axis
            match = re.search(r'RAXIS\s*=(\s*'+num_form+')*',
                              command, re.IGNORECASE)
            if match:
                numbers = [float(x) for x in re.findall(
                    num_form, match.group(0)) if re.search(r'\d', x)]
                for k in range(len(numbers)):
                    if k > Ntor:
                        l = -k+Ntor+1
                    else:
                        l = k
                    idx = np.where(axis[:, 0] == l)[0]
                    if np.size(idx) > 0:
                        axis[idx[0], 1] = numbers[k]
                    else:
                        axis = np.pad(axis, ((0, 1), (0, 0)), mode='constant')
                        axis[-1, :] = np.array([l, numbers[k], 0.0])
            match = re.search(r'ZAXIS\s*=(\s*'+num_form+')*',
                              command, re.IGNORECASE)
            if match:
                numbers = [float(x) for x in re.findall(
                    num_form, match.group(0)) if re.search(r'\d', x)]
                for k in range(len(numbers)):
                    if k > Ntor:
                        l = k-Ntor-1
                    else:
                        l = -k
                    idx = np.where(axis[:, 0] == l)[0]
                    if np.size(idx) > 0:
                        axis[idx[0], 2] = numbers[k]
                    else:
                        axis = np.pad(axis, ((0, 1), (0, 0)), mode='constant')
                        axis[-1, :] = np.array([l, 0.0, numbers[k]])

            # boundary shape
            # RBS*sin(m*t-n*p) = RBS*sin(m*t)*cos(n*p) - RBS*cos(m*t)*sin(n*p)
            match = re.search(r'RBS\(\s*'+num_form+'\s*,\s*'+num_form +
                              '\s*\)\s*=\s*'+num_form, command, re.IGNORECASE)
            if match:
                numbers = [float(x) for x in re.findall(
                    num_form, match.group(0)) if re.search(r'\d', x)]
                n = int(numbers[0])
                m = int(numbers[1])
                n_sgn = np.sign(np.array([n]))[0]
                n *= n_sgn
                if np.sign(m) < 0:
                    warnings.warn(colored('m is negative!', 'yellow'))
                RBS = numbers[2]
                if m != 0:
                    m_idx = np.where(bdry[:, 0] == -m)[0]
                    n_idx = np.where(bdry[:, 1] == n)[0]
                    idx = np.where(np.isin(m_idx, n_idx))[0]
                    if np.size(idx) > 0:
                        bdry[m_idx[idx[0]], 2] = RBS
                    else:
                        bdry = np.pad(bdry, ((0, 1), (0, 0)), mode='constant')
                        bdry[-1, :] = np.array([-m, n, RBS, 0.0])
                if n != 0:
                    m_idx = np.where(bdry[:, 0] == m)[0]
                    n_idx = np.where(bdry[:, 1] == -n)[0]
                    idx = np.where(np.isin(m_idx, n_idx))[0]
                    if np.size(idx) > 0:
                        bdry[m_idx[idx[0]], 2] = -n_sgn*RBS
                    else:
                        bdry = np.pad(bdry, ((0, 1), (0, 0)), mode='constant')
                        bdry[-1, :] = np.array([m, -n, -n_sgn*RBS, 0.0])
            # RBC*cos(m*t-n*p) = RBC*cos(m*t)*cos(n*p) + RBC*sin(m*t)*sin(n*p)
            match = re.search(r'RBC\(\s*'+num_form+'\s*,\s*'+num_form +
                              '\s*\)\s*=\s*'+num_form, command, re.IGNORECASE)
            if match:
                numbers = [float(x) for x in re.findall(
                    num_form, match.group(0)) if re.search(r'\d', x)]
                n = int(numbers[0])
                m = int(numbers[1])
                n_sgn = np.sign(np.array([n]))[0]
                n *= n_sgn
                if np.sign(m) < 0:
                    warnings.warn(colored('m is negative!', 'yellow'))
                RBS = numbers[2]
                if m != 0:
                    m_idx = np.where(bdry[:, 0] == -m)[0]
                    n_idx = np.where(bdry[:, 1] == n)[0]
                    idx = np.where(np.isin(m_idx, n_idx))[0]
                    if np.size(idx) > 0:
                        bdry[m_idx[idx[0]], 2] = RBS
                    else:
                        bdry = np.pad(bdry, ((0, 1), (0, 0)), mode='constant')
                        bdry[-1, :] = np.array([-m, n, RBS, 0.0])
                if n != 0:
                    m_idx = np.where(bdry[:, 0] == m)[0]
                    n_idx = np.where(bdry[:, 1] == -n)[0]
                    idx = np.where(np.isin(m_idx, n_idx))[0]
                    if np.size(idx) > 0:
                        bdry[m_idx[idx[0]], 2] = -n_sgn*RBS
                    else:
                        bdry = np.pad(bdry, ((0, 1), (0, 0)), mode='constant')
                        bdry[-1, :] = np.array([m, -n, -n_sgn*RBS, 0.0])
            # ZBS*sin(m*t-n*p) = ZBS*sin(m*t)*cos(n*p) - ZBS*cos(m*t)*sin(n*p)
            match = re.search(r'ZBS\(\s*'+num_form+'\s*,\s*'+num_form +
                              '\s*\)\s*=\s*'+num_form, command, re.IGNORECASE)
            if match:
                numbers = [float(x) for x in re.findall(
                    num_form, match.group(0)) if re.search(r'\d', x)]
                n = int(numbers[0])
                m = int(numbers[1])
                n_sgn = np.sign(np.array([n]))[0]
                n *= n_sgn
                if np.sign(m) < 0:
                    warnings.warn(colored('m is negative!', 'yellow'))
                ZBS = numbers[2]
                if m != 0:
                    m_idx = np.where(bdry[:, 0] == -m)[0]
                    n_idx = np.where(bdry[:, 1] == n)[0]
                    idx = np.where(np.isin(m_idx, n_idx))[0]
                    if np.size(idx) > 0:
                        bdry[m_idx[idx[0]], 3] = ZBS
                    else:
                        bdry = np.pad(bdry, ((0, 1), (0, 0)), mode='constant')
                        bdry[-1, :] = np.array([-m, n, 0.0, ZBS])
                if n != 0:
                    m_idx = np.where(bdry[:, 0] == m)[0]
                    n_idx = np.where(bdry[:, 1] == -n)[0]
                    idx = np.where(np.isin(m_idx, n_idx))[0]
                    if np.size(idx) > 0:
                        bdry[m_idx[idx[0]], 3] = -n_sgn*ZBS
                    else:
                        bdry = np.pad(bdry, ((0, 1), (0, 0)), mode='constant')
                        bdry[-1, :] = np.array([m, -n, 0.0, -n_sgn*ZBS])
            # ZBC*cos(m*t-n*p) = ZBC*cos(m*t)*cos(n*p) + ZBC*sin(m*t)*sin(n*p)
            match = re.search(r'ZBC\(\s*'+num_form+'\s*,\s*'+num_form +
                              '\s*\)\s*=\s*'+num_form, command, re.IGNORECASE)
            if match:
                numbers = [float(x) for x in re.findall(
                    num_form, match.group(0)) if re.search(r'\d', x)]
                n = int(numbers[0])
                m = int(numbers[1])
                n_sgn = np.sign(np.array([n]))[0]
                n *= n_sgn
                if np.sign(m) < 0:
                    warnings.warn(colored('m is negative!', 'yellow'))
                ZBC = numbers[2]
                m_idx = np.where(bdry[:, 0] == m)[0]
                n_idx = np.where(bdry[:, 1] == n)[0]
                idx = np.where(np.isin(m_idx, n_idx))[0]
                if np.size(idx) > 0:
                    bdry[m_idx[idx[0]], 3] = ZBC
                else:
                    bdry = np.pad(bdry, ((0, 1), (0, 0)), mode='constant')
                    bdry[-1, :] = np.array([m, n, 0.0, ZBC])
                if m != 0 and n != 0:
                    m_idx = np.where(bdry[:, 0] == -m)[0]
                    n_idx = np.where(bdry[:, 1] == -n)[0]
                    idx = np.where(np.isin(m_idx, n_idx))[0]
                    if np.size(idx) > 0:
                        bdry[m_idx[idx[0]], 3] = n_sgn*ZBC
                    else:
                        bdry = np.pad(bdry, ((0, 1), (0, 0)), mode='constant')
                        bdry[-1, :] = np.array([-m, -n, 0.0, n_sgn*ZBC])

            # catch multi-line inputs
            match = re.search(r'=', command)
            if not match:
                numbers = [float(x) for x in re.findall(
                    num_form, command) if re.search(r'\d', x)]
                if len(numbers) > 0:
                    raise IOError(
                        colored('Cannot handle multi-line VMEC inputs!', 'red'))

        p_l *= pres_scale
        desc_file.write('\n')
        desc_file.write('# pressure and rotational transform profiles\n')
        for k in range(max(p_l.size, i_l.size)):
            if k >= p_l.size:
                desc_file.write(
                    'l: {:3d}\tp = {:16.8E}\ti = {:16.8E}\n'.format(k, 0.0, i_l[k]))
            elif k >= i_l.size:
                desc_file.write(
                    'l: {:3d}\tp = {:16.8E}\ti = {:16.8E}\n'.format(k, p_l[k], 0.0))
            else:
                desc_file.write(
                    'l: {:3d}\tp = {:16.8E}\ti = {:16.8E}\n'.format(k, p_l[k], i_l[k]))

        desc_file.write('\n')
        desc_file.write('# fixed-boundary surface shape\n')
        for k in range(np.shape(bdry)[0]):
            desc_file.write('m: {:3d}\tn: {:3d}\tR1 = {:16.8E}\tZ1 = {:16.8E}\n'.format(
                int(bdry[k, 0]), int(bdry[k, 1]), bdry[k, 2], bdry[k, 3]))

        desc_file.write('\n')
        desc_file.write('# magnetic axis initial guess\n')
        for k in range(np.shape(axis)[0]):
            desc_file.write('n: {:3d}\tR0 = {:16.8E}\tZ0 = {:16.8E}\n'.format(
                int(axis[k, 0]), axis[k, 1], axis[k, 2]))

        desc_file.truncate()

        # close files
        vmec_file.close()
        desc_file.close()


def get_parser():
    """Standalone function that gets parser for command line arguments.

    Parameters
    ----------

    Returns
    -------
    parser : argparse object
        argument parser

    """

    parser = argparse.ArgumentParser(prog='DESC',
                                     description='DESC computes equilibria by solving the force balance equations. '
                                     + 'It can also be used for perturbation analysis and sensitivity studies '
                                     + 'to see how the equilibria change as input parameters are varied.')
    parser.add_argument('input_file', nargs='*',
                        help='Path to input file')
    parser.add_argument('-o', '--output', metavar='output_file',
                        help='Path to output file. If not specified, defaults to <input_name>.output')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='Plot results after solver finishes')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Do not display any progress information')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Display detailed progress information')
    parser.add_argument('--vmec', metavar='vmec_path',
                        help='Path to VMEC data for comparison plot')
    parser.add_argument('--gpu', '-g', action='store', nargs='?', default=False, const=True, metavar='gpuID',
                        help='Use GPU if available, and an optional device ID to use a specific GPU.'
                        + ' If no ID is given, default is to select the GPU with most available memory.'
                        + ' Note that not all of the computation will be done '
                        + 'on the gpu, only the most expensive parts where the I/O efficiency is worth it.')
    parser.add_argument('--numpy', action='store_true', help="Use numpy backend.Performance will be much slower,"
                        + " and autodiff won't work but may be useful for debugging")
    parser.add_argument('--version', action='store_true',
                        help='Display version number and exit')
    return parser
