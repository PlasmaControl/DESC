import argparse
import pathlib
import sys
import warnings
import os
import re
import h5py
import numpy as np
from datetime import datetime

from desc.backend import TextColors



class InputReader:
    def __init__(self):
        self.parse_args()

        print("Reading input from {}".format(self.input_path))
        print("Outputs will be written to {}".format(self.output_path))

        self.inputs = self.parse_inputs()

    def parse_args(self):
        self.parser = self._get_parser_()
        self.args = self.parser.parse_args()

        if len(args.input_file) == 0:
            raise NameError('Input file path must be specified')
            #print('Input file path must be specified')
            #return None

        if args.numpy:
            os.environ['DESC_USE_NUMPY'] = 'True'
        else:
            os.environ['DESC_USE_NUMPY'] = ''

        self.input_path = str(pathlib.Path(args.input_file[0]).resolve())
        if args.output:
            self.output_path = args.output
        else:
            self.output_path = self.input_path+'.output'

        return None

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
        'stell_sym': False,
        'NFP': 1,
        'Psi_lcfs': 1.0,
        'Mpol': np.atleast_1d(0),
        'Ntor': np.atleast_1d(0),
        'delta_lm': np.atleast_1d(None),
        'Mnodes': np.atleast_1d(0),
        'Nnodes': np.atleast_1d(0),
        'bdry_ratio': np.atleast_1d(1.0),
        'pres_ratio': np.atleast_1d(1.0),
        'zeta_ratio': np.atleast_1d(1.0),
        'errr_ratio': np.atleast_1d(1e-2),
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
        'cP': np.atleast_1d(0.0),
        'cI': np.atleast_1d(0.0),
        'axis': np.atleast_2d((0, 0.0, 0.0)),
        'bdry': np.atleast_2d((0, 0, 0.0, 0.0))
    }

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
            fname_desc = fname + '_desc'
            vmec_to_desc_input(fname, fname_desc)
            print('Generated DESC input file {}:'.format(fname_desc))
            return read_input(fname_desc)

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
        match = re.search(r'stell_sym', argument, re.IGNORECASE)
        if match:
            inputs['stell_sym'] = int(numbers[0])
        match = re.search(r'NFP', argument, re.IGNORECASE)
        if match:
            inputs['NFP'] = int(numbers[0])
        match = re.search(r'Psi_lcfs', argument, re.IGNORECASE)
        if match:
            inputs['Psi_lcfs'] = numbers[0]

        # spectral resolution
        match = re.search(r'Mpol', argument, re.IGNORECASE)
        if match:
            inputs['Mpol'] = np.array(numbers).astype(int)
        match = re.search(r'Ntor', argument, re.IGNORECASE)
        if match:
            inputs['Ntor'] = np.array(numbers).astype(int)
        match = re.search(r'delta_lm', argument, re.IGNORECASE)
        if match:
            inputs['delta_lm'] = np.array(numbers).astype(int)
        match = re.search(r'Mnodes', argument, re.IGNORECASE)
        if match:
            inputs['Mnodes'] = np.array(numbers).astype(int)
        match = re.search(r'Nnodes', argument, re.IGNORECASE)
        if match:
            inputs['Nnodes'] = np.array(numbers).astype(int)

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
        match = re.search(r'errr_ratio', argument, re.IGNORECASE)
        if match:
            inputs['errr_ratio'] = np.array(numbers).astype(float)
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
        match = re.search(r'errr_ratio', argument, re.IGNORECASE)
        if match:
            inputs['errr_ratio'] = np.array(numbers).astype(float)
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
        match = re.search(r'cP\s*=\s*'+num_form, command, re.IGNORECASE)
        if match:
            cP = [float(x) for x in re.findall(
                num_form, match.group(0)) if re.search(r'\d', x)][0]
            if inputs['cP'].size < l+1:
                inputs['cP'] = np.pad(
                    inputs['cP'], (0, l+1-inputs['cP'].size), mode='constant')
            inputs['cP'][l] = cP
        match = re.search(r'cI\s*=\s*'+num_form, command, re.IGNORECASE)
        if match:
            cI = [float(x) for x in re.findall(
                num_form, match.group(0)) if re.search(r'\d', x)][0]
            if inputs['cI'].size < l+1:
                inputs['cI'] = np.pad(
                    inputs['cI'], (0, l+1-inputs['cI'].size), mode='constant')
            inputs['cI'][l] = cI

        # magnetic axis Fourier modes
        match = re.search(r'aR\s*=\s*'+num_form, command, re.IGNORECASE)
        if match:
            aR = [float(x) for x in re.findall(
                num_form, match.group(0)) if re.search(r'\d', x)][0]
            axis_idx = np.where(inputs['axis'][:, 0] == n)[0]
            if axis_idx.size == 0:
                axis_idx = np.atleast_1d(inputs['axis'].shape[0])
                inputs['axis'] = np.pad(
                    inputs['axis'], ((0, 1), (0, 0)), mode='constant')
                inputs['axis'][axis_idx[0], 0] = n
            inputs['axis'][axis_idx[0], 1] = aR
        match = re.search(r'aZ\s*=\s*'+num_form, command, re.IGNORECASE)
        if match:
            aZ = [float(x) for x in re.findall(
                num_form, match.group(0)) if re.search(r'\d', x)][0]
            axis_idx = np.where(inputs['axis'][:, 0] == n)[0]
            if axis_idx.size == 0:
                axis_idx = np.atleast_1d(inputs['axis'].shape[0])
                inputs['axis'] = np.pad(
                    inputs['axis'], ((0, 1), (0, 0)), mode='constant')
                inputs['axis'][axis_idx[0], 0] = n
            inputs['axis'][axis_idx[0], 2] = aZ

        # boundary Fourier modes
        match = re.search(r'bR\s*=\s*'+num_form, command, re.IGNORECASE)
        if match:
            bR = [float(x) for x in re.findall(
                num_form, match.group(0)) if re.search(r'\d', x)][0]
            bdry_m = np.where(inputs['bdry'][:, 0] == m)[0]
            bdry_n = np.where(inputs['bdry'][:, 1] == n)[0]
            bdry_idx = bdry_m[np.in1d(bdry_m, bdry_n)]
            if bdry_idx.size == 0:
                bdry_idx = np.atleast_1d(inputs['bdry'].shape[0])
                inputs['bdry'] = np.pad(
                    inputs['bdry'], ((0, 1), (0, 0)), mode='constant')
                inputs['bdry'][bdry_idx[0], 0] = m
                inputs['bdry'][bdry_idx[0], 1] = n
            inputs['bdry'][bdry_idx[0], 2] = bR
        match = re.search(r'bZ\s*=\s*'+num_form, command, re.IGNORECASE)
        if match:
            bZ = [float(x) for x in re.findall(
                num_form, match.group(0)) if re.search(r'\d', x)][0]
            bdry_m = np.where(inputs['bdry'][:, 0] == m)[0]
            bdry_n = np.where(inputs['bdry'][:, 1] == n)[0]
            bdry_idx = bdry_m[np.in1d(bdry_m, bdry_n)]
            if bdry_idx.size == 0:
                bdry_idx = np.atleast_1d(inputs['bdry'].shape[0])
                inputs['bdry'] = np.pad(
                    inputs['bdry'], ((0, 1), (0, 0)), mode='constant')

    # error handling
    if np.any(inputs['Mpol'] == 0):
        raise IOError(TextColors.FAIL +
                      'Mpol is not assigned' + TextColors.ENDC)
    if np.sum(inputs['bdry']) == 0:
        raise IOError(
            TextColors.FAIL + 'Fixed-boundary surface is not assigned' + TextColors.ENDC)
    arrs = ['Mpol', 'Ntor', 'delta_lm', 'Mnodes', 'Nnodes', 'bdry_ratio',
            'pres_ratio', 'zeta_ratio', 'errr_ratio', 'pert_order',
            'ftol', 'xtol', 'gtol', 'nfev']
    arr_len = 0
    for a in arrs:
        arr_len = max(arr_len, len(inputs[a]))
    for a in arrs:
        if inputs[a].size == 1:
            inputs[a] = np.broadcast_to(inputs[a], arr_len, subok=True).copy()
        elif inputs[a].size != arr_len:
            raise IOError(TextColors.FAIL +
                          'Continuation parameter arrays are not proper lengths' + TextColors.ENDC)

    # unsupplied values
    if np.sum(inputs['Mnodes']) == 0:
        inputs['Mnodes'] = np.rint(1.5*inputs['Mpol']).astype(int)
    if np.sum(inputs['Nnodes']) == 0:
        inputs['Nnodes'] = np.rint(1.5*inputs['Ntor']).astype(int)
    if np.sum(inputs['axis']) == 0:
        axis_idx = np.where(inputs['bdry'][:, 0] == 0)[0]
        inputs['axis'] = inputs['bdry'][axis_idx, 1:]
    if None in inputs['delta_lm']:
        default_deltas = {'fringe': 2*inputs['Mpol'],
                          'ansi': inputs['Mpol'],
                          'chevron': inputs['Mpol'],
                          'house': 2*inputs['Mpol']}
        inputs['delta_lm'] = default_deltas[inputs['zern_mode']]

    return inputs



