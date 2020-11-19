import argparse
import pathlib
import sys
import warnings
import os


def get_device(gpuID=False):
    """Checks available GPUs and selects the one with the most available memory

    Parameters
    ----------
    gpuID: bool or int
        whether to use GPU, or the device ID of a specific GPU to use. If False,
        use only CPU. If True, attempts to find the GPU with most available memory.

    Returns
    -------
    device : jax.device
        handle to gpu or cpu device selected

    """

    import jax

    if gpuID is False:
        return jax.devices('cpu')[0]

    try:
        gpus = jax.devices('gpu')
        # did the user request a specific GPU?
        if isinstance(gpuID, int) and gpuID < len(gpus):
            return gpus[gpuID]
        if isinstance(gpuID, int):
            from desc.backend import TextColors
            # ID was not valid
            warnings.warn(
                TextColors.WARNING + 'gpuID did not match any found devices, trying default gpu option' + TextColors.ENDC)
        # find all available options and see which has the most space
        import nvidia_smi
        nvidia_smi.nvmlInit()
        maxmem = 0
        gpu = gpus[0]
        for i in range(len(gpus)):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            if info.free > maxmem:
                maxmem = info.free
                gpu = gpus[i]

        nvidia_smi.nvmlShutdown()
        return gpu

    except:
        from desc.backend import TextColors
        warnings.warn(TextColors.WARNING +
                      'No GPU found, falling back to CPU' + TextColors.ENDC)
        return jax.devices('cpu')[0]


def get_parser():
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


def main(args=sys.argv[1:]):
    """Runs the main DESC code from the command line.
    Reads and parses user input from command line, runs the code,
    and prints and plots the resulting equilibrium.
    """
    parser = get_parser()
    args = parser.parse_args(args)

    if args.version:
        import desc
        print(desc.__version__)
        return

    if len(args.input_file) == 0:
        print('Input file path must be specified')
        return

    if args.numpy:
        os.environ['DESC_USE_NUMPY'] = 'True'
    else:
        os.environ['DESC_USE_NUMPY'] = ''

    import desc

    print(desc.BANNER)

    from desc.continuation import solve_eq_continuation
    from desc.plotting import plot_comparison, plot_vmec_comparison, plot_fb_err
    from desc.input_output import read_input, output_to_file
    from desc.backend import use_jax
    from desc.vmec import read_vmec_output, vmec_error, convert_vmec_to_desc

    if use_jax:
        device = get_device(args.gpu)
        print("Using device: " + str(device))
    else:
        device = None

    in_fname = str(pathlib.Path(args.input_file[0]).resolve())
    out_fname = args.output if args.output else in_fname+'.output'

    print("Reading input from {}".format(in_fname))
    inputs = read_input(in_fname)
    print("Output will be written to {}".format(out_fname))

    if args.quiet:
        inputs['verbose'] = 0
    elif args.verbose:
        inputs['verbose'] = 2
    else:
        inputs['verbose'] = 1

    # solve equilibrium
    iterations, timer = solve_eq_continuation(
        inputs, checkpoint_filename=out_fname, device=device)

    if args.plot:

        equil_init = iterations['init']
        equil = iterations['final']
        print('Plotting flux surfaces, this may take a few moments...')
        # plot comparison to initial guess
        plot_comparison(equil_init, equil, 'Initial', 'Solution')

        # plot comparison to VMEC
        if args.vmec:
            print('Plotting comparison to VMEC, this may take a few moments...')
            vmec_data = read_vmec_output(pathlib.Path(args.vmec).resolve())
            plot_vmec_comparison(vmec_data, equil)
            vmec_equil = convert_vmec_to_desc(vmec_data, equil['zern_idx'], equil['lambda_idx'])
            plot_comparison(equil, vmec_equil, 'DESC', 'VMEC')
            err = vmec_error(equil, vmec_data, Npol=8, Ntor=8)
            print("Error relative to VMEC solution: {} mm".format(err*1e3))

        # plot force balance error
        print('Plotting force balance error, this may take a few moments...')
        plot_fb_err(equil, domain='real', normalize='global',
                    log=True, cmap='plasma')


if __name__ == '__main__':
    main(sys.argv[1:])
