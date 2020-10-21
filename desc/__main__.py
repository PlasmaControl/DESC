import argparse
import pathlib
import sys
import warnings
from desc.continuation import solve_eq_continuation
from desc.plotting import plot_comparison, plot_vmec_comparison, plot_fb_err
from desc.input_output import read_input, output_to_file, read_vmec_output
from desc.backend import use_jax


def get_device(use_gpu=False, gpuID=None):
    """Checks available GPUs and selects the one with the most available memory"""

    import jax
    import nvidia_smi

    if not use_gpu:
        return jax.devices('cpu')[0]

    try:
        gpus = jax.devices('gpu')
        # did the user request a specific GPU?
        if gpuID is not None and gpuID < len(gpus):
            return gpus[gpuID]
        # find all available options and see which has the most space
        else:
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
        warnings.warn('No GPU found, falling back to CPU')
        return jax.devices('cpu')[0]


def parse_args(args):
    """Parses command line arguments
    Args:
        args (list): List of arguments to parse, ie sys.argv
    Returns:
        parsed_args (argparse object): argparse object containing parsed arguments.
    """

    parser = argparse.ArgumentParser(prog='DESC',
                                     description='DESC computes equilibria by solving the force balance equations. '
                                     + 'It can also be used for perturbation analysis and sensitivity studies '
                                     + 'to see how the equilibria change as input parameters are varied.')
    parser.add_argument('input_file', help='Path to input file')
    parser.add_argument('-o', '--output',
                        help='Path to output file. If not specified, defaults to <input_name>.output')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='Plot results after solver finishes')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Do not display any progress information')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Display detailed progress information')
    parser.add_argument('--vmec', help='Path to VMEC data for comparison plot')
    parser.add_argument('--gpu', '-g', action='store_true',
                        help='Use GPU if available. Note that not all of the computation will be done '
                        + 'on the gpu, only the most expensive parts where the I/O efficiency is worth it.')
    parser.add_argument('--gpuID', action='store', default=None,
                        help='device ID of GPU to use (usually 0,1,2 etc). Can be obtained by running '
                        + '`nvidia-smi`. Default is to select the GPU with most available memory.')
    return parser.parse_args(args)


def main(args=sys.argv[1:]):
    """Runs the main DESC code from the command line.
    Reads and parses user input from command line, runs the code,
    and prints and plots the resulting equilibrium.
    """
    args = parse_args(args)

    if use_jax:
        device = get_device(args.gpu, args.gpuID)
        print("Using device: " + str(device))
    else:
        device = None

    in_fname = str(pathlib.Path(args.input_file).resolve())
    out_fname = args.output if args.output else in_fname+'.output'

    print('Reading input from {}'.format(in_fname))
    inputs = read_input(in_fname)

    if args.quiet:
        inputs['verbose'] = 0
    elif args.verbose:
        inputs['verbose'] = 2
    else:
        inputs['verbose'] = 1

    # solve equilibrium
    iterations = solve_eq_continuation(
        inputs, checkpoint_filename=out_fname, device=device)

    # output
#     print('Writing output to {}'.format(out_fname))
#     output_to_file(out_fname, equil)

    if args.plot:

        equil_init = iterations['init']
        equil = iterations['final']

        # plot comparison to initial guess
        plot_comparison(equil_init, equil, 'Initial', 'Solution')

        # plot comparison to VMEC
        if args.vmec:
            vmec_data = read_vmec_output(pathlib.Path(args.vmec).resolve())
            plot_vmec_comparison(vmec_data, equil)

        # plot force balance error
        plot_fb_err(equil, domain='real', normalize='global',
                    log=True, cmap='plasma')


if __name__ == '__main__':
    main(sys.argv[1:])
