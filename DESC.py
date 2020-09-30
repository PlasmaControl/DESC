from continuation import solve_eq_continuation
from plotting import plot_comparison, plot_vmec_comparison, plot_fb_err
from input_output import read_input, output_to_file, read_vmec_output
import argparse
import pathlib
import sys


def parse_args(args):
    """Parses command line arguments
    Args:
        args (list): List of arguments to parse, ie sys.argv
    Returns:
        parsed_args (argparse object): argparse object containing parsed arguments.
    """

    parser = argparse.ArgumentParser(prog='DESC',
        description="""The DESC code computes equilibria by solving the force balance equations. 
        It can also be used for perturbation analysis and sensitivity studies 
        to see how the equilibria change as input parameters are varied.""")
    parser.add_argument("input_file", help="path to input file")
    parser.add_argument("-o", "--output", help="output path and filename. If not specified, defaults to <input_name>.output")
    parser.add_argument("-p", "--plot", action='store_true',
                        help="""Plot results after solver finishes""")
    parser.add_argument("--vmec", help='path to VMEC data for comparison plot')
#     parser.add_argument("-t", "--num_threads", type=int,
#                         help="""number of threads to use. If not specified,
#                         defaults to what is in config.""", metavar='')

#     group = parser.add_mutually_exclusive_group()
#     group.add_argument("-q", "--quiet", action="store_true",
#                        help="hide progress display window")
#     group.add_argument("-d", "--display", action='store_true',
#                        help="show progress display window")
    return parser.parse_args(args)


def main(args=sys.argv[1:]):
    """Runs the main DESC code from the command line.
    Reads and parses user input from command line, runs the code,
    and prints and plots the resulting equilibrium.
    """    
    args = parse_args(args)
    
    in_fname = str(pathlib.Path(args.input_file).resolve())
    print('Reading input from {}'.format(in_fname))
    inputs = read_input(in_fname)
    out_fname = args.output if args.output else inputs['out_fname']
    
    # solve equilibrium
    equil_init,equil = solve_eq_continuation(inputs)

    # output
    print('Writing output to {}'.format(out_fname))
    output_to_file(out_fname,equil)

    if args.plot:
        # plot comparison to initial guess
        plot_comparison(equil_init,equil,'Initial','Solution')

        # plot comparison to VMEC
        if args.vmec:
            vmec_data = read_vmec_output(pathlib.Path(args.vmec).resolve())
            plot_vmec_comparison(vmec_data,equil)

        # plot force balance error
        plot_fb_err(equil,domain='real',normalize='global',log=True,cmap='plasma')

if __name__ == '__main__':
    main(sys.argv[1:])