import sys
import warnings
from termcolor import colored

from desc.io import InputReader


def main(cl_args=sys.argv[1:]):
    """Run the main DESC code from the command line.

    Reads and parses user input from command line, runs the code,
    and prints and plots the resulting equilibrium.

    """
    ir = InputReader(cl_args=cl_args)

    if ir.args.version:
        return

    import desc

    if ir.args.verbose:
        print(desc.BANNER)

    from desc.backend import use_jax
    from desc.equilibrium import EquilibriaFamily
    from desc.plotting import plot_surfaces, plot_section
    import matplotlib.pyplot as plt

    if ir.args.verbose:
        print("Reading input from {}".format(ir.input_path))
        print("Outputs will be written to {}".format(ir.output_path))

    # initialize
    equil_fam = EquilibriaFamily(ir.inputs)
    # check vmec path input
    if ir.args.guess is not None:
        if ir.args.verbose:
            print("Initial guess from {}".format(ir.args.guess))
        equil_fam[0].set_initial_guess(ir.args.guess)
    # solve equilibrium
    equil_fam.solve_continuation(
        verbose=ir.args.verbose, checkpoint_path=ir.output_path
    )

    if ir.args.plot > 1:
        for i, eq in enumerate(equil_fam[:-1]):
            print("Plotting solution at step {}".format(i + 1))
            ax = plot_surfaces(eq)
            plt.show()
            ax = plot_section(eq, "|F|", log=True, norm_F=True)
            plt.show()
    if ir.args.plot > 0:
        print("Plotting final solution")
        ax = plot_surfaces(equil_fam[-1])
        plt.show()
        ax = plot_section(equil_fam[-1], "|F|", log=True, norm_F=True)
        plt.show()


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
