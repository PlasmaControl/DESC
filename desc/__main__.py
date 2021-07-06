import sys
import warnings
from termcolor import colored

from desc.io import InputReader


def main(cl_args=sys.argv[1:]):
    """Run. the main DESC code from the command line.

    Reads and parses user input from command line, runs the code,
    and prints and plots the resulting equilibrium.

    """
    ir = InputReader(cl_args=cl_args)

    if ir.args.version:
        return

    import desc

    if ir.args.verbose:
        print(desc.BANNER)
        print("Reading input from {}".format(ir.input_path))
        print("Outputs will be written to {}".format(ir.output_path))

    from desc.backend import use_jax
    from desc.equilibrium import EquilibriaFamily
    from desc.vmec import VMECIO

    from desc.plotting import plot_surfaces, plot_section
    import matplotlib.pyplot as plt

    # initialize
    equil_fam = EquilibriaFamily(ir.inputs)
    # check vmec path input
    if ir.args.vmec is not None:
        equil_fam[0] = VMECIO.load(
            ir.args.vmec,
            L=ir.inputs[0]["L"],
            M=ir.inputs[0]["M"],
            N=ir.inputs[0]["N"],
            spectral_indexing=ir.inputs[0]["spectral_indexing"],
        )
        equil_fam[0].inputs = ir.inputs[0]
        equil_fam[0].objective = ir.inputs[0]["objective"]
        equil_fam[0].optimizer = ir.inputs[0]["optimizer"]

    # solve equilibrium
    equil_fam.solve_continuation(
        verbose=ir.args.verbose, checkpoint_path=ir.output_path
    )

    if ir.args.plot > 1:
        print("Plotting initial guess")
        ax = plot_surfaces(equil_fam[0].initial)
        plt.show()
        ax = plot_section(equil_fam[0].initial, "|F|", log=True, norm_F=True)
        plt.show()
    if ir.args.plot > 2:
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
