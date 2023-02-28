"""Main command line interface to DESC for solving fixed boundary equilibria."""


import logging
import sys

from desc.io import InputReader

logger = logging.getLogger("DESC_logger")


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

    import matplotlib.pyplot as plt

    from desc.equilibrium import EquilibriaFamily
    from desc.plotting import plot_section, plot_surfaces

    logger.info("Reading input from %s" % ir.input_path)
    logger.info("Outputs will be written to %s" % ir.output_path)

    # initialize
    inputs = ir.inputs
    equil_fam = EquilibriaFamily(inputs)
    # check vmec path input
    if ir.args.guess is not None:
        logger.info("Initial guess from %s" % ir.args.guess)
        equil_fam[0].set_initial_guess(ir.args.guess)
    # solve equilibrium
    equil_fam.solve_continuation(
        objective=inputs[0]["objective"],
        optimizer=inputs[0]["optimizer"],
        pert_order=[inp["pert_order"] for inp in inputs],
        ftol=[inp["ftol"] for inp in inputs],
        xtol=[inp["xtol"] for inp in inputs],
        gtol=[inp["gtol"] for inp in inputs],
        nfev=[inp["nfev"] for inp in inputs],
        verbose=ir.args.verbose,
        checkpoint_path=ir.output_path,
    )

    if ir.args.plot > 1:
        for i, eq in enumerate(equil_fam[:-1]):
            logger.info("Plotting solution at step %s" % (i + 1))
            _ = plot_surfaces(eq)
            plt.show()
            _ = plot_section(eq, "|F|", log=True, norm_F=True)
            plt.show()
    if ir.args.plot > 0:
        logger.info("Plotting final solution")
        _ = plot_surfaces(equil_fam[-1])
        plt.show()
        _ = plot_section(equil_fam[-1], "|F|", log=True, norm_F=True)
        plt.show()


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
