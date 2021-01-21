import numpy as np
import sys
import warnings
from termcolor import colored
from desc.io import InputReader


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
        return jax.devices("cpu")[0]

    try:
        gpus = jax.devices("gpu")
        # did the user request a specific GPU?
        if isinstance(gpuID, int) and gpuID < len(gpus):
            return gpus[gpuID]
        if isinstance(gpuID, int):
            # ID was not valid
            warnings.warn(
                colored(
                    "gpuID did not match any found devices, trying default gpu option",
                    "yellow",
                )
            )
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
        warnings.warn(colored("No GPU found, falling back to CPU", "yellow"))
        return jax.devices("cpu")[0]


def main(cl_args=None):
    """Runs the main DESC code from the command line.
    Reads and parses user input from command line, runs the code,
    and prints and plots the resulting equilibrium.
    """

    ir = InputReader(cl_args=cl_args)

    if ir.args.version:
        return

    import desc

    if ir.args.verbose:
        print(desc.BANNER)

    from desc.equilibrium import EquilibriaFamily, Equilibrium
    from desc.vmec import VMECIO
    from desc.backend import use_jax
    from desc.plotting import Plot
    import matplotlib.pyplot as plt

    if use_jax:
        device = get_device(ir.args.gpu)
        if ir.args.verbose:
            print("Using device: " + str(device))
    else:
        device = None

    # initialize
    equil_fam = EquilibriaFamily(ir.inputs)
    # check vmec path input
    if ir.args.vmec is not None:
        equil_fam[0] = VMECIO.load(
            ir.args.vmec,
            L=ir.inputs[0]["L"],
            M=ir.inputs[0]["M"],
            N=ir.inputs[0]["N"],
            index=ir.inputs[0]["zern_mode"],
        )
        equil_fam[0].inputs = ir.inputs[0]
        equil_fam[0].objective = ir.inputs[0]["errr_mode"]
        equil_fam[0].optimizer = ir.inputs[0]["optim_method"]

    # solve equilibrium
    equil_fam.solve_continuation(
        verbose=ir.args.verbose, checkpoint_path=ir.output_path, device=device
    )

    if ir.args.plot > 1:
        print("Plotting initial guess")
        print("Axis location: {}".format(equil_fam[0].initial.compute_axis_location()))
        ax = Plot().plot_surfaces(equil_fam[0].initial)
        plt.show()
        ax = Plot().plot_2d(equil_fam[0].initial, "log(|F|)")
        plt.show()
    if ir.args.plot > 2:
        for i, eq in enumerate(equil_fam[:-1]):
            print("Plotting solution at step {}".format(i + 1))
            print("Axis location: {}".format(eq.compute_axis_location()))
            ax = Plot().plot_surfaces(eq)
            plt.show()
            ax = Plot().plot_2d(eq, "log(|F|)")
            plt.show()
    if ir.args.plot > 0:
        print("Plotting final solution")
        print("Axis location: {}".format(equil_fam[-1].compute_axis_location()))
        ax = Plot().plot_surfaces(equil_fam[-1])
        plt.show()
        ax = Plot().plot_2d(equil_fam[-1], "log(|F|)")
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
