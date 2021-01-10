import pathlib
import sys
import warnings
from termcolor import colored
from desc.input_reader import InputReader


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
            # ID was not valid
            warnings.warn(
                colored('gpuID did not match any found devices, trying default gpu option', 'yellow'))
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
        warnings.warn(colored('No GPU found, falling back to CPU', 'yellow'))
        return jax.devices('cpu')[0]


def main(cl_args=None):
    """Runs the main DESC code from the command line.
    Reads and parses user input from command line, runs the code,
    and prints and plots the resulting equilibrium.
    """

    ir = InputReader(cl_args=cl_args)

    if ir.args.version:
        return

    import desc

    print(desc.BANNER)

    from desc.continuation import solve_eq_continuation
    from desc.backend import use_jax
    from desc.plotting import Plot
    import matplotlib.pyplot as plt

    if use_jax:
        device = get_device(ir.args.gpu)
        print("Using device: " + str(device))
    else:
        device = None

    # solve equilibrium
    equil_fam, timer = solve_eq_continuation(
        ir.inputs, file_name=ir.output_path, device=device)

    if ir.args.plot:
        print('plotting flux surfaces')
        for eq in equil_fam:
            ax = Plot().plot_surfaces(eq)
            plt.show()
            ax = Plot().plot_2d(eq, 'r')
            plt.show()
            ax = Plot().plot_2d(eq, 'log(|F|)')
            plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
