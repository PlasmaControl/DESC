import pathlib
import sys
import warnings

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


def main(cl_args=None):
    """Runs the main DESC code from the command line.
    Reads and parses user input from command line, runs the code,
    and prints and plots the resulting equilibrium.
    """

    ir = InputReader(cl_args=cl_args)

    if ir.args.version:
        import desc
        print(desc.__version__)
        return

    import desc

    print(desc.BANNER)

    from desc.continuation import solve_eq_continuation
    from desc.plotting import plot_comparison, plot_vmec_comparison
    #from desc.input_output import read_input, output_to_file
    from desc.backend import use_jax
    from desc.vmec import read_vmec_output, vmec_error

    if use_jax:
        device = get_device(ir.args.gpu)
        print("Using device: " + str(device))
    else:
        device = None

    # solve equilibrium
    equil_fam, timer = solve_eq_continuation(
        ir.inputs, checkpoint_filename=None, device=device)
      # ir.inputs, checkpoint_filename=ir.output_path, device=device)

    if ir.args.plot:

        equil_init = equil_fam[0].initial
        equil = equil_fam[-1]
        print('Plotting flux surfaces, this may take a few moments...')
        # plot comparison to initial guess
        plot_comparison(equil_init, equil, 'Initial', 'Solution')

        # plot comparison to VMEC
        if ir.args.vmec:
            print('Plotting comparison to VMEC, this may take a few moments...')
            vmec_data = read_vmec_output(pathlib.Path(ir.args.vmec).resolve())
            plot_vmec_comparison(vmec_data, equil)
            err = vmec_error(equil, vmec_data)
            print("Average error relative to VMEC solution: {:.3f} meters".format(err))


if __name__ == '__main__':
    main(sys.argv[1:])
