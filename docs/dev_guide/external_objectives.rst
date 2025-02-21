Using external codes
--------------------

The ``ExternalObjective`` class allows external codes to be used within a DESC
optimization. This takes a user-supplied function that does not need to be JAX
transformable, and wraps it with forward finite differences. This is most useful for
cases when the function needs to call external codes that are written in another
language besides Python, or cannot be rewritten with JAX for some other reason. If the
user function can be made JAX transformable, it is recommended to use the
``ObjectiveFromUser`` class instead.

This guide begins with a simple example of how to use a custom function with the
``ExternalObjective``. It then uses the ``TERPSICHORE`` objective as a more realistic
example, which can be used as a template for wrapping other codes. Finally, an example
script is shown for how to run the TERPSICHORE optimization with multiprocessing. (You
must obtain access to the TERPSICHORE source code to run this example, it is not
included with DESC.)

Simple example
--------------

The following is a simple example of a custom function that is not JAX transformable.
It was adapted from the test ``test_external_vs_generic_objectives`` in
``tests/test_examples.py``.

The function must take a single positional argument, which can be either a single
Equilibrium or a list of Equilibria. Additional inputs can be passed as keyword
arguments. In this example, the function returns the three scalar quatities
:math:`\langle\beta\rangle_{vol}`, :math:`\langle\beta\rangle_{pol}`, and
:math:`\langle\beta\rangle_{tor}`. It writes these quantities to a NetCDF file and then
reads them back from the file. This mimics I/O from the common VMEC "wout" format, and
also demonstrates an operation that is not automatic-differentiable.

::

    def beta_from_file(eq, path=""):
        # write data
        file = Dataset(path, mode="w", format="NETCDF3_64BIT_OFFSET")
        data = eq.compute(["<beta>_vol", "<beta_pol>_vol", "<beta_tor>_vol"])
        betatotal = file.createVariable("betatotal", np.float64)
        betatotal[:] = data["<beta>_vol"]
        betapol = file.createVariable("betapol", np.float64)
        betapol[:] = data["<beta_pol>_vol"]
        betator = file.createVariable("betator", np.float64)
        betator[:] = data["<beta_tor>_vol"]
        file.close()
        # read data
        file = Dataset(path, mode="r")
        betatot = float(file.variables["betatotal"][0])
        betapol = float(file.variables["betapol"][0])
        betator = float(file.variables["betator"][0])
        file.close()
        return np.atleast_1d([betatot, betapol, betator])

This function can then be used in an optimization with the ``ExternalObjective`` class.
This objective must be initialized with the Equilibrium to be optimized, ``eq``, and the
following other arguments:

* ``fun``, the external function. In this example it is the function ``beta_from_file``.
* ``dim_f``, the dimension of the output of the function ``fun``. In this example
  ``dim_f=3`` because ``beta_from_file`` returns an array of size 3.
* ``vectorized``, whether ``fun`` expects a single Equilibrium or a list of
  Equilibria. Since the function ``beta_from_file`` takes a single Equilibrium as its
  only positional argument, ``vectorized=False``.

All other input parameters that the external function requires can be passed as keyword
arguments to ``ExternalObjective``. In this example, ``path`` specifies the file name
of the NetCDF file that the function writes/reads. Since ``vectorized=False``, the
function ``beta_from_file`` will be evaluated sequentially when computing the finite
differences (rather than in parallel), so the same file will be overwritten each time
for the different Equilibria.

::

    from desc.objectives import ExternalObjective, ObjectiveFunction

    objective = ObjectiveFunction(
        ExternalObjective(
            eq=eq,
            fun=beta_from_file,
            dim_f=3,
            vectorized=False,
            path="path/to/file.nc",
        )
    )

Note that for this example, the same optimization objective could be accomplished
naitively in DESC with automatic differentiation instead of finite differences as:

::

    from desc.objectives import GenericObjective, ObjectiveFunction

    objective = ObjectiveFunction(
        (
            GenericObjective("<beta>_vol", thing=eq),
            GenericObjective("<beta_pol>_vol", thing=eq),
            GenericObjective("<beta_tor>_vol", thing=eq),
        )
    )


TERPSICHORE objective
---------------------

This section walks through the implementation of wrapping the ideal MHD linear stability
code TERPSICHORE, which is written in FORTRAN. It summarizes how the external function
was written to call TERPSICHORE from DESC, and can be used as a template for wrapping
other codes. The code shown here is abbreviated and slightly modified for brevity, but
the full code can be found in ``desc/experimental/_terpsichore.py``.

The external function in this case is named ``terpsichore``, and takes the following
arguments:

* ``eq``, a list of Equilibria.
* ``processes``, the maximum number of processes to use for multiprocessing.
* ``path``, the path to the directory where the optimization script is being executed.
* ``exec``, the file name of the TERPSICHORE executable.

The ``TERPSICHORE`` objective takes other arguments that have been omitted for
simplicity. The outline of this function is:

1. Create a temporary directory where I/O files will be written.
2. Write the equilibria data to files in the format that TERPSICHORE expects from a VMEC
   input.
3. Write the TERPSICHORE input files for each equilibrium.
4. Run TERPSICHORE with the inputs from steps 2 and 3.
5. Read the instability growth rates from the TERPSICHORE output files for each
   equilibrium.
6. Return the growth rates.

Running TERPSICHORE is relatively slow compared to other computations in DESC. The
bottleneck of an optimization is computing the Jacobian matrix with finite differences,
which scales with the number of optimization degrees of freedom. Evaluating the
TERPSICHORE growth rates for each Equilibrium can be performed in parallel on different
CPU threads using Python multiprocessing. Note that writing the equilibria data in step
2 cannot be easily parallelized, since it involves computations using JAX that has
issues with multiprocessing.

::

    # TERPSICHORE only runs on a CPU, but DESC is optimized to run on a GPU.
    # This decorator will run this function on a CPU, even if other functions are being
    # run on a GPU.
    @execute_on_cpu
    def terpsichore(eq, processes=1, path="", exec=""):
        """TERPSICHORE driver function."""
        # create temporary directory to store I/O files
        tmp_path = os.path.join(path, "tmp-TERPS")
        os.mkdir(tmp_path)

        # write input files for each equilibrium in serial
        # these indices are used to give each equilibrium's I/O files unique file names
        idxs = list(range(len(eq)))  # equilibrium indices
        for k in idxs:
            # create a sub-directory for each equilibrium
            idx_path = os.path.join(tmp_path, str(k))
            os.mkdir(idx_path)
            exec_path = os.path.join(idx_path, exec)
            input_path = os.path.join(idx_path, "input")
            wout_path = os.path.join(idx_path, "wout.nc")
            shutil.copy(os.path.join(path, exec), exec_path)
            _write_wout(eq=eq[k], path=wout_path)  # write equilibrium input data
            _write_terps_input(path=input_path)  # write TERPSICHORE input file

        # run TERPSICHORE on list of equilibria in parallel
        if len(eq) == 1:  # no multiprocessing if only one equilibrium
            result = jnp.atleast_1d(_pool_fun(0, path=tmp_path, exec=exec))
        else:  # use multiprocessing if there are multiple equilibria
            with mp.Pool(processes=min(processes, len(eq))) as pool:
                results = pool.map(
                    functools.partial(_pool_fun, path=tmp_path, exec=exec),
                    idxs,
                )
                pool.close()
                pool.join()
                result = jnp.vstack(results, dtype=float)

        # remove temporary directory and all sub-directories
        shutil.rmtree(tmp_path)

        return result

The function ``_write_wout`` is a simplified version of ``VMECIO.save`` that only saves
the output quantities that TERPSICHORE needs. Avoiding computation of the unnecessary
quantities greatly reduces the overall run time. The function
``_write_terps_input`` writes the TERPSICHORE input file, which is a text file with a
specific format. The details of these two functions are not important for the scope of
this guide.

``_pool_fun`` is the function that is run in parallel for each Equilibrium. It calls
``_run_terps`` (also shown below) to execute the TERPSICHORE Fortran code through a
Python subprocess call, and ``_read_terps_output`` (not shown) to parse the output file
and extract the instability growth rate. If TERPSICHORE fails to execute for any reason
or takes too long to run, a large unstable growth rate is returned.

::

    def _pool_fun(k, path, exec):
        """Run TERPSICHORE and read output for equilibrium with index k."""
        idx_path = os.path.join(path, str(k))
        exec_path = os.path.join(idx_path, exec)
        fort16_path = os.path.join(idx_path, "fort.16")
        input_path = os.path.join(idx_path, "input")
        wout_path = os.path.join(idx_path, "wout.nc")

        try:  # try to run TERPSICHORE and read the growth rate from the output file
            _run_terps(dir=idx_path, exec=exec_path, input=input_path, wout=wout_path)
            output = _read_terps_output(path=fort16_path)
        except RuntimeError:
            output = 1.0  # default value if TERPSICHORE failed to run

        return np.atleast_1d(output)


    def _run_terps(dir, exec, input, wout):
        """Run TERPSICHORE."""
        stdout_path = os.path.join(dir, "stdout.terps")
        stderr_path = os.path.join(dir, "stderr.terps")

        fout = open(stdout_path, "w")
        ferr = open(stderr_path, "w")

        # execute TERPSICHORE
        cmd = exec + " < " + input + " " + wout
        terps_subprocess = subprocess.run(
            cmd, cwd=dir, shell=True, stdout=fout, stderr=ferr
        )

        # not shown: a delay to wait until TERPSICHORE finishes running
        terps_subprocess.terminate()

        fout.close()
        ferr.close()

Finally, the ``TERPSICHORE`` objective function simply inherits from
``ExternalObjective`` and passes ``fun=terpsichore`` as the external function.
``dim_f=1`` because TERPSICHORE is returning a scalar growth rate in this example, and
``vectorized=True`` because the function ``terpsichore`` expects a list of Equilibria
as its only positional argument. (Parts of the full class definition have been omitted
here for simplicity.)

::

    class TERPSICHORE(ExternalObjective):
        """Computes ideal MHD linear stability from calls to the code TERPSICHORE."""

        def __init__(self, eq, processes=1, path="", exec=""):
            super().__init__(
                eq=eq,
                fun=terpsichore,
                dim_f=1,
                vectorized=True,
                processes=processes,
                path=path,
                exec=exec,
            )

Multiprocessing
---------------

Due to complexities of Python multiprocessing, one must guard against spawning unwanted
child processes. The following is a simple example script for performing an optimization
with the ``TERPSICHORE`` objective function. Note that the step size used in the finite
differencing of ``ExternalObjective`` can be controlled with the arguments ``abs_step``
and ``rel_step``. ``processes=os.cpu_count()`` will use the maximum number of CPU
threads that are available.

::

    import os
    import sys

    import multiprocessing as mp
    import numpy as np

    # this ensures that this driver code only runs once, for the main process
    if mp.current_process().name == "MainProcess":
        from desc import set_device

        set_device("gpu")

        from desc.examples import get
        from desc.experimental import TERPSICHORE
        from desc.objectives import (
            ForceBalance,
            FixBoundaryR,
            FixIota,
            FixPressure,
            FixPsi,
            ObjectiveFunction,
        )
        from desc.optimize import Optimizer

        eq = get("W7-X")
        optimizer = Optimizer("proximal-lsq-exact")
        objective = ObjectiveFunction(
            (
                TERPSICHORE(
                    eq=eq,
                    abs_step=1e-2,
                    rel_step=0,
                    processes=os.cpu_count(),
                    path=os.getcwd(),
                    exec="terps_exec.x",
                ),
            ),
        )
        constraints = (
            FixBoundaryR(eq=eq, modes=np.array([[0, 0, 0]])),
            FixIota(eq=eq),
            FixPressure(eq=eq),
            FixPsi(eq=eq),
            ForceBalance(eq=eq),
        )
        [eq], _ = optimizer.optimize(
            things=eq,
            objective=objective,
            constraints=constraints,
        )
