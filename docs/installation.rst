============
Installation
============

Follow these instructions to install DESC and its dependencies.

We only test installation with conda environments, so your mileage may vary with other managers.
For information on using conda, see `here <https://conda.io/projects/conda/en/latest/user-guide/getting-started.html#starting-conda>`__.

.. attention::

    DESC requires ``python>=3.10``. If you have ``python2`` also locally installed, replace all ``pip`` commands with ``pip3`` and all ``python`` commands with ``python3`` to ensure the correct python version is used.

.. attention::

    We do not support DESC on Windows OS.
    There have been instances where numerical discrepancies on Windows cause incorrect results.
    For these reasons, we recommend using Windows Subsystem for Linux (WSL) if you have a Windows machine.
    For instructions to install WSL see `here <https://learn.microsoft.com/en-us/windows/wsl/install>`__.
    To use WSL in VS code see `here <https://code.visualstudio.com/docs/remote/wsl>`__.


On Your Local Machine
*********************

.. tab-set::

    .. tab-item:: CPU

        .. dropdown:: PyPI

            .. code-block:: sh

                pip install desc-opt

        .. dropdown:: GitHub

            .. code-block:: sh

                git clone https://github.com/PlasmaControl/DESC.git
                cd DESC
                conda create --name desc-env 'python>=3.10, <=3.13'
                conda activate desc-env
                pip install --editable .

            You may optionally install developer requirements if you want to run tests.

            .. code-block:: sh

                pip install -r devtools/dev-requirements.txt

            These instructions were tested to work on an M1 Macbook device on May 3, 2023.

        .. dropdown:: uv

            One could use `uv <https://docs.astral.sh/uv>`_, a new python package management tool, instead of pip.
            For a project that modifies DESC and also uses it to perform analysis,
            it can be nice to separate the DESC folder from the project's data, scripts, jupyter notebooks, etc.
            This will show how to set up a new ``uv`` project called ``myproject`` with DESC as an editable dependency (Either on local machine or on the cluster, this method can work with both),
            and with the ability to use DESC in a jupyter notebook.

            .. code-block:: sh

                # download UV; it installs into .local/bin
                curl -LsSf https://astral.sh/uv/install.sh | sh

                # the depth=1 option reduces the quantity of older data downloaded
                git clone --depth=1 git@github.com:PlasmaControl/DESC.git

                # initialize a project
                uv init myproject
                cd myproject

                # add dependencies
                uv add --editable "../DESC"

                # test the installation
                uv run python

                >>> from desc.backend import print_backend_info
                >>> print_backend_info()

                # Jupyter Notebooks
                # ----------------
                # install a jupyter kernel
                uv add --dev ipykernel
                uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=myproject

                # run jupyter
                uv run --with jupyter jupyter lab

    .. tab-item:: CPU+GPU (without FINUFFT)

        For GPU support, you must install the JAX library as discussed in `JAX installation docs <https://github.com/google/jax#installation>`__.
        For example, below are the instructions to install on compatible devices with an NVIDIA GPU.
        These instructions do not install GPU support for ``jax-finufft`` which is used in ``Bounce2D`` computations
        such as ``GammaC`` and ``EffectiveRipple`` objectives. Set ``nufft_eps=0`` in these objectives if necessary.

        .. code-block:: sh

            git clone https://github.com/PlasmaControl/DESC.git
            cd DESC
            conda create --name desc-env 'python>=3.10, <=3.13'
            conda activate desc-env
            sed -i '1 s/^jax/jax[cuda12]/' requirements.txt
            pip install --editable .

        Note that on BSD systems, the ``sed`` command that replaces ``jax`` with ``jax[cuda12]``
        in the ``requirements.txt`` file is ``sed -i '' '1 s/^jax/jax[cuda12]/' requirements.txt``.

        You may optionally install developer requirements if you want to run tests.

        .. code-block:: sh

            pip install -r devtools/dev-requirements.txt

    .. tab-item:: CPU+GPU (with FINUFFT)

        .. code-block:: sh

            git clone https://github.com/PlasmaControl/DESC.git
            cd DESC
            conda create --name desc-env -c conda-forge 'python>=3.10, <=3.13' 'fftw' 'gxx<12'
            conda activate desc-env

            sed -i '1 s/^jax/jax[cuda12]/' requirements.txt
            sed -i '/^jax-finufft/d' requirements.txt

            export CMAKE_PREFIX_PATH=$CONDA_PREFIX:$CMAKE_PREFIX_PATH
            pip install --editable .
            pip install -Ccmake.define.JAX_FINUFFT_USE_CUDA=ON --no-binary=jax-finufft 'jax-finufft >= 1.1.0'

        Note that on BSD systems, the ``sed`` command that replaces ``jax`` with ``jax[cuda12]``
        in the ``requirements.txt`` file is ``sed -i '' '1 s/^jax/jax[cuda12]/' requirements.txt``.

        You may optionally install developer requirements if you want to run tests.

        .. code-block:: sh

            pip install -r devtools/dev-requirements.txt


On Most Linux Computing Clusters
********************************

.. tab-set::

    .. tab-item:: CPU

        .. dropdown:: PyPI

            .. code-block:: sh

                pip install desc-opt

        .. dropdown:: GitHub

            On computing clusters one typically types ``module load anaconda`` to use conda.
            Some clusters require specifying the version of anaconda too.

            .. code-block:: sh

                module load anaconda

            .. code-block:: sh

                git clone https://github.com/PlasmaControl/DESC.git
                cd DESC
                conda create --name desc-env 'python>=3.10, <=3.13'
                conda activate desc-env
                pip install --editable .

            You may optionally install developer requirements if you want to run tests.

            .. code-block:: sh

                pip install -r devtools/dev-requirements.txt

    .. tab-item:: CPU+GPU

        For GPU support, you must install the JAX library as discussed in `JAX installation docs <https://github.com/google/jax#installation>`__.
        We will show instructions that work for the clusters we have tested.
        Most of these clusters are ``x86_64`` architectures.
        You may try the instructions for the cluster most resembling your own, or see if your cluster has
        JAX GPU installation instructions, as that is the main cause for installation differences.

        .. attention::

            DESC does not always guarantee support for the latest version of JAX (which does not have a stable 1.0 release yet).
            Older versions of ``jax`` may need to be installed, which may in turn require lower versions of ``jaxlib``, as well as CUDA and CuDNN.

        .. dropdown:: Perlmutter (NERSC)

            These instructions were verified to work on the Perlmutter supercomputer at NERSC on Feb 11, 2026
            for both CPU and GPU runs.
            These instructions do not install GPU support for ``jax-finufft`` which is used in ``Bounce2D`` computations
            such as ``GammaC`` and ``EffectiveRipple`` objectives. Set ``nufft_eps=0`` in these objectives if necessary.

            .. code-block:: sh

                module load conda

            .. code-block:: sh

                conda create --name desc-env python=3.12
                conda activate desc-env

            Now clone the DESC repository and enter the DESC directory

            .. code-block:: sh

                git clone https://github.com/PlasmaControl/DESC.git
                cd DESC

                pip install --no-cache-dir -r devtools/dev-requirements.txt
                pip install --no-cache-dir --editable .
                pip install --no-cache-dir "jax[cuda12]==0.6.2"

            The ``--no-cache-dir`` avoids conflicts with existing DESC environments or other software that use CUDA on your system.

            Before running a DESC script, you may also need to execute ``unset LD_LIBRARY_PATH`` either in your interactive node (for interactive jobs)
            or in your SLURM script (for submitted jobs).

        .. dropdown:: Della and Stellar Clusters (Princeton)

            We base our instructions below off of `this tutorial <https://github.com/PrincetonUniversity/intro_ml_libs/tree/master/jax>`__.
            If this does not work, please check the link to install JAX with the most recent recommendations from the Princeton computing services.

            These instructions were verified to work on the Della and Stellar clusters at Princeton on 2025 September 9.

            .. code-block:: sh

                module load anaconda3/2024.10
                module load cudatoolkit/12.9

                git clone https://github.com/PlasmaControl/DESC.git
                cd DESC
                conda create --name desc-env -c conda-forge 'python=3.12' 'fftw' 'gxx<12'
                conda activate desc-env

                sed -i '1 s/^jax/jax[cuda12]/' requirements.txt
                sed -i '/^jax-finufft/d' requirements.txt

                export CMAKE_PREFIX_PATH=$CONDA_PREFIX:$CMAKE_PREFIX_PATH
                pip install --editable .
                pip install -Ccmake.define.JAX_FINUFFT_USE_CUDA=ON --no-binary=jax-finufft 'jax-finufft >= 1.1.0'

            Note that on BSD systems, the ``sed`` command that replaces ``jax`` with ``jax[cuda12]``
            in the ``requirements.txt`` file is ``sed -i '' '1 s/^jax/jax[cuda12]/' requirements.txt``.

            You may optionally install developer requirements if you want to run tests.

            .. code-block:: sh

                pip install -r devtools/dev-requirements.txt


        .. dropdown:: RAVEN (IPP, Germany)

            These instructions were verified to work on the RAVEN cluster at IPP on Aug 18, 2024.
            They do not install FINUFFT with GPU support.

            .. code-block:: sh

                module load anaconda/3/2023.03
                CONDA_OVERRIDE_CUDA="12.2" conda create --name desc-env "jax==0.4.23" "jaxlib==0.4.23=cuda12*" -c conda-forge
                conda activate desc-env

                git clone https://github.com/PlasmaControl/DESC
                cd DESC

            Top pin the allowed ``scipy`` version as follows by editing the ``requirements.txt`` file in the current directory.

            .. code-block:: sh

                scipy >= 1.7.0, <= 1.11.3

            Now install DESC.

            .. code-block:: sh

                pip install --editable .

            You may optionally install developer requirements if you want to run tests.

            .. code-block:: sh

                pip install -r devtools/dev-requirements.txt


Verifying your Installation
***************************

To verify your installation works, try the following.

.. tab-set::

    .. tab-item:: CPU

        The following command should show an output stating the DESC version, the JAX version, and your device.

        .. code-block:: python

            from desc.backend import print_backend_info
            print_backend_info()

        You can try running an example equilibrium solve.
        (The filepath shown here is from the ``DESC`` folder if you have cloned the git repo. Otherwise the file can be downloaded `here <https://github.com/PlasmaControl/DESC/blob/master/desc/examples/SOLOVEV>`__.)

        .. code-block:: sh

            python -m desc -vv desc/examples/SOLOVEV

    .. tab-item:: CPU+GPU

        The following command should show an output stating the DESC version, the JAX version, and your device.

        .. code-block:: python

            from desc import set_device
            set_device('gpu')

            from desc.backend import print_backend_info
            print_backend_info()

        You can try running an example equilibrium solve.
        (The filepath shown here is from the ``DESC`` folder if you have cloned the git repo. Otherwise the file can be downloaded `here <https://github.com/PlasmaControl/DESC/blob/master/desc/examples/SOLOVEV>`__.)

        .. code-block:: sh

            python -m desc -vv desc/examples/SOLOVEV -g

        If you installed DESC with FINUFFT on GPU, the following code should complete without error.

        .. code-block:: python

            from desc import set_device
            set_device("gpu")

            from desc.examples import get
            from desc.objectives import ObjectiveFunction, GammaC

            obj = ObjectiveFunction(GammaC(get("W7-X"), num_transit=1, num_pitch=1))
            obj.build()
            x = obj.x()
            obj.compute_scaled_error(x).block_until_ready()


Troubleshooting
***************
We list common problems and their possible solutions.
If you encounter other problems, please `make an issue on Github <https://github.com/PlasmaControl/DESC/issues>`__ and we will help.

.. tip::

    **Problem**: My installation yields the error :code:`ModuleNotFoundError: No module named 'desc'`.

    **Solution**:
    First ensure you have activated the conda environment where DESC is installed (``conda activate desc-env``).
    If the issue persists, it is possible that DESC has not been added to your ``PYTHONPATH``.
    Try adding the DESC directory to your ``PYTHONPATH`` manually by adding ``export PYTHONPATH="$PYTHONPATH:path/to/DESC"`` to the end of your ``~/.bashrc`` (or other shell configuration) file.
    Note that ``/path/to/DESC`` should be replaced with the actual path to the DESC directory on your machine.
    You will also need to run ``source ~/.bashrc`` after making that change to ensure that your path updates for your current terminal session.

.. tip::

    **Problem**: My installation yields the error ``ModuleNotFoundError: No module named 'termcolor'`` (or another module which is not ``desc``).

    **Solution**:
    Ensure you have activated the conda environment where DESC is installed (``conda activate desc-env``).

.. tip::

    **Problem**: Attempts to install yield ``ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behavior is the source of the following dependency conflicts.
    desc-opt ...`` with a list of incompatibilities.

    **Solution**:
    This may be due to another version of DESC or ``jax`` that is installed in the conda ``base`` environment.
    Try deleting the ``DESC`` folder, ensuring that ``pip list`` in the conda ``base`` environment no longer lists ``desc-opt`` or ``jax``, then repeating the installation.

.. tip::

    **Problem**: I am using JAX version 0.6.1 and getting errors like ``XlaRuntimeError: INTERNAL: cuSolver internal error``

    **Solution**:
    It is recommended to upgrade JAX to a newer version where these issues are resolved.
    If you must use version 0.6.1, then you must install the following package.

    .. code-block:: sh

        pip install 'nvidia-cublas-cu12==12.9.0.13'

.. tip::

    **Problem**: Using ``pytest`` to run tests leads to import errors `as discussed here <https://github.com/PlasmaControl/DESC/issues/1859>`__.

    **Solution**:
    This issue occurs because ``pip`` is an imperfect package manager, and the packages
    it installs have a tendency to leak out of the environment when ``pip`` thinks
    it can cache files globally to share among local environments.
    One way to resolve the issue is to prepend ``python -m`` to any command with ``pytest``.
    Alternatively one can fix the broken ``pytest`` as follows.
    Since ``pytest`` has leaked out of the environment, first remove it globally.
    If you use ``conda`` it should suffice to remove it from the ``base`` environment, then install
    in the local environment as follows.

    .. code-block:: sh

        conda deactivate
        conda activate base
        pip uninstall pytest
        conda activate desc-env
        pip install pytest
