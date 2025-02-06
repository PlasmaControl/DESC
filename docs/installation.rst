============
Installation
============

Follow these instructions to install DESC and its dependencies.
Note that most of the installation options here do not install JAX with GPU support.
We do include installation instructions to install JAX with GPU support on some computing clusters that we have tested.
In general, to install JAX with GPU support, please refer to the `JAX installation docs <https://github.com/google/jax#installation>`__.

For information on using conda, see `here <https://conda.io/projects/conda/en/latest/user-guide/getting-started.html#starting-conda>`__.
Other package managers like venv could be used instead of conda, we have just chosen conda as our package manager of choice, and only test with conda environments, so your mileage may vary with other managers.

**NOTE: DESC requires python>=3.9.**
**If you have python2 also locally installed, replace all `pip` commands with `pip3` and all `python` commands with `python3` to ensure the correct python version is used.**

**NOTE: If you are on Windows, consider using the Windows Subsystem for Linux (WSL) to install DESC.**

We don't test or support DESC on Windows OS, and there have been some instances that numerical discrepancies on Windows can cause failures or wrong results. For these reasons, we recommend using WSL if you have a Windows machine. For instructions on how to install WSL see `here <https://learn.microsoft.com/en-us/windows/wsl/install>`__. For using WSL in VS Code see `here <https://code.visualstudio.com/docs/remote/wsl>`__.


On Your Local Machine
*********************

**Install from PyPI**

.. code-block:: console

    pip install desc-opt

**Or from GitHub (for development builds)**

First download the repository from GitHub.

.. code-block:: sh

    git clone https://github.com/PlasmaControl/DESC.git
    cd DESC

Now use pip to install packages (this will only install DESC + JAX with CPU capabilities, NOT GPU)

`tested to work on M1 Macbook on May 3, 2023`

.. code-block:: sh

    conda create --name desc-env 'python>=3.9, <=3.12'
    conda activate desc-env
    pip install --editable .
    # optionally install developer requirements (if you want to run tests)
    pip install -r devtools/dev-requirements.txt


On Most Linux Computing Clusters
********************************

These examples use conda environments.
On computing clusters you must ensure to `module load anaconda` in order to use conda (or in some clusters, you must specify the version of anaconda module you want).


With CPU support only
---------------------

**Install from PyPI**

.. code-block:: console

    pip install desc-opt

**Or from GitHub (for development builds)**

First download the repository from GitHub.

.. code-block:: sh

    git clone https://github.com/PlasmaControl/DESC.git
    cd DESC
    # load your python module
    module load anaconda  # this command may vary depending on cluster

Now use pip to install packages (this will only install DESC + JAX with CPU capabilities, NOT GPU)

.. code-block:: sh

    conda create --name desc-env 'python>=3.9, <=3.12'
    conda activate desc-env
    pip install --editable .
    # optionally install developer requirements (if you want to run tests)
    pip install -r devtools/dev-requirements.txt


With CPU+GPU support
--------------------

We will show the installation instructions that work for the clusters we've tested.
If your cluster is not shown, try the installation for the cluster most resembling your own, or see if your cluster has
specific JAX GPU installation instructions, as that is the main installation difference between clusters.
(note, most of these clusters below are `x86_64` architectures, see the `JAX installation docs <https://github.com/google/jax#installation>`__ for more info if you have a different architecture ).

**Note that DESC does not always test on or guarantee support of the latest version of JAX (which does not have a stable 1.0 release yet), and thus older versions of GPU-accelerated versions of JAX may need to be installed, which may in turn require lower versions of JaxLib, as well as CUDA and CuDNN.**


Perlmutter (NERSC)
++++++++++++++++++++++++++++++
These instructions were tested and confirmed to work on the Perlmutter supercomputer at NERSC on December 17, 2024.

Set up the correct cuda environment for jax installation

.. code-block:: sh

    module load cudatoolkit/12.4
    module load cudnn/8.9.3_cuda12
    module load python/3.11

Check that you have loaded these modules

.. code-block:: sh

    module list

Create a conda environment for DESC (`following these instructions <https://docs.nersc.gov/development/languages/python/using-python-perlmutter/#jax>`__ )

.. code-block:: sh

    conda create -n desc-env python=3.11
    conda activate desc-env
    pip install --upgrade "jax[cuda12]"

Clone and install DESC

.. code-block:: sh

    git clone https://github.com/PlasmaControl/DESC.git
    cd DESC
    # installation for users
    pip install --editable .
    # optionally install developer requirements (if you want to run tests)
    pip install -r devtools/dev-requirements.txt


Della and Stellar Clusters (Princeton)
++++++++++++++++++++++++++++++++++++++

First, install JAX for the latest version of `jaxlib` available on the Princeton clusters.

We base our instructions below off of `this tutorial <https://github.com/PrincetonUniversity/intro_ml_libs/tree/master/jax>`__, if the below instructions do not work please
check the link to install JAX with the most up-to-date recommendations from the Princeton computing services. We first will install DESC as usual, then we will install the
version of the gpu-compatible JAX.

.. code-block:: sh

    conda create --name desc-env python=3.12 -y
    conda activate desc-env
    git clone https://github.com/PlasmaControl/DESC.git
    cd DESC
    # install DESC
    pip install --editable .
    # optionally install developer requirements (if you want to run tests)
    pip install -r devtools/dev-requirements.txt
    # finally, install the gpu-compatible JAX that matches the version needed by the DESC requirements
    # It is important to NOT use the --upgrade or -U flag here! otherwise you may get incompatible JAX versions
    pip install "jax[cuda12]"

Tested and confirmed to work on the Della and Stellar clusters at Princeton as of January 30, 2025.


RAVEN (IPP, Germany)
++++++++++++++++++++++++++++++
These instructions were tested and confirmed to work on the RAVEN cluster at IPP on Aug 18, 2024

Create a conda environment for DESC

.. code-block:: sh

    module load anaconda/3/2023.03
    CONDA_OVERRIDE_CUDA="12.2" conda create --name desc-env "jax==0.4.23" "jaxlib==0.4.23=cuda12*" -c conda-forge
    conda activate desc-env

Clone DESC

.. code-block:: sh

    git clone https://github.com/PlasmaControl/DESC
    cd DESC

In the requirements.txt file, change the scipy version from

.. code-block:: sh

    scipy >= 1.7.0, < 2.0.0

to

.. code-block:: sh

    scipy >= 1.7.0, <= 1.11.3

Install DESC

.. code-block:: sh

    # installation for users
    pip install --editable .
    # optionally install developer requirements (if you want to run tests)
    pip install -r devtools/dev-requirements.txt


On Clusters with IBM Power Architecture
***************************************

If pre-built JAX binaries are not available, you will first need to build JAX from source.
More info can be found here: https://jax.readthedocs.io/en/latest/developer.html

These instructions were tested and confirmed to work on the Traverse supercomputer at Princeton as of Nov. 6, 2023.

NOTE: You must use an older version of DESC in order to use Traverse, as there are some compatibility issues with JAX and the architecture.
Commit `a2fe711ffa3f` (an older version of the `master` branch) was tested to work fine on Traverse with these instructions.

.. code-block:: sh

    git clone https://github.com/PlasmaControl/DESC.git
    cd DESC

    module load anaconda3/2020.11 cudatoolkit/11.1 cudnn/cuda-11.1/8.0.4

    conda create --name desc-env python=3.10
    conda activate desc-env
    # install what you can of the requirements with conda, ends up being all but jax, jaxlib and nvgpu
    conda install colorama "h5py>=3.0.0" "matplotlib>=3.3.0,<=3.6.0,!=3.4.3" "mpmath>=1.0.0" "netcdf4>=1.5.4" "numpy>=1.20.0,<1.25.0" psutil "scipy>=1.5.0,<1.11.0" termcolor
    pip install nvgpu

Build and install JAX with GPU support:

.. code-block:: sh

    cd ..
    git clone https://github.com/google/jax.git
    cd jax
    # last commit of JAX that we got to work with Traverse
    git checkout 6c08702489b33f6c51d5cf0ccadc45e997ab406e

    python build/build.py --enable_cuda --cuda_path /usr/local/cuda-11.1 --cuda_version=11.1 --cudnn_version=8.0.4 --cudnn_path /usr/local/cudnn/cuda-11.1/8.0.4 --noenable_mkl_dnn --bazel_path /usr/bin/bazel --target_cpu=ppc
    pip install dist/*.whl
    pip install .

Add DESC to your Python path:

.. code-block:: sh

    cd ../DESC
    pip install --no-deps --editable .


Checking your Installation
**************************

To check that you have properly installed DESC and its dependencies, try the following:

.. code-block:: python

    python
    >>> from desc import set_device  # only needed if running on a GPU
    >>> set_device('gpu')  # only needed if running on a GPU
    >>> from desc.backend import print_backend_info
    >>> print_backend_info()

You should see an output stating the DESC version, the JAX version, and your device (CPU or GPU).

You can also try running an example input file (filepath shown here is from the ``DESC`` folder, if you have cloned the git repo, otherwise the file can be found and downloaded `here <https://github.com/PlasmaControl/DESC/blob/master/desc/examples/SOLOVEV>`__):

.. code-block:: console

    python -m desc -vv desc/examples/SOLOVEV


Troubleshooting
***************
We list here some common problems encountered during installation and their possible solutions.
If you encounter issues during installation, please `leave us an issue on Github <https://github.com/PlasmaControl/DESC/issues>`__ and we will try our best to help!

**Problem**: I've installed DESC, but when I check my installation I get an error :code:`ModuleNotFoundError: No module named 'desc'`.

**Solution**:

This may be caused by DESC not being on your PYTHONPATH, or your environment containing DESC not being activated.

Try adding the DESC directory to your PYTHONPATH manually by adding the line ``export PYTHONPATH="$PYTHONPATH:path/to/DESC"`` (where ``/path/to/DESC`` is the path to the DESC folder on your machine) to the end of your ``~/.bashrc`` (or other shell configuration) file. You will also need to run ``source ~/.bashrc`` after making the change to ensure that your path updates properly for your current terminal session.

Try ensuring you've activated the conda environment that DESC is in ( ``conda activate desc-env`` ), then retry using DESC.

**Problem**: I've installed DESC, but when I check my installation I get an error ``ModuleNotFoundError: No module named 'termcolor'`` (or another module which is not ``desc``).

**Solution**:

You likely are not running python from the environment in which you've installed DESC. Try ensuring you've activated the conda environment that DESC is in( ``conda activate desc-env`` ), then retry using DESC.

**Problem**: I'm attempting to install jax with pip on a cluster, I get an error ``ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
desc-opt 0.9.2+587.gc0b44414.dirty...`` with a list of incompatiblities.

**Solution**:

This may be caused by a version of DESC already having been installed in your base conda environment.

Try removing the ``DESC`` folder completely, ensuring that ``pip list`` in your base conda environment no longer lists ``desc-opt`` as a package, then redo the installation instructions.
