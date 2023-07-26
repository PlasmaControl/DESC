============
Installation
============

Follow these instructions to install DESC and its dependencies.
Note that most of the installation options here do not install JAX with GPU support.
We do include installation instructions to install JAX with GPU support on some computing clusters that we have tested.
In general, to install JAX with GPU support, please refer to the `JAX installation docs <https://github.com/google/jax#installation>`__.

For information on using conda, see `here <https://conda.io/projects/conda/en/latest/user-guide/getting-started.html#starting-conda>`__.
Other package managers like venv could be used instead of conda, we have just chosen conda as our package manager of choice, and only test with conda environments, so your mileage may vary with other managers.

**NOTE: DESC requires python>=3.8.**
**If you have python2 also locally installed, replace all `pip` commands with `pip3` and all `python` commands with `python3` to ensure the correct python version is used.**

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

Now pick one of the installation options below.

Option 1: Using pip to install packages (this will only install DESC + JAX with CPU capabilities, NOT GPU)

`Option 1 tested to work on M1 Macbook on 5-3-23`

.. code-block:: sh

    conda create --name desc-env 'python>=3.8'
    conda activate desc-env
    pip install --editable .
    # optionally install developer requirements (if you want to run tests)
    pip install -r devtools/dev-requirements.txt

Option 2: Using conda to install packages (this will only install DESC + JAX with CPU capabilities, NOT GPU)

.. code-block:: sh

    # only need to do one of these conda env create commands, not both
    # option A: without developer requirements
    conda env create --file requirements_conda.yml
    # option B: with developer requirements (if you want to run tests)
    conda env create --file devtools/dev-requirements_conda.yml

    # to add DESC to your Python path
    conda activate desc-env
    pip install --no-deps --editable .

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

Now pick one of the installation options below.

Option 1: Using pip to install packages (this will only install DESC + JAX with CPU capabilities, NOT GPU)

.. code-block:: sh

    conda create --name desc-env 'python>=3.8'
    conda activate desc-env
    pip install --editable .
    # optionally install developer requirements (if you want to run tests)
    pip install -r devtools/dev-requirements.txt

Option 2: Using conda to install packages (this will only install DESC + JAX with CPU capabilities, NOT GPU)

.. code-block:: sh

    # only need to do one of these conda env create commands, not both
    # option A: without developer requirements
    conda env create --file requirements_conda.yml
    # option B: with developer requirements (if you want to run tests)
    conda env create --file devtools/dev-requirements_conda.yml

    # to add DESC to your Python path
    conda activate desc-env
    pip install --no-deps --editable .

With CPU+GPU support
--------------------

We will show the installation instructions that work for the clusters we've tested.
If your cluster is not shown, try the installation for the cluster most resembling your own, or see if your cluster has
specific JAX GPU installation instructions, as that is the main installation difference between clusters.
(note, most of these clusters below are `x86_64` architectures, see the `JAX installation docs <https://github.com/google/jax#installation>`__ for more info if you have a different architecture ).

**Note that DESC does not always test on or guarantee support of the latest version of JAX (which does not have a stable 1.0 release yet), and thus older versions of GPU-accelerated versions of JAX may need to be installed, which may in turn require lower versions of JaxLib, as well as CUDA and CuDNN.**

Della Cluster (Princeton)
+++++++++++++++++++++++++
These instructions were tested and confirmed to work on the Della cluster at Princeton as of 7-11-2023.

First, install JAX (commands taken from `this tutorial <https://github.com/PrincetonUniversity/intro_ml_libs/tree/master/jax>`__ ):

.. code-block:: sh

    module load anaconda3/2023.3
    CONDA_OVERRIDE_CUDA="11.2" conda create --name desc-env jax "jaxlib==0.4.1=cuda112*" -c conda-forge
    conda activate desc-env

Then, we install DESC:

.. code-block:: sh

    git clone https://github.com/PlasmaControl/DESC.git
    cd DESC
    # remove the jax lines from requirements.txt, as we already have installed them above
    sed -i '/jax/d' ./requirements.txt
    # then install as usual
    pip install --editable .
    # optionally install developer requirements (if you want to run tests)
    pip install -r devtools/dev-requirements.txt


Stellar Cluster (Princeton)
+++++++++++++++++++++++++++
Using pip install and including GPU capabilities.
These instructions were tested and confirmed to work on the Stellar cluster at Princeton as of 5-2-2023.

First, install JAX with GPU support (commands taken from `this tutorial <https://github.com/PrincetonUniversity/intro_ml_libs/tree/master/jax>`__ ):

.. code-block:: sh

    module load anaconda3/2022.5
    CONDA_OVERRIDE_CUDA="11.2" conda create --name desc-env jax "jaxlib==0.4.1=cuda112*" -c conda-forge
    conda activate desc-env

Then, we install DESC:

.. code-block:: sh

    git clone https://github.com/PlasmaControl/DESC.git
    cd DESC
    # remove the jax lines from requirements.txt, as we already have installed them above
    sed -i '/jax/d' ./requirements.txt
    # then install as usual
    pip install --editable .
    # optionally install developer requirements (if you want to run tests)
    pip install -r devtools/dev-requirements.txt

On Clusters with IBM Power Architecture
***************************************

If pre-built JAX binaries are not available, you will first need to build JAX from source.
More info can be found here: https://jax.readthedocs.io/en/latest/developer.html

These instructions were tested and confirmed to work on the Traverse supercomputer at Princeton as of 7-10-2023.

.. code-block:: sh

    git clone https://github.com/PlasmaControl/DESC.git
    cd DESC

    module load anaconda3/2020.11 cudatoolkit/11.1 cudnn/cuda-11.1/8.0.4

    conda create --name desc-env python=3.8
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
    >>> import desc.equilibrium


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
