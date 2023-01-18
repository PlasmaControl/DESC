============
Installation
============

Follow these instructions to install DESC and its dependencies.
Note that the default installation instructions here (except for the IBM Power architecture instructions) do not install JAX with GPU support.
To install JAX with GPU support, please refer to the `JAX installation docs <https://github.com/google/jax#installation>`_.

On Your Local Machine
*********************

Install from PyPI:

.. code-block:: console

    pip install desc-opt

Or from GitHub (for development builds)

.. code-block:: sh

    git clone https://github.com/PlasmaControl/DESC.git
    cd DESC

    # OPTION 1: install with pip
    # standard build
    pip install -r requirements.txt
    # developer build (if you want to run tests)
    pip install -r devtools/dev-requirements.txt

    # OPTION 2: install with conda
    # standard build
    conda env create --file requirements_conda.yml
    # developer build (if you want to run tests)
    conda env create --file devtools/dev-requirements_conda.yml
    conda activate desc-env

On Most Linux Computing Clusters
********************************

Install from PyPI:

.. code-block:: console

    pip install desc-opt

Or from GitHub (for development builds)

.. code-block:: sh

    git clone https://github.com/PlasmaControl/DESC.git
    cd DESC

    module load anaconda  # load your python module

    # standard build
    conda env create --file requirements_conda.yml
    # developer build (if you want to run tests)
    conda env create --file devtools/dev-requirements_conda.yml
    conda activate desc-env

On Clusters with IBM Power Architecture
***************************************

If pre-built JAX binaries are not available, you will first need to build JAX from source.
More info can be found here: https://jax.readthedocs.io/en/latest/developer.html

The following are instructions tested to work on the Traverse supercomputer at Princeton:

.. code-block:: sh

    git clone https://github.com/PlasmaControl/DESC.git
    cd DESC

    module load anaconda3/2020.11 cudatoolkit/11.1 cudnn/cuda-11.1/8.0.4

    conda create --name desc-env python=3.8
    conda activate desc-env
    # install what you can of the requirements with conda, ends up being all but jax, jaxlib and nvgpu
    conda install colorama "h5py>=3.0.0" "matplotlib>=3.3.0,<=3.6.0,!=3.4.3" "mpmath>=1.0.0" "netcdf4>=1.5.4" "numpy>=1.20.0" psutil "scipy>=1.5.0" termcolor
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

Optionally, if you want to be able to use pytest and other development tools:

.. code-block:: sh

    cd ../DESC
    pip install -r devtools/dev-requirements.txt

Checking your Installation
**************************

To check that you have properly installed DESC and its dependencies, try the following:

.. code-block:: pycon

    python
    >>> from desc import set_device  # only needed if running on a GPU
    >>> set_device('gpu')  # only needed if running on a GPU
    >>> import desc.equilibrium


You should see an output stating the DESC version, the JAX version, and your device (CPU or GPU).

You can also try running an example input file:

.. code-block:: console

    python -m desc -vvv examples/DESC/SOLOVEV
