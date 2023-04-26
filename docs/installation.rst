============
Installation
============

Follow these instructions to install DESC and its dependencies.
Note that the default installation instructions here (except for the IBM Power architecture instructions) do not install JAX with GPU support.
To install JAX with GPU support, please refer to the `JAX installation docs <https://github.com/google/jax#installation>`_.
For information on using conda, see `here <https://conda.io/projects/conda/en/latest/user-guide/getting-started.html#starting-conda>`_.
**NOTE: DESC requires python>=3.8, and if you have python2 also locally installed, replace all `pip` commands with `pip3` and all `python` commands with `python3` to ensure the correct python version is used**

On Your Local Machine
*********************

Install from PyPI:

.. code-block:: console

    pip install desc-opt

Or from GitHub (for development builds)

.. code-block:: sh

    git clone https://github.com/PlasmaControl/DESC.git
    cd DESC

    # OPTION 1: install with pip after first creating a conda environment
    conda create
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
    python setup.py install

On Most Linux Computing Clusters
********************************

These examples use conda environments and either installing with conda or with pip.
On computing clusters you must ensure to `module load anaconda` in order to use conda (or in some clusters, you must specify the version of anaconda module you want)

With CPU support only
---------------------

Install from PyPI:

.. code-block:: console

    pip install desc-opt

Or from GitHub (for development builds):

.. code-block:: sh

    git clone https://github.com/PlasmaControl/DESC.git
    cd DESC

    module load anaconda  # load your python module

Using conda to install packages (note, this will only install DESC + JAX with CPU capabilities, NOT GPU):

.. code-block:: sh

    # standard build
    conda env create --file requirements_conda.yml
    # developer build (if you want to run tests)
    conda env create --file devtools/dev-requirements_conda.yml
    conda activate desc-env
    python setup.py install

Using pip install (note, this will only install DESC + JAX with CPU capabilities, NOT GPU):

.. code-block:: sh

    # standard build
    pip install -r requirements.txt
    # developer build (if you want to run tests)
    pip install -r devtools/dev-requirements.txt
    conda activate desc-env
    python setup.py install

With CPU+GPU support
--------------------

We will show the installation instructions that work for the clusters we've tested.
If your cluster is not shown, try the installation for the cluster most resembling your own, or see if your cluster has
specific JAX GPU installation instructions, as that is the main installation difference between clusters.
(note, most of these clusters below are `x86_64` architectures, see the `JAX installation docs <https://github.com/google/jax#installation>`_ for more info if you have a different architecture ).

Della Cluster (Princeton)
+++++++++++++++++++++++
These instructions were tested and confirmed to work on the Della cluster at Princeton as of 10-13-2022.

First, install JAX (commands taken from `this tutorial <https://github.com/PrincetonUniversity/intro_ml_libs/tree/master/jax>`_ ):

.. code-block:: sh

    module load anaconda3/2021.11
    conda create --name desc-env python=3.9
    conda activate desc-env
    pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Then, we install DESC:
.. code-block:: sh

    git clone https://github.com/PlasmaControl/DESC.git
    # then go into requirements.txt and remove the jax lines, as we already have installed them above
    sed -i '/jax/d' ./requirements.txt
    # then install as usual
    pip install -r requirements.txt
    # developer build (if you want to be able to run tests)
    pip install -r devtools/dev-requirements.txt
    python setup.py install


Stellar Cluster (Princeton)
+++++++++++++++++++++++
Using pip install and including GPU capabilities.
These instructions were tested and confirmed to work on the Stellar cluster at Princeton as of 1-12-2023.

First, install JAX with GPU support (commands taken from `this tutorial <https://github.com/PrincetonUniversity/intro_ml_libs/tree/master/jax>`_ ):

.. code-block:: sh

    module load anaconda3/2022.5
    CONDA_OVERRIDE_CUDA="11.2" conda create --name desc-env jax "jaxlib==0.4.1=cuda112*" -c conda-forge

Then, we install DESC:
.. code-block:: sh

    conda activate desc-env
    git clone https://github.com/PlasmaControl/DESC.git
    # then use sed on requirements.txt to remove the jax line, as we already have installed it above
    cd DESC
    sed '/jax/d' ./requirements.txt > ./requirements_no_jax.txt
    # then install as usual
    pip install -r requirements_no_jax.txt
    # developer build (if you want to be able to run tests)
    pip install -r devtools/dev-requirements.txt
    python setup.py install


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

.. code-block:: python

    python
    >>> from desc import set_device  # only needed if running on a GPU
    >>> set_device('gpu')  # only needed if running on a GPU
    >>> import desc.equilibrium


You should see an output stating the DESC version, the JAX version, and your device (CPU or GPU).

You can also try running an example input file:

.. code-block:: console

    python -m desc -vvv examples/DESC/SOLOVEV


Troubleshooting
***************
We list here some common problems encountered during installation and their possible solutions.
If you encounter issues during installation, please `leave us an issue on Github <https://github.com/PlasmaControl/DESC/issues>`_ and we will try our best to help!

 - **Problem**: I've installed DESC, but when I check my installation I get an error `ModuleNotFoundError: No module named 'desc'`
   - **Solution**: This may be caused by DESC not being on your PYTHONPATH, or your environment containing DESC not being activated.
     - Try re-running `python setup.py install` step, or manually add the DESC directory to your PYTHONPATH, like `export PYTHONPATH="$PYTHONPATH:path/to/DESC"`
     - Try ensuring you've activated the conda environment that DESC is in( `conda activate desc-env` ), then retry using DESC.
 - **Problem**: I've installed DESC, but when I check my installation I get an error `ModuleNotFoundError: No module named 'termcolor'` (or another module which is not `desc```)
   - Solution: you likely are not running python from the environment in which you've installed DESC.
     Try ensuring you've activated the conda environment that DESC is in( `conda activate desc-env` ), then retry using DESC
