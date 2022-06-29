============
Installation
============

Follow these instructions to install DESC and its dependencies.

On Your Local Machine
*********************

Install from Pypi:

.. code-block:: bash

    pip install desc-opt

Or from github (for development builds)

.. code-block:: bash

    git clone https://github.com/PlasmaControl/DESC.git
    cd DESC
    pip install -r requirements.txt
    pip install -r devtools/dev-requirements.txt

On Most Linux Computing Clusters
********************************

Install from Pypi:

.. code-block:: bash

    pip install desc-opt

Or from github (for development builds)

.. code-block:: bash

    git clone https://github.com/PlasmaControl/DESC.git
    cd DESC
    module load anaconda # load your python module
    conda create --name desc-env # create a new conda virtual environment
    conda activate desc-env # activate the environment
    conda install pip
    pip install -r requirements.txt # install dependencies
    pip install -r devtools/dev-requirements.txt # optional, if you want to run DESC/tests

On Clusters with IBM Power Architecture
***************************************

If pre-built JAX binaries are not available, you will first need to build JAX from source.
More info can be found here: https://jax.readthedocs.io/en/latest/developer.html

First get the latest stable release and load the necessary modules:

.. code-block:: bash

    git clone https://github.com/PlasmaControl/DESC.git   
    wget https://github.com/google/jax/archive/jaxlib-v0.1.55.tar.gz
    tar zxf jaxlib-v0.1.55.tar.gz # you can extract this tarball into any directory
    module load anaconda3 cudatoolkit cudnn/cuda-11.0/8.0.1

Then install python dependencies:

.. code-block:: bash

   conda create --name desc-env python=3.9 # create a new conda virtual environment
   conda activate desc-env
   conda install numpy scipy cython six # install JAX dependencies
   conda install h5py netcdf4 matplotlib # install other DESC dependencies
   conda install pytest pytest-cov codecov # optional, if you also want to run DESC/tests

Finally, build and install JAX:

.. code-block:: bash

   cd jax-jaxlib-v0.1.55 # or wherever you put the contents of the tarball
   python build/build.py --enable_cuda --cudnn_path /usr/local/cudnn/cuda-11.0/8.0.1 --noenable_march_native --noenable_mkl_dnn --cuda_compute_capabilities 7.0 --bazel_path /usr/bin/bazel
   pip install -e build
   pip install -e .

Checking your Installation
**************************

To check that you have properly installed DESC and its dependencies, try the following:

.. code-block:: bash

    python
    >>> from desc.equilibrium import Equilibrium

You should see an output stating the DESC version, the JAX version, and your device (CPU or GPU).

You can also try running an example input file:

.. code-block:: bash

   python -m desc -vvv examples/DESC/SOLOVEV
