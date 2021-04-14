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

    git clone https://github.com/dpanici/DESC.git
    cd DESC
    pip install -r requirements.txt


On PPPL's Sunfire Cluster
*************************
Other linux based clusters should follow a similar process but may differ. 

Install from Pypi:

.. code-block:: bash

    pip install desc-opt
    
Or from github (for development builds)

.. code-block:: bash

    git clone https://github.com/dpanici/DESC.git
    cd DESC
    conda create --prefix ./env
    conda init <shell>
    conda activate ./env
    conda install anaconda
    conda install netcdf4 h5py matplotlib numpy scipy six cython
    conda install -c conda-forge jax=0.1.77
    conda deactivate
    unset PYTHONPATH
    conda activate ./env

On Princeton's Adroit Cluster
*****************************
Other linux based clusters should follow a similar process but may differ. 

Install from Pypi:

.. code-block:: bash

    pip install desc-opt
    
Or from github (for development builds)

.. code-block:: bash

    git clone https://github.com/dpanici/DESC.git
    cd DESC
    module load anaconda
    conda create --prefix ./env
    conda init <shell>
    conda activate ./env
    conda install anaconda
    conda install netcdf4 h5py matplotlib numpy scipy six cython
    conda install -c conda-forge jax=0.1.77
    conda deactivate
    unset PYTHONPATH
    conda activate ./env

Then, to build the documentation locally, from the DESC directory:

.. code-block:: bash

    cd docs
    pip install sphinx
    pip install sphinx-rtd-theme
    pip install sphinx-argparse
    make html

will build the documentation in the _build/html/ folder of the docs folder.

On Traverse
***********
(or other IBM Power based architecture where pre-built JAX binaries are not available) you will first need to build JAX from source.
More info can be found here: https://jax.readthedocs.io/en/latest/developer.html

For Traverse, first get the latest stable release and load the necessary modules:

.. code-block:: bash

    git clone https://github.com/dpanici/DESC.git   
    wget https://github.com/google/jax/archive/jaxlib-v0.1.55.tar.gz
    tar zxf jaxlib-v0.1.55.tar.gz # this puts it in the current directory, you can put it anywhere that is convenient
    module load anaconda3 cudatoolkit cudnn/cuda-11.0/8.0.1

Then install python dependencies:

.. code-block:: bash

   conda create --name jax python=3.7 # suggested you make a new environment
   conda activate jax
   conda install numpy scipy cython six # python packages JAX needs
   conda install h5py netcdf4 matplotlib # other DESC dependencies that JAX doesn't require
   conda install pytest pytest-cov codecov #if you also want to run the DESC tests

Finally, build and install JAX:

.. code-block:: bash

   cd jax-jaxlib-v0.1.55 # or wherever else you put the contents of the tarball		
   python build/build.py --enable_cuda --cudnn_path /usr/local/cudnn/cuda-11.0/8.0.1 --noenable_march_native --noenable_mkl_dnn --cuda_compute_capabilities 7.0 --bazel_path /usr/bin/bazel
   pip install -e build 
   pip install -e . 

