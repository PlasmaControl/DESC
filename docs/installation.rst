Installation
============

Follow these instructions to install DESC and its dependencies on PPPL's sunfire. 
Other systems should follow a similar process but may differ. 

.. code-block:: bash

    git clone https://github.com/ddudt/DESC.git
    cd DESC
    conda create --prefix ./env
    conda init <shell>
    conda activate ./env
    conda install anaconda
    conda install netcdf4
    conda install -c conda-forge jax=0.1.77
    conda deactivate
    unset PYTHONPATH
    conda activate ./env
