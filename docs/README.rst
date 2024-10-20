To build the documentation locally, install the developer requirements and run from the DESC directory:

.. code-block:: sh

    cd docs
    conda install -c conda-forge pandoc
    make html

will build the documentation in the _build/html/ folder of the docs folder.

To build the pdf, type ``make latexpdf`` from the docs/ directory.
