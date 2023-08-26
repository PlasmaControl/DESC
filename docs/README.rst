To build the documentation locally, from the DESC directory:

.. code-block:: sh

    cd docs
    conda install -c conda-forge pandoc
    make clean html

will build the documentation in the _build/html/ folder of the docs folder.

To build the pdf, type ``make clean latexpdf`` from the docs/ directory.
