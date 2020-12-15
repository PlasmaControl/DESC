To build the documentation locally, from the DESC directory:

.. code-block:: bash

    cd docs
    pip install sphinx
    pip install sphinx-rtd-theme
    pip install sphinx-argparse
    make html

will build the documentation in the _build/html/ folder of the docs folder.

To build the pdf, type ``make pdf`` from the docs/ directory.
