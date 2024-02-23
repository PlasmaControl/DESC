======================
Command Line Interface
======================

DESC is executed with the following command line syntax:

.. argparse::
   :module: desc.input_reader
   :func: get_parser
   :prog: desc
   :nodefault:
   :nodescription:

As an example, the following command will run the 'SOLOVEV' input file contained in the DESC repository on a GPU
with timing output and save the solution to the file `solution.h5`:

.. code-block:: console

   python -m desc --gpu -vvv -o solution.h5 desc/examples/SOLOVEV
