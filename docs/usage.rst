=====
Usage
=====

DESC is executed with the command line syntax 

.. code-block:: bash

   python -m desc [-h] [-o <path/to/output_file>] [-p] [-q] [-v] [--vmec <path/to/VMEC_solution>] <path/to/input_file>

and the following arguments: 

- ``<path/to/input_file>`` is the path to the input file that is to be executed. 
- ``-o, --output`` specifies that the output files will be saved at the path ``<path/to/output_file>``. If omitted, it defaults to ``<path/to/input_file>.'output'``. 
- ``-p, --plot`` will plot the results after the solver finishes. 
- ``-q, --quiet`` will not display any progress information. 
- ``-v, --verbose`` will display all progress information, including minor iterations. The default is to display a summary at each major interation. 
- ``--vmec`` will plot a comparison of the DESC solution to a VMEC solution given by a NetCDF file at ``<path/to/VMEC_solution>``. 
