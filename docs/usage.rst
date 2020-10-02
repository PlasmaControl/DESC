=====
Usage
=====

.. role:: bash(code)
   :language: bash

DESC is executed with the command line syntax 
:bash:`python desc/DESC.py [-h] [-o <path/to/output_file>] [-p] [-q] [-v] [--vmec <path/to/VMEC_solution>] <path/to/input_file>`
with the following arguments: 
- :bash:`<path/to/input_file>` is the path to the input file that is to be executed.
- :bash:`-o, --output` specifies that the output files will be saved at the path :bash:`<path/to/output_file>`. 
If omitted, it defaults to :bash:`<path/to/input_file>.'output'`. 
- :bash:`-p, --plot` will plot the results after the solver finishes. 
- :bash:`-q, --quiet` will not display any progress information. 
- :bash:`-v, --verbose` will display all progress information, including minor iterations. 
The default is to display a summary at each major interation. 
- :bash:`--vmec` will plot a comparison of the DESC solution to a VMEC solution given by a NetCDF file at :bash:`<path/to/VMEC_solution>`. 
