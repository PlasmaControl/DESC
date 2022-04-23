##########################################
Scripts for Linting with Pylint and Flake8
##########################################

Useage
======

To use, run either the run_pylint_linting.sh or run_flake8_linting.sh scripts

These will generate a pylint.output file, or flake8_errors.ouput and flake8_summary.output files.

Currently, black is still ran through github actions


Outputs
=======

Flake8
------
flake8_errors.output will contain a line-by-line listing of errors
flake8_summary.output will total these up into categories in a quick summary

Pylint
------
Pylint.output currently will contain, in the following order :
	1.  a series of line-specific reports, then 
	2.  show duplicated code blocks 
	3.  a dependency tree 
	4.  an error summary



Configuration
=============
Flake8
------
Currently, no error messages are disabled, but error signatures of the format "E302" can be added to flake.ini after "ignore=" seperated by commas

Pylint:
-------
To disable certain classes of error message, go to the [MESSAGES] section of pylinrc, and add the error to "disable=" separated by commas

Currently the only suppressed errors are 
	1.  'invalid-name', which can be used to specify with regex how functions, objects, attributes, and variable names should look
		This can be defined under the [BASIC] section of pylinrc; currently common defaults that disagree with DESC are there
	2.  Various errors that have strong opinions on how objects should inhereit, and what objects should really be functions