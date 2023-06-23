##########################################
Scripts for Linting with Pylint and Flake8
##########################################

Usage
======

To use, run either the run_pylint_linting.sh or run_flake8_linting.sh scripts

These will generate a pylint.output file, or flake8_errors.ouput and flake8_summary.output files.

Currently, black is still ran through GitHub actions.

Most formatting-level errors are being suppressed; errors that touch the code logic are primarily the ones being raised.

Currently, black is still ran through GitHub actions.



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

Currently, error messages about whitespace and indenting that black does not care about have been suppressed.
These will be fixed in a future cleanup branch.
More errors can be added to the [flake8] section of settings.cfg after "ignore=", separated by commas.

Pylint:
-------
Currently several classes of error are being suppressed, mostly to do with preferring encapsulating behavior into simpler classes and modules.
To disable certain classes of error message, go to the [MESSAGES] section of pylinrc, and add the error to "disable=" separated by commas.
Additionally, several very minor errors are being suppressed to be fixed in a future cleanup branch.
