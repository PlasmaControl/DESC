###############################
Scripts for Linting with Flake8
###############################

Usage
======

It is recommended to run ``pre-commit install`` in a terminal under the main ``DESC`` directory.

Then anytime you commit with ``git commit`` the linting suite will run automatically.

To manually generate a linting report you need to execute the shell script ``run_flake8_linting.sh`` by typing ``sh run_flake8_linting.sh`` in a terminal from the ``devtools`` directory.

This will generate the ``flake8_errors.output`` and ``flake8_summary.output`` files.

Currently, black is still ran through GitHub actions.

Most formatting-level errors are being suppressed; errors that touch the code logic are primarily the ones being raised.


Outputs
=======

Flake8
------
flake8_errors.output will contain a line-by-line listing of errors
flake8_summary.output will total these up into categories in a quick summary


Configuration
=============
Flake8
------

Currently, error messages about whitespace and indenting that black does not care about have been suppressed.
These will be fixed in a future cleanup branch.
More errors can be added to the [flake8] section of settings.cfg after "ignore=", separated by commas.
