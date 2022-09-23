#!/bin/sh

echo 'Removing old output files...'
rm --force 'flake8_errors.output' 'flake8_summary.output'

echo 'Running Flake8 using settings defined in settings.cfg...'
flake8 --config '../setup.cfg' --output-file 'flake8_errors.output' '../desc/' '../tests/'

echo 'Generating Flake8 report...'
flake8 --config '../setup.cfg' -qqq --statistics --output-file 'flake8_summary.output' '../desc/' '../tests/'
