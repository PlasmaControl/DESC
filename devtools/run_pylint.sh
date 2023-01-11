#!/bin/sh

echo 'Removing old output files...'
rm --force 'pylint.output'

echo 'Running Pylint on settings defined in pylintrc...'
pylint '../desc' '../tests' --output 'pylint.output' 1>/dev/null
echo 'Done!'
