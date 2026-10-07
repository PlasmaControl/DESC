#!/bin/sh

if [ -f "devtools/pre-commit.log" ]; then
    echo "Error in earlier pre-commit! Skipping unmarked tests check."
    exit 1
fi

echo "Files to check: $@"
python devtools/check_unmarked_tests.py "$@"
