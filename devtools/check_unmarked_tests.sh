#!/bin/sh

unmarked=$(pytest tests/ --collect-only -m "not unit and not regression" -q | head -n -2) 2> /dev/null
num_unmarked=$(echo "$unmarked"  | sed '/^\s*$/d' | wc -l)

if [ $num_unmarked -gt 0 ]
then
    echo "----found unmarked tests----"
    echo "$unmarked"
    exit 1
fi
