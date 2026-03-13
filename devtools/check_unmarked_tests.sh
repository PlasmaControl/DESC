#!/bin/sh

if [ -f "devtools/pre-commit.log" ]; then
    echo "Error in earlier pre-commit! Skipping unmarked tests check."
    exit 1
fi

# Start the timer using date (in seconds since epoch)
start_time=$(date +%s)

echo "Files to check: $@"
# Collect unmarked tests for the specific file and suppress errors
unmarked=$(pytest "$@" --collect-only -m "not unit and not regression and not benchmark and not memory and not mpi_run and not mpi_setup" -q 2> /dev/null | head -n 2)

# Count the number of unmarked tests found, ignoring empty lines and the line emitted if pytest found no unmarked tests
num_unmarked=$(echo "$unmarked" | sed '/^\s*$/d;/no tests collected/d' | wc -l)

# If there are any unmarked tests, print them and exit with status 1
if [ "$num_unmarked" -gt 0 ]; then
    echo "----found $num_unmarked unmarked tests----"
    echo "$unmarked"
    # Calculate the elapsed time and print with a newline
    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))
    printf "\nTime taken: %d seconds" "$elapsed_time"
    exit 1
fi
