"""Sort the compute functions and print to specified output.

First command line argument is module name, e.g. "_basis_vectors".
Second command line output is file to print to,
e.g. "sorted_basis_vectors.txt".
"""

import inspect
import re
import sys

import desc.compute


def get_matches(fun, pattern):
    """Return all matches of ``pattern`` in source code of function ``fun``."""
    src = inspect.getsource(fun)
    matches = pattern.findall(src)
    return matches


# Gather all compute function source code and map to quantity name.
src_codes = {}
pattern_name = re.compile(r"(?<=name=)[^,]+")
for module_name, module in inspect.getmembers(desc.compute, inspect.ismodule):
    if module_name == sys.argv[1]:
        for _, fun in inspect.getmembers(module, inspect.isfunction):
            # quantities that this function computes
            name = get_matches(fun, pattern_name)
            if len(name) > 0:  # skip imported functions
                src_codes[name[0]] = inspect.getsource(fun)

# Write compute functions sorted by name to file.
with open(sys.argv[2], "w") as output_file:
    for name in sorted(src_codes):
        output_file.write(src_codes[name])
        output_file.write("\n\n")
