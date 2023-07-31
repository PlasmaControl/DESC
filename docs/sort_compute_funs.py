"""Sort the compute functions and print to specified output.

First command line argument is module name, e.g. "_basis_vectors".
Second command line output is file to print to,
e.g. "sorted_basis_vectors.txt".
"""

import inspect
import re
import sys

import desc.compute

# Gather all compute function source code and map quantity name to source code.
source_codes = {}
pattern = re.compile(r"(?<=name=)[^,]+")
for module_name, module in inspect.getmembers(desc.compute, inspect.ismodule):
    if module_name == sys.argv[1]:
        for _, fun in inspect.getmembers(module, inspect.isfunction):
            source_code = inspect.getsource(fun)
            # quantities that this function computes
            matches = pattern.findall(source_code)
            if matches:  # skip imported functions
                source_codes[matches[0]] = source_code

# Write compute functions sorted by name to file.
with open(sys.argv[2], "w") as output_file:
    for name in sorted(source_codes):
        output_file.write("\n")
        output_file.write(source_codes[name])
        output_file.write("\n")
