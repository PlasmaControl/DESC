"""Sort the compute functions and print to specified output.

First command line argument is module name, e.g. "_basis_vectors".
Second command line argument is output file to print to,
e.g. "sorted_basis_vectors.txt".
"""

import inspect
import re
import sys

import desc.compute

preamble = ""

# Gather all compute function source code and map name to source code.
source_codes = {}
pattern = re.compile(r"(?<=name=)[^,]+")
for module_name, module in inspect.getmembers(desc.compute, inspect.ismodule):
    if module_name == sys.argv[1]:
        # everything in the module until the first compute function
        # (import statements, etc.)
        preamble = inspect.getsource(module).partition("@register_compute_fun")[0]
        for function_name, fun in inspect.getmembers(module, inspect.isfunction):
            source_code = inspect.getsource(fun)
            matches = pattern.findall(source_code)
            if matches:  # skip imported functions
                # matches[0] is the thing this function says it computes, e.g. x
                # while function_name is e.g. _x_ZernikeRZLToroidalSection
                if function_name in source_codes:
                    raise ValueError("Can't sort when things have same name.")
                source_codes[function_name] = source_code

# Write functions sorted to file.
with open(sys.argv[2], "w") as output_file:
    output_file.write(preamble)
    for function_name in sorted(source_codes):
        output_file.write(source_codes[function_name])
        output_file.write("\n\n")
