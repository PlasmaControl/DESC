#!/usr/bin/env python3
"""
Run this script to update all the DESC equilibria in desc/examples
and tests/inputs.

If you added a new IO attribute to the DESC equilibrium, this code
will update output files to include the new attribute. Please make sure
you add special handler to eq._set_up() function for new attribute. By default
the new attribute will be set to None.

python3 update_all_equilibria_backwards_compatibility.py

Copy the two lines below to the main code to run it on a GPU
from desc import set_device
set_device("gpu")
"""

import glob
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../"))

from desc.io import load

os.chdir("../desc/examples")
pwd = os.getcwd()
print(f"Updating files in {pwd}")

for fname in glob.glob(pwd + "/*.h5"):
    foutputname = fname.split(".")[0].split("/")[-1]
    print(f"Updating the output file {foutputname}")
    # load old output file
    # this will automatically update it by calling _set_up
    eqfam = load(fname)
    # save new output file
    eqfam.save(fname)

os.chdir("../../tests/inputs")
pwd = os.getcwd()
print(f"\nUpdating files in {pwd}")

for fname in glob.glob(pwd + "/*.h5"):
    foutputname = fname.split(".")[0].split("/")[-1]
    if foutputname != "DSHAPE_output_saved_without_current":
        print(f"Updating the output file {foutputname}")
        # load old output file
        # this will automatically update it by calling _set_up
        eqfam = load(fname)
        # save new output file
        eqfam.save(fname)
