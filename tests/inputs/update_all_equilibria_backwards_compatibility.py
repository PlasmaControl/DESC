#!/usr/bin/env python3
"""
Run this script to update all the DESC equilibria in tests/inputs.

If you added a new IO attribute to the DESC equilibrium, this code
will update output files to include the new attribute. Please make sure
you add special handler to eq._set_up() function for new attribute. By default
the new attribute will be set to None.

python3 update_all_equilibria_backwards_compatibility.py
"""
import glob
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../"))

from desc.io import load  # noqa: E402

pwd = os.getcwd()

for fname in glob.glob(pwd + "/*.h5"):
    foutputname = fname.split(".")[0].split("/")[-1]
    print(f"\nUpdating the output file {foutputname} \n")
    # load old output file
    eqfam = load(fname)
    print(eqfam)
    for i, eq in enumerate(eqfam):
        print(eq)
        # update the output file
        eq._set_up()
    # save new output file
    eqfam.save(fname)
