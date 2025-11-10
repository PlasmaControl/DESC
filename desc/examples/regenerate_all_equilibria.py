#!/usr/bin/env python3
"""
Run this script to regenerate all the DESC equilibria in desc/examples.

python3 regenerate_all_equilibria.py

Copy the two lines below to the main code to run it on a GPU
from desc import set_device
set_device("gpu")
"""
import glob
import os
import subprocess as spr

from desc.__main__ import main
from desc.io import load
from desc.vmec import VMECIO

pwd = os.getcwd()

for fname in glob.glob(pwd + "/*.h5"):
    if (
        fname.split(".")[-1] == "h5"
        and os.path.isfile(fname.split(".")[-1] + ".nc") is True
        and fname.split(".")[0].split("/")[-1].split("_")[0] != "precise"
    ):
        finputname = fname.split(".")[0].split("/")[-1].split("_")[0]
        print(f"Running the input file {finputname} \n")
        main(cl_args=[str(f"{finputname}"), "-vv"])
        # save wout file
        eq = load(f"{pwd}/{finputname}")[-1]
        VMECIO.save(eq, "wout_" + fname + ".nc", surfs=256)
    elif (
        fname.split(".")[-1] == "h5"
        and os.path.isfile(fname + ".nc") is False
        and fname.split(".")[0].split("/")[-1].split("_")[0] != "precise"
    ):
        finputname = fname.split(".")[0].split("/")[-1].split("_")[0]
        print(f"Running the input file {finputname} \n")
        main(cl_args=[str(f"{finputname}"), "-vv"])
    else:
        continue

main(cl_args=["DSHAPE_CURRENT", "-vv"])
spr.call(["python3 -u precise_QA.py"], shell=True)
spr.call(["python3 -u precise_QH.py"], shell=True)
