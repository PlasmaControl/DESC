#!/usr/bin/env python3
"""
Copy this script into the directory desc/examples and execute
"""
from desc import set_device
set_device("gpu")

from desc.io import InputReader
from desc.__main__ import main
from desc.vmec import VMECIO
import subprocess as spr
import os
import glob

pwd = os.getcwd()

#for fname in glob.glob(pwd + "/*.h5"):
#    if fname.split(".")[-1] == "h5" and os.path.isfile(fname.split(".")[-1]+".nc") is True and fname.split(".")[0].split("/")[-1].split("_")[0] != "precise":
#        finputname = fname.split(".")[0].split("/")[-1].split("_")[0]
#        print(f"Running the input file {finputname} \n")
#        main(cl_args=[str(f"{finputname}")])
#        # save wout file
#        VMECIO.save(eq, "wout_" + fname + ".nc", surfs=256)
#    elif fname.split(".")[-1] == "h5" and os.path.isfile(fname+".nc") is False and fname.split(".")[0].split("/")[-1].split("_")[0]  != "precise":
#        finputname = fname.split(".")[0].split("/")[-1].split("_")[0] 
#        print(f"Running the input file {finputname} \n")
#        main(cl_args=[str(f"{finputname}")])
#    else:
#        continue
#
#spr.call(["python3 -u precise_QA.py"], shell=True)
#spr.call(["python3 -u precise_QH.py"], shell=True) 
main(cl_args=[str(f"DSHAPE_CURRENT")])
