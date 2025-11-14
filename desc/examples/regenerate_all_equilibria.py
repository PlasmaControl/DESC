#!/usr/bin/env python3
"""
Run this script to regenerate all the DESC equilibria in desc/examples.

python3 regenerate_all_equilibria.py

"""
import argparse
import glob
import os
import subprocess as spr
import sys
from pathlib import Path

parser = argparse.ArgumentParser(
    prog="regenerate_all_equilibria",
)
parser.add_argument(
    "-o",
    "--output",
    metavar="output_dir",
    help="Path to output files. If not specified, defaults to input dir",
)
parser.add_argument(
    "--gpu",
    "-g",
    action="store_true",
    help="Use GPU if available. If more than one are available, selects the "
    + "GPU with most available memory. ",
)
parser.add_argument(
    "--noprecise",
    action="store_true",
    help="Don't regenerate precise QS equilibria.",
)


args = parser.parse_args(sys.argv[1:])

pwd = os.getcwd()
if args.output:
    output_path = os.path.abspath(args.output)
else:
    output_path = pwd

Path(output_path).mkdir(parents=True, exist_ok=True)

files = sorted(glob.glob(pwd + "/*.h5"))
print("Regenerating files:")
for f in files:
    print(f)
print("saving to: ", output_path)
names = [f.split("/")[-1].split(".")[0].replace("_output", "") for f in files]
names = [f for f in names if "precise" not in f]

for fname in names:
    print(f"Running the input file {fname} \n")
    cargs = [
        "desc",
        fname,
        "-vv",
        "-o",
        f"{os.path.join(output_path, fname + '_output.h5')}",
    ]
    if args.gpu:
        cargs += ["-g"]
    spr.run(cargs)

if not args.noprecise:
    spr.run(["python3", "-u", "precise_QA.py"])
    spr.run(["python3", "-u", "precise_QH.py"])
    spr.run(["python3", "-u", "reactor_QA.py"])
