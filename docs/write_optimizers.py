import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../"))
import csv

from desc.optimize import optimizers

with open("optimizers.csv", "w", newline="") as f:
    fieldnames = [
        "Name",
        "Scalar",
        "Equality Constraints",
        "Inequality Constraints",
        "Stochastic",
        "Hessian",
        "GPU",
        "Description",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")

    writer.writeheader()
    keys = optimizers.keys()
    for key in keys:
        d = {}
        d["Name"] = "``" + key + "``"
        d["Scalar"] = optimizers[key]["scalar"]
        d["Equality Constraints"] = optimizers[key]["equality_constraints"]
        d["Inequality Constraints"] = optimizers[key]["inequality_constraints"]
        d["Stochastic"] = optimizers[key]["stochastic"]
        d["Hessian"] = optimizers[key]["hessian"]
        d["GPU"] = optimizers[key]["GPU"]
        d["Description"] = optimizers[key]["description"]
        writer.writerow(d)
