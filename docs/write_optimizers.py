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
    for key, val in optimizers.items():
        d = {
            "Name": "``" + key + "``",
            "Scalar": val["scalar"],
            "Equality Constraints": val["equality_constraints"],
            "Inequality Constraints": val["inequality_constraints"],
            "Stochastic": val["stochastic"],
            "Hessian": val["hessian"],
            "GPU": val["GPU"],
            "Description": val["description"],
        }
        writer.writerow(d)
