import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../"))
from desc.compute.data_index import data_index
import csv

with open("_build/variables.csv", "w", newline="") as f:
    fieldnames = ["Name", "Label", "Units", "Description", "Compute function"]
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")

    writer.writeheader()
    keys = data_index.keys()
    for key in keys:
        d = {}
        d["Name"] = "``" + key + "``"
        d["Label"] = ":math:`" + data_index[key]["label"].replace("$", "") + "`"
        d["Units"] = data_index[key]["units_long"]
        d["Description"] = data_index[key]["description"]
        d["Compute function"] = "``" + data_index[key]["fun"] + "``"
        writer.writerow(d)
