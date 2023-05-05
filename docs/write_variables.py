import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../"))
import csv

from desc.compute import data_index

with open("variables.csv", "w", newline="") as f:
    fieldnames = ["Name", "Label", "Units", "Description", "Module"]
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")

    writer.writeheader()
    keys = data_index.keys()
    for key in keys:
        d = {}
        d["Name"] = "``" + key + "``"
        d["Label"] = ":math:`" + data_index[key]["label"].replace("$", "") + "`"
        d["Units"] = data_index[key]["units_long"]
        d["Description"] = data_index[key]["description"]
        d["Module"] = "``" + data_index[key]["fun"].__module__ + "``"
        writer.writerow(d)
