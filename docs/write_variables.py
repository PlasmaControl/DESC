import os
import re
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../"))
import csv

from desc.compute import data_index


def _escape(line):
    match = re.findall(r"\|.*\|", line)
    if match:
        sub = r"\|" + match[0][1:-1] + "|"
        line = line.replace(match[0], sub)
    return line


with open("variables.csv", "w", newline="") as f:
    fieldnames = ["Name", "Label", "Units", "Description", "Module"]
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")

    writer.writeheader()
    for key, val in data_index.items():
        d = {
            "Name": "``" + key + "``",
            "Label": ":math:`" + val["label"].replace("$", "") + "`",
            "Units": val["units_long"],
            "Description": val["description"],
            "Module": "``" + val["fun"].__module__ + "``",
        }
        # stuff like |x| is interpreted as a substitution by rst, need to escape
        d["Description"] = _escape(d["Description"])
        writer.writerow(d)
