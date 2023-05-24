"""Test the CSV database input creator."""
import os

from database_converter import desc_to_csv

names = []

inputs = {}
outputs = {}
is_current = {
    "precise_QA": True,
    "precise_QH": True,
    "QAS": True,
    "DSHAPE_CURRENT": True,
    "ESTELL": True,
}
for root, dirs, files in os.walk("desc/examples"):
    for file in files:
        if file.find(".h5") != -1:
            name = file.split("_output.h5")[0]
            names.append(name)
            outputs[name] = file
for root, dirs, files in os.walk("desc/examples"):
    for file in files:
        if file.find(".h5") == -1:
            for name in names:
                if file.find(name) != -1:
                    inputs[name] = file
for name in names:
    if "SOLOVEV" not in name or "CURRENT" in name:
        continue
    try:
        used_current = is_current[name]
    except KeyError:
        used_current = False
    print(f"Saving example {name}")
    desc_to_csv(
        f"desc/examples/{name}_output.h5",
        name="SOLOVEV",
        provenance="Example Equilibrium From DESC Repository desc/examples folder",
        inputfilename=inputs[name],
        current=used_current,
    )
