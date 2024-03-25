"""Test the CSV database input creator."""

import os

from desc.io.equilibrium_io import desc_to_csv

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
is_device = {
    "W7-X": "W7-X",
    "NCSX": "NCSX",
    "ATF": "ATF",
}  # QA, QH, QI, OT, OH, QP, AS
symmetry = {
    "W7-X": "QI",
    "NCSX": "QA",
    "ARIES-CS": "QA",
    "ESTELL": "QA",
    "precise_QA": "QA",
    "precise_QH": "QH",
    "QAS": "QA",
    "WISTELL-A": "QH",
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
    try:
        device = is_device[name]
    except KeyError:
        device = False
    try:
        sym_class = symmetry[name]
    except KeyError:
        sym_class = None

    print(f"Saving example {name}")
    desc_to_csv(
        f"desc/examples/{name}_output.h5",  # output filename
        name=name,  # some string descriptive name, not necessarily unique
        provenance="Example Equilibrium From DESC Repository desc/examples folder",
        inputfilename=inputs[name],
        current=used_current,
        deviceid=device,
        config_class=sym_class,
        user_updated="yge",
        user_created="dpanici",
        # TODO: Add config class (symmetry)
    )
