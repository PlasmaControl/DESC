# made with chatgpt + claude !!

import sys
import numpy as np


def parse_focus_file(path):
    with open(path) as f:
        lines = f.read().split("\n")

    coils = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#coil_type"):
            i += 1
            meta = lines[i].split()
            coil_name = meta[2]
            i += 2  # skip to Nseg/current line
            current = float(lines[i].split()[1])
            i += 2  # skip to NFcoil
            nf = int(lines[i].strip())
            i += 2  # skip to first data row
            # 6 rows: xcos, xsin, ycos, ysin, zcos, zsin — each has (nf+1) values
            rows = []
            for _ in range(6):
                rows.append(np.array(list(map(float, lines[i].split()))))
                i += 1
            coils.append(dict(
                name=coil_name, current=current, nf=nf,
                xcos=rows[0], xsin=rows[1],
                ycos=rows[2], ysin=rows[3],
                zcos=rows[4], zsin=rows[5],
            ))
        else:
            i += 1
    return coils


def coils_to_arrays(coils):
    """Convert list of coil dicts to stacked numpy arrays in DESC mode format."""
    N = coils[0]["nf"]  # e.g. 500
    # DESC modes: positive int n → cos(n·t), negative int n → sin(|n|·t), 0 → constant
    modes = np.array(list(range(0, N + 1)) + list(range(-1, -(N + 1), -1)))
    X_n, Y_n, Z_n, currents, names = [], [], [], [], []
    for c in coils:
        # cos coefficients [0..N], then sin coefficients [1..N] (sin[0] ≡ 0, drop it)
        X_n.append(np.concatenate([c["xcos"], c["xsin"][1:]]))
        Y_n.append(np.concatenate([c["ycos"], c["ysin"][1:]]))
        Z_n.append(np.concatenate([c["zcos"], c["zsin"][1:]]))
        currents.append(c["current"])
        names.append(c["name"])
    return (np.array(X_n), np.array(Y_n), np.array(Z_n),
            modes, np.array(currents), np.array(names))


if "--focus" in sys.argv:
    focus_path = sys.argv[sys.argv.index("--focus") + 1]
    print(f"Parsing FOCUS file: {focus_path}")
    coils = parse_focus_file(focus_path)
    X_n, Y_n, Z_n, modes, currents, names = coils_to_arrays(coils)
else:
    npz_path = "tf_coils_desc.npz"
    print(f"Loading pre-parsed data from: {npz_path}")
    data     = np.load(npz_path, allow_pickle=True)
    X_n      = data["X_n"]
    Y_n      = data["Y_n"]
    Z_n      = data["Z_n"]
    modes    = data["modes"]
    currents = data["currents"]
    names    = data["names"]

n_coils, n_modes = X_n.shape
print(f"Loaded {n_coils} coils, {n_modes} modes each (N_max = {modes.max()})")

from desc.coils import FourierXYZCoil, CoilSet

coil_list = []
for i in range(n_coils):
    coil = FourierXYZCoil(
        current = float(currents[i]),
        X_n     = X_n[i],
        Y_n     = Y_n[i],
        Z_n     = Z_n[i],
        modes   = modes,
        name    = str(names[i]),
    )
    coil_list.append(coil)
    print(f"  Created {names[i]} (I = {currents[i]:.3f} A)")

coilset = CoilSet(coil_list, name="tf_coils")
print(f"\nCoilSet created: {coilset}")

output_path = "tf_coils_desc.h5"
coilset.save(output_path)
print(f"\nSaved CoilSet → {output_path}")

try:
    import matplotlib.pyplot as plt
    from desc.plotting import plot_coils

    fig, ax = plot_coils(coilset, figsize=(8, 6))
    fig.savefig("tf_coils_preview.png", dpi=120)
    print("Preview saved → tf_coils_preview.png")
except Exception as e:
    print(f"(Plotting skipped: {e})")