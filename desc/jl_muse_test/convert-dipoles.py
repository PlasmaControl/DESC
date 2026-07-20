# made with the help of chatgpt !

import numpy as np
import pandas as pd

file_path = "muse-magnets.focus"
data = np.genfromtxt(file_path, delimiter=",", skip_header=3)

(
    coiltype,
    symmetry,
    coilname,
    ox,
    oy,
    oz,
    Ic,
    M_0,
    pho,
    Lc,
    mp,
    mt,
) = data.T

desc_df = pd.DataFrame({
    "x (m)": ox,
    "y (m)": oy,
    "z (m)": oz,
    "phi (rad)": mp,
    "theta (rad)": mt,
    "m0": M_0,
    "rho (unitless)": pho,
    "Ic": Ic,
})

output_path = "muse_dipoles_desc.csv"
desc_df.to_csv(output_path, index=False)