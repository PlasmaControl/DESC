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
    "rho (unitless)": pho,
    "phi (rad)": mp,
    "theta (rad)": mt,
})

output_path = "muse_dipoles_desc.csv"
desc_df.to_csv(output_path, index=False)