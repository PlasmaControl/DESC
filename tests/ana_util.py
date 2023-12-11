import numpy as np
from desc.basis import FourierZernikeBasis

def modes_gen(disc_lmn: dict, basis: FourierZernikeBasis):
    lmn = np.zeros(basis.num_modes)
    for key, val in disc_lmn.items():
        lmn[basis.get_idx(*key)] = val
    return lmn