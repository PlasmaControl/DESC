from desc import set_device

# if GPU is available, use it. If not, will run on CPU without any issues
set_device("gpu")
import os
import subprocess
import sys

from desc.field_line_tracing_DESC_with_current_potential_python_regcoil import (
    trace_from_curr_pot,
)
from desc.find_helical_contours_from_python_regcoil_equal_curr_line_integral import (
    find_helical_coils,
)
from desc.regcoil import run_regcoil

eqname = "./ellNFP4_init_smaller.h5"
dirname = f"{eqname.split('/')[-1].strip('.h5')}"
numCoils = 19
step = 2
alpha = 0
coilsFilename = (
    f"{dirname}/coils_ncoils_{numCoils}_{dirname}_lam_{alpha:1.2e}_step_{step}.txt"
)

##### Run Python REGCOIL #####
printstring = (
    "Running python REGCOIL with regularization param ="
    + str(alpha)
    + " on  DESC equilibrium "
    + str(eqname)
)
print("#" * len(printstring) + "\n" + printstring + "\n" + "#" * len(printstring))

M = 20
N = 20
NFP = 4
# this takes <1 min on GPU on Della, but can take ~20 minutes on CPU on Dario's laptop
(
    phi_mns,
    alphas,
    curr_pot_trans2,
    I2,
    G,
    phi_total_function,
    B0,
    chiBs,
    ind,
) = run_regcoil(
    eqname,
    alpha=alpha,
    helicity_ratio=-1,
    scan=True,
    basis_M=M,
    basis_N=N,
    eval_grid_M=40,
    eval_grid_N=40,
    source_grid_M=50,
    source_grid_N=300,  # this must be at least 2*basis_N*NFP, if not higher
    nscan=10,
    scan_upper=-3,
    external_TF_scan=False,
    show_plots=True,
    external_TF_scan_n=5,
    verbose=2,
)

# FIXME: this is can be slow, even on GPU
trace_from_curr_pot(
    phi_mns[7],
    curr_pot_trans2,
    eqname,
    I2,
    G,
    alpha=alpha,
    M=50,
    N=160,
    ntransit=5,
    external_TF=B0,
)

phi_mn_opt = phi_mns[7]

# TODO: the equal-current optimization can be slow ( minutes for 5 coils, hour or longer for 15+ coils),
# should JIT compile some of the functions to improve speed, and see other ways to optimize
coilset2 = find_helical_coils(
    phi_mn_opt,
    curr_pot_trans2.basis,
    eqname,
    I2,
    G,
    alphas[7],
    desirednumcoils=numCoils,
    coilsFilename=coilsFilename,
    maxiter=500,
    method="Nelder-Mead",
)

from desc.coils import CoilSet

coilset = CoilSet.from_makegrid_coilfile(coilsFilename)

### calculate BNORM ###
from desc.calc_BNORM_from_coilset import calc_BNORM_from_coilset

printstring = "Finding Bnormal on plasma surface from the Helical Coilset"
print("#" * len(printstring) + "\n" + printstring + "\n" + "#" * len(printstring))
# /home/dpanici/regcoil/work/opt_nt_tao/working_scripts/
calc_BNORM_from_coilset(coilset, eqname, alpha=alpha, step=step)


# ### field line trace with coils ###
from desc.field_line_tracing_DESC_from_coilset import field_trace_from_coilset

field_trace_from_coilset(coilsFilename, eqname, 25)
