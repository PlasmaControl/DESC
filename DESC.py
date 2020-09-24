from continuation import solve_eq_continuation
from plotting import plot_comparison, plot_vmec_comparison, plot_fb_err
from input_output import read_input, output_to_file, read_vmec_output
import jax
# from jax.config import config
# config.update("jax_debug_nans", True)


# input & output filenames
filename = 'HELIOTRON'
in_fname = 'benchmarks/DESC/'+filename+'.input'
inputs = read_input(in_fname)
out_fname = inputs['out_fname']

# solve equilibrium
equil_init,equil = solve_eq_continuation(inputs)

# output
output_to_file(out_fname,equil)

# plot comparison to initial guess
plot_comparison(equil_init,equil,'Initial','Solution')

# plot comparison to VMEC
vmec_data = read_vmec_output('benchmarks/VMEC/wout_'+filename+'.nc')
plot_vmec_comparison(vmec_data,equil)

# plot force balance error
plot_fb_err(equil,domain='real',normalize='global',log=True,cmap='plasma')
