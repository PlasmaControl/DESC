from desc.plotting import plot_section, plot_comparison, plot_boozer_surface, plot_qs_error
import desc.io
import sys
import matplotlib.pyplot as plt

fname = sys.argv[1]
eq = desc.io.load(fname)

print("The aspect ratio is " + str(eq.compute("V")["R0/a"]))
fig, ax = plot_comparison(eqs=[eq],labels=["constrained"])
#fig, ax = plot_boozer_surface(eq)
#fig, ax = plot_qs_error(eq,helicity=(1,eq.NFP),fT=True,fB=False,fC=False,rho=10)
#fig, ax = plot_section(eq=eq,name='|F|',log=True)
plt.show()
