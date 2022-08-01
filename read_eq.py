from desc.plotting import plot_section, plot_comparison, plot_boozer_surface
import desc.io
import sys
import matplotlib.pyplot as plt

fname = sys.argv[1]
eq = desc.io.load(fname)

print("The aspect ratio is " + str(eq.compute("V")["R0/a"]))
#fig, ax = plot_comparison(eqs=[eq],labels=["constrained"])
fig, ax = plot_boozer_surface(eq)
plt.show()
