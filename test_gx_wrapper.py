import desc.io
from desc.objectives._gx_wrapper import GX_Wrapper

eq = desc.io.load('desc/examples/W7X_output.h5')[-1]
g = GX_Wrapper(eq=eq,nzgrid=64)
g.write_geo()
