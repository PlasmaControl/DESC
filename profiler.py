from desc.input_reader import InputReader
from desc.continuation import solve_eq_continuation

ir = InputReader(cl_args=['examples/DESC/HELIOTRON'])
iterations, timer = solve_eq_continuation(ir.inputs)

###

from desc.backend import Tristate
from desc.grid import LinearGrid, ConcentricGrid
from desc.basis import PowerSeries, DoubleFourierSeries, FourierZernikeBasis
from desc.transform import Transform

L = 8
M = 8
N = 8
Mnodes = 12
Nnodes = 12
NFP = 5

stell_sym = True
if stell_sym:
    R_sym = Tristate(True)
    Z_sym = Tristate(False)
    L_sym = Tristate(False)
else:
    R_sym = Tristate(None)
    Z_sym = Tristate(None)
    L_sym = Tristate(None)

zern_mode = 'ansi'
node_mode = 'cheb1'

# grids
RZ_grid = ConcentricGrid(Mnodes, Nnodes, NFP=NFP, sym=stell_sym,
                         axis=False, index=zern_mode, surfs=node_mode)
L_grid = LinearGrid(M=Mnodes, N=2*Nnodes+1, NFP=NFP, sym=stell_sym)

# bases
R_basis = FourierZernikeBasis(L=L, M=M, N=N,
                              NFP=NFP, sym=R_sym, index=zern_mode)
Z_basis = FourierZernikeBasis(L=L, M=M, N=N,
                              NFP=NFP, sym=Z_sym, index=zern_mode)
L_basis = DoubleFourierSeries(M=M, N=N, NFP=NFP, sym=L_sym)
Rb_basis = DoubleFourierSeries(M=M, N=N, NFP=NFP, sym=L_sym)
Zb_basis = DoubleFourierSeries(M=M, N=N, NFP=NFP, sym=L_sym)
P_basis = PowerSeries(L=L)
I_basis = PowerSeries(L=L)

# transforms
R_transform = Transform(RZ_grid, R_basis, derivs=3)
Z_transform = Transform(RZ_grid, Z_basis, derivs=3)
R1_transform = Transform(L_grid, R_basis)
Z1_transform = Transform(L_grid, Z_basis)
L_transform = Transform(L_grid,  L_basis, derivs=0)
P_transform = Transform(RZ_grid, P_basis, derivs=1)
I_transform = Transform(RZ_grid, I_basis, derivs=1)
