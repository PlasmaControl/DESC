import numpy as np
from desc.equilibrium import Equilibrium

eq_sym1 = Equilibrium(M=3, N=2, sym=True)
eq_asym1 = Equilibrium(M=3, N=2, sym=False)

eq_sym2 = eq_asym1.copy()
eq_asym2 = eq_sym1.copy()

eq_sym2.change_resolution(sym=True)
eq_asym2.change_resolution(sym=False)

np.testing.assert_allclose(eq_sym1.R_lmn, eq_sym2.R_lmn)
np.testing.assert_allclose(eq_sym1.Z_lmn, eq_sym2.Z_lmn)
np.testing.assert_allclose(eq_sym1.L_lmn, eq_sym2.L_lmn)
np.testing.assert_allclose(eq_sym1.Rb_lmn, eq_sym2.Rb_lmn)
np.testing.assert_allclose(eq_sym1.Zb_lmn, eq_sym2.Zb_lmn)

np.testing.assert_allclose(eq_asym1.R_lmn, eq_asym2.R_lmn)
np.testing.assert_allclose(eq_asym1.Z_lmn, eq_asym2.Z_lmn)
np.testing.assert_allclose(eq_asym1.L_lmn, eq_asym2.L_lmn)
np.testing.assert_allclose(eq_asym1.Rb_lmn, eq_asym2.Rb_lmn)
np.testing.assert_allclose(eq_asym1.Zb_lmn, eq_asym2.Zb_lmn)
