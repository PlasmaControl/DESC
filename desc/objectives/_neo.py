import numpy as np
import subprocess
import re

from jax import core
from jax.interpreters import ad, batching

from desc.backend import jnp
from desc.derivatives import FiniteDiffDerivative
from desc.equilibrium import Equilibrium
from desc.vmec import VMECIO
from .objective_funs import _Objective


class NEOWrapper(_Objective):

    _scalar = False
    _linear = False

    def __init__(self, eq=None, target=0, weight=1, name="NEO"):

        super().__init__(eq=eq, target=target, weight=weight, name=name)

    def build(self, eq, use_jit=False, verbose=1):
        self._args = ["R_lmn", "Z_lmn", "L_lmn", "i_l", "p_l", "Psi"]
        self._dim_f = 1

        self.path = "/u/ddudt/DESC/"

        # equilibrium parameters
        self.sym = eq.sym
        self.L = eq.L
        self.M = eq.M
        self.N = eq.N
        self.NFP = eq.NFP
        self.spectral_indexing = eq.spectral_indexing
        self.pressure = eq.pressure
        self.iota = eq.iota

        # wout parameters
        self.ns = 8

        # booz parameters
        self.M_booz = 2 * eq.M
        self.N_booz = 2 * eq.N

        # neo parameters
        pass

        self.neo_compute = core.Primitive("neo")
        self.neo_compute.def_impl(self.compute_impl)
        ad.primitive_jvps[self.neo_compute] = self.compute_neo_jvp
        batching.primitive_batchers[self.neo_compute] = self.compute_neo_batch

        self._check_dimensions()
        self._set_dimensions(eq)

        self._built = True

    def compute(self, R_lmn, Z_lmn, L_lmn, p_l, i_l, Psi):
        args = (R_lmn, Z_lmn, L_lmn, p_l, i_l, Psi)
        return self.neo_compute.bind(*args)

    def compute_impl(self, *args):
        (R_lmn, Z_lmn, L_lmn, p_l, i_l, Psi) = args

        self.pressure.params = p_l
        self.iota.params = i_l
        eq = Equilibrium(
            sym=self.sym,
            L=self.L,
            M=self.M,
            N=self.N,
            NFP=self.NFP,
            spectral_indexing=self.spectral_indexing,
            R_lmn=R_lmn,
            Z_lmn=Z_lmn,
            L_lmn=L_lmn,
            pressure=self.pressure,
            iota=self.iota,
            Psi=float(Psi),
        )
        eq.surface = eq.get_surface_at(rho=1)
        eq.solved = True

        print("Saving VMEC wout file")
        VMECIO.save(eq, self.path + "wout_desc.nc", surfs=self.ns, verbose=0)

        self.write_booz()
        self.run_booz()

        self.write_neo()
        self.run_neo()

        eps_eff_32 = self.read_neo()
        return self._shift_scale(jnp.atleast_1d(eps_eff_32))

    def compute_neo_jvp(self, values, tangents):
        R_lmn, Z_lmn, L_lmn, p_l, i_l, Psi = values
        primal_out = jnp.atleast_1d(0.0)

        n = len(values)
        argnum = np.arange(0, n, 1)

        jvp = FiniteDiffDerivative.compute_jvp(
            self.compute, argnum, tangents, *values, rel_step=1e-4
        )

        return (primal_out, jvp)

    def compute_neo_batch(self, values, axis):
        numdiff = len(values[0])

        res = jnp.array([0.0])

        for i in range(numdiff):
            R_lmn = values[0][i]
            Z_lmn = values[1][i]
            L_lmn = values[2][i]
            p_l = values[3][i]
            i_l = values[4][i]
            Psi = values[5][i]

            res = jnp.vstack([res, self.compute(R_lmn, Z_lmn, L_lmn, p_l, i_l, Psi)])

        res = res[1:]

        return res, axis[0]

    def write_booz(self):
        """Write BOOZ_XFORM input file."""
        print("Writing BOOZ_XFORM input file")
        f = open(self.path + "in_booz.desc", "w")
        f.write("{} {}\n".format(self.M_booz, self.N_booz))
        f.write("'desc'\n")
        f.write("{}\n".format(self.ns))
        f.close()

    def write_neo(self):
        """Write NEO input file."""
        print("Writing NEO input file")
        f = open(self.path + "neo_in.desc", "w")
        f.write("'#'\n'#'\n'#'\n")
        f.write("boozmn_desc.nc\n")
        f.write("neo_out.desc\n")
        f.write(" 1\n {}\n".format(self.ns))
        f.write(" 200\n 200\n 0\n 0\n 50\n 1\n 0.01\n 100\n 50\n 500\n 5000\n")
        f.write(" 0\n 1\n 0\n 0\n 2\n 0\n 0\n 0\n 0\n 0\n")
        f.write("'#'\n'#'\n'#'\n")
        f.write(" 0\n")
        f.write("neo_cur_desc\n")
        f.write(" 200\n 2\n 0\n")
        f.close()

    def read_neo(self):
        """Read NEO output file."""
        print("Reading NEO output file")
        num_form = r"[-+]?\ *\d*\.?\d*(?:[Ee]\ *[-+]?\ *\d+)?"
        f = open(self.path + "neo_out.desc", "r")
        f.seek(0)
        line = f.readline()
        nums = [float(x) for x in re.findall(num_form, line) if re.search(r"\d", x)]
        f.close()
        # nums = [SURFACE_LABEL, EPSTOT, REFF, IOTA, B_REF, R_REF]
        return nums[1]

    def run_booz(self):
        """Run BOOZ_XFORM."""
        print("Running BOOZ_XFORM")
        f = open("booz.out", "w")
        cmd = ["xbooz_xform", "in_booz.desc"]
        p = subprocess.run(cmd, stdout=f)
        f.close()

    def run_neo(self):
        """Run NEO."""
        print("Running NEO")
        f = open("neo.out", "w")
        cmd = ["xneo", "desc"]
        p = subprocess.run(cmd, stdout=f)
        f.close()
