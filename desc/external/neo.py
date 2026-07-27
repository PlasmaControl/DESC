"""A class to interface with the NEO code."""

from datetime import datetime

import numpy as np

from desc.utils import setdefault
from desc.vmec import VMECIO


class NeoIO:
    """Class to interface with NEO.

    Notes
    -----
    It is recommended to use DESC instead of NEO to compute or perform
    optimization with the effective ripple objective.
    """

    def __init__(self, name, eq, ns=256, M_booz=None, N_booz=None):
        self.name = name
        self.vmec_file = f"wout_{name}.nc"
        self.booz_file = f"boozmn_{name}.nc"
        self.neo_in_file = f"neo_in.{name}"
        self.neo_out_file = f"neo_out.{name}"

        self.eq = eq
        self.ns = ns  # number of surfaces
        self.M_booz = setdefault(M_booz, 6 * eq.M + 1)
        self.N_booz = setdefault(N_booz, 6 * eq.N)

    @staticmethod
    def read(name):
        """Return ρ and ε¹ᐧ⁵ from NEO output with given name."""
        neo_eps = np.loadtxt(name)[:, 1]
        neo_rho = np.sqrt(np.linspace(1 / (neo_eps.size + 1), 1, neo_eps.size))
        # replace bad values with linear interpolation
        good = np.isfinite(neo_eps)
        neo_eps[~good] = np.interp(neo_rho[~good], neo_rho[good], neo_eps[good])
        return neo_rho, neo_eps

    def write(self, *, theta_n=300, phi_n=300, num_pitch=150, nstep_per=500, **kwargs):
        """Write neo input file.

        Parameters
        ----------
        theta_n, phi_n : int
            Spline resolution.
        num_pitch : int
            Resolution for quadrature over velocity coordinate.
            Default is 150.
        nstep_per : int
            Runge-Kutta steps to perform within each field period. Default is 500.

            NEO employs a constant step size (2π/NFP/``nstep_per``),
            explicit Runge-Kutta scheme which has algebraic convergence of order 3/2
            for the bounce integrals of interest. The accuracy of the bounce points is
            also only found with first order accuracy in (2π/NFP/``nstep_per``) so the
            algebraic convergence is capped at 3/2 independent of the integrator.

        """
        print(f"Writing VMEC wout to {self.vmec_file}")
        VMECIO.save(self.eq, self.vmec_file, surfs=self.ns, verbose=0)
        self._write_booz()
        self._write_neo(
            theta_n=theta_n,
            phi_n=phi_n,
            num_pitch=num_pitch,
            nstep_per=nstep_per,
            **kwargs,
        )

    def _write_booz(self):
        print(f"Writing boozer output file to {self.booz_file}")
        import booz_xform as bx

        b = bx.Booz_xform()
        b.read_wout(self.vmec_file)
        b.mboz = self.M_booz
        b.nboz = self.N_booz
        b.run()
        b.write_boozmn(self.booz_file)

    def _write_neo(
        self,
        theta_n,
        phi_n,
        num_pitch,
        nstep_per,
        multra=1,
        acc_req=0.01,
        nbins=100,
        nstep_min=500,
        nstep_max=2000,
        verbose=2,
    ):
        """Write NEO file."""
        print(f"Writing NEO input file to {self.neo_in_file}")
        f = open(self.neo_in_file, "w")

        def writeln(s):
            f.write(str(s))
            f.write("\n")

        # https://princetonuniversity.github.io/STELLOPT/NEO
        writeln(f"'#' {datetime.now()}")
        writeln(f"'#' {self.vmec_file}")
        writeln(f"'#' M_booz={self.M_booz}. N_booz={self.N_booz}.")
        writeln(self.booz_file)
        writeln(self.neo_out_file)
        # Neo computes things on the so-called "half grid" between the full grid.
        # There are only ns - 1 surfaces there.
        writeln(self.ns - 1)
        # NEO starts indexing at 1 and does not compute on axis (index 1).
        surface_indices = " ".join(str(i) for i in range(2, self.ns + 1))
        writeln(surface_indices)
        writeln(theta_n)
        writeln(phi_n)
        writeln(0)
        writeln(0)
        writeln(num_pitch)
        writeln(multra)
        writeln(acc_req)
        writeln(nbins)
        writeln(nstep_per)
        writeln(nstep_min)
        writeln(nstep_max)
        writeln(0)
        writeln(verbose)
        writeln(0)
        writeln(0)
        writeln(2)
        writeln(0)
        writeln(0)
        writeln(0)
        writeln(0)
        writeln(0)
        writeln("'#'\n'#'\n'#'")
        writeln(0)
        writeln(f"neo_cur.{self.name}")
        writeln(200)
        writeln(2)
        writeln(0)
        f.close()
