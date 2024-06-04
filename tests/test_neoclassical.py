"""Test for neoclassical transport compute functions."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from desc.backend import jnp
from desc.equilibrium import Equilibrium
from desc.equilibrium.coords import rtz_grid
from desc.vmec import VMECIO


@pytest.mark.unit
def test_field_line_average():
    """Test that field line average converges to surface average."""
    eq = Equilibrium.load(
        "tests/inputs/DESC_from_NAE_O_r1_precise_QI_plunk_fixed_bdry_r0"
        ".15_L_9_M_9_N_24_output.h5"
    )
    rho = np.linspace(0, 1, 5)
    alpha = np.array([0])
    L = 10 * np.pi  # Large enough to pass the test, apparently.
    zeta = np.linspace(0, L, 20)
    grid = rtz_grid(
        eq, rho, alpha, zeta, coordinates="raz", period=(np.inf, 2 * np.pi, np.inf)
    )
    data = eq.compute(["L|r,a", "G|r,a", "V_r(r)"], grid=grid)
    np.testing.assert_allclose(
        np.squeeze(data["L|r,a"] / data["G|r,a"]),
        grid.compress(data["V_r(r)"]) / (4 * np.pi**2),
        rtol=2e-2,
    )


@pytest.mark.unit
def test_effective_ripple():
    """Compare DESC effective ripple against NEO STELLOPT."""
    eq = Equilibrium.load(
        "tests/inputs/DESC_from_NAE_O_r1_precise_QI_plunk_fixed_bdry_r0"
        ".15_L_9_M_9_N_24_output.h5"
    )
    rho = np.linspace(0, 1, 40)
    # TODO: Here's a potential issue, resolve with 2d spline.
    knots = np.linspace(0, 100 * np.pi, 1000)
    grid = rtz_grid(
        eq,
        rho,
        np.array([0]),
        knots,
        coordinates="raz",
        period=(np.inf, 2 * np.pi, np.inf),
    )
    data = eq.compute(
        "effective ripple",
        grid=grid,
        batch=False,
    )
    assert np.isfinite(data["effective ripple"]).all()
    eps_eff = grid.compress(data["effective ripple"])

    # Plot DESC effective ripple.
    fig, ax = plt.subplots()
    rho = grid.compress(grid.nodes[:, 0])
    ax.plot(rho, eps_eff, marker="o")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\epsilon_{\text{eff}}^{3/2}$")
    ax.set_title("DESC effective ripple ε¹ᐧ⁵")
    plt.tight_layout()
    plt.show()
    plt.close()

    # Plot NEO effective ripple.
    # This should be ε¹ᐧ⁵, but need to check if it's just ε.
    neo_eps = np.array(
        read_neo_out("tests/inputs/neo_out.QI_plunk_fixed_surf_r0.15_N_24_hires_ns99")
    )
    assert neo_eps.ndim == 1
    neo_rho = np.sqrt(np.linspace(1 / (neo_eps.size + 1), 1, neo_eps.size))
    fig, ax = plt.subplots()
    ax.plot(neo_rho, neo_eps, marker="o")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\epsilon_{\text{eff}}^{3/2}$")
    ax.set_title("NEO effective ripple ε¹ᐧ⁵?")
    plt.tight_layout()
    plt.show()
    plt.close()

    # Plot DESC vs NEO effective ripple.
    fig, ax = plt.subplots()
    ax.plot(rho, eps_eff, marker="o", label="ε¹ᐧ⁵ DESC")
    # Looks more similar when neo_eps -> neo_eps**1.5...
    ax.plot(neo_rho, neo_eps, marker="o", label="ε¹ᐧ⁵? NEO")
    ax.legend()
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\epsilon_{\text{eff}}^{3/2}$")
    ax.set_title("DESC vs. NEO effective ripple")
    plt.tight_layout()
    plt.show()
    plt.close()


@pytest.mark.unit
def test_Gamma_c():
    """Compare DESC effective ripple against NEO STELLOPT."""
    eq = Equilibrium.load(
        "tests/inputs/DESC_from_NAE_O_r1_precise_QI_plunk_fixed_bdry_r0"
        ".15_L_9_M_9_N_24_output.h5"
    )
    rho = jnp.linspace(0, 1, 20)
    alpha = jnp.array([0])
    # TODO: Here's a potential issue, resolve with 2d spline.
    knots = jnp.linspace(-30 * jnp.pi, 30 * jnp.pi, 2000)
    grid = rtz_grid(
        eq, rho, alpha, knots, coordinates="raz", period=(jnp.inf, 2 * jnp.pi, jnp.inf)
    )
    data = eq.compute("Gamma_c", grid=grid)
    assert np.isfinite(data["Gamma_c"]).all()
    Gamma_c = grid.compress(data["Gamma_c"])

    fig, ax = plt.subplots()
    ax.plot(rho, Gamma_c, marker="o")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\Gamma_{c}$")
    ax.set_title(r"DESC $\Gamma_{c}$")
    plt.tight_layout()
    plt.show()
    plt.close()


class NEOWrapper:
    """Class to easily make NEO and BOOZxform inputs from DESC equilibria."""

    def __init__(self, basename, eq=None, ns=None, M_booz=None, N_booz=None):

        self.basename = basename
        self.M_booz = M_booz
        self.N_booz = N_booz

        if eq:
            self.build(eq, basename, ns=ns)

    def build(self, eq, basename, **kwargs):
        """Pass as input an already-solved Equilibrium from DESC."""
        # equilibrium parameters
        self.eq = eq
        self.sym = eq.sym
        self.L = eq.L
        self.M = eq.M
        self.N = eq.N
        self.NFP = eq.NFP
        self.spectral_indexing = eq.spectral_indexing
        self.pressure = eq.pressure
        self.iota = eq.iota
        self.current = eq.current

        # wout parameters
        self.ns = kwargs.get("ns", 256)

        # booz parameters
        if self.M_booz is None:
            self.M_booz = 3 * eq.M + 1
        if self.N_booz is None:
            self.N_booz = 3 * eq.N

        # basename for files
        self.basename = basename

    def save_VMEC(self):
        """Save VMEC file."""
        self.eq.solved = True  # must set this for NEO to run correctly
        print(f"Saving VMEC wout file to wout_{self.basename}.nc")
        VMECIO.save(self.eq, f"wout_{self.basename}.nc", surfs=self.ns, verbose=0)

    def write_booz(self):
        """Write BOOZ_XFORM input file."""
        print(f"Writing BOOZ_XFORM input file to in_booz.{self.basename}")
        with open(f"in_booz.{self.basename}", "w+") as f:
            f.write("{} {}\n".format(self.M_booz, self.N_booz))
            f.write(f"'{self.basename}'\n")
            f.write(
                "\n".join([str(x) for x in range(2, self.ns + 1)])
            )  # surface indices

    def write_neo(self, N_particles=150):
        """Write NEO input file."""
        print(f"Writing NEO input file neo_in.{self.basename}")
        with open(f"neo_in.{self.basename}", "w+") as f:
            f.write("'#'\n'#'\n'#'\n")
            f.write(f" boozmn_{self.basename}.nc\n")  # booz out file
            f.write(f" neo_out.{self.basename}\n")  # desired NEO out file
            f.write(f" {self.ns-1}\n")  # number of surfaces
            f.write(
                " ".join([str(x) for x in range(2, self.ns + 1)]) + "\n"
            )  # surface indices
            f.write(
                " 300 ! number of theta points\n "
                "300 ! number of zeta points\n "
                + f"\n 0\n 0\n {N_particles} ! number of test particles\n "
                f"1 ! 1 = singly trapped particles\n "
                + "0.001 ! integration accuracy\n 100 ! number of poloidal bins\n "
                "10 ! integration steps per field period\n "
                + "500 ! min number of field periods\n "
                "5000 ! max number of field periods\n"
            )  # default values
            f.write(
                " 0\n 1\n 0\n 0\n 2 ! 2 = reference |B| used is max on each surface"
                " \n 0\n 0\n 0\n 0\n 0\n"
            )
            f.write("'#'\n'#'\n'#'\n")
            f.write(" 0\n")
            f.write(f"neo_cur_{self.basename}\n")
            f.write(" 200\n 2\n 0\n")


def read_neo_out(fname):
    """Read ripple from text file."""

    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    # import all data from text file as an array
    with open(f"{fname}") as f:
        array = np.array([[float(x) for x in line.split()] for line in f])

    eps_eff = array[:, 1]  # epsilon_eff^(3/2) is the second column
    nans, x = nan_helper(eps_eff)  # find NaN values

    # replace NaN values with linear interpolation
    eps_eff[nans] = np.interp(x(nans), x(~nans), eps_eff[~nans])

    return eps_eff
