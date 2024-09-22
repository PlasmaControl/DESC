"""Test for neoclassical transport compute functions."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from tests.test_plotting import tol_1d

from desc.examples import get
from desc.grid import LinearGrid
from desc.utils import setdefault
from desc.vmec import VMECIO


@pytest.mark.unit
def test_field_line_average():
    """Test that field line average converges to surface average."""
    rho = np.array([1])
    alpha = np.array([0])
    eq = get("DSHAPE")
    iota_grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
    iota = iota_grid.compress(eq.compute("iota", grid=iota_grid)["iota"]).item()
    # For axisymmetric devices, one poloidal transit must be exact.
    zeta = np.linspace(0, 2 * np.pi / iota, 25)
    grid = eq.get_rtz_grid(rho, alpha, zeta, coordinates="raz")
    data = eq.compute(["<L|r,a>", "<G|r,a>", "V_r(r)"], grid=grid)
    np.testing.assert_allclose(
        data["<L|r,a>"] / data["<G|r,a>"], data["V_r(r)"] / (4 * np.pi**2), rtol=1e-3
    )
    assert np.all(np.sign(data["<L|r,a>"]) > 0)
    assert np.all(np.sign(data["<G|r,a>"]) > 0)

    # Otherwise, many toroidal transits are necessary to sample surface.
    eq = get("W7-X")
    zeta = np.linspace(0, 40 * np.pi, 300)
    grid = eq.get_rtz_grid(rho, alpha, zeta, coordinates="raz")
    data = eq.compute(["<L|r,a>", "<G|r,a>", "V_r(r)"], grid=grid)
    np.testing.assert_allclose(
        data["<L|r,a>"] / data["<G|r,a>"], data["V_r(r)"] / (4 * np.pi**2), rtol=1e-3
    )
    assert np.all(np.sign(data["<L|r,a>"]) > 0)
    assert np.all(np.sign(data["<G|r,a>"]) > 0)


@pytest.mark.unit
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_effective_ripple():
    """Test effective ripple with W7-X."""
    eq = get("W7-X")
    rho = np.linspace(0, 1, 10)
    alpha = np.array([0])
    zeta = np.linspace(0, 20 * np.pi, 1000)
    grid = eq.get_rtz_grid(rho, alpha, zeta, coordinates="raz")
    data = eq.compute("effective ripple", grid=grid)
    assert np.isfinite(data["effective ripple"]).all()
    eps_eff = grid.compress(data["effective ripple"])

    neo_rho, neo_eps = NeoIO.read("tests/inputs/neo_out.w7x")
    np.testing.assert_allclose(
        eps_eff, np.interp(rho, neo_rho, neo_eps), rtol=0.16, atol=1e-5
    )
    fig, ax = plt.subplots()
    ax.plot(rho, eps_eff, marker="o")
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_Gamma_c_Velasco():
    """Test Γ_c with W7-X."""
    eq = get("W7-X")
    rho = np.linspace(0, 1, 10)
    grid = eq.get_rtz_grid(
        rho, np.array([0]), np.linspace(0, 20 * np.pi, 1000), coordinates="raz"
    )
    data = eq.compute("Gamma_c Velasco", grid=grid)
    assert np.isfinite(data["Gamma_c Velasco"]).all()
    fig, ax = plt.subplots()
    ax.plot(rho, grid.compress(data["Gamma_c Velasco"]), marker="o")
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_Gamma_c():
    """Test Γ_c Nemov with W7-X."""
    eq = get("W7-X")
    rho = np.linspace(0, 1, 10)
    grid = eq.get_rtz_grid(
        rho, np.array([0]), np.linspace(0, 20 * np.pi, 1000), coordinates="raz"
    )
    data = eq.compute("Gamma_c", grid=grid)
    assert np.isfinite(data["Gamma_c"]).all()
    fig, ax = plt.subplots()
    ax.plot(rho, grid.compress(data["Gamma_c"]), marker="o")
    return fig


class NeoIO:
    """Class to interface with NEO."""

    def __init__(self, name, eq, ns=256, M_booz=None, N_booz=None):
        self.name = name
        self.vmec_file = f"{name}/wout_{name}.nc"
        self.booz_out_file = f"{name}/boozmn_{name}.nc"
        self.neo_in_file = f"{name}/neo_in.{name}"
        self.neo_out_file = f"neo_out.{name}"

        self.eq = eq
        self.ns = ns  # number of surfaces
        self.M_booz = setdefault(M_booz, 3 * eq.M + 1)
        self.N_booz = setdefault(N_booz, 3 * eq.N)

    @staticmethod
    def read(name):
        """Return ρ and ε¹ᐧ⁵ from NEO output with given name."""
        with open(name) as f:
            array = np.array([[float(x) for x in line.split()] for line in f])

        neo_eps = array[:, 1]
        neo_rho = np.sqrt(np.linspace(1 / (neo_eps.size + 1), 1, neo_eps.size))
        # replace bad values with linear interpolation
        good = np.isfinite(neo_eps)
        neo_eps[~good] = np.interp(neo_rho[~good], neo_rho[good], neo_eps[good])
        return neo_rho, neo_eps

    def write(self):
        """Write vmec, booz_xform, and neo files."""
        self._write_VMEC()
        self._write_neo()

        import booz_xform as bx

        b = bx.Booz_xform()
        b.read_wout(self.vmec_file)
        b.mboz = self.M_booz
        b.nboz = self.N_booz
        b.run()
        b.write_boozmn(self.booz_out_file)

    def _write_VMEC(self):
        self.eq.solved = True  # must set this for NEO to run correctly
        print(f"Writing VMEC wout to {self.vmec_file}")
        VMECIO.save(self.eq, self.vmec_file, surfs=self.ns, verbose=0)

    def _write_neo(self, num_pitch=125):
        """Write NEO input file."""
        print(f"Writing NEO input file to {self.neo_in_file}")
        with open(self.neo_in_file, "w+") as f:
            f.write("'#'\n'#'\n'#'\n")
            f.write(f" {self.booz_out_file}\n")
            f.write(f" {self.neo_out_file}\n")
            f.write(f" {self.ns-1}\n")
            # surface indices
            # https://github.com/PrincetonUniversity/STELLOPT/blob/develop/NEO/Sources/neo.f90
            f.write(" ".join([str(x) for x in range(2, self.ns + 1)]) + "\n")
            f.write(
                " 300 ! number of theta points\n "
                "300 ! number of zeta points\n "
                + f"\n 0\n 0\n {num_pitch} ! number of test particles\n "
                f"50 ! 1 = singly trapped particles\n "
                + "0.001 ! integration accuracy\n 100 ! number of poloidal bins\n "
                "50 ! integration steps per field period\n "
                + "500 ! min number of field periods\n "
                "2000 ! max number of field periods\n"
            )
            f.write(
                " 0\n 1\n 0\n 0\n 2 ! 2 = reference |B| used is max on each surface"
                " \n 0\n 0\n 0\n 0\n 0\n"
            )
            f.write("'#'\n'#'\n'#'\n")
            f.write(" 0\n")
            f.write(f"neo_cur_{self.name}\n")
            f.write(" 200\n 2\n 0\n")
