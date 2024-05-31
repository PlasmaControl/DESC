"""Test for neoclassical transport compute functions."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import quadax

from desc.compute._neoclassical import _poloidal_average, poloidal_leggauss, vec_quadax
from desc.equilibrium import Equilibrium
from desc.equilibrium.coords import rtz_grid
from desc.equilibrium.equilibrium import compute_raz_data
from desc.vmec import VMECIO


@pytest.mark.unit
def test_field_line_average():
    """Test that field line average converges to surface average."""
    eq = Equilibrium.load(
        "tests/inputs/DESC_from_NAE_O_r1_precise_QI_plunk_fixed_bdry_r0"
        ".15_L_9_M_9_N_24_output.h5"
    )
    resolution = 10
    rho = np.linspace(0, 1, resolution)

    # Surface average field lines truncated at 1 toroidal transit.
    alpha, w = poloidal_leggauss(resolution)
    L = 2 * np.pi
    zeta = np.linspace(0, L, resolution)
    grid = rtz_grid(eq, rho, alpha, zeta, coordinates="raz")
    grid.source_grid.poloidal_weight = w
    data = compute_raz_data(eq, grid, ["L|r,a", "G|r,a"], names_1dr=["V_r(r)"])
    np.testing.assert_allclose(
        _poloidal_average(grid.source_grid, data["L|r,a"] / data["G|r,a"]),
        grid.compress(data["V_r(r)"]) / (4 * np.pi**2),
        rtol=2e-2,
    )

    # Now for field line with large L.
    L = 10 * np.pi  # Large enough to pass the test, apparently.
    zeta = np.linspace(0, L, resolution * 2)
    grid = rtz_grid(eq, rho, 0, zeta, coordinates="raz")
    data = compute_raz_data(eq, grid, ["L|r,a", "G|r,a"], names_1dr=["V_r(r)"])
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
    grid = rtz_grid(
        eq,
        radial=np.linspace(0, 1, 40),
        poloidal=np.array([0]),
        toroidal=np.linspace(0, 100 * np.pi, 1000),
        coordinates="raz",
    )
    data = compute_raz_data(
        eq,
        grid,
        ["B^zeta", "|B|", "|B|_z|r,a", "|grad(psi)|", "kappa_g", "L|r,a"],
        names_0d=["R0"],
        names_1dr=["min_tz |B|", "max_tz |B|", "V_r(r)", "psi_r", "S(r)"],
    )
    quad = vec_quadax(quadax.quadgk)  # noqa: F841
    data = eq.compute(
        "effective ripple raw",
        grid=grid,
        data=data,
        override_grid=False,
        # batch=True,  # noqa: E800
        # quad=quad,  # noqa: E800
    )
    assert np.all(np.isfinite(data["effective ripple raw"]))
    rho = grid.compress(grid.nodes[:, 0])
    raw = grid.compress(data["effective ripple raw"])

    # Workaround until eq.compute() is fixed to only compute dependencies
    # that are needed for the requested computation. (So don't compute
    # dependencies of things already in data).
    data_R0 = eq.compute("R0")
    for key in data_R0:
        if key not in data:
            # Need to add R0's dependencies which are surface functions of zeta
            # aren't attempted to be recomputed on grid_desc.
            data[key] = data_R0[key]
    data = eq.compute("effective ripple", grid=grid, data=data)
    assert np.all(np.isfinite(data["effective ripple"]))
    eps_eff = grid.compress(data["effective ripple"])

    # Plot DESC effective ripple.
    fig, ax = plt.subplots(2)
    ax[0].plot(rho, raw, marker="o")
    ax[0].set_xlabel(r"$\rho$")
    ax[0].set_ylabel("effective ripple raw")
    ax[0].set_title(r"∫ dλ λ⁻² B₀⁻¹ $\langle$ ∑ⱼ Hⱼ²/Iⱼ $\rangle$")
    ax[1].plot(rho, eps_eff, marker="o")
    ax[1].set_xlabel(r"$\rho$")
    ax[1].set_ylabel(r"$\epsilon_{\text{eff}}^{3/2}$")
    ax[1].set_title(
        r"ε¹ᐧ⁵ = π/(8√2) (R₀(∂V/∂ψ)/S)² ∫ dλ λ⁻² B₀⁻¹ $\langle$ ∑ⱼ Hⱼ²/Iⱼ $\rangle$"
    )
    fig.suptitle("DESC effective ripple")
    plt.tight_layout()
    plt.show()
    plt.close()

    # Plot neo effective ripple. What are its units?
    # This should be ε¹ᐧ⁵, but need to check if it's just ε.
    neo_eps = np.array(
        read_neo_out("tests/inputs/neo_out.QI_plunk_fixed_surf_r0.15_N_24_hires_ns99")
    )
    assert neo_eps.ndim == 1
    neo_rho = np.sqrt(np.linspace(1 / (neo_eps.size + 1), 1, neo_eps.size))
    fig, ax = plt.subplots()
    ax.plot(neo_rho, neo_eps, marker="o", label="NEO high resolution")
    ax.legend()
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\epsilon_{\text{eff}}^{3/2}$")
    ax.set_title("NEO effective ripple")
    plt.tight_layout()
    plt.show()
    plt.close()

    # Plot DESC vs NEO effective ripple.
    fig, ax = plt.subplots()
    ax.plot(rho, eps_eff, marker="o", label="ε¹ᐧ⁵ DESC (low?) resolution")
    ax.plot(neo_rho, neo_eps, marker="o", label="(ε¹ᐧ⁵?) NEO high resolution")
    ax.legend()
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\epsilon_{\text{eff}}^{3/2}$")
    ax.set_title(
        r"ε¹ᐧ⁵ = π/(8√2) (R₀(∂V/∂ψ)/S)² ∫ dλ λ⁻² B₀⁻¹ $\langle$ ∑ⱼ Hⱼ²/Iⱼ $\rangle$"
    )
    fig.suptitle("DESC vs. NEO effective ripple")
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
