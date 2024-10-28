"""Test for neoclassical transport compute functions."""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pytest
from tests.test_plotting import tol_1d

from desc.examples import get
from desc.grid import LinearGrid
from desc.objectives import (
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixPressure,
    FixPsi,
    ForceBalance,
    GammaC,
    Gammad,
    ObjectiveFunction,
)
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
    grid = eq._get_rtz_grid(rho, alpha, zeta, coordinates="raz")
    data = eq.compute(["<L|r,a>", "<G|r,a>", "V_r(r)"], grid=grid)
    np.testing.assert_allclose(
        data["<L|r,a>"] / data["<G|r,a>"], data["V_r(r)"] / (4 * np.pi**2), rtol=1e-3
    )
    assert np.all(np.sign(data["<L|r,a>"]) > 0)
    assert np.all(np.sign(data["<G|r,a>"]) > 0)

    # Otherwise, many toroidal transits are necessary to sample surface.
    eq = get("W7-X")
    zeta = np.linspace(0, 40 * np.pi, 300)
    grid = eq._get_rtz_grid(rho, alpha, zeta, coordinates="raz")
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
    grid = eq._get_rtz_grid(rho, alpha, zeta, coordinates="raz")
    data = eq.compute("effective ripple", grid=grid)
    assert np.isfinite(data["effective ripple"]).all()
    np.testing.assert_allclose(
        data["effective ripple 3/2"] ** (2 / 3),
        data["effective ripple"],
        err_msg="Bug in source grid logic in eq.compute.",
    )
    eps_32 = grid.compress(data["effective ripple 3/2"])
    fig, ax = plt.subplots()
    ax.plot(rho, eps_32, marker="o")

    neo_rho, neo_eps_32 = NeoIO.read("tests/inputs/neo_out.w7x")
    np.testing.assert_allclose(
        eps_32, np.interp(rho, neo_rho, neo_eps_32), rtol=0.16, atol=1e-5
    )
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_Gamma_c_Velasco():
    """Test Γ_c with W7-X."""
    eq = get("W7-X")
    rho = np.linspace(0, 1, 10)
    grid = eq._get_rtz_grid(
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
    grid = eq._get_rtz_grid(
        rho, np.array([0]), np.linspace(0, 20 * np.pi, 1000), coordinates="raz"
    )
    data = eq.compute("Gamma_c", grid=grid)
    assert np.isfinite(data["Gamma_c"]).all()
    fig, ax = plt.subplots()
    ax.plot(rho, grid.compress(data["Gamma_c"]), marker="o")
    return fig


@pytest.mark.unit
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_Gamma_d():
    """Test Γ_d computation with W7-X."""
    eq = get("W7-X")
    rho = np.linspace(0, 1, 10)
    grid = eq._get_rtz_grid(
        rho, np.array([0]), np.linspace(0, 20 * np.pi, 1000), coordinates="raz"
    )
    data = eq.compute("Gamma_d", grid=grid)
    assert np.isfinite(data["Gamma_d"]).all()
    fig, ax = plt.subplots()
    ax.plot(rho, grid.compress(data["Gamma_d"]), marker="o")
    return fig


@pytest.mark.regression
def test_Gamma_d_opt():
    """Test that an optimizatin with Gamma_d works without failing."""
    eq = get("ESTELL")
    with pytest.warns(UserWarning):
        eq.change_resolution(4, 4, 4, 8, 8, 8)
    k = 1

    alpha = np.array([0.0])
    rho = np.linspace(0.80, 0.95, 2)

    objective = ObjectiveFunction(
        (
            Gammad(
                eq=eq,
                rho=rho,
                alpha=alpha,
                deriv_mode="fwd",
                batch=False,
                num_pitch=3,
                num_quad=3,
                num_transit=2,
            ),
        ),
    )
    R_modes = np.vstack(
        (
            [0, 0, 0],
            eq.surface.R_basis.modes[
                np.max(np.abs(eq.surface.R_basis.modes), 1) > k, :
            ],
        )
    )
    Z_modes = eq.surface.Z_basis.modes[
        np.max(np.abs(eq.surface.Z_basis.modes), 1) > k, :
    ]
    constraints = (
        ForceBalance(eq),
        FixBoundaryR(eq=eq, modes=R_modes),
        FixBoundaryZ(eq=eq, modes=Z_modes),
        FixPressure(eq=eq),
        FixCurrent(eq=eq),
        FixPsi(eq=eq),
    )

    eq.optimize(
        objective=objective,
        constraints=constraints,
        maxiter=2,  # just testing that no errors occur during JIT/AD of the objective
    )


@pytest.mark.regression
def test_Gamma_d_opt_batch_true():
    """Test that an optimizatin with Gamma_d works without failing w/ batch=True."""
    eq = get("ESTELL")
    with pytest.warns(UserWarning):
        eq.change_resolution(4, 4, 4, 8, 8, 8)
    k = 1

    alpha = np.array([0.0])
    rho = np.linspace(0.80, 0.95, 2)

    objective = ObjectiveFunction(
        (
            Gammad(
                eq=eq,
                rho=rho,
                alpha=alpha,
                deriv_mode="fwd",
                batch=True,
                num_pitch=3,
                num_quad=3,
                num_transit=2,
            ),
        ),
    )
    R_modes = np.vstack(
        (
            [0, 0, 0],
            eq.surface.R_basis.modes[
                np.max(np.abs(eq.surface.R_basis.modes), 1) > k, :
            ],
        )
    )
    Z_modes = eq.surface.Z_basis.modes[
        np.max(np.abs(eq.surface.Z_basis.modes), 1) > k, :
    ]
    constraints = (
        ForceBalance(eq),
        FixBoundaryR(eq=eq, modes=R_modes),
        FixBoundaryZ(eq=eq, modes=Z_modes),
        FixPressure(eq=eq),
        FixCurrent(eq=eq),
        FixPsi(eq=eq),
    )

    eq.optimize(
        objective=objective,
        constraints=constraints,
        maxiter=2,  # just testing that no errors occur during JIT/AD of the objective
    )


@pytest.mark.regression
def test_Gamma_c_opt():
    """Test that an optimizatin with Gamma_c works without failing."""
    eq = get("ESTELL")
    with pytest.warns(UserWarning):
        eq.change_resolution(4, 4, 4, 8, 8, 8)
    k = 1

    alpha = np.array([0.0])
    rho = np.linspace(0.80, 0.95, 2)

    objective = ObjectiveFunction(
        (
            GammaC(
                eq=eq,
                rho=rho,
                alpha=alpha,
                deriv_mode="fwd",
                batch=False,
                num_pitch=3,
                num_quad=3,
                num_transit=2,
            ),
        ),
    )
    R_modes = np.vstack(
        (
            [0, 0, 0],
            eq.surface.R_basis.modes[
                np.max(np.abs(eq.surface.R_basis.modes), 1) > k, :
            ],
        )
    )
    Z_modes = eq.surface.Z_basis.modes[
        np.max(np.abs(eq.surface.Z_basis.modes), 1) > k, :
    ]
    constraints = (
        ForceBalance(eq),
        FixBoundaryR(eq=eq, modes=R_modes),
        FixBoundaryZ(eq=eq, modes=Z_modes),
        FixPressure(eq=eq),
        FixCurrent(eq=eq),
        FixPsi(eq=eq),
    )

    eq.optimize(
        objective=objective,
        constraints=constraints,
        maxiter=2,  # just testing that no errors occur during JIT/AD of the objective
    )


@pytest.mark.regression
def test_Gamma_c_opt_batch_True():
    """Test that an optimizatin with Gamma_c works without failing w/ batch=True."""
    eq = get("ESTELL")
    with pytest.warns(UserWarning):
        eq.change_resolution(4, 4, 4, 8, 8, 8)
    k = 1

    alpha = np.array([0.0])
    rho = np.linspace(0.80, 0.95, 2)

    objective = ObjectiveFunction(
        (
            GammaC(
                eq=eq,
                rho=rho,
                alpha=alpha,
                deriv_mode="fwd",
                batch=True,
                num_pitch=3,
                num_quad=3,
                num_transit=2,
            ),
        ),
    )
    R_modes = np.vstack(
        (
            [0, 0, 0],
            eq.surface.R_basis.modes[
                np.max(np.abs(eq.surface.R_basis.modes), 1) > k, :
            ],
        )
    )
    Z_modes = eq.surface.Z_basis.modes[
        np.max(np.abs(eq.surface.Z_basis.modes), 1) > k, :
    ]
    constraints = (
        ForceBalance(eq),
        FixBoundaryR(eq=eq, modes=R_modes),
        FixBoundaryZ(eq=eq, modes=Z_modes),
        FixPressure(eq=eq),
        FixCurrent(eq=eq),
        FixPsi(eq=eq),
    )

    eq.optimize(
        objective=objective,
        constraints=constraints,
        maxiter=2,  # just testing that no errors occur during JIT/AD of the objective
    )


class NeoIO:
    """Class to interface with NEO."""

    def __init__(self, name, eq, ns=256, M_booz=None, N_booz=None):
        self.name = name
        self.vmec_file = f"wout_{name}.nc"
        self.booz_file = f"boozmn.{name}"
        self.neo_in_file = f"neo_in.{name}"
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
        """Write neo input file."""
        self.eq.solved = True  # must set this for NEO to run correctly
        print(f"Writing VMEC wout to {self.vmec_file}")
        VMECIO.save(self.eq, self.vmec_file, surfs=self.ns, verbose=0)
        self._write_booz()
        self._write_neo()

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
        theta_n=200,
        phi_n=200,
        num_pitch=50,
        multra=1,
        acc_req=0.01,
        nbins=100,
        nstep_per=75,
        nstep_min=500,
        nstep_max=2000,
        verbose=2,
    ):
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
