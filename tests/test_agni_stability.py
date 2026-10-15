"""Tests for the AgniStability objective (finite-n ideal MHD stability by AGNI)."""

import numpy as np
import pytest

from desc.equilibrium import Equilibrium
from desc.io import load
from desc.objectives import AgniStability, ObjectiveFunction

agnimhd = pytest.importorskip("agnimhd")

# low resolution QH equilibrium, NFP=4, with kinetic profiles
QH_FILE = "./tests/inputs/AGNI_QH_lowres.h5"
# gamma^2 of family 0 on the basis below, from a dense solve
GAMMA2_8x8x3 = 2.278446950686394e-3


def qh_basis():
    """8 x 8 x 3 nodes, radial nodes clustered at rho = 0.65 by a staircase map."""
    staircase = dict(eps=1e-2, x_0=0.65, m_1=2.0, m_2=3.0)
    return agnimhd.Basis(8, 8, 3, radial="lobatto", automorphism=staircase)


@pytest.mark.unit
def test_agni_stability_value():
    """The objective gives the reference gamma^2 and agnimhd's own value.

    agnimhd's own path is from_desc + growth_rate, on the same equilibrium and basis.
    """
    eq = load(QH_FILE)
    basis = qh_basis()
    obj = AgniStability(eq, basis)
    obj.build(verbose=0)
    gamma2 = obj.compute(eq.params_dict)
    assert gamma2.shape == (1,)
    np.testing.assert_allclose(gamma2[0], GAMMA2_8x8x3, rtol=1e-8)

    eq_data, diffmat = agnimhd.from_desc(eq, basis)
    np.testing.assert_allclose(
        gamma2[0], agnimhd.growth_rate(eq_data, diffmat), rtol=1e-10
    )


@pytest.mark.unit
def test_agni_stability_jacobian():
    """Reverse-mode Jacobian matches central finite differences.

    The Jacobian is computed as the optimizer computes it, and checked in the three
    coefficients gamma^2 depends on most.
    """
    eq = load(QH_FILE)
    obj = ObjectiveFunction(AgniStability(eq, qh_basis()), deriv_mode="blocked")
    obj.build(verbose=0)
    x = obj.x(eq)
    jac = np.asarray(obj.jac_scaled_error(x))[0]
    assert np.all(np.isfinite(jac))
    h = 1e-6
    for i in np.argsort(-np.abs(jac))[:3]:
        dx = np.zeros_like(x)
        dx[i] = h
        fd = (obj.compute_scaled_error(x + dx) - obj.compute_scaled_error(x - dx)) / 2
        np.testing.assert_allclose(fd[0] / h, jac[i], rtol=1e-4)


@pytest.mark.unit
def test_agni_stability_jd():
    """Jacobi-Davidson with its coarse level gives the default solver's gamma^2.

    The matrix-free Jacobi-Davidson solver is compared to the default (eigsh).
    """
    eq = load(QH_FILE)
    # JD needs the Fourier truncation (mpol, ntor) fixed on both levels
    basis = agnimhd.Basis(16, 12, 8, mpol=5, ntor=1)
    solver = agnimhd.SolverConfig(
        eigensolver="jd", sigma=2e-4, jd_tol=1e-4, jd_theta_tol=0.0
    )
    obj = AgniStability(eq, basis, solver=solver, coarse=basis.coarse(12, 6))
    obj.build(verbose=0)
    # eigsh on the same basis gives 9.138782581409812e-05
    np.testing.assert_allclose(
        obj.compute(eq.params_dict)[0], 9.138782581409812e-05, rtol=1e-8
    )


@pytest.mark.unit
def test_agni_stability_errors():
    """Inputs the objective cannot use are refused when it is made."""
    basis = qh_basis()
    with pytest.raises(ValueError, match="density"):
        AgniStability(Equilibrium(), basis, density=True)
    with pytest.raises(ValueError, match="coarse"):
        jd = agnimhd.SolverConfig(eigensolver="jd")
        AgniStability(Equilibrium(), basis, solver=jd, coarse=agnimhd.Basis(6, 4, 3))
