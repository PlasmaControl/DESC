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
def test_agni_stability():
    """Value is the dense reference and the Jacobian matches finite differences.

    The Jacobian is the optimizer's reverse-mode one, checked in the two
    coefficients gamma^2 depends on most. One build, so it compiles once.
    """
    eq = load(QH_FILE)
    obj = ObjectiveFunction(AgniStability(eq, qh_basis()), deriv_mode="blocked")
    obj.build(verbose=0)
    x = obj.x(eq)
    np.testing.assert_allclose(obj.compute_scaled_error(x), [GAMMA2_8x8x3], rtol=1e-8)

    jac = np.asarray(obj.jac_scaled_error(x))[0]
    assert np.all(np.isfinite(jac))
    h = 1e-6
    for i in np.argsort(-np.abs(jac))[:2]:
        dx = np.zeros_like(x)
        dx[i] = h
        fd = (obj.compute_scaled_error(x + dx) - obj.compute_scaled_error(x - dx)) / 2
        np.testing.assert_allclose(fd[0] / h, jac[i], rtol=1e-4)


@pytest.mark.unit
def test_agni_stability_errors():
    """Inputs the objective cannot use are refused when it is made."""
    basis = qh_basis()
    with pytest.raises(ValueError, match="density"):
        AgniStability(Equilibrium(), basis, density=True)
    with pytest.raises(ValueError, match="coarse"):
        jd = agnimhd.SolverConfig(eigensolver="jd")
        AgniStability(Equilibrium(), basis, solver=jd, coarse=agnimhd.Basis(6, 4, 3))
