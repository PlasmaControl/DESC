"""Tests for the AgniStability objective (finite-n ideal MHD stability by AGNI)."""

import sys

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


def qh_basis(**kwargs):
    """8 x 8 x 3 nodes, radial nodes clustered at rho = 0.65 by a staircase map."""
    staircase = dict(eps=1e-2, x_0=0.65, m_1=2.0, m_2=3.0)
    return agnimhd.Basis(8, 8, 3, radial="lobatto", automorphism=staircase, **kwargs)


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


@pytest.mark.unit
def test_agni_stability_coarse_level():
    """With JD, build evaluates the equilibrium on the coarse nodes like from_desc."""
    eq = load(QH_FILE)
    basis = qh_basis(mpol=2, ntor=1)  # JD keeps mpol, ntor on both levels
    coarse = basis.coarse(6, 3)
    jd = agnimhd.SolverConfig(eigensolver="jd", sigma=1e-4)
    obj = AgniStability(eq, basis, solver=jd, coarse=coarse)
    obj.build(verbose=0)
    data = obj._equilibrium_data(eq.params_dict, obj._constants, coarse=True)
    ref, _ = agnimhd.from_desc(eq, coarse)
    assert data.resolution == ref.resolution == (8, 6, 3)
    for key in ("g_rr", "g_vv", "sqrtg", "J_sup_zeta", "iota", "p", "J_cross_grad_rho"):
        np.testing.assert_allclose(getattr(data, key), getattr(ref, key), rtol=1e-12)


@pytest.mark.unit
def test_agni_stability_needs_agnimhd(monkeypatch):
    """Without agnimhd the objective says how to install it."""
    monkeypatch.setitem(sys.modules, "agnimhd", None)
    with pytest.raises(ImportError, match="pip install agnimhd"):
        AgniStability(Equilibrium(), basis=None)
