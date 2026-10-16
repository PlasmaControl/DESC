"""Test that desc.external.agni is compatible with the installed agnimhd."""

import numpy as np
import pytest

from desc.io import load
from desc.objectives import ObjectiveFunction

agnimhd = pytest.importorskip("agnimhd")


@pytest.mark.unit
def test_agni_stability():
    """The objective builds and computes through ObjectiveFunction."""
    from desc.external.agni import AgniStability

    eq = load("./tests/inputs/AGNI_QH_lowres.h5")  # low resolution QH, NFP=4
    staircase = dict(eps=1e-2, x_0=0.65, m_1=2.0, m_2=3.0)
    basis = agnimhd.Basis(8, 8, 3, radial="lobatto", automorphism=staircase)
    obj = ObjectiveFunction(AgniStability(eq, basis))
    obj.build(verbose=0)
    # gamma^2 of family 0 on this basis, from a dense solve
    np.testing.assert_allclose(obj.compute_scaled_error(obj.x(eq)), 2.278446950686e-3)
