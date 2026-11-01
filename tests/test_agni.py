"""Test that desc.external.agni is compatible with the installed agnimhd."""

import numpy as np
import pytest

from desc.io import load
from desc.objectives import ObjectiveFunction

agnimhd = pytest.importorskip("agnimhd")


@pytest.mark.unit
def test_agni_stability():
    """The objective builds and computes a finite γ² with the installed agnimhd."""
    from desc.external.agni import AgniStability

    eq = load("./tests/inputs/AGNI_QH_lowres.h5")  # low resolution QH, NFP=4
    basis = agnimhd.Basis(8, 8, 3, radial="lobatto")
    obj = ObjectiveFunction(AgniStability(eq, basis))
    obj.build(verbose=0)
    gamma2 = obj.compute_scaled_error(obj.x(eq))
    assert gamma2.shape == (1,) and np.isfinite(gamma2).all()
