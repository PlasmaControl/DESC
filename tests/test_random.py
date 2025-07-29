"""Tests for random surfaces, profiles etc."""

import numpy as np
import pytest

from desc.equilibrium import Equilibrium
from desc.random import random_pressure, random_surface


@pytest.mark.unit
def test_random_pressure():
    """Test that randomly generated profile is monotonic, has correct scaling etc."""
    rng = np.random.default_rng(0)
    p = random_pressure(L=8, p0=(1e3, 1e4), rng=rng)
    assert p.basis.sym == "even"
    assert 1e3 <= p(np.array([0.0])) <= 1e4
    assert p.basis.L == 8  # symmetric, so should be 4 params up to order 8
    dp = p(np.linspace(0, 1, 10), dr=1)
    assert np.all(dp <= 0)  # can't use array_less because that doesn't do <=


@pytest.mark.unit
def test_random_surface():
    """Test that randomly generated surface is "sensible"."""
    rng = np.random.default_rng(0)
    surf = random_surface(
        M=4,
        N=4,
        R0=(5, 10),
        R_scale=(0.5, 2),
        Z_scale=(0.5, 2),
        NFP=(1, 3),
        sym=True,
        alpha=(1, 4),
        beta=(1, 4),
        rng=rng,
    )
    assert surf.sym
    assert 1 <= surf.NFP <= 3
    assert surf.M == 4
    assert surf.N == 4
    assert surf._compute_orientation() == 1

    eq = Equilibrium(surface=surf)
    R0 = eq.compute("R0")["R0"]
    assert 5 <= R0 <= 10
    AR = eq.compute("R0/a")["R0/a"]
    # should be ~ R0/sqrt(R_scale*Z_scale), allowing for random variation
    assert 2.5 <= AR <= 20
    assert eq.is_nested()

    # same stuff for non-symmetric
    rng = np.random.default_rng(0)
    surf = random_surface(
        M=4,
        N=4,
        R0=(5, 10),
        R_scale=(0.5, 2),
        Z_scale=(0.5, 2),
        NFP=(1, 3),
        sym=False,
        alpha=(1, 4),
        beta=(1, 4),
        rng=rng,
    )
    assert not surf.sym
    assert 1 <= surf.NFP <= 3
    assert surf.M == 4
    assert surf.N == 4
    assert surf._compute_orientation() == 1

    eq = Equilibrium(surface=surf)
    R0 = eq.compute("R0")["R0"]
    assert 5 <= R0 <= 10
    AR = eq.compute("R0/a")["R0/a"]
    # should be ~ R0/sqrt(R_scale*Z_scale), allowing for random variation
    assert 2.5 <= AR <= 20
    assert eq.is_nested()
