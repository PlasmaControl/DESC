"""Tests for SLAMPerpElongation and the fixed-bounce-field J|| objectives."""

import numpy as np
import pytest

from desc.backend import jax
from desc.equilibrium import Equilibrium
from desc.examples import get
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid
from desc.objectives import (
    JparDjdaNorm,
    JparDjdaTopK,
    JparOrder,
    JparSingleWell,
    SLAMPerpElongation,
)


def _ellipse_torus(kappa):
    # Z sin coefficient negative: right-handed (theta, zeta)
    surf = FourierRZToroidalSurface(
        R_lmn=[3.0, 1.0], Z_lmn=[-kappa], modes_R=[[0, 0], [1, 0]], modes_Z=[[-1, 0]]
    )
    return Equilibrium(surface=surf, L=2, M=2, N=0, sym=True)


@pytest.mark.unit
@pytest.mark.parametrize("kappa", [1.0, 2.0])
def test_perp_elongation_ellipse(kappa):
    """A torus of elliptical section reads a/b on planes normal to its axis."""
    eq = _ellipse_torus(kappa)
    zeta = np.array([0.5, 1.0, 1.5])
    grid = LinearGrid(
        rho=1.0,
        theta=np.linspace(0, 2 * np.pi, 256, endpoint=False),
        zeta=np.linspace(0.2, 1.8, 801),
        NFP=1,
        sym=False,
    )
    obj = SLAMPerpElongation(eq, zeta, grid, bandwidth=0.01, rcut=5.0)
    obj.build(verbose=0)
    np.testing.assert_allclose(obj.sections(eq.params_dict), kappa, rtol=2e-3)
    np.testing.assert_allclose(obj.compute(eq.params_dict), kappa, rtol=2e-3)


@pytest.mark.unit
def test_perp_elongation_gradient():
    """The perpendicular elongation is differentiable in the boundary."""
    eq = _ellipse_torus(2.0)
    grid = LinearGrid(
        rho=1.0,
        theta=np.linspace(0, 2 * np.pi, 64, endpoint=False),
        zeta=np.linspace(0.6, 1.4, 201),
        NFP=1,
        sym=False,
    )
    obj = SLAMPerpElongation(eq, [1.0], grid, rcut=5.0)
    obj.build(verbose=0)
    g = jax.grad(obj.compute)(eq.params_dict)
    assert all(np.all(np.isfinite(v)) for v in jax.tree_util.tree_leaves(g))
    assert np.linalg.norm(g["R_lmn"]) > 0


@pytest.mark.unit
def test_jparallel_objectives():
    """J|| objectives build, give finite residuals, and top-k is the worst of norm."""
    eq = get("precise_QA")
    kw = dict(num_alpha=8, nzeta=200, M_booz=4, N_booz=4)
    rho, t = [0.5, 0.7, 0.9], [0.3, 0.6]
    norm = JparDjdaNorm(eq, rho, t, rho_lo=0.5, rho_hi=0.9, **kw)
    topk = JparDjdaTopK(eq, rho, t, top_k=3, rho_lo=0.5, rho_hi=0.9, **kw)
    order = JparOrder(eq, rho, t, **kw)
    well = JparSingleWell(eq, rho, **kw)
    for obj in (norm, topk, order, well):
        obj.build(verbose=0)
        f = np.asarray(obj.compute(eq.params_dict))
        assert f.shape == (obj.dim_f,)
        assert np.all(np.isfinite(f))
    r = np.asarray(norm.compute(eq.params_dict)) * np.sqrt(3 * 8)
    r = r.reshape(len(t), 3, 8)
    top = np.sort(np.abs(r), axis=-1)[..., ::-1][..., :3]
    rk = np.asarray(topk.compute(eq.params_dict)) * np.sqrt(3 * 3)
    np.testing.assert_allclose(np.abs(rk).reshape(len(t), 3, 3), top, rtol=1e-10)
