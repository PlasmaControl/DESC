"""Tests for high order singular integration.

Hyperparams from Dhairya for greens ID test:

 M  q  Nu Nv   N=Nu*Nv    error
13 10  15 13       195 0.246547
13 10  30 13       390 0.0313301
13 12  45 13       585 0.0022925
13 12  60 13       780 0.00024359
13 12  75 13       975 1.97686e-05
19 16  90 19      1710 1.2541e-05
19 16 105 19      1995 2.91152e-06
19 18 120 19      2280 7.03463e-07
19 18 135 19      2565 1.60672e-07
25 20 150 25      3750 7.59613e-09
31 22 210 31      6510 1.04357e-09
37 24 240 37      8880 1.80728e-11
43 28 300 43     12900 2.14129e-12

"""

import numpy as np
import pytest

import desc
from desc.compute.geom_utils import xyz2rpz_vec
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid
from desc.singularities import (
    FFTInterpolator,
    singular_integral,
    virtual_casing_biot_savart,
)


@pytest.mark.unit
def test_singular_integral_greens_id():
    """Test high order singular integration using greens identity.

    Any harmonic function can be represented as the sum of a single layer and double
    layer potential:

    Φ(r) = -1/2π ∫ Φ(r) n⋅(r-r')/|r-r'|³ da + 1/2π ∫ dΦ/dn 1/|r-r'| da

    If we choose Φ(r) == 1, then we get

    1 + 1/2π ∫ n⋅(r-r')/|r-r'|³ da = 0

    So we integrate the kernel n⋅(r-r')/|r-r'|³ and can benchmark the residual.

    """
    eq = Equilibrium()
    Nv = np.array([30, 45, 60, 90, 120, 150, 240])
    Nu = np.array([13, 13, 13, 19, 19, 25, 37])
    ss = np.array([13, 13, 13, 19, 19, 25, 37])
    qs = np.array([10, 12, 12, 16, 18, 20, 24])
    es = np.array([0.4, 2e-2, 3e-3, 5e-5, 4e-6, 1e-6, 1e-9])
    eval_grid = LinearGrid(M=5, N=6, NFP=eq.NFP)

    for i, (m, n) in enumerate(zip(Nu, Nv)):
        src_grid = LinearGrid(M=m // 2, N=n // 2, NFP=eq.NFP)
        src_data = eq.compute(
            ["R", "Z", "phi", "e^rho", "|e_theta x e_zeta|"], grid=src_grid
        )
        eval_data = eq.compute(
            ["R", "Z", "phi", "e^rho", "|e_theta x e_zeta|"], grid=eval_grid
        )
        s = ss[i]
        q = qs[i]
        interpolator = FFTInterpolator(eval_grid, src_grid, s, q)

        err = singular_integral(
            eval_data,
            eval_grid,
            src_data,
            src_grid,
            "nr_over_r3",
            interpolator,
            loop=True,
        )
        np.testing.assert_array_less(np.abs(2 * np.pi + err), es[i])


@pytest.mark.unit
def test_singular_integral_vac_estell():
    """Test calculating Bplasma for vacuum estell, which should be near 0."""
    eq = desc.examples.get("ESTELL")
    eval_grid = LinearGrid(M=8, N=8, NFP=int(eq.NFP))

    src_grid = LinearGrid(M=18, N=18, NFP=int(eq.NFP))

    keys = [
        "K_vc",
        "B",
        "|B|^2",
        "R",
        "zeta",
        "Z",
        "e^rho",
        "n_rho",
        "|e_theta x e_zeta|",
    ]

    src_data = eq.compute(keys, grid=src_grid)
    eval_data = eq.compute(keys, grid=eval_grid)

    k = min(src_grid.num_theta, src_grid.num_zeta)
    s = k // 2 + int(np.sqrt(k))
    q = k // 2 + int(np.sqrt(k))

    interpolator = FFTInterpolator(eval_grid, src_grid, s, q)
    Bplasma = -virtual_casing_biot_savart(
        eval_data,
        eval_grid,
        src_data,
        src_grid,
        interpolator,
        loop=True,
    )
    # need extra factor of B/2 bc we're evaluating on plasma surface
    Bplasma = xyz2rpz_vec(Bplasma, phi=eval_data["zeta"]) + eval_data["B"] / 2
    Bplasma = np.linalg.norm(Bplasma, axis=-1)
    # scale by total field magnitude
    B = Bplasma / np.mean(np.linalg.norm(eval_data["B"], axis=-1))
    # this isn't a perfect vacuum equilibrium (|J| ~ 1e3 A/m^2), so increasing
    # resolution of singular integral won't really make Bplasma less.
    np.testing.assert_array_less(B, 0.05)
