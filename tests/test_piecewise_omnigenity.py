"""Regression for collocation of a pwO field in the equilibrium's Boozer chart."""

import numpy as np
import pytest
from desc.backend import jnp
from desc.equilibrium import Equilibrium
from desc.grid import Grid, LinearGrid
from desc.magnetic_fields import PiecewiseOmnigenousField
from desc.objectives import PiecewiseOmnigenity


@pytest.mark.parametrize("nt", [8, 9])
@pytest.mark.parametrize("mode", ["field_fixed", "both_free", "eq_fixed"])
def test_piecewise_target_uses_boozer_coordinates(nt, mode):
    """Evaluate the target at each matching Boozer point without rolling rows."""
    eq = Equilibrium(L=2, M=2, N=1, NFP=2, iota=0.42)
    grid = LinearGrid(rho=np.sqrt(0.5), theta=nt, zeta=7, NFP=eq.NFP, sym=False)
    field = PiecewiseOmnigenousField(
        B_min=1.0,
        B_max=1.35,
        theta_C=np.pi,
        zeta_C=np.pi / 2,
        t_1=-0.3,
        t_2=2.0,
        NFP=2,
        p=2,
    )
    obj = PiecewiseOmnigenity(
        eq=eq,
        field=field,
        eq_grid=grid,
        field_fixed=mode == "field_fixed",
        eq_fixed=mode == "eq_fixed",
        normalize=False,
        overlap_penalty=20.0,
    )
    obj.build(use_jit=False, verbose=0)
    data = eq.compute(["|B|", "theta_B", "zeta_B", "iota"], grid=grid)
    theta_shift = (grid.num_theta // 2) * (2 * np.pi / grid.num_theta)
    target_grid = Grid(
        nodes=jnp.column_stack(
            (
                grid.nodes[:, 0],
                (data["theta_B"] - theta_shift) % (2 * np.pi),
                data["zeta_B"] % (2 * np.pi / eq.NFP),
            )
        ),
        NFP=eq.NFP,
        sort=False,
        jitable=True,
    )
    target = field.compute(
        ["|B|_pwO", "Q_pwO"], grid=target_grid, iota=data["iota"][0], p=2
    )
    q = target["Q_pwO"]
    penalty = 20 * (q + 0.1) * (q >= -0.1)
    expected = data["|B|"] - target["|B|_pwO"] + penalty
    if mode == "field_fixed":
        actual = obj.compute(eq.params_dict)
    elif mode == "eq_fixed":
        actual = obj.compute(field.params_dict)
    else:
        actual = obj.compute(eq.params_dict, field.params_dict)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_piecewise_boozer_coordinate_jit_jvp():
    """The changing Boozer target grid remains in the differentiated path."""
    import jax

    eq = Equilibrium(L=2, M=2, N=1, NFP=2, iota=0.42)
    grid = LinearGrid(
        rho=np.sqrt(0.5),
        theta=0.13 + 2 * np.pi * np.arange(8) / 8,
        zeta=0.11 + np.pi * np.arange(7) / 7,
        NFP=2,
        sym=False,
    )
    field = PiecewiseOmnigenousField(
        B_min=1.0,
        B_max=1.35,
        theta_C=np.pi,
        zeta_C=np.pi / 2,
        t_1=-0.3,
        t_2=2.0,
        NFP=2,
        p=2,
    )
    obj = PiecewiseOmnigenity(
        eq=eq, field=field, eq_grid=grid, field_fixed=True, normalize=False
    )
    obj.build(use_jit=False, verbose=0)
    base = eq.params_dict
    fun = jax.jit(obj.compute)
    y = fun(base)
    tangent = {k: jnp.zeros_like(v) for k, v in base.items()}
    tangent["R_lmn"] = jnp.ones_like(base["R_lmn"]) * 0.01
    _, dy = jax.jvp(fun, (base,), (tangent,))
    h = 1e-5
    plus = {k: base[k] + h * tangent[k] for k in base}
    minus = {k: base[k] - h * tangent[k] for k in base}
    fd = (fun(plus) - fun(minus)) / (2 * h)
    assert bool(jnp.all(jnp.isfinite(y)))
    assert bool(jnp.all(jnp.isfinite(dy)))
    np.testing.assert_allclose(dy, fd, rtol=5e-5, atol=1e-7)
