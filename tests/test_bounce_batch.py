"""Regression tests for current and legacy bounce batching APIs."""

import warnings
from functools import partial

import numpy as np
import pytest

from desc.backend import jax, jnp
from desc.equilibrium import Equilibrium
from desc.grid import Grid, LinearGrid
from desc.integrals import Bounce1D, Bounce2D
from desc.integrals.bounce_integral import BounceOptions, Options
from desc.objectives import EffectiveRipple, GammaC


def _call_batch(bounce, fun, data, grid, angle, batch_size, sparse, shard, style):
    """Exercise the supported positional and keyword spellings."""
    common = {"sparse": sparse, "flux_data": {"profile": data["profile"]}}
    angles = {"angle": angle} if bounce is Bounce2D else {}
    if bounce is Bounce2D:
        common["shard"] = shard
    if style.startswith("current"):
        common.update(
            batch_size=batch_size,
            names="extra",
            custom_data={"custom": data["custom"]},
            **angles,
        )
        if style == "current positional":
            return bounce.batch(fun, data, grid, **common)
        return bounce.batch(fun=fun, data=data, grid=grid, **common)

    # The old API gives desc_data precedence for required fields and extrema.
    fun_data = {name: data[name] for name in ("extra", "custom")}
    fun_data.update(
        {
            name: jnp.zeros_like(data["|B|"])
            for name in (*bounce.required_names, "min_tz |B|", "max_tz |B|")
        }
    )
    original = fun_data.copy()
    common.update(surf_batch_size=batch_size, num_pitch=5, expand_out=True)
    if style == "legacy positional":
        args = (
            (fun_data, data, angle, grid)
            if bounce is Bounce2D
            else (fun_data, data, grid)
        )
        result = bounce.batch(fun, *args, **common)
    elif style == "legacy keyword":
        result = bounce.batch(
            fun=fun,
            fun_data=fun_data,
            desc_data=data,
            grid=grid,
            **angles,
            **common,
        )
    elif style == "legacy mixed":
        result = bounce.batch(fun, fun_data, data, grid=grid, **angles, **common)
    elif style == "legacy desc keyword":
        result = bounce.batch(
            fun, fun_data, desc_data=data, grid=grid, **angles, **common
        )
    else:
        args = (fun_data, data, angle) if bounce is Bounce2D else (fun_data, data)
        result = bounce.batch(fun, *args, grid=grid, **common)
    assert fun_data.keys() == original.keys()
    assert all(fun_data[name] is original[name] for name in original)
    return result


@pytest.mark.unit
@pytest.mark.parametrize(
    "bounce,shard", [(Bounce1D, False), (Bounce2D, False), (Bounce2D, True)]
)
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2, None])
def test_bounce_batch_signatures(bounce, shard, sparse, batch_size):
    """Legacy and current calls preserve values and reverse-mode derivatives."""
    rho = jnp.array([0.2, 0.5, 0.9])
    if bounce is Bounce2D:
        grid = LinearGrid(rho=rho, theta=4, zeta=5)
        names = ("B^zeta", "|B|", "extra", "custom")
        axes = (-3, -2, -1)
    else:
        grid = Grid.create_meshgrid(
            [rho, jnp.array([0.0, 1.0]), jnp.linspace(0, 2 * jnp.pi, 5)],
            coordinates="raz",
        )
        names = (*Bounce1D.required_names, "extra", "custom")
        axes = (-2, -1)
    q = 2 + grid.nodes[:, 0] + jnp.sin(grid.nodes[:, 1]) + jnp.cos(grid.nodes[:, 2])
    angle = jnp.arange(18.0).reshape(3, 2, 3) / 18
    scales = jnp.array([0.7, 1.1, 1.6])

    def inputs(scale):
        values = {
            name: (i + 1) * q * grid.expand(scale) for i, name in enumerate(names)
        }
        values.update(
            {
                "iota": grid.expand(rho),
                "min_tz |B|": grid.expand(rho + 1),
                "max_tz |B|": grid.expand(rho + 3),
                "profile": grid.expand(scale**2),
            }
        )
        return values

    def fun(data):
        result = sum(jnp.sum(jnp.abs(data[name]) ** 2, axis=axes) for name in names)
        result += (data["max_tz |B|"] - data["min_tz |B|"]) ** 2 + data["profile"]
        if bounce is Bounce2D:
            result += data["iota"] ** 2 + data["angle"].sum(axis=(-2, -1))
        return result

    def reference(scale):
        values = inputs(scale)
        prepared = {name: bounce.reshape(grid, values[name]) for name in names}
        if bounce is Bounce2D:
            prepared = {
                name: Bounce2D.fourier(value) for name, value in prepared.items()
            }
            prepared["angle"] = angle
            prepared["iota"] = rho
        for name in ("min_tz |B|", "max_tz |B|", "profile"):
            prepared[name] = grid.compress(values[name])
        return fun(prepared)

    def evaluate(scale, style):
        return _call_batch(
            bounce, fun, inputs(scale), grid, angle, batch_size, sparse, shard, style
        )

    expected = reference(scales)
    expected_jac = jax.jacrev(reference)(scales)
    styles = [
        "current positional",
        "current keyword",
        "legacy positional",
        "legacy keyword",
        "legacy mixed",
        "legacy desc keyword",
    ]
    if bounce is Bounce2D:
        styles.append("legacy angle positional")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Older JAX versions intentionally fall back to ordinary batching.
        warnings.filterwarnings(
            "ignore",
            message=r"shard=True requires JAX 0\.10\.2 or newer; "
            r"falling back to ordinary chunking\.",
            category=RuntimeWarning,
        )
        for style in styles:
            call = partial(evaluate, style=style)
            np.testing.assert_allclose(jax.jit(call)(scales), expected, rtol=1e-12)
            np.testing.assert_allclose(
                jax.jit(jax.jacrev(call))(scales), expected_jac, rtol=1e-12, atol=1e-12
            )
    assert not caught, [str(warning.message) for warning in caught]


@pytest.mark.unit
def test_bounce_master_option_names():
    """Master's field-period spelling reaches the existing option container."""
    assert BounceOptions is Options
    grid = LinearGrid(rho=[0.5], M=1, N=1)
    opts = Options.guess(-1, grid, field_period_transits=3)
    assert opts.num_field_periods == opts.field_period_transits == 3
    assert opts.num_well == Options.guess(-1, grid, num_field_periods=3).num_well
    eq = Equilibrium()
    for objective in (EffectiveRipple, GammaC):
        obj = objective(eq, field_period_transits=3, jac_chunk_size=2, nufft_eps=0)
        assert obj._hyperparam["num_field_periods"] == 3
        assert obj._jac_chunk_size == 2
