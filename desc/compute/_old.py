"""Old compute functions.

These do not appear in the public documentation under the list of variables.
They are kept for verification and correctness testing.

References
----------
.. [1] V. V. Nemov, S. V. Kasilov, W. Kernbichler, and M. F. Heyn,
       "Evaluation of 1/ν neoclassical transport in stellarators,"
       Phys. Plasmas 6, 4622 (1999). https://doi.org/10.1063/1.873749.
.. [2] V. V. Nemov, S. V. Kasilov, W. Kernbichler, and G. O. Leitold,
       "Poloidal motion of trapped particle orbits in real-space coordinates,"
       Phys. Plasmas 15, 052501 (2008). https://doi.org/10.1063/1.2912456.
.. [3] J. L. Velasco, I. Calvo, S. Mulas, E. Sanchez, F. I. Parra, A. Cappa,
       and the W7-X Team, "A model for the fast evaluation of prompt losses of
       energetic ions in stellarators," Nucl. Fusion 61, 116059 (2021).
       https://doi.org/10.1088/1741-4326/ac2994.

"""

from functools import partial

from desc.backend import jit, jnp

from ..integrals.bounce_integral import Bounce1D, Options
from ..utils import safediv
from ._drift import _radial_drift, _radial_drift_wb_inverse, _v_tau, _vartheta_drift
from ._fast_ion import _gamma_c_data, _reduction_gamma_c
from ._neoclassical import _I_1, _I_2
from .data_index import register_compute_fun

_bounce1D_doc = {
    "num_well": Options._doc["num_well"],
    "num_quad": Options._doc["num_quad"],
    "num_pitch": Options._doc["num_pitch"],
    "surf_batch_size": Options._doc["surf_batch_size"],
    "quad": Options._doc["quad"],
}


@register_compute_fun(
    name="old effective ripple 3/2",
    label="\\epsilon_{\\mathrm{eff}}^{3/2}",
    units="~",
    units_long="None",
    description="Effective ripple modulation amplitude to 3/2 power",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "min_tz |B|",
        "max_tz |B|",
        "R0",
        "|grad(rho)|*kappa_g",
        "<|grad(rho)|>",
        "fieldline length",
    ]
    + Bounce1D.required_names,
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
    public=False,
    **_bounce1D_doc,
)
@partial(jit, static_argnames=["num_well", "num_quad", "num_pitch", "surf_batch_size"])
def _old_epsilon_32(params, transforms, profiles, data, **kwargs):
    """Effective ripple modulation amplitude to 3/2 power.

    References [1]_.

    """
    # noqa: unused dependency
    grid = transforms["grid"].source_grid
    opts = Options.guess(eta=1, grid=grid, Y_B=grid.num_zeta, **kwargs)
    num_well = kwargs.get("num_well", -1)

    def foreach_surface(data):
        pitch_inv, weight = Bounce1D.pitch_quad(
            data["min_tz |B|"], data["max_tz |B|"], opts.pitch_quad
        )
        I_1, I_2 = Bounce1D(grid, data, opts.quad).integrate(
            [_I_1, _I_2], pitch_inv, data, ["|grad(rho)|*kappa_g"], num_well=num_well
        )
        return jnp.sum(
            safediv(I_1**2, I_2).sum(-1).mean(-2) * (weight / pitch_inv**3),
            axis=-1,
        )

    B0 = data["max_tz |B|"]
    scalar = jnp.pi / (8 * 2**0.5) * data["R0"] ** 2
    out = Bounce1D.batch(
        foreach_surface,
        data,
        grid,
        names=("|grad(rho)|*kappa_g",),
        batch_size=opts.surf_batch_size,
    )
    assert out.ndim == 1
    data["old effective ripple 3/2"] = (
        (B0 / data["<|grad(rho)|>"]) ** 2
        * grid.expand(out * scalar)
        / data["fieldline length"]
    )
    return data


@register_compute_fun(
    name="old effective ripple",
    label="\\epsilon_{\\mathrm{eff}}",
    units="~",
    units_long="None",
    description="Neoclassical transport in the banana regime",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["old effective ripple 3/2"],
    public=False,
)
def _old_effective_ripple(params, transforms, profiles, data, **kwargs):
    """Proxy for neoclassical transport in the banana regime."""
    data["old effective ripple"] = data["old effective ripple 3/2"] ** (2 / 3)
    return data


@register_compute_fun(
    name="old Gamma_c",
    label=(
        # Γ_c = π/(8√2) ∫ dλ 〈 ∑ⱼ [v τ γ_c²]ⱼ 〉
        "\\Gamma_c = \\frac{\\pi}{8 \\sqrt{2}} "
        "\\int d\\lambda \\langle \\sum_j (v \\tau \\gamma_c^2)_j \\rangle"
    ),
    units="~",
    units_long="None",
    description="Fast ion confinement proxy",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "min_tz |B|",
        "max_tz |B|",
        "B^phi",
        "B^phi_r|v,p",
        "|B|_r|v,p",
        "b",
        "grad(phi)",
        "grad(psi)",
        "|grad(psi)|*kappa_g",
        "|grad(rho)|",
        "|e_alpha|r,p|",
        "iota_r",
        "fieldline length",
    ]
    + Bounce1D.required_names,
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
    public=False,
    **_bounce1D_doc,
)
@partial(jit, static_argnames=["num_well", "num_quad", "num_pitch", "surf_batch_size"])
def _old_Gamma_c(params, transforms, profiles, data, **kwargs):
    """Fast ion confinement proxy as defined by Nemov et al.

    Equation 61 of reference [2]_.

    """
    # noqa: unused dependency
    grid = transforms["grid"].source_grid
    opts = Options.guess(eta=-2, grid=grid, Y_B=grid.num_zeta, **kwargs)
    num_well = kwargs.get("num_well", -1)

    def foreach_surface(data):
        pitch_inv, weight = Bounce1D.pitch_quad(
            data["min_tz |B|"], data["max_tz |B|"], opts.pitch_quad
        )
        bounce = Bounce1D(grid, data, opts.quad)
        points = bounce.points(pitch_inv, num_well)
        v_tau, radial, poloidal = bounce.integrate(
            [_v_tau, _radial_drift, _vartheta_drift],
            pitch_inv,
            data,
            ["|grad(psi)|*kappa_g", "|B|_r|v,p", "K"],
            points,
        )
        return jnp.sum(
            _reduction_gamma_c(
                v_tau,
                radial,
                poloidal
                * bounce.interp_to_argmin(data["|grad(rho)|*|e_alpha|r,p|"], points),
            )
            * (weight / pitch_inv**2),
            axis=-1,
        )

    out = Bounce1D.batch(
        foreach_surface,
        data,
        grid,
        custom_data=_gamma_c_data(data),
        batch_size=opts.surf_batch_size,
    )
    assert out.ndim == 1
    data["old Gamma_c"] = (
        grid.expand(out * (jnp.pi / 2**3.5)) / data["fieldline length"]
    )
    return data


@register_compute_fun(
    name="old Gamma_c Velasco",
    label=(
        "\\check{\\Gamma}_c = \\frac{1}{2} "
        "\\left\\langle \\int d\\lambda \\frac{B}{\\sqrt{1 - \\lambda B}} "
        "\\gamma_c^2"
        "\\right\\rangle"
    ),
    units="~",
    units_long="None",
    description="Fast ion confinement proxy "
    "as defined by Velasco et al. (doi:10.1088/1741-4326/ac2994)",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["min_tz |B|", "max_tz |B|", "cvdrift0", "gbdrift", "fieldline length"]
    + Bounce1D.required_names,
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
    public=False,
    **_bounce1D_doc,
)
@partial(jit, static_argnames=["num_well", "num_quad", "num_pitch", "surf_batch_size"])
def _old_Gamma_c_Velasco(params, transforms, profiles, data, **kwargs):
    """Fast ion confinement proxy as defined by Velasco et al.

    Equation 20 of reference [3]_.

    """
    # noqa: unused dependency
    grid = transforms["grid"].source_grid
    opts = Options.guess(eta=-1, grid=grid, Y_B=grid.num_zeta, **kwargs)
    num_well = kwargs.get("num_well", -1)

    def _poloidal_drift_secular_wb_inverse(data, B, pitch):
        return safediv(
            data["gbdrift"] * (1 - 0.5 * pitch * B),
            jnp.sqrt(jnp.abs(1 - pitch * B)),
        )

    def foreach_surface(data):
        pitch_inv, weight = Bounce1D.pitch_quad(
            data["min_tz |B|"], data["max_tz |B|"], opts.pitch_quad
        )
        v_tau, radial, poloidal = Bounce1D(grid, data, opts.quad).integrate(
            [_v_tau, _radial_drift_wb_inverse, _poloidal_drift_secular_wb_inverse],
            pitch_inv,
            data,
            names,
            num_well=num_well,
        )
        return jnp.sum(
            _reduction_gamma_c(v_tau, radial, poloidal) * (weight / pitch_inv**2),
            axis=-1,
        )

    names = ("cvdrift0", "gbdrift")
    out = Bounce1D.batch(
        foreach_surface,
        data,
        grid,
        names=names,
        batch_size=opts.surf_batch_size,
    )
    assert out.ndim == 1
    data["old Gamma_c Velasco"] = (
        grid.expand(out * (jnp.pi**2 / 2**5)) / data["fieldline length"]
    )
    return data
