"""Compute functions for stability objectives.

Notes
-----
Some quantities require additional work to compute at the magnetic axis.
A Python lambda function is used to lazily compute the magnetic axis limits
of these quantities. These lambda functions are evaluated only when the
computational grid has a node on the magnetic axis to avoid potentially
expensive computations.
"""

from functools import partial

from scipy.constants import mu_0

from desc.backend import eigh_tridiagonal, jax, jit, jnp, scan

from ..diffmat_utils import cheb_D1, cheb_D2, fourier_diffmat
from ..integrals.surface_integral import surface_integrals_map
from ..utils import dot
from .data_index import register_compute_fun


@register_compute_fun(
    name="D_shear",
    label="D_{\\mathrm{shear}}",
    units="Wb^{-2}",
    units_long="Inverse Webers squared",
    description="Mercier stability criterion magnetic shear term",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["iota_psi"],
)
def _D_shear(params, transforms, profiles, data, **kwargs):
    # Implements equation 4.16 in M. Landreman & R. Jorge (2020)
    # doi:10.1017/S002237782000121X.
    data["D_shear"] = data["iota_psi"] ** 2 / (16 * jnp.pi**2)
    return data


@register_compute_fun(
    name="D_current",
    label="D_{\\mathrm{current}}",
    units="Wb^{-2}",
    units_long="Inverse Webers squared",
    description="Mercier stability criterion toroidal current term",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "psi_r",
        "iota_psi",
        "B",
        "J",
        "G",
        "I_r",
        "|grad(psi)|",
        "|e_theta x e_zeta|",
    ],
    resolution_requirement="tz",
)
def _D_current(params, transforms, profiles, data, **kwargs):
    # Implements equation 4.17 in M. Landreman & R. Jorge (2020)
    # doi:10.1017/S002237782000121X.
    Xi = mu_0 * data["J"] - (data["I_r"] / data["psi_r"] * data["B"].T).T
    integrate = surface_integrals_map(transforms["grid"])
    data["D_current"] = (
        -jnp.sign(data["G"])
        / (2 * jnp.pi) ** 4
        * data["iota_psi"]
        * transforms["grid"].replace_at_axis(
            integrate(
                data["|e_theta x e_zeta|"]
                / data["|grad(psi)|"] ** 3
                * dot(Xi, data["B"])
            ),
            # TODO(#671): implement equivalent of equation 4.3 in desc coordinates
            jnp.nan,
        )
    )
    return data


@register_compute_fun(
    name="D_well",
    label="D_{\\mathrm{well}}",
    units="Wb^{-2}",
    units_long="Inverse Webers squared",
    description="Mercier stability criterion magnetic well term",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "p_r",
        "psi",
        "psi_r",
        "psi_rr",
        "V_rr(r)",
        "V_r(r)",
        "|B|^2",
        "|grad(psi)|",
        "|e_theta x e_zeta|",
    ],
    resolution_requirement="tz",
)
def _D_well(params, transforms, profiles, data, **kwargs):
    # Implements equation 4.18 in M. Landreman & R. Jorge (2020)
    # doi:10.1017/S002237782000121X.
    integrate = surface_integrals_map(transforms["grid"])
    dp_dpsi = mu_0 * data["p_r"] / data["psi_r"]
    d2V_dpsi2 = (
        data["V_rr(r)"] * data["psi_r"] - data["V_r(r)"] * data["psi_rr"]
    ) / data["psi_r"] ** 3
    data["D_well"] = (
        dp_dpsi
        * (
            jnp.sign(data["psi"]) * d2V_dpsi2
            - dp_dpsi
            * integrate(
                data["|e_theta x e_zeta|"] / (data["|B|^2"] * data["|grad(psi)|"])
            )
        )
        * integrate(
            data["|e_theta x e_zeta|"] * data["|B|^2"] / data["|grad(psi)|"] ** 3
        )
        / (2 * jnp.pi) ** 6
    )
    # Axis limit does not exist as ∂ᵨ ψ and ‖∇ ψ‖ terms dominate so that D_well
    # is of the order ρ⁻² near axis.
    return data


@register_compute_fun(
    name="D_geodesic",
    label="D_{\\mathrm{geodesic}}",
    units="Wb^{-2}",
    units_long="Inverse Webers squared",
    description="Mercier stability criterion geodesic curvature term",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["|grad(psi)|", "J*B", "|B|^2", "|e_theta x e_zeta|"],
    resolution_requirement="tz",
)
def _D_geodesic(params, transforms, profiles, data, **kwargs):
    # Implements equation 4.19 in M. Landreman & R. Jorge (2020)
    # doi:10.1017/S002237782000121X.
    integrate = surface_integrals_map(transforms["grid"])
    data["D_geodesic"] = transforms["grid"].replace_at_axis(
        (
            integrate(
                data["|e_theta x e_zeta|"]
                * mu_0
                * data["J*B"]
                / data["|grad(psi)|"] ** 3
            )
            ** 2
            - integrate(
                data["|e_theta x e_zeta|"] * data["|B|^2"] / data["|grad(psi)|"] ** 3
            )
            * integrate(
                data["|e_theta x e_zeta|"]
                * mu_0**2
                * data["J*B"] ** 2
                / (data["|B|^2"] * data["|grad(psi)|"] ** 3),
            )
        )
        / (2 * jnp.pi) ** 6,
        jnp.nan,  # enforce manually because our integration replaces nan with 0
    )
    # Axis limit does not exist as ‖∇ ψ‖ terms dominate so that D_geodesic
    # is of the order ρ⁻² near axis.
    return data


@register_compute_fun(
    name="D_Mercier",
    label="D_{\\mathrm{Mercier}}",
    units="Wb^{-2}",
    units_long="Inverse Webers squared",
    description="Mercier stability criterion (positive/negative value "
    + "denotes stability/instability)",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["D_shear", "D_current", "D_well", "D_geodesic"],
)
def _D_Mercier(params, transforms, profiles, data, **kwargs):
    # Implements equation 4.20 in M. Landreman & R. Jorge (2020)
    # doi:10.1017/S002237782000121X.
    data["D_Mercier"] = (
        data["D_shear"] + data["D_current"] + data["D_well"] + data["D_geodesic"]
    )
    # The axis limit does not exist as D_Mercier is of the order ρ⁻² near axis.
    return data


@register_compute_fun(
    name="magnetic well",
    label="\\mathrm{Magnetic~Well}",
    units="~",
    units_long="None",
    description="Magnetic well proxy for MHD stability (positive/negative value "
    + "denotes stability/instability)",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["V(r)", "V_r(r)", "p_r", "<|B|^2>", "<|B|^2>_r"],
)
def _magnetic_well(params, transforms, profiles, data, **kwargs):
    # Implements equation 3.2 in M. Landreman & R. Jorge (2020)
    # doi:10.1017/S002237782000121X.
    # pressure = thermal + magnetic = 2 mu_0 p + |B|^2
    # The surface average operation is an additive homomorphism.
    # Thermal pressure is constant over a rho surface.
    # surface average(pressure) = thermal + surface average(magnetic)
    # The sign of sqrt(g) is enforced to be non-negative.
    data["magnetic well"] = transforms["grid"].replace_at_axis(
        data["V(r)"]
        * (2 * mu_0 * data["p_r"] + data["<|B|^2>_r"])
        / (data["V_r(r)"] * data["<|B|^2>"]),
        0,  # coefficient of limit is V_r / V_rr, rest is finite
    )
    return data


@register_compute_fun(
    name="gds2",
    # |∇(α + ι ζ₀ sign ι)|² ρ²
    label="\\vert \\nabla (\\alpha + "
    "\\iota \\zeta_0 \\mathrm{sign} \\iota) \\vert^2 \\rho^2",
    units="m^{-2}",
    units_long="inverse square meters",
    description="Parameter in ideal ballooning equation",
    dim=2,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["rho", "g^aa", "g^rr", "g^ra", "shear", "iota"],
    zeta0="array: points of vanishing integrated local shear to scan over. "
    "Default 15 points linearly spaced in [-π/2,π/2]. "
    "The values ``zeta0`` correspond to values of ι ζ₀ and not ζ₀.",
    public=False,
)
def _gds2(params, transforms, profiles, data, **kwargs):
    zeta0 = kwargs.get("zeta0", jnp.linspace(-0.5 * jnp.pi, 0.5 * jnp.pi, 15))
    zeta0 = zeta0.reshape(-1, 1)

    data["gds2"] = (
        data["g^aa"] * data["rho"] ** 2
        - 2
        * data["g^ra"]
        * data["rho"]
        * jnp.sign(data["iota"])
        * data["shear"]
        * zeta0
        + data["g^rr"] * data["shear"] ** 2 * zeta0**2
    )
    return data


@register_compute_fun(
    name="c ballooning",
    # c = 2 a³ Bₙ μ₀ sign(ψ) dp/dψ / (|B|² b⋅∇ζ) (b × 𝛋) ⋅ ∇(α + ι ζ₀) ρ²
    label="2 a^3 B_n \\mu_0 \\mathrm{sign}(\\psi) (\\partial_{\\psi} p) / "
    "(\\vert B \\vert^2 b \\cdot \\nabla ζ) (b \\times \\kappa) \\cdot "
    "\\nabla (\\alpha + \\iota \\zeta_0) \\rho^2",
    units="~",
    units_long="None",
    description="Parameter in ideal ballooning equation",
    dim=2,
    params=["Psi"],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["a", "p_r", "psi", "psi_r", "B^zeta", "rho", "cvdrift", "cvdrift0", "shear"],
    zeta0="array: points of vanishing integrated local shear to scan over. "
    "Default 15 points linearly spaced in [-π/2,π/2]. "
    "The values ``zeta0`` correspond to values of ι ζ₀ and not ζ₀.",
)
def _c_balloon(params, transforms, profiles, data, **kwargs):
    """Dimensionless c sign(ψ) ρ².

    Where c mentioned immediately prior is defined in
    eq. 25b of arxiv.org/abs/2410.04576. Also α = α_{DESC} + ι ζ₀ here,
    consistent with above link.
    """
    zeta0 = kwargs.get("zeta0", jnp.linspace(-0.5 * jnp.pi, 0.5 * jnp.pi, 15))
    zeta0 = zeta0.reshape(-1, 1)

    psi_boundary = params["Psi"] / (2 * jnp.pi)
    data["c ballooning"] = (
        (2 * psi_boundary * data["a"] * mu_0)  # a³ Bₙ μ₀
        * jnp.sign(data["psi"])
        * data["p_r"]
        / data["psi_r"]
        / data["B^zeta"]
        * (
            2 * data["rho"] ** 2 * data["cvdrift"]
            - data["cvdrift0"] * data["shear"] * zeta0
        )
    )
    return data


@register_compute_fun(
    name="f ballooning",
    # f = a Bₙ³ |B|⁻² / (B⋅∇ζ) |∇(α + ι ζ₀ sign ι)|² ρ²
    label="a B_n^3 \\vert B \\vert^{-2} / (B \\cdot \\nabla ζ) "
    "\\vert \\nabla (\\alpha + \\iota \\zeta_0 \\mathrm{sign} \\iota) \\vert^2 \\rho^2",
    units="~",
    units_long="None",
    description="Parameter in ideal ballooning equation",
    dim=2,
    params=["Psi"],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["a", "|B|^2", "B^zeta", "gds2"],
)
def _f_balloon(params, transforms, profiles, data, **kwargs):
    """Dimensionless f ρ² where f is defined in eq. 25c of arxiv.org/abs/2410.04576.

    Also α = α_{DESC} + ι ζ₀ sign ι here whereas above link has α = α_{DESC} + ι ζ₀.
    """
    psi_boundary = params["Psi"] / (2 * jnp.pi)
    B_n = 2 * psi_boundary / data["a"] ** 2
    data["f ballooning"] = (
        data["a"] * B_n**3 / data["|B|^2"] / data["B^zeta"]
    ) * data["gds2"]
    return data


@register_compute_fun(
    name="g ballooning",
    # g = a³ Bₙ |B|⁻² (B⋅∇ζ) |∇(α + ι ζ₀ sign ι)|² ρ²
    label="a^3 B_n \\vert B \\vert^{-2} (B \\cdot \\nabla ζ) "
    "\\vert \\nabla (\\alpha + \\iota \\zeta_0 \\mathrm{sign} \\iota) \\vert^2 \\rho^2",
    units="~",
    units_long="None",
    description="Parameter in ideal ballooning equation",
    dim=2,
    params=["Psi"],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["a", "|B|^2", "B^zeta", "gds2"],
)
def _g_balloon(params, transforms, profiles, data, **kwargs):
    """Dimensionless ρ² g where g is defined in eq. 25a of arxiv.org/abs/2410.04576.

    Also α = α_{DESC} + ι ζ₀ sign ι here whereas above link has α = α_{DESC} + ι ζ₀.
    """
    psi_boundary = params["Psi"] / (2 * jnp.pi)
    B_n = 2 * psi_boundary / data["a"] ** 2
    data["g ballooning"] = (
        data["a"] ** 3 * B_n * data["B^zeta"] / data["|B|^2"]
    ) * data["gds2"]
    return data


@register_compute_fun(
    name="ideal ballooning lambda",
    label="\\lambda_{\\mathrm{ballooning}}=\\gamma^2",
    units="~",
    units_long="None",
    description="Normalized squared ideal ballooning growth rate",
    dim=3,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=["c ballooning", "f ballooning", "g ballooning"],
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
    Neigvals="int: number of largest eigenvalues to return, default value is 1.`"
    "If `Neigvals=2` eigenvalues are `[-1, 0, 1]` we get `[1, 0]`",
)
@partial(jit, static_argnames=["Neigvals"])
def _ideal_ballooning_lambda(params, transforms, profiles, data, **kwargs):
    """Eigenvalues of ideal-ballooning equation.

    A finite-difference method is used to calculate the maximum
    growth rate against the infinite-n ideal ballooning mode.
    The equation being solved is

    d/dζ(g dX/dζ) + c X = λ f X, g, f > 0

    where

      λ = a² / v_A² * γ²
    v_A = Bₙ / sqrt(μ₀ n₀ M) is the Alfven speed

    Returns
    -------
    Ideal-ballooning lambda eigenvalues
        Shape (num_rho, num alpha, num zeta0, num eigvals).

    """
    Neigvals = kwargs.get("Neigvals", 1)
    grid = transforms["grid"].source_grid
    # toroidal step size between points along field lines is assumed uniform
    dz = grid.nodes[grid.unique_zeta_idx[:2], 2]
    dz = dz[1] - dz[0]
    num_zeta0 = data["c ballooning"].shape[0]

    def reshape(f):
        assert f.shape == (num_zeta0, grid.num_nodes)
        f = jnp.swapaxes(grid.meshgrid_reshape(f.T, "raz"), -1, -2)
        assert f.shape == (grid.num_rho, grid.num_alpha, num_zeta0, grid.num_zeta)
        return f

    c = reshape(data["c ballooning"])
    f = reshape(data["f ballooning"])
    g = reshape(data["g ballooning"])

    # Approximate derivative along field line with second order finite differencing.
    # Use g on the half grid for numerical stability.
    g_half = (g[..., 1:] + g[..., :-1]) / (2 * dz**2)
    b_inv = jnp.reciprocal(f[..., 1:-1])
    diag_inner = (c[..., 1:-1] - g_half[..., 1:] - g_half[..., :-1]) * b_inv
    diag_outer = g_half[..., 1:-1] * jnp.sqrt(b_inv[..., :-1] * b_inv[..., 1:])

    # TODO: Issue #1750
    w, v = eigh_tridiagonal(diag_inner, diag_outer)

    w, top_idx = jax.lax.top_k(w, k=Neigvals)
    assert w.shape == (grid.num_rho, grid.num_alpha, num_zeta0, Neigvals)
    data["ideal ballooning lambda"] = w

    # v becomes less than the machine precision at some points which gives NaNs.
    # stop_gradient prevents that.
    v = jax.lax.stop_gradient(v)
    v = jnp.take_along_axis(v, top_idx[..., jnp.newaxis, :], axis=-1)
    assert v.shape == (
        grid.num_rho,
        grid.num_alpha,
        num_zeta0,
        grid.num_zeta - 2,
        Neigvals,
    )
    data["ideal ballooning eigenfunction"] = v

    return data


@register_compute_fun(
    name="ideal ballooning eigenfunction",
    label="X_{\\mathrm{ballooning}}",
    units="~",
    units_long="None",
    description="Ideal ballooning eigenfunction",
    dim=4,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["ideal ballooning lambda"],
)
def _ideal_ballooning_eigenfunction(params, transforms, profiles, data, **kwargs):
    """Eigenfunctions of ideal-ballooning equation.

    Returns
    -------
    Ideal-ballooning lambda eigenfunctions
        Shape (num_rho, num alpha, num zeta0, num zeta - 2, num eigvals).

    """
    return data  # noqa: unused dependency


@register_compute_fun(
    name="Newcomb ballooning metric",
    label="\\mathrm{Newcomb-ballooning-metric}",
    units="~",
    units_long="None",
    description="A measure of Newcomb's distance from marginal ballooning stability",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["c ballooning", "g ballooning"],
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
)
@jit
def _Newcomb_ball_metric(params, transforms, profiles, data, **kwargs):
    """Ideal-ballooning growth rate proxy.

    A finite-difference method is used to integrate the
    marginal stability ideal-ballooning equation

    d/dζ(g dX/dζ) + c X = 0, g > 0

    where

      λ = a² / v_A² * γ²
    v_A = Bₙ / sqrt(μ₀ n₀ M) is the Alfven speed

    The Newcomb's stability criterion is used.
    We define the Newcomb metric as follows:
    If zero crossing is at -inf (root finder failed), use the Y coordinate as a
    metric of stability. Otherwise use the zero-crossing point on the X-axis.
    This idea behind Newcomb's method is explained further in Appendix D of
    [Gaur _et al._](https://doi.org/10.1017/S0022377823000107).

    """
    grid = transforms["grid"].source_grid
    # toroidal step size between points along field lines is assumed uniform
    zeta = grid.compress(grid.nodes[:, 2], surface_label="zeta")
    dz = zeta[1] - zeta[0]
    num_zeta0 = data["c ballooning"].shape[0]

    def reshape(f):
        assert f.shape == (num_zeta0, grid.num_nodes)
        f = jnp.moveaxis(grid.meshgrid_reshape(f.T, "raz"), -2, 0)
        assert f.shape == (grid.num_zeta, grid.num_rho, grid.num_alpha, num_zeta0)
        return f

    c = reshape(data["c ballooning"])[:-1]
    g = reshape(data["g ballooning"])

    def integrator(carry, x):
        """Update ``y`` and its derivative using leapfrog-like method.

        Assumed that y starts nonnegative with positive dy.

        Returns
        -------
        Cumulative integration of ``y`` and markers for the sign change.
        """
        y, dy = carry
        c, g = x
        y_new = y + dz * dy / g
        dy_new = dy - c * y_new * dz
        return (y_new, dy_new), (y_new, y_new < 0)

    dy_dz_initial = 5e-3
    _, (y, is_root) = scan(
        integrator,
        init=(jnp.zeros(c.shape[1:]), jnp.full(c.shape[1:], dy_dz_initial)),
        # Use g on the half grid for numerical stability.
        xs=(c, (g[1:] + g[:-1]) / 2),
    )

    idx_right_root = jnp.argmax(is_root.at[-1].set(True), axis=0, keepdims=True)
    y_left_root = jnp.take_along_axis(y, idx_right_root - 1, axis=0)
    # derivative of linear approximation of ζ ↦ y(ζ) near root
    dy_dz = (jnp.take_along_axis(y, idx_right_root, axis=0) - y_left_root) / dz

    # crossing from stable to unstable regime
    x = zeta[idx_right_root] - jnp.where(
        idx_right_root < (is_root.shape[0] - 1), y_left_root / dy_dz * dz, 0
    )
    # We take the signed distance X - ζ max < 0 as the distance to stability.
    # If there was no crossing we take y[ζ = ζ max] > 0.
    # This metric is only C0. Maybe think of something better?
    # RG: Peak of the metric does not match mean peak of the growth rate in ρ.
    data["Newcomb ballooning metric"] = (
        jnp.where(
            idx_right_root < (is_root.shape[0] - 1),
            (x - zeta[-1]) / (zeta[-1] - zeta[0]),
            y[-1],
        )
        .min((-1, -2))
        .squeeze(0)
    )
    return data


@register_compute_fun(
    name="finite-n lambda",
    label="low-\\n \\lambda = \\gamma^2",
    units="~",
    units_long="None",
    description="Normalized squared growth rate",
    dim=1,
    params=["Psi", "NFP"],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=[
        "a",
        "g_rr|PEST",
        "g_rt|PEST",
        "g_rz|PEST",
        "g_tt|PEST",
        "g_tz|PEST",
        "g_zz|PEST",
        "J^theta|PEST",
        "J^zeta|PEST",
        "sqrt(g)_PEST",
        "sqrt(g)_r|PEST",
        "sqrt(g)_t|PEST",
        "sqrt(g)_z|PEST",
        "iota",
        "iota_r",
        "iota_rr",
        "p",
        "p_r",
        "p_rr",
        "psi_r",
        "psi_rr",
        "(psi_r/sqrt(g)_PEST)",
        "(chi_r/sqrt(g)_PEST)",
    ],
    n_rho_max="int: 2 x maximum radial mode number",
    n_theta_max="int: 2 x maximum radial mode number",
    n_zeta_max="int: 2 x maximum toroidal mode number",
    axisym="bool: if the equilibrium is axisymmetric",
)
def _AGNI(params, transforms, profiles, data, **kwargs):
    """
    AGNI: Analysis of Global Normal-modes in Ideal MHD.

    Based on the original source here:
    https://github.com/rahulgaur104/AGNI/tree/master

    A finite-n stability eigenvalue solver.
    Currenly only finds fixed boundary unstable modes at
    low to medium resolution.
    """
    a_N = data["a"]
    B_N = params["Psi"] / (jnp.pi * a_N**2)

    # --no-verify NFP = params["NFP"]
    NFP = 1

    iota = data["iota"][:, None]
    iota_r = data["iota_r"][:, None]
    iota_rr = data["iota_rr"][:, None]

    p = mu_0 * data["p"][:, None] / B_N**2
    p_r = mu_0 * data["p_r"][:, None] / B_N**2
    p_rr = mu_0 * data["p_rr"][:, None] / B_N**2
    psi_r = data["psi_r"][:, None] / (a_N**2 * B_N)
    psi_rr = data["psi_rr"][:, None] / (a_N**2 * B_N)

    psi_r_sqrt_g = data["(psi_r/sqrt(g)_PEST)"][:, None] * (a_N / B_N)
    chi_r_sqrt_g = data["(chi_r/sqrt(g)_PEST)"][:, None] * (a_N / B_N)

    axisym = kwargs.get("axisym", False)

    n_rho_max = kwargs.get("n_rho_max", 8)
    n_theta_max = kwargs.get("n_theta_max", 8)
    n_zeta_max = kwargs.get("n_zeta_max", 4)

    if axisym:
        D_zeta0 = n_zeta_max * jnp.array([[0, -1], [1, 0]])
        n_zeta_max = 2
    else:
        D_zeta0 = fourier_diffmat(n_zeta_max)

    def _eval_1D(f, x):
        return vmap(lambda x_val: f(x_val))(x)

    def _f(x):
        x_0 = 0.5
        m_1 = 2.1
        m_2 = 2.1
        lower = x_0 * (1 - jnp.exp(-m_1 * (x + 1)) + 0.5 * (x + 1) * jnp.exp(-2 * m_1))
        upper = (1 - x_0) * (jnp.exp(m_2 * (x - 1)) + 0.5 * (x - 1) * jnp.exp(-2 * m_2))
        eps = 1.0e-3
        return eps + (1 - eps) * (lower + upper)

    dx_f = jax.grad(_f)
    dxx_f = jax.grad(dx_f)

    # The points in the supplied grid must be consistent with how
    # the kronecker product is created
    x = transforms["grid"].nodes[:: n_theta_max * n_zeta_max, 0]

    scale_vector1 = (_eval_1D(dx_f, x)) ** -1
    scale_vector2 = (_eval_1D(dxx_f, x)) * scale_vector1

    scale_x1 = scale_vector1[:, None]
    scale_x2 = scale_vector2[:, None]

    # Get differentiation matrices
    # RG: setting the gradient to 0 to save some memory?
    D_rho0 = jax.lax.stop_gradient(cheb_D1(n_rho_max) * scale_x1)
    D_rho0 = jax.lax.stop_gradient(D_rho0)

    D_theta0 = jax.lax.stop_gradient(fourier_diffmat(n_theta_max))

    D_zeta0 = jax.lax.stop_gradient(NFP * D_zeta0)

    I_rho0 = jax.lax.stop_gradient(jnp.eye(n_rho_max))
    I_theta0 = jax.lax.stop_gradient(jnp.eye(n_theta_max))
    I_zeta0 = jax.lax.stop_gradient(jnp.eye(n_zeta_max))

    D_rho = jnp.kron(D_rho0, jnp.kron(I_theta0, I_zeta0))
    D_theta = jnp.kron(I_rho0, jnp.kron(D_theta0, I_zeta0))
    D_zeta = jnp.kron(I_rho0, jnp.kron(I_theta0, D_zeta0))

    D_rho_D_rho = jax.lax.stop_gradient(
        jnp.kron(
            (cheb_D2(n_rho_max) - cheb_D1(n_rho_max) * scale_x2) * scale_x1**2,
            jnp.kron(I_theta0, I_zeta0),
        )
    )

    D_theta_D_theta = D_theta @ D_theta
    D_zeta_D_zeta = D_zeta @ D_zeta

    D_rho_D_theta = D_rho @ D_theta
    D_rho_D_zeta = D_rho @ D_zeta
    D_theta_D_zeta = D_theta @ D_zeta

    n_total = n_rho_max * n_theta_max * n_zeta_max
    # Create the full matrix
    A = jnp.zeros((3 * n_total, 3 * n_total))

    sqrt_g_Q_sup_rho = jnp.zeros((n_total, 3 * n_total))
    sqrt_g_Q_sup_theta = jnp.zeros((n_total, 3 * n_total))
    sqrt_g_Q_sup_zeta = jnp.zeros((n_total, 3 * n_total))

    partial_zeta_Q_rho = jnp.zeros((n_total, 3 * n_total))
    partial_rho_Q_zeta = jnp.zeros((n_total, 3 * n_total))

    partial_theta_Q_rho = jnp.zeros((n_total, 3 * n_total))
    partial_rho_Q_theta = jnp.zeros((n_total, 3 * n_total))

    partial_theta_Q_zeta = jnp.zeros((n_total, 3 * n_total))
    partial_zeta_Q_theta = jnp.zeros((n_total, 3 * n_total))

    # Define field component indices
    rho_idx = slice(0, n_total)
    theta_idx = slice(n_total, 2 * n_total)
    zeta_idx = slice(2 * n_total, 3 * n_total)

    all_idx = slice(0, 3 * n_total)

    g_rr = data["g_rr|PEST"][:, None] / a_N**2
    g_tt = data["g_tt|PEST"][:, None] / a_N**2
    g_zz = data["g_zz|PEST"][:, None] / a_N**2

    g_rt = data["g_rt|PEST"][:, None] / a_N**2
    g_tr = g_rt

    g_rz = data["g_rz|PEST"][:, None] / a_N**2
    # --no-verify g_zr = g_rz

    g_tz = data["g_tz|PEST"][:, None] / a_N**2
    g_zt = g_tz

    j_sup_theta = data["J^theta|PEST"][:, None] * a_N**2 / B_N
    j_sup_zeta = data["J^zeta|PEST"][:, None] * a_N**2 / B_N

    sqrt_g = data["sqrt(g)_PEST"][:, None] / a_N**3
    sqrt_g_r = data["sqrt(g)_r|PEST"][:, None] / a_N**3
    sqrt_g_t = data["sqrt(g)_t|PEST"][:, None] / a_N**3
    sqrt_g_z = data["sqrt(g)_z|PEST"][:, None] / a_N**3

    C_zeta = (sqrt_g_z / sqrt_g) + D_zeta
    C_theta = (sqrt_g_t / sqrt_g) + D_theta
    C_rho = (sqrt_g_r / sqrt_g) + D_rho

    # √g Q^ρ
    sqrt_g_Q_sup_rho = sqrt_g_Q_sup_rho.at[rho_idx, rho_idx].add(
        iota * D_theta + D_zeta
    )

    # √g Q^θ
    sqrt_g_Q_sup_theta = sqrt_g_Q_sup_theta.at[rho_idx, rho_idx].add(
        -iota * D_rho - iota_r
    )
    sqrt_g_Q_sup_theta = sqrt_g_Q_sup_theta.at[rho_idx, theta_idx].add(D_zeta)
    sqrt_g_Q_sup_theta = sqrt_g_Q_sup_theta.at[rho_idx, zeta_idx].add(-D_zeta)

    # √g Q^ζ
    sqrt_g_Q_sup_zeta = sqrt_g_Q_sup_zeta.at[rho_idx, rho_idx].add(-1 * D_rho)
    sqrt_g_Q_sup_zeta = sqrt_g_Q_sup_zeta.at[rho_idx, theta_idx].add(-1 * D_theta)
    sqrt_g_Q_sup_zeta = sqrt_g_Q_sup_zeta.at[rho_idx, zeta_idx].add(1 * D_theta)

    # ∂ᵨ (√g Q_ϑ) rho block
    partial_rho_Q_theta = partial_rho_Q_theta.at[rho_idx, rho_idx].add(
        1
        / sqrt_g
        * (
            g_tr * (iota * D_rho_D_theta + iota_r * D_theta + D_rho_D_zeta)
            - g_tt * (D_rho_D_rho + 2 * iota_r * D_rho + iota_rr)
            - g_tz * D_rho_D_rho
            - g_tr * sqrt_g_r / sqrt_g * (iota * D_theta + D_zeta)
            + g_tt * sqrt_g_r / sqrt_g * (iota_r + iota * D_rho)
            + g_tz * sqrt_g_r / sqrt_g * D_rho
        )
    )

    # ∂ᵨ (√g Q_ϑ) theta block
    partial_rho_Q_theta = partial_rho_Q_theta.at[rho_idx, theta_idx].add(
        1
        / sqrt_g
        * (
            g_tt * D_rho_D_zeta
            - g_tz * D_rho_D_theta
            - g_tt * sqrt_g_r / sqrt_g * D_zeta
            + g_tz * sqrt_g_r / sqrt_g * D_theta
        )
    )

    # ∂ᵨ (√g Q_ϑ) zeta block
    partial_rho_Q_theta = partial_rho_Q_theta.at[rho_idx, zeta_idx].add(
        -1
        / sqrt_g
        * (
            g_tt * D_rho_D_zeta
            - g_tz * D_rho_D_theta
            - g_tt * sqrt_g_r / sqrt_g * D_zeta
            + g_tz * sqrt_g_r / sqrt_g * D_theta
        )
    )

    # ∂_θ (√g Qᵨ) rho bloc
    partial_theta_Q_rho = partial_theta_Q_rho.at[rho_idx, rho_idx].add(
        1
        / sqrt_g
        * (
            g_rr * (iota * D_theta_D_theta + 0 * iota_r * D_theta + D_theta_D_zeta)
            - g_rt * (D_rho_D_theta + 1.0 * iota_r * D_theta)
            - g_rz * D_rho_D_theta
            - g_rr * sqrt_g_t / sqrt_g * (iota * D_theta + D_zeta)
            + g_rt * sqrt_g_t / sqrt_g * (iota_r + iota * D_rho)
            + g_rz * sqrt_g_t / sqrt_g * D_rho
        )
    )

    # ∂_θ (√g Qᵨ) theta block
    partial_theta_Q_rho = partial_theta_Q_rho.at[rho_idx, theta_idx].add(
        1
        / sqrt_g
        * (
            g_rt * D_theta_D_zeta
            - g_rz * D_theta_D_theta
            - g_rt * sqrt_g_t / sqrt_g * D_zeta
            + g_rz * sqrt_g_t / sqrt_g * D_theta
        )
    )

    # ∂_θ (√g Qᵨ) zeta block
    partial_theta_Q_rho = partial_theta_Q_rho.at[rho_idx, zeta_idx].add(
        -1
        / sqrt_g
        * (
            g_rt * D_theta_D_zeta
            - g_rz * D_theta_D_theta
            - g_rt * sqrt_g_t / sqrt_g * D_zeta
            + g_rz * sqrt_g_t / sqrt_g * D_theta
        )
    )

    # rho block
    partial_zeta_Q_theta = partial_zeta_Q_theta.at[rho_idx, rho_idx].add(
        1
        / sqrt_g
        * (
            g_rt * (iota * D_theta_D_zeta + 0 * iota_r * D_theta + D_zeta_D_zeta)
            - g_tt * (D_rho_D_zeta + 1 * iota_r * D_zeta)
            - g_tz * D_rho_D_zeta
            - g_rt * sqrt_g_z / sqrt_g * (iota * D_theta + D_zeta)
            + g_tt * sqrt_g_z / sqrt_g * (iota_r + iota * D_rho)
            + g_tz * sqrt_g_z / sqrt_g * D_rho
        )
    )

    # theta block
    partial_zeta_Q_theta = partial_zeta_Q_theta.at[rho_idx, theta_idx].add(
        1
        / sqrt_g
        * (
            g_tt * D_zeta_D_zeta
            - g_tz * D_theta_D_zeta
            - g_tt * sqrt_g_z / sqrt_g * D_zeta
            + g_tz * sqrt_g_z / sqrt_g * D_theta
        )
    )

    # zeta block
    partial_zeta_Q_theta = partial_zeta_Q_theta.at[rho_idx, zeta_idx].add(
        -1
        / sqrt_g
        * (
            g_tt * D_zeta_D_zeta
            - g_tz * D_theta_D_zeta
            - g_tt * sqrt_g_z / sqrt_g * D_zeta
            + g_tz * sqrt_g_z / sqrt_g * D_theta
        )
    )

    # rhobloc
    partial_theta_Q_zeta = partial_theta_Q_zeta.at[rho_idx, rho_idx].add(
        1
        / sqrt_g
        * (
            g_rz * (iota * D_theta_D_theta + 0 * iota_r * D_theta + D_theta_D_zeta)
            - g_tz * (D_rho_D_theta + 1 * iota_r * D_theta)
            - g_zz * D_rho_D_theta
            - g_rz * sqrt_g_t / sqrt_g * (iota * D_theta + D_zeta)
            + g_tz * sqrt_g_t / sqrt_g * (iota_r + iota * D_rho)
            + g_zz * sqrt_g_t / sqrt_g * D_rho
        )
    )

    # theta block
    partial_theta_Q_zeta = partial_theta_Q_zeta.at[rho_idx, theta_idx].add(
        1
        / sqrt_g
        * (
            g_zt * D_theta_D_zeta
            - g_zz * D_theta_D_theta
            - g_zt * sqrt_g_t / sqrt_g * D_zeta
            + g_zz * sqrt_g_t / sqrt_g * D_theta
        )
    )

    # zeta block
    partial_theta_Q_zeta = partial_theta_Q_zeta.at[rho_idx, zeta_idx].add(
        -1
        / sqrt_g
        * (
            g_zt * D_theta_D_zeta
            - g_zz * D_theta_D_theta
            - g_zt * sqrt_g_t / sqrt_g * D_zeta
            + g_zz * sqrt_g_t / sqrt_g * D_theta
        )
    )

    # rho block
    partial_zeta_Q_rho = partial_zeta_Q_rho.at[rho_idx, rho_idx].add(
        g_rr * (iota * D_theta_D_zeta + 0 * iota_r * D_theta + D_zeta_D_zeta)
        - g_rt * (D_rho_D_zeta + 1 * iota_r * D_zeta + 0.0 * iota_rr)
        - g_rz * D_rho_D_zeta
        - g_rr * sqrt_g_z / sqrt_g * (iota * D_theta + D_zeta)
        + g_rt * sqrt_g_z / sqrt_g * (iota_r + iota * D_rho)
        + g_rz * sqrt_g_z / sqrt_g * D_rho
    )

    # theta block
    partial_zeta_Q_rho = partial_zeta_Q_rho.at[rho_idx, theta_idx].add(
        1
        / sqrt_g
        * (
            g_rt * D_zeta_D_zeta
            - g_rz * D_theta_D_zeta
            - g_rt * sqrt_g_z / sqrt_g * D_zeta
            + g_rz * sqrt_g_z / sqrt_g * D_theta
        )
    )

    # zeta block
    partial_zeta_Q_rho = partial_zeta_Q_rho.at[rho_idx, zeta_idx].add(
        -1
        / sqrt_g
        * (
            g_rt * D_zeta_D_zeta
            - g_rz * D_theta_D_zeta
            - g_rt * sqrt_g_z / sqrt_g * D_zeta
            + g_rz * sqrt_g_z / sqrt_g * D_theta
        )
    )

    # rho block
    partial_rho_Q_zeta = partial_rho_Q_zeta.at[rho_idx, rho_idx].add(
        g_rz * (iota * D_rho_D_theta + iota_r * D_theta + D_rho_D_zeta)
        - g_tz * (D_rho_D_rho + 2 * iota_r * D_rho + iota_rr)
        - g_zz * D_rho_D_rho
        - g_rz * sqrt_g_r / sqrt_g * (iota * D_theta + D_zeta)
        + g_tt * sqrt_g_r / sqrt_g * (iota_r + iota * D_rho)
        + g_zz * sqrt_g_r / sqrt_g * D_rho
    )

    # theta block
    partial_rho_Q_zeta = partial_rho_Q_zeta.at[rho_idx, theta_idx].add(
        1
        / sqrt_g
        * (
            g_tz * D_rho_D_zeta
            - g_zz * D_rho_D_theta
            - g_tz * sqrt_g_r / sqrt_g * D_zeta
            + g_zz * sqrt_g_r / sqrt_g * D_theta
        )
    )

    # zeta block
    partial_rho_Q_zeta = partial_rho_Q_zeta.at[rho_idx, zeta_idx].add(
        -1
        / sqrt_g
        * (
            g_tz * D_rho_D_zeta
            - g_zz * D_rho_D_theta
            - g_tz * sqrt_g_r / sqrt_g * D_zeta
            + g_zz * sqrt_g_r / sqrt_g * D_theta
        )
    )

    A = A.at[rho_idx, all_idx].add(
        +1.0 * psi_r_sqrt_g * (partial_zeta_Q_rho - partial_rho_Q_zeta)
        - 1.0 * chi_r_sqrt_g * (partial_rho_Q_theta - partial_theta_Q_rho)
        + 1.0 * (j_sup_theta * sqrt_g_Q_sup_zeta - j_sup_zeta * sqrt_g_Q_sup_theta)
    )
    A = A.at[theta_idx, all_idx].add(
        -1.0 * psi_r_sqrt_g * (partial_theta_Q_zeta - partial_zeta_Q_theta)
        + j_sup_zeta * sqrt_g_Q_sup_rho
    )
    A = A.at[zeta_idx, all_idx].add(
        1.0 * chi_r_sqrt_g * (partial_theta_Q_zeta - partial_zeta_Q_theta)
        - j_sup_theta * sqrt_g_Q_sup_rho
    )

    # pressure-driven instability term
    A = A.at[rho_idx, rho_idx].add(
        (p_rr / psi_r - p_r * psi_rr / psi_r**2) + p_r / psi_r * D_rho
    )
    A = A.at[theta_idx, rho_idx].add(p_r / psi_r * D_theta)
    A = A.at[zeta_idx, rho_idx].add(p_r / psi_r * D_zeta)

    gamma0 = 5 / 3

    compress_rho = jnp.zeros((n_total, 3 * n_total))
    compress_theta = jnp.zeros((n_total, 3 * n_total))
    compress_zeta = jnp.zeros((n_total, 3 * n_total))

    # compressional term
    D_rho_C_rho = D_rho @ C_rho
    D_rho_C_theta = D_rho @ C_theta
    D_rho_C_zeta = D_rho @ C_zeta

    compress_rho = compress_rho.at[rho_idx, rho_idx].set(p_r * C_rho + p * D_rho_C_rho)
    compress_rho = compress_rho.at[rho_idx, theta_idx].set(
        p_r * C_theta + p * D_rho_C_theta
    )
    compress_rho = compress_rho.at[rho_idx, zeta_idx].set(
        p_r * C_zeta + p * D_rho_C_zeta
    )

    compress_theta = compress_theta.at[rho_idx, rho_idx].set(p * D_theta @ C_rho)
    compress_theta = compress_theta.at[rho_idx, theta_idx].set(p * D_theta @ C_theta)
    compress_theta = compress_theta.at[rho_idx, zeta_idx].set(p * D_theta @ C_zeta)

    compress_zeta = compress_zeta.at[rho_idx, rho_idx].set(p * D_zeta @ C_rho)
    compress_zeta = compress_zeta.at[rho_idx, theta_idx].set(p * D_zeta @ C_theta)
    compress_zeta = compress_zeta.at[rho_idx, zeta_idx].set(p * D_zeta @ C_zeta)

    A = A.at[rho_idx, all_idx].add(gamma0 * compress_rho)
    A = A.at[theta_idx, all_idx].add(gamma0 * compress_theta)
    A = A.at[zeta_idx, all_idx].add(gamma0 * compress_zeta)

    # apply dirichlet BC to ξ^ρ
    keep_1 = jnp.arange(n_theta_max * n_zeta_max, n_total - n_theta_max * n_zeta_max)
    keep_2 = jnp.arange(n_total, 3 * n_total)
    keep = jnp.concatenate([keep_1, keep_2])

    w, v = jnp.linalg.eig(A[jnp.ix_(keep, keep)])

    data["finite-n lambda"] = w

    return data
