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

from ..integrals.quad_utils import leggauss_lob
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
    dim=4,
    params=[],
    transforms={"grid": [], "diffmat": []},
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

    num_zeta0 = data["c ballooning"].shape[0]

    def reshape(f):
        assert f.shape == (num_zeta0, grid.num_nodes)
        f = jnp.swapaxes(grid.meshgrid_reshape(f.T, "raz"), -1, -2)
        assert f.shape == (grid.num_rho, grid.num_alpha, num_zeta0, grid.num_zeta)
        return f

    c = reshape(data["c ballooning"])
    f = reshape(data["f ballooning"])
    g = reshape(data["g ballooning"])

    if transforms["diffmat"].zeta_diffmat is not None:

        def _eval_1D(f, x, scale, shift):
            return jax.vmap(lambda x_val: f(x_val, scale, shift))(x)

        def _f(x, scale, shift):
            x0 = 0.0
            x1 = 0.4
            m1 = 1.0
            m2 = 1.0
            m3 = 20
            m4 = 20

            wL = 0.5 * (1.0 + x0)  # left-side weigh
            wR = 0.5 * (1.0 - x0)  # right-side weight

            lower = wL * (
                1.0 - jnp.exp(-m1 * (x + 1.0)) + 0.5 * (x + 1.0) * jnp.exp(-2.0 * m1)
            )
            upper = wR * (
                jnp.exp(m2 * (x - 1.0)) + 0.5 * (x - 1.0) * jnp.exp(-2.0 * m2)
            )

            # Rescale to span [0,1] and shift to [-1,1]
            g_cluster = 2.0 * (lower + upper) - 1.0

            # Left logistic function
            s_axis = 1.0 / (1.0 + jnp.exp(-m3 * (x + 1.0)))
            s_axis0 = 1.0 / (1.0 + jnp.exp(-m3 * 0.0))
            s_axis1 = 1.0 / (1.0 + jnp.exp(-m3 * 2.0))
            axis = wL * (s_axis - s_axis0) / (s_axis1 - s_axis0)

            # Right logistic fn, also increasing after the flip
            s_edge_raw = 1.0 / (1.0 + jnp.exp(m4 * (x - 1.0)))
            s_edge = 1.0 - s_edge_raw
            s_edge0 = 1.0 - 1.0 / (1.0 + jnp.exp(m4 * -2.0))
            s_edge1 = 1.0 - 1.0 / (1.0 + jnp.exp(0.0))
            edge = wR * (s_edge - s_edge0) / (s_edge1 - s_edge0)

            g_axisedge = 2.0 * (axis + edge) - 1.0

            # Identity map contributes (1-x1) · x
            return (1.0 - x1) * x + x1 * (g_cluster + g_axisedge - x)

        dx_f = jax.grad(_f, argnums=0)

        num_alpha = grid.num_alpha
        num_rho = grid.num_rho
        num_zeta = grid.num_zeta

        ## The points in the supplied grid must be consistent with how
        ## the kronecker product is created
        x0 = grid.nodes[:: num_alpha * num_rho, 2]

        # The factor of two because we are mapping from [-1, 1] -> (-ntor pi, ntor pi)
        scale = (x0[-1] - x0[0]) / 2
        shift = 1 - x0[0] / scale

        x, w = leggauss_lob(num_zeta)

        scale_vector1 = (_eval_1D(dx_f, x, scale, shift)) ** -1 * 1 / scale

        scale_x1 = scale_vector1[:, None]

        # Check that the gradients of D_zeta are not calculated
        D_zeta = transforms["diffmat"].zeta_diffmat * scale_x1

        # 2D matrices stacked in rho, alpha and zeta_0 dimensions
        w = (1 / scale_vector1) * w
        wg = -1 * w * g

        # Row scaling D_rho by wg
        A = D_zeta.T @ (wg[..., :, None] * D_zeta)

        # the scale due to the derivative
        wc = w * c
        idx = jnp.arange(num_zeta)

        A = A.at[..., idx, idx].add(wc)

        b_inv = jnp.sqrt(jnp.reciprocal(w * f))

        A = (b_inv[..., :, None] * A) * b_inv[..., None, :]

        # apply dirichlet BC to X
        w, v = jnp.linalg.eigh(A[..., 1:-1, 1:-1])

    else:
        # toroidal step size between points along field lines is assumed uniform
        dz = grid.nodes[grid.unique_zeta_idx[:2], 2]
        dz = dz[1] - dz[0]

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
    dim=5,
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
    description="Normalized squared growth rate"
    + "using the most compact representation of diffmatrices",
    dim=1,
    params=["Psi"],
    transforms={"grid": [], "diffmat": []},
    profiles=[],
    coordinates="rtz",
    data=[
        "g_rr|PEST",
        "g_rv|PEST",
        "g_rp|PEST",
        "g_vv|PEST",
        "g_vp|PEST",
        "g_pp|PEST",
        "g^rr",
        "g^rv",
        "g^rz",
        "J^theta_PEST",
        "J^zeta",
        "|J|",
        "sqrt(g)_PEST",
        "(sqrt(g)_PEST_r)|PEST",
        "(sqrt(g)_PEST_v)|PEST",
        "(sqrt(g)_PEST_p)|PEST",
        "finite-n instability drive",
        "iota",
        "psi_r",
        "psi_rr",
        "p",
        "a",
    ],
    n_rho_max="int: 2 x maximum radial mode number",
    n_theta_max="int: 2 x maximum poloidal mode number",
    n_zeta_max="int: 2 x maximum toroidal mode number",
    axisym="bool: if the equilibrium is axisymmetric",
    n_mode_axisym="int: toroidal mode number to study",
    incompressible="bool: imposes incompressibility",
    stable_only="bool: for testing only, materialize "
    + "and eigendecompose the stable part of the matrix",
)
def _AGNI(params, transforms, profiles, data, **kwargs):
    """
    AGNI: Analysis of Global Normal-modes in Ideal MHD.

    Based on the original source here:
    https://github.com/rahulgaur104/AGNI/tree/master

    A finite-n stability eigenvalue solver.
    Currenly only finds fixed boundary unstable modes at
    low to medium resolution.

    This version of the code keeps the derivatives of the form
    partial_rho (iota psi' xi^rho) more compact which leads to
    fewer terms and even order derivatives. For this version
    the PSD version of A is actually very close to PSD ~ 1e-12.
    B is perfectly PSD
    """
    a_N = data["a"]
    B_N = params["Psi"] / (jnp.pi * a_N**2)

    iota = data["iota"][:, None]

    psi_r = data["psi_r"][:, None] / (0.5 * a_N**2 * B_N)

    psi_rr = data["psi_rr"][:, None] / (0.5 * a_N**2 * B_N)

    psi_r2 = psi_r**2
    psi_r3 = psi_r**3

    iota_psi_r2 = iota * psi_r2

    # Add a tiny shift because sometimes the pressure can be
    # slightly negative in the edge
    p0 = mu_0 * data["p"][:, None] / B_N**2 + 1e-12

    # Arbitrary choice. Mostly used to decide the range of eigenvalues of
    # the mass matrix. Pre-conditioning should remove this factor
    n0 = 1e2

    axisym = kwargs.get("axisym", False)

    # For axisymmetric equilibria n_mode_axisym will decide the toroidal
    # mode number to analyze.
    n_mode_axisym = kwargs.get("n_mode_axisym", 1)
    incompressible = kwargs.get("incompressible", False)
    # --no-verify stable_only = kwargs.get("stable_only", False)

    n_rho_max = kwargs.get("n_rho_max", 8)
    n_theta_max = kwargs.get("n_theta_max", 8)

    if axisym:
        if n_mode_axisym == 0 and incompressible:
            return NotImplementedError
        else:
            # Each componenet of xi can be written as the Fourier sum of
            # two modes in the toroidal direction
            D_zeta0 = n_mode_axisym * jnp.array([[0, -1], [1, 0]])
            n_zeta_max = 2
    else:
        n_zeta_max = kwargs.get("n_zeta_max", 4)
        D_zeta0 = transforms["diffmat"].zeta_diffmat

    def _f(x):
        x_0 = 0.8
        m_1 = 3.0
        m_2 = 1.0
        lower = x_0 * (1 - jnp.exp(-m_1 * (x + 1)) + 0.5 * (x + 1) * jnp.exp(-2 * m_1))
        upper = (1 - x_0) * (jnp.exp(m_2 * (x - 1)) + 0.5 * (x - 1) * jnp.exp(-2 * m_2))
        eps = 1.0e-3
        return eps + (1 - eps) * (lower + upper)

    # ∫dρₛ (∂X/∂ρₛ) = ∫dρ f'(ρ) (∂ρ/∂ρₛ) (∂X/∂ρ)
    dx_f = jax.vmap(jax.grad(_f))

    x, w = leggauss_lob(n_rho_max)
    scale_vector1 = dx_f(x) ** -1

    # --no-verify scale_vector1 = jnp.ones_like(x0) * (1 - 1e-3)
    # --no-verify h = (1-1e-3)/(n_rho_max-1)

    scale_x1 = scale_vector1[:, None]

    # Get differentiation matrices
    D_rho0 = transforms["diffmat"].rho_diffmat * scale_x1
    D_theta0 = transforms["diffmat"].theta_diffmat

    wrho = jnp.diag(1 / scale_vector1 * w)
    wrho = wrho.at[jnp.abs(wrho) < 1e-12].set(0.0)

    ### assuming uniform spacing in and θ and ζ
    wtheta = 2 * jnp.pi / n_theta_max
    wzeta = 2 * jnp.pi / n_zeta_max

    I_rho0 = jax.lax.stop_gradient(jnp.eye(n_rho_max))
    I_theta0 = jax.lax.stop_gradient(jnp.eye(n_theta_max))
    I_zeta0 = jax.lax.stop_gradient(jnp.eye(n_zeta_max))

    D_rho = jax.lax.stop_gradient(jnp.kron(D_rho0, jnp.kron(I_theta0, I_zeta0)))
    D_theta = jax.lax.stop_gradient(jnp.kron(I_rho0, jnp.kron(D_theta0, I_zeta0)))
    D_zeta = jax.lax.stop_gradient(jnp.kron(I_rho0, jnp.kron(I_theta0, D_zeta0)))

    D_thetaT = jax.lax.stop_gradient(jnp.kron(I_rho0, jnp.kron(D_theta0.T, I_zeta0)))
    D_zetaT = jax.lax.stop_gradient(jnp.kron(I_rho0, jnp.kron(I_theta0, D_zeta0.T)))

    W = jnp.diag(jnp.kron(wrho * wtheta * wzeta, jnp.kron(I_theta0, I_zeta0)))[:, None]

    n_total = n_rho_max * n_theta_max * n_zeta_max

    # Define block indices
    rho_idx = slice(0, n_total)
    theta_idx = slice(n_total, 2 * n_total)
    zeta_idx = slice(2 * n_total, 3 * n_total)

    ## Create the full matrix
    A = jnp.zeros((3 * n_total, 3 * n_total))
    B = jnp.zeros((3 * n_total, 3 * n_total))

    sqrtg = data["sqrt(g)_PEST"][:, None] * 1 / a_N**3

    sqrtg_r = data["(sqrt(g)_PEST_r)|PEST"][:, None] * 1 / a_N**3
    sqrtg_v = data["(sqrt(g)_PEST_v)|PEST"][:, None] * 1 / a_N**3
    sqrtg_p = data["(sqrt(g)_PEST_p)|PEST"][:, None] * 1 / a_N**3

    partial_z_log_sqrtg = (sqrtg_p / sqrtg).flatten()
    partial_r_log_sqrtg = (sqrtg_r / sqrtg).flatten()
    partial_v_log_sqrtg = (sqrtg_v / sqrtg).flatten()

    psi_r_over_sqrtg = psi_r / sqrtg

    g_rr = data["g_rr|PEST"][:, None] * 1 / a_N**2
    g_vv = data["g_vv|PEST"][:, None] * 1 / a_N**2
    g_pp = data["g_pp|PEST"][:, None] * 1 / a_N**2  # finite on-axis

    g_rv = data["g_rv|PEST"][:, None] * 1 / a_N**2
    g_rp = data["g_rp|PEST"][:, None] * 1 / a_N**2
    g_vp = data["g_vp|PEST"][:, None] * 1 / a_N**2

    g_sup_rr = data["g^rr"][:, None] * a_N**2
    g_sup_rv = data["g^rv"][:, None] * a_N**2
    g_sup_rp = data["g^rz"][:, None] * a_N**2

    J2 = ((mu_0 * data["|J|"]) ** 2)[:, None] * (a_N / B_N) ** 2
    j_sup_theta = mu_0 * data["J^theta_PEST"][:, None] * a_N**2 / B_N
    j_sup_zeta = mu_0 * data["J^zeta"][:, None] * a_N**2 / B_N

    # instability drive term
    F = -1 * mu_0 * data["finite-n instability drive"][:, None] * (1 / B_N) ** 2

    C_zeta = jnp.diag(iota * partial_z_log_sqrtg) + iota * D_zeta
    C_rho = (
        jnp.diag(psi_r * partial_r_log_sqrtg + psi_rr) + psi_r * D_rho
    )  # (n_total, n_total)
    C_theta = jnp.diag(partial_v_log_sqrtg) + D_theta

    ####################
    ####----Q²_ρρ----####
    ####################
    A = A.at[rho_idx, rho_idx].add(
        D_thetaT @ ((psi_r_over_sqrtg * iota**2 * psi_r3 * W * g_rr) * D_theta)
        + D_zetaT @ ((psi_r_over_sqrtg * W * psi_r3 * g_rr) * D_zeta)
        + D_thetaT @ ((psi_r_over_sqrtg * iota * psi_r3 * W * g_rr) * D_zeta)
        + ((psi_r_over_sqrtg * iota * psi_r3 * W * g_rr) * D_zeta).T @ D_theta
    )

    ####################
    ####----Q²_ϑϑ ----####
    ####################
    # enforcing symmetry exactly
    A = A.at[theta_idx, theta_idx].add(
        0.5
        * (
            D_zetaT @ ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta)
            + ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta).T @ D_zeta
        )
    )
    A = A.at[zeta_idx, zeta_idx].add(
        0.5
        * (
            D_zetaT @ ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta)
            + ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta).T @ D_zeta
        )
    )

    A = A.at[theta_idx, zeta_idx].add(
        -1.0 * (D_zetaT @ ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta))
    )

    A = A.at[rho_idx, rho_idx].add(
        +(D_rho * iota_psi_r2.T).T
        @ ((psi_r_over_sqrtg * W * g_vv / psi_r) * (D_rho * iota_psi_r2.T))
    )

    A = A.at[rho_idx, theta_idx].add(
        -1 * (D_rho * iota_psi_r2.T).T @ ((psi_r_over_sqrtg * W * g_vv) * D_zeta)
    )

    A = A.at[rho_idx, zeta_idx].add(
        1 * (D_rho * iota_psi_r2.T).T @ ((psi_r_over_sqrtg * W * g_vv) * D_zeta)
    )

    ####################
    ####----Q²_ζζ----####
    ####################
    A = A.at[theta_idx, theta_idx].add(
        0.5
        * (
            D_theta.T @ ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta)
            + ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta).T @ D_theta
        )
    )
    A = A.at[zeta_idx, zeta_idx].add(
        0.5
        * (
            D_theta.T @ ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta)
            + ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta).T @ D_theta
        )
    )

    A = A.at[theta_idx, zeta_idx].add(
        -1 * D_theta.T @ ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta)
    )

    A = A.at[rho_idx, rho_idx].add(
        +(D_rho * psi_r2.T).T
        @ ((psi_r_over_sqrtg * W * g_pp / psi_r) * (D_rho * psi_r2.T))
    )

    A = A.at[rho_idx, theta_idx].add(
        1 * (D_rho * psi_r2.T).T @ ((psi_r_over_sqrtg * W * g_pp) * D_theta)
    )

    A = A.at[rho_idx, zeta_idx].add(
        -1 * (D_rho * psi_r2.T).T @ ((psi_r_over_sqrtg * W * g_pp) * D_theta)
    )

    ####################
    ####----Q²_ρϑ----####
    ####################
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            D_theta.T
            @ ((iota * psi_r * psi_r_over_sqrtg * W * g_rv) * (D_rho * iota_psi_r2.T))
            + D_zeta.T
            @ ((psi_r * psi_r_over_sqrtg * W * g_rv) * (D_rho * iota_psi_r2.T))
        )
    )

    ## transposed part of the mixed term along the ρ-ρ block diagonal
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            ((iota * psi_r * psi_r_over_sqrtg * W * g_rv) * (D_rho * iota_psi_r2.T)).T
            @ D_theta
            + ((psi_r * psi_r_over_sqrtg * W * g_rv) * (D_rho * iota_psi_r2.T)).T
            @ D_zeta
        )
    )

    A = A.at[rho_idx, theta_idx].add(
        D_theta.T @ ((iota * psi_r2 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
        + D_zeta.T @ ((psi_r2 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
    )
    A = A.at[rho_idx, zeta_idx].add(
        -1
        * (
            D_theta.T @ ((iota * psi_r2 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
            + D_zeta.T @ ((psi_r2 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
        )
    )

    ######################
    ####-----Q²_ρζ-----####
    ######################
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            D_theta.T
            @ ((iota * psi_r * psi_r_over_sqrtg * W * g_rp) * (D_rho * psi_r2.T))
            + D_zeta.T @ ((psi_r * psi_r_over_sqrtg * W * g_rp) * (D_rho * psi_r2.T))
        )
    )

    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            ((iota * psi_r * psi_r_over_sqrtg * W * g_rp) * (D_rho * psi_r2.T)).T
            @ D_theta
            + ((psi_r * psi_r_over_sqrtg * W * g_rp) * (D_rho * psi_r2.T)).T @ D_zeta
        )
    )

    A = A.at[rho_idx, theta_idx].add(
        -1
        * (
            D_theta.T @ ((iota * psi_r2 * psi_r_over_sqrtg * W * g_rp) * D_theta)
            + D_zeta.T @ ((psi_r2 * psi_r_over_sqrtg * W * g_rp) * D_theta)
        )
    )
    A = A.at[rho_idx, zeta_idx].add(
        1
        * (
            D_theta.T @ ((iota * psi_r2 * psi_r_over_sqrtg * W * g_rp) * D_theta)
            + D_zeta.T @ ((psi_r2 * psi_r_over_sqrtg * W * g_rp) * D_theta)
        )
    )

    ##########################
    #######-----Q²_ϑζ-----#####
    ##########################
    A = A.at[theta_idx, theta_idx].add(
        -1
        * (
            D_zeta.T @ ((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta)
            + ((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta).T @ D_zeta
        )
    )

    A = A.at[zeta_idx, zeta_idx].add(
        -1
        * (
            D_zeta.T @ ((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta)
            + ((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta).T @ D_zeta
        )
    )

    A = A.at[rho_idx, theta_idx].add(
        -1
        * (
            (D_rho * psi_r2.T).T @ ((psi_r_over_sqrtg * W * g_vp) * D_zeta)
            - (D_rho * iota_psi_r2.T).T @ ((psi_r_over_sqrtg * W * g_vp) * D_theta)
        )
    )
    A = A.at[rho_idx, zeta_idx].add(
        1
        * (
            (D_rho * psi_r2.T).T @ ((psi_r_over_sqrtg * W * g_vp) * D_zeta)
            - (D_rho * iota_psi_r2.T).T @ ((psi_r_over_sqrtg * W * g_vp) * D_theta)
        )
    )

    A = A.at[theta_idx, zeta_idx].add(
        D_zeta.T @ ((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta)
        + ((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta).T @ D_zeta
    )

    A = A.at[rho_idx, rho_idx].add(
        1
        * (
            (D_rho * iota_psi_r2.T).T
            @ ((psi_r_over_sqrtg * W * g_vp / psi_r) * (D_rho * psi_r2.T))
        )
    )
    # ρ-ρ symmetrizing term
    A = A.at[rho_idx, rho_idx].add(
        1
        * (
            ((psi_r_over_sqrtg * W * g_vp / psi_r) * (D_rho * psi_r2.T)).T
            @ (D_rho * iota_psi_r2.T)
        )
    )

    # Mixed Q-J term ξ^ρ (𝐉 × ∇ρ)/|∇ ρ|² ⋅ 𝐐
    # \xi^{\rho} (\mathbf{J} \times \nabla\rho)/|\nabla \rho|^2 \cdot \mathbf{Q}
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            (
                W
                * psi_r3
                * sqrtg
                * (j_sup_theta * g_sup_rp + j_sup_zeta * g_sup_rv)
                / g_sup_rr
            )
            * (iota * D_theta + D_zeta)
            + (W * sqrtg * psi_r * j_sup_zeta) * (D_rho * iota_psi_r2.T)
            + (W * sqrtg * psi_r * j_sup_theta) * (D_rho * psi_r2.T)
        )
    )

    # ρ-ρ block transposed for symmetry
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            (
                (
                    W
                    * psi_r3
                    * sqrtg
                    * (j_sup_theta * g_sup_rp + j_sup_zeta * g_sup_rv)
                    / g_sup_rr
                )
                * (iota * D_theta + D_zeta)
            ).T
            + ((W * sqrtg * psi_r * j_sup_zeta) * (D_rho * iota_psi_r2.T)).T
            + ((W * sqrtg * psi_r * j_sup_theta) * (D_rho * psi_r2.T)).T
        )
    )

    A = A.at[rho_idx, theta_idx].add(
        -(W * psi_r2 * sqrtg * j_sup_theta) * D_theta
        + (W * psi_r2 * sqrtg * j_sup_zeta) * D_zeta
    )
    A = A.at[rho_idx, zeta_idx].add(
        +(W * psi_r2 * sqrtg * j_sup_theta) * D_theta
        - (W * psi_r2 * sqrtg * j_sup_zeta) * D_zeta
    )

    ## diagonal |J|² term
    A = A.at[rho_idx, rho_idx].add(jnp.diag((psi_r2 * W * sqrtg * J2).flatten()))

    # Mass matrix (must be symmetric positive definite)
    B = B.at[rho_idx, rho_idx].add(jnp.diag(n0 * (W * psi_r2 * sqrtg * g_rr).flatten()))
    B = B.at[theta_idx, theta_idx].add(jnp.diag(n0 * (W * sqrtg * g_vv).flatten()))
    B = B.at[zeta_idx, zeta_idx].add(
        jnp.diag(n0 * (W * iota**2 * sqrtg * g_pp).flatten())
    )

    B = B.at[rho_idx, theta_idx].add(
        jnp.diag(n0 * (W * psi_r * sqrtg * g_rv).flatten())
    )
    B = B.at[rho_idx, zeta_idx].add(
        jnp.diag(n0 * (W * psi_r * iota * sqrtg * g_rp).flatten())
    )
    B = B.at[theta_idx, zeta_idx].add(
        jnp.diag(n0 * (W * iota * sqrtg * g_vp).flatten())
    )

    if incompressible is False:
        # purely stabilizing and doesn't change the marginal stability
        # To improve performance set exact to False
        exact = False
        if exact:
            gamma = 5 / 3
            A = A.at[rho_idx, rho_idx].add(C_rho.T @ ((gamma * sqrtg * W * p0) * C_rho))
            A = A.at[theta_idx, theta_idx].add(
                C_theta.T @ ((gamma * sqrtg * W * p0) * C_theta)
            )
            A = A.at[zeta_idx, zeta_idx].add(
                C_zeta.T @ ((gamma * sqrtg * W * p0) * C_zeta)
            )
            A = A.at[rho_idx, theta_idx].add(
                C_rho.T @ ((gamma * sqrtg * W * p0) * C_theta)
            )
            A = A.at[rho_idx, zeta_idx].add(
                C_rho.T @ ((gamma * sqrtg * W * p0) * C_zeta)
            )
            A = A.at[theta_idx, zeta_idx].add(
                C_theta.T @ ((gamma * sqrtg * W * p0) * C_zeta)
            )

    ### Instability drive term
    Au = jnp.zeros((3 * n_total, 3 * n_total))
    Au = Au.at[rho_idx, rho_idx].add(jnp.diag((W * psi_r2 * sqrtg * F).flatten()))

    A = A.at[theta_idx, rho_idx].set(A[rho_idx, theta_idx].T)
    A = A.at[zeta_idx, rho_idx].set(A[rho_idx, zeta_idx].T)
    A = A.at[zeta_idx, theta_idx].set(A[theta_idx, zeta_idx].T)

    B = B.at[theta_idx, rho_idx].set(B[rho_idx, theta_idx].T)
    B = B.at[zeta_idx, rho_idx].set(B[rho_idx, zeta_idx].T)
    B = B.at[zeta_idx, theta_idx].set(B[theta_idx, zeta_idx].T)

    D = jnp.diag(1 / jnp.sqrt(jnp.diag(B)))

    # Preconditioning improves B, does not affect A
    A = D @ (A @ D.T)
    Au = D @ (Au @ D.T)
    B = D @ (B @ D.T)

    def component_to_node_permutn(N: int) -> jnp.ndarray:
        """
        Build the permutation that converts component-major ordering to node-major.

        Component-major vector layout (length 3N):
            [ rho_1..N | theta_1..N | zeta_1..N ]

        Node-major vector layout (length 3N):
            [ rho_1, theta_1, zeta_1 | ... | rho_N, theta_N, zeta_N ]

        The returned permutation `p` satisfies:
            x_node = x_comp[p]
            M_node = M_comp[p][:, p]

        Parameters
        ----------
        N : int
            Number of spatial nodes per component.

        Returns
        -------
        jnp.ndarray, shape (3*N,)
            Permutation indices from component-major to node-major.
        """
        k = jnp.arange(N, dtype=jnp.int64)

        perm = jnp.empty(3 * N, dtype=jnp.int64)
        perm = perm.at[3 * k + 0].set(k)
        perm = perm.at[3 * k + 1].set(N + k)
        perm = perm.at[3 * k + 2].set(2 * N + k)

        return perm

    def _assemble_diagblocks_comp_major(
        blocks, rho_idx, theta_idx, zeta_idx, sym=False
    ):
        """
        blocks: (N,3,3). Works for L (lower-tri) or B_blocks (symmetric).

        *_idx:  python slices for component-major ranges.

        NOTE that it currently only works for assembling lower diagonal
        matrices such as the ones formed by cholesky. Generalize logic later.
        """
        N = blocks.shape[0]
        big = jnp.zeros((3 * N, 3 * N))

        # Diagonal sub-blocks
        big = big.at[rho_idx, rho_idx].set(jnp.diag(blocks[:, 0, 0]))
        big = big.at[theta_idx, theta_idx].set(jnp.diag(blocks[:, 1, 1]))
        big = big.at[zeta_idx, zeta_idx].set(jnp.diag(blocks[:, 2, 2]))

        # Off-diagonal (lower) subblocks — upper are zero for a Cholesky L anyway
        big = big.at[theta_idx, rho_idx].set(jnp.diag(blocks[:, 1, 0]))
        big = big.at[zeta_idx, rho_idx].set(jnp.diag(blocks[:, 2, 0]))
        big = big.at[zeta_idx, theta_idx].set(jnp.diag(blocks[:, 2, 1]))

        if sym:
            big = big.at[rho_idx, theta_idx].set(jnp.diag(blocks[:, 0, 1]))
            big = big.at[rho_idx, zeta_idx].set(jnp.diag(blocks[:, 0, 2]))
            big = big.at[theta_idx, zeta_idx].set(jnp.diag(blocks[:, 1, 2]))

        return big

    if axisym and incompressible:

        # store indices needed to apply dirichlet BC to ξ^ρ
        keep_1 = jnp.arange(
            n_theta_max * n_zeta_max, n_total - n_theta_max * n_zeta_max
        )
        keep_2 = jnp.arange(n_total, 2 * n_total)
        keep = jnp.concatenate([keep_1, keep_2])

        d_r = 1.0 / jnp.diag(D)[rho_idx]
        d_v = 1.0 / jnp.diag(D)[theta_idx]
        d_z = 1.0 / jnp.diag(D)[zeta_idx]

        C_zeta = C_zeta * d_z[None, :]

        # TODO: convert to batched inversion for speed
        C_rho = C_rho * d_r[None, :]
        C_zeta_inv_C_rho = jnp.linalg.solve(C_zeta, C_rho)

        C_theta = C_theta * d_v[None, :]
        C_zeta_inv_C_theta = jnp.linalg.solve(C_zeta, C_theta)

        ## ξ^ζ = −((C_ζ D⁻¹)⁻¹ C_ρ D⁻¹ ξₛ^ρ + (C_ζ D⁻¹)⁻¹ C_θ D⁻¹ ξₛ^θ)

        # Impose incompressibility
        A = A.at[rho_idx, rho_idx].add(
            -1
            * (
                A[rho_idx, zeta_idx] @ C_zeta_inv_C_rho
                + (A[rho_idx, zeta_idx] @ C_zeta_inv_C_rho).T
            )
            + C_zeta_inv_C_rho.T @ A[zeta_idx, zeta_idx] @ C_zeta_inv_C_rho
        )

        A = A.at[rho_idx, theta_idx].add(
            -1
            * (
                A[rho_idx, zeta_idx] @ C_zeta_inv_C_theta
                + C_zeta_inv_C_rho.T @ A[zeta_idx, theta_idx]
            )
            + C_zeta_inv_C_rho.T @ A[zeta_idx, zeta_idx] @ C_zeta_inv_C_theta
        )

        A = A.at[theta_idx, theta_idx].add(
            -1
            * (
                A[theta_idx, zeta_idx] @ C_zeta_inv_C_theta
                + (A[theta_idx, zeta_idx] @ C_zeta_inv_C_theta).T
            )
            + C_zeta_inv_C_theta.T @ A[zeta_idx, zeta_idx] @ C_zeta_inv_C_theta
        )

        # Fill out the lower part using symmetry
        A = A.at[theta_idx, rho_idx].set(A[rho_idx, theta_idx].T)

        # A2u only has a non-zero rho-rho component
        Au = Au.at[rho_idx, rho_idx].add(
            C_zeta_inv_C_rho.T @ Au[zeta_idx, zeta_idx] @ C_zeta_inv_C_rho
        )

        B = B.at[rho_idx, rho_idx].add(
            -1
            * (
                B[rho_idx, zeta_idx] @ C_zeta_inv_C_rho
                + (B[rho_idx, zeta_idx] @ C_zeta_inv_C_rho).T
            )
            + C_zeta_inv_C_rho.T @ B[zeta_idx, zeta_idx] @ C_zeta_inv_C_rho
        )

        B = B.at[rho_idx, theta_idx].add(
            -1
            * (
                B[rho_idx, zeta_idx] @ C_zeta_inv_C_theta
                + C_zeta_inv_C_rho.T @ B[zeta_idx, theta_idx]
            )
            + C_zeta_inv_C_rho.T @ B[zeta_idx, zeta_idx] @ C_zeta_inv_C_theta
        )

        B = B.at[theta_idx, theta_idx].add(
            -1
            * (
                B[theta_idx, zeta_idx] @ C_zeta_inv_C_theta
                + (B[theta_idx, zeta_idx] @ C_zeta_inv_C_theta).T
            )
            + C_zeta_inv_C_theta.T @ B[zeta_idx, zeta_idx] @ C_zeta_inv_C_theta
        )

        # Fill out the lower part using symmetry
        B = B.at[theta_idx, rho_idx].set(B[rho_idx, theta_idx].T)

        # The will become one of the most expensive parts
        L = jnp.linalg.cholesky(B)

        ## (L⁻¹ A L⁻ᵀ)
        A_LTinv = jax.lax.linalg.triangular_solve(L.T, A, left_side=False, lower=False)
        Ahat = jax.lax.linalg.triangular_solve(L, A_LTinv, left_side=True, lower=True)

        # (L⁻¹ Aᵤ L⁻ᵀ)
        Au_LTinv = jax.lax.linalg.triangular_solve(
            L.T, Au, left_side=False, lower=False
        )
        Auhat = jax.lax.linalg.triangular_solve(L, Au_LTinv, left_side=True, lower=True)

        A2 = Ahat[jnp.ix_(keep, keep)] + Auhat[jnp.ix_(keep, keep)]

        w, v = jnp.linalg.eigh((A2 + A2.T) / 2)

    else:  # if non-axisymmetric or compressible or both
        B_blocks = jnp.zeros((n_total, 3, 3))

        B_blocks = B_blocks.at[:, 0, 0].set(jnp.diag(B[rho_idx, rho_idx]))
        B_blocks = B_blocks.at[:, 1, 1].set(jnp.diag(B[theta_idx, theta_idx]))
        B_blocks = B_blocks.at[:, 2, 2].set(jnp.diag(B[zeta_idx, zeta_idx]))

        B_blocks = B_blocks.at[:, 0, 1].set(jnp.diag(B[rho_idx, theta_idx]))
        B_blocks = B_blocks.at[:, 1, 0].set(jnp.diag(B[theta_idx, rho_idx]))

        B_blocks = B_blocks.at[:, 2, 0].set(jnp.diag(B[rho_idx, zeta_idx]))
        B_blocks = B_blocks.at[:, 0, 2].set(jnp.diag(B[zeta_idx, rho_idx]))

        B_blocks = B_blocks.at[:, 1, 2].set(jnp.diag(B[theta_idx, zeta_idx]))
        B_blocks = B_blocks.at[:, 2, 1].set(jnp.diag(B[zeta_idx, theta_idx]))

        L = jnp.linalg.cholesky(B_blocks)  # (N,3,3)
        I3 = jnp.tile(jnp.eye(3), (L.shape[0], 1, 1))
        Linv = jax.lax.linalg.triangular_solve(
            L, I3, left_side=True, lower=True
        )  # (N,3,3)

        # components to node permutations
        p = component_to_node_permutn(n_total)
        A2 = A[p][:, p]
        A2u = Au[p][:, p]

        # L^-1 A L^-T
        A2 = A2.reshape(n_total, 3, n_total, 3)
        A2 = jnp.einsum("ikl,iljq,jbq->ikjb", Linv, A2, Linv)
        A2 = A2.reshape(3 * n_total, 3 * n_total)

        A2u = A2u.reshape(n_total, 3, n_total, 3)
        A2u = jnp.einsum("ikl,iljq,jbq->ikjb", Linv, A2u, Linv)
        A2u = A2u.reshape(3 * n_total, 3 * n_total)

        # node to component permutation
        pinv = jnp.empty_like(p)
        pinv = pinv.at[p].set(jnp.arange(3 * n_total))

        A2 = A2[pinv][:, pinv]
        A2u = A2u[pinv][:, pinv]

        # store indices needed to apply dirichlet BC to ξ^ρ
        keep_1 = jnp.arange(
            n_theta_max * n_zeta_max, n_total - n_theta_max * n_zeta_max
        )
        keep_2 = jnp.arange(n_total, 3 * n_total)
        keep = jnp.concatenate([keep_1, keep_2])

        if incompressible:  # Only enforce incompressibility here
            # ∇⋅𝛏 = C_ρ ξ^ρ + C_θ ξ^θ + C_ζ ξ^ζ

            ## Assemble L_full from blocks of L (only for comparison)
            # --no-verify Linv_full = _assemble_diagblocks_comp_major(Linv, rho_idx)
            # --no-verify L_test = jnp.linalg.cholesky(B)
            ##max|Linv_full - L_test⁻¹| ≈ 3.55e-15

            # C.shape (N, 3N)
            C = jnp.concatenate([C_rho, C_theta, C_zeta], axis=1)

            d = jnp.diag(D)  # (3N,)
            C_scaled = C * d[None, :]  # right-multiply by D via column scaling

            # Apply L2⁻ᵀ per node using the existing L2inv
            # Ĉ = C D L⁻ᵀ
            Linv_T = jnp.swapaxes(Linv, 1, 2)  # (N, 3, 3)
            C_node = C_scaled[:, p].reshape(n_total, n_total, 3)
            Chat_node = jnp.einsum("mil, ilk -> mik", C_node, Linv_T)
            Chat = Chat_node.reshape(n_total, 3 * n_total)[:, pinv]

            Chat = Chat[keep_1][:, keep]
            row_norm = jnp.clip(
                jnp.linalg.norm(Chat, axis=1, keepdims=True), 1e-300, jnp.inf
            )
            Chat = Chat / row_norm

            # Orthogonal projector P = I - C^T (L_G L_G^T)⁻¹ Ĉ
            G = Chat @ Chat.T
            G = (G + G.T) / 2 + 1e-14 * jnp.eye(
                n_total - 2 * n_theta_max * n_zeta_max
            )  # Gram matrix w ridge

            # The will become one of the most expensive parts
            L_G = jnp.linalg.cholesky(G)

            Y = jax.lax.linalg.triangular_solve(L_G, Chat, left_side=True, lower=True)
            S = jax.lax.linalg.triangular_solve(L_G.T, Y, left_side=True, lower=False)
            CTS = Chat.T @ S  # = C^T (L_G L_G^T)⁻¹ Ĉ

            ## applying the boundary condition first
            ## BCs before projection ≠ projection before BCs
            A2_bc = A2[jnp.ix_(keep, keep)]
            A2u_bc = A2u[jnp.ix_(keep, keep)]

            ## Projected operator A_proj = P A P without forming P
            A2_proj = A2_bc - A2_bc @ CTS - CTS @ A2_bc + CTS @ A2_bc @ CTS
            A2_proj = (A2_proj + A2_proj.T) / 2

            A2_proj = A2_proj.at[jnp.diag_indices_from(A2_proj)].add(1e-9)

            # --no-verify w0, v0 = jnp.linalg.eigh((A2_proj + A2_proj.T) / 2)
            # --no-verify print(w0)

            A2u_proj = A2u_bc - A2u_bc @ CTS - CTS @ A2u_bc + CTS @ A2u_bc @ CTS
            A2u_proj = (A2u_proj + A2u_proj.T) / 2

            A3_proj = A2u_proj + A2_proj

            w, v = jnp.linalg.eigh((A3_proj + A3_proj.T) / 2)

            # Small for modes far from marginality
            print(Chat @ v[:, 0])

            # --no-verify P = jnp.eye(CTS.shape[0], CTS.dtype) - CTS
            # --no-verify print("sym=", float(jnp.linalg.norm(P - P.T)),
            # --no-verify       "idem=", float(jnp.linalg.norm(P@P - P)),
            # --no-verify        "CP=", float(jnp.linalg.norm(Chat @ P)))

        else:
            ## Shift the diagonal of A to ensure positive definiteness
            ## The estimate must be accurate. If A is diagonally dominant
            ## use Gerhsgorin theorem to estimate the lowest eigenvalue
            A2 = A2.at[jnp.diag_indices_from(A2)].add(1e-9)

            A3 = A2[jnp.ix_(keep, keep)] + A2u[jnp.ix_(keep, keep)]

            w, v = jnp.linalg.eigh((A3 + A3.T) / 2)
            print(w)

    data["finite-n lambda"] = w
    data["finite-n eigenfunction"] = v

    return data
