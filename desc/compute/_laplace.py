"""Compute functions for multiply connected Laplace solver as described in [1]_.

References
----------
.. [1] Unalmis et al. New high-order accurate free surface stellarator
        equilibria optimization and boundary integral methods in DESC.

"""

from functools import partial
from typing import NamedTuple, Optional

import equinox as eqx
import jax
import lineax as lx

try:
    import optimistix as optx
except ImportError:
    pass

from interpax_fft import rfft_interp2d

from desc.backend import jnp
from desc.integrals.singularities import (
    _kernel_BS_plus_grad_S,
    _kernel_dipole,
    _kernel_dipole_plus_half,
    _kernel_monopole,
    _nonsingular_part,
    _prune_data,
    get_interpolator,
    singular_integral,
)
from desc.utils import cross, dot, errorif

from .data_index import register_compute_fun


class Options(NamedTuple):
    """Laplace solver options."""

    Phi_tilde_0: Optional[jax.Array] = None
    """Initial guess for iteration."""

    atol: float = 1e-6
    """Absolute error tolerance for the iterative linear solve. Default is ``1e-6``."""

    rtol: float = 1e-6
    """Relative error tolerance for the iterative linear solve. Default is ``1e-6``."""

    max_steps: int = 50
    """Maximum number of steps for iterative linear solve.

    For ``"gmres"``, this is the number of restart cycles. Lineax's default
    restart length is 20, so one GMRES step applies the operator at
    most 20 times. Two restart cycles are typically sufficient for a primal
    solve, but differentiated solves can require more cycles.
    For ``"bicgstab"``, each step applies the operator twice, and
    ``max_steps=50`` is typically sufficient for convergence. The default of
    50 is shared by all iterative methods; converged solvers terminate early.
    """

    problem: str = "interior Neumann"
    """Boundary value problem to solve.

    One of ``"interior Neumann"``, ``"exterior Neumann"``, or ``"interior Dirichlet"``.
    (In some routines this may be determined automatically.)
    """

    solve_method: str = "gmres"
    """Method to use for the scalar potential solve.

    One of ``"gmres"``, ``"bicgstab"``, or ``"direct"``. Default is GMRES.
    Fastest is typically BiCGStab (heed the note in the max_steps option).
    If an iterative solver errors due to incompatibility with old JAX versions,
    ``"fixed_point"`` can be selected instead if ``optimistix`` is installed.
    """

    full_output: bool = False
    """Whether to return diagnostic output of the iterative potential solve.

    If ``True``, computes the maximum error ``Phi_tilde error`` and stores the number
    of steps ``num_steps`` used by the scalar potential solver. Default is
    ``False``.
    """

    chunk_size: Optional[int] = None
    """Size to split integral computation into chunks.

    If no chunking should be done or the chunk size is the full input then
    supply ``None``. Default is ``None``. Recommend to verify computation with
    ``chunk_size`` set to a small number due to bugs in JAX or XLA.
    """

    B_coil_chunk_size: Optional[int] = None
    """Size to split coil integral computation into chunks.

    If no chunking should be done or the chunk size is the full input then
    supply ``None``. Default is ``None``.
    """

    D_quad: bool = False
    """Developer option for double-layer potential quadrature.

    Set to ``True`` to perform double-layer potential quadrature without removing
    singularities. Default is ``False``.
    """

    throw: bool = True
    """Whether to raise an error if an iterative solve fails.

    Default is ``True`` so an unconverged potential is never used silently.
    Set this to ``False`` only when the returned status is intentionally being
    inspected, for example when constructing a convergence scan.
    """


def _check_solve_method(solve_method):
    """Check that a scalar potential solve method is valid."""
    errorif(
        solve_method not in {"fixed_point", "gmres", "bicgstab", "direct"},
        msg=(
            "solve_method must be one of 'fixed_point', 'gmres', 'bicgstab', "
            f"or 'direct', got {solve_method!r}."
        ),
    )


def _D_plus_half(
    eval_data,
    source_data,
    interpolator,
    basis=None,
    chunk_size=None,
    prune_data=True,
    _D_quad=False,
):
    """Compute (D[Φ] + Φ/2)(x).

    D[Φ](x) = ∫_y Φ(y)〈∇_x G(x−y),ds(y)〉.

    Parameters
    ----------
    basis : DoubleFourierSeries
        If not supplied, then computes (D[Φ] + Φ/2)(x).
        If supplied, then constructs the operator which
        acts on the spectral coefficients of Φ in the supplied basis.
    prune_data : bool
        Whether the data should be pruned. Default is True.
    _D_quad : bool
        Set to ``True`` to perform double layer potential quadrature without removing
        singularities. Default is ``False``. This is intended for developer use.

    """
    if basis is None:
        ndim = 1
        known_map = None
    else:
        ndim = basis.num_modes
        known_map = ("Phi_tilde", basis.evaluate)

    kernel = _kernel_dipole if _D_quad else _kernel_dipole_plus_half

    result = singular_integral(
        eval_data,
        source_data,
        interpolator,
        kernel,
        known_map=known_map,
        ndim=ndim,
        chunk_size=chunk_size,
        _prune_data=prune_data,
    )
    if ndim == 1:
        result = result.squeeze(-1)

    if _D_quad:
        result += eval_data["Phi_tilde(x)"] / 2

    return result


@eqx.filter_jit
def _direct_solve(
    boundary_condition, potential_data, source_data, interpolator, basis, options
):
    potential_grid = interpolator.eval_grid
    source_grid = interpolator.source_grid

    assert basis.M <= potential_grid.M
    assert basis.N <= potential_grid.N
    well_posed = potential_grid.num_nodes == basis.num_modes
    if not well_posed:
        well_posed = None

    potential_data, source_data = _prune_data(
        potential_data,
        potential_grid,
        source_data,
        source_grid,
        _kernel_dipole_plus_half,
    )
    Phi_tilde = basis.evaluate(potential_grid)
    potential_data["Phi_tilde(x)"] = Phi_tilde
    source_data["Phi_tilde"] = (
        Phi_tilde
        if (
            potential_grid.num_theta == source_grid.num_theta
            and potential_grid.num_zeta == source_grid.num_zeta
        )
        else basis.evaluate(source_grid)
    )

    D = _D_plus_half(
        potential_data,
        source_data,
        interpolator,
        basis,
        options.chunk_size,
        prune_data=False,
        _D_quad=options.D_quad,
    )
    assert D.shape == (potential_grid.num_nodes, basis.num_modes)

    if options.problem in ("exterior Neumann", "interior Dirichlet"):
        # This system is negative definite, but perhaps not symmetric.
        # Lineax assumes negative semidefinite means the operator is symmetric.
        # Hence we do not set that tag.
        D -= Phi_tilde
    elif options.problem == "interior Neumann" and basis.gauge_idx.size:
        # This system is positive definite, but the same logic above applies.
        D = jnp.delete(D, basis.gauge_idx, axis=1, assume_unique_indices=True)
        if well_posed:
            D = D[:-1]
            boundary_condition = boundary_condition[:-1]

    D = lx.MatrixLinearOperator(D)
    Phi_tilde_mn = lx.linear_solve(
        D, boundary_condition, solver=lx.AutoLinearSolver(well_posed=well_posed)
    ).value
    if options.problem == "interior Neumann" and basis.gauge_idx.size:
        Phi_tilde_mn = jnp.insert(Phi_tilde_mn, basis.gauge_idx, 0.0)

    return Phi_tilde_mn


@eqx.filter_jit
def _iterative_solve(
    boundary_condition, potential_data, source_data, interpolator, options
):
    potential_grid = interpolator.eval_grid
    source_grid = interpolator.source_grid

    potential_data, source_data = _prune_data(
        potential_data,
        potential_grid,
        source_data,
        source_grid,
        _kernel_dipole_plus_half,
    )
    Phi_tilde_0 = options.Phi_tilde_0
    if Phi_tilde_0 is None:
        Phi_tilde_0 = jnp.zeros(potential_grid.num_nodes)
    assert Phi_tilde_0.size == potential_grid.num_nodes

    subtract_phi = options.problem in ("exterior Neumann", "interior Dirichlet")
    solvers = {"gmres": lx.GMRES, "bicgstab": lx.BiCGStab}
    if options.solve_method in solvers:
        operator = lx.FunctionLinearOperator(
            partial(
                _linear_potential_operator,
                potential_data=potential_data,
                source_data=source_data,
                interpolator=interpolator,
                chunk_size=options.chunk_size,
                subtract_phi=subtract_phi,
            ),
            jax.ShapeDtypeStruct(Phi_tilde_0.shape, Phi_tilde_0.dtype),
        )
        solution = lx.linear_solve(
            operator,
            boundary_condition,
            solver=solvers[options.solve_method](
                atol=options.atol,
                rtol=options.rtol,
                max_steps=options.max_steps,
            ),
            options={"y0": Phi_tilde_0},
            throw=options.throw,
        )
        if options.full_output:
            err = jnp.abs(operator.mv(solution.value) - boundary_condition).max()
            return solution.value, (err, solution.stats["num_steps"])
        return solution.value

    # Some JAX versions fail to transpose scan, so we keep fixed point.
    xi = 2 / 3
    args = (
        boundary_condition,
        potential_data,
        source_data,
        interpolator,
        options.chunk_size,
        xi,
        subtract_phi,
    )
    solution = optx.fixed_point(
        _iteration_operator,
        optx.FixedPointIteration(rtol=options.rtol, atol=options.atol),
        Phi_tilde_0,
        args,
        max_steps=options.max_steps,
        adjoint=optx.ImplicitAdjoint(
            lx.GMRES(
                rtol=options.rtol,
                atol=options.atol,
                max_steps=options.max_steps,
            )
        ),
        throw=options.throw,
    )
    if options.full_output:
        err = jnp.abs(
            _linear_potential_operator(
                solution.value,
                potential_data,
                source_data,
                interpolator,
                options.chunk_size,
                subtract_phi,
            )
            - boundary_condition
        ).max()
        return solution.value, (err, solution.stats["num_steps"])
    return solution.value


def _iteration_operator(Phi_tilde, args):
    """Fixed-point iteration for the selected boundary integral equation."""
    (
        rhs,
        potential_data,
        source_data,
        interpolator,
        chunk_size,
        xi,
        subtract_phi,
    ) = args
    potential_data["Phi_tilde(x)"] = Phi_tilde
    source_data["Phi_tilde"] = _interp(
        Phi_tilde, interpolator.eval_grid, interpolator.source_grid
    )
    out = _D_plus_half(
        potential_data,
        source_data,
        interpolator,
        chunk_size=chunk_size,
        prune_data=False,
    )
    if subtract_phi:
        out = ((xi - 1) * Phi_tilde + out - rhs) / xi
    else:
        out = (xi * Phi_tilde - out + rhs) / xi
    return out


def _linear_potential_operator(
    Phi_tilde, potential_data, source_data, interpolator, chunk_size, subtract_phi
):
    """Equation solved by the iterative linear solver."""
    potential_data["Phi_tilde(x)"] = Phi_tilde
    source_data["Phi_tilde"] = _interp(
        Phi_tilde, interpolator.eval_grid, interpolator.source_grid
    )
    out = _D_plus_half(
        potential_data,
        source_data,
        interpolator,
        chunk_size=chunk_size,
        prune_data=False,
    )
    if subtract_phi:
        out -= Phi_tilde
    return out


def _interp(x, input_grid, output_grid):
    if (
        input_grid.num_theta == output_grid.num_theta
        and input_grid.num_zeta == output_grid.num_zeta
    ):
        return x
    return rfft_interp2d(
        input_grid.meshgrid_reshape(x, "rtz")[0],
        output_grid.num_theta,
        output_grid.num_zeta,
        dx=2 * jnp.pi / input_grid.num_theta,
        dy=2 * jnp.pi / input_grid.num_zeta / input_grid.NFP,
    ).ravel(order="F")


@register_compute_fun(
    name="interpolator",
    label="",
    units="",
    units_long="",
    description="Interpolator for singular integrals.",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={"grid": []},
    profiles=[],
    data=["|e_theta x e_zeta|", "e_theta", "e_zeta"],
    parameterization=["desc.geometry.surface.FourierRZToroidalSurface"],
    q="int : Order of quadrature in polar domain.",
    potential_grid="""LinearGrid :
        Grid to evaluate potential on boundary.
        If not given, default is to interpolate to source grid.
        """,
    warn_fft="""bool :
        Whether to warn if the interpolation will be lossy. Default is ``True``.
        """,
    use_dft="""bool :
        Whether to use the DFT interpolator instead of the FFT interpolator.
        Default is ``False``.
        """,
)
def _interpolator(params, transforms, profiles, data, **kwargs):
    # noqa: unused dependency
    grid = transforms["grid"]
    potential_grid = kwargs.get("potential_grid", grid)
    data["interpolator"] = get_interpolator(potential_grid, grid, data, **kwargs)

    # TODO: interpolate Rb_mn, Zb_mn, and omegab_mn directly
    data["potential data"] = {
        "R": _interp(data["R"], grid, potential_grid),
        "omega": _interp(data["omega"], grid, potential_grid),
        "Z": _interp(data["Z"], grid, potential_grid),
    }
    zeta = potential_grid.nodes[:, 2]
    data["potential data"]["phi"] = zeta + data["potential data"]["omega"]

    return data


@register_compute_fun(
    name="potential data",
    label="potential data",
    units="~",
    units_long="not applicable",
    description="RpZ position on the potential grid",
    dim=1,
    coordinates="rtz",
    params=[],
    transforms={},
    profiles=[],
    data=["interpolator"],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
    public=False,
)
def _potential_grid_position(params, transforms, profiles, data, **kwargs):
    # noqa: unused dependency
    return data


@register_compute_fun(
    name="B0 (surface)",
    label="B_0",
    units="T",
    units_long="Tesla",
    description="Auxiliary harmonic field evaluated on the boundary surface",
    dim=3,
    coordinates="tz",
    params=[],
    transforms={"grid": []},
    profiles=[],
    data=["x"],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
    B0="_MagneticField : Field object to compute with.",
    B0_params="dict : Optional I and Y overrides for the auxiliary field.",
    field_grid="Grid : Source grid used to compute magnetic field.",
    options=Options.__doc__,
    public=False,
)
def _B0_surface(params, transforms, profiles, data, **kwargs):
    """Evaluate the physical harmonic representative used by the BIE."""
    options = kwargs.get("options", Options())
    field_kwargs = {
        "coords": data["x"],
        "source_grid": kwargs.get("field_grid", None),
        "chunk_size": options.chunk_size,
    }
    if "B0_params" in kwargs:
        field_kwargs["params"] = kwargs["B0_params"]
    data["B0 (surface)"] = kwargs["B0"].compute_magnetic_field(**field_kwargs)
    return data


@register_compute_fun(
    name="S[B0*n]",
    label="S[B_0 \\cdot n_{\\rho}]",
    units="T m",
    units_long="Tesla meter",
    description="Single layer potential of monopole density B0*n",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={},
    profiles=[],
    data=_kernel_monopole.keys + ["interpolator"],
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
    options=Options.__doc__,
    public=False,
)
def _S_B0_n(params, transforms, profiles, data, **kwargs):
    # noqa: unused dependency
    options = kwargs.get("options", Options())
    data["S[B0*n]"] = singular_integral(
        data.get("potential data", data),
        data,
        data["interpolator"],
        _kernel_monopole,
        chunk_size=options.chunk_size,
    ).squeeze(-1)
    return data


@register_compute_fun(
    name="Phi_tilde_mn",
    label="\\widetilde{\\Phi}_{m n}",
    units="T m",
    units_long="Tesla meter",
    description="Fourier coefficients of the globally defined harmonic remainder",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={"Phi_tilde": [[0, 0, 0]]},
    profiles=[],
    data=list(set(_kernel_dipole_plus_half.keys) - {"Phi_tilde"})
    + ["S[B0*n]", "interpolator"],
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
    options=Options.__doc__,
)
def _scalar_potential_mn_Neumann(params, transforms, profiles, data, **kwargs):
    # noqa: unused dependency
    options = kwargs.get("options", Options())
    _check_solve_method(options.solve_method)

    if options.solve_method == "direct":
        data["Phi_tilde_mn"] = _direct_solve(
            data["S[B0*n]"],
            data.get("potential data", data),
            data,
            data["interpolator"],
            transforms["Phi_tilde"].basis,
            options,
        )
    else:
        data["Phi_tilde"] = _iterative_solve(
            data["S[B0*n]"],
            data.get("potential data", data),
            data,
            data["interpolator"],
            options,
        )
        if options.full_output:
            data["Phi_tilde"], (data["Phi_tilde error"], data["num_steps"]) = data[
                "Phi_tilde"
            ]

        assert data["Phi_tilde"].size == transforms["Phi_tilde"].grid.num_nodes
        data["Phi_tilde_mn"] = transforms["Phi_tilde"].fit(data["Phi_tilde"])
    return data


@register_compute_fun(
    name="Phi_tilde",
    label="\\widetilde{\\Phi}",
    units="T m",
    units_long="Tesla meter",
    description=(
        "Single-valued potential remainder solved by the boundary integral equation"
    ),
    dim=1,
    coordinates="tz",
    params=[],
    transforms={"Phi_tilde": [[0, 0, 0]]},
    profiles=[],
    data=["Phi_tilde_mn"],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
)
def _Phi_tilde(params, transforms, profiles, data, **kwargs):
    assert data["Phi_tilde_mn"].size == transforms["Phi_tilde"].basis.num_modes
    data["Phi_tilde"] = transforms["Phi_tilde"].transform(data["Phi_tilde_mn"])
    return data


@register_compute_fun(
    name="Phi_tilde_t",
    label="\\partial_{\\theta} \\widetilde{\\Phi}",
    units="T m",
    units_long="Tesla meter",
    description="Poloidal derivative of the single-valued potential remainder",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={"Phi_tilde": [[0, 1, 0]]},
    profiles=[],
    data=["Phi_tilde_mn"],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
)
def _Phi_tilde_t(params, transforms, profiles, data, **kwargs):
    assert data["Phi_tilde_mn"].size == transforms["Phi_tilde"].basis.num_modes
    data["Phi_tilde_t"] = transforms["Phi_tilde"].transform(data["Phi_tilde_mn"], dt=1)
    return data


@register_compute_fun(
    name="Phi_tilde_z",
    label="\\partial_{\\zeta} \\widetilde{\\Phi}",
    units="T m",
    units_long="Tesla meter",
    description="Toroidal derivative of the single-valued potential remainder",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={"Phi_tilde": [[0, 0, 1]]},
    profiles=[],
    data=["Phi_tilde_mn"],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
)
def _Phi_tilde_z(params, transforms, profiles, data, **kwargs):
    assert data["Phi_tilde_mn"].size == transforms["Phi_tilde"].basis.num_modes
    data["Phi_tilde_z"] = transforms["Phi_tilde"].transform(data["Phi_tilde_mn"], dz=1)
    return data


@register_compute_fun(
    name="K_vc",
    label="\\nabla \\widetilde{\\Phi} \\times n",
    units="T",
    units_long="Tesla",
    description=(
        "Surface-current contribution from the single-valued potential remainder"
    ),
    dim=3,
    coordinates="tz",
    params=[],
    transforms={},
    profiles=[],
    data=[
        "n_rho x grad(theta)",
        "n_rho x grad(zeta)",
        "Phi_tilde_t",
        "Phi_tilde_z",
    ],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
)
def _virtual_surface_current(params, transforms, profiles, data, **kwargs):
    data["K_vc"] = -(
        data["Phi_tilde_t"][:, None] * data["n_rho x grad(theta)"]
        + data["Phi_tilde_z"][:, None] * data["n_rho x grad(zeta)"]
    )
    return data


@register_compute_fun(
    name="Phi_tilde error",
    label="\\widetilde{\\Phi}_{\\text{error}}",
    units="T m",
    units_long="Tesla meter",
    description="Magnetic scalar potential error",
    dim=0,
    coordinates="",
    params=[],
    transforms={},
    profiles=[],
    data=["Phi_tilde_mn"],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
    public=False,
)
def _Phi_tilde_error(params, transforms, profiles, data, **kwargs):
    # noqa: unused dependency
    return data


@register_compute_fun(
    name="num_steps",
    label="\\text{number of steps}",
    units="",
    units_long="",
    description="Magnetic scalar potential number of steps for inversion",
    dim=0,
    coordinates="",
    params=[],
    transforms={},
    profiles=[],
    data=["Phi_tilde_mn"],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
    public=False,
)
def _Phi_tilde_num_steps(params, transforms, profiles, data, **kwargs):
    # noqa: unused dependency
    return data


@register_compute_fun(
    name="B0 x n",
    label="B_0 \\times n",
    units="T",
    units_long="Tesla",
    description="Tangential trace of the auxiliary harmonic field",
    dim=3,
    coordinates="tz",
    params=[],
    transforms={},
    profiles=[],
    data=["B0 (surface)", "n_rho"],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
    public=False,
)
def _B0_cross_n(params, transforms, profiles, data, **kwargs):
    data["B0 x n"] = cross(data["B0 (surface)"], data["n_rho"])
    return data


@register_compute_fun(
    name="B x n",
    label="(\\nabla \\varphi + B_0) \\times n",
    units="T",
    units_long="Tesla",
    description="Tangential trace of the physical boundary field",
    dim=3,
    coordinates="tz",
    params=[],
    transforms={},
    profiles=[],
    data=["K_vc", "B0 x n"],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
)
def _B_cross_n(params, transforms, profiles, data, **kwargs):
    data["B x n"] = data["K_vc"] + data["B0 x n"]
    return data


@register_compute_fun(
    name="|B x n|^2",
    label="\\lvert B \\times n \\rvert^2",
    units="T^2",
    units_long="Tesla squared",
    description="Squared norm of the physical tangential boundary field",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={},
    profiles=[],
    data=["B x n"],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
)
def _B_cross_n_squared(params, transforms, profiles, data, **kwargs):
    data["|B x n|^2"] = dot(data["B x n"], data["B x n"])
    return data


@register_compute_fun(
    name="B_remainder",
    label="\\nabla \\varphi",
    units="T",
    units_long="Tesla",
    description="Magnetic field due to potential which solves the"
    " boundary value problem.",
    dim=3,
    coordinates="RpZ",
    params=[],
    transforms={},
    profiles=[],
    data=_kernel_BS_plus_grad_S.keys,
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
    options=Options.__doc__,
    eval_interpolator="""_BIESTInterpolator :
        Interpolator from source grid to evaluation grid on boundary.
        If not given, default is to interpolate to source grid.
        """,
    on_boundary="bool : Whether RpZcoords are on boundary surface.",
    public=False,
)
def _B_remainder(params, transforms, profiles, data, RpZ_data, **kwargs):
    # noqa: unused dependency
    options = kwargs.get("options", Options())
    sign = 1 - 2 * int("exterior" in options.problem)

    if kwargs["on_boundary"]:
        RpZ_data["B_remainder"] = (
            sign
            * 2
            * singular_integral(
                RpZ_data,
                data,
                kwargs.get("eval_interpolator", data.get("interpolator", None)),
                _kernel_BS_plus_grad_S,
                chunk_size=options.chunk_size,
            )
        )
    else:
        grid = transforms["grid"]
        eval_data, source_data = _prune_data(
            RpZ_data, None, data, grid, _kernel_BS_plus_grad_S
        )
        RpZ_data["B_remainder"] = sign * _nonsingular_part(
            eval_data,
            None,
            source_data,
            grid,
            st=jnp.nan,
            sz=jnp.nan,
            kernel=_kernel_BS_plus_grad_S,
            chunk_size=options.chunk_size,
        )
    return RpZ_data


@register_compute_fun(
    name="B",
    label="B",
    units="T",
    units_long="Tesla",
    description="Magnetic field",
    dim=3,
    coordinates="RpZ",
    params=[],
    transforms={},
    profiles=[],
    data=["B_remainder", "B0"],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
)
def _total_B(params, transforms, profiles, data, RpZ_data, **kwargs):
    RpZ_data["B"] = RpZ_data["B_remainder"] + RpZ_data["B0"]
    return RpZ_data


@register_compute_fun(
    name="B0*n",
    label="B_0 \\cdot n_{\\rho}",
    units="T",
    units_long="Tesla",
    description="Auxiliary field dotted into flux surface normal",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={},
    profiles=[],
    data=["B0 (surface)", "n_rho"],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
)
def _B0_dot_n(params, transforms, profiles, data, **kwargs):
    data["B0*n"] = dot(data["B0 (surface)"], data["n_rho"])
    return data


@register_compute_fun(
    name="B0",
    label="B0",
    units="T",
    units_long="Tesla",
    description="Auxiliary field",
    dim=3,
    coordinates="RpZ",
    params=[],
    transforms={"grid": []},
    profiles=[],
    data=[],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
    B0="_MagneticField : Field object to compute with.",
    B0_params="dict : Optional I and Y overrides for the auxiliary field.",
    field_grid="Grid : Source grid used to compute magnetic field.",
    options=Options.__doc__,
    public=False,
)
def _B0_field(params, transforms, profiles, data, RpZ_data, **kwargs):
    options = kwargs.get("options", Options())
    coords = jnp.column_stack([RpZ_data["R"], RpZ_data["phi"], RpZ_data["Z"]])
    field_kwargs = {
        "coords": coords,
        "source_grid": kwargs.get("field_grid", None),
        "chunk_size": options.chunk_size,
    }
    if "B0_params" in kwargs:
        field_kwargs["params"] = kwargs["B0_params"]
    RpZ_data["B0"] = kwargs["B0"].compute_magnetic_field(**field_kwargs)
    return RpZ_data


@register_compute_fun(
    name="B_coil",
    label="B_{\\text{coil}}",
    units="T",
    units_long="Tesla",
    description="Magnetic field due to coils",
    dim=3,
    coordinates="rtz",
    params=[],
    transforms={"grid": []},
    profiles=[],
    data=["x"],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
    options=Options.__doc__,
    B_coil="_MagneticField : Field object to compute with.",
    field_grid="Grid : Source grid used to compute magnetic field.",
)
def _B_coil_field(params, transforms, profiles, data, **kwargs):
    options = kwargs.get("options", Options())
    data["B_coil"] = kwargs["B_coil"].compute_magnetic_field(
        coords=data["x"],
        source_grid=kwargs.get("field_grid", None),
        chunk_size=options.B_coil_chunk_size,
    )
    return data


@register_compute_fun(
    name="n_rho x B_coil",
    label="n_{\\rho} \\times B_{\\text{coil}}",
    units="T",
    units_long="Tesla",
    description="Flux surface normal cross magnetic field due to coils",
    dim=3,
    coordinates="rtz",
    params=[],
    transforms={},
    profiles=[],
    data=["n_rho", "B_coil"],
    parameterization="desc.magnetic_fields._laplace.SourceFreeField",
)
def _n_rho_x_B_coil(params, transforms, profiles, data, **kwargs):
    data["n_rho x B_coil"] = cross(data["n_rho"], data["B_coil"])
    return data


@register_compute_fun(
    name="Y_coil",
    label="Y_{\\text{coil}}",
    units="T m",
    units_long="Tesla meter",
    description="Net poloidal current produced by magnetic coils",
    dim=0,
    coordinates="",
    params=["Y"],
    transforms={},
    profiles=[],
    data=["e_zeta", "B_coil"],
    grid_requirement={"can_fft2": True},
    options=Options.__doc__,
    parameterization="desc.magnetic_fields._laplace.FreeSurfaceOuterField",
)
def _Y_coil(params, transforms, profiles, data, **kwargs):
    if params.get("Y", None) is not None:
        data["Y_coil"] = params["Y"]
        return data
    # Equation B.2 in [1]_.
    data["Y_coil"] = dot(data["B_coil"], data["e_zeta"]).mean()
    return data


@register_compute_fun(
    name="varphi_tilde_mn",
    label="\\widetilde{\\varphi}_{m n}",
    units="T m",
    units_long="Tesla meter",
    description="Fourier coefficients of globally defined coil-potential remainder",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={"varphi_tilde": [[0, 0, 0]]},
    profiles=[],
    data=[
        "n_rho x B_coil",
        "n_rho x grad(theta)",
        "n_rho x grad(zeta)",
        "n_rho",
        "grad(phi)",
        "Y_coil",
    ],
    parameterization="desc.magnetic_fields._laplace.FreeSurfaceOuterField",
)
def _varphi_tilde_mn(params, transforms, profiles, data, **kwargs):
    """Return harmonics of the globally defined coil-potential remainder.

    ``B_coil`` must be smooth and divergence free for correctness of inversion.
    TODO: Compute this from scalar potential integral, without inversion.
    """
    grid = transforms["grid"]
    assert grid.num_rho == 1

    basis = transforms["varphi_tilde"].basis
    # TODO: could compute these in objective build
    #       and avoid computing if they are passed in as kwargs
    _t = basis.evaluate(grid, [0, 1, 0])[:, None]
    _z = basis.evaluate(grid, [0, 0, 1])[:, None]

    mat = (
        _t * data["n_rho x grad(theta)"][..., None]
        + _z * data["n_rho x grad(zeta)"][..., None]
    ).reshape(grid.num_nodes * 3, basis.num_modes)
    if basis.gauge_idx.size:
        mat = jnp.delete(mat, basis.gauge_idx, axis=1, assume_unique_indices=True)
    mat = lx.MatrixLinearOperator(mat)

    # Equation 5.22 in [1]_.
    varphi_tilde_mn = lx.linear_solve(
        mat,
        (
            data["n_rho x B_coil"]
            - data["Y_coil"] * cross(data["n_rho"], data["grad(phi)"])
        ).ravel(),
        solver=lx.AutoLinearSolver(well_posed=None),
    ).value
    if basis.gauge_idx.size:
        varphi_tilde_mn = jnp.insert(varphi_tilde_mn, basis.gauge_idx, 0.0)

    data["varphi_tilde_mn"] = varphi_tilde_mn
    return data


@register_compute_fun(
    name="varphi_tilde",
    label="\\widetilde{\\varphi}",
    units="T m",
    units_long="Tesla meter",
    description="Globally defined remainder of the coil scalar potential",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={"varphi_tilde": [[0, 0, 0]]},
    profiles=[],
    data=["varphi_tilde_mn"],
    parameterization="desc.magnetic_fields._laplace.FreeSurfaceOuterField",
)
def _varphi_tilde(params, transforms, profiles, data, **kwargs):
    data["varphi_tilde"] = transforms["varphi_tilde"].transform(data["varphi_tilde_mn"])
    return data


@register_compute_fun(
    name="varphi_coordinate_secular",
    label="Y_{\\mathrm{coil}}\\zeta",
    units="T m",
    units_long="Tesla meter",
    description="Coordinate-secular term in the boundary trace of the coil potential",
    dim=1,
    coordinates="z",
    params=["Y"],
    transforms={},
    profiles=[],
    data=["zeta"],
    parameterization="desc.magnetic_fields._laplace.FreeSurfaceOuterField",
)
def _varphi_coordinate_secular(params, transforms, profiles, data, **kwargs):
    data["varphi_coordinate_secular"] = params["Y"] * data["zeta"]
    return data


@register_compute_fun(
    name="varphi_secular",
    label="\\varphi_{\\mathrm{secular}}",
    units="T m",
    units_long="Tesla meter",
    description="Physical secular potential of the coil field",
    dim=1,
    coordinates="tz",
    params=["Y"],
    transforms={},
    profiles=[],
    data=["phi"],
    parameterization="desc.magnetic_fields._laplace.FreeSurfaceOuterField",
)
def _varphi_secular(params, transforms, profiles, data, **kwargs):
    data["varphi_secular"] = params["Y"] * data["phi"]
    return data


@register_compute_fun(
    name="varphi_periodic",
    label="\\varphi_{\\mathrm{periodic}}",
    units="T m",
    units_long="Tesla meter",
    description="Coordinate-periodic part of the coil scalar potential",
    dim=1,
    coordinates="tz",
    params=["Y"],
    transforms={},
    profiles=[],
    data=["varphi_tilde", "omega", "potential data"],
    parameterization="desc.magnetic_fields._laplace.FreeSurfaceOuterField",
)
def _varphi_periodic(params, transforms, profiles, data, **kwargs):
    if "potential data" in data:
        omega = data["potential data"]["omega"]
    else:
        omega = data["omega"]
    # Equations 5.10, 5.13, and 5.15 in [1]_, with phi = zeta + omega.
    data["varphi_periodic"] = data["varphi_tilde"] + params["Y"] * omega
    return data


@register_compute_fun(
    name="varphi",
    label="\\varphi",
    units="T m",
    units_long="Tesla meter",
    description="Magnetic scalar potential of coil field",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={},
    profiles=[],
    data=["varphi_tilde", "varphi_secular"],
    parameterization="desc.magnetic_fields._laplace.FreeSurfaceOuterField",
)
def _varphi(params, transforms, profiles, data, **kwargs):
    # Equation 5.15 in [1]_.
    data["varphi"] = data["varphi_tilde"] + data["varphi_secular"]
    return data


@register_compute_fun(
    name="Phi_tilde_mn",
    label="\\widetilde{\\Phi}_{m n}",
    units="T m",
    units_long="Tesla meter",
    description="Fourier coefficients of the globally defined dipole density",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={"Phi_tilde": [[0, 0, 0]]},
    profiles=[],
    data=list(set(_kernel_dipole_plus_half.keys) - {"Phi_tilde"})
    + ["varphi_tilde", "S[B0*n]", "interpolator"],
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    parameterization="desc.magnetic_fields._laplace.FreeSurfaceOuterField",
    options=Options.__doc__,
)
def _dipole_density_mn_free_surface(params, transforms, profiles, data, **kwargs):
    """Solve for tilde-Phi, the remainder relative to physical period fields."""
    # noqa: unused dependency
    options = kwargs.get("options", Options())._replace(problem="interior Dirichlet")
    _check_solve_method(options.solve_method)

    boundary_condition = data["S[B0*n]"] - data["varphi_tilde"]
    if options.solve_method == "direct":
        data["Phi_tilde_mn"] = _direct_solve(
            boundary_condition,
            data.get("potential data", data),
            data,
            data["interpolator"],
            transforms["Phi_tilde"].basis,
            options,
        )
    else:
        data["Phi_tilde"] = _iterative_solve(
            boundary_condition,
            data.get("potential data", data),
            data,
            data["interpolator"],
            options,
        )
        if options.full_output:
            data["Phi_tilde"], (data["Phi_tilde error"], data["num_steps"]) = data[
                "Phi_tilde"
            ]

        assert data["Phi_tilde"].size == transforms["Phi_tilde"].grid.num_nodes
        data["Phi_tilde_mn"] = transforms["Phi_tilde"].fit(data["Phi_tilde"])
    return data


@register_compute_fun(
    name="γ potential",
    label="\\gamma",
    units="T m",
    units_long="Tesla meter",
    description="Double layer potential with dipole density -Φ̃",
    dim=1,
    coordinates="tz",
    params=[],
    transforms={},
    profiles=[],
    data=_kernel_dipole_plus_half.keys + ["interpolator"],
    resolution_requirement="tz",
    grid_requirement={"can_fft2": True},
    parameterization="desc.magnetic_fields._laplace.FreeSurfaceOuterField",
    options=Options.__doc__,
    public=False,
)
def _gamma_potential(params, transforms, profiles, data, **kwargs):
    # noqa: unused dependency
    options = kwargs.get("options", Options())
    data["Phi_tilde(x)"] = data["Phi_tilde"]
    # Left hand side of equation 5.20 in [1]_ computed by evaluating
    # the right hand side. This is used for testing.
    data["γ potential"] = data["Phi_tilde"] - _D_plus_half(
        data,
        data,
        data["interpolator"],
        chunk_size=options.chunk_size,
    )
    return data
