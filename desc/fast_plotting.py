"""Fast drop-in replacement for desc.plotting.

All function names and signatures are identical to desc.plotting.
Import this module instead of desc.plotting for faster rendering.

Performance improvements
------------------------
plot_3d
    Replaces go.Mesh3d + custom triangulation with go.Surface, which uses
    WebGL's native structured-surface renderer.  On a structured (theta × zeta)
    grid this is 10-50x faster in the browser and requires no triangle index
    arrays.  Also batches the two separate eq.compute() calls (quantity +
    coordinates) into a single JAX pass.

plot_coils
    Replaces one go.Scatter3d trace *per coil* with one trace *per unique
    style* (color/linewidth/linestyle) using None-separated polylines.  With
    100+ coils this collapses O(N_coils) WebGL draw calls to O(N_styles) ≈ 1-4.

Everything else
    Imported unchanged from desc.plotting.
"""

import warnings
from collections import defaultdict

import numpy as np
import plotly.graph_objects as go
from pylatexenc.latex2text import LatexNodes2Text

# ------------------------------------------------------------------
# Re-export everything from desc.plotting so this module is a
# complete drop-in replacement.  We override only the slow functions.
# ------------------------------------------------------------------
from desc.plotting import (  # noqa: F401 – re-exported
    plot_1d,
    plot_2d,
    plot_basis,
    plot_boozer_modes,
    plot_boozer_surface,
    plot_boundaries,
    plot_boundary,
    plot_coefficients,
    plot_comparison,
    plot_fsa,
    plot_gammac,
    plot_grid,
    plot_logo,
    plot_qs_error,
    plot_section,
    plot_surfaces,
    plot_field_lines,
    poincare_plot,
    # private helpers used below
    _compute,
    _compute_Bn,
    _get_grid,
    colorblind_colors,
    sequential_colors,
    dashes,
)

from desc.coils import CoilSet, _Coil
from desc.compute import data_index
from desc.compute.utils import _parse_parameterization
from desc.grid import LinearGrid
from desc.utils import errorif, only1, setdefault

try:
    import tkinter
    _dpi = tkinter.Tk().winfo_fpixels("1i")
except Exception:
    _dpi = 72

_LATEX = LatexNodes2Text()  # reuse one instance — instantiation is not free

__all__ = [
    "plot_1d",
    "plot_2d",
    "plot_3d",
    "plot_basis",
    "plot_boozer_modes",
    "plot_boozer_surface",
    "plot_boundaries",
    "plot_boundary",
    "plot_coefficients",
    "plot_coils",
    "plot_comparison",
    "plot_fsa",
    "plot_gammac",
    "plot_grid",
    "plot_logo",
    "plot_qs_error",
    "plot_section",
    "plot_surfaces",
    "plot_field_lines",
    "poincare_plot",
]


# ======================================================================
# plot_3d — fast version
# ======================================================================

def plot_3d(
    eq,
    name,
    grid=None,
    log=False,
    fig=None,
    return_data=False,
    **kwargs,
):
    """Plot 3D surfaces.  Drop-in replacement for desc.plotting.plot_3d.

    Uses go.Surface (structured-grid WebGL renderer) instead of go.Mesh3d,
    eliminating triangulation overhead and dramatically reducing browser render
    time.  Also batches coordinate + quantity computation into one JAX call.

    Parameters
    ----------
    eq : Equilibrium or Surface
        Object to plot.
    name : str
        Quantity to plot (e.g. ``"|F|"``, ``"B*n"``).
    grid : LinearGrid, optional
        2-D grid (exactly one dimension must be size 1).
        Default: M=50 poloidal × N=50*NFP toroidal, rho=1.
    log : bool
        Log10 colour scale.
    fig : plotly Figure, optional
        Figure to add the trace to.
    return_data : bool
        Also return a dict of the plotted arrays.
    **kwargs
        figsize, alpha, cmap, title, levels, component,
        showgrid, zeroline, showscale, showticklabels, showaxislabels,
        field, field_grid, chunk_size, B_plasma_chunk_size.

    Returns
    -------
    fig : plotly Figure
    plot_data : dict  (only if return_data=True)
    """
    # ------------------------------------------------------------------
    # Grid / defaults
    # ------------------------------------------------------------------
    if grid is None:
        grid = _get_grid(M=50, N=int(50 * eq.NFP), NFP=1, endpoint=True)
    assert isinstance(grid, LinearGrid), "grid must be a LinearGrid for 3-D plotting"
    assert only1(
        grid.num_rho == 1, grid.num_theta == 1, grid.num_zeta == 1
    ), "Grid must be 2-D (exactly one dimension of size 1)"

    figsize        = kwargs.pop("figsize", (10, 10))
    alpha          = kwargs.pop("alpha", 1.0)
    cmap           = kwargs.pop("cmap", "RdBu_r")
    title          = kwargs.pop("title", "")
    levels         = kwargs.pop("levels", None)
    component      = kwargs.pop("component", None)
    showgrid       = kwargs.pop("showgrid", True)
    zeroline       = kwargs.pop("zeroline", True)
    showscale      = kwargs.pop("showscale", True)
    showticklabels = kwargs.pop("showticklabels", True)
    showaxislabels = kwargs.pop("showaxislabels", True)

    # ------------------------------------------------------------------
    # Data computation
    # ------------------------------------------------------------------
    if name == "B*n":
        # B·n path: field evaluation is always separate from coordinates
        data, label = _compute_Bn(
            eq=eq,
            field=kwargs.pop("field", None),
            plot_grid=grid,
            field_grid=kwargs.pop("field_grid", None),
            chunk_size=kwargs.pop("chunk_size", None),
            B_plasma_chunk_size=kwargs.pop("B_plasma_chunk_size", None),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coords = eq.compute(["X", "Y", "Z"], grid=grid)
        X = coords["X"].reshape(
            (grid.num_theta, grid.num_rho, grid.num_zeta), order="F"
        )
        Y = coords["Y"].reshape(
            (grid.num_theta, grid.num_rho, grid.num_zeta), order="F"
        )
        Z = coords["Z"].reshape(
            (grid.num_theta, grid.num_rho, grid.num_zeta), order="F"
        )
    else:
        # Fast path: batch quantity + coordinates in ONE JAX call
        parameterization = _parse_parameterization(eq)
        errorif(
            name not in data_index[parameterization],
            msg=f"Unrecognized value '{name}' for parameterization {parameterization}.",
        )
        assert component in [None, "R", "phi", "Z"], (
            f"component must be one of [None, 'R', 'phi', 'Z'], got {component}"
        )
        _ = kwargs.pop("field", None)
        _ = kwargs.pop("field_grid", None)
        _ = kwargs.pop("chunk_size", None)
        _ = kwargs.pop("B_plasma_chunk_size", None)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            computed = eq.compute([name, "X", "Y", "Z"], grid=grid)

        raw   = computed[name]
        dim   = data_index[parameterization][name]["dim"]
        label = data_index[parameterization][name]["label"]
        units = data_index[parameterization][name]["units"]

        if dim > 1:
            components = {"R": 0, "phi": 1, "Z": 2}
            if component is None:
                raw   = np.linalg.norm(raw, axis=-1)
                label = "|" + label + "|"
            else:
                raw   = raw[:, components[component]]
                label = "(" + label + ")_"
                label += component if component in ["R", "Z"] else r"\phi"

        label = r"$" + label + r"~(" + units + r")$"
        data  = raw.reshape(
            (grid.num_theta, grid.num_rho, grid.num_zeta), order="F"
        )
        X = computed["X"].reshape(
            (grid.num_theta, grid.num_rho, grid.num_zeta), order="F"
        )
        Y = computed["Y"].reshape(
            (grid.num_theta, grid.num_rho, grid.num_zeta), order="F"
        )
        Z = computed["Z"].reshape(
            (grid.num_theta, grid.num_rho, grid.num_zeta), order="F"
        )

    errorif(
        len(kwargs) != 0,
        msg=f"plot_3d got unexpected keyword argument(s): {list(kwargs.keys())}",
    )

    # ------------------------------------------------------------------
    # Squeeze the singleton dimension for go.Surface (needs 2-D arrays)
    # ------------------------------------------------------------------
    if grid.num_rho == 1:
        # Surface at fixed rho (most common): theta × zeta
        X2d   = X[:, 0, :]
        Y2d   = Y[:, 0, :]
        Z2d   = Z[:, 0, :]
        data2d = data[:, 0, :]
    elif grid.num_theta == 1:
        # Fixed theta: rho × zeta
        X2d   = X[0, :, :]
        Y2d   = Y[0, :, :]
        Z2d   = Z[0, :, :]
        data2d = data[0, :, :]
    else:
        # Fixed zeta (cross-section): theta × rho
        X2d   = X[:, :, 0]
        Y2d   = Y[:, :, 0]
        Z2d   = Z[:, :, 0]
        data2d = data[:, :, 0]

    # ------------------------------------------------------------------
    # Log scale
    # ------------------------------------------------------------------
    cmin = cmax = None
    if log:
        data2d = np.log10(np.abs(data2d))
        cmin   = int(np.floor(np.nanmin(data2d)))
        cmax   = int(np.ceil(np.nanmax(data2d)))
        levels = setdefault(levels, np.logspace(cmin, cmax, cmax - cmin + 1))
        ticks  = np.log10(levels)
        cbar   = dict(
            title=_LATEX.latex_to_text(label),
            ticktext=[f"{lv:.0e}" for lv in levels],
            tickvals=ticks,
        )
    else:
        cbar = dict(
            title=_LATEX.latex_to_text(label),
            ticktext=levels,
            tickvals=levels,
        )

    # ------------------------------------------------------------------
    # go.Surface — structured WebGL surface renderer (no triangulation)
    # ------------------------------------------------------------------
    surface = go.Surface(
        x=X2d,
        y=Y2d,
        z=Z2d,
        surfacecolor=data2d,
        opacity=alpha,
        cmin=cmin,
        cmax=cmax,
        colorscale=cmap,
        name=_LATEX.latex_to_text(label),
        colorbar=cbar,
        showscale=showscale,
    )

    if fig is None:
        fig = go.Figure()
    fig.add_trace(surface)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    xaxis_title = "X (m)" if showaxislabels else ""
    yaxis_title = "Y (m)" if showaxislabels else ""
    zaxis_title = "Z (m)" if showaxislabels else ""

    _axis_style = dict(
        backgroundcolor="white",
        gridcolor="darkgrey",
        showbackground=False,
        zerolinecolor="darkgrey",
        showgrid=showgrid,
        zeroline=zeroline,
        showticklabels=showticklabels,
    )

    fig.update_layout(
        scene=dict(
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            zaxis_title=zaxis_title,
            aspectmode="data",
            xaxis=_axis_style,
            yaxis=_axis_style,
            zaxis=_axis_style,
        ),
        width=figsize[0] * _dpi,
        height=figsize[1] * _dpi,
        title=dict(text=title, y=0.9, x=0.5, xanchor="center", yanchor="top"),
        font=dict(family="Times"),
    )

    plot_data = {"X": X, "Y": Y, "Z": Z, name: data}
    if return_data:
        return fig, plot_data
    return fig


# ======================================================================
# plot_coils — fast version
# ======================================================================

def plot_coils(coils, grid=None, fig=None, return_data=False, **kwargs):
    """Plot 3-D coil geometry.  Drop-in replacement for desc.plotting.plot_coils.

    Groups all coils that share the same (color, linewidth, linestyle) into a
    *single* go.Scatter3d trace using None-separated polylines.  With N coils
    this reduces O(N) WebGL draw calls to O(unique styles) ≈ 1–4.

    Parameters
    ----------
    coils : _Coil, CoilSet, or iterable
    grid : LinearGrid, optional
        Discretisation grid.  Default: N=400.
    fig : plotly Figure, optional
    return_data : bool
    **kwargs
        unique, figsize, lw, ls, color,
        showgrid, zeroline, showticklabels, showaxislabels,
        check_intersection.

    Returns
    -------
    fig : plotly Figure
    plot_data : dict  (only if return_data=True)
    """
    lw                 = kwargs.pop("lw", 5)
    ls                 = kwargs.pop("ls", "solid")
    figsize            = kwargs.pop("figsize", (10, 10))
    color              = kwargs.pop("color", "black")
    unique             = kwargs.pop("unique", False)
    showgrid           = kwargs.pop("showgrid", True)
    zeroline           = kwargs.pop("zeroline", True)
    showticklabels     = kwargs.pop("showticklabels", True)
    showaxislabels     = kwargs.pop("showaxislabels", True)
    check_intersection = kwargs.pop("check_intersection", False)

    errorif(
        len(kwargs) != 0,
        msg=f"plot_coils got unexpected keyword argument(s): {list(kwargs.keys())}",
    )
    errorif(
        not isinstance(coils, _Coil),
        ValueError,
        f"Expected `coils` to be of type `_Coil`, got {type(coils)}",
    )

    # Normalise style args to lists
    if not isinstance(lw, (list, tuple)):    lw    = [lw]
    if not isinstance(ls, (list, tuple)):    ls    = [ls]
    if not isinstance(color, (list, tuple)): color = [color]

    if grid is None:
        grid = LinearGrid(N=400, endpoint=True)

    # ------------------------------------------------------------------
    # Flatten CoilSet (expand symmetry copies unless unique=True)
    # ------------------------------------------------------------------
    def _flatten(coilset):
        if hasattr(coilset, "__len__"):
            if hasattr(coilset, "_NFP") and hasattr(coilset, "_sym"):
                if not unique and (coilset.NFP > 1 or coilset.sym):
                    coilset = CoilSet.from_symmetry(
                        coilset,
                        NFP=coilset.NFP,
                        sym=coilset.sym,
                        check_intersection=check_intersection,
                    )
            return [c for item in coilset for c in _flatten(item)]
        return [coilset]

    coils_list = _flatten(coils)

    # ------------------------------------------------------------------
    # Batch coil coordinates by style — one trace per unique style
    # ------------------------------------------------------------------
    # style key → {'x': [], 'y': [], 'z': []}
    groups  = defaultdict(lambda: {"x": [], "y": [], "z": []})
    # keep per-coil data for return_data
    all_x, all_y, all_z = [], [], []

    for i, coil in enumerate(coils_list):
        pts = np.array(coil.compute("x", grid=grid, basis="xyz")["x"])
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        all_x.append(x);  all_y.append(y);  all_z.append(z)

        key = (color[i % len(color)], lw[i % len(lw)], ls[i % len(ls)])
        groups[key]["x"].extend(x.tolist() + [None])
        groups[key]["y"].extend(y.tolist() + [None])
        groups[key]["z"].extend(z.tolist() + [None])

    # ------------------------------------------------------------------
    # Emit one Scatter3d trace per style group
    # ------------------------------------------------------------------
    if fig is None:
        fig = go.Figure()

    for (c, w, s), gdata in groups.items():
        fig.add_trace(go.Scatter3d(
            x=gdata["x"],
            y=gdata["y"],
            z=gdata["z"],
            mode="lines",
            marker=dict(size=0, opacity=0),
            line=dict(color=c, width=w, dash=s),
            connectgaps=False,
            showlegend=False,
            name=f"coils [{c}]",
        ))

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    xaxis_title = "X (m)" if showaxislabels else ""
    yaxis_title = "Y (m)" if showaxislabels else ""
    zaxis_title = "Z (m)" if showaxislabels else ""

    _axis_style = dict(
        backgroundcolor="white",
        gridcolor="darkgrey",
        showbackground=False,
        zerolinecolor="darkgrey",
        showgrid=showgrid,
        zeroline=zeroline,
        showticklabels=showticklabels,
    )

    fig.update_layout(
        scene=dict(
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            zaxis_title=zaxis_title,
            xaxis=_axis_style,
            yaxis=_axis_style,
            zaxis=_axis_style,
            aspectmode="data",
        ),
        width=figsize[0] * _dpi,
        height=figsize[1] * _dpi,
    )

    plot_data = {"X": all_x, "Y": all_y, "Z": all_z}
    if return_data:
        return fig, plot_data
    return fig
